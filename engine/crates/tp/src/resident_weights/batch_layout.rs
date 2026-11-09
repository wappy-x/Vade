//! Checked chunk-major copy ranges. No floating-point operation enters the layout.
use super::{Rows,Result};
use std::ops::Range;
pub(super) struct Layout {pub stride:usize,pub chunks:usize,pub ranges:Vec<Vec<Range<usize>>>}
impl Layout {
    /// Every origin owns one complete rank-major chunk, including replicated rows.
    pub fn placement(&self,rank:usize,parts:usize)->Result<(usize,Vec<(usize,usize)>)> {
        ((0..parts).contains(&rank) && self.chunks.eq(&parts)).then_some(()).ok_or("resident device batch topology differs")?;
        let own=rank.checked_mul(self.stride).ok_or("resident device batch rank offset overflow")?;
        let remote=(0..rank).chain(rank+1..parts).map(|origin|origin.checked_mul(self.stride)
            .map(|offset|(origin,offset)).ok_or_else(||"resident device batch rank offset overflow".into())).collect::<Result<Vec<_>>>()?;
        Ok((own,remote))
    }
    /// Map one wire chunk to final job-major ranges. Replicated rows use origin zero.
    pub fn scatter(&self,origin:usize)->Result<Vec<[usize;3]>> {
        if origin>=self.chunks {return Err("resident scatter origin differs".into());}
        let mut output=0usize;let mut rows=Vec::new();
        for ranges in &self.ranges {for range in ranges {
            let words=range.len();if range.start/self.stride==origin {rows.push([range.start%self.stride,output,words]);}
            output=output.checked_add(words).ok_or("resident scatter output overflow")?;
        }}Ok(rows)
    }
    pub fn output_words(&self)->Result<usize> {self.ranges.iter().flatten().try_fold(0usize,|n,r|n.checked_add(r.len()).ok_or("resident scatter output overflow".into()))}
    pub fn new(rows:&[Rows])->Result<Self> {
        if rows.is_empty() || rows.iter().any(|r|r.count==0 || r.total==0 || r.total%r.count!=0 || !r.split && r.total!=r.count) {return Err("resident device batch geometry differs".into());}
        let chunks=rows.iter().find(|r|r.split).map_or(1,|r|r.total/r.count);
        if rows.iter().any(|r|r.split && r.total/r.count!=chunks) {return Err("resident device batch row groups differ".into());}
        let stride=rows.iter().try_fold(0usize,|n,r|n.checked_add(r.count)).ok_or("resident device batch stride overflow")?;
        let extent=stride.checked_mul(chunks).ok_or("resident device batch extent overflow")?;
        let mut offset=0usize;let mut ranges=Vec::with_capacity(rows.len());
        for row in rows {let mut parts=Vec::new();for chunk in 0..if row.split {chunks} else {1} {
            let start=chunk.checked_mul(stride).and_then(|n|n.checked_add(offset)).ok_or("resident device batch source overflow")?;
            let end=start.checked_add(row.count).filter(|&n|n<=extent).ok_or("resident device batch source exceeds extent")?;parts.push(start..end);
        }ranges.push(parts);offset+=row.count;}
        Ok(Self {stride,chunks,ranges})
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn ranges_match_the_existing_rank_major_oracle_including_replicated_rows() {
        for chunks in 1..=5 {for jobs in 1..=9 {let rows=(0..jobs).map(|i|{let count=i+1;let split=chunks>1 && i%3!=0;Rows {start:0,count,total:if split {count*chunks} else {count},split}}).collect::<Vec<_>>();
            let p=Layout::new(&rows).unwrap();let values=(0..p.stride*p.chunks).map(|i|f32::from_bits(if i%7==0 {0x80000000} else {i as u32})).collect::<Vec<_>>();
            let expected=super::super::gathered_mixed(&values,&rows).unwrap();
            for (ranges,expected) in p.ranges.iter().zip(expected) {let bits=ranges.iter().flat_map(|r|values[r.clone()].iter().map(|v|v.to_bits())).collect::<Vec<_>>();assert_eq!(bits,expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());}
        }}
    }
    #[test] fn mailbox_gather_places_every_origin_for_one_two_and_four_ranks() {
        for parts in [1,2,4] {let rows=[Rows {start:0,count:1,total:parts,split:parts>1},Rows {start:0,count:2,total:2*parts,split:parts>1},Rows {start:0,count:1,total:1,split:false}];
            let layout=Layout::new(&rows).unwrap();let sources=(0..parts).map(|origin|vec![(origin+1)*10,origin*3+1,origin*3+2,99]).collect::<Vec<_>>();
            let expected=sources.iter().flatten().copied().collect::<Vec<_>>();
            for rank in 0..parts {let (own,remote)=layout.placement(rank,parts).unwrap();let mut packed=vec![0;expected.len()];
                packed[own..own+layout.stride].copy_from_slice(&sources[rank]);
                if parts>1 {use spark_gpu_probe::tp_route::{Channels,Links};let channels=Channels::new(&Links::new(parts,false).unwrap(),layout.stride).unwrap();
                    let mut wire=vec![Vec::new();parts-1];for &(origin,_) in &remote {wire[channels.channel(origin,rank)]=sources[origin].clone();}
                    // Reverse delivery order to expose accidental channel-zero or peer-offset use.
                    for &(origin,offset) in remote.iter().rev() {packed[offset..offset+layout.stride].copy_from_slice(&wire[channels.channel(origin,rank)]);}
                }
                assert_eq!(packed,expected,"parts={parts} rank={rank}");
                let values=packed.iter().map(|&v|v as f32).collect::<Vec<_>>();let oracle=super::super::gathered_mixed(&values,&rows).unwrap();
                for (ranges,oracle) in layout.ranges.iter().zip(oracle) {assert_eq!(ranges.iter().flat_map(|r|values[r.clone()].iter().copied()).collect::<Vec<_>>(),oracle);}
            }
            assert!(layout.placement(parts,parts).is_err());assert!(layout.placement(0,parts+1).is_err());
        }
    }
    #[test] fn scatter_matches_gathered_rows_for_every_origin_and_replication() {
        for chunks in 1..=5 {for jobs in 1..=19 {
            let rows=(0..jobs).map(|i|Rows {start:0,count:i+1,total:if i%3==0 {i+1} else {(i+1)*chunks},split:chunks>1 && i%3!=0}).collect::<Vec<_>>();
            let layout=Layout::new(&rows).unwrap();let source=(0..layout.stride*layout.chunks).map(|i|[0,0x80000000,0x7fc01234,i as u32][i%4]).collect::<Vec<_>>();
            let expected=layout.ranges.iter().flatten().flat_map(|r|source[r.clone()].iter().copied()).collect::<Vec<_>>();
            let mut actual=vec![0xdeadbeef;layout.output_words().unwrap()];let mut writes=vec![0;actual.len()];
            for origin in (0..layout.chunks).rev() {for [s,d,n] in layout.scatter(origin).unwrap() {actual[d..d+n].copy_from_slice(&source[origin*layout.stride+s..origin*layout.stride+s+n]);for count in &mut writes[d..d+n] {*count+=1;}}}
            assert_eq!(actual,expected);assert!(writes.iter().all(|&n|n==1));assert!(layout.scatter(layout.chunks).is_err());
        }}
    }
    #[test] fn empty_zero_mismatched_groups_and_overflow_are_refused() {

        assert!(Layout::new(&[]).is_err());
        for r in [Rows {start:0,count:0,total:1,split:true},Rows {start:0,count:2,total:3,split:true},Rows {start:0,count:1,total:2,split:false}] {assert!(Layout::new(&[r]).is_err());}
        assert!(Layout::new(&[Rows {start:0,count:1,total:2,split:true},Rows {start:0,count:1,total:3,split:true}]).is_err());
        assert!(Layout::new(&[Rows {start:0,count:usize::MAX,total:usize::MAX,split:false},Rows {start:0,count:1,total:1,split:false}]).is_err());
        assert!(Layout::new(&[Rows {start:0,count:usize::MAX/2+1,total:usize::MAX/2+1,split:false},Rows {start:0,count:1,total:2,split:true}]).is_err());
    }
}
