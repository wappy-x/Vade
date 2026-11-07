//! Checked raw-word placement. Each row is [source offset, target offset, words].
//! Target rows are disjoint; source rows may overlap because they are immutable.
#[derive(Debug)]
pub struct Plan {pub rows:Vec<[usize;3]>,pub longest:usize,pub words:usize}
/// CTAs per row of a scatter take. Grid-y multiplies this by the row count, so the budget
/// `most` of an ordinary take is shared over the rows. Fewer blocks change only how many
/// threads store a row, never which words reach which place.
pub fn take_blocks(longest:usize,rows:usize,most:usize)->u32 {longest.div_ceil(256).clamp(1,(most/rows.max(1)).max(1)) as u32}
/// A receive channel of a world of `world` ranks. Malformed channels return false, never overflow.
pub fn channel_inside(channel:usize,world:usize)->bool {channel.checked_add(1).is_some_and(|next|next<world)}
impl Plan {
    pub fn checked(source:usize,target:usize,rows:&[[usize;3]])->Result<Self,String> {
        if rows.is_empty() || rows.len()>65535 {return Err("scatter row count exceeds grid".into());}
        let mut ends=Vec::with_capacity(rows.len());let mut longest=0;let mut words=0usize;
        for &[from,to,n] in rows {
            if n==0 || n>u32::MAX as usize || from.checked_add(n).is_none_or(|end|end>source)
                || to.checked_add(n).is_none_or(|end|end>target) {return Err("scatter row exceeds storage".into());}
            ends.push((to,to+n));longest=longest.max(n);words=words.checked_add(n).ok_or("scatter extent overflow")?;
        }
        ends.sort_unstable();if ends.windows(2).any(|p|p[0].1>p[1].0) {return Err("scatter targets overlap".into());}
        Ok(Self {rows:rows.to_vec(),longest,words})
    }
    /// The copy grid is rows times the blocks of the longest row. Cut unequal rows
    /// as resident_join does, so idle blocks do not launch. The same words reach
    /// the same places, in the same source and target ranges.
    pub fn cut(self)->Result<Self,String> {
        let smallest=self.rows.iter().map(|row|row[2]).min().unwrap_or(self.longest);
        if self.longest==smallest {return Ok(self);}
        let mut size=smallest.div_ceil(128)*128;
        let chosen=loop {
            let count=self.rows.iter().map(|row|row[2].div_ceil(size)).sum::<usize>();
            if count<=8192 {break (size.div_ceil(128).saturating_mul(count)<self.longest.div_ceil(128).saturating_mul(self.rows.len())).then_some(size);}
            size=size.checked_mul(2).ok_or("scatter row size overflow")?;
        };
        let Some(size)=chosen else {return Ok(self);};
        let mut rows=Vec::new();let mut longest=0;
        for &[from,to,n] in &self.rows {let mut done=0;
            while done<n {let words=(n-done).min(size);rows.push([from+done,to+done,words]);longest=longest.max(words);done+=words;}
        }
        if rows.len()>65535 {return Err("scatter row count exceeds grid".into());}
        Ok(Self {rows,longest,words:self.words})
    }
    /// Encode only initialized numeric fields of HybridCopyRow, never host padding.
    pub fn encode(&self,source:u64,target:u64)->Result<Vec<u8>,String> {
        let address=|base:u64,offset:usize|u64::try_from(offset).ok().and_then(|v|v.checked_mul(4)).and_then(|v|base.checked_add(v)).ok_or_else(||"scatter address overflow".to_string());
        let mut bytes=Vec::with_capacity(self.rows.len()*24);
        for &[from,to,n] in &self.rows {
            address(source,from+n)?;address(target,to+n)?;
            bytes.extend_from_slice(&address(source,from)?.to_ne_bytes());bytes.extend_from_slice(&address(target,to)?.to_ne_bytes());
            bytes.extend_from_slice(&(n as u32).to_ne_bytes());bytes.extend_from_slice(&0u32.to_ne_bytes());
        }Ok(bytes)
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn disjoint_placements_preserve_bits_and_guards() {
        let src:[u32;6]=[0,0x80000000,0x7fc01234,0xffffffff,4,5];let p=Plan::checked(6,12,&[[3,1,3],[0,7,3]]).unwrap();
        let mut dst=[0xdeadbeef;12];for &[s,d,n] in &p.rows {dst[d..d+n].copy_from_slice(&src[s..s+n]);}
        assert_eq!(dst,[0xdeadbeef,0xffffffff,4,5,0xdeadbeef,0xdeadbeef,0xdeadbeef,0,0x80000000,0x7fc01234,0xdeadbeef,0xdeadbeef]);
        let bytes=p.encode(16,128).unwrap();assert_eq!(bytes.len(),48);assert_eq!(&bytes[..8],&28u64.to_ne_bytes());assert_eq!(&bytes[8..16],&132u64.to_ne_bytes());assert_eq!(&bytes[20..24],&[0;4]);
    }
    #[test] fn rejects_bounds_overlap_and_address_overflow() {
        for rows in [vec![],vec![[0,0,0]],vec![[0,0,4]],vec![[2,0,2]],vec![[0,2,2]],vec![[0,0,2],[0,1,1]],vec![[usize::MAX,0,1]],vec![[0,usize::MAX,1]]] {assert!(Plan::checked(3,3,&rows).is_err());}
        let p=Plan::checked(1,1,&[[0,0,1]]).unwrap();assert!(p.encode(u64::MAX-2,0).is_err());assert!(p.encode(0,u64::MAX-2).is_err());
    }
    #[test] fn channel_guard_refuses_every_outside_value_without_overflow() {
        assert!(channel_inside(0,2));assert!(channel_inside(2,4));assert!(!channel_inside(1,2));assert!(!channel_inside(3,4));assert!(!channel_inside(0,1));assert!(!channel_inside(0,0));
        assert!(!channel_inside(usize::MAX,2));assert!(!channel_inside(usize::MAX,usize::MAX));assert!(!channel_inside(usize::MAX-1,usize::MAX));
    }
    /// The eight gather families of the traced head, plus unequal shapes: a cut plan
    /// writes each target word once from the same source word, and keeps the checks.
    #[test] fn cut_keeps_every_word_and_lowers_idle_blocks() {
        for sizes in [vec![4096,4096,4096,64,64,32],vec![4096,4096],vec![6144,6144],vec![144,1024,1024],[vec![1024;16],vec![2048]].concat(),vec![2048;8],vec![768,256,64,64,16],vec![8192,2048],vec![1,129],vec![1,1,1,70000]] {
            let words=sizes.iter().sum::<usize>();let mut rows=Vec::new();let (mut from,mut to)=(0,3);
            for &n in sizes.iter().rev() {rows.push([words-from-n,to,n]);from+=n;to+=n+2;}
            let before=Plan::checked(words,to,&rows).unwrap();let blocks=before.longest.div_ceil(128)*before.rows.len();let total=before.words;
            let cut=before.cut().unwrap();assert_eq!(cut.words,total);
            assert!(cut.longest.div_ceil(128)*cut.rows.len()<=blocks);assert!(Plan::checked(words,to,&cut.rows).is_ok());
            let mut direct=vec![u64::MAX;to];for &[s,d,n] in &rows {for i in 0..n {direct[d+i]=(s+i) as u64;}}
            let mut viacut=vec![u64::MAX;to];for &[s,d,n] in &cut.rows {for i in 0..n {assert_eq!(viacut[d+i],u64::MAX);viacut[d+i]=(s+i) as u64;}}
            assert_eq!(direct,viacut);
        }
        let cut=Plan::checked(1168,1168,&[[0,0,768],[768,768,256],[1024,1024,64],[1088,1088,64],[1152,1152,16]]).unwrap().cut().unwrap();
        assert_eq!((cut.rows.len(),cut.longest),(11,128));
    }
    /// The kernel's copy loops, on the host: every word of each row is stored once, by
    /// any grid the host can choose, on the scalar and the four-word path. Failure of
    /// the acquire (a CTA that returns first) stores nothing, by the loop's position.
    #[test] fn take_scatter_addressing_covers_each_row_once_on_both_paths() {
        for most in [1,2,6,96] {for words in [4usize,16,64,144,256,1024,4096,8192,16384,24576,32768] {for longest in [words,words*2,words*5] {
            for rows in [1,2,3,5,6,8,17,219] {
            let blocks=take_blocks(longest,rows,most) as usize;assert!((1..=most).contains(&blocks));assert!(blocks*rows<=most.max(rows));let step=blocks*256;
            for quads in [false,true] {
                let mut stored=vec![0u32;words];
                for block in 0..blocks {for thread in 0..256 {let mut i=block*256+thread;
                    if quads {let q=words/4;while i<q {for k in 0..4 {stored[4*i+k]+=1;}i+=step;}}
                    else {while i<words {stored[i]+=1;i+=step;}}
                }}
                assert!(stored.iter().all(|&n|n==1),"most={most} words={words} longest={longest} rows={rows} quads={quads}");
            }
            }
        }}}
    }
    /// The command words of a scatter take are the five words the kernel adapter reads.
    #[test] fn take_scatter_kind_has_its_own_entry() {
        use crate::resident_launch::{Command,Kind};
        assert_eq!(Kind::TakeScatter as u32,22);assert_eq!(Kind::TakeScatter.kernel(),"resident_launch_take_scatter");
        let mut c=Command::new(Kind::TakeScatter);for i in 0..5 {c.word(i,i as u64+100);}
        for i in 0..5 {assert_eq!(u64::from_ne_bytes(c.payload[i*8..i*8+8].try_into().unwrap()),i as u64+100);}
    }
    /// The counter reads the word count where encode writes it.
    #[test] fn copy_counters_read_the_encoded_extent() {
        use crate::resident_launch::{CopyCounts,Kind};
        let c=crate::resident_launch_encode::copy(crate::kernel_params::HybridCopy {source:core::ptr::null(),destination:core::ptr::null_mut(),words:7,padding:0,rows:0});
        assert_eq!(c.kind,Kind::Copy as u32);assert_eq!(u32::from_ne_bytes(c.payload[16..20].try_into().unwrap()),7);
        let after=CopyCounts {launches:[3,2,1],bytes:[30,20,10],merged_zero_launches:1,merged_zero_bytes:8,..Default::default()};
        let before=CopyCounts {launches:[1,1,1],bytes:[10,10,10],merged_zero_launches:1,merged_zero_bytes:8,..Default::default()};
        assert_eq!(after.since(before).line(),"resident_launch_copy=2 resident_launch_copy_bytes=20 resident_launch_copy_words=1 resident_launch_copy_words_bytes=10 resident_launch_zero_words=0 resident_launch_zero_words_bytes=0 resident_merged_zero_launches=0 resident_merged_zero_bytes=0 stage_full_flushes=0 piece_evictions=0");
    }
}
