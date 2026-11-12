//! Deterministic compact expert bounds and token/rank destination maps.
//! First-party implementation of the ordered expert-binning idea in llama.cpp mmq.
use crate::fp8_abi::{GroupedRouteParams as Route,RoutingLayout,valid_route};
pub fn tasks(tokens: usize) -> usize { let pairs=tokens*10; pairs.div_ceil(128)+pairs.min(512) }
pub fn words(tokens: usize) -> usize { 512+513+3*tokens*10+1+2*tasks(tokens) }
/// # Safety
/// All Route spans are live with tokens*10 pairs, expert counts/starts and tasks()
/// capacity. Each token is validated once before count/prefix/order run.
#[inline(always)]
pub unsafe fn validate(p: Route,token: usize) {
    if p.failed(token) { return; }
    let x=p.pairs as *const f32;
    for slot in 0..10 { let at=(token*10+slot)*3;
        if *x.add(at)!=token as f32 || !valid_route(*x.add(at),*x.add(at+1),*x.add(at+2),p.tokens,p.experts)
            || (0..slot).any(|j| *x.add((token*10+j)*3+1)==*x.add(at+1)) {
            *(p.error as *mut u32).add(token)=1; return;
        }
    }
}
/// # Safety
/// As validate; the expert owns its count and compact inverse-map slice.
#[inline(always)]
pub unsafe fn count(p: Route,expert: usize) {
    let mut n=0;
    for pair in 0..p.pair_count as usize {
        n+=u32::from(!p.failed(pair/10) && *(p.pairs as *const f32).add(pair*3+1)==expert as f32);
    }
    *(p.counts as *mut u32).add(expert)=n;
}
/// # Safety
/// Counts are complete; one owner writes prefix bounds and ordered tasks.
#[inline(always)]
pub unsafe fn prefix(p: Route) {
    let (mut offset,mut task)=(0u32,0usize); let out=p.tasks as *mut u32;
    for expert in 0..p.experts as usize {
        *(p.starts as *mut u32).add(expert)=offset;
        let n=*(p.counts as *const u32).add(expert);
        for first in (0..n).step_by(128) { *out.add(1+task*2)=expert as u32; *out.add(2+task*2)=first; task+=1; }
        offset+=n;
    }
    *(p.starts as *mut u32).add(p.experts as usize)=offset; *out=task as u32;
}
/// # Safety
/// Validated counts/prefix are complete; one owner per expert scans pairs in
/// token/slot order. sorted_row gathers compact inputs; pair_to_row addresses
/// token/rank output slots with the shared eleventh slot reserved.
#[inline(always)]
pub unsafe fn order(p: Route,expert: usize) {
    let mut row=*(p.starts as *const u32).add(expert) as usize;
    let x=p.pairs as *const f32;
    for pair in 0..p.pair_count as usize {
        if p.failed(pair/10) || *x.add(pair*3+1)!=expert as f32 { continue; }
        *(p.sorted_row as *mut u32).add(row)=pair as u32;
        let rank=RoutingLayout::FLASH.ordered_slot(pair%10,|slot| *x.add((pair/10*10+slot)*3+1) as u32);
        *(p.token_pairs as *mut u32).add(pair/10*10+rank)=pair as u32;
        *(p.pair_to_row as *mut u32).add(pair)=(pair/10*11+rank) as u32;
        row+=1;
    }
}
/// # Safety
/// sorted has the compact live pair map. Tail rows return before any load.
#[inline(always)]
pub unsafe fn input_row(sorted: u64,start: usize,first: usize,local: usize,count: usize,middle: bool) -> Option<usize> {
    let row=first+local;
    if row>=count { None } else if middle { Some(start+row) }
        else { Some((*(sorted as *const u32).add(start+row)/10) as usize) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_bounds_inverse_and_tail_masks_are_invariant() {
        let plan=|ids: &[usize],reject: bool| {
            let n=ids.len(); let pairs: Vec<f32>=ids.iter().enumerate().flat_map(|(t,&id)| (0..10).flat_map(move |s|
                [t as f32,((id*17+s*41)%512) as f32,(s+1) as f32/55.0])).collect();
            let (mut flags,mut counts,mut starts)=(vec![0u32;n],vec![0u32;512],vec![0u32;513]);
            if reject { flags[n-1]=1; }
            let (mut sorted,mut inverse,mut token_pairs,mut tasks_out)=(vec![u32::MAX;n*10],vec![u32::MAX;n*10],vec![u32::MAX;n*10],vec![0u32;1+2*tasks(n)]);
            let p=Route { error:flags.as_mut_ptr() as u64,pairs:pairs.as_ptr() as u64,counts:counts.as_mut_ptr() as u64,
                starts:starts.as_mut_ptr() as u64,sorted_row:sorted.as_mut_ptr() as u64,pair_to_row:inverse.as_mut_ptr() as u64,
                token_pairs:token_pairs.as_mut_ptr() as u64,tasks:tasks_out.as_mut_ptr() as u64,token_count:0,output_row:0,
                tokens:n as u32,experts:512,pair_count:(n*10) as u32,row_capacity:(n*10) as u32 };
            unsafe { for t in 0..n { validate(p,t); } for e in 0..512 { count(p,e); } prefix(p); for e in 0..512 { order(p,e); } }
            assert_eq!(starts[512] as usize,(n-usize::from(reject))*10); assert!(tasks_out[0] as usize<=tasks(n));
            for e in 0..512 { let (start,m)=(starts[e] as usize,counts[e] as usize);
                assert_eq!(starts[e+1] as usize,start+m);
                for (r,&pair) in sorted[start..start+m].iter().enumerate() {
                    assert_eq!(pairs[pair as usize*3+1],e as f32);
                    let slot=inverse[pair as usize] as usize;
                    assert_eq!(slot/11,pair as usize/10);
                    assert_eq!(token_pairs[slot/11*10+slot%11],pair);
                    if r>0 { assert!(sorted[start+r-1]<pair); }
                }
                for first in (0..m.max(1)).step_by(16) { for local in 0..16 {
                    let got=unsafe { input_row(p.sorted_row,start,first,local,m,false) };
                    assert_eq!(got,(first+local<m).then(|| sorted[start+first+local] as usize/10));
                } }
            }
            let round=|v: f32| f32::from_bits(crate::next_math::bf16(v)<<16);
            (0..n-usize::from(reject)).map(|t| {
                let mut value=0.0;
                for rank in 0..10 { let pair=token_pairs[t*10+rank] as usize;
                    let down=round((ids[t] as f32*0.3+pairs[pair*3+1]*0.05)*1.003);
                    value=crate::next_math::routed_add(value,down,pairs[pair*3+2]); }
                value.to_bits()
            }).collect::<Vec<_>>()
        };
        let ids: Vec<_>=(0..513).map(|i| i%37).collect(); let expected=plan(&ids,false);
        for chunk in [1,2,7,16,63,64,65,127,512] { assert_eq!(ids.chunks(chunk).flat_map(|c| plan(c,false)).collect::<Vec<_>>(),expected); }
        for width in 1..=16 { let reversed: Vec<_>=ids[..width].iter().rev().copied().collect();
            assert_eq!(plan(&reversed,false),expected[..width].iter().rev().copied().collect::<Vec<_>>());
            assert_eq!(plan(&ids[..width],true),expected[..width-1]);
        }
    }
}
