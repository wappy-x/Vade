//! Which `SelectRows` shapes take the block-per-token selection kernel (`kernels/src/select_fast.rs`). By shape only.
//! Any other shape keeps `prompt_rows`. The kernel computes the same bytes (the order is total), so this rule
//! only keeps the shapes the kernel was written for: complete pool budget, room for every id, tail room.
use crate::prompt_ops::{Op,Shape};

/// Threads of the selection block.
pub const THREADS:u32=1024;
pub fn admitted(s:Shape)->bool {
    let (d,t)=(s.dim as usize,s.taps as usize);
    s.op==Op::SelectRows && d>0 && t>0 && t%d==0 && s.cols<=1 && s.tokens>0
        && s.slots as usize >= t+if s.cols!=0 {d-1} else {0}
        // The swept envelope (kernel-bench --causal-probe select-bytes). Positions per pool and kept pools are structure: only the
        // swept values are admitted (positions 1, 2, 4, 8; 1, 2, 3 and 512 kept pools, every pair). Row counts (up to 32,768 pool
        // scores per token) and tokens (up to 256 a launch) are intervals whose edges were swept. Beyond it prompt_rows keeps the work.
        && matches!(d,1|2|4|8) && matches!(t/d,1|2|3|512) && s.rows<=32768 && s.tokens<=256
        // The old heap stores a winning pool id as f32 and casts it back (prompt_ops.rs select_top), and the output holds expanded ids as f32.
        // Every pool id and expanded id is an exact f32 integer only below 2^24, so the contexts of the token block stay below it; beyond
        // that the original kernel keeps the work and its rounding.
        && (s.first as u64)+(s.tokens as u64)<=(1u64<<24)
}

/// The whole output row of a selection that keeps every complete pool: the ids and the
/// count follow from the context alone, for any finite or infinite scores (`select_all`).
/// A NaN score keeps these ids and writes count -2.0, so a caller that uses this row must
/// still compare the device's count word before any result leaves. None: selection is real.
pub fn known_selection(s:Shape)->Option<Vec<f32>> {
    if s.tokens!=1 {return None;}
    known_selection_rows(s)
}

/// Multi-row counterpart of `known_selection`: when even the last token of `s` keeps every
/// complete pool (`t / d >= (first + tokens - 1) / d`), every token's selected IDs and count
/// follow from the context alone for any finite or infinite scores (`select_all`). A NaN score
/// writes count `-2.0` for that token, so a caller that uses these rows must still compare
/// every token's count word on the device before any result leaves.
pub fn known_selection_rows(s:Shape)->Option<Vec<f32>> {
    let (m,d,t,first,slots)=(s.tokens as usize,s.dim as usize,s.taps as usize,s.first as usize,s.slots as usize);
    if s.op!=Op::SelectRows || m==0 || d==0 || t==0 || t%d!=0 || s.cols>1 {return None;}
    let last=first.checked_add(m-1)?;if last>=1<<24 {return None;}
    let last_groups=last/d;if t/d<last_groups || s.rows as usize!=last_groups {return None;}
    let words=slots.checked_add(1)?;let mut out=vec![-1.0;m.checked_mul(words)?];
    for token in 0..m {
        let context=first+token;let groups=context/d;
        let count=if s.cols!=0 {context} else {groups*d};if count>slots {return None;}
        let row=&mut out[token*words..(token+1)*words];
        for (id,word) in row[..count].iter_mut().enumerate() {*word=id as f32;}
        row[slots]=count as f32;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn glm(context:usize)->Shape {glm_rows(context,1)}
    fn glm_rows(first:usize,m:usize)->Shape {
        let mut s=Shape::new(Op::SelectRows,m).unwrap();
        s.dim=4;s.taps=2048;s.cols=1;s.slots=2051;s.first=first as u32;s.rows=((first+m-1)/4) as u32;s
    }
    fn scores(kind:usize,n:usize,seed:&mut u64)->Vec<f32> {(0..n).map(|i|{*seed^=*seed<<13;*seed^=*seed>>7;*seed^=*seed<<17;let r=*seed;match kind {
        0=>f32::from_bits(r as u32&0x7f7fffff|((r>>40&1) as u32)<<31),1=>0.5,2=>(r%3) as f32,3=>if i%2==0 {0.0} else {-0.0},
        4=>if i%2==0 {f32::INFINITY} else {f32::NEG_INFINITY},5=>f32::from_bits((r as u32)&0x007fffff),_=>if i%2==0 {f32::MAX} else {f32::MIN},}}).collect()}
    fn bits(v:&[f32])->Vec<u32> {v.iter().map(|x|x.to_bits()).collect()}
    /// The kernel's bytes are pinned to `reference` by the device sweep; this pins the prediction to it.
    #[test]
    fn known_row_equals_the_selection_up_to_2051_and_stops_at_2052() {
        let mut seed=0x9e3779b97f4a7c15u64;
        for context in (1..=64).chain(2040..=2051) {for kind in 0..7 {for _ in 0..4 {
            let s=glm(context);let x=scores(kind,context/4,&mut seed);
            let got=crate::prompt_ops::reference(s,[&x,&[],&[],&[],&[]],&[]).unwrap();
            let known=known_selection(s).unwrap_or_else(||panic!("context {context}"));assert_eq!(bits(&got),bits(&known),"context={context} kind={kind}");
            assert_eq!(known[2051],context as f32);if context>=1 {assert_eq!(known[context-1],(context-1) as f32);}
        }}}
        for context in [2052,2053,2055,2056,3000,4158] {assert!(known_selection(glm(context)).is_none(),"context {context}");}
        // At 2052 selection is real: the worst pool leaves, so its ids differ from 0..2052.
        let s=glm(2052);let x=(0..513).map(|i|i as f32).collect::<Vec<_>>();let got=crate::prompt_ops::reference(s,[&x,&[],&[],&[],&[]],&[]).unwrap();
        assert_eq!(got[0],4.0);assert_eq!(got[2051],2048.0);
    }
    #[test]
    fn known_rows_equal_multi_row_selection_up_to_2051_and_catch_nan_counts() {
        let mut seed=0xd1b54a32d192ed03u64;
        for m in 1..=4 {
            for first in (1..=20).chain(2044..=2052-m) {
                for kind in 0..7 {
                    let s=glm_rows(first,m);let x=scores(kind,m*s.rows as usize,&mut seed);
                    let got=crate::prompt_ops::reference(s,[&x,&[],&[],&[],&[]],&[]).unwrap();
                    let known=known_selection_rows(s).unwrap_or_else(||panic!("first={first} m={m}"));
                    assert_eq!(bits(&got),bits(&known),"first={first} m={m} kind={kind}");
                }
            }
            for first in [2053-m,2056,4158] {
                assert!(known_selection_rows(glm_rows(first,m)).is_none(),"first={first} m={m}");
            }
        }
        // Any token whose pool score has NaN writes count -2.0 for that token while keeping IDs.
        for m in 2..=4 {
            let s=glm_rows(64,m);let r=s.rows as usize;
            for bad_token in 0..m {
                let mut x=vec![0.25;m*r];x[bad_token*r+r/2]=f32::NAN;
                let got=crate::prompt_ops::reference(s,[&x,&[],&[],&[],&[]],&[]).unwrap();
                let known=known_selection_rows(s).unwrap();
                for token in 0..m {
                    let at=token*2052+2051;
                    if token==bad_token {assert_eq!(got[at],-2.0);assert_ne!(got[at].to_bits(),known[at].to_bits());}
                    else {assert_eq!(got[at].to_bits(),known[at].to_bits());}
                }
            }
        }
    }
    #[test]
    fn a_nan_score_keeps_the_known_ids_and_writes_count_minus_two() {
        for context in [4,5,1040,2048,2051] {for at in [0,context/8,context/4-1] {
            let s=glm(context);let mut x=vec![0.25;context/4];x[at]=f32::NAN;
            let got=crate::prompt_ops::reference(s,[&x,&[],&[],&[],&[]],&[]).unwrap();let known=known_selection(s).unwrap();
            assert_eq!(got[2051],-2.0);assert_ne!(known[2051].to_bits(),got[2051].to_bits());assert_eq!(bits(&got[..2051]),bits(&known[..2051]));
        }}
    }
    #[test]
    fn known_row_refuses_other_shapes() {
        let mut s=glm(100);s.op=Op::ScoresRows;assert!(known_selection(s).is_none());
        let mut s=glm(100);s.tokens=2;assert!(known_selection(s).is_none());
        let mut s=glm(100);s.rows=24;assert!(known_selection(s).is_none());assert!(known_selection_rows(s).is_none());
        let mut s=glm(100);s.taps=2047;assert!(known_selection(s).is_none());assert!(known_selection_rows(s).is_none());
        let mut s=glm(100);s.dim=0;assert!(known_selection(s).is_none());assert!(known_selection_rows(s).is_none());
        let mut s=glm(100);s.cols=2;assert!(known_selection(s).is_none());assert!(known_selection_rows(s).is_none());
        let mut s=glm(100);s.slots=99;assert!(known_selection(s).is_none());assert!(known_selection_rows(s).is_none());
        let mut s=glm(100);s.cols=0;s.slots=2048;let row=known_selection(s).unwrap();assert_eq!(row[2048],100.0);assert_eq!(row[100],-1.0);
        let mut s=glm(103);s.cols=0;s.slots=2048;s.rows=25;let row=known_selection(s).unwrap();assert_eq!(row[2048],100.0);
    }
    fn shape(dim:u32,taps:u32,cols:u32,slots:u32)->Shape {let mut s=Shape::new(Op::SelectRows,1).unwrap();s.dim=dim;s.taps=taps;s.cols=cols;s.slots=slots;s.rows=1000;s.first=4000;s}
    #[test]
    fn real_selection_shapes_are_admitted_and_odd_ones_keep_the_generic_kernel() {
        assert!(admitted(shape(4,2048,1,2051)) && admitted(shape(4,2048,0,2048)) && admitted(shape(4,2048,0,3000)));
        assert!(!admitted(shape(4,2048,1,2050)) && !admitted(shape(4,2046,0,2048)) && !admitted(shape(0,2048,0,2048)) && !admitted(shape(4,0,0,2048)));
        assert!(!admitted(shape(4,2048,2,2051)));
        let mut other=shape(4,2048,1,2051);other.op=Op::ScoresRows;assert!(!admitted(other));
        let mut none=shape(4,2048,1,2051);none.tokens=0;assert!(!admitted(none));
        let mut many=shape(4,2048,1,2051);many.tokens=256;assert!(admitted(many));many.tokens=257;assert!(!admitted(many));
        let mut wide=shape(4,2048,1,2051);wide.rows=32768;assert!(admitted(wide));wide.rows=32769;assert!(!admitted(wide));
        // The review's counterexample: ids above 2^24 round in the old kernel, so it must keep them.
        let mut big=shape(3,3,0,3);big.tokens=1;big.rows=16777218;big.first=50331654;assert!(!admitted(big));
        let mut edge=shape(4,2048,1,2051);edge.tokens=1;edge.first=(1<<24)-1;assert!(admitted(edge));edge.first=1<<24;assert!(!admitted(edge));
        let mut edge3=shape(4,2048,1,2051);edge3.tokens=3;edge3.first=(1<<24)-3;assert!(admitted(edge3));edge3.first=(1<<24)-2;assert!(!admitted(edge3));
        assert!(!admitted(shape(16,2048,0,2048)) && admitted(shape(8,4096,0,4096)) && !admitted(shape(4,2052,0,2052)) && admitted(shape(4,2048,0,2048)) && !admitted(shape(1,513,0,513)) && admitted(shape(1,512,0,512)));
        // Review: dim 3 with 4 kept pools (taps 12), and any other unswept pair, keep the generic kernel.
        let mut review=shape(3,12,0,12);review.tokens=1;review.rows=10;review.first=30;assert!(!admitted(review));
        for (d,pools) in [(3usize,512usize),(5,1),(6,2),(7,3),(4,4),(4,511),(4,5),(2,100),(8,513)] {assert!(!admitted(shape(d as u32,(d*pools) as u32,0,(d*pools) as u32)),"d={d} pools={pools}");}
        for d in [1usize,2,4,8] {for pools in [1usize,2,3,512] {assert!(admitted(shape(d as u32,(d*pools) as u32,0,(d*pools) as u32)),"d={d} pools={pools}");}}
    }
}
