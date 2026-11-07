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

#[cfg(test)]
mod tests {
    use super::*;
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
        // Astra's train 16 review: dim 3 with 4 kept pools (taps 12), and any other unswept pair, keep the generic kernel.
        let mut review=shape(3,12,0,12);review.tokens=1;review.rows=10;review.first=30;assert!(!admitted(review));
        for (d,pools) in [(3usize,512usize),(5,1),(6,2),(7,3),(4,4),(4,511),(4,5),(2,100),(8,513)] {assert!(!admitted(shape(d as u32,(d*pools) as u32,0,(d*pools) as u32)),"d={d} pools={pools}");}
        for d in [1usize,2,4,8] {for pools in [1usize,2,3,512] {assert!(admitted(shape(d as u32,(d*pools) as u32,0,(d*pools) as u32)),"d={d} pools={pools}");}}
    }
}
