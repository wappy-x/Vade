//! Admission and launch geometry of the small pool scores kernel `prompt_pool_scores` (`Op::PoolScoresRows`,
//! `kernels/src/prompt_fast.rs`): a warp for each (token, pool) word, one lane a head. For the 1 to 16 token rows of a
//! draft verify pass or a step of several chats. There `pool_scores_tile` wastes its 64-token block, and `prompt_rows`
//! walks every head's chain in one thread (32 x 128 adds, about 340 us a launch on one box). Same bits: each head's
//! chain in key order, then the heads added in head order from zero.
//!
//! One file, two readers: `admitted` takes a shape ONLY when its key dim, head count and pool size are listed below,
//! and the device sweep (`kernel-bench --pool-scores`, which includes this file by path) runs every listed value
//! against `prompt_rows`, every token count from 1 to MAX_TOKENS, the pool counts at the listed edges and the causal
//! context (`first`) at both sides of the pool boundaries. So the admitted set cannot grow without the sweep growing.
pub const THREADS: u32 = 128;
/// (token, pool) words of one block: a warp each.
pub const WORDS: usize = THREADS as usize / 32;
/// The token rows of a verify pass or a step of several chats (`prompt_writes::SMALL`). More tokens take the tile.
pub const MAX_TOKENS: usize = 16;
/// Pools of one launch: the model's largest context (32,768 tokens) in pools of 4 is 8,192.
pub const MAX_ROWS: usize = 16384;
/// Key dims swept: the model's 128, counts that are not whole vectors of 4, and powers of two to 256.
pub const DIMS: &[usize] = &[1, 4, 8, 12, 16, 32, 64, 127, 128, 132, 256];
/// Head counts swept: the model's 32 (one round of lanes), fewer, and more than one round (33, 64).
pub const HEADS: &[usize] = &[1, 2, 3, 4, 8, 16, 31, 32, 33, 64];
/// Pool sizes swept (`taps`: tokens a pool). The model's is 4. They only move which pools are complete at a token.
pub const TAPS: &[usize] = &[1, 2, 3, 4, 5, 8, 16, 64];
/// Pool counts the sweep runs: both ends, both sides of a block of WORDS, and the pools of the model's contexts.
pub const ROW_EDGES: &[usize] = &[1, 2, 3, 4, 5, 7, 8, 9, 63, 64, 65, 127, 128, 129, 1000, 2047, 2048, 2049, 8191, 8192, 16383, 16384];
/// True when `prompt_pool_scores` may run this shape.
pub fn admitted(rows: usize, tokens: usize, heads: usize, dim: usize, taps: usize) -> bool {
    (1..=MAX_ROWS).contains(&rows) && (1..=MAX_TOKENS).contains(&tokens) && HEADS.contains(&heads) && DIMS.contains(&dim) && TAPS.contains(&taps)
}
/// Blocks for the launch: WORDS words a block.
pub fn blocks(rows: usize, tokens: usize) -> Option<u32> { u32::try_from(rows.checked_mul(tokens)?.div_ceil(WORDS)).ok() }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_model_shape_is_admitted_and_other_shapes_are_not() {
        // The model: 32 index heads of 128, pools of 4. A 4-row check at a context of 356 tokens scores 89 pools.
        assert!(admitted(89, 4, 32, 128, 4) && admitted(64, 1, 32, 128, 4) && admitted(8192, 16, 32, 128, 4) && admitted(16384, 16, 64, 256, 64));
        assert!(!admitted(89, 0, 32, 128, 4) && !admitted(89, 17, 32, 128, 4) && !admitted(0, 4, 32, 128, 4) && !admitted(16385, 4, 32, 128, 4));
        assert!(!admitted(89, 4, 5, 128, 4) && !admitted(89, 4, 32, 100, 4) && !admitted(89, 4, 32, 128, 6));
        assert_eq!(MAX_TOKENS, crate::prompt_writes::SMALL as usize);
        assert_eq!((blocks(89, 4), blocks(1, 1), blocks(5, 1), blocks(16384, 16)), (Some(89), Some(1), Some(2), Some(65536)));
    }
}
