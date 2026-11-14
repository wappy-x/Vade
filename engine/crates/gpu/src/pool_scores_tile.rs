//! Admission and launch geometry for the tiled pool scores kernel `pool_scores_tile` (`Op::PoolScoresRows`).
//! Shape and alignment only. Any other shape runs on `prompt_rows`, with the same bits.
//!
//! One file, two readers: `admitted` takes a shape ONLY when its key dim, head count and pool size are listed below,
//! and the device sweep (`kernel-bench --pool-scores`, which includes this file by path) runs every listed value
//! against `prompt_rows`, with the pool and token counts at every tile edge of their intervals and at the interval
//! ends, and the causal context (`first`) at both sides of the pool boundaries. So the admitted set cannot grow
//! without the sweep growing: the host test walks every value up to past the largest listed one.
pub const THREADS: u32 = 256;
const TILE: usize = 64;
/// Pools of one launch: the model's largest context (32,768 tokens) in pools of 4 is 8,192.
pub const MAX_ROWS: usize = 16384;
/// The prompt group cap. Device checks reach this limit.
pub const MAX_TOKENS: usize = 1024;
/// Fewer tokens keep `prompt_rows`: a verify pass and a step of several chats have 1 to 16 token rows
/// (`prompt_writes::SMALL`), and a block of this kernel works 64 token columns whatever the count, so few tokens
/// waste most of it. With this floor no decode pass changes its launch; only prompt groups take the tile.
pub const MIN_TOKENS: usize = 17;
/// Fewer (pool x token) words keep `prompt_rows` too. A launch of the tile costs about 390 us whatever it holds (32 heads
/// of staged chunks in a block); the row kernel is faster below about 6,000 words. MEASURED on one box, one launch:
/// 31 pools x 127 tokens (a 127-token prompt) 339 us on the row kernel, 391 us on the tile; 64 x 64: 344 against 392 us;
/// 260 x 62: 658 against 396 us; 259 x 127: 2,026 against 395 us. A group of 1,024 tokens passes the floor from 8 pools on.
pub const MIN_WORDS: usize = 8192;
/// Key dims swept: the model's 128, every remainder of the 32-column chunk, and powers of two to 256.
pub const DIMS: &[usize] = &[4, 8, 12, 16, 20, 24, 28, 32, 36, 60, 64, 68, 96, 128, 132, 192, 256];
/// Head counts swept: the model's 32, and small counts.
pub const HEADS: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 64];
/// Pool sizes swept (`taps`: tokens a pool). The model's is 4. They only move which pools are complete at a token.
pub const TAPS: &[usize] = &[1, 2, 3, 4, 5, 8, 16, 64];
/// Pool and token counts the sweep runs: both ends of each interval and both sides of tile edges.
pub const ROW_EDGES: &[usize] = &[1, 2, 3, 63, 64, 65, 127, 128, 129, 1000, 2047, 2048, 2049, 8191, 8192, 16383, 16384];
pub const TOKEN_EDGES: &[usize] = &[17, 18, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024];
/// True when `pool_scores_tile` may run this shape. `queries` and `keys` are device addresses.
pub fn admitted(rows: usize, tokens: usize, heads: usize, dim: usize, taps: usize, queries: u64, keys: u64) -> bool {
    (1..=MAX_ROWS).contains(&rows) && (MIN_TOKENS..=MAX_TOKENS).contains(&tokens) && rows * tokens >= MIN_WORDS && HEADS.contains(&heads) && DIMS.contains(&dim) && TAPS.contains(&taps)
        && (queries | keys) & 15 == 0
}
/// Blocks for the launch: token tiles fastest, so neighbouring blocks share key lines.
pub fn blocks(rows: usize, tokens: usize) -> Option<u32> { u32::try_from(rows.div_ceil(TILE).checked_mul(tokens.div_ceil(TILE))?).ok() }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_shape_and_edges_are_admitted() {
        // The model: 32 index heads of 128, pools of 4; a group of 1,024 tokens at a context of 8,192 scores 2,048 pools.
        assert!(admitted(2048, 1024, 32, 128, 4, 256, 512) && admitted(8192, 1024, 32, 128, 4, 16, 32) && admitted(482, 17, 1, 4, 1, 16, 32) && admitted(16384, 1024, 64, 256, 64, 0, 0));
        // The work floor: a 127-token prompt (31 pools) and other small groups keep the row kernel; a full group passes it from 8 pools on.
        assert!(!admitted(31, 127, 32, 128, 4, 256, 512) && !admitted(64, 64, 32, 128, 4, 256, 512) && !admitted(481, 17, 32, 128, 4, 256, 512) && !admitted(7, 1024, 32, 128, 4, 256, 512));
        assert!(admitted(8, 1024, 32, 128, 4, 256, 512) && admitted(260, 62, 32, 128, 4, 256, 512) && admitted(128, 64, 32, 128, 4, 256, 512) && admitted(1039, 62, 32, 128, 4, 256, 512));
        // A verify pass or a step of several chats (1 to 16 token rows) keeps the row kernel.
        for tokens in 1..=16 { assert!(!admitted(2048, tokens, 32, 128, 4, 256, 512) && !admitted(16384, tokens, 32, 128, 4, 256, 512), "tokens {tokens}"); }
        assert_eq!(MIN_TOKENS, crate::prompt_writes::SMALL as usize + 1);
        assert_eq!(blocks(2048, 1024), Some(512));
        assert_eq!(blocks(8192, 1024), Some(2048));
        assert_eq!(blocks(1, 1), Some(1));
        assert_eq!(blocks(65, 64), Some(2));
        assert_eq!(blocks(64, 65), Some(2));
    }
    #[test]
    fn unaligned_or_other_shapes_keep_the_row_kernel() {
        assert!(admitted(8, 1024, 1, 4, 4, 0, 0) && !admitted(8, 1025, 1, 4, 4, 0, 0) && admitted(1000, 17, 1, 4, 4, 0, 0) && !admitted(1000, 16, 1, 4, 4, 0, 0));
        assert!(!admitted(0, 64, 1, 4, 4, 0, 0) && !admitted(1000, 0, 1, 4, 4, 0, 0) && !admitted(1000, 64, 0, 4, 4, 0, 0));
        assert!(!admitted(1000, 64, 1, 6, 4, 0, 0) && !admitted(1000, 64, 1, 4, 0, 0, 0) && !admitted(16385, 64, 1, 4, 4, 0, 0) && !admitted(1000, 64, 1, 0, 4, 0, 0));
        assert!(admitted(1000, 64, 1, 4, 4, 0, 0) && !admitted(1000, 64, 1, 4, 4, 4, 0) && !admitted(1000, 64, 1, 4, 4, 0, 8));
    }
    #[test]
    fn admits_only_the_swept_lists_and_intervals() {
        // Every dim, head count and pool size from 0 past the largest listed value: admitted only if listed.
        for dim in 0..=DIMS[DIMS.len() - 1] + 64 { assert_eq!(admitted(130, 65, 2, dim, 4, 0, 0), DIMS.contains(&dim), "dim {dim}"); }
        for heads in 0..=HEADS[HEADS.len() - 1] + 64 { assert_eq!(admitted(130, 65, heads, 4, 4, 0, 0), HEADS.contains(&heads), "heads {heads}"); }
        for taps in 0..=TAPS[TAPS.len() - 1] + 64 { assert_eq!(admitted(130, 65, 2, 4, taps, 0, 0), TAPS.contains(&taps), "taps {taps}"); }
        // The work floor is exact: one word under it keeps the row kernel.
        assert!(admitted(128, 64, 2, 4, 4, 0, 0) && !admitted(127, 64, 2, 4, 4, 0, 0) && !admitted(481, 17, 2, 4, 4, 0, 0) && admitted(482, 17, 2, 4, 4, 0, 0));
        assert!(DIMS.iter().all(|d| d % 4 == 0) && ROW_EDGES.iter().all(|&r| r <= MAX_ROWS) && TOKEN_EDGES.iter().all(|&t| t <= MAX_TOKENS));
        assert!(ROW_EDGES.contains(&MAX_ROWS) && TOKEN_EDGES.contains(&MAX_TOKENS) && ROW_EDGES.contains(&1) && TOKEN_EDGES.contains(&MIN_TOKENS));
        assert!(TOKEN_EDGES.iter().all(|&t| (MIN_TOKENS..=MAX_TOKENS).contains(&t)));
    }
}
