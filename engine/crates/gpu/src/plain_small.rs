//! Admission and launch geometry for the small-count plain token rows kernels `plain_rows_small` (1 to 16 tokens) and `plain_rows_few` (1 to 4 tokens, at
//! most 512 rows: 8 rows a block, one warp for each token).
//! Shape and alignment only. The admitted shapes are exactly the lists of `small_sweep` (rows, columns, 1 to 16 tokens), which the
//! device sweep (`kernel-bench --rows-small`) walks in full against the head's `prompt_rows`; any other shape keeps the tile kernel
//! or `prompt_rows`.
use crate::small_sweep::{MAX_TOKENS, PLAIN_COLS, PLAIN_FEW_MAX_ROWS, PLAIN_FEW_MAX_TOKENS, PLAIN_ROWS};
/// Tokens a thread chains at once.
pub const TOKENS_PER_WARP: usize = 4;
/// Rows one block covers: one row for each lane of a warp.
pub const ROWS_PER_BLOCK: usize = 32;
/// True when `plain_rows_small` may replace the other plain kernels for output `[rows, tokens]` over `cols` columns.
/// `input` and `weight` are the device addresses of the two operands.
pub fn admitted(rows: usize, tokens: usize, cols: usize, input: u64, weight: u64) -> bool {
    (1..=MAX_TOKENS).contains(&tokens) && PLAIN_ROWS.contains(&rows) && PLAIN_COLS.contains(&cols) && (input | weight) & 15 == 0
}
/// Rows one `plain_rows_few` block covers.
pub const FEW_ROWS_PER_BLOCK: usize = 8;
/// True when `plain_rows_few` runs the shape: it is admitted for `plain_rows_small` and has at most 4 tokens and 512 rows.
pub fn few_admitted(rows: usize, tokens: usize, cols: usize, input: u64, weight: u64) -> bool {
    admitted(rows, tokens, cols, input, weight) && tokens <= PLAIN_FEW_MAX_TOKENS && rows <= PLAIN_FEW_MAX_ROWS
}
/// Blocks for a `plain_rows_few` launch (128 threads each).
pub fn few_blocks(rows: usize) -> Option<u32> { u32::try_from(rows.div_ceil(FEW_ROWS_PER_BLOCK)).ok() }
/// Threads per block: four warps. A warp computes one group of 4 tokens; the others only copy.
pub fn threads(_tokens: usize) -> u32 { 128 }
/// Blocks for the launch: 32 weight rows each.
pub fn blocks(rows: usize) -> Option<u32> { u32::try_from(rows.div_ceil(ROWS_PER_BLOCK)).ok() }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_model_plain_matrix_is_admitted() {
        // rank-local and logical rows x columns of the plain matrices (lane A's WEIGHT-BYTES): gate, f_a, g_a, b_proj, f_b, g_b, wq_b, wk, indexer, weights_proj, kv_b.
        for (rows, cols) in [(288, 4096), (144, 4096), (128, 4096), (64, 4096), (32, 4096), (8192, 128), (4096, 128), (4096, 1536), (2048, 1536), (16, 4096), (16384, 512)] {
            for tokens in [1, 2, 4, 15, 16] { assert!(admitted(rows, tokens, cols, 256, 512), "{rows}x{cols} tokens {tokens}"); }
        }
    }
    #[test]
    fn the_admitted_set_is_the_sweep_set() {
        // Every dimension is scanned over its whole range with the other two at a listed value: nothing outside the lists is admitted.
        for rows in 0..=20000 { assert_eq!(admitted(rows, 4, 512, 16, 32), PLAIN_ROWS.contains(&rows), "rows {rows}"); }
        for cols in 0..=9000 { assert_eq!(admitted(64, 4, cols, 16, 32), PLAIN_COLS.contains(&cols), "cols {cols}"); }
        for tokens in 0..=20 { assert_eq!(admitted(64, tokens, 512, 16, 32), (1..=16).contains(&tokens), "tokens {tokens}"); }
        // The reviewer's examples, and the boundaries of each list.
        assert!(admitted(1, 16, 4088, 16, 32) && !admitted(1, 16, 4084, 16, 32) && !admitted(1, 16, 4092, 16, 32) && !admitted(1000 + 1, 4, 512, 16, 32));
        assert!(!admitted(16385, 4, 512, 16, 32) && !admitted(64, 17, 512, 16, 32) && !admitted(64, 4, 4100, 16, 32) && !admitted(64, 4, 510, 16, 32));
        assert!(!admitted(64, 4, 512, 4, 32) && !admitted(64, 4, 512, 16, 8));
    }
    #[test]
    fn few_admits_exactly_the_swept_product() {
        // The device sweep runs every listed (rows, cols) with 1 to 4 tokens and at most 512 rows through `plain_rows_few`.
        for rows in 0..=20000 { assert_eq!(few_admitted(rows, 3, 512, 16, 32), PLAIN_ROWS.contains(&rows) && rows <= PLAIN_FEW_MAX_ROWS, "rows {rows}"); }
        for cols in 0..=9000 { assert_eq!(few_admitted(64, 3, cols, 16, 32), PLAIN_COLS.contains(&cols), "cols {cols}"); }
        for tokens in 0..=20 { assert_eq!(few_admitted(64, tokens, 512, 16, 32), (1..=PLAIN_FEW_MAX_TOKENS).contains(&tokens), "tokens {tokens}"); }
        assert!(!few_admitted(64, 3, 512, 4, 32) && !few_admitted(64, 3, 512, 16, 8));
        assert_eq!([1, 8, 9, 64, 512].map(|r| few_blocks(r).unwrap()), [1, 1, 2, 8, 64]);
        // Every few shape is also a plain_rows_small shape (the order in small_choice decides which one runs).
        for rows in PLAIN_ROWS { for tokens in 1..=MAX_TOKENS { assert!(!few_admitted(*rows, tokens, 512, 16, 32) || admitted(*rows, tokens, 512, 16, 32)); } }
    }
    #[test]
    fn geometry_counts_warps_and_blocks() {
        assert_eq!([1, 4, 5, 8, 9, 16].map(threads), [128; 6]);
        assert_eq!([1, 32, 33, 64, 16384].map(|r| blocks(r).unwrap()), [1, 1, 2, 2, 512]);
    }
}
