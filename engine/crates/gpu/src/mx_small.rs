//! Admission and entry choice for the small-count MXFP8 rows kernels `mx_rows_small2/4/8/8w` (a draft verify pass, a batch of chats):
//! 2 to 8 tokens of one projection on the one-token kernel's load schedule (kernels/src/mx_rows_small.rs). The admitted calls are the
//! lists of `small_sweep` (rows, columns, 2 to 8 tokens) with the alignment of `mx_rows::admitted`; the device sweep
//! (`kernel-bench --mx-small`) walks every listed (rows, columns) at every token count against the production M kernels, and
//! against the kernels these entries replaced. Any other call keeps `mx_rows_tile` or the old kernels.
use crate::small_sweep::{MX_COLS, MX_MAX_BYTES, MX_MAX_TOKENS, MX_MIN_TOKENS, MX_ROWS};
/// True when this MXFP8 projection of `tokens` rows may use the small entries (`weights`, `input`, `output` are device addresses).
pub fn admitted(rows: usize, cols: usize, tokens: usize, weights: u64, input: u64, output: u64) -> bool {
    (MX_MIN_TOKENS..=MX_MAX_TOKENS).contains(&tokens) && MX_ROWS.contains(&rows) && MX_COLS.contains(&cols) && rows * cols <= MX_MAX_BYTES && weights & 15 == 0 && input & 15 == 0 && output & 3 == 0
}
/// Past this many columns a call of 5 to 8 tokens takes the entry with 4 weight rows a warp (`mx_rows_small8w`), else the one
/// with 2 (`mx_rows_small8`). The split comes from a sweep of test variants on one Spark (2026-11-12; the variants are not in the
/// tree and `kernel-bench --mx-small time` does not time the two forms against each other): 4 rows a warp read the 8,192 and
/// 12,288-column matrices faster, 2 rows a warp the 4,096-column ones. Not measured again with the entries as they are now.
pub const WIDE_COLS: usize = 4096;
/// The entry name, the tokens a block takes and the weight rows a warp owns, for a call of `tokens` tokens over `cols` columns;
/// None for 0, 1 or more than 8 tokens.
pub fn entry(tokens: usize, cols: usize) -> Option<(&'static str, usize, usize)> {
    match tokens {
        2 => Some(("mx_rows_small2", 2, 2)),
        3 | 4 => Some(("mx_rows_small4", 4, 4)),
        5..=8 => Some(if cols > WIDE_COLS { ("mx_rows_small8w", 8, 4) } else { ("mx_rows_small8", 8, 2) }),
        _ => None,
    }
}
/// Threads of a `mx_rows_small*` block (the tile entry has its own, `mx_rows::THREADS`): 4 warps.
pub const THREADS: u32 = 128;
/// Grid (row tiles of 4 warps x the entry's rows a warp, token tiles).
pub fn grid(rows: usize, cols: usize, tokens: usize) -> Option<[u32; 2]> {
    let (_, per, warp_rows) = entry(tokens, cols)?;
    Some([u32::try_from(rows.div_ceil(4 * warp_rows)).ok()?, u32::try_from(tokens.div_ceil(per)).ok()?])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_model_mxfp8_projections_are_admitted() {
        // q, k, v 4096 x 4096; o 2048 x 8192; dense gate and up 6144 x 4096; down 2048 x 12288 (rank-local).
        for (rows, cols) in [(4096, 4096), (2048, 8192), (6144, 4096), (2048, 12288)] { for tokens in 2..=8 { assert!(admitted(rows, cols, tokens, 256, 512, 4), "{rows}x{cols} tokens {tokens}"); } }
    }
    #[test]
    fn the_admitted_set_is_the_sweep_set() {
        for rows in 0..=7000 { assert_eq!(admitted(rows, 4096, 4, 16, 16, 4), MX_ROWS.contains(&rows) && rows * 4096 <= MX_MAX_BYTES, "rows {rows}"); }
        for cols in 0..=13000 { assert_eq!(admitted(2049, cols, 4, 16, 16, 4), MX_COLS.contains(&cols) && 2049 * cols <= MX_MAX_BYTES, "cols {cols}"); }
        for tokens in 0..=20 { assert_eq!(admitted(4096, 4096, tokens, 16, 16, 4), (2..=8).contains(&tokens), "tokens {tokens}"); }
        assert!(!admitted(4096, 4096, 4, 8, 16, 4) && !admitted(4096, 4096, 4, 16, 8, 4) && !admitted(4096, 4096, 4, 16, 16, 2) && !admitted(6144, 12288, 4, 16, 16, 4));
    }
    #[test]
    fn entries_follow_the_token_count_and_the_width() {
        assert_eq!((1..=10).map(|t| entry(t, 4096).map(|e| e.0)).collect::<Vec<_>>(), [None, Some("mx_rows_small2"), Some("mx_rows_small4"), Some("mx_rows_small4"),
            Some("mx_rows_small8"), Some("mx_rows_small8"), Some("mx_rows_small8"), Some("mx_rows_small8"), None, None]);
        // The wide matrices (o 8192 and down 12288 columns) take 4 rows a warp at 5 to 8 tokens; 2 to 4 tokens do not look at the width.
        for cols in [8192, 12288] { assert_eq!((2..=8).map(|t| entry(t, cols).unwrap()).collect::<Vec<_>>(), [("mx_rows_small2", 2, 2), ("mx_rows_small4", 4, 4), ("mx_rows_small4", 4, 4),
            ("mx_rows_small8w", 8, 4), ("mx_rows_small8w", 8, 4), ("mx_rows_small8w", 8, 4), ("mx_rows_small8w", 8, 4)]); }
        assert_eq!(entry(8, 4096), Some(("mx_rows_small8", 8, 2)));
        // A block is 4 warps of the entry's rows a warp; a token tile is the entry's token count.
        assert_eq!(grid(4096, 4096, 1), None);
        assert_eq!(grid(4096, 4096, 2), Some([512, 1]));
        assert_eq!(grid(4097, 4096, 3), Some([257, 1]));
        assert_eq!(grid(100, 4096, 2), Some([13, 1]));
        assert_eq!(grid(100, 4096, 4), Some([7, 1]));
        assert_eq!(grid(4096, 4096, 5), Some([512, 1]));
        assert_eq!(grid(2048, 8192, 8), Some([128, 1]));
        assert_eq!(grid(2048, 12288, 7), Some([128, 1]));
        assert_eq!(grid(100, 4096, 9), None);
        assert_eq!(THREADS, 128);
    }
}
