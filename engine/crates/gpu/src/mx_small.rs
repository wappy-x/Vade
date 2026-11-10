//! Admission and entry choice for the small-count MXFP8 rows kernels `mx_rows_small2/4` (a draft verify pass): the token tile of
//! `mx_rows_tile` as a parameter. The admitted calls are the lists of `small_sweep` (rows, columns, 2 to 4 tokens) with the alignment
//! of `mx_rows::admitted`; the device sweep (`kernel-bench --mx-small`) walks every listed (rows, columns) at every token count
//! against the production M kernels. Any other call keeps `mx_rows_tile` or the old kernels.
use crate::small_sweep::{MX_COLS, MX_MAX_BYTES, MX_MAX_TOKENS, MX_MIN_TOKENS, MX_ROWS};
/// True when this MXFP8 projection of `tokens` rows may use the small entries (`weights`, `input`, `output` are device addresses).
pub fn admitted(rows: usize, cols: usize, tokens: usize, weights: u64, input: u64, output: u64) -> bool {
    (MX_MIN_TOKENS..=MX_MAX_TOKENS).contains(&tokens) && MX_ROWS.contains(&rows) && MX_COLS.contains(&cols) && rows * cols <= MX_MAX_BYTES && weights & 15 == 0 && input & 15 == 0 && output & 3 == 0
}
/// The entry name and the tokens a block takes for a call of `tokens` tokens; None for 0, 1 or more than 4.
pub fn entry(tokens: usize) -> Option<(&'static str, usize)> {
    match tokens { 2 => Some(("mx_rows_small2", 2)), 3 | 4 => Some(("mx_rows_small4", 4)), _ => None }
}
/// Grid (row tiles of 16 rows, token tiles).
/// Threads of a `mx_rows_small*` block (the tile entry has its own, `mx_rows::THREADS`).
pub const THREADS: u32 = 128;
pub fn grid(rows: usize, tokens: usize) -> Option<[u32; 2]> {
    let (_, per) = entry(tokens)?;
    Some([u32::try_from(rows.div_ceil(16)).ok()?, u32::try_from(tokens.div_ceil(per)).ok()?])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_model_mxfp8_projections_are_admitted() {
        // q, k, v 4096 x 4096; o 2048 x 8192; dense gate and up 6144 x 4096; down 2048 x 12288 (rank-local).
        for (rows, cols) in [(4096, 4096), (2048, 8192), (6144, 4096), (2048, 12288)] { for tokens in 2..=4 { assert!(admitted(rows, cols, tokens, 256, 512, 4), "{rows}x{cols} tokens {tokens}"); } }
    }
    #[test]
    fn the_admitted_set_is_the_sweep_set() {
        for rows in 0..=7000 { assert_eq!(admitted(rows, 4096, 4, 16, 16, 4), MX_ROWS.contains(&rows) && rows * 4096 <= MX_MAX_BYTES, "rows {rows}"); }
        for cols in 0..=13000 { assert_eq!(admitted(2049, cols, 4, 16, 16, 4), MX_COLS.contains(&cols) && 2049 * cols <= MX_MAX_BYTES, "cols {cols}"); }
        for tokens in 0..=20 { assert_eq!(admitted(4096, 4096, tokens, 16, 16, 4), (2..=4).contains(&tokens), "tokens {tokens}"); }
        assert!(!admitted(4096, 4096, 4, 8, 16, 4) && !admitted(4096, 4096, 4, 16, 8, 4) && !admitted(4096, 4096, 4, 16, 16, 2) && !admitted(6144, 12288, 4, 16, 16, 4));
    }
    #[test]
    fn entries_follow_the_token_count() {
        assert_eq!((1..=6).map(|t| entry(t).map(|e| e.0)).collect::<Vec<_>>(), [None, Some("mx_rows_small2"), Some("mx_rows_small4"), Some("mx_rows_small4"), None, None]);
        assert_eq!(grid(4096, 1), None);
        assert_eq!(grid(4096, 2), Some([256, 1]));
        assert_eq!(grid(4097, 3), Some([257, 1]));
        assert_eq!(grid(100, 2), Some([7, 1]));
        assert_eq!(grid(100, 5), None);
        assert_eq!(grid(100, 1), None);
    }
}
