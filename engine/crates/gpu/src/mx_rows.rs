//! Admission and launch geometry for `mx_rows_tile`, the tiled MXFP8 prompt-row projection.
//! Shape and alignment only. The kernel is byte-equal to the production path (`flash_next_mxfp8_*` in groups
//! of 16 tokens) on the family below (device sweep, benchmarks/2026-11-05/lane-k/m2-*); any other call
//! keeps the old kernels.
pub const ROWS_PER_BLOCK: usize = 48;
pub const TOKENS_PER_BLOCK: usize = 16;
pub const THREADS: u32 = 384;
pub const MAX_ROWS: usize = 8192;
pub const MAX_COLS: usize = 8192;
/// The dense down projection of layers 0 to 2 (2,048 x 12,288 a rank): the one column count past `MAX_COLS`. The kernel
/// walks 256-column tiles and holds no column bound of its own; the count is admitted because the device sweep covers it
/// (`swept::MX_COLS_DENSE`). Before, a prompt group ran this matrix on the 16-token decode kernels, 64 launches a group.
pub const DENSE_COLS: usize = 12288;
pub const MAX_TOKENS: usize = 1024;
/// Wider launches win in the device sweep through 16 MiB of weight codes.
/// Larger matrices keep the existing 256-token pieces.
pub fn token_limit(rows:usize,cols:usize)->usize {
    if rows.checked_mul(cols).is_some_and(|n|n<=16*1024*1024) {MAX_TOKENS} else {256}
}
/// True when one MXFP8 projection of `tokens` rows may use `mx_rows_tile`. `round` is 1 for the GDN
/// boundary (old kernels only), `input_bf16` 1 for packed BF16 input (old kernels only).
pub fn admitted(rows: usize, cols: usize, tokens: usize, round: u32, input_bf16: u32, weights: u64, input: u64, output: u64) -> bool {
    (1..=MAX_ROWS).contains(&rows) && ((32..=MAX_COLS).contains(&cols) || cols==DENSE_COLS) && cols.is_multiple_of(32) && (1..=token_limit(rows,cols)).contains(&tokens)
        && round == 0 && input_bf16 == 0 && weights & 15 == 0 && input & 15 == 0 && output & 3 == 0
}
/// Grid (row tiles, token tiles).
pub fn grid(rows: usize, tokens: usize) -> Option<[u32; 2]> {
    Some([u32::try_from(rows.div_ceil(ROWS_PER_BLOCK)).ok()?, u32::try_from(tokens.div_ceil(TOKENS_PER_BLOCK)).ok()?])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_groups_are_admitted() {
        assert!(admitted(4096, 4096, 256, 0, 0, 256, 512, 4) && admitted(2048, 8192, 17, 0, 0, 0, 0, 0) && admitted(1, 32, 1, 0, 0, 16, 16, 4));
        // The dense down projection: a rank's 2,048 rows of 12,288 columns, in the 256-token pieces of the larger weights.
        assert!(admitted(2048, DENSE_COLS, 256, 0, 0, 16, 16, 4) && admitted(2048, DENSE_COLS, 5, 0, 0, 0, 0, 0) && !admitted(2048, DENSE_COLS, 257, 0, 0, 16, 16, 4));
    }
    #[test]
    fn unproven_calls_keep_the_old_kernels() {
        assert!(!admitted(0, 4096, 8, 0, 0, 0, 0, 0) && !admitted(8193, 4096, 8, 0, 0, 0, 0, 0) && !admitted(4096, 4112, 8, 0, 0, 0, 0, 0) && !admitted(4096, 8224, 8, 0, 0, 0, 0, 0));
        assert!(!admitted(4096, 4096, 0, 0, 0, 0, 0, 0) && !admitted(4096, 4096, 1025, 0, 0, 0, 0, 0));
        assert!(!admitted(2048, DENSE_COLS - 32, 8, 0, 0, 0, 0, 0) && !admitted(2048, DENSE_COLS + 32, 8, 0, 0, 0, 0, 0) && !admitted(2048, 2 * DENSE_COLS, 8, 0, 0, 0, 0, 0) && !admitted(2048, 10240, 8, 0, 0, 0, 0, 0));
        assert!(!admitted(4096, 4096, 8, 1, 0, 0, 0, 0) && !admitted(4096, 4096, 8, 0, 1, 0, 0, 0));
        assert!(!admitted(4096, 4096, 8, 0, 0, 8, 0, 0) && !admitted(4096, 4096, 8, 0, 0, 0, 8, 0) && !admitted(4096, 4096, 8, 0, 0, 0, 0, 2));
    }
    /// Grok's finding of 2026-11-06: the predicate admitted rows, columns and token counts the device sweep never reached. The standing rule:
    /// every admitted value of each axis is on the sweep's axis list (`swept.rs`, walked by `kernel-bench --mx-rows`), and the corners are swept.
    #[test]
    fn every_admitted_value_is_on_the_device_sweep_axes() {
        use crate::swept::*;
        let (rows_axis, cols_axis, tokens_axis) = (mx_rows_axis(), mx_cols_axis(), mx_tokens_axis());
        for rows in 0..=MAX_ROWS + 70 { if admitted(rows, 4096, 16, 0, 0, 0, 0, 0) { assert!(rows_axis.binary_search(&rows).is_ok(), "rows {rows}"); } }
        for cols in 0..=DENSE_COLS + 70 { if admitted(1024, cols, 16, 0, 0, 0, 0, 0) { assert!(cols_axis.binary_search(&cols).is_ok(), "cols {cols}"); } }
        for &cols in &cols_axis { assert!(admitted(1024, cols, 16, 0, 0, 0, 0, 0), "axis cols {cols}"); }
        for tokens in 0..=MAX_TOKENS + 70 { if admitted(1024, 4096, tokens, 0, 0, 0, 0, 0) { assert!(tokens_axis.binary_search(&tokens).is_ok(), "tokens {tokens}"); } }
        assert_eq!((rows_axis.len(), cols_axis.len(), tokens_axis.len()), (MAX_ROWS, MAX_COLS / 32 + 1, MAX_TOKENS));
        assert_eq!((MX_ROWS_MAX, MX_COLS_MAX, MX_COLS_DENSE, MX_TOKENS_MAX), (MAX_ROWS, MAX_COLS, DENSE_COLS, MAX_TOKENS));
        let corners = mx_corners();
        assert_eq!(corners.len(), 12);
        for (r, c, t) in corners { assert_eq!(admitted(r, c, t, 0, 0, 16, 16, 4),t<=token_limit(r,c), "corner {r} {c} {t}"); }
        // One past each end of every axis is refused.
        assert!(!admitted(MAX_ROWS + 1, 32, 1, 0, 0, 0, 0, 0) && !admitted(1, MAX_COLS + 32, 1, 0, 0, 0, 0, 0) && !admitted(1, 32, MAX_TOKENS + 1, 0, 0, 0, 0, 0));
    }
    #[test]
    fn grid_counts_tiles() {
        assert_eq!(grid(4096, 256), Some([86, 16]));
        assert_eq!(grid(49, 17), Some([2, 2]));
        assert_eq!(grid(48, 16), Some([1, 1]));
        assert_eq!(THREADS as usize, ROWS_PER_BLOCK / 4 * 32);
        assert_eq!(grid(16, 16), Some([1, 1]));
    }
    #[test]
    fn larger_weights_keep_the_existing_tile_pieces() {
        for (r,c,limit) in [(4096,4096,1024),(2048,8192,1024),(8192,512,1024),(6144,4096,256),(8192,8192,256),(2048,DENSE_COLS,256),(1365,DENSE_COLS,1024)] {
            assert_eq!(token_limit(r,c),limit);assert!(admitted(r,c,limit,0,0,16,16,4));
            assert!(!admitted(r,c,limit+1,0,0,16,16,4));
            assert_eq!(crate::small_choice::mx(r,c,1024,16,16,4),crate::small_choice::Mx::Tile);
        }
        assert_eq!(token_limit(usize::MAX,2),256);
    }
}
