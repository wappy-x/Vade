//! Admission and launch geometry for the tiled plain token rows kernel `plain_rows_tile`.
//! Shape and alignment only. The kernel is byte-equal to the `prompt_rows` ProjectRows path on the
//! family below (device sweep, benchmarks/2026-11-05/lane-k/t1-*); any other shape keeps `prompt_rows`.
pub const THREADS: u32 = 256;
const TILE: usize = 64;
pub const MAX_ROWS: usize = 16384;
/// The prompt group cap. Device checks reach this limit.
pub const MAX_TOKENS: usize = 1024;
pub const MAX_COLS: usize = 4096;
/// True when `plain_rows_tile` may replace `prompt_rows` for output `[rows, tokens]` over `cols` columns.
/// `input` and `weight` are the device addresses of the two operands.
pub fn admitted(rows: usize, tokens: usize, cols: usize, input: u64, weight: u64) -> bool {
    (1..=MAX_ROWS).contains(&rows) && (1..=MAX_TOKENS).contains(&tokens) && (4..=MAX_COLS).contains(&cols) && cols % 4 == 0 && (input | weight) & 15 == 0
}
/// `plain_rows_narrow` (16 rows x 32 tokens, 128 threads): projections of up to 192 rows, where the 64 x 64 tile launches only 1 to 3
/// row blocks per token tile and leaves most SMs idle. Same bytes as `prompt_rows` on the shapes below (device sweep, `kernel-bench --rows-tile`);
/// every admitted rows, tokens and columns value is swept (`swept.rs` axis lists, walked by `kernel-bench --rows-tile base cand narrow`), the rows x tokens
/// grid at columns 4 and 36 in full, the columns axis at eight (rows, tokens) shapes, and the eight corners.
pub const NARROW_THREADS: u32 = 128;
pub const NARROW_MAX_ROWS: usize = 192;
pub const NARROW_MAX_TOKENS: usize = 512;
pub fn narrow_admitted(rows: usize, tokens: usize, cols: usize, input: u64, weight: u64) -> bool {
    (1..=NARROW_MAX_ROWS).contains(&rows) && (1..=NARROW_MAX_TOKENS).contains(&tokens) && admitted(rows, tokens, cols, input, weight)
}
/// Blocks of the narrow launch: token tiles of 32 fastest.
pub fn narrow_blocks(rows: usize, tokens: usize) -> Option<u32> { u32::try_from(rows.div_ceil(16).checked_mul(tokens.div_ceil(32))?).ok() }
/// Blocks for the launch: token tiles fastest, so neighbouring blocks share weight lines.
pub fn blocks(rows: usize, tokens: usize) -> Option<u32> { u32::try_from(rows.div_ceil(TILE).checked_mul(tokens.div_ceil(TILE))?).ok() }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_shape_and_edges_are_admitted() {
        assert!(admitted(16384, 256, 512, 256, 512) && admitted(1, 1, 4, 16, 32) && admitted(16384, 1024, 4096, 0, 0));
    }
    #[test]
    fn unproven_or_unaligned_shapes_keep_the_old_kernel() {
        assert!(!admitted(0, 256, 512, 0, 0) && !admitted(16385, 256, 512, 0, 0) && !admitted(16, 0, 512, 0, 0) && !admitted(16, 1025, 512, 0, 0));
        assert!(!admitted(16, 16, 510, 0, 0) && !admitted(16, 16, 0, 0, 0) && !admitted(16, 16, 4100, 0, 0) && !admitted(16, 16, 2, 0, 0));
        assert!(!admitted(16, 16, 512, 4, 16) && !admitted(16, 16, 512, 16, 8));
    }
    /// Astra's finding of 2026-11-06 (and the standing rule): every admitted value of each axis is on the device sweep's axis list.
    #[test]
    fn every_narrow_value_is_on_the_device_sweep_axes() {
        use crate::swept::*;
        let (rows_axis, tokens_axis, cols_axis) = (narrow_rows_axis(), narrow_tokens_axis(), narrow_cols_axis());
        for rows in 0..=NARROW_MAX_ROWS + 70 { if narrow_admitted(rows, 64, 512, 16, 16) { assert!(rows_axis.binary_search(&rows).is_ok(), "rows {rows}"); } }
        for tokens in 0..=NARROW_MAX_TOKENS + 70 { if narrow_admitted(64, tokens, 512, 16, 16) { assert!(tokens_axis.binary_search(&tokens).is_ok(), "tokens {tokens}"); } }
        for cols in 0..=MAX_COLS + 70 { if narrow_admitted(64, 64, cols, 16, 16) { assert!(cols_axis.binary_search(&cols).is_ok(), "cols {cols}"); } }
        assert_eq!((rows_axis.len(), tokens_axis.len(), cols_axis.len()), (NARROW_MAX_ROWS, NARROW_MAX_TOKENS, MAX_COLS / 4));
        assert_eq!((NARROW_ROWS_MAX, NARROW_TOKENS_MAX, NARROW_COLS_MAX), (NARROW_MAX_ROWS, NARROW_MAX_TOKENS, MAX_COLS));
        let corners = narrow_corners();
        assert_eq!(corners.len(), 8);
        for (r, c, t) in corners { assert!(narrow_admitted(r, t, c, 16, 16), "corner {r} {c} {t}"); }
    }
    #[test]
    fn narrow_set_is_the_interval_and_counts_tiles() {
        assert!(narrow_admitted(64, 256, 4096, 16, 16) && narrow_admitted(1, 1, 4, 0, 0) && narrow_admitted(192, 512, 4096, 0, 0));
        assert!(!narrow_admitted(193, 256, 512, 0, 0) && !narrow_admitted(0, 256, 512, 0, 0) && !narrow_admitted(64, 513, 512, 0, 0) && !narrow_admitted(64, 0, 512, 0, 0));
        assert!(!narrow_admitted(64, 256, 510, 0, 0) && !narrow_admitted(64, 256, 4100, 0, 0) && !narrow_admitted(64, 256, 512, 4, 16));
        for rows in 1..=300usize {assert_eq!(narrow_admitted(rows, 100, 512, 0, 0), rows <= 192);}
        assert_eq!(narrow_blocks(64, 256), Some(32));assert_eq!(narrow_blocks(192, 256), Some(96));assert_eq!(narrow_blocks(1, 1), Some(1));assert_eq!(narrow_blocks(17, 33), Some(4));
    }
    #[test]
    fn grid_counts_tiles() {
        assert_eq!(blocks(16384, 256), Some(256 * 4));
        assert_eq!(blocks(65, 1), Some(2));
        assert_eq!(blocks(64, 64), Some(1));
        assert_eq!(blocks(1, 65), Some(2));
    }
}
