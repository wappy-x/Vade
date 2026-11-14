//! Admission and launch geometry for the tiled attention scores kernel `scores_rows_tile`.
//! Shape and alignment only. Any other shape runs on `prompt_rows`, with the same bits.
//!
//! One file, two readers: `admitted` takes a shape ONLY when its key dim, value width (taps) and head count are
//! listed below, and the device sweep (`kernel-bench --scores-tile`, which includes this file by path) runs every
//! listed value against `prompt_rows`, with the row and token counts at every tile edge of their intervals and at
//! the interval ends. So the admitted set cannot grow without the sweep growing: the host test walks every dim,
//! taps and head count up to past the largest listed value and fails if an unlisted one is admitted.
pub const THREADS: u32 = 256;
const TILE: usize = 64;
pub const MAX_ROWS: usize = 16384;
/// The prompt group cap. Device checks reach this limit.
pub const MAX_TOKENS: usize = 1024;
/// Key dims swept: the model's 256, every remainder of the 32-column chunk (4 to 28, 36, 60, 68, 132, 252, 260),
/// and the powers of two to 4,096.
pub const DIMS: &[usize] = &[4, 8, 12, 16, 20, 24, 28, 32, 36, 60, 64, 68, 96, 128, 132, 192, 252, 256, 260, 512, 576, 1024, 2048, 4096];
/// Value widths swept. They only move the key row stride `heads * (dim + taps)`; the model's is 256, the device agreement check (resident-decode-check prompt-rows) and the layer fixture use 16.
pub const TAPS: &[usize] = &[4, 8, 12, 16, 32, 36, 64, 128, 252, 256, 260, 512];
/// Head counts swept: the model's 64, 32 and 16 for 2 and 4 ranks, and small counts.
pub const HEADS: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 64, 128];
/// Row and token counts the sweep runs: both ends of each interval and both sides of tile edges.
pub const ROW_EDGES: &[usize] = &[1, 2, 3, 63, 64, 65, 127, 128, 129, 1000, 4095, 4096, 4097, 16383, 16384];
pub const TOKEN_EDGES: &[usize] = &[1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024];
/// True when `scores_rows_tile_packed` may run this shape: the model geometry only (key and value
/// width 256, the head counts above), swept on the device by latent-rows --bench2 and the prompt-rows
/// fixture against `prompt_rows`. The packed row base and queries stay 16-byte aligned, so each head
/// segment (256 packed words) and each 4-column group (2 packed words) keeps its load alignment.
pub fn packed_admitted(rows: usize, tokens: usize, heads: usize, dim: usize, taps: usize, queries: u64, keys: u64) -> bool {
    (1..=MAX_ROWS).contains(&rows) && (1..=MAX_TOKENS).contains(&tokens) && HEADS.contains(&heads) && dim == 256 && taps == 256
        && (queries | keys) & 15 == 0
}
/// True when `scores_rows_tile` may run this shape. `queries` and `keys` are device addresses.
pub fn admitted(rows: usize, tokens: usize, heads: usize, dim: usize, taps: usize, queries: u64, keys: u64) -> bool {
    (1..=MAX_ROWS).contains(&rows) && (1..=MAX_TOKENS).contains(&tokens) && HEADS.contains(&heads) && DIMS.contains(&dim) && TAPS.contains(&taps)
        && (queries | keys) & 15 == 0
}
/// Blocks for the launch: each head's token tiles fastest, so neighbouring blocks share key lines.
pub fn blocks(rows: usize, tokens: usize, heads: usize) -> Option<u32> { u32::try_from(rows.div_ceil(TILE).checked_mul(tokens.div_ceil(TILE))?.checked_mul(heads)?).ok() }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_shape_and_edges_are_admitted() {
        assert!(admitted(8192, 256, 64, 256, 256, 256, 512) && admitted(16384, 1024, 32, 256, 256, 16, 32) && admitted(1, 1, 1, 4, 4, 16, 32));
    }
    #[test]
    fn unaligned_or_other_shapes_keep_the_row_kernel() {
        assert!(admitted(1, 1024, 1, 4, 4, 0, 0) && !admitted(1, 1025, 1, 4, 4, 0, 0));
        assert!(!admitted(0, 1, 1, 4, 4, 0, 0) && !admitted(1, 0, 1, 4, 4, 0, 0) && !admitted(1, 1, 0, 4, 4, 0, 0));
        assert!(!admitted(1, 1, 1, 6, 4, 0, 0) && !admitted(1, 1, 1, 4, 2, 0, 0) && !admitted(16385, 1, 1, 4, 4, 0, 0));
        assert!(!admitted(1, 1, 1, 4, 4, 4, 0) && !admitted(1, 1, 1, 4, 4, 0, 8));
        // Review: taps 0 is never launched (Shape::lengths refuses it) and is no longer admitted either.
        assert!(!admitted(1, 1, 1, 4, 0, 16, 32) && !admitted(1, 1, 1, 0, 4, 16, 32));
    }
    #[test]
    fn admits_only_the_swept_lists_and_intervals() {
        // Every dim, taps and head count from 0 past the largest listed value: admitted only if listed.
        for dim in 0..=DIMS[DIMS.len() - 1] + 64 { assert_eq!(admitted(65, 65, 2, dim, 4, 0, 0), DIMS.contains(&dim), "dim {dim}"); }
        for taps in 0..=TAPS[TAPS.len() - 1] + 64 { assert_eq!(admitted(65, 65, 2, 4, taps, 0, 0), TAPS.contains(&taps), "taps {taps}"); }
        for heads in 0..=HEADS[HEADS.len() - 1] + 64 { assert_eq!(admitted(65, 65, heads, 4, 4, 0, 0), HEADS.contains(&heads), "heads {heads}"); }
        // Rows and tokens are intervals; the sweep runs both ends and every edge listed, all inside.
        for rows in 0..=MAX_ROWS + 1 { assert_eq!(admitted(rows, 1, 1, 4, 4, 0, 0), (1..=MAX_ROWS).contains(&rows)); }
        for tokens in 0..=MAX_TOKENS + 1 { assert_eq!(admitted(1, tokens, 1, 4, 4, 0, 0), (1..=MAX_TOKENS).contains(&tokens)); }
        assert!(ROW_EDGES.contains(&1) && ROW_EDGES.contains(&MAX_ROWS) && TOKEN_EDGES.contains(&1) && TOKEN_EDGES.contains(&MAX_TOKENS));
        assert!(ROW_EDGES.iter().all(|&r| (1..=MAX_ROWS).contains(&r)) && TOKEN_EDGES.iter().all(|&t| (1..=MAX_TOKENS).contains(&t)));
        // The kernel's own guard: dim and taps multiples of 4 (16-byte copies), so every listed value keeps it.
        assert!(DIMS.iter().chain(TAPS).all(|&v| v > 0 && v % 4 == 0));
    }
    #[test]
    fn packed_admission_is_the_model_geometry_only() {
        assert!(packed_admitted(2051, 256, 64, 256, 256, 0, 0) && packed_admitted(1, 1, 16, 256, 256, 16, 32));
        assert!(!packed_admitted(2051, 256, 64, 128, 256, 0, 0) && !packed_admitted(2051, 256, 64, 256, 128, 0, 0));
        assert!(!packed_admitted(2051, 256, 7, 256, 256, 0, 0) && !packed_admitted(0, 256, 64, 256, 256, 0, 0));
        assert!(!packed_admitted(2051, 1025, 64, 256, 256, 0, 0) && !packed_admitted(2051, 256, 64, 256, 256, 8, 0));
    }
    #[test]
    fn grid_counts_heads_and_tiles() {
        assert_eq!(blocks(8192, 256, 64), Some(128 * 4 * 64));
        assert_eq!(blocks(65, 1, 2), Some(4));
        assert_eq!(blocks(MAX_ROWS, MAX_TOKENS, HEADS[HEADS.len() - 1]), Some(256 * 16 * 128));
    }
}
