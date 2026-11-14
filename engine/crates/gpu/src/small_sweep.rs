//! The shapes the small-count kernels may run, as lists. One file, two readers: the host predicates (`plain_small`,
//! `rows_small`, `nv_small`, `fp8_small`, `mx_small`) admit a shape ONLY if every dimension is listed here, and
//! the device sweeps (`kernel-bench --rows-small`, `--nv-small`, `--fp8-small`, `--mx-small`, which include this file by path)
//! walk the product of the same lists. So the admitted set cannot grow without the sweep growing: each predicate's host test
//! enumerates its admitted set over the whole range of every dimension and fails if a shape outside the lists is admitted.
//! Alignment is the other half of admission (16-byte aligned operands); the sweeps run on 16-byte aligned buffers.
//! A dimension listed here but not meaningful for a kernel is not a promise: the predicate still applies its own limits.

/// Most tokens a call of the row kernels may hold (the sweeps run every count 1 to this).
pub const MAX_TOKENS: usize = 16;

// Plain token rows (`plain_rows_small`): output [rows, tokens] over `cols` columns. Includes every plain matrix of the model (rank-local
// rows and logical rows): gate 288 / 144, f_a / g_a / wk / indexer 128 / 64, b_proj 64 / 32, f_b / g_b 8192 / 4096 x 128,
// wq_b 4096 / 2048 x 1536, weights_proj 32 / 16, kv_b 16384 x 512, and 4096 columns for the rest.
pub const PLAIN_ROWS: &[usize] = &[1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 144, 192, 193, 255, 256, 257, 288, 512, 1000, 1024, 2048, 4096, 8192, 16384];
pub const PLAIN_COLS: &[usize] = &[4, 8, 12, 28, 32, 36, 60, 64, 68, 100, 128, 132, 252, 256, 508, 512, 516, 1024, 1536, 2048, 2052, 4088, 4096];

/// `plain_rows_few` (8 rows a block, one warp for each token) takes the plain shapes with at most this many rows and tokens; the sweep runs those
/// shapes of the lists above with it as well. Larger shapes keep `plain_rows_small`.
pub const PLAIN_FEW_MAX_ROWS: usize = 512;
pub const PLAIN_FEW_MAX_TOKENS: usize = 4;

// StepRows (`prompt_rows_small`, dim 128): head counts. (Wrapped Prepare, GatedNorm and Route are lane P's `prompt_prepare`, `prompt_gated_norm`, `prompt_route`.)
pub const STEP_HEADS: &[usize] = &[1, 2, 3, 4, 8, 32, 33, 64];
pub const STEP_DIM: usize = 128;
pub fn step_swept(heads: usize, dim: usize) -> bool { STEP_HEADS.contains(&heads) && dim == STEP_DIM }

// NVFP4 expert rows (`nv_rows_small`): one rows x cols for the whole launch; the parts' token counts are 1 to 16 with a mean of at most 2,
// at most 128 parts. The sweeps run these (rows, cols) with the count families in `NV_COUNT_FAMILIES` and 200 seeded random count lists.
pub const NV_ROWS: &[usize] = &[1, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 200, 256, 1024, 2048];
pub const NV_COLS: &[usize] = &[128, 256, 384, 512, 1024, 2048, 4096, 8192];
pub const NV_MAX_PARTS: usize = 128;
pub const NV_MAX_PART_TOKENS: usize = 16;
pub const NV_MAX_MEAN_TOKENS: usize = 2;

// Block-FP8 expert rows (`fp8_rows_small`): rows, cols, block shape; parts of 1 to 8 tokens, at most 128 parts.
// The model's rank-local block-FP8 matrices on two ranks (`resident_weights::Rows::new`: equal halves in units of the scale block's 128 rows):
// gate and up 1024 x 4096, down 2048 x 2048, o_proj 2048 x 16384, q_b 8192 x 1536, q_a 768 x 4096, kv_a 256 x 4096 (512 rows in all). 256 was
// missing here (288 is the router gate, a plain matrix), so kv_a took `fp8_rows_tile` at every row count: 11 launches of a decode step.
pub const FP8_ROWS: &[usize] = &[1, 2, 15, 16, 17, 31, 32, 33, 65, 127, 129, 256, 288, 768, 1024, 2048, 8192];
pub const FP8_COLS: &[usize] = &[16, 32, 48, 64, 128, 144, 256, 272, 384, 512, 1024, 1536, 2048, 4096, 16384];
pub const FP8_BLOCKS: &[(usize, usize)] = &[(128, 128), (1, 16), (16, 32), (64, 64), (1, 128)];
pub const FP8_MAX_PARTS: usize = 128;
pub const FP8_MAX_PART_TOKENS: usize = 8;
/// `fp8_rows_multi` takes parts of up to 16 tokens (same rows, columns and block shapes).
pub const FP8_MULTI_MAX_PART_TOKENS: usize = 16;
/// A shape larger than this many weight bytes is not in the list (the sweep builds every listed shape).
pub const FP8_MAX_BYTES: usize = 134_217_728;
/// The sweep runs launches of up to `FP8_MAX_PARTS` parts only for shapes of at most `FP8_BIG_BYTES` weight bytes; larger shapes are swept with
/// launches of at most `FP8_BIG_MAX_PARTS` parts (their weights dominate the sweep's time). The predicates admit exactly that.
pub const FP8_BIG_BYTES: usize = 4_000_000;
pub const FP8_BIG_MAX_PARTS: usize = 12;
pub fn fp8_parts_swept(rows: usize, cols: usize, parts: usize) -> bool { parts <= FP8_MAX_PARTS && (rows * cols <= FP8_BIG_BYTES || parts <= FP8_BIG_MAX_PARTS) }

// MXFP8 rows (`mx_rows_small2/4/8/8w`): rows, cols, tokens 2 to 8 (1 token: the old M1 kernel; 9 and more: `mx_rows_tile`).
pub const MX_ROWS: &[usize] = &[1, 3, 4, 5, 15, 16, 17, 31, 33, 64, 100, 129, 640, 641, 1030, 2048, 2049, 4096, 6144];
pub const MX_COLS: &[usize] = &[32, 64, 96, 224, 256, 288, 512, 800, 1024, 2048, 4096, 8192, 12288];
pub const MX_MIN_TOKENS: usize = 2;
pub const MX_MAX_TOKENS: usize = 8;
/// A shape larger than this many weight bytes is not in the list.
pub const MX_MAX_BYTES: usize = 50_331_648;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lists_are_sorted_and_without_repeats() {
        fn ok(v: &[usize]) -> bool { v.windows(2).all(|w| w[0] < w[1]) }
        for v in [PLAIN_ROWS, PLAIN_COLS, STEP_HEADS, NV_ROWS, NV_COLS, FP8_ROWS, FP8_COLS, MX_ROWS, MX_COLS] { assert!(ok(v)); }
    }
}
