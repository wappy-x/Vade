//! Admission and launch geometry for `fp8_rows_tile`, the tiled block-FP8 prompt-row projection.
//! Shape and alignment only. The kernel is byte-equal to the `expert_group` Fp8Rows path on the family
//! below (device sweep, benchmarks/2026-11-05/lane-k/f*-fp8rows-*); any other launch keeps `expert_group`.
/// TENSOR-CORE REBASE (block FP8): a block of the prompt tile is `WARPS` warps (16 weight rows a warp) and
/// `TOKENS_PER_BLOCK` token rows (four tensor-core tiles of 8 a warp), the one body of `kernels/src/fp8_tc.rs`.
pub const WARPS: usize = 8;
pub const ROWS_PER_BLOCK: usize = 16 * WARPS;
pub const TOKENS_PER_BLOCK: usize = 32;
/// Stages of 128 columns that a block keeps in flight.
pub const DEPTH: usize = 3;
pub const THREADS: u32 = (32 * WARPS) as u32;
/// Dynamic shared bytes of a launch: the stages of the longest part, at most one block of tokens.
pub fn shared(parts: &[Part]) -> u32 {
    let most = parts.iter().map(|p| p.tokens).max().unwrap_or(1).clamp(1, TOKENS_PER_BLOCK);
    crate::expert_ops::fp8_tc_shared(WARPS as u32, most as u32, DEPTH as u32)
}
pub const MAX_ROWS: usize = 8192;
pub const MAX_COLS: usize = 16384;
/// The prompt group cap. Device checks reach this limit.
pub const MAX_TOKENS: usize = 1024;
/// One part of a group launch.
#[derive(Clone, Copy)]
pub struct Part { pub rows: usize, pub cols: usize, pub block_rows: usize, pub block_cols: usize, pub tokens: usize, pub weight: u64, pub scales: u64 }
/// True when the whole launch may use `fp8_rows_tile`: every part has the same rows and cols, cols a multiple of
/// 16, block_cols a power of two of at least 16, block_rows at least 1, 16-byte aligned weights and input rows
/// (cols a multiple of 4 floats), 4-byte aligned scales. `input` is the device address of the activations.
pub fn admitted(parts: &[Part], input: u64) -> bool {
    let Some(first) = parts.first() else { return false; };
    (1..=MAX_ROWS).contains(&first.rows) && (16..=MAX_COLS).contains(&first.cols) && first.cols % 16 == 0 && input & 15 == 0
        && parts.iter().all(|p| p.rows == first.rows && p.cols == first.cols && p.block_rows >= 1 && p.block_cols >= 16 && p.block_cols.is_power_of_two()
            && (1..=MAX_TOKENS).contains(&p.tokens) && p.weight & 15 == 0 && p.scales & 3 == 0)
}
/// Grid (row groups, token tiles, parts). The row groups of one token tile are neighbours: they share its token rows.
/// None when a dimension overflows the ABI.
pub fn grid(parts: &[Part]) -> Option<[u32; 3]> {
    let first = parts.first()?;let most = parts.iter().map(|p| p.tokens).max()?;
    Some([u32::try_from(first.rows.div_ceil(ROWS_PER_BLOCK)).ok()?, u32::try_from(most.div_ceil(TOKENS_PER_BLOCK)).ok()?, u32::try_from(parts.len()).ok()?])
}
/// TENSOR-CORE REBASE (block FP8): the THIRD form of the prompt tile, the kernel `fp8_rows_tile3` (body `tile3_body`
/// of `kernels/src/fp8_tc.rs`). Its word is the word of `fp8_rows_tile`, bit for bit (device proof: kernel bench
/// `--fp8-tc`, modes `all` and `tile3`); the body gets its operands another way: a stage is one scale block, the
/// tensor core decodes the weights, the next stage is prefetched.
/// MEASURED (kernel bench run `f8tc-t3-5`, cold weights, microseconds a launch at 1,024 token rows, first form then
/// this form): 2,048 x 2,048 660.0 to 281.6; 1,024 x 4,096 698.6 to 259.9; 2,048 x 16,384 5,090.7 to 1,944.5;
/// 8,192 x 1,536 2,002.7 to 897.3; 768 x 4,096 470.8 to 178.8; 288 x 4,096 291.6 to 85.0. A pass of the model's 170
/// calls: 172.82 to 67.82 ms at 1,024 token rows, 51.04 to 22.05 at 256, 22.06 to 12.41 at 64, 22.25 to 12.21 at 40,
/// 20.84 to 11.30 at 24. The driver gives the kernel 128 registers a thread (512 threads a block would still launch).
pub const THIRD_KERNEL: &str = "fp8_rows_tile3";
/// Warps a block of the third form (16 weight rows a warp) and tiles of 8 token rows a warp.
pub const THIRD_WARPS: usize = 8;
pub const THIRD_TILES: usize = 4;
pub const THIRD_THREADS: u32 = (32 * THIRD_WARPS) as u32;
/// Dynamic shared bytes of a block: 2 double-buffered stages of 8 steps plus `PARK=2` shared partial/low-join
/// storage (`fp8_tc::tile3_shared(4, 8, 2)` = 49,152 B -> 2 CTAs/SM, 0 spills, zero global output parking).
pub const THIRD_SHARED: u32 = (2 * 8 * 8 * 4 * THIRD_TILES * 8 + THIRD_WARPS * 32 * THIRD_TILES * 32) as u32;
/// Token rows of the longest part from which the third form is launched: where the prompt tile is the host's choice
/// at all (more than 16 token rows, `small_choice`). The smallest size measured is 24, one token tile, as 17 is.
pub const THIRD_MIN_TOKENS: usize = 17;
/// The grid (row groups, token tiles, parts) of the third form for these parts, or None: keep `fp8_rows_tile`. The
/// kernel takes scale blocks of 128 columns and a multiple of 16 rows, and whole stages of 128 columns (it traps on
/// anything else). Call it only for parts that `admitted` takes.
pub fn third(parts: &[Part]) -> Option<[u32; 3]> {
    let first = parts.first()?;let most = parts.iter().map(|p| p.tokens).max()?;
    if most < THIRD_MIN_TOKENS || parts.iter().any(|p| p.block_cols != 128 || p.cols % 128 != 0 || p.block_rows == 0 || p.block_rows % 16 != 0) { return None; }
    Some([u32::try_from(first.rows.div_ceil(16 * THIRD_WARPS)).ok()?, u32::try_from(most.div_ceil(8 * THIRD_TILES)).ok()?, u32::try_from(parts.len()).ok()?])
}
/// Wide-prompt admission for `fp8_rows_tile3` (`NF=4, PARK=0`, `16,384 B` dynamic smem, 2 CTAs/SM, L2 CTA swizzling)
/// with no arbitrary 1,024-token cap (Rule 17).
pub fn third_wide(parts: &[Part], input: u64) -> Option<[u32; 3]> {
    let first = parts.first()?;let most = parts.iter().map(|p| p.tokens).max()?;
    if most < THIRD_MIN_TOKENS || first.rows == 0 || first.cols == 0 || !first.cols.is_multiple_of(128) || input & 15 != 0 { return None; }
    if parts.iter().any(|p| p.rows != first.rows || p.cols != first.cols || p.tokens == 0
        || p.block_cols != 128 || p.block_rows == 0 || !p.block_rows.is_multiple_of(16)
        || p.weight & 15 != 0 || p.scales & 3 != 0) { return None; }
    Some([u32::try_from(first.rows.div_ceil(16 * THIRD_WARPS)).ok()?, u32::try_from(most.div_ceil(8 * THIRD_TILES)).ok()?, u32::try_from(parts.len()).ok()?])
}
/// Wide-prompt third form (`fp8_rows_tile3_n8`, `NF=8, PARK=2`): 64 tokens x 128 rows per 256-thread block, 98,304 B
/// dynamic shared memory, zero global output parking, and no arbitrary 1,024-token cap (Rule 17).
pub const THIRD_N8_KERNEL: &str = "fp8_rows_tile3_n8";
pub const THIRD_N8_TILES: usize = 8;
pub const THIRD_N8_SHARED: u32 = (2 * 8 * 8 * 4 * THIRD_N8_TILES * 8 + THIRD_WARPS * 32 * THIRD_N8_TILES * 32) as u32;
pub const THIRD_N8_MIN_TOKENS: usize = 33;
/// Grid (row groups, 64-token tiles, parts) for `fp8_rows_tile3_n8`, or None when parts do not meet `group_tile3`.
pub fn third_n8(parts: &[Part], input: u64) -> Option<[u32; 3]> {
    let first = parts.first()?;let most = parts.iter().map(|p| p.tokens).max()?;
    if most < THIRD_N8_MIN_TOKENS || first.rows == 0 || first.cols == 0 || !first.cols.is_multiple_of(128) || input & 15 != 0 { return None; }
    if parts.iter().any(|p| p.rows != first.rows || p.cols != first.cols || p.tokens == 0
        || p.block_cols != 128 || p.block_rows == 0 || !p.block_rows.is_multiple_of(16)
        || p.weight & 15 != 0 || p.scales & 3 != 0) { return None; }
    Some([u32::try_from(first.rows.div_ceil(16 * THIRD_WARPS)).ok()?, u32::try_from(most.div_ceil(8 * THIRD_N8_TILES)).ok()?, u32::try_from(parts.len()).ok()?])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_third_form_takes_whole_stages_of_the_model_block_only() {
        // The six shapes of the model, one and two parts: the longest part decides the token tiles.
        assert_eq!(third(&[part(2048, 2048, 1024)]), Some([16, 32, 1]));
        assert_eq!(third(&[part(1024, 4096, 100), part(1024, 4096, 1000)]), Some([8, 32, 2]));
        assert_eq!(third(&[part(2048, 16384, 64)]), Some([16, 2, 1]));
        assert_eq!(third(&[part(2048, 2048, 17)]), Some([16, 1, 1]));
        assert_eq!(third(&[part(8192, 1536, 65)]), Some([64, 3, 1]));
        assert_eq!(third(&[part(768, 4096, 257)]), Some([6, 9, 1]));
        assert_eq!(third(&[part(288, 4096, 1024)]), Some([3, 32, 1]));
        // Short groups keep the first form; so does every block or column count the kernel would trap on.
        assert_eq!(third(&[part(2048, 2048, 16)]), None);
        assert_eq!(third(&[part(2048, 2048, 1), part(2048, 2048, 16)]), None);
        assert_eq!(third(&[]), None);
        let ok = part(2048, 2048, 256);
        assert!(third(&[ok]).is_some());
        for wrong in [Part { block_cols: 64, ..ok }, Part { block_cols: 256, ..ok }, Part { cols: 2064, ..ok }, Part { block_rows: 8, ..ok }, Part { block_rows: 0, ..ok }, Part { block_rows: 24, ..ok }] {
            assert_eq!(third(&[wrong]), None);
            assert_eq!(third(&[ok, wrong]), None);
        }
        assert!(third(&[Part { block_rows: 16, ..ok }]).is_some() && third(&[Part { block_rows: 256, ..ok }]).is_some());
        // The launch constants are the kernel's: 256 threads, 2 double-buffered stages + PARK=2 shared storage for 4 tiles a warp.
        assert_eq!((THIRD_THREADS, THIRD_SHARED), (256, 49152));
    }
    fn part(rows: usize, cols: usize, tokens: usize) -> Part { Part { rows, cols, block_rows: 128, block_cols: 128, tokens, weight: 256, scales: 512 } }
    #[test]
    fn dense_prompt_groups_are_admitted() {
        assert!(admitted(&[part(2048, 2048, 256)], 256) && admitted(&[part(1024, 4096, 17)], 0) && admitted(&[part(2048, 16384, 512)], 16) && admitted(&[part(1, 16, 1)], 16));
    }
    #[test]
    fn unproven_launches_keep_expert_group() {
        let ok = part(2048, 2048, 256);
        assert!(!admitted(&[], 0) && !admitted(&[part(8193, 2048, 8)], 0) && !admitted(&[part(2048, 2056, 8)], 0) && !admitted(&[part(2048, 16400, 8)], 0) && !admitted(&[part(2048, 2048, 1025)], 0));
        assert!(!admitted(&[Part { block_cols: 8, ..ok }], 0) && !admitted(&[Part { block_cols: 96, ..ok }], 0) && !admitted(&[Part { block_rows: 0, ..ok }], 0));
        assert!(!admitted(&[ok], 8) && !admitted(&[Part { weight: 8, ..ok }], 0) && !admitted(&[Part { scales: 2, ..ok }], 0) && !admitted(&[ok, part(1024, 2048, 8)], 0));
    }
    #[test]
    fn grid_counts_tiles() {
        assert_eq!(grid(&[part(2048, 2048, 256)]), Some([16, 8, 1]));
        assert_eq!(grid(&[part(65, 64, 65)]), Some([1, 3, 1]));
        assert_eq!(grid(&[]), None);
    }
}
