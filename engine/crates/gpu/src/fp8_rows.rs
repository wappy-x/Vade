//! Admission and launch geometry for `fp8_rows_tile`, the tiled block-FP8 prompt-row projection.
//! Shape and alignment only. The kernel is byte-equal to the `expert_group` Fp8Rows path on the family
//! below (device sweep, benchmarks/2026-11-05/lane-k/f*-fp8rows-*); any other launch keeps `expert_group`.
pub const ROWS_PER_BLOCK: usize = 64;
pub const TOKENS_PER_BLOCK: usize = 64;
pub const THREADS: u32 = 256;
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
/// Grid (token tiles, row tiles, parts). None when a dimension overflows the ABI.
pub fn grid(parts: &[Part]) -> Option<[u32; 3]> {
    let first = parts.first()?;let most = parts.iter().map(|p| p.tokens).max()?;
    Some([u32::try_from(most.div_ceil(TOKENS_PER_BLOCK)).ok()?, u32::try_from(first.rows.div_ceil(ROWS_PER_BLOCK)).ok()?, u32::try_from(parts.len()).ok()?])
}
#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(grid(&[part(2048, 2048, 256)]), Some([4, 32, 1]));
        assert_eq!(grid(&[part(65, 64, 65)]), Some([2, 2, 1]));
        assert_eq!(grid(&[]), None);
    }
}
