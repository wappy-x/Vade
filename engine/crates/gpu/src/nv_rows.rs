//! Admission and launch geometry for `nv_rows_tile`, the tiled NVFP4 prompt-row projection.
//! Shape and alignment only. The kernel is byte-equal to the `expert_group` NvRows path on the family
//! below (device sweep, benchmarks/2026-11-05/lane-k/r6-*); any other launch keeps `expert_group`.
pub const ROWS_PER_BLOCK: usize = 128;
pub const TOKENS_PER_BLOCK: usize = 8;
pub const THREADS: u32 = 64;
pub const MAX_ROWS: usize = 2048;
pub const MAX_COLS: usize = 8192;
/// The prompt group cap. Device checks reach this limit.
pub const MAX_TOKENS: usize = 1024;
pub const MAX_PARTS: usize = 512;
/// One part of a group launch: expert rows, columns, tokens, and the raw owners' device addresses.
#[derive(Clone, Copy)]
pub struct Part { pub rows: usize, pub cols: usize, pub tokens: usize, pub weight: u64, pub scales: u64 }
/// True when the whole launch may use `nv_rows_tile`: every part has the same rows and cols, cols a multiple
/// of 128, 16-byte aligned weights, 8-byte aligned scales, and the decode table is 8-byte aligned.
pub fn admitted(parts: &[Part], input: u64, lut: u64) -> bool {
    let Some(first) = parts.first() else { return false; };
    parts.len() <= MAX_PARTS && (1..=MAX_ROWS).contains(&first.rows) && (128..=MAX_COLS).contains(&first.cols) && first.cols % 128 == 0
        && input & 15 == 0 && lut & 7 == 0
        && parts.iter().all(|p| p.rows == first.rows && p.cols == first.cols && (1..=MAX_TOKENS).contains(&p.tokens) && p.weight & 15 == 0 && p.scales & 7 == 0)
}
/// Grid (token tiles, row tiles, parts) for the launch. None when a dimension overflows the ABI.
pub fn grid(parts: &[Part]) -> Option<[u32; 3]> {
    let first = parts.first()?;let most = parts.iter().map(|p| p.tokens).max()?;
    Some([u32::try_from(most.div_ceil(TOKENS_PER_BLOCK)).ok()?, u32::try_from(first.rows.div_ceil(ROWS_PER_BLOCK)).ok()?, u32::try_from(parts.len()).ok()?])
}
#[cfg(test)]
mod tests {
    use super::*;
    fn part(rows: usize, cols: usize, tokens: usize) -> Part { Part { rows, cols, tokens, weight: 256, scales: 512 } }
    #[test]
    fn layer_launches_are_admitted() {
        let parts: Vec<Part> = (0..288).map(|i| part(1024, 4096, 1 + i % 13)).collect();
        assert!(admitted(&parts, 256, 1024) && admitted(&[part(2048, 2048, 128)], 0, 0) && admitted(&[part(1, 128, 1)], 16, 8));
    }
    #[test]
    fn unproven_launches_keep_expert_group() {
        let ok = part(1024, 4096, 7);
        assert!(!admitted(&[], 0, 0) && !admitted(&[part(2049, 4096, 7)], 0, 0) && !admitted(&[part(1024, 4160, 7)], 0, 0) && !admitted(&[part(1024, 4096, 1025)], 0, 0));
        assert!(!admitted(&[ok, part(1024, 2048, 7)], 0, 0) && !admitted(&[ok, part(512, 4096, 7)], 0, 0) && !admitted(&[ok; 513], 0, 0));
        assert!(!admitted(&[ok], 8, 0) && !admitted(&[ok], 0, 4) && !admitted(&[Part { weight: 8, ..ok }], 0, 0) && !admitted(&[Part { scales: 4, ..ok }], 0, 0));
    }
    #[test]
    fn grid_counts_tiles() {
        let parts: Vec<Part> = (0..288).map(|i| part(1024, 4096, 1 + i % 13)).collect();
        assert_eq!(grid(&parts), Some([2, 8, 288]));
        assert_eq!(grid(&[part(129, 128, 8)]), Some([1, 2, 1]));
        assert_eq!(grid(&[part(129, 128, 9)]), Some([2, 2, 1]));
        assert_eq!(grid(&[]), None);
    }
}
