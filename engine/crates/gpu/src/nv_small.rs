//! Admission and launch geometry for `nv_rows_small`, the NVFP4 expert-row tile for a SMALL token count (a draft
//! verify pass): one block for each (part, token) job and 16 rows. Shape and alignment only. The admitted launches are the lists of
//! `small_sweep` (rows, columns) with at most 128 parts of 1 to 16 tokens each and at most 2 tokens a part on average; the device sweep
//! (`kernel-bench --nv-small`) walks every listed (rows, columns) against `expert_group` with the count families of
//! `kernel_nvsmall.rs` and seeded random count lists (two for each listed shape). Any other launch keeps `nv_rows_tile` or `expert_group`.
use crate::nv_rows::Part;
use crate::small_sweep::{NV_COLS, NV_MAX_MEAN_TOKENS, NV_MAX_PARTS, NV_MAX_PART_TOKENS, NV_ROWS};
pub const ROWS_PER_BLOCK: usize = 16;
pub const THREADS: u32 = 32 + ROWS_PER_BLOCK as u32 * 8;
/// Dynamic shared bytes: 4 stages of (16 rows x 64 weight bytes + 16 x 8 scale bytes + 8 activation blocks of 80 bytes)
/// and 2 product buffers of 16 rows x 132 words (same layout as the decode `nv_tile`).
pub const SHARED: u32 = (4 * (ROWS_PER_BLOCK * 64 + ROWS_PER_BLOCK * 8 + 8 * 80) + 2 * ROWS_PER_BLOCK * 132 * 4) as u32;
/// True when the whole launch may use `nv_rows_small`: every part has the same listed rows and cols, 1 to 16 tokens and at most
/// 2 tokens a part on average (lane K's `nv_rows_tile` is faster where parts hold about 3 tokens or more), 16-byte aligned weights and
/// scales, and 16-byte aligned activation rows.
pub fn admitted(parts: &[Part], input: u64) -> bool {
    let Some(first) = parts.first() else { return false; };
    parts.len() <= NV_MAX_PARTS && NV_ROWS.contains(&first.rows) && NV_COLS.contains(&first.cols) && input & 15 == 0
        && parts.iter().all(|p| p.rows == first.rows && p.cols == first.cols && (1..=NV_MAX_PART_TOKENS).contains(&p.tokens) && (p.weight | p.scales) & 15 == 0)
        && parts.iter().map(|p| p.tokens).sum::<usize>() <= NV_MAX_MEAN_TOKENS * parts.len()
}
/// Column-tile parts of few tokens (`expert_ops::NV_TILE_ROWS`): the same launch rule as `admitted` (at most NV_MAX_PARTS parts of 1 to
/// NV_MAX_PART_TOKENS tokens, at most NV_MAX_MEAN_TOKENS tokens a part on average), rows and cols inside the storage rule, 16-byte aligned
/// weights and scales and a word-aligned input. Each (part, token) then runs as one job of the decode kernel `nv_cols`.
pub fn admitted_columns(parts: &[Part], input: u64) -> bool {
    let Some(first) = parts.first() else { return false; };
    parts.len() <= NV_MAX_PARTS && crate::expert_ops::nv_tile_layout_admits(first.rows, first.cols) && input & 3 == 0
        && parts.iter().all(|p| p.rows == first.rows && p.cols == first.cols && (1..=NV_MAX_PART_TOKENS).contains(&p.tokens) && (p.weight | p.scales) & 15 == 0)
        && parts.iter().map(|p| p.tokens).sum::<usize>() <= NV_MAX_MEAN_TOKENS * parts.len()
}
/// The `nv_cols` jobs of a column-tile group: one for each (part, token), in table order. A job reads the token's quantized row and
/// writes its token column of the part's `[row][token]` output (first word `first + token`, stride `tokens`).
#[cfg(not(target_arch = "nvptx64"))]
pub fn column_jobs(table: &[crate::expert_ops::GroupEntry], input: u64, output: u64, stride: usize) -> Vec<crate::expert_ops::Arguments> {
    let mut jobs = Vec::new();
    for e in table { for token in 0..e.shape.tokens as usize {
        jobs.push(crate::expert_ops::Arguments { input: [input + ((e.first_row as usize + token) * stride * 4) as u64, 4, 4, 4], weight: e.weight, scales: e.scales,
            output: output + ((e.first as usize + token) * 4) as u64, shape: crate::expert_ops::Shape { op: crate::expert_ops::Op::NvProject, ..e.shape } });
    } }
    jobs
}
/// The kernel finds the part of a job by its input row, so the parts' first input rows must be cumulative: part i starts where part i-1 ends,
/// from row 0 (the group builder takes `first_row` from the caller). A table that does not satisfy this keeps `expert_group`.
pub fn contiguous(entries: impl Iterator<Item = (usize, usize)>) -> bool {
    let mut next = 0usize;
    for (first_row, tokens) in entries { if first_row != next { return false; } next += tokens; }
    true
}
/// Blocks: `ceil(rows/16)` for each job (a job is one token of one part).
pub fn blocks(parts: &[Part]) -> Option<u32> {
    let first = parts.first()?; let jobs: usize = parts.iter().map(|p| p.tokens).sum();
    u32::try_from(first.rows.div_ceil(ROWS_PER_BLOCK).checked_mul(jobs)?).ok()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn part(rows: usize, cols: usize, tokens: usize) -> Part { Part { rows, cols, tokens, weight: 256, scales: 512 } }
    #[test]
    fn model_expert_launches_are_admitted() {
        // gate/up 1024 x 4096 and down 2048 x 2048, the parts of a 1, 2, 4, 8 and 16 row pass and a hot expert of a 15-token prompt.
        for (rows, cols) in [(1024, 4096), (2048, 2048)] {
            for counts in [vec![1; 7], vec![1; 15], vec![1; 30], { let mut v = vec![1; 80]; v.extend([2, 3, 4]); v }, { let mut v = vec![1; 100]; v.extend([15, 9, 2]); v }] {
                let parts: Vec<Part> = counts.iter().map(|&t| part(rows, cols, t)).collect();
                assert!(admitted(&parts, 256), "{rows}x{cols} {} parts", parts.len());
            }
        }
        assert_eq!(blocks(&[part(1024, 4096, 1); 8]), Some(64 * 8));
        assert_eq!(blocks(&[part(17, 4096, 2), part(17, 4096, 1)]), Some(2 * 3));
    }
    #[test]
    fn only_cumulative_input_rows_are_sent_to_the_job_per_token_kernels() {
        assert!(contiguous([(0, 2), (2, 1), (3, 5)].into_iter()) && contiguous([].into_iter()) && contiguous([(0, 1)].into_iter()));
        assert!(!contiguous([(0, 2), (3, 1)].into_iter()) && !contiguous([(0, 2), (1, 2)].into_iter()) && !contiguous([(1, 1)].into_iter()) && !contiguous([(2, 1), (0, 2)].into_iter()));
    }
    #[test]
    fn the_admitted_set_is_the_sweep_set() {
        for rows in 0..=2100 { assert_eq!(admitted(&[part(rows, 4096, 1)], 16), NV_ROWS.contains(&rows), "rows {rows}"); }
        for cols in 0..=9000 { assert_eq!(admitted(&[part(1024, cols, 1)], 16), NV_COLS.contains(&cols), "cols {cols}"); }
        for tokens in 0..=20 { assert_eq!(admitted(&[part(1024, 4096, tokens), part(1024, 4096, 1), part(1024, 4096, 1)], 16), (1..=16).contains(&tokens) && tokens + 2 <= 2 * 3, "tokens {tokens}"); }
        // Reviewer example: one part, rows 1, cols 8192 is in the list now (and swept); 8200 is not. Boundaries of parts and means.
        assert!(admitted(&[part(1, 8192, 1)], 16) && !admitted(&[part(1, 8200, 1)], 16) && !admitted(&[part(1, 16384, 1)], 16));
        assert!(!admitted(&[part(1024, 4096, 1); 129], 16) && admitted(&[part(1024, 4096, 1); 128], 16) && !admitted(&[part(1024, 4096, 3); 40], 16) && admitted(&[part(1024, 4096, 2); 64], 16));
        assert!(!admitted(&[], 0) && !admitted(&[part(1024, 4096, 0)], 16) && !admitted(&[part(1024, 4096, 17)], 16) && !admitted(&[part(1024, 4096, 2), part(1024, 2048, 2)], 16));
        assert!(!admitted(&[part(1024, 4096, 2), part(512, 4096, 2)], 16) && !admitted(&[part(1024, 4096, 2)], 8) && !admitted(&[Part { weight: 8, ..part(1024, 4096, 2) }], 16) && !admitted(&[Part { scales: 8, ..part(1024, 4096, 2) }], 16));
    }
}
