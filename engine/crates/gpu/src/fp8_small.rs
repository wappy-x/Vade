//! Admission and launch geometry for `fp8_rows_small`, the block-FP8 expert-row tile for a SMALL token count (a draft
//! verify pass): one block for each (part, token) job and 16 rows. Shape and alignment only. The admitted launches are the lists of
//! `small_sweep` (rows, columns, block shape) with at most 128 parts of 1 to 8 tokens each; the device sweep (`kernel-bench
//! --fp8-small`) walks every listed (rows, columns, block shape) against `expert_group` with the count families of
//! `kernel_fp8small.rs` and one seeded random count list for each listed shape and block shape. Any other launch keeps `fp8_rows_tile` or `expert_group`.
use crate::fp8_rows::Part;
use crate::small_sweep::{fp8_parts_swept, FP8_BLOCKS, FP8_COLS, FP8_MAX_BYTES, FP8_MAX_PART_TOKENS, FP8_MULTI_MAX_PART_TOKENS, FP8_ROWS};
pub const ROWS_PER_BLOCK: usize = 16;
pub const QN: usize = 8;
pub const DEPTH: usize = 4;
pub const THREADS: u32 = 32 + (ROWS_PER_BLOCK * QN) as u32;
/// Dynamic shared bytes (same layout as the decode `fp8_tile`): D stages of raw weights (16 rows of 128 bytes), scales (128 words)
/// and raw inputs (128 words), three input buffers of 128 words and two product buffers of 16 rows x 132 words.
pub const SHARED: u32 = (DEPTH * ROWS_PER_BLOCK * QN * 16 + DEPTH * ROWS_PER_BLOCK * QN * 4 + DEPTH * QN * 16 * 4 + 3 * QN * 16 * 4 + 2 * ROWS_PER_BLOCK * (QN * 16 + 4) * 4) as u32;
/// True when the whole launch may use `fp8_rows_small`: every part has the same listed rows and cols and a listed block shape, 1 to 8
/// tokens (each token is a job that decodes the weights again; at 16 tokens lane K's tile is faster), a 16-byte aligned weight and a
/// 4-byte aligned scale array; the activation rows are 16-byte aligned.
pub fn admitted(parts: &[Part], input: u64) -> bool {
    let Some(first) = parts.first() else { return false; };
    fp8_parts_swept(first.rows, first.cols, parts.len()) && FP8_ROWS.contains(&first.rows) && FP8_COLS.contains(&first.cols) && first.rows * first.cols <= FP8_MAX_BYTES && input & 15 == 0
        && parts.iter().all(|p| p.rows == first.rows && p.cols == first.cols && FP8_BLOCKS.contains(&(p.block_rows, p.block_cols))
            && (1..=FP8_MAX_PART_TOKENS).contains(&p.tokens) && p.weight & 15 == 0 && p.scales & 3 == 0)
}
/// Rows up to which ONE part of 3 or 4 tokens takes `fp8_rows_small` (it has one block for each 16 rows of each token) instead of `fp8_rows_multi` (one block for 32 rows).
pub const SMALL_RULE_ROWS: usize = 768;
/// `fp8_rows_multi`: the same lists and alignment, parts of 1 to 16 tokens; a thread owns a row, a warp up to 4 tokens, 128 threads, 32 rows a block.
pub const MULTI_THREADS: u32 = 128;
pub const MULTI_ROWS_PER_BLOCK: usize = 32;
pub fn multi_admitted(parts: &[Part], input: u64) -> bool {
    let Some(first) = parts.first() else { return false; };
    fp8_parts_swept(first.rows, first.cols, parts.len()) && FP8_ROWS.contains(&first.rows) && FP8_COLS.contains(&first.cols) && first.rows * first.cols <= FP8_MAX_BYTES && input & 15 == 0
        && parts.iter().all(|p| p.rows == first.rows && p.cols == first.cols && FP8_BLOCKS.contains(&(p.block_rows, p.block_cols))
            && (1..=FP8_MULTI_MAX_PART_TOKENS).contains(&p.tokens) && p.weight & 15 == 0 && p.scales & 3 == 0)
}
/// SHARED ROWS: 2 to `SHARED_MAX_PARTS` parts of ONE shape and block shape that all read the SAME token rows (block-FP8 matrices on one input: the
/// model's gate and up) in one `fp8_rows_multi` launch. The kernel finds a part by its block and reads the part's own first row, so parts on one
/// first row are parts like any other; `kernel-bench --fp8-small shared` walks exactly these tables (2 to 4 parts, 1 to 16 tokens) over every listed
/// rows x columns x block shape. The caller says that the first rows are equal (`small_choice::Group::shared_rows`).
pub const SHARED_MAX_PARTS: usize = 4;
pub fn shared_admitted(parts: &[Part], input: u64) -> bool {
    let Some(first) = parts.first() else { return false; };
    (2..=SHARED_MAX_PARTS).contains(&parts.len()) && multi_admitted(parts, input)
        && parts.iter().all(|p| p.tokens == first.tokens && p.block_rows == first.block_rows && p.block_cols == first.block_cols)
}
/// `fp8_rows_few`: ONE part of 1 to 4 tokens, the lists and alignment of `multi_admitted`; a thread owns a row, a warp one token, 128 threads, 16 rows a block.
pub const FEW_THREADS: u32 = 128;
pub const FEW_ROWS_PER_BLOCK: usize = 16;
pub const FEW_MAX_PART_TOKENS: usize = 4;
/// ONE part of 1 to 4 tokens (the model's FP8 projections are single matrices): exactly the launches `kernel-bench --fp8-small few` walks in full as count lists `[1]`, `[2]`, `[3]`, `[4]`
/// over every listed rows x columns x block shape (several parts, mixed block shapes and mixed counts are checked there too, but are not admitted: they keep `fp8_rows_multi`).
pub fn few_admitted(parts: &[Part], input: u64) -> bool { parts.len() == 1 && multi_admitted(parts, input) && parts[0].tokens <= FEW_MAX_PART_TOKENS }
/// Blocks of `fp8_rows_few`: `ceil(rows/16)` for each part.
pub fn few_blocks(parts: &[Part]) -> Option<u32> {
    let first = parts.first()?;
    u32::try_from(first.rows.div_ceil(FEW_ROWS_PER_BLOCK).checked_mul(parts.len())?).ok()
}
/// Where `fp8_rows_few` is taken instead of `fp8_rows_multi` (ONE part of 3 or 4 tokens): over more than SMALL_RULE_ROWS rows when the weights are at most 4,194,304
/// bytes (gate_up 1024 x 4096 and down 2048 x 2048). Since `fp8_rows_multi` spreads its tokens over its 4 warps, 4 tokens cost what 3 cost and the two kernels
/// are level on those shapes (one Spark 2026-11-12, kernel-bench --fp8-small time, few against multi: 41 against 39 and 34 against 33 us), while on the large
/// shapes multi wins at 3 AND at 4 tokens (o_proj 2048 x 16384: 215 against 172 us; q_b 8192 x 1536: 87 against 83), so 4 tokens follow the 3-token rule.
pub fn few_wins(rows: usize, cols: usize, longest_part_tokens: usize) -> bool {
    rows > SMALL_RULE_ROWS && (3..=4).contains(&longest_part_tokens) && rows * cols <= 4_194_304
}
/// Blocks of `fp8_rows_multi`: `ceil(rows/32)` for each part.
pub fn multi_blocks(parts: &[Part]) -> Option<u32> {
    let first = parts.first()?;
    u32::try_from(first.rows.div_ceil(MULTI_ROWS_PER_BLOCK).checked_mul(parts.len())?).ok()
}
/// Blocks: `ceil(rows/16)` for each job (a job is one token of one part).
pub fn blocks(parts: &[Part]) -> Option<u32> {
    let first = parts.first()?; let jobs: usize = parts.iter().map(|p| p.tokens).sum();
    u32::try_from(first.rows.div_ceil(ROWS_PER_BLOCK).checked_mul(jobs)?).ok()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn part(rows: usize, cols: usize, tokens: usize) -> Part { Part { rows, cols, block_rows: 128, block_cols: 128, tokens, weight: 256, scales: 512 } }
    #[test]
    fn model_fp8_projections_are_admitted_with_the_decode_tile_geometry() {
        assert_eq!(THREADS, 160);
        assert_eq!(SHARED, 30720);
        // gate_up 1024 x 4096, down 2048 x 2048, o_proj 2048 x 16384, q_b 8192 x 1536, q_a 768 x 4096, kv_a 256 x 4096 (the model's rank-local FP8 rows), 1 to 8 tokens.
        for (rows, cols) in [(1024, 4096), (2048, 2048), (2048, 16384), (8192, 1536), (768, 4096), (256, 4096), (288, 4096)] {
            for tokens in 1..=8 { assert!(admitted(&[part(rows, cols, tokens)], 256), "{rows}x{cols} tokens {tokens}"); }
        }
        assert_eq!(blocks(&[part(1024, 4096, 4)]), Some(64 * 4));
        assert_eq!(blocks(&[part(17, 4096, 2), part(17, 4096, 1)]), Some(2 * 3));
    }
    #[test]
    fn the_multi_admitted_set_is_the_sweep_set() {
        for rows in 0..=8200 { assert_eq!(multi_admitted(&[part(rows, 4096, 1)], 16), FP8_ROWS.contains(&rows), "rows {rows}"); }
        for cols in 0..=16400 { assert_eq!(multi_admitted(&[part(1024, cols, 1)], 16), FP8_COLS.contains(&cols) && 1024 * cols <= FP8_MAX_BYTES, "cols {cols}"); }
        for tokens in 0..=20 { assert_eq!(multi_admitted(&[part(1024, 4096, tokens)], 16), (1..=16).contains(&tokens), "tokens {tokens}"); }
        for br in 0..=200 { for bc in 0..=200 { assert_eq!(multi_admitted(&[Part { block_rows: br, block_cols: bc, ..part(1024, 4096, 5) }], 16), FP8_BLOCKS.contains(&(br, bc))); } }
        for (rows, cols) in [(1024, 4096), (2048, 2048), (2048, 16384), (8192, 1536), (768, 4096), (256, 4096), (288, 4096)] { for tokens in 1..=16 { assert!(multi_admitted(&[part(rows, cols, tokens)], 256)); } }
        assert!(!multi_admitted(&[], 0) && !multi_admitted(&[part(1024, 4096, 2); 129], 16) && !multi_admitted(&[part(1024, 4096, 2)], 8) && !multi_admitted(&[part(1024, 4096, 17)], 16));
        assert!(!multi_admitted(&[part(1024, 4096, 2), part(512, 4096, 2)], 16) && !multi_admitted(&[Part { weight: 8, ..part(1024, 4096, 2) }], 16) && !multi_admitted(&[Part { scales: 2, ..part(1024, 4096, 2) }], 16));
        assert_eq!(multi_blocks(&[part(1024, 4096, 5)]), Some(32));
        assert_eq!(multi_blocks(&[part(33, 4096, 5), part(33, 4096, 2)]), Some(4));
        assert_eq!((MULTI_THREADS, MULTI_ROWS_PER_BLOCK), (128, 32));
    }
    #[test]
    fn the_few_admitted_set_is_the_sweep_set() {
        for rows in 0..=8200 { assert_eq!(few_admitted(&[part(rows, 4096, 3)], 16), FP8_ROWS.contains(&rows), "rows {rows}"); }
        for cols in 0..=16400 { assert_eq!(few_admitted(&[part(1024, cols, 3)], 16), FP8_COLS.contains(&cols) && 1024 * cols <= FP8_MAX_BYTES, "cols {cols}"); }
        for tokens in 0..=20 { assert_eq!(few_admitted(&[part(1024, 4096, tokens)], 16), (1..=4).contains(&tokens), "tokens {tokens}"); }
        for br in 0..=200 { for bc in 0..=200 { assert_eq!(few_admitted(&[Part { block_rows: br, block_cols: bc, ..part(1024, 4096, 3) }], 16), FP8_BLOCKS.contains(&(br, bc))); } }
        assert!(!few_admitted(&[], 0) && !few_admitted(&[part(1024, 4096, 2); 129], 16) && !few_admitted(&[part(1024, 4096, 2)], 8) && !few_admitted(&[part(1024, 4096, 2), part(1024, 4096, 5)], 16));
        // Only ONE part: the swept count lists are [1], [2], [3], [4]; two parts (even of the same shape) keep multi.
        assert!(!few_admitted(&[part(1024, 4096, 3), part(1024, 4096, 4)], 16) && !few_admitted(&[part(1024, 4096, 2), part(1024, 4096, 2)], 16) && few_admitted(&[part(1024, 4096, 4)], 16));
        assert!(!few_admitted(&[part(1024, 4096, 2), part(512, 4096, 2)], 16) && !few_admitted(&[Part { weight: 8, ..part(1024, 4096, 2) }], 16) && !few_admitted(&[Part { scales: 2, ..part(1024, 4096, 2) }], 16));
        assert_eq!(few_blocks(&[part(1024, 4096, 4)]), Some(64));
        assert_eq!(few_blocks(&[part(17, 4096, 4), part(17, 4096, 1)]), Some(4), "the block count is defined for any table");
        assert_eq!((FEW_THREADS, FEW_ROWS_PER_BLOCK), (128, 16));
        // where it wins
        assert!(few_wins(1024, 4096, 3) && few_wins(2048, 2048, 3) && !few_wins(2048, 16384, 3) && !few_wins(8192, 1536, 3) && few_wins(1024, 4096, 4) && few_wins(2048, 2048, 4) && !few_wins(2048, 16384, 4) && !few_wins(8192, 1536, 4)
            && !few_wins(768, 4096, 4) && !few_wins(1024, 4096, 2) && !few_wins(1024, 4096, 5));
    }
    #[test]
    fn shared_rows_admit_two_to_four_parts_of_one_shape_block_shape_and_count() {
        for parts in 0..=6usize { for tokens in 0..=17usize {
            assert_eq!(shared_admitted(&vec![part(1024, 4096, tokens); parts], 16), (2..=SHARED_MAX_PARTS).contains(&parts) && (1..=16).contains(&tokens), "{parts} parts of {tokens} tokens");
        } }
        // Every listed shape of the model, 2 parts, 1 to 16 tokens.
        for (rows, cols) in [(1024, 4096), (2048, 2048), (2048, 16384), (8192, 1536), (768, 4096), (256, 4096), (288, 4096)] { for tokens in 1..=16 { assert!(shared_admitted(&[part(rows, cols, tokens); 2], 256)); } }
        // One shape, one block shape, one count; the alignment and the lists of `multi_admitted`.
        let two = |second: Part| shared_admitted(&[part(1024, 4096, 4), second], 16);
        assert!(two(part(1024, 4096, 4)) && !two(part(1024, 4096, 3)) && !two(part(2048, 4096, 4)) && !two(part(1024, 2048, 4)));
        assert!(!two(Part { block_rows: 1, block_cols: 16, ..part(1024, 4096, 4) }) && !two(Part { weight: 8, ..part(1024, 4096, 4) }) && !two(Part { scales: 2, ..part(1024, 4096, 4) }));
        assert!(shared_admitted(&[Part { block_rows: 1, block_cols: 16, ..part(1024, 4096, 4) }; 2], 16) && !shared_admitted(&[part(1024, 4096, 4); 2], 8) && !shared_admitted(&[part(130, 4096, 4); 2], 16));
        assert_eq!(multi_blocks(&[part(1024, 4096, 8); 2]), Some(64));
    }
    #[test]
    fn the_admitted_set_is_the_sweep_set() {
        for rows in 0..=8200 { assert_eq!(admitted(&[part(rows, 4096, 1)], 16), FP8_ROWS.contains(&rows), "rows {rows}"); }
        for cols in 0..=16400 { assert_eq!(admitted(&[part(1024, cols, 1)], 16), FP8_COLS.contains(&cols) && 1024 * cols <= FP8_MAX_BYTES, "cols {cols}"); }
        for tokens in 0..=20 { assert_eq!(admitted(&[part(1024, 4096, tokens)], 16), (1..=8).contains(&tokens), "tokens {tokens}"); }
        for br in 0..=200 { for bc in 0..=200 { assert_eq!(admitted(&[Part { block_rows: br, block_cols: bc, ..part(1024, 4096, 2) }], 16), FP8_BLOCKS.contains(&(br, bc))); } }
        assert!(!admitted(&[], 0) && !admitted(&[part(16385, 4096, 1)], 16) && !admitted(&[part(8192, 16384 + 16, 1)], 16));
        assert!(!admitted(&[part(1024, 4096, 2), part(1024, 2048, 2)], 16) && !admitted(&[part(1024, 4096, 2), part(512, 4096, 2)], 16) && !admitted(&[part(1024, 4096, 2); 129], 16) && !admitted(&[part(1024, 4096, 2)], 8));
        assert!(!admitted(&[Part { weight: 8, ..part(1024, 4096, 2) }], 16) && !admitted(&[Part { scales: 2, ..part(1024, 4096, 2) }], 16));
    }
}
