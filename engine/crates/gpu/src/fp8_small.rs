//! Admission and launch geometry for `fp8_rows_small`, the block-FP8 expert-row tile for a SMALL token count (a draft
//! verify pass): one block for each (part, token) job and 16 rows. Shape and alignment only. The admitted launches are the lists of
//! `small_sweep` (rows, columns, block shape) with at most 128 parts of 1 to 8 tokens each; the device sweep (`kernel-bench
//! --fp8-small`) walks every listed (rows, columns, block shape) against `expert_group` with the count families of
//! `kernel_fp8small.rs` and one seeded random count list for each listed shape and block shape. Any other launch keeps `fp8_rows_tile` or `expert_group`.
use crate::fp8_rows::Part;
use crate::small_sweep::{fp8_parts_swept, FP8_BLOCKS, FP8_COLS, FP8_MAX_BYTES, FP8_MAX_PART_TOKENS, FP8_MULTI_MAX_PART_TOKENS, FP8_ROWS};
/// TENSOR-CORE REBASE (block FP8): the three small entries (`fp8_rows_small`, `fp8_rows_few`, `fp8_rows_multi`) run the
/// row body of `kernels/src/fp8_tc.rs` (`row_body`) with ONE geometry: a block is one tile of `ROWS_PER_BLOCK` weight rows
/// and one tensor-core tile of at most `TOKENS_PER_BLOCK` token rows of one part, on `warps` warps. So 2 to 8 token
/// rows read the weights once, as one row does.
pub const ROWS_PER_BLOCK: usize = 16;
pub const TOKENS_PER_BLOCK: usize = 8;
/// The most scale blocks a row of a part has, and the longest part's tokens inside one block.
fn extent(parts: &[Part]) -> (u32, u32) {
    (parts.iter().map(|p| p.cols.div_ceil(p.block_cols.max(1))).max().unwrap_or(1) as u32, parts.iter().map(|p| p.tokens).max().unwrap_or(1).clamp(1, TOKENS_PER_BLOCK) as u32)
}
/// Warps of a block of a launch of these parts (`expert_ops::fp8_row_warps`: a speed choice; the word does not depend on it).
pub fn warps(parts: &[Part]) -> u32 { let (blocks, most) = extent(parts); crate::expert_ops::fp8_row_warps(blocks, most, false) }
/// Threads of a block of a launch of these parts.
pub fn threads(parts: &[Part]) -> u32 { 32 * warps(parts) }
/// Dynamic shared bytes of a launch: the row body's bytes for the part with the most scale blocks and the longest part's tokens.
pub fn shared(parts: &[Part]) -> u32 { let (blocks, most) = extent(parts); crate::expert_ops::fp8_row_shared(warps(parts), blocks, most, false) }
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
pub const MULTI_ROWS_PER_BLOCK: usize = ROWS_PER_BLOCK;
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
pub const FEW_ROWS_PER_BLOCK: usize = ROWS_PER_BLOCK;
pub const FEW_MAX_PART_TOKENS: usize = 4;
/// ONE part of 1 to 4 tokens (the model's FP8 projections are single matrices): exactly the launches `kernel-bench --fp8-small few` walks in full as count lists `[1]`, `[2]`, `[3]`, `[4]`
/// over every listed rows x columns x block shape (several parts, mixed block shapes and mixed counts are checked there too, but are not admitted: they keep `fp8_rows_multi`).
pub fn few_admitted(parts: &[Part], input: u64) -> bool { parts.len() == 1 && multi_admitted(parts, input) && parts[0].tokens <= FEW_MAX_PART_TOKENS }
/// Blocks of `fp8_rows_few`: `ceil(rows/16)` for each part.
pub fn few_blocks(parts: &[Part]) -> Option<u32> { blocks(parts) }
/// Where `fp8_rows_few` is taken instead of `fp8_rows_multi` (ONE part of 3 or 4 tokens): over more than SMALL_RULE_ROWS rows when the weights are at most 4,194,304
/// bytes (gate_up 1024 x 4096 and down 2048 x 2048). Since `fp8_rows_multi` spreads its tokens over its 4 warps, 4 tokens cost what 3 cost and the two kernels
/// are level on those shapes (one Spark 2026-11-12, kernel-bench --fp8-small time, few against multi: 41 against 39 and 34 against 33 us), while on the large
/// shapes multi wins at 3 AND at 4 tokens (o_proj 2048 x 16384: 215 against 172 us; q_b 8192 x 1536: 87 against 83), so 4 tokens follow the 3-token rule.
pub fn few_wins(rows: usize, cols: usize, longest_part_tokens: usize) -> bool {
    rows > SMALL_RULE_ROWS && (3..=4).contains(&longest_part_tokens) && rows * cols <= 4_194_304
}
/// Blocks of `fp8_rows_multi`: `ceil(rows/32)` for each part.
pub fn multi_blocks(parts: &[Part]) -> Option<u32> { blocks(parts) }
/// Blocks: `ceil(rows/16)` for each job (a job is one token of one part).
pub fn blocks(parts: &[Part]) -> Option<u32> {
    // The grid `fp8_tc::linear` reads: parts x token tiles of the longest part x row groups, in one dimension.
    let first = parts.first()?; let most = parts.iter().map(|p| p.tokens).max()?;
    u32::try_from(parts.len().checked_mul(most.div_ceil(TOKENS_PER_BLOCK))?.checked_mul(first.rows.div_ceil(ROWS_PER_BLOCK))?).ok()
}
/// `fp8_rows_share`: parts of ONE token count of 2 to 8, the lists and alignment of `multi_admitted`. A block takes 16 rows of a part, as the one-row tile
/// does; 128 producer threads copy and decode a weight once for all tokens, and a chain warp holds 2 tokens (`kernels/src/fp8_rows_share.rs`).
pub const SHARE_ROWS_PER_BLOCK: usize = 16;
pub const SHARE_MIN_TOKENS: usize = 2;
pub const SHARE_MAX_TOKENS: usize = 8;
/// Parts of a launch whose parts all read the same rows (gate and up of the shared expert).
pub const SHARE_SHARED_PARTS: usize = 2;
/// Threads of a block: a chain warp for each 2 tokens, then the 128 producers.
pub fn share_threads(tokens: usize) -> u32 { (32 * tokens.div_ceil(2) + 128) as u32 }
/// Dynamic shared bytes of a block: 3 stages of raw weights (16 rows of 128 bytes) and of scales (128 words), two buffers of first products
/// (16 rows of 132 words) and 3 stages of activations (a row of 132 words for each token slot of the chain warps).
pub fn share_shared(tokens: usize) -> u32 { (3 * 16 * 128 + 3 * 128 * 4 + 2 * 16 * 132 * 4 + (if tokens > 4 { 2 } else { 3 }) * 2 * tokens.div_ceil(2) * 132 * 4) as u32 }
/// The launches `kernel-bench --fp8-small share` walks in full over every listed rows x columns x block shape: ONE part of 2 to 8 tokens, or
/// (`shared_rows`) exactly two parts of one shape, block shape and count on the same rows. Any other table keeps the older kernels.
pub fn share_admitted(parts: &[Part], input: u64, shared_rows: bool) -> bool {
    let Some(first) = parts.first() else { return false; };
    parts.len() == (if shared_rows { SHARE_SHARED_PARTS } else { 1 }) && (SHARE_MIN_TOKENS..=SHARE_MAX_TOKENS).contains(&first.tokens) && multi_admitted(parts, input)
        && parts.iter().all(|p| p.tokens == first.tokens && p.block_rows == first.block_rows && p.block_cols == first.block_cols)
}
/// Blocks of `fp8_rows_share`: `ceil(rows/16)` for each part.
pub fn share_blocks(parts: &[Part]) -> Option<u32> {
    let first = parts.first()?;
    u32::try_from(first.rows.div_ceil(SHARE_ROWS_PER_BLOCK).checked_mul(parts.len())?).ok()
}
/// Where `fp8_rows_share` is faster than the entry it would replace: the six block-FP8 tables a rank serves with two ranks, timed on one
/// Spark (`kernel-bench --fp8-small {OLD} {X} sharetime`, us a launch; the numbers and run tags are in `docs/measurements/share-rows.md`).
/// The shared expert's gate and up (two parts on the same rows, 1,024 x 4,096) at 2 to 6 tokens; its down (2,048 x 2,048) and q_b
/// (8,192 x 1,536) at 2 to 8; o_proj (2,048 x 16,384) at 2 to 5; q_a (768 x 4,096) at 3 to 8; kv_a (256 x 4,096) at 5 to 8.
/// It LOSES and is not chosen for gate and up at 7 and 8 tokens and o_proj at 6 to 8 (four chain warps a block: the chains are the bound),
/// q_a at 2 tokens and kv_a at 2 to 4 (16 to 48 blocks: the chain of one block is the bound and the small kernel has more blocks).
/// Any other table (another rank count, another part count) keeps the entries of main.
pub fn share_wins(rows: usize, cols: usize, parts: usize, tokens: usize) -> bool {
    matches!((rows, cols, parts, tokens), (1024, 4096, 2, 2..=6) | (2048, 2048, 1, 2..=8) | (2048, 16384, 1, 2..=5) | (8192, 1536, 1, 2..=8) | (768, 4096, 1, 3..=8) | (256, 4096, 1, 5..=8))
}
/// The existing few-row form's measured two-row winners. Three/four rows retain the base rule.
pub fn two_wins(rows:usize,cols:usize)->bool {
    matches!((rows,cols),(1024,4096)|(2048,2048)|(2048,16384)|(8192,1536))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn part(rows: usize, cols: usize, tokens: usize) -> Part { Part { rows, cols, block_rows: 128, block_cols: 128, tokens, weight: 256, scales: 512 } }
    #[test]
    fn model_fp8_projections_are_admitted_with_the_decode_tile_geometry() {
        // 8 warps a block; one token of 32 scale blocks: the join slots and 4 handing warps of 4 blocks of 128 bytes.
        assert_eq!(threads(&[part(1024, 4096, 1)]), 256);
        assert_eq!(shared(&[part(1024, 4096, 1)]), 2048 + 4 * 4 * 128);
        // Eight tokens of o_proj's 128 scale blocks: 4 handing warps of 16 blocks of 512 bytes, inside the limit.
        assert_eq!((threads(&[part(2048, 16384, 8)]), shared(&[part(2048, 16384, 8)])), (256, 2048 + 4 * 16 * 512));
        for (rows, cols) in [(1024, 4096), (2048, 2048), (2048, 16384), (8192, 1536), (768, 4096), (256, 4096), (288, 4096)] {
            for tokens in 1..=16 { assert!(shared(&[part(rows, cols, tokens)]) <= crate::expert_ops::FP8_ROW_SHARED_LIMIT && crate::expert_ops::fp8_row_threads(threads(&[part(rows, cols, tokens)]))); }
        }
        // gate_up 1024 x 4096, down 2048 x 2048, o_proj 2048 x 16384, q_b 8192 x 1536, q_a 768 x 4096, kv_a 256 x 4096 (the model's rank-local FP8 rows), 1 to 8 tokens.
        for (rows, cols) in [(1024, 4096), (2048, 2048), (2048, 16384), (8192, 1536), (768, 4096), (256, 4096), (288, 4096)] {
            for tokens in 1..=8 { assert!(admitted(&[part(rows, cols, tokens)], 256), "{rows}x{cols} tokens {tokens}"); }
        }
        assert_eq!(blocks(&[part(1024, 4096, 4)]), Some(64));
        assert_eq!(blocks(&[part(17, 4096, 2), part(17, 4096, 1)]), Some(4));
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
        assert_eq!(multi_blocks(&[part(1024, 4096, 5)]), Some(64));
        assert_eq!(multi_blocks(&[part(33, 4096, 5), part(33, 4096, 2)]), Some(6));
        assert_eq!(MULTI_ROWS_PER_BLOCK, 16);
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
        assert_eq!(FEW_ROWS_PER_BLOCK, 16);
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
        assert_eq!(multi_blocks(&[part(1024, 4096, 8); 2]), Some(128));
    }
    #[test]
    fn the_share_admitted_set_is_one_part_or_two_parts_on_the_same_rows_of_2_to_8_tokens() {
        for rows in 0..=8200 { assert_eq!(share_admitted(&[part(rows, 4096, 3)], 16, false), FP8_ROWS.contains(&rows), "rows {rows}"); }
        for cols in 0..=16400 { assert_eq!(share_admitted(&[part(1024, cols, 3)], 16, false), FP8_COLS.contains(&cols) && 1024 * cols <= FP8_MAX_BYTES, "cols {cols}"); }
        for br in 0..=200 { for bc in 0..=200 { assert_eq!(share_admitted(&[Part { block_rows: br, block_cols: bc, ..part(1024, 4096, 3) }], 16, false), FP8_BLOCKS.contains(&(br, bc))); } }
        for parts in 0..=5usize { for tokens in 0..=17usize { for shared in [false, true] {
            assert_eq!(share_admitted(&vec![part(1024, 4096, tokens); parts], 16, shared), parts == (if shared { 2 } else { 1 }) && (2..=8).contains(&tokens), "{parts} parts of {tokens} tokens, shared {shared}");
        } } }
        // Every block-FP8 shape of the model, 2 to 8 tokens: one part, and two parts on the same rows.
        for (rows, cols) in [(1024, 4096), (2048, 2048), (2048, 16384), (8192, 1536), (768, 4096), (256, 4096)] { for tokens in 2..=8 {
            assert!(share_admitted(&[part(rows, cols, tokens)], 256, false) && share_admitted(&[part(rows, cols, tokens); 2], 256, true));
        } }
        // One count, one shape and one block shape; the alignment of `multi_admitted`.
        let two = |second: Part| share_admitted(&[part(1024, 4096, 4), second], 16, true);
        assert!(two(part(1024, 4096, 4)) && !two(part(1024, 4096, 3)) && !two(part(2048, 4096, 4)) && !two(Part { block_rows: 1, block_cols: 16, ..part(1024, 4096, 4) }));
        assert!(!two(Part { weight: 8, ..part(1024, 4096, 4) }) && !two(Part { scales: 2, ..part(1024, 4096, 4) }) && !share_admitted(&[part(1024, 4096, 4); 2], 8, true) && !share_admitted(&[part(1024, 4096, 4)], 8, false));
        // Geometry: 16 rows a block for each part; a chain warp for each 2 tokens before the 128 producers.
        assert_eq!(share_blocks(&[part(1024, 4096, 4)]), Some(64));
        assert_eq!(share_blocks(&[part(1024, 4096, 8); 2]), Some(128));
        assert_eq!(share_blocks(&[part(17, 4096, 2)]), Some(2));
        assert_eq!((2..=8).map(share_threads).collect::<Vec<_>>(), [160, 192, 192, 224, 224, 256, 256]);
        assert_eq!((2..=8).map(share_shared).collect::<Vec<_>>(), [27744, 30912, 30912, 30912, 30912, 33024, 33024]);
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
