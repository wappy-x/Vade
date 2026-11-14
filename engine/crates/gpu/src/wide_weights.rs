//! Key ranges and word addresses of the BF16 weight tile of the wide global prefill (`Backend::prefill_attention_wide`:
//! 256 or more rows, head width 256 or 512, no window; Gemma's global layers). Its softmax writes the tile
//! (`attention_bounded_softmax`, form 2, `kernels/src/decode_attention.rs`) and its values pass reads it
//! (`prefill_attention_wide_values`, `kernels/src/attention.rs`). Both kernels take their bounds here, so
//! `tests/gemma_prefix_weights.rs` checks on the host the arithmetic the device runs.
//!
//! The values pass has no causal mask on this tile. It reads whole tiles of 64 keys from key 0, the same tiles for every row
//! of a block of 64 rows, and it loads the V row of every key the block's LAST row attends. Nothing clears the tile between
//! calls. So each word it reads must be written by the softmax of the same call: the row's weight, or zero past the row's
//! own keys.

/// Rows of one values block, and keys of one weight tile.
pub const BLOCK: usize = 64;
/// One past the last key whose V row the values pass loads for the block of rows starting at `block`.
#[inline(always)]
pub const fn block_end(position: usize, rows: usize, block: usize) -> usize {
    position + if block + BLOCK < rows { block + BLOCK } else { rows }
}
/// One past the last key whose weight the values pass reads, for every row of the block starting at `block`.
// The plain form on purpose: with `div_ceil` here the NVPTX compiler emits two unrelated kernels with other instructions.
#[allow(clippy::manual_div_ceil)]
#[inline(always)]
pub const fn read_end(position: usize, rows: usize, block: usize) -> usize {
    (block_end(position, rows, block) + BLOCK - 1) / BLOCK * BLOCK
}
/// One past the last key whose weight the softmax writes for `row`: the row's own keys, then zeros up to the last key the
/// values pass reads for the row's block. The row's own next multiple of 64 keys is the same bound only when `position` is a
/// multiple of 64. After a kept prompt of any other length it stops one tile early for the first rows of each block, and the
/// values pass then multiplies whatever the scratch held by the V rows of later tokens.
#[inline(always)]
pub const fn write_end(position: usize, rows: usize, row: usize) -> usize {
    read_end(position, rows, row / BLOCK * BLOCK)
}
/// The u16 word of the weight of (`row`, `head`, `key`): tensor-core A fragments of 16 rows by 16 keys, `capacity` keys a row.
#[inline(always)]
pub const fn word(row: usize, head: usize, heads: usize, capacity: usize, key: usize) -> usize {
    let tile = ((row / 16 * heads + head) * (capacity / 16) + key / 16) * 256;
    tile + (row % 8 * 4 + key % 8 / 2) * 8 + row % 16 / 8 * 2 + key % 16 / 8 * 4 + key % 2
}
