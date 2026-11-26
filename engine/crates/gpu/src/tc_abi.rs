//! Launch ABI of the tensor-core kernel family (the second kernel image, `sm_121a`). Host and device share this file.
//!
//! `fp4_cohort` (and its block-shape forms): the routed-expert NVFP4 products of one layer for any number of token
//! rows, on contract (b) (`tc_contract`). One part = one routed expert and the token rows that picked it.
//! `tc_nv_pack`: the quantizer's FP32 activation rows (17 words a 16-column block: 16 decoded E2M1 values and the
//! decoded UE4M3 block scale) to the packed codes and scale bytes the tensor-core instruction reads. A one-to-one map.

/// One routed expert of a launch.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvPart {
    /// Weight codes and block scales in the column-tile layout (`expert_ops::NV_TILE_ROWS`), or row-major (`layout` 0).
    pub weight: u64,
    pub scales: u64,
    /// Address of the part's first output word; word (row, token) is at output + 4 * (row * tokens + token).
    pub output: u64,
    /// First token row of the part in the activation arrays, and its token count.
    pub first: u32,
    pub tokens: u32,
    pub global: f32,
    pub activation: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvRows {
    /// `count` x `NvPart`.
    pub table: u64,
    /// Activation E2M1 codes, token-major, cols/2 bytes a token row (low nibble = even column).
    pub codes: u64,
    /// Activation UE4M3 block scale codes, token-major, cols/16 bytes a token row.
    pub scales: u64,
    pub count: u32,
    pub rows: u32,
    pub cols: u32,
    /// 1 = column tiles (production), 0 = row-major (bench only).
    pub layout: u32,
    /// 1 = a block with one live token does not read weight columns whose activation code is +-0.
    pub skip: u32,
    /// Bench switch: 1 = write the accumulator bits as they leave the K chain (no global scale, no BF16 round).
    pub raw: u32,
    /// The word function (`tc_contract`): 1 = one chain (first matrices), 4 = quarter chains (second matrices).
    pub chains: u32,
    /// Zero: the rectangular grid. Otherwise work_count NvTile records, grid
    /// (rows/128, work_count, 1), with unused records marked part=u32::MAX.
    pub work: u64,
    pub work_count: u32,
    /// PAIRED PARTS, work map only. 0 or 1: a work record names one part. S in 2..=8: a work record names the first
    /// of S consecutive parts of ONE expert (the same weights, scales, global and activation), and its token tiles
    /// run over the S parts' tokens in part order; each token keeps its own operand row and its own output word.
    pub merge: u32,
    /// DIRECT PLACE: 0 = a part's word (row, token) at its record's output + 4 * (row * tokens + token). Otherwise
    /// the u64 address of each operand row's word (row 0), indexed by the operand row (part first + token): the word
    /// (row, token) is at that address + 4 * row (built by `prompt_route_native` from the route's checked pairs).
    pub rowmap: u64,
}
#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct NvTile {pub part:u32,pub tile:u32}
const _:()={assert!(core::mem::size_of::<NvPart>()==40);assert!(core::mem::size_of::<NvRows>()==80);assert!(core::mem::offset_of!(NvRows,work)==56);assert!(core::mem::size_of::<NvTile>()==8);};
/// Threads a block and the grid of `fp4_cohort`: (ceil(max tokens / NV_ROWS_BLOCK_TOKENS), rows / 128, count).
pub const NV_ROWS_THREADS: u32 = 256;
pub const NV_ROWS_BLOCK_TOKENS: u32 = 128;
pub const NV_ROWS_MAX_COLS: u32 = 8192;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvPack {
    /// `rows` activation rows of `stride` FP32 words; `stride` must be cols / 16 * 17.
    pub input: u64,
    /// Output: rows * cols / 2 bytes of codes and rows * cols / 16 bytes of scale codes (both 4-byte aligned).
    pub codes: u64,
    pub scales: u64,
    pub rows: u32,
    pub cols: u32,
    pub stride: u32,
    pub pad: u32,
}
/// One thread a 16-column block: ceil(rows * cols / 16 / NV_PACK_THREADS) blocks.
pub const NV_PACK_THREADS: u32 = 128;
pub fn nv_pack_blocks(rows: usize, cols: usize) -> Option<u32> {
    u32::try_from(rows.checked_mul(cols / 16)?.div_ceil(NV_PACK_THREADS as usize)).ok()
}

/// `nv_quantize_pack` (the production image): the activation quantizer and `tc_nv_pack` in one pass. FP32 activation rows
/// straight to the packed operands, the same bytes as `tc_nv_pack` of the quantizer's rows (`expert_ops::quantize_packed`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvQuantizePack {
    /// `rows` activation rows of `cols` FP32 words.
    pub input: u64,
    /// Output: rows * cols / 2 bytes of codes and rows * cols / 16 bytes of scale codes (both 4-byte aligned).
    pub codes: u64,
    pub scales: u64,
    /// `rows` FP32 words: the calibrated input scale of the part that owns each row.
    pub activations: u64,
    pub rows: u32,
    pub cols: u32,
}
/// `nv_quantize_gather` (the production image): `nv_quantize_pack` of rows that are still where they were made. Row r of
/// the operands is row `index[r]` of `input`, so the rows are not copied into expert order first (`Op::PermuteRows`).
/// The same bytes as `nv_quantize_pack` of the copied rows: the copy moves words, and a row's bytes depend on that row
/// and its input scale only.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvQuantizeGather {
    /// `source_rows` activation rows of `cols` FP32 words.
    pub input: u64,
    /// Output: rows * cols / 2 bytes of codes and rows * cols / 16 bytes of scale codes (both 4-byte aligned).
    pub codes: u64,
    pub scales: u64,
    /// `rows` FP32 words: the calibrated input scale of the part that owns each row.
    pub activations: u64,
    /// `rows` FP32 words: the input row of each operand row, the table `Op::PermuteRows` reads (a float cast to an index).
    pub index: u64,
    pub rows: u32,
    pub cols: u32,
    pub source_rows: u32,
    pub pad: u32,
}
/// One thread a 16-column block: ceil(rows * cols / 16 / NV_QUANTIZE_PACK_THREADS) blocks.
pub const NV_QUANTIZE_PACK_THREADS: u32 = 128;
pub fn nv_quantize_pack_blocks(rows: usize, cols: usize) -> Option<u32> {
    u32::try_from(rows.checked_mul(cols / 16)?.div_ceil(NV_QUANTIZE_PACK_THREADS as usize)).ok()
}
