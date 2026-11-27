//! Block-FP8 prompt rows on tensor cores. The word is the contract of `tc_f8` (`docs/tc-dense-contract.md`, section
//! Block FP8), made by the one body of `fp8_tc`: the same word as the decode step (`fp8_tile`), the small row groups
//! (`fp8_rows_small`, `fp8_rows_few`, `fp8_rows_multi`), the generic body (`expert_group`) and the host reference.
//!
//! A block takes 16 weight rows a warp and 32 token rows of one part: each warp holds four tensor-core tiles of 8
//! tokens, and the warps of the block share the staged token rows.
//!
//! Launch: entry `fp8_rows_tile`, the same `GroupArguments` and table as `expert_group`, grid
//! (ceil(rows / (16 * warps)), ceil(max_tokens / 32), parts): the row groups of one token tile are neighbours, so the
//! token rows they share stay in the cache. Whole warps a block (`fp8_rows::THREADS`), dynamic shared
//! `fp8_rows::shared(max_tokens)` bytes. Needs op Fp8Rows, cols % 16 == 0, block_cols a power of two of at least 16,
//! weight and input 16-byte aligned, scales 4-byte aligned, one word an output; else keep `expert_group`.
use crate::expert_ops::GroupArguments;

/// Stages of the prompt tile.
const D:usize=3;
/// Tensor-core tiles of 8 tokens a warp.
const NF:usize=4;

#[no_mangle]
pub extern "ptx-kernel" fn fp8_rows_tile(p:GroupArguments) {
    // SAFETY: Only the three block indices are read here. Every check of the launch, its parts and its operands, and
    // every memory access, is in `fp8_tc::group` and `fp8_tc::run` (their contracts): the host checks every part
    // (shape, raw owners, input span) and keeps all owners and the table alive through stream completion, as for
    // expert_group; each (part, row, token) output word has exactly one writing thread.
    let (row_group,tile,part)=unsafe {(core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize,core::arch::nvptx::_block_idx_z() as usize)};
    crate::fp8_tc::group::<NF,D>(p,part,tile,row_group);
}

/// The third form of the prompt tile (`fp8_tc::tile3_body`): the word of `fp8_rows_tile`, 4 tiles of 8 token rows a
/// warp, the tensor core decodes the weights, the next stage is prefetched, the waiting range sum in registers and
/// the low join in the word's output slot. Grid (row groups of 16 rows a warp, token tiles of 32 rows, parts);
/// dynamic shared `fp8_tc::tile3_shared(4, warps, 0)` = 8,192 bytes. The host launches it where
/// `fp8_rows::third` gives a grid (scale blocks of 128 columns and a multiple of 16 rows, whole stages).
#[no_mangle]
pub extern "ptx-kernel" fn fp8_rows_tile3(p:GroupArguments) {
    // SAFETY: Only the grid/block indices are read here. Every check of the launch, its parts and its operands, and
    // every memory access, is in `fp8_tc::group_tile3` and `fp8_tc::tile3_body` (their contracts): the host checks
    // every part (shape, raw owners, input span) and keeps all owners and the table alive through stream completion,
    // as for `fp8_rows_tile`; each (part, row, token) output word has exactly one thread that writes and reads it.
    // L2 CTA swizzling is a pure bijection on `(row_group, tile)` within `[0..gx) x [0..gy)` that pins `GROUP_R = 4`
    // row groups in L2 across all token tiles without changing any word.
    let (bx,by,part,gx,gy)=unsafe {(
        core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize,core::arch::nvptx::_block_idx_z() as usize,
        core::arch::nvptx::_grid_dim_x() as usize,core::arch::nvptx::_grid_dim_y() as usize,
    )};
    const GROUP_R:usize=4;
    let (row_group,tile)=if gx>=GROUP_R && gy>=4 {
        let bid=by*gx+bx;
        let group_blocks=GROUP_R*gy;
        let group=bid/group_blocks;
        let first_r=group*GROUP_R;
        let group_r=(gx-first_r).min(GROUP_R);
        let rem=bid-group*group_blocks;
        (first_r+rem%group_r,rem/group_r)
    } else {
        (bx,by)
    };
    crate::fp8_tc::group_tile3::<4,0,2,true>(p,part,tile,row_group);
}

/// Wide-prompt third form (`NF=8`, `PARK=2`): 8 tiles of 8 token rows (64 tokens) and 128 weight rows per 256-thread
/// block, parking both range 0/2 and the low join strictly in dynamic shared memory (`98,304 B`, zero global parking,
/// 2x fewer weight reads and tensor-core E4M3 decodes than `fp8_rows_tile3`). Bit-identical word.
#[no_mangle]
pub extern "ptx-kernel" fn fp8_rows_tile3_n8(p:GroupArguments) {
    // SAFETY: Only the three block indices are read here. Every check of the launch, its parts and its operands, and
    // every memory access, is in `fp8_tc::group_tile3` and `fp8_tc::tile3_body` (their contracts): the host checks
    // every part (shape, raw owners, input span) and keeps all owners and the table alive through stream completion,
    // as for `fp8_rows_tile`; each (part, row, token) output word has exactly one writer.
    let (row_group,tile,part)=unsafe {(core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize,core::arch::nvptx::_block_idx_z() as usize)};
    crate::fp8_tc::group_tile3::<8,0,2,true>(p,part,tile,row_group);
}
