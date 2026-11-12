//! `tc_nv_pack`: the activation quantizer's FP32 rows to the packed operands of the NVFP4 tensor-core instruction.
//! A block of an input row is 17 words: 16 decoded E2M1 values (+-0, +-0.5 .. +-6) and the decoded UE4M3 block scale.
//! Output: 8 bytes of E2M1 codes a block (low nibble = even column) and one UE4M3 scale code a block. The map is
//! one-to-one (`tc_contract::e2m1_code_f32`, `ue4m3_code_f32`); a row's bytes depend on that row only.
use crate::tc_abi::{NvPack, NV_PACK_THREADS};
use crate::tc_contract as tc;

/// SAFETY: the host (`Backend::nv_rows_tc`) checks before the launch: `input` holds rows * stride FP32 words with stride == cols / 16 * 17, cols a multiple of 16;
/// `codes` holds rows * cols / 2 bytes and `scales` rows * cols / 16 bytes, both 4-byte aligned, private to this
/// launch and disjoint from the input; all owners live through stream completion. Thread i owns block i of the
/// rows * cols / 16 blocks: it reads that block's 17 words and writes that block's two code words and one scale byte.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn tc_nv_pack(p: NvPack) {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
    if _block_dim_x() as u32 != NV_PACK_THREADS || p.cols == 0 || p.cols % 16 != 0 || p.stride != p.cols / 16 * 17 || (p.codes | p.scales | p.input) & 3 != 0 {
        core::arch::asm!("trap;");
    }
    let blocks = p.cols as usize / 16;
    let i = _block_idx_x() as usize * NV_PACK_THREADS as usize + _thread_idx_x() as usize;
    if i >= p.rows as usize * blocks { return; }
    let (row, b) = (i / blocks, i % blocks);
    let src = (p.input as *const u32).add(row * p.stride as usize + b * 17);
    let (mut lo, mut hi) = (0u32, 0u32);
    let mut e = 0;
    while e < 8 {
        lo |= tc::e2m1_code_f32(*src.add(e)) << (4 * e);
        hi |= tc::e2m1_code_f32(*src.add(8 + e)) << (4 * e);
        e += 1;
    }
    let dst = (p.codes as *mut u32).add(row * (blocks * 2) + b * 2);
    *dst = lo;
    *dst.add(1) = hi;
    *(p.scales as *mut u8).add(row * blocks + b) = tc::ue4m3_code_f32(*src.add(16)) as u8;
}
