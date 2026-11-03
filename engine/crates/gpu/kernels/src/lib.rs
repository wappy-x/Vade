//! Genuine Rust-authored kernels compiled by the official rustc NVPTX backend.
//! No CUDA C, external Rust GPU crate, or runtime library.
#![no_std]
#![feature(abi_ptx, stdarch_nvptx, link_llvm_intrinsics, asm_experimental_arch)]
#![feature(core_float_math)]

use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
use core::panic::PanicInfo;
#[path = "../../kernel_params.rs"]
mod kernel_params;
#[path = "../../src/projection_params.rs"]
mod projection_params;
mod math;
#[path = "../../src/next_math.rs"]
mod next_math;
#[path = "../../src/compact.rs"]
mod compact;
mod bf16;
mod canonical;
mod matvec;
mod moe;
mod ptx;
#[path = "../../src/fp8_abi.rs"]
mod fp8_abi;
#[path = "../../src/fp4_abi.rs"]
mod fp4_abi;
mod layer;
mod swiglu;
mod decode_attention;
mod flash_ops;
#[path = "../../src/frontend_math.rs"]
mod frontend_math;
mod next_frontend;
mod flash_gdn;
#[path = "../../src/next_params.rs"]
mod next_params;
mod next_decode;
mod flash_gdn_chunk;
mod flash_qsa;
mod qsa_common;
mod qsa_indexer;
mod flash_moe;
#[path = "../../src/router_abi.rs"]
mod router_abi;
mod corrected_router;
mod ungated_moe;
#[cfg(feature = "tensor-cores")]
mod gemm;
#[cfg(feature = "tensor-cores")]
mod fp8;
#[cfg(feature = "tensor-cores")]
mod fp8_bf16;
#[cfg(feature = "tensor-cores")]
mod fp8_decode;
#[cfg(feature = "tensor-cores")]
mod fp8_routing;
#[cfg(feature = "tensor-cores")]
mod fp8_accurate_gate;
#[cfg(feature = "tensor-cores")]
mod fp8_accurate_decode;
#[cfg(feature = "fp4")]
mod fp4;
#[cfg(feature = "tensor-cores")]
mod fp4_experts;
#[cfg(feature = "tensor-cores")]
mod fp4_exact;
#[cfg(feature = "tensor-cores")]
mod flash_next;
#[cfg(feature = "tensor-cores")]
mod next_head;
#[cfg(feature = "tensor-cores")]
mod head_bounds;
#[cfg(feature = "tensor-cores")]
mod verify_solo97;
#[cfg(feature = "tensor-cores")]
mod next_head32;
#[cfg(feature = "tensor-cores")]
mod qsa_tc;
#[cfg(feature = "tensor-cores")]
mod next_pool;
#[path = "../../src/gdn_params.rs"]
mod gdn_params;
mod flash_next_gdn_chunk;
mod qsa_rows;

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    // Any panic is a programming defect. Trap ensures the host observes a
    // failed launch instead of a kernel that silently returns partial output.
    unsafe { core::arch::nvptx::trap() }
}

#[inline(always)]
unsafe fn index() -> usize {
    (_block_idx_x() as usize) * (_block_dim_x() as usize) + (_thread_idx_x() as usize)
}


// Official LLVM NVVM intrinsic ABI, pinned by this crate's nightly compiler.
// This declaration asks LLVM to generate shuffle instructions; it is neither
// handwritten PTX nor a copied CUDA kernel. The ABI is separately qualified.
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.shfl.sync.bfly.i32"]
    fn shuffle_xor(mask: u32, value: u32, lane_mask: u32, clamp: u32) -> u32;
}

/// Experimental coalesced matvec: one full warp cooperates on one output row.
/// BF16 weights and FP32 input, FP32 partial sums and a fixed butterfly tree.
/// This changes reduction order relative to the scalar reference kernel.
/// # Safety
/// `weights` must be readable for `rows * width` row-major BF16 elements,
/// `input` readable for `width` FP32 elements, and `output` writable for `rows`
/// FP32 elements. Buffers must be aligned for their element types and valid on
/// the device until the launch completes. The output range must not overlap
/// either input range, and inputs must remain unchanged during the launch.
/// Buffer-size and launch-index arithmetic must not overflow. Launch a 1D grid
/// with a block width divisible by 32 and at least `rows * 32` threads. All 32
/// lanes of each valid row must participate in every synchronized shuffle with
/// mask `0xffffffff`.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn matvec_bf16_f32_warp32(
    weights: *const u16, input: *const f32, output: *mut f32, rows: u32, width: u32,
) {
    let thread = index();
    let row = thread / 32;
    let lane = thread % 32;
    if row < rows as usize {
        let mut sum = 0.0f32;
        let mut col = lane;
        while col < width as usize {
            let x = *input.add(col);
            sum += math::bf16_to_f32(*weights.add(row * width as usize + col)) * x;
            col += 32;
        }
        let mut distance = 16u32;
        while distance != 0 {
            let other = f32::from_bits(shuffle_xor(u32::MAX, sum.to_bits(), distance, 31));
            sum += other;
            distance /= 2;
        }
        if lane == 0 { *output.add(row) = sum; }
    }
}
mod latency_group2;
mod latency_rows;
mod tp_mailbox;
