//! The tensor-core kernel family: VADE's second kernel image, built for `sm_121a` (`cargo kernels --tc`).
//! GB10's block-scaled NVFP4 tensor-core instruction needs the architecture-specific target; the production image
//! stays on `sm_121` with its own proof. This image holds only the kernels that need the instruction (and their
//! helpers), so it compiles in seconds and its register table is short. No CUDA C, no outside crate.
// Off the device target this crate is EMPTY: a host build of the whole workspace (the installer's
// `cargo build --workspace --exclude spark-rust-kernels`) then passes over it. `cargo kernels` builds it for NVPTX.
#![cfg(target_arch = "nvptx64")]
#![no_std]
#![feature(abi_ptx, stdarch_nvptx, asm_experimental_arch)]
#![feature(core_float_math)]
#![allow(dead_code)]

use core::panic::PanicInfo;
#[path = "../../kernels/src/ptx.rs"]
mod ptx;
#[path = "../../src/next_math.rs"]
mod next_math;
#[path = "../../src/tc_contract.rs"]
mod tc_contract;
#[path = "../../src/tc_abi.rs"]
mod tc_abi;
/// The four FP32 words of one thread's m16n8 accumulator fragment.
mod canonical {
    #[derive(Clone, Copy)]
    pub(crate) struct Acc { pub a: f32, pub b: f32, pub c: f32, pub d: f32 }
    pub(crate) const ZERO: Acc = Acc { a: 0.0, b: 0.0, c: 0.0, d: 0.0 };
}
/// The one function of `expert_ops` these kernels use: the BF16 round of the epilogue.
mod expert_ops {
    #[inline(always)]
    pub fn round(x: f32) -> f32 { f32::from_bits(crate::next_math::bf16(x) << 16) }
}
#[path = "../../kernels/src/fp4_cohort.rs"]
mod fp4_cohort;
mod nv_pack;

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    // Any panic is a programming defect. Trap, so the host sees a failed launch.
    // SAFETY: `trap` takes no operand and touches no memory; it ends the thread's kernel, which is what a panic means here.
    unsafe { core::arch::nvptx::trap() }
}
