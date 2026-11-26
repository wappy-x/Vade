//! Bit-exact W4A16 NVFP4 routed-expert path: the correctness fallback and
//! GPU oracle for the BF16-MMA kernels in fp4_experts.rs. GB10's native FP4
//! MMA (fp4::mma, m16n8k64 e2m1 x e2m1) needs FP4 activations, so a W4A16
//! kernel that must match the FP32 CPU oracle bit-for-bit dequantizes in
//! registers and accumulates with scalar FMA instead. Every output element
//! is one thread summing strictly ascending K: the result is independent
//! of token count, grid shape, and expert grouping. Scales arrive already
//! expanded from CSF storage by the host (csf_expand in the model crate).
use crate::kernel_params::{Fp4ExactProject, Fp4ExactSwiglu};

use crate::next_math as math;
use math::{bf16, exp as exp_exact};

/// One thread per (token, row) output. Grid 1D covering tokens*rows.
/// # Safety
/// Pointers follow Fp4ExactProject's layout; cols % 16 == 0, scales are
/// nonnegative finite E4M3 codes, global_scale and inputs finite. Output
/// aliases no input. Launch with crate::index() coverage of tokens*rows.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_exact_project(p: Fp4ExactProject) {
    let at = crate::index();
    let (tokens, rows, cols) = (p.tokens as usize, p.rows as usize, p.cols as usize);
    if at >= tokens * rows { return; }
    let (token, row) = (at / rows, at % rows);
    let input = (p.input as *const u32).add(token * (cols / 2));
    let weights = (p.weights as *const u8).add(row * (cols / 2));
    let scales = (p.scales as *const u8).add(row * (cols / 16));
    *(p.output as *mut f32).add(at) = math::project(input, weights, scales, p.global_scale, cols);
}

/// One thread per output word (two middle elements). Grid 1D covering pairs.
/// # Safety
/// gate/up hold 2*pairs finite FP32 values; output holds pairs u32 words
/// and aliases neither input.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_exact_swiglu(p: Fp4ExactSwiglu) {
    let pair = crate::index();
    if pair >= p.pairs as usize { return; }
    let middle = |i: usize| {
        let gate = *(p.gate as *const f32).add(i);
        let up = *(p.up as *const f32).add(i);
        bf16(gate / (1.0 + exp_exact(-gate)) * up)
    };
    *(p.output as *mut u32).add(pair) = middle(pair * 2) | (middle(pair * 2 + 1) << 16);
}
