//! TP's optional projection overlay shares the engine's canonical BF16 MMA order.
#![no_std]
#![feature(abi_ptx, stdarch_nvptx, link_llvm_intrinsics, asm_experimental_arch)]
#![allow(internal_features, unsafe_op_in_unsafe_fn)]
use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
#[path = "../../../gpu/kernel_params.rs"]
#[allow(dead_code)] // TP uses only the shared canonical helpers.
mod kernel_params;
#[path = "../../../gpu/kernels/src/canonical.rs"]
#[allow(dead_code)] // TP uses only the projection helpers.
mod canonical;
use canonical::{Acc, ZERO, madd, add, range, weight_tile};
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { unsafe { core::arch::nvptx::trap() } }

#[inline(always)]
unsafe fn pair(input: *const f32) -> u32 {
    let packed: u32;
    core::arch::asm!("cvt.rn.bf16x2.f32 {p}, {b}, {a};", p = out(reg32) packed,
        a = in(reg32) *input, b = in(reg32) *input.add(1));
    packed
}

#[inline(always)]
unsafe fn dot_range(w: *const u32, x: *const f32, tile: usize, lane: usize,
    tiles: usize, part: usize) -> Acc {
    let (mut col, end) = range(tiles, part);
    let mut sum = ZERO;
    while col < end {
        let at = weight_tile(tile, col, tiles) * 64 + lane * 2;
        let b = [*w.add(at), *w.add(at + 1)];
        let col_x = col * 16 + lane % 4 * 2;
        let lo = pair(x.add(col_x));
        let hi = pair(x.add(col_x + 8));
        sum = madd([lo, lo, hi, hi], b, sum);
        col += 1;
    }
    sum
}

/// Same ABI and blocked weights as shared decode; block64, grid ceil(N/16).
/// # Safety
/// W/X aligned8, K positive and divisible by16; output disjoint from inputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn tp_matvec_packed_prefetch(
    w: *const u32, x: *const f32, y: *mut f32, n: u32, k: u32,
) {
    let tid = _block_idx_x() as usize * 64 + _thread_idx_x() as usize;
    let (tile, lane) = (tid / 32, tid % 32);
    if tile * 8 >= n as usize { return; }
    let tiles = k as usize / 16;
    let a = dot_range(w, x, tile, lane, tiles, 0);
    let b = dot_range(w, x, tile, lane, tiles, 1);
    let c = dot_range(w, x, tile, lane, tiles, 2);
    let d = dot_range(w, x, tile, lane, tiles, 3);
    let sum = add(add(a, b), add(c, d));
    if lane < 4 {
        let row = tile * 8 + lane * 2;
        if row < n as usize { *y.add(row) = sum.a; }
        if row + 1 < n as usize { *y.add(row + 1) = sum.b; }
    }
}
