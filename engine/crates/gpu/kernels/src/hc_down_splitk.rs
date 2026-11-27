//! Standalone single-launch 48-SM 4-way split-K BF16 HC-down + injection kernel for M=3..=16.
//! Splits each 8-row group's K=10240 across its four canonical K16 ranges (1 warp / CTA ->
//! 41 * 4 = 164 CTAs across all 48 SMs). Each CTA stores its canonical partial to L2 and
//! increments a per-task completion counter with GPU-scope release/acquire fencing; the 4th
//! CTA deterministically merges (p0+p1)+(p2+p3), resets its counter to 0, applies hc_sigmoid,
//! and writes hc_planes! — zero extra merge kernel launch overhead and full PDL _chain overlap.
//! Kept in its own compilation unit so dense_latency.rs codegen stays completely untouched.
#![cfg(target_arch = "nvptx64")]

use crate::{
    canonical::{self, Acc, ZERO},
    kernel_params::Bf16Pairs,
    ptx,
};
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _thread_idx_x},
};

macro_rules! pick_hc {
    ($g:expr, $down:expr) => {{
        let a = $g.pairs[0];
        let (weight, output, rows, stride) = if $down {
            (a.weight2, a.output2, a.rows2, a.stride2)
        } else {
            (a.weight, a.output, a.rows, a.stride)
        };
        (
            weight,
            output,
            rows as usize,
            stride as usize,
            a.input,
            a.cols as usize,
            a.tokens as usize,
            a.input_plane_words as usize,
        )
    }};
}

macro_rules! tmadd {
    ($b:expr, $x:expr, $c:expr) => {{
        let b: [u32; 2] = $b;
        let x: [u32; 4] = $x;
        canonical::madd([b[0], 0, b[1], 0], [x[0], x[2]], $c)
    }};
}

macro_rules! untranspose {
    ($c:expr, $lane:expr) => {{
        let c: Acc = $c;
        let lane: usize = $lane;
        let v = (c.a + c.b).to_bits();
        let src = (lane % 4 * 8 + lane / 4 % 4) as u32;
        let (a, b): (u32, u32);
        asm!(
            "shfl.sync.idx.b32 {a}, {v}, {s}, 31, -1;",
            "shfl.sync.idx.b32 {b}, {v}, {t}, 31, -1;",
            a = out(reg32) a,
            b = out(reg32) b,
            v = in(reg32) v,
            s = in(reg32) src,
            t = in(reg32) src + 4,
        );
        Acc { a: f32::from_bits(a), b: f32::from_bits(b), c: 0.0, d: 0.0 }
    }};
}

macro_rules! setup_b {
    ($weight:expr, $n:expr, $k:expr, $row:expr, $lane:expr) => {{
        let (n, k, row, lane): (usize, usize, usize, usize) = ($n, $k, $row, $lane);
        let kt = k / 16;
        let r = row + lane / 4;
        let wbase = $weight as *const u8;
        if r >= n {
            (wbase, 0u32, 0u32, 0u32, true)
        } else if n % 64 == 0 {
            let at = ((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 256 + lane * 8;
            (wbase.add(at), 8192u32, 256u32, (lane % 2 == 0) as u32, false)
        } else if n % 4 == 0 && k % 16 == 0 {
            (wbase.add(r / 4 * kt * 128 + lane % 16 * 8), 512u32, 128u32, (lane % 2 == 0) as u32, false)
        } else {
            (wbase.add((r * k + lane % 4 * 2) * 2), 128u32, 32u32, 2u32, false)
        }
    }};
}

macro_rules! hc_planes_320 {
    ($planes:expr, $rows:expr, $t:expr, $f:expr, $a:expr, $b:expr) => {{
        let (rows, t, f): (usize, usize, usize) = ($rows, $t, $f);
        const KT: usize = 320 / 16;
        let plane = rows.div_ceil(16) * KT * 128;
        let word = ((t / 16 * KT + f / 16) * 32 + t % 8 * 4 + f % 8 / 2) * 4
            + t % 16 / 8
            + f % 16 / 8 * 2;
        let (hi, lo) = crate::gemm::split2_pair($a, $b);
        let p = $planes as *mut u32;
        *p.add(word) = hi;
        *p.add(plane + word) = lo;
        if t % 16 < 8 && t + 8 >= rows {
            *p.add(word + 1) = 0;
            *p.add(plane + word + 1) = 0;
        }
    }};
}

macro_rules! fused_m16 {
    ($g:expr, $partials:expr, $planes:expr, $counters:expr, $l:literal, $d:literal, $w:literal, $chain:literal) => {{
        let lane = _thread_idx_x() as usize;
        let first = ($g.pairs[0].rows as usize).div_ceil(8);
        let tasks = first + ($g.pairs[0].rows2 as usize).div_ceil(8);
        let cta = _block_idx_x() as usize;
        let (part, task) = (cta / tasks, cta % tasks);
        let down = task >= first;
        let (weight, output, n, stride, input, k, rows, plane_words) = pick_hc!($g, down);
        let row = (if down { task - first } else { task }) * 8;
        let (wptr, s4, s1, mode, zero) = setup_b!(weight, n, k, row, lane);
        let xptr = (input as *const u8).add(lane * 16);
        let plane = plane_words * 4;
        let size = ($l * 32 + 256) as u32;
        let (start, end) = canonical::range(k / 16, part);
        let (start, end) = (start as u32, end as u32);
        if zero {
            let mut dst = ($l * 32 + lane * 8) as u32;
            let mut i = 0u32;
            while i < $d as u32 {
                ptx::store(dst, 0.0);
                ptx::store(dst + 4, 0.0);
                dst += size;
                i += 1;
            }
        }
        let stage_w = |at: u32, tile: u32| {
            let w = wptr.add(((tile >> 2) * s4 + (tile & 3) * s1) as usize);
            let dst = at + ($l * 32 + lane * 8) as u32;
            if mode == 1 {
                ptx::copy16(dst, w.cast());
            } else if mode == 2 {
                ptx::copy4(dst, w.cast());
                ptx::copy4(dst + 4, w.add(16).cast());
            }
        };
        let stage_x = |at: u32, tile: u32| {
            if lane < $l {
                let x = xptr.add((tile * 512) as usize);
                ptx::copy16(at + (lane * 16) as u32, x.cast());
                ptx::copy16(at + ($l * 16 + lane * 16) as u32, x.add(plane).cast());
            }
        };
        let e: u32 = if $chain { $d as u32 } else if $l == 4 { 2 } else { 0 };
        let mut at = 0u32;
        let mut i = 0u32;
        while i < e {
            if start + i < end {
                stage_w(at, start + i);
            }
            ptx::commit();
            at += size;
            i += 1;
        }
        if $chain {
            asm!("griddepcontrol.wait;", options(nostack));
        }
        let mut at = 0u32;
        let mut i = 0u32;
        while i < $d as u32 {
            if start + i < end {
                if i < e {
                    stage_x(at, start + i);
                    ptx::commit();
                } else {
                    stage_x(at, start + i);
                    stage_w(at, start + i);
                    ptx::commit();
                }
            } else {
                ptx::commit();
            }
            at += size;
            i += 1;
        }
        let (mut hi, mut lo) = (ZERO, ZERO);
        let mut current = 0u32;
        let last = ($d as u32 - 1) * size;
        let mut tile = start;
        while tile < end {
            asm!("cp.async.wait_group {w};", w = const $w);
            ptx::warp_sync();
            let b = ptx::load2(current + ($l * 32 + lane * 8) as u32);
            if $l <= 16 {
                let src = lane / 8 * 4 + lane % 4;
                let x = if src < $l {
                    ptx::load4(current + (lane / 4 % 2 * $l * 16 + src * 16) as u32)
                } else {
                    [0; 4]
                };
                hi = tmadd!(b, x, hi);
            } else {
                let h = ptx::load4(current + (lane * 16) as u32);
                let l = ptx::load4(current + ($l * 16 + lane * 16) as u32);
                hi = canonical::madd(h, b, hi);
                lo = canonical::madd(l, b, lo);
            }
            ptx::warp_sync();
            if tile + ($d as u32) < end {
                stage_x(current, tile + ($d as u32));
                stage_w(current, tile + ($d as u32));
                ptx::commit();
            } else {
                ptx::commit();
            }
            current = if current == last { 0 } else { current + size };
            tile += 1;
        }
        ptx::wait();
        let partial = if $l <= 16 { untranspose!(hi, lane) } else { canonical::add(hi, lo) };
        let dst = ($partials as *mut f32).add(((task * 4 + part) * 32 + lane) * 4);
        asm!(
            "st.global.cg.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};",
            p = in(reg64) dst,
            a = in(reg32) partial.a,
            b = in(reg32) partial.b,
            c = in(reg32) partial.c,
            d = in(reg32) partial.d,
        );
        ptx::warp_sync();
        asm!("fence.acq_rel.gpu;", options(nostack));
        let counter_ptr = ($counters as *mut u32).add(task);
        let mut prev = 0u32;
        if lane == 0 {
            asm!(
                "atom.global.add.u32 {old}, [{p}], 1;",
                old = out(reg32) prev,
                p = in(reg64) counter_ptr,
                options(nostack),
            );
        }
        let done: u32;
        asm!(
            "shfl.sync.idx.b32 {d}, {v}, 0, 31, -1;",
            d = out(reg32) done,
            v = in(reg32) prev,
            options(nostack),
        );
        if done != 3 {
            return;
        }
        asm!("fence.acq_rel.gpu;", options(nostack));
        if lane == 0 {
            asm!("st.global.cg.u32 [{p}], 0;", p = in(reg64) counter_ptr, options(nostack));
        }
        let load_part = |q: usize| -> Acc {
            let src = ($partials as *const f32).add(((task * 4 + q) * 32 + lane) * 4);
            let (a, b, c, d): (f32, f32, f32, f32);
            asm!(
                "ld.global.cg.v4.f32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                a = out(reg32) a,
                b = out(reg32) b,
                c = out(reg32) c,
                d = out(reg32) d,
                p = in(reg64) src,
            );
            Acc { a, b, c, d }
        };
        let out = canonical::add(
            canonical::add(load_part(0), load_part(1)),
            canonical::add(load_part(2), load_part(3)),
        );
        let t = lane / 4;
        let col = row + lane % 4 * 2;
        let y = output as *mut f32;
        if !down {
            if t < rows && col < n { *y.add(t * stride + col) = out.a; }
            if t < rows && col + 1 < n { *y.add(t * stride + col + 1) = out.b; }
            if t + 8 < rows && col < n { *y.add((t + 8) * stride + col) = out.c; }
            if t + 8 < rows && col + 1 < n { *y.add((t + 8) * stride + col + 1) = out.d; }
        } else {
            let act = |v: f32| {
                let value = v * 0.25;
                value * crate::flash_next::hc_sigmoid(value)
            };
            if t < rows {
                let (a, b) = (act(out.a), act(out.b));
                *y.add(t * stride + col) = a;
                *y.add(t * stride + col + 1) = b;
                hc_planes_320!($planes, rows, t, col, a, b);
            }
            if t + 8 < rows {
                let (a, b) = (act(out.c), act(out.d));
                *y.add((t + 8) * stride + col) = a;
                *y.add((t + 8) * stride + col + 1) = b;
                hc_planes_320!($planes, rows, t + 8, col, a, b);
            }
        }
    }};
}

/// Single-launch 48-SM 4-way split-K BF16 decode HC down + injection (M=3..=16): 164 CTAs x 32
/// threads with deterministic last-CTA (p0+p1)+(p2+p3) merge, hc_sigmoid, and hc_planes!.
/// # Safety
/// Launch (ceil(rows/8)+ceil(rows2/8))*4 CTAs x 32 threads, 3 <= M <= 16, K % 64 == 0,
/// plain BF16 weights, rows2 == 320, pairs[1] empty, dynamic shared >= ring size for M,
/// `partials` covering tasks * 4 * 512 B, `planes` covering 2*320 planes, and `counters`
/// holding >= tasks zero-initialized u32 words (reset to 0 by the 4th CTA of each task).
// SAFETY: Backend::flash_hc_down_splitk_group validates all bounds via hc_down_splitk_check
// and uses the dedicated zero-initialized counter tail reserved in prepare_flash_dense.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_hc_down_splitk_fused(
    g: Bf16Pairs,
    partials: u64,
    planes: u64,
    counters: u64,
) {
    match g.pairs[0].tokens {
        3..=4 => fused_m16!(g, partials, planes, counters, 16, 8, 7, false),
        _ => fused_m16!(g, partials, planes, counters, 32, 8, 7, false),
    }
}

/// Programmatic-launch chain twin of flash_hc_down_splitk_fused for M=3..=4: prefetches all D
/// BF16 weight stages before griddepcontrol.wait and signals downstream hc_up_chain on entry.
/// # Safety
/// As flash_hc_down_splitk_fused with 3 <= M <= 4, on a plain or programmatic launch.
// SAFETY: griddepcontrol touches no memory; before wait this kernel reads only constant weights,
// and downstream kernels wait via griddepcontrol.wait before reading planes or outputs.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_hc_down_splitk_fused_chain(
    g: Bf16Pairs,
    partials: u64,
    planes: u64,
    counters: u64,
) {
    asm!("griddepcontrol.launch_dependents;", options(nostack));
    fused_m16!(g, partials, planes, counters, 16, 8, 7, true);
}
