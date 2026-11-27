//! Standalone MXFP8 decode latency kernels for K % 128 == 0 && K % 256 != 0 (e.g. K = 640,
//! MoE shared_down). Kept in its own compilation unit so dense_latency.rs codegen stays untouched.
//! Every output keeps the exact canonical four-range K16 MMA graph and (p0+p1)+(p2+p3) merge.
use crate::{
    canonical::{self, Acc, ZERO},
    kernel_params::Bf16Rows,
    ptx,
};
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _grid_dim_x, _thread_idx_x};

macro_rules! mx8_frag {
    ($w:expr, $s:expr) => {{
        let w: u32 = $w;
        let s: u32 = $s;
        let m = w & 0x7f7f_7f7f;
        let sign = w ^ m;
        let f = ((s + 120) << 7) * 0x0001_0001;
        let (lo, hi): (u32, u32);
        core::arch::asm!(
            "{{ .reg .b32 t0, t1, t2, t3, z;",
            "mov.b32 z, 0;",
            "prmt.b32 t0, {m}, z, 0x4140;",
            "prmt.b32 t1, {g}, z, 0x1404;",
            "prmt.b32 t2, {m}, z, 0x4342;",
            "prmt.b32 t3, {g}, z, 0x3424;",
            "shl.b32 t0, t0, 4;",
            "shl.b32 t2, t2, 4;",
            "or.b32 t0, t0, t1;",
            "or.b32 t2, t2, t3;",
            "mul.rn.bf16x2 {lo}, t0, {f};",
            "mul.rn.bf16x2 {hi}, t2, {f};",
            "}}",
            m = in(reg32) m,
            g = in(reg32) sign,
            f = in(reg32) f,
            lo = out(reg32) lo,
            hi = out(reg32) hi,
            options(pure, nomem, nostack),
        );
        [lo, hi]
    }};
}

macro_rules! k32_tmadd {
    ($b:expr, $x:expr, $c:expr) => {{
        let b: [u32; 2] = $b;
        let x: [u32; 4] = $x;
        canonical::madd([b[0], 0, b[1], 0], [x[0], x[2]], $c)
    }};
}

macro_rules! k32_untranspose1 {
    ($c:expr, $lane:expr) => {{
        let c: Acc = $c;
        let lane: usize = $lane;
        let v = (c.a + c.b).to_bits();
        let src = (lane % 4 * 8 + lane / 4 % 4) as u32;
        let (a, b): (u32, u32);
        core::arch::asm!(
            "shfl.sync.idx.b32 {a}, {v}, {s}, 31, -1;",
            "shfl.sync.idx.b32 {b}, {v}, {t}, 31, -1;",
            a = out(reg32) a, b = out(reg32) b, v = in(reg32) v, s = in(reg32) src, t = in(reg32) src + 4,
        );
        Acc { a: f32::from_bits(a), b: f32::from_bits(b), c: 0.0, d: 0.0 }
    }};
}

macro_rules! k32_untranspose8 {
    ($c0:expr, $c1:expr, $lane:expr) => {{
        let (c0, c1): (Acc, Acc) = ($c0, $c1);
        let lane: usize = $lane;
        let (v0, v1) = ((c0.a + c0.b).to_bits(), (c1.a + c1.b).to_bits());
        let src = (lane % 4 * 8 + lane / 4 % 4) as u32;
        let (a0, a1, b0, b1): (u32, u32, u32, u32);
        core::arch::asm!(
            "shfl.sync.idx.b32 {a0}, {v0}, {s}, 31, -1;",
            "shfl.sync.idx.b32 {a1}, {v1}, {s}, 31, -1;",
            "shfl.sync.idx.b32 {b0}, {v0}, {t}, 31, -1;",
            "shfl.sync.idx.b32 {b1}, {v1}, {t}, 31, -1;",
            a0 = out(reg32) a0, a1 = out(reg32) a1, b0 = out(reg32) b0, b1 = out(reg32) b1,
            v0 = in(reg32) v0, v1 = in(reg32) v1, s = in(reg32) src, t = in(reg32) src + 4,
        );
        let (a, b) = if lane < 16 { (a0, b0) } else { (a1, b1) };
        Acc { a: f32::from_bits(a), b: f32::from_bits(b), c: 0.0, d: 0.0 }
    }};
}

macro_rules! k32_m1_body {
    ($p:expr, $chain:literal) => {{
        const WSTRIDE: u32 = 1536;
        const STAGES: u32 = 5;
        let p: Bf16Rows = $p;
        let tid = _thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let row = _block_idx_x() as usize * 8;
        let (k, n) = (p.cols as usize, p.rows as usize);
        if row >= n { return; }
        let kt = k / 16;
        let (start_t, end_t) = canonical::range(kt, warp);
        let (start, end) = (start_t as u32 / 2, end_t as u32 / 2);
        let weight = (p.weight & !1) as *const u8;
        let sc = weight.add((n * k).div_ceil(256) * 256 + row / 8 * (k / 32) * 8);
        let wsh = (warp as u32) * WSTRIDE;
        let sb = wsh + STAGES * 256;
        let cin = 4 * WSTRIDE;
        let merge = cin + (kt * 64) as u32;

        let stage = |at: u32, pair: u32| {
            if lane < 16 {
                let q = pair as usize / 2;
                let half = pair as usize % 2;
                let w = weight.add(((row / 64 * (kt / 4) + q) * 32 + row / 8 % 8 * 4 + half * 2) * 128 + lane * 16);
                ptx::copy16(at + (lane * 16) as u32, w.cast());
            }
            ptx::commit();
        };
        let mut c = lane as u32;
        while c < end - start {
            ptx::copy8(sb + c * 8, sc.add(((start + c) * 8) as usize).cast());
            c += 32;
        }
        for i in 0..STAGES {
            if start + i < end { stage(wsh + i * 256, start + i); } else { ptx::commit(); }
        }
        if $chain { core::arch::asm!("griddepcontrol.wait;", options(nostack)); }
        let mut idx = tid;
        while idx < kt * 8 {
            let (t, l) = (idx / 8, idx % 8);
            let src = (p.input as *const u8).add(t * 512 + l % 4 * 16 + l / 4 * p.input_plane_words as usize * 4);
            let dst = cin + (idx * 8) as u32;
            ptx::copy4(dst, src.cast());
            ptx::copy4(dst + 4, src.add(8).cast());
            idx += 128;
        }
        ptx::commit();
        ptx::wait();
        core::arch::nvptx::_syncthreads();

        let mut acc = ZERO;
        let mut current = wsh;
        let last = wsh + (STAGES - 1) * 256;
        let mut pair = start;
        while pair < end {
            if pair >= start + STAGES {
                core::arch::asm!("cp.async.wait_group 4;");
                ptx::warp_sync();
            }
            let s = ptx::load_u8(sb + (pair - start) * 8 + (lane / 4) as u32);
            for j in 0..2u32 {
                let b = mx8_frag!(ptx::load(current + j * 128 + (lane * 4) as u32).to_bits(), s);
                let x = if lane < 8 { ptx::load2(cin + ((pair * 2 + j) * 64) as u32 + (lane * 8) as u32) } else { [0; 2] };
                acc = k32_tmadd!(b, [x[0], 0, x[1], 0], acc);
            }
            if pair + STAGES < end {
                ptx::warp_sync();
                stage(current, pair + STAGES);
            }
            current = if current == last { wsh } else { current + 256 };
            pair += 1;
        }
        let part = k32_untranspose1!(acc, lane);
        if lane < 4 {
            let at = merge + (warp * 32 + lane * 8) as u32;
            ptx::store(at, part.a);
            ptx::store(at + 4, part.b);
        }
        core::arch::nvptx::_syncthreads();
        if warp == 0 && lane < 4 {
            let ld = |w: usize| (ptx::load(merge + (w * 32 + lane * 8) as u32), ptx::load(merge + (w * 32 + lane * 8 + 4) as u32));
            let (p0, p1, p2, p3) = (ld(0), ld(1), ld(2), ld(3));
            let a = (p0.0 + p1.0) + (p2.0 + p3.0);
            let b = (p0.1 + p1.1) + (p2.1 + p3.1);
            let y = (p.output as *mut f32).add(row + lane * 2);
            *y = a;
            *y.add(1) = b;
        }
    }};
}

/// # Safety
/// M = 1, N % 64 == 0, K % 128 == 0, K <= 2560, MXFP8 weight, grid N/8 x 1, block 128,
/// dynamic shared >= 5632 + (K/16)*64 + 128 bytes.
// SAFETY: blocks::next_rows_strided checks M == 1, N > 640, N % 64 == 0, K % 128 == 0, K <= 2560,
// MXFP8 weight flag, and passes grid N/8 x 128 threads with the exact shared size.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_mx8_mma_latency_k32_m1(p: Bf16Rows) {
    k32_m1_body!(p, false);
}

/// # Safety
/// Same as flash_mx8_mma_latency_k32_m1, on a plain or programmatic launch.
// SAFETY: griddepcontrol touches no memory; only weight bytes are copied before wait.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_mx8_mma_latency_k32_m1_chain(p: Bf16Rows) {
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    k32_m1_body!(p, true);
}

macro_rules! k32_wide_body {
    ($p:expr, $m16:literal) => {{
        const WSTRIDE: u32 = 1536;
        const STAGES: u32 = 5;
        let p: Bf16Rows = $p;
        let tid = _thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (k, n) = (p.cols as usize, p.rows as usize);
        let kt = k / 16;
        let weight = (p.weight & !1) as *const u8;
        let total = n / 8;
        let wsh = (warp as u32) * WSTRIDE;
        let sb = wsh + STAGES * 256;
        let cin = 4 * WSTRIDE;

        let mut base = _block_idx_x() as usize * 4;
        while base < total {
            let task = base + warp;
            let row = task * 8;
            let live = task < total;
            let sc = if live { weight.add((n * k).div_ceil(256) * 256 + row / 8 * (k / 32) * 8) } else { core::ptr::null() };
            let (mut left, mut right) = (ZERO, ZERO);
            for part in 0..4usize {
                let (start_t, end_t) = canonical::range(kt, part);
                let (start, end) = (start_t as u32 / 2, end_t as u32 / 2);
                core::arch::nvptx::_syncthreads();
                if $m16 {
                    let mut c = tid;
                    while c < (end_t - start_t) * 128 {
                        let (t, h, l) = (start_t + c / 128, c / 32 % 4, c % 32);
                        let (token, plane) = (4 * (h % 2) + l / 8, l / 4 % 2);
                        let src = (p.input as *const u8).add(t * 512 + (token * 4 + l % 4) * 16 + plane * p.input_plane_words as usize * 4 + h / 2 * 4);
                        let dst = cin + (c * 8) as u32;
                        ptx::copy4(dst, src.cast());
                        ptx::copy4(dst + 4, src.add(8).cast());
                        c += _block_dim_x() as usize;
                    }
                } else {
                    let mut c = tid;
                    while c < (end_t - start_t) * 64 {
                        let (t, h, l) = (start_t + c / 64, c / 32 % 2, c % 32);
                        let (token, plane) = (4 * h + l / 8, l / 4 % 2);
                        let src = (p.input as *const u8).add(t * 512 + (token * 4 + l % 4) * 16 + plane * p.input_plane_words as usize * 4);
                        let dst = cin + (c * 8) as u32;
                        ptx::copy4(dst, src.cast());
                        ptx::copy4(dst + 4, src.add(8).cast());
                        c += _block_dim_x() as usize;
                    }
                }
                ptx::commit();
                ptx::wait();
                core::arch::nvptx::_syncthreads();
                if live {
                    let stage = |at: u32, pair: u32| {
                        if lane < 16 {
                            let q = pair as usize / 2;
                            let half = pair as usize % 2;
                            let w = weight.add(((row / 64 * (kt / 4) + q) * 32 + row / 8 % 8 * 4 + half * 2) * 128 + lane * 16);
                            ptx::copy16(at + (lane * 16) as u32, w.cast());
                        }
                        ptx::commit();
                    };
                    let mut c = lane as u32;
                    while c < end - start {
                        ptx::copy8(sb + c * 8, sc.add(((start + c) * 8) as usize).cast());
                        c += 32;
                    }
                    for i in 0..STAGES {
                        if start + i < end { stage(wsh + i * 256, start + i); } else { ptx::commit(); }
                    }
                    let (mut c0, mut c1, mut c2, mut c3) = (ZERO, ZERO, ZERO, ZERO);
                    let mut current = wsh;
                    let last = wsh + (STAGES - 1) * 256;
                    let mut pair = start;
                    while pair < end {
                        core::arch::asm!("cp.async.wait_group 4;");
                        ptx::warp_sync();
                        let s = ptx::load_u8(sb + (pair - start) * 8 + (lane / 4) as u32);
                        for j in 0..2u32 {
                            let b = mx8_frag!(ptx::load(current + j * 128 + (lane * 4) as u32).to_bits(), s);
                            if $m16 {
                                let x = cin + (((pair - start) * 2 + j) * 1024) as u32 + (lane * 8) as u32;
                                let (x0, x1, x2, x3) = (ptx::load2(x), ptx::load2(x + 256), ptx::load2(x + 512), ptx::load2(x + 768));
                                c0 = k32_tmadd!(b, [x0[0], 0, x0[1], 0], c0);
                                c1 = k32_tmadd!(b, [x1[0], 0, x1[1], 0], c1);
                                c2 = k32_tmadd!(b, [x2[0], 0, x2[1], 0], c2);
                                c3 = k32_tmadd!(b, [x3[0], 0, x3[1], 0], c3);
                            } else {
                                let x = cin + (((pair - start) * 2 + j) * 512) as u32 + (lane * 8) as u32;
                                let (x0, x1) = (ptx::load2(x), ptx::load2(x + 256));
                                c0 = k32_tmadd!(b, [x0[0], 0, x0[1], 0], c0);
                                c1 = k32_tmadd!(b, [x1[0], 0, x1[1], 0], c1);
                            }
                        }
                        ptx::warp_sync();
                        if pair + STAGES < end { stage(current, pair + STAGES); } else { ptx::commit(); }
                        current = if current == last { wsh } else { current + 256 };
                        pair += 1;
                    }
                    let z = if $m16 {
                        let (lo, hi) = (k32_untranspose8!(c0, c1, lane), k32_untranspose8!(c2, c3, lane));
                        Acc { a: lo.a, b: lo.b, c: hi.a, d: hi.b }
                    } else {
                        k32_untranspose8!(c0, c1, lane)
                    };
                    match part {
                        0 => left = z,
                        1 => left = canonical::add(left, z),
                        2 => right = z,
                        _ => right = canonical::add(right, z),
                    }
                }
            }
            if live {
                let v = canonical::add(left, right);
                let (t, col, m, stride) = (lane / 4, row + lane % 4 * 2, p.tokens as usize, p.stride as usize);
                let y = p.output as *mut f32;
                if t < m {
                    if col < n { *y.add(t * stride + col) = v.a; }
                    if col + 1 < n { *y.add(t * stride + col + 1) = v.b; }
                }
                if t + 8 < m {
                    if col < n { *y.add((t + 8) * stride + col) = v.c; }
                    if col + 1 < n { *y.add((t + 8) * stride + col + 1) = v.d; }
                }
            }
            base += _grid_dim_x() as usize * 4;
        }
    }};
}

/// # Safety
/// 5 <= M <= 16, N % 64 == 0, K % 128 == 0, K <= 2560, MXFP8 weight, grid >= 1, block 128,
/// dynamic shared >= 6144 + (K/64) * (if M <= 8 { 512 } else { 1024 }) bytes.
// SAFETY: blocks::next_rows_strided validates M in 5..=16, N > 640, N % 64 == 0, K % 128 == 0,
// K <= 2560, MXFP8 weight flag, and launches grid (N/32).min(192) x 128 threads with the exact shared size.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_mx8_mma_latency_k32_wide(p: Bf16Rows) {
    if p.tokens <= 8 {
        k32_wide_body!(p, false);
    } else {
        k32_wide_body!(p, true);
    }
}

/// M=17..32 twin of flash_mx8_mma_latency_k32_wide: stages both 16-token tiles per canonical
/// range into CTA shared memory and reads each MXFP8 K32 weight pair once across all 32 tokens.
/// # Safety
/// 17 <= M <= 32, N % 64 == 0, K % 128 == 0, K <= 2560, MXFP8 weight, grid >= 1, block 128,
/// dynamic shared >= 6144 + (K/64) * 2048 bytes.
// SAFETY: blocks::next_rows_strided validates M in 17..=32, N > 640, N % 64 == 0, K % 128 == 0,
// K <= 2560, MXFP8 weight flag, and launches grid (N/32).min(192) x 128 threads with the exact shared size.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_mx8_mma_latency_k32_wide32(p: Bf16Rows) {
    const WSTRIDE: u32 = 1536;
    const STAGES: u32 = 5;
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (k, n) = (p.cols as usize, p.rows as usize);
    let kt = k / 16;
    let weight = (p.weight & !1) as *const u8;
    let total = n / 8;
    let wsh = (warp as u32) * WSTRIDE;
    let sb = wsh + STAGES * 256;
    let cin = 4 * WSTRIDE;

    let mut base = _block_idx_x() as usize * 4;
    while base < total {
        let task = base + warp;
        let row = task * 8;
        let live = task < total;
        let sc = if live { weight.add((n * k).div_ceil(256) * 256 + row / 8 * (k / 32) * 8) } else { core::ptr::null() };
        let (mut left0, mut right0) = (ZERO, ZERO);
        let (mut left1, mut right1) = (ZERO, ZERO);
        for part in 0..4usize {
            let (start_t, end_t) = canonical::range(kt, part);
            let (start, end) = (start_t as u32 / 2, end_t as u32 / 2);
            core::arch::nvptx::_syncthreads();
            let mut c = tid;
            while c < (end_t - start_t) * 256 {
                let (t, mt, h, l) = (start_t + c / 256, c / 128 % 2, c / 32 % 4, c % 32);
                let (token, plane) = (4 * (h % 2) + l / 8, l / 4 % 2);
                let src = (p.input as *const u8).add((mt * kt + t) * 512 + (token * 4 + l % 4) * 16 + plane * p.input_plane_words as usize * 4 + h / 2 * 4);
                let dst = cin + (c * 8) as u32;
                ptx::copy4(dst, src.cast());
                ptx::copy4(dst + 4, src.add(8).cast());
                c += _block_dim_x() as usize;
            }
            ptx::commit();
            ptx::wait();
            core::arch::nvptx::_syncthreads();
            if live {
                let stage = |at: u32, pair: u32| {
                    if lane < 16 {
                        let q = pair as usize / 2;
                        let half = pair as usize % 2;
                        let w = weight.add(((row / 64 * (kt / 4) + q) * 32 + row / 8 % 8 * 4 + half * 2) * 128 + lane * 16);
                        ptx::copy16(at + (lane * 16) as u32, w.cast());
                    }
                    ptx::commit();
                };
                let mut c = lane as u32;
                while c < end - start {
                    ptx::copy8(sb + c * 8, sc.add(((start + c) * 8) as usize).cast());
                    c += 32;
                }
                for i in 0..STAGES {
                    if start + i < end { stage(wsh + i * 256, start + i); } else { ptx::commit(); }
                }
                let (mut c0, mut c1, mut c2, mut c3) = (ZERO, ZERO, ZERO, ZERO);
                let (mut c4, mut c5, mut c6, mut c7) = (ZERO, ZERO, ZERO, ZERO);
                let mut current = wsh;
                let last = wsh + (STAGES - 1) * 256;
                let mut pair = start;
                while pair < end {
                    core::arch::asm!("cp.async.wait_group 4;");
                    ptx::warp_sync();
                    let s = ptx::load_u8(sb + (pair - start) * 8 + (lane / 4) as u32);
                    for j in 0..2u32 {
                        let b = mx8_frag!(ptx::load(current + j * 128 + (lane * 4) as u32).to_bits(), s);
                        let x = cin + (((pair - start) * 2 + j) * 2048) as u32 + (lane * 8) as u32;
                        let (x0, x1, x2, x3) = (ptx::load2(x), ptx::load2(x + 256), ptx::load2(x + 512), ptx::load2(x + 768));
                        c0 = k32_tmadd!(b, [x0[0], 0, x0[1], 0], c0);
                        c1 = k32_tmadd!(b, [x1[0], 0, x1[1], 0], c1);
                        c2 = k32_tmadd!(b, [x2[0], 0, x2[1], 0], c2);
                        c3 = k32_tmadd!(b, [x3[0], 0, x3[1], 0], c3);
                        let (x4, x5, x6, x7) = (ptx::load2(x + 1024), ptx::load2(x + 1280), ptx::load2(x + 1536), ptx::load2(x + 1792));
                        c4 = k32_tmadd!(b, [x4[0], 0, x4[1], 0], c4);
                        c5 = k32_tmadd!(b, [x5[0], 0, x5[1], 0], c5);
                        c6 = k32_tmadd!(b, [x6[0], 0, x6[1], 0], c6);
                        c7 = k32_tmadd!(b, [x7[0], 0, x7[1], 0], c7);
                    }
                    ptx::warp_sync();
                    if pair + STAGES < end { stage(current, pair + STAGES); } else { ptx::commit(); }
                    current = if current == last { wsh } else { current + 256 };
                    pair += 1;
                }
                ptx::wait();
                ptx::warp_sync();
                let (lo0, hi0) = (k32_untranspose8!(c0, c1, lane), k32_untranspose8!(c2, c3, lane));
                let z0 = Acc { a: lo0.a, b: lo0.b, c: hi0.a, d: hi0.b };
                let (lo1, hi1) = (k32_untranspose8!(c4, c5, lane), k32_untranspose8!(c6, c7, lane));
                let z1 = Acc { a: lo1.a, b: lo1.b, c: hi1.a, d: hi1.b };
                match part {
                    0 => { left0 = z0; left1 = z1; }
                    1 => { left0 = canonical::add(left0, z0); left1 = canonical::add(left1, z1); }
                    2 => { right0 = z0; right1 = z1; }
                    _ => { right0 = canonical::add(right0, z0); right1 = canonical::add(right1, z1); }
                }
            }
        }
        if live {
            let v0 = canonical::add(left0, right0);
            let v1 = canonical::add(left1, right1);
            let (t, col, m, stride) = (lane / 4, row + lane % 4 * 2, p.tokens as usize, p.stride as usize);
            let y = p.output as *mut f32;
            if t < m {
                if col < n { *y.add(t * stride + col) = v0.a; }
                if col + 1 < n { *y.add(t * stride + col + 1) = v0.b; }
            }
            if t + 8 < m {
                if col < n { *y.add((t + 8) * stride + col) = v0.c; }
                if col + 1 < n { *y.add((t + 8) * stride + col + 1) = v0.d; }
            }
            if t + 16 < m {
                if col < n { *y.add((t + 16) * stride + col) = v1.a; }
                if col + 1 < n { *y.add((t + 16) * stride + col + 1) = v1.b; }
            }
            if t + 24 < m {
                if col < n { *y.add((t + 24) * stride + col) = v1.c; }
                if col + 1 < n { *y.add((t + 24) * stride + col + 1) = v1.d; }
            }
        }
        base += _grid_dim_x() as usize * 4;
    }
}
