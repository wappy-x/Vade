//! Standalone M=17..32 mixed MoE group kernel ([router(512), shared_gate(1), shared[0](640),
//! shared[1](640)], K=2560). Kept in its own module so dense_latency.rs codegen stays untouched.
//! Every output keeps the exact canonical four-range K16 MMA graph and (p0+p1)+(p2+p3) merge.
#![cfg(target_arch = "nvptx64")]

use crate::{
    canonical::{self, Acc, ZERO},
    kernel_params::{Bf16Pairs, Bf16Rows},
    ptx,
};
use core::arch::nvptx::{_block_idx_x, _grid_dim_x, _syncthreads, _thread_idx_x};

#[inline(always)]
fn m32_pick(g: Bf16Pairs, j: usize) -> Bf16Rows {
    let (a, b) = (g.pairs[0], g.pairs[1]);
    let mut q = a;
    (q.weight, q.output, q.rows, q.stride) = match j {
        0 => (a.weight, a.output, a.rows, a.stride),
        1 => (a.weight2, a.output2, a.rows2, a.stride2),
        2 => (b.weight, b.output, b.rows, b.stride),
        _ => (b.weight2, b.output2, b.rows2, b.stride2),
    };
    (q.weight2, q.output2, q.rows2, q.stride2) = (0, 0, 0, 0);
    q
}

macro_rules! m32_tmadd {
    ($b:expr, $x:expr, $c:expr) => {{
        let b: [u32; 2] = $b;
        let x: [u32; 4] = $x;
        canonical::madd([b[0], 0, b[1], 0], [x[0], x[2]], $c)
    }};
}

macro_rules! m32_mx8_frag {
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

macro_rules! m32_untranspose8 {
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

macro_rules! m32_ld4 {
    ($src:expr) => {{
        let src: *const u8 = $src;
        let (a, b, c, d): (u32, u32, u32, u32);
        core::arch::asm!(
            "ld.global.nc.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
            p = in(reg64) src, options(readonly, nostack),
        );
        [a, b, c, d]
    }};
}

macro_rules! m32_write_tile {
    ($p:expr, $v:expr, $t_base:expr, $row:expr) => {{
        let (p, v, t_base, row): (Bf16Rows, Acc, usize, usize) = ($p, $v, $t_base, $row);
        let y = p.output as *mut f32;
        let (n, s, m) = (p.rows as usize, p.stride as usize, p.tokens as usize);
        if t_base < m && row < n { *y.add(t_base * s + row) = v.a; }
        if t_base < m && row + 1 < n { *y.add(t_base * s + row + 1) = v.b; }
        if t_base + 8 < m && row < n { *y.add((t_base + 8) * s + row) = v.c; }
        if t_base + 8 < m && row + 1 < n { *y.add((t_base + 8) * s + row + 1) = v.d; }
    }};
}

/// M=17..32 mixed MoE group (N=512+1+640+640, K=2560): 1-wave 225-CTA launch
/// (4 * 4096 = 16,384 B shared -> 6 CTAs/SM = 288 CTAs/wave) staging each weight tile
/// once across both 16-token tiles. MXFP8 members (shared gate/up) use a 7-stage K64
/// weight ring; BF16 members (router, shared_gate) use an 8-stage K16 weight ring.
/// # Safety
/// Production mixed MoE group (N=512+1+640+640, K=2560), 17 <= M <= 32, latency packed
/// input (two M16 tiles); grid 225 / block 128, 16,384 B dynamic shared.
// SAFETY: blocks::next_rows_group_at validates tokens in 17..=32, cols == 2560, total == 1793
// and launches 225 CTAs x 128 threads with 16,384 B dynamic shared.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_verify_mx4_mixed32(g: Bf16Pairs, _partials: u64) {
    const WSTRIDE: usize = 4096;
    const _: () = assert!(7 * 512 + 256 <= WSTRIDE && 8 * 256 <= WSTRIDE && 32 * 32 <= WSTRIDE);
    let total = (g.pairs[0].rows + g.pairs[0].rows2 + g.pairs[1].rows + g.pairs[1].rows2) as usize;
    let m_tok = g.pairs[0].tokens as usize;
    if !(17..=32).contains(&m_tok) || g.pairs[0].cols != 2560 || total != 1793 {
        return;
    }
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let sh = (warp * WSTRIDE) as u32;
    let stride = _grid_dim_x() as usize;
    let mut task = _block_idx_x() as usize;
    let (mut j, mut first) = (0usize, 0usize);
    while j < 4 {
        let p = m32_pick(g, j);
        let tasks = (p.rows as usize).div_ceil(8);
        if task >= first + tasks {
            first += tasks;
            j += 1;
            continue;
        }
        let row = (task - first) * 8;
        let (k, n) = (p.cols as usize, p.rows as usize);
        let kt = k / 16;
        let (start_t, end_t) = (kt * warp / 4, kt * (warp + 1) / 4);
        let plane = p.input_plane_words as usize * 4;
        let second = kt * 512;

        let (part0, part1) = if p.weight & 1 != 0 {
            const D: usize = 7;
            const SIZE: u32 = 512;
            let (start, end) = (start_t as u32 / 4, end_t as u32 / 4);
            let weight = (p.weight & !1) as *const u8;
            let (w, ws, wd) = if n % 64 == 0 {
                (weight.add(((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 128 + lane * 16), 4096u32, lane * 16)
            } else {
                let (q, jj) = (lane / 16, lane % 16);
                let w = if row + q * 4 < n { weight.add((row / 4 + q) * kt * 64 + jj * 16) } else { core::ptr::null() };
                (w, 256u32, jj / 4 * 128 + q * 64 + jj % 4 * 16)
            };
            let sc = weight.add((n * k).div_ceil(256) * 256 + row / 8 * (k / 32) * 8);
            if row + lane / 4 >= n {
                let mut dst = sh + (lane * 4) as u32;
                for _ in 0..D {
                    for jj in 0..4u32 { ptx::store(dst + jj * 128, 0.0); }
                    dst += SIZE;
                }
            }
            let stage = |at: u32, quad: u32| {
                if !w.is_null() { ptx::copy16(at + wd as u32, w.add((quad * ws) as usize).cast()); }
                ptx::commit();
            };
            let sb = sh + D as u32 * SIZE;
            let mut c = lane as u32;
            while c < end - start {
                ptx::copy16(sb + c * 16, sc.add(((start + c) * 16) as usize).cast());
                c += 32;
            }
            let mut at = sh;
            for i in 0..D as u32 {
                if start + i < end { stage(at, start + i); } else { ptx::commit(); }
                at += SIZE;
            }
            let (mut c0, mut c1, mut c2, mut c3) = (ZERO, ZERO, ZERO, ZERO);
            let (mut c4, mut c5, mut c6, mut c7) = (ZERO, ZERO, ZERO, ZERO);
            let mut current = sh;
            let last = sh + (D as u32 - 1) * SIZE;
            let in_base = (lane / 8 * 4 + lane % 4) * 16 + (lane / 4 % 2) * plane;
            let mut quad = start;
            while quad < end {
                core::arch::asm!("cp.async.wait_group 6;");
                ptx::warp_sync();
                for jj in 0..4u32 {
                    let s = ptx::load_u8(sb + (quad - start) * 16 + jj / 2 * 8 + (lane / 4) as u32);
                    let b = m32_mx8_frag!(ptx::load(current + jj * 128 + (lane * 4) as u32).to_bits(), s);
                    let src0 = (p.input as *const u8).add(((quad * 4 + jj) * 512) as usize + in_base);
                    let a = m32_ld4!(src0);
                    let d = m32_ld4!(src0.add(256));
                    c0 = m32_tmadd!(b, [a[0], 0, a[2], 0], c0);
                    c1 = m32_tmadd!(b, [d[0], 0, d[2], 0], c1);
                    c2 = m32_tmadd!(b, [a[1], 0, a[3], 0], c2);
                    c3 = m32_tmadd!(b, [d[1], 0, d[3], 0], c3);
                    let src1 = src0.add(second);
                    let e = m32_ld4!(src1);
                    let f = m32_ld4!(src1.add(256));
                    c4 = m32_tmadd!(b, [e[0], 0, e[2], 0], c4);
                    c5 = m32_tmadd!(b, [f[0], 0, f[2], 0], c5);
                    c6 = m32_tmadd!(b, [e[1], 0, e[3], 0], c6);
                    c7 = m32_tmadd!(b, [f[1], 0, f[3], 0], c7);
                }
                ptx::warp_sync();
                if quad + (D as u32) < end { stage(current, quad + D as u32); } else { ptx::commit(); }
                current = if current == last { sh } else { current + SIZE };
                quad += 1;
            }
            ptx::wait();
            ptx::warp_sync();
            let (lo0, hi0) = (m32_untranspose8!(c0, c1, lane), m32_untranspose8!(c2, c3, lane));
            let (lo1, hi1) = (m32_untranspose8!(c4, c5, lane), m32_untranspose8!(c6, c7, lane));
            (
                Acc { a: lo0.a, b: lo0.b, c: hi0.a, d: hi0.b },
                Acc { a: lo1.a, b: lo1.b, c: hi1.a, d: hi1.b },
            )
        } else {
            const D: u32 = 8;
            const SIZE: u32 = 256;
            let weight = p.weight as *const u8;
            let r = row + lane / 4;
            let (w, s4, s1, mode) = if r >= n {
                (weight, 0u32, 0u32, 0u32)
            } else if n % 64 == 0 {
                let at = ((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 256 + lane * 8;
                (weight.add(at), 8192u32, 256u32, (lane % 2 == 0) as u32)
            } else if n % 4 == 0 {
                (weight.add(r / 4 * kt * 128 + lane % 16 * 8), 512u32, 128u32, (lane % 2 == 0) as u32)
            } else {
                (weight.add((r * k + lane % 4 * 2) * 2), 128u32, 32u32, 2u32)
            };
            if mode == 0 {
                let mut dst = sh + (lane * 8) as u32;
                let mut i = 0;
                while i < D {
                    ptx::store(dst, 0.0);
                    ptx::store(dst + 4, 0.0);
                    dst += SIZE;
                    i += 1;
                }
            }
            let stage = |at: u32, tile: u32| {
                let src = w.add(((tile >> 2) * s4 + (tile & 3) * s1) as usize);
                let dst = at + (lane * 8) as u32;
                if mode == 1 {
                    ptx::copy16(dst, src.cast());
                } else if mode == 2 {
                    ptx::copy4(dst, src.cast());
                    ptx::copy4(dst + 4, src.add(16).cast());
                }
                ptx::commit();
            };
            let (start, end) = (start_t as u32, end_t as u32);
            let mut at = sh;
            let mut i = 0;
            while i < D {
                if start + i < end { stage(at, start + i); } else { ptx::commit(); }
                at += SIZE;
                i += 1;
            }
            let (mut h0, mut l0, mut h1, mut l1) = (ZERO, ZERO, ZERO, ZERO);
            let mut current = sh;
            let last = sh + (D - 1) * SIZE;
            let x = (p.input as *const u8).add(lane * 16);
            let mut tile = start;
            while tile < end {
                core::arch::asm!("cp.async.wait_group 7;");
                ptx::warp_sync();
                let b = ptx::load2(current + (lane * 8) as u32);
                let t0 = x.add((tile * 512) as usize);
                let a0 = m32_ld4!(t0);
                let a1 = m32_ld4!(t0.add(plane));
                let a2 = m32_ld4!(t0.add(second));
                let a3 = m32_ld4!(t0.add(second + plane));
                h0 = canonical::madd(a0, b, h0);
                l0 = canonical::madd(a1, b, l0);
                h1 = canonical::madd(a2, b, h1);
                l1 = canonical::madd(a3, b, l1);
                ptx::warp_sync();
                if tile + D < end { stage(current, tile + D); } else { ptx::commit(); }
                current = if current == last { sh } else { current + SIZE };
                tile += 1;
            }
            ptx::wait();
            ptx::warp_sync();
            (canonical::add(h0, l0), canonical::add(h1, l1))
        };

        let slot = sh + (lane * 32) as u32;
        ptx::store(slot, part0.a); ptx::store(slot + 4, part0.b);
        ptx::store(slot + 8, part0.c); ptx::store(slot + 12, part0.d);
        ptx::store(slot + 16, part1.a); ptx::store(slot + 20, part1.b);
        ptx::store(slot + 24, part1.c); ptx::store(slot + 28, part1.d);
        _syncthreads();
        if warp == 0 {
            let get = |w: usize, off: u32| {
                let at = (w * WSTRIDE + lane * 32) as u32 + off;
                Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
            };
            let v0 = canonical::add(canonical::add(get(0, 0), get(1, 0)), canonical::add(get(2, 0), get(3, 0)));
            let v1 = canonical::add(canonical::add(get(0, 16), get(1, 16)), canonical::add(get(2, 16), get(3, 16)));
            let (t, col) = (lane / 4, row + lane % 4 * 2);
            m32_write_tile!(p, v0, t, col);
            m32_write_tile!(p, v1, t + 16, col);
        }
        _syncthreads();
        task += stride;
    }
}
