//! Direct-register verify input with K64 weight-only rings; VADE_VERIFY_MX4 default on.
//! First-party direct-group implementation of exact MXFP8 verification projections.
//! REDUCTION-PLAN: m16n8k16 BF16, transposed as main; four shape-only absolute K
//! ranges, ascending K16 chains, hi+lo per range, then (p0+p1)+(p2+p3).
//! M5-8 (H = 2: _part_group8, _mixed8): two transposed MMAs per K16 tile on range_w8's
//! operands (tokens 0-3, then 4-7) and untranspose8, so each output keeps the chains
//! and merge of main's M5-8 entries (wide8 / intra). The M3/4 entries are unchanged.
//! No new precision, atomics, spin waits or barriers. Input loads bypass the shared ring.
use super::*;

/// One canonical K range for H token halves: H = 1 tokens 0-3 (M3/4), H = 2 tokens
/// 0-7 (M5-8; the second MMA reads the words 256 bytes on, range_w8's x1).
#[inline(always)]
unsafe fn cached_range<const D: usize, const W: usize, const RELATIVE: bool, const DIRECT: bool, const H: usize, const ISSUE: bool>(
    p: Bf16Rows,
    sh: u32,
    row: usize,
    part: usize,
    lane: usize,
    cin: u32,
    primed: bool,
) -> canonical::Acc {
    const SIZE: u32 = 512;
    let (k, n) = (p.cols as usize, p.rows as usize);
    let kt = k / 16;
    let (start, end) = canonical::range(kt, part);
    let (start, end) = (start as u32 / 4, end as u32 / 4);
    let weight = mx8_base(p.weight) as *const u8;
    let (w, ws, wd) = if n % 64 == 0 {
        (weight.add(((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 128 + lane * 16), 4096u32, lane * 16)
    } else {
        let (q, j) = (lane / 16, lane % 16);
        let w = if row + q * 4 < n { weight.add((row / 4 + q) * kt * 64 + j * 16) } else { core::ptr::null() };
        (w, 256u32, j / 4 * 128 + q * 64 + j % 4 * 16)
    };
    let sc = weight.add((n * k).div_ceil(256) * 256 + row / 8 * (k / 32) * 8);
    if !primed && row + lane / 4 >= n {
        let mut dst = sh + (lane * 4) as u32;
        for _ in 0..D {
            for j in 0..4u32 {
                ptx::store(dst + j * 128, 0.0);
            }
            dst += SIZE;
        }
    }
    let stage = |at: u32, quad: u32| {
        if !w.is_null() {
            ptx::copy16(at + wd as u32, w.add((quad * ws) as usize).cast());
        }
        ptx::commit();
    };
    let sb = sh + D as u32 * SIZE;
    if !primed {
        let mut c = lane as u32;
        while c < end - start {
            ptx::copy16(sb + c * 16, sc.add(((start + c) * 16) as usize).cast());
            c += 32;
        }
        let mut at = sh;
        for i in 0..D as u32 {
            if start + i < end {
                stage(at, start + i);
            } else {
                ptx::commit();
            }
            at += SIZE;
        }
    }
    if ISSUE {
        return ZERO;
    }
    let (mut c0, mut c1) = (ZERO, ZERO);
    let mut current = sh;
    let last = sh + (D as u32 - 1) * SIZE;
    let mut quad = start;
    while quad < end {
        core::arch::asm!("cp.async.wait_group {w};", w = const W);
        ptx::warp_sync();
        for j in 0..4u32 {
            let s = ptx::load_u8(sb + (quad - start) * 16 + j / 2 * 8 + (lane / 4) as u32);
            let b = mx8_fragment(ptx::load(current + j * 128 + (lane * 4) as u32).to_bits(), s);
            if DIRECT {
                let src = (p.input as *const u8).add(((quad * 4 + j) * 512) as usize
                    + (lane / 8 * 4 + lane % 4) * 16
                    + (lane / 4 % 2) * p.input_plane_words as usize * 4);
                let x = [src.cast::<u32>().read(), src.add(8).cast::<u32>().read()];
                c0 = tmadd(b, [x[0], 0, x[1], 0], c0);
                if H == 2 {
                    let y = [src.add(256).cast::<u32>().read(), src.add(264).cast::<u32>().read()];
                    c1 = tmadd(b, [y[0], 0, y[1], 0], c1);
                }
            } else {
                let at = cin + ((quad - if RELATIVE { start } else { 0 }) * 4 + j) * (256 * H as u32) + (lane * 8) as u32;
                let x = ptx::load2(at);
                c0 = tmadd(b, [x[0], 0, x[1], 0], c0);
                if H == 2 {
                    let y = ptx::load2(at + 256);
                    c1 = tmadd(b, [y[0], 0, y[1], 0], c1);
                }
            }
        }
        ptx::warp_sync();
        if quad + (D as u32) < end {
            stage(current, quad + D as u32);
        } else {
            ptx::commit();
        }
        current = if current == last { sh } else { current + SIZE };
        quad += 1;
    }
    if H == 2 {
        return untranspose8(c0, c1, lane);
    }
    untranspose(c0, lane)
}

/// Same REDUCTION-PLAN, with direct read-only input words and weight-only K64 rings.
/// Removes the CTA input image and its barriers; 19,456 shared bytes, no new waits.
/// # Safety
/// All matrices MXFP8, M3/4, K<=2560 and K%256==0, total rows>4096;
/// 192 CTAs x 128 threads. Input uses the shipped packed-input layout.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_verify_mx4_group(g: crate::kernel_params::Bf16Pairs, partials: u64) {
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let m = group(g);
    let total: usize = (0..m.count).map(|j| rows_of(&m, j)).sum();
    if !(3..=4).contains(&m.a.tokens) || m.a.cols > 2560 || m.a.cols % 256 != 0
        || crate::kernel_params::latency_split(total, true) {
        return dispatch::<true, 2>(m, partials);
    }
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let sh = (warp * crate::kernel_params::LATENCY_W8_RING) as u32;
    let mut task = _block_idx_x() as usize * 4 + warp;
    let (mut j, mut first) = (0, 0);
    while j < m.count {
        let tasks = rows_of(&m, j).div_ceil(8);
        if task >= first + tasks { first += tasks; j += 1; continue; }
        let p = pick(&m, j);
        let row = (task - first) * 8;
        let part = |i| if p.weight & 1 != 0 { cached_range::<9, 8, false, true, 1, false>(p, sh, row, i, lane, 0, false) }
            else { pick_range::<16, 6, 5, 9, 8, 0, 0, false>(p, sh, row, i, lane) };
        let (a, b, c, d) = (part(0), part(1), part(2), part(3));
        write(p, canonical::add(canonical::add(a, b), canonical::add(c, d)), lane / 4, row + lane % 4 * 2);
        task += 192 * 4;
    }
}
#[no_mangle]
pub static FLASH_VERIFY_MX4_GROUP_SHARED: u32 = 19456;


/// VADE_VERIFY_MX4_MIXED (default off): MoE's two MXFP8 matrices use weight-only
/// K64 rings; router/shared scalar keep the production BF16 range function.
/// REDUCTION-PLAN: four warps own the four absolute K ranges, each with its
/// original ascending K16 chain; the existing CTA tree is (p0+p1)+(p2+p3).
/// # Safety
/// Production mixed MoE group (N=512+1+640+640,K2560), M3/4, packed input;
/// grid192/block128, four LATENCY_RING-byte shared regions. The exported
/// allocation is derived from that same stride (not a separate byte literal).
/// No new barriers/waits. The original mixed entry is quarantined: its bench
/// metadata underfunded these regions; r2 fixes that allocation contract.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_verify_mx4_mixed_r2(g: crate::kernel_params::Bf16Pairs, partials: u64) {
    const _: () = assert!(9 * 512 + 256 <= RING);
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let m = group(g);
    let total: usize = (0..m.count).map(|j| rows_of(&m, j)).sum();
    if !(3..=4).contains(&m.a.tokens) || m.a.cols != 2560 || total != 1793 {
        return dispatch::<true, 2>(m, partials);
    }
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let sh = (warp * RING) as u32;
    let mut task = _block_idx_x() as usize;
    let (mut j, mut first) = (0, 0);
    while j < m.count {
        let tasks = rows_of(&m, j).div_ceil(8);
        if task >= first + tasks { first += tasks; j += 1; continue; }
        let p = pick(&m, j);
        let row = (task - first) * 8;
        let partial = if p.weight & 1 != 0 {
            cached_range::<9, 8, false, true, 1, false>(p, sh, row, warp, lane, 0, false)
        } else { pick_range::<16, 6, 5, 9, 8, 0, 0, false>(p, sh, row, warp, lane) };
        let slot = sh + (lane * 16) as u32;
        ptx::store(slot, partial.a); ptx::store(slot + 4, partial.b);
        ptx::store(slot + 8, partial.c); ptx::store(slot + 12, partial.d);
        ptx::sync();
        if warp == 0 {
            let get = |w: usize| {
                let at = (w * RING + lane * 16) as u32;
                canonical::Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
            };
            write(p, canonical::add(canonical::add(get(0), get(1)), canonical::add(get(2), get(3))), lane / 4, row + lane % 4 * 2);
        }
        ptx::sync();
        task += 192;
    }
}
#[no_mangle]
pub static FLASH_VERIFY_MX4_MIXED_R2_SHARED: u32 = (4 * RING) as u32;

/// M5-8 twin of flash_verify_mx4_mixed_r2 (VADE_VERIFY_MX4_MIXED): 1-wave 225-CTA launch
/// (4 * 4096 = 16,384 B shared -> 6 CTAs/SM = 288 CTAs/wave on GB10 so all 225 tasks fit in
/// one single wave instead of 192 + 33 stragglers). Same K16 chains and (p0+p1)+(p2+p3) merge.
/// # Safety
/// Production mixed MoE group (N=512+1+640+640,K2560), M5-8, packed input;
/// grid 225 / block 128, four 4096-byte shared regions (16,384 B).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_verify_mx4_mixed8(g: crate::kernel_params::Bf16Pairs, _partials: u64) {
    const WSTRIDE: usize = 4096;
    const _: () = assert!(7 * 512 + 256 <= WSTRIDE && 3 * (32 * 32 + 256) <= WSTRIDE);
    use core::arch::nvptx::{_block_idx_x, _grid_dim_x, _thread_idx_x};
    let m = group(g);
    let total: usize = (0..m.count).map(|j| rows_of(&m, j)).sum();
    if !(5..=8).contains(&m.a.tokens) || m.a.cols != 2560 || total != 1793 {
        return;
    }
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let sh = (warp * WSTRIDE) as u32;
    let stride = _grid_dim_x() as usize;
    let mut task = _block_idx_x() as usize;
    let (mut j, mut first) = (0, 0);
    while j < m.count {
        let tasks = rows_of(&m, j).div_ceil(8);
        if task >= first + tasks { first += tasks; j += 1; continue; }
        let p = pick(&m, j);
        let row = (task - first) * 8;
        let partial = if p.weight & 1 != 0 {
            cached_range::<7, 6, false, true, 2, false>(p, sh, row, warp, lane, 0, false)
        } else { pick_range::<32, 3, 2, 5, 4, 0, 0, false>(p, sh, row, warp, lane) };
        let slot = sh + (lane * 16) as u32;
        ptx::store(slot, partial.a); ptx::store(slot + 4, partial.b);
        ptx::store(slot + 8, partial.c); ptx::store(slot + 12, partial.d);
        core::arch::nvptx::_syncthreads();
        if warp == 0 {
            let get = |w: usize| {
                let at = (w * WSTRIDE + lane * 16) as u32;
                canonical::Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
            };
            write(p, canonical::add(canonical::add(get(0), get(1)), canonical::add(get(2), get(3))), lane / 4, row + lane % 4 * 2);
        }
        core::arch::nvptx::_syncthreads();
        task += stride;
    }
}
#[no_mangle]
pub static FLASH_VERIFY_MX4_MIXED8_SHARED: u32 = (4 * 4096) as u32;

/// M9-16 twin of flash_verify_mx4_mixed8 (VADE_VERIFY_MX4_MIXED): 1-wave 225-CTA launch
/// (4 * 4096 = 16,384 B shared -> 6 CTAs/SM = 288 CTAs/wave) reading MXFP8 twins in four-half
/// direct ranges; BF16 router/shared scalar keep M9-16 range. Same K16 chains and (p0+p1)+(p2+p3) merge.
/// # Safety
/// Production mixed MoE group (N=512+1+640+640,K2560), M9-16, packed input;
/// grid 225 / block 128, four 4096-byte shared regions (16,384 B).
// SAFETY: blocks::next_rows_group_at checks tokens in 9..=16, cols == 2560, total == 1793 and passes 225 CTAs x 128 with 16,384 B shared.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_verify_mx4_mixed16(g: crate::kernel_params::Bf16Pairs, _partials: u64) {
    const WSTRIDE: usize = 4096;
    const _: () = assert!(7 * 512 + 256 <= WSTRIDE && 3 * (32 * 32 + 256) <= WSTRIDE);
    use core::arch::nvptx::{_block_idx_x, _grid_dim_x, _thread_idx_x};
    let m = group(g);
    let total: usize = (0..m.count).map(|j| rows_of(&m, j)).sum();
    if !(9..=16).contains(&m.a.tokens) || m.a.cols != 2560 || total != 1793 {
        return;
    }
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let sh = (warp * WSTRIDE) as u32;
    let stride = _grid_dim_x() as usize;
    let mut task = _block_idx_x() as usize;
    let (mut j, mut first) = (0, 0);
    while j < m.count {
        let tasks = rows_of(&m, j).div_ceil(8);
        if task >= first + tasks { first += tasks; j += 1; continue; }
        let p = pick(&m, j);
        let row = (task - first) * 8;
        let partial = if p.weight & 1 != 0 {
            const D: usize = 7;
            const W: usize = 6;
            const SIZE: u32 = 512;
            let (k, n) = (p.cols as usize, p.rows as usize);
            let kt = k / 16;
            let (start, end) = canonical::range(kt, warp);
            let (start, end) = (start as u32 / 4, end as u32 / 4);
            let weight = mx8_base(p.weight) as *const u8;
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
            let mut current = sh;
            let last = sh + (D as u32 - 1) * SIZE;
            let in_base = (lane / 8 * 4 + lane % 4) * 16 + (lane / 4 % 2) * p.input_plane_words as usize * 4;
            let mut quad = start;
            while quad < end {
                core::arch::asm!("cp.async.wait_group {w};", w = const W);
                ptx::warp_sync();
                for jj in 0..4u32 {
                    let s = ptx::load_u8(sb + (quad - start) * 16 + jj / 2 * 8 + (lane / 4) as u32);
                    let b = mx8_fragment(ptx::load(current + jj * 128 + (lane * 4) as u32).to_bits(), s);
                    let src = (p.input as *const u8).add(((quad * 4 + jj) * 512) as usize + in_base);
                    let (a0, a1, a2, a3): (u32, u32, u32, u32);
                    let (d0, d1, d2, d3): (u32, u32, u32, u32);
                    core::arch::asm!("ld.global.nc.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                        a = out(reg32) a0, b = out(reg32) a1, c = out(reg32) a2, d = out(reg32) a3,
                        p = in(reg64) src, options(readonly, nostack));
                    core::arch::asm!("ld.global.nc.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                        a = out(reg32) d0, b = out(reg32) d1, c = out(reg32) d2, d = out(reg32) d3,
                        p = in(reg64) src.add(256), options(readonly, nostack));
                    c0 = tmadd(b, [a0, 0, a2, 0], c0);
                    c1 = tmadd(b, [d0, 0, d2, 0], c1);
                    c2 = tmadd(b, [a1, 0, a3, 0], c2);
                    c3 = tmadd(b, [d1, 0, d3, 0], c3);
                }
                ptx::warp_sync();
                if quad + (D as u32) < end { stage(current, quad + D as u32); } else { ptx::commit(); }
                current = if current == last { sh } else { current + SIZE };
                quad += 1;
            }
            let (lo, hi) = (untranspose8(c0, c1, lane), untranspose8(c2, c3, lane));
            canonical::Acc { a: lo.a, b: lo.b, c: hi.a, d: hi.b }
        } else { pick_range::<32, 3, 2, 5, 4, 0, 0, false>(p, sh, row, warp, lane) };
        let slot = sh + (lane * 16) as u32;
        ptx::store(slot, partial.a); ptx::store(slot + 4, partial.b);
        ptx::store(slot + 8, partial.c); ptx::store(slot + 12, partial.d);
        core::arch::nvptx::_syncthreads();
        if warp == 0 {
            let get = |w: usize| {
                let at = (w * WSTRIDE + lane * 16) as u32;
                canonical::Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
            };
            write(p, canonical::add(canonical::add(get(0), get(1)), canonical::add(get(2), get(3))), lane / 4, row + lane % 4 * 2);
        }
        core::arch::nvptx::_syncthreads();
        task += stride;
    }
}
#[no_mangle]
pub static FLASH_VERIFY_MX4_MIXED16_SHARED: u32 = (4 * 4096) as u32;

#[path = "verify_mx4_part.rs"]
mod part;
