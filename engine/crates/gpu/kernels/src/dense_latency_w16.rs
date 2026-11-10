// Included by dense_latency.rs (same module): MXFP8 at verify rows M9-16.

/// M9-16 input of canonical range `part`, once per CTA at `at`: [K16 tile of the range][four
/// 4-token groups][lane] 8 bytes = the transposed MMA's a0/a2-position words of tokens
/// 4h' + lane / 8 (h' = h % 2; h >= 2 takes the token + 8 words), plane lane / 4 % 2.
#[cfg(target_arch = "nvptx64")]
unsafe fn cta_input16(p: Bf16Rows, at: u32, part: usize) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    let (start, end) = canonical::range(p.cols as usize / 16, part);
    let mut c = _thread_idx_x() as usize;
    while c < (end - start) * 128 {
        let (t, h, l) = (start + c / 128, c / 32 % 4, c % 32);
        let (token, plane) = (4 * (h % 2) + l / 8, l / 4 % 2);
        let src = (p.input as *const u8).add(t * 512 + (token * 4 + l % 4) * 16 + plane * p.input_plane_words as usize * 4 + h / 2 * 4);
        let dst = at + (c * 8) as u32;
        ptx::copy4(dst, src.cast());
        ptx::copy4(dst + 4, src.add(8).cast());
        c += _block_dim_x() as usize;
    }
    ptx::commit();
    ptx::wait();
    ptx::sync();
}

/// range_w8 at M9-16: the same fragment ring and scale burst, four transposed MMAs per K16
/// tile (tokens 0-3, 4-7, 8-11, 12-15) on the CTA copy of this range (cta_input16).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn range_w16<const D: usize, const W: usize>(
    p: Bf16Rows,
    sh: u32,
    row: usize,
    part: usize,
    lane: usize,
    cin: u32,
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
    if row + lane / 4 >= n {
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
    let (mut c0, mut c1, mut c2, mut c3) = (ZERO, ZERO, ZERO, ZERO);
    let mut current = sh;
    let last = sh + (D as u32 - 1) * SIZE;
    let mut quad = start;
    while quad < end {
        core::arch::asm!("cp.async.wait_group {w};", w = const W);
        ptx::warp_sync();
        for j in 0..4u32 {
            let s = ptx::load_u8(sb + (quad - start) * 16 + j / 2 * 8 + (lane / 4) as u32);
            let b = mx8_fragment(ptx::load(current + j * 128 + (lane * 4) as u32).to_bits(), s);
            let x = cin + (((quad - start) * 4 + j) * 1024) as u32 + (lane * 8) as u32;
            let (x0, x1, x2, x3) = (ptx::load2(x), ptx::load2(x + 256), ptx::load2(x + 512), ptx::load2(x + 768));
            c0 = tmadd(b, [x0[0], 0, x0[1], 0], c0);
            c1 = tmadd(b, [x1[0], 0, x1[1], 0], c1);
            c2 = tmadd(b, [x2[0], 0, x2[1], 0], c2);
            c3 = tmadd(b, [x3[0], 0, x3[1], 0], c3);
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
    let (lo, hi) = (untranspose8(c0, c1, lane), untranspose8(c2, c3, lane));
    canonical::Acc { a: lo.a, b: lo.b, c: hi.a, d: hi.b }
}

/// MXFP8 at M9-16, K <= 2560, four warps (latency_mx8_wide16): per CTA round the four warps
/// take four row groups; range by range the CTA copies that range's input (cta_input16) and
/// each warp runs range_w16 on it; the ranges merge (p0+p1)+(p2+p3) as in wide8.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide16(m: Mats) {
    use crate::kernel_params::LATENCY_W8_RING;
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let cin = (4 * LATENCY_W8_RING) as u32;
    let sh = (warp * LATENCY_W8_RING) as u32;
    let mut total = 0;
    for j in 0..m.count {
        total += rows_of(&m, j).div_ceil(8);
    }
    let mut base = _block_idx_x() as usize * 4;
    while base < total {
        let task = base + warp;
        let (mut j, mut first_task) = (0, 0);
        while j + 1 < m.count && task >= first_task + rows_of(&m, j).div_ceil(8) {
            first_task += rows_of(&m, j).div_ceil(8);
            j += 1;
        }
        let p = pick(&m, j);
        let row = task.saturating_sub(first_task) * 8;
        let (mut left, mut right) = (ZERO, ZERO);
        for part in 0..4 {
            ptx::sync();
            cta_input16(m.a, cin, part);
            if task < total {
                let z = range_w16::<9, 8>(p, sh, row, part, lane, cin);
                match part {
                    0 => left = z,
                    1 => left = canonical::add(left, z),
                    2 => right = z,
                    _ => right = canonical::add(right, z),
                }
            }
        }
        if task < total {
            write(p, canonical::add(left, right), lane / 4, row + lane % 4 * 2);
        }
        base += 192 * 4;
    }
}
