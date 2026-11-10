//! Standalone exact sixteen-output-row MXFP8 verification entry.
//! Shares canonical arithmetic helpers without a dependency on the wide-row entry.
#[cfg(target_arch = "nvptx64")]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.mma.m16n8k16.row.col.bf16"]
    fn lr_mma(a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32,
        c0: f32, c1: f32, c2: f32, c3: f32) -> crate::canonical::Acc;
}
#[cfg(target_arch = "nvptx64")]
use crate::canonical::Acc;
#[cfg(target_arch = "nvptx64")]
use crate::kernel_params::Bf16Rows;
/// canonical::madd and canonical::add (same intrinsic, same adds).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lr_madd(a: [u32; 4], b: [u32; 2], c: Acc) -> Acc {
    lr_mma(a[0], a[1], a[2], a[3], b[0], b[1], c.a, c.b, c.c, c.d)
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn lr_add(a: Acc, b: Acc) -> Acc {
    Acc { a: a.a + b.a, b: a.b + b.b, c: a.c + b.c, d: a.d + b.d }
}
#[cfg(target_arch = "nvptx64")]
const LR_ZERO: Acc = Acc { a: 0.0, b: 0.0, c: 0.0, d: 0.0 };
/// canonical::range: K16 tile bounds of canonical range `part`.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn lr_range(tiles: usize, part: usize) -> (usize, usize) {
    (tiles * part / 4, tiles * (part + 1) / 4)
}
/// kernel_params::latency_wide_chunk_tiles.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn lr_chunk(cols: usize, g: usize) -> usize {
    let range = cols / 64;
    let (bytes, tiles) = ((range * g * 256 + 81919) / 81920, (range + 63) / 64);
    let chunks = if bytes > tiles { bytes } else { tiles };
    ((range + chunks - 1) / chunks + 3) / 4 * 4
}
/// dense_latency's mx8_fragment: E4M3 word + UE8M0 scale -> the two BF16x2 words of the expansion.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lr_fragment(w: u32, s: u32) -> [u32; 2] {
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
    );
    [lo, hi]
}
/// dense_latency's untranspose8: hi + lo per (row, token) of two transposed accumulators.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lr_untranspose8(c0: Acc, c1: Acc, lane: usize) -> Acc {
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
}
/// output[token][row] (flash_bf16.rs put/write for one matrix, rows2 = 0).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lr_write(p: Bf16Rows, v: Acc, token: usize, row: usize) {
    let y = p.output as *mut f32;
    let (n, s, m) = (p.rows as usize, p.stride as usize, p.tokens as usize);
    if token < m && row < n { *y.add(token * s + row) = v.a; }
    if token < m && row + 1 < n { *y.add(token * s + row + 1) = v.b; }
    if token + 8 < m && row < n { *y.add((token + 8) * s + row) = v.c; }
    if token + 8 < m && row + 1 < n { *y.add((token + 8) * s + row + 1) = v.d; }
}
/// dense_latency_w16's w16n_slot: swizzled 16-byte slot of input chunk k within a 1 KB tile.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn lr_slot(k: usize) -> usize {
    k ^ (k >> 3 & 7)
}
/// dense_latency_w16's cta_input_n::<4>: K16 tiles [t0, t1) of the latency input pack (one or
/// both 16-token tiles, 1 KB each per K16 tile) into the CTA copy at `at`.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lr_input(p: Bf16Rows, at: u32, t0: usize, t1: usize) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    let (mts, kt) = (if p.tokens > 16 { 2 } else { 1 }, p.cols as usize / 16);
    let mut c = _thread_idx_x() as usize;
    while c < (t1 - t0) * 64 * mts {
        let (t, mt, k) = (t0 + c / (64 * mts), c / 64 % mts, c % 64);
        let src = (p.input as *const u8).add((mt * kt + t) * 512 + k % 32 * 16 + k / 32 * p.input_plane_words as usize * 4);
        crate::ptx::copy16(at + ((c / (64 * mts)) * 1024 * mts + mt * 1024 + lr_slot(k) * 16) as u32, src.cast());
        c += _block_dim_x() as usize;
    }
    crate::ptx::commit();
    crate::ptx::wait();
    crate::ptx::sync();
}

/// One row group's hi + lo per (row, token) of a range: lo tokens from c0/c1, tokens + 8 from
/// c2/c3 (wide_n's untranspose8 pair); `upper` takes D rows 8-15 (c, d) instead of rows 0-7 (a, b).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lr_group(c0: Acc, c1: Acc, c2: Acc, c3: Acc, upper: bool, lane: usize) -> Acc {
    let g = |x: Acc| if upper { Acc { a: x.c, b: x.d, c: 0.0, d: 0.0 } } else { x };
    let lo = lr_untranspose8(g(c0), g(c1), lane);
    let hi = lr_untranspose8(g(c2), g(c3), lane);
    Acc { a: lo.a, b: lo.b, c: hi.a, d: hi.b }
}

/// range_wn (G 4) over 16 weight rows: rows 0-7 in the transposed MMA's a0/a2 words, rows 8-15
/// in a1/a3 (zero in range_wn), so D rows 8-15 (c, d) hold rows 8-15's sums. MMA rows are
/// independent: each row's K16 chain sees range_wn's operands in range_wn's order. Ring: LR_D
/// stages of 1 KB (both 8-row groups' K64 quad, contiguous in the 64-row block), then both
/// groups' scale bursts (256 B each); 4 x 1 KB + 512 B fits latency_mx8_shared_rows's 4864 B.
#[cfg(target_arch = "nvptx64")]
const LR_D: u32 = 4;

/// the 1-4 token entry in its own child module.
mod r1 {
    use super::*;
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn lr_range_rows1(
        p: Bf16Rows,
        sh: u32,
        row: usize,
        lane: usize,
        cin: u32,
        (q0, q1, t0): (u32, u32, u32),
        mut acc: Acc,
        fill: bool,
    ) -> Acc {
        const SIZE: u32 = 1024;
        let (k, n) = (p.cols as usize, p.rows as usize);
        let kt = k / 16;
        let weight = (p.weight & !1) as *const u8;
        let w = weight.add(((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 128 + lane * 16);
        let sc = weight.add((n * k + 255) / 256 * 256 + row / 8 * (k / 32) * 8);
        let sb = sh + LR_D * SIZE;
        if fill {
            let span = q1 - q0;
            let mut c = lane as u32;
            while c < 2 * span {
                let (g, i) = (c / span, c % span);
                crate::ptx::copy16(sb + g * 256 + i * 16, sc.add(g as usize * (k / 32) * 8 + ((q0 + i) * 16) as usize).cast());
                c += 32;
            }
            let mut at = sh;
            let mut i = 0;
            while i < LR_D {
                if q0 + i < q1 {
                    crate::ptx::copy16(at + (lane * 16) as u32, w.add(((q0 + i) * 4096) as usize).cast());
                    crate::ptx::copy16(at + 512 + (lane * 16) as u32, w.add(((q0 + i) * 4096 + 512) as usize).cast());
                }
                crate::ptx::commit();
                at += SIZE;
                i += 1;
            }
            return acc;
        }
        let k0 = lane / 4 % 2 * 32 + lane / 8 * 4 + lane % 4;
        let s0 = (lr_slot(k0) * 16) as u32;
        let mut current = sh;
        let last = sh + (LR_D - 1) * SIZE;
        let mut quad = q0;
        while quad < q1 {
            core::arch::asm!("cp.async.wait_group 3;");
            crate::ptx::warp_sync();
            let mut j = 0u32;
            while j < 4 {
                let at = sb + (quad - q0) * 16 + j / 2 * 8 + (lane / 4) as u32;
                let b = lr_fragment(crate::ptx::load(current + j * 128 + (lane * 4) as u32).to_bits(), crate::ptx::load_u8(at));
                let b2 = lr_fragment(crate::ptx::load(current + 512 + j * 128 + (lane * 4) as u32).to_bits(), crate::ptx::load_u8(at + 256));
                let a = [b[0], b2[0], b[1], b2[1]];
                let x = cin + (quad * 4 + j - t0) * 1024;
                let v0 = crate::ptx::load4(x + s0);
                acc = lr_madd(a, [v0[0], v0[2]], acc);
                j += 1;
            }
            crate::ptx::warp_sync();
            let next = quad + LR_D;
            if next < q1 {
                crate::ptx::copy16(current + (lane * 16) as u32, w.add((next * 4096) as usize).cast());
                crate::ptx::copy16(current + 512 + (lane * 16) as u32, w.add((next * 4096 + 512) as usize).cast());
            }
            crate::ptx::commit();
            current = if current == last { sh } else { current + SIZE };
            quad += 1;
        }
        acc
    }

    /// flash_mx8_mma_latency_r at 1-4 tokens: one token group (tokens 0-3, acc[0] of
    /// lr_range_rows) per 16-row warp task, one transposed MMA and one input load per K16 tile, on
    /// the latency input pack of M <= 4 (its unwritten lanes feed only the B columns of tokens >= 4,
    /// which are never written: MMA columns are independent). Same per-row chains, untranspose and
    /// (p0+p1)+(p2+p3) merge as _r for those tokens: same bytes.
    /// # Safety
    /// As flash_mx8_mma_latency_r with 1 <= M <= 4: grid 192, block 128, dynamic shared
    /// latency_mx8_shared_rows(9, K, 4, N, false) bytes (_r's one-tile sizing), N % 64 == 0.
    #[cfg(target_arch = "nvptx64")]
    #[no_mangle]
    pub unsafe extern "ptx-kernel" fn flash_verify_mx_solo97(p: Bf16Rows, _scales: u64, _partials: u64) {
        solo97::<false>(p);
    }
    /// flash_verify_mx_solo97 for a chained decode step: the first ring fills before the wait.
    /// # Safety
    /// As flash_verify_mx_solo97, on a plain or programmatic launch.
    // SAFETY: griddepcontrol touches no memory. Before its wait this kernel reads weights and
    // scales only, which no kernel writes during a step; it reads the input after the wait.
    #[cfg(target_arch = "nvptx64")]
    #[no_mangle]
    pub unsafe extern "ptx-kernel" fn flash_verify_mx_solo97_chain(p: Bf16Rows, _scales: u64, _partials: u64) {
        core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
        solo97::<true>(p);
    }
    // SAFETY: main's body of flash_verify_mx_solo97, shared with its chain twin. The CHAIN
    // instantiation adds only griddepcontrol and the weight prologue before the wait.
    #[cfg(target_arch = "nvptx64")]
    #[inline(always)]
    unsafe fn solo97<const CHAIN: bool>(p: Bf16Rows) {
        use crate::kernel_params::LATENCY_W8_RING;
        use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
        let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
        let cin = (4 * LATENCY_W8_RING) as u32;
        let sh = (warp * LATENCY_W8_RING) as u32;
        let kt = p.cols as usize / 16;
        let chunk = lr_chunk(p.cols as usize, 4);
        let total = p.rows as usize / 16;
        let mut base = _block_idx_x() as usize * 4;
        while base < total {
            let task = base + warp;
            let row = task * 16;
            let (mut left, mut right) = ([LR_ZERO; 2], [LR_ZERO; 2]);
            let mut part = 0;
            while part < 4 {
                let (start, end) = lr_range(kt, part);
                let mut acc = LR_ZERO;
                let mut t0 = start;
                while t0 < end {
                    let t1 = if t0 + chunk < end { t0 + chunk } else { end };
                    let q = ((t0 / 4) as u32, (t1 / 4) as u32, t0 as u32);
                    if task < total {
                        acc = lr_range_rows1(p, sh, row, lane, cin, q, acc, true);
                    }
                    crate::ptx::sync();
                    if CHAIN { core::arch::asm!("griddepcontrol.wait;", options(nostack)); }
                    lr_input(p, cin, t0, t1);
                    if task < total {
                        acc = lr_range_rows1(p, sh, row, lane, cin, q, acc, false);
                    }
                    t0 = t1;
                }
                let z = [
                    lr_group(acc, LR_ZERO, LR_ZERO, LR_ZERO, false, lane),
                    lr_group(acc, LR_ZERO, LR_ZERO, LR_ZERO, true, lane),
                ];
                match part {
                    0 => left = z,
                    1 => left = [lr_add(left[0], z[0]), lr_add(left[1], z[1])],
                    2 => right = z,
                    _ => right = [lr_add(right[0], z[0]), lr_add(right[1], z[1])],
                }
                part += 1;
            }
            if task < total {
                let (t, r) = (lane / 4, row + lane % 4 * 2);
                lr_write(p, lr_add(left[0], right[0]), t, r);
                lr_write(p, lr_add(left[1], right[1]), t, r + 8);
            }
            base += 192 * 4;
        }
    }
}
