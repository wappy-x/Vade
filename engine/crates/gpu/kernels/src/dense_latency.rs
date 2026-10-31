//! N8 pipeline of warp-private K16 stages; persistent tasks retain the canonical MMA graph.
//! K % 64 == 0 (every production shape): per-task lane addressing, and a stage carries
//! only the input lanes that hold live rows at this M (L lanes x 16 bytes per plane), so
//! small M rings deeper (D slots) in the same RING shared bytes per warp.
#[inline(always)]
fn input_word(tiles: usize, token: usize, tile: usize, lane: usize) -> usize {
    ((token / 16 * tiles + tile) * 32 + lane) * 4
}

#[cfg(target_arch = "nvptx64")]
use super::{canonical, packed, ptx, weight_fragment, write, Bf16Rows, ZERO};
#[cfg(target_arch = "nvptx64")]
use crate::kernel_params::LATENCY_RING as RING;

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn stage(p: Bf16Rows, sh: u32, row: usize, tile: usize, lane: usize) {
    let x = (p.input as *const u32).add(input_word((p.cols as usize).div_ceil(16), 0, tile, lane));
    ptx::copy16(sh + (lane * 16) as u32, x);
    ptx::copy16(
        sh + 512 + (lane * 16) as u32,
        x.add(p.input_plane_words as usize),
    );
    let (r, c) = (row + lane / 4, tile * 16 + lane % 4 * 2);
    let dst = sh + 1024 + (lane * 8) as u32;
    if r < p.rows as usize && packed(p, r) {
        if lane.is_multiple_of(2) {
            let at =
                crate::kernel_params::flash_bf16_element(r, c, p.cols as usize, p.rows as usize);
            ptx::copy16(dst, (p.weight as *const u16).add(at).cast());
        }
    } else {
        let b = weight_fragment(p, r, c);
        ptx::store(dst, f32::from_bits(b[0]));
        ptx::store(dst + 4, f32::from_bits(b[1]));
    }
    ptx::commit();
}

/// Per-task lane addressing: tile t's bytes sit at x + 512t (A hi plane; lo at +plane)
/// and w + (t>>2)*s4 + (t&3)*s1 (this lane's B fragment). mode 1: even lanes copy 16
/// bytes (lane and lane+1 share a row); 2: unpacked rows copy two BF16 pairs; 0: no copy
/// (odd lanes of packed rows, or rows >= N whose slot bytes are zeroed per range).
#[cfg(target_arch = "nvptx64")]
#[derive(Clone, Copy)]
struct Lane {
    x: *const u8,
    plane: u32,
    w: *const u8,
    s4: u32,
    s1: u32,
    mode: u32,
    zero: bool,
    /// F (MXFP8): this row group's UE8M0 scales, [row / 8][K / 32][8] bytes.
    sc: *const u8,
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lane_setup<const F: bool>(p: Bf16Rows, row: usize, lane: usize) -> Lane {
    let (k, n) = (p.cols as usize, p.rows as usize);
    let kt = k / 16;
    let r = row + lane / 4;
    let weight = mx8_base(p.weight) as *const u8;
    let (w, s4, s1, mode) = if r >= n {
        (weight, 0, 0, 0)
    } else if F {
        // MXFP8: the packed BF16 element order at one byte per element (N % 4 == 0);
        // lane 4g copies 16 bytes = the 4-byte fragments of lanes 4g..4g+3 (row g).
        if n % 64 == 0 {
            let at = ((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 128 + lane * 4;
            (weight.add(at), 4096, 128, (lane % 4 == 0) as u32)
        } else {
            (weight.add(r / 4 * kt * 64 + lane % 16 * 4), 256, 64, (lane % 4 == 0) as u32)
        }
    } else if packed(p, r) {
        if n % 64 == 0 {
            let at = ((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 256 + lane * 8;
            (weight.add(at), 8192, 256, (lane % 2 == 0) as u32)
        } else {
            (weight.add(r / 4 * kt * 128 + lane % 16 * 8), 512, 128, (lane % 2 == 0) as u32)
        }
    } else {
        (weight.add((r * k + lane % 4 * 2) * 2), 128, 32, 2)
    };
    Lane {
        x: (p.input as *const u8).add(lane * 16),
        plane: p.input_plane_words as u32 * 4,
        w,
        s4,
        s1,
        mode,
        zero: r >= n,
        sc: weight.wrapping_add((n * k).div_ceil(256) * 256 + row / 8 * (k / 32) * 8),
    }
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn stage_input<const L: usize>(ln: Lane, sh: u32, tile: u32, lane: usize) {
    if lane < L {
        let x = ln.x.add((tile * 512) as usize);
        ptx::copy16(sh + (lane * 16) as u32, x.cast());
        ptx::copy16(sh + (L * 16 + lane * 16) as u32, x.add(ln.plane as usize).cast());
    }
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn stage_weight<const L: usize, const F: bool>(ln: Lane, sh: u32, tile: u32, lane: usize) {
    let w = ln.w.add(((tile >> 2) * ln.s4 + (tile & 3) * ln.s1) as usize);
    if F {
        // 128 fragment bytes, then this K32 block's 8 row scales (both K16 halves copy it).
        if ln.mode == 1 {
            ptx::copy16(sh + (L * 32 + lane * 4) as u32, w.cast());
        }
        if lane == 0 {
            ptx::copy8(sh + (L * 32 + 128) as u32, ln.sc.add((tile / 2 * 8) as usize).cast());
        }
        return;
    }
    let dst = sh + (L * 32 + lane * 8) as u32;
    if ln.mode == 1 {
        ptx::copy16(dst, w.cast());
    } else if ln.mode == 2 {
        ptx::copy4(dst, w.cast());
        ptx::copy4(dst + 4, w.add(16).cast());
    }
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn stage_fast<const L: usize, const F: bool>(ln: Lane, sh: u32, tile: u32, lane: usize) {
    stage_input::<L>(ln, sh, tile, lane);
    stage_weight::<L, F>(ln, sh, tile, lane);
    ptx::commit();
}

/// MXFP8 B fragment: four E4M3 bytes (k = 2t, 2t+1, 2t+8, 2t+9 of row g) and the row's
/// UE8M0 scale `s` -> the two BF16x2 words main's load-time expand_to_bf16 produces.
/// Bits s.eeee.mmm placed at BF16 bit 15 and bits 10..4 equal value * 2^-120 for every
/// code (eeee = 0 lands on BF16 subnormals with the same ratio); one multiply by the
/// exact power of two 2^(s-7) gives value * 2^(s-127), exact for 10 <= s <= 134.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn mx8_fragment(w: u32, s: u32) -> [u32; 2] {
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

/// Pull tile `tile`'s B fragment toward L2 ahead of its ring stage (same addresses).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn prefetch_fast(ln: Lane, tile: u32, lane: usize) {
    if ln.mode != 0 && lane % 8 == 0 {
        let w = ln.w.add(((tile >> 2) * ln.s4 + (tile & 3) * ln.s1) as usize);
        core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) w);
    }
}

/// Transposed M<=4 MMA: the warp's eight weight rows in A rows 0-7 (a0, a2 = the B fragment
/// words), token t's hi plane in B column 2t and its lo plane in column 2t + 1 (lanes
/// 8t..8t+3 and 8t+4..8t+7). D'[r][2t] and D'[r][2t+1] equal the hi and lo chains' D[t][r]
/// bit for bit (mma.m16n8k16 is exact under the operand swap: 294,912 of 294,912 random
/// elements, owners tool mmaswap.py), so one MMA per K16 tile carries both chains.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub(super) unsafe fn tmadd(b: [u32; 2], x: [u32; 4], c: canonical::Acc) -> canonical::Acc {
    canonical::madd([b[0], 0, b[1], 0], [x[0], x[2]], c)
}

/// hi + lo per (row, token) (lane L holds row L/4, token L%4), in the normal layout: lane l
/// gets token l/4's rows 2(l%4) and 2(l%4) + 1 (tokens >= 4 and c, d are never written).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub(super) unsafe fn untranspose(c: canonical::Acc, lane: usize) -> canonical::Acc {
    let v = (c.a + c.b).to_bits();
    let src = (lane % 4 * 8 + lane / 4 % 4) as u32;
    let (a, b): (u32, u32);
    core::arch::asm!(
        "shfl.sync.idx.b32 {a}, {v}, {s}, 31, -1;",
        "shfl.sync.idx.b32 {b}, {v}, {t}, 31, -1;",
        a = out(reg32) a, b = out(reg32) b, v = in(reg32) v, s = in(reg32) src, t = in(reg32) src + 4,
    );
    canonical::Acc { a: f32::from_bits(a), b: f32::from_bits(b), c: 0.0, d: 0.0 }
}

/// One canonical K range: ascending K16 hi and lo chains through a D-slot ring with
/// W = D-1 younger copy groups in flight (empty groups pad the tail), then hi + lo.
/// Lanes >= L hold rows >= M, whose A fragments are zero; MMA rows are independent.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn range<const L: usize, const D: usize, const W: usize, const P: usize, const F: bool>(
    p: Bf16Rows,
    sh: u32,
    row: usize,
    part: usize,
    lane: usize,
) -> canonical::Acc {
    // F: 128 E4M3 fragment bytes + 8 scale bytes (+8 pad) per K16 stage instead of 256.
    let size = (L * 32 + if F { 144 } else { 256 }) as u32;
    let (start, end) = canonical::range(p.cols as usize / 16, part);
    let (start, end) = (start as u32, end as u32);
    let ln = lane_setup::<F>(p, row, lane);
    if ln.zero {
        // Rows >= N: zero B fragments in every slot once; copies never touch them.
        let mut dst = sh + (L * 32 + lane * if F { 4 } else { 8 }) as u32;
        for _ in 0..D {
            ptx::store(dst, 0.0);
            if !F {
                ptx::store(dst + 4, 0.0);
            }
            dst += size;
        }
    }
    if P > 0 {
        let mut t = start + D as u32;
        while t < (start + (D + P) as u32).min(end) {
            prefetch_fast(ln, t, lane);
            t += 1;
        }
    }
    // Weights first: the first E stages' DRAM weight copies are issued before any
    // (L2-resident) input copy. Groups: E weight halves, E input halves, D-E whole
    // stages, then one per tile, so wait_group(D-1) still covers tile t's two halves.
    // M1 only (L = 4, D = 13): at M4..16 the split start measured neutral to +1.6 us.
    let e: u32 = if L == 4 { 2 } else { 0 };
    let mut at = sh;
    for i in 0..e {
        if start + i < end {
            stage_weight::<L, F>(ln, at, start + i, lane);
        }
        ptx::commit();
        at += size;
    }
    let mut at = sh;
    for i in 0..D as u32 {
        if start + i < end {
            if i < e {
                stage_input::<L>(ln, at, start + i, lane);
                ptx::commit();
            } else {
                stage_fast::<L, F>(ln, at, start + i, lane);
            }
        } else {
            ptx::commit();
        }
        at += size;
    }
    let (mut hi, mut lo) = (ZERO, ZERO);
    let mut current = sh;
    let last = sh + (D as u32 - 1) * size;
    let mut tile = start;
    while tile < end {
        core::arch::asm!("cp.async.wait_group {w};", w = const W);
        ptx::warp_sync();
        let b = if F {
            let w = ptx::load(current + (L * 32 + lane * 4) as u32).to_bits();
            mx8_fragment(w, ptx::load_u8(current + (L * 32 + 128 + lane / 4) as u32))
        } else {
            ptx::load2(current + (L * 32 + lane * 8) as u32)
        };
        if L <= 16 {
            // Lane 8t + 4p + q reads plane p of token t from normal-layout lane 4t + q.
            let src = lane / 8 * 4 + lane % 4;
            let x = if src < L { ptx::load4(current + (lane / 4 % 2 * L * 16 + src * 16) as u32) } else { [0; 4] };
            hi = tmadd(b, x, hi);
        } else {
            let (h, l) = if lane < L {
                (ptx::load4(current + (lane * 16) as u32), ptx::load4(current + (L * 16 + lane * 16) as u32))
            } else {
                ([0; 4], [0; 4])
            };
            hi = canonical::madd(h, b, hi);
            lo = canonical::madd(l, b, lo);
        }
        ptx::warp_sync();
        if P > 0 && tile + ((D + P) as u32) < end {
            prefetch_fast(ln, tile + (D + P) as u32, lane);
        }
        if tile + (D as u32) < end {
            stage_fast::<L, F>(ln, current, tile + D as u32, lane);
        } else {
            ptx::commit();
        }
        current = if current == last { sh } else { current + size };
        tile += 1;
    }
    if L <= 16 {
        return untranspose(hi, lane);
    }
    canonical::add(hi, lo)
}

/// MXFP8 K32 stage: input tiles 2u and 2u+1 (hi planes, then lo planes), their two
/// 128-byte fragment blocks, then K32 block u's eight row scales.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn stage_pair<const L: usize>(ln: Lane, sh: u32, pair: u32, lane: usize) {
    let t = pair * 2;
    if lane < L {
        let x = ln.x.add((t * 512) as usize);
        let lo = x.add(ln.plane as usize);
        ptx::copy16(sh + (lane * 16) as u32, x.cast());
        ptx::copy16(sh + (L * 16 + lane * 16) as u32, x.add(512).cast());
        ptx::copy16(sh + (2 * L * 16 + lane * 16) as u32, lo.cast());
        ptx::copy16(sh + (3 * L * 16 + lane * 16) as u32, lo.add(512).cast());
    }
    if ln.mode == 1 {
        // t is even, so tile t + 1 sits s1 bytes after tile t in the same group of four.
        let w = ln.w.add(((t >> 2) * ln.s4 + (t & 3) * ln.s1) as usize);
        ptx::copy16(sh + (L * 64 + lane * 4) as u32, w.cast());
        ptx::copy16(sh + (L * 64 + 128 + lane * 4) as u32, w.add(ln.s1 as usize).cast());
    }
    if lane == 0 {
        ptx::copy8(sh + (L * 64 + 256) as u32, ln.sc.add((pair * 8) as usize).cast());
    }
    ptx::commit();
}

/// MXFP8 canonical range in K32 stages (the range's K16 tile count is even): one wait,
/// sync and scale load per two tiles, and the K16 tiles still enter the hi and lo chains
/// one by one in ascending order, so the sums are range()'s.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn range_mx2<const L: usize, const D: usize, const W: usize>(
    p: Bf16Rows,
    sh: u32,
    row: usize,
    part: usize,
    lane: usize,
) -> canonical::Acc {
    let size = (L * 64 + 272) as u32;
    let (start, end) = canonical::range(p.cols as usize / 16, part);
    let (start, end) = (start as u32 / 2, end as u32 / 2);
    let ln = lane_setup::<true>(p, row, lane);
    if ln.zero {
        let mut dst = sh + (L * 64 + lane * 4) as u32;
        for _ in 0..D {
            ptx::store(dst, 0.0);
            ptx::store(dst + 128, 0.0);
            dst += size;
        }
    }
    let mut at = sh;
    for i in 0..D as u32 {
        if start + i < end {
            stage_pair::<L>(ln, at, start + i, lane);
        } else {
            ptx::commit();
        }
        at += size;
    }
    let (mut hi, mut lo) = (ZERO, ZERO);
    let mut current = sh;
    let last = sh + (D as u32 - 1) * size;
    let mut pair = start;
    while pair < end {
        core::arch::asm!("cp.async.wait_group {w};", w = const W);
        ptx::warp_sync();
        let s = ptx::load_u8(current + (L * 64 + 256 + lane / 4) as u32);
        for h in 0..2u32 {
            let w = ptx::load(current + h * 128 + (L * 64 + lane * 4) as u32).to_bits();
            let b = mx8_fragment(w, s);
            if L <= 16 {
                // Lane 8t + 4p + q reads plane p of token t (lo planes 2 * L * 16 bytes on).
                let src = lane / 8 * 4 + lane % 4;
                let at = current + h * (L * 16) as u32 + (lane / 4 % 2 * 2 * L * 16 + src * 16) as u32;
                let x = if src < L { ptx::load4(at) } else { [0; 4] };
                hi = tmadd(b, x, hi);
                continue;
            }
            let (a, l) = if lane < L {
                let at = current + h * (L * 16) as u32 + (lane * 16) as u32;
                (ptx::load4(at), ptx::load4(at + (2 * L * 16) as u32))
            } else {
                ([0; 4], [0; 4])
            };
            hi = canonical::madd(a, b, hi);
            lo = canonical::madd(l, b, lo);
        }
        ptx::warp_sync();
        if pair + (D as u32) < end {
            stage_pair::<L>(ln, current, pair + D as u32, lane);
        } else {
            ptx::commit();
        }
        current = if current == last { sh } else { current + size };
        pair += 1;
    }
    if L <= 16 {
        return untranspose(hi, lane);
    }
    canonical::add(hi, lo)
}

/// MXFP8 K64 stage at M1 (L = 4): tiles 4u..4u+3 in three copies: lanes 0-15 the hi
/// input planes and lanes 16-31 the lo planes (16 bytes each), every lane 16 of the 512
/// fragment bytes (two 256-byte row quads when N % 64 != 0); the range's scales follow the
/// ring (one burst per range). Slot: hi 256 B | lo 256 B | fragments 4 x 128 B | 16 B unused. The per-lane
/// sources are set up once per range and advance by quad (input 2048 B, scales 16 B).
#[cfg(target_arch = "nvptx64")]
#[derive(Clone, Copy)]
struct Quad {
    x: *const u8,
    xd: u32,
    w: *const u8,
    ws: u32,
    wd: u32,
    sc: *const u8,
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn stage_quad(q: Quad, sh: u32, quad: u32, lane: usize) {
    ptx::copy16(sh + q.xd, q.x.add((quad * 2048) as usize).cast());
    if !q.w.is_null() {
        ptx::copy16(sh + q.wd, q.w.add((quad * q.ws) as usize).cast());
    }
    let _ = lane;
    ptx::commit();
}

/// MXFP8 canonical range at M1 in K64 stages (the range's K16 tile count is a multiple of
/// four): a fifth of range_mx2's shared-memory copies per tile; the K16 tiles still enter
/// the hi and lo chains one by one in ascending order, so the sums are range()'s.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn range_mx4<const D: usize, const W: usize>(
    p: Bf16Rows,
    sh: u32,
    row: usize,
    part: usize,
    lane: usize,
) -> canonical::Acc {
    const SIZE: u32 = 1040;
    let (k, n) = (p.cols as usize, p.rows as usize);
    let kt = k / 16;
    let (start, end) = canonical::range(kt, part);
    let (start, end) = (start as u32 / 4, end as u32 / 4);
    let weight = mx8_base(p.weight) as *const u8;
    let (plane, tile, col) = (lane / 16, lane % 16 / 4, lane % 4);
    // Fragment bytes: one 4096-byte K64 block per row group when N % 64 == 0, else two
    // K-contiguous row quads (null = rows >= N, left zero).
    let (w, ws, wd) = if n % 64 == 0 {
        (weight.add(((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 128 + lane * 16), 4096, 512 + lane * 16)
    } else {
        let (q, j) = (lane / 16, lane % 16);
        let w = if row + q * 4 < n { weight.add((row / 4 + q) * kt * 64 + j * 16) } else { core::ptr::null() };
        (w, 256, 512 + j / 4 * 128 + q * 64 + j % 4 * 16)
    };
    let q = Quad {
        x: (p.input as *const u8).add(tile * 512 + col * 16 + plane * p.input_plane_words as usize * 4),
        xd: (plane * 256 + tile * 64 + col * 16) as u32,
        w,
        ws,
        wd: wd as u32,
        sc: weight.add((n * k).div_ceil(256) * 256 + row / 8 * (k / 32) * 8),
    };
    if row + lane / 4 >= n {
        let mut dst = sh + (512 + lane * 4) as u32;
        for _ in 0..D {
            for j in 0..4u32 {
                ptx::store(dst + j * 128, 0.0);
            }
            dst += SIZE;
        }
    }
    // The range's UE8M0 scales (16 contiguous bytes per quad) in one burst after the ring,
    // part of the first stage's copy group, instead of a 16-byte read per stage.
    let sb = sh + D as u32 * SIZE;
    let mut c = lane as u32;
    while c < end - start {
        ptx::copy16(sb + c * 16, q.sc.add(((start + c) * 16) as usize).cast());
        c += 32;
    }
    let mut at = sh;
    for i in 0..D as u32 {
        if start + i < end {
            stage_quad(q, at, start + i, lane);
        } else {
            ptx::commit();
        }
        at += SIZE;
    }
    let mut hi = ZERO;
    let mut current = sh;
    let last = sh + (D as u32 - 1) * SIZE;
    let mut quad = start;
    while quad < end {
        core::arch::asm!("cp.async.wait_group {w};", w = const W);
        ptx::warp_sync();
        for j in 0..4u32 {
            let s = ptx::load_u8(sb + (quad - start) * 16 + j / 2 * 8 + (lane / 4) as u32);
            let b = mx8_fragment(ptx::load(current + 512 + j * 128 + (lane * 4) as u32).to_bits(), s);
            // Lanes 4-7 read the lo planes (256 bytes after the hi planes).
            let at = current + j * 64 + (lane / 4 * 256 + lane % 4 * 16) as u32;
            let x = if lane < 8 { ptx::load4(at) } else { [0; 4] };
            hi = tmadd(b, x, hi);
        }
        ptx::warp_sync();
        if quad + (D as u32) < end {
            stage_quad(q, current, quad + D as u32, lane);
        } else {
            ptx::commit();
        }
        current = if current == last { sh } else { current + SIZE };
        quad += 1;
    }
    untranspose(hi, lane)
}

/// M5-8 input for the transposed MMA, once per CTA at `at`: [K16 tile][half h][lane L]
/// 8 bytes = the a0 and a2 words of token 4h + L / 8, plane (L / 4) % 2, k quad L % 4
/// (normal-layout lane 4 * token + L % 4 of the packed M16 tile).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn cta_input8(p: Bf16Rows, at: u32) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    let tiles = p.cols as usize / 16;
    let mut c = _thread_idx_x() as usize;
    while c < tiles * 64 {
        let (t, h, l) = (c / 64, c / 32 % 2, c % 32);
        let (token, plane) = (4 * h + l / 8, l / 4 % 2);
        let src = (p.input as *const u8).add(t * 512 + (token * 4 + l % 4) * 16 + plane * p.input_plane_words as usize * 4);
        let dst = at + (c * 8) as u32;
        ptx::copy4(dst, src.cast());
        ptx::copy4(dst + 4, src.add(8).cast());
        c += _block_dim_x() as usize;
    }
    ptx::commit();
    ptx::wait();
    ptx::sync();
}

/// M1 input for the transposed MMA, once per CTA at `at`: [K16 tile][lane L < 8] 8 bytes =
/// the a0 and a2 words of token 0, plane L / 4, k quad L % 4.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn cta_input1(p: Bf16Rows, at: u32) {
    use core::arch::nvptx::{_block_dim_x, _thread_idx_x};
    let tiles = p.cols as usize / 16;
    let mut c = _thread_idx_x() as usize;
    while c < tiles * 8 {
        let (t, l) = (c / 8, c % 8);
        let src = (p.input as *const u8).add(t * 512 + l % 4 * 16 + l / 4 * p.input_plane_words as usize * 4);
        let dst = at + (c * 8) as u32;
        ptx::copy4(dst, src.cast());
        ptx::copy4(dst + 4, src.add(8).cast());
        c += _block_dim_x() as usize;
    }
    ptx::commit();
    ptx::wait();
    ptx::sync();
}

/// hi + lo per (row, token) of the two transposed halves (lane L of half h holds row L/4,
/// token 4h + L%4), in the normal layout: lane l gets token l/4's rows 2(l%4), 2(l%4)+1.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn untranspose8(c0: canonical::Acc, c1: canonical::Acc, lane: usize) -> canonical::Acc {
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
    canonical::Acc { a: f32::from_bits(a), b: f32::from_bits(b), c: 0.0, d: 0.0 }
}

/// MXFP8 canonical range at M5-8 from the CTA input copy: K64 stages of fragments only
/// (512 bytes; the range's scales in one burst after the ring), two transposed MMAs per
/// K16 tile (tokens 0-3 and 4-7), tiles in ascending order: the sums are range()'s.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn range_w8<const D: usize, const W: usize, const H: usize>(
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
            if H == 2 {
                let x = cin + ((quad * 4 + j) * 512) as u32 + (lane * 8) as u32;
                let (x0, x1) = (ptx::load2(x), ptx::load2(x + 256));
                c0 = tmadd(b, [x0[0], 0, x0[1], 0], c0);
                c1 = tmadd(b, [x1[0], 0, x1[1], 0], c1);
            } else {
                // M1: token 0's hi (lanes 0-3) and lo (lanes 4-7) words, 64 bytes per tile.
                let x = if lane < 8 { ptx::load2(cin + ((quad * 4 + j) * 64) as u32 + (lane * 8) as u32) } else { [0; 2] };
                c0 = tmadd(b, [x[0], 0, x[1], 0], c0);
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
    if H == 1 {
        return untranspose(c0, lane);
    }
    untranspose8(c0, c1, lane)
}

/// MXFP8 at M5-8, K <= 2560, four warps (latency_mx8_wide): one CTA input copy, then the
/// non-split task walk of fast() with range_w8 ranges and the (p0+p1)+(p2+p3) merge.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn wide8<const H: usize>(m: Mats) {
    use crate::kernel_params::LATENCY_W8_RING;
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let cin = (4 * LATENCY_W8_RING) as u32;
    if H == 2 { cta_input8(m.a, cin) } else { cta_input1(m.a, cin) }
    let sh = (warp * LATENCY_W8_RING) as u32;
    let mut task = _block_idx_x() as usize * 4 + warp;
    let (mut j, mut first_task) = (0, 0);
    while j < m.count {
        let tasks = rows_of(&m, j).div_ceil(8);
        if task >= first_task + tasks {
            first_task += tasks;
            j += 1;
            continue;
        }
        let p = pick(&m, j);
        let row = (task - first_task) * 8;
        // The ring (D x 512 bytes) and the range's scale burst (16 bytes per K64 quad) share
        // LATENCY_W8_RING = 4864 bytes: 9 stages up to K 2560, 8 up to K 10240.
        let (z0, z1, z2, z3) = if p.cols <= 2560 {
            (range_w8::<9, 8, H>(p, sh, row, 0, lane, cin), range_w8::<9, 8, H>(p, sh, row, 1, lane, cin),
             range_w8::<9, 8, H>(p, sh, row, 2, lane, cin), range_w8::<9, 8, H>(p, sh, row, 3, lane, cin))
        } else {
            (range_w8::<8, 7, H>(p, sh, row, 0, lane, cin), range_w8::<8, 7, H>(p, sh, row, 1, lane, cin),
             range_w8::<8, 7, H>(p, sh, row, 2, lane, cin), range_w8::<8, 7, H>(p, sh, row, 3, lane, cin))
        };
        write(p, canonical::add(canonical::add(z0, z1), canonical::add(z2, z3)), lane / 4, row + lane % 4 * 2);
        task += 192 * 4;
    }
}

/// Up to four matrices sharing one packed input (count >= 1); matrix j's tasks follow
/// matrix j-1's, so each output keeps the solo launch's canonical ranges and merge.
#[cfg(target_arch = "nvptx64")]
#[derive(Clone, Copy)]
struct Mats {
    a: Bf16Rows,
    b: Bf16Rows,
    count: usize,
}

/// Matrix j of a launch = a.first, a.second, b.first, b.second. Field picks (no
/// dynamically indexed array) keep the parameters out of local memory.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn rows_of(m: &Mats, j: usize) -> usize {
    (match j {
        0 => m.a.rows,
        1 => m.a.rows2,
        2 => m.b.rows,
        _ => m.b.rows2,
    }) as usize
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn pick(m: &Mats, j: usize) -> Bf16Rows {
    let mut q = m.a;
    let (w, y, n, s) = match j {
        0 => (m.a.weight, m.a.output, m.a.rows, m.a.stride),
        1 => (m.a.weight2, m.a.output2, m.a.rows2, m.a.stride2),
        2 => (m.b.weight, m.b.output, m.b.rows, m.b.stride),
        _ => (m.b.weight2, m.b.output2, m.b.rows2, m.b.stride2),
    };
    (q.weight, q.output, q.rows, q.stride) = (w, y, n, s);
    (q.weight2, q.output2, q.rows2, q.stride2) = (0, 0, 0, 0);
    q
}

/// Split groups (<= 640 rows) in 128-thread CTAs: CTA task = one row group, warp w
/// computes canonical range w, and warp 0 merges (p0+p1)+(p2+p3) from shared memory
/// (each warp parks its partial in its own idle ring slot) - the merge kernel's sums.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn intra<
    const L: usize,
    const D: usize,
    const W: usize,
    const DF: usize,
    const WF: usize,
    const P: usize,
    const S: u8,
>(
    m: Mats,
) {
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let sh = (warp * RING) as u32;
    let mut task = _block_idx_x() as usize;
    let (mut j, mut first_group) = (0, 0);
    while j < m.count {
        let groups = rows_of(&m, j).div_ceil(8);
        if task >= first_group + groups {
            first_group += groups;
            j += 1;
            continue;
        }
        let p = pick(&m, j);
        let row = (task - first_group) * 8;
        let partial = pick_range::<L, D, W, DF, WF, P, S>(p, sh, row, warp, lane);
        let slot = sh + (lane * 16) as u32;
        ptx::store(slot, partial.a);
        ptx::store(slot + 4, partial.b);
        ptx::store(slot + 8, partial.c);
        ptx::store(slot + 12, partial.d);
        ptx::sync();
        if warp == 0 {
            let get = |w: u32| {
                let at = (w as usize * RING + lane * 16) as u32;
                canonical::Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
            };
            let out = canonical::add(canonical::add(get(0), get(1)), canonical::add(get(2), get(3)));
            write(p, out, lane / 4, row + lane % 4 * 2);
        }
        ptx::sync();
        task += 192;
    }
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn solo(p: Bf16Rows) -> Mats {
    Mats { a: p, b: p, count: 1 }
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn group(g: crate::kernel_params::Bf16Pairs) -> Mats {
    Mats { a: g.pairs[0], b: g.pairs[1], count: 4 }
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn fast<
    const L: usize,
    const D: usize,
    const W: usize,
    const DF: usize,
    const WF: usize,
    const P: usize,
    const ICS: bool,
    const S: u8,
>(
    m: Mats,
    partials: u64,
) {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let warps = _block_dim_x() as usize / 32;
    let mut total = 0;
    for j in 0..m.count {
        total += rows_of(&m, j);
    }
    // Groups merge their four ranges inside the CTA, so they need no partials and can
    // split wider launches (a 1793-row MoE group otherwise keeps 225 of 768 warps busy);
    // (p0+p1)+(p2+p3) is the unsplit association too, so the bytes do not change.
    let split = crate::kernel_params::latency_split(total, ICS);
    if S == 1 && L == 32 && !split && warps == 4 && crate::kernel_params::latency_mx8_wide(m.a.tokens as usize, m.a.cols as usize) {
        return wide8::<2>(m);
    }
    if S == 1 && L == 32 && !split && warps == 4 && crate::kernel_params::latency_mx8_wide16(m.a.tokens as usize, m.a.cols as usize) {
        return wide16(m);
    }
    if S == 1 && L == 4 && !split && warps == 4 && crate::kernel_params::latency_mx8_wide1(m.a.tokens as usize, m.a.cols as usize) {
        return wide8::<1>(m);
    }
    if ICS && split {
        return intra::<L, D, W, DF, WF, P, S>(m);
    }
    let parts = if split { 4 } else { 1 };
    let mut task = _block_idx_x() as usize * warps + warp;
    let sh = (warp * RING) as u32;
    let (mut j, mut first_task, mut first_row) = (0, 0, 0);
    while j < m.count {
        let tasks = rows_of(&m, j).div_ceil(8) * parts;
        if task >= first_task + tasks {
            first_task += tasks;
            first_row += rows_of(&m, j);
            j += 1;
            continue;
        }
        let p = pick(&m, j);
        let local = task - first_task;
        let row = local / parts * 8;
        let (mut left, mut right) = (ZERO, ZERO);
        let first = if split { local % 4 } else { 0 };
        for part in first..if split { first + 1 } else { 4 } {
            let partial = pick_range::<L, D, W, DF, WF, P, S>(p, sh, row, part, lane);
            if split {
                let mut out = p;
                out.output = partials
                    + ((4 * first_row + part * p.rows as usize) * p.tokens as usize * 4) as u64;
                out.stride = p.rows;
                write(out, partial, lane / 4, row + lane % 4 * 2);
                continue;
            }
            match part {
                0 => left = partial,
                1 => left = canonical::add(left, partial),
                2 => right = partial,
                _ => right = canonical::add(right, partial),
            }
        }
        if !split {
            write(p, canonical::add(left, right), lane / 4, row + lane % 4 * 2);
        }
        task += 192 * warps;
    }
}

/// MXFP8 matrices are flagged by bit 0 of their (256-byte aligned) weight address; their
/// UE8M0 scales follow the E4M3 bytes at the next 256-byte boundary.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn mx8_base(weight: u64) -> u64 {
    weight & !1
}

/// The matrix's own storage picks the stage format: F stages are 112 bytes smaller, so
/// they ring DF deep in the same RING bytes per warp. Same K walk, chains and merge.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn pick_range<
    const L: usize,
    const D: usize,
    const W: usize,
    const DF: usize,
    const WF: usize,
    const P: usize,
    const S: u8,
>(
    p: Bf16Rows,
    sh: u32,
    row: usize,
    part: usize,
    lane: usize,
) -> canonical::Acc {
    if S == 1 || S == 2 && p.weight & 1 != 0 {
        // K32 stages need even range boundaries: (K / 16) % 8 == 0 (every production K).
        // M3..16 keep K16 stages (four K32 stages at M4 measured +1.8 us).
        if (p.cols as usize / 16) % 8 == 0 {
            match L {
                // K64 stages in single-storage launches only: in the mixed-storage group entry
                // they spill under the 128-register cap (measured +5..+14 us per group).
                4 if S == 1 && crate::kernel_params::latency_mx4(p.cols as usize) => return range_mx4::<5, 4>(p, sh, row, part, lane),
                4 => return range_mx2::<4, 11, 10>(p, sh, row, part, lane),
                8 => return range_mx2::<8, 7, 6>(p, sh, row, part, lane),
                _ => {}
            }
        }
        range::<L, DF, WF, P, true>(p, sh, row, part, lane)
    } else {
        range::<L, D, W, P, false>(p, sh, row, part, lane)
    }
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
/// S: matrix storage. 0 = BF16 and 1 = MXFP8 for solo launches (one code path each),
/// 2 = per matrix from the weight flag (groups mix BF16 and MXFP8 matrices).
unsafe fn dispatch<const ICS: bool, const S: u8>(m: Mats, partials: u64) {
    match m.a.tokens {
        // BF16 stages: 384/512/768/1280 B; MXFP8 stages: 272/400/656/1168 B.
        1 => fast::<4, 13, 12, 22, 21, 0, ICS, S>(m, partials),
        2 => fast::<8, 10, 9, 15, 14, 0, ICS, S>(m, partials),
        3..=4 => fast::<16, 6, 5, 9, 8, 0, ICS, S>(m, partials),
        _ => fast::<32, 4, 3, 5, 4, 0, ICS, S>(m, partials),
    }
}

/// K % 64 != 0: four warp-private K16 stages, with three younger copy groups in flight.
/// Small N tasks are exactly the four canonical ranges; a second pass merges.
/// # Safety
/// Launch 192 CTAs, M<=16, rows2=0. N<=640: 32 threads, partials=4*M*N floats;
/// otherwise 128 threads. Dynamic shared memory is RING bytes per warp.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn generic(p: Bf16Rows, partials: u64) {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let warps = _block_dim_x() as usize / 32;
    let split = p.rows <= 640;
    let parts = if split { 4 } else { 1 };
    let mut task = _block_idx_x() as usize * warps + warp;
    let sh = (warp * RING) as u32;
    while task < (p.rows as usize).div_ceil(8) * parts {
        let row = task / parts * 8;
        let tiles = (p.cols as usize).div_ceil(16);
        let (mut left, mut right) = (ZERO, ZERO);
        let first = if split { task % 4 } else { 0 };
        for part in first..if split { first + 1 } else { 4 } {
            let (mut hi, mut lo) = (ZERO, ZERO);
            let (mut tile, end) = canonical::range(tiles, part);
            for t in tile..(tile + 4).min(end) {
                stage(p, sh + (t % 4 * 1280) as u32, row, t, lane);
            }
            while tile < end {
                match (end - tile - 1).min(3) {
                    3 => core::arch::asm!("cp.async.wait_group 3;"),
                    2 => core::arch::asm!("cp.async.wait_group 2;"),
                    1 => ptx::wait_one(),
                    _ => ptx::wait(),
                }
                ptx::warp_sync();
                let current = sh + (tile % 4 * 1280) as u32;
                let b = ptx::load2(current + 1024 + (lane * 8) as u32);
                hi = canonical::madd(ptx::load4(current + (lane * 16) as u32), b, hi);
                lo = canonical::madd(ptx::load4(current + 512 + (lane * 16) as u32), b, lo);
                ptx::warp_sync();
                if tile + 4 < end {
                    stage(p, current, row, tile + 4, lane);
                }
                tile += 1;
            }
            let partial = canonical::add(hi, lo);
            if split {
                let mut out = p;
                out.output = partials + (part * p.rows as usize * p.tokens as usize * 4) as u64;
                out.stride = p.rows;
                write(out, partial, lane / 4, row + lane % 4 * 2);
                continue;
            }
            match part {
                0 => left = partial,
                1 => left = canonical::add(left, partial),
                2 => right = partial,
                _ => right = canonical::add(right, partial),
            }
        }
        if !split {
            write(p, canonical::add(left, right), lane / 4, row + lane % 4 * 2);
        }
        task += 192 * warps;
    }
}

/// Warp-private K16 rings sized by M; small N tasks are exactly the four canonical
/// ranges and a second pass merges (p0+p1)+(p2+p3).
/// # Safety
/// Launch 192 CTAs, M<=16, rows2=0. N<=640: 32 threads, partials=4*M*N floats;
/// otherwise 128 threads. Dynamic shared memory is RING bytes per warp.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_latency(p: Bf16Rows, partials: u64) {
    if p.cols % 64 != 0 {
        generic(p, partials);
    } else {
        dispatch::<false, 0>(solo(p), partials);
    }
}

/// MXFP8 read-as-stored twin of flash_bf16_mma_latency: E4M3 weights in the packed BF16
/// element order at one byte per element, then at the next 256-byte boundary UE8M0
/// scales as [row / 8][K / 32][8] bytes (padding rows hold 127; `_scales` is unused). Each B fragment expands in registers to the BF16 words
/// main expands at load, and the K walk, MMA chains and merges are the BF16 kernel's, so
/// the output bytes are main's. Half the weight bytes per token.
/// # Safety
/// As flash_bf16_mma_latency, plus K % 64 == 0, N % 4 == 0 and every scale in 10..=134.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_mx8_mma_latency(mut p: Bf16Rows, _scales: u64, partials: u64) {
    p.weight |= 1;
    dispatch::<false, 1>(solo(p), partials);
}

/// One launch for up to four projections of the same decode input (pairs[i] first and
/// second matrices; rows 0 = absent). Matrix j's ranges and (p0+p1)+(p2+p3) merge are
/// the solo launch's; groups of <= 640 rows merge inside the CTA (no merge launch).
/// # Safety
/// Launch 192 CTAs x 128 threads, M<=16, K%64==0, 20480 dynamic shared bytes.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_latency_group(
    g: crate::kernel_params::Bf16Pairs,
    partials: u64,
) {
    dispatch::<true, 2>(group(g), partials);
}

/// flash_bf16_mma_latency_group when every present matrix is MXFP8 (flagged): one storage
/// path, so the MXFP8 code needs no BF16 registers. Same ranges, chains and merges.
/// # Safety
/// As flash_bf16_mma_latency_group, plus every matrix flagged and N % 4 == 0.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_mx8_mma_latency_group(
    g: crate::kernel_params::Bf16Pairs,
    partials: u64,
) {
    dispatch::<true, 1>(group(g), partials);
}

/// # Safety
/// Stream ordered after all four range writers; partials contains 4*M*N floats.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_latency_merge(p: Bf16Rows, partials: u64) {
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let at = _block_idx_x() as usize * 128 + _thread_idx_x() as usize;
    let count = p.rows as usize * p.tokens as usize;
    if at < count {
        let x = (partials as *const f32).add(at);
        super::put(
            p,
            (*x + *x.add(count)) + (*x.add(2 * count) + *x.add(3 * count)),
            at / p.rows as usize,
            at % p.rows as usize,
        );
    }
}

// LBF12 head projection and encoder. The projection uses next_head::project's
// walk, accumulation chains and argmax order.
#[cfg(target_arch = "nvptx64")]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.mma.m16n8k16.row.col.bf16"]
    fn lbf_mma(a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32, c0: f32, c1: f32, c2: f32, c3: f32) -> crate::canonical::Acc;
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf_madd(a: [u32; 4], b: [u32; 2], c: crate::canonical::Acc) -> crate::canonical::Acc { lbf_mma(a[0], a[1], a[2], a[3], b[0], b[1], c.a, c.b, c.c, c.d) }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn lbf_range(tiles: usize, part: usize) -> (usize, usize) { (tiles * part / 4, tiles * (part + 1) / 4) }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn lbf_better(a: (f32, u32), b: (f32, u32)) -> (f32, u32) { if b.0 > a.0 || b.0 == a.0 && b.1 < a.1 { b } else { a } }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf_shared() -> u32 { let p: u32; core::arch::asm!("{{ .shared .align 16 .b8 scratch[1024]; mov.u32 {p}, scratch; }}", p = out(reg32) p); p }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf_sload(p: u32) -> f32 { let v: f32; core::arch::asm!("ld.shared.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p); v }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf_store(p: u32, v: f32) { core::arch::asm!("st.shared.f32 [{p}], {v};", p = in(reg32) p, v = in(reg32) v); }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf_sync() { core::arch::asm!("bar.sync 0;"); }
#[cfg(target_arch = "nvptx64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct LbfFragment([u32; 4]);
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf_load(p: *const LbfFragment) -> [u32; 4] {
    let (a,b,c,d): (u32,u32,u32,u32);
    core::arch::asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];", a=out(reg32)a,b=out(reg32)b,
        c=out(reg32)c,d=out(reg32)d,p=in(reg64)p);
    [a,b,c,d]
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn lbf_bits(x: f32) -> u32 { unsafe { core::mem::transmute(x) } }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn lbf_float(x: u32) -> f32 { unsafe { core::mem::transmute(x) } }
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf_shuffle(v: u32, d: u32) -> u32 { crate::shuffle_xor(u32::MAX,v,d,31) }
/// LBF12: a packed 512-byte head slot (8 rows x K32) holds per lane 8 values x 12
/// bits (sign << 11 | exponent offset << 7 | mantissa; offset 15 = signed zero) in its first
/// 384 bytes (lane * 12); the 8 row base exponents sit in bases[slot][8], four slots per
/// 32-byte sector. The decode gives back the identical BF16 bits.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub(super) unsafe fn lbf12_decode(slot: *const u8, bases: *const u8, lane: usize) -> [u32; 4] {
    let w = slot.add(lane*12) as *const u32;
    let lo = *w as u64 | (*w.add(1) as u64) << 32;
    let hi = *w.add(2) as u64;
    let base = *bases.add(lane/4) as u32;
    let f = |j: u32| -> u32 { (if j < 5 { lo >> (12*j) } else if j == 5 { lo >> 60 | hi << 4 } else { hi >> (12*j-64) }) as u32 & 0xFFF };
    let v = |x: u32| -> u32 { let off = (x >> 7) & 15; (x >> 11) << 15 | if off == 15 { 0 } else { (base+off) << 7 | (x & 0x7F) } };
    [v(f(0)) | v(f(1)) << 16, v(f(2)) | v(f(3)) << 16, v(f(4)) | v(f(5)) << 16, v(f(6)) | v(f(7)) << 16]
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn lbf_project<const GROUPS: usize>(p: crate::kernel_params::Bf16Rows, partial: u64, flags: u64, bases: u64) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp,lane) = (tid/32,tid%32);
    let (n,k,m) = (p.rows as usize,p.cols as usize,p.tokens as usize);
    let blocks = core::arch::nvptx::_grid_dim_x() as usize; let block = core::arch::nvptx::_block_idx_x() as usize;
    let span = (n/8 + blocks - 1)/blocks;
    let stop = if (block+1)*span < n/8 { (block+1)*span } else { n/8 };
    let mut best = [(f32::NEG_INFINITY,u32::MAX); GROUPS]; let mut valid = [1u32; GROUPS];
    let mut nt = block*span+warp;
    while nt < stop {
        let mut left = [[0.0;2]; GROUPS]; let mut right = left;
        let packed = lbf_load((flags as *const LbfFragment).add(nt));
        let mut part = 0;
        while part < 4 {
            let mut sums = [crate::canonical::ZERO; GROUPS];
            let (start,end) = lbf_range(k/16,part);
            let mut pair = start/2;
            while pair < end/2 {
                let b = if (packed[pair/32] >> (pair%32)) & 1 != 0 {
                    lbf12_decode((p.weight as *const u8).add((nt*(k/32)+pair)*512), (bases as *const u8).add((nt*(k/32)+pair)*8), lane) }
                    else { lbf_load((p.weight as *const LbfFragment).add((nt*(k/32)+pair)*32+lane)) };
                let mut g = 0;
                while g < GROUPS {
                    let a = p.input as *const LbfFragment;
                    sums[g] = lbf_madd((*a.add((g*(k/16)+pair*2)*32+lane)).0,[b[0],b[1]],sums[g]);
                    sums[g] = lbf_madd((*a.add((g*(k/16)+pair*2+1)*32+lane)).0,[b[2],b[3]],sums[g]);
                    g += 1;
                }
                pair += 1;
            }
            let mut g = 0;
            while g < GROUPS {
                let z = [sums[g].a+sums[g].c,sums[g].b+sums[g].d];
                let mut j = 0;
                while j < 2 { match part { 0 => left[g][j]=z[j], 1 => left[g][j]+=z[j],
                    2 => right[g][j]=z[j], _ => right[g][j]+=z[j] } j += 1; }
                g += 1;
            }
            part += 1;
        }
        let mut g = 0;
        while g < GROUPS { let t=g*8+lane/4;
            if t<m { let mut j = 0; while j < 2 { let id=nt*8+lane%4*2+j; let value=left[g][j]+right[g][j];
                *(p.output as *mut f32).add(t*n+id)=value;
                valid[g] &= u32::from(lbf_bits(value) & 0x7f80_0000 != 0x7f80_0000); best[g]=lbf_better(best[g],(value,id as u32));
                j += 1; } }
            g += 1;
        }
        nt += 4;
    }
    let sh = lbf_shared();
    let mut g = 0;
    while g < GROUPS {
        let mut d = 2;
        while d > 0 { best[g]=lbf_better(best[g],(lbf_float(lbf_shuffle(lbf_bits(best[g].0),d)),lbf_shuffle(best[g].1,d)));
            valid[g] &= lbf_shuffle(valid[g],d); d -= 1; }
        if lane%4==0 { let at=sh+((warp*GROUPS*8+g*8+lane/4)*12) as u32;
            lbf_store(at,best[g].0); lbf_store(at+4,best[g].1 as f32); lbf_store(at+8,valid[g] as f32); }
        g += 1;
    }
    lbf_sync();
    if tid<m { let (g,t)=(tid/8,tid%8); let mut best=(f32::NEG_INFINITY,u32::MAX); let mut valid=1;
        let mut w = 0;
        while w < 4 { let at=sh+((w*GROUPS*8+g*8+t)*12) as u32;
            best=lbf_better(best,(lbf_sload(at),lbf_sload(at+4) as u32)); valid &= lbf_sload(at+8) as u32; w += 1; }
        let out=(partial as *mut f32).add((tid*blocks+block)*3);
        *out=best.0; *out.add(1)=best.1 as f32; *out.add(2)=valid as f32;
    }
}
/// flash_next_head on an LBF12 head (flash_next_head_lbf12_encode): same walk, chains and
/// argmax; packed slots decode to the identical BF16 fragments. flags: [N/8][4] u32 bitmaps;
/// bases: [N/8 * K/32][8] bytes.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_lbf12(p: crate::kernel_params::Bf16Rows, partial: u64, flags: u64, bases: u64) {
    if p.tokens == 0 || p.tokens > 16 || p.cols as usize > 4096 { return; }
    if p.tokens<=8 { lbf_project::<1>(p,partial,flags,bases); } else { lbf_project::<2>(p,partial,flags,bases); }
}



// LBF12 head encoder, next to the head projection implementation.
#[cfg(target_arch = "nvptx64")]
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct EncFragment([u32; 4]);
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn enc_load(p: *const EncFragment) -> [u32; 4] {
    let (a,b,c,d): (u32,u32,u32,u32);
    core::arch::asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];", a=out(reg32)a,b=out(reg32)b,
        c=out(reg32)c,d=out(reg32)d,p=in(reg64)p);
    [a,b,c,d]
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn enc_shuffle(v: u32, d: u32) -> u32 { crate::shuffle_xor(u32::MAX,v,d,31) }
/// Re-encodes the resident head in place, one warp per 8-row x K32 slot: if every row's nonzero
/// exponents span <= 14 (no subnormal), the slot becomes LBF12 and its bit in flags is set
/// (flags zeroed by the host); any other slot stays raw BF16. N % 8 == 0, K % 32 == 0, K <= 4096.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_head_lbf12_encode(p: crate::kernel_params::FlashBf16Pack, flags: u64, bases: u64) {
    let i = core::arch::nvptx::_block_idx_x() as usize*core::arch::nvptx::_block_dim_x() as usize+core::arch::nvptx::_thread_idx_x() as usize;
    let (tile, lane) = (i/32, i%32);
    let (n, k) = (p.rows as usize, p.cols as usize);
    if tile >= n/8*(k/32) { return; }
    let slot = (p.weight as *mut u8).add(tile*512);
    let v = enc_load((slot as *const EncFragment).add(lane));
    let h = |j: usize| (v[j/2] >> (16*(j%2))) & 0xFFFF;
    let (mut lo, mut hi, mut bad) = (255u32, 0u32, 0u32);
    let mut j = 0;
    while j < 8 { let x = h(j); if x & 0x7FFF != 0 { let e = (x >> 7) & 0xFF; bad |= u32::from(e == 0);
        if e < lo { lo = e; } if e > hi { hi = e; } } j += 1; }
    let mut d = 1;
    while d <= 2 { let (l, u) = (enc_shuffle(lo, d), enc_shuffle(hi, d)); if l < lo { lo = l; } if u > hi { hi = u; }
        bad |= enc_shuffle(bad, d); d += 1; }
    let ok = u32::from(bad == 0 && (hi < lo || hi - lo <= 14));
    let all: u32;
    core::arch::asm!("{{ .reg .pred p, q; setp.ne.u32 p, {o}, 0; vote.sync.all.pred q, p, -1; selp.u32 {a}, 1, 0, q; }}", o = in(reg32) ok, a = out(reg32) all);
    if all == 0 { return; }
    let base = if hi < lo { 0 } else { lo };
    let f = |j: usize| -> u64 { let x = h(j); let off = if x & 0x7FFF == 0 { 15 } else { ((x >> 7) & 0xFF) - base }; ((x >> 15) << 11 | off << 7 | (x & 0x7F)) as u64 };
    let lo64 = f(0) | f(1) << 12 | f(2) << 24 | f(3) << 36 | f(4) << 48 | f(5) << 60;
    let hi64 = f(5) >> 4 | f(6) << 8 | f(7) << 20;
    let w = slot.add(lane*12) as *mut u32;
    *w = lo64 as u32; *w.add(1) = (lo64 >> 32) as u32; *w.add(2) = hi64 as u32;
    if lane % 4 == 0 { *(bases as *mut u8).add(tile*8+lane/4) = base as u8; }
    if lane == 0 {
        let (nt, pair) = (tile/(k/32), tile%(k/32));
        core::arch::asm!("red.global.or.b32 [{a}], {v};", a = in(reg64) flags + ((nt*4 + pair/32)*4) as u64, v = in(reg32) 1u32 << (pair%32));
    }
}

include!("dense_latency_w16.rs");
include!("dense_latency_fold.rs");

#[cfg(test)]
mod tests {
    use super::input_word;
    // (N,K,launches/token), from the pooled rows callers; sum = 675.
    const SHAPES: [(usize, usize, usize); 12] = [
        (4, 10240, 96),
        (320, 10240, 97),
        (10240, 2560, 37),
        (6144, 2560, 36),
        (48, 2560, 72),
        (2560, 6144, 48),
        (12288, 2560, 12),
        (512, 2560, 72),
        (640, 2560, 108),
        (1, 2560, 48),
        (2560, 640, 48),
        (2560, 2560, 1),
    ];

    #[test]
    fn pipeline_ring_persistent_ranges_and_partial_storage() {
        use crate::kernel_params::{
            flash_bf16_element as element, flash_bf16_packed, FLASH_BF16_INPUT_BYTES,
        };
        for (n, k, _) in SHAPES {
            let parts = if n <= 640 { 4 } else { 1 };
            let mut owners = vec![0; n.div_ceil(8) * parts];
            let stride = 192 * if n <= 640 { 1 } else { 4 };
            for block in 0..stride {
                for task in (block..owners.len()).step_by(stride) {
                    owners[task] += 1;
                }
            }
            assert!(owners.iter().all(|&v| v == 1));
            for m in [1usize, 2, 8, 16] {
                let input_bytes = m.div_ceil(16) * k.div_ceil(16) * 128 * 8;
                assert!(input_bytes + 4 * m * n.min(640) * 4 <= FLASH_BF16_INPUT_BYTES);
            }
            for part in 0..4 {
                let (start, end) = (k / 16 * part / 4, k / 16 * (part + 1) / 4);
                let mut ring = [None; 4];
                for t in start..(start + 4).min(end) {
                    ring[t % 4] = Some(t);
                }
                for t in start..end {
                    assert_eq!(ring[t % 4].take(), Some(t));
                    if t + 4 < end {
                        ring[t % 4] = Some(t + 4);
                    }
                    if flash_bf16_packed(n, k) {
                        for r in 0..n {
                            for lane in [0, 2] {
                                let at = element(r, t * 16 + lane * 2, k, n);
                                assert_eq!(at % 8, 0);
                                assert_eq!(at + 4, element(r, t * 16 + (lane + 1) * 2, k, n));
                                assert!(at + 8 <= n * k);
                            }
                        }
                    }
                }
                assert_eq!(ring, [None; 4]);
            }
        }
    }

    #[test]
    fn dense_latency_operand_bytes_and_ownership() {
        assert_eq!(SHAPES.iter().map(|s| s.2).sum::<usize>(), 675);
        for (n, k, _) in SHAPES {
            for m in [1usize, 2, 8, 16, 17, 65] {
                let kt = k / 16;
                for mt in (0..m).step_by(16) {
                    for tile in 0..kt {
                        let staged: [usize; 1024] = core::array::from_fn(|i| {
                            let (plane, tid, word) = (i / 512, i % 512 / 4, i % 4);
                            plane * m.div_ceil(16) * kt * 128
                                + ((mt / 16 * kt + tile / 4 * 4 + tid / 32) * 32 + tid % 32) * 4
                                + word
                        });
                        for lane in 0..32 {
                            let direct = input_word(kt, mt, tile, lane);
                            for plane in 0..2 {
                                for word in 0..4 {
                                    assert_eq!(
                                        staged[plane * 512 + tile % 4 * 128 + lane * 4 + word],
                                        plane * m.div_ceil(16) * kt * 128 + direct + word
                                    );
                                }
                            }
                        }
                    }
                }
                for row in (0..n).step_by(8) {
                    for lane in 0..32 {
                        for tile in 0..kt {
                            let b = (lane / 16 * 4 + tile % 4) * 16 + lane % 16;
                            let i = ((row % 32 / 8) * 64 + b * 8 / 16) % 256;
                            let r = row / 32 * 32 + i / 32 * 4 + i % 8 * 2 / 4;
                            let c = tile / 4 * 64 + i % 32 / 8 * 16 + i % 8 * 2 % 4 * 2;
                            for word in 0..4 {
                                let cc = c + (b % 2) * 2 + word % 2 + word / 2 * 8;
                                let direct_r = row + lane / 4;
                                let direct_c = tile * 16 + lane % 4 * 2 + word % 2 + word / 2 * 8;
                                let off = (r < n).then_some((r, cc));
                                let on = (direct_r < n).then_some((direct_r, direct_c));
                                assert_eq!(off, on, "N{n} K{k} row{row} lane{lane} tile{tile}");
                                if r < n && crate::kernel_params::flash_bf16_packed(n, k) {
                                    let element = crate::kernel_params::flash_bf16_element;
                                    assert_eq!(
                                        element(r, c, k, n) + b % 2 * 4 + word,
                                        element(direct_r, tile * 16 + lane % 4 * 2, k, n) + word
                                    );
                                }
                            }
                        }
                    }
                }
                let mut owners = vec![0u8; m * n];
                for slot in 0..n.div_ceil(8) * m.div_ceil(16) * 128 {
                    let (block, lane) = (slot / 128, slot / 4 % 32);
                    let t = block % m.div_ceil(16) * 16 + lane / 4 + slot % 4 / 2 * 8;
                    let r = block / m.div_ceil(16) * 8 + lane % 4 * 2 + slot % 2;
                    if t < m && r < n {
                        owners[t * n + r] += 1;
                    }
                }
                assert!(owners.iter().all(|&v| v == 1));
            }
        }
    }
}

#[cfg(target_arch = "nvptx64")]
#[path = "verify_mx4.rs"]
mod verify_mx4;
