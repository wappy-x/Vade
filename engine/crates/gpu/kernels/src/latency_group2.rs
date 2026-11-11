//! Small BF16 groups at M17-32 in one launch: both 16-token input tiles per K16
//! stage, so each weight fragment is read once for every token. Private g2_ copies of every
//! helper (no existing entry's codegen changes); its own module so the codegen-unit merge
//! leaves every other entry alone.
#[cfg(target_arch = "nvptx64")]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.mma.m16n8k16.row.col.bf16"]
    fn g2_mma(a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32,
        c0: f32, c1: f32, c2: f32, c3: f32) -> crate::canonical::Acc;
}
/// canonical::madd and canonical::add (same intrinsic, same adds).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn g2_madd(a: [u32; 4], b: [u32; 2], c: crate::canonical::Acc) -> crate::canonical::Acc {
    g2_mma(a[0], a[1], a[2], a[3], b[0], b[1], c.a, c.b, c.c, c.d)
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn g2_add(a: crate::canonical::Acc, b: crate::canonical::Acc) -> crate::canonical::Acc {
    crate::canonical::Acc { a: a.a + b.a, b: a.b + b.b, c: a.c + b.c, d: a.d + b.d }
}
/// Matrix j of the group (dense_latency's pick): weight, output, rows, stride; rows2 = 0.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn g2_pick(g: crate::kernel_params::Bf16Pairs, j: usize) -> crate::kernel_params::Bf16Rows {
    // A match, not g.pairs[j / 2]: a dynamically indexed parameter array goes to local memory.
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
/// Stage bytes per K16 tile: token tile 0 hi and lo planes, token tile 1 hi and lo (512 each),
/// then the B fragments of the task's two 8-row groups (32 x 8 bytes each).
#[cfg(target_arch = "nvptx64")]
const G2_STAGE: u32 = 2560;
/// Ring depth per warp (dynamic shared = 4 warps x G2_D x G2_STAGE bytes).
#[cfg(target_arch = "nvptx64")]
const G2_D: u32 = 8;
/// One 8-row group's B-fragment source for lane `lane` (flash_bf16_packed's layouts, or row-major
/// for N % 4 != 0): pointer, K64 and K16 strides, mode (0 = rows past N, 1 = 16-byte copy on even
/// lanes, 2 = two 4-byte copies).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn g2_source(p: crate::kernel_params::Bf16Rows, row: usize, lane: usize) -> (*const u8, u32, u32, u32) {
    let (k, n) = (p.cols as usize, p.rows as usize);
    let kt = k / 16;
    let r = row + lane / 4;
    let weight = p.weight as *const u8;
    if r >= n {
        (weight, 0u32, 0u32, 0u32)
    } else if n != 0 && n % 4 == 0 && k != 0 && k % 16 == 0 {
        // flash_bf16_packed(n, k)
        if n % 64 == 0 {
            let at = ((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 256 + lane * 8;
            (weight.add(at), 8192, 256, (lane % 2 == 0) as u32)
        } else {
            (weight.add(r / 4 * kt * 128 + lane % 16 * 8), 512, 128, (lane % 2 == 0) as u32)
        }
    } else {
        (weight.add((r * k + lane % 4 * 2) * 2), 128, 32, 2)
    }
}
/// One canonical range of a 16-row task (two 8-row groups) for both token tiles: the
/// stage holds the input once and both groups' B fragments, so the input is staged once per 16
/// rows instead of once per 8. Per (row, token) the K16 chains are the g2_range chains (hi and
/// lo chains in ascending K16 tiles, then hi + lo). Returns [tile 0, tile 1] x [group 0, group 1].
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn g2_range16(p: crate::kernel_params::Bf16Rows, sh: u32, row: usize, part: usize, lane: usize) -> [crate::canonical::Acc; 4] {
    let kt = p.cols as usize / 16;
    // canonical::range(kt, part)
    let (start, end) = ((kt * part / 4) as u32, (kt * (part + 1) / 4) as u32);
    let (w0, s40, s10, m0) = g2_source(p, row, lane);
    let (w1, s41, s11, m1) = g2_source(p, row + 8, lane);
    let x = (p.input as *const u8).add(lane * 16);
    let (plane, second) = (p.input_plane_words as usize * 4, kt * 512);
    let mut g = 0u32;
    while g < 2 {
        if (if g == 0 { m0 } else { m1 }) == 0 {
            let mut dst = sh + 2048 + g * 256 + (lane * 8) as u32;
            let mut i = 0;
            while i < G2_D {
                crate::ptx::store(dst, 0.0);
                crate::ptx::store(dst + 4, 0.0);
                dst += G2_STAGE;
                i += 1;
            }
        }
        g += 1;
    }
    let stage = |at: u32, tile: u32| {
        let t = x.add((tile * 512) as usize);
        crate::ptx::copy16(at + (lane * 16) as u32, t.cast());
        crate::ptx::copy16(at + 512 + (lane * 16) as u32, t.add(plane).cast());
        crate::ptx::copy16(at + 1024 + (lane * 16) as u32, t.add(second).cast());
        crate::ptx::copy16(at + 1536 + (lane * 16) as u32, t.add(second + plane).cast());
        let src = w0.add(((tile >> 2) * s40 + (tile & 3) * s10) as usize);
        let dst = at + 2048 + (lane * 8) as u32;
        if m0 == 1 {
            crate::ptx::copy16(dst, src.cast());
        } else if m0 == 2 {
            crate::ptx::copy4(dst, src.cast());
            crate::ptx::copy4(dst + 4, src.add(16).cast());
        }
        let src = w1.add(((tile >> 2) * s41 + (tile & 3) * s11) as usize);
        let dst = at + 2304 + (lane * 8) as u32;
        if m1 == 1 {
            crate::ptx::copy16(dst, src.cast());
        } else if m1 == 2 {
            crate::ptx::copy4(dst, src.cast());
            crate::ptx::copy4(dst + 4, src.add(16).cast());
        }
        crate::ptx::commit();
    };
    let mut at = sh;
    let mut i = 0;
    while i < G2_D {
        if start + i < end {
            stage(at, start + i);
        } else {
            crate::ptx::commit();
        }
        at += G2_STAGE;
        i += 1;
    }
    let z = crate::canonical::ZERO;
    let (mut h0, mut l0, mut h1, mut l1) = (z, z, z, z);
    let (mut g0, mut k0, mut g1, mut k1) = (z, z, z, z);
    let mut current = sh;
    let last = sh + (G2_D - 1) * G2_STAGE;
    let mut tile = start;
    while tile < end {
        core::arch::asm!("cp.async.wait_group 7;");
        crate::ptx::warp_sync();
        let (b, c) = (crate::ptx::load2(current + 2048 + (lane * 8) as u32), crate::ptx::load2(current + 2304 + (lane * 8) as u32));
        let (a0, a1) = (crate::ptx::load4(current + (lane * 16) as u32), crate::ptx::load4(current + 512 + (lane * 16) as u32));
        let (a2, a3) = (crate::ptx::load4(current + 1024 + (lane * 16) as u32), crate::ptx::load4(current + 1536 + (lane * 16) as u32));
        h0 = g2_madd(a0, b, h0);
        l0 = g2_madd(a1, b, l0);
        h1 = g2_madd(a2, b, h1);
        l1 = g2_madd(a3, b, l1);
        g0 = g2_madd(a0, c, g0);
        k0 = g2_madd(a1, c, k0);
        g1 = g2_madd(a2, c, g1);
        k1 = g2_madd(a3, c, k1);
        crate::ptx::warp_sync();
        if tile + G2_D < end {
            stage(current, tile + G2_D);
        } else {
            crate::ptx::commit();
        }
        current = if current == last { sh } else { current + G2_STAGE };
        tile += 1;
    }
    [g2_add(h0, l0), g2_add(h1, l1), g2_add(g0, k0), g2_add(g1, k1)]
}
/// output[token][row] (flash_bf16.rs put/write for one matrix with rows2 = 0).
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn g2_write(p: crate::kernel_params::Bf16Rows, v: crate::canonical::Acc, token: usize, row: usize) {
    let y = p.output as *mut f32;
    let (n, s, m) = (p.rows as usize, p.stride as usize, p.tokens as usize);
    if token < m && row < n { *y.add(token * s + row) = v.a; }
    if token < m && row + 1 < n { *y.add(token * s + row + 1) = v.b; }
    if token + 8 < m && row < n { *y.add((token + 8) * s + row) = v.c; }
    if token + 8 < m && row + 1 < n { *y.add((token + 8) * s + row + 1) = v.d; }
}
/// flash_bf16_mma_latency_group's split path (intra: warp w = canonical range w, warp 0 merges
/// (p0+p1)+(p2+p3)) for BF16 groups at 17 <= M <= 32 in 16-row CTA tasks (the earlier path ran
/// 8-row tasks): each warp streams its range once for both token tiles and both 8-row groups
/// (g2_range16), so the bytes of every (row, token) are intra's at M <= 16.
/// # Safety
/// Grid 192 x 1, block 128, dynamic shared 4 x G2_D x G2_STAGE bytes; g as the group entry's
/// (BF16 weights, the latency input pack of M tokens, two 16-token tiles), 17 <= M <= 32.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_latency_group2(g: crate::kernel_params::Bf16Pairs) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let sh = (warp as u32) * G2_D * G2_STAGE;
    let mut task = core::arch::nvptx::_block_idx_x() as usize;
    let mut j = 0;
    let mut first = 0;
    while j < 4 {
        let p = g2_pick(g, j);
        let tasks = (p.rows as usize + 15) / 16;
        if task >= first + tasks {
            first += tasks;
            j += 1;
            continue;
        }
        let row = (task - first) * 16;
        let z = g2_range16(p, sh, row, warp, lane);
        let slot = sh + (lane * 64) as u32;
        let mut i = 0;
        while i < 4 {
            let v = match i { 0 => z[0], 1 => z[1], 2 => z[2], _ => z[3] };
            crate::ptx::store(slot + i * 16, v.a);
            crate::ptx::store(slot + i * 16 + 4, v.b);
            crate::ptx::store(slot + i * 16 + 8, v.c);
            crate::ptx::store(slot + i * 16 + 12, v.d);
            i += 1;
        }
        crate::ptx::sync();
        if warp == 0 {
            let get = |w: u32, i: u32| {
                let at = w * G2_D * G2_STAGE + (lane * 64) as u32 + i * 16;
                crate::canonical::Acc { a: crate::ptx::load(at), b: crate::ptx::load(at + 4), c: crate::ptx::load(at + 8), d: crate::ptx::load(at + 12) }
            };
            // i = tile + 2 x group: (p0+p1)+(p2+p3) per (row group, token tile).
            let mut i = 0;
            while i < 4 {
                let out = g2_add(g2_add(get(0, i), get(1, i)), g2_add(get(2, i), get(3, i)));
                g2_write(p, out, (i % 2) as usize * 16 + lane / 4, row + (i / 2) as usize * 8 + lane % 4 * 2);
                i += 1;
            }
        }
        crate::ptx::sync();
        task += 192;
    }
}
