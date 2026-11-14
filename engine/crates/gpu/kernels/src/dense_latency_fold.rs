// Included by dense_latency.rs (same module, so no codegen unit of its own).
// BF16 M1 solo launches with N <= 64 (kernel_params::latency_fold1): one CTA per
// eight-row group, warp w = canonical range w, in the MXFP8 M1 structure on BF16 fragments:
// a CTA copy of the input's token-0 a0/a2 words (64 bytes per K16 tile, cta_input1) and per
// warp a ring of K64 stages that carry only the eight rows' B fragments (1 KB), one
// transposed MMA per K16 tile in ascending order (range()'s chains). The four range sums
// merge (p0+p1)+(p2+p3) in shared memory, as the merge pass does. Same bytes.

/// K64 stages per warp ring (RING bytes).
#[cfg(target_arch = "nvptx64")]
const FOLD_D: usize = RING / 1024;

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
unsafe fn range_b1(p: Bf16Rows, sh: u32, row: usize, part: usize, lane: usize, cin: u32) -> canonical::Acc {
    const SIZE: u32 = 1024;
    let (k, n) = (p.cols as usize, p.rows as usize);
    let kt = k / 16;
    let (start, end) = canonical::range(kt, part);
    let (start, end) = (start as u32 / 4, end as u32 / 4);
    let weight = p.weight as *const u8;
    // Stage layout [K16 tile j][lane] 8 bytes = the packed weight's own quad block; lane
    // copies 32 bytes of it (N % 64 == 0: the whole block; else its 4-row group's half).
    let (w, ws, wd) = if n % 64 == 0 {
        (weight.add(((row / 64 * (kt / 4)) * 32 + row / 8 % 8 * 4) * 256 + lane * 32), 8192u32, lane * 32)
    } else {
        let (q, j) = (lane / 16, lane % 16);
        let w = if row + q * 4 < n { weight.add((row / 4 + q) * kt * 128 + j * 32) } else { core::ptr::null() };
        (w, 512u32, j / 4 * 256 + q * 128 + j % 4 * 32)
    };
    if row + lane / 4 >= n {
        // Rows >= N: zero B fragments in every slot once; copies never touch them.
        let mut dst = sh + (lane * 8) as u32;
        for _ in 0..FOLD_D {
            for j in 0..4u32 {
                ptx::store(dst + j * 256, 0.0);
                ptx::store(dst + j * 256 + 4, 0.0);
            }
            dst += SIZE;
        }
    }
    let stage = |at: u32, quad: u32| {
        if !w.is_null() {
            let src = w.add((quad * ws) as usize);
            ptx::copy16(at + wd as u32, src.cast());
            ptx::copy16(at + wd as u32 + 16, src.add(16).cast());
        }
        ptx::commit();
    };
    let mut at = sh;
    for i in 0..FOLD_D as u32 {
        if start + i < end {
            stage(at, start + i);
        } else {
            ptx::commit();
        }
        at += SIZE;
    }
    let mut c0 = ZERO;
    let mut current = sh;
    let last = sh + (FOLD_D as u32 - 1) * SIZE;
    let mut quad = start;
    while quad < end {
        core::arch::asm!("cp.async.wait_group {w};", w = const FOLD_D - 1);
        ptx::warp_sync();
        for j in 0..4u32 {
            let b = ptx::load2(current + j * 256 + (lane * 8) as u32);
            // Token 0's hi (lanes 0-3) and lo (lanes 4-7) words, 64 bytes per tile.
            let x = if lane < 8 { ptx::load2(cin + ((quad * 4 + j) * 64) as u32 + (lane * 8) as u32) } else { [0; 2] };
            c0 = tmadd(b, [x[0], 0, x[1], 0], c0);
        }
        ptx::warp_sync();
        if quad + (FOLD_D as u32) < end {
            stage(current, quad + FOLD_D as u32);
        } else {
            ptx::commit();
        }
        current = if current == last { sh } else { current + SIZE };
        quad += 1;
    }
    untranspose(c0, lane)
}

/// # Safety
/// 128 threads; latency_fold1(1, K, N); dynamic shared latency_fold_shared(1, K, N) bytes.
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub(super) unsafe fn fold1(p: Bf16Rows) {
    use core::arch::nvptx::{_block_idx_x, _grid_dim_x, _thread_idx_x};
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let groups = (p.rows as usize).div_ceil(8);
    let mut task = _block_idx_x() as usize;
    if task >= groups {
        return;
    }
    let cin = (4 * RING) as u32;
    cta_input1(p, cin);
    let sh = (warp * RING) as u32;
    while task < groups {
        let row = task * 8;
        let partial = range_b1(p, sh, row, warp, lane, cin);
        // Each warp parks its range sum in its own (idle) ring slot; warp 0 merges.
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
        task += _grid_dim_x() as usize;
    }
}
