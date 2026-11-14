// The decode HC block's down group; part of dense_latency.rs (include!, same module).
/// One BF16 decode HC block: intra's CTA tasks for pairs[0] = injection (matrix 0), down
/// (matrix 1), with flash_next_hc_activate in warp 0's merge epilogue for down: same ranges,
/// chains, merges and hc_sigmoid, and the activated rows' packed planes as that kernel writes them.
/// A macro over the ranges' stage shape (L, D, W), so the body sits under the entry's one contract.
#[cfg(target_arch = "nvptx64")]
macro_rules! hc_group { ($m:ident, $planes:ident, $l:literal, $d:literal, $w:literal) => {{
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    let (warp, lane) = (_thread_idx_x() as usize / 32, _thread_idx_x() as usize % 32);
    let sh = (warp * RING) as u32;
    let (task, first) = (_block_idx_x() as usize, rows_of(&$m, 0).div_ceil(8));
    let down = task >= first;
    let p = pick(&$m, down as usize);
    let row = (if down { task - first } else { task }) * 8;
    let partial = range::<$l, $d, $w, 0, false, false>(p, sh, row, warp, lane);
    let slot = sh + (lane * 16) as u32;
    ptx::store(slot, partial.a);
    ptx::store(slot + 4, partial.b);
    ptx::store(slot + 8, partial.c);
    ptx::store(slot + 12, partial.d);
    ptx::sync();
    if warp != 0 { return; }
    let get = |w: u32| {
        let at = (w as usize * RING + lane * 16) as u32;
        canonical::Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
    };
    let out = canonical::add(canonical::add(get(0), get(1)), canonical::add(get(2), get(3)));
    let (t, n, rows) = (lane / 4, row + lane % 4 * 2, p.tokens as usize);
    if !down { return write(p, out, t, n); }
    let act = |v: f32| { let value = v * 0.25; value * crate::flash_next::hc_sigmoid(value) };
    if t < rows {
        let (a, b) = (act(out.a), act(out.b));
        put(p, a, t, n);
        put(p, b, t, n + 1);
        hc_planes!(320, $planes, rows, t, n, a, b);
    }
    if t + 8 < rows {
        let (a, b) = (act(out.c), act(out.d));
        put(p, a, t + 8, n);
        put(p, b, t + 8, n + 1);
        hc_planes!(320, $planes, rows, t + 8, n, a, b);
    }
}} }

/// flash_bf16_mma_latency_group for one decode HC block, activation and HC up's planes included.
/// # Safety
/// Launch ceil(injection rows / 8) + ceil(down rows / 8) CTAs x 128 threads, M <= 16, K % 64 == 0,
/// both matrices BF16, pairs[1] empty, 4 * RING dynamic shared bytes; planes covers the two dense
/// planes of M x 320 and does not overlap the input planes.
// SAFETY: Backend::flash_hc_down_group, the production launch, checks M, K, the weights, pairs[1]
// and the planes range, and passes this grid and shared size. The kernel bench's launch_down_act
// (kernel_head.rs) meets the same list on its own fixture.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_hc_down_latency_group(g: crate::kernel_params::Bf16Pairs, planes: u64) {
    let m = group(g);
    match m.a.tokens {
        1 => hc_group!(m, planes, 4, 13, 12),
        2 => hc_group!(m, planes, 8, 10, 9),
        3..=4 => hc_group!(m, planes, 16, 6, 5),
        _ => hc_group!(m, planes, 32, 4, 3),
    }
}
