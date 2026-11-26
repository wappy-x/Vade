//! First-party fixed-order QSA kernels; decode and chunked prefill share rows.
use super::qsa_common::*;
use core::slice::{from_raw_parts as slice, from_raw_parts_mut as slice_mut};

#[no_mangle]
pub unsafe extern "ptx-kernel" fn qsa_fp32_prepare(p: Prepare) {
    let i = crate::index();
    let (past, rows) = (p.past as usize, p.rows as usize);
    let groups = (past + rows) / RATIO - past / RATIO;
    let mut x = [0.0; DIM];
    let (weight, rope, target);
    if i < rows * HEADS {
        let (row, h) = (i / HEADS, i % HEADS);
        x.copy_from_slice(slice((p.qk as *const f32).add(row * 640 + h * DIM), DIM));
        weight = p.qw;
        rope = (p.query_rope as *const Rope).add(row);
        target = (p.query as *mut f32).add(i * DIM);
    } else {
        let g = i - rows * HEADS;
        if g >= groups { return; }
        let first = (past / RATIO + g) * RATIO;
        for j in 0..RATIO {
            let pos = first + j;
            let k = if pos < past { (p.ring as *const f32).add(pos % RATIO * DIM) }
                else { (p.qk as *const f32).add((pos - past) * 640 + HEADS * DIM) };
            for d in 0..DIM { x[d] += *k.add(d); }
        }
        for v in &mut x { *v /= RATIO as f32; }
        weight = p.kw;
        rope = (p.group_rope as *const Rope).add(g);
        target = (p.cache as *mut f32).add((past / RATIO + g) * DIM);
    }
    let y = normalize_rotate(x, &*(weight as *const [f32; DIM]), &*rope, p.epsilon);
    slice_mut(target, DIM).copy_from_slice(&y);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn qsa_fp32_ring(p: Prepare) {
    let i = crate::index();
    let count = (p.rows as usize).min(RATIO);
    if i >= count * DIM { return; }
    let row = p.rows as usize - count + i / DIM;
    *(p.ring as *mut f32).add((p.past as usize + row) % RATIO * DIM + i % DIM) =
        *(p.qk as *const f32).add(row * 640 + HEADS * DIM + i % DIM);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn qsa_fp32_scores(p: Select) {
    let b = crate::index();
    let row = core::arch::nvptx::_block_idx_y() as usize;
    if b >= p.stride as usize { return; }
    let value = if b < (p.past as usize + row + 1) / RATIO {
        score(slice((p.query as *const f32).add(row * HEADS * DIM), HEADS * DIM),
            slice((p.cache as *const f32).add(b * DIM), DIM))
    } else { f32::NEG_INFINITY };
    *(p.scores as *mut f32).add(row * p.stride as usize + b) = value;
}
/// Radix threshold for nonnegative scores, followed by ordered prefix writes.
/// Each CTA owns one row. Integer counts use shared memory, never atomics.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn qsa_fp32_select(p: Select) {
    use crate::ptx::{load, store, shared, sync};
    let (t, row) = (core::arch::nvptx::_thread_idx_x() as usize, core::arch::nvptx::_block_idx_x() as usize);
    let n = (p.past as usize + row + 1) / RATIO;
    let keep = n.min(BUDGET / RATIO);
    let out = (p.selected as *mut u32).add(row * WIDTH);
    let scores = (p.scores as *const f32).add(row * p.stride as usize);
    let base = shared::<3076>();
    let get = |i: usize| load(base + (i * 4) as u32).to_bits() as usize;
    let put = |i: usize, x: usize| store(base + (i * 4) as u32, f32::from_bits(x as u32));
    for i in (t..WIDTH).step_by(256) { *out.add(i) = u32::MAX; }
    sync();
    let (start, end) = ((t * n.div_ceil(256)).min(n), ((t + 1) * n.div_ceil(256)).min(n));
    if n > keep {
        let mut threshold = 0u32;
        for bit in (0..31).rev() {
            let candidate = threshold | 1 << bit;
            let mut count = 0;
            for b in (t..n).step_by(256) { count += usize::from((*scores.add(b)).to_bits() >= candidate); }
            put(t, count); sync();
            if t == 0 { let mut sum = 0; for j in 0..256 { sum += get(j); } put(768, if sum >= keep { candidate } else { threshold } as usize); }
            sync(); threshold = get(768) as u32;
        }
        let (mut greater, mut ties) = (0, 0);
        for b in start..end { let v = (*scores.add(b)).to_bits(); greater += usize::from(v > threshold); ties += usize::from(v == threshold); }
        put(t, greater); put(256 + t, ties); sync();
        if t == 0 {
            let mut total = 0; for j in 0..256 { total += get(j); }
            let (mut need, mut offset) = (keep - total, 0);
            for j in 0..256 { let take = get(256 + j).min(need); need -= take; put(512 + j, offset); offset += get(j) + take; put(256 + j, take); }
        }
        sync();
        let (mut offset, mut ties) = (get(512 + t), get(256 + t));
        for b in start..end {
            let v = (*scores.add(b)).to_bits();
            if v > threshold || (v == threshold && ties > 0) {
                if v == threshold { ties -= 1; }
                for j in 0..RATIO { *out.add(offset * RATIO + j) = (b * RATIO + j) as u32; }
                offset += 1;
            }
        }
    } else { for b in (t..n).step_by(256) { for j in 0..RATIO { *out.add(b * RATIO + j) = (b * RATIO + j) as u32; } } }
    // Initialization must finish before tail writes (a different thread may own it).
    sync();
    if t == 0 {
        let tail = (p.past as usize + row + 1) % RATIO;
        for j in 0..tail { *out.add(keep * RATIO + j) = (n * RATIO + j) as u32; }
        *(p.counts as *mut u32).add(row) = (keep * RATIO + tail) as u32;
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn qsa_fp32_attention_scores(p: Attention) {
    let i = crate::index();
    if i >= p.rows as usize * 24 * WIDTH { return; }
    let (row, h, slot) = (i / (24 * WIDTH), i / WIDTH % 24, i % WIDTH);
    if slot >= *(p.counts as *const u32).add(row) as usize { return; }
    let pos = *(p.selected as *const u32).add(row * WIDTH + slot) as usize;
    let q = slice((p.query as *const f32).add((row * 24 + h) * 256), 256);
    let k = slice((p.keys as *const f32).add(pos * 512 + h / 12 * 256), 256);
    *(p.weights as *mut f32).add(i) = dot(q, k) * (1.0 / 16.0);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn qsa_fp32_softmax(p: Attention) {
    let head = crate::index();
    if head >= p.rows as usize * 24 { return; }
    let n = *(p.counts as *const u32).add(head / 24) as usize;
    let w = slice_mut((p.weights as *mut f32).add(head * WIDTH), n);
    let mut high = f32::NEG_INFINITY;
    for &v in w.iter() { high = high.max(v); }
    let mut den = 0.0;
    for v in w.iter_mut() { *v = exponential(*v - high); den += *v; }
    for v in w { *v /= den; }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn qsa_fp32_values(p: Attention) {
    let i = crate::index();
    if i >= p.rows as usize * 24 * 256 { return; }
    let (row, h, d) = (i / (24 * 256), i / 256 % 24, i % 256);
    let mut sum = 0.0;
    for s in 0..*(p.counts as *const u32).add(row) as usize {
        let pos = *(p.selected as *const u32).add(row * WIDTH + s) as usize;
        sum += *(p.weights as *const f32).add((row * 24 + h) * WIDTH + s) * *(p.values as *const f32).add(pos * 512 + h / 12 * 256 + d);
    }
    *(p.output as *mut f32).add(i) = sum;
}
