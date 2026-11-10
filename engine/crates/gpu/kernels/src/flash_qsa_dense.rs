// Dense causal attention kernels (flash_dense_*, flash_next_dense_attention), moved out of
// flash_qsa.rs unchanged and pasted back by include! at its end: same module, same items.

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DenseRow { keys: *mut f32, values: *mut f32, past: u32, padding: u32 }

#[inline(always)]
unsafe fn dense_row(p: &crate::kernel_params::DenseAttention, row: usize) -> DenseRow {
    if p.rows == 0 { DenseRow { keys: p.keys as *mut f32, values: p.values as *mut f32, past: p.past, padding: 0 } }
    else { *(p.rows as *const DenseRow).add(row) }
}

/// Independent scores retain the lane-strided dot and reduction tree. One
/// block per chat, key head and 32 positions: each warp loads a cached key row
/// once and scores it against every query head of the group. Only score
/// production is parallel: the consumer keeps sequential softmax/V sums.
/// Rows vary fastest across blocks, so a chat's verify rows read each key
/// tile together and share it through L2.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_dense_scores(p: crate::kernel_params::DenseAttention) {
    let block = core::arch::nvptx::_block_idx_x() as usize;
    let (row, rest) = (block % p.count as usize, block / p.count as usize);
    let (key_head, tile) = (rest % p.key_heads as usize, rest / p.key_heads as usize);
    let state = dense_row(&p, row);
    let replay = p.replay_position as *const u32;
    let length = state.past as usize + if replay.is_null() { 0 } else { *replay as usize } + 1;
    if length <= 512 { return; }
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let group = (p.query_heads / p.key_heads) as usize;
    let first_head = row * p.query_heads as usize + key_head * group;
    for step in 0..8 {
        let pos = tile * 32 + warp + step * 4;
        if pos < length {
            let key = state.keys.cast::<u16>().add((pos * p.key_heads as usize + key_head) * 256).cast::<f32>();
            let key = row_features(key, lane, 1);
            for head_row in first_head..first_head + group {
                let score = score(row_features((p.query as *const f32).add(head_row * 256), lane, 0), key, 1);
                if lane == 0 { *(p.scores as *mut f32).add(head_row * p.score_stride as usize + pos) = score; }
            }
        }
    }
}

/// Independent chats share the launch; each warp keeps its solo head arithmetic.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_dense_prepare_batch(
    qg: *const f32, k: *const f32, v: *const f32, qw: *const f32, kw: *const f32,
    q: *mut f32, rows: *const DenseRow, qheads: u32, kheads: u32, count: u32,
    frequencies: *const f32, epsilon: f32, replay_position: *const u32,
) {
    let threads = (qheads + kheads) as usize * 32;
    let thread = index();
    let row = thread / threads;
    if row >= count as usize { return; }
    let state = &*rows.add(row);
    prepare(qg.add(row * qheads as usize * 512), k.add(row * kheads as usize * 256),
        v.add(row * kheads as usize * 256), qw, kw, q.add(row * qheads as usize * 256),
        state.keys, state.values, qheads, kheads, 1, state.past, 0, frequencies,
        1, epsilon, replay_position, thread % threads);
}

/// Dense causal attention over the FP32 Flash KV cache (precision 0): every
/// visible position, sequential score/softmax/value order, sigmoid output
/// gate, no BF16 rounding and no score staging at any length. Params use the
/// single-cache DenseAttention form (rows=0, scores=0, replay_position=0);
/// launch (query_heads*count).div_ceil(4) blocks of 128, as the batch entry.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_dense_attention(p: crate::kernel_params::DenseAttention) {
    let state = dense_row(&p, 0);
    attention::<false>(p.query as *const f32, p.query_gate as *const f32, state.keys, state.values,
        core::ptr::null(), p.output as *mut f32, p.query_heads, p.key_heads, 0, 0,
        p.count, state.past, 0, p.query_heads / p.key_heads, 0, 0,
        core::ptr::null(), index(), core::ptr::null_mut());
}

#[no_mangle]
/// Four query-head warps per CTA; the short dense path caches 512 scores each.
/// Above 512 positions it only turns the saved scores into weights; the value
/// sums run in flash_dense_values.
/// # Safety
/// The batch launch uses block128 and complete per-row KV descriptors.
pub unsafe extern "ptx-kernel" fn flash_dense_attention_batch(p: crate::kernel_params::DenseAttention) {
    let (q, qg, y) = (p.query as *const f32, p.query_gate as *const f32, p.output as *mut f32);
    let (qheads, kheads, count, replay_position) = (p.query_heads, p.key_heads, p.count, p.replay_position as *const u32);
    let threads = qheads as usize * 32;
    let thread = index();
    let row = thread / threads;
    if row >= count as usize { return; }
    let state = dense_row(&p, row);
    attention::<true>(q.add(row * threads * 8), qg.add(row * threads * 16),
        state.keys, state.values, core::ptr::null(), y.add(row * threads * 8),
        qheads, kheads, 0, 0, 1, state.past, 0, qheads / kheads, 1, 0,
        replay_position, thread % threads, if p.scores == 0 { core::ptr::null_mut() }
            else { (p.scores as *mut f32).add(thread / 32 * p.score_stride as usize) });
}

/// Positions per staged value tile.
const VALUE_TILE: usize = 128;

/// Long-context value sums after flash_dense_attention_batch left the exact
/// weights rounded(probability / denominator) in the score rows. One block per
/// chat, key head and 64 output features; thread (query head, feature). Each V
/// tile is copied from memory once into shared storage (double-buffered
/// cp.async) and read there by every query head of the group. Each feature
/// still adds positions 0..length in order with the solo walk's multiply then
/// add, so neither the split of the 256 features nor the tiling changes a sum.
/// Rows vary fastest across blocks, so a chat's verify rows stream each V
/// tile together and share it through L2.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_dense_values(p: crate::kernel_params::DenseAttention) {
    use crate::ptx::{commit, copy4, copy16, load, load_u16, shared, sync, wait, wait_one};
    const STAGE: usize = VALUE_TILE * 64 * 2 + crate::kernel_params::DENSE_VALUE_GROUP as usize * VALUE_TILE * 4;
    let block = core::arch::nvptx::_block_idx_x() as usize;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let threads = core::arch::nvptx::_block_dim_x() as usize;
    let (row, rest) = (block % p.count as usize, block / p.count as usize);
    let (key_head, slice) = (rest / 4, rest % 4);
    let state = dense_row(&p, row);
    let replay = p.replay_position as *const u32;
    let length = state.past as usize + if replay.is_null() { 0 } else { *replay as usize } + 1;
    if length <= 512 { return; }
    let group = (p.query_heads / p.key_heads) as usize;
    let first_head = row * p.query_heads as usize + key_head * group;
    let (head, local) = (tid / 64, tid % 64);
    let values = state.values.cast::<u16>().add(key_head * 256 + slice * 64);
    let stride = p.key_heads as usize * 256;
    let base = shared::<{ 2 * STAGE }>();
    // Stage one tile: 64 features (128 bytes) per position, then each head's weights.
    let issue = |tile: usize| {
        let (start, buffer) = (tile * VALUE_TILE, base + (tile % 2 * STAGE) as u32);
        let count = VALUE_TILE.min(length - start);
        let mut chunk = tid;
        while chunk < count * 8 {
            let (pos, part) = (start + chunk / 8, chunk % 8);
            copy16(buffer + (chunk * 16) as u32, values.add(pos * stride + part * 8).cast());
            chunk += threads;
        }
        let weights = buffer + (VALUE_TILE * 128) as u32;
        let mut chunk = tid;
        while chunk < group * count {
            let (g, i) = (chunk / count, chunk % count);
            let source = (p.scores as *const f32).add((first_head + g) * p.score_stride as usize + start + i);
            copy4(weights + ((g * VALUE_TILE + i) * 4) as u32, source.cast());
            chunk += threads;
        }
        commit();
    };
    let tiles = length.div_ceil(VALUE_TILE);
    issue(0);
    let mut total = 0.0f32;
    for tile in 0..tiles {
        if tile + 1 < tiles { issue(tile + 1); wait_one(); } else { wait(); }
        sync();
        let buffer = base + (tile % 2 * STAGE) as u32;
        let (weights, count) = (buffer + (VALUE_TILE * 128 + head * VALUE_TILE * 4) as u32, VALUE_TILE.min(length - tile * VALUE_TILE));
        let value = |i: usize| f32::from_bits(load_u16(buffer + (i * 128 + local * 2) as u32) << 16);
        let mut i = 0;
        while i + 8 <= count {
            let (mut w, mut v) = ([0.0f32; 8], [0.0f32; 8]);
            for j in 0..8 { w[j] = load(weights + ((i + j) * 4) as u32); v[j] = value(i + j); }
            for j in 0..8 { total += w[j] * v[j]; }
            i += 8;
        }
        while i < count { total += load(weights + (i * 4) as u32) * value(i); i += 1; }
        sync();
    }
    let (head_row, feature) = (first_head + head, slice * 64 + local);
    let gate = output_gate(*(p.query_gate as *const f32).add(head_row * 512 + 256 + feature), 0);
    *(p.output as *mut f32).add(head_row * 256 + feature) = gated_bf16(total, gate);
}

// Prompt rows of one KV cache (host-validated) replay the staged path's FP32
// operations, only tiled: score()'s lane partials and shuffle-tree pairing, an
// order-free maximum, position-order denominator and value sums. Constant
// indices keep the register tiles out of local memory.
macro_rules! unroll4 { ($i:ident, $body:block) => {
    { const $i: usize = 0; $body } { const $i: usize = 1; $body } { const $i: usize = 2; $body } { const $i: usize = 3; $body }
}; }
macro_rules! unroll8 { ($i:ident, $body:block) => {
    { const $i: usize = 0; $body } { const $i: usize = 1; $body } { const $i: usize = 2; $body } { const $i: usize = 3; $body }
    { const $i: usize = 4; $body } { const $i: usize = 5; $body } { const $i: usize = 6; $body } { const $i: usize = 7; $body }
}; }
#[inline(always)]
unsafe fn store4(p: u32, v: [u32; 4]) {
    core::arch::asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
        p = in(reg32) p, a = in(reg32) v[0], b = in(reg32) v[1], c = in(reg32) v[2], d = in(reg32) v[3]);
}
/// BF16 element i of eight packed in four words (low half first).
#[inline(always)]
fn element(w: [u32; 4], i: usize) -> u32 { if i % 2 == 0 { w[i / 2] & 0xffff } else { w[i / 2] >> 16 } }
/// Leaf-major tile row: 16-byte slot l holds features l, l+32, ..., l+224,
/// the eight products lane l of score() adds. Slots are XOR-swizzled by row.
#[inline(always)]
fn leaf_slot(row: usize, leaf: usize) -> u32 { (row * 512 + ((leaf ^ (row & 7)) << 4)) as u32 }
/// Natural rows: chunk c (features 8c..8c+7) of each source holds slot c/4
/// of leaves 8(c%4)..8(c%4)+7. Transpose sources 4i+j, i = 0..7, to leaf slots.
#[inline(always)]
unsafe fn store_leaves(base: u32, row: usize, j: usize, n: [[u32; 4]; 8]) {
    unroll8!(E, {
        store4(base + leaf_slot(row, 8 * j + E), [0, 1, 2, 3].map(|m| element(n[2 * m], E) | element(n[2 * m + 1], E) << 16));
    });
}
type Tile = [[f32; 4]; 4];
#[inline(always)]
fn tile_add(x: Tile, mut y: Tile) -> Tile {
    unroll4!(A, { unroll4!(B, { y[A][B] = x[A][B] + y[A][B]; }); });
    y
}
#[inline(always)]
fn unpack(w: [u32; 4], i: usize) -> f32 { f32::from_bits(if i % 2 == 0 { w[i / 2] << 16 } else { w[i / 2] & 0xffff_0000 }) }

/// One lane partial of score() for 4 query rows x 4 keys: 0.0 plus eight
/// products in feature order lane, lane+32, ..., lane+224.
#[inline(always)]
unsafe fn leaf(q: [u32; 4], k: [u32; 4], l: usize) -> Tile {
    use crate::ptx::load4;
    let (mut qs, mut ks, mut out) = ([[0u32; 4]; 4], [[0u32; 4]; 4], [[0.0f32; 4]; 4]);
    unroll4!(A, { qs[A] = load4(q[A] + ((l ^ (q[A] as usize >> 9 & 7)) << 4) as u32); });
    unroll4!(B, { ks[B] = load4(k[B] + ((l ^ (k[B] as usize >> 9 & 7)) << 4) as u32); });
    unroll4!(A, { unroll4!(B, {
        let mut dot = 0.0;
        unroll8!(I, { dot += unpack(qs[A], I) * unpack(ks[B], I); });
        out[A][B] = dot;
    }); });
    out
}
/// score()'s reduction: sum() pairs lane l with l^16, then l^8, l^4, l^2,
/// l^1, so T(s, l) = T(2s, l) + T(2s, l+s). Pairs T(16, l) are visited in
/// bit-reversed l (a binary counter) and folded on a four-level stack.
#[inline(always)]
unsafe fn tree(q: [u32; 4], k: [u32; 4]) -> Tile {
    let (mut stack, mut root) = ([[[0.0f32; 4]; 4]; 4], [[0.0f32; 4]; 4]);
    for i in 0..16 {
        let l = (i & 1) << 3 | (i & 2) << 1 | (i & 4) >> 1 | (i & 8) >> 3;
        let mut v = tile_add(leaf(q, k, l), leaf(q, k, l + 16));
        if i & 1 == 0 { stack[0] = v; continue; }
        v = tile_add(stack[0], v);
        if i & 2 == 0 { stack[1] = v; continue; }
        v = tile_add(stack[1], v);
        if i & 4 == 0 { stack[2] = v; continue; }
        v = tile_add(stack[2], v);
        if i & 8 == 0 { stack[3] = v; } else { root = tile_add(stack[3], v); }
    }
    root
}

/// Block256, grid (ceil(count/8) * (query_heads/8), ceil(longest/1024)), dynamic
/// shared 98,304: retain Q across four independent 256-position spans. Each
/// span keeps the reference 64-key steps, lane mapping and maximum order.
/// Separate natural/transposed K buffers preserve the reference staging.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_dense_prompt_scores(p: crate::kernel_params::DenseAttention) {
    use crate::ptx::{commit, copy16, load4, sync, wait};
    const QUERY: u32 = 0; const STAGE: u32 = 32768; const KEYS: u32 = 65536;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let tiles = (p.count as usize).div_ceil(8);
    let block = core::arch::nvptx::_block_idx_x() as usize;
    let (first, head_group, begin) = (block % tiles * 8, block / tiles, core::arch::nvptx::_block_idx_y() as usize * 1024);
    let tokens = 8.min(p.count as usize - first);
    let state = dense_row(&p, first);
    let longest = state.past as usize + tokens;
    if begin >= longest { return; }
    let (heads, kheads, stride) = (p.query_heads as usize, p.key_heads as usize, p.score_stride as usize);
    let kh = head_group * 8 / (heads / kheads);
    let (r, j) = (tid % 64, tid / 64);
    if r / 8 < tokens {
        let q = (p.query as *const f32).add(((first + r / 8) * heads + head_group * 8 + r % 8) * 256);
        let mut n = [[0u32; 4]; 8];
        unroll8!(I, { let f = q.add(32 * I + 8 * j);
            unroll4!(W, { n[I][W] = (*f.add(2 * W)).to_bits() >> 16 | (*f.add(2 * W + 1)).to_bits() & 0xffff_0000; }); });
        store_leaves(QUERY, r, j, n);
    }
    let keys = state.keys.cast::<u16>().add(kh * 256);
    let issue = |start: usize| {
        unroll8!(M, {
            let (n, c) = ((tid + 256 * M) / 32, tid % 32);
            if start + n < longest { copy16(STAGE + leaf_slot(n, c), keys.add((start + n) * kheads * 256 + c * 8).cast()); }
        });
        commit();
    };
    let (tq, tk) = (tid / 16, tid % 16);
    let (mut qrows, mut krows) = ([0u32; 4], [0u32; 4]);
    unroll4!(A, { qrows[A] = QUERY + ((tq + 16 * A) * 512) as u32; krows[A] = KEYS + ((tk + 16 * A) * 512) as u32; });
    let scores = (p.scores as *mut u16).add((first * heads + head_group * 8) * stride);
    for offset in 0..4 {
        let span = begin + offset * 256;
        if span >= longest { break; }
        let mut best = [f32::NEG_INFINITY; 4];
        issue(span);
        for step in 0..4 {
            let start = span + step * 64;
            if start >= longest { break; }
            wait(); sync();
            let mut n = [[0u32; 4]; 8];
            unroll8!(I, { n[I] = load4(STAGE + leaf_slot(r, 4 * I + j)); });
            store_leaves(KEYS, r, j, n);
            sync();
            if step < 3 && start + 64 < longest { issue(start + 64); }
            let total = tree(qrows, krows);
            unroll4!(A, { unroll4!(B, {
                let (row, pos) = (tq + 16 * A, start + tk + 16 * B);
                if row / 8 < tokens && pos <= state.past as usize + row / 8 {
                    let bits = rounded(total[A][B]).to_bits();
                    best[A] = best[A].max(f32::from_bits(bits & 0xffff_0000) * (1.0 / 16.0));
                    *scores.add((row / 8 * heads + row % 8) * stride + pos) = (bits >> 16) as u16;
                }
            }); });
        }
        // Maxima are order-free: reduce the 16 key columns sharing each row.
        let maxima = (p.scores as *mut f32).add(p.count as usize * heads * stride / 2);
        unroll4!(A, {
            let mut m = best[A];
            for d in [8, 4, 2, 1] { m = m.max(f32::from_bits(shuffle_xor(u32::MAX, m.to_bits(), d, 31))); }
            let row = tq + 16 * A;
            if tk == 0 && row / 8 < tokens { *maxima.add(((first + row / 8) * heads + head_group * 8 + row % 8) * stride.div_ceil(256) + span / 256) = m; }
        });
    }
}

/// Block128 (warp = token, lane = 8 features), grid ceil(count/4) * (query_heads/8),
/// dynamic shared 20,480. Each lane < 8 takes its head's maximum over the span
/// maxima and adds exp(score - maximum) over positions 0..length in order; the
/// weights rounded(probability / denominator) then multiply staged V and add
/// in position order into 8 heads x 8 features per lane.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_dense_prompt_values(p: crate::kernel_params::DenseAttention) {
    use crate::ptx::{commit, copy16, load4, sync, wait};
    const STAGE: usize = 16 * 512 + 4 * 16 * 8 * 4;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let tiles = (p.count as usize).div_ceil(4);
    let block = core::arch::nvptx::_block_idx_x() as usize;
    let (first, head_group) = (block % tiles * 4, block / tiles);
    let tokens = 4.min(p.count as usize - first);
    let state = dense_row(&p, first);
    let longest = state.past as usize + tokens;
    if longest <= 512 { return; }
    let (heads, kheads, stride) = (p.query_heads as usize, p.key_heads as usize, p.score_stride as usize);
    let kh = head_group * 8 / (heads / kheads);
    let length = state.past as usize + warp + 1;
    let live = warp < tokens && length > 512;
    let row = (first + warp) * heads + head_group * 8;
    let scores = (p.scores as *const u16).add(row * stride);
    let score = |h: usize, pos: usize| bf(*scores.add(h * stride + pos)) * (1.0 / 16.0);
    let (mut highest, mut denominator) = (f32::NEG_INFINITY, 0.0f32);
    if live && lane < 8 {
        let maxima = (p.scores as *const f32).add(p.count as usize * heads * stride / 2 + (row + lane) * stride.div_ceil(256));
        for s in 0..length.div_ceil(256) { highest = highest.max(*maxima.add(s)); }
        let mut pos = 0;
        while pos + 8 <= length {
            let mut e = [0.0f32; 8];
            unroll8!(I, { e[I] = exp(score(lane, pos + I) - highest); });
            unroll8!(I, { denominator += e[I]; });
            pos += 8;
        }
        while pos < length { denominator += exp(score(lane, pos) - highest); pos += 1; }
    }
    // Lane m*32.. needs head lane/16 + 2m for its four weights per tile.
    let pick = |v: f32, m: usize| f32::from_bits(shuffle_xor(u32::MAX, v.to_bits(), (lane ^ (lane / 16 + 2 * m)) as u32, 31));
    let (mut his, mut dens) = ([0.0f32; 4], [0.0f32; 4]);
    unroll4!(M, { his[M] = pick(highest, M); dens[M] = pick(denominator, M); });
    let values = state.values.cast::<u16>().add(kh * 256);
    let issue = |t: usize| {
        let buffer = (t % 2 * STAGE) as u32;
        unroll4!(M, {
            let (n, c) = ((tid + 128 * M) / 32, tid % 32);
            if t * 16 + n < longest { copy16(buffer + (n * 512 + c * 16) as u32, values.add((t * 16 + n) * kheads * 256 + c * 8).cast()); }
        });
        commit();
    };
    // Weight (head lane/16 + 2m, position lane%16) of tile t, stored [position][head].
    let raw = |t: usize| -> [u16; 4] {
        let (mut out, pos) = ([0u16; 4], t * 16 + lane % 16);
        unroll4!(M, { if live && pos < length { out[M] = *scores.add((lane / 16 + 2 * M) * stride + pos); } });
        out
    };
    let weights = |t: usize, raw: [u16; 4]| {
        let base = (t % 2 * STAGE + 16 * 512 + warp * 512) as u32;
        unroll4!(M, {
            let w = rounded(exp(bf(raw[M]) * (1.0 / 16.0) - his[M]) / dens[M]);
            crate::ptx::store(base + ((lane % 16 * 8 + lane / 16 + 2 * M) * 4) as u32, w);
        });
    };
    let tiles = longest.div_ceil(16);
    issue(0);
    weights(0, raw(0));
    let mut total = [[0.0f32; 8]; 8];
    for t in 0..tiles {
        wait(); sync();
        let next = if t + 1 < tiles { issue(t + 1); raw(t + 1) } else { [0; 4] };
        if live {
            let base = (t % 2 * STAGE) as u32;
            for n in 0..16.min(length.saturating_sub(t * 16)) {
                let w = [load4(base + (16 * 512 + warp * 512 + n * 32) as u32), load4(base + (16 * 512 + warp * 512 + n * 32 + 16) as u32)];
                let v = load4(base + (n * 512 + lane * 16) as u32);
                unroll8!(H, {
                    let w = f32::from_bits(w[H / 4][H % 4]);
                    unroll8!(E, { total[H][E] += w * bf(element(v, E) as u16); });
                });
            }
        }
        if t + 1 < tiles { weights(t + 1, next); }
    }
    if !live { return; }
    unroll8!(H, { unroll8!(E, {
        let (head_row, feature) = (row + H, lane * 8 + E);
        let gate = output_gate(*(p.query_gate as *const f32).add(head_row * 512 + 256 + feature), 0);
        *(p.output as *mut f32).add(head_row * 256 + feature) = gated_bf16(total[H][E], gate);
    }); });
}

macro_rules! four { ($i:ident, $body:block) => {
    { const $i: usize = 0; $body } { const $i: usize = 1; $body }
    { const $i: usize = 2; $body } { const $i: usize = 3; $body }
}; }
/// flash_qsa_scores_t (thread-owned index scores) for verify rows, one
/// warp per CTA: 4 queries x 32 groups (lane = group, all four queries per
/// thread), the same per-score tree (bit-reversed lane order, level stack),
/// same bytes. Block 32, grid ceil(tokens / 4) * ceil(blocks / 32); static shared 18,432 B.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_scores_t4(mut p: crate::kernel_params::QsaIndex, cursor: *const u32) {
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    use crate::ptx::{load4, shared, sync};
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    const T4: usize = 4;
    const G4: usize = 32;
    let (q, pooled, scores) = (p.query as *const f32, p.pooled as *const f32, p.scores as *mut f32);
    let (tokens, past, blocks) = (p.tokens as usize, p.past as usize, p.blocks as usize);
    let lane = _thread_idx_x() as usize;
    let tiles = blocks.div_ceil(G4);
    let tile = _block_idx_x() as usize;
    if tile >= tokens.div_ceil(T4) * tiles { return; }
    let (token, first) = (tile / tiles * T4, tile % tiles * G4);
    let end = (token + T4).min(tokens);
    if (past + end) / 4 <= 512 || first >= (past + end) / 4 { return; }
    let keys = shared::<{ G4 * 32 * 16 + T4 * 32 * 16 }>();
    let queries = keys + (G4 * 32 * 16) as u32;
    let mut g = 0;
    while g < G4 {
        let mut v = [0.0f32; 4];
        if first + g < blocks { let key = pooled.add((first + g) * 4 * 128); four!(F, { v[F] = *key.add(lane + F * 32); }); }
        store4f(keys + ((g * 32 + (lane ^ (g & 7))) * 16) as u32, v);
        g += 1;
    }
    let mut score = [0.0f32; 4];
    for head in 0..4 {
        sync();
        four!(T, {
            let mut v = [0.0f32; 4];
            if token + T < tokens { let source = q.add(((token + T) * 4 + head) * 128); four!(F, { v[F] = *source.add(lane + F * 32); }); }
            store4f(queries + ((T * 32 + lane) * 16) as u32, v);
        });
        sync();
        let mut stack = [[0.0f32; 5]; 4];
        x32!(K, {
            const L: usize = bitrev5(K);
            const LEVEL: u32 = (!K).trailing_zeros();
            let mut qv = [[0.0f32; 4]; 4];
            four!(T, { qv[T] = load4(queries + ((T * 32 + L) * 16) as u32).map(f32::from_bits); });
            let kv = load4(keys + ((lane * 32 + (L ^ (lane & 7))) * 16) as u32).map(f32::from_bits);
            four!(T, {
                let mut dot = 0.0;
                four!(F, { dot += qv[T][F] * kv[F]; });
                let mut c = dot;
                if K & 1 != 0 { c = stack[T][0] + c;
                if K & 2 != 0 { c = stack[T][1] + c;
                if K & 4 != 0 { c = stack[T][2] + c;
                if K & 8 != 0 { c = stack[T][3] + c;
                if K & 16 != 0 { c = stack[T][4] + c; } } } } }
                if LEVEL < 5 { stack[T][LEVEL as usize] = c; } else { score[T] += c.max(0.0); }
            });
        });
    }
    four!(T, {
        let position = token + T;
        let visible = (past + position + 1) / 4;
        if position < end && visible > 512 {
            let g = first + lane;
            if g < visible { *scores.add(position * blocks + g) = score[T] / crate::layer::sqrt(128.0); }
        }
    });
}
