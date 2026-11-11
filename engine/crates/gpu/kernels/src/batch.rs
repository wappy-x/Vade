//! Causal batch kernels with FP32, BF16, or explicitly selected scaled I16 KV.
use super::*;

#[no_mangle]
pub unsafe extern "ptx-kernel" fn copy_strided(
    source: *const f32, destination: *mut f32,
    source_stride: u32, destination_stride: u32, width: u32, rows: u32,
) {
    let i = index();
    if i < width as usize * rows as usize {
        let (row, col) = (i / width as usize, i % width as usize);
        *destination.add(row * destination_stride as usize + col) = *source.add(row * source_stride as usize + col);
    }
}

/// Copy selected normalized rows for one vocabulary projection.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_gather_rows(p: crate::kernel_params::GatherRows) {
    let i = index();
    if i < p.rows as usize * p.width as usize {
        let row = i / p.width as usize;
        let source = *(p.indices as *const f32).add(row) as usize;
        *(p.output as *mut f32).add(i) = *(p.input as *const f32).add(source * p.width as usize + i % p.width as usize);
    }
}

#[inline(always)]
pub(crate) unsafe fn cache_load(p: *const f32, i: usize, format: u32) -> f32 {
    match format {
        1 => f32::from_bits((*p.cast::<u16>().add(i) as u32) << 16),
        4 => {
            let block = p.add(i / 1024 * 520);
            *block.cast::<i16>().add(i % 1024) as f32 * *block.add(512 + i % 1024 / 128)
        }
        _ => *p.add(i),
    }
}
#[inline(always)]
unsafe fn stage_cache4(dst: u32, src: *const f32, at: usize, format: u32) {
    if format == 0 {
        crate::ptx::copy16(dst, src.add(at).cast());
    } else if format == 1 {
        let packed = *src.cast::<u64>().add(at / 4);
        core::arch::asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
            p=in(reg32)dst, a=in(reg32)(packed as u32)<<16,
            b=in(reg32)((packed>>16) as u32)<<16,
            c=in(reg32)((packed>>32) as u32)<<16,
            d=in(reg32)((packed>>48) as u32)<<16);
    } else {
        for d in 0..4 { crate::ptx::store(dst+(d*4) as u32, cache_load(src,at+d,format)); }
    }
}
#[inline(always)]
unsafe fn save(p: *mut f32, i: usize, x: f32, format: u32) {
    if format == 1 {
        let bits = x.to_bits();
        *p.cast::<u16>().add(i) = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
    } else { *p.add(i) = x; }
}

#[inline(always)]
unsafe fn round_i16(x: f32) -> i16 {
    let rounded: i32;
    core::arch::asm!("cvt.rni.s32.f32 {out}, {value};", out = out(reg32) rounded,
        value = in(reg32) x.max(-32767.0).min(32767.0));
    rounded as i16
}

/// Identical per-head quantization for single-session and ragged Multi writes.
#[inline(always)]
pub(crate) unsafe fn save_i16_head(input: *const f32, value: *const f32, weight: *const f32,
    rope: *const f32, keys: *mut f32, values: *mut f32, start: usize, scale: f32,
    position: usize, kh: usize, key_head: usize, lane: usize) {
    let (mut k, mut v) = ([0.0f32; 4], [0.0f32; 4]);
    for j in 0..2 {
        let i = lane + j * 32;
        let a = *input.add(start + i) * scale * *weight.add(i);
        let b = *input.add(start + i + 64) * scale * *weight.add(i + 64);
        let sin = *rope.add(position * 128 + 2 * i);
        let cos = *rope.add(position * 128 + 2 * i + 1);
        k[j] = a * cos - b * sin; k[j + 2] = b * cos + a * sin;
    }
    let (mut km, mut vm, mut ki, mut vi) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for j in 0..4 {
        v[j] = *value.add(start + lane + j * 32);
        km = km.max(k[j].abs()); vm = vm.max(v[j].abs());
        if !k[j].is_finite() { ki = 1.0; }
        if !v[j].is_finite() { vi = 1.0; }
    }
    km = maximum(km); vm = maximum(vm);
    let ks = if sum(ki) != 0.0 { f32::NAN } else if km == 0.0 { 1.0 } else { (km / 32767.0).max(f32::MIN_POSITIVE) };
    let vs = if sum(vi) != 0.0 { f32::NAN } else if vm == 0.0 { 1.0 } else { (vm / 32767.0).max(f32::MIN_POSITIVE) };
    // Eight heads share a scale trailer, keeping every payload warp sector-aligned.
    let head = position * kh + key_head;
    let block = head / 8 * 520;
    let within = head % 8;
    let (kp, vp) = (keys.add(block + within * 64), values.add(block + within * 64));
    for j in 0..4 {
        *kp.cast::<i16>().add(lane + j * 32) = round_i16(k[j] / ks);
        *vp.cast::<i16>().add(lane + j * 32) = round_i16(v[j] / vs);
    }
    if lane == 0 { *keys.add(block + 512 + within) = ks; *values.add(block + 512 + within) = vs; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_embedding(args: crate::kernel_params::BatchEmbedding) {
    let i = index();
    let row = _block_idx_y() as usize;
    if i < args.width as usize {
        let out = (args.output as *mut f32).add(row * args.width as usize + i);
        let Some(token) = crate::kernel_params::id_index(*(args.ids as *const f32).add(row), args.rows) else {
            *out = 0.0;
            return;
        };
        let offset = crate::canonical::embedding_element(token, i, args.width as usize, args.packed);
        *out = f32::from_bits((*(args.table as *const u16).add(offset) as u32) << 16);
    }
}

/// IDs initially contain row indices into argmax triples; each thread replaces
/// its own index with the selected token, on the same stream as the next depth.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_selected_ids(args: crate::kernel_params::SelectedIds) {
    let row = index();
    if row < args.rows as usize {
        let id = (args.ids as *mut f32).add(row);
        let Some(source) = crate::kernel_params::id_index(*id, args.result_rows) else {
            *id = -1.0;
            return;
        };
        let result = (args.result as *const f32).add(source * 3);
        *id = if *result.add(2) == 1.0 && (*result).is_finite() { *result.add(1) } else { -1.0 };
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_qk_rope(
    query: *mut f32, key: *const f32, value: *const f32,
    q_weight: *const f32, k_weight: *const f32, rope: *const f32, keys: *mut f32, values: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, epsilon: f32, bf16: u32, first_head: u32, last_head: u32, cache_base: u32,
) { qk_rope(query, key, value, q_weight, k_weight, rope, keys, values, heads, kv_heads, dim, position, epsilon, bf16, first_head, last_head, cache_base, core::ptr::null(), 0, false, core::ptr::null()); }
#[no_mangle]
pub unsafe extern "ptx-kernel" fn graph_batch_qk_rope(
    query: *mut f32, key: *const f32, value: *const f32,
    q_weight: *const f32, k_weight: *const f32, rope: *const f32, keys: *mut f32, values: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, epsilon: f32, bf16: u32, first_head: u32, last_head: u32, cache_base: u32, replay_position: *const u32,
) { qk_rope(query, key, value, q_weight, k_weight, rope, keys, values, heads, kv_heads, dim, position, epsilon, bf16, first_head, last_head, cache_base, replay_position, 0, false, core::ptr::null()); }
#[inline(always)]
unsafe fn qk_rope(
    query: *mut f32, key: *const f32, value: *const f32,
    q_weight: *const f32, k_weight: *const f32, rope: *const f32,
    keys: *mut f32, values: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, epsilon: f32, bf16: u32, first_head: u32, last_head: u32, cache_base: u32,
    replay_position: *const u32, ring: usize, model: bool, metadata: *const crate::kernel_params::BoundedCache,
) {
    let row = _block_idx_y() as usize;
    let (keys, values, position, ring) = if metadata.is_null() {
        (keys, values, position as usize + row + if replay_position.is_null() { 0 } else { *replay_position as usize }, ring)
    } else {
        let cache = &*metadata.add(row);
        (cache.keys as *mut f32, cache.values as *mut f32, cache.position as usize, cache.capacity as usize)
    };
    let cache_position = if ring == 0 {position - cache_base as usize} else {position % ring};
    let (h, kh, d) = (heads as usize, kv_heads as usize, dim as usize);
    let query = query.add(row * h * d);
    let key = key.add(row * kh * d);
    let value = value.add(row * kh * d);
    let head = index() / 32 + first_head as usize;
    let lane = index() % 32;
    if head >= last_head as usize { return; }
    let is_key = head >= h;
    let start = (if is_key { head - h } else { head }) * d;
    let input = if is_key { key } else { query as *const f32 };
    let weight = if is_key { k_weight } else { q_weight };
    let mut squares = 0.0;
    let mut i = lane;
    while i < d { let x = *input.add(start + i); squares += x * x; i += 32; }
    let scale = if epsilon < 0.0 { 1.0 } else { 1.0 / sqrt(sum(squares) / dim as f32 + epsilon) };
    if bf16 == 4 && is_key {
        save_i16_head(input, value, weight, rope, keys, values, start, scale, position, kh, head - h, lane);
        return;
    }
    let mut v_squares = 0.0;
    if model && is_key {
        i = lane;
        while i < d { let x = *value.add(start + i); v_squares += x * x; i += 32; }
    }
    let v_scale = if model && is_key {1.0 / sqrt(sum(v_squares) / dim as f32 + epsilon)} else {1.0};
    i = lane;
    while i < d / 2 {
        let a = *input.add(start + i) * scale * *weight.add(i);
        let b = *input.add(start + i + d / 2) * scale * *weight.add(i + d / 2);
        let sin = *rope.add(position * d + 2 * i);
        let cos = *rope.add(position * d + 2 * i + 1);
        let (a, b) = if model {
            let (a, b) = (round_bf16(a), round_bf16(b));
            (round_bf16(round_bf16(a * cos) - round_bf16(b * sin)),
             round_bf16(round_bf16(b * cos) + round_bf16(a * sin)))
        } else { crate::layer::rope_pair(a, b, sin, cos, epsilon < 0.0) };
        if is_key {
            save(keys, cache_position * kh * d + start + i, a, bf16);
            save(keys, cache_position * kh * d + start + i + d / 2, b, bf16);
        } else { *query.add(start + i) = a; *query.add(start + i + d / 2) = b; }
        i += 32;
    }
    if is_key {
        i = lane;
        while i < d { save(values, cache_position * kh * d + start + i, if model {round_bf16(*value.add(start + i) * v_scale)} else if epsilon < 0.0 {crate::layer::round_bf16(*value.add(start + i))} else {*value.add(start + i)}, bf16); i += 32; }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn model_qkv(a: crate::kernel_params::ModelQkv) {
    qk_rope(a.query as *mut f32, a.key as *const f32, a.value as *const f32,
        a.q_weight as *const f32, a.k_weight as *const f32, a.rope as *const f32,
        a.keys as *mut f32, a.values as *mut f32, a.heads, a.kv_heads, a.dim,
        a.position, a.epsilon, 1, 0, a.heads + a.kv_heads, 0, core::ptr::null(),
        a.cache_capacity as usize, true, a.metadata as *const crate::kernel_params::BoundedCache);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_scores(
    query: *const f32, keys: *const f32, scores: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, bf16: u32,
) {
    bounded_scores(query, keys, scores, heads, kv_heads, dim, position, capacity,
        bf16, 0, 0, 1.0 / sqrt(dim as f32), false);
}
#[inline(always)]
pub(crate) unsafe fn bounded_scores(query: *const f32, keys: *const f32, scores: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, format: u32,
    window: u32, ring: u32, scale: f32, rounded: bool) {
    let row = _block_idx_z() as usize;
    let slot = index() / 32;
    let (begin, end) = attention_bounds(position as usize + row, window as usize);
    let t = begin + slot;
    let lane = index() % 32;
    let head = _block_idx_y() as usize;
    if t >= end { return; }
    let (h, kh, d) = (heads as usize, kv_heads as usize, dim as usize);
    let token = if ring == 0 {t} else {t % ring as usize};
    let key = (token * kh + head / (h / kh)) * d;
    let q = query.add((row * h + head) * d);
    let mut dot = 0.0;
    let mut i = lane;
    while i < d { dot += *q.add(i) * cache_load(keys, key + i, format); i += 32; }
    let result = if rounded {round_bf16(sum(dot) * scale)} else {sum(dot) / sqrt(dim as f32)};
    if lane == 0 { *scores.add((row * h + head) * capacity as usize + slot) = result; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_softmax(scores: *mut f32, heads: u32, position: u32, capacity: u32) {
    softmax(scores, heads, position, capacity, core::ptr::null(), 0, false);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn graph_batch_softmax(scores: *mut f32, heads: u32, position: u32, capacity: u32, replay_position: *const u32) {
    softmax(scores, heads, position, capacity, replay_position, 0, false);
}
#[inline(always)]
pub(crate) unsafe fn softmax(scores: *mut f32, heads: u32, position: u32, capacity: u32, replay_position: *const u32, window: u32, rounded: bool) {
    let position = position + if replay_position.is_null() { 0 } else { *replay_position };
    let row = _block_idx_y() as usize;
    let head = index() / 32;
    let lane = index() % 32;
    if head >= heads as usize { return; }
    let (begin, limit) = attention_bounds(position as usize + row, window as usize);
    let end = limit - begin;
    let scores = scores.add((row * heads as usize + head) * capacity as usize);
    let mut peak = f32::NEG_INFINITY;
    let mut t = lane;
    while t < end { peak = peak.max(*scores.add(t)); t += 32; }
    peak = maximum(peak);
    let mut denominator = 0.0;
    t = lane;
    while t < end { let p = exp(*scores.add(t) - peak); *scores.add(t) = p; denominator += p; t += 32; }
    denominator = sum(denominator);
    t = lane;
    while t < end { *scores.add(t) = if rounded {round_bf16(*scores.add(t) / denominator)} else {*scores.add(t) / denominator}; t += 32; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_values(
    scores: *const f32, values: *const f32, output: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, bf16: u32,
) {
    bounded_values(scores, values, output, heads, kv_heads, dim, position, capacity, bf16, 0, 0, false, false, 0);
}
#[inline(always)]
pub(crate) unsafe fn bounded_values(scores: *const f32, values: *const f32, output: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, format: u32,
    window: u32, ring: u32, rounded: bool, split: bool, chunks: usize) {
    let row = if split {_block_idx_x() as usize/heads as usize} else {_block_idx_z() as usize};
    let head = if split {_block_idx_x() as usize%heads as usize} else {_block_idx_y() as usize};
    let feature = if split {_thread_idx_x() as usize} else {index()};

    let (h, kh, d) = (heads as usize, kv_heads as usize, dim as usize);
    if feature >= d { return; }
    let scores = scores.add((row * h + head) * capacity as usize);
    let start = head / (h / kh) * d + feature;
    let mut result = 0.0;
    let mut t = if split {_block_idx_y() as usize*64} else {0};
    let (begin, limit) = attention_bounds(position as usize + row, window as usize);
    let end = if split {(limit-begin).min(t+64)} else {limit-begin};
    let mut token = cache_token(begin+t, ring);
    let ring_end = if ring == 0 {usize::MAX} else {ring as usize};
    // Match scalar decode and ragged batches: reduce each 64-key partial
    // before merging, independently of the number of verification rows.
    while t < end {
        let chunk_end = (t + 64).min(end);
        let mut partial = 0.0;
        while t < chunk_end {
            if t+4 <= chunk_end && token+4 <= ring_end {
                let a = cache_load(values, token * kh * d + start, format);
                let b = cache_load(values, (token+1) * kh * d + start, format);
                let c = cache_load(values, (token+2) * kh * d + start, format);
                let e = cache_load(values, (token+3) * kh * d + start, format);
                partial += *scores.add(t) * a; partial += *scores.add(t + 1) * b;
                partial += *scores.add(t + 2) * c; partial += *scores.add(t + 3) * e;
                t += 4; token += 4;
            } else {
                partial += *scores.add(t) * cache_load(values, token*kh*d+start, format);
                t += 1; token += 1;
            }
            if token == ring_end {token=0;}
        }
        result += partial;
    }
    let at = if split {((row*h+head)*chunks+_block_idx_y() as usize)*d+feature} else {(row*h+head)*d+feature};
    *output.add(at) = if rounded {round_bf16(result)} else {result};
}

/// PV normalizes while staging probabilities, avoiding a global normalization pass.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_softmax_staged(
    scores: *mut f32, denominators: *mut f32, heads: u32, position: u32, capacity: u32,
) {
    let row = _block_idx_y() as usize;
    let head = index() / 32;
    let lane = index() % 32;
    if head >= heads as usize { return; }
    let end = position as usize + row + 1;
    let scores = scores.add((row * heads as usize + head) * capacity as usize);
    let mut peak = f32::NEG_INFINITY;
    let mut t = lane;
    while t < end { peak = peak.max(*scores.add(t)); t += 32; }
    peak = maximum(peak);
    let mut denominator = 0.0;
    t = lane;
    while t < end { let p = exp(*scores.add(t) - peak); *scores.add(t) = p; denominator += p; t += 32; }
    denominator = sum(denominator);
    if lane == 0 { *denominators.add(row * heads as usize + head) = denominator; }
}

/// Eight query warps share each key tile. Two half-warps evaluate two keys,
/// folding the canonical distance-16 reduction before shuffling 8,4,2,1.
#[inline(always)]
unsafe fn scores_gqa8_tile(
    query: *const f32, keys: *const f32, scores: *mut f32,
    heads: u32, kv_heads: u32, position: u32, capacity: u32, format: u32, rows: u32,
) {
    let tid = _thread_idx_x() as usize;
    let row_base = _block_idx_z() as usize * 8;
    let row = row_base + tid / 32;
    let lane = tid % 32;
    let kh = _block_idx_y() as usize;
    let head = kh * 8;
    let begin = _block_idx_x() as usize * 64;
    let end = (begin + 64).min(position as usize + row + 1);
    let all_end = (begin + 64).min(position as usize + (row_base + 8).min(rows as usize));
    if begin >= all_end { return; }
    let shared = crate::ptx::shared::<49152>();
    let mut i = tid * 4;
    while i < 64 * 128 {
        let t = begin + i / 128;
        // Alternate keys occupy opposite half-warp shared-memory banks.
        let staged = i / 128 * 128 + (i % 128 + i / 128 % 2 * 16) % 128;
        let dst = shared + (staged * 4) as u32;
        let at = (t * kv_heads as usize + kh) * 128 + i % 128;
        if t < all_end {
            stage_cache4(dst, keys, at, format);
        } else {
            for d in 0..4 { crate::ptx::store(dst + (d * 4) as u32, 0.0); }
        }
        i += 256 * 4;
    }
    crate::ptx::wait();
    crate::ptx::sync();
    let q = query.add((row.min(rows as usize - 1) * heads as usize + head) * 128 + lane % 16);
    let mut query = [[0.0f32; 8]; 8];
    for g in 0..8 { for d in 0..4 {
        query[g][d] = *q.add(g * 128 + d * 32);
        query[g][d + 4] = *q.add(g * 128 + 16 + d * 32);
    } }
    let raw = shared + 32768 + (tid / 32 * 8 * 64 * 4) as u32;
    let mut t = lane / 16;
    while t < 64 {
        let mut dot = [[0.0f32; 2]; 8];
        for d in 0..4 {
            let rotation = t % 2 * 16;
            let a = crate::ptx::load(shared + ((t * 128 + (lane % 16 + d * 32 + rotation) % 128) * 4) as u32);
            let b = crate::ptx::load(shared + ((t * 128 + (lane % 16 + 16 + d * 32 + rotation) % 128) * 4) as u32);
            for g in 0..8 {
                dot[g][0] += query[g][d] * a;
                dot[g][1] += query[g][d + 4] * b;
            }
        }
        for g in 0..8 {
            let mut score = dot[g][0] + dot[g][1];
            let mut distance = 8;
            while distance != 0 {
                score += f32::from_bits(shuffle(u32::MAX, score.to_bits(), distance, 31));
                distance /= 2;
            }
            if lane % 16 == 0 { crate::ptx::store(raw + ((g * 64 + t) * 4) as u32, score); }
        }
        t += 2;
    }
    crate::ptx::warp_sync();
    if row < rows as usize {
        for g in 0..8 {
            let mut t = lane;
            while t < 64 {
                if begin + t < end {
                    let score = crate::ptx::load(raw + ((g * 64 + t) * 4) as u32) / sqrt(128.0);
                    *scores.add((row * heads as usize + head + g) * capacity as usize + begin + t) = score;
                }
                t += 32;
            }
        }
    }
}

/// Four or eight query heads share each key without changing the reduction tree.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_scores_gqa4(
    query: *const f32, keys: *const f32, scores: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, bf16: u32, rows: u32,
) { scores_gqa4(query, keys, scores, heads, kv_heads, dim, position, capacity, bf16, rows, core::ptr::null()); }
#[no_mangle]
pub unsafe extern "ptx-kernel" fn graph_batch_scores_gqa4(
    query: *const f32, keys: *const f32, scores: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, bf16: u32, rows: u32, replay_position: *const u32,
) { scores_gqa4(query, keys, scores, heads, kv_heads, dim, position, capacity, bf16, rows, replay_position); }
#[inline(always)]
unsafe fn scores_gqa4(
    query: *const f32, keys: *const f32, scores: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, bf16: u32, rows: u32, replay_position: *const u32,
) {
    let position = position + if replay_position.is_null() { 0 } else { *replay_position };
    let group = heads / kv_heads;
    if group != 4 && group != 5 && group != 8 { core::arch::nvptx::trap(); }
    if group == 8 && dim == 128 {
        scores_gqa8_tile(query, keys, scores, heads, kv_heads, position, capacity, bf16, rows);
        return;
    }
    let row = _block_idx_z() as usize;
    let t = index() / 32;
    let lane = index() % 32;
    let kh = _block_idx_y() as usize;
    if t > position as usize + row { return; }
    let (h, d) = (heads as usize, dim as usize);
    let key = (t * kv_heads as usize + kh) * d;
    let q = query.add((row * h + kh * group as usize) * d);
    let (mut a, mut b, mut c, mut e) = (0.0, 0.0, 0.0, 0.0);
    let (mut f, mut g, mut j, mut l) = (0.0, 0.0, 0.0, 0.0);
    let mut i = lane;
    while i < d {
        let k = cache_load(keys, key + i, bf16);
        a += *q.add(i) * k; b += *q.add(d + i) * k;
        c += *q.add(2 * d + i) * k; e += *q.add(3 * d + i) * k;
        if group >= 5 { f += *q.add(4 * d + i) * k; }
        if group == 8 {
            g += *q.add(5 * d + i) * k;
            j += *q.add(6 * d + i) * k; l += *q.add(7 * d + i) * k;
        }
        i += 32;
    }
    let denominator = sqrt(dim as f32);
    let (a, b, c, e) = (sum(a)/denominator, sum(b)/denominator, sum(c)/denominator, sum(e)/denominator);
    if lane == 0 {
        let stride = capacity as usize;
        let out = scores.add((row * h + kh * group as usize) * stride + t);
        *out = a; *out.add(stride) = b; *out.add(stride * 2) = c; *out.add(stride * 3) = e;
    }
    if group == 5 {
        let f = sum(f) / denominator;
        if lane == 0 { *scores.add((row * h + kh * 5 + 4) * capacity as usize + t) = f; }
    } else if group == 8 {
        let (f, g, j, l) = (sum(f)/denominator, sum(g)/denominator, sum(j)/denominator, sum(l)/denominator);
        if lane == 0 {
            let stride = capacity as usize;
            let out = scores.add((row * h + kh * 8 + 4) * stride + t);
            *out = f; *out.add(stride) = g; *out.add(stride * 2) = j; *out.add(stride * 3) = l;
        }
    }
}

/// Four query rows share each value tile across all eight query heads.
/// Keep canonical 64-key partials in registers, avoiding global partial storage.
/// Grid(ceil(rows/4),KV,1), block512, static shared40960; FP32/BF16 KV, dim128.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_values_gqa8_full(
    scores: *const f32, values: *const f32, output: *mut f32,
    heads: u32, kv_heads: u32, position: u32, capacity: u32, rows: u32,
    denominators: *const f32, format: u32,
) {
    let tid = _thread_idx_x() as usize;
    let feature = tid % 32 * 4;
    let row_base = _block_idx_x() as usize * 4;
    let r = tid / 128;
    let row = row_base + r;
    let group = tid / 32 % 4 * 2;
    let kh = _block_idx_y() as usize;
    let sh = crate::ptx::shared::<40960>();
    let end = position as usize + (row_base + 4).min(rows as usize);
    // Each warp owns two heads and four adjacent features per lane: eight
    // output accumulators instead of all 32 row/head combinations per lane.
    let mut sums = [[0.0f32; 4]; 2];
    let mut begin = 0;
    while begin < end {
        let mut i = tid * 4;
        while i < 64 * 128 {
            let key = begin + i / 128;
            if key < end {
                stage_cache4(sh + (i * 4) as u32, values,
                    (key * kv_heads as usize + kh) * 128 + i % 128, format);
            } else {
                for d in 0..4 { crate::ptx::store(sh + ((i + d) * 4) as u32, 0.0); }
            }
            i += 512 * 4;
        }
        i = tid;
        while i < 2048 {
            let (r, g, key) = (i / 512, i / 64 % 8, i % 64);
            let value = if row_base + r < rows as usize && begin + key <= position as usize + row_base + r {
                let head = (row_base + r) * heads as usize + kh * 8 + g;
                *scores.add(head * capacity as usize + begin + key) / *denominators.add(head)
            } else { 0.0 };
            crate::ptx::store(sh + 32768 + (((r * 64 + key) * 8 + g) * 4) as u32, value);
            i += 512;
        }
        crate::ptx::wait();
        crate::ptx::sync();
        let count = if row < rows as usize { (position as usize + row + 1).saturating_sub(begin).min(64) } else { 0 };
        let mut partial = [[0.0f32; 4]; 2];
        let mut key = 0;
        while key < count {
            let raw = crate::ptx::load4(sh + ((key * 128 + feature) * 4) as u32);
            let probabilities = crate::ptx::load2(sh + 32768 + (((r * 64 + key) * 8 + group) * 4) as u32);
            let a = f32::from_bits(probabilities[0]);
            let b = f32::from_bits(probabilities[1]);
            for d in 0..4 {
                let value = f32::from_bits(raw[d]);
                partial[0][d] += a * value;
                partial[1][d] += b * value;
            }
            key += 1;
        }
        for g in 0..2 { for d in 0..4 { sums[g][d] += partial[g][d]; } }
        crate::ptx::sync();
        begin += 64;
    }
    if row < rows as usize {
        for g in 0..2 { for d in 0..4 {
            *output.add((row * heads as usize + kh * 8 + group + g) * 128 + feature + d) = sums[g][d];
        } }
    }
}

/// Independent 64-key reductions bound the per-thread load dependency chain.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_values_split_gqa4(
    scores: *const f32, values: *const f32, partials: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, bf16: u32, chunks: u32,
) { values_split_gqa4(scores, values, partials, heads, kv_heads, dim, position, capacity, bf16, chunks, core::ptr::null()); }
#[no_mangle]
pub unsafe extern "ptx-kernel" fn graph_batch_values_split_gqa4(
    scores: *const f32, values: *const f32, partials: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, bf16: u32, chunks: u32, replay_position: *const u32,
) { values_split_gqa4(scores, values, partials, heads, kv_heads, dim, position, capacity, bf16, chunks, replay_position); }
#[inline(always)]
unsafe fn values_split_gqa4(
    scores: *const f32, values: *const f32, partials: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, capacity: u32, bf16: u32, chunks: u32, replay_position: *const u32,
) {
    let position = position + if replay_position.is_null() { 0 } else { *replay_position };
    let group = heads / kv_heads;
    if group != 4 && group != 5 && group != 8 { core::arch::nvptx::trap(); }
    let row = _block_idx_z() as usize / chunks as usize;
    let chunk = _block_idx_z() as usize % chunks as usize;
    let kh = _block_idx_y() as usize;
    let feature = index();
    let (h, d, stride) = (heads as usize, dim as usize, kv_heads as usize * dim as usize);
    if feature >= d { return; }
    let s = scores.add((row * h + kh * group as usize) * capacity as usize);
    let mut t = chunk * 64;
    let end = (t + 64).min(position as usize + row + 1);
    let (mut a, mut b, mut c, mut e) = (0.0, 0.0, 0.0, 0.0);
    let (mut f, mut g, mut j, mut l) = (0.0, 0.0, 0.0, 0.0);
    while t < end {
        let v = cache_load(values, t * stride + kh * d + feature, bf16);
        a += *s.add(t) * v; b += *s.add(capacity as usize + t) * v;
        c += *s.add(2 * capacity as usize + t) * v; e += *s.add(3 * capacity as usize + t) * v;
        if group >= 5 { f += *s.add(4 * capacity as usize + t) * v; }
        if group == 8 {
            g += *s.add(5 * capacity as usize + t) * v;
            j += *s.add(6 * capacity as usize + t) * v; l += *s.add(7 * capacity as usize + t) * v;
        }
        t += 1;
    }
    let out = partials.add(((row * chunks as usize + chunk) * h + kh * group as usize) * d + feature);
    *out = a; *out.add(d) = b; *out.add(2 * d) = c; *out.add(3 * d) = e;
    if group >= 5 { *out.add(4 * d) = f; }
    if group == 8 { *out.add(5 * d) = g; *out.add(6 * d) = j; *out.add(7 * d) = l; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_values_merge(partials: *const f32, output: *mut f32, width: u32, chunks: u32) {
    let i = index();
    let row = _block_idx_y() as usize;
    if i >= width as usize { return; }
    let mut value = 0.0;
    let mut chunk = 0;
    while chunk < chunks as usize { value += *partials.add((row * chunks as usize + chunk) * width as usize + i); chunk += 1; }
    *output.add(row * width as usize + i) = value;
}


#[inline(always)]
pub(crate) fn attention_bounds(position: usize, window: usize) -> (usize, usize) {
    let end = position + 1;
    (if window == 0 {0} else {end.saturating_sub(window)}, end)
}
#[inline(always)]
fn cache_token(position: usize, ring: u32) -> usize {
    if ring == 0 {position} else {position % ring as usize}
}
