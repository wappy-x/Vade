//! Layer kernels compiled with the projection kernels for the same GPU target.
//! Buffers and dimensions are checked by the Rust host backend before launch.
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _block_idx_y, _block_idx_z, _thread_idx_x};
#[path = "batch.rs"]
pub(crate) mod batch;
#[path = "multi.rs"]
mod multi;

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.shfl.sync.bfly.i32"]
    fn shuffle(mask: u32, value: u32, distance: u32, clamp: u32) -> u32;
    #[link_name = "llvm.sqrt.f32"]
    pub(crate) fn sqrt(value: f32) -> f32;
    #[link_name = "llvm.nvvm.ex2.approx.ftz.f"]
    fn exp2(value: f32) -> f32;
}

#[inline(always)]
unsafe fn index() -> usize {
    _block_idx_x() as usize * _block_dim_x() as usize + _thread_idx_x() as usize
}
#[inline(always)]
pub(crate) unsafe fn other(value: f32, distance: u32) -> f32 {
    f32::from_bits(shuffle(u32::MAX, value.to_bits(), distance, 31))
}
#[inline(always)]
pub(crate) unsafe fn sum(mut value: f32) -> f32 {
    let mut distance = 16;
    while distance != 0 { value += other(value, distance); distance /= 2; }
    value
}
#[inline(always)]
pub(crate) unsafe fn maximum(mut value: f32) -> f32 {
    let mut distance = 16;
    while distance != 0 { value = value.max(other(value, distance)); distance /= 2; }
    value
}
#[inline(always)]
// NVVM lowers this to ex2.approx.ftz: deterministic GPU arithmetic, not host exp.
pub(crate) unsafe fn exp(value: f32) -> f32 { exp2(value * core::f32::consts::LOG2_E) }

#[no_mangle]
pub unsafe extern "ptx-kernel" fn embedding_bf16(
    table: *const u16, output: *mut f32, token: u32, width: u32, packed: u32,
) {
    embedding(table, output, token, width, packed);
}
#[inline(always)]
unsafe fn embedding(table: *const u16, output: *mut f32, token: u32, width: u32, packed: u32) {
    let i = index();
    if i < width as usize {
        let row = token as usize;
        let offset = crate::canonical::embedding_element(row, i, width as usize, packed);
        *output.add(i) = f32::from_bits((*table.add(offset) as u32) << 16);
    }
}

/// Read a preceding argmax result on the same stream, without a host round trip.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn embedding_selected_bf16(
    table: *const u16, output: *mut f32, selected: *const f32, width: u32, rows: u32, packed: u32,
) {
    // An invalid reduction is reported after the chain finishes; never index it.
    let token = if *selected.add(2) == 1.0 && (*selected).is_finite() {
        crate::kernel_params::id_index(*selected.add(1), rows)
    } else { None };
    if let Some(token) = token {
        embedding(table, output, token as u32, width, packed);
    } else {
        let i = index();
        if i < width as usize { *output.add(i) = 0.0; }
    }
}

/// Warp-sized partials keep hundreds of serial loads out of normalization.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn layer_rms(
    input: *const f32, partials: *mut f32, width: u32,
) {
    rms_partials(input, partials, width);
}
#[inline(always)]
unsafe fn rms_partials(input: *const f32, partials: *mut f32, width: u32) {
    let row = _block_idx_y() as usize;
    let input = input.add(row * width as usize);
    let partials = partials.add(row * ((width as usize + 127) / 128));
    let thread = index();
    let lane = thread % 32;
    let chunk = thread / 32;
    let begin = chunk * 128;
    if begin >= width as usize { return; }
    let end = (begin + 128).min(width as usize);
    let mut squares = 0.0;
    let mut i = begin + lane;
    while i < end { let x = *input.add(i); squares += x * x; i += 32; }
    squares = sum(squares);
    if lane == 0 { *partials.add(chunk) = squares; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn residual_rms(
    state: *mut f32, residual: *const f32, partials: *mut f32, width: u32,
) {
    let row = _block_idx_y() as usize;
    let state = state.add(row * width as usize);
    let residual = residual.add(row * width as usize);
    let partials = partials.add(row * ((width as usize + 127) / 128));
    let thread = index();
    let lane = thread % 32;
    let chunk = thread / 32;
    let begin = chunk * 128;
    if begin >= width as usize { return; }
    let end = (begin + 128).min(width as usize);
    let mut squares = 0.0;
    let mut i = begin + lane;
    while i < end {
        let x = *state.add(i) + *residual.add(i);
        *state.add(i) = x;
        squares += x * x;
        i += 32;
    }
    squares = sum(squares);
    if lane == 0 { *partials.add(chunk) = squares; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn layer_rms_apply(
    input: *const f32, weight: *const f32, partials: *const f32,
    output: *mut f32, width: u32, epsilon: f32,
) {
    rms_apply(input, weight, partials, output, width, epsilon, false);
}
#[inline(always)]
unsafe fn rms_apply(input: *const f32, weight: *const f32, partials: *const f32,
    output: *mut f32, width: u32, epsilon: f32, rounded: bool) {
    let row = _block_idx_y() as usize;
    let input = input.add(row * width as usize);
    let output = output.add(row * width as usize);
    let partials = partials.add(row * ((width as usize + 127) / 128));
    let i = index();
    if i / 32 * 32 >= width as usize { return; }
    let mut j = i % 32;
    let chunks = (width as usize + 127) / 128;
    let mut squares = 0.0;
    while j < chunks { squares += *partials.add(j); j += 32; }
    let scale = 1.0 / sqrt(sum(squares) / width as f32 + epsilon.abs());
    if i < width as usize {
        let value = if epsilon < 0.0 { round_bf16(round_bf16(*input.add(i) * scale) * *weight.add(i)) }
            else { *input.add(i) * scale * if weight.is_null() {1.0} else {*weight.add(i)} };
        *output.add(i) = if rounded { round_bf16(value) } else { value };
    }
}

#[inline(always)]
unsafe fn rms_fused<const RESIDUAL: bool, const PAIR: bool>(
    input: *mut f32, residual: *const f32, peer: *const f32, weight: *const f32,
    output: *mut f32, width: u32, epsilon: f32,
) {
    let row = _block_idx_y() as usize * width as usize;
    let input = input.add(row);
    let output = output.add(row);
    let residual = if RESIDUAL { residual.add(row) } else { residual };
    let peer = if PAIR { peer.add(row) } else { peer };
    let tid = _thread_idx_x() as usize;
    let lane = tid % 32;
    let chunk = tid / 32;
    let begin = chunk * 128;
    let end = (begin + 128).min(width as usize);
    let shared = crate::ptx::shared::<128>();
    let mut squares = 0.0;
    let mut i = begin + lane;
    while i < end {
        let mut x = *input.add(i);
        if RESIDUAL {
            let residual=if PAIR{*residual.add(i)+*peer.add(i)}else{*residual.add(i)};
            x += residual;
            *input.add(i) = x;
        }
        squares += x * x;
        i += 32;
    }
    squares = sum(squares);
    if lane == 0 { crate::ptx::store(shared + (chunk * 4) as u32, squares); }
    crate::ptx::sync();
    // Match layer_rms_apply, including its initial addition to positive zero.
    squares = 0.0;
    let chunks = (width as usize + 127) / 128;
    let mut j = lane;
    while j < chunks {
        squares += crate::ptx::load(shared + (j * 4) as u32);
        j += 32;
    }
    let scale = 1.0 / sqrt(sum(squares) / width as f32 + epsilon.abs());
    i = begin + lane;
    while i < end {
        *output.add(i) = if epsilon < 0.0 {
            round_bf16(round_bf16(*input.add(i) * scale) * *weight.add(i))
        } else { *input.add(i) * scale * *weight.add(i) };
        i += 32;
    }
}

/// Exact two-stage RMS arithmetic in one CTA. One warp per 128 elements;
/// 0 < width <= 4096, FP32 input/weight/output of width elements, disjoint output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn layer_rms_fused(
    input: *const f32, weight: *const f32, output: *mut f32,
    width: u32, epsilon: f32,
) {
    rms_fused::<false,false>(input.cast_mut(), core::ptr::null(), core::ptr::null(), weight, output, width, epsilon);
}

/// Add residual to state and normalize with the exact two-stage RMS arithmetic.
/// One warp per 128 elements; 0 < width <= 4096, buffers FP32[width] and disjoint.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn residual_rms_fused(
    state: *mut f32, residual: *const f32, weight: *const f32,
    output: *mut f32, width: u32, epsilon: f32,
) {
    rms_fused::<true,false>(state, residual, core::ptr::null(), weight, output, width, epsilon);
}


/// Q/K head normalization, half-split RoPE, and append of this position's KV.
/// RoPE sin/cos are computed once on the host using the reference arithmetic.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn qk_rope_cache(
    query: *mut f32, key: *const f32, value: *const f32,
    q_weight: *const f32, k_weight: *const f32, rope: *const f32,
    keys: *mut f32, values: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, position: u32, epsilon: f32,
) {
    let thread = index();
    let head = thread / 32;
    let lane = thread % 32;
    if head >= (heads + kv_heads) as usize { return; }
    let is_key = head >= heads as usize;
    let local_head = if is_key { head - heads as usize } else { head };
    let input = if is_key { key } else { query as *const f32 };
    let weight = if is_key { k_weight } else { q_weight };
    let start = local_head * dim as usize;
    let output = if is_key { keys.add(position as usize * kv_heads as usize * dim as usize) } else { query };
    let half = dim as usize / 2;
    let mut squares = 0.0;
    let mut i = lane;
    while i < dim as usize { let x = *input.add(start + i); squares += x * x; i += 32; }
    let scale = if epsilon < 0.0 { 1.0 } else { 1.0 / sqrt(sum(squares) / dim as f32 + epsilon) };
    i = lane;
    while i < half {
        let a = *input.add(start + i) * scale * *weight.add(i);
        let b = *input.add(start + i + half) * scale * *weight.add(i + half);
        let sin = *rope.add(position as usize * dim as usize + 2 * i);
        let cos = *rope.add(position as usize * dim as usize + 2 * i + 1);
        let (a, b) = rope_pair(a, b, sin, cos, epsilon < 0.0);
        *output.add(start + i) = a;
        *output.add(start + i + half) = b;
        i += 32;
    }
    if is_key {
        i = lane;
        while i < dim as usize {
            let x = *value.add(start + i);
            *values.add(position as usize * kv_heads as usize * dim as usize + start + i) = if epsilon < 0.0 { round_bf16(x) } else { x };
            i += 32;
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_scores(
    query: *const f32, keys: *const f32, scores: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, positions: u32, capacity: u32,
) {
    let thread = index();
    let t = thread / 32;
    let lane = thread % 32;
    let head = _block_idx_y() as u32;
    if head >= heads || t >= positions as usize { return; }
    let kh = head / (heads / kv_heads);
    let key = keys.add((t * kv_heads as usize + kh as usize) * dim as usize);
    let q = query.add(head as usize * dim as usize);
    let mut dot = 0.0;
    let mut i = lane;
    while i < dim as usize { dot += *q.add(i) * *key.add(i); i += 32; }
    let score = sum(dot) / sqrt(dim as f32);
    if lane == 0 { *scores.add(head as usize * capacity as usize + t) = score; }
}

/// Qwen3-8B's four query heads share a KV head. Read each key element once.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_scores_gqa4(
    query: *const f32, keys: *const f32, scores: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, positions: u32, capacity: u32,
) {
    let thread = index();
    let t = thread / 32;
    let lane = thread % 32;
    let kh = _block_idx_y() as usize;
    if heads != kv_heads * 4 { core::arch::nvptx::trap(); }
    if kh >= kv_heads as usize || t >= positions as usize { return; }
    let d = dim as usize;
    let key = keys.add((t * kv_heads as usize + kh) * d);
    let q = query.add(kh * 4 * d);
    let (mut a, mut b, mut c, mut e) = (0.0, 0.0, 0.0, 0.0);
    let mut i = lane;
    while i < d {
        let k = *key.add(i);
        a += *q.add(i) * k;
        b += *q.add(d + i) * k;
        c += *q.add(2 * d + i) * k;
        e += *q.add(3 * d + i) * k;
        i += 32;
    }
    let denominator = sqrt(dim as f32);
    let (a, b, c, e) = (sum(a) / denominator, sum(b) / denominator, sum(c) / denominator, sum(e) / denominator);
    if lane == 0 {
        let stride = capacity as usize;
        let out = scores.add(kh * 4 * stride + t);
        *out = a;
        *out.add(stride) = b;
        *out.add(2 * stride) = c;
        *out.add(3 * stride) = e;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_softmax(
    scores: *mut f32, heads: u32, positions: u32, capacity: u32,
) {
    let thread = index();
    let head = thread / 32;
    let lane = thread % 32;
    if head >= heads as usize { return; }
    let scores = scores.add(head * capacity as usize);
    let mut peak = f32::NEG_INFINITY;
    let mut t = lane;
    while t < positions as usize { peak = peak.max(*scores.add(t)); t += 32; }
    peak = maximum(peak);
    let mut denominator = 0.0;
    t = lane;
    while t < positions as usize {
        let p = exp(*scores.add(t) - peak);
        *scores.add(t) = p;
        denominator += p;
        t += 32;
    }
    denominator = sum(denominator);
    t = lane;
    while t < positions as usize { *scores.add(t) /= denominator; t += 32; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_values(
    scores: *const f32, values: *const f32, output: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, positions: u32, capacity: u32,
) {
    let feature = index();
    let head = _block_idx_y() as u32;
    if head >= heads || feature >= dim as usize { return; }
    let kh = head / (heads / kv_heads);
    *output.add(head as usize * dim as usize + feature) = weighted_values(scores.add(head as usize * capacity as usize),
        values.add(kh as usize * dim as usize + feature), kv_heads as usize * dim as usize, 0, positions as usize);
}

#[inline(always)]
unsafe fn weighted_values(scores: *const f32, values: *const f32, stride: usize, begin: usize, end: usize) -> f32 {
    let mut value = 0.0;
    let mut t = begin;
    // Issue independent reads together; preserve the reference accumulation
    // order while avoiding a dependent memory round trip for every position.
    while t + 8 <= end {
        let p = [*scores.add(t), *scores.add(t + 1), *scores.add(t + 2), *scores.add(t + 3),
            *scores.add(t + 4), *scores.add(t + 5), *scores.add(t + 6), *scores.add(t + 7)];
        let v = [*values.add(t * stride), *values.add((t + 1) * stride), *values.add((t + 2) * stride), *values.add((t + 3) * stride),
            *values.add((t + 4) * stride), *values.add((t + 5) * stride), *values.add((t + 6) * stride), *values.add((t + 7) * stride)];
        let mut j = 0;
        while j < 8 { value += p[j] * v[j]; j += 1; }
        t += 8;
    }
    while t < end {
        value += *scores.add(t) * *values.add(t * stride);
        t += 1;
    }
    value
}

/// At longer contexts, independent 64-position chunks expose enough concurrent
/// loads to avoid a long latency chain in each output feature.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_values_split(
    scores: *const f32, values: *const f32, partials: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, positions: u32, capacity: u32,
) {
    let feature = index();
    let head = _block_idx_y() as u32;
    let chunk = _block_idx_z() as usize;
    let begin = chunk * 64;
    if head >= heads || feature >= dim as usize || begin >= positions as usize { return; }
    let kh = head / (heads / kv_heads);
    *partials.add((chunk * heads as usize + head as usize) * dim as usize + feature) = weighted_values(scores.add(head as usize * capacity as usize),
        values.add(kh as usize * dim as usize + feature),
        kv_heads as usize * dim as usize, begin, (begin + 64).min(positions as usize));
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_values_gqa4(
    scores: *const f32, values: *const f32, partials: *mut f32,
    heads: u32, kv_heads: u32, dim: u32, positions: u32, capacity: u32,
) {
    let feature = index();
    let kh = _block_idx_y() as usize;
    let chunk = _block_idx_z() as usize;
    let begin = chunk * 64;
    if heads != kv_heads * 4 { core::arch::nvptx::trap(); }
    if kh >= kv_heads as usize || feature >= dim as usize || begin >= positions as usize { return; }
    let end = (begin + 64).min(positions as usize);
    let stride = kv_heads as usize * dim as usize;
    let values = values.add(kh * dim as usize + feature);
    let s0 = scores.add(kh * 4 * capacity as usize);
    let s1 = s0.add(capacity as usize);
    let s2 = s1.add(capacity as usize);
    let s3 = s2.add(capacity as usize);
    let (mut a, mut b, mut c, mut e) = (0.0, 0.0, 0.0, 0.0);
    let mut t = begin;
    while t + 8 <= end {
        let v = [*values.add(t * stride), *values.add((t + 1) * stride), *values.add((t + 2) * stride), *values.add((t + 3) * stride),
            *values.add((t + 4) * stride), *values.add((t + 5) * stride), *values.add((t + 6) * stride), *values.add((t + 7) * stride)];
        let p0 = [*s0.add(t), *s0.add(t + 1), *s0.add(t + 2), *s0.add(t + 3), *s0.add(t + 4), *s0.add(t + 5), *s0.add(t + 6), *s0.add(t + 7)];
        let p1 = [*s1.add(t), *s1.add(t + 1), *s1.add(t + 2), *s1.add(t + 3), *s1.add(t + 4), *s1.add(t + 5), *s1.add(t + 6), *s1.add(t + 7)];
        let p2 = [*s2.add(t), *s2.add(t + 1), *s2.add(t + 2), *s2.add(t + 3), *s2.add(t + 4), *s2.add(t + 5), *s2.add(t + 6), *s2.add(t + 7)];
        let p3 = [*s3.add(t), *s3.add(t + 1), *s3.add(t + 2), *s3.add(t + 3), *s3.add(t + 4), *s3.add(t + 5), *s3.add(t + 6), *s3.add(t + 7)];
        let mut j = 0;
        while j < 8 { a += p0[j] * v[j]; b += p1[j] * v[j]; c += p2[j] * v[j]; e += p3[j] * v[j]; j += 1; }
        t += 8;
    }
    while t < end {
        let v = *values.add(t * stride);
        a += *s0.add(t) * v; b += *s1.add(t) * v; c += *s2.add(t) * v; e += *s3.add(t) * v;
        t += 1;
    }
    let d = dim as usize;
    let out = partials.add((chunk * heads as usize + kh * 4) * d + feature);
    *out = a; *out.add(d) = b; *out.add(2 * d) = c; *out.add(3 * d) = e;
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_values_merge(partials: *const f32, output: *mut f32, width: u32, chunks: u32) {
    let i = index();
    if i >= width as usize { return; }
    let mut value = 0.0;
    let mut chunk = 0;
    while chunk < chunks as usize { value += *partials.add(chunk * width as usize + i); chunk += 1; }
    *output.add(i) = value;
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn layer_swiglu(a: crate::kernel_params::SwiGlu) {
    let (gate, up) = (a.gate as *mut f32, a.up as *const f32);
    let i = index();
    if i < a.width as usize {
        let g = *gate.add(i);
        *gate.add(i) = if a.bf16 != 0 {
            let g = round_bf16(g);
            round_bf16(round_bf16(g / (1.0 + exp(-g))) * round_bf16(*up.add(i)))
        } else { g / (1.0 + exp(-g)) * *up.add(i) };
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn layer_residual(state: *mut f32, residual: *const f32, width: u32) {
    let i = index();
    if i < width as usize { *state.add(i) += *residual.add(i); }
}


#[inline(always)]
unsafe fn reduce_best(mut best: f32, mut id: u32, mut valid: u32) -> (f32, u32, u32) {
    let mut distance = 16;
    while distance != 0 {
        let value = other(best, distance);
        let index = shuffle(u32::MAX, id, distance, 31);
        valid &= shuffle(u32::MAX, valid, distance, 31);
        if value > best || (value == best && index < id) { best = value; id = index; }
        distance /= 2;
    }
    (best, id, valid)
}

/// The BF16 output head returns BF16 logits, promoted for the sampling API.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn logits_bf16(logits: *mut f32, width: u32) {
    let i = index();
    if i < width as usize {
        let bits = (*logits.add(i)).to_bits();
        if bits & 0x7f800000 != 0x7f800000 {
            *logits.add(i) = f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn logits_argmax(logits: *const f32, partials: *mut f32, width: u32) {
    let row = _block_idx_y() as usize;
    let logits = logits.add(row * width as usize);
    let partials = partials.add(row * ((width as usize + 1023)/1024) * 3);
    let thread = index();
    let warp = thread / 32;
    let lane = thread % 32;
    let begin = warp * 1024;
    if begin >= width as usize { return; }
    let end = (begin + 1024).min(width as usize);
    let (mut best, mut id, mut valid) = (f32::NEG_INFINITY, u32::MAX, 1u32);
    let mut i = begin + lane;
    while i < end {
        let value = *logits.add(i);
        valid &= value.is_finite() as u32;
        if value > best { best = value; id = i as u32; }
        i += 32;
    }
    let (best, id, valid) = reduce_best(best, id, valid);
    if lane == 0 {
        *partials.add(warp * 3) = best;
        *partials.add(warp * 3 + 1) = id as f32;
        *partials.add(warp * 3 + 2) = valid as f32;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn logits_argmax_finish(partials: *const f32, result: *mut f32, count: u32) {
    let row = _block_idx_y() as usize;
    let partials = partials.add(row * count as usize * 3);
    let result = result.add(row * 3);
    let lane = index();
    if lane >= 32 { return; }
    let (mut best, mut id, mut valid) = (f32::NEG_INFINITY, u32::MAX, 1u32);
    let mut i = lane;
    while i < count as usize {
        let value = *partials.add(i * 3);
        let index = *partials.add(i * 3 + 1) as u32;
        valid &= *partials.add(i * 3 + 2) as u32;
        if value > best || (value == best && index < id) { best = value; id = index; }
        i += 32;
    }
    let (best, id, valid) = reduce_best(best, id, valid);
    if lane == 0 { *result = best; *result.add(1) = id as f32; *result.add(2) = valid as f32; }
}

/// Each warp selects one disjoint vocabulary tile. Ordering matches the host:
/// descending finite logit, then ascending ID, including ties at signed zero.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn logits_top_k(p: crate::kernel_params::LogitsTopK) {
    let row = _block_idx_y() as usize;
    let (width, k) = (p.width as usize, p.k as usize);
    let thread = index();
    let (tile, lane) = (thread / 32, thread % 32);
    let begin = tile * 1024;
    if begin >= width { return; }
    let logits = (p.logits as *const f32).add(row * width);
    let out = (p.partials as *mut f32).add((row * width.div_ceil(1024) + tile) * k * 3);
    let (mut previous, mut previous_id) = (f32::INFINITY, u32::MAX);
    for rank in 0..k {
        let (mut best, mut id, mut valid) = (f32::NEG_INFINITY, u32::MAX, 1);
        let mut i = begin + lane;
        while i < (begin + 1024).min(width) {
            let value = *logits.add(i);
            valid &= value.is_finite() as u32;
            if (rank == 0 || value < previous || value == previous && i as u32 > previous_id)
                && (value > best || value == best && (i as u32) < id) {
                best = value; id = i as u32;
            }
            i += 32;
        }
        let (best, id, valid) = reduce_best(best, id, valid);
        if lane == 0 {
            *out.add(rank*3) = best;
            *out.add(rank*3+1) = id as f32;
            *out.add(rank*3+2) = valid as f32;
        }
        previous = best; previous_id = id;
    }
}

/// Merge the tile shortlists. All rows run independently on the GPU.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn logits_top_k_finish(p: crate::kernel_params::LogitsTopK) {
    let row = _block_idx_y() as usize;
    let lane = index();
    if lane >= 32 { return; }
    let k = p.k as usize;
    let count = (p.width as usize).div_ceil(1024) * k;
    let input = (p.partials as *const f32).add(row * count * 3);
    let out = (p.result as *mut f32).add(row * k * 3);
    let (mut previous, mut previous_id) = (f32::INFINITY, u32::MAX);
    for rank in 0..k {
        let (mut best, mut id, mut valid) = (f32::NEG_INFINITY, u32::MAX, 1);
        let mut i = lane;
        while i < count {
            let value = *input.add(i*3);
            let token = *input.add(i*3+1) as u32;
            valid &= *input.add(i*3+2) as u32;
            if (rank == 0 || value < previous || value == previous && token > previous_id)
                && (value > best || value == best && token < id) {
                best = value; id = token;
            }
            i += 32;
        }
        let (best, id, valid) = reduce_best(best, id, valid);
        if lane == 0 {
            *out.add(rank*3) = best;
            *out.add(rank*3+1) = id as f32;
            *out.add(rank*3+2) = valid as f32;
        }
        previous = best; previous_id = id;
    }
}

#[inline(always)]
pub(crate) fn rope_pair(a: f32, b: f32, sin: f32, cos: f32, no_norm: bool) -> (f32, f32) {
    if no_norm {
        let (a, b, sin, cos) = (round_bf16(a), round_bf16(b), round_bf16(sin), round_bf16(cos));
        (round_bf16(round_bf16(a*cos)-round_bf16(b*sin)),
            round_bf16(round_bf16(b*cos)+round_bf16(a*sin)))
    } else { (a*cos-b*sin, b*cos+a*sin) }
}

/// Shared model-dtype boundary helpers; FP32 reductions, BF16 outputs.
#[inline(always)]
pub(crate) fn round_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    if bits & 0x7f800000 == 0x7f800000 { return x; }
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}
#[inline(always)]
unsafe fn tanh(x: f32) -> f32 {
    let result: f32;
    core::arch::asm!("tanh.approx.f32 {y}, {x};", y=out(reg32)result, x=in(reg32)x);
    result
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn model_rms_partials(a: crate::kernel_params::ModelRms) {
    rms_partials(a.input as *const f32, a.partials as *mut f32, a.width);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn model_rms_apply(a: crate::kernel_params::ModelRms) {
    rms_apply(a.input as *const f32, a.weight as *const f32, a.partials as *const f32,
        a.output as *mut f32, a.width, a.epsilon, true);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn model_elementwise(a: crate::kernel_params::ModelElementwise) {
    let i = index();
    if i >= a.elements as usize { return; }
    let x = *(a.input as *const f32).add(i);
    let y = match a.operation {
        1 => round_bf16(x + *(a.peer as *const f32).add(i)),
        2 => {
            let z = 0.7978845608028654 * (x + 0.044715 * x * x * x);
            let gate = round_bf16(0.5 * x * (1.0 + tanh(z)));
            round_bf16(gate * *(a.peer as *const f32).add(i))
        },
        3 => round_bf16(round_bf16(tanh(round_bf16(x / a.scalar))) * a.scalar),
        _ => round_bf16(x * a.scalar),
    };
    *(a.output as *mut f32).add(i) = y;
}
