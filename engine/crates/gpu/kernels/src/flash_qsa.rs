//! Head-sharded indexed attention. All activations and caches remain FP32.
//! The indexer consumes all 640 gathered projection coordinates before norms.
use crate::{index, layer::{exp, sqrt, sum}, shuffle_xor};
use crate::{flash_gdn::rounded, math::bf16_to_f32 as bf, ptx::ballot};

// FP32 1 / (10_000_000 ** (i/32)), matching the official config computation.
const INVERSE_FREQUENCIES: [u32; 32] = [
    0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
    0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
    0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
    0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44,
];

#[inline(always)]
unsafe fn sigmoid(value: f32) -> f32 { 1.0 / (1.0 + exp(-value)) }

#[inline(always)]
unsafe fn rotate(a: f32, b: f32, coordinate: usize, position: usize, frequencies: *const f32, precision: u32) -> (f32, f32) {
    let (sine, cosine) = sincos(coordinate, position, frequencies);
    turn(a, b, sine, cosine, precision)
}
// theta=10^7, first 64 coordinates, rotate_half pairs i with i+32. Split so prepare runs the angle off its load chain.
#[inline(always)]
unsafe fn sincos(coordinate: usize, position: usize, frequencies: *const f32) -> (f32, f32) {
    let frequency = if frequencies.is_null() { f32::from_bits(INVERSE_FREQUENCIES[coordinate]) } else { *frequencies.add(coordinate) };
    let angle = position as f32 * frequency;
    let turns = (angle as f64 / core::f64::consts::TAU) as i64;
    let angle = (angle as f64 - turns as f64 * core::f64::consts::TAU) as f32;
    let (sine, cosine): (f32, f32);
    core::arch::asm!("sin.approx.ftz.f32 {s}, {x}; cos.approx.ftz.f32 {c}, {x};", s=out(reg32)sine, c=out(reg32)cosine, x=in(reg32)angle);
    (sine, cosine)
}
#[inline(always)]
unsafe fn turn(a: f32, b: f32, sine: f32, cosine: f32, precision: u32) -> (f32, f32) {
    if precision == 0 { (a * cosine - b * sine, b * cosine + a * sine) }
    else {
        let (s, c) = (rounded(sine), rounded(cosine));
        (rounded(rounded(a * c) - rounded(b * s)), rounded(rounded(b * c) + rounded(a * s)))
    }
}

#[inline(always)]
unsafe fn cached(p: *const f32, i: usize, precision: u32) -> f32 {
    if precision == 0 { *p.add(i) } else { bf(*p.cast::<u16>().add(i)) }
}
#[inline(always)]
unsafe fn store(p: *mut f32, i: usize, x: f32, precision: u32) {
    if precision == 0 { *p.add(i) = x; } else { *p.cast::<u16>().add(i) = (rounded(x).to_bits() >> 16) as u16; }
}

/// Warp per token/head. Q/gate uses complete interleaved 512-row heads.
/// Q is [tokens,qheads,256]; cached K/V are [capacity,kheads,256].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_prepare(
    qg: *const f32, k: *const f32, v: *const f32,
    qw: *const f32, kw: *const f32,
    q: *mut f32, keys: *mut f32, values: *mut f32,
    qheads: u32, kheads: u32, tokens: u32, past: u32, position_offset: u32,
    frequencies: *const f32, precision: u32, epsilon: f32,
    replay_position: *const u32,
) {
    prepare(qg, k, v, qw, kw, q, keys, values, qheads, kheads, tokens, past,
        position_offset, frequencies, precision, epsilon, replay_position, index());
}
#[inline(always)]
pub(crate) unsafe fn prepare(
    qg: *const f32, k: *const f32, v: *const f32,
    qw: *const f32, kw: *const f32, q: *mut f32, keys: *mut f32, values: *mut f32,
    qheads: u32, kheads: u32, tokens: u32, past: u32, position_offset: u32,
    frequencies: *const f32, precision: u32, epsilon: f32,
    replay_position: *const u32, thread: usize,
) {
    let past = if replay_position.is_null() { past } else { past + *replay_position };
    let row = thread / 32;
    let lane = thread % 32;
    let heads = (qheads + kheads) as usize;
    if row >= tokens as usize * heads { return; }
    let token = row / heads;
    let head = row % heads;
    let is_key = head >= qheads as usize;
    let h = if is_key { head - qheads as usize } else { head };
    let source = if is_key { k.add((token * kheads as usize + h) * 256) }
        else { qg.add((token * qheads as usize + h) * 512) };
    let weight = if is_key { kw } else { qw };
    let offset = ((past as usize + token) * kheads as usize + h) * 256;
    let target = if is_key { if precision == 0 { keys.add(offset) } else { keys.cast::<u16>().add(offset).cast::<f32>() } }
        else { q.add((token * qheads as usize + h) * 256) };
    // row loads (x, weight, V) once, up front, in registers; same per-feature arithmetic and order.
    let (mut x, mut w, mut y) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
    let (vsource, mut part) = (v.add((token * kheads as usize + h) * 256), 0);
    while part < 8 { let i = lane + part * 32; x[part] = *source.add(i); w[part] = cached(weight, i, precision); if is_key { y[part] = *vsource.add(i); } part += 1; }
    let (sine, cosine) = sincos(lane, position_offset as usize + past as usize + token, frequencies);
    let mut squares = 0.0; part = 0;
    while part < 8 { squares += x[part] * x[part]; part += 1; }
    let scale = 1.0 / sqrt(sum(squares) / 256.0 + epsilon);
    let normalize = |i: usize| { let x = (x[i] * scale) * (1.0 + w[i]); if precision == 0 { x } else { rounded(x) } };
    let (a, b) = turn(normalize(0), normalize(1), sine, cosine, precision);
    let qp = if is_key { precision } else { 0 };
    store(target, lane, a, qp); store(target, lane + 32, b, qp);
    part = 2;
    while part < 8 { store(target, lane + part * 32, normalize(part), qp); part += 1; }
    if is_key {
        let target = if precision == 0 { values.add(offset) } else { values.cast::<u16>().add(offset).cast::<f32>() };
        part = 0; while part < 8 { store(target, lane + part * 32, y[part], precision); part += 1; }
    }
}

/// Gathered input is [tokens,640]: four complete Q heads then one raw K.
/// Append unnormalized, unrotated K and normalize/rotate Q independently.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_index_prepare(
    x: *const f32, qw: *const f32, q: *mut f32, raw_keys: *mut f32,
    tokens: u32, past: u32, position_offset: u32, cursor: *const u32,
) {
    let past = if cursor.is_null() { past } else { *cursor };
    index_prepare(x, qw, q, raw_keys, tokens, past, position_offset, index());
}
#[inline(always)]
pub(crate) unsafe fn index_prepare(
    x: *const f32, qw: *const f32, q: *mut f32, raw_keys: *mut f32,
    tokens: u32, past: u32, position_offset: u32, thread: usize,
) {
    let row = thread / 32;
    let lane = thread % 32;
    if row >= tokens as usize * 5 { return; }
    let token = row / 5;
    let head = row % 5;
    let source = x.add(token * 640 + head * 128);
    if head == 4 {
        let target = raw_keys.add((past as usize + token) * 128);
        let mut feature = lane;
        while feature < 128 { *target.add(feature) = *source.add(feature); feature += 32; }
        return;
    }
    let mut a = [0.0f32; 4];
    let mut squares = 0.0;
    let mut part = 0;
    while part < 4 {
        a[part] = *source.add(lane + part * 32);
        squares += a[part] * a[part];
        part += 1;
    }
    let scale = 1.0 / sqrt(sum(squares) / 128.0 + 1e-6);
    part = 0;
    while part < 4 {
        a[part] = (a[part] * scale) * (1.0 + *qw.add(lane + part * 32));
        part += 1;
    }
    let (a0, a1) = rotate(a[0], a[1], lane, position_offset as usize + past as usize + token, core::ptr::null(), 0);
    a[0] = a0;
    a[1] = a1;
    let target = q.add((token * 4 + head) * 128);
    part = 0;
    while part < 4 { *target.add(lane + part * 32) = a[part]; part += 1; }
}

/// Complete four-token groups share a pooled key across causal query rows.
/// Average raw FP32 keys first, then RMSNorm and RoPE at the first position.
/// Output replaces each group's fourth raw row after all four rows are read.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_index_pool(mut p: crate::kernel_params::QsaIndexPool, cursor: *const u32) {
    if !cursor.is_null() {
        let first = *cursor / 4;
        p.raw_keys += first as u64 * 2048; p.pooled += first as u64 * 2048;
        p.position_offset = first * 4;
    }
    index_pool(p);
}
#[inline(always)]
unsafe fn index_pool(p: crate::kernel_params::QsaIndexPool) {
    let (raw_keys, kw, pooled) = (p.raw_keys as *const f32, p.weight as *const f32, p.pooled as *mut f32);
    let (blocks, position_offset) = (p.blocks, p.position_offset);
    let thread = index();
    let block = thread / 32;
    let lane = thread % 32;
    if block >= blocks as usize { return; }
    let source = raw_keys.add(block * 4 * 128);
    let mut a = [0.0f32; 4];
    let mut squares = 0.0;
    let mut part = 0;
    while part < 4 {
        let feature = lane + part * 32;
        let mut total = 0.0;
        let mut token = 0;
        while token < 4 { total += *source.add(token * 128 + feature); token += 1; }
        a[part] = total * 0.25;
        squares += a[part] * a[part];
        part += 1;
    }
    let scale = 1.0 / sqrt(sum(squares) / 128.0 + 1e-6);
    part = 0;
    while part < 4 {
        a[part] = (a[part] * scale) * (1.0 + *kw.add(lane + part * 32));
        part += 1;
    }
    let (a0, a1) = rotate(a[0], a[1], lane, position_offset as usize + block * 4, core::ptr::null(), 0);
    a[0] = a0;
    a[1] = a1;
    let target = pooled.add(block * 4 * 128);
    part = 0;
    while part < 4 { *target.add(lane + part * 32) = a[part]; part += 1; }
}

/// Warp per neighboring-query tile/group. Scores are [tokens,blocks], with
/// only complete groups visible to each causal query eligible for selection.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_scores(mut p: crate::kernel_params::QsaIndex, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    index_scores(p);
}
#[inline(always)]
unsafe fn index_scores(p: crate::kernel_params::QsaIndex) {
    if p.tokens as usize >= crate::kernel_params::QSA_INDEX_QUERIES { index_scores_tile(p); }
    else { index_scores_single(p); }
}
// Constant subscripts retain the independent scores in registers.
macro_rules! index4 { ($i:ident, $body:block) => {
    { const $i: usize = 0; $body } { const $i: usize = 1; $body }
    { const $i: usize = 2; $body } { const $i: usize = 3; $body }
}; }
/// Four neighboring queries x four keys per warp. Queries and keys are only
/// loaded once per head/tile; each score retains its four feature additions,
/// the same lane shuffle tree, then heads 0,1,2,3 and the final division.
#[inline(always)]
unsafe fn index_scores_tile(p: crate::kernel_params::QsaIndex) {
    const Q: usize = crate::kernel_params::QSA_INDEX_QUERIES;
    const K: usize = crate::kernel_params::QSA_INDEX_GROUPS;
    let (q, pooled, scores) = (p.query as *const f32, p.pooled as *const f32, p.scores as *mut f32);
    let (tokens, past, blocks) = (p.tokens as usize, p.past as usize, p.blocks as usize);
    let (tile, lane) = (index() / 32, index() % 32);
    let groups = blocks.div_ceil(K);
    if tile >= tokens.div_ceil(Q) * groups { return; }
    let (token, first) = (tile / groups * Q, tile % groups * K);
    let end = (token + Q).min(tokens);
    let visible = (past + end) / 4;
    if visible <= 512 || first >= visible { return; }
    let mut keys = [[0.0f32; 4]; K];
    index4!(G, { if first + G < visible {
        let key = pooled.add((first + G) * 4 * 128);
        index4!(F, { keys[G][F] = *key.add(lane + F * 32); });
    } });
    let mut score = [[0.0f32; K]; Q];
    for head in 0..4 {
        index4!(R, {
            let position = token + R;
            let visible = (past + position + 1) / 4;
            if position < end && visible > 512 {
                let source = q.add((position * 4 + head) * 128);
                let query = [*source.add(lane), *source.add(lane + 32), *source.add(lane + 64), *source.add(lane + 96)];
                index4!(G, { if first + G < visible {
                    let mut dot = 0.0;
                    index4!(F, { dot += query[F] * keys[G][F]; });
                    score[R][G] += sum(dot).max(0.0);
                } });
            }
        });
    }
    if lane == 0 {
        index4!(R, {
            let position = token + R;
            let visible = (past + position + 1) / 4;
            if position < end && visible > 512 {
                index4!(G, { if first + G < visible {
                    *scores.add(position * blocks + first + G) = score[R][G] / sqrt(128.0);
                } });
            }
        });
    }
}
#[inline(always)]
unsafe fn index_scores_single(p: crate::kernel_params::QsaIndex) {
    const QUERIES: usize = crate::kernel_params::QSA_INDEX_QUERIES;
    let (q, pooled, scores) = (p.query as *const f32, p.pooled as *const f32, p.scores as *mut f32);
    let (tokens, past, blocks) = (p.tokens, p.past, p.blocks);
    let thread = index();
    let tile = thread / 32;
    let lane = thread % 32;
    if tile >= (tokens as usize).div_ceil(QUERIES) * blocks as usize { return; }
    let token = tile / blocks as usize * QUERIES;
    let block = tile % blocks as usize;
    let end = (token + QUERIES).min(tokens as usize);
    let visible = (past as usize + end) / 4;
    if visible <= 512 || block >= visible { return; }
    let key = pooled.add(block * 4 * 128);
    // Four lane features stay live across every head and neighboring query.
    // Each query keeps its original feature chain and warp reduction tree.
    let key = [*key.add(lane), *key.add(lane + 32), *key.add(lane + 64), *key.add(lane + 96)];
    let mut score = [0.0; QUERIES];
    let mut head = 0;
    while head < 4 {
        let mut query = 0;
        while query < QUERIES {
            let position = token + query;
            let visible = (past as usize + position + 1) / 4;
            if position < end && visible > 512 && block < visible {
                let source = q.add((position * 4 + head) * 128);
                let mut dot = 0.0;
                let mut part = 0;
                while part < 4 {
                    dot += *source.add(lane + part * 32) * key[part];
                    part += 1;
                }
                score[query] += sum(dot).max(0.0);
            }
            query += 1;
        }
        head += 1;
    }
    if lane == 0 {
        let mut query = 0;
        while query < QUERIES {
            let position = token + query;
            let visible = (past as usize + position + 1) / 4;
            if position < end && visible > 512 && block < visible {
                *scores.add(position * blocks as usize + block) = score[query] / sqrt(128.0);
            }
            query += 1;
        }
    }
}

/// Prompt index scores without lane shuffles, the same bytes as
/// index_scores_tile / index_scores_single: a thread owns 4 queries x 2 groups
/// and folds each score's 32 lane partials (lane l adds features l, l + 32,
/// l + 64, l + 96 in that order, from 0) in bit-reversed lane order with a
/// 5-level pairwise stack, which is exactly `sum`'s butterfly tree (16, 8, 4,
/// 2, 1); then relu, heads 0..3 from 0, / sqrt(128). CTA = 16 queries x 64
/// groups, 128 threads (warp w: queries 4w .. 4w + 3, lane: groups lane and
/// lane + 32); static shared 40,960 B = keys [64][32 lanes][4 features]
/// (16-byte entries XOR-swizzled by group % 8) + one head's queries [16][32][4].
const XT: usize = 16;
const XG: usize = 64;
const fn bitrev5(k: usize) -> usize { ((k & 1) << 4) | ((k & 2) << 2) | (k & 4) | ((k & 8) >> 2) | ((k & 16) >> 4) }
macro_rules! x32 {
    ($i:ident, $body:block) => { x32!(@ $i, $body, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31); };
    (@ $i:ident, $body:block, $($n:literal)*) => { $( { const $i: usize = $n; $body } )* };
}
macro_rules! index2 { ($i:ident, $body:block) => { { const $i: usize = 0; $body } { const $i: usize = 1; $body } }; }
#[inline(always)]
unsafe fn store4f(p: u32, v: [f32; 4]) {
    core::arch::asm!("st.shared.v4.f32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) p, a = in(reg32) v[0], b = in(reg32) v[1],
        c = in(reg32) v[2], d = in(reg32) v[3]);
}
/// Block 128, grid ceil(tokens / 16) * ceil(blocks / 64).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_scores_t(mut p: crate::kernel_params::QsaIndex, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    index_scores_thread(p);
}
#[inline(always)]
unsafe fn index_scores_thread(p: crate::kernel_params::QsaIndex) {
    use crate::ptx::{load4, shared, sync};
    let (q, pooled, scores) = (p.query as *const f32, p.pooled as *const f32, p.scores as *mut f32);
    let (tokens, past, blocks) = (p.tokens as usize, p.past as usize, p.blocks as usize);
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let tiles = blocks.div_ceil(XG);
    let tile = core::arch::nvptx::_block_idx_x() as usize;
    if tile >= tokens.div_ceil(XT) * tiles { return; }
    let (token, first) = (tile / tiles * XT, tile % tiles * XG);
    let end = (token + XT).min(tokens);
    if (past + end) / 4 <= 512 || first >= (past + end) / 4 { return; }
    let keys = shared::<{ XG * 32 * 16 + XT * 32 * 16 }>();
    let queries = keys + (XG * 32 * 16) as u32;
    let mut e = tid;
    while e < XG * 32 {
        let (g, l) = (e / 32, e % 32);
        let mut v = [0.0f32; 4];
        if first + g < blocks { let key = pooled.add((first + g) * 4 * 128); index4!(F, { v[F] = *key.add(l + F * 32); }); }
        store4f(keys + ((g * 32 + (l ^ (g & 7))) * 16) as u32, v);
        e += 128;
    }
    let mut score = [[0.0f32; 2]; 4];
    for head in 0..4 {
        sync();
        let mut e = tid;
        while e < XT * 32 {
            let (t, l) = (e / 32, e % 32);
            let mut v = [0.0f32; 4];
            if token + t < tokens { let source = q.add(((token + t) * 4 + head) * 128); index4!(F, { v[F] = *source.add(l + F * 32); }); }
            store4f(queries + ((t * 32 + l) * 16) as u32, v);
            e += 128;
        }
        sync();
        let mut stack = [[[0.0f32; 5]; 2]; 4];
        x32!(K, {
            const L: usize = bitrev5(K);
            const LEVEL: u32 = (!K).trailing_zeros();
            let mut qv = [[0.0f32; 4]; 4];
            index4!(T, { qv[T] = load4(queries + (((4 * warp + T) * 32 + L) * 16) as u32).map(f32::from_bits); });
            let mut kv = [[0.0f32; 4]; 2];
            index2!(J, { let g = lane + 32 * J; kv[J] = load4(keys + ((g * 32 + (L ^ (g & 7))) * 16) as u32).map(f32::from_bits); });
            index4!(T, { index2!(J, {
                let mut dot = 0.0;
                index4!(F, { dot += qv[T][F] * kv[J][F]; });
                // Level n holds the sum of 2^n consecutive partials (left + right).
                let mut c = dot;
                if K & 1 != 0 { c = stack[T][J][0] + c;
                if K & 2 != 0 { c = stack[T][J][1] + c;
                if K & 4 != 0 { c = stack[T][J][2] + c;
                if K & 8 != 0 { c = stack[T][J][3] + c;
                if K & 16 != 0 { c = stack[T][J][4] + c; } } } } }
                if LEVEL < 5 { stack[T][J][LEVEL as usize] = c; } else { score[T][J] += c.max(0.0); }
            }); });
        });
    }
    index4!(T, {
        let position = token + 4 * warp + T;
        let visible = (past + position + 1) / 4;
        if position < end && visible > 512 {
            index2!(J, {
                let g = first + lane + 32 * J;
                if g < visible { *scores.add(position * blocks + g) = score[T][J] / sqrt(128.0); }
            });
        }
    });
}

/// Index scores with 4 queries x 4 groups per thread: CTA = 32 queries
/// x 64 groups (warp w: queries 8w + 4 (lane / 16) .., groups lane % 16 + 16j);
/// same per-score tree, same bytes. Block 128, grid ceil(tokens / 32) * ceil(blocks / 64).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_scores_t44(mut p: crate::kernel_params::QsaIndex, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    index_scores_thread44(p);
}
#[inline(always)]
unsafe fn index_scores_thread44(p: crate::kernel_params::QsaIndex) {
    const XT: usize = 32;
    use crate::ptx::{load4, shared, sync};
    let (q, pooled, scores) = (p.query as *const f32, p.pooled as *const f32, p.scores as *mut f32);
    let (tokens, past, blocks) = (p.tokens as usize, p.past as usize, p.blocks as usize);
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let tiles = blocks.div_ceil(XG);
    let tile = core::arch::nvptx::_block_idx_x() as usize;
    if tile >= tokens.div_ceil(XT) * tiles { return; }
    let (token, first) = (tile / tiles * XT, tile % tiles * XG);
    let end = (token + XT).min(tokens);
    if (past + end) / 4 <= 512 || first >= (past + end) / 4 { return; }
    let keys = shared::<{ XG * 32 * 16 + XT * 32 * 16 }>();
    let queries = keys + (XG * 32 * 16) as u32;
    let mut e = tid;
    while e < XG * 32 {
        let (g, l) = (e / 32, e % 32);
        let mut v = [0.0f32; 4];
        if first + g < blocks { let key = pooled.add((first + g) * 4 * 128); index4!(F, { v[F] = *key.add(l + F * 32); }); }
        store4f(keys + ((g * 32 + (l ^ (g & 7))) * 16) as u32, v);
        e += 128;
    }
    let quad = lane / 16;
    let mut score = [[0.0f32; 4]; 4];
    for head in 0..4 {
        sync();
        let mut e = tid;
        while e < XT * 32 {
            let (t, l) = (e / 32, e % 32);
            let mut v = [0.0f32; 4];
            if token + t < tokens { let source = q.add(((token + t) * 4 + head) * 128); index4!(F, { v[F] = *source.add(l + F * 32); }); }
            store4f(queries + ((t * 32 + l) * 16) as u32, v);
            e += 128;
        }
        sync();
        let mut stack = [[[0.0f32; 5]; 4]; 4];
        x32!(K, {
            const L: usize = bitrev5(K);
            const LEVEL: u32 = (!K).trailing_zeros();
            let mut qv = [[0.0f32; 4]; 4];
            index4!(T, { qv[T] = load4(queries + (((8 * warp + 4 * quad + T) * 32 + L) * 16) as u32).map(f32::from_bits); });
            let mut kv = [[0.0f32; 4]; 4];
            index4!(J, { let g = lane % 16 + 16 * J; kv[J] = load4(keys + ((g * 32 + (L ^ (g & 7))) * 16) as u32).map(f32::from_bits); });
            index4!(T, { index4!(J, {
                let mut dot = 0.0;
                index4!(F, { dot += qv[T][F] * kv[J][F]; });
                // Level n holds the sum of 2^n consecutive partials (left + right).
                let mut c = dot;
                if K & 1 != 0 { c = stack[T][J][0] + c;
                if K & 2 != 0 { c = stack[T][J][1] + c;
                if K & 4 != 0 { c = stack[T][J][2] + c;
                if K & 8 != 0 { c = stack[T][J][3] + c;
                if K & 16 != 0 { c = stack[T][J][4] + c; } } } } }
                if LEVEL < 5 { stack[T][J][LEVEL as usize] = c; } else { score[T][J] += c.max(0.0); }
            }); });
        });
    }
    index4!(T, {
        let position = token + 8 * warp + 4 * quad + T;
        let visible = (past + position + 1) / 4;
        if position < end && visible > 512 {
            index4!(J, {
                let g = first + lane % 16 + 16 * J;
                if g < visible { *scores.add(position * blocks + g) = score[T][J] / sqrt(128.0); }
            });
        }
    });
}

/// Prompt attention scores (>= 64 tokens, switch off) without lane shuffles,
/// the same bytes as flash_qsa_attention_scores / _k8: main's lane l adds
/// features l + 32i (i = 0..7) from 0, the butterfly (16, 8, 4, 2, 1) sums
/// the 32 lanes, then * (1/16). Here a thread owns 3 heads x 1 slot: the
/// partial of lane 4a + b is element b of 16-byte chunks a + 8i, and the tree
/// is rebuilt exactly as an 8-leaf bit-reversed fold over a for each b (bits
/// 4, 3, 2 of l), then (S0 + S2) + (S1 + S3) (bits 1, 0). CTA = token x
/// 12-head group (warp w: heads 3w .. 3w + 2, lane: slot) x a slot range of
/// 32-slot stages (grid y distributes ranges only); 128 threads, static shared
/// 45,056 B = Q [12][256] + K stage [32][256] (chunks XOR-swizzled by slot % 8).
const TSLOTS: usize = 32;
#[inline(always)]
unsafe fn opaque8() -> usize {
    let n: u32;
    core::arch::asm!("mov.u32 {n}, 8;", n = out(reg32) n, options(pure, nomem, nostack));
    n as usize
}
const fn bitrev3(k: usize) -> usize { ((k & 1) << 2) | (k & 2) | ((k & 4) >> 2) }
macro_rules! index8 { ($i:ident, $body:block) => {
    index4!($i, $body);
    { const $i: usize = 4; $body } { const $i: usize = 5; $body } { const $i: usize = 6; $body } { const $i: usize = 7; $body }
}; }
macro_rules! index6 { ($i:ident, $body:block) => { index3!($i, $body); { const $i: usize = 3; $body } { const $i: usize = 4; $body } { const $i: usize = 5; $body } }; }
macro_rules! index3 { ($i:ident, $body:block) => { { const $i: usize = 0; $body } { const $i: usize = 1; $body } { const $i: usize = 2; $body } }; }
/// Block 128, grid (groups * tokens, slot ranges).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores_t(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_scores_thread::<3>(p);
}
#[inline(always)]
unsafe fn attention_scores_thread<const HPT: usize>(p: crate::kernel_params::QsaAttention) {
    let threads = 32 * (crate::kernel_params::QSA_SCORE_HEADS / HPT);
    use crate::ptx::{commit, copy16, load4, shared, sync, wait};
    const HEADS: usize = crate::kernel_params::QSA_SCORE_HEADS;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let groups = (p.query_start as usize % HEADS + p.query_heads as usize).div_ceil(HEADS);
    let group = core::arch::nvptx::_block_idx_x() as usize;
    if group >= groups * p.tokens as usize { return; }
    let token = group / groups;
    let first = (p.query_start as usize / HEADS + group % groups) * HEADS;
    let length = p.past as usize + token + 1;
    let complete = length / 4;
    let visible = selected_length(length);
    let range = crate::kernel_params::QSA_MAX_POSITIONS.div_ceil(core::arch::nvptx::_grid_dim_y() as usize).div_ceil(TSLOTS) * TSLOTS;
    let begin = core::arch::nvptx::_block_idx_y() as usize * range;
    if begin >= visible { return; }
    let end = (begin + range).min(visible);
    let key_head = first / p.queries_per_key as usize - p.key_start as usize;
    let (qs, qe) = (p.query_start as usize, (p.query_start + p.query_heads) as usize);
    let row_of = |r: usize| if first + r >= qs && first + r < qe { token * p.query_heads as usize + first + r - qs } else { usize::MAX };
    let qbase = shared::<{ HEADS * 1024 + TSLOTS * 1024 + ID_TILE }>();
    let kbase = qbase + (HEADS * 1024) as u32;
    let id_tile = kbase + (TSLOTS * 1024) as u32;
    let mut e = tid;
    while e < HEADS * 64 {
        let (h, c) = (e / 64, e % 64);
        let row = row_of(h);
        let dst = qbase + (h * 1024 + c * 16) as u32;
        if row != usize::MAX { copy16(dst, (p.query as *const f32).add(row * 256 + 4 * c).cast()); } else { store4f(dst, [0.0; 4]); }
        e += threads;
    }
    // the token's IDs in shared memory before the first K stage.
    stage_ids(id_tile, p.ids as *const u32, complete, token, p.blocks, tid, threads);
    commit();
    wait();
    sync();
    let keys = (p.keys as *const f32).add(key_head * 256);
    let kstride = p.key_heads as usize * 256;
    let scores = p.scores as *mut f32;
    let rows: [usize; 6] = core::array::from_fn(|h| if h < HPT { row_of(HPT * warp + h) } else { usize::MAX });
    let mut start = begin;
    while start < end {
        let mut e = tid;
        while e < TSLOTS * 64 {
            let (s, c) = (e / 64, e % 64);
            let dst = kbase + (s * 1024 + ((c ^ (s & 7)) * 16)) as u32;
            if start + s < end {
                copy16(dst, keys.add(tile_position(id_tile, complete, start + s) * kstride + 4 * c).cast());
            } else { store4f(dst, [0.0; 4]); }
            e += threads;
        }
        commit();
        wait();
        sync();
        let s = lane;
        let steps = opaque8();
        let mut stack = [[[0.0f32; 3]; 4]; 6];
        let mut total = [[0.0f32; 4]; 6];
        index8!(AK, {
            const A: usize = bitrev3(AK);
            const LEVEL: u32 = (!AK).trailing_zeros();
            let mut v = [[0.0f32; 4]; 6];
            // Rolled (opaque trip count): fully unrolled, ptxas hoists every load and spills.
            let mut i = 0;
            while i < steps {
                let c = A + 8 * i;
                let k = load4(kbase + (s * 1024 + ((c ^ (s & 7)) * 16)) as u32).map(f32::from_bits);
                index6!(H, { if H < HPT {
                    let q = load4(qbase + ((HPT * warp + H) * 1024 + c * 16) as u32).map(f32::from_bits);
                    index4!(B, { v[H][B] += q[B] * k[B]; });
                } });
                i += 1;
            }
            index6!(H, { if H < HPT { index4!(B, {
                let mut c = v[H][B];
                if AK & 1 != 0 { c = stack[H][B][0] + c;
                if AK & 2 != 0 { c = stack[H][B][1] + c;
                if AK & 4 != 0 { c = stack[H][B][2] + c; } } }
                if LEVEL < 3 { stack[H][B][LEVEL as usize] = c; } else { total[H][B] = c; }
            }); } });
        });
        if start + s < end {
            index6!(H, {
                if H < HPT && rows[H] != usize::MAX {
                    let t = (total[H][0] + total[H][2]) + (total[H][1] + total[H][3]);
                    *scores.add(rows[H] * p.stride as usize + start + s) = t * (1.0 / 16.0);
                }
            });
        }
        sync();
        start += TSLOTS;
    }
}

/// Unsigned key ordered like the score; -0 counts as +0, as `==` does.
#[inline(always)]
fn ordered(score: f32) -> u32 {
    let bits = if score == 0.0 { 0 } else { score.to_bits() };
    if bits >> 31 != 0 { !bits } else { bits | 1 << 31 }
}
#[inline(always)]
unsafe fn match_any(value: u32) -> u32 {
    let peers: u32;
    core::arch::asm!("match.any.sync.b32 {p}, {v}, 0xffffffff;", p = out(reg32) peers, v = in(reg32) value);
    peers
}
#[inline(always)]
unsafe fn shared_u32(p: u32) -> u32 {
    let v: u32;
    core::arch::asm!("ld.shared.u32 {v}, [{p}];", v = out(reg32) v, p = in(reg32) p);
    v
}
#[inline(always)]
unsafe fn set_shared_u32(p: u32, v: u32) { core::arch::asm!("st.shared.u32 [{p}], {v};", p = in(reg32) p, v = in(reg32) v) }

/// CTA per query selects the 512 complete groups a full sort by descending
/// score, then ascending group, would: 8-bit radix passes narrow the 512th
/// key; emit groups above it and the earliest groups equal to it in ascending
/// position order. Same groups as picking the best group 512 times.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_select(mut p: crate::kernel_params::QsaIndex, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    index_select(p);
}
#[inline(always)]
unsafe fn index_select(p: crate::kernel_params::QsaIndex) { index_select_n::<256, 1072>(p) }
/// THREADS = 256 (main) or 1024 (qsa_tc.rs _w entries); the picks and their order do not depend on it.
#[inline(always)]
pub(crate) unsafe fn index_select_n<const THREADS: usize, const SHARED: usize>(p: crate::kernel_params::QsaIndex) {
    let token = core::arch::nvptx::_block_idx_x() as usize;
    let thread = core::arch::nvptx::_thread_idx_x() as usize;
    let (warp, lane) = (thread / 32, thread % 32);
    if token >= p.tokens as usize { return; }
    let visible = (p.past as usize + token + 1) / 4;
    if visible <= 512 { return; }
    let scores = (p.scores as *mut f32).add(token * p.blocks as usize);
    // [0, 256) digit counts, [256, 256 + W) warp tie counts, then prefix, still needed, bucket size (W = THREADS / 32).
    let sh = crate::ptx::shared::<SHARED>(); // 1072 + (THREADS / 32 - 8) * 4
    let w = THREADS / 32;
    let at = |i: usize| sh + (i * 4) as u32;
    let rounds = visible.div_ceil(THREADS);
    let (mut prefix, mut high, mut need, mut tied) = (0u32, 0u32, 512u32, 0u32);
    let mut shift = 32;
    while shift != 0 && tied != need {
        shift -= 8;
        if THREADS == 256 || thread < 256 { set_shared_u32(at(thread), 0); }
        crate::ptx::sync();
        let mut round = 0;
        while round < rounds {
            let i = round * THREADS + thread;
            let key = if i < visible { ordered(*scores.add(i)) } else { 0 };
            let digit = if i < visible && key & high == prefix { key >> shift & 255 } else { 256 };
            let peers = match_any(digit);
            if digit != 256 && lane as u32 == peers.trailing_zeros() {
                core::arch::asm!("red.shared.add.u32 [{p}], {n};", p = in(reg32) at(digit as usize), n = in(reg32) peers.count_ones());
            }
            round += 1;
        }
        crate::ptx::sync();
        if warp == 0 {
            // Lane l owns digits 255 - 8l down to 248 - 8l; scan from the top.
            let mut total = 0;
            for j in 0..8 { total += shared_u32(at(255 - 8 * lane - j)); }
            let mut inclusive = total;
            let mut distance = 1;
            while distance < 32 {
                let other: u32;
                core::arch::asm!("shfl.sync.up.b32 {o}, {v}, {d}, 0, 0xffffffff;", o = out(reg32) other, v = in(reg32) inclusive, d = in(reg32) distance);
                if lane as u32 >= distance { inclusive += other; }
                distance *= 2;
            }
            let mut above = inclusive - total;
            if above < need && need <= inclusive {
                let mut digit = 255 - 8 * lane;
                let mut size = shared_u32(at(digit));
                while above + size < need { above += size; digit -= 1; size = shared_u32(at(digit)); }
                set_shared_u32(at(256 + w), prefix | (digit as u32) << shift);
                set_shared_u32(at(257 + w), need - above);
                set_shared_u32(at(258 + w), size);
            }
        }
        crate::ptx::sync();
        prefix = shared_u32(at(256 + w));
        need = shared_u32(at(257 + w));
        tied = shared_u32(at(258 + w));
        high |= 255 << shift;
    }
    // Every key above the prefix is taken, plus the first `need` equal to it.
    let mut taken = 0;
    let ids = scores.cast::<u32>();
    let mut total = 0u32;
    let mut round = 0;
    while round < rounds && total < 512 {
        let i = round * THREADS + thread;
        let key = if i < visible { ordered(*scores.add(i)) & high } else { 0 };
        let equal = i < visible && key == prefix;
        let mut pick = i < visible && key > prefix;
        if tied == need { pick |= equal; }
        else if taken < need {
            let votes = ballot(equal);
            if lane == 0 { set_shared_u32(at(256 + warp), votes.count_ones()); }
            crate::ptx::sync();
            let mut before = taken + (votes & ((1u32 << lane) - 1)).count_ones();
            let mut v = 0;
            while v < THREADS / 32 {
                let count = shared_u32(at(256 + v));
                if v < warp { before += count; }
                taken += count;
                v += 1;
            }
            pick |= equal && before < need;
            crate::ptx::sync();
        }
        let votes = ballot(pick);
        if lane == 0 { set_shared_u32(at(256 + warp), votes.count_ones()); }
        // All this round's scores are in registers before IDs overwrite them.
        // A selected ID's offset is <= its source index, so future rounds'
        // scores cannot be overwritten by this ascending compaction.
        crate::ptx::sync();
        let mut offset = total + (votes & ((1u32 << lane) - 1)).count_ones();
        let mut v = 0;
        while v < THREADS / 32 {
            let count = shared_u32(at(256 + v));
            if v < warp { offset += count; }
            total += count;
            v += 1;
        }
        if pick { *ids.add(offset as usize) = i as u32; }
        // Out-of-range lanes also wait; counts cannot be reused early.
        crate::ptx::sync();
        round += 1;
    }
}

/// Selected groups plus the open tail; short rows see every position.
#[inline(always)]
pub(crate) fn selected_length(length: usize) -> usize {
    if length / 4 <= 512 { length } else { 4 * 512 + length % 4 }
}
/// Dense position of an ascending visible slot. Short rows never read IDs.
#[inline(always)]
pub(crate) unsafe fn selected_position(ids: *const u32, complete: usize, token: usize, blocks: u32, slot: usize) -> usize {
    if complete <= 512 { slot }
    else if slot < 4 * 512 { 4 * *ids.add(token * blocks as usize + slot / 4) as usize + slot % 4 }
    else { 4 * complete + slot - 4 * 512 }
}
/// Bytes of an ID tile: one token's 512 selected group IDs in shared memory.
pub(crate) const ID_TILE: usize = 4 * 512;
/// Issues the copies of a token's 512 selected group IDs into the ID tile
/// (rows with more than 512 complete groups only; the caller commits and
/// waits). Prompt kernels then gather without a global ID load in front.
#[inline(always)]
pub(crate) unsafe fn stage_ids(tile: u32, ids: *const u32, complete: usize, token: usize, blocks: u32, tid: usize, threads: usize) {
    if complete <= 512 { return; }
    let mut e = tid;
    while e < 512 {
        crate::ptx::copy4(tile + (4 * e) as u32, ids.add(token * blocks as usize + e));
        e += threads;
    }
}
/// selected_position with the IDs read from a staged ID tile (same positions).
#[inline(always)]
pub(crate) unsafe fn tile_position(tile: u32, complete: usize, slot: usize) -> usize {
    if complete <= 512 { slot }
    else if slot < 4 * 512 { 4 * shared_u32(tile + (4 * (slot / 4)) as u32) as usize + slot % 4 }
    else { 4 * complete + slot - 4 * 512 }
}
/// This lane's eight features of one cached row (lane, lane + 32, ...).
#[inline(always)]
unsafe fn row_features(row: *const f32, lane: usize, precision: u32) -> [f32; 8] {
    [cached(row, lane, precision), cached(row, lane + 32, precision), cached(row, lane + 64, precision),
        cached(row, lane + 96, precision), cached(row, lane + 128, precision), cached(row, lane + 160, precision),
        cached(row, lane + 192, precision), cached(row, lane + 224, precision)]
}
#[inline(always)]
unsafe fn score(query: [f32; 8], key: [f32; 8], precision: u32) -> f32 {
    let mut dot = 0.0;
    macro_rules! part { ($i:expr) => { dot += query[$i] * key[$i]; }; }
    part!(0); part!(1); part!(2); part!(3); part!(4); part!(5); part!(6); part!(7);
    let dot = sum(dot);
    if precision == 0 { dot * (1.0 / 16.0) } else { rounded(dot) * (1.0 / 16.0) }
}
#[inline(always)]
unsafe fn attention_score(query: [f32; 8], key: *const f32, lane: usize, precision: u32) -> f32 {
    score(query, row_features(key, lane, precision), precision)
}
#[inline(always)]
pub(crate) unsafe fn output_gate(gate: f32, swish: u32) -> f32 { sigmoid(gate) * if swish == 0 { 1.0 } else { gate } }
#[inline(always)]
fn gated_bf16(value: f32, gate: f32) -> f32 { rounded(rounded(value) * rounded(gate)) }

/// Warp per local query head/token. Two-pass stable FP32 softmax, followed
/// by the original per-feature sigmoid gate. Output is [tokens,qheads,256].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_gate(params: crate::kernel_params::QsaGate) {
    let crate::kernel_params::QsaGate { query_gate: qg, attention: y, packed, width, tokens } = params;
    let i = index();
    if i < width as usize * tokens as usize {
        let gate = rounded(sigmoid(*qg.add(i / 256 * 512 + 256 + i % 256)));
        let value = rounded(rounded(*y.add(i)) * gate);
        if packed.is_null() { *y.add(i) = value; }
        else { crate::gemm::store_bf16_input_element(packed, i / width as usize, i % width as usize, width as usize, value); }
    }
}

/// Softmax over one indexed-attention row's stored scores: a parallel maximum
/// of the visible scores, each replaced by its probability, and the
/// denominator summed in position order (source lanes 0..31 per block of
/// 32 visible keys), as the one-warp indexed kernel did.
#[inline(always)]
unsafe fn stored_softmax(saved_scores: *mut f32, length: usize, lane: usize) -> f32 {
    let mut highest = f32::NEG_INFINITY;
    for at in (lane..length).step_by(32) {
        highest = highest.max(*saved_scores.add(at));
    }
    highest = crate::layer::maximum(highest);
    let mut denominator = 0.0;
    let mut slot = 0;
    while slot < length {
        let mine = slot + lane < length;
        let probability = if mine { exp(*saved_scores.add(slot + lane) - highest) } else { 0.0 };
        for source in 0..32 {
            let value = f32::from_bits(shuffle_xor(u32::MAX, probability.to_bits(), (lane ^ source) as u32, 31));
            if slot + source < length { denominator += value; }
        }
        if mine { *saved_scores.add(slot + lane) = probability; }
        slot += 32;
    }
    denominator
}

/// Indexed attention, step 2 of 3: one warp per (query row): the stored
/// scores become probabilities and the row's denominator is kept.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_softmax(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_softmax(p);
}
#[inline(always)]
unsafe fn attention_softmax(p: crate::kernel_params::QsaAttention) {
    let (row, lane) = (index() / 32, index() % 32);
    if row >= p.query_heads as usize * p.tokens as usize { return; }
    let token = row / p.query_heads as usize;
    let length = p.past as usize + token + 1;
    let scores = (p.scores as *mut f32).add(row * p.stride as usize);
    let denominator = stored_softmax(scores, selected_length(length), lane);
    if lane == 0 { *(p.denominators as *mut f32).add(row) = denominator; }
}

/// Indexed attention, step 3 of 3, for every query count (decode, MTP
/// verification and prompt tiles alike): one block per token, KV head and 64
/// features. Each feature adds probability x value over its visible slots in
/// ascending order, divides by the row's denominator and applies the gate.
/// The query heads of a KV head reuse selected FP32 values from two
/// shared-memory tiles; only loads overlap, never the per-feature order.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_values(p);
}
#[inline(always)]
unsafe fn attention_values(p: crate::kernel_params::QsaAttention) {
    use crate::ptx::{commit, copy4, copy16, load, shared, sync, wait, wait_one};
    const TILE: usize = 64;
    const STAGE: usize = TILE * 64 * 4 + crate::kernel_params::QSA_VALUE_HEADS * TILE * 4;
    let block = core::arch::nvptx::_block_idx_x() as usize;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let threads = core::arch::nvptx::_block_dim_x() as usize;
    let (token, rest) = (block % p.tokens as usize, block / p.tokens as usize);
    let (key_head, slice) = (rest / 4, rest % 4);
    let first = ((key_head + p.key_start as usize) * p.queries_per_key as usize).max(p.query_start as usize);
    let end = ((key_head + p.key_start as usize + 1) * p.queries_per_key as usize)
        .min((p.query_start + p.query_heads) as usize);
    let group = end - first;
    let first_row = token * p.query_heads as usize + first - p.query_start as usize;
    let (head, feature) = (tid / 64, slice * 64 + tid % 64);
    let length = p.past as usize + token + 1;
    let visible = selected_length(length);
    let values = (p.values as *const f32).add(key_head * 256 + slice * 64);
    let stride = p.key_heads as usize * 256;
    let base = shared::<{ 2 * STAGE }>();
    let issue = |tile: usize| {
        let (start, buffer) = (tile * TILE, base + (tile % 2 * STAGE) as u32);
        let count = TILE.min(visible - start);
        let mut chunk = tid;
        while chunk < count * 16 {
            let pos = selected_position(p.ids as *const u32, length / 4, token, p.blocks, start + chunk / 16);
            copy16(buffer + (chunk * 16) as u32, values.add(pos * stride + chunk % 16 * 4).cast());
            chunk += threads;
        }
        let weights = buffer + (TILE * 64 * 4) as u32;
        let mut chunk = tid;
        while chunk < group * count {
            let (h, i) = (chunk / count, chunk % count);
            let source = (p.scores as *const f32).add((first_row + h) * p.stride as usize + start + i);
            copy4(weights + ((h * TILE + i) * 4) as u32, source.cast());
            chunk += threads;
        }
        commit();
    };
    let tiles = visible.div_ceil(TILE);
    issue(0);
    let mut total = 0.0f32;
    for tile in 0..tiles {
        if tile + 1 < tiles { issue(tile + 1); wait_one(); } else { wait(); }
        sync();
        if head < group {
            let buffer = base + (tile % 2 * STAGE) as u32;
            let weights = buffer + ((TILE * 64 + head * TILE) * 4) as u32;
            let count = TILE.min(visible - tile * TILE);
            let value = |i: usize| load(buffer + ((i * 64 + tid % 64) * 4) as u32);
            let mut i = 0;
            while i + 8 <= count {
                let (mut w, mut v) = ([0.0f32; 8], [0.0f32; 8]);
                for j in 0..8 { w[j] = load(weights + ((i + j) * 4) as u32); v[j] = value(i + j); }
                for j in 0..8 { total += w[j] * v[j]; }
                i += 8;
            }
            while i < count { total += load(weights + (i * 4) as u32) * value(i); i += 1; }
        }
        sync();
    }
    if head < group {
        let row = first_row + head;
        let denominator = *(p.denominators as *const f32).add(row);
        let gate = output_gate(*(p.query_gate as *const f32).add(row * 512 + 256 + feature), 0);
        *(p.output as *mut f32).add(row * 256 + feature) = (total / denominator) * gate;
    }
}

/// Small launches (decode rows, MTP verify, prompt tails under 64 tokens,
/// the values16 schedule of the qsa-decode-longctx probe): main's values with
/// 16-feature slices instead of 64, so a KV head takes 16 blocks of 192 threads
/// (12 heads x 16 features) instead of 4 of 768, and the token's IDs staged once
/// in shared memory. Same 64-slot tiles and the same per-output arithmetic:
/// ascending slot chain, then / d, * gate. The host takes it for one-token rows.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_s16(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_values_s::<16>(p);
}
/// Batch rows of flash_qsa_attention_values_s16 (grid y = row).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_s16_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    attention_values_s::<16>(batch_attention(p, r, row));
}
#[inline(always)]
unsafe fn attention_values_s<const S: usize>(p: crate::kernel_params::QsaAttention) {
    use crate::ptx::{commit, copy4, copy16, load, shared, sync, wait, wait_one};
    const TILE: usize = 64;
    let stage = TILE * S * 4 + crate::kernel_params::QSA_VALUE_HEADS * TILE * 4;
    let block = core::arch::nvptx::_block_idx_x() as usize;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let threads = core::arch::nvptx::_block_dim_x() as usize;
    let (token, rest) = (block % p.tokens as usize, block / p.tokens as usize);
    let (key_head, slice) = (rest / (256 / S), rest % (256 / S));
    let first = ((key_head + p.key_start as usize) * p.queries_per_key as usize).max(p.query_start as usize);
    let end = ((key_head + p.key_start as usize + 1) * p.queries_per_key as usize)
        .min((p.query_start + p.query_heads) as usize);
    let group = end - first;
    let first_row = token * p.query_heads as usize + first - p.query_start as usize;
    let (head, feature) = (tid / S, slice * S + tid % S);
    let length = p.past as usize + token + 1;
    let visible = selected_length(length);
    let values = (p.values as *const f32).add(key_head * 256 + slice * S);
    let stride = p.key_heads as usize * 256;
    // The token's IDs are staged once (the shared ID tile; same positions).
    let base = shared::<{ 2 * (TILE * 16 * 4 + crate::kernel_params::QSA_VALUE_HEADS * TILE * 4) + ID_TILE }>();
    let id_tile = base + (2 * stage) as u32;
    stage_ids(id_tile, p.ids as *const u32, length / 4, token, p.blocks, tid, threads);
    commit();
    wait();
    sync();
    let issue = |tile: usize| {
        let (start, buffer) = (tile * TILE, base + (tile % 2 * stage) as u32);
        let count = TILE.min(visible - start);
        let mut chunk = tid;
        while chunk < count * (S / 4) {
            let pos = tile_position(id_tile, length / 4, start + chunk / (S / 4));
            copy16(buffer + (chunk * 16) as u32, values.add(pos * stride + chunk % (S / 4) * 4).cast());
            chunk += threads;
        }
        let weights = buffer + (TILE * S * 4) as u32;
        let mut chunk = tid;
        while chunk < group * count {
            let (h, i) = (chunk / count, chunk % count);
            let source = (p.scores as *const f32).add((first_row + h) * p.stride as usize + start + i);
            copy4(weights + ((h * TILE + i) * 4) as u32, source.cast());
            chunk += threads;
        }
        commit();
    };
    let tiles = visible.div_ceil(TILE);
    issue(0);
    let mut total = 0.0f32;
    for tile in 0..tiles {
        if tile + 1 < tiles { issue(tile + 1); wait_one(); } else { wait(); }
        sync();
        if head < group {
            let buffer = base + (tile % 2 * stage) as u32;
            let weights = buffer + ((TILE * S + head * TILE) * 4) as u32;
            let count = TILE.min(visible - tile * TILE);
            let value = |i: usize| load(buffer + ((i * S + tid % S) * 4) as u32);
            let mut i = 0;
            while i + 8 <= count {
                let (mut w, mut v) = ([0.0f32; 8], [0.0f32; 8]);
                for j in 0..8 { w[j] = load(weights + ((i + j) * 4) as u32); v[j] = value(i + j); }
                for j in 0..8 { total += w[j] * v[j]; }
                i += 8;
            }
            while i < count { total += load(weights + (i * 4) as u32) * value(i); i += 1; }
        }
        sync();
    }
    if head < group {
        let row = first_row + head;
        let denominator = *(p.denominators as *const f32).add(row);
        let gate = output_gate(*(p.query_gate as *const f32).add(row * 512 + 256 + feature), 0);
        *(p.output as *mut f32).add(row * 256 + feature) = (total / denominator) * gate;
    }
}

/// Prompt tiles (64 or more tokens): one block per token and KV head, thread =
/// (feature, four query heads), 16-slot double-buffered tiles of whole FP32
/// rows. Each (head, feature) adds probability x value over its visible slots
/// in ascending order (multiply, then add) exactly as
/// flash_qsa_attention_values (bytes equal); every staged value is read once
/// per thread for four heads, the weights four slots at a time.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_wide(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_values_wide::<1, 16>(p);
}
/// Two adjacent features per thread (block 384 = 128 feature pairs x 3 head
/// quads): one 8-byte V load and the same weight broadcast feed both; every
/// output keeps its ascending slot chain (bytes == values_wide).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_wide2(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_values_wide::<2, 16>(p);
}
/// Four adjacent features per thread (block 192 = 64 feature quads x 3 head
/// quads; one 16-byte V load feeds four outputs per head); same bytes.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_wide4(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_values_wide::<4, 16>(p);
}
/// values_wide4 with 12-slot tiles (25 KB shared, three CTAs per SM); same bytes.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_wide4t(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_values_wide::<4, 12>(p);
}
#[inline(always)]
unsafe fn attention_values_wide<const FPT: usize, const TILE: usize>(p: crate::kernel_params::QsaAttention) {
    use crate::ptx::{commit, copy4, copy16, load, load4, shared, sync, wait, wait_one};
    const HEADS: usize = crate::kernel_params::QSA_VALUE_HEADS;
    let stage = TILE * 256 * 4 + HEADS * TILE * 4;
    let block = core::arch::nvptx::_block_idx_x() as usize;
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let threads = core::arch::nvptx::_block_dim_x() as usize;
    let (token, key_head) = (block % p.tokens as usize, block / p.tokens as usize);
    let first = ((key_head + p.key_start as usize) * p.queries_per_key as usize).max(p.query_start as usize);
    let end = ((key_head + p.key_start as usize + 1) * p.queries_per_key as usize)
        .min((p.query_start + p.query_heads) as usize);
    let group = end - first;
    let first_row = token * p.query_heads as usize + first - p.query_start as usize;
    // Warps are 32 consecutive features of one head quad: V reads are
    // conflict-free rows, weight reads are warp-wide broadcasts.
    let (feature, quad) = if FPT == 4 { (tid % 64 * 4, tid / 64) } else if FPT == 2 { (tid % 128 * 2, tid / 128) } else { (tid % 256, tid / 256) };
    let length = p.past as usize + token + 1;
    let visible = selected_length(length);
    let values = (p.values as *const f32).add(key_head * 256);
    let stride = p.key_heads as usize * 256;
    // 12-slot tiles (values_wide4t) also stage the token's IDs after the two stages.
    let base = if TILE == 12 { shared::<{ 2 * (12 * 1024 + 12 * 12 * 4) + ID_TILE }>() } else { shared::<{ 2 * (16 * 1024 + 12 * 16 * 4) }>() };
    let id_tile = base + (2 * stage) as u32;
    let issue = |tile: usize| {
        let (start, buffer) = (tile * TILE, base + (tile % 2 * stage) as u32);
        let count = TILE.min(visible - start);
        let mut chunk = tid;
        while chunk < count * 64 {
            let pos = if TILE == 12 { tile_position(id_tile, length / 4, start + chunk / 64) }
                else { selected_position(p.ids as *const u32, length / 4, token, p.blocks, start + chunk / 64) };
            copy16(buffer + (chunk * 16) as u32, values.add(pos * stride + chunk % 64 * 4).cast());
            chunk += threads;
        }
        let weights = buffer + (TILE * 256 * 4) as u32;
        let mut chunk = tid;
        while chunk < group * count {
            let (h, i) = (chunk / count, chunk % count);
            let source = (p.scores as *const f32).add((first_row + h) * p.stride as usize + start + i);
            copy4(weights + ((h * TILE + i) * 4) as u32, source.cast());
            chunk += threads;
        }
        commit();
    };
    let tiles = visible.div_ceil(TILE);
    if TILE == 12 {
        stage_ids(id_tile, p.ids as *const u32, length / 4, token, p.blocks, tid, threads);
        commit();
        wait();
        sync();
    }
    issue(0);
    let mut total = [0.0f32; 4];
    let mut total2 = [0.0f32; 4];
    let mut total3 = [0.0f32; 4];
    let mut total4 = [0.0f32; 4];
    let live = [0, 1, 2, 3].map(|j| 4 * quad + j < group);
    for tile in 0..tiles {
        if tile + 1 < tiles { issue(tile + 1); wait_one(); } else { wait(); }
        sync();
        let buffer = base + (tile % 2 * stage) as u32;
        let weights = buffer + ((TILE * 256 + 4 * quad * TILE) * 4) as u32;
        let count = TILE.min(visible - tile * TILE);
        let value = |i: usize| load(buffer + ((i * 256 + feature) * 4) as u32);
        let value2 = |i: usize| crate::ptx::load2(buffer + ((i * 256 + feature) * 4) as u32).map(f32::from_bits);
        let value4 = |i: usize| load4(buffer + ((i * 256 + feature) * 4) as u32).map(f32::from_bits);
        let mut i = 0;
        while i + 4 <= count {
            if FPT == 4 {
                let v = [value4(i), value4(i + 1), value4(i + 2), value4(i + 3)];
                macro_rules! head { ($j:literal) => { if live[$j] {
                    let w = load4(weights + (($j * TILE + i) * 4) as u32).map(f32::from_bits);
                    total[$j] += w[0] * v[0][0]; total[$j] += w[1] * v[1][0]; total[$j] += w[2] * v[2][0]; total[$j] += w[3] * v[3][0];
                    total2[$j] += w[0] * v[0][1]; total2[$j] += w[1] * v[1][1]; total2[$j] += w[2] * v[2][1]; total2[$j] += w[3] * v[3][1];
                    total3[$j] += w[0] * v[0][2]; total3[$j] += w[1] * v[1][2]; total3[$j] += w[2] * v[2][2]; total3[$j] += w[3] * v[3][2];
                    total4[$j] += w[0] * v[0][3]; total4[$j] += w[1] * v[1][3]; total4[$j] += w[2] * v[2][3]; total4[$j] += w[3] * v[3][3];
                } }; }
                head!(0); head!(1); head!(2); head!(3);
            } else if FPT == 2 {
                let v = [value2(i), value2(i + 1), value2(i + 2), value2(i + 3)];
                macro_rules! head { ($j:literal) => { if live[$j] {
                    let w = load4(weights + (($j * TILE + i) * 4) as u32).map(f32::from_bits);
                    total[$j] += w[0] * v[0][0]; total[$j] += w[1] * v[1][0];
                    total[$j] += w[2] * v[2][0]; total[$j] += w[3] * v[3][0];
                    total2[$j] += w[0] * v[0][1]; total2[$j] += w[1] * v[1][1];
                    total2[$j] += w[2] * v[2][1]; total2[$j] += w[3] * v[3][1];
                } }; }
                head!(0); head!(1); head!(2); head!(3);
            } else {
                let v = [value(i), value(i + 1), value(i + 2), value(i + 3)];
                macro_rules! head { ($j:literal) => { if live[$j] {
                    let w = load4(weights + (($j * TILE + i) * 4) as u32).map(f32::from_bits);
                    total[$j] += w[0] * v[0]; total[$j] += w[1] * v[1];
                    total[$j] += w[2] * v[2]; total[$j] += w[3] * v[3];
                } }; }
                head!(0); head!(1); head!(2); head!(3);
            }
            i += 4;
        }
        while i < count {
            if FPT == 4 {
                let v = value4(i);
                macro_rules! head { ($j:literal) => { if live[$j] {
                    let w = load(weights + (($j * TILE + i) * 4) as u32);
                    total[$j] += w * v[0]; total2[$j] += w * v[1]; total3[$j] += w * v[2]; total4[$j] += w * v[3];
                } }; }
                head!(0); head!(1); head!(2); head!(3);
            } else if FPT == 2 {
                let v = value2(i);
                macro_rules! head { ($j:literal) => { if live[$j] {
                    let w = load(weights + (($j * TILE + i) * 4) as u32);
                    total[$j] += w * v[0];
                    total2[$j] += w * v[1];
                } }; }
                head!(0); head!(1); head!(2); head!(3);
            } else {
                let v = value(i);
                macro_rules! head { ($j:literal) => { if live[$j] {
                    total[$j] += load(weights + (($j * TILE + i) * 4) as u32) * v;
                } }; }
                head!(0); head!(1); head!(2); head!(3);
            }
            i += 1;
        }
        sync();
    }
    let mut j = 0;
    while j < 4 {
        if live[j] {
            let row = first_row + 4 * quad + j;
            let denominator = *(p.denominators as *const f32).add(row);
            let gate = output_gate(*(p.query_gate as *const f32).add(row * 512 + 256 + feature), 0);
            *(p.output as *mut f32).add(row * 256 + feature) = (total[j] / denominator) * gate;
            if FPT == 4 {
                let t = [total2[j], total3[j], total4[j]];
                for k in 0..3 {
                    let g = output_gate(*(p.query_gate as *const f32).add(row * 512 + 256 + feature + 1 + k), 0);
                    *(p.output as *mut f32).add(row * 256 + feature + 1 + k) = (t[k] / denominator) * g;
                }
            }
            if FPT == 2 {
                let gate2 = output_gate(*(p.query_gate as *const f32).add(row * 512 + 256 + feature + 1), 0);
                *(p.output as *mut f32).add(row * 256 + feature + 1) = (total2[j] / denominator) * gate2;
            }
        }
        j += 1;
    }
}

/// Indexed attention, step 1 of 3, for every query count: scores of every
/// visible key in ascending selected-slot order. A block is one group of
/// QSA_SCORE_HEADS globally aligned query heads of one token; each warp scores
/// QSA_SCORE_KEYS keys for every head of the group, sharing the key loads. Each
/// dot keeps the single-head feature order and warp reduction. IDs reuse the
/// selector's dead [tokens,blocks] scores.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_scores::<{ crate::kernel_params::QSA_SCORE_KEYS }>(p);
}
/// Prompt tiles (64 or more tokens): eight keys per warp share each query
/// load. Every score is the same dot in the same order as
/// flash_qsa_attention_scores (bytes equal); grid-y = ceil(slots / 32).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores_k8(mut p: crate::kernel_params::QsaAttention, cursor: *const u32) {
    if !cursor.is_null() { p.past = *cursor; p.blocks = *cursor.add(1); }
    attention_scores::<8>(p);
}
#[inline(always)]
unsafe fn attention_scores<const KEYS: usize>(p: crate::kernel_params::QsaAttention) {
    const HEADS: usize = crate::kernel_params::QSA_SCORE_HEADS;
    let lane = index() % 32;
    let groups = (p.query_start as usize % HEADS + p.query_heads as usize).div_ceil(HEADS);
    let group = core::arch::nvptx::_block_idx_x() as usize;
    let slot = (core::arch::nvptx::_block_idx_y() as usize * 4 + core::arch::nvptx::_thread_idx_x() as usize / 32) * KEYS;
    if group >= groups * p.tokens as usize { return; }
    let token = group / groups;
    let first = (p.query_start as usize / HEADS + group % groups) * HEADS;
    let length = p.past as usize + token + 1;
    let complete = length / 4;
    let visible = selected_length(length);
    if slot >= visible { return; }
    let positions: [usize; KEYS] = core::array::from_fn(|i|
        if slot + i < visible { selected_position(p.ids as *const u32, complete, token, p.blocks, slot + i) } else { 0 });
    // HEADS divides queries_per_key, so even a partial local group shares K.
    let key_head = first / p.queries_per_key as usize - p.key_start as usize;
    let keys: [[f32; 8]; KEYS] = core::array::from_fn(|i| if slot + i < visible {
        row_features((p.keys as *const f32).add((positions[i] * p.key_heads as usize + key_head) * 256), lane, 0)
    } else { [0.0; 8] });
    macro_rules! write_score { ($query:expr, $row:expr, $i:literal) => { if $i < KEYS && slot + $i < visible {
        let value = score($query, keys[$i], 0);
        if lane == 0 { *(p.scores as *mut f32).add($row * p.stride as usize + slot + $i) = value; }
    } }; }
    macro_rules! emit { ($i:literal) => { if $i < HEADS && first + $i >= p.query_start as usize
        && first + $i < (p.query_start + p.query_heads) as usize {
        let row = token * p.query_heads as usize + first + $i - p.query_start as usize;
        let query = row_features((p.query as *const f32).add(row * 256), lane, 0);
        write_score!(query, row, 0); write_score!(query, row, 1);
        write_score!(query, row, 2); write_score!(query, row, 3);
        write_score!(query, row, 4); write_score!(query, row, 5);
        write_score!(query, row, 6); write_score!(query, row, 7);
    } }; }
    emit!(0); emit!(1); emit!(2); emit!(3);
    emit!(4); emit!(5); emit!(6); emit!(7);
    emit!(8); emit!(9); emit!(10); emit!(11);
}

// A session chooses only pointers and position, never a reduction. Inactive
// planes return uniformly before reading the descriptor or reaching a barrier.
#[inline(always)]
pub(crate) unsafe fn batch_row(p: crate::kernel_params::QsaBatch, row: usize)
    -> Option<crate::kernel_params::QsaBatchRow> {
    if row >= p.count as usize { return None; }
    let record=*(p.rows as *const crate::kernel_params::QsaBatchRow).add(row);
    if record.keys==0 { None } else { Some(record) }
}
#[inline(always)]
pub(crate) unsafe fn batch_index(r: crate::kernel_params::QsaBatchRow) -> crate::kernel_params::QsaIndex {
    crate::kernel_params::QsaIndex { query: r.index_query, pooled: r.pooled, scores: r.ids,
        tokens: 1, past: r.past, blocks: (r.past + 1) / 4, padding: 0 }
}
#[inline(always)]
pub(crate) unsafe fn batch_attention(p: crate::kernel_params::QsaBatch, r: crate::kernel_params::QsaBatchRow, row: usize)
    -> crate::kernel_params::QsaAttention {
    crate::kernel_params::QsaAttention {
        query: r.query, query_gate: p.query_gate + row as u64 * p.query_heads as u64 * 512 * 4,
        keys: r.keys, values: r.values, ids: r.ids,
        output: p.output + row as u64 * p.query_heads as u64 * 256 * 4, scores: r.scores,
        query_heads: p.query_heads, key_heads: p.key_heads, query_start: p.query_start, key_start: p.key_start,
        tokens: 1, past: r.past, blocks: (r.past + 1) / 4, queries_per_key: 12,
        stride: r.stride, padding: 0, denominators: r.denominators,
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_index_prepare_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    index_prepare((p.index as *const f32).add(row * 640), (r.norm as *const f32).add(512),
        r.index_query as *mut f32, r.raw_index as *mut f32, 1, r.past, 0, index());
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_prepare_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    prepare((p.query_gate as *const f32).add(row * p.query_heads as usize * 512),
        (p.key as *const f32).add(row * p.key_heads as usize * 256),
        (p.value as *const f32).add(row * p.key_heads as usize * 256),
        r.norm as *const f32, (r.norm as *const f32).add(256), r.query as *mut f32,
        r.keys as *mut f32, r.values as *mut f32, p.query_heads, p.key_heads, 1, r.past, 0,
        core::ptr::null(), 0, 1e-6, core::ptr::null(), index());
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_index_pool_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    if r.past % 4 != 3 { return; }
    let first = r.past / 4;
    let offset = (first as u64) * 4 * 128 * 4;
    index_pool(crate::kernel_params::QsaIndexPool { raw_keys: r.raw_index + offset, weight: r.norm + 640 * 4,
        pooled: r.pooled + offset, blocks: 1, position_offset: first * 4 });
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_scores_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    index_scores(batch_index(r));
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_select_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    index_select(batch_index(r));
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_scores_batch(p: crate::kernel_params::QsaBatch) {
    // Grid-y remains selected-slot planes, preserving the shared-head kernel.
    let row = core::arch::nvptx::_block_idx_z() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    attention_scores::<{ crate::kernel_params::QSA_SCORE_KEYS }>(batch_attention(p, r, row));
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_softmax_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    attention_softmax(batch_attention(p, r, row));
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_qsa_attention_values_batch(p: crate::kernel_params::QsaBatch) {
    let row = core::arch::nvptx::_block_idx_y() as usize;
    let Some(r) = batch_row(p, row) else { return; };
    attention_values(batch_attention(p, r, row));
}

#[inline(always)]
unsafe fn attention<const CACHE_SCORES: bool>(
    q: *const f32, qg: *const f32, k: *const f32, v: *const f32,
    mask: *const u32, y: *mut f32, qheads: u32, kheads: u32, qstart: u32, kstart: u32,
    tokens: u32, past: u32, blocks: u32, queries_per_key: u32, precision: u32, swish: u32,
    replay_position: *const u32, thread: usize, saved_scores: *mut f32,
) {
    let past = if replay_position.is_null() { past } else { past + *replay_position };
    let row = thread / 32;
    let lane = thread % 32;
    if row >= qheads as usize * tokens as usize { return; }
    let token = row / qheads as usize;
    let head = row % qheads as usize;
    let key_head = (head + qstart as usize) / queries_per_key as usize - kstart as usize;
    let length = past as usize + token + 1;
    let complete = length / 4;
    let query = row_features(q.add(row * 256), lane, 0);
    // The short dense path computes each score once. Shared storage is
    // private to each warp; max, denominator and value sums keep their order.
    let cache_scores = CACHE_SCORES && precision != 0 && mask.is_null() && length <= 512;
    let global_scores = !saved_scores.is_null() && length > 512;
    let scores = if CACHE_SCORES { crate::ptx::shared::<8192>()
        + (core::arch::nvptx::_thread_idx_x() as u32 / 32 * 2048) } else { 0 };
    let mut highest = f32::NEG_INFINITY;
    let mut denominator = 0.0;
    let mut pos = 0;
    if global_scores {
        // Coalesced score reads; only the maximum is a parallel reduction.
        // Each lane still takes its maxima in position order, eight loads in flight.
        let mut at = lane;
        while at + 224 < length {
            let s = row_features(saved_scores.add(at), 0, 0);
            highest = highest.max(s[0]).max(s[1]).max(s[2]).max(s[3]).max(s[4]).max(s[5]).max(s[6]).max(s[7]);
            at += 256;
        }
        while at < length { highest = highest.max(*saved_scores.add(at)); at += 32; }
        highest = crate::layer::maximum(highest);
        // The denominator still adds positions 0..length in the solo order;
        // the next 32 scores load while this group is added.
        let mut next = if lane < length { *saved_scores.add(lane) } else { 0.0 };
        while pos < length {
            let current = next;
            if pos + 32 + lane < length { next = *saved_scores.add(pos + 32 + lane); }
            let probability = if pos + lane < length { exp(current - highest) } else { 0.0 };
            for source in 0..32 {
                let value = f32::from_bits(shuffle_xor(u32::MAX, probability.to_bits(), (lane ^ source) as u32, 31));
                if pos + source < length { denominator += value; }
            }
            if pos + lane < length { *saved_scores.add(pos + lane) = probability; }
            pos += 32;
        }
        // Leave the solo walk's exact weights, rounded(probability / denominator),
        // for flash_dense_values. Each lane rewrites only the positions it wrote.
        let mut at = lane;
        while at + 224 < length {
            let p = row_features(saved_scores.add(at), 0, 0);
            for i in 0..8 { *saved_scores.add(at + i * 32) = rounded(p[i] / denominator); }
            at += 256;
        }
        while at < length { *saved_scores.add(at) = rounded(*saved_scores.add(at) / denominator); at += 32; }
        return;
    } else {
        while pos < length {
            if mask.is_null() || complete <= 512 || pos >= complete * 4 || *mask.add(token * blocks as usize + pos / 4) != 0 {
                let offset = (pos * kheads as usize + key_head) * 256;
                let key = if precision == 0 { k.add(offset) } else { k.cast::<u16>().add(offset).cast::<f32>() };
                let score = attention_score(query, key, lane, precision);
                highest = highest.max(score);
                if cache_scores && lane == 0 { crate::ptx::store(scores + (pos * 4) as u32, score); }
            }
            pos += 1;
        }
        if cache_scores { crate::ptx::warp_sync(); }
        if precision != 0 {
            pos = 0;
            while pos < length {
                let offset = (pos * kheads as usize + key_head) * 256;
                let key = k.cast::<u16>().add(offset).cast::<f32>();
                let score = if cache_scores { crate::ptx::load(scores + (pos * 4) as u32) }
                    else { attention_score(query, key, lane, precision) };
                let probability = exp(score - highest);
                denominator += probability;
                if cache_scores && lane == 0 { crate::ptx::store(scores + (pos * 4) as u32, probability); }
                pos += 1;
            }
        }
    }
    if cache_scores { crate::ptx::warp_sync(); }
    let mut values = [0.0f32; 8];
    pos = 0;
    while pos < length {
        if mask.is_null() || complete <= 512 || pos >= complete * 4 || *mask.add(token * blocks as usize + pos / 4) != 0 {
            let offset = (pos * kheads as usize + key_head) * 256;
            let key = if precision == 0 { k.add(offset) } else { k.cast::<u16>().add(offset).cast::<f32>() };
            let probability = if cache_scores { crate::ptx::load(scores + (pos * 4) as u32) }
                else { exp(attention_score(query, key, lane, precision) - highest) };
            let product_probability = if precision == 0 { probability } else { rounded(probability / denominator) };
            if precision == 0 { denominator += probability; }
            let mut part = 0;
            while part < 8 {
                values[part] += product_probability * cached(v, offset + lane + part * 32, precision);
                part += 1;
            }
        }
        pos += 1;
    }
    let gate = qg.add(row * 512 + 256);
    let mut part = 0;
    while part < 8 {
        let feature = lane + part * 32;
        let gate = output_gate(*gate.add(feature), swish);
        let value = if precision == 0 { values[part] / denominator } else { values[part] };
        let value = if precision == 0 { value * gate } else { gated_bf16(value, gate) };
        *y.add(row * 256 + feature) = value;
        part += 1;
    }
}

include!("flash_qsa_dense.rs");
