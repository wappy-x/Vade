//! Grouped decode/verify attention: one K/V read, online softmax, bounded summaries.
use crate::{layer::{sum, maximum, sqrt}, ptx};
// The scalar formats need 4160 bytes; BF16 KV needs one canonical split's tiles.
pub(crate) const STREAM_SHARED: usize = MMA_SHARED;

use core::arch::nvptx::{_block_idx_x, _block_idx_y, _block_idx_z, _thread_idx_x};

#[inline(always)]
unsafe fn load<const FORMAT: u32>(p: *const f32, i: usize) -> f32 {
    crate::layer::batch::cache_load(p, i, FORMAT)
}

// All four lane components belong to one head and use the same I16 scale.
#[inline(always)]
unsafe fn load_key<const FORMAT: u32>(p: *const f32, head: usize, lane: usize) -> [f32; 4] {
    if FORMAT == 4 {
        let block = p.add(head / 8 * 520);
        let scale = *block.add(512 + head % 8);
        let values = block.cast::<i16>().add(head % 8 * 128 + lane);
        [*values as f32 * scale, *values.add(32) as f32 * scale,
            *values.add(64) as f32 * scale, *values.add(96) as f32 * scale]
    } else {
        let i = head * 128 + lane;
        [load::<FORMAT>(p, i), load::<FORMAT>(p, i + 32),
            load::<FORMAT>(p, i + 64), load::<FORMAT>(p, i + 96)]
    }
}

// Four query heads share every global K/V load. A CTA streams one split;
// a single split can write final output; multi-session callers retain summaries.
// Split boundaries never depend on verification width, so a row uses exactly
// the decode arithmetic order. BF16 KV uses the canonical MMA split.
#[inline(always)]
pub(crate) unsafe fn stream<const FORMAT: u32, const FINAL: bool, const GROUPS: usize>(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, length: usize, part: usize, parts: u32, span: u32, sh: u32) {
    if FORMAT == 1 {
        stream_mma::<FINAL>(q, k, v, out, kv, length, part, parts, span, 4 * GROUPS, None, sh);
    } else {
        stream_group::<FORMAT, FINAL, 4, GROUPS>(q, k, v, out, kv, length, part, parts, span, sh);
    }
}
#[inline(always)]
pub(crate) unsafe fn stream_spans<const GROUPS: usize>(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, length: usize, part: usize, parts: u32, span: u32,
    prefix_k: *const f32, prefix_v: *const f32, base: usize, sh: u32) {
    let prefix = if base == 0 { None } else { Some((prefix_k.cast::<u16>(), prefix_v.cast::<u16>(), base)) };
    stream_mma::<false>(q, k, v, out, kv, length, part, parts, span, 4 * GROUPS, prefix, sh);
}

#[inline(always)]
pub(crate) unsafe fn stream_mma<const FINAL: bool>(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, length: usize, part: usize, parts: u32, span: u32,
    heads: usize, prefix: Option<(*const u16, *const u16, usize)>, sh: u32) {
    let group = _block_idx_y() as usize;
    let (parts, begin) = (parts as usize, part * span as usize);
    let finalize = FINAL && parts == 1;
    // BF16 launches pack all query heads of a KV head into one MMA tile.
    // Mixed-format cohorts retain the larger scalar grid; surplus BF16 CTAs exit.
    if group >= kv as usize { return; }
    let kh = group;
    split_mma(q.add(group * heads * 128), 0, heads, 1, k.cast(), v.cast(), prefix, kv as usize, kh,
        begin, length.min(begin + span as usize), finalize,
        |_, h| if finalize { out.add((group * heads + h) * 128) } else { out.add(((group * heads + h) * parts + part) * 132) }, sh);
}

/// Shared memory of one canonical split: a swizzled 64-token K tile and V tile.
pub(crate) const MMA_SHARED: usize = 32768;

#[inline(always)]
fn bf16x2(lo: f32, hi: f32) -> u32 {
    let packed: u32;
    unsafe { core::arch::asm!("cvt.rn.bf16x2.f32 {p}, {b}, {a};", p = out(reg32) packed,
        a = in(reg32) lo, b = in(reg32) hi, options(pure, nomem, nostack)); }
    packed
}
// Constant indices keep each ldmatrix swizzle an immediate XOR and offset.
macro_rules! unroll {
    ($count:expr, $n:ident, $body:block) => {{
        { const $n: usize = 0; if $n < $count $body } { const $n: usize = 1; if $n < $count $body }
        { const $n: usize = 2; if $n < $count $body } { const $n: usize = 3; if $n < $count $body }
        { const $n: usize = 4; if $n < $count $body } { const $n: usize = 5; if $n < $count $body }
        { const $n: usize = 6; if $n < $count $body } { const $n: usize = 7; if $n < $count $body }
    }};
}
#[inline(always)]
unsafe fn matrix4<const TRANS: bool, const XOR: u32, const OFFSET: u32>(base: u32, lane: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    if TRANS {
        core::arch::asm!("{{ .reg .u32 p; xor.b32 p, {lane}, {xor}; add.u32 p, p, {base}; ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{a}, {b}, {c}, {d}}}, [p+{offset}]; }}",
            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
            lane = in(reg32) lane, base = in(reg32) base, xor = const XOR, offset = const OFFSET);
    } else {
        core::arch::asm!("{{ .reg .u32 p; xor.b32 p, {lane}, {xor}; add.u32 p, p, {base}; ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a}, {b}, {c}, {d}}}, [p+{offset}]; }}",
            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
            lane = in(reg32) lane, base = in(reg32) base, xor = const XOR, offset = const OFFSET);
    }
    [a, b, c, d]
}
#[inline(always)]
unsafe fn quad(value: f32, distance: u32) -> f32 {
    let other: u32;
    core::arch::asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;", o = out(reg32) other, v = in(reg32) value.to_bits(), d = in(reg32) distance);
    f32::from_bits(other)
}
// Token-major 16-byte chunks, XOR-swizzled by token so ldmatrix rows hit distinct banks.
// Tokens at or past `end` are zero, so masked probabilities multiply exact zeros.
#[inline(always)]
unsafe fn stage_tile(dst: u32, src: *const u16, prefix: Option<(*const u16, usize)>,
    tile: usize, end: usize, kv: usize, kh: usize, tid: usize, threads: usize) {
    let mut i = tid;
    while i < 64 * 16 {
        let (key, chunk) = (i / 16, i % 16);
        let at = dst + (key * 256 + (chunk ^ (key & 7)) * 16) as u32;
        let t = tile + key;
        if t < end {
            let (source, token) = match prefix {
                Some((shared, base)) if t < base => (shared, t),
                Some((_, base)) => (src, t - base),
                None => (src, t),
            };
            ptx::copy16(at, source.add((token * kv + kh) * 128 + chunk * 8).cast());
        } else {
            core::arch::asm!("st.shared.v4.u32 [{p}], {{0, 0, 0, 0}};", p = in(reg32) at);
        }
        i += threads;
    }
}

/// The canonical BF16 decode attention split (order in crates/gpu/README.md,
/// "Decode attention order"). M row m = row * heads + head; warp w owns M rows
/// 16w..16w+15 and every key of each 64-token tile, so a row's bits never
/// depend on the other rows.
/// All threads stage K/V; a K tile reloads during PV and a V tile during QK.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn split_mma(q: *const f32, q_stride: usize, heads: usize, rows: usize,
    k: *const u16, v: *const u16, prefix: Option<(*const u16, *const u16, usize)>, kv: usize, kh: usize,
    begin: usize, end: usize, finalize: bool, out: impl Fn(usize, usize) -> *mut f32, sh: u32) {
    use crate::canonical::{madd, ZERO};
    let tid = _thread_idx_x() as usize;
    let threads = core::arch::nvptx::_block_dim_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let m_rows = rows * heads;
    let compute = warp * 16 < m_rows;
    let first = warp * 16 + lane / 4;
    let col = lane % 4 * 2;
    let (kp, vp) = (prefix.map(|(pk, _, b)| (pk, b)), prefix.map(|(_, pv, b)| (pv, b)));
    let (ksh, vsh) = (sh, sh + 16384);
    // Lane parts of the swizzled ldmatrix addresses (see stage_tile).
    let k_lane = (lane % 8 * 256 + ((lane / 8) ^ (lane % 8)) * 16) as u32;
    let v_lane = ((lane / 8 % 2 * 8 + lane % 8) * 256 + ((lane / 16) ^ (lane % 8)) * 16) as u32;
    let load_row = |m: usize| -> [[u32; 2]; 8] {
        let mut pairs = [[0u32; 2]; 8];
        if compute && m < m_rows {
            let row = q.add(m / heads * q_stride + m % heads * 128 + col);
            let mut s = 0;
            while s < 8 {
                let x = row.add(s * 16);
                pairs[s] = [bf16x2(*x, *x.add(1)), bf16x2(*x.add(8), *x.add(9))];
                s += 1;
            }
        }
        pairs
    };
    let (upper, lower) = (load_row(first), load_row(first + 8));
    let mut qa = [[0u32; 4]; 8];
    let mut s = 0;
    while s < 8 { qa[s] = [upper[s][0], lower[s][0], upper[s][1], lower[s][1]]; s += 1; }
    let mut o = [ZERO; 16];
    let mut peak = [f32::NEG_INFINITY; 2];
    let mut denominator = [0.0f32; 2];
    stage_tile(ksh, k, kp, begin, end, kv, kh, tid, threads);
    ptx::commit();
    stage_tile(vsh, v, vp, begin, end, kv, kh, tid, threads);
    ptx::commit();
    let mut tile = begin;
    while tile < end {
        let next = tile + 64;
        ptx::wait_one();
        ptx::sync();
        let mut p = [ZERO; 8];
        let mut correction = [0.0f32; 2];
        if compute {
            unroll!(8, J, {
                let mut s = ZERO;
                unroll!(4, P, {
                    let b = matrix4::<false, { (P * 64) as u32 }, { (J * 2048) as u32 }>(ksh, k_lane);
                    s = madd(qa[2 * P], [b[0], b[1]], s);
                    s = madd(qa[2 * P + 1], [b[2], b[3]], s);
                });
                let key = tile + J * 8 + col;
                let (even, odd) = (key < end, key + 1 < end);
                s.a = if even { s.a / sqrt(128.0) } else { f32::NEG_INFINITY };
                s.b = if odd { s.b / sqrt(128.0) } else { f32::NEG_INFINITY };
                s.c = if even { s.c / sqrt(128.0) } else { f32::NEG_INFINITY };
                s.d = if odd { s.d / sqrt(128.0) } else { f32::NEG_INFINITY };
                p[J] = s;
            });
            let mut top = peak;
            let mut j = 0;
            while j < 8 {
                top[0] = top[0].max(p[j].a).max(p[j].b);
                top[1] = top[1].max(p[j].c).max(p[j].d);
                j += 1;
            }
            let mut h = 0;
            while h < 2 {
                top[h] = top[h].max(quad(top[h], 1));
                top[h] = top[h].max(quad(top[h], 2));
                correction[h] = ptx::exp(peak[h] - top[h]);
                h += 1;
            }
            let mut total = [0.0f32; 2];
            j = 0;
            while j < 8 {
                p[j].a = ptx::exp(p[j].a - top[0]); total[0] += p[j].a;
                p[j].b = ptx::exp(p[j].b - top[0]); total[0] += p[j].b;
                p[j].c = ptx::exp(p[j].c - top[1]); total[1] += p[j].c;
                p[j].d = ptx::exp(p[j].d - top[1]); total[1] += p[j].d;
                j += 1;
            }
            h = 0;
            while h < 2 {
                total[h] += quad(total[h], 1);
                total[h] += quad(total[h], 2);
                denominator[h] = denominator[h] * correction[h] + total[h];
                peak[h] = top[h];
                h += 1;
            }
        }
        ptx::sync();
        if next < end { stage_tile(ksh, k, kp, next, end, kv, kh, tid, threads); }
        ptx::commit();
        ptx::wait_one();
        ptx::sync();
        if compute {
            let mut n = 0;
            while n < 16 {
                o[n].a *= correction[0]; o[n].b *= correction[0];
                o[n].c *= correction[1]; o[n].d *= correction[1];
                n += 1;
            }
            unroll!(4, B, {
                let (x, y) = (p[2 * B], p[2 * B + 1]);
                let a = [bf16x2(x.a, x.b), bf16x2(x.c, x.d), bf16x2(y.a, y.b), bf16x2(y.c, y.d)];
                unroll!(8, P, {
                    let b = matrix4::<true, { (P * 32) as u32 }, { (B * 4096) as u32 }>(vsh, v_lane);
                    o[2 * P] = madd(a, [b[0], b[1]], o[2 * P]);
                    o[2 * P + 1] = madd(a, [b[2], b[3]], o[2 * P + 1]);
                });
            });
        }
        ptx::sync();
        if next < end { stage_tile(vsh, v, vp, next, end, kv, kh, tid, threads); }
        ptx::commit();
        tile = next;
    }
    ptx::wait();
    if !compute { return; }
    let mut h = 0;
    while h < 2 {
        let m = first + h * 8;
        if m < m_rows {
            let target = out(m / heads, m % heads);
            let mut n = 0;
            while n < 16 {
                let (x, y) = if h == 0 { (o[n].a, o[n].b) } else { (o[n].c, o[n].d) };
                let at = target.add(n * 8 + col);
                if finalize {
                    *at = x / denominator[h];
                    *at.add(1) = y / denominator[h];
                } else {
                    *at = x;
                    *at.add(1) = y;
                }
                n += 1;
            }
            if !finalize && lane % 4 == 0 {
                *target.add(128) = peak[h];
                *target.add(129) = denominator[h];
            }
        }
        h += 1;
    }
}

// Scalar order for FP32 and scaled-I16 KV and the two-head BF16 model.
#[inline(always)]
unsafe fn stream_group<const FORMAT: u32, const FINAL: bool, const GROUP: usize, const GROUPS: usize>(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, length: usize, part: usize, parts: u32, span: u32, sh: u32) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let group = _block_idx_y() as usize;
    let kh = group / GROUPS;
    let (kv, parts) = (kv as usize, parts as usize);
    let begin = part * span as usize;
    let end = length.min(begin + span as usize);
    let tile_size = if span <= 64 {64} else {256};
    let q = q.add(group * GROUP * 128);
    let mut query = [[0.0f32; 4]; 4];
    let mut g = 0;
    while g < GROUP { let mut d = 0; while d < 4 { query[g][d] = *q.add(g * 128 + lane + d * 32); d += 1; } g += 1; }
    let mut accum = [0.0f32; 4];
    let (mut peak, mut denominator) = (f32::NEG_INFINITY, 0.0f32);
    let mut tile = begin;
    while tile < end {
        let tile_end = end.min(tile + tile_size);
        let mut t = warp;
        while t < tile_size {
            let mut dot = [0.0f32; 4];
            if tile + t < tile_end {
                let key = if FORMAT == 4 { load_key::<4>(k, (tile + t) * kv + kh, lane) } else { [0.0; 4] };
                let mut d = 0;
                while d < 4 {
                    let x = if FORMAT == 4 { key[d] } else { load::<FORMAT>(k, ((tile + t) * kv + kh) * 128 + lane + d * 32) };
                    let mut g = 0;
                    while g < GROUP { dot[g] += query[g][d] * x; g += 1; }
                    d += 1;
                }
            }
            let mut g = 0;
            while g < GROUP {
                let score = if tile + t < tile_end { sum(dot[g]) / sqrt(128.0) } else { f32::NEG_INFINITY };
                if lane == 0 { ptx::store(sh + ((g * 256 + t) * 4) as u32, score); }
                g += 1;
            }
            t += 4;
        }
        ptx::sync();
        // Active warps own the queries' online normalization states; short
        // splits skip the padded scores (exp(-inf) adds exact zeros).
        if warp < GROUP {
            let mut next_peak = peak;
            let mut i = lane;
            while i < tile_size { next_peak = next_peak.max(ptx::load(sh + ((warp * 256 + i) * 4) as u32)); i += 32; }
            next_peak = maximum(next_peak);
            let correction = ptx::exp(peak - next_peak);
            let mut tile_sum = 0.0;
            i = lane;
            while i < tile_size {
                let at = sh + ((warp * 256 + i) * 4) as u32;
                let probability = ptx::exp(ptx::load(at) - next_peak);
                ptx::store(at, probability);
                tile_sum += probability;
                i += 32;
            }
            denominator = denominator * correction + sum(tile_sum);
            peak = next_peak;
            if lane == 0 { ptx::store(sh + 4096 + (warp * 4) as u32, correction); }
        }
        ptx::sync();
        let mut g = 0;
        while g < GROUP { accum[g] *= ptx::load(sh + 4096 + (g * 4) as u32); g += 1; }
        t = tile;
        while t < tile_end {
            let value = load::<FORMAT>(v, (t * kv + kh) * 128 + tid);
            stream_accumulate::<GROUP>(&mut accum, sh, t - tile, value);
            t += 1;
        }
        ptx::sync();
        tile += tile_size;
    }
    if FINAL && parts == 1 {
        // Reproduce a one-summary merge, including its additions to zero.
        // Each warp owns one head's scale and denominator; all threads own
        // one output feature for each of the four heads.
        let mut merged = 0.0;
        if lane == 0 && warp < GROUP {
            let scale = ptx::exp(peak - peak);
            ptx::store(sh + 4096 + (warp * 4) as u32, scale);
            merged += denominator * scale;
        }
        merged = sum(merged);
        if lane == 0 && warp < GROUP {
            let mut total = 0.0;
            total += merged;
            for _ in 0..3 { total += 0.0; }
            ptx::store(sh + 4112 + (warp * 4) as u32, total);
        }
        ptx::sync();
        let mut g = 0;
        while g < GROUP {
            let mut value = 0.0;
            value += accum[g] * ptx::load(sh + 4096 + (g * 4) as u32);
            *out.add((group * GROUP + g) * 128 + tid) =
                value / ptx::load(sh + 4112 + (g * 4) as u32);
            g += 1;
        }
        return;
    }
    if lane == 0 && warp < GROUP {
        let at = ((group * GROUP + warp) * parts + part) * 132;
        *out.add(at + 128) = peak;
        *out.add(at + 129) = denominator;
    }
    let mut g = 0;
    while g < GROUP { *out.add(((group * GROUP + g) * parts + part) * 132 + tid) = accum[g]; g += 1; }
}

#[inline(always)]
unsafe fn stream_accumulate<const GROUP: usize>(accum: &mut [f32; 4], sh: u32, token: usize, value: f32) {
    let mut g = 0;
    while g < GROUP {
        accum[g] += ptx::load(sh + ((g * 256 + token) * 4) as u32) * value;
        g += 1;
    }
}

/// Qwen3-0.6B: two query heads share BF16 K/V. Independent 64-key ranges
/// expose more CTAs and bound the serial value loop as the prefix grows.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_attention_gqa2(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, base: u32, parts: u32) {
    stream_group::<1, true, 2, 1>(q, k, v, out, kv, base as usize+1, _block_idx_x() as usize, parts, 64, ptx::shared::<4160>());
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn graph_decode_attention_gqa2(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, base: u32, parts: u32, replay_position: *const u32) {
    let base = base + if replay_position.is_null() { 0 } else { *replay_position };
    stream_group::<1, true, 2, 1>(q, k, v, out, kv, base as usize+1, _block_idx_x() as usize, parts, 64, ptx::shared::<4160>());
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_attention_stream(args: crate::kernel_params::DecodeStream) {
    stream_args::<1>(args, core::ptr::null());
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn graph_decode_attention_stream(args: crate::kernel_params::DecodeStream, replay_position: *const u32) {
    stream_args::<1>(args, replay_position);
}

/// BF16: all eight query heads share one CTA and one K/V tile.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_attention_stream_gqa8(args: crate::kernel_params::DecodeStream) {
    stream_args::<2>(args, core::ptr::null());
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn graph_decode_attention_stream_gqa8(args: crate::kernel_params::DecodeStream, replay_position: *const u32) {
    stream_args::<2>(args, replay_position);
}

#[inline(always)]
unsafe fn stream_args<const GROUPS: usize>(args: crate::kernel_params::DecodeStream, replay_position: *const u32) {
    stream_rows::<GROUPS>(args.query as *const f32, args.keys as *const f32, args.values as *const f32,
        args.output as *mut f32, args.kv_heads, args.base, args.parts, args.span, args.format, replay_position);
}

#[inline(always)]
unsafe fn stream_rows<const GROUPS: usize>(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, base: u32, parts: u32, span: u32, format: u32, replay_position: *const u32) {
    let base = base + if replay_position.is_null() { 0 } else { *replay_position };
    let row = _block_idx_z() as usize;
    let part = _block_idx_x() as usize;
    let length = base as usize + row + 1;
    let q = q.add(row * kv as usize * 4 * GROUPS * 128);
    let stride = if parts == 1 { 128 } else { parts as usize * 132 };
    let out = out.add(row * kv as usize * 4 * GROUPS * stride);
    let sh = ptx::shared::<STREAM_SHARED>();
    match format {
        1 => stream::<1, true, GROUPS>(q, k, v, out, kv, length, part, parts, span, sh),
        4 => stream::<4, true, GROUPS>(q, k, v, out, kv, length, part, parts, span, sh),
        _ => stream::<0, true, GROUPS>(q, k, v, out, kv, length, part, parts, span, sh),
    }
}

// One warp per query row handles two keys in parallel. Each half-warp folds
// the canonical distance-16 dot reduction locally, then shuffles 8,4,2,1.
// The remaining normalization and ordered value sums match stream exactly.
// Dynamic shared: rows * 4096 bytes of probabilities + 32768 bytes of K/V.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_attention_verify(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, base: u32, parts: u32, span: u32, rows: u32) {
    shared_stream(q, k, v, out, kv, base, parts, span, rows);
}

#[inline(always)]
pub(crate) unsafe fn draft_stage4(at: u32, source: *const f32, index: usize) {
    core::arch::asm!("cp.async.ca.shared.global [{p}], [{s}], 8;",
        p=in(reg32)at, s=in(reg64)source.cast::<u16>().add(index));
}
#[inline(always)]
pub(crate) unsafe fn draft_load(at: u32) -> f32 {
    let bits: u32;
    core::arch::asm!("ld.shared.u16 {x}, [{p}];", x=out(reg32)bits, p=in(reg32)at);
    f32::from_bits(bits << 16)
}

/// FP32 multi-token verification at long contexts: scalar decode reductions.
/// Raw FP32 scores become probabilities once in shared memory, before ordered PV.
#[inline(always)]
unsafe fn shared_stream(q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, kv: u32, base: u32, parts: u32, span: u32, rows: u32) {
    let tid = _thread_idx_x() as usize;
    let (lane, row) = (tid % 32, tid / 32);
    let (group, part) = (_block_idx_y() as usize, _block_idx_x() as usize);
    let kh = group;
    let (kv, parts, rows) = (kv as usize, parts as usize, rows as usize);
    let begin = part * span as usize;
    let end = (base as usize + row + 1).min(begin + span as usize);
    let all_end = (base as usize + rows).min(begin + span as usize);
    let tile_size = if span <= 64 { 64 } else { 256 };
    let cohort = core::arch::nvptx::_block_dim_x() as usize / 32;
    let stage = (cohort * 4096) as u32;
    let element_bytes = 4;
    let scores = (row * 4096) as u32;
    let qrow = if row < rows { row } else { 0 };
    let q = q.add((qrow * kv + group) * 4 * 128);
    let mut query = [[0.0f32; 8]; 4];
    let mut accum = [[0.0f32; 4]; 4];
    let mut peak = [f32::NEG_INFINITY; 4];
    let mut denominator = [0.0f32; 4];
    let mut g = 0;
    while g < 4 {
        let mut d = 0;
        while d < 4 {
            query[g][d] = *q.add(g * 128 + lane % 16 + d * 32);
            query[g][d + 4] = *q.add(g * 128 + lane % 16 + 16 + d * 32);
            d += 1;
        }
        g += 1;
    }
    let mut tile = begin;
    while tile < all_end {
        let tile_end = all_end.min(tile + tile_size);
        let mut chunk = 0;
        while chunk < tile_size {
            let mut i = tid * 4;
            while i < 64 * 128 {
                let t = tile + chunk + i / 128;
                // Rotate alternate keys to separate the two half-warp bank sets.
                let staged = i / 128 * 128 + ((i % 128 + (i / 128 % 2) * 16) % 128);
                if t < tile_end {
                    ptx::copy16(stage + (staged * element_bytes) as u32, k.add((t * kv + kh) * 128 + i % 128).cast());
                } else {
                    let mut d = 0;
                    while d < 4 {
                        ptx::store(stage + (staged * element_bytes + d * 4) as u32, 0.0); d += 1;
                    }
                }
                i += cohort * 32 * 4;
            }
            ptx::wait();
            ptx::sync();
            let mut t = lane / 16;
            let rotation = t % 2 * 16;
            while t < 64 {
                let valid = tile + chunk + t < end;
                let mut dot = [[0.0f32; 2]; 4];
                let mut d = 0;
                while d < 4 {
                    let x = ptx::load(stage + ((t * 128 + (lane % 16 + d * 32 + rotation) % 128) * element_bytes) as u32);
                    let y = ptx::load(stage + ((t * 128 + (lane % 16 + 16 + d * 32 + rotation) % 128) * element_bytes) as u32);
                    let mut g = 0;
                    while g < 4 {
                        dot[g][0] += query[g][d] * x;
                        dot[g][1] += query[g][d + 4] * y;
                        g += 1;
                    }
                    d += 1;
                }
                let mut reduced = [0.0f32; 4];
                let mut g = 0;
                while g < 4 {
                    let mut score = dot[g][0] + dot[g][1];
                    let mut distance = 8;
                    while distance != 0 {
                        let other: u32;
                        core::arch::asm!("shfl.sync.bfly.b32 {other}, {value}, {distance}, 31, -1;",
                            other = out(reg32) other, value = in(reg32) score.to_bits(), distance = in(reg32) distance);
                        score += f32::from_bits(other);
                        distance /= 2;
                    }
                    if !valid { score = f32::NEG_INFINITY; }
                    reduced[g] = score;
                    g += 1;
                }
                if lane % 16 == 0 {
                    let at = scores + ((chunk + t) * 16) as u32;
                    core::arch::asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
                        p = in(reg32) at, a = in(reg32) reduced[0].to_bits(), b = in(reg32) reduced[1].to_bits(),
                        c = in(reg32) reduced[2].to_bits(), d = in(reg32) reduced[3].to_bits());
                }
                t += 2;
            }
            ptx::sync();
            chunk += 64;
        }
        if tile < end {
            // Defer score division until all 32 lanes have independent scores.
            let mut next_peak = peak;
            let mut i = lane;
            while i < tile_size {
                let at = scores + (i * 16) as u32;
                let raw = ptx::load4(at);
                let mut scaled = [0.0f32; 4];
                let mut g = 0;
                while g < 4 {
                    scaled[g] = f32::from_bits(raw[g]) / sqrt(128.0);
                    next_peak[g] = next_peak[g].max(scaled[g]);
                    g += 1;
                }
                core::arch::asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
                    p = in(reg32) at, a = in(reg32) scaled[0].to_bits(), b = in(reg32) scaled[1].to_bits(),
                    c = in(reg32) scaled[2].to_bits(), d = in(reg32) scaled[3].to_bits());
                i += 32;
            }
            let mut correction = [0.0f32; 4];
            let mut g = 0;
            while g < 4 {
                next_peak[g] = maximum(next_peak[g]);
                correction[g] = ptx::exp(peak[g] - next_peak[g]);
                g += 1;
            }
            let mut tile_sum = [0.0f32; 4];
            i = lane;
            while i < tile_size {
                let at = scores + (i * 16) as u32;
                let raw = ptx::load4(at);
                let mut probability = [0.0f32; 4];
                let mut g = 0;
                while g < 4 {
                    probability[g] = ptx::exp(f32::from_bits(raw[g]) - next_peak[g]);
                    tile_sum[g] += probability[g];
                    g += 1;
                }
                core::arch::asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
                    p = in(reg32) at, a = in(reg32) probability[0].to_bits(), b = in(reg32) probability[1].to_bits(),
                    c = in(reg32) probability[2].to_bits(), d = in(reg32) probability[3].to_bits());
                i += 32;
            }
            g = 0;
            while g < 4 {
                denominator[g] = denominator[g] * correction[g] + sum(tile_sum[g]);
                peak[g] = next_peak[g];
                let mut d = 0;
                while d < 4 { accum[g][d] *= correction[g]; d += 1; }
                g += 1;
            }
        }
        ptx::sync();
        chunk = 0;
        while tile + chunk < tile_end {
            let mut i = tid * 4;
            while i < 64 * 128 {
                let t = tile + chunk + i / 128;
                if t < tile_end {
                    ptx::copy16(stage + (i * element_bytes) as u32, v.add((t * kv + kh) * 128 + i % 128).cast());
                } else {
                    let mut d = 0;
                    while d < 4 {
                        ptx::store(stage + (i * element_bytes + d * 4) as u32, 0.0); d += 1;
                    }
                }
                i += cohort * 32 * 4;
            }
            ptx::wait();
            ptx::sync();
            let count = end.saturating_sub(tile + chunk).min(64);
            let mut t = 0;
            while t < count {
                let raw = ptx::load4(scores + ((chunk + t) * 16) as u32);
                let probability = [f32::from_bits(raw[0]), f32::from_bits(raw[1]), f32::from_bits(raw[2]), f32::from_bits(raw[3])];
                let values = ptx::load4(stage + ((t * 128 + lane * 4) * element_bytes) as u32).map(f32::from_bits);
                let mut d = 0;
                while d < 4 {
                    let value = values[d];
                    let mut g = 0;
                    while g < 4 {
                        accum[g][d] += probability[g] * value;
                        g += 1;
                    }
                    d += 1;
                }
                t += 1;
            }
            ptx::sync();
            chunk += 64;
        }
        tile += tile_size;
    }
    if row >= rows { return; }
    let mut g = 0;
    while g < 4 {
        let at = (((row * kv + group) * 4 + g) * parts + part) * 132;
        if lane == 0 { *out.add(at + 128) = peak[g]; *out.add(at + 129) = denominator[g]; }
        let mut d = 0;
        while d < 4 { *out.add(at + lane * 4 + d) = accum[g][d]; d += 1; }
        g += 1;
    }
}

// The merge reads summaries only, never K/V. Row-local limits ensure extra
// splits in a wider verification launch do not change the reduction tree.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn decode_attention_merge(input: *const f32, out: *mut f32,
    heads: u32, parts: u32, base: u32, span: u32) {
    let row = _block_idx_y() as usize;
    let head = _block_idx_x() as usize + row * heads as usize;
    merge(input.add(head * parts as usize * 132), out.add(head * 128), (base as usize + row + 1).div_ceil(span as usize));
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn graph_decode_attention_merge(input: *const f32, out: *mut f32,
    heads: u32, parts: u32, base: u32, span: u32, replay_position: *const u32) {
    let base = base + if replay_position.is_null() { 0 } else { *replay_position };
    let row = _block_idx_y() as usize;
    let head = _block_idx_x() as usize + row * heads as usize;
    let count = (base as usize + row + 1).div_ceil(span as usize);
    merge(input.add(head * parts as usize * 132), out.add(head * 128), count);
}

#[inline(always)]
pub(crate) unsafe fn merge(input: *const f32, out: *mut f32, count: usize) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let sh = ptx::shared::<4160>();
    let mut peak = f32::NEG_INFINITY;
    let mut p = tid;
    while p < count { peak = peak.max(*input.add(p * 132 + 128)); p += 128; }
    peak = maximum(peak);
    if lane == 0 { ptx::store(sh + 4096 + (warp * 4) as u32, peak); }
    ptx::sync();
    for w in 0..4 { peak = peak.max(ptx::load(sh + 4096 + w * 4)); }
    let mut denominator = 0.0;
    p = tid;
    while p < count {
        let scale = ptx::exp(*input.add(p * 132 + 128) - peak);
        ptx::store(sh + (p * 4) as u32, scale);
        denominator += *input.add(p * 132 + 129) * scale;
        p += 128;
    }
    denominator = sum(denominator);
    if lane == 0 { ptx::store(sh + 4112 + (warp * 4) as u32, denominator); }
    ptx::sync();
    denominator = 0.0;
    for w in 0..4 { denominator += ptx::load(sh + 4112 + w * 4); }
    let mut value = 0.0;
    p = 0;
    while p < count { value += *input.add(p * 132 + tid) * ptx::load(sh + (p * 4) as u32); p += 1; }
    *out.add(tid) = value / denominator;
}

/// The same generic attention reduction handles wide prefill and decode.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_bounded_scores(a: crate::kernel_params::BoundedAttention) {
    crate::layer::batch::bounded_scores(a.query as *const f32, a.keys as *const f32,
        a.scores as *mut f32, a.heads, a.kv_heads, a.dim, a.position, a.capacity,
        1, a.window, a.cache_capacity, a.scale, a.round_bf16 != 0);
}

/// Eight adjacent queries reuse each K load, retaining every lane dot/tree.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_bounded_scores_grouped(a: crate::kernel_params::BoundedAttention) {
    let lane = _thread_idx_x() as usize%32;
    let slot = _block_idx_x() as usize*(core::arch::nvptx::_block_dim_x() as usize/32)+_thread_idx_x() as usize/32;
    let first = _block_idx_z() as usize*8;
    let head = _block_idx_y() as usize;
    let (h,kh,d) = (a.heads as usize,a.kv_heads as usize,a.dim as usize);
    let base = a.position as usize+first;
    let begin = if a.window == 0 {0} else {(base+1).saturating_sub(a.window as usize)};
    let token = begin+slot;
    if token > base+7 || first >= a.rows as usize {return;}
    let key = (token%a.cache_capacity as usize*kh+head/(h/kh))*d;
    let mut dot = [0.0;8];
    let mut feature = lane;
    while feature < d {
        let k = load::<1>(a.keys as *const f32,key+feature);
        for r in 0..8 {
            if first+r < a.rows as usize {dot[r] += *(a.query as *const f32).add(((first+r)*h+head)*d+feature)*k;}
        }
        feature += 32;
    }
    for r in 0..8 {
        let value = sum(dot[r])*a.scale;
        let position = base+r;
        let start = if a.window == 0 {0} else {(position+1).saturating_sub(a.window as usize)};
        if lane == 0 && first+r < a.rows as usize && token >= start && token <= position {
            *(a.scores as *mut f32).add(((first+r)*h+head)*a.capacity as usize+token-start) = crate::layer::round_bf16(value);
        }
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_bounded_softmax(a: crate::kernel_params::BoundedAttention) {
    if a.round_bf16 == 2 {
        // Full-row FP32 normalization, then BF16 P in the shared MMA layout.
        let row = _block_idx_y() as usize;
        let head = crate::index()/32; let lane = crate::index()%32;
        if head >= a.heads as usize {return;}
        let h = a.heads as usize; let capacity = a.capacity as usize;
        let count = a.position as usize+row+1;
        let scores = (a.scores as *const u16).add((row*h+head)*capacity);
        let load = |i: usize| f32::from_bits((*scores.add(i) as u32)<<16);
        let mut peak = f32::NEG_INFINITY;
        let mut t = lane;
        while t < count {peak = peak.max(load(t)); t += 32;}
        peak = maximum(peak);
        let mut denominator = 0.0;
        t = lane;
        while t < count {denominator += crate::layer::exp(load(t)-peak); t += 32;}
        denominator = sum(denominator);
        t = lane;
        while t < (count+63)/64*64 {
            let p = if t < count {crate::layer::exp(load(t)-peak)/denominator} else {0.0};
            let tile = ((row/16*h+head)*(capacity/16)+t/16)*256;
            let at = tile+(row%8*4+t%8/2)*8+row%16/8*2+t%16/8*4+t%2;
            *(a.metadata as *mut u16).add(at) = (crate::layer::round_bf16(p).to_bits()>>16) as u16;
            t += 32;
        }
    } else {
        crate::layer::batch::softmax(a.scores as *mut f32, a.heads, a.position, a.capacity,
            core::ptr::null(), a.window, a.round_bf16 != 0);
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_bounded_values(a: crate::kernel_params::BoundedAttention) {
    crate::layer::batch::bounded_values(a.scores as *const f32, a.values as *const f32,
        a.output as *mut f32, a.heads, a.kv_heads, a.dim, a.position, a.capacity,
        1, a.window, a.cache_capacity, a.round_bf16 != 0, false, 0);
}

#[inline(always)]
unsafe fn bounded_shared_load(p: u32) -> f32 {
    let bits: u32;
    core::arch::asm!("ld.shared.u16 {v}, [{p}];", v=out(reg32)bits, p=in(reg32)p);
    f32::from_bits(bits << 16)
}
#[inline(always)]
unsafe fn bounded_shared_store(p: u32, x: f32) {
    core::arch::asm!("{{ .reg .b16 v; cvt.u16.u32 v, {x}; st.shared.b16 [{p}], v; }}",
        x=in(reg32)(crate::layer::round_bf16(x).to_bits() >> 16), p=in(reg32)p);
}

/// Keep rounded scores/probabilities on chip. The full-row softmax and the
/// 64-key PV partials retain the bounded scalar path's arithmetic order.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_bounded_flash(a: crate::kernel_params::BoundedAttention) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let row = _block_idx_x() as usize;
    let head = _block_idx_y() as usize;
    let (h, kh, d) = (a.heads as usize, a.kv_heads as usize, a.dim as usize);
    let (keys, values, position, capacity) = if a.metadata == 0 {
        (a.keys as *const f32, a.values as *const f32, a.position as usize+row, a.cache_capacity as usize)
    } else {
        let cache = &*(a.metadata as *const crate::kernel_params::BoundedCache).add(row);
        (cache.keys as *const f32, cache.values as *const f32, cache.position as usize, cache.capacity as usize)
    };
    let begin = if a.window == 0 {0} else {(position+1).saturating_sub(a.window as usize)};
    let count = position+1-begin;
    let sh = 0u32;
    let q = (a.query as *const f32).add((row*h+head)*d);
    let kv_head = head/(h/kh);
    let mut t = warp;
    while t < count {
        let token = (begin+t) % capacity;
        let mut dot = 0.0;
        let mut feature = lane;
        while feature < d {
            dot += *q.add(feature) * load::<1>(keys, (token*kh+kv_head)*d+feature);
            feature += 32;
        }
        dot = sum(dot)*a.scale;
        if lane == 0 { bounded_shared_store(sh+(t*2) as u32, dot); }
        t += core::arch::nvptx::_block_dim_x() as usize/32;
    }
    ptx::sync();
    if warp == 0 {
        let mut peak = f32::NEG_INFINITY;
        t = lane;
        while t < count { peak = peak.max(bounded_shared_load(sh+(t*2) as u32)); t += 32; }
        peak = maximum(peak);
        let mut denominator = 0.0;
        t = lane;
        while t < count { denominator += crate::layer::exp(bounded_shared_load(sh+(t*2) as u32)-peak); t += 32; }
        denominator = sum(denominator);
        t = lane;
        while t < count {
            bounded_shared_store(sh+(t*2) as u32, crate::layer::exp(bounded_shared_load(sh+(t*2) as u32)-peak)/denominator);
            t += 32;
        }
    }
    ptx::sync();
    if tid < d {
        let mut result = 0.0;
        t = 0;
        let mut token = begin % capacity;
        while t < count {
            let end = (t+64).min(count);
            let mut partial = 0.0;
            while t < end {
                partial += bounded_shared_load(sh+(t*2) as u32) * load::<1>(values, (token*kh+kv_head)*d+tid);
                t += 1;
                token += 1;
                if token == capacity {token=0;}
            }
            result += partial;
        }
        *(a.output as *mut f32).add((row*h+head)*d+tid) = crate::layer::round_bf16(result);
    }
}

/// Parallel 64-key PV partials preserve the bounded path's exact boundaries.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_bounded_values_split(a: crate::kernel_params::BoundedAttention) {
    crate::layer::batch::bounded_values(a.scores as *const f32, a.values as *const f32,
        a.output as *mut f32, a.heads, a.kv_heads, a.dim, a.position, a.capacity,
        1, a.window, a.cache_capacity, false, true, (a.capacity as usize+63)/64);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_bounded_merge(a: crate::kernel_params::BoundedAttention) {
    let i = _block_idx_x() as usize*core::arch::nvptx::_block_dim_x() as usize+_thread_idx_x() as usize;
    let (h,d) = (a.heads as usize, a.dim as usize);
    if i >= a.rows as usize*h*d {return;}
    let row = i/(h*d); let head = i/d%h; let feature = i%d;
    let length = if a.window == 0 {a.position as usize+row+1} else {(a.position as usize+row+1).min(a.window as usize)};
    let chunks = (a.capacity as usize+63)/64;
    let mut result = 0.0;
    for part in 0..(length+63)/64 {
        result += *(a.scores as *const f32).add(((row*h+head)*chunks+part)*d+feature);
    }
    *(a.output as *mut f32).add(i) = crate::layer::round_bf16(result);
}
