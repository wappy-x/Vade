//! Ragged Qwen3 GQA rows with exactly the serial reduction order.
use super::*;

#[repr(C)]
struct Row {
    keys: *mut f32, values: *mut f32, position: u32, format: u32,
    scores: u32, parts: u32, span: u32, chunks: u32,
    prefix_keys: *const f32, prefix_values: *const f32, base: u32, _pad: u32,
}
#[repr(C)]
struct Tile { row: u32, chunk: u32 }
#[inline(always)]
unsafe fn load(p: *const f32, i: usize, format: u32) -> f32 {
    super::batch::cache_load(p, i, format)
}
#[inline(always)]
unsafe fn save(p: *mut f32, i: usize, x: f32, format: u32) {
    if format == 1 { let bits = x.to_bits(); *p.cast::<u16>().add(i) = ((bits.wrapping_add(0x7fff+((bits>>16)&1)))>>16) as u16; }
    else { *p.add(i) = x; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_qk_rope(p: crate::kernel_params::MultiQk) {
    let (query, key, value) = (p.query as *mut f32, p.key as *const f32, p.value as *const f32);
    let (q_weight, k_weight, rope, metadata) = (p.q_weight as *const f32,
        p.k_weight as *const f32, p.rope as *const f32, p.metadata as *const u32);
    let (heads, kv_heads, dim, epsilon) = (p.heads, p.kv_heads, p.dim, p.epsilon);
    let row = _block_idx_y() as usize;
    let meta = &*metadata.cast::<Row>().add(row);
    let position = meta.position as usize;
    let cache_position = position - meta.base as usize;
    let (h,kh,d) = (heads as usize,kv_heads as usize,dim as usize);
    let query = query.add(row*h*d);
    let key = key.add(row*kh*d);
    let value = value.add(row*kh*d);
    let head = index()/32;
    let lane = index()%32;
    if head >= h+kh { return; }
    let is_key = head >= h;
    let start = (if is_key {head-h} else {head})*d;
    let input = if is_key {key} else {query as *const f32};
    let weight = if is_key {k_weight} else {q_weight};
    let mut squares = 0.0;
    let mut i = lane;
    while i < d { let x = *input.add(start+i); squares += x*x; i += 32; }
    let scale = if epsilon < 0.0 { 1.0 } else { 1.0/sqrt(sum(squares)/dim as f32+epsilon) };
    if meta.format == 4 && is_key {
        super::batch::save_i16_head(input, value, weight, rope, meta.keys, meta.values,
            start, scale, position, kh, head-h, lane);
        return;
    }
    i = lane;
    while i < d/2 {
        let a = *input.add(start+i)*scale* *weight.add(i);
        let b = *input.add(start+i+d/2)*scale* *weight.add(i+d/2);
        let sin = *rope.add(position*d+2*i);
        let cos = *rope.add(position*d+2*i+1);
        let (a,b) = crate::layer::rope_pair(a, b, sin, cos, epsilon < 0.0);
        if is_key {
            save(meta.keys,cache_position*kh*d+start+i,a,meta.format);
            save(meta.keys,cache_position*kh*d+start+i+d/2,b,meta.format);
        } else { *query.add(start+i) = a; *query.add(start+i+d/2) = b; }
        i += 32;
    }
    if is_key {
        i = lane;
        while i < d {
            let x = *value.add(start+i);
            save(meta.values,cache_position*kh*d+start+i,if epsilon < 0.0 { crate::layer::round_bf16(x) } else { x },meta.format);
            i += 32;
        }
    }
}

// A descriptor covers exactly one row's 64-key tile. Scores and partials
// retain the serial arithmetic; neither launch nor storage pads to another row.
#[inline(always)]
unsafe fn score_head(query: *const f32, dim: usize, key: impl Fn(usize) -> f32) -> f32 {
    let mut dot = 0.0;
    let mut i = _thread_idx_x() as usize % 32;
    while i < dim { dot += *query.add(i) * key(i); i += 32; }
    sum(dot) / sqrt(dim as f32)
}

#[inline(always)]
unsafe fn value_head(scores: *const f32, begin: usize, end: usize, value: impl Fn(usize) -> f32) -> f32 {
    let mut accum = 0.0;
    let mut t = begin;
    while t < end { accum += *scores.add(t) * value(t); t += 1; }
    accum
}

#[inline(always)]
unsafe fn scores_group<const GROUP: usize>(query: *const f32, meta: &Row, scores: *mut f32,
    row: usize, chunk: usize, kh: usize, heads: usize, kv_heads: usize, dim: usize) {
    let lane = _thread_idx_x() as usize % 32;
    let q = query.add((row*heads+kh*GROUP)*dim);
    let stride = meta.position as usize+1;
    let end = (chunk*64+64).min(stride);
    let mut t = chunk*64+_thread_idx_x() as usize/32;
    while t < end {
        let key = (t*kv_heads+kh)*dim;
        let mut dot = [0.0; GROUP];
        let mut i = lane;
        while i < dim {
            let k = load(meta.keys,key+i,meta.format);
            let mut g = 0;
            while g < GROUP { dot[g] += *q.add(g*dim+i)*k; g += 1; }
            i += 32;
        }
        let denominator = sqrt(dim as f32);
        let mut g = 0;
        while g < GROUP {
            let score = sum(dot[g])/denominator;
            if lane == 0 { *scores.add(meta.scores as usize+(kh*GROUP+g)*stride+t) = score; }
            g += 1;
        }
        t += 4;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_scores(query: *const f32, metadata: *const u32, tiles: *const u32,
    scores: *mut f32, heads: u32, kv_heads: u32, dim: u32) {
    let tile = &*tiles.cast::<Tile>().add(_block_idx_x() as usize);
    let row = tile.row as usize;
    let meta = &*metadata.cast::<Row>().add(row);
    let lane = _thread_idx_x() as usize % 32;
    let kh = _block_idx_y() as usize;
    let (h,d) = (heads as usize,dim as usize);
    if heads == kv_heads * 8 {
        scores_group::<8>(query,meta,scores,row,tile.chunk as usize,kh,h,kv_heads as usize,d);
    } else if heads == kv_heads * 4 {
        scores_group::<4>(query,meta,scores,row,tile.chunk as usize,kh,h,kv_heads as usize,d);
    } else if heads == kv_heads * 5 {
        scores_group::<5>(query,meta,scores,row,tile.chunk as usize,kh,h,kv_heads as usize,d);
    } else if heads == kv_heads * 2 {
        scores_group::<2>(query,meta,scores,row,tile.chunk as usize,kh,h,kv_heads as usize,d);
    } else {
        let group = h / kv_heads as usize;
        let end = (tile.chunk as usize*64+64).min(meta.position as usize+1);
        for g in 0..group {
            let head = kh*group+g;
            let q = query.add((row*h+head)*d);
            let mut t = tile.chunk as usize*64+_thread_idx_x() as usize/32;
            while t < end {
                let key = (t*kv_heads as usize+kh)*d;
                let score = score_head(q, d, |i| load(meta.keys, key+i, meta.format));
                if lane == 0 { *scores.add(meta.scores as usize+head*(meta.position as usize+1)+t) = score; }
                t += 4;
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_softmax(scores: *mut f32, metadata: *const u32, heads: u32) {
    let row = _block_idx_y() as usize;
    let head = index()/32;
    let lane = index()%32;
    let meta = &*metadata.cast::<Row>().add(row);
    if head >= heads as usize || meta.span != 0 { return; }
    let end = meta.position as usize+1;
    let scores = scores.add(meta.scores as usize+head*end);
    let mut peak = f32::NEG_INFINITY;
    let mut t = lane;
    while t < end { peak = peak.max(*scores.add(t)); t += 32; }
    peak = maximum(peak);
    let mut denominator = 0.0;
    t = lane;
    while t < end { let p = exp(*scores.add(t)-peak); *scores.add(t) = p; denominator += p; t += 32; }
    denominator = sum(denominator);
    t = lane;
    while t < end { *scores.add(t) /= denominator; t += 32; }
}

// Share each V load while retaining ascending-key sums for every query head.
#[inline(always)]
unsafe fn values_group<const GROUP: usize>(scores: *const f32, meta: &Row, partials: *mut f32,
    chunk: usize, kh: usize, heads: usize, stride: usize, dim: usize, feature: usize) {
    let capacity = meta.position as usize+1;
    let s = scores.add(meta.scores as usize+kh*GROUP*capacity);
    let mut t = chunk*64;
    let end = (t+64).min(capacity);
    let mut accum = [0.0; GROUP];
    while t < end {
        let v = load(meta.values,t*stride+kh*dim+feature,meta.format);
        let mut g = 0;
        while g < GROUP { accum[g] += *s.add(g*capacity+t)*v; g += 1; }
        t += 1;
    }
    let out = partials.add(meta.parts as usize+(chunk*heads+kh*GROUP)*dim+feature);
    let mut g = 0;
    while g < GROUP { *out.add(g*dim) = accum[g]; g += 1; }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_values(scores: *const f32, metadata: *const u32, tiles: *const u32,
    partials: *mut f32, heads: u32, kv_heads: u32, dim: u32) {
    let tile = &*tiles.cast::<Tile>().add(_block_idx_x() as usize);
    let meta = &*metadata.cast::<Row>().add(tile.row as usize);
    let kh = _block_idx_y() as usize;
    let feature = _thread_idx_x() as usize;
    let (h,d,stride) = (heads as usize,dim as usize,kv_heads as usize*dim as usize);
    if feature >= d { return; }
    if heads == kv_heads*8 && d <= 1024 {
        values_group::<8>(scores,meta,partials,tile.chunk as usize,kh,h,stride,d,feature);
    } else if heads == kv_heads*4 && d <= 1024 {
        values_group::<4>(scores,meta,partials,tile.chunk as usize,kh,h,stride,d,feature);
    } else if heads == kv_heads*5 && d <= 1024 {
        values_group::<5>(scores,meta,partials,tile.chunk as usize,kh,h,stride,d,feature);
    } else if heads == kv_heads*2 && d <= 1024 {
        values_group::<2>(scores,meta,partials,tile.chunk as usize,kh,h,stride,d,feature);
    } else {
        let group = h / kv_heads as usize;
        let capacity = meta.position as usize+1;
        let mut feature = feature;
        while feature < d {
            for g in 0..group {
                let head = kh*group+g;
                let s = scores.add(meta.scores as usize+head*capacity);
                let begin = tile.chunk as usize*64;
                let end = (begin+64).min(capacity);
                let value = value_head(s, begin, end, |t| load(meta.values, t*stride+kh*d+feature, meta.format));
                *partials.add(meta.parts as usize+(tile.chunk as usize*h+head)*d+feature) = value;
            }
            feature += _block_dim_x() as usize;
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_merge(partials: *const f32, metadata: *const u32, output: *mut f32, width: u32) {
    let i = index();
    let row = _block_idx_y() as usize;
    let meta = &*metadata.cast::<Row>().add(row);
    if i >= width as usize || meta.span != 0 { return; }
    let mut value = 0.0;
    let mut chunk = 0;
    while chunk < meta.chunks as usize { value += *partials.add(meta.parts as usize+chunk*width as usize+i); chunk += 1; }
    *output.add(row*width as usize+i) = value;
}

// Two 128-thread groups stage their next 16 values while accumulating the
// current slice. The 64-key partials and their merge keep the scalar order.
#[inline(always)]
unsafe fn draft_stage_values(sh: u32, meta: &Row, kv: usize, kh: usize,
    pair: usize, slice: usize, tid: usize, length: usize) {
    let group = tid / 128;
    let mut i = tid % 128 * 4;
    while i < 16 * 128 {
        let t = pair * 128 + group * 64 + slice * 16 + i / 128;
        if t < length {
            let (values, at) = if t < meta.base as usize { (meta.prefix_values, t) }
                else { (meta.values as *const f32, t - meta.base as usize) };
            crate::decode_attention::draft_stage4(sh + (group * 4096 + i * 2) as u32,
                values, (at * kv + kh) * 128 + i % 128);
        }
        i += 128 * 4;
    }
}

// Full-row softmax and 64-key partials keep the canonical arithmetic. Two
// 128-thread groups compute separate chunks; their final merge stays ordered.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_draft_attention(p: crate::kernel_params::DraftAttention) {
    let (query, metadata, tiles, output, kv_heads) = (p.query as *const f32,
        p.metadata as *const u32, p.tiles as *const u32, p.output as *mut f32, p.kv_heads);
    let tile = &*tiles.cast::<Tile>().add(_block_idx_x() as usize);
    let meta = &*metadata.cast::<Row>().add(tile.row as usize);
    let kv = kv_heads as usize;
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let kh = _block_idx_y() as usize;
    let length = meta.position as usize+1;
    let q = query.add((tile.row as usize*kv+kh)*2*128);
    let sh = crate::ptx::shared::<28672>();
    let mut t = warp;
    while t < length {
        let mut dot = [0.0; 2];
        let mut i = lane;
        while i < 128 {
            let (keys, at) = if t < meta.base as usize { (meta.prefix_keys, t) }
                else { (meta.keys as *const f32, t - meta.base as usize) };
            let k = load(keys,(at*kv+kh)*128+i,1);
            dot[0] += *q.add(i)*k;
            dot[1] += *q.add(128+i)*k;
            i += 32;
        }
        let a = sum(dot[0])/sqrt(128.0);
        let b = sum(dot[1])/sqrt(128.0);
        if lane == 0 {
            crate::ptx::store(sh+(t*4) as u32,a);
            crate::ptx::store(sh+((512+t)*4) as u32,b);
        }
        t += 8;
    }
    crate::ptx::sync();
    if warp < 2 {
        let base = sh+(warp*512*4) as u32;
        let mut peak = f32::NEG_INFINITY;
        t = lane;
        while t < length { peak = peak.max(crate::ptx::load(base+(t*4) as u32)); t += 32; }
        peak = maximum(peak);
        let mut denominator = 0.0;
        t = lane;
        while t < length {
            let p = exp(crate::ptx::load(base+(t*4) as u32)-peak);
            crate::ptx::store(base+(t*4) as u32,p);
            denominator += p;
            t += 32;
        }
        denominator = sum(denominator);
        t = lane;
        while t < length {
            let at = base+(t*4) as u32;
            crate::ptx::store(at,crate::ptx::load(at)/denominator);
            t += 32;
        }
    }
    crate::ptx::sync();
    let feature = tid % 128;
    let chunks = (length+63)/64;
    let mut pair = 0;
    while pair * 2 < chunks {
        let chunk = pair * 2 + tid / 128;
        let mut partial = [0.0; 2];
        draft_stage_values(sh + 12288, meta, kv, kh, pair, 0, tid, length);
        crate::ptx::commit(); crate::ptx::wait(); crate::ptx::sync();
        let mut slice = 0;
        while slice < 4 {
            let buffer = slice % 2;
            if slice < 3 {
                draft_stage_values(sh + 12288 + ((1 - buffer) * 8192) as u32,
                    meta, kv, kh, pair, slice + 1, tid, length);
                crate::ptx::commit();
            }
            t = chunk * 64 + slice * 16;
            let end = (t + 16).min(length);
            while t < end {
                let at = sh + 12288 + (buffer * 8192 + tid / 128 * 4096
                    + ((t % 16) * 128 + feature) * 2) as u32;
                let v = crate::decode_attention::draft_load(at);
                partial[0] += crate::ptx::load(sh+(t*4) as u32)*v;
                partial[1] += crate::ptx::load(sh+((512+t)*4) as u32)*v;
                t += 1;
            }
            crate::ptx::wait(); crate::ptx::sync();
            slice += 1;
        }
        if chunk < chunks {
            crate::ptx::store(sh+(4096+(chunk*2*128+feature)*4) as u32,partial[0]);
            crate::ptx::store(sh+(4096+((chunk*2+1)*128+feature)*4) as u32,partial[1]);
        }
        pair += 1;
    }
    let mut chunk;
    crate::ptx::sync();
    if tid < 128 {
        let mut result = [0.0; 2];
        chunk = 0;
        while chunk < chunks {
            result[0] += crate::ptx::load(sh+(4096+(chunk*2*128+feature)*4) as u32);
            result[1] += crate::ptx::load(sh+(4096+((chunk*2+1)*128+feature)*4) as u32);
            chunk += 1;
        }
        let at = (tile.row as usize*kv+kh)*2*128+feature;
        *output.add(at) = result[0];
        *output.add(at+128) = result[1];
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_stream(args: crate::kernel_params::MultiStream) {
    let (query, metadata, tiles, parts) = (args.query as *const f32, args.metadata as *const u32,
        args.tiles as *const u32, args.parts as *mut f32);
    let (kv_heads, heads) = (args.kv_heads, args.heads);
    // Mutually exclusive formats and prefix paths share one CTA workspace.
    let sh = crate::ptx::shared::<{ crate::decode_attention::STREAM_SHARED }>();
    if heads == kv_heads * 2 {
        let tile = &*tiles.cast::<Tile>().add(_block_idx_x() as usize);
        let meta = &*metadata.cast::<Row>().add(tile.row as usize);
        let prefix = (meta.base != 0).then_some((meta.prefix_keys.cast::<u16>(), meta.prefix_values.cast::<u16>(), meta.base as usize));
        crate::decode_attention::stream_mma::<false>(query.add(tile.row as usize * heads as usize * 128),
            meta.keys, meta.values, parts.add(meta.parts as usize), kv_heads, meta.position as usize + 1,
            tile.chunk as usize, meta.chunks, meta.span, 2, prefix, sh);
    } else if heads == kv_heads * 8 {
        stream_rows::<2>(query, metadata, tiles, parts, kv_heads, sh);
    } else {
        stream_rows::<1>(query, metadata, tiles, parts, kv_heads, sh);
    }
}

#[inline(always)]
unsafe fn stream_rows<const GROUPS: usize>(query: *const f32, metadata: *const u32, tiles: *const u32,
    parts: *mut f32, kv_heads: u32, sh: u32) {
    let tile = &*tiles.cast::<Tile>().add(_block_idx_x() as usize);
    let meta = &*metadata.cast::<Row>().add(tile.row as usize);
    let q = query.add(tile.row as usize*kv_heads as usize*4*GROUPS*128);
    let out = parts.add(meta.parts as usize);
    let length = meta.position as usize+1;
    let part = tile.chunk as usize;
    match meta.format {
        1 => crate::decode_attention::stream_spans::<GROUPS>(q,meta.keys,meta.values,out,kv_heads,length,part,meta.chunks,meta.span,
            meta.prefix_keys,meta.prefix_values,meta.base as usize,sh),
        4 => crate::decode_attention::stream::<4, false, GROUPS>(q,meta.keys,meta.values,out,kv_heads,length,part,meta.chunks,meta.span,sh),
        _ => crate::decode_attention::stream::<0, false, GROUPS>(q,meta.keys,meta.values,out,kv_heads,length,part,meta.chunks,meta.span,sh),
    }
}

/// Scalar groups retain full-row softmax and ascending 64-key value sums.
/// Hold the queries in registers and reuse each staged K/V load across GROUP
/// heads. QK owns a warp per row/group; PV owns 128 feature lanes per row/group.
#[inline(always)]
unsafe fn cascade_scalar<const GROUP: usize>(args: &crate::kernel_params::Cascade,
    keys: *const f32, values: *const f32, sh: u32) {
    let (tid, threads) = (_thread_idx_x() as usize, _block_dim_x() as usize);
    let lane = tid % 32;
    let (heads, kv) = (args.heads as usize, args.kv_heads as usize);
    let (group, kh, chunk) = (heads / kv, _block_idx_y() as usize, _block_idx_x() as usize);
    let begin = chunk * 64;
    let source = if args.values == 0 { keys } else { values };
    let mut i = tid;
    while i < 64 * 128 {
        let value = load(source, ((begin + i / 128) * kv + kh) * 128 + i % 128, 1);
        crate::ptx::store(sh + (i * 4) as u32, value);
        i += threads;
    }
    crate::ptx::sync();
    let lanes = if args.values == 0 { 32 } else { 128 };
    let mut m = tid / lanes * GROUP;
    while m < args.rows as usize * group {
        let (row, head) = (m / group, kh * group + m % group);
        let meta = &*(args.metadata as *const Row).add(row);
        let stride = meta.position as usize + 1;
        let scores = (args.scores as *mut f32).add(meta.scores as usize + head * stride);
        if args.values == 0 {
            let query = (args.query as *const f32).add((row * heads + head) * 128 + lane);
            let mut q = [[0.0f32; 4]; GROUP];
            for g in 0..GROUP {
                for d in 0..4 { q[g][d] = *query.add(g * 128 + d * 32); }
            }
            let mut saved = [0.0f32; GROUP];
            let mut t = 0;
            while t < 64 {
                let mut dot = [0.0f32; GROUP];
                for d in 0..4 {
                    let k = crate::ptx::load(sh + ((t * 128 + d * 32 + lane) * 4) as u32);
                    for g in 0..GROUP { dot[g] += q[g][d] * k; }
                }
                for g in 0..GROUP {
                    let score = sum(dot[g]) / sqrt(128.0);
                    if lane == t % 32 { saved[g] = score; }
                }
                if t % 32 == 31 {
                    for g in 0..GROUP { *scores.add(g * stride + begin + t - 31 + lane) = saved[g]; }
                }
                t += 1;
            }
        } else {
            let d = tid % 128;
            let out = (args.parts as *mut f32).add(meta.parts as usize + (chunk * heads + head) * 128 + d);
            let mut accum = [0.0f32; GROUP];
            for half in 0..2 {
                let mut p = [0.0f32; GROUP];
                for g in 0..GROUP { p[g] = *scores.add(g * stride + begin + half * 32 + lane); }
                for t in 0..32 {
                    let v = crate::ptx::load(sh + (((half * 32 + t) * 128 + d) * 4) as u32);
                    for g in 0..GROUP { accum[g] += other(p[g], (lane ^ t) as u32) * v; }
                }
            }
            for g in 0..GROUP { *out.add(g * 128) = accum[g]; }
        }
        m += threads / lanes * GROUP;
    }
}

/// Keep the scalar launch separate: the MMA split's register allocation would
/// otherwise limit this low-register reuse path to one resident CTA per SM.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_cascade_scalar(args: crate::kernel_params::Cascade) {
    let source = &*(args.metadata as *const Row);
    let (keys, values) = if source.base > 0 { (source.prefix_keys, source.prefix_values) }
        else { (source.keys as *const f32, source.values as *const f32) };
    let sh = crate::ptx::shared::<32768>();
    if args.heads == args.kv_heads * 5 { cascade_scalar::<5>(&args, keys, values, sh); }
    else { cascade_scalar::<1>(&args, keys, values, sh); }
}

/// Shared-prefix chunks reuse K/V with the canonical MMA order.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_cascade(args: crate::kernel_params::Cascade) {
    let metadata = args.metadata as *const u32;
    let source = &*metadata.cast::<Row>();
    let (keys, values) = if source.base > 0 { (source.prefix_keys, source.prefix_values) }
        else { (source.keys as *const f32, source.values as *const f32) };
    let sh = crate::ptx::shared::<{ crate::decode_attention::MMA_SHARED }>();
    let (heads, kv) = (args.heads as usize, args.kv_heads as usize);
    let (group, kh, part) = (heads / kv, _block_idx_y() as usize, _block_idx_x() as usize);
    let (span, parts) = (source.span as usize, args.parts as *mut f32);
    let begin = part * span;
    crate::decode_attention::split_mma((args.query as *const f32).add(kh * group * 128), heads * 128, group,
        args.rows as usize, keys.cast(), values.cast(), None, kv, kh, begin, begin + span, false,
        |row, h| {
            let meta = metadata.add(row * 16);
            parts.add(*meta.add(7) as usize + ((kh * group + h) * *meta.add(9) as usize + part) * 132)
        }, sh);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_stream_merge(parts: *const f32, metadata: *const u32, output: *mut f32, heads: u32) {
    let row = _block_idx_y() as usize;
    let head = _block_idx_x() as usize;
    let meta = &*metadata.cast::<Row>().add(row);
    if meta.span == 0 || meta.span == u32::MAX { return; }
    crate::decode_attention::merge(parts.add(meta.parts as usize+head*meta.chunks as usize*132),
        output.add((row*heads as usize+head)*128), meta.chunks as usize);
}
