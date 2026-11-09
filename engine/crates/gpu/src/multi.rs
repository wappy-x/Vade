//! Arena row views and queued reductions for independent-session batches.
use super::*;

/// Reused shortlists; no duplicate model weights or vocabulary-sized host rows.
pub struct TopKScratch<'a> { partials: DeviceVector<'a>, result: DeviceVector<'a> }
impl TopKScratch<'_> {
    pub fn allocations(&self, out: &mut Vec<(usize, usize)>) {
        self.partials.allocations(out); self.result.allocations(out);
    }
}

impl<'a> Backend<'a> {
    pub fn top_k_device(&self, logits: &DeviceVector<'_>, scratch: &mut Option<TopKScratch<'a>>,
        rows: usize, k: usize) -> Result<Vec<Result<Vec<(u32, f32)>, String>>, String> {
        if rows == 0 || !logits.len.is_multiple_of(rows) || !(1..=64).contains(&k) {
            return Err("invalid top-k batch".into());
        }
        let width = logits.len / rows;
        if width < k || width > 1 << 24 { return Err("invalid top-k vocabulary".into()); }
        self.check_vector(logits, logits.len)?;
        let count = width.div_ceil(1024);
        let words = rows * k * 3;
        if scratch.as_ref().is_none_or(|s| s.partials.len < words*count || s.result.len < words) {
            let mut buffers = self.vector_arena(&[(words*count).next_power_of_two(), words.next_power_of_two()])?.into_iter();
            *scratch = Some(TopKScratch { partials: buffers.next().unwrap(), result: buffers.next().unwrap() });
        }
        let scratch = scratch.as_ref().unwrap();
        let mut p = crate::kernel_params::LogitsTopK { logits: logits.buffer.address,
            partials: scratch.partials.buffer.address, result: scratch.result.buffer.address,
            width: width as u32, k: k as u32 };
        let kernels = self.layer_kernels()?;
        unsafe {
            self.enqueue_grid(kernels.top_k, count*32, rows as u32, 1, &mut [parameter(&mut p)])?;
            self.enqueue_grid(kernels.top_k_finish, 32, rows as u32, 1, &mut [parameter(&mut p)])?;
        }
        Ok(self.read_vector_values(&scratch.result.slice(0..words)?)?.chunks_exact(k*3).map(|row| {
            row.as_chunks::<3>().0.iter().map(|v| {
                if v[2] != 1.0 || !v[0].is_finite() { return Err("nonfinite logits".into()); }
                let id = crate::kernel_params::id_index(v[1], width as u32).ok_or("invalid top-k ID")?;
                Ok((id as u32, v[0]))
            }).collect()
        }).collect())
    }
}

pub struct CacheRows<'s, 'a> {
    pub keys: &'s mut DeviceVector<'a>, pub values: &'s mut DeviceVector<'a>,
    pub prefix: Option<(&'s DeviceVector<'a>, &'s DeviceVector<'a>, usize)>,
    pub position: usize, pub rows: usize, pub format: KvFormat,
}
pub(super) struct Kernels {
    qk: Handle, scores: Handle, softmax: Handle, values: Handle, merge: Handle,
    stream: Handle, draft_attention: Handle, verify: Handle, stream_merge: Handle,
    selected_ids: Handle, gather_rows: Handle, cascade: Handle, scalar_cascade: Handle, prefill: Handle,
}
fn row_shape(length: usize, format: KvFormat, heads: usize, kv_heads: usize, dim: usize) -> (usize, usize) {
    // Two-head BF16 draft rows retain full-row softmax in shared memory.
    if length <= 512 && dim == 128 && heads == kv_heads * 2 && format == KvFormat::Bf16 {
        return (u32::MAX as usize, 1);
    }
    let span = if dim == 128 && (heads == kv_heads * 4 || heads == kv_heads * 8
        || heads == kv_heads * 2 && format == KvFormat::Bf16 && length > 4096) {
        if length <= 4096 {64} else if length <= 16384 {256} else {1024}
    } else {0};
    (span, length.div_ceil(if span == 0 {64} else {span}))
}
impl Kernels {
    #[allow(clippy::if_same_then_else)] // the branches are separate test cases.
    pub(super) fn kind(&self, function: Handle) -> Option<usize> {
        if function == self.qk { Some(2) } else if function == self.scores { Some(3) }
        else if [self.stream,self.draft_attention,self.verify,self.cascade,self.scalar_cascade,self.stream_merge,self.prefill].contains(&function) { Some(3) }
        else if function == self.softmax { Some(4) } else if [self.values,self.merge].contains(&function) { Some(5) } else { None }
    }
}

impl<'a> DeviceVector<'a> {
    /// A zero-copy subrange of vector_arena storage, retaining its allocation.
    /// Inputs and outputs of a launch must refer to disjoint ranges.
    pub fn slice(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        if range.start >= range.end || range.end > self.len { return Err("invalid device vector slice".into()); }
        let owner = self.buffer.owner.as_ref().ok_or("slicing requires vector_arena storage")?;
        let offset = (self.buffer.address - owner.address) as usize + range.start * 4;
        Ok(Self { buffer: Buffer::view(owner, offset, range.len() * 4)?, len: range.len(), managed: self.managed,
            #[cfg(debug_assertions)] projection_scratch: self.projection_scratch.clone() })
    }
}

impl Backend<'_> {
    /// Model-normalized BF16 Q/K/V and windowed attention for independent rows.
    pub fn model_attention_multi_device(&self, query: &mut DeviceVector<'_>, key: &DeviceVector<'_>, value: &DeviceVector<'_>,
        q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>, caches: &mut [CacheRows<'_, '_>],
        metadata: &mut DeviceVector<'_>, scores: &mut DeviceVector<'_>, output: &mut DeviceVector<'_>,
        heads: usize, kv_heads: usize, dim: usize, window: usize, epsilon: f32) -> Result<(), String> {
        let rows = caches.len();
        if rows == 0 || heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || !matches!(dim, 256 | 512)
            || caches.iter().any(|c| c.rows != 1 || c.format != KvFormat::Bf16 || c.prefix.is_some()) {
            return Err("invalid model attention batch".into());
        }
        self.check_vector_prefix(query, rows*heads*dim)?; self.check_vector_prefix(output, rows*heads*dim)?;
        self.check_vector_prefix(key, rows*kv_heads*dim)?; self.check_vector_prefix(value, rows*kv_heads*dim)?;
        self.check_vector_prefix(q_norm, dim)?; self.check_vector_prefix(k_norm, dim)?;
        let capacity = caches.iter().map(|c| if window == 0 {c.position+1} else {window.min(c.position+1)}).max().unwrap();
        if capacity > 32768 {
            for (row, cache) in caches.iter_mut().enumerate() {
                let mut q = query.slice(row*heads*dim..(row+1)*heads*dim)?;
                let k = key.slice(row*kv_heads*dim..(row+1)*kv_heads*dim)?;
                let v = value.slice(row*kv_heads*dim..(row+1)*kv_heads*dim)?;
                let mut out = output.slice(row*heads*dim..(row+1)*heads*dim)?;
                let ring = cache.keys.len()*2/(kv_heads*dim);
                self.model_qkv_device(&mut q, &k, &v, q_norm, k_norm, rope, cache.keys, cache.values,
                    heads, kv_heads, dim, cache.position, 1, ring, epsilon)?;
                self.bounded_attention_device(&q, cache.keys, cache.values, scores, &mut out,
                    heads, kv_heads, dim, cache.position, 1, window, ring, 1.0)?;
            }
            return Ok(());
        }
        let mut descriptors = Vec::with_capacity(rows*6);
        for cache in caches {
            self.check_vector(cache.keys, cache.keys.len())?; self.check_vector(cache.values, cache.keys.len())?;
            self.check_vector_prefix(rope, (cache.position+1)*dim)?;
            let ring = cache.keys.len()*2/(kv_heads*dim);
            if ring < capacity.min(cache.position+1) { return Err("model attention ring too small".into()); }
            let (k,v) = (cache.keys.buffer.address, cache.values.buffer.address);
            descriptors.extend([k as u32, (k>>32) as u32, v as u32, (v>>32) as u32,
                cache.position as u32, ring as u32].map(f32::from_bits));
        }
        self.write_vector(&mut metadata.slice(0..descriptors.len())?, &descriptors)?;
        let mut qkv = crate::kernel_params::ModelQkv { query: query.buffer.address, key: key.buffer.address, value: value.buffer.address,
            q_weight: q_norm.buffer.address, k_weight: k_norm.buffer.address, rope: rope.buffer.address,
            keys: 0, values: 0, metadata: metadata.buffer.address, heads: heads as u32, kv_heads: kv_heads as u32,
            dim: dim as u32, position: 0, rows: rows as u32, cache_capacity: 0, epsilon };
        let mut attention = crate::kernel_params::BoundedAttention { query: query.buffer.address, keys: 0, values: 0,
            scores: 0, output: output.buffer.address, metadata: metadata.buffer.address, heads: heads as u32, kv_heads: kv_heads as u32,
            dim: dim as u32, position: 0, rows: rows as u32, capacity: capacity as u32, window: window as u32,
            cache_capacity: 0, scale: 1.0, round_bf16: 1 };
        let k = self.layer_kernels()?;
        unsafe {
            self.enqueue_grid(k.model_qkv, (heads+kv_heads)*32, rows as u32, 1, &mut [parameter(&mut qkv)])?;
            self.enqueue_dims_shared(k.bounded_flash, rows as u32, heads as u32, 1, dim as u32,
                (capacity*2).div_ceil(256) as u32*256, &mut [parameter(&mut attention)])
        }
    }
    pub fn require_multi_kernels(&self) -> Result<(), String> {
        if self.multi.borrow().is_none() {
            let m = self.layer_module()?;
            *self.multi.borrow_mut() = Some(Kernels { qk: m.function("multi_qk_rope")?, scores: m.function("multi_scores")?,
                softmax: m.function("multi_softmax")?, values: m.function("multi_values")?, merge: m.function("multi_merge")?,
                stream: m.function("multi_stream")?, draft_attention: m.function("multi_draft_attention")?,
                selected_ids: m.function("batch_selected_ids")?, gather_rows: m.function("batch_gather_rows")?, verify: m.function_with_shared("decode_attention_verify", 98304)?,
                cascade: m.function("multi_cascade")?, scalar_cascade: m.function("multi_cascade_scalar")?,
                prefill: m.function_with_shared("prefill_attention_multi_bf16", 65536)?,
                stream_merge: m.function("multi_stream_merge")? });
        }
        Ok(())
    }
    /// Queue ragged BF16 prompt tiles together so short suffixes fill GB10.
    pub fn prefill_attention_multi_device(&self, query: &DeviceVector<'_>, caches: &[CacheRows<'_, '_>],
        metadata: &mut DeviceVector<'_>, tiles: &mut DeviceVector<'_>, output: &mut DeviceVector<'_>,
        heads: usize, kv_heads: usize) -> Result<(), String> {
        if caches.is_empty() || kv_heads == 0 || !heads.is_multiple_of(kv_heads)
            || caches.iter().any(|c| c.rows == 0 || c.format != KvFormat::Bf16) {
            return Err("ragged prefill requires BF16 GQA rows".into());
        }
        let rows = caches.iter().map(|c| c.rows).sum::<usize>();
        self.check_vector(query, rows * heads * 128)?;
        self.check_vector(output, rows * heads * 128)?;
        self.check_vector(metadata, metadata.len())?;
        self.check_vector(tiles, tiles.len())?;
        if metadata.len() < caches.len() * 12 { return Err("ragged prefill metadata too small".into()); }
        let mut descriptors = Vec::with_capacity(caches.len() * 12);
        let mut blocks = Vec::new();
        let group = heads / kv_heads;
        let tile_rows = if group == 4 || group == 8 { 16 } else if group <= 8 && 8 % group == 0 { 128 / group } else { 64 };
        let mut first = 0;
        for (i, c) in caches.iter().enumerate() {
            self.check_vector(c.keys, c.keys.len)?; self.check_vector(c.values, c.keys.len)?;
            let base = c.prefix.map_or(0, |(_, _, base)| base);
            if c.position < base || c.position + c.rows - base > c.format.element_capacity(c.keys.len) / (kv_heads * 128) {
                return Err("ragged prefill exceeds cache".into());
            }
            let (k, v) = (c.keys.buffer.address, c.values.buffer.address);
            let (pk, pv, base) = c.prefix.map_or((k, v, 0), |(k, v, base)| (k.buffer.address, v.buffer.address, base));
            descriptors.extend([k as u32, (k >> 32) as u32, v as u32, (v >> 32) as u32,
                c.position as u32, c.rows as u32, first as u32, pk as u32, (pk >> 32) as u32,
                pv as u32, (pv >> 32) as u32, base as u32].map(f32::from_bits));
            for tile in 0..c.rows.div_ceil(tile_rows) { blocks.extend([i as u32, tile as u32].map(f32::from_bits)); }
            first += c.rows;
        }
        if tiles.len() < blocks.len() { return Err("ragged prefill tiles too small".into()); }
        self.write_vector(&mut metadata.slice(0..descriptors.len())?, &descriptors)?;
        self.write_vector(&mut tiles.slice(0..blocks.len())?, &blocks)?;
        self.require_multi_kernels()?;
        let mut args = crate::kernel_params::PrefillAttentionMulti { query: query.buffer.address,
            metadata: metadata.buffer.address, tiles: tiles.buffer.address, output: output.buffer.address,
            heads: heads as u32, kv_heads: kv_heads as u32, exp2: u32::from(heads == kv_heads * 8) };
        let (_, gy, threads, shared) = Self::prefill_attention_grid(1, heads, kv_heads, true);
        unsafe { self.enqueue_dims_shared(self.multi.borrow().as_ref().unwrap().prefill, (blocks.len()/2) as u32,
            gy, 1, threads, shared, &mut [parameter(&mut args)]) }
    }
    /// Compact scratch in FP32 words: scores, partials/summaries, tile descriptors.
    pub fn multi_attention_scratch(&self, lengths: impl Iterator<Item = (usize, KvFormat)>, heads: usize, kv_heads: usize, dim: usize) -> [usize; 3] {
        let (mut scores, mut parts, mut tiles) = (0, 0, 0);
        for (length, format) in lengths {
            let (span, chunks) = row_shape(length, format, heads, kv_heads, dim);
            if span == 0 { scores += length*heads; }
            parts += chunks*heads*if span == 0 {dim} else {132};
            tiles += chunks*2;
        }
        [scores.max(1), parts.max(1), tiles.max(1)]
    }
    /// Every row owns its score/summary range and every tile identifies a row.
    /// Long FP32 rows share lcdecode's streaming reductions with scalar decode.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_multi_device(&self, query: &mut DeviceVector<'_>, key: &DeviceVector<'_>, value: &DeviceVector<'_>,
        q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>, caches: &mut [CacheRows<'_, '_>],
        metadata: &mut DeviceVector<'_>, tiles: &mut DeviceVector<'_>, scores: &mut DeviceVector<'_>, parts: &mut DeviceVector<'_>, output: &mut DeviceVector<'_>,
        heads: usize, kv_heads: usize, dim: usize, epsilon: f32) -> Result<(), String> {
        self.require_multi_kernels()?;
        let rows = caches.iter().map(|c| c.rows).sum::<usize>();
        if rows == 0 || rows > 2048 || heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || dim == 0 || !dim.is_multiple_of(2) {
            return Err("invalid multi attention dimensions".into());
        }
        self.check_vector(query, rows*heads*dim)?; self.check_vector(key, rows*kv_heads*dim)?;
        self.check_vector(value, key.len)?; self.check_vector(output, query.len)?;
        self.check_vector(q_norm, dim)?; self.check_vector(k_norm, dim)?; self.check_vector(rope, rope.len)?;
        self.check_vector(metadata, rows*16)?;
        for buffer in [&*tiles, &*scores, &*parts] { self.check_vector(buffer, buffer.len)?; }
        // Only alias identities authorize shared reads. Equal tokens alone do
        // not prove that two caches contain the same computed KV.
        // A private owner must not disable reuse for its shared followers.
        let mut cascades = Vec::new();
        let mut scalar_cascades = Vec::new();
        let mut shared_chunks = vec![0; rows];
        if (2..=16).contains(&rows) && dim == 128
            && caches.iter().all(|c| c.rows == 1 && c.format == KvFormat::Bf16) {
            let mut first = 0;
            while first < rows {
                let a = &caches[first];
                let span = row_shape(a.position + 1, a.format, heads, kv_heads, dim).0;
                // Short GQA2 draft attention owns the complete row. All other
                // groups share their existing split, including scalar GQA5.
                if span == u32::MAX as usize { first += 1; continue; }
                let (ak, av, ab) = a.prefix.unwrap_or((&*a.keys, &*a.values, a.position));
                let (mut end, mut chunks) = (first + 1, usize::MAX);
                while end < rows {
                    let b = &caches[end];
                    if row_shape(b.position + 1, b.format, heads, kv_heads, dim).0 != span { break; }
                    let (bk, bv, bb) = b.prefix.unwrap_or((&*b.keys, &*b.values, b.position));
                    let common = (ak.shared_prefix_bytes(bk).min(av.shared_prefix_bytes(bv)) / (kv_heads * dim * 2))
                        .min(ab).min(bb).min(a.position).min(b.position) / if span == 0 { 64 } else { span };
                    if common == 0 { break; }
                    chunks = chunks.min(common);
                    end += 1;
                }
                if end > first + 1 {
                    shared_chunks[first..end].fill(chunks);
                    if span == 0 { scalar_cascades.push((first, end - first, chunks)); }
                    else { cascades.push((first, end - first, chunks)); }
                }
                first = end;
            }
        }
        if self.profiling.get() && rows == 16 {
            let aliases: Vec<_> = caches.iter().map(|a| {
                let (ak, av, ab) = a.prefix.unwrap_or((&*a.keys, &*a.values, a.position));
                caches.iter().filter(|b| {
                    let (bk, bv, bb) = b.prefix.unwrap_or((&*b.keys, &*b.values, b.position));
                    (ak.shared_prefix_bytes(bk).min(av.shared_prefix_bytes(bv)) / (kv_heads * dim * 2)).min(ab).min(bb) >= 256
                }).count()
            }).collect();
            eprintln!("shared_attention_profile position={} cascades={cascades:?} scalar_cascades={scalar_cascades:?} alias_rows={aliases:?}", caches[0].position);
        }
        let mut descriptors = Vec::with_capacity(rows*16);
        let (mut legacy, mut streaming, mut draft) = (Vec::new(), Vec::new(), Vec::new());
        let mut grouped = Vec::new();
        let (mut score_offset, mut part_offset, mut row) = (0, 0, 0u32);
        for cache in caches.iter_mut() {
            if cache.format == KvFormat::ScaledI16 && dim != 128 { return Err("scaled I16 KV requires head dimension 128".into()); }
            self.check_vector(cache.keys, cache.keys.len)?; self.check_vector(cache.values, cache.keys.len)?;
            let end = cache.position.checked_add(cache.rows).ok_or("cache position overflow")?;
            let (pk, pv, base) = cache.prefix.map_or((cache.keys.buffer.address, cache.values.buffer.address, 0),
                |(k, v, base)| (k.buffer.address, v.buffer.address, base));
            if cache.rows == 0 || end > rope.len/dim || cache.position < base || end - base > cache.format.element_capacity(cache.keys.len)/(kv_heads*dim) {
                return Err("multi attention exceeds cache".into());
            }
            let (k,v) = (cache.keys.buffer.address, cache.values.buffer.address);
            let mut position = cache.position;
            while position < end {
                let (span, chunks) = row_shape(position+1, cache.format, heads, kv_heads, dim);
                if span != 0 && chunks > 1024 { return Err("streaming attention exceeds summary capacity".into()); }
                // A shared pass needs a constant summary stride for every row.
                let count = if span != 0 && heads == kv_heads * 4 && cache.format == KvFormat::Fp32 && position >= 16384 {
                    (end-position).min(16).min(chunks*span-position)
                } else { 1 };
                if count > 1 { grouped.push((row, count, k, v, position, part_offset, span, chunks)); }
                for at in position..position+count {
                    descriptors.extend([k as u32, (k>>32) as u32, v as u32, (v>>32) as u32, at as u32,
                        cache.format as u32, score_offset as u32, part_offset as u32, span as u32, chunks as u32,
                        pk as u32, (pk >> 32) as u32, pv as u32, (pv >> 32) as u32, base as u32, 0].map(f32::from_bits));
                    if count == 1 {
                        let list = if span == 0 { &mut legacy } else if span == u32::MAX as usize { &mut draft } else { &mut streaming };
                        for chunk in shared_chunks[row as usize]..chunks { list.extend([row, chunk as u32].map(f32::from_bits)); }
                    }
                    if span == 0 { score_offset += (at+1)*heads; }
                    part_offset += chunks*heads*if span == 0 {dim} else {132};
                    row += 1;
                }
                position += count;
            }
        }
        if score_offset > scores.len || part_offset > parts.len || legacy.len()+streaming.len()+draft.len() > tiles.len {
            return Err("multi attention scratch too small".into());
        }
        let legacy_tiles = legacy.len()/2;
        let stream_tiles = streaming.len()/2;
        let draft_tiles = draft.len()/2;
        legacy.extend(streaming);
        legacy.extend(draft);
        self.write_vector(metadata, &descriptors)?;
        if !legacy.is_empty() { self.write_vector(&mut tiles.slice(0..legacy.len())?, &legacy)?; }
        let kernels = self.multi.borrow(); let kernels = kernels.as_ref().unwrap();
        let (mut q, k, v, mut meta) = (query.buffer.address, key.buffer.address, value.buffer.address, metadata.buffer.address);
        let (qw, kw, r, eps) = (q_norm.buffer.address, k_norm.buffer.address, rope.buffer.address, epsilon);
        let (mut h, mut kh, mut d) = (heads as u32, kv_heads as u32, dim as u32);
        let (mut tile, mut scores, mut parts, mut out) = (tiles.buffer.address, scores.buffer.address, parts.buffer.address, output.buffer.address);
        unsafe {
            let mut qk = crate::kernel_params::MultiQk { query: q, key: k, value: v,
                q_weight: qw, k_weight: kw, rope: r, metadata: meta, heads: h, kv_heads: kh, dim: d, epsilon: eps };
            self.enqueue_grid(kernels.qk, (heads+kv_heads)*32, rows as u32, 1,
                &mut [parameter(&mut qk)])?;
            let scalar_prefix = move |values| -> Result<(), String> {
                for &(first, count, chunks) in &scalar_cascades {
                    let mut args = crate::kernel_params::Cascade { query: q + (first * heads * 128 * 4) as u64,
                        metadata: meta + (first * 16 * 4) as u64, parts, scores,
                        kv_heads: kh, rows: count as u32, heads: h, values };
                    self.profile_phase(if values == 0 { 3 } else { 5 }, ||
                        self.enqueue_dims(kernels.scalar_cascade, chunks as u32, kh, 1, 256, &mut [parameter(&mut args)]))?;
                }
                Ok(())
            };
            if legacy_tiles > 0 {
                self.enqueue_dims(kernels.scores, legacy_tiles as u32, kh, 1, 128,
                    &mut [parameter(&mut q),parameter(&mut meta),parameter(&mut tile),parameter(&mut scores),parameter(&mut h),parameter(&mut kh),parameter(&mut d)])?;
                scalar_prefix(0)?;
                self.enqueue_grid(kernels.softmax, heads*32, rows as u32, 1,
                    &mut [parameter(&mut scores),parameter(&mut meta),parameter(&mut h)])?;
                self.enqueue_dims(kernels.values, legacy_tiles as u32, kh, 1, dim.div_ceil(32).min(32) as u32*32,
                    &mut [parameter(&mut scores),parameter(&mut meta),parameter(&mut tile),parameter(&mut parts),parameter(&mut h),parameter(&mut kh),parameter(&mut d)])?;
                scalar_prefix(1)?;
                let mut width = (heads*dim) as u32;
                self.enqueue_grid(kernels.merge, heads*dim, rows as u32, 1,
                    &mut [parameter(&mut parts),parameter(&mut meta),parameter(&mut out),parameter(&mut width)])?;
            }
            if stream_tiles > 0 {
                let stream_tile = tile + (legacy_tiles*8) as u64;
                let bf16 = caches.iter().all(|cache| cache.format == KvFormat::Bf16);
                let groups = if bf16 { kh } else { h / 4 };
                let threads = if bf16 { 64 } else { 128 };
                let mut args = crate::kernel_params::MultiStream { query: q, metadata: meta,
                    tiles: stream_tile, parts, kv_heads: kh, heads: h };
                self.enqueue_dims(kernels.stream, stream_tiles as u32, groups, 1, threads,
                    &mut [parameter(&mut args)])?;
            }
            for &(first, count, chunks) in &cascades {
                let mut args = crate::kernel_params::Cascade { query: q + (first * heads * 128 * 4) as u64,
                    metadata: meta + (first * 16 * 4) as u64, parts, scores,
                    kv_heads: kh, rows: count as u32, heads: h, values: 0 };
                self.enqueue_dims(kernels.cascade, chunks as u32, kh, 1,
                    (count * heads / kv_heads).div_ceil(16) as u32 * 32, &mut [parameter(&mut args)])?;
            }
            if draft_tiles > 0 {
                let draft_tile = tile + ((legacy_tiles+stream_tiles)*8) as u64;
                let mut args = crate::kernel_params::DraftAttention { query: q, metadata: meta,
                    tiles: draft_tile, output: out, kv_heads: kh };
                self.enqueue_dims(kernels.draft_attention, draft_tiles as u32, kh, 1, 256,
                    &mut [parameter(&mut args)])?;
            }
            for &(row, count, mut k, mut v, base, offset, span, chunks) in &grouped {
                let mut query = q + row as u64 * heads as u64 * 128 * 4;
                let mut target = parts + offset as u64 * 4;
                let (mut base, mut count, mut span, mut chunks) = (base as u32, count as u32, span as u32, chunks as u32);
                self.enqueue_dims_shared(kernels.verify, chunks, kh, 1, count * 32, count * 4096 + 32768,
                    &mut [parameter(&mut query),parameter(&mut k),parameter(&mut v),parameter(&mut target),parameter(&mut kh),
                        parameter(&mut base),parameter(&mut chunks),parameter(&mut span),parameter(&mut count)])?;
            }
            if stream_tiles > 0 || !cascades.is_empty() || !grouped.is_empty() {
                self.enqueue_dims(kernels.stream_merge, h, rows as u32, 1, 128,
                    &mut [parameter(&mut parts),parameter(&mut meta),parameter(&mut out),parameter(&mut h)])?;
            }
        }
        Ok(())
    }
    /// Prepare short GQA2 rows once before graph capture; KV addresses remain
    /// indirect so cache growth and changing positions do not invalidate kernels.
    pub fn prepare_draft_metadata(&self, caches: &[CacheRows<'_, '_>], metadata: &mut DeviceVector<'_>,
        heads: usize, kv_heads: usize) -> Result<(), String> {
        let mut descriptors = Vec::new();
        let mut row = 0;
        for c in caches {
            if c.rows == 0 || c.position + c.rows > 512 || c.format != KvFormat::Bf16 || heads != kv_heads * 2
                || c.position < c.prefix.map_or(0, |(_, _, base)| base) {
                return Err("captured draft requires short BF16 GQA2 decode rows".into());
            }
            let (k,v) = (c.keys.buffer.address, c.values.buffer.address);
            let (pk,pv,base) = c.prefix.map_or((k,v,0), |(k,v,base)| (k.buffer.address,v.buffer.address,base));
            for position in c.position..c.position+c.rows {
                descriptors.extend([k as u32,(k>>32) as u32,v as u32,(v>>32) as u32,position as u32,
                    1,0,(row*heads*132) as u32,u32::MAX,1,pk as u32,(pk>>32) as u32,
                    pv as u32,(pv>>32) as u32,base as u32,0].map(f32::from_bits));
                row += 1;
            }
        }
        self.write_vector(metadata, &descriptors)
    }
    /// Replay the same canonical short-row attention launches with fresh metadata.
    pub fn draft_attention_prepared_device(&self, query: &DeviceVector<'_>, key: &DeviceVector<'_>,
        value: &DeviceVector<'_>, q_norm: &DeviceVector<'_>, k_norm: &DeviceVector<'_>, rope: &DeviceVector<'_>,
        metadata: &DeviceVector<'_>, tiles: &DeviceVector<'_>, output: &DeviceVector<'_>,
        heads: usize, kv_heads: usize, rows: usize, epsilon: f32) -> Result<(), String> {
        let kernels = self.multi.borrow(); let kernels = kernels.as_ref().unwrap();
        let mut qk = crate::kernel_params::MultiQk { query: query.buffer.address, key: key.buffer.address,
            value: value.buffer.address, q_weight: q_norm.buffer.address, k_weight: k_norm.buffer.address,
            rope: rope.buffer.address, metadata: metadata.buffer.address, heads: heads as u32,
            kv_heads: kv_heads as u32, dim: 128, epsilon };
        let mut args = crate::kernel_params::DraftAttention { query: query.buffer.address,
            metadata: metadata.buffer.address, tiles: tiles.buffer.address, output: output.buffer.address, kv_heads: kv_heads as u32 };
        unsafe {
            self.enqueue_grid(kernels.qk, (heads+kv_heads)*32, rows as u32, 1, &mut [parameter(&mut qk)])?;
            self.enqueue_dims(kernels.draft_attention, rows as u32, kv_heads as u32, 1, 256,
                &mut [parameter(&mut args)])
        }
    }
    pub fn gather_output_rows_device(&self, input: &DeviceVector<'_>, indices: &DeviceVector<'_>,
        output: &DeviceVector<'_>, width: usize, rows: usize) -> Result<(), String> {
        self.check_vector(output, rows*width)?; self.check_vector(indices, rows)?;
        let mut p = crate::kernel_params::GatherRows { input: input.buffer.address, indices: indices.buffer.address,
            output: output.buffer.address, width: width as u32, rows: rows as u32 };
        unsafe { self.enqueue(self.multi.borrow().as_ref().unwrap().gather_rows, rows*width, &mut [parameter(&mut p)]) }
    }
    /// One completion boundary for every row; only ID, score and validity return.
    pub fn argmax_batch_device(&self, logits: &DeviceVector<'_>, partials: &mut DeviceVector<'_>,
        result: &mut DeviceVector<'_>, rows: usize) -> Result<Vec<Result<(u32, f32), String>>, String> {
        self.enqueue_argmax_batch_device(logits, partials, result, rows)?;
        self.read_argmax_batch_device(result)
    }
    /// Read selections already queued eagerly or by a whole-step graph.
    pub fn read_argmax_batch_device(&self, result: &DeviceVector<'_>) -> Result<Vec<Result<(u32, f32), String>>, String> {
        Ok(self.read_vector_values(result)?.as_chunks::<3>().0.iter().map(|r| {
            if r[2] != 1.0 || !r[0].is_finite() || !r[1].is_finite() { Err("nonfinite logits".into()) } else { Ok((r[1] as u32, r[0])) }
        }).collect())
    }
    /// Queue every selected ID without introducing a host completion boundary.
    pub fn enqueue_argmax_batch_device(&self, logits: &DeviceVector<'_>, partials: &mut DeviceVector<'_>,
        result: &mut DeviceVector<'_>, rows: usize) -> Result<(), String> {
        if rows == 0 || !logits.len.is_multiple_of(rows) { return Err("invalid argmax batch".into()); }
        let width = logits.len / rows;
        if width == 0 || width > 1 << 24 { return Err("argmax ID exceeds exact FP32 integer range".into()); }
        let count = width.div_ceil(1024);
        self.check_vector(logits, rows * width)?;
        self.check_vector(partials, rows * count * 3)?;
        self.check_vector(result, rows * 3)?;
        let kernels = self.layer_kernels()?;
        {
            let (mut x, mut p, mut y) = (logits.buffer.address, partials.buffer.address, result.buffer.address);
            let (mut width, mut count) = (width as u32, count as u32);
            unsafe {
                self.enqueue_grid(kernels.argmax, count as usize * 32, rows as u32, 1,
                    &mut [parameter(&mut x), parameter(&mut p), parameter(&mut width)])?;
                self.enqueue_grid(kernels.argmax_finish, 32, rows as u32, 1,
                    &mut [parameter(&mut p), parameter(&mut y), parameter(&mut count)])?;
            }
        }
        Ok(())
    }
    /// Compact the surviving draft cohort's previous device-selected IDs.
    pub fn selected_ids_device(&self, result: &DeviceVector<'_>, indices: &[usize], ids: &mut DeviceVector<'_>) -> Result<(), String> {
        self.require_multi_kernels()?;
        if indices.is_empty() || result.len/3 > 1 << 24 || indices.iter().any(|&i| i >= result.len/3) { return Err("selected ID row outside result".into()); }
        self.check_vector(result, result.len)?;
        self.check_vector(ids, indices.len())?;
        self.write_vector(ids, &indices.iter().map(|&i| i as f32).collect::<Vec<_>>())?;
        let mut args = crate::kernel_params::SelectedIds { result: result.buffer.address, ids: ids.buffer.address,
            rows: u32::try_from(indices.len()).map_err(|_| "selected ID count exceeds kernel ABI")?, result_rows: (result.len/3) as u32 };
        unsafe { self.enqueue(self.multi.borrow().as_ref().unwrap().selected_ids, indices.len(),
            &mut [parameter(&mut args)]) }
    }
}

impl Backend<'_> {
    /// Release spare buffers before checking room for an optional cache copy.
    /// Live session buffers and captured execution storage keep their owners.
    pub fn prepare_cache_allocation(&self, bytes: usize) -> Result<(), String> {
        self.context.activate()?;
        self.context.release_deferred();
        self.context.make_room(bytes).and_then(|()| self.fatal_error().map_or(Ok(()), Err))
    }
}
