//! Persistent Qwen activations and selectable FP32/BF16 KV per session.
use super::{Layer, Mlp, Model, Result};
use engine_model::qwen3::Config;
use spark_gpu_probe::driver::{Backend, DecodeGraph, DeviceVector, KvFormat, Matrix, MoeScratch, MoeBatchScratch, MOE_ROUTING_ERROR};
use std::{rc::Rc, time::{Duration, Instant}};

pub(super) type Observer<'a> = Option<&'a mut dyn FnMut(&str, &[f32]) -> Result<()>>;
pub(super) type RoutingObserver<'a> = Option<&'a mut dyn FnMut(usize, usize, &[(u32, f32)]) -> Result<()>>;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Logits { None, Last, All }

struct Norms<'a> {
    input: DeviceVector<'a>, post: DeviceVector<'a>, q: DeviceVector<'a>, k: DeviceVector<'a>,
}
pub(super) struct DeviceModel<'a> {
    pub embedding: Option<Matrix<'a>>, norm: Option<DeviceVector<'a>>,
    norms: Vec<Norms<'a>>, rope: DeviceVector<'a>,
}
impl<'a> DeviceModel<'a> {
    pub(super) fn projection_norms(&self, index: usize) -> crate::layer::Norms<'_, 'a> {
        let n = &self.norms[index];
        crate::layer::Norms { input: &n.input, post: &n.post, q: &n.q, k: &n.k,
            next: self.norms.get(index + 1).map(|n| &n.input) }
    }
    pub(super) fn rope(&self) -> &DeviceVector<'a> { &self.rope }
    pub(super) fn layer_norms(&self, layer: usize) -> (&DeviceVector<'a>, &DeviceVector<'a>) {
        (&self.norms[layer].input, &self.norms[layer].post)
    }
    pub(super) fn final_norm(&self) -> Result<&DeviceVector<'a>> {
        self.norm.as_ref().ok_or_else(|| "partition does not own final norm".into())
    }
    pub(super) fn attention_inputs(&self, layer: usize) -> (&DeviceVector<'a>, &DeviceVector<'a>, &DeviceVector<'a>) {
        (&self.norms[layer].q, &self.norms[layer].k, &self.rope)
    }
    pub fn new(backend: &Backend<'a>, config: &Config, layers: &[Layer<'a>],
        norm: Option<&[f32]>, embedding: Option<Matrix<'a>>, max_tokens: usize) -> Result<Self> {
        backend.require_layer_kernels()?;
        let norms = layers.iter().map(|layer| Ok(Norms {
            input: backend.upload_vector(&layer.input_norm)?, post: backend.upload_vector(&layer.post_norm)?,
            q: backend.upload_vector(&layer.q_norm)?, k: backend.upload_vector(&layer.k_norm)?,
        })).collect::<Result<Vec<_>>>()?;
        Ok(Self { embedding, norm: norm.map(|n| backend.upload_vector(n)).transpose()?,
            norms, rope: crate::layer::rope(backend, config, max_tokens)? })
    }
}

pub(super) struct Kv<'a> { keys: DeviceVector<'a>, values: DeviceVector<'a>, prefix: Option<Rc<Vec<Kv<'a>>>>, base: usize }
pub(super) fn cache_bytes(kv: &[Kv<'_>]) -> usize { cache_allocations(kv).iter().map(|(_, bytes)| bytes).sum() }
pub(super) fn cache_allocations(kv: &[Kv<'_>]) -> Vec<(usize, usize)> {
    let mut allocations = Vec::new();
    for layer in kv { layer.keys.allocations(&mut allocations); layer.values.allocations(&mut allocations); }
    if let Some(prefix) = kv.first().and_then(|kv| kv.prefix.as_ref()) { allocations.extend(cache_allocations(prefix)); }
    allocations.sort_unstable_by_key(|&(id, _)| id);
    allocations.dedup_by_key(|entry| entry.0);
    allocations
}
/// Retain one rounded prefix copy; admission separately funds tail growth
/// and materialization. Logical shortening within that copy needs no remap.
pub(super) fn snapshot_fits(model: &Model<'_, '_>, kv: &[Kv<'_>], format: KvFormat, tokens: usize) -> bool {
    let page = 2 * 1024 * 1024;
    let bytes = format.storage_len(tokens * model.kv_capacity.stride) * 4;
    let capacity = if bytes < page {
        (format.storage_len(tokens.max(1).next_power_of_two().min(model.max_tokens) * model.kv_capacity.stride) * 4).div_ceil(64) * 64
    } else { bytes.div_ceil(page) * page };
    cache_bytes(kv) <= capacity * kv.len() * 2
}
pub(super) fn prefix_allocations(kv: &[Kv<'_>], words: usize, words_per_token: usize) -> Vec<(usize, usize)> {
    let mut allocations = Vec::new();
    for (i, layer) in kv.iter().enumerate() {
        if let Some(prefix) = &layer.prefix {
            // Prefix vectors retain their original complete physical pages.
            let retained = words.min(layer.base * words_per_token);
            prefix[i].keys.prefix_allocations(retained, &mut allocations);
            prefix[i].values.prefix_allocations(retained, &mut allocations);
            layer.keys.prefix_allocations(words.saturating_sub(retained), &mut allocations);
            layer.values.prefix_allocations(words.saturating_sub(retained), &mut allocations);
        } else {
            layer.keys.prefix_allocations(words, &mut allocations); layer.values.prefix_allocations(words, &mut allocations);
        }
    }
    allocations
}
pub(super) struct DeviceState<'a> {
    graphs: Vec<(usize, std::ops::Range<usize>, DecodeGraph<'a>)>,
    graph_input: Option<DeviceVector<'a>>,
    packed_experts: bool,
    pub hidden: DeviceVector<'a>, input: DeviceVector<'a>,
    norm_sums: DeviceVector<'a>, input_pack: DeviceVector<'a>, weight_pack: DeviceVector<'a>,
    q: DeviceVector<'a>, k: DeviceVector<'a>, v: DeviceVector<'a>,
    attention: DeviceVector<'a>, projected: DeviceVector<'a>,
    gate: DeviceVector<'a>, up: DeviceVector<'a>,
    moe: Option<MoeScratch<'a>>,
    scores: DeviceVector<'a>, attention_parts: DeviceVector<'a>, kv: Rc<Vec<Kv<'a>>>,
    pub logits: DeviceVector<'a>, partials: DeviceVector<'a>, result: DeviceVector<'a>,
    pub(super) top_k: Option<spark_gpu_probe::driver::TopKScratch<'a>>,
    format: KvFormat, batch: Option<BatchState<'a>>, sequence: Option<SequenceState<'a>>, layer_cost: Duration, head_cost: Duration, pub peak_bytes: usize,
}
impl<'a> DeviceState<'a> {
    pub(super) fn split_cache(&self) -> bool { self.kv[0].prefix.is_some() }
    pub(super) fn cache_rows(&mut self, layer: usize, position: usize, rows: usize) -> Result<spark_gpu_probe::driver::CacheRows<'_, 'a>> {
        let kv = &mut Rc::get_mut(&mut self.kv).ok_or("multi forward requires reserved KV")?[layer];
        let prefix = kv.prefix.as_ref().map(|p| (&p[layer].keys, &p[layer].values, kv.base));
        Ok(spark_gpu_probe::driver::CacheRows { keys: &mut kv.keys, values: &mut kv.values, prefix, position, rows, format: self.format })
    }
    pub fn new(model: &Model<'a, '_>, kv_format: Option<KvFormat>) -> Result<Self> {
        Self::with_cache(model, kv_format, None)
    }
    pub(super) fn with_cache(model: &Model<'a, '_>, kv_format: Option<KvFormat>, shared: Option<Rc<Vec<Kv<'a>>>>) -> Result<Self> {
        let b = model.backend;
        let c = &model.config;
        let packed_experts = c.num_experts != 0 && c.hidden_size == 2048
            && c.attention_heads * c.head_dim == 4096 && c.kv_heads * c.head_dim == 512
            && model.layer_start == 0 && model.layers.len() == c.layers as usize;
        let hidden = c.hidden_size as usize;
        let query = (c.attention_heads * c.head_dim) as usize;
        let kv_width = model.kv_capacity.stride;
        let intermediate = if c.num_experts == 0 { c.intermediate_size as usize } else { 1 };
        let up_scratch = if c.num_experts == 0 { intermediate } else if model.layers.iter().all(|layer| match &layer.mlp {
            Mlp::Dense { gate, up, .. } => b.matrix_is_packed(gate) && b.matrix_is_packed(up),
            Mlp::Experts { .. } => true,
        }) {1} else {intermediate};
        let moe = model.layers.iter().find_map(|layer| match &layer.mlp {
            Mlp::Experts { experts, .. } => Some(b.moe_scratch(experts, c.num_experts_per_tok as usize)),
            _ => None,
        }).transpose()?;
        let vocabulary = if model.head.is_some() { c.vocab_size as usize } else { 1 };
        let format = kv_format.unwrap_or(KvFormat::Bf16);
        let streaming = c.head_dim == 128 && (c.attention_heads == c.kv_heads * 4 || c.attention_heads == c.kv_heads * 8
            || c.attention_heads == c.kv_heads * 2 && format == KvFormat::Bf16 && model.max_tokens <= 4096);
        let kv = if let Some(shared) = shared { shared } else {
            let mut buffers = b.kv_arena(&vec![format.storage_len(kv_width); model.layers.len()*2])?.into_iter();
            Rc::new((0..model.layers.len()).map(|_| Kv { keys: buffers.next().unwrap(), values: buffers.next().unwrap(), prefix: None, base: 0 }).collect())
        };
        let mut scratch = b.vector_arena(&[hidden, hidden, hidden.div_ceil(128), query, kv_width, kv_width,
            query, hidden, intermediate, up_scratch, if streaming {1} else {c.attention_heads as usize * model.max_tokens},
            if streaming {1} else {query * model.max_tokens.div_ceil(64)}, vocabulary, vocabulary.div_ceil(1024)*3,
            if c.num_experts == 0 || packed_experts { 8 * intermediate.max(query).max(hidden) } else { 1 }, 1])?.into_iter();
        let mut next = || scratch.next().unwrap();
        Ok(Self {
            graphs: Vec::new(), graph_input: None, packed_experts, top_k: None,
            hidden: next(), input: next(), norm_sums: next(), q: next(), k: next(), v: next(), moe,
            attention: next(), projected: next(), gate: next(), up: next(), scores: next(), kv,
            attention_parts: next(), logits: next(), partials: next(), input_pack: next(), weight_pack: next(), result: b.vector(3)?,
            format, batch: None, sequence: None, layer_cost: Duration::ZERO, head_cost: Duration::from_millis(5), peak_bytes: b.allocated_bytes(),
        })
    }
    pub fn cache_bytes(&self) -> usize { cache_bytes(&self.kv) }
    pub fn cache_allocations(&self) -> Vec<(usize, usize)> { cache_allocations(&self.kv) }
    pub(super) fn allocations(&self) -> Vec<(usize, usize)> {
        let mut out = self.cache_allocations();
        for vector in [&self.hidden, &self.input, &self.norm_sums, &self.input_pack, &self.weight_pack,
            &self.q, &self.k, &self.v, &self.attention, &self.projected, &self.gate, &self.up,
            &self.scores, &self.attention_parts, &self.logits, &self.partials, &self.result] {
            vector.allocations(&mut out);
        }
        if let Some(input) = &self.graph_input { input.allocations(&mut out); }
        if let Some(moe) = &self.moe { moe.allocations(&mut out); }
        if let Some(top_k) = &self.top_k { top_k.allocations(&mut out); }
        if let Some(batch) = &self.batch {
            for vector in [&batch.ids, &batch.hidden, &batch.input, &batch.sums, &batch.q, &batch.k,
                &batch.v, &batch.attention, &batch.projected, &batch.gate, &batch.up, &batch.scores,
                &batch.parts, &batch.weight_pack, &batch.input_pack, &batch.logits, &batch.packed_keys, &batch.packed_values] {
                vector.allocations(&mut out);
            }
            if let Some(moe) = &batch.moe { moe.allocations(&mut out); }
        }
        if let Some(sequence) = &self.sequence { sequence.hidden.allocations(&mut out); }
        out
    }
    pub fn prefix_allocations(&self, model: &Model<'a, '_>, tokens: usize) -> Vec<(usize, usize)> {
        prefix_allocations(&self.kv, self.format.storage_len(tokens * model.kv_capacity.stride), self.format.storage_len(model.kv_capacity.stride))
    }
    pub fn reserve_prompt_cache(&mut self, model: &Model<'a, '_>, committed: usize, needed: usize) -> Result<()> {
        let reserved = self.kv[0].base + self.format.element_capacity(self.kv[0].keys.len()) / model.kv_capacity.stride;
        // Reserve shareable physical blocks only once, before the first chunk.
        // batch_run detaches any shared tail immediately before writing it.
        for page in reserved / 512 + 1..=needed.div_ceil(512) {
            self.reserve_cache(model, committed, (page * 512).min(needed))?;
        }
        Ok(())
    }
    pub fn reserve_cache(&mut self, model: &Model<'a,'_>, committed: usize, needed: usize) -> Result<()> {
        if let Some(prefix) = &self.kv[0].prefix {
            self.graphs.clear();
            let base = self.kv[0].base.min(committed);
            let sources = prefix.iter().flat_map(|kv| [&kv.keys, &kv.values]).collect::<Vec<_>>();
            let start = self.format.storage_len(base * model.kv_capacity.stride);
            let len = self.format.storage_len(needed * model.kv_capacity.stride);
            let mut buffers = model.backend.fork_vectors(&sources, start, len, self.format.storage_len(model.max_tokens * model.kv_capacity.stride))?.into_iter();
            self.peak_bytes = self.peak_bytes.max(model.backend.allocated_bytes());
            let mut replacement = Vec::with_capacity(self.kv.len());
            for source in self.kv.iter() {
                let mut next = Kv { keys: buffers.next().unwrap(), values: buffers.next().unwrap(), prefix: None, base: 0 };
                if committed > source.base {
                    let count = self.format.storage_len((committed - source.base) * model.kv_capacity.stride);
                    let offset = self.format.storage_len(source.base * model.kv_capacity.stride);
                    model.backend.copy_vector_range(&source.keys, &mut next.keys, 0, offset, count)?;
                    model.backend.copy_vector_range(&source.values, &mut next.values, 0, offset, count)?;
                }
                replacement.push(next);
            }
            self.kv = Rc::new(replacement);
        }
        if model.backend.decode_attention_parts(needed) > 1 && model.config.head_dim == 128
            && (model.config.attention_heads == model.config.kv_heads * 4
                || model.config.attention_heads == model.config.kv_heads * 8
                || model.config.attention_heads == model.config.kv_heads * 2 && self.format == KvFormat::Bf16 && needed <= 4096) {
            let parts = model.config.attention_heads as usize * (model.config.head_dim as usize + 4) * model.backend.decode_attention_parts(needed);
            if self.attention_parts.len() < parts {
                self.graphs.clear();
                let replacement = model.backend.vector(parts)?;
                self.peak_bytes = self.peak_bytes.max(model.backend.allocated_bytes());
                self.attention_parts = replacement;
            }
        }
        // Arena growth copies the prefix; keep it geometric until vectors can use pages.
        let arena = self.format.storage_len(needed * model.kv_capacity.stride) * 4 < 2 * 1024 * 1024;
        let capacity = if needed < 512 || arena { needed.max(1).next_power_of_two() } else { needed }.min(model.max_tokens);
        let len = self.format.storage_len(capacity * model.kv_capacity.stride);
        let copy = self.format.storage_len(committed * model.kv_capacity.stride);
        // A sharded compact cache can end inside an eight-head group. Its next
        // payload precedes that group's scale trailer, so detach the whole group.
        let shared = if self.format == KvFormat::ScaledI16 { committed * model.kv_capacity.stride / 1024 * 520 } else { copy };
        let writable = Rc::strong_count(&self.kv) == 1 && self.kv.iter().all(|kv| kv.keys.writable_from(shared) && kv.values.writable_from(shared));
        if writable && self.kv.iter().all(|kv| kv.keys.len() >= len && kv.values.len() >= len) { return Ok(()); }
        self.graphs.clear();
        if !writable && self.kv[0].keys.is_segmented() {
            let sources = self.kv.iter().flat_map(|kv| [&kv.keys, &kv.values]).collect::<Vec<_>>();
            let mut forks = model.backend.fork_vectors(&sources, shared, len, self.format.storage_len(model.max_tokens * model.kv_capacity.stride))?.into_iter();
            let replacement = self.kv.iter().map(|kv| {
                let mut next = Kv {
                    keys: forks.next().unwrap(), values: forks.next().unwrap(),
                    prefix: None, base: 0,
                };
                if shared < copy {
                    model.backend.copy_vector_range(&kv.keys, &mut next.keys, shared, shared, copy - shared)?;
                    model.backend.copy_vector_range(&kv.values, &mut next.values, shared, shared, copy - shared)?;
                }
                Ok(next)
            }).collect::<Result<Vec<_>>>()?;
            self.peak_bytes = self.peak_bytes.max(model.backend.allocated_bytes());
            self.kv = Rc::new(replacement);
            return Ok(());
        }
        if let Some(cache) = Rc::get_mut(&mut self.kv).filter(|kv| writable && kv[0].keys.is_segmented()) {
            for kv in cache {
                model.backend.grow_vector(&mut kv.keys, len)?;
                model.backend.grow_vector(&mut kv.values, len)?;
            }
            self.peak_bytes = self.peak_bytes.max(model.backend.allocated_bytes());
            return Ok(());
        }
        // Small contexts avoid CUDA's 2 MiB minimum page per vector. Once a
        // vector reaches a page, growth maps only its new physical segments.
        let mut buffers = if len * 4 >= 2 * 1024 * 1024 {
            let maximum = self.format.storage_len(model.max_tokens * model.kv_capacity.stride);
            (0..self.kv.len()*2).map(|_| model.backend.segmented_vector(maximum, len)).collect::<Result<Vec<_>>>()?
        } else { model.backend.kv_arena(&vec![len; self.kv.len()*2])? }.into_iter();
        let mut replacement = self.kv.iter().map(|_| Kv { keys: buffers.next().unwrap(), values: buffers.next().unwrap(), prefix: None, base: 0 }).collect::<Vec<_>>();
        self.peak_bytes = self.peak_bytes.max(model.backend.allocated_bytes());
        for (source, destination) in self.kv.iter().zip(&mut replacement) {
            model.backend.copy_prefix(&source.keys,&mut destination.keys,copy)?;
            model.backend.copy_prefix(&source.values,&mut destination.values,copy)?;
        }
        self.kv = Rc::new(replacement);
        Ok(())
    }
    /// Batched GQA4/GQA8 reads immutable prefix addresses directly, owning
    /// a compact private tail. Scalar/sequence callers materialize above.
    pub(super) fn reserve_multi_cache(&mut self, model: &Model<'a, '_>, committed: usize, needed: usize) -> Result<()> {
        if self.format != KvFormat::Bf16 || (model.config.attention_heads != model.config.kv_heads * 4
            && model.config.attention_heads != model.config.kv_heads * 8)
            || committed == 0 || self.kv[0].prefix.is_none() && Rc::strong_count(&self.kv) == 1 {
            return self.reserve_cache(model, committed, needed);
        }
        let base = self.kv[0].prefix.as_ref().map_or(committed, |_| self.kv[0].base.min(committed));
        let stride = model.kv_capacity.stride;
        let prefix = self.kv[0].prefix.as_ref().unwrap_or(&self.kv);
        // A short fork of a longer snapshot must shed uncommitted mappings;
        // the cache may evict its original owner while this request runs.
        if (self.kv[0].prefix.is_none() || Rc::strong_count(&self.kv) > 1)
            && !snapshot_fits(model, prefix, self.format, needed) {
            return self.reserve_cache(model, committed, needed);
        }
        // Long tails join the append-only paged cache before compact doubling
        // could retain a second long allocation during materialization.
        if (needed - base) * stride * 2 >= 2 * 1024 * 1024 {
            return self.reserve_cache(model, committed, needed);
        }
        let capacity = (needed - base).max(1).next_power_of_two().min(model.max_tokens - base);
        let len = self.format.storage_len(capacity * stride);
        if self.kv[0].base == base && Rc::strong_count(&self.kv) == 1
            && self.kv.iter().all(|kv| kv.keys.len() >= len) { return Ok(()); }
        let prefix = self.kv[0].prefix.clone().unwrap_or_else(|| self.kv.clone());
        self.graphs.clear();
        let mut buffers = model.backend.kv_arena(&vec![len; self.kv.len() * 2])?.into_iter();
        let mut replacement = Vec::with_capacity(self.kv.len());
        for source in self.kv.iter() {
            let mut next = Kv { keys: buffers.next().unwrap(), values: buffers.next().unwrap(), prefix: Some(prefix.clone()), base };
            if committed > source.base && source.prefix.is_some() {
                let count = self.format.storage_len((committed - source.base) * stride);
                model.backend.copy_vector_range(&source.keys, &mut next.keys, 0, 0, count)?;
                model.backend.copy_vector_range(&source.values, &mut next.values, 0, 0, count)?;
            }
            replacement.push(next);
        }
        self.peak_bytes = self.peak_bytes.max(model.backend.allocated_bytes());
        self.kv = Rc::new(replacement);
        Ok(())
    }
    pub(super) fn snapshot(&self) -> Rc<Vec<Kv<'a>>> { self.kv.clone() }
    pub fn observe(backend: &Backend<'a>, name: &str, vector: &DeviceVector<'a>, observer: &mut Observer<'_>) -> Result<()> {
        if let Some(observer) = observer { observer(name, &backend.read_vector(vector)?)?; }
        Ok(())
    }
    pub fn layers(&mut self, model: &Model<'a, '_>, device: &DeviceModel<'a>, position: usize, observer: &mut Observer<'_>) -> Result<()> {
        self.reserve_cache(model,position,position+1)?;
        self.layers_reserved(model, device, position, observer)
    }
    pub(super) fn layers_reserved(&mut self, model: &Model<'a, '_>, device: &DeviceModel<'a>, position: usize, observer: &mut Observer<'_>) -> Result<()> {
        let b = model.backend;
        let c = &model.config;
        let eps = c.norm_epsilon();
        b.rms_device(&self.hidden, &device.norms[0].input, &mut self.input, &mut self.norm_sums, eps)?;
        let first_experts = model.layers.iter().position(|layer| matches!(layer.mlp, Mlp::Experts { .. }));
        for (i, ((layer, norms), kv)) in model.layers.iter().zip(&device.norms).zip(Rc::get_mut(&mut self.kv).unwrap()).enumerate() {
            model.profile_layer(position + 1, self.format);
            if c.num_experts == 0 {
                b.matmul_device(&layer.q, &self.input, &mut self.q, &mut self.weight_pack, &mut self.input_pack, 1, false)?;
                b.matmul_reusing_input(&layer.k, &self.input, &mut self.k, &mut self.weight_pack, &mut self.input_pack, 1, false, true)?;
                b.matmul_reusing_input(&layer.v, &self.input, &mut self.v, &mut self.weight_pack, &mut self.input_pack, 1, false, true)?;
            } else if self.packed_experts {
                b.pack_prefill_bf16_device(&self.input, &mut self.input_pack, c.hidden_size as usize, 1)?;
                b.prefill_qkv_device(&layer.q, &layer.k, &layer.v, &self.input_pack,
                    &mut self.q, &mut self.k, &mut self.v, 1, false)?;
            } else { b.qkv_device(&layer.q, &layer.k, &layer.v, &self.input, &mut self.q, &mut self.k, &mut self.v)?; }
            if self.format != KvFormat::Fp32 {
                b.qk_batch_device(&mut self.q, &self.k, &self.v, &norms.q, &norms.k, &device.rope,
                    &mut kv.keys, &mut kv.values, position, c.qk_epsilon(), 1, self.format)?;
                b.attention_batch_device(&self.q, &kv.keys, &kv.values, &mut self.scores, &mut self.attention_parts, &mut self.attention,
                    c.attention_heads as usize, c.kv_heads as usize, c.head_dim as usize, position, 1, self.format)?;
            } else {
                b.qk_rope_cache_device(&mut self.q, &self.k, &self.v, &norms.q, &norms.k, &device.rope,
                &mut kv.keys, &mut kv.values, position, c.qk_epsilon())?;
            b.attention_device(&self.q, &kv.keys, &kv.values, &mut self.scores, &mut self.attention_parts, &mut self.attention,
                c.attention_heads as usize, c.kv_heads as usize, c.head_dim as usize, position + 1)?;
            }
            if c.num_experts == 0 {
                b.matmul_device(&layer.o, &self.attention, &mut self.projected, &mut self.weight_pack, &mut self.input_pack, 1, false)?;
            } else if self.packed_experts {
                b.pack_prefill_bf16_device(&self.attention, &mut self.input_pack, self.attention.len(), 1)?;
                b.matmul_packed_device(&layer.o, &self.input_pack, &mut self.projected, 1)?;
            } else { b.matvec_device(&layer.o, &self.attention, &mut self.projected)?; }
            b.residual_rms_device(&mut self.hidden, &self.projected, &norms.post, &mut self.input, &mut self.norm_sums, eps)?;
            match &layer.mlp {
                Mlp::Dense { gate, up, down } => {
                    let _class = b.prefill_class(1);
                    b.matmul_device(gate, &self.input, &mut self.gate, &mut self.weight_pack, &mut self.input_pack, 1, false)?;
                    b.matmul_reusing_input(up, &self.input, &mut self.up, &mut self.weight_pack, &mut self.input_pack, 1, false, true)?;
                    b.swiglu_precision_device(&mut self.gate, &self.up, !c.qk_norm)?;
                    b.matmul_device(down, &self.gate, &mut self.projected, &mut self.weight_pack, &mut self.input_pack, 1, false)?;
                }
                Mlp::Experts { router, experts } => {
                    let scratch = self.moe.as_mut().ok_or("missing expert scratch")?;
                    b.moe_device(router, experts, &self.input, &mut self.projected, scratch, c.norm_topk_prob, first_experts == Some(i))?;
                    if let Some(observer) = observer {
                        let routing: Vec<_> = b.moe_routing(scratch)?.into_iter()
                            .flat_map(|(id, weight)| [id as f32, weight]).collect();
                        observer(&format!("routing-{}", model.layer_start + i), &routing)?;
                    }
                }
            }
            if let Some(next) = device.norms.get(i + 1) {
                b.residual_rms_device(&mut self.hidden, &self.projected, &next.input, &mut self.input, &mut self.norm_sums, eps)?;
            } else { b.residual_checkpoint_device(&mut self.hidden, &self.projected, !c.qk_norm)?; }
            if observer.is_some() { Self::observe(b, &format!("layer-{}", model.layer_start + i), &self.hidden, observer)?; }
        }
        Ok(())
    }
    pub fn project(&mut self, model: &Model<'a, '_>, device: &DeviceModel<'a>, observer: &mut Observer<'_>) -> Result<()> {
        let b = model.backend;
        model.profile_head();
        b.rms_device(&self.hidden, device.norm.as_ref().ok_or("partition does not own final norm")?, &mut self.input, &mut self.norm_sums, model.config.norm_epsilon())?;
        Self::observe(b, "final_norm", &self.input, observer)?;
        b.profile_output_head(|| {
            let head = model.head.as_ref().ok_or("partition does not own output")?;
            if model.config.num_experts == 0 {
                b.matmul_device(head, &self.input, &mut self.logits, &mut self.weight_pack, &mut self.input_pack, 1, false)?;
            } else { b.matvec_device(head, &self.input, &mut self.logits)?; }
            b.round_logits_bf16(&mut self.logits)
        })
    }
    pub fn greedy(&mut self, backend: &Backend<'a>) -> Result<(u32, f32)> {
        let result = backend.argmax_device(&self.logits, &mut self.partials, &mut self.result)?;
        self.routing(backend)?;
        Ok(result)
    }
    /// After a forward's output read: fail its request if the router rejected the row.
    pub(super) fn routing(&self, backend: &Backend<'a>) -> Result<()> {
        match &self.moe {
            Some(moe) if backend.moe_rejected(moe)? => Err(MOE_ROUTING_ERROR.into()),
            _ => Ok(()),
        }
    }
    pub(super) fn reserve_greedy(&mut self, backend: &Backend<'a>, rows: usize) -> Result<()> {
        if self.result.len() < rows * 3 {
            self.graphs.clear();
            self.result = backend.vector(rows * 3)?;
            self.peak_bytes = self.peak_bytes.max(backend.allocated_bytes());
        }
        Ok(())
    }
    pub(super) fn greedy_enqueue(&mut self, backend: &Backend<'a>, row: usize) -> Result<()> {
        backend.enqueue_argmax_device(&self.logits, &mut self.partials, &mut self.result, row)
    }
    pub(super) fn embedding_selected(&mut self, model: &Model<'a, '_>, row: usize) -> Result<()> {
        model.backend.embedding_selected_device(model.device.embedding.as_ref().ok_or("missing draft embedding")?,
            &self.result, row, &mut self.hidden)
    }
    pub(super) fn greedy_results(&self, backend: &Backend<'a>, rows: usize) -> Result<Vec<(u32, f32)>> {
        let results = backend.read_argmax_results(&self.result, rows)?;
        self.routing(backend)?;
        Ok(results)
    }
    /// One graph for each active proposal width; scalar positions come from a
    /// 16-byte stream upload. Keep the canonical attention partition tree.
    pub(super) fn greedy_graph(&mut self, model: &Model<'a, '_>, position: usize, token: u32, rows: usize) -> Result<bool> {
        if model.backend.is_profiling() { return Ok(false); }
        let c = &model.config;
        let group = c.attention_heads / c.kv_heads;
        if self.format != KvFormat::Bf16 || c.head_dim != 128
            || !matches!(group, 2 | 4 | 8) || group == 2 && position + rows > 4096 {
            return Ok(false);
        }
        let b = model.backend;
        self.reserve_cache(model, position, position + rows)?;
        self.reserve_greedy(b, rows)?;
        b.prepare_decode_graph(c.hidden_size <= 2048 && !self.packed_experts, c.num_experts != 0)?;
        if self.graph_input.is_none() { self.graph_input = Some(b.vector(4)?); }
        b.write_vector(self.graph_input.as_mut().unwrap(), &[0.0, token as f32, 1.0, f32::from_bits(position as u32)])?;
        let cached = self.graphs.iter().position(|(count, window, _)| *count == rows && window.contains(&position));
        let index = if let Some(index) = cached { index } else {
            self.graphs.retain(|(count, _, _)| *count != rows);
            let window = (0..rows).fold(0..usize::MAX, |window, row| {
                let span = if group == 2 { 64 } else { b.decode_attention_span(position + row + 1) };
                let begin = (position + row) / span * span;
                window.start.max(begin.saturating_sub(row))..window.end.min(begin + span - row)
            });
            // Move only the handle while recording, so the closure can borrow
            // the session scratch. Storage is restored even on capture failure.
            let input = self.graph_input.take().unwrap();
            let captured = unsafe { b.capture_decode(&input, position, model.layers.len(), rows, || {
                for row in 0..rows {
                    if row == 0 {
                        b.embedding_selected_device(model.device.embedding.as_ref().ok_or("missing embedding")?, &input, 0, &mut self.hidden)?;
                    } else { self.embedding_selected(model, row - 1)?; }
                    self.layers_reserved(model, &model.device, position + row, &mut None)?;
                    self.project(model, &model.device, &mut None)?;
                    self.greedy_enqueue(b, row)?;
                }
                Ok(())
            }) };
            self.graph_input = Some(input);
            self.graphs.push((rows, window, captured?));
            self.graphs.len() - 1
        };
        // Cache growth and argmax reallocation clear all execs above. All other
        // captured scratch and the input buffer belong to this DeviceState.
        unsafe { b.replay_decode(&self.graphs[index].2)?; }
        Ok(true)
    }
}

/// All projections span T rows. Attention tiles queries to bound score scratch.
struct BatchProgress { position: usize, tokens: Vec<u32>, output: Logits, next_layer: usize }
struct BatchState<'a> {
    progress: Option<BatchProgress>,
    tokens: usize, capacity: usize, prompt_end: usize, all_logits: bool, exact: bool, fast_prefill: bool,
    ids: DeviceVector<'a>, hidden: DeviceVector<'a>, input: DeviceVector<'a>, sums: DeviceVector<'a>,
    q: DeviceVector<'a>, k: DeviceVector<'a>, v: DeviceVector<'a>, attention: DeviceVector<'a>,
    projected: DeviceVector<'a>, gate: DeviceVector<'a>, up: DeviceVector<'a>,
    scores: DeviceVector<'a>, parts: DeviceVector<'a>, weight_pack: DeviceVector<'a>, input_pack: DeviceVector<'a>, logits: DeviceVector<'a>,
    packed_keys: DeviceVector<'a>, packed_values: DeviceVector<'a>,
    moe: Option<MoeBatchScratch<'a>>,
}
// Hidden rows survive exchange; Q is projected only when its chunk can attend.
struct SequenceState<'a> {
    hidden: DeviceVector<'a>,
    position: usize, tokens: usize, cache_tokens: usize, prompt_end: usize, chunk_size: usize, next_chunk: usize,
}
use crate::context::{prefill_workspace, sequence_chunk_size};
impl<'a> BatchState<'a> {
    fn new(model: &Model<'a, '_>, tokens: usize, capacity: usize, all_logits: bool, exact: bool, format: KvFormat, prompt_end: usize, fast_prefill: bool, hidden: Option<DeviceVector<'a>>) -> Result<Self> {
        let b = model.backend;
        b.require_batch_kernels()?;
        let c = &model.config;
        let (h, q, k, m) = (c.hidden_size as usize, (c.attention_heads * c.head_dim) as usize,
            model.kv_capacity.stride, if c.num_experts == 0 { c.intermediate_size as usize } else { 1 });
        let moe = model.layers.iter().find_map(|layer| match &layer.mlp {
            Mlp::Experts { experts, .. } => Some(b.moe_batch_scratch(experts, c.num_experts_per_tok as usize, tokens)),
            _ => None,
        }).transpose()?;
        let padded = capacity.div_ceil(16)*16;
        let tc = !exact && c.head_dim == 128;
        let streaming = c.head_dim == 128 && (c.attention_heads == c.kv_heads * 4 || c.attention_heads == c.kv_heads * 8);
        let fast_attention = tc && c.num_experts == 0 && c.head_dim == 128 && c.attention_heads == c.kv_heads * 4 && tokens >= 64 && format == KvFormat::Fp32
            && b.fast_prefill_attention(prompt_end);
        let swizzled_attention = tc && format == KvFormat::Bf16 && c.attention_heads == c.kv_heads * 8
            && tokens >= if c.num_experts != 0 { 64 } else { 512 };
        let pack_attention = tc && format != KvFormat::Bf16;
        let projection_exact = exact;
        let fast_ffn = c.qk_norm && !projection_exact && h % 128 == 0 && m % 128 == 0;
        let packed_norms = !fast_prefill && fast_ffn && q % 128 == 0 && k % 128 == 0;
        let pack_width = if fast_ffn { q.max(h) } else { m.max(q).max(h) };
        let projection_input = tokens.div_ceil(16) * 16 * pack_width / 2;
        if fast_attention && fast_ffn {
            let sequence = hidden.is_some();
            let packed_k = if sequence { (c.kv_heads as usize).div_ceil(2) * 128 } else { k };
            let mut workspace = b.vector_workspace(&prefill_workspace(tokens, capacity, h, q, k, m, packed_k, sequence))?.into_iter();
            let mut next = || workspace.next().unwrap();
            return Ok(Self {
                progress: None, tokens, capacity, prompt_end, all_logits, exact, fast_prefill, moe,
                ids: b.vector(tokens)?, hidden: match hidden { Some(hidden) => hidden, None => b.vector(tokens*h)? }, input: next(), sums: b.vector(tokens*h.div_ceil(128))?,
                q: next(), k: next(), v: next(), attention: next(), projected: next(), gate: next(),
                up: b.vector(1)?, scores: b.vector(1)?, parts: b.vector(1)?, weight_pack: b.vector(1)?,
                input_pack: next(), packed_keys: next(), packed_values: next(),
                logits: b.shared_vector(if all_logits {tokens*c.vocab_size as usize} else {1})?,
            });
        }
        let mut scratch = b.vector_arena(&[
            tokens, if hidden.is_some() {1} else {tokens * h}, if packed_norms && !all_logits {1} else {tokens * h}, tokens * h.div_ceil(128),
            tokens * q, tokens * k, tokens * k, tokens * q, tokens * h,
            if fast_ffn { tokens.div_ceil(16) * 16 * m / 2 } else { tokens * m },
            if fast_ffn {1} else {tokens * m},
            if tc || streaming {1} else {tokens.min(64)*c.attention_heads as usize*padded},
            if tc || streaming && b.decode_attention_parts(capacity) <= 1 {1}
                else if streaming {tokens.min(16)*c.attention_heads as usize*(c.head_dim as usize+4)*b.decode_attention_parts(capacity)} else {tokens.min(64)*q*padded.div_ceil(64)},
            1, projection_input,
            if fast_attention || swizzled_attention {capacity.div_ceil(64)*64*k/2} else if pack_attention {if format == KvFormat::ScaledI16 {capacity*k} else {(capacity*k).div_ceil(2)}} else {1},
            if fast_attention || swizzled_attention {capacity.div_ceil(64)*64*k/2} else if pack_attention {if format == KvFormat::ScaledI16 {capacity*k} else {(capacity*k).div_ceil(2)}} else {1},
        ])?.into_iter();
        let mut next = || scratch.next().unwrap();
        Ok(Self {
            progress: None, tokens, capacity, prompt_end, all_logits, exact, fast_prefill, moe,
            ids: next(), hidden: { let allocated = next(); hidden.unwrap_or(allocated) }, input: next(), sums: next(),
            q: next(), k: next(), v: next(), attention: next(),
            projected: next(), gate: next(), up: next(), scores: next(), parts: next(),
            weight_pack: next(), input_pack: next(), packed_keys: next(), packed_values: next(),
            logits: b.shared_vector(if all_logits { tokens * c.vocab_size as usize } else { 1 })?,
        })
    }
}
impl<'a> DeviceState<'a> {
    pub fn sequence_abort(&mut self) { self.sequence = None; self.batch = None; }
    fn sequence_scratch(&mut self, model: &Model<'a, '_>, start: usize, tokens: usize, capacity: usize) -> Result<()> {
        let sequence = self.sequence.as_ref().ok_or("missing sequence state")?;
        let (prompt_end, rows) = (sequence.prompt_end, sequence.tokens.min(sequence.chunk_size));
        if self.batch.as_ref().is_none_or(|s| s.capacity < capacity || s.all_logits || s.exact) {
            self.batch = None;
            let hidden = Some(sequence.hidden.slice(0..rows * model.config.hidden_size as usize)?);
            self.batch = Some(BatchState::new(model, rows, capacity, false, false, KvFormat::Fp32, prompt_end, false, hidden)?);
        }
        let s = self.batch.as_mut().unwrap();
        let h = model.config.hidden_size as usize;
        s.hidden = sequence.hidden.slice(start * h..(start + tokens) * h)?;
        if s.tokens != tokens {
            let c = &model.config;
            let (h, q, k, m) = (c.hidden_size as usize, (c.attention_heads * c.head_dim) as usize,
                model.kv_capacity.stride, c.intermediate_size as usize);
            // Rank boundaries align to GEMM tiles, not whole projection chunks.
            // Only logical row shapes change; queued kernels keep their arguments.
            for (vector, len) in [(&mut s.ids, tokens),
                (&mut s.sums, tokens * h.div_ceil(128)), (&mut s.q, tokens * q),
                (&mut s.k, tokens * k), (&mut s.v, tokens * k),
                (&mut s.attention, tokens * q), (&mut s.projected, tokens * h)]
            {
                vector.resize_within_capacity(len)?;
            }
            if s.input.len() != 1 { s.input.resize_within_capacity(tokens * h)?; }
            let fast_ffn = c.qk_norm && h % 128 == 0 && m % 128 == 0;
            s.gate.resize_within_capacity(if fast_ffn { tokens.div_ceil(16) * 16 * m / 2 } else { tokens * m })?;
            if s.up.len() != 1 { s.up.resize_within_capacity(tokens * m)?; }
            s.tokens = tokens;
        }
        self.peak_bytes = self.peak_bytes.max(model.backend.allocated_bytes());
        Ok(())
    }
    pub fn sequence_begin_slice(&mut self, model: &Model<'a, '_>, tokens: &[u32], position: usize, total: usize, prompt_end: usize, budget: Duration) -> Result<bool> {
        let started = Instant::now();
        if self.format != KvFormat::Fp32 { return Err("sequence prefill requires FP32 KV".into()); }
        if budget.is_zero() { return Ok(false); }
        let b = model.backend;
        let h = model.config.hidden_size as usize;
        let q = (model.config.attention_heads * model.config.head_dim) as usize;
        let (k, m) = (model.kv_capacity.stride, model.config.intermediate_size as usize);
        let packed_k = (model.config.kv_heads as usize).div_ceil(2) * 128;
        // Chunk shape follows the memory budget, independently of scheduler
        // quantum. Sliced execution still yields after a completed GPU chunk.
        let chunk_size = self.sequence.as_ref().map_or_else(
            || sequence_chunk_size(tokens.len(), position + tokens.len(), h, q, k, m, packed_k),
            |sequence| sequence.chunk_size);
        let setup_chunk = super::PREFILL_CHUNK;
        if self.sequence.is_none() {
            self.batch = None;
            // Map one page per layer, then grow each just before projection.
            self.reserve_cache(model, 0, total.min(setup_chunk))?;
            self.sequence = Some(SequenceState {
                hidden: b.vector_arena(&[tokens.len() * h])?.pop().unwrap(),
                position, tokens: tokens.len(), cache_tokens: total, prompt_end, chunk_size, next_chunk: 0,
            });
            self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
            let elapsed = started.elapsed();
            if elapsed.saturating_add(elapsed) >= budget { return Ok(false); }
        }
        loop {
            let reserved = self.kv[0].keys.len() / model.kv_capacity.stride;
            if reserved >= total || self.kv[0].keys.is_segmented() { break; }
            let phase = Instant::now();
            self.reserve_cache(model, 0, ((reserved / setup_chunk + 1) * setup_chunk).min(total))?;
            // Leave room for another quantum instead of overrunning knowingly.
            if started.elapsed().saturating_add(phase.elapsed()) >= budget { return Ok(false); }
        }
        let first = self.sequence.as_ref().unwrap().next_chunk;
        for start in (first..tokens.len()).step_by(chunk_size) {
            let phase = Instant::now();
            let chunk = &tokens[start..(start + chunk_size).min(tokens.len())];
            self.sequence_scratch(model, start, chunk.len(), position + tokens.len())?;
            let s = self.batch.as_mut().unwrap();
            let seq = self.sequence.as_mut().unwrap();
            b.write_vector(&mut s.ids, &chunk.iter().map(|&id| id as f32).collect::<Vec<_>>())?;
            b.batch_embedding_device(model.device.embedding.as_ref().ok_or("missing embedding")?, &s.ids, &mut s.hidden)?;
            seq.next_chunk = start + chunk.len();
            if budget != Duration::MAX {
                b.synchronize()?;
                if seq.next_chunk < tokens.len() && started.elapsed().saturating_add(phase.elapsed()) >= budget { return Ok(false); }
            }
        }
        b.synchronize()?;
        self.sequence.as_mut().unwrap().next_chunk = 0;
        Ok(true)
    }
    pub fn sequence_project_slice(&mut self, model: &Model<'a, '_>, index: usize, budget: Duration) -> Result<bool> {
        let started = Instant::now();
        let b = model.backend;
        let _order = b.prompt_order(true);
        let c = &model.config;
        let (tokens, end, first) = { let seq = self.sequence.as_ref().ok_or("missing sequence state")?; (seq.tokens, seq.position + seq.tokens, seq.next_chunk) };
        let layer = &model.layers[index];
        let norms = &model.device.norms[index];
        let chunk_size = self.sequence.as_ref().unwrap().chunk_size;
        // No later layer is read before projection. Grow its two fixed-address
        // vectors here so helper startup need not map the whole request's KV.
        let needed = self.sequence.as_ref().unwrap().cache_tokens * model.kv_capacity.stride;
        let kv = &mut Rc::get_mut(&mut self.kv).ok_or("shared sequence cache")?[index];
        if kv.keys.len() < needed { b.grow_vector(&mut kv.keys, needed)?; }
        if kv.values.len() < needed { b.grow_vector(&mut kv.values, needed)?; }
        self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
        for start in (first..tokens).step_by(chunk_size) {
            let t = (tokens - start).min(chunk_size);
            self.sequence_scratch(model, start, t, end)?;
            let seq = self.sequence.as_mut().unwrap();
            let s = self.batch.as_mut().unwrap();
            let kv = &mut Rc::get_mut(&mut self.kv).ok_or("shared sequence cache")?[index];
            if c.qk_norm && c.hidden_size % 128 == 0 && layer.q.rows() % 128 == 0 && layer.k.rows() % 128 == 0 {
                let half = false;
                b.rms_pack_device(&s.hidden, &norms.input, &mut s.input_pack, &mut s.sums, c.norm_epsilon(), half)?;
                b.matmul_packed_device(&layer.k, &s.input_pack, &mut s.k, t)?;
                b.matmul_packed_device(&layer.v, &s.input_pack, &mut s.v, t)?;
            } else {
                b.rms_batch_device(&s.hidden, &norms.input, &mut s.input, &mut s.sums, c.norm_epsilon())?;
                b.matmul_device(&layer.k, &s.input, &mut s.k, &mut s.weight_pack, &mut s.input_pack, t, false)?;
                b.matmul_reusing_input(&layer.v, &s.input, &mut s.v, &mut s.weight_pack, &mut s.input_pack, t, false, true)?;
            }
            b.qk_batch_range_device(&mut s.q, &s.k, &s.v, &norms.q, &norms.k, &model.device.rope,
                &mut kv.keys, &mut kv.values, seq.position + start, c.qk_epsilon(), t, KvFormat::Fp32, Some(false))?;
            seq.next_chunk = start + t;
            if budget != Duration::MAX {
                b.synchronize()?;
                if seq.next_chunk < tokens && started.elapsed() >= budget { return Ok(false); }
            }
        }
        b.synchronize()?;
        self.sequence.as_mut().unwrap().next_chunk = 0;
        Ok(true)
    }
    pub fn sequence_export(&self, model: &Model<'a, '_>, layer: usize, start: usize, count: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.format != KvFormat::Fp32 { return Err("sequence export requires FP32 KV".into()); }
        let kv = &self.kv[layer];
        let width = model.kv_capacity.stride;
        Ok((model.backend.read_vector_range(&kv.keys, start * width, count * width)?,
            model.backend.read_vector_range(&kv.values, start * width, count * width)?))
    }
    pub fn sequence_export_into(&self, model: &Model<'a, '_>, layer: usize, start: usize, keys: &mut [f32], values: &mut [f32]) -> Result<()> {
        if self.format != KvFormat::Fp32 { return Err("sequence export requires FP32 KV".into()); }
        let kv = &self.kv[layer];
        let offset = start * model.kv_capacity.stride;
        model.backend.read_vector_range_into(&kv.keys, offset, keys)?;
        model.backend.read_vector_range_into(&kv.values, offset, values)
    }
    pub fn sequence_device_kv(&self, layer: usize) -> Result<(&DeviceVector<'a>, &DeviceVector<'a>)> {
        if self.format != KvFormat::Fp32 { return Err("sequence device cache requires FP32 KV".into()); }
        let kv = &self.kv[layer];
        Ok((&kv.keys, &kv.values))
    }
    pub fn take_sequence_device_kv(&mut self) -> Result<Vec<(DeviceVector<'a>, DeviceVector<'a>)>> {
        if self.format != KvFormat::Fp32 { return Err("sequence device cache requires FP32 KV".into()); }
        let cache = Rc::get_mut(&mut self.kv).ok_or("shared sequence cache cannot be transferred")?;
        if cache.iter().any(|kv| !kv.keys.writable_from(0) || !kv.values.writable_from(0)) {
            return Err("shared sequence pages cannot be transferred".into());
        }
        Ok(std::mem::take(cache).into_iter().map(|kv| (kv.keys, kv.values)).collect())
    }
    pub fn sequence_import(&mut self, model: &Model<'a, '_>, layer: usize, start: usize, keys: &[f32], values: &[f32]) -> Result<()> {
        let kv = &mut Rc::get_mut(&mut self.kv).ok_or("shared sequence cache")?[layer];
        let offset = start * model.kv_capacity.stride;
        model.backend.write_vector_range(&mut kv.keys, offset, keys)?;
        model.backend.write_vector_range(&mut kv.values, offset, values)
    }
    pub fn sequence_finish_layer_slice(&mut self, model: &Model<'a, '_>, index: usize, budget: Duration) -> Result<bool> {
        let started = Instant::now();
        let b = model.backend;
        let _order = b.prompt_order(true);
        let c = &model.config;
        let (tokens, end, first) = { let seq = self.sequence.as_ref().ok_or("missing sequence state")?; (seq.tokens, seq.position + seq.tokens, seq.next_chunk) };
        let layer = &model.layers[index];
        let (gate, up, down) = layer.mlp.dense()?;
        let norms = &model.device.norms[index];
        let chunk_size = self.sequence.as_ref().unwrap().chunk_size;
        for start in (first..tokens).step_by(chunk_size) {
            let t = (tokens - start).min(chunk_size);
            self.sequence_scratch(model, start, t, end)?;
            let seq = self.sequence.as_mut().unwrap();
            let s = self.batch.as_mut().unwrap();
            let kv = &mut Rc::get_mut(&mut self.kv).ok_or("shared sequence cache")?[index];
            if c.qk_norm && c.hidden_size % 128 == 0 && layer.q.rows() % 128 == 0 {
                b.rms_pack_device(&s.hidden, &norms.input, &mut s.input_pack, &mut s.sums, c.norm_epsilon(), false)?;
                b.matmul_packed_device(&layer.q, &s.input_pack, &mut s.q, t)?;
            } else {
                b.rms_batch_device(&s.hidden, &norms.input, &mut s.input, &mut s.sums, c.norm_epsilon())?;
                b.matmul_device(&layer.q, &s.input, &mut s.q, &mut s.weight_pack, &mut s.input_pack, t, false)?;
            }
            b.qk_batch_range_device(&mut s.q, &s.k, &s.v, &norms.q, &norms.k, &model.device.rope,
                &mut kv.keys, &mut kv.values, seq.position + start, c.qk_epsilon(), t, KvFormat::Fp32, Some(true))?;
            if c.head_dim == 128 {
                b.prefill_attention_device(&s.q, &kv.keys, &kv.values, &mut s.attention,
                    &mut s.packed_keys, &mut s.packed_values,
                    c.attention_heads as usize, c.kv_heads as usize, seq.position + start, t, KvFormat::Fp32, seq.prompt_end, false)?;
            } else {
                b.attention_batch_device(&s.q, &kv.keys, &kv.values, &mut s.scores, &mut s.parts, &mut s.attention,
                    c.attention_heads as usize, c.kv_heads as usize, c.head_dim as usize, seq.position + start, t, false)?;
            }
            b.matmul_device(&layer.o, &s.attention, &mut s.projected, &mut s.weight_pack, &mut s.input_pack, t, false)?;
            b.residual_checkpoint_device(&mut s.hidden, &s.projected, !c.qk_norm)?;
            if c.qk_norm && c.hidden_size % 128 == 0 && c.intermediate_size % 128 == 0 {
                b.rms_pack_device(&s.hidden, &norms.post, &mut s.input_pack, &mut s.sums, c.norm_epsilon(), false)?;
                b.prefill_gate_up_packed_device(gate, up, &s.input_pack, &mut s.gate, t)?;
                b.matmul_packed_device(down, &s.gate, &mut s.projected, t)?;
            } else {
                b.rms_batch_device(&s.hidden, &norms.post, &mut s.input, &mut s.sums, c.norm_epsilon())?;
                b.matmul_device(gate, &s.input, &mut s.gate, &mut s.weight_pack, &mut s.input_pack, t, false)?;
                b.matmul_reusing_input(up, &s.input, &mut s.up, &mut s.weight_pack, &mut s.input_pack, t, false, true)?;
                b.swiglu_precision_device(&mut s.gate, &s.up, !c.qk_norm)?;
                b.matmul_device(down, &s.gate, &mut s.projected, &mut s.weight_pack, &mut s.input_pack, t, false)?;
            }
            b.residual_checkpoint_device(&mut s.hidden, &s.projected, !c.qk_norm)?;
            seq.next_chunk = start + t;
            if budget != Duration::MAX {
                b.synchronize()?;
                if seq.next_chunk < tokens && started.elapsed() >= budget { return Ok(false); }
            }
        }
        b.synchronize()?;
        self.sequence.as_mut().unwrap().next_chunk = 0;
        Ok(true)
    }
    pub fn sequence_finish(&mut self, model: &Model<'a, '_>) -> Result<Vec<f32>> {
        let logits = self.sequence_logits(model)?;
        if logits.iter().any(|v| !v.is_finite()) { return Err("nonfinite device output".into()); }
        self.sequence_abort();
        Ok(logits)
    }
    pub fn sequence_finish_values(&mut self, model: &Model<'a, '_>) -> Result<Vec<f32>> {
        let logits = self.sequence_logits(model)?;
        self.sequence_abort();
        Ok(logits)
    }
    fn sequence_logits(&mut self, model: &Model<'a, '_>) -> Result<Vec<f32>> {
        let seq = self.sequence.as_ref().ok_or("missing sequence state")?;
        model.backend.copy_row(&seq.hidden, &mut self.hidden, seq.tokens - 1)?;
        self.project(model, &model.device, &mut None)?;
        model.backend.read_vector_values(&self.logits)
    }
    pub fn release_batch(&mut self) { self.batch = None; }
    pub fn prefill_pending(&self) -> bool { self.batch.as_ref().is_some_and(|s| s.progress.is_some()) }
    pub fn batch(&mut self, model: &Model<'a, '_>, device: &DeviceModel<'a>, position: usize, tokens: &[u32], output: Logits, exact: bool, prompt_end: usize, fast_prefill: bool, observer: &mut RoutingObserver<'_>) -> Result<Vec<Vec<f32>>> {
        if self.prefill_pending() { return Err("prefill slice is still pending".into()); }
        let result = self.batch_run(model, device, position, tokens, output, exact, None, prompt_end, fast_prefill, observer);
        if result.is_err() { self.release_batch(); }
        result?.ok_or_else(|| "unbounded batch yielded".into())
    }
    pub fn prefill_slice(&mut self, model: &Model<'a, '_>, device: &DeviceModel<'a>, position: usize,
        tokens: &[u32], output: Logits, budget: Option<Duration>, prompt_end: usize, fast_prefill: bool) -> Result<Option<Vec<Vec<f32>>>> {
        let result = self.batch_run(model, device, position, tokens, output, false, budget, prompt_end, fast_prefill, &mut None);
        if result.is_err() { self.release_batch(); }
        result
    }
    fn batch_run(&mut self, model: &Model<'a, '_>, device: &DeviceModel<'a>, position: usize,
        tokens: &[u32], output: Logits, exact: bool, budget: Option<Duration>, prompt_end: usize, fast_prefill: bool, observer: &mut RoutingObserver<'_>) -> Result<Option<Vec<Vec<f32>>>> {
        model.backend.generic_profile(!exact, || (position, tokens.len(), engine_model::plan::profile::ProfilePlan::qwen(&model.config).classes(), 1),
            || self.batch_run_profiled(model, device, position, tokens, output, exact, budget, prompt_end, fast_prefill, observer))
    }
    fn batch_run_profiled(&mut self, model: &Model<'a, '_>, device: &DeviceModel<'a>, position: usize,
        tokens: &[u32], output: Logits, exact: bool, budget: Option<Duration>, prompt_end: usize, fast_prefill: bool, observer: &mut RoutingObserver<'_>) -> Result<Option<Vec<Vec<f32>>>> {
        let started = Instant::now();
        let _order = model.backend.prompt_order(!exact);
        let pending = self.batch.as_ref().and_then(|s| s.progress.as_ref());
        if pending.is_some_and(|p| p.position != position || p.tokens != tokens || p.output != output)
            || pending.is_some() && self.batch.as_ref().unwrap().prompt_end != prompt_end {
            return Err("resumed prefill must use the same tokens and output mode".into());
        }
        let resume = pending.is_some();
        let t = tokens.len();
        let all_logits = output == Logits::All;
        if !resume { self.reserve_cache(model,position,position+t)?; }
        let capacity = (self.format.element_capacity(self.kv[0].keys.len()) / model.kv_capacity.stride).min(model.max_tokens);
        if self.batch.as_ref().is_none_or(|s| s.tokens != t || s.capacity != capacity || s.prompt_end != prompt_end || s.all_logits != all_logits || s.exact != exact || s.fast_prefill != fast_prefill) {
            self.batch = None;
            self.batch = Some(BatchState::new(model, t, capacity, all_logits, exact, self.format, prompt_end, fast_prefill, None)?);
        }
        let s = self.batch.as_mut().unwrap();
        let b = model.backend;
        self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
        let c = &model.config;
        let eps = c.norm_epsilon();
        if !resume {
            b.write_vector(&mut s.ids, &tokens.iter().map(|&id| id as f32).collect::<Vec<_>>())?;
            b.batch_embedding_device(device.embedding.as_ref().ok_or("partition does not own embedding")?, &s.ids, &mut s.hidden)?;
            s.progress = Some(BatchProgress { position, tokens: tokens.to_vec(), output, next_layer: 0 });
            if let Some(limit) = budget {
                b.synchronize()?;
                if started.elapsed().saturating_add(self.layer_cost.mul_f64(1.2)) >= limit { return Ok(None); }
            }
        }
        let first_layer = s.progress.as_ref().unwrap().next_layer;
        let projection_exact = exact;
        let fused_norms = c.qk_norm && !exact && !fast_prefill && c.num_experts == 0 && c.hidden_size % 128 == 0 && c.intermediate_size % 128 == 0;
        let attention_fragments = fused_norms && c.head_dim == 128 && c.attention_heads == c.kv_heads * 4
            && self.format == KvFormat::Bf16 && t % 256 == 0;
        // GQA8 swizzled prefill rounds the same quotients into the O projection's fragments.
        let staged = (position + t).div_ceil(64) * 64 * c.kv_heads as usize * 128;
        let gqa8_fragments = fused_norms && c.head_dim == 128 && c.attention_heads == c.kv_heads * 8
            && self.format == KvFormat::Bf16 && t % 256 == 0 && t >= 512 && staged <= u32::MAX as usize
            && s.packed_keys.len() >= staged / 2 && s.packed_values.len() >= staged / 2;
        let mut ran_layer = false;
        let first_experts = model.layers.iter().position(|layer| matches!(layer.mlp, Mlp::Experts { .. }));
        for (index, ((layer, norms), kv)) in model.layers.iter().zip(&device.norms)
            .zip(Rc::get_mut(&mut self.kv).ok_or("pending prefill KV became shared")?).enumerate().skip(first_layer) {
            if ran_layer && budget.is_some_and(|limit| started.elapsed().saturating_add(self.layer_cost.mul_f64(1.2)) > limit) { return Ok(None); }
            let layer_started = Instant::now();
            model.profile_layer(position + t, self.format);
            if c.qk_norm && !projection_exact && !fast_prefill && c.hidden_size % 128 == 0 && layer.q.rows() % 128 == 0 && layer.k.rows() % 128 == 0 {
                // The model uses one BF16 operand for every prompt row, including tails.
                let half = false;
                if index > 0 && (fused_norms || matches!(model.layers[index - 1].mlp, Mlp::Experts { .. })) {
                    b.rms_pack_apply_device(&s.hidden, &norms.input, &mut s.input_pack, &s.sums, eps, half)?;
                } else {
                    b.rms_pack_device(&s.hidden, &norms.input, &mut s.input_pack, &mut s.sums, eps, half)?;
                }
                b.prefill_qkv_device(&layer.q, &layer.k, &layer.v, &s.input_pack, &mut s.q, &mut s.k, &mut s.v, t, half)?;
            } else {
                b.rms_batch_device(&s.hidden, &norms.input, &mut s.input, &mut s.sums, eps)?;
                b.matmul_prefill_device(&layer.q, &s.input, &mut s.q, &mut s.weight_pack, &mut s.input_pack, t, projection_exact, false, fast_prefill)?;
                b.matmul_prefill_device(&layer.k, &s.input, &mut s.k, &mut s.weight_pack, &mut s.input_pack, t, projection_exact, true, fast_prefill)?;
                b.matmul_prefill_device(&layer.v, &s.input, &mut s.v, &mut s.weight_pack, &mut s.input_pack, t, projection_exact, true, fast_prefill)?;
            }
            b.qk_batch_device(&mut s.q, &s.k, &s.v, &norms.q, &norms.k, &device.rope, &mut kv.keys, &mut kv.values, position, c.qk_epsilon(), t, self.format)?;
            let packed_attention = !exact && c.num_experts != 0 && c.head_dim == 128
                && c.attention_heads == c.kv_heads * 8 && t >= 64 && self.format == KvFormat::Bf16;
            if attention_fragments {
                b.prefill_attention_fragments_device(&s.q, &kv.keys, &kv.values, &mut s.attention,
                    c.attention_heads as usize, c.kv_heads as usize, position, t)?;
            } else if packed_attention || gqa8_fragments {
                b.prefill_attention_output_packed_device(&s.q, &kv.keys, &kv.values, &mut s.attention,
                    &mut s.packed_keys, &mut s.packed_values, c.attention_heads as usize, c.kv_heads as usize, position, t)?;
            } else if !exact && c.head_dim == 128 {
                b.prefill_attention_device(&s.q, &kv.keys, &kv.values, &mut s.attention,
                    &mut s.packed_keys, &mut s.packed_values,
                    c.attention_heads as usize, c.kv_heads as usize, position, t, self.format, prompt_end, c.num_experts != 0)?;
            } else {
                b.attention_batch_device(&s.q, &kv.keys, &kv.values, &mut s.scores, &mut s.parts, &mut s.attention,
                    c.attention_heads as usize, c.kv_heads as usize, c.head_dim as usize, position, t, self.format)?;
            }
            b.profile_phase(15, || {
                if fused_norms { b.matmul_residual_rms_device(&layer.o, &s.attention,
                    &mut s.hidden, &mut s.sums, &mut s.input_pack, t, attention_fragments || gqa8_fragments) }
                else if packed_attention { b.matmul_packed_device(&layer.o, &s.attention, &mut s.projected, t) }
                else { b.matmul_prefill_device(&layer.o, &s.attention, &mut s.projected, &mut s.weight_pack, &mut s.input_pack, t, projection_exact, false, fast_prefill) }
            })?;
            match &layer.mlp {
                Mlp::Experts { router, experts } => {
                    b.residual_rms_batch_device(&mut s.hidden, &s.projected, &norms.post, &mut s.input, &mut s.sums, eps)?;
                    b.moe_batch_device(router, experts, &s.input, &mut s.hidden,
                        s.moe.as_mut().ok_or("missing batched expert scratch")?,
                        &mut s.weight_pack, &mut s.input_pack, false, c.norm_topk_prob, first_experts == Some(index), Some(&mut s.sums))?;
                    if let Some(observer) = observer {
                        let routing = b.moe_batch_routing(s.moe.as_ref().unwrap(), t)?;
                        for (row, selected) in routing.chunks(c.num_experts_per_tok as usize).enumerate() {
                            observer(model.layer_start + index, position + row, selected)?;
                        }
                    }
                }
                Mlp::Dense { gate, up, down } => {
                    let _class = b.prefill_class(1);
                    if !fused_norms { b.residual_checkpoint_device(&mut s.hidden, &s.projected, !c.qk_norm)?; }
                    if fast_prefill && !exact && c.hidden_size % 128 == 0 && c.intermediate_size % 128 == 0 {
                        b.rms_batch_device(&s.hidden, &norms.post, &mut s.input, &mut s.sums, eps)?;
                        b.prefill_gate_up_fast_device(gate, up, &s.input, &mut s.gate, &mut s.input_pack, t)?;
                        b.matmul_prefill_device(down, &s.input, &mut s.projected, &mut s.weight_pack, &mut s.gate, t, false, true, true)?;
                    } else if c.qk_norm && !projection_exact && c.hidden_size % 128 == 0 && c.intermediate_size % 128 == 0 {
                        if fused_norms {
                            b.rms_pack_apply_device(&s.hidden, &norms.post, &mut s.input_pack, &s.sums, eps, false)?;
                        } else { b.rms_pack_device(&s.hidden, &norms.post, &mut s.input_pack, &mut s.sums, eps, false)?; }
                        b.prefill_gate_up_packed_device(gate, up, &s.input_pack, &mut s.gate, t)?;
                        b.profile_phase(17, || {
                            if fused_norms { b.matmul_residual_rms_device(down, &s.gate,
                                &mut s.hidden, &mut s.sums, &mut s.input_pack, t, true) }
                            else { b.matmul_packed_device(down, &s.gate, &mut s.projected, t) }
                        })?;
                    } else {
                        b.rms_batch_device(&s.hidden, &norms.post, &mut s.input, &mut s.sums, eps)?;
                        b.matmul_device(gate, &s.input, &mut s.gate, &mut s.weight_pack, &mut s.input_pack, t, projection_exact)?;
                        b.matmul_reusing_input(up, &s.input, &mut s.up, &mut s.weight_pack, &mut s.input_pack, t, projection_exact, true)?;
                        b.swiglu_precision_device(&mut s.gate, &s.up, !c.qk_norm)?;
                        b.matmul_device(down, &s.gate, &mut s.projected, &mut s.weight_pack, &mut s.input_pack, t, projection_exact)?;
                    }
                }
            }
            if !fused_norms && matches!(layer.mlp, Mlp::Dense { .. }) { b.residual_checkpoint_device(&mut s.hidden, &s.projected, !c.qk_norm)?; }
            if budget.is_some() {
                b.synchronize()?;
                self.layer_cost = self.layer_cost.mul_f64(0.8).max(layer_started.elapsed());
            }
            s.progress.as_mut().unwrap().next_layer = index + 1;
            ran_layer = true;
        }
        if output != Logits::None && ran_layer
            && budget.is_some_and(|limit| started.elapsed().saturating_add(self.head_cost.mul_f64(1.2)) > limit) { return Ok(None); }
        let head_started = Instant::now();
        let rows = if output == Logits::None {
            b.synchronize()?;
            Vec::new()
        } else if all_logits {
            model.profile_head();
            b.rms_batch_device(&s.hidden, device.norm.as_ref().ok_or("missing final norm")?, &mut s.input, &mut s.sums, eps)?;
            // MoE prompt-All heads use canonical arithmetic, matching Last and shared serving.
            b.profile_output_head(|| b.matmul_device(model.head.as_ref().ok_or("missing head")?, &s.input, &mut s.logits, &mut s.weight_pack, &mut s.input_pack, t, projection_exact || c.num_experts != 0))?;
            b.round_logits_bf16(&mut s.logits)?;
            let logits = b.read_vector(&s.logits)?;
            logits.chunks_exact(c.vocab_size as usize).map(<[f32]>::to_vec).collect()
        } else {
            b.copy_row(&s.hidden, &mut self.hidden, t - 1)?;
            self.project(model, device, &mut None)?;
            vec![b.read_vector(&self.logits)?]
        };
        if budget.is_some() && output != Logits::None { self.head_cost = self.head_cost.mul_f64(0.8).max(head_started.elapsed()); }
        // Every row belongs to this request: any rejected router row fails it.
        if let Some(moe) = &self.batch.as_ref().unwrap().moe {
            if !b.moe_batch_rejected(moe, t)?.is_empty() { return Err(MOE_ROUTING_ERROR.into()); }
        }
        self.batch.as_mut().unwrap().progress = None;
        // Intermediate chunks retain scratch across scheduler yields.
        if output != Logits::None && (t > 16 || !all_logits) { self.batch = None; }
        Ok(Some(rows))
    }
}
