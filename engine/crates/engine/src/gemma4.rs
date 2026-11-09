//! Dense Gemma text cartridge; projections, normalization and attention are shared.
use crate::{layer, qwen::Loader};
use engine_model::{gemma4::{Attention, Config}, safetensors::{Index, Limits, TensorFile}};
use spark_gpu_probe::driver::{Backend, DeviceVector, KvFormat, Matrix};
use std::{cell::RefCell, collections::BTreeMap, path::Path};
mod assistant;
pub use assistant::Assistant;
type Result<T> = std::result::Result<T, String>;
pub const PREFILL_CHUNK: usize = 4096;
const DECODE_ROWS: usize = 64;

struct Layer<'a> {
    input: DeviceVector<'a>, attention: DeviceVector<'a>, pre_mlp: DeviceVector<'a>, post_mlp: DeviceVector<'a>,
    q_norm: DeviceVector<'a>, k_norm: Option<DeviceVector<'a>>, scalar: f32,
    q: Matrix<'a>, k: Option<Matrix<'a>>, v: Option<Matrix<'a>>, o: Matrix<'a>,
    gate: Matrix<'a>, up: Matrix<'a>, down: Matrix<'a>,
}
pub struct Model<'a, 'b> {
    pub backend: &'b Backend<'a>, pub config: Config, layers: Vec<Layer<'a>>,
    embedding: Matrix<'a>, head: Matrix<'a>, norm: DeviceVector<'a>,
    local_rope: DeviceVector<'a>, global_rope: DeviceVector<'a>, pub max_tokens: usize,
    eos: Vec<u32>, assistant: Option<Assistant<'a>>, multi: RefCell<Option<State<'a>>>,
}
impl<'a, 'b> Model<'a, 'b> {
    pub fn load(b: &'b Backend<'a>, root: &Path, max_tokens: usize) -> Result<Self> {
        let cartridge = engine_model::cartridge::Cartridge::preflight(root).map_err(|e| e.to_string())?;
        let c = cartridge.gemma4.ok_or("expected Gemma 4 cartridge")?;
        if max_tokens == 0 || max_tokens as u64 > c.max_positions || !c.tie_word_embeddings {
            return Err("unsupported Gemma context or untied head".into());
        }
        b.require_layer_kernels()?; b.require_batch_kernels()?;
        let index = Index::open(&root.join("model.safetensors.index.json"), Limits::default()).map_err(|e| e.to_string())?;
        let mut files = BTreeMap::new();
        for shard in index.weight_map.values() {
            if !files.contains_key(shard) {
                files.insert(shard.clone(), TensorFile::open(&root.join(shard), Limits::default()).map_err(|e| e.to_string())?);
            }
        }
        // Validate the descriptors retained by the loader, excluding vision tensors.
        for (name, shape) in c.expected_shapes() {
            let shard = index.weight_map.get(&name).ok_or("missing text tensor")?;
            let t = files[shard].header().tensors.get(&name).ok_or("missing retained tensor")?;
            if t.shape != shape || t.dtype != engine_model::safetensors::Dtype::Bf16 { return Err("text weight binding mismatch".into()); }
        }
        let mut loader = Loader { files, map: index.weight_map, matrices: BTreeMap::new() };
        loader.load_matrices(b, 0..c.layer_types.len(), true, true, true, true, "model.language_model", None)?;
        let embedding = loader.matrix("model.language_model.embed_tokens.weight")?;
        let head = embedding.rows_view(0..c.vocab_size as usize)?;
        let norm = b.upload_vector(&loader.host("model.language_model.norm.weight", 1024*1024)?)?;
        let mut layers = Vec::new();
        for (i, &kind) in c.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            let mut norm = |name: &str| -> Result<DeviceVector<'a>> { b.upload_vector(&loader.host(&format!("{p}.{name}.weight"), 1024*1024)?) };
            let (input, attention, pre_mlp, post_mlp, q_norm, k_norm) = (norm("input_layernorm")?, norm("post_attention_layernorm")?,
                norm("pre_feedforward_layernorm")?, norm("post_feedforward_layernorm")?, norm("self_attn.q_norm")?, norm("self_attn.k_norm")?);
            let scalar = loader.host(&format!("{p}.layer_scalar"), 16)?[0];
            let mut mat = |name: &str| loader.matrix(&format!("{p}.{name}.weight"));
            layers.push(Layer { input, attention, pre_mlp, post_mlp, q_norm, k_norm: Some(k_norm), scalar,
                q: mat("self_attn.q_proj")?, k: Some(mat("self_attn.k_proj")?),
                v: if kind == Attention::Sliding || !c.attention_k_eq_v {Some(mat("self_attn.v_proj")?)} else {None},
                o: mat("self_attn.o_proj")?, gate: mat("mlp.gate_proj")?, up: mat("mlp.up_proj")?, down: mat("mlp.down_proj")? });
        }
        let local_rope = layer::model_rope(b, c.head_dim as usize, c.sliding_rope.theta, c.sliding_rope.rotary_fraction, max_tokens)?;
        let global_rope = layer::model_rope(b, c.global_head_dim as usize, c.full_rope.theta, c.full_rope.rotary_fraction, max_tokens)?;
        let mut eos = c.eos_tokens.iter().map(|&n| n as u32).collect::<Vec<_>>();
        // Generation metadata can add end-of-turn stops; never treat BOS as EOS.
        let path = root.join("generation_config.json");
        if path.exists() {
            let bytes = engine_model::safetensors::read_bounded(&path, 1024*1024).map_err(|e| e.to_string())?;
            let doc = engine_model::json::parse(&bytes, Default::default()).map_err(|e| e.to_string())?;
            if let Some(ids) = doc.object().map_err(|e| e.to_string())?.get("eos_token_id") {
                let values = match ids { engine_model::json::Value::Array(a) => a.as_slice(), _ => std::slice::from_ref(ids) };
                for id in values { let id = id.u64().map_err(|e| e.to_string())?;
                    if id >= c.vocab_size { return Err("generation EOS outside vocabulary".into()); }
                    if !eos.contains(&(id as u32)) { eos.push(id as u32); }
                }
            }
        }
        eprintln!("gemma_text_weight_bytes={} device_bytes={} peak_device_bytes={}", c.weight_bytes().map_err(|e| e.to_string())?, b.allocated_bytes(), b.peak_allocated_bytes());
        Ok(Self { backend: b, config: c, layers, embedding, head, norm, local_rope, global_rope, max_tokens, eos, assistant: None, multi: RefCell::new(None) })
    }
    pub fn eos_token_ids(&self) -> &[u32] { &self.eos }
    pub fn session(&self) -> Session<'_, 'a, 'b> { Session { model: self, len: 0, state: None, draft: None } }
    /// Read each projection once for up to 64 independent decode rows.
    pub fn step_batch(&self, rows: &mut [(&mut Session<'_, 'a, 'b>, u32)]) -> Result<Vec<Vec<f32>>> {
        let (b, c) = (self.backend, &self.config);
        if rows.is_empty() || rows.len() > DECODE_ROWS { return Err("Gemma decode width outside 1..64".into()); }
        for (session, token) in rows.iter_mut() {
            if !std::ptr::eq(session.model, self) || session.len >= self.max_tokens || *token as u64 >= c.vocab_size {
                return Err("invalid Gemma batch row".into());
            }
            if session.state.is_none() { session.state = Some(State::new(b, c, self.max_tokens, DECODE_ROWS)?); }
        }
        let mut scratch = self.multi.borrow_mut();
        if scratch.is_none() {
            let mut state = State::scratch(b, c, self.max_tokens, DECODE_ROWS)?;
            state.normalized = b.vector(DECODE_ROWS*c.hidden_size as usize)?;
            state.logits = b.vector(DECODE_ROWS*c.vocab_size as usize)?;
            *scratch = Some(state);
        }
        let s = scratch.as_mut().unwrap();
        let count = rows.len(); let h = c.hidden_size as usize;
        s.resize(c, count)?;
        b.write_vector(&mut s.ids, &rows.iter().map(|r| r.1 as f32).collect::<Vec<_>>())?;
        b.batch_embedding_device(&self.embedding, &s.ids, &mut s.hidden)?;
        b.model_elementwise_device(&s.hidden, None, &s.hidden, count*h, 0, bf16((h as f32).sqrt()))?;
        for (i, l) in self.layers.iter().enumerate() {
            let local = c.layer_types[i] == Attention::Sliding;
            let (kv, dim) = c.geometry(c.layer_types[i]); let (kv, dim) = (kv as usize, dim as usize);
            let qwidth = c.attention_heads as usize*dim; let kwidth = kv*dim;
            s.q.resize_within_capacity(count*qwidth)?; s.attention.resize_within_capacity(s.q.len())?;
            s.k.resize_within_capacity(count*kwidth)?; s.v.resize_within_capacity(s.k.len())?;
            layer::forward_normalized_attention(b, layer::Weights { q: Some(&l.q), k: l.k.as_ref(), v: l.v.as_ref(), o: Some(&l.o),
                gate: Some(&l.gate), up: Some(&l.up), down: Some(&l.down) },
                layer::BlockNorms { input: &l.input, attention: &l.attention, pre_mlp: &l.pre_mlp, post_mlp: &l.post_mlp,
                    q: &l.q_norm, k: l.k_norm.as_ref().unwrap(), scalar: l.scalar },
                layer::Buffers { hidden: &mut s.hidden, input: &mut s.input, sums: &mut s.sums, q: &mut s.q, k: &mut s.k, v: &mut s.v,
                    attention: &mut s.attention, projected: &mut s.projected, gate: &mut s.gate, up: &mut s.up, scores: &mut s.scores,
                    parts: &mut s.last, weight_pack: &mut s.weight_pack, input_pack: &mut s.input_pack,
                    packed_keys: &mut s.normalized, packed_values: &mut s.logits },
                layer::Run { position: 0, tokens: count, first: i == 0, prompt_end: 0,
                    heads: c.attention_heads as usize, kv_heads: kv, head_dim: dim, epsilon: c.rms_norm_eps as f32,
                    exact: true, format: KvFormat::Bf16, decode: true, sums_ready: false },
                |q, k, v, scores, output| {
                    let mut caches = Vec::with_capacity(count);
                    for (session, _) in rows.iter_mut() {
                        let cache = &mut session.state.as_mut().unwrap().cache[i];
                        cache.prepare(b, session.len, 1, kv*dim, c.sliding_window as usize, self.max_tokens)?;
                        caches.push(spark_gpu_probe::driver::CacheRows { keys: &mut cache.keys, values: &mut cache.values,
                            prefix: None, position: session.len, rows: 1, format: KvFormat::Bf16 });
                    }
                    b.model_attention_multi_device(q, k, v, &l.q_norm, l.k_norm.as_ref().unwrap(),
                        if local {&self.local_rope} else {&self.global_rope}, &mut caches, &mut s.metadata, scores, output,
                        c.attention_heads as usize, kv, dim, if local {c.sliding_window as usize} else {0}, c.rms_norm_eps as f32)?;
                    Ok(())
                })?;
        }
        s.normalized.resize_within_capacity(count*h)?;
        s.logits.resize_within_capacity(count*c.vocab_size as usize)?;
        b.model_rms_device(&s.hidden, Some(&self.norm), &mut s.normalized, &mut s.sums, h, count, c.rms_norm_eps as f32)?;
        layer::project(b, &self.head, &s.normalized, &mut s.logits, &mut s.weight_pack, &mut s.input_pack, count, true, false, false)?;
        b.model_elementwise_device(&s.logits, None, &s.logits, s.logits.len(), 0, 1.0)?;
        b.model_elementwise_device(&s.logits, None, &s.logits, s.logits.len(), 3, c.final_logit_softcapping as f32)?;
        let logits = b.read_vector(&s.logits)?;
        for (i, (session, _)) in rows.iter_mut().enumerate() {
            session.prepare_draft()?;
            if let Some(d) = &mut session.draft {
                b.copy_vector_range(&s.normalized, &mut d.carry, i*h, 0, h)?; d.end = 0;
            }
            session.len += 1;
        }
        Ok(logits.chunks(c.vocab_size as usize).map(|r| r.to_vec()).collect())
    }
    pub fn session_bytes(c: &Config, max_tokens: usize, tokens: usize) -> Result<usize> {
        let tokens = tokens.min(max_tokens).max(1).div_ceil(DECODE_ROWS)*DECODE_ROWS;
        let rows = if tokens > DECODE_ROWS {tokens.min(PREFILL_CHUNK)} else {DECODE_ROWS};
        let mut kv = 0usize;
        for &kind in &c.layer_types {
            let (heads, dim) = c.geometry(kind);
            let capacity = if kind == Attention::Sliding { (c.sliding_window as usize+rows-1).min(tokens).min(max_tokens) } else {tokens.min(max_tokens)};
            // VMM page rounding and a private tail on a fork.
            kv = kv.checked_add((capacity*heads as usize*dim as usize*2).div_ceil(2*1024*1024)*4*1024*1024).ok_or("KV reservation overflow")?;
        }
        let scratch: usize = scratch_lengths(c, max_tokens, rows).iter().map(|n| (n+15)/16*64).sum();
        kv.checked_add(scratch).and_then(|n| n.checked_add(64*1024*1024)).ok_or_else(|| "session reservation overflow".into())
    }
}
fn scratch_lengths(c: &Config, max_tokens: usize, rows: usize) -> Vec<usize> {
    let t = rows;
    let h = c.hidden_size as usize;
    let q = (c.attention_heads*c.head_dim.max(c.global_head_dim)) as usize;
    let k = (c.kv_heads*c.head_dim).max(c.global_kv_heads*c.global_head_dim) as usize;
    let m = c.intermediate_size as usize;
    vec![t, t*h, t*h, t*h.div_ceil(128), t*q, t*k, t*k, t*q, t*h,
        t*m, t*m, if t >= 256 {256*c.attention_heads as usize*max_tokens.div_ceil(64)*64} else if max_tokens <= 32768 {1} else {8*c.attention_heads as usize*max_tokens}, 8*t.div_ceil(16)*h.max(q).max(m), 1, c.vocab_size as usize, h, h, t*6]
}
struct Cache<'a> { keys: DeviceVector<'a>, values: DeviceVector<'a>, capacity: usize, local: bool }
impl<'a> Cache<'a> {
    fn prepare(&mut self, b: &Backend<'a>, position: usize, rows: usize, stride: usize, window: usize, max_tokens: usize) -> Result<()> {
        if self.local {
            let capacity = (position+rows).div_ceil(DECODE_ROWS)*DECODE_ROWS;
            return self.grow_local(b, capacity.min(window+rows-1).min(max_tokens), position, stride, window);
        }
        let capacity = (position+rows).div_ceil(DECODE_ROWS)*DECODE_ROWS;
        let capacity = capacity.min(max_tokens);
        let words = capacity*stride/2;
        let committed = position*stride/2;
        if !self.keys.writable_from(committed) {
            self.keys = b.fork_vector(&self.keys, committed, words)?;
            self.values = b.fork_vector(&self.values, committed, words)?;
        }
        b.grow_vector(&mut self.keys, words)?; b.grow_vector(&mut self.values, words)?;
        self.capacity = capacity;
        Ok(())
    }
    fn grow_local(&mut self, b: &Backend<'a>, capacity: usize, position: usize, stride: usize, window: usize) -> Result<()> {
        if capacity <= self.capacity { return Ok(()); }
        let words = capacity*stride/2;
        let mut keys = b.vector(words)?;
        let mut values = b.vector(words)?;
        let mut token = position.saturating_sub(window);
        while token < position {
            let source = token % self.capacity;
            let destination = token % capacity;
            let count = (position-token).min(self.capacity-source).min(capacity-destination);
            b.copy_vector_range(&self.keys, &mut keys, source*stride/2, destination*stride/2, count*stride/2)?;
            b.copy_vector_range(&self.values, &mut values, source*stride/2, destination*stride/2, count*stride/2)?;
            token += count;
        }
        self.keys = keys; self.values = values; self.capacity = capacity;
        Ok(())
    }
}
struct State<'a> {
    ids: DeviceVector<'a>, hidden: DeviceVector<'a>, input: DeviceVector<'a>, sums: DeviceVector<'a>,
    q: DeviceVector<'a>, k: DeviceVector<'a>, v: DeviceVector<'a>, attention: DeviceVector<'a>, projected: DeviceVector<'a>,
    gate: DeviceVector<'a>, up: DeviceVector<'a>, scores: DeviceVector<'a>, input_pack: DeviceVector<'a>, weight_pack: DeviceVector<'a>,
    metadata: DeviceVector<'a>, rows: usize, logits: DeviceVector<'a>, last: DeviceVector<'a>, normalized: DeviceVector<'a>, cache: Vec<Cache<'a>>,
}
impl<'a> State<'a> {
    fn resize(&mut self, c: &Config, rows: usize) -> Result<()> {
        let h = c.hidden_size as usize;
        self.ids.resize_within_capacity(rows)?; self.hidden.resize_within_capacity(rows*h)?;
        self.input.resize_within_capacity(rows*h)?; self.projected.resize_within_capacity(rows*h)?;
        self.gate.resize_within_capacity(rows*c.intermediate_size as usize)?;
        self.up.resize_within_capacity(self.gate.len())
    }
    fn scratch(b: &Backend<'a>, c: &Config, max_tokens: usize, rows: usize) -> Result<Self> {
        let mut buffers = b.vector_arena(&scratch_lengths(c, max_tokens, rows))?.into_iter();
        let mut next = || buffers.next().unwrap();
        let s = Self { ids: next(), hidden: next(), input: next(), sums: next(), q: next(), k: next(), v: next(),
            attention: next(), projected: next(), gate: next(), up: next(), scores: next(), input_pack: next(), weight_pack: next(),
            rows, logits: next(), last: next(), normalized: next(), metadata: next(), cache: Vec::new() };
        Ok(s)
    }
    fn new(b: &Backend<'a>, c: &Config, max_tokens: usize, rows: usize) -> Result<Self> {
        let mut s = Self::scratch(b, c, max_tokens, rows)?;
        for &kind in &c.layer_types {
            let (heads, dim) = c.geometry(kind);
            let local = kind == Attention::Sliding;
            // Chunk slack preserves every earlier row's window until attention completes.
            let capacity = if local {rows.min(max_tokens)} else {1};
            let words = capacity*heads as usize*dim as usize/2;
            let maximum = max_tokens*heads as usize*dim as usize/2;
            s.cache.push(Cache { keys: if local {b.vector(words)?} else {b.segmented_vector(maximum, words)?},
                values: if local {b.vector(words)?} else {b.segmented_vector(maximum, words)?}, capacity, local });
        }
        Ok(s)
    }
    fn allocations(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for v in [&self.ids, &self.hidden, &self.input, &self.sums, &self.q, &self.k, &self.v, &self.attention,
            &self.projected, &self.gate, &self.up, &self.scores, &self.input_pack, &self.weight_pack, &self.logits, &self.last, &self.normalized, &self.metadata] {
            v.allocations(&mut out);
        }
        for cache in &self.cache { cache.keys.allocations(&mut out); cache.values.allocations(&mut out); }
        out.sort_unstable(); out.dedup_by_key(|a| a.0); out
    }
}
pub struct Session<'m, 'a, 'b> { model: &'m Model<'a, 'b>, len: usize, state: Option<State<'a>>, draft: Option<assistant::DraftState<'a>> }
impl<'m, 'a, 'b> Session<'m, 'a, 'b> {
    pub fn allocations(&self) -> Vec<(usize, usize)> {
        let mut out = self.state.as_ref().map_or_else(Vec::new, State::allocations);
        if let Some(d) = &self.draft { d.allocations(&mut out); }
        out.sort_unstable(); out.dedup_by_key(|a| a.0); out
    }
    pub fn bytes(&self) -> usize { self.allocations().iter().map(|a| a.1).sum() }
    pub fn peak_bytes(&self) -> usize { self.model.backend.peak_allocated_bytes() }
    pub fn truncate(&mut self, position: usize) -> Result<()> {
        if let Some(d) = &mut self.draft {
            if d.end > d.entry && position >= d.entry && position <= self.len && self.len <= d.end {
                let h = self.model.config.hidden_size as usize;
                let b = self.model.backend;
                if position == d.entry { b.copy_vector_range(&d.prior, &mut d.carry, 0, 0, h)?; }
                else { b.copy_vector_range(&d.normalized, &mut d.carry, (position-d.entry-1)*h, 0, h)?; }
                self.len = position;
                return Ok(());
            }
        }
        if position != self.len { return Err("Gemma truncation outside the retained verification tail".into()); }
        Ok(())
    }
    pub fn fork(&self) -> Result<Self> {
        let mut result = Self { model: self.model, len: self.len, state: None, draft: None };
        if let Some(source) = &self.state {
            let b = self.model.backend;
            let mut state = State::new(b, &self.model.config, self.model.max_tokens, DECODE_ROWS)?;
            for (i, (from, to)) in source.cache.iter().zip(&mut state.cache).enumerate() {
                if from.local {
                    let (heads, dim) = self.model.config.geometry(self.model.config.layer_types[i]);
                    to.grow_local(b, from.capacity, 0, (heads*dim) as usize, self.model.config.sliding_window as usize)?;
                    b.copy_vector_range(&from.keys, &mut to.keys, 0, 0, from.keys.len())?;
                    b.copy_vector_range(&from.values, &mut to.values, 0, 0, from.values.len())?;
                } else {
                    let stride = self.model.config.geometry(Attention::Full);
                    let words = self.len*stride.0 as usize*stride.1 as usize/2;
                    to.keys = b.fork_vector(&from.keys, words, words.max(1))?;
                    to.values = b.fork_vector(&from.values, words, words.max(1))?;
                    to.capacity = from.capacity;
                }
            }
            result.state = Some(state);
        }
        if let Some(source) = &self.draft {
            result.prepare_draft()?;
            let d = result.draft.as_mut().unwrap();
            self.model.backend.copy_vector_range(&source.carry, &mut d.carry, 0, 0, source.carry.len())?;
        }
        Ok(result)
    }
    pub fn prefill(&mut self, tokens: &[u32], logits: bool) -> Result<Vec<f32>> {
        if tokens.is_empty() || self.len.checked_add(tokens.len()).is_none_or(|n| n > self.model.max_tokens)
            || tokens.iter().any(|&t| t as u64 >= self.model.config.vocab_size) { return Err("invalid Gemma prompt or context".into()); }
        let mut output = Vec::new();
        for (i, chunk) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            output = self.run(chunk, logits && (i+1)*PREFILL_CHUNK >= tokens.len(), false)?;
        }
        if self.state.as_ref().is_some_and(|s| s.rows > DECODE_ROWS) {
            let mut state = State::scratch(self.model.backend, &self.model.config, self.model.max_tokens, DECODE_ROWS)?;
            state.cache = std::mem::take(&mut self.state.as_mut().unwrap().cache);
            self.state = Some(state);
        }
        Ok(output)
    }
    /// One target weight pass for a chronological verification tail. Local
    /// rings retain window+rows entries, so rejecting the tail evicts no
    /// history needed by the next canonical step.
    pub fn verify_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() || tokens.len() > 16 || !self.model.has_assistant()
            || self.len.checked_add(tokens.len()).is_none_or(|n| n > self.model.max_tokens)
            || tokens.iter().any(|&t| t as u64 >= self.model.config.vocab_size) {
            return Err("invalid Gemma verification tail".into());
        }
        self.prepare_draft()?;
        let d = self.draft.as_mut().unwrap();
        let b = self.model.backend;
        b.copy_vector_range(&d.carry, &mut d.prior, 0, 0, d.carry.len())?;
        d.entry = self.len; d.end = self.len+tokens.len();
        let entry = self.len;
        match self.run(tokens, true, true) {
            Ok(logits) => Ok(logits.chunks(self.model.config.vocab_size as usize).map(|r| r.to_vec()).collect()),
            Err(error) => Err(match self.truncate(entry) {
                Ok(()) => error, Err(rollback) => format!("{error}; rollback failed: {rollback}")
            }),
        }
    }
    fn run(&mut self, tokens: &[u32], logits: bool, all: bool) -> Result<Vec<f32>> {
        self.prepare_draft()?;
        if !all { if let Some(d) = &mut self.draft { d.end = 0; } }
        let m = self.model; let c = &m.config; let b = m.backend;
        if self.state.is_none() { self.state = Some(State::new(b, c, m.max_tokens, tokens.len().max(DECODE_ROWS))?); }
        if self.state.as_ref().unwrap().rows < tokens.len() {
            let mut state = State::scratch(b, c, m.max_tokens, tokens.len())?;
            state.cache = std::mem::take(&mut self.state.as_mut().unwrap().cache);
            self.state = Some(state);
        }
        let s = self.state.as_mut().unwrap();
        let rows = tokens.len(); let h = c.hidden_size as usize;
        s.resize(c, rows)?;
        b.write_vector(&mut s.ids, &tokens.iter().map(|&t| t as f32).collect::<Vec<_>>())?;
        b.batch_embedding_device(&m.embedding, &s.ids, &mut s.hidden)?;
        let scale = bf16((c.hidden_size as f32).sqrt());
        b.model_elementwise_device(&s.hidden, None, &s.hidden, rows*h, 0, scale)?;
        for (i, l) in m.layers.iter().enumerate() {
            let kind = c.layer_types[i]; let (kv, dim) = c.geometry(kind); let (kv, dim) = (kv as usize, dim as usize);
            let cache = &mut s.cache[i];
            cache.prepare(b, self.len, rows+usize::from(all), kv*dim, c.sliding_window as usize, m.max_tokens)?;
            s.q.resize_within_capacity(rows*c.attention_heads as usize*dim)?; s.attention.resize_within_capacity(s.q.len())?;
            s.k.resize_within_capacity(rows*kv*dim)?; s.v.resize_within_capacity(s.k.len())?;
            layer::forward_normalized(b, layer::Weights { q: Some(&l.q), k: l.k.as_ref(), v: l.v.as_ref(), o: Some(&l.o),
                gate: Some(&l.gate), up: Some(&l.up), down: Some(&l.down) },
                layer::BlockNorms { input: &l.input, attention: &l.attention, pre_mlp: &l.pre_mlp, post_mlp: &l.post_mlp,
                    q: &l.q_norm, k: l.k_norm.as_ref().unwrap(), scalar: l.scalar },
                if cache.local {&m.local_rope} else {&m.global_rope}, &mut cache.keys, &mut cache.values,
                layer::Buffers { hidden: &mut s.hidden, input: &mut s.input, sums: &mut s.sums, q: &mut s.q, k: &mut s.k, v: &mut s.v,
                    attention: &mut s.attention, projected: &mut s.projected, gate: &mut s.gate, up: &mut s.up, scores: &mut s.scores,
                    parts: &mut s.last, weight_pack: &mut s.weight_pack, input_pack: &mut s.input_pack,
                    packed_keys: &mut s.normalized, packed_values: &mut s.logits },
                layer::Run { position: self.len, tokens: rows, first: i == 0, prompt_end: self.len+rows,
                    heads: c.attention_heads as usize, kv_heads: kv, head_dim: dim, epsilon: c.rms_norm_eps as f32,
                    exact: true, format: KvFormat::Bf16, decode: rows == 1, sums_ready: false },
                if cache.local {c.sliding_window as usize} else {0}, cache.capacity)?;
        }
        self.len += rows;
        if all {
            let d = self.draft.as_mut().ok_or("missing verification state")?;
            d.normalized.resize_within_capacity(rows*h)?;
            d.logits.resize_within_capacity(rows*c.vocab_size as usize)?;
            b.model_rms_device(&s.hidden, Some(&m.norm), &mut d.normalized, &mut s.sums, h, rows, c.rms_norm_eps as f32)?;
            b.copy_vector_range(&d.normalized, &mut d.carry, (rows-1)*h, 0, h)?;
            b.matmul_bf16_device(&m.head, &d.normalized, &mut d.logits, &mut s.weight_pack, &mut s.input_pack, rows, false)?;
            b.model_elementwise_device(&d.logits, None, &d.logits, d.logits.len(), 3, c.final_logit_softcapping as f32)?;
            return b.read_vector(&d.logits);
        }
        if !logits && self.draft.is_none() { return Ok(Vec::new()); }
        b.copy_vector_range(&s.hidden, &mut s.last, (rows-1)*h, 0, h)?;
        b.model_rms_device(&s.last, Some(&m.norm), &mut s.normalized, &mut s.sums, h, 1, c.rms_norm_eps as f32)?;
        if let Some(d) = &mut self.draft { b.copy_vector_range(&s.normalized, &mut d.carry, 0, 0, h)?; }
        if !logits { return Ok(Vec::new()); }
        layer::project(b, &m.head, &s.normalized, &mut s.logits, &mut s.weight_pack, &mut s.input_pack, 1, true, false, false)?;
        b.model_elementwise_device(&s.logits, None, &s.logits, s.logits.len(), 0, 1.0)?;
        b.model_elementwise_device(&s.logits, None, &s.logits, s.logits.len(), 3, c.final_logit_softcapping as f32)?;
        b.read_vector(&s.logits)
    }
}
fn bf16(x: f32) -> f32 { f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits()>>16)&1)) & 0xffff0000) }
