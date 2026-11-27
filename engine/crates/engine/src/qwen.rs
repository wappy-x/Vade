//! Dense and expert Qwen3 execution with immutable BF16 GPU matrices and per-request KV.
//! Device-resident FP32 activations, FP32 KV by default, opt-in BF16/scaled I16 KV.
//! Checkpoint files must remain immutable; metadata is not authentication.
use engine_model::{qwen3::{Checkpoint, Config}, safetensors::{Limits, TensorFile}};
use spark_gpu_probe::driver::{Backend, DeviceVector, ExpertTable, KvFormat, Matrix};
use std::{collections::BTreeMap, path::Path, time::Instant};
use crate::context::{admit_kv, KvCapacity, DEFAULT_KV_BUDGET_BYTES};
type Result<T> = std::result::Result<T, String>;
#[path = "device.rs"]
mod device;
#[path = "multi.rs"]
pub mod multi;
use device::{DeviceModel, DeviceState, Observer, Logits};
struct Layer<'a> {
    input_norm: Vec<f32>, post_norm: Vec<f32>, q_norm: Vec<f32>, k_norm: Vec<f32>,
    q: Matrix<'a>, k: Matrix<'a>, v: Matrix<'a>, o: Matrix<'a>,
    mlp: Mlp<'a>,
}
#[allow(clippy::large_enum_variant)] // boxing would add an allocation per value.
enum Mlp<'a> {
    Dense { gate: Matrix<'a>, up: Matrix<'a>, down: Matrix<'a> },
    Experts { router: Matrix<'a>, experts: ExpertTable<'a> },
}
impl<'a> Mlp<'a> {
    fn dense(&self) -> Result<(&Matrix<'a>, &Matrix<'a>, &Matrix<'a>)> {
        match self {
            Self::Dense { gate, up, down } => Ok((gate, up, down)),
            Self::Experts { .. } => Err("this operation requires a dense MLP".into()),
        }
    }
}
pub struct Model<'a, 'b> {
    backend: &'b Backend<'a>, config: Config, layers: Vec<Layer<'a>>,
    head: Option<Matrix<'a>>, max_tokens: usize, layer_start: usize,
    kv_capacity: KvCapacity, kv_budget_bytes: usize,
    device: DeviceModel<'a>,
    draft_catchup_ms_per_token: std::cell::Cell<f64>,
}
impl<'a, 'b> Model<'a, 'b> {
    fn profile_layer(&self, length: usize, format: KvFormat) {
        use engine_model::{plan::profile::ProfilePlan, safetensors::Dtype};
        let b = self.backend;
        b.generic_plan(|| {
            let plan = ProfilePlan::qwen(&self.config);
            b.prefill_profile_layer(|| plan.layer_name());
            for read in plan.reads(Dtype::Bf16) { b.generic_read_bytes(|| read); }
            b.generic_read_bytes(|| (2, 2 * format.storage_len(length * self.kv_capacity.stride) * 4));
        });
    }
    fn profile_head(&self) {
        self.backend.prefill_profile_layer(|| "output");
        self.backend.generic_read_bytes(|| (8, self.head.as_ref().map_or(0, |m| m.rows() * m.cols() * 2)));
        self.backend.generic_read_bytes(|| (7, self.config.hidden_size as usize * 4));
    }
    pub fn load(backend: &'b Backend<'a>, root: &Path, max_tokens: usize) -> Result<Self> {
        Self::load_with_kv_budget(backend, root, max_tokens, DEFAULT_KV_BUDGET_BYTES)
    }
    /// Explicit per-session FP32 K+V payload admission. Model weights, transient
    /// activations and aggregate live/cached sessions require separate budgets.
    pub fn load_with_kv_budget(backend: &'b Backend<'a>, root: &Path, max_tokens: usize,
        kv_budget_bytes: usize) -> Result<Self> {
        Self::load_with_yarn(backend, root, max_tokens, kv_budget_bytes, false)
    }
    pub fn load_with_yarn(backend: &'b Backend<'a>, root: &Path, max_tokens: usize,
        kv_budget_bytes: usize, yarn: bool) -> Result<Self> {
        Self::load_range(backend, root, max_tokens, kv_budget_bytes, None, yarn)
    }
    fn load_range(backend: &'b Backend<'a>, root: &Path, max_tokens: usize,
        kv_budget_bytes: usize,
        partition: Option<(usize, usize, bool, bool)>, yarn: bool) -> Result<Self> {
        let started = std::time::Instant::now();
        let checkpoint = Checkpoint::inspect(root).map_err(|e| e.to_string())?;
        let mut config = checkpoint.config;
        if yarn { config.enable_yarn().map_err(|e| e.to_string())?; }
        // Every projection uses packed BF16 tiles. Reject unsupported tiles
        // before allocating weights, with the checkpoint tensor that needs them.
        let mut expected = config.expected_shapes();
        if config.tie_word_embeddings && checkpoint.index.weight_map.contains_key("lm_head.weight") {
            expected.insert("lm_head.weight".into(), vec![config.vocab_size, config.hidden_size]);
        }
        for (name, shape) in &expected {
            if shape.len() == 2 && name != "model.embed_tokens.weight"
                && (shape[0] % 8 != 0 || shape[1] % 16 != 0) {
                return Err(format!("unsupported Qwen3 tensor {name} shape {shape:?}: packed projections require rows divisible by 8 and columns divisible by 16"));
            }
        }
        if max_tokens > u32::MAX as usize {
            return Err("unsupported Qwen3 context: GPU positions must fit u32".into());
        }
        if config.head_dim == 128 && (config.attention_heads == config.kv_heads * 4 || config.num_experts != 0 && config.attention_heads == config.kv_heads * 8)
            && backend.decode_attention_parts(max_tokens) > 1024 {
            return Err(format!("unsupported Qwen3 context {max_tokens}: streaming attention supports at most 1024 summary tiles"));
        }
        backend.require_layer_kernels()?;
        let total_layers = usize::try_from(config.layers).map_err(|_| "layer count exceeds host address space")?;
        let (start, end, owns_embedding, owns_output) = partition.unwrap_or((0, total_layers, true, true));
        if start >= end || end > total_layers || owns_embedding != (start == 0) || owns_output != (end == total_layers) {
            return Err("invalid contiguous partition range or boundary ownership".into());
        }
        let kv_capacity = admit_kv(start..end, config.layers, config.kv_heads, config.head_dim,
            max_tokens, config.max_positions, kv_budget_bytes)?;
        let tokens = max_tokens as u64;
        let query = config.attention_heads * config.head_dim;
        let streaming = config.head_dim == 128 && (config.attention_heads == config.kv_heads * 4 || config.num_experts != 0 && config.attention_heads == config.kv_heads * 8);
        let attention_rows = tokens.min(PREFILL_CHUNK.min(64) as u64);
        let prefill_rows = tokens.min(if max_tokens >= 6144 { 4096 } else { PREFILL_CHUNK as u64 }).div_ceil(16) * 16;
        let padded_tokens = tokens.div_ceil(16) * 16;
        let chunks = tokens.div_ceil(64);
        if !streaming && chunks > 65_535 {
            return Err(spark_gpu_probe::admission::token_refusal(max_tokens as u128, 65535 * 64, &format!("CONTEXT={max_tokens}; OUTPUT included in CONTEXT"), "generic attention's GPU grid-z limit of 65535 tiles x 64 tokens", spark_gpu_probe::admission::SERVE));
        }
        for (name, elements) in [
            ("KV", kv_capacity.elements_per_layer as u64),
            ("RoPE", tokens * config.head_dim),
            ("prefill activations", prefill_rows * config.hidden_size.max(query).max(config.intermediate_size)),
            ("multi logits", 2048 * config.vocab_size),
            ("packed prefill KV", if config.head_dim == 128 { tokens.div_ceil(64) * 64 * config.kv_heads * config.head_dim } else { 1 }),
            ("attention scores", if streaming { 1 } else { tokens * config.attention_heads }),
            ("attention partials", if streaming { 1 } else { query * chunks }),
            ("batch attention scores", if streaming { 1 } else { attention_rows * config.attention_heads * padded_tokens }),
            ("batch attention partials", if streaming { 1 } else { attention_rows * query * padded_tokens.div_ceil(64) }),
        ] {
            if elements > u32::MAX as u64 {
                return Err(format!("unsupported Qwen3 {name} allocation: {elements} elements exceed GPU u32 vector capacity"));
            }
        }
        let mut files = BTreeMap::new();
        for shard in checkpoint.shards.keys() {
            files.insert(shard.clone(), TensorFile::open(&root.join(shard), Limits::default()).map_err(|e| e.to_string())?);
        }
        // Validate shapes again against the retained file descriptors actually consumed.
        let mut count = 0usize;
        for (shard, file) in &files {
            for (name, tensor) in &file.header().tensors {
                if expected.get(name) != Some(&tensor.shape)
                    || tensor.dtype != engine_model::safetensors::Dtype::Bf16
                    || checkpoint.index.weight_map.get(name) != Some(shard) {
                    return Err("retained checkpoint tensor binding mismatch".into());
                }
                count += 1;
            }
        }
        if count != expected.len() { return Err("retained checkpoint tensor count mismatch".into()); }
        let mut loader = Loader { files, map: checkpoint.index.weight_map, matrices: BTreeMap::new() };
        if config.tie_word_embeddings && loader.map.contains_key("lm_head.weight") { loader.validate_tied_embeddings()?; }
        // All packed projections use the canonical blocked BF16 layout.
        loader.load_matrices(backend, start..end, owns_embedding, owns_output,
            config.num_experts != 0 || config.tie_word_embeddings, config.tie_word_embeddings, "model", Some(&config))?;
        let norm = if owns_output { Some(loader.host("model.norm.weight", 1024 * 1024)?) } else { None };
        let table = if owns_embedding || owns_output && config.tie_word_embeddings {
            Some(loader.matrix("model.embed_tokens.weight")?)
        } else { None };
        let head = if owns_output {
            Some(if config.tie_word_embeddings { table.as_ref().unwrap().rows_view(0..config.vocab_size as usize)? }
                else { loader.matrix("lm_head.weight")? })
        } else { None };
        let mut layers = Vec::new();
        for i in start..end {
            let p = format!("model.layers.{i}");
            let mut host = |s: &str| loader.host(&format!("{p}.{s}.weight"), 1024 * 1024);
            let (input_norm, post_norm) = (host("input_layernorm")?, host("post_attention_layernorm")?);
            let (q_norm, k_norm) = if config.qk_norm { (host("self_attn.q_norm")?, host("self_attn.k_norm")?) }
                else { (vec![1.0; config.head_dim as usize], vec![1.0; config.head_dim as usize]) };
            let mut mat = |s: &str| loader.matrix(&format!("{p}.{s}.weight"));
            let mlp = if config.num_experts == 0 {
                Mlp::Dense { gate: mat("mlp.gate_proj")?, up: mat("mlp.up_proj")?, down: mat("mlp.down_proj")? }
            } else {
                let router = mat("mlp.gate")?;
                let mut gate = Vec::with_capacity(config.num_experts as usize);
                let mut up = Vec::with_capacity(config.num_experts as usize);
                let mut down = Vec::with_capacity(config.num_experts as usize);
                for expert in 0..config.num_experts {
                    let p = format!("mlp.experts.{expert}");
                    gate.push(mat(&format!("{p}.gate_proj"))?);
                    up.push(mat(&format!("{p}.up_proj"))?);
                    down.push(mat(&format!("{p}.down_proj"))?);
                }
                Mlp::Experts { router, experts: backend.expert_table(gate, up, down)? }
            };
            layers.push(Layer { input_norm, post_norm, q_norm, k_norm,
                q: mat("self_attn.q_proj")?, k: mat("self_attn.k_proj")?, v: mat("self_attn.v_proj")?, o: mat("self_attn.o_proj")?,
                mlp });
            eprintln!("loaded_layer={} gpu_bytes={}", i + 1, backend.allocated_bytes());
        }
        let table = if owns_embedding { table } else { None };
        let device = DeviceModel::new(backend, &config, &layers, norm.as_deref(), table, max_tokens)?;
        for layer in &mut layers {
            layer.input_norm = Vec::new(); layer.post_norm = Vec::new();
            layer.q_norm = Vec::new(); layer.k_norm = Vec::new();
        }
        eprintln!("model_load_seconds={:.3} device_bytes={} peak_device_bytes={} pinned_upload_bytes={}",
            started.elapsed().as_secs_f64(), backend.allocated_bytes(), backend.peak_allocated_bytes(), backend.pinned_upload_bytes());
        Ok(Self { backend, config, layers, head, max_tokens, layer_start: start,
            kv_capacity, kv_budget_bytes, device,
            // GB10 0.6B prefill catchup: 2.07 s for 28,672 tokens.
            draft_catchup_ms_per_token: std::cell::Cell::new(2070.0 / 28672.0) })
    }
    fn observe_draft_catchup(&self, tokens: usize, ms: f64) {
        // Short suffixes are dominated by launches, not long-prefill throughput.
        if tokens >= 4096 && ms.is_finite() && ms > 0.0 {
            self.draft_catchup_ms_per_token.set(ms / tokens as f64);
        }
    }
    /// Projections arrive packed directly from the loader; no conversion allocation.
    pub fn prepare_batch_weights(&mut self) -> Result<usize> {
        self.backend.require_batch_kernels()?;
        Ok(self.backend.allocated_bytes())
    }
    fn embed(&self, token: u32) -> Result<Vec<f32>> {
        if token as u64 >= self.config.vocab_size { return Err("token outside vocabulary".into()); }
        self.backend.shared_bf16_row(self.device.embedding.as_ref().ok_or("partition does not own embedding")?, token as usize)
    }
    fn check_hidden(&self, hidden: &[f32]) -> Result<()> {
        if hidden.len() != self.config.hidden_size as usize || hidden.iter().any(|v| !v.is_finite()) {
            return Err("hidden state has invalid width or nonfinite value".into());
        }
        Ok(())
    }
    pub fn config(&self) -> &Config { &self.config }
    /// Borrow the canonical weights for execution with a different partition.
    pub fn backend(&self) -> &'b Backend<'a> { self.backend }
    /// Expert layers expose no dense gate/up/down projections.
    pub fn projection_weights(&self, index: usize) -> crate::layer::Weights<'_, 'a> {
        let l = &self.layers[index];
        let dense = l.mlp.dense().ok();
        crate::layer::Weights { q: Some(&l.q), k: Some(&l.k), v: Some(&l.v), o: Some(&l.o),
            gate: dense.map(|d| d.0), up: dense.map(|d| d.1), down: dense.map(|d| d.2) }
    }
    pub fn projection_norms(&self, index: usize) -> crate::layer::Norms<'_, 'a> { self.device.projection_norms(index) }
    pub fn rope(&self) -> &spark_gpu_probe::driver::DeviceVector<'a> { self.device.rope() }
    pub fn output_norm(&self) -> Result<&spark_gpu_probe::driver::DeviceVector<'a>> { self.device.final_norm() }
    pub fn output_weights(&self) -> Result<&Matrix<'a>> { self.head.as_ref().ok_or_else(|| "partition has no output weights".into()) }
    pub fn embedding(&self, token: u32) -> Result<Vec<f32>> { self.embed(token) }
    /// Logical BF16 projection and FP32 normalization payload read by one full
    /// decode token. Selected experts remain in place; cache/activation traffic
    /// and repeated hardware cache-line fetches are not weight payload.
    pub fn active_weight_bytes_per_token(&self) -> u64 {
        let c = &self.config;
        let attention = 2 * c.hidden_size * c.head_dim * (c.attention_heads + c.kv_heads);
        let mlp = if c.num_experts == 0 { 3 * c.hidden_size * c.intermediate_size }
            else { c.hidden_size * c.num_experts + 3 * c.hidden_size * c.moe_intermediate_size * c.num_experts_per_tok };
        let per_layer = 2 * (attention + mlp) + 4 * (2 * c.hidden_size + 2 * c.head_dim);
        per_layer * self.layers.len() as u64
            + if self.device.embedding.is_some() { 2 * c.hidden_size } else { 0 }
            + if self.head.is_some() { 2 * c.vocab_size * c.hidden_size + 4 * c.hidden_size } else { 0 }
    }
    pub fn max_tokens(&self) -> usize { self.max_tokens }
    pub fn kv_budget_bytes(&self) -> usize { self.kv_budget_bytes }
    pub fn max_cache_bytes(&self) -> usize { self.kv_capacity.bytes }
    pub fn memory_bytes(&self) -> (usize, usize, usize) {
        (self.backend.allocated_bytes(), self.backend.peak_allocated_bytes(), self.backend.pinned_host_bytes())
    }
    pub fn maintain_kv_reserve(&self, idle: bool) -> Result<()> { self.backend.maintain_kv_reserve(idle) }
    pub fn fatal_error(&self) -> Option<String> { self.backend.fatal_error() }
    pub fn session(&self) -> Session<'_, 'a, 'b> {
        Session { model: self, len: 0, device: None, snapshot: None, kv_format: None, sequence: None, fast_prefill: false,
            draft: None, draft_k: 0, draft_limit: 0, draft_kv_peak: 0 }
    }
}
pub(crate) struct Loader<'a> { pub(crate) files: BTreeMap<String, TensorFile>, pub(crate) map: BTreeMap<String, String>, pub(crate) matrices: BTreeMap<String, Matrix<'a>> }
impl<'a> Loader<'a> {
    pub(crate) fn load_matrices(&mut self, backend: &Backend<'a>, layers: std::ops::Range<usize>, embedding: bool, head: bool, single_arena: bool, tied: bool, namespace: &str, config: Option<&Config>) -> Result<()> {
        use spark_gpu_probe::load::{WeightRows, WeightShard};
        let mut plans = Vec::new();
        for file in self.files.values() {
            let mut tensors = file.header().tensors.iter().filter(|(name, t)| {
                t.shape.len() == 2 && (if name.as_str() == format!("{namespace}.embed_tokens.weight") { embedding || head && tied }
                    else if name.as_str() == "lm_head.weight" { head && !tied }
                    else { name.strip_prefix(&format!("{namespace}.layers.")).and_then(|s| s.split('.').next())
                        .and_then(|s| s.parse::<usize>().ok()).is_some_and(|i| layers.contains(&i)) })
            }).collect::<Vec<_>>();
            tensors.sort_unstable_by_key(|(_, t)| t.start);
            if tensors.is_empty() { continue; }
            let mut names = Vec::new();
            let mut ranges = Vec::new();
            for (name, tensor) in tensors {
                let rows = usize::try_from(tensor.shape[0]).map_err(|_| "rows overflow")?;
                let columns = usize::try_from(tensor.shape[1]).map_err(|_| "columns overflow")?;
                let mut append = |name: String, rows: std::ops::Range<usize>, packed: bool| {
                    names.push(name);
                    ranges.push(WeightRows { file_offset: file.header().data_start + tensor.start,
                        rows, columns, packed, fp8: false });
                };
                if let Some(c) = config.filter(|c| c.fused_projections) {
                    if let Some(base) = name.strip_suffix("qkv_proj.weight") {
                        let q = (c.attention_heads * c.head_dim) as usize;
                        let k = (c.kv_heads * c.head_dim) as usize;
                        append(format!("{base}q_proj.weight"), 0..q, true);
                        append(format!("{base}k_proj.weight"), q..q+k, true);
                        append(format!("{base}v_proj.weight"), q+k..rows, true);
                        continue;
                    }
                    if let Some(base) = name.strip_suffix("gate_up_proj.weight") {
                        let m = c.intermediate_size as usize;
                        append(format!("{base}gate_proj.weight"), 0..m, true);
                        append(format!("{base}up_proj.weight"), m..rows, true);
                        continue;
                    }
                }
                append(name.clone(), 0..rows, name.as_str() != format!("{namespace}.embed_tokens.weight") || tied);
            }
            plans.push((file.file(), names, ranges));
        }
        let shards = plans.iter().map(|(file, _, matrices)| WeightShard { file, matrices }).collect::<Vec<_>>();
        let loaded = if single_arena { backend.load_bf16_shards_in_arena(&shards)? } else { backend.load_bf16_shards(&shards)? };
        for ((_, names, _), matrices) in plans.iter().zip(loaded) {
            self.matrices.extend(names.iter().cloned().zip(matrices));
        }
        Ok(())
    }
    fn file(&mut self, name: &str) -> Result<&mut TensorFile> {
        let shard = self.map.get(name).ok_or("missing indexed tensor")?;
        self.files.get_mut(shard).ok_or_else(|| "missing retained shard".into())
    }
    fn validate_tied_embeddings(&mut self) -> Result<()> {
        let bytes = self.file("model.embed_tokens.weight")?.header().tensors["model.embed_tokens.weight"].byte_len();
        let mut embedding = vec![0u8; (bytes as usize).min(4*1024*1024)];
        let mut head = vec![0u8; embedding.len()];
        let mut offset = 0;
        while offset < bytes {
            let n = (bytes-offset).min(embedding.len() as u64) as usize;
            self.file("model.embed_tokens.weight")?.read_range("model.embed_tokens.weight", offset, &mut embedding[..n]).map_err(|e| e.to_string())?;
            self.file("lm_head.weight")?.read_range("lm_head.weight", offset, &mut head[..n]).map_err(|e| e.to_string())?;
            if embedding[..n] != head[..n] { return Err("checkpoint declares tied embeddings but stored head differs".into()); }
            offset += n as u64;
        }
        Ok(())
    }
    pub(crate) fn matrix(&mut self, name: &str) -> Result<Matrix<'a>> {
        self.matrices.remove(name).ok_or_else(|| "missing loaded matrix".into())
    }
    pub(crate) fn host(&mut self, name: &str, limit: usize) -> Result<Vec<f32>> {
        let file = self.file(name)?;
        let bytes = usize::try_from(file.header().tensors.get(name).ok_or("missing tensor")?.byte_len()).map_err(|_| "host tensor size overflow")?;
        if bytes.checked_mul(2).ok_or("decoded tensor size overflow")? > limit { return Err(spark_gpu_probe::admission::refusal(bytes as u128 * 2, limit, "decoded host weight tensor; weights/rank, per-session bytes, sessions, CONTEXT and OUTPUT not supplied to loader", "the caller's host tensor budget", spark_gpu_probe::admission::SERVE)); }
        let mut result = Vec::with_capacity(bytes / 2);
        let mut chunk = vec![0u8; bytes.min(4 * 1024 * 1024)];
        let mut offset = 0;
        while offset < bytes {
            let n = chunk.len().min(bytes - offset);
            file.read_range(name, offset as u64, &mut chunk[..n]).map_err(|e| e.to_string())?;
            result.extend(chunk[..n].as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)));
            offset += n;
        }
        if result.iter().any(|v| !v.is_finite()) { return Err("nonfinite host tensor".into()); }
        Ok(result)
    }
}
struct SequenceProgress {
    position: usize, tokens: usize, total: usize, prompt_end: usize, layer: usize,
    beginning: bool, projected: bool, invalid: bool, covered: Vec<Vec<bool>>,
}
pub struct Session<'m, 'a, 'b> {
    model: &'m Model<'a, 'b>, len: usize, device: Option<DeviceState<'a>>, snapshot: Option<std::rc::Rc<Vec<device::Kv<'a>>>>, kv_format: Option<KvFormat>,
    sequence: Option<SequenceProgress>, fast_prefill: bool,
    draft: Option<Box<Session<'m, 'a, 'b>>>, draft_k: usize, draft_limit: usize, draft_kv_peak: usize,
}
impl Drop for Session<'_, '_, '_> {
    fn drop(&mut self) {
        let bytes = self.cache_bytes();
        let target_bytes = bytes - self.draft.as_ref().map_or(0, |s| s.cache_bytes());
        self.model.backend.kv_conversation_ended(bytes);
        let profile = std::env::var("SPARK_PROFILE").as_deref() == Ok("1") && self.device.is_some();
        let started = Instant::now();
        drop(self.device.take());
        drop(self.snapshot.take());
        let target_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        drop(self.draft.take());
        if profile {
            eprintln!("session_retire_profile layers={} target_ms={target_ms:.3} draft_ms={:.3} target_kv_bytes={target_bytes}",
                self.model.layers.len(), started.elapsed().as_secs_f64() * 1000.0);
        }
    }
}
impl<'m, 'a, 'b> Session<'m, 'a, 'b> {
    /// The small model shares immutable weights; only its KV belongs to this conversation.
    pub fn with_draft_model(mut self, model: &'m Model<'a, 'b>, k: usize) -> Self {
        self.draft = Some(Box::new(model.session()));
        self.draft_k = k;
        self.draft_limit = model.max_tokens;
        self
    }
    pub fn disable_model_draft(&mut self) { self.draft = None; }
    /// A cached fork whose snapshot released its proposal KV gets an empty
    /// draft; the shared proposal pass catches it up from the target prefix.
    pub fn restore_model_draft(&mut self, model: &'m Model<'a, 'b>, k: usize) {
        if self.draft.is_none() { self.draft = Some(Box::new(model.session())); self.draft_k = k; }
    }
    pub fn reserve_model_draft(&mut self, tokens: usize) { self.draft_limit = tokens; }
    pub fn draft_model_width(&self) -> usize {
        self.draft.as_ref().map_or(0, |draft|
            self.draft_k.min(draft.model.max_tokens.min(self.draft_limit).saturating_sub(self.len)))
    }
    pub fn draft_model_exhausted(&self) -> bool {
        self.draft.as_ref().is_some_and(|draft| self.len >= draft.model.max_tokens)
    }
    pub fn draft_model_catchup(&self) -> (usize, f64) {
        self.draft.as_ref().map_or((0, 0.0), |draft|
            (self.len.saturating_sub(draft.len), draft.model.draft_catchup_ms_per_token.get()))
    }
    pub fn draft_model_kv_bytes(&self) -> (usize, usize) {
        (self.draft.as_ref().map_or(0, |s| s.cache_bytes()), self.draft_kv_peak)
    }
    pub fn retain_draft_prefix(&mut self, source: &Self) -> Result<bool> {
        if !std::ptr::eq(self.model, source.model) { return Err("draft cache model mismatch".into()); }
        let Some(source) = &source.draft else { return Ok(false); };
        let tokens = self.len.min(source.len);
        if tokens <= self.draft.as_ref().map_or(0, |draft| draft.len) { return Ok(false); }
        // Retention must not queue child copies ahead of the next target
        // token. Keep the admitted page slack, or omit an oversized draft.
        let vector = tokens * source.model.kv_capacity.stride * 2;
        let page = 2 * 1024 * 1024;
        let bound = (vector.div_ceil(page) + 2) * page * source.model.layers.len() * 2;
        if source.cache_bytes() > bound { return Ok(false); }
        let mut draft = source.fork()?;
        draft.len = tokens;
        self.draft_kv_peak = self.draft_kv_peak.max(draft.cache_bytes());
        self.draft = Some(Box::new(draft));
        Ok(true)
    }
    pub fn propose_model(&mut self, history: &[u32], next: u32, limit: usize) -> Result<Option<crate::draft::ModelProposal>> {
        let width = self.draft_model_width();
        if width == 0 { return Ok(None); }
        let Some(draft) = &mut self.draft else { return Ok(None); };
        if history.len() != self.len { return Err("draft prefix differs from target checkpoint".into()); }
        let count = limit.min(width);
        if count == 0 { return Ok(Some(crate::draft::ModelProposal { tokens: Vec::new(), proposal_ms: 0.0, catchup_ms: 0.0 })); }
        let catchup = std::time::Instant::now();
        // Target truncation already discarded every rejected draft suffix.
        // Catch up accepted bonus tokens and any intervening plain decoding.
        if draft.len > history.len() { draft.truncate(history.len())?; }
        if draft.len < history.len() {
            let missing = &history[draft.len..];
            if missing.len() <= 8 {
                // Bonus-token catchup needs KV only. Queue on the same stream
                // as the following proposal, without scratch allocation or a sync.
                for &token in missing { draft.device_run(token, false, &mut None)?; draft.len += 1; }
            } else {
                let started = std::time::Instant::now();
                draft.prefill_advance(missing)?;
                draft.model.observe_draft_catchup(missing.len(), started.elapsed().as_secs_f64()*1000.0);
                draft.device.as_mut().unwrap().release_batch();
            }
        }
        let catchup_ms = catchup.elapsed().as_secs_f64() * 1000.0;
        let started = std::time::Instant::now();
        if next as u64 >= draft.model.config.vocab_size || count > draft.model.max_tokens - draft.len {
            return Err("draft proposal exceeds token capacity or vocabulary".into());
        }
        draft.prepare_device()?;
        let state = draft.device.as_mut().unwrap();
        let m = draft.model;
        // Reserve the whole chain before queueing; growth/drop must not insert
        // a synchronization between the device-selected proposal tokens.
        state.reserve_cache(m, draft.len, draft.len + count)?;
        state.reserve_greedy(m.backend, count)?;
        if !state.greedy_graph(m, draft.len, next, count)? {
            for row in 0..count {
                if row == 0 { m.backend.embedding_device(m.device.embedding.as_ref().ok_or("missing draft embedding")?, next, &mut state.hidden)?; }
                else { state.embedding_selected(m, row - 1)?; }
                state.layers_reserved(m, &m.device, draft.len + row, &mut None)?;
                state.project(m, &m.device, &mut None)?;
                state.greedy_enqueue(m.backend, row)?;
            }
        }
        let results = state.greedy_results(m.backend, count)?;
        let proposals = results[..count].iter().map(|v| v.0).collect();
        draft.len += count;
        self.draft_kv_peak = self.draft_kv_peak.max(draft.cache_bytes());
        Ok(Some(crate::draft::ModelProposal { tokens: proposals,
            proposal_ms: started.elapsed().as_secs_f64() * 1000.0, catchup_ms }))
    }
    pub fn config(&self) -> &Config { self.model.config() }
    pub fn backend_stats(&self) -> spark_gpu_probe::driver::BackendStats { self.model.backend.stats() }
    pub fn kv_bf16(&self) -> bool { self.kv_format() == KvFormat::Bf16 }
    pub fn kv_format(&self) -> KvFormat { self.kv_format.unwrap_or(KvFormat::Bf16) }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn reset(&mut self) {
        if let Some(draft) = &mut self.draft { draft.reset(); }
        if let Some(state) = &mut self.device && state.prefill_pending() { state.release_batch(); }
        self.len = 0;
        if self.sequence.take().is_some()
            && let Some(state) = &mut self.device { state.sequence_abort(); }
    }
    /// Logical checkpoint; only positions <= current length can be restored.
    pub fn checkpoint(&self) -> usize { self.len }
    pub fn truncate(&mut self, position: usize) -> Result<()> {
        self.check_sequence_idle()?;
        if position > self.len { return Err("cannot truncate beyond current position".into()); }
        // Prefix-cache lookup shortens a fork while its original owner is
        // still charged. Shed oversized mappings before that owner can be
        // evicted; a prefix within the rounded allowance stays lazy.
        if position < self.len && self.device.is_none() && self.snapshot.as_ref().is_some_and(|kv|
            position == 0 || !device::snapshot_fits(self.model, kv, self.kv_format(), position)) {
            let started = std::time::Instant::now();
            if position == 0 { self.snapshot = None; }
            else {
                self.prepare_device()?;
                self.device.as_mut().unwrap().reserve_cache(self.model, position, position)?;
            }
            if std::env::var("SPARK_PROFILE").as_deref() == Ok("1") {
                eprintln!("snapshot_truncate_profile from={} to={position} elapsed_ms={:.3}", self.len, started.elapsed().as_secs_f64()*1000.0);
            }
        }
        if let Some(state) = &mut self.device && state.prefill_pending() { state.release_batch(); }
        self.len = position;
        // Shorten the child's logical history before a divergent target
        // suffix, but defer its physical remapping until proposal catchup.
        // Actual retained pages stay charged on the first-token path.
        if let Some(draft) = &mut self.draft { draft.len = position.min(draft.len); }
        Ok(())
    }
    /// Reachable KV allocation; snapshots can share it until their next forward.
    /// Service admission must bound the SUM across live and cached sessions.
    pub fn cache_bytes(&self) -> usize {
        self.device.as_ref().map_or_else(|| self.snapshot.as_ref().map_or(0, |kv| device::cache_bytes(kv)), |device| device.cache_bytes())
            + self.draft.as_ref().map_or(0, |s| s.cache_bytes())
    }
    /// Physical allocations reachable from the session; shared KV appears once.
    pub fn cache_allocations(&self) -> Vec<(usize, usize)> {
        let mut allocations = self.snapshot_allocations();
        if let Some(draft) = &self.draft { allocations.extend(draft.cache_allocations()); }
        allocations
    }
    /// Completed sessions cannot grow. Charge their actual KV and scratch
    /// instead of the active reservation's extra allowance for a writable copy.
    pub fn resident_bytes(&self) -> usize {
        let mut allocations = self.device.as_ref().map_or_else(|| self.snapshot_allocations(), |s| s.allocations());
        allocations.sort_unstable_by_key(|&(id, _)| id);
        allocations.dedup_by_key(|entry| entry.0);
        allocations.iter().map(|&(_, bytes)| bytes).sum::<usize>()
            + self.draft.as_ref().map_or(0, |s| s.resident_bytes())
    }
    /// Physical target KV pages; cache_allocations also includes the draft.
    pub fn snapshot_allocations(&self) -> Vec<(usize, usize)> {
        self.device.as_ref().map_or_else(|| self.snapshot.as_ref().map_or_else(Vec::new, |kv| device::cache_allocations(kv)), |state| state.cache_allocations())
    }
    /// Immutable whole pages that a fork at this position will reference.
    pub fn prefix_allocations(&self, tokens: usize) -> Vec<(usize, usize)> {
        if self.kv_format() != KvFormat::Bf16 { return Vec::new(); }
        let tokens = tokens.min(self.len);
        let mut allocations = self.device.as_ref().map_or_else(|| self.snapshot.as_ref().map_or_else(Vec::new, |kv|
            device::prefix_allocations(kv, self.kv_format.unwrap_or(KvFormat::Bf16).storage_len(tokens * self.model.kv_capacity.stride),
                self.kv_format.unwrap_or(KvFormat::Bf16).storage_len(self.model.kv_capacity.stride))),
            |state| state.prefix_allocations(self.model, tokens));
        if let Some(draft) = &self.draft { allocations.extend(draft.prefix_allocations(tokens)); }
        allocations
    }
    /// Peak CUDA allocations observed by this session (includes its model).
    pub fn device_peak_bytes(&self) -> usize {
        self.device.as_ref().map_or(0, |state| state.peak_bytes)
            .max(self.draft.as_ref().map_or(0, |s| s.device_peak_bytes()))
    }
    /// Select storage before this session allocates KV; BF16 is the model default.
    pub fn configure_kv_bf16(&mut self, enabled: bool) -> Result<()> {
        self.configure_kv_format(enabled.into())
    }
    /// Below-model-precision cache quantization requires explicit selection.
    pub fn configure_kv_format(&mut self, format: KvFormat) -> Result<()> {
        if self.device.is_some() || self.snapshot.is_some() || self.len != 0 { return Err("KV format must be selected before first forward".into()); }
        if format == KvFormat::ScaledI16 && self.model.config.head_dim != 128 {
            return Err("scaled I16 KV requires head dimension 128".into());
        }
        self.kv_format = Some(format);
        Ok(())
    }
    /// Select the unfused prefill packing route; both routes use BF16 operands.
    pub fn configure_fast_prefill(&mut self, enabled: bool) -> Result<()> {
        if self.device.is_some() || self.len != 0 { return Err("prefill precision must be selected before first forward".into()); }
        self.fast_prefill = enabled;
        Ok(())
    }
    /// Planned FP32 KV payload at the admitted token cap; cache_bytes reports
    /// actual retained storage. Neither includes allocator metadata,
    /// weights, temporary activations or other sessions.
    pub fn max_cache_bytes(&self) -> usize {
        self.model.max_cache_bytes()
    }
    pub fn kv_budget_bytes(&self) -> usize { self.model.kv_budget_bytes() }
    /// Snapshot immutable KV; the next forward copies its committed prefix before writing.
    pub fn fork(&self) -> Result<Self> {
        self.check_sequence_idle()?;
        self.require_idle_prefill()?;
        let snapshot = self.device.as_ref().map(|state| state.snapshot()).or_else(|| self.snapshot.clone());
        Ok(Self { model: self.model, len: self.len, device: None, snapshot, kv_format: self.kv_format, sequence: None, fast_prefill: self.fast_prefill,
            draft: self.draft.as_ref().map(|s| s.fork().map(Box::new)).transpose()?, draft_k: self.draft_k, draft_limit: self.draft_limit, draft_kv_peak: 0 })
    }
    fn check_sequence_idle(&self) -> Result<()> {
        if self.sequence.is_some() { return Err("sequence prefill active; finish or reset it first".into()); }
        Ok(())
    }
    fn check_sequence_stage(&self, layer: usize, projected: bool) -> Result<()> {
        let seq = self.sequence.as_ref().ok_or("sequence prefill is not active")?;
        if seq.invalid { return Err("sequence prefill failed; reset required".into()); }
        if seq.beginning || layer != seq.layer || layer >= self.model.layers.len() || seq.projected != projected {
            return Err("sequence layer operation out of order".into());
        }
        Ok(())
    }
    fn sequence_result<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err()
            && let Some(seq) = &mut self.sequence { seq.invalid = true; }
        result
    }
    /// Begin FP32 sequence-parallel prefill on an empty full-model session.
    /// `position` is this rank's first global position; `total_tokens` is the
    /// retained prefix length (the whole prompt on the decode rank).
    /// `prompt_end` is the full prompt length used to select attention arithmetic.
    /// Only hidden rows span the local slice; Q and shared projection scratch
    /// are chunked. Logical length stays zero until finish.
    pub fn sequence_begin(&mut self, local_tokens: &[u32], position: usize, total_tokens: usize, prompt_end: usize) -> Result<()> {
        self.check_sequence_idle()?;
        self.sequence_begin_slice(local_tokens, position, total_tokens, prompt_end, std::time::Duration::MAX).map(|_| ())
    }
    /// Resume setup with identical arguments, yielding between KV allocations
    /// and memory-bounded embedding chunks. True permits the first projection.
    pub fn sequence_begin_slice(&mut self, local_tokens: &[u32], position: usize, total_tokens: usize, prompt_end: usize,
        budget: std::time::Duration) -> Result<bool> {
        if self.model.config.num_experts != 0 { return Err("sequence-parallel expert prefill is not implemented".into()); }
        let started = std::time::Instant::now();
        if let Some(seq) = &self.sequence {
            if seq.invalid || !seq.beginning || seq.position != position || seq.tokens != local_tokens.len()
                || seq.total != total_tokens || seq.prompt_end != prompt_end {
                return Err("sequence setup resume mismatch or failed state".into());
            }
        } else {
            self.require_idle_prefill()?;
            if self.len != 0 || self.kv_format() != KvFormat::Fp32 { return Err("sequence prefill requires an empty FP32 session".into()); }
            if self.model.layer_start != 0 || self.model.layers.len() != self.model.config.layers as usize {
                return Err("sequence prefill requires the full model".into());
            }
            if local_tokens.is_empty() || total_tokens > self.model.max_tokens || total_tokens > prompt_end
                || prompt_end as u64 > self.model.config.max_positions
                || position.checked_add(local_tokens.len()).is_none_or(|end| end > total_tokens) {
                return Err("sequence slice outside prompt capacity".into());
            }
            if local_tokens.iter().any(|&token| token as u64 >= self.model.config.vocab_size) {
                return Err("sequence token outside vocabulary".into());
            }
            self.prepare_device()?;
            self.sequence = Some(SequenceProgress { position, tokens: local_tokens.len(), total: total_tokens,
                prompt_end, layer: 0, beginning: true, projected: false, invalid: false,
                covered: vec![vec![false; total_tokens]; self.model.layers.len()] });
        }
        let remaining = if budget == std::time::Duration::MAX { budget } else { budget.saturating_sub(started.elapsed()) };
        let result = self.device.as_mut().unwrap().sequence_begin_slice(self.model, local_tokens,
            position, total_tokens, prompt_end, remaining);
        if !self.sequence_result(result)? { return Ok(false); }
        self.sequence.as_mut().unwrap().beginning = false;
        Ok(true)
    }
    /// Compute this layer's local Q/K/V and RoPE at global cache positions.
    /// Synchronizes the compute stream before returning for phase measurement.
    pub fn sequence_project(&mut self, layer: usize) -> Result<()> {
        self.sequence_project_slice(layer, std::time::Duration::MAX).map(|_| ())
    }
    /// Resume projection between fixed prefill chunks. False retains unfinished
    /// work; true makes this layer's complete local K/V available for exchange.
    pub fn sequence_project_slice(&mut self, layer: usize, budget: std::time::Duration) -> Result<bool> {
        self.check_sequence_stage(layer, false)?;
        let result = self.device.as_mut().unwrap().sequence_project_slice(self.model, layer, budget);
        if !self.sequence_result(result)? { return Ok(false); }
        let seq = self.sequence.as_mut().unwrap();
        seq.projected = true;
        seq.covered[layer][seq.position..seq.position + seq.tokens].fill(true);
        Ok(true)
    }
    fn check_sequence_export(&self, layer: usize, start: usize, count: usize) -> Result<()> {
        if layer >= self.model.layers.len() { return Err("sequence export layer outside model".into()); }
        let limit = self.sequence.as_ref().map_or(self.len, |seq| seq.total);
        let end = start.checked_add(count).filter(|&end| end <= limit).ok_or("sequence export outside prompt")?;
        if let Some(seq) = &self.sequence
            && (seq.invalid || layer > seq.layer || (layer == seq.layer && !seq.projected)
                || !seq.covered[layer][start..end].iter().all(|&ready| ready)) {
                return Err("sequence export rows are not ready".into());
            }
        Ok(())
    }
    /// Export token-major FP32 cache rows. Also available after ordinary prefill
    /// and after sequence_finish, so callers can compare complete KV caches.
    pub fn sequence_export(&self, layer: usize, start: usize, count: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.check_sequence_export(layer, start, count)?;
        self.device.as_ref().ok_or("missing device cache")?.sequence_export(self.model, layer, start, count)
    }
    /// Export directly into reusable K/V exchange buffers, preserving FP32 bits.
    #[allow(clippy::manual_is_multiple_of)] // a zero divisor must still panic in %.
    pub fn sequence_export_into(&self, layer: usize, start: usize, keys: &mut [f32], values: &mut [f32]) -> Result<()> {
        let width = self.model.kv_capacity.stride;
        if keys.len() != values.len() || keys.len() % width != 0 { return Err("sequence export K/V shape mismatch".into()); }
        self.check_sequence_export(layer, start, keys.len() / width)?;
        self.device.as_ref().ok_or("missing device cache")?.sequence_export_into(self.model, layer, start, keys, values)
    }
    /// Read-only FP32 device cache views for a ready token range. Consumers can
    /// copy head shards on the same backend without a host round trip.
    pub fn sequence_device_kv(&self, layer: usize, start: usize, count: usize)
        -> Result<(&DeviceVector<'_>, &DeviceVector<'_>)> {
        self.check_sequence_export(layer, start, count)?;
        self.device.as_ref().ok_or("missing device cache")?.sequence_device_kv(layer)
    }
    /// Move completed FP32 cache storage to a decode owner without allocating a
    /// second cache. The source becomes empty and releases its compute scratch.
    pub fn take_sequence_device_kv(&mut self) -> Result<Vec<(DeviceVector<'a>, DeviceVector<'a>)>> {
        self.check_sequence_idle()?;
        self.require_idle_prefill()?;
        if self.len == 0 { return Err("sequence cache is empty".into()); }
        let cache = self.device.as_mut().ok_or("missing device cache")?.take_sequence_device_kv()?;
        self.device = None;
        self.snapshot = None;
        self.len = 0;
        Ok(cache)
    }
    /// Import raw token-major FP32 K/V without changing their bits. Completed
    /// layers also accept future rows, permitting causal compute/exchange overlap.
    #[allow(clippy::manual_is_multiple_of)] // a zero divisor must still panic in %.
    pub fn sequence_import(&mut self, layer: usize, start: usize, keys: &[f32], values: &[f32]) -> Result<()> {
        let seq = self.sequence.as_ref().ok_or("sequence prefill is not active")?;
        if seq.invalid || layer >= self.model.layers.len() || layer > seq.layer || (layer == seq.layer && !seq.projected) {
            return Err("sequence import layer is not ready".into());
        }
        let width = self.model.kv_capacity.stride;
        if keys.len() != values.len() || keys.is_empty() || keys.len() % width != 0 {
            return Err("sequence import K/V shape mismatch".into());
        }
        let seq = self.sequence.as_ref().unwrap();
        let end = start.checked_add(keys.len() / width).filter(|&end| end <= seq.total).ok_or("sequence import outside prompt")?;
        if layer < seq.layer && start < seq.position + seq.tokens {
            return Err("cannot change a completed layer's causal prefix".into());
        }
        let result = self.device.as_mut().unwrap().sequence_import(self.model, layer, start, keys, values);
        self.sequence_result(result)?;
        self.sequence.as_mut().unwrap().covered[layer][start..end].fill(true);
        Ok(())
    }
    /// Run causal attention and O/FFN once the local slice's causal prefix is ready.
    /// Synchronizes before returning; each layer must finish before the next.
    pub fn sequence_finish_layer(&mut self, layer: usize) -> Result<()> {
        self.sequence_finish_layer_slice(layer, std::time::Duration::MAX).map(|_| ())
    }
    /// Resume attention/O/FFN between fixed prefill chunks. Each call completes
    /// at least one chunk, keeping its GPU work on the session's owning thread.
    pub fn sequence_finish_layer_slice(&mut self, layer: usize, budget: std::time::Duration) -> Result<bool> {
        self.check_sequence_stage(layer, true)?;
        let seq = self.sequence.as_ref().unwrap();
        if !seq.covered[layer][..seq.position + seq.tokens].iter().all(|&ready| ready) {
            return Err("sequence layer is missing K/V rows".into());
        }
        let result = self.device.as_mut().unwrap().sequence_finish_layer_slice(self.model, layer, budget);
        if !self.sequence_result(result)? { return Ok(false); }
        let seq = self.sequence.as_mut().unwrap();
        seq.layer += 1;
        seq.projected = false;
        Ok(true)
    }
    /// Commit the complete gathered cache and return last-local-token logits.
    /// The rank owning the prompt's final token can continue ordinary decoding.
    pub fn sequence_finish(&mut self) -> Result<Vec<f32>> {
        self.require_finished_sequence()?;
        let result = self.device.as_mut().unwrap().sequence_finish(self.model);
        let logits = self.sequence_result(result)?;
        self.len = self.sequence.take().unwrap().total;
        Ok(logits)
    }
    /// TP callers exchange the final row before agreeing on its validity.
    pub fn sequence_finish_values(&mut self) -> Result<Vec<f32>> {
        self.require_finished_sequence()?;
        let result = self.device.as_mut().unwrap().sequence_finish_values(self.model);
        let logits = self.sequence_result(result)?;
        self.len = self.sequence.take().unwrap().total;
        Ok(logits)
    }
    fn require_finished_sequence(&self) -> Result<()> {
        let seq = self.sequence.as_ref().ok_or("sequence prefill is not active")?;
        if seq.invalid || seq.layer != self.model.layers.len() || seq.projected
            || !seq.covered.iter().all(|rows| rows.iter().all(|&ready| ready)) {
            return Err("sequence prefill has unfinished or failed layers".into());
        }
        Ok(())
    }
    /// Appends one token only after all layers and logits succeed. Failed steps
    /// leave every cache and logical position unchanged; no partial request state.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        self.device_token(token, true, None)
    }
    pub fn step_top_k(&mut self, token: u32, k: usize) -> Result<Vec<(u32, f32)>> {
        let (m, position) = (self.model, self.len);
        m.backend.generic_profile(false, || (position, 1, engine_model::plan::profile::ProfilePlan::qwen(&m.config).classes(), 1), || self.step_top_k_profiled(token, k))
    }
    fn step_top_k_profiled(&mut self, token: u32, k: usize) -> Result<Vec<(u32, f32)>> {
        self.device_run(token, true, &mut None)?;
        let state = self.device.as_mut().ok_or("device state missing")?;
        let values = self.model.backend.top_k_device(&state.logits, &mut state.top_k, 1, k)?
            .pop().ok_or("missing top-k row")??;
        state.routing(self.model.backend)?;
        state.peak_bytes = state.peak_bytes.max(self.model.backend.allocated_bytes());
        self.len += 1;
        Ok(values)
    }
    /// Greedy decode returns only the selected ID. Device mode keeps logits on
    /// the GPU and reads three unified-memory reduction values after one sync.
    pub fn step_greedy(&mut self, token: u32) -> Result<u32> {
        self.step_greedy_with_score(token).map(|(token, _)| token)
    }
    pub fn step_greedy_with_score(&mut self, token: u32) -> Result<(u32, f32)> {
        let b = self.model.backend;
        let profile = self.len == 28673 && std::env::var("SPARK_PROFILE").as_deref() == Ok("1") && !b.is_profiling();
        if !profile { return self.greedy_step(token); }
        b.synchronize()?;
        let before = b.stats();
        let start = Instant::now();
        b.set_profiling(true);
        let result = self.greedy_step(token);
        b.set_profiling(false);
        eprintln!("long_decode_profile position=28673 wall_ms={:.3} kv_bytes={} device_bytes={} peak_device_bytes={} stats={:?}",
            start.elapsed().as_secs_f64()*1000.0, self.cache_bytes(), b.allocated_bytes(), self.device_peak_bytes(), b.stats().since(before));
        result
    }
    fn greedy_step(&mut self, token: u32) -> Result<(u32, f32)> {
        let (m, position) = (self.model, self.len);
        m.backend.generic_profile(false, || (position, 1, engine_model::plan::profile::ProfilePlan::qwen(&m.config).classes(), 1), || self.greedy_step_profiled(token))
    }
    fn greedy_step_profiled(&mut self, token: u32) -> Result<(u32, f32)> {
        self.require_idle_prefill()?;
        self.check_sequence_idle()?;
        if token as u64 >= self.model.config.vocab_size || self.len >= self.model.max_tokens {
            return Err("greedy step exceeds token capacity or vocabulary".into());
        }
        self.prepare_device()?;
        let state = self.device.as_mut().unwrap();
        if state.greedy_graph(self.model, self.len, token, 1)? {
            let result = state.greedy_results(self.model.backend, 1)?[0];
            self.len += 1;
            return Ok(result);
        }
        self.device_run(token, true, &mut None)?;
        let result = self.device.as_mut().ok_or("missing device session")?.greedy(self.model.backend)?;
        self.len += 1;
        Ok(result)
    }
    fn prepare_device(&mut self) -> Result<()> {
        if self.device.is_none() {
            self.device = Some(if self.snapshot.is_some() {
                DeviceState::with_cache(self.model, self.kv_format, self.snapshot.clone())?
            } else { DeviceState::new(self.model, self.kv_format)? });
            self.snapshot = None;
        }
        Ok(())
    }
    fn device_run(&mut self, token: u32, include_logits: bool, observer: &mut Observer<'_>) -> Result<()> {
        self.require_idle_prefill()?;
        self.check_sequence_idle()?;
        let m = self.model;
        if token as u64 >= m.config.vocab_size { return Err("token outside vocabulary".into()); }
        if self.len >= m.max_tokens { return Err("request token capacity reached".into()); }
        self.prepare_device()?;
        let device = &m.device;
        let state = self.device.as_mut().ok_or("device state missing")?;
        if include_logits && observer.is_none() && state.greedy_graph(m, self.len, token, 1)? { return Ok(()); }
        m.backend.embedding_device(device.embedding.as_ref().ok_or("partition does not own embedding")?, token, &mut state.hidden)?;
        DeviceState::observe(m.backend, "embedding", &state.hidden, observer)?;
        state.layers(m, device, self.len, observer)?;
        if include_logits { state.project(m, device, observer)?; }
        Ok(())
    }
    fn device_token(&mut self, token: u32, include_logits: bool, mut observer: Observer<'_>) -> Result<Vec<f32>> {
        let (m, position) = (self.model, self.len);
        m.backend.generic_profile(false, || (position, 1, engine_model::plan::profile::ProfilePlan::qwen(&m.config).classes(), 1), || self.device_token_profiled(token, include_logits, &mut observer))
    }
    fn device_token_profiled(&mut self, token: u32, include_logits: bool, observer: &mut Observer<'_>) -> Result<Vec<f32>> {
        self.device_run(token, include_logits, observer)?;
        let state = self.device.as_ref().ok_or("device state missing")?;
        let output = self.model.backend.read_vector(if include_logits { &state.logits } else { &state.hidden })?;
        state.routing(self.model.backend)?;
        if include_logits
            && let Some(observer) = observer { observer("logits", &output)?; }
        self.len += 1;
        Ok(if include_logits { output } else { Vec::new() })
    }
    fn require_idle_prefill(&self) -> Result<()> {
        if self.device.as_ref().is_some_and(|state| state.prefill_pending()) {
            return Err("prefill slice is still pending".into());
        }
        Ok(())
    }
    fn check_batch(&self, tokens: &[u32]) -> Result<()> {
        self.require_idle_prefill()?;
        self.check_batch_tokens(tokens)
    }
    fn check_batch_tokens(&self, tokens: &[u32]) -> Result<()> {
        self.check_sequence_idle()?;
        if tokens.is_empty() { return Err("empty token batch".into()); }
        if tokens.len() > self.model.max_tokens - self.len { return Err("batch exceeds request capacity".into()); }
        if tokens.iter().any(|&token| token as u64 >= self.model.config.vocab_size) {
            return Err("batch token outside vocabulary".into());
        }
        Ok(())
    }
    /// Consecutive positions with one next-token logit row per input token.
    /// Device batches commit together; truncate rolls back a suffix.
    pub fn forward_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.check_batch(tokens)?;
        if tokens.len() > 512 { return Err("forward_batch accepts at most 512 rows".into()); }
        if tokens.len() > 1 { return self.device_batch(tokens, Logits::All, false, self.len + tokens.len(), false); }
        tokens.iter().map(|&token| self.step(token)).collect()
    }
    /// Canonical speculative verification: each logit bit matches sequential
    /// native MMA decode from the same prefix. Prepare packed weights for batching.
    pub fn verify_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.check_batch(tokens)?;
        if tokens.len() > 512 { return Err("verify_batch accepts at most 512 rows".into()); }
        if self.model.config.num_experts != 0 && tokens.len() > 1 {
            // Expert chains use the same canonical rows as shared serving.
            // Route once per layer so adjacent positions reuse expert weights.
            let mut multi = multi::Multi::new(self.model, tokens.len())?;
            return multi.step(&mut [multi::Rows { session: self, tokens, output: multi::Output::All }])
                .map(|mut rows| rows.remove(0)).map_err(String::from);
        }
        // A single chain reuses session scratch; exact attention has no prompt-end dispatch.
        if tokens.len() > 1 { return self.device_batch(tokens, Logits::All, true, 0, false); }
        tokens.iter().map(|&token| self.step(token)).collect()
    }
    /// Preferred prompt rows when no other request is runnable.
    pub fn prefill_chunk_size(&self, prompt_end: usize) -> usize {
        if self.model.config.num_experts != 0 {
            if prompt_end < 6144 { return 1024; }
            // Equal full chunks keep the large expert arena alive through the
            // final chunk, instead of freeing and reallocating it for a tail.
            return prompt_end.div_ceil(prompt_end.div_ceil(16384)).div_ceil(128) * 128;
        }
        if prompt_end >= 6144 { 4096 } else { PREFILL_CHUNK }
    }
    /// Consecutive prompt positions, projecting only the final vocabulary row.
    /// Successful chunks commit independently; scratch follows the selected row cap.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.prefill_observed(tokens, &mut None)
    }
    /// Observe each expert layer/position on the ordinary batched prefill path.
    /// Download routing metadata only when explicitly requested.
    pub fn prefill_with_routing_observer(&mut self, tokens: &[u32],
        mut observer: impl FnMut(usize, usize, &[(u32, f32)]) -> Result<()>) -> Result<Vec<f32>> {
        self.prefill_observed(tokens, &mut Some(&mut observer))
    }
    fn prefill_observed(&mut self, tokens: &[u32], observer: &mut device::RoutingObserver<'_>) -> Result<Vec<f32>> {
        self.check_batch(tokens)?;
        let prompt_end = self.len + tokens.len();
        self.prepare_device()?;
        self.device.as_mut().unwrap().reserve_cache(self.model, self.len, self.len+tokens.len())?;
        let mut logits = Vec::new();
        let chunk_size = self.prefill_chunk_size(prompt_end);
        for (i, chunk) in tokens.chunks(chunk_size).enumerate() {
            let output = if (i + 1) * chunk_size >= tokens.len() { Logits::Last } else { Logits::None };
            if let Some(row) = self.device_batch_observed(chunk, output, false, prompt_end, self.fast_prefill, observer)?.pop() { logits = row; }
        }
        Ok(logits)
    }
    /// Batched prompt KV only; intermediate chunks retain their scratch.
    pub fn prefill_advance(&mut self, tokens: &[u32]) -> Result<()> {
        self.check_batch(tokens)?;
        let prompt_end = self.len + tokens.len();
        for chunk in tokens.chunks(self.prefill_chunk_size(prompt_end)) { self.device_batch(chunk, Logits::None, false, prompt_end, self.fast_prefill)?; }
        Ok(())
    }
    /// Resume one fixed prompt chunk at layer boundaries. None retains its
    /// activations and uncommitted KV; repeat with the same tokens/output mode.
    /// Some commits the chunk and returns final logits only when requested.
    /// One indivisible layer or output projection always makes progress.
    /// No budget completes the chunk without per-layer synchronization.
    pub fn prefill_slice(&mut self, tokens: &[u32], final_chunk: bool,
        budget: Option<std::time::Duration>, prompt_end: usize) -> Result<Option<Vec<f32>>> {
        let started = std::time::Instant::now();
        self.check_batch_tokens(tokens)?;
        if tokens.len() > self.prefill_chunk_size(prompt_end) { return Err("prefill slice exceeds the model row cap".into()); }
        if prompt_end < self.len + tokens.len() || prompt_end > self.model.max_tokens {
            return Err("prefill prompt end outside context".into());
        }
        self.prepare_device()?;
        if !self.device.as_ref().unwrap().prefill_pending() {
            self.device.as_mut().unwrap().reserve_prompt_cache(self.model, self.len, prompt_end)?;
        }
        let budget = budget.map(|limit| limit.saturating_sub(started.elapsed()));
        let output = if final_chunk { Logits::Last } else { Logits::None };
        let rows = self.device.as_mut().ok_or("missing device state")?.prefill_slice(
            self.model, &self.model.device, self.len, tokens, output, budget, prompt_end, self.fast_prefill)?;
        if let Some(mut rows) = rows {
            self.len += tokens.len();
            Ok(Some(rows.pop().unwrap_or_default()))
        } else { Ok(None) }
    }
    fn device_batch(&mut self, tokens: &[u32], output: Logits, exact: bool, prompt_end: usize, fast_prefill: bool) -> Result<Vec<Vec<f32>>> {
        self.device_batch_observed(tokens, output, exact, prompt_end, fast_prefill, &mut None)
    }
    fn device_batch_observed(&mut self, tokens: &[u32], output: Logits, exact: bool, prompt_end: usize, fast_prefill: bool,
        observer: &mut device::RoutingObserver<'_>) -> Result<Vec<Vec<f32>>> {
        self.prepare_device()?;
        let rows = self.device.as_mut().ok_or("missing device state")?.batch(self.model,
            &self.model.device, self.len, tokens, output, exact, prompt_end, fast_prefill, observer)?;
        self.len += tokens.len();
        Ok(rows)
    }
    /// Appends one prompt token's KV without computing unused next-token logits.
    /// Use advance for nonfinal prompt tokens, then step for the final token.
    /// Each call is a separate transaction; earlier successful calls remain
    /// committed if a later token fails. The skipped output head is not validated.
    pub fn advance(&mut self, token: u32) -> Result<()> {
        self.device_token(token, false, None).map(|_| ())
    }
    /// Validation observer sees embedding, each completed layer, final norm, and
    /// logits. Observer failures abort without appending any token KV.
    pub fn step_with_observer(&mut self, token: u32,
        mut observer: impl FnMut(&str, &[f32]) -> Result<()>) -> Result<Vec<f32>> {
        self.device_token(token, true, Some(&mut observer))
    }
}

/// Short prompt row cap; prefill arithmetic is identical for every tail size.
pub const PREFILL_CHUNK: usize = 512;

/// A contiguous pipeline stage. Only its layer matrices and owned boundary
/// embedding/output are allocated; all shard headers remain validated.
pub struct PartitionModel<'a, 'b> { inner: Model<'a, 'b>, start: usize, end: usize }
impl<'a, 'b> PartitionModel<'a, 'b> {
    pub fn load(backend: &'b Backend<'a>, root: &Path, max_tokens: usize,
        start_layer: usize, end_layer: usize, owns_embedding: bool, owns_output: bool) -> Result<Self> {
        Self::load_with_kv_budget(backend, root, max_tokens, start_layer, end_layer,
            owns_embedding, owns_output, DEFAULT_KV_BUDGET_BYTES)
    }
    /// Charge only this stage's actual layer range; other stages and sessions
    /// remain the caller's aggregate admission responsibility.
    pub fn load_with_kv_budget(backend: &'b Backend<'a>, root: &Path, max_tokens: usize,
        start_layer: usize, end_layer: usize, owns_embedding: bool, owns_output: bool,
        kv_budget_bytes: usize) -> Result<Self> {
        Ok(Self { inner: Model::load_range(backend, root, max_tokens, kv_budget_bytes,
            Some((start_layer, end_layer, owns_embedding, owns_output)), false)?, start: start_layer, end: end_layer })
    }
    pub fn config(&self) -> &Config { self.inner.config() }
    pub fn max_tokens(&self) -> usize { self.inner.max_tokens() }
    pub fn kv_budget_bytes(&self) -> usize { self.inner.kv_budget_bytes() }
    pub fn max_cache_bytes(&self) -> usize { self.inner.max_cache_bytes() }
    pub fn layer_range(&self) -> std::ops::Range<usize> { self.start..self.end }
    pub fn prepare_batch_weights(&mut self) -> Result<usize> { self.inner.prepare_batch_weights() }
    pub fn session(&self) -> PartitionSession<'_, 'a, 'b> {
        PartitionSession { inner: self.inner.session(), invalid: false }
    }
}
pub struct PartitionSession<'m, 'a, 'b> { inner: Session<'m, 'a, 'b>, invalid: bool }
impl PartitionSession<'_, '_, '_> {
    pub fn len(&self) -> usize { self.inner.len() }
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }
    pub fn cache_bytes(&self) -> usize { self.inner.cache_bytes() }
    pub fn max_cache_bytes(&self) -> usize { self.inner.max_cache_bytes() }
    pub fn kv_budget_bytes(&self) -> usize { self.inner.kv_budget_bytes() }
    pub fn is_invalid(&self) -> bool { self.invalid }
    pub fn reset(&mut self) { self.inner.reset(); self.invalid = false; }
    pub fn embed(&self, token: u32) -> Result<Vec<f32>> {
        if self.invalid { return Err("partition session invalid; reset required".into()); }
        self.inner.model.embed(token)
    }
    /// Any failed stage operation invalidates this request until reset. KV is
    /// committed only after the complete local layer range succeeds.
    pub fn step_hidden(&mut self, position: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        let result = (|| {
            if self.invalid { return Err("partition session invalid; reset required".into()); }
            if position != self.inner.len || position >= self.inner.model.max_tokens { return Err("partition position mismatch or capacity reached".into()); }
            self.inner.model.check_hidden(hidden)?;
            let device = &self.inner.model.device;
            {
                self.inner.prepare_device()?;
                let state = self.inner.device.as_mut().ok_or("device state missing")?;
                self.inner.model.backend.write_vector(&mut state.hidden, hidden)?;
                state.layers(self.inner.model, device, position, &mut None)?;
                let output = self.inner.model.backend.read_vector(&state.hidden)?;
                state.routing(self.inner.model.backend)?;
                self.inner.len += 1;
                Ok(output)
            }
        })();
        if result.is_err() { self.invalid = true; }
        result
    }
    pub fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        if self.invalid { return Err("partition session invalid; reset required".into()); }
        let result = (|| {
            let m = self.inner.model;
            m.check_hidden(hidden)?;
            let device = &m.device;
            {
                self.inner.prepare_device()?;
                let state = self.inner.device.as_mut().ok_or("device state missing")?;
                m.backend.write_vector(&mut state.hidden, hidden)?;
                state.project(m, device, &mut None)?;
                m.backend.read_vector(&state.logits)
            }
        })();
        if result.is_err() { self.invalid = true; }
        result
    }
}
