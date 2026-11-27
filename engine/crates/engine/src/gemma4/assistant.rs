//! Shared-KV official assistant; untrusted proposals go through draft::Speculator.
use super::*;
use spark_gpu_probe::load::{WeightRows, WeightShard};
use std::time::Instant;

pub struct Assistant<'a> {
    config: engine_model::gemma4_assistant::Config, layers: Vec<Layer<'a>>,
    pre: Matrix<'a>, post: Matrix<'a>, head: Matrix<'a>, norm: DeviceVector<'a>,
    width: usize,
}
pub(super) struct DraftState<'a> {
    pub carry: DeviceVector<'a>, pub prior: DeviceVector<'a>, pub normalized: DeviceVector<'a>, pub logits: DeviceVector<'a>,
    decoder: State<'a>, concat: DeviceVector<'a>, recurrent: DeviceVector<'a>,
    argmax: DeviceVector<'a>, result: DeviceVector<'a>,
    pub entry: usize, pub end: usize,
}
impl<'a> Assistant<'a> {
    pub fn open_config(root: &Path) -> Result<engine_model::gemma4_assistant::Config> {
        let bytes = engine_model::safetensors::read_bounded(&root.join("config.json"), 1024*1024).map_err(|e| e.to_string())?;
        engine_model::gemma4_assistant::Config::parse(&bytes).map_err(|e| e.to_string())
    }
    fn load(b: &Backend<'a>, root: &Path, target: &Config) -> Result<Self> {
        let config = Self::open_config(root)?;
        config.validate_target(target).map_err(|e| e.to_string())?;
        let file = TensorFile::open(&root.join("model.safetensors"), Limits::default()).map_err(|e| e.to_string())?;
        let expected = config.expected_shapes();
        if file.header().tensors.len() != expected.len() { return Err("assistant tensor count mismatch".into()); }
        for (name, shape) in &expected {
            let t = file.header().tensors.get(name).ok_or_else(|| format!("missing assistant tensor {name}"))?;
            if t.shape != *shape || t.dtype != engine_model::safetensors::Dtype::Bf16 {
                return Err(format!("assistant tensor binding mismatch: {name}"));
            }
        }
        let map = expected.keys().map(|n| (n.clone(), "model.safetensors".into())).collect();
        let mut files = BTreeMap::new(); files.insert("model.safetensors".into(), file);
        let mut loader = Loader { files, map, matrices: BTreeMap::new() };
        loader.load_matrices(b, 0..4, true, true, true, true, "model", None)?;
        let file = &loader.files["model.safetensors"];
        let extras = ["pre_projection.weight", "post_projection.weight"];
        let plans: Vec<_> = extras.iter().map(|n| {
            let t = &file.header().tensors[*n];
            WeightRows { file_offset: file.header().data_start+t.start, rows: 0..t.shape[0] as usize,
                columns: t.shape[1] as usize, packed: true, fp8: false }
        }).collect();
        let mut projections = b.load_bf16_shards_in_arena(&[WeightShard { file: file.file(), matrices: &plans }])?.remove(0).into_iter();
        let pre = projections.next().unwrap(); let post = projections.next().unwrap();
        let head = loader.matrix("model.embed_tokens.weight")?;
        let norm = b.upload_vector(&loader.host("model.norm.weight", 1024*1024)?)?;
        let mut layers = Vec::new();
        for i in 0..4 {
            let p = format!("model.layers.{i}");
            let mut norm = |n: &str| b.upload_vector(&loader.host(&format!("{p}.{n}.weight"), 1024*1024)?);
            let (input, attention, pre_mlp, post_mlp, q_norm) = (norm("input_layernorm")?, norm("post_attention_layernorm")?,
                norm("pre_feedforward_layernorm")?, norm("post_feedforward_layernorm")?, norm("self_attn.q_norm")?);
            let scalar = loader.host(&format!("{p}.layer_scalar"), 16)?[0];
            let mut mat = |n: &str| loader.matrix(&format!("{p}.{n}.weight"));
            layers.push(Layer { input, attention, pre_mlp, post_mlp, q_norm, k_norm: None, scalar,
                q: mat("self_attn.q_proj")?, k: None, v: None, o: mat("self_attn.o_proj")?,
                gate: mat("mlp.gate_proj")?, up: mat("mlp.up_proj")?, down: mat("mlp.down_proj")? });
        }
        let weight_bytes = config.weight_bytes() as usize;
        eprintln!("gemma_assistant_weight_bytes={weight_bytes} device_bytes={} peak_device_bytes={}", b.allocated_bytes(), b.peak_allocated_bytes());
        Ok(Self { config, layers, pre, post, head, norm, width: 4 })
    }
    fn state(&self, m: &Model<'a, '_>) -> Result<DraftState<'a>> {
        let b = m.backend; let h = m.config.hidden_size as usize;
        Ok(DraftState { carry: b.vector(h)?, prior: b.vector(h)?, normalized: b.vector(16*h)?,
            logits: b.vector(16*m.config.vocab_size as usize)?, decoder: State::scratch(b, &self.config.text, m.max_tokens, 1)?,
            concat: b.vector_arena(&[2*h])?.remove(0), recurrent: b.vector(h)?, argmax: b.vector((self.config.text.vocab_size as usize).div_ceil(1024)*3)?,
            result: b.vector(3)?, entry: 0, end: 0 })
    }
    fn step(&self, m: &Model<'a, '_>, target: &State<'a>, position: usize,
        token: u32, d: &mut DraftState<'a>) -> Result<u32> {
        let (b,c) = (m.backend, &self.config.text); let s = &mut d.decoder;
        let h = m.config.hidden_size as usize;
        b.write_vector(&mut s.ids, &[token as f32])?;
        // The reference calls the target's scaled embedding module: the
        // target BF16 sqrt(H) scale applies before concatenation.
        let mut embedding = d.concat.slice(0..h)?;
        b.batch_embedding_device(&m.embedding, &s.ids, &mut embedding)?;
        b.model_elementwise_device(&embedding, None, &embedding, h, 0, bf16((h as f32).sqrt()))?;
        b.copy_vector_range(&d.recurrent, &mut d.concat, 0, h, h)?;
        b.matmul_bf16_device(&self.pre, &d.concat, &mut s.hidden, &mut s.weight_pack, &mut s.input_pack, 1, false)?;
        for (i,l) in self.layers.iter().enumerate() {
            let kind = c.layer_types[i]; let (kv,dim) = c.geometry(kind); let (kv,dim) = (kv as usize, dim as usize);
            let source = m.config.layer_types.iter().rposition(|&k| k == kind).ok_or("missing shared target KV")?;
            let cache = &target.cache[source];
            s.q.resize_within_capacity(c.attention_heads as usize*dim)?; s.attention.resize_within_capacity(s.q.len())?;
            let rope = if kind == Attention::Sliding {&m.local_rope} else {&m.global_rope};
            layer::forward_normalized_attention(b, layer::Weights { q: Some(&l.q), k: None, v: None, o: Some(&l.o),
                gate: Some(&l.gate), up: Some(&l.up), down: Some(&l.down) },
                layer::BlockNorms { input: &l.input, attention: &l.attention, pre_mlp: &l.pre_mlp, post_mlp: &l.post_mlp,
                    q: &l.q_norm, k: &l.q_norm, scalar: l.scalar },
                layer::Buffers { hidden: &mut s.hidden, input: &mut s.input, sums: &mut s.sums, q: &mut s.q, k: &mut s.k, v: &mut s.v,
                    attention: &mut s.attention, projected: &mut s.projected, gate: &mut s.gate, up: &mut s.up, scores: &mut s.scores,
                    parts: &mut s.last, weight_pack: &mut s.weight_pack, input_pack: &mut s.input_pack,
                    packed_keys: &mut s.normalized, packed_values: &mut s.logits },
                layer::Run { position, tokens: 1, first: i == 0, prompt_end: position+1,
                    heads: c.attention_heads as usize, kv_heads: kv, head_dim: dim, epsilon: c.rms_norm_eps as f32,
                    exact: true, format: KvFormat::Bf16, decode: true, sums_ready: false },
                |q, _, _, scores, output| {
                    b.model_query_device(q, &l.q_norm, rope, c.attention_heads as usize, dim, position, c.rms_norm_eps as f32)?;
                    b.bounded_attention_device(q, &cache.keys, &cache.values, scores, output,
                        c.attention_heads as usize, kv, dim, position, 1, if cache.local {c.sliding_window as usize} else {0}, cache.capacity, 1.0)
                })?;
        }
        b.model_rms_device(&s.hidden, Some(&self.norm), &mut s.normalized, &mut s.sums, c.hidden_size as usize, 1, c.rms_norm_eps as f32)?;
        b.matmul_bf16_device(&self.post, &s.normalized, &mut d.recurrent, &mut s.weight_pack, &mut s.input_pack, 1, false)?;
        b.matmul_bf16_device(&self.head, &s.normalized, &mut s.logits, &mut s.weight_pack, &mut s.input_pack, 1, true)?;
        b.argmax_device(&s.logits, &mut d.argmax, &mut d.result).map(|r| r.0)
    }
}
impl<'a,'b> Model<'a,'b> {
    pub fn load_assistant(&mut self, root: &Path) -> Result<()> {
        if self.assistant.is_some() { return Err("assistant already loaded".into()); }
        self.assistant = Some(Assistant::load(self.backend, root, &self.config)?); Ok(())
    }
    pub fn set_assistant_width(&mut self, width: usize) -> Result<()> {
        if !(1..=8).contains(&width) { return Err("assistant draft width must be 1..8".into()); }
        self.assistant.as_mut().ok_or("assistant is not loaded")?.width = width; Ok(())
    }
    pub fn has_assistant(&self) -> bool { self.assistant.is_some() }
    pub fn assistant_reservation_bytes(&self) -> usize { if self.has_assistant() {32*1024*1024} else {0} }
}
impl<'m,'a,'b> Session<'m,'a,'b> {
    pub fn has_assistant(&self) -> bool { self.model.has_assistant() }
    pub(super) fn prepare_draft(&mut self) -> Result<()> {
        if self.draft.is_none()
            && let Some(a) = &self.model.assistant { self.draft = Some(a.state(self.model)?); }
        Ok(())
    }
    pub fn propose_model(&mut self, history: &[u32], next: u32, limit: usize) -> Result<Option<crate::draft::ModelProposal>> {
        let Some(a) = &self.model.assistant else { return Ok(None); };
        if history.len() != self.len || self.len == 0 || limit > 15 || next as u64 >= self.model.config.vocab_size {
            return Err("invalid assistant proposal prefix or width".into());
        }
        let d = self.draft.as_mut().ok_or("assistant hidden state unavailable")?;
        let s = self.state.as_ref().ok_or("assistant target KV unavailable")?;
        let start = Instant::now();
        self.model.backend.copy_vector_range(&d.carry, &mut d.recurrent, 0, 0, d.carry.len())?;
        let mut tokens = Vec::with_capacity(limit);
        if limit > 0 {
            // Warm the recurrent projection with the last committed token, then
            // force the target's already known next token instead of its draft.
            a.step(self.model, s, self.len-1, *history.last().unwrap(), d)?;
            let mut token = next;
            for _ in 0..limit {
                token = a.step(self.model, s, self.len-1, token, d)?;
                tokens.push(token);
                if self.model.eos.contains(&token) { break; }
            }
        }
        Ok(Some(crate::draft::ModelProposal { tokens, proposal_ms: start.elapsed().as_secs_f64()*1000.0, catchup_ms: 0.0 }))
    }
}
impl crate::draft::Target for Session<'_, '_, '_> {
    fn checkpoint(&self) -> usize { self.len }
    fn truncate(&mut self, p: usize) -> Result<()> { Session::truncate(self,p) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>> { self.verify_batch(&[token]).map(|mut r| r.remove(0)) }
    fn verify_batch(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> { Session::verify_batch(self,tokens) }
    fn kv_bf16(&self) -> bool { true }
    fn model_draft_width(&self) -> usize { self.model.assistant.as_ref().map_or(0, |a| a.width) }
    fn propose_model(&mut self, h: &[u32], n: u32, l: usize) -> Result<Option<crate::draft::ModelProposal>> { Session::propose_model(self,h,n,l) }
}

impl DraftState<'_> {
    pub fn allocations(&self, out: &mut Vec<(usize, usize)>) {
        out.extend(self.decoder.allocations());
        for v in [&self.carry, &self.prior, &self.normalized, &self.logits, &self.concat, &self.recurrent, &self.argmax, &self.result] {
            v.allocations(out);
        }
    }
}
