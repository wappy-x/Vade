//! Row-sharded Qwen using the single-Spark backend and arithmetic kernels.
use crate::{Result, rows, transport::Ring};
use engine_model::{qwen3::{Checkpoint, Config}, safetensors::{Limits, TensorFile}};
use spark_gpu_probe::driver::{Backend, DeviceVector, Fence, Matrix};
use spark_engine::layer;
use crate::service_commands::{candidate,greedy as gathered_greedy,finite,finite_bytes,OUTPUT_ERROR};
use std::{cell::RefCell, collections::BTreeMap, path::Path, time::{Duration, Instant}};

mod multi;
use multi::{MultiAttention, MultiState};

struct Shard<'a> { matrix: Option<Matrix<'a>>, first: usize }
impl Shard<'_> {
    fn count(&self) -> usize { self.matrix.as_ref().map_or(0, Matrix::rows) }
}

struct Weights<'a, 'b> {
    config: Config, backend: &'b Backend<'a>,
    files: BTreeMap<String, TensorFile>, map: BTreeMap<String, String>, peak_bytes: usize,
}
impl<'a, 'b> Weights<'a, 'b> {
    fn open(root: &Path, backend: &'b Backend<'a>) -> Result<Self> {
        let checkpoint = Checkpoint::inspect(root).map_err(|e| e.to_string())?;
        let mut files = BTreeMap::new();
        let expected = checkpoint.config.expected_shapes();
        let mut count = 0;
        for shard in checkpoint.shards.keys() {
            let file = TensorFile::open(&root.join(shard), Limits::default()).map_err(|e| e.to_string())?;
            for (name, tensor) in &file.header().tensors {
                if expected.get(name) != Some(&tensor.shape) || tensor.dtype != engine_model::safetensors::Dtype::Bf16
                    || checkpoint.index.weight_map.get(name) != Some(shard) { return Err("checkpoint tensor binding mismatch".into()); }
                count += 1;
            }
            files.insert(shard.clone(), file);
        }
        if count != expected.len() { return Err("checkpoint tensor count mismatch".into()); }
        Ok(Self { config: checkpoint.config, backend, files, map: checkpoint.index.weight_map, peak_bytes: 0 })
    }
    fn file(&mut self, name: &str) -> Result<&mut TensorFile> {
        self.files.get_mut(self.map.get(name).ok_or("tensor missing from index")?).ok_or_else(|| "missing shard".into())
    }
    fn matrix_units(&mut self, name: &str, rank: usize, world: usize, unit: usize) -> Result<Shard<'a>> {
        if rank >= world { return Err("invalid rank/world".into()); }
        let backend = self.backend;
        let file = self.file(name)?;
        let shape = &file.header().tensors.get(name).ok_or("missing matrix")?.shape;
        if shape.len() != 2 { return Err("matrix must have two dimensions".into()); }
        let (total, cols) = (shape[0] as usize, shape[1] as usize);
        if unit == 0 || total % unit != 0 { return Err("invalid head shard".into()); }
        let units = rows(total / unit, rank, world);
        let range = units.start * unit..units.end * unit;
        let offset = range.start * cols * 2;
        let matrix = if range.is_empty() { None } else {
            let matrix = backend.upload_packed_bf16(range.len(), cols, 4 * 1024 * 1024, |at, out|
                file.read_range(name, (offset + at) as u64, out).map_err(|e| e.to_string()))?;
            self.peak_bytes = self.peak_bytes.max(backend.allocated_bytes());
            Some(matrix)
        };
        Ok(Shard { matrix, first: range.start })
    }
    fn norm(&mut self, name: &str) -> Result<DeviceVector<'a>> {
        let file = self.file(name)?;
        let bytes = file.header().tensors.get(name).ok_or("missing norm")?.byte_len() as usize;
        if bytes > 1024 * 1024 { return Err("norm tensor too large".into()); }
        let mut raw = vec![0u8; bytes];
        file.read_range(name, 0, &mut raw).map_err(|e| e.to_string())?;
        self.backend.upload_vector(&raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect::<Vec<_>>())
    }
}
struct Norms<'a> {
    input_norm: DeviceVector<'a>, post_norm: DeviceVector<'a>, q_norm: DeviceVector<'a>, k_norm: DeviceVector<'a>,
}
struct Layer<'a> {
    norms: Option<Norms<'a>>,
    q: Shard<'a>, k: Shard<'a>, v: Shard<'a>, o: Shard<'a>,
    gate: Shard<'a>, up: Shard<'a>, down: Shard<'a>,
}
pub struct Model<'a, 'b> {
    backend: &'b Backend<'a>, config: Config, layers: Vec<Layer<'a>>,
    embedding: Option<RefCell<TensorFile>>, norm: Option<DeviceVector<'a>>, head: Shard<'a>, rope: Option<DeviceVector<'a>>,
    full: Option<&'b spark_engine::qwen::Model<'a, 'b>>,
    /// Hybrid sessions run the opt-in FP32-KV sequence prefill.
    sequence: bool,
    /// SPARK_PROFILE=1, read once: per-layer profiling, which runs prompt chunks serially
    /// instead of as halves. Every rank must agree (tp-serve checks it at startup).
    profile: bool,
    max_tokens: usize, rank: usize, world: usize, local_kv_heads: usize,
    pub resident_bytes: usize, pub load_peak_bytes: usize,
}
impl<'a, 'b> Model<'a, 'b> {
    /// Device admission for shard-only weights and one live FP32 session.
    /// Packed row tails are included; no full checkpoint staging is reserved.
    pub fn device_budget(config: &Config, rank: usize, world: usize, max_tokens: usize) -> Result<usize> {
        if rank >= world || max_tokens == 0 || max_tokens as u64 > config.max_positions || max_tokens > u32::MAX as usize {
            return Err("invalid rank/world/capacity".into());
        }
        let (h,d)=(config.hidden_size as usize,config.head_dim as usize);
        if config.kv_heads == 0 || config.attention_heads % config.kv_heads != 0
            || h % 128 != 0 || config.intermediate_size % 64 != 0 {
            return Err("unsupported TP packed projection dimensions".into());
        }
        let k=rows(config.kv_heads as usize,rank,world).len()*d;
        let q=k*config.attention_heads as usize/config.kv_heads as usize;
        let mid=rows(config.intermediate_size as usize/64,rank,world).len()*64;
        let projection=rows(h/128,rank,world).len()*128;
        let full_q=(config.attention_heads*config.head_dim) as usize;
        let vocab=rows(config.vocab_size as usize,rank,world).len().div_ceil(64)*64;
        let layer=2*h as u128*(q as u128+2*(k.div_ceil(64)*64) as u128+2*mid as u128)
            +2*projection as u128*(full_q as u128+config.intermediate_size as u128)+8*(h+d) as u128;
        let resident=layer*config.layers as u128+2*(vocab as u128)*(h as u128)
            +4*h as u128+4*max_tokens as u128*d as u128;
        let resident=usize::try_from(resident).map_err(|_|"TP model byte count overflow")?;
        resident.checked_add(Self::shard_session_bytes(config,rank,world,max_tokens)).ok_or_else(||"TP device budget overflow".into())
    }
    pub fn shard_session_bytes(c:&Config,rank:usize,world:usize,max_tokens:usize)->usize {
        let cap=max_tokens.div_ceil(16) as u128*16;let t=max_tokens.min(4096) as u128;
        let h=c.hidden_size as u128;let d=c.head_dim as u128;
        let heads=rows(c.kv_heads as usize,rank,world).len() as u128*c.attention_heads as u128/c.kv_heads as u128;
        let k=rows(c.kv_heads as usize,rank,world).len() as u128*d;let q=heads*d;
        let mid=rows(c.intermediate_size as usize/64,rank,world).len() as u128*64;
        let vocab=rows(c.vocab_size as usize,rank,world).len() as u128;
        let kv=2*c.layers as u128*cap*k*4;
        // Local projections, full operand gathering/packing and residuals.
        let projections=4*t*(8*h+4*q+4*k+6*mid+2*(c.intermediate_size as u128).max(c.attention_heads as u128*d))
            +16*max_tokens.min(512) as u128*vocab;
        let attention=4*(64*heads*cap+64*q*cap.div_ceil(64)+2*cap*k);
        usize::try_from(2*kv+projections+attention+128*1024*1024).unwrap_or(usize::MAX)
    }
    /// HTTP uses native BF16 KV. Charge only this rank's heads and the
    /// workspaces actually used by streaming attention, including arena growth.
    pub fn service_session_bytes(c:&Config,rank:usize,world:usize,max_tokens:usize)->usize {
        let cap=max_tokens as u128;
        let h=c.hidden_size as u128;let d=c.head_dim as u128;
        let k=rows(c.kv_heads as usize,rank,world).len() as u128*d;
        let q=k*c.attention_heads as u128/c.kv_heads as u128;
        let mid=rows(c.intermediate_size as usize/64,rank,world).len() as u128*64;
        let width=(c.intermediate_size as u128).max(c.attention_heads as u128*d).max(h);
        let page=2*1024*1024u128;
        let kv_vector=if cap*k*2<page {(cap.max(1).next_power_of_two()*k*2).min(page)}
            else {(cap*k*2).div_ceil(page)*page};
        let kv=2*c.layers as u128*kv_vector;
        // A growing small arena may coexist with its replacement; segmented
        // caches grow in place, so a second context-sized cache is not needed.
        let growth=2*c.layers as u128*kv_vector.min(page);
        let t=cap.min(4096).div_ceil(16)*16;
        let vocab=rows(c.vocab_size as usize,rank,world).len() as u128;
        // Includes retained single-row decode plus one prompt workspace. The
        // unpacked FFN bound also covers uneven head/MLP shards and MultiState.
        let projections=4*(t+16)*(6*h+2*q+2*k+2*mid+2*width)+4*64*vocab;
        let attention=4*64*(q/d)*132*cap.div_ceil(256);
        usize::try_from(kv+growth+projections+attention+16*1024*1024).unwrap_or(usize::MAX)
    }
    pub fn load(backend: &'b Backend<'a>, root: &Path, rank: usize, world: usize, max_tokens: usize) -> Result<Self> {
        if rank >= world || max_tokens == 0 { return Err("invalid rank/world/capacity".into()); }
        backend.require_batch_kernels()?;
        let mut weights = Weights::open(root, backend)?;
        let config = weights.config.clone();
        if max_tokens as u64 > config.max_positions || max_tokens > u32::MAX as usize {
            return Err(spark_gpu_probe::admission::token_refusal(max_tokens as u128, (config.max_positions as usize).min(u32::MAX as usize), &format!("CONTEXT={max_tokens}; OUTPUT included in CONTEXT"), "the lower of checkpoint positions and GPU position range", spark_gpu_probe::admission::SERVE));
        }
        // Keep Q and O shards aligned to whole KV groups, independently of the
        // hidden width: Qwen3-32B projects hidden 5120 into 64 * 128 = 8192.
        // Every local projection is packed directly from its checkpoint slice.
        let local_kv_heads = rows(config.kv_heads as usize, rank, world).len();
        let d = config.head_dim as usize;
        let group = config.attention_heads as usize / config.kv_heads as usize;
        if config.hidden_size % 16 != 0 || local_kv_heads * group * d % 16 != 0 || config.intermediate_size % 64 != 0 {
            return Err("TP packed projections require hidden/local query widths divisible by 16 and MLP width divisible by 64".into());
        }
        let norm = weights.norm("model.norm.weight")?;
        let head = weights.matrix_units("lm_head.weight", rank, world, 1)?;
        let mut layers = Vec::with_capacity(config.layers as usize);
        for i in 0..config.layers {
            let p = format!("model.layers.{i}");
            let mut norm = |s| weights.norm(&format!("{p}.{s}.weight"));
            let (input_norm, post_norm, q_norm, k_norm) = (norm("input_layernorm")?, norm("post_attention_layernorm")?, norm("self_attn.q_norm")?, norm("self_attn.k_norm")?);
            let q=weights.matrix_units(&format!("{p}.self_attn.q_proj.weight"),rank,world,group*d)?;
            let k=weights.matrix_units(&format!("{p}.self_attn.k_proj.weight"),rank,world,d)?;
            let v=weights.matrix_units(&format!("{p}.self_attn.v_proj.weight"),rank,world,d)?;
            let gate=weights.matrix_units(&format!("{p}.mlp.gate_proj.weight"),rank,world,64)?;
            let up=weights.matrix_units(&format!("{p}.mlp.up_proj.weight"),rank,world,64)?;
            // Complete K accumulators own output rows. Summing column shards
            // would round partial FP32 accumulators before the next MMA.
            let o=weights.matrix_units(&format!("{p}.self_attn.o_proj.weight"),rank,world,128)?;
            let down=weights.matrix_units(&format!("{p}.mlp.down_proj.weight"),rank,world,128)?;
            layers.push(Layer { norms: Some(Norms { input_norm, post_norm, q_norm, k_norm }),
                q,k,v,o,gate,up,down });
            if i % 12 == 11 { eprintln!("rank={rank} loaded_layers={} device_bytes={}", i + 1, backend.allocated_bytes()); }
        }
        let embedding = weights.files.remove(weights.map.get("model.embed_tokens.weight").ok_or("missing embedding")?).ok_or("missing embedding file")?;
        let rope = layer::rope(backend, &config, max_tokens)?;
        let resident_bytes = backend.allocated_bytes();
        let load_peak_bytes = weights.peak_bytes.max(resident_bytes);
        Ok(Self { backend, config, layers, embedding: Some(RefCell::new(embedding)), norm: Some(norm), head, rope: Some(rope), full: None, sequence: false, profile: Self::profile_requested(),
            max_tokens, rank, world, local_kv_heads, resident_bytes, load_peak_bytes })
    }
    /// TP decode views the same full matrices used by sequence-parallel prefill.
    pub fn from_full(full: &'b spark_engine::qwen::Model<'a, 'b>, rank: usize, world: usize) -> Result<Self> {
        if rank >= world { return Err("invalid rank/world".into()); }
        let backend = full.backend();
        backend.require_batch_kernels()?;
        let config = full.config().clone();
        let d = config.head_dim as usize;
        let group = config.attention_heads as usize / config.kv_heads as usize;
        let shard_rows = |matrix: &Matrix<'a>, unit: usize| -> Result<Shard<'a>> {
            let range = rows(matrix.rows() / unit, rank, world);
            let first = range.start * unit;
            Ok(Shard { matrix: if range.is_empty() { None } else { Some(matrix.rows_view(first..range.end * unit)?) }, first })
        };
        let mut layers = Vec::with_capacity(config.layers as usize);
        for i in 0..config.layers as usize {
            let w = full.projection_weights(i);
            layers.push(Layer { norms: None,
                q: shard_rows(w.q.ok_or("missing full Q weights")?, group*d)?,
                k: shard_rows(w.k.ok_or("missing full K weights")?, d)?,
                v: shard_rows(w.v.ok_or("missing full V weights")?, d)?,
                o: shard_rows(w.o.ok_or("missing full O weights")?, 128)?,
                gate: shard_rows(w.gate.ok_or("missing full gate weights")?, 64)?,
                up: shard_rows(w.up.ok_or("missing full up weights")?, 64)?,
                down: shard_rows(w.down.ok_or("missing full down weights")?, 128)?,
            });
        }
        let head = shard_rows(full.output_weights()?, 1)?;
        let local_kv_heads = rows(config.kv_heads as usize, rank, world).len();
        let resident_bytes = backend.allocated_bytes();
        Ok(Self { backend, config, layers, embedding: None, norm: None, head, rope: None, full: Some(full), sequence: false, profile: Self::profile_requested(),
            max_tokens: full.max_tokens(), rank, world, local_kv_heads, resident_bytes, load_peak_bytes: resident_bytes })
    }
    pub fn config(&self) -> &Config { &self.config }
    /// Whether this process was started with SPARK_PROFILE=1.
    pub fn profile_requested() -> bool { std::env::var("SPARK_PROFILE").as_deref()==Ok("1") }
    fn layer_norms(&self, i: usize) -> Result<layer::Norms<'_, 'a>> {
        if let Some(full) = self.full { return Ok(full.projection_norms(i)); }
        let n = self.layers[i].norms.as_ref().ok_or("missing layer norms")?;
        Ok(layer::Norms { input: &n.input_norm, post: &n.post_norm, q: &n.q_norm, k: &n.k_norm,
            next: self.layers.get(i+1).and_then(|l| l.norms.as_ref()).map(|n| &n.input_norm) })
    }
    /// Every rank owns KV heads, MLP columns and hidden rows (prefill_halves).
    fn every_rank_shares(&self) -> bool {
        let c = &self.config;
        self.local_kv_heads > 0 && (0..self.world).all(|r| !rows(c.intermediate_size as usize/64, r, self.world).is_empty()
            && !rows(c.hidden_size as usize/128, r, self.world).is_empty() && !rows(c.kv_heads as usize, r, self.world).is_empty())
    }
    fn layer_rope(&self) -> Result<&DeviceVector<'a>> {
        if let Some(full) = self.full { Ok(full.rope()) } else { self.rope.as_ref().ok_or("missing rope".into()) }
    }
    /// Rank-major shape (units, width, rows) of a prompt layer's link step:
    /// attention operands, O output, gate operands (packed SwiGLU tiles), down output.
    fn prefill_gather_shape(&self, t: usize, stage: usize) -> (usize, usize, usize) {
        let c = &self.config;
        match stage {
            0 => (c.kv_heads as usize, (c.attention_heads/c.kv_heads*c.head_dim) as usize, t),
            2 if self.packed_ffn() => (c.intermediate_size as usize/64, 64*8, t.div_ceil(16)),
            2 => (c.intermediate_size as usize/64, 64, t),
            _ => (c.hidden_size as usize/128, 128, t),
        }
    }
    fn packed_ffn(&self) -> bool {
        self.config.hidden_size%128==0 && (0..self.world).all(|rank|
            rows(self.config.intermediate_size as usize/64,rank,self.world).len()%2==0)
    }
    pub fn max_tokens(&self) -> usize { self.max_tokens }
    pub fn memory_bytes(&self) -> (usize, usize, usize) {
        (self.backend.allocated_bytes(), self.backend.peak_allocated_bytes(), self.backend.pinned_host_bytes())
    }
    pub fn fatal_error(&self) -> Option<String> { self.backend.fatal_error() }
    pub fn full_model(&self) -> Option<&'b spark_engine::qwen::Model<'a, 'b>> { self.full }
    /// Conservative HTTP admission for KV growth, 4096-row prefill and decode.
    pub fn session_reservation_bytes(&self)->usize {
        let c=&self.config;let cap=self.max_tokens.div_ceil(16)*16;
        let sequence=self.full.filter(|_|self.sequence).map_or(0,|full| {
            let q=(c.attention_heads*c.head_dim) as usize;
            let k=(c.kv_heads*c.head_dim) as usize;
            let h=c.hidden_size as usize;
            // Source and destination coexist during general-ring hand-off.
            // The pair reuses it; this bound also covers bounded hand-off slots.
            full.max_cache_bytes()+4*cap*(h+q+2*k)
                +4*cap.min(4096)*(8*h+4*q+4*k+2*c.intermediate_size as usize)
        });
        if !self.sequence { Self::service_session_bytes(c,self.rank,self.world,self.max_tokens) }
        else { Self::shard_session_bytes(c,self.rank,self.world,self.max_tokens).saturating_add(sequence) }
    }
    pub fn request_reservation_bytes(&self,tokens:usize)->usize {
        if !self.sequence {Self::service_session_bytes(&self.config,self.rank,self.world,tokens.min(self.max_tokens))}
        else{self.session_reservation_bytes()}
    }
    /// The one switch for the opt-in sequence prefill (hybrid only): its charge, FP32 KV and prompt splitting.
    pub fn enable_sequence_prefill(&mut self) { self.sequence = self.full.is_some(); }
    pub fn sequence_prefill(&self) -> bool { self.sequence }
    pub fn session<'m, 'r>(&'m self, ring: &'r RefCell<Ring>) -> Session<'m, 'r, 'a, 'b> {
        // Shards follow the checkpoint; sequence hand-off still imports FP32 KV.
        Session { model: self, ring, kv: Vec::new(), len: 0, bf16: self.full.is_none(), invalid: false, collect_logits: true, decode: None, batch: None, second: None, peak_bytes: self.resident_bytes, pinned_peak:self.backend.peak_pinned_staging_bytes(), buffered:Vec::new(), sequence:None, sequence_pending:false, sequence_work:None }
    }
    fn embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let width = self.config.hidden_size as usize;
        let mut output = Vec::with_capacity(tokens.len()*width);
        if let Some(full) = self.full {
            for &token in tokens { output.extend(full.embedding(token)?); }
            return Ok(output);
        }
        let mut raw = vec![0u8; width*2];
        let file = self.embedding.as_ref().ok_or("missing embedding")?.borrow_mut();
        for &token in tokens {
            file.read_range("model.embed_tokens.weight", token as u64*width as u64*2, &mut raw).map_err(|e| e.to_string())?;
            output.extend(raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16)));
        }
        Ok(output)
    }
}
struct Kv<'a> { keys: DeviceVector<'a>, values: DeviceVector<'a> }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Logits { None, Last, All, Greedy }
struct State<'a> {
    tokens: usize, capacity: usize, prompt_end: usize, all_logits: bool, exact: bool,
    hidden: DeviceVector<'a>, input: DeviceVector<'a>, sums: DeviceVector<'a>,
    q: DeviceVector<'a>, k: DeviceVector<'a>, v: DeviceVector<'a>, attention: DeviceVector<'a>,
    projected: DeviceVector<'a>, partial: DeviceVector<'a>, gathered: DeviceVector<'a>, gate: DeviceVector<'a>, up: DeviceVector<'a>,
    scores: DeviceVector<'a>, parts: DeviceVector<'a>, weight_pack: DeviceVector<'a>, input_pack: DeviceVector<'a>,
    packed_keys: DeviceVector<'a>, packed_values: DeviceVector<'a>, exchange: Option<DeviceVector<'a>>,
    last: DeviceVector<'a>, final_input: DeviceVector<'a>, final_sums: DeviceVector<'a>, logits: DeviceVector<'a>, arg_parts: DeviceVector<'a>, arg_result: DeviceVector<'a>,
}
impl<'a> State<'a> {
    fn new(m: &Model<'a,'_>, t: usize, capacity: usize, all_logits: bool, exact: bool, bf16: bool, prompt_end: usize) -> Result<Self> {
        let b=m.backend; let c=&m.config;
        let (h,d,g,mid)=(c.hidden_size as usize,c.head_dim as usize,c.attention_heads as usize/c.kv_heads as usize,c.intermediate_size as usize);
        let k=m.local_kv_heads*d; let q=k*g;
        let local_mid=rows(mid/64,m.rank,m.world).len()*64;
        let vocab=m.head.count(); let ht=if all_logits {t} else {1};
        let padded=capacity.div_ceil(16)*16;
        let tc=!exact && d==128;
        let streaming=d==128 && (g==4 || g==8);
        let fast_attention=tc && d==128 && g==4 && t>=64 && !bf16 && b.fast_prefill_attention(prompt_end);
        let fast_ffn=!exact && m.packed_ffn();
        let packed_norms=fast_ffn && q%128==0 && k%128==0;
        let score=if tc || streaming {1} else {t.min(64)*m.local_kv_heads*g*padded};
        let parts=if tc || streaming && b.decode_attention_parts(capacity)<=1 {1} else if streaming {t.min(16)*m.local_kv_heads*g*132*b.decode_attention_parts(capacity)} else {t.min(64)*q*padded.div_ceil(64)};
        let pack_width=mid.max((c.attention_heads*c.head_dim) as usize).max(h);
        let vec=|n:usize| b.vector(n.max(1));
        let shared=|n:usize| b.shared_vector(n.max(1));
        let projected=vec(t*h)?;
        let gather_len=if fast_ffn {(t*(c.attention_heads*c.head_dim) as usize).max(t.div_ceil(16)*16*mid/2)}else{t*pack_width};
        let (partial_len,gathered_len)=if m.world>1 {
            (t*rows(h/128,m.rank,m.world).len()*128,gather_len)
        }else{(1,1)};
        let partial=vec(partial_len)?;
        let exchange=if !exact&&m.world>1 {Some(b.mapped_vector(t.div_ceil(16)*16*pack_width)?)}else{None};
        let attention=if let Some(exchange)=exchange.as_ref().filter(|_|q>0) {
            let start=rows(c.kv_heads as usize,m.rank,m.world).start*g*d*t;
            exchange.slice(start..start+t*q)?
        }else{vec(t*q)?};
        let gate_len=if fast_ffn {t.div_ceil(16)*16*local_mid/2}else{t*local_mid};
        let gate=if let Some(exchange)=exchange.as_ref().filter(|_|gate_len>0) {
            let start=rows(mid/64,m.rank,m.world).start*64*if fast_ffn{t.div_ceil(16)*8}else{t};
            exchange.slice(start..start+gate_len)?
        }else{vec(gate_len)?};
        Ok(Self { tokens:t,capacity,prompt_end,all_logits,exact,
            hidden:vec(t*h)?,input:vec(if packed_norms {1}else{t*h})?,sums:vec(t*h.div_ceil(128))?,
            q:vec(t*q)?,k:vec(t*k)?,v:vec(t*k)?,attention,
            projected,partial,gate,up:vec(if fast_ffn{1}else{t*local_mid})?,
            gathered:vec(gathered_len)?,scores:vec(score)?,parts:vec(parts)?,
            exchange,
            weight_pack:vec(1)?,input_pack:vec(t.div_ceil(16)*16*pack_width/2)?,
            packed_keys:vec(if fast_attention {capacity.div_ceil(64)*64*k} else if tc&&!bf16{capacity*k}else{1})?,
            packed_values:vec(if fast_attention {capacity.div_ceil(64)*64*k/2} else if tc&&!bf16{capacity*k}else{1})?,
            last:vec(ht*h)?,final_input:vec(ht*h)?,final_sums:vec(ht*h.div_ceil(128))?,logits:shared(ht*vocab)?,
            arg_parts:vec(vocab.div_ceil(1024)*3)?,arg_result:shared(3)?,
        })
    }
}

/// Gather operands, then gather complete output rows. Each MMA traverses the
/// full K in the single-Spark order; no independently rounded sums are added.
fn project_rows(b: &Backend<'_>, ring: &mut Ring, weight: &Shard<'_>, input: &DeviceVector<'_>,
    gathered: &mut DeviceVector<'_>, local: &mut DeviceVector<'_>, output: &mut DeviceVector<'_>,
    weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, t: usize,
    units: usize, unit: usize, hidden: usize, packed: bool, exact: bool,
    exchange: Option<&mut DeviceVector<'_>>) -> Result<()> {
    let project = |input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
        weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>|
        project(b, weight, input, output, weight_pack, input_pack, t, packed, exact);
    if ring.world() == 1 { return project(input, output, weight_pack, input_pack); }
    if let Some(exchange)=exchange {
        let (rows,width)=if packed{(t.div_ceil(16),unit*8)}else{(t,unit)};
        gather_device(b,ring,input,gathered,exchange,units,width,rows)?;
        let own=crate::rows(hidden/128,ring.rank(),ring.world());
        if own.is_empty() {return gather_device(b,ring,local,output,exchange,hidden/128,128,t);}
        let mut projected=exchange.slice(own.start*128*t..own.end*128*t)?;
        project(gathered,&mut projected,weight_pack,input_pack)?;
        return gather_device(b,ring,&projected,output,exchange,hidden/128,128,t);
    }
    // Packed SwiGLU is [row tile][K tile][MMA fragment], with eight FP32
    // storage words per K column in a 16-row tile. Transport preserves bits.
    let (rows, width) = if packed { (t.div_ceil(16), unit*8) } else { (t, unit) };
    let values=ring.gather_batch_with(units,width,rows,|out|b.read_vector_range_into(input,0,out))?;
    gathered.resize_within_capacity(values.len())?;
    b.write_vector(gathered,values)?;
    project(gathered, local, weight_pack, input_pack)?;
    b.write_vector(output, ring.gather_batch_with(hidden/128,128,t,|out|b.read_vector_range_into(local,0,out))?)
}

#[allow(clippy::too_many_arguments)]
fn project(b: &Backend<'_>, weight: &Shard<'_>, input: &DeviceVector<'_>, output: &mut DeviceVector<'_>,
    weight_pack: &mut DeviceVector<'_>, input_pack: &mut DeviceVector<'_>, t: usize, packed: bool, exact: bool) -> Result<()> {
    if let Some(matrix) = &weight.matrix {
        if packed { b.matmul_packed_device(matrix, input, output, t) }
        else { b.matmul_device(matrix, input, output, weight_pack, input_pack, t, exact) }
    } else { Ok(()) }
}

/// No CPU activation copies: DMA/GPU copy into a pinned rank block, exchange,
/// then interleave the unchanged words with the shared strided-copy kernel.
fn gather_device(b:&Backend<'_>,ring:&mut Ring,input:&DeviceVector<'_>,output:&mut DeviceVector<'_>,
    exchange:&mut DeviceVector<'_>,units:usize,width:usize,tokens:usize)->Result<()> {
    own_block(b,input,exchange,ring.rank(),ring.world(),units,width,tokens)?;
    b.synchronize()?;
    exchange_blocks(b,ring,exchange,units,width,tokens,false)?;
    spread_blocks(b,ring.world(),exchange,output,units,width,tokens)
}
/// Place this rank's block at its rank-major offset in `exchange`; producers
/// that write there directly (exchange slices) need no copy.
#[allow(clippy::too_many_arguments)]
fn own_block(b:&Backend<'_>,input:&DeviceVector<'_>,exchange:&mut DeviceVector<'_>,rank:usize,world:usize,
    units:usize,width:usize,tokens:usize)->Result<()> {
    exchange.resize_within_capacity(units.checked_mul(width).and_then(|n|n.checked_mul(tokens)).ok_or("device gather overflow")?)?;
    let own=rows(units,rank,world);
    if input.device_address()!=exchange.device_address()+(own.start*width*tokens*4) as u64 {
        b.copy_range(input,exchange,0,own.start*width*tokens,own.len()*width*tokens)?;
    }
    Ok(())
}
/// The link step of gather_device: the rank blocks are complete in `exchange`
/// once the queued GPU writes before it have finished.
pub(crate) fn exchange_blocks(b:&Backend<'_>,ring:&mut Ring,exchange:&mut DeviceVector<'_>,units:usize,width:usize,tokens:usize,finite:bool)->Result<()> {
    let total=units.checked_mul(width).and_then(|n|n.checked_mul(tokens)).ok_or("device gather overflow")?;
    exchange.resize_within_capacity(total)?;
    let host=b.mapped_exchange_host(exchange)?;
    let blocks=unsafe{std::slice::from_raw_parts_mut(host,total)};
    ring.gather_rank_blocks(blocks,units,width,tokens)?;
    if finite && blocks.iter().fold(0u32,|any,v|any|u32::from(v.to_bits()&0x7f80_0000==0x7f80_0000))!=0 {return Err("exchanged blocks are not finite".into());}
    Ok(())
}
fn spread_blocks(b:&Backend<'_>,world:usize,exchange:&DeviceVector<'_>,output:&mut DeviceVector<'_>,units:usize,width:usize,tokens:usize)->Result<()> {
    output.resize_within_capacity(units*width*tokens)?;
    for rank in 0..world {
        let r=rows(units,rank,world);
        b.copy_strided(exchange,output,r.start*width*tokens,r.start*width,r.len()*width,units*width,r.len()*width,tokens)?;
    }
    Ok(())
}

/// Prompt chunks of at least this many rows run as two halves (prefill_halves).
const HALVES_ROWS: usize = 1024;

/// Two row halves of one prompt chunk take turns on the GPU and the link: while
/// one half's rank blocks cross the link, the stream computes the other half's
/// next stage. Each half runs the kernels of a prompt chunk of its own rows, the
/// second after the first's cache rows, as HTTP prefill chunks already do.
fn prefill_halves<'a>(m: &Model<'a, '_>, ring: &mut Ring, kv: &mut [Kv<'_>], mut halves: [&mut State<'_>; 2], position: usize) -> Result<()> {
    let b=m.backend;
    let positions=[position,position+halves[0].tokens];
    let steps=4*m.layers.len();
    // One fence per half, recorded after each stage so that stage's exchange block is complete.
    let mut start=|x:usize|->Result<Fence<'a>> { prefill_stage(m,halves[x],kv,positions[x],0)?; b.fence() };
    let mut fences=[start(0)?,start(1)?];
    for step in 0..steps {
        for (x,s) in halves.iter_mut().enumerate() {
            fences[x].wait()?;
            let (units,width,rows)=m.prefill_gather_shape(s.tokens,step%4);
            exchange_blocks(b,ring,s.exchange.as_mut().ok_or("missing exchange")?,units,width,rows,false)?;
            prefill_stage(m,s,kv,positions[x],step+1)?;
            if step+1<steps { fences[x]=b.fence()?; }
        }
    }
    Ok(())
}
/// GPU work of one half between link steps: spread link step `step-1`, then
/// compute up to link step `step` (four per layer: attention operands, O
/// output, gate operands, down output), or the last residual.
fn prefill_stage(m: &Model<'_, '_>, s: &mut State<'_>, kv: &mut [Kv<'_>], position: usize, step: usize) -> Result<()> {
    let b=m.backend; let t=s.tokens; let (i,stage)=(step/4,step%4);
    if step>0 {
        let (units,width,rows)=m.prefill_gather_shape(t,(step-1)%4);
        let output=if stage%2==1 {&mut s.gathered} else {&mut s.projected};
        spread_blocks(b,m.world,s.exchange.as_ref().ok_or("missing exchange")?,output,units,width,rows)?;
    }
    if i==m.layers.len() { return layer_output(m,&m.layer_norms(i-1)?,s,false); }
    let (layer,norms)=(&m.layers[i],m.layer_norms(i)?);
    match stage {
        0 => {
            if i>0 { layer_output(m,&m.layer_norms(i-1)?,s,false)?; }
            layer_attention(m,layer,&norms,m.layer_rope()?,kv.get_mut(i).map(|kv|(&mut kv.keys,&mut kv.values)),s,position,false,true,i==0)?;
        }
        2 => layer_mlp(m,layer,&norms,s,false)?,
        _ => {}
    }
    // This rank's block of the step's operands: copied (attention, gate) or
    // written straight by the O/down GEMM.
    let (units,width,n)=m.prefill_gather_shape(t,stage);
    let exchange=s.exchange.as_mut().ok_or("missing exchange")?;
    match stage {
        0 => own_block(b,&s.attention,exchange,m.rank,m.world,units,width,n),
        2 => own_block(b,&s.gate,exchange,m.rank,m.world,units,width,n),
        _ => {
            let (weight,packed)=if stage==1 {(&layer.o,false)} else {(&layer.down,m.packed_ffn())};
            exchange.resize_within_capacity(units*width*n)?;
            let own=rows(units,m.rank,m.world);
            let mut block=exchange.slice(own.start*width*n..own.end*width*n)?;
            project(b,weight,&s.gathered,&mut block,&mut s.weight_pack,&mut s.input_pack,t,packed,false)
        }
    }
}

fn forward_layer(m: &Model<'_, '_>, ring: &mut Ring, layer: &Layer<'_>, norms: layer::Norms<'_, '_>,
    rope: &DeviceVector<'_>, cache: Option<(&mut DeviceVector<'_>, &mut DeviceVector<'_>)>,
    s: &mut State<'_>, position: usize, exact: bool, bf16: bool, first: bool) -> Result<()> {
    let b=m.backend; let c=&m.config; let t=s.tokens; let h=c.hidden_size as usize;
    let group=c.attention_heads as usize/c.kv_heads as usize;
    layer_attention(m,layer,&norms,rope,cache,s,position,exact,bf16,first)?;
    project_rows(b,ring,&layer.o,&s.attention,&mut s.gathered,&mut s.partial,&mut s.projected,
        &mut s.weight_pack,&mut s.input_pack,t,c.kv_heads as usize,group*c.head_dim as usize,h,false,exact,s.exchange.as_mut())?;
    layer_mlp(m,layer,&norms,s,exact)?;
    let packed=!exact && m.packed_ffn();
    project_rows(b,ring,&layer.down,&s.gate,&mut s.gathered,&mut s.partial,&mut s.projected,
        &mut s.weight_pack,&mut s.input_pack,t,c.intermediate_size as usize/64,64,h,packed,exact,s.exchange.as_mut())?;
    layer_output(m,&norms,s,exact)
}

/// Input norm, QKV, rope, cache rows and attention into `s.attention`.
#[allow(clippy::too_many_arguments)]
fn layer_attention(m: &Model<'_, '_>, layer: &Layer<'_>, norms: &layer::Norms<'_, '_>,
    rope: &DeviceVector<'_>, cache: Option<(&mut DeviceVector<'_>, &mut DeviceVector<'_>)>,
    s: &mut State<'_>, position: usize, exact: bool, bf16: bool, first: bool) -> Result<()> {
    let b=m.backend; let c=&m.config; let t=s.tokens; let h=c.hidden_size as usize;
    let eps=c.rms_norm_eps as f32; let d=c.head_dim as usize;
    let group=c.attention_heads as usize/c.kv_heads as usize;
    let decode=t==1 && exact;
    if decode && first { b.rms_device(&s.hidden, norms.input, &mut s.input, &mut s.sums, eps)?; }
    if let Some((keys, values))=cache {
        let (q,k,v)=(layer.q.matrix.as_ref().unwrap(),layer.k.matrix.as_ref().unwrap(),layer.v.matrix.as_ref().unwrap());
        if !decode && q.cols()%128==0 && q.rows()%128==0 && k.rows()%128==0 {
            if first { b.rms_partials_device(&s.hidden, &mut s.sums, h)?; }
            b.rms_pack_apply_device(&s.hidden, norms.input, &mut s.input_pack, &s.sums, eps, false)?;
            b.prefill_qkv_device(q,k,v,&s.input_pack,&mut s.q,&mut s.k,&mut s.v,t,false)?;
        } else {
            if !decode { b.rms_batch_device(&s.hidden,norms.input,&mut s.input,&mut s.sums,eps)?; }
            b.matmul_device(q,&s.input,&mut s.q,&mut s.weight_pack,&mut s.input_pack,t,exact)?;
            b.matmul_reusing_input(k,&s.input,&mut s.k,&mut s.weight_pack,&mut s.input_pack,t,exact,true)?;
            b.matmul_reusing_input(v,&s.input,&mut s.v,&mut s.weight_pack,&mut s.input_pack,t,exact,true)?;
        }
        let heads=m.local_kv_heads*group;
        if decode && !bf16 {
            b.qk_rope_cache_device(&mut s.q,&s.k,&s.v,norms.q,norms.k,rope,keys,values,position,eps)?;
            b.attention_device(&s.q,keys,values,&mut s.scores,&mut s.parts,&mut s.attention,heads,m.local_kv_heads,d,position+1)?;
        } else {
            b.qk_batch_device(&mut s.q,&s.k,&s.v,norms.q,norms.k,rope,keys,values,position,eps,t,bf16)?;
            if !decode && !exact && d==128 {
                b.prefill_attention_device(&s.q,keys,values,&mut s.attention,&mut s.packed_keys,&mut s.packed_values,
                    heads,m.local_kv_heads,position,t,bf16,s.prompt_end,false)?;
            } else {
                b.attention_batch_device(&s.q,keys,values,&mut s.scores,&mut s.parts,&mut s.attention,
                    heads,m.local_kv_heads,d,position,t,bf16)?;
            }
        }
    }
    Ok(())
}

/// Residual of the gathered O projection, post norm, gate/up into `s.gate`.
fn layer_mlp(m: &Model<'_, '_>, layer: &Layer<'_>, norms: &layer::Norms<'_, '_>, s: &mut State<'_>, exact: bool) -> Result<()> {
    let b=m.backend; let c=&m.config; let t=s.tokens; let h=c.hidden_size as usize;
    let eps=c.rms_norm_eps as f32;
    let decode=t==1 && exact;
    if decode { b.residual_rms_device(&mut s.hidden,&s.projected,norms.post,&mut s.input,&mut s.sums,eps)?; }
    else { b.residual_device(&mut s.hidden,&s.projected)?; b.rms_partials_device(&s.hidden,&mut s.sums,h)?; }
    let packed=!decode && !exact && m.packed_ffn();
    if let (Some(gate),Some(up))=(&layer.gate.matrix,&layer.up.matrix) {
        if packed {
            b.rms_pack_apply_device(&s.hidden,norms.post,&mut s.input_pack,&s.sums,eps,false)?;
            b.prefill_gate_up_packed_device(gate,up,&s.input_pack,&mut s.gate,t)?;
        } else {
            if !decode { b.rms_batch_device(&s.hidden,norms.post,&mut s.input,&mut s.sums,eps)?; }
            b.matmul_device(gate,&s.input,&mut s.gate,&mut s.weight_pack,&mut s.input_pack,t,exact)?;
            b.matmul_reusing_input(up,&s.input,&mut s.up,&mut s.weight_pack,&mut s.input_pack,t,exact,true)?;
            b.swiglu_device(&mut s.gate,&s.up)?;
        }
    }
    Ok(())
}

/// Residual of the gathered down projection.
fn layer_output(m: &Model<'_, '_>, norms: &layer::Norms<'_, '_>, s: &mut State<'_>, exact: bool) -> Result<()> {
    let b=m.backend; let c=&m.config; let h=c.hidden_size as usize;
    let eps=c.rms_norm_eps as f32;
    let decode=s.tokens==1 && exact;
    if let Some(next)=norms.next.filter(|_|decode) {
        b.residual_rms_device(&mut s.hidden,&s.projected,next,&mut s.input,&mut s.sums,eps)
    } else {
        b.residual_device(&mut s.hidden,&s.projected)?;
        if !decode { b.rms_partials_device(&s.hidden,&mut s.sums,h)?; }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum SequenceStage { Begin, Project, Export, Exchange, Import, FinishLayer, Complete }
struct SequenceWork {
    stage:SequenceStage, layer:usize, row:usize, done:bool,
    keys:Vec<f32>, values:Vec<f32>,
    started:Instant, stats:spark_seqpar::prefill::PrefillStats,
}

/// Shared projections across independent sessions. Prefill concatenates prompt
/// rows; each session retains only its rank's heads in its own cache.
pub struct Multi<'m, 'a, 'b> {
    model: &'m Model<'a, 'b>, state: Option<MultiState<'a>>, attention: Option<MultiAttention<'a>>, peak_bytes: usize,
}
/// Bind one ordered rank transport once; forward/verify/rollback then have core's signatures.
/// All ranks must make the same calls. A transport failure poisons the session.
pub struct Session<'m,'r,'a,'b> {
    model: &'m Model<'a,'b>, ring: &'r RefCell<Ring>, kv: Vec<Kv<'a>>, len: usize, bf16: bool, invalid: bool,
    pub(crate) collect_logits: bool, // Service workers participate in gathers without copying unused rows.
    decode: Option<State<'a>>, batch: Option<State<'a>>,
    /// Second row half of a split prompt chunk (prefill_halves).
    second: Option<State<'a>>, peak_bytes: usize, pinned_peak:usize,
    buffered:Vec<u32>, sequence:Option<(spark_engine::qwen::Session<'b,'a,'b>,spark_seqpar::Partition)>, sequence_pending:bool,
    sequence_work:Option<SequenceWork>,
}
impl Drop for Session<'_,'_,'_,'_> {
    fn drop(&mut self){self.ring.borrow_mut().clear_registered();}
}
impl<'m,'r,'a,'b> Session<'m,'r,'a,'b> {
    pub fn kv_bf16(&self)->bool {self.bf16}
    pub fn valid(&self)->bool {!self.invalid}
    pub fn len(&self)->usize {if self.sequence_pending {self.sequence.as_ref().unwrap().1.tokens()}else{self.len+self.buffered.len()}}
    pub fn is_empty(&self)->bool {self.len()==0}
    pub fn checkpoint(&self)->usize {self.len()}
    fn invalidate_to(&mut self,position:usize) {
        self.len=position;self.buffered.clear();self.sequence_pending=false;self.invalid=true;
    }
    pub fn truncate(&mut self,position:usize)->Result<()> {
        if self.invalid&&position==self.len(){return Ok(());}
        if self.invalid {return Err("TP session invalid after failed collective".into());}
        if position>self.len() {return Err("cannot truncate beyond current position".into());}
        if !self.buffered.is_empty(){self.buffered.truncate(position);return Ok(());}
        self.handoff_sequence()?;
        self.len=position; Ok(())
    }
    pub fn reset(&mut self) {self.len=0;self.buffered.clear();self.sequence_pending=false;self.sequence=None;self.sequence_work=None;}
    pub fn configure_kv_bf16(&mut self,enabled:bool)->Result<()> {
        if !self.kv.is_empty() || !self.is_empty() || self.sequence.is_some() {return Err("KV format must be selected before first forward".into());}
        self.bf16=enabled; Ok(())
    }
    pub fn can_fork(&self)->bool { !self.invalid&&self.buffered.is_empty()&&!self.sequence_pending&&self.sequence_work.is_none() }
    pub fn fork(&self)->Result<Self> {
        if self.invalid {return Err("cannot fork invalid TP session".into());}
        if !self.can_fork(){return Err("sequence prefill has no reusable TP cache yet".into());}
        let mut branch=self.model.session(self.ring);
        branch.bf16=self.bf16;
        branch.collect_logits=self.collect_logits;
        if self.len>0 {
            let stride=self.model.local_kv_heads*self.model.config.head_dim as usize;
            let storage=if self.bf16 {2}else{1};
            let count=(self.len*stride).div_ceil(storage);
            if self.kv.first().is_some_and(|kv|kv.keys.is_segmented()) {
                let sources=self.kv.iter().flat_map(|kv|[&kv.keys,&kv.values]).collect::<Vec<_>>();
                let mut buffers=self.model.backend.fork_vectors(&sources,count,count,(self.model.max_tokens*stride).div_ceil(storage))?.into_iter();
                branch.kv=(0..self.kv.len()).map(|_|Kv{keys:buffers.next().unwrap(),values:buffers.next().unwrap()}).collect();
            }else{
                branch.reserve(self.len)?;
                for(src,dst)in self.kv.iter().zip(&mut branch.kv){
                    self.model.backend.copy_prefix(&src.keys,&mut dst.keys,count)?;
                    self.model.backend.copy_prefix(&src.values,&mut dst.values,count)?;
                }
            }
            self.model.backend.synchronize()?;
        }
        branch.len=self.len;Ok(branch)
    }
    pub fn cache_bytes(&self)->usize {self.kv.iter().map(|k|(k.keys.len()+k.values.len())*4).sum::<usize>()+self.sequence.as_ref().map_or(0,|(source,_)|source.cache_bytes())}
    pub fn cache_allocations(&self)->Option<Vec<(usize,usize)>> {
        if self.sequence.is_some(){return None;}
        let mut out=Vec::new();
        for kv in &self.kv {kv.keys.allocations(&mut out);kv.values.allocations(&mut out);}
        out.sort_unstable_by_key(|&(id,_)|id);out.dedup_by_key(|entry|entry.0);Some(out)
    }
    pub fn prefix_allocations(&self,tokens:usize)->Option<Vec<(usize,usize)>> {
        if self.sequence.is_some(){return None;}
        let words=(tokens.min(self.len)*self.model.local_kv_heads*self.model.config.head_dim as usize).div_ceil(if self.bf16{2}else{1});
        let mut out=Vec::new();
        for kv in &self.kv {kv.keys.prefix_allocations(words,&mut out);kv.values.prefix_allocations(words,&mut out);}
        Some(out)
    }
    pub fn device_peak_bytes(&self)->usize {self.peak_bytes}
    /// Backend's pinned transfer pool uses a 4 MiB minimum and power-of-two
    /// growth. Account for old+new allocations during replacement as well.
    pub fn pinned_peak_bytes(&self)->usize {self.pinned_peak}
    pub fn collective_stats(&self)->(u64,Duration) {let r=self.ring.borrow();(r.exchanges,r.elapsed)}
    pub fn barrier(&mut self)->Result<()> {self.ring.borrow_mut().barrier()}
    pub fn step(&mut self,token:u32)->Result<Vec<f32>> {self.run(&[token],Logits::All,true)?.pop().or_else(||(!self.collect_logits).then(Vec::new)).ok_or("missing logit row".into())}
    pub fn forward(&mut self,token:u32)->Result<Vec<f32>> {self.step(token)}
    pub fn step_greedy(&mut self,token:u32)->Result<u32> {
        self.run(&[token],Logits::Greedy,true)?.first().map(|row|row[0] as u32).ok_or("missing greedy token".into())
    }
    pub fn forward_batch(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>> {self.run(tokens,Logits::All,true)}
    /// Strict by default, including for callers using core's new explicit name.
    pub fn verify_batch(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>> {self.forward_batch(tokens)}
    /// Native BF16 prefill attention; strict verification keeps decode reductions.
    pub fn forward_batch_fast(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>> {self.run(tokens,Logits::All,false)}
    pub fn prefill(&mut self,tokens:&[u32])->Result<Vec<f32>> {
        self.prefill_chunk(tokens,self.len().saturating_add(tokens.len()),true)
    }
    pub fn prefill_chunk_size(&self,prompt_end:usize)->usize {
        // BF16 sessions (shard and hybrid) prefill on the shards in 4096-row chunks.
        if self.bf16||self.model.full.is_none()&&self.model.backend.fast_prefill_attention(prompt_end){4096}else{512}
    }
    /// Keep full-prompt attention dispatch consistent across HTTP prefill chunks.
    pub fn prefill_chunk(&mut self,tokens:&[u32],prompt_end:usize,final_chunk:bool)->Result<Vec<f32>> {
        self.handoff_sequence()?;
        if !self.buffered.is_empty(){return Err("sequence prompt must finish before TP prefill".into());}
        if tokens.is_empty()||tokens.len()>self.model.max_tokens-self.len||tokens.iter().any(|&t|t as u64>=self.model.config.vocab_size){return Err("prefill outside vocabulary or capacity".into());}
        let end=self.len+tokens.len();
        if prompt_end<end||prompt_end>self.model.max_tokens||final_chunk&&prompt_end!=end{return Err("prefill prompt end outside context".into());}
        // Keep physical pages small enough to share a prefix ending inside a
        // long prompt; one context-sized allocation would copy on every fork.
        let stride=self.model.local_kv_heads*self.model.config.head_dim as usize;
        if stride>0 && self.bf16 {
            let page_tokens=(2*1024*1024/(stride*2)).max(1);
            let reserved=self.kv.first().map_or(0,|kv|kv.keys.len()*2/stride);
            for page in reserved/page_tokens+1..=prompt_end.div_ceil(page_tokens) {
                self.reserve((page*page_tokens).min(prompt_end))?;
            }
        }else{self.reserve(prompt_end)?;}
        let chunk_size=self.prefill_chunk_size(prompt_end);
        let entry=self.len;
        let result=(||{let mut last=Vec::new();
            for(i,chunk)in tokens.chunks(chunk_size).enumerate(){
                let output=if final_chunk&&(i+1)*chunk_size>=tokens.len(){Logits::Last}else{Logits::None};
                if let Some(row)=self.run_to(chunk,output,false,prompt_end)?.pop(){last=row;}
            }Ok(last)
        })();
        if result.as_ref().is_err_and(|error|error==OUTPUT_ERROR){self.invalidate_to(entry);}
        if final_chunk||result.is_err(){let mut ring=self.ring.borrow_mut();ring.clear_registered();self.batch=None;self.second=None;ring.release_scratch();}
        result
    }
    pub fn advance(&mut self,token:u32)->Result<()> {self.step(token).map(|_|())}
    pub fn can_sequence(&self)->bool {self.model.full.is_some()&&self.len==0&&self.sequence.is_none()&&!self.bf16&&!self.invalid}
    /// Prepare a resumable hybrid job. Every rank receives the same prompt;
    /// GPU work and the layer exchange advance only under scheduler commands.
    pub fn start_sequence(&mut self,tokens:&[u32],ring:Option<&spark_seqpar::Ring>)->Result<()> {
        if !self.can_sequence()||!self.buffered.is_empty()||tokens.is_empty()||tokens.len()>self.model.max_tokens
            ||tokens.iter().any(|&t|t as u64>=self.model.config.vocab_size) {
            return Err("invalid resumable sequence prompt".into());
        }
        let weight=std::env::var("SEQPAR_ATTENTION_WEIGHT").map_or(Ok(spark_seqpar::Partition::prefill_weight(tokens.len())),|value|value.parse::<f64>().map_err(|e|e.to_string()))?;
        let partition=spark_seqpar::Partition::prefill(tokens.len(),self.model.world,weight)?;
        let width=(self.model.config.kv_heads*self.model.config.head_dim) as usize;
        let exchange=match ring {Some(ring)=>ring.prefix_tokens(&partition)?,None if self.model.world==1=>0,None=>return Err("missing sequence ring".into())};
        let mut source=self.model.full.unwrap().session();
        source.configure_kv_bf16(false)?;
        self.buffered.extend_from_slice(tokens);
        self.sequence=Some((source,partition));
        self.sequence_work=Some(SequenceWork{stage:SequenceStage::Begin,layer:0,row:0,done:false,
            keys:vec![0.0;exchange*width],values:vec![0.0;exchange*width],started:Instant::now(),
            stats:spark_seqpar::prefill::PrefillStats::default()});
        Ok(())
    }
    /// All ranks rendezvous after a bounded GPU slice, so a different session
    /// can use TP collectives between slices without crossing command streams.
    pub fn advance_sequence(&mut self,budget:Duration,ring:Option<&mut spark_seqpar::Ring>)->Result<Option<Vec<f32>>> {
        let b=self.model.backend;
        b.use_vec8(false)?;
        let result=self.advance_sequence_inner(budget,ring);
        let restored=b.use_vec8(true);
        self.peak_bytes=self.peak_bytes.max(b.allocated_bytes()).max(self.sequence.as_ref().map_or(0,|(s,_)|s.device_peak_bytes()));
        self.pinned_peak=self.pinned_peak.max(b.peak_pinned_staging_bytes());
        if result.is_err()||restored.is_err(){self.invalid=true;}
        let result=result?;restored?;Ok(result)
    }
    fn advance_sequence_inner(&mut self,budget:Duration,mut ring:Option<&mut spark_seqpar::Ring>)->Result<Option<Vec<f32>>> {
        if self.invalid{return Err("TP session invalid after failed collective".into());}
        let m=self.model;
        let device_before=m.backend.stats();
        let (source,partition)=self.sequence.as_mut().ok_or("sequence prefill not started")?;
        let work=self.sequence_work.as_mut().ok_or("sequence prefill already completed")?;
        let local=partition.range(m.rank);
        let width=(m.config.kv_heads*m.config.head_dim) as usize;
        if matches!(work.stage,SequenceStage::Complete) {
            let phase=Instant::now();
            #[cfg(target_os="linux")]
            if let Some(ring)=ring.as_deref_mut(){ring.retain_regions(false);}
            let logits=source.sequence_finish_values()?;
            let vocab=m.config.vocab_size as usize;
            let mut bytes=vec![0u8;vocab*4];
            if m.rank+1==m.world {
                if logits.len()!=vocab{return Err("missing sequence prefill logits".into());}
                for(dst,logit)in bytes.chunks_exact_mut(4).zip(logits){dst.copy_from_slice(&logit.to_le_bytes());}
            }
            self.ring.borrow_mut().with_transfer_regions(|ring|ring.transfer_from_last(0,&mut bytes))?;
            finite_bytes(&bytes)?;
            work.stats.head=phase.elapsed();
            work.stats.add_backend(m.backend.stats().since(device_before));
            eprintln!("hybrid_prefill rank={} tokens={} elapsed_ms={:.3} kv_bytes={} peak_device_bytes={}",
                m.rank,partition.tokens(),work.started.elapsed().as_secs_f64()*1000.0,source.cache_bytes(),source.device_peak_bytes());
            work.stats.report("hybrid",m.rank);
            self.buffered.clear();self.sequence_work=None;self.sequence_pending=true;
            return Ok(Some(if self.collect_logits{bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect()}else{Vec::new()}));
        }
        if !work.done {
            let phase=Instant::now();
            work.done=match work.stage {
                SequenceStage::Begin=>{
                    let done=source.sequence_begin_slice(&self.buffered[local.clone()],local.start,local.end,partition.tokens(),budget)?;
                    #[cfg(target_os="linux")]
                    if done { if let Some(ring)=ring.as_deref_mut(){
                        ring.retain_regions(true);
                        unsafe{ring.prepare_prefix_regions(&mut work.keys,&mut work.values)?;}
                    }}
                    done
                },
                SequenceStage::Project=>source.sequence_project_slice(work.layer,budget)?,
                SequenceStage::Export=>{
                    while m.rank+1<m.world&&work.row<local.len() {
                        let start=local.start+work.row;
                        let end=(start+spark_engine::qwen::PREFILL_CHUNK).min(local.end);
                        source.sequence_export_into(work.layer,start,&mut work.keys[start*width..end*width],&mut work.values[start*width..end*width])?;
                        work.row=end-local.start;
                        if phase.elapsed()>=budget {break;}
                    }
                    m.rank+1==m.world||work.row==local.len()
                },
                SequenceStage::Exchange=>{
                    if let Some(ring)=ring {
                        #[cfg(target_os="linux")]
                        {
                            let before=ring.stats;
                            ring.exchange_prefix_kv(&mut work.keys,&mut work.values,partition,width)?;
                            let delta=ring.stats.since(before);
                            work.stats.transport.sent+=delta.sent;
                            work.stats.transport.received+=delta.received;
                        }
                    }
                    true
                },
                SequenceStage::Import=>{
                    while work.row<local.start {
                        let end=(work.row+spark_engine::qwen::PREFILL_CHUNK).min(local.start);
                        source.sequence_import(work.layer,work.row,&work.keys[work.row*width..end*width],&work.values[work.row*width..end*width])?;
                        work.row=end;
                        if phase.elapsed()>=budget {break;}
                    }
                    work.row==local.start
                },
                SequenceStage::FinishLayer=>source.sequence_finish_layer_slice(work.layer,budget)?,
                SequenceStage::Complete=>unreachable!(),
            };
            let elapsed=phase.elapsed();
            match work.stage {
                SequenceStage::Begin=>work.stats.begin+=elapsed,
                SequenceStage::Project=>work.stats.project+=elapsed,
                SequenceStage::Export=>work.stats.export+=elapsed,
                SequenceStage::Exchange=>work.stats.exchange+=elapsed,
                SequenceStage::Import=>work.stats.import+=elapsed,
                SequenceStage::FinishLayer=>work.stats.attention_ffn+=elapsed,
                SequenceStage::Complete=>{},
            }
        }
        let mut completed=vec![0u8;m.world];completed[m.rank]=u8::from(work.done);
        let phase=Instant::now();
        self.ring.borrow_mut().all_gather(&mut completed,1)?;
        work.stats.schedule_wait+=phase.elapsed();
        work.stats.add_backend(m.backend.stats().since(device_before));
        if completed.iter().all(|&done|done!=0) {
            work.stage=match work.stage {
                SequenceStage::Begin=>SequenceStage::Project,
                SequenceStage::Project=>SequenceStage::Export,
                SequenceStage::Export=>SequenceStage::Exchange,
                SequenceStage::Exchange=>SequenceStage::Import,
                SequenceStage::Import=>SequenceStage::FinishLayer,
                SequenceStage::FinishLayer=>{work.layer+=1;if work.layer==m.layers.len(){SequenceStage::Complete}else{SequenceStage::Project}},
                SequenceStage::Complete=>unreachable!(),
            };
            work.row=0;work.done=false;
        }
        Ok(None)
    }
    /// Buffer scheduler chunks, then publish the final rank's prompt logits.
    /// The first decode performs cache hand-off after HTTP emits that token.
    pub fn prefill_sequence(&mut self,tokens:&[u32],prompt_end:usize,final_chunk:bool,ring:Option<&mut spark_seqpar::Ring>)->Result<Vec<f32>> {
        let entry=self.len();
        if !self.can_sequence()||tokens.is_empty()||prompt_end>self.model.max_tokens
            || self.buffered.len()+tokens.len()>prompt_end
            || tokens.iter().any(|&t|t as u64>=self.model.config.vocab_size)
            || final_chunk&&self.buffered.len()+tokens.len()!=prompt_end {
            return Err("invalid sequence prefill chunk".into());
        }
        self.buffered.extend_from_slice(tokens);
        if !final_chunk{return Ok(Vec::new());}
        let weight=std::env::var("SEQPAR_ATTENTION_WEIGHT").map_or(Ok(spark_seqpar::Partition::prefill_weight(prompt_end)),|value|value.parse::<f64>().map_err(|e|e.to_string()))?;
        let partition=spark_seqpar::Partition::prefill(prompt_end,self.model.world,weight)?;
        let mut source=self.model.full.unwrap().session();
        source.configure_kv_bf16(false)?;
        let b=self.model.backend;
        b.use_vec8(false)?;
        self.invalid=true;
        let options=spark_seqpar::prefill::PrefillOptions {
            overlap:std::env::var("SEQPAR_OVERLAP").as_deref()!=Ok("0"),..Default::default()
        };
        let result=spark_seqpar::prefill::prefill_with_finish(&mut source,&self.buffered,ring,&partition,options,|source,pending|{
            if let Some(layer)=pending{source.sequence_finish_layer(layer)?;}
            source.sequence_finish_values()
        });
        let restored=b.use_vec8(true);
        let (logits,_)=result?;restored?;
        let vocab=self.model.config.vocab_size as usize;
        let mut bytes=vec![0u8;vocab*4];
        if self.model.rank+1==self.model.world {
            if logits.len()!=vocab{return Err("missing sequence prefill logits".into());}
            for(dst,logit)in bytes.chunks_exact_mut(4).zip(logits){dst.copy_from_slice(&logit.to_le_bytes());}
        }
        self.ring.borrow_mut().with_transfer_regions(|ring|ring.transfer_from_last(0,&mut bytes))?;
        if let Err(error)=finite_bytes(&bytes){self.invalidate_to(entry);return Err(error);}
        self.peak_bytes=self.peak_bytes.max(source.device_peak_bytes()).max(b.allocated_bytes());
        self.pinned_peak=self.pinned_peak.max(b.peak_pinned_staging_bytes());
        self.buffered.clear();self.sequence=Some((source,partition));self.sequence_pending=true;self.invalid=false;
        Ok(if self.collect_logits{bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect()}else{Vec::new()})
    }
    fn handoff_sequence(&mut self)->Result<()> {
        if !self.sequence_pending{return Ok(());}
        let (mut source,partition)=self.sequence.take().unwrap();
        self.sequence_pending=false;
        let result=self.import_sequence_with_finish(&mut source,&partition,|_|Ok(()));
        result.map(|_|())
    }
    /// Pipeline bounded suffix pieces with GPU packing and cache installation.
    /// The first transfer overlaps the caller's final prefill computation. CUDA
    /// stays on this thread; transport owns each queued pinned slot until return.
    pub fn import_sequence_with_finish(&mut self, source: &mut spark_engine::qwen::Session<'_, 'a, '_>,
        partition: &spark_seqpar::Partition,
        finish: impl FnOnce(&mut spark_engine::qwen::Session<'_, 'a, '_>) -> Result<()>) -> Result<usize> {
        let started = Instant::now();
        let m = self.model;
        let b = m.backend;
        let count = partition.tokens();
        if !self.is_empty() || self.bf16 || source.kv_bf16() || partition.world() != m.world
            || count > m.max_tokens {
            return Err("sequence hand-off requires fresh FP32 TP state and matching causal prefixes".into());
        }
        self.invalid = true;
        let d = m.config.head_dim as usize;
        let stride = m.config.kv_heads as usize * d;
        let local_heads = rows(m.config.kv_heads as usize, m.rank, m.world);
        let width = local_heads.len() * d;
        let prefix = partition.range(m.rank).end;
        let first_heads = rows(m.config.kv_heads as usize, 0, m.world);
        // Reusing the source saves a full TP cache. Keep eight transfer slots
        // within that saving, bounded to 32 MiB total and the previous peak.
        let reuse_pair = m.world == 2 && first_heads.len() * d * 2 == stride
            && partition.range(0).end * stride >= count * first_heads.len() * d;
        let slot_count = if reuse_pair { 8 } else { 2 };
        let slot_bytes = if reuse_pair { (count * width * 8 * m.layers.len() / 128).clamp(64 * 1024, 4 * 1024 * 1024) }
            else { 3 * 1024 * 1024 };
        let piece_rows = (slot_bytes / (first_heads.len() * d * 8)).max(1);
        #[derive(Clone, Copy)]
        struct Piece { layer: usize, target: usize, start: usize, rows: usize, width: usize }
        let mut pieces = Vec::new();
        for layer in 0..m.layers.len() {
            for target in 0..m.world - 1 {
                let heads = rows(m.config.kv_heads as usize, target, m.world);
                let shard_width = heads.len() * d;
                if shard_width == 0 { continue; }
                let first = partition.range(target).end;
                for start in (first..count).step_by(piece_rows) {
                    pieces.push(Piece { layer, target, start, rows: (count - start).min(piece_rows), width: shard_width });
                }
            }
        }
        let transferred = pieces.iter().map(|p| p.rows * p.width * 8).sum();
        let (mut prefix_elapsed,mut prepare_elapsed,mut install_elapsed)=(Duration::ZERO,Duration::ZERO,Duration::ZERO);
        let copy_prefix = |this: &mut Self, source: &mut spark_engine::qwen::Session<'_, 'a, '_>| -> Result<Duration> {
            let start=Instant::now();
            if source.len() != prefix { return Err("sequence hand-off causal prefix mismatch".into()); }
            if reuse_pair && width > 0
                && source.sequence_device_kv(0, 0, prefix)?.0.len() >= count * width {
                this.kv = source.take_sequence_device_kv()?.into_iter().map(|(keys, values)| Kv { keys, values }).collect();
                return Ok(start.elapsed());
            }
            // Fit the whole context inside the VMM page bound of the next
            // decode row; one arena avoids per-layer allocation/map calls.
            let page=2*1024*1024;
            let bytes=count*width*4;
            if this.kv.is_empty() && bytes>=page && m.max_tokens*width*4<=(bytes+width*4).div_ceil(page)*page {
                let mut buffers=b.vector_arena(&vec![m.max_tokens*width;m.layers.len()*2])?.into_iter();
                this.kv=(0..m.layers.len()).map(|_|Kv{keys:buffers.next().unwrap(),values:buffers.next().unwrap()}).collect();
                this.peak_bytes=this.peak_bytes.max(b.allocated_bytes());
            }else{this.reserve(count)?;}
            if width > 0 {
                for layer in 0..m.layers.len() {
                    let (keys, values) = source.sequence_device_kv(layer, 0, prefix)?;
                    b.copy_strided(keys, &mut this.kv[layer].keys, local_heads.start * d, 0, stride, width, width, prefix)?;
                    b.copy_strided(values, &mut this.kv[layer].values, local_heads.start * d, 0, stride, width, width, prefix)?;
                }
            }
            Ok(start.elapsed())
        };
        if pieces.is_empty() {
            finish(source)?;
            prefix_elapsed+=copy_prefix(self, source)?;
        } else {
            let mut prepare = |this: &Self, source: &spark_engine::qwen::Session<'_, '_, '_>, adopted: bool, piece: Piece, packed: &mut DeviceVector<'_>| -> Result<()> {
                let start=Instant::now();
                if m.rank + 1 == m.world {
                    let (keys, values) = if adopted {
                        (&this.kv[piece.layer].keys, &this.kv[piece.layer].values)
                    } else { source.sequence_device_kv(piece.layer, piece.start, piece.rows)? };
                    let n = piece.rows * piece.width;
                    let heads = rows(m.config.kv_heads as usize, piece.target, m.world);
                    let offset = piece.start * stride + heads.start * d;
                    b.copy_strided(keys, packed, offset, 0, stride, piece.width, piece.width, piece.rows)?;
                    b.copy_strided(values, packed, offset, n, stride, piece.width, piece.width, piece.rows)?;
                }
                // Complete packing or the previous cache install before the
                // transport can read or overwrite this mapped slot.
                b.synchronize()?;prepare_elapsed+=start.elapsed();Ok(())
            };
            let elements = piece_rows * first_heads.len() * d * 2;
            let ring_ref = self.ring;
            let mut ring = ring_ref.borrow_mut();
            b.with_exchange_slots(elements, slot_count, |slots, host| {
                self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
                let host = host as usize;
                let result = std::thread::scope(|scope| {
                    let (send, requests) = std::sync::mpsc::sync_channel::<(Piece, usize)>(slot_count);
                    let (returns, receive) = std::sync::mpsc::sync_channel(slot_count);
                    let transport = &mut *ring;
                    let worker = scope.spawn(move || {
                        // A spawned thread inherits the compute thread's mask.
                        // Give polling its own CPU so GPU submission can overlap.
                        if let Ok(cpu)=std::env::var("TP_CPU") {
                            crate::runtime::pin_cpu(cpu.parse::<usize>().map_err(|e|e.to_string())? ^ 1)?;
                        }
                        transport.with_transfer_regions(|ring| {
                        unsafe { ring.reserve_transfer_region(host as *mut u8, elements * 4 * slot_count)?; }
                        while let Ok((piece, slot)) = requests.recv() {
                            let bytes = unsafe { std::slice::from_raw_parts_mut((host as *mut u8).add(slot * elements * 4), piece.rows * piece.width * 8) };
                            let mut pieces = bytes.chunks_mut(1024 * 1024).collect::<Vec<_>>();
                            let result = ring.transfer_from_last_batch(piece.target, &mut pieces);
                            let failed = result.is_err();
                            if returns.send((piece, slot, result)).is_err() { break; }
                            if failed { break; }
                        }
                        Ok(())
                        })
                    });
                    let result: Result<()> = (|| {
                        prepare(self, source, false, pieces[0], &mut slots[0])?;
                        send.send((pieces[0], 0)).map_err(|_| "sequence hand-off worker stopped")?;
                        finish(source)?;
                        // The first token precedes cache adoption/compaction or
                        // the general ring's destination allocation and copies.
                        prefix_elapsed+=copy_prefix(self, source)?;
                        let adopted=source.is_empty();
                        let (mut next, mut pending) = (1, 1);
                        let mut available=(1..slot_count).collect::<std::collections::VecDeque<_>>();
                        loop {
                            while next < pieces.len() && !available.is_empty() {
                                let slot = available.pop_front().unwrap();
                                prepare(self, source, adopted, pieces[next], &mut slots[slot])?;
                                send.send((pieces[next], slot)).map_err(|_| "sequence hand-off worker stopped")?;
                                next += 1; pending += 1;
                            }
                            let (piece, slot, completed) = receive.recv().map_err(|_| "sequence hand-off worker stopped")?;
                            completed?;
                            pending -= 1;
                            // Compact one completed layer while queued slots
                            // transfer. The sender has packed its final piece;
                            // the receiver compacts before installing any suffix.
                            if adopted && (m.rank == piece.target && piece.start == prefix
                                || m.rank + 1 == m.world && piece.start + piece.rows == count) {
                                let start=Instant::now();
                                let kv=&mut self.kv[piece.layer];
                                b.compact_heads(&mut kv.keys, local_heads.start*d, stride, width, prefix)?;
                                b.compact_heads(&mut kv.values, local_heads.start*d, stride, width, prefix)?;
                                prefix_elapsed+=start.elapsed();
                            }
                            if m.rank == piece.target {
                                let start=Instant::now();
                                let n = piece.rows * piece.width;
                                b.copy_vector_range(&slots[slot], &mut self.kv[piece.layer].keys, 0, piece.start * width, n)?;
                                b.copy_vector_range(&slots[slot], &mut self.kv[piece.layer].values, n, piece.start * width, n)?;
                                install_elapsed+=start.elapsed();
                            }
                            available.push_back(slot);
                            if pending == 0 && next == pieces.len() { break; }
                        }
                        Ok(())
                    })();
                    drop(send); drop(receive);
                    let transport = worker.join().unwrap_or_else(|_| Err("sequence hand-off worker panicked".into()));
                    result?;
                    transport
                });
                ring.clear_registered();
                result
            })?;
        }
        b.synchronize()?;
        self.ring.borrow_mut().clear_registered();
        // Every outgoing piece has completed. General rings no longer need the
        // source, and adopted pair caches now contain only packed local heads.
        let before = b.allocated_bytes();
        if !source.is_empty() { drop(source.take_sequence_device_kv()?); }
        let elements = (count + 1).min(m.max_tokens) * width;
        if elements != 0 {
            for kv in &mut self.kv {
                for vector in [&mut kv.keys, &mut kv.values] {
                    if vector.is_segmented() && vector.len() > elements { b.trim_vector(vector, elements)?; }
                    if vector.len() * 4 <= (elements * 4).div_ceil(2 * 1024 * 1024) * (2 * 1024 * 1024) { continue; }
                    // Replace one vector at a time to bound the transient peak.
                    let mut compact = b.segmented_vector(m.max_tokens * width, elements)?;
                    self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
                    b.copy_prefix(vector, &mut compact, count * width)?;
                    b.synchronize()?;
                    *vector = compact;
                }
            }
        }
        eprintln!("handoff_reclaim rank={} before_device_bytes={before} after_device_bytes={} kv_bytes={}",
            m.rank, b.allocated_bytes(), self.cache_bytes());
        self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
        self.pinned_peak = self.pinned_peak.max(b.peak_pinned_staging_bytes());
        self.len = count;
        self.invalid = false;
        eprintln!("handoff rank={} bytes={transferred} seconds={:.9} prefix_s={:.9} prepare_s={:.9} install_s={:.9} device_peak_bytes={} pinned_peak_bytes={}",
            m.rank, started.elapsed().as_secs_f64(),prefix_elapsed.as_secs_f64(),prepare_elapsed.as_secs_f64(),install_elapsed.as_secs_f64(), self.peak_bytes, self.pinned_peak);
        Ok(transferred)
    }
    fn reserve(&mut self,needed:usize)->Result<usize> {
        let m=self.model; let b=m.backend;
        let stride=m.local_kv_heads*m.config.head_dim as usize;
        let storage=if self.bf16 {2} else {1};
        // Grow arenas geometrically until page-backed vectors can grow in place.
        let arena=(needed*stride).div_ceil(storage)*4<2*1024*1024;
        let capacity=if needed<512||arena {needed.max(1).next_power_of_two()}else{needed}.min(m.max_tokens);
        if stride==0 {return Ok(capacity);}
        let elements=(capacity*stride).div_ceil(storage); let copy=(self.len*stride).div_ceil(storage);
        if self.kv.first().is_some_and(|kv|kv.keys.is_segmented())
            && self.kv.iter().any(|kv|!kv.keys.writable_from(copy)||!kv.values.writable_from(copy)) {
            let sources=self.kv.iter().flat_map(|kv|[&kv.keys,&kv.values]).collect::<Vec<_>>();
            let mut buffers=b.fork_vectors(&sources,copy,elements,(m.max_tokens*stride).div_ceil(storage))?.into_iter();
            self.kv=(0..self.kv.len()).map(|_|Kv{keys:buffers.next().unwrap(),values:buffers.next().unwrap()}).collect();
        }else if self.kv.first().is_some_and(|kv| kv.keys.is_segmented()) {
            for kv in &mut self.kv {b.grow_vector(&mut kv.keys,elements)?;b.grow_vector(&mut kv.values,elements)?;}
        } else if self.kv.first().is_none_or(|kv| kv.keys.len()<elements) {
            let mut buffers=if elements*4>=2*1024*1024 {
                let maximum=(m.max_tokens*stride).div_ceil(storage);
                (0..m.layers.len()*2).map(|_|b.segmented_vector(maximum,elements)).collect::<Result<Vec<_>>>()?
            }else{b.vector_arena(&vec![elements;m.layers.len()*2])?}.into_iter();
            let mut replacement=(0..m.layers.len()).map(|_|Kv{keys:buffers.next().unwrap(),values:buffers.next().unwrap()}).collect::<Vec<_>>();
            self.peak_bytes=self.peak_bytes.max(b.allocated_bytes());
            for(src,dst)in self.kv.iter().zip(&mut replacement){
                b.copy_prefix(&src.keys,&mut dst.keys,copy)?;b.copy_prefix(&src.values,&mut dst.values,copy)?;
            }
            self.kv=replacement;
        }
        self.peak_bytes=self.peak_bytes.max(b.allocated_bytes());
        Ok((self.kv[0].keys.len()*storage/stride).min(m.max_tokens))
    }
    fn run(&mut self,tokens:&[u32],output:Logits,exact:bool)->Result<Vec<Vec<f32>>> {
        if tokens.len()>512{return Err("batch accepts at most 512 rows".into());}
        self.handoff_sequence()?;
        if !self.buffered.is_empty(){return Err("sequence prompt must finish before decode".into());}
        self.run_to(tokens,output,exact,self.len+tokens.len())
    }
    fn run_to(&mut self,tokens:&[u32],output:Logits,exact:bool,prompt_end:usize)->Result<Vec<Vec<f32>>> {
        if self.invalid {return Err("TP session invalid after failed collective".into());}
        let m=self.model;
        if self.ring.borrow().rank()!=m.rank || self.ring.borrow().world()!=m.world {return Err("model/transport rank mismatch".into());}
        if tokens.is_empty() || tokens.len()>4096 || tokens.len()>m.max_tokens-self.len || tokens.iter().any(|&t|t as u64>=m.config.vocab_size) {return Err("batch outside vocabulary or capacity (maximum 4096 prefill rows)".into());}
        let result=self.run_inner(tokens,output,exact,prompt_end);
        if result.is_err() {self.invalid=true;}
        result
    }
    fn run_inner(&mut self,tokens:&[u32],output:Logits,exact:bool,prompt_end:usize)->Result<Vec<Vec<f32>>> {
        let t=tokens.len(); let capacity=self.reserve(self.len+t)?;
        let all_logits=matches!(output,Logits::All|Logits::Greedy);
        let m=self.model; let b=m.backend; let c=&m.config;
        // Prompt chunks keep one Spark's prefill order, also a chunk of 64 rows or fewer.
        let _order=b.prompt_order(!exact);
        let mut ring=self.ring.borrow_mut();
        let staging=t*c.hidden_size as usize*if m.world>2||(m.world>1&&t>16) {m.world}else{1};
        // Grow only the upload buffer, clearing its RDMA registration first.
        if !(m.world==2&&t>16)&&staging*4>b.pinned_upload_bytes() {
            ring.clear_registered();
            b.reserve_exchange(staging)?;
            self.pinned_peak=self.pinned_peak.max(b.peak_pinned_staging_bytes());
        }
        let decode=t==1 && exact;
        let profile=t>1&&m.profile;
        // Long BF16 prompt chunks on several ranks run as two halves (prefill_halves).
        // The decision must be the same on every rank: it changes the link step sequence.
        let split=(m.world>1&&self.bf16&&!exact&&!all_logits&&!profile&&t>=HALVES_ROWS&&m.every_rank_shares()).then_some(t/2/512*512);
        let rows=[split.unwrap_or(t),t-split.unwrap_or(t)];
        let len=self.len;
        let stale=|s:&State,n:usize| s.tokens!=n || s.capacity<len+t || !decode && s.prompt_end!=prompt_end || s.all_logits!=all_logits || s.exact!=exact;
        let slot=if decode {&mut self.decode} else {&mut self.batch};
        if slot.as_ref().is_none_or(|s|stale(s,rows[0])) {
            ring.clear_registered();
            *slot=None; *slot=Some(State::new(m,rows[0],capacity,all_logits,exact,self.bf16,prompt_end)?);
        }
        if split.is_none() { if self.second.is_some() { ring.clear_registered(); self.second=None; } }
        else if self.second.as_ref().is_none_or(|s|stale(s,rows[1])) {
            ring.clear_registered();
            self.second=None; self.second=Some(State::new(m,rows[1],capacity,all_logits,exact,self.bf16,prompt_end)?);
        }
        self.peak_bytes=self.peak_bytes.max(b.allocated_bytes());
        let slot=if decode {&mut self.decode} else {&mut self.batch};
        b.write_vector(&mut slot.as_mut().unwrap().hidden,&m.embeddings(&tokens[..rows[0]])?)?;
        if let Some(second)=self.second.as_mut() { b.write_vector(&mut second.hidden,&m.embeddings(&tokens[rows[0]..])?)?; }
        self.pinned_peak=self.pinned_peak.max(b.peak_pinned_staging_bytes());
        let eps=c.rms_norm_eps as f32;
        if let Some(second)=self.second.as_mut() {
            let first=self.batch.as_mut().unwrap();
            ring.with_transfer_regions(|ring|prefill_halves(m,ring,&mut self.kv,[first,second],len))?;
        } else {
        let s=slot.as_mut().unwrap();
        if profile {b.set_profiling(true);}
        let mut layers=|ring:&mut Ring|->Result<()> {
        for (i,layer) in m.layers.iter().enumerate() {
            let timing=if profile {b.synchronize()?;Some((Instant::now(),b.stats(),ring.elapsed))}else{None};
            forward_layer(m,ring,layer,m.layer_norms(i)?,m.layer_rope()?,
                self.kv.get_mut(i).map(|kv|(&mut kv.keys,&mut kv.values)),s,self.len,exact,self.bf16,i==0)?;
            if let Some((start,before,exchange))=timing {
                b.synchronize()?;
                let line=format!("tp_layer_profile rank={} layer={i} position={} rows={t} wall_ms={:.3} transport_ms={:.3} stats={:?}",
                    m.rank,self.len,start.elapsed().as_secs_f64()*1000.,
                    (ring.elapsed-exchange).as_secs_f64()*1000.,b.stats().since(before));
                eprintln!("{line}");
            }
        }
            Ok(())
        };
        if !decode && m.world>1 {
            ring.with_transfer_regions(layers)?;
        } else { layers(&mut ring)?; }
        if profile {b.set_profiling(false);}
        }
        if output==Logits::None {b.synchronize()?;self.len+=t;return Ok(Vec::new());}
        // The head reads the chunk's last row: the second half's, when split.
        let s=if let Some(second)=self.second.as_mut() {second} else if decode {self.decode.as_mut().unwrap()} else {self.batch.as_mut().unwrap()};
        let ht=if all_logits {t} else {1};
        if all_logits {b.copy_prefix(&s.hidden,&mut s.last,t*c.hidden_size as usize)?;}
        else {b.copy_row(&s.hidden,&mut s.last,s.tokens-1)?;}
        let norm = if let Some(full) = m.full { full.output_norm()? } else { m.norm.as_ref().ok_or("missing output norm")? };
        if ht==1 { b.rms_device(&s.last,norm,&mut s.final_input,&mut s.final_sums,eps)?; }
        else { b.rms_batch_device(&s.last,norm,&mut s.final_input,&mut s.final_sums,eps)?; }
        if let Some(head)=&m.head.matrix {
            b.matmul_device(head,&s.final_input,&mut s.logits,&mut s.weight_pack,&mut s.input_pack,ht,true)?;
            b.round_logits_bf16(&mut s.logits)?;
        }
        let output=if output==Logits::Greedy {
            let local=if m.head.count()>0 {b.argmax_batch_device(&s.logits,&mut s.arg_parts,&mut s.arg_result,1)?.pop().ok_or("missing TP argmax row")?} else {Ok((0,f32::NEG_INFINITY))};
            let (id,value)=candidate(local,m.head.first);
            let mut candidates=vec![0u8;m.world*8]; let at=m.rank*8;
            candidates[at..at+4].copy_from_slice(&id.to_le_bytes());
            candidates[at+4..at+8].copy_from_slice(&value.to_le_bytes());
            ring.all_gather(&mut candidates,8)?;
            let id=gathered_greedy(candidates.chunks_exact(8).map(|v|(u32::from_le_bytes(v[..4].try_into().unwrap()),f32::from_le_bytes(v[4..].try_into().unwrap()))),c.vocab_size as usize)?;
            vec![vec![id as f32]]
        } else {
            let local=if m.head.count()==0 {Vec::new()} else {b.read_vector_values(&s.logits)?};
            let gathered=ring.gather_batch(&local,c.vocab_size as usize,1,ht)?;
            finite(gathered)?;
            if self.collect_logits {gathered.chunks_exact(c.vocab_size as usize).map(<[f32]>::to_vec).collect()} else {Vec::new()}
        };
        b.synchronize()?;
        self.len+=t;
        Ok(output)
    }
}
impl spark_engine::draft::Target for Session<'_,'_,'_,'_> {
    fn checkpoint(&self)->usize {self.checkpoint()}
    fn truncate(&mut self,p:usize)->Result<()> {self.truncate(p)}
    fn step(&mut self,t:u32)->Result<Vec<f32>> {self.step(t)}
    fn step_greedy(&mut self,t:u32)->Result<u32> {self.step_greedy(t)}
    fn verify_batch(&mut self,t:&[u32])->Result<Vec<Vec<f32>>> {self.verify_batch(t)}
    fn forward_batch(&mut self,t:&[u32])->Result<Vec<Vec<f32>>> {self.forward_batch_fast(t)}
    fn kv_bf16(&self)->bool {self.kv_bf16()}
}
