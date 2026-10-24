//! Head-sharded indexed attention. Official BF16 weights, FP32 activations and
//! index history; output partials must be reduced before HC injection.
use super::{Weights, gdn::{bf16, launch, projection, projection_pair, Arg::{Ptr, U32, F32}}};
use engine_model::flash::{partition, plan::RankPlan};
use spark_gpu_probe::{driver::{Backend, DeviceVector}, kernel_params::{QsaAttention, QsaBatchRow, QsaIndex, QsaIndexPool, QSA_INDEX_QUERIES, QSA_INDEX_GROUPS, QSA_MAX_POSITIONS, QSA_SCORE_HEADS, QSA_SCORE_KEYS, QSA_VALUE_HEADS}};
use std::{marker::PhantomData, ops::Range};

type Result<T> = std::result::Result<T, String>;
const HIDDEN: usize = 2560;
const DIM: usize = 256;
const INDEX: usize = 640;
#[path = "../hybrid/attention.rs"]
pub mod attention;
/// Tensor-core QSA attention, an ORDER change (default on; VADE_QSA_TC=0 restores the scalar path), read once
/// as (score planes, value planes), 0 = main's FP32 entry: VADE_QSA_TC=1 is
/// (3, 3): FP32 operands split into three BF16 planes (six products, no
/// operand bit lost); s1 flips only the scores, v1 only the values (the
/// two-plane _tc2 entries are a benchmark-only screen, rejected). Row and batch
/// entries flip together, so alone == batched == cached == chunked and MTP
/// on == off hold with the switch on. The softmax is main's.
pub(crate) fn qsa_tc() -> (u8, u8) {
    let (s, v, _) = qsa_tc3();
    (s, v)
}
/// The switch with the index-score stage: VADE_QSA_TC=1i flips scores,
/// values and index scores (the selection then sees tensor-core index
/// scores: a quality check decides), i1 only the index scores.
pub(crate) fn qsa_tc3() -> (u8, u8, u8) {
    qsa_tc_switch().unwrap_or((0, 0, 0))
}
/// VADE_QSA_TC parsed once: default on; unset = 1
/// (scores + values on tensor cores, indexer off); VADE_QSA_TC=0 restores the scalar path.
/// 1, 1i, s1, v1, i1 as above.
/// Any other value is an error at model load (NextModel::load_with_mtp), when QSA
/// scratch is created (QsaScratch::allocate) or norms are gathered, never a silent off.
pub(crate) fn qsa_tc_switch() -> Result<(u8, u8, u8)> {
    static TC: std::sync::OnceLock<Result<(u8, u8, u8)>> = std::sync::OnceLock::new();
    TC.get_or_init(|| match std::env::var("VADE_QSA_TC") {
        Err(std::env::VarError::NotPresent) => Ok((3, 3, 0)),
        Ok(v) => match v.as_str() {
            "0" => Ok((0, 0, 0)), "1" => Ok((3, 3, 0)), "1i" => Ok((3, 3, 3)),
            "s1" => Ok((3, 0, 0)), "v1" => Ok((0, 3, 0)), "i1" => Ok((0, 0, 3)),
            _ => Err(format!("VADE_QSA_TC must be unset, 0, 1, 1i, s1, v1 or i1 (got {v:?})")),
        },
        Err(e) => Err(format!("VADE_QSA_TC: {e}")),
    }).clone()
}
/// Score entry for the switch, and its grid-y: the tensor-core entries take
/// slot ranges of 16 * ceil(ceil(2051 / y) / 16), which only distribute work.
fn score_entry(batch: bool, tokens: usize, planes: usize) -> (&'static str, usize) {
    match (qsa_tc().0, batch) {
        // Prompt rows: warp-owned 8-slot tiles, no barrier in the loop (same bytes).
        (3, false) if tokens >= 64 => ("flash_qsa_attention_scores_tc_w", 8),
        (3, false) => ("flash_qsa_attention_scores_tc", planes),
        (3, true) => ("flash_qsa_attention_scores_tc_batch", planes),
        // Prompt tiles: thread-owned scores, main's lane tree without shuffles (same bytes).
        (_, false) if tokens >= 64 => ("flash_qsa_attention_scores_t", 8),
        (_, false) => ("flash_qsa_attention_scores", planes),
        (_, true) => ("flash_qsa_attention_scores_batch", planes),
    }
}
/// Value entry for the switch: (name, blocks per token, threads). Main's entry
/// takes four 64-feature slices per KV head; the tensor-core one a whole head.
fn value_entry(batch: bool, tokens: usize) -> (&'static str, usize, u32) {
    let main = (QSA_VALUE_HEADS * 64) as u32;
    match (qsa_tc().1, batch) {
        // Under 64 tokens: four 64-feature slices per KV head, same arithmetic.
        (3, false) if tokens >= 64 => ("flash_qsa_attention_values_tc", 1, 128),
        (3, false) => ("flash_qsa_attention_values_tc_m", 4, 128),
        (3, true) => ("flash_qsa_attention_values_tc_m_batch", 4, 128),
        // Prompt tiles: one block per token and KV head, four adjacent features
        // per thread (one 16-byte V load feeds them), 12-slot tiles, same bytes.
        (_, false) if tokens >= 64 => ("flash_qsa_attention_values_wide4t", 1, 192),
        // One-token rows (decode, batch rows): 16-feature slices, 16 blocks of
        // 192 threads per KV head, the same arithmetic as main's entry.
        (_, false) if tokens == 1 => ("flash_qsa_attention_values_s16", 16, 192),
        (_, false) => ("flash_qsa_attention_values", 4, main),
        (_, true) => ("flash_qsa_attention_values_s16_batch", 16, 192),
    }
}
// Only newly completed groups need pooling, including after a rewind.
fn index_pool_range(past: usize, tokens: usize) -> Range<usize> { past / 4..(past + tokens) / 4 }
fn index_score_grid(tokens: usize, blocks: usize) -> usize {
    let groups = if tokens >= QSA_INDEX_QUERIES { blocks.div_ceil(QSA_INDEX_GROUPS) } else { blocks };
    (tokens.div_ceil(QSA_INDEX_QUERIES) * groups).div_ceil(4)
}
// Cursor-backed graphs launch capacity-bounded grids; inactive warps return
// before reads in index_scores/attention_scores. Eager calls pass their length.
pub(super) fn decode_shape(past: usize, tokens: usize, capacity: usize) -> [usize; 3] {
    let blocks = (past + tokens) / 4;
    [index_pool_range(past, tokens).len(), if blocks > 512 { index_score_grid(tokens, capacity / 4) } else { 0 },
        capacity.min(QSA_MAX_POSITIONS).div_ceil(QSA_SCORE_KEYS).div_ceil(4)]
}
const NORMS: [(&str, usize); 4] = [("q_norm.weight", 256), ("k_norm.weight", 256),
    ("indexer.q_layernorm.weight", 128), ("indexer.k_layernorm.weight", 128)];

/// Projection rows and stable descriptors for independent decode caches.
/// Histories and attention scratch remain session-owned.
pub(super) struct QsaBatch<'a> {
    qg: DeviceVector<'a>,
    k: DeviceVector<'a>,
    v: DeviceVector<'a>,
    index: DeviceVector<'a>,
    gated: DeviceVector<'a>,
    sessions: DeviceVector<'a>,
    shape: [usize; 2],
    bytes: usize,
    prompt: Option<[u64; 4]>,
}
impl<'a> QsaBatch<'a> {
    fn lengths(plan: &RankPlan, rows: usize) -> [usize; 6] {
        let heads = (plan.query_heads.end - plan.query_heads.start) as usize;
        let keys = (plan.kv_heads.end - plan.kv_heads.start) as usize;
        [heads * 512 * rows, keys * DIM * rows, keys * DIM * rows,
            INDEX * rows, heads * DIM * rows, rows * std::mem::size_of::<QsaBatchRow>() / 4].map(|n| n.max(1))
    }
    pub(super) fn new(b: &Backend<'a>, plan: &RankPlan, rows: usize) -> Result<Self> {
        let lengths = Self::lengths(plan, rows);
        let bytes = lengths.iter().map(|n| (n * 4).div_ceil(64) * 64).sum();
        let mut v = b.vector_arena(&lengths)?.into_iter();
        Ok(Self { qg: v.next().unwrap(), k: v.next().unwrap(), v: v.next().unwrap(),
            index: v.next().unwrap(), gated: v.next().unwrap(), sessions: v.next().unwrap(), shape: [0; 2], bytes, prompt: None })
    }
    /// Prompt projections borrow idle MLP partials; the descriptor view is unused.
    /// Every view starts on a 64-byte boundary and retains the arena allocation.
    /// Borrowed views add no resident bytes beyond the owning MLP workspace.
    pub(super) fn workspace(plan: &RankPlan, rows: usize, arena: &DeviceVector<'a>) -> Result<Self> {
        if rows == 0 || rows > crate::flash_next_rows::prompt_tile().max(4096) { return Err("invalid QSA prompt group".into()); }
        let mut offset = 0;
        let mut lengths = Self::lengths(plan, rows);
        lengths[5] = arena.len().checked_sub(lengths[..5].iter().map(|n| n.div_ceil(16) * 16).sum::<usize>())
            .ok_or("QSA projections exceed idle arena")?;
        let [qg, k, v, index, gated, sessions] = lengths.map(|len| {
            let start = offset;
            offset += len.div_ceil(16) * 16;
            arena.slice(start..start + len)
        });
        Ok(Self { qg: qg?, k: k?, v: v?, index: index?, gated: gated?, sessions: sessions?, shape: [0; 2], bytes: 0, prompt: None })
    }
    pub(super) fn pool_project(&mut self,b: &Backend<'_>,proj: [u64;5],input: &DeviceVector<'_>,count: usize) -> Result<()> {
        self.index.resize_within_capacity(count*INDEX)?;
        // One grouped launch at decode (each output keeps its solo K16 graph and merge).
        super::blocks::next_rows_group(b,input.device_address(),HIDDEN,count,[(proj[0],self.qg.device_address(),12288),
            (proj[1],self.k.device_address(),512),(proj[2],self.v.device_address(),512),(proj[3],self.index.device_address(),INDEX)])
    }
    pub(super) fn pool_attend(&mut self,b: &Backend<'_>,plan: &RankPlan,templates: &DeviceVector<'_>,
        norm: u64,capacity: usize,mut step: spark_gpu_probe::next_math::SlotStep) -> Result<()> {
        let (mut source,mut out,mut norm)=(templates.device_address(),self.sessions.device_address(),norm);
        unsafe { b.launch_flash("flash_next_qsa_slot",1,128,&mut [(&mut source as *mut u64).cast(),(&mut out as *mut u64).cast(),
            (&mut norm as *mut u64).cast(),(&mut step as *mut spark_gpu_probe::next_math::SlotStep).cast()])?; }
        self.shape=[capacity.min(QSA_MAX_POSITIONS),if capacity/4>512 { (capacity/4).next_power_of_two() } else { 0 }];
        // Inactive descriptors return before touching KV. Clear their pooled
        // output so the common O projection cannot consume an old occupant.
        b.zero_vector(&mut self.gated)?;
        Qsa::attend_pool(b,plan,self,step.rows as usize)
    }
    pub(super) fn pool_finish(&self,b: &Backend<'_>,weight: u64,output: &DeviceVector<'_>,count: usize) -> Result<()> {
        super::blocks::next_rows(b,weight,self.gated.device_address(),output.device_address(),HIDDEN,6144,count)
    }
    /// Cohort maxima bound launches only; descriptors retain each row's exact
    /// past and score stride. Padding returns before reading the table.
    pub(super) fn upload_sessions(&mut self, b: &Backend<'_>, rows: &[QsaBatchRow]) -> Result<[usize; 2]> {
        if rows.is_empty() || std::mem::size_of_val(rows) > self.sessions.len() * 4 {
            return Err("invalid Flash QSA descriptor count".into());
        }
        let length = rows.iter().map(|r| r.past as usize + 1).max().unwrap();
        let blocks = length / 4;
        let shape = [length.min(QSA_MAX_POSITIONS).next_power_of_two().min(QSA_MAX_POSITIONS),
            if blocks > 512 { blocks.next_power_of_two() } else { 0 }];
        let bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) };
        b.write_vector_bytes(&mut self.sessions, bytes)?;
        self.shape = shape;
        Ok(shape)
    }
    pub(super) fn device_bytes(&self) -> usize {
        self.bytes
    }
}

/// First row's past when every descriptor names the same K/V cache and the
/// pasts run consecutively, as a prompt chunk's rows do.
pub(crate) fn prompt_past(rows: &[f32]) -> Option<usize> {
    let first = rows.get(..6)?;
    rows.chunks_exact(6).enumerate().all(|(i, r)| r[..4].iter().zip(first).all(|(a, b)| a.to_bits() == b.to_bits())
        && first[4].to_bits().checked_add(i as u32) == Some(r[4].to_bits()))
        .then_some(first[4].to_bits() as usize)
}

/// Full gated attention uses the same prepare/softmax/gate kernels as QSA,
/// with no index mask and BF16 KV. Only the active layer needs this scratch.
pub struct DenseScratch<'a> {
    pub(crate) qg: DeviceVector<'a>,
    pub(crate) k: DeviceVector<'a>,
    pub(crate) v: DeviceVector<'a>,
    q: DeviceVector<'a>,
    pub(crate) gated: DeviceVector<'a>,
    decode_scores: Option<DeviceVector<'a>>,
    score_stride: usize,
    score_length: usize,
    bytes: usize,
}
pub struct DenseState<'a> {
    keys: DeviceVector<'a>,
    values: DeviceVector<'a>,
}
impl<'a> DenseState<'a> {
    pub(crate) fn new(b: &Backend<'a>, c: &engine_model::qwen3_5::Config, capacity: usize) -> Result<Self> {
        let words = capacity * (c.kv_heads * c.head_dim) as usize / 2;
        let mut v = b.vector_arena(&[words, words])?.into_iter();
        Ok(Self { keys: v.next().unwrap(), values: v.next().unwrap() })
    }
    /// Hybrid target KV uses the existing GB10 virtual pages once each vector
    /// reaches a page. Complete allocations alias; the writable tail is private.
    pub(crate) fn fork_layers(b: &Backend<'a>, c: &engine_model::qwen3_5::Config,
        source: &[Option<Self>], past: usize, capacity: usize, maximum: usize) -> Result<Vec<Option<Self>>> {
        let stride = (c.kv_heads * c.head_dim) as usize / 2;
        let sources: Vec<_> = source.iter().flatten().flat_map(|s| [&s.keys, &s.values]).collect();
        let words = capacity * stride;
        let count = c.layers.iter().filter(|&&kind| kind == engine_model::flash::LayerKind::SparseAttention).count() * 2;
        if count == 0 { return Ok(c.layers.iter().map(|_| None).collect()); }
        let vectors = if words * 4 < 2 * 1024 * 1024 {
            let mut copies = b.vector_arena(&vec![words; count])?;
            for (source, copy) in sources.iter().zip(&mut copies) { b.copy_prefix(source, copy, past * stride)?; }
            copies
        } else if sources.is_empty() {
            (0..count).map(|_| b.segmented_vector(maximum * stride, words)).collect::<Result<Vec<_>>>()?
        } else { b.fork_vectors(&sources, past * stride, words, maximum * stride)? };
        let mut vectors = vectors.into_iter();
        Ok(c.layers.iter().map(|kind| if *kind == engine_model::flash::LayerKind::SparseAttention {
            Some(Self { keys: vectors.next().unwrap(), values: vectors.next().unwrap() })
        } else { None }).collect())
    }
    pub(crate) fn writable_from(&self, c: &engine_model::qwen3_5::Config, past: usize) -> bool {
        let words = past * (c.kv_heads * c.head_dim) as usize / 2;
        self.keys.writable_from(words) && self.values.writable_from(words)
    }
    pub(crate) fn paged(&self) -> bool { self.keys.is_segmented() && self.values.is_segmented() }
    pub(crate) fn grow(&mut self, b: &Backend<'a>, c: &engine_model::qwen3_5::Config, capacity: usize) -> Result<()> {
        let words = capacity * (c.kv_heads * c.head_dim) as usize / 2;
        b.grow_vector(&mut self.keys, words)?;
        b.grow_vector(&mut self.values, words)
    }
    pub(crate) fn allocations(&self, out: &mut Vec<(usize, usize)>) {
        self.keys.allocations(out); self.values.allocations(out);
    }
    pub(crate) fn prefix_allocations(&self, c: &engine_model::qwen3_5::Config, tokens: usize, out: &mut Vec<(usize, usize)>) {
        let words = tokens * (c.kv_heads * c.head_dim) as usize / 2;
        self.keys.prefix_allocations(words, out); self.values.prefix_allocations(words, out);
    }
    pub fn capacity(&self, c: &engine_model::qwen3_5::Config) -> usize {
        self.keys.len() * 2 / (c.kv_heads * c.head_dim) as usize
    }
    pub fn device_bytes(&self) -> usize { (self.keys.len() + self.values.len()) * 4 }
    /// The one-layer MTP prompt only needs K/V; no query or attention output
    /// participates in the next position's state. Reuse the canonical prepare.
    pub(crate) fn append(&mut self, b: &Backend<'_>, c: &engine_model::qwen3_5::Config,
        k: &DeviceVector<'_>, v: &DeviceVector<'_>, norm: u64, frequencies: u64,
        tokens: usize, past: usize) -> Result<()> {
        self.append_replaying(b,c,k,v,norm,frequencies,tokens,past,0)
    }
    pub(crate) fn append_replaying(&mut self, b: &Backend<'_>, c: &engine_model::qwen3_5::Config,
        k: &DeviceVector<'_>, v: &DeviceVector<'_>, norm: u64, frequencies: u64,
        tokens: usize, past: usize, replay_position: u64) -> Result<()> {
        if past.checked_add(tokens).is_none_or(|n| n > self.capacity(c))
            || k.len() != tokens * (c.kv_heads * c.head_dim) as usize || v.len() != k.len() {
            return Err("MTP KV insertion exceeds cache or projection shape".into());
        }
        let mut p = spark_gpu_probe::kernel_params::QsaPrepare {
            query_gate: 0, key: k.device_address(), value: v.device_address(),
            query_weight: 0, key_weight: norm, query: 0,
            keys: self.keys.device_address(), values: self.values.device_address(),
            heads: 0, kv_heads: c.kv_heads as u32, tokens: tokens as u32, past: if replay_position == 0 { past as u32 } else { 0 },
            position_offset: 0, frequencies, precision: 1, epsilon: c.epsilon as f32, replay_position,
        };
        b.profile_phase(2, || unsafe { b.launch_flash_shared("flash_qsa_prepare",
            (tokens * c.kv_heads as usize * 32).div_ceil(128) as u32, 128, 0,
            &mut [(&mut p.query_gate as *mut u64).cast(), (&mut p.key as *mut u64).cast(),
                (&mut p.value as *mut u64).cast(), (&mut p.query_weight as *mut u64).cast(),
                (&mut p.key_weight as *mut u64).cast(), (&mut p.query as *mut u64).cast(),
                (&mut p.keys as *mut u64).cast(), (&mut p.values as *mut u64).cast(),
                (&mut p.heads as *mut u32).cast(), (&mut p.kv_heads as *mut u32).cast(),
                (&mut p.tokens as *mut u32).cast(), (&mut p.past as *mut u32).cast(),
                (&mut p.position_offset as *mut u32).cast(), (&mut p.frequencies as *mut u64).cast(),
                (&mut p.precision as *mut u32).cast(), (&mut p.epsilon as *mut f32).cast(),
                (&mut p.replay_position as *mut u64).cast()]) })
    }
    pub(crate) fn descriptor(&self, past: usize) -> [f32; 6] {
        let (k, v) = (self.keys.device_address(), self.values.device_address());
        [k as u32, (k >> 32) as u32, v as u32, (v >> 32) as u32, past as u32, 0]
            .map(f32::from_bits)
    }
    pub(crate) fn copy_prefix(&self, b: &Backend<'a>, c: &engine_model::qwen3_5::Config,
        next: &mut Self, past: usize) -> Result<()> {
        let words = past * (c.kv_heads * c.head_dim) as usize / 2;
        b.copy_vector_range(&self.keys, &mut next.keys, 0, 0, words)?;
        b.copy_vector_range(&self.values, &mut next.values, 0, 0, words)?;
        Ok(())
    }
}
impl<'a> DenseScratch<'a> {
    fn lengths(c: &engine_model::qwen3_5::Config, tile: usize) -> [usize; 5] {
        let query = tile * c.query_heads as usize * c.head_dim as usize;
        let key = tile * c.kv_heads as usize * c.head_dim as usize;
        [2 * query, key, key, query, query]
    }
    pub fn reservation_bytes(c: &engine_model::qwen3_5::Config, tile: usize) -> usize {
        Self::lengths(c, tile).iter().map(|n| (n * 4).div_ceil(64) * 64).sum()
    }
    pub fn new(b: &Backend<'a>, c: &engine_model::qwen3_5::Config, tile: usize) -> Result<Self> {
        let lengths = Self::lengths(c, tile);
        let bytes = Self::reservation_bytes(c, tile);
        let mut v = b.vector_arena(&lengths)?.into_iter();
        Ok(Self { qg: v.next().unwrap(), k: v.next().unwrap(), v: v.next().unwrap(),
            q: v.next().unwrap(), gated: v.next().unwrap(), decode_scores: None, score_stride: 0, score_length: 0, bytes })
    }
    pub fn device_bytes(&self) -> usize { self.bytes + self.decode_scores.as_ref().map_or(0, |v| v.len() * 4) }
    pub(crate) fn reserve_decode_scores(&mut self, b: &Backend<'a>, heads: usize,
        rows: usize, length: usize, capacity: usize) -> Result<bool> {
        let stride = if length <= 512 { 0 } else { length.next_power_of_two().min(capacity) };
        let changed = self.score_length != stride;
        self.score_length = stride;
        if stride == 0 { return Ok(changed); }
        // Kernels lay rows out with the kept stride, so a kept buffer must hold rows at that stride.
        if self.score_stride >= stride && self.decode_scores.as_ref().is_some_and(|v| v.len() >= rows * heads * self.score_stride) {
            return Ok(changed);
        }
        self.decode_scores = Some(b.vector(rows * heads * stride)?);
        self.score_stride = stride;
        Ok(true)
    }
    pub fn resize(&mut self, c: &engine_model::qwen3_5::Config, tokens: usize) -> Result<()> {
        let query = tokens * c.query_heads as usize * c.head_dim as usize;
        let key = tokens * c.kv_heads as usize * c.head_dim as usize;
        self.qg.resize_within_capacity(2 * query)?;
        self.k.resize_within_capacity(key)?;
        self.v.resize_within_capacity(key)?;
        self.q.resize_within_capacity(query)?;
        self.gated.resize_within_capacity(query)
    }
    pub fn attend(&mut self, b: &Backend<'a>, c: &engine_model::qwen3_5::Config, state: &mut DenseState<'_>,
        norms: [u64; 2], frequencies: u64, tokens: usize, past: usize, packed_output: u64) -> Result<()> {
        let _class = b.prefill_class(2);
        let (heads, keys) = (c.query_heads as u32, c.kv_heads as u32);
        if tokens == 0 || c.head_dim != 256 || keys == 0 || heads == 0 || heads % keys != 0
            || heads / keys > spark_gpu_probe::kernel_params::DENSE_VALUE_GROUP
            || self.qg.len() != tokens * heads as usize * 512 || self.k.len() != tokens * keys as usize * 256
            || self.v.len() != self.k.len() || self.q.len() != tokens * heads as usize * 256 || self.gated.len() != self.q.len()
            || past.checked_add(tokens).is_none_or(|n| n > state.capacity(c)) { return Err("invalid causal attention extent".into()); }
        b.generic_read_bytes(|| (2, 4 * (past + tokens) * (c.kv_heads * c.head_dim) as usize));
        let (past, replay_position) = b.decode_position(past);
        b.profile_phase(2, || launch(b, "flash_qsa_prepare", (heads + keys) as usize * tokens * 32,
            &mut [Ptr(self.qg.device_address()), Ptr(self.k.device_address()), Ptr(self.v.device_address()),
                Ptr(norms[0]), Ptr(norms[1]), Ptr(self.q.device_address()), Ptr(state.keys.device_address()), Ptr(state.values.device_address()),
                U32(heads), U32(keys), U32(tokens as u32), U32(past), U32(0), Ptr(frequencies), U32(1), F32(c.epsilon as f32), Ptr(replay_position)]))?;
        // Only untyped hybrid callers enter this legacy prompt path.
        if tokens > 1 {
            b.prefill_attention_bf16_dim_device(&self.q, &state.keys, &state.values, &mut self.gated,
                heads as usize, keys as usize, c.head_dim as usize, past as usize, tokens)?;
            return b.profile_phase(6, || {
                let mut params = spark_gpu_probe::kernel_params::QsaGate {
                    query_gate: self.qg.device_address() as _, attention: self.gated.device_address() as _,
                    packed: packed_output as _, width: heads * c.head_dim as u32, tokens: tokens as u32,
                };
                unsafe { b.launch_flash_shared("flash_qsa_gate", self.gated.len().div_ceil(128) as u32, 128, 0,
                    &mut [(&mut params as *mut spark_gpu_probe::kernel_params::QsaGate).cast()]) }
            });
        }
        self.decode_attention(b, heads, keys, 1, 0, state.keys.device_address(), state.values.device_address(), past, replay_position, 0)
    }
    pub(crate) fn attend_batch(&mut self, b: &Backend<'_>, c: &engine_model::qwen3_5::Config,
        norms: [u64; 2], frequencies: u64, descriptors: &DeviceVector<'_>, count: usize,
        replay_position: u64) -> Result<()> {
        let _class = b.prefill_class(2);
        self.prepare_rows(b, c, norms, frequencies, descriptors, count, replay_position)?;
        self.decode_attention(b, c.query_heads as u32, c.kv_heads as u32, count, descriptors.device_address(), 0, 0, 0, replay_position, 0)
    }
    /// Consecutive prompt rows of one cache (see prompt_past) whose rows all
    /// pass 512 positions take the tiled exact kernels; others stay staged.
    pub(crate) fn attend_prompt(&mut self, b: &Backend<'_>, c: &engine_model::qwen3_5::Config,
        norms: [u64; 2], frequencies: u64, descriptors: &DeviceVector<'_>, count: usize, past: usize) -> Result<()> {
        let (heads, keys) = (c.query_heads as u32, c.kv_heads as u32);
        let Some(scores) = self.decode_scores.as_ref().map(|v| v.device_address()).filter(|_| self.score_length >= past + count && past >= 512
            && keys != 0 && heads != 0 && heads % keys == 0 && (heads / keys) % 8 == 0 && c.head_dim == 256) else {
            return self.attend_batch(b, c, norms, frequencies, descriptors, count, 0);
        };
        let _class = b.prefill_class(2);
        self.prepare_rows(b, c, norms, frequencies, descriptors, count, 0)?;
        let mut params = spark_gpu_probe::kernel_params::DenseAttention {
            query: self.q.device_address(), query_gate: self.qg.device_address(), output: self.gated.device_address(),
            rows: descriptors.device_address(), scores, replay_position: 0, keys: 0, values: 0,
            query_heads: heads, key_heads: keys, count: count as u32, score_stride: self.score_stride as u32,
            score_tiles: 0, past: 0,
        };
        let p = (&mut params as *mut spark_gpu_probe::kernel_params::DenseAttention).cast();
        b.profile_phase(3, || unsafe {
            b.launch_flash_rows("flash_dense_prompt_scores", count.div_ceil(8) as u32 * (heads / 8), (past + count).div_ceil(1024) as u32, 256, 98304, &mut [p])?;
            b.launch_flash_shared("flash_dense_prompt_values", count.div_ceil(4) as u32 * (heads / 8), 128, 20480, &mut [p])
        })
    }
    fn prepare_rows(&mut self, b: &Backend<'_>, c: &engine_model::qwen3_5::Config,
        norms: [u64; 2], frequencies: u64, descriptors: &DeviceVector<'_>, count: usize,
        replay_position: u64) -> Result<()> {
        b.generic_byte_probe(2, || Ok(b.read_vector_values(descriptors)?.chunks_exact(6).take(count)
            .map(|r| 4 * (r[4].to_bits() as usize + 1) * (c.kv_heads * c.head_dim) as usize).sum()))?;
        let (heads, keys) = (c.query_heads as u32, c.kv_heads as u32);
        launch(b, "flash_dense_prepare_batch", (heads + keys) as usize * count * 32,
            &mut [Ptr(self.qg.device_address()), Ptr(self.k.device_address()), Ptr(self.v.device_address()),
                Ptr(norms[0]), Ptr(norms[1]), Ptr(self.q.device_address()), Ptr(descriptors.device_address()),
                U32(heads), U32(keys), U32(count as u32), Ptr(frequencies), F32(c.epsilon as f32), Ptr(replay_position)])
    }
    fn decode_attention(&self, b: &Backend<'_>, heads: u32, keys: u32, count: usize,
        rows: u64, key_cache: u64, value_cache: u64, past: u32, replay_position: u64, token: usize) -> Result<()> {
        let mut params = spark_gpu_probe::kernel_params::DenseAttention {
            query: self.q.device_address() + (token * heads as usize * 256 * 4) as u64,
            query_gate: self.qg.device_address() + (token * heads as usize * 512 * 4) as u64,
            output: self.gated.device_address() + (token * heads as usize * 256 * 4) as u64,
            rows, scores: if self.score_length == 0 { 0 }
                else { self.decode_scores.as_ref().map_or(0, |v| v.device_address()) },
            keys: key_cache, values: value_cache, past,
            replay_position, query_heads: heads, key_heads: keys, count: count as u32, score_stride: self.score_stride as u32,
            score_tiles: self.score_length.div_ceil(32) as u32,
        };
        launch_dense_rows(b, &mut params)
    }
}

/// One non-indexed causal row for prompt, decode, replay and verification.
/// # Safety
/// All addresses must retain correctly sized, non-overlapping allocations in
/// this backend's context through stream completion; KV and norms are BF16.
/// Pass descriptors constructed by FullAttentionBinding::row without mutation.
pub unsafe fn causal_row(b: &Backend<'_>, prepare: spark_gpu_probe::kernel_params::QsaPrepare,
    mut row: spark_gpu_probe::kernel_params::DenseAttention) -> Result<()> {
    if row.key_heads == 0 || row.query_heads == 0 || row.query_heads % row.key_heads != 0
        || row.query_heads / row.key_heads > spark_gpu_probe::kernel_params::DENSE_VALUE_GROUP
        || prepare.tokens != 1 || row.count != 1 || row.rows != 0 || prepare.precision != 1
        || prepare.replay_position != 0 || row.replay_position != 0
        || prepare.heads != row.query_heads || prepare.kv_heads != row.key_heads
        || prepare.past != row.past || prepare.query != row.query || prepare.query_gate != row.query_gate
        || prepare.keys != row.keys || prepare.values != row.values {
        return Err("causal attention row descriptors disagree".into());
    }
    launch(b, "flash_qsa_prepare", (prepare.heads + prepare.kv_heads) as usize * 32,
        &mut [Ptr(prepare.query_gate), Ptr(prepare.key), Ptr(prepare.value), Ptr(prepare.query_weight),
            Ptr(prepare.key_weight), Ptr(prepare.query), Ptr(prepare.keys), Ptr(prepare.values),
            U32(prepare.heads), U32(prepare.kv_heads), U32(1), U32(prepare.past), U32(prepare.position_offset),
            Ptr(prepare.frequencies), U32(1), F32(prepare.epsilon), Ptr(0)])?;
    launch_dense_rows(b, &mut row)
}

fn launch_dense_rows(b: &Backend<'_>, params: &mut spark_gpu_probe::kernel_params::DenseAttention) -> Result<()> {
        let (heads, keys, count) = (params.query_heads, params.key_heads, params.count as usize);
        let (long, tiles) = (params.scores != 0, params.score_tiles);
        if long && heads / keys > spark_gpu_probe::kernel_params::DENSE_VALUE_GROUP {
            return Err(format!("dense decode attention supports at most {} query heads per key head",
                spark_gpu_probe::kernel_params::DENSE_VALUE_GROUP));
        }
        let p = (params as *mut spark_gpu_probe::kernel_params::DenseAttention).cast();
        b.profile_phase(3, || unsafe {
            if long { b.launch_flash_shared("flash_dense_scores", keys * count as u32 * tiles, 128, 0, &mut [p])?; }
            b.launch_flash_shared("flash_dense_attention_batch", (heads as usize * count).div_ceil(4) as u32, 128, 0, &mut [p])?;
            // Long rows: one block per key head and 64 features sums the values of its query group.
            if long { b.launch_flash_shared("flash_dense_values", keys * count as u32 * 4, heads / keys * 64, 0, &mut [p])?; }
            Ok(())
        })
}

/// Bounded rank-major staging, reordered to original coordinates after an
/// all-gather. Neither normalization nor selection can see an incomplete row.
pub struct QsaGather {
    wire: Vec<f32>,
    offsets: Vec<usize>,
    ranges: Vec<Vec<Range<usize>>>,
    tokens: usize,
    width: usize,
}
impl QsaGather {
    fn layout(plan: &RankPlan) -> Result<Self> {
        let mut ranges = Vec::new();
        let mut offsets = vec![0];
        for rank in 0..plan.ranks {
            let mut base = 0;
            let mut columns = Vec::new();
            for (_, width) in NORMS {
                let r = partition(width as u64, rank, plan.ranks).map_err(|e| e.to_string())?;
                columns.push(base + r.start as usize..base + r.end as usize);
                base += width;
            }
            offsets.push(offsets.last().unwrap() + columns.iter().map(|r| r.len()).sum::<usize>());
            ranges.push(columns);
        }
        Ok(Self { wire: vec![0.0; *offsets.last().unwrap()], offsets, ranges, tokens: 1, width: 768 })
    }
    /// Gather short norm vectors from their resident slices once per layer.
    pub fn norms(w: &Weights<'_>, layer: usize) -> Result<Self> {
        Self::norms_prefix(w, &format!("model.language_model.layers.{layer}.self_attn."))
    }
    pub(super) fn norms_prefix(w: &Weights<'_>, prefix: &str) -> Result<Self> {
        qsa_tc_switch()?;
        let mut result = Self::layout(&w.plan)?;
        let mut at = result.offsets[w.plan.rank as usize];
        for (suffix, _) in NORMS {
            let name = format!("{prefix}{suffix}");
            for value in w.values_f32(&name)? { result.wire[at] = value; at += 1; }
        }
        Ok(result)
    }
    pub fn host_bytes(&self) -> usize { self.wire.len() * 4 }
    pub fn gather(mut self, exchange: impl FnOnce(&mut [f32], &[usize]) -> Result<()>) -> Result<Vec<f32>> {
        exchange(&mut self.wire, &self.offsets)?;
        let mut canonical = vec![0.0; self.tokens * self.width];
        for (rank, ranges) in self.ranges.iter().enumerate() {
            let mut source = self.offsets[rank];
            for token in 0..self.tokens {
                for range in ranges {
                    canonical[token * self.width + range.start..token * self.width + range.end]
                        .copy_from_slice(&self.wire[source..source + range.len()]);
                    source += range.len();
                }
            }
        }
        Ok(canonical)
    }
}

/// Allocate one active layer's workspace, then reuse it for every indexed
/// layer. GDN's projection/launch path is shared; no weight expansion is held.
pub struct QsaScratch<'a> {
    heads: usize,
    keys: usize,
    capacity: usize,
    history: usize,
    qg: DeviceVector<'a>,
    k: DeviceVector<'a>,
    v: DeviceVector<'a>,
    index: DeviceVector<'a>,
    q: DeviceVector<'a>,
    index_q: DeviceVector<'a>,
    scores: DeviceVector<'a>,
    gated: DeviceVector<'a>,
    output: DeviceVector<'a>,
    norm: DeviceVector<'a>,
    /// [heads * tokens, min(history, QSA_MAX_POSITIONS)] selected-slot scores, then probabilities.
    attention_scores: DeviceVector<'a>,
    /// One softmax denominator per query head and token.
    denominators: DeviceVector<'a>,
    /// One-token TC values: k-step tiles [KV head][slice][k-step][16][64].
    split: DeviceVector<'a>,
    norm_ready: bool,
    norm_source: Option<DeviceVector<'a>>,
    pub(super) cursor: u64,
    bytes: usize,
}
impl<'a> QsaScratch<'a> {
    pub(super) fn pool_template(&self) -> QsaBatchRow {
        QsaBatchRow { keys:0,values:0,raw_index:0,pooled:0,norm:0,past:0,stride:self.history.min(QSA_MAX_POSITIONS) as u32,
            query:self.q.device_address(),index_query:self.index_q.device_address(),ids:self.scores.device_address(),
            scores:self.attention_scores.device_address(),denominators:self.denominators.device_address() }
    }

    pub fn new(b: &Backend<'a>, plan: &RankPlan, max_tokens: usize, history: usize) -> Result<Self> {
        Self::allocate(b, plan, max_tokens, history, false)
    }
    /// MTP catchup writes KV but never uses attention scratch.
    pub(super) fn cache_only(b: &Backend<'a>, plan: &RankPlan, max_tokens: usize, history: usize) -> Result<Self> {
        Self::allocate(b, plan, max_tokens, history, true)
    }
    fn lengths(plan: &RankPlan, max_tokens: usize, history: usize, cache_only: bool) -> [usize; 13] {
        let heads = (plan.query_heads.end - plan.query_heads.start) as usize;
        let keys = (plan.kv_heads.end - plan.kv_heads.start) as usize;
        let blocks = if cache_only { 0 } else { history / 4 };
        let queries = if cache_only { 0 } else { max_tokens * heads };
        [queries * 512, max_tokens * keys * DIM, max_tokens * keys * DIM,
            max_tokens * INDEX, queries * DIM,
            max_tokens * 512, max_tokens * blocks,
            queries * DIM, if cache_only { 0 } else { max_tokens * HIDDEN }, 770, queries * history.min(QSA_MAX_POSITIONS),
            queries, if cache_only { 0 } else { keys * 4 * QSA_MAX_POSITIONS.div_ceil(16) * 16 * 64 }].map(|n| n.max(1))
    }
    pub(super) fn reservation_bytes(plan: &RankPlan, max_tokens: usize, history: usize, cache_only: bool) -> usize {
        Self::lengths(plan, max_tokens, history, cache_only).iter().map(|n| (n * 4).div_ceil(64) * 64).sum()
    }
    fn allocate(b: &Backend<'a>, plan: &RankPlan, max_tokens: usize, history: usize, cache_only: bool) -> Result<Self> {
        qsa_tc_switch()?;
        if max_tokens == 0 || history < max_tokens || history > 262144 {
            return Err("invalid indexed-attention workspace capacity".into());
        }
        let lengths = Self::lengths(plan, max_tokens, history, cache_only);
        Ok(Self::from_vectors(plan, max_tokens, history, &lengths, b.vector_arena(&lengths)?))
    }
    fn from_vectors(plan: &RankPlan, max_tokens: usize, history: usize,
        lengths: &[usize; 13], vectors: Vec<DeviceVector<'a>>) -> Self {
        let heads = (plan.query_heads.end - plan.query_heads.start) as usize;
        let keys = (plan.kv_heads.end - plan.kv_heads.start) as usize;
        let bytes = lengths.iter().map(|n| (n * 4).div_ceil(64) * 64).sum();
        let mut v = vectors.into_iter();
        Self { heads, keys, capacity: max_tokens, history,
            qg: v.next().unwrap(), k: v.next().unwrap(), v: v.next().unwrap(),
            index: v.next().unwrap(), q: v.next().unwrap(),
            index_q: v.next().unwrap(), scores: v.next().unwrap(),
            gated: v.next().unwrap(), output: v.next().unwrap(),
            norm: v.next().unwrap(), attention_scores: v.next().unwrap(), denominators: v.next().unwrap(), split: v.next().unwrap(),
            norm_ready: false, norm_source: None, cursor: 0, bytes }
    }
    pub fn set_norms(&mut self, b: &Backend<'_>, norms: &[f32]) -> Result<()> {
        if norms.len() != 768 { return Err("QSA needs all four gathered norm vectors".into()); }
        self.norm_source = None;
        self.cursor = 0;
        b.write_vector(&mut self.norm.slice(0..768)?, norms)?;
        self.norm_ready = true;
        Ok(())
    }
    /// VADE_MTP_BATCH draft observe: this scratch's own norms, for a wider scratch to borrow.
    pub(super) fn own_norms(&self) -> Result<DeviceVector<'a>> { self.norm.slice(0..768) }
    pub(super) fn max_tokens(&self) -> usize { self.capacity }
    pub fn use_norm(&mut self, norm: &DeviceVector<'a>) -> Result<()> {
        if norm.len() != 768 { return Err("invalid resident QSA norms".into()); }
        self.norm_source = Some(norm.slice(0..768)?);
        self.norm_ready = true;
        self.cursor = 0;
        Ok(())
    }
    fn norm_address(&self) -> u64 { self.norm_source.as_ref().unwrap_or(&self.norm).device_address() }
    pub(super) fn cursor_address(&self) -> u64 { self.norm.device_address() + 768 * 4 }
    pub(super) fn upload_cursor(&mut self, b: &Backend<'_>, past: usize, tokens: usize) -> Result<()> {
        b.write_vector(&mut self.norm.slice(768..770)?, &[f32::from_bits(past as u32), f32::from_bits(((past + tokens) / 4) as u32)])
    }
    pub fn output(&self) -> &DeviceVector<'a> { &self.output }
    pub fn output_mut(&mut self) -> &mut DeviceVector<'a> { &mut self.output }
    pub fn device_bytes(&self) -> usize { self.bytes }
}

/// Session KV and index history survive a streamed weight allocation.
/// Each complete group's fourth raw row holds its pooled key. Rewinding into
/// a group preserves its first three raw rows; append rewrites the fourth
/// before pooling again. Prefix copies therefore retain every needed value.
pub struct QsaState<'a> {
    layer: usize,
    rank: u64,
    keys: DeviceVector<'a>,
    values: DeviceVector<'a>,
    raw_index: DeviceVector<'a>,
    capacity: usize,
    past: usize,
    pending: Option<usize>,
}
impl<'a> QsaState<'a> {
    pub(super) fn pool_addresses(&self) -> [u64;3] { [self.keys.device_address(),self.values.device_address(),self.raw_index.device_address()] }
    pub(super) fn commit_next(&mut self) { self.past+=1; }

    pub(super) fn snapshot(&self, b: &Backend<'a>, capacity: usize) -> Result<Self> {
        if self.pending.is_some() || capacity < self.past { return Err("invalid QSA snapshot capacity".into()); }
        let width = if self.keys.len() < DIM { 0 } else { self.keys.len() / self.capacity };
        let mut next = Self { layer: self.layer, rank: self.rank,
            keys: b.vector((capacity * width).max(1))?, values: b.vector((capacity * width).max(1))?,
            raw_index: b.vector(capacity * 128)?, capacity, past: self.past, pending: None };
        if width != 0 {
            b.copy_range(&self.keys, &mut next.keys, 0, 0, self.past * width)?;
            b.copy_range(&self.values, &mut next.values, 0, 0, self.past * width)?;
        }
        b.copy_range(&self.raw_index, &mut next.raw_index, 0, 0, self.past * 128)?;
        Ok(next)
    }
    pub(super) fn rewind(&mut self, position: usize) { self.past = position; self.pending = None; }
    /// No attention write in flight (a finished session's arena may be adopted).
    pub(super) fn idle(&self) -> bool { self.pending.is_none() }
    pub(super) fn device_bytes(&self) -> usize { (self.keys.len() + self.values.len() + self.raw_index.len()) * 4 }
    /// Prefix cache: FP32 lengths of `copy_rows_into`'s keys, values and raw index.
    pub(super) fn rows_lengths(&self, rows: usize) -> [usize; 3] {
        let width = if self.keys.len() < DIM { 0 } else { self.keys.len() / self.capacity };
        [(rows * width).max(1), (rows * width).max(1), (rows * 128).max(1)]
    }
    /// Prefix cache: an independent copy of committed rows 0..rows in caller-allocated
    /// vectors of `rows_lengths(rows)` (one arena per prefix).
    pub(super) fn copy_rows_into(&self, b: &Backend<'a>, rows: usize, [keys, values, raw_index]: [DeviceVector<'a>; 3]) -> Result<Self> {
        if self.pending.is_some() || rows > self.past { return Err("QSA prefix copy exceeds committed rows".into()); }
        let width = if self.keys.len() < DIM { 0 } else { self.keys.len() / self.capacity };
        if [keys.len(), values.len(), raw_index.len()] != self.rows_lengths(rows) { return Err("QSA prefix storage shape".into()); }
        let mut next = Self { layer: self.layer, rank: self.rank, keys, values,
            raw_index, capacity: rows.max(1), past: rows, pending: None };
        if width != 0 && rows != 0 {
            b.copy_range(&self.keys, &mut next.keys, 0, 0, rows * width)?;
            b.copy_range(&self.values, &mut next.values, 0, 0, rows * width)?;
        }
        if rows != 0 { b.copy_range(&self.raw_index, &mut next.raw_index, 0, 0, rows * 128)?; }
        Ok(next)
    }
    /// Bytes `copy_rows` allocates for `rows` rows of a state with this capacity.
    pub(super) fn rows_bytes(&self, rows: usize) -> usize {
        let width = if self.keys.len() < DIM { 0 } else { self.keys.len() / self.capacity };
        (2 * (rows * width).max(1) + (rows * 128).max(1)) * 4 + 3 * 256
    }
    /// Install a `copy_rows` prefix into this (fresh) state of the same layer and width.
    pub(super) fn restore_rows(&mut self, b: &Backend<'a>, prefix: &Self) -> Result<()> {
        let width = if self.keys.len() < DIM { 0 } else { self.keys.len() / self.capacity };
        let source = if prefix.keys.len() < DIM { 0 } else { prefix.keys.len() / prefix.capacity };
        if prefix.layer != self.layer || prefix.rank != self.rank || prefix.past > self.capacity
            || prefix.pending.is_some() || (prefix.past != 0 && source != width) {
            return Err("QSA prefix differs in layer, rank, width or capacity".into());
        }
        if width != 0 && prefix.past != 0 {
            b.copy_range(&prefix.keys, &mut self.keys, 0, 0, prefix.past * width)?;
            b.copy_range(&prefix.values, &mut self.values, 0, 0, prefix.past * width)?;
        }
        if prefix.past != 0 { b.copy_range(&prefix.raw_index, &mut self.raw_index, 0, 0, prefix.past * 128)?; }
        self.rewind(prefix.past);
        Ok(())
    }
}

/// VADE_FLASHNEXT_KV_RESERVE (default on; =0 off): Flash-Next session KV arenas come from
/// the backend KV reserve, so a session of the same capacity reuses a freed arena instead of
/// a fresh allocation. Arena contents beyond `past` are never read either way.
pub fn kv_reserve() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_FLASHNEXT_KV_RESERVE").as_deref() != Ok("0"))
}

/// VADE_VERIFY_POOL_NODE (default on; =0 the pool launches only when a four-token group
/// completes): a verify pass captures the same kernel list whether or not its rows complete
/// a group, so the retained segment executable is updated in place (cuGraphExecUpdate)
/// instead of re-instantiated on every flip. The empty launch computes and writes nothing.
fn verify_pool_node() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_VERIFY_POOL_NODE").as_deref() != Ok("0"))
}

pub struct Qsa<'a, 'w> {
    next: bool,
    /// VADE_FLASHNEXT_DRAFT_MX8_LAYER=1 (MTP drafter only): launches of M <= 4 take the latency
    /// entries, which read the registered MXFP8 twins of these official BF16 matrices.
    draft_mx8: bool,
    layer: usize,
    plan: RankPlan,
    qg: u64,
    k: u64,
    v: u64,
    index: u64,
    out: u64,
    native: Option<[super::next_native::Matrix; 5]>,
    keys: DeviceVector<'a>,
    values: DeviceVector<'a>,
    raw_index: DeviceVector<'a>,
    capacity: usize,
    past: usize,
    pending: Option<usize>,
    weight_bytes: usize,
    _weights: PhantomData<&'w ()>,
}
impl<'a, 'w> Qsa<'a, 'w> {
    fn pooled_index(&self) -> u64 { self.raw_index.device_address() + 3 * 128 * 4 }
    pub fn load(b: &Backend<'a>, w: &'w Weights<'_>, layer: usize, capacity: usize) -> Result<Self> {
        Self::load_cached(b, w, layer, capacity, None)
    }
    pub fn load_cached(b: &Backend<'a>, w: &'w Weights<'_>, layer: usize, capacity: usize, cached: Option<QsaState<'a>>) -> Result<Self> {
        if layer >= 48 || layer % 4 != 3 || capacity == 0 || capacity > 262144 {
            return Err("invalid indexed layer or cache capacity".into());
        }
        Self::load_prefix(b, w, &format!("model.language_model.layers.{layer}.self_attn."), layer, capacity, cached)
    }
    pub(super) fn load_prefix(b: &Backend<'a>, w: &'w Weights<'_>, p: &str, layer: usize, capacity: usize, cached: Option<QsaState<'a>>) -> Result<Self> {
        let heads = (w.plan.query_heads.end - w.plan.query_heads.start) as usize;
        let keys = (w.plan.kv_heads.end - w.plan.kv_heads.start) as usize;
        let get = |part: &str, shape: &[u64], elements| bf16(w, &format!("{p}{part}"), shape, elements);
        let cache = if let Some(cache) = cached {
            if cache.layer != layer || cache.rank != w.plan.rank || cache.capacity != capacity
                || cache.keys.len() != (capacity * keys * DIM).max(1) {
                return Err("QSA cache does not belong to this layer, rank and capacity".into());
            }
            if cache.pending.is_some() { return Err("finish QSA attention before detaching its weights".into()); }
            cache
        } else {
            let mut v = b.vector_arena(&[(capacity * keys * DIM).max(1), (capacity * keys * DIM).max(1), capacity * 128])?.into_iter();
            QsaState { layer, rank: w.plan.rank, keys: v.next().unwrap(), values: v.next().unwrap(), raw_index: v.next().unwrap(), capacity, past: 0, pending: None }
        };
        Ok(Self { next: false, draft_mx8: false, layer, plan: w.plan.clone(),
            qg: get("q_proj.weight", &[12288, 2560], heads * 512 * HIDDEN)?,
            k: get("k_proj.weight", &[512, 2560], keys * DIM * HIDDEN)?,
            v: get("v_proj.weight", &[512, 2560], keys * DIM * HIDDEN)?,
            index: get("indexer.index_qk_proj.weight", &[640, 2560], INDEX * HIDDEN)?,
            out: get("o_proj.weight", &[2560, 6144], HIDDEN * heads * DIM)?,
            native: None, keys: cache.keys, values: cache.values, raw_index: cache.raw_index,
            capacity, past: cache.past, pending: cache.pending,
            weight_bytes: w.prefix_bytes(&p),
            _weights: PhantomData })
    }
    /// Mixed-checkpoint layer: MXFP8 projections arrive as expanded packed
    /// BF16 addresses [q, k, v, index, o]; norms stay in the resident arena.
    pub fn load_next(b: &Backend<'a>, w: &'w Weights<'_>, layer: usize, capacity: usize,
        proj: [u64; 5], cached: Option<QsaState<'a>>) -> Result<Self> {
        if layer >= 48 || layer % 4 != 3 || capacity == 0 || capacity > 262144 {
            return Err("invalid indexed layer or cache capacity".into());
        }
        if proj.iter().any(|a| a % 16 != 0) { return Err("expanded QSA projections need 16-byte alignment".into()); }
        let keys = (w.plan.kv_heads.end - w.plan.kv_heads.start) as usize;
        let cache = if let Some(cache) = cached {
            if cache.layer != layer || cache.rank != w.plan.rank || cache.capacity != capacity
                || cache.keys.len() != (capacity * keys * DIM).max(1) { return Err("QSA cache does not belong to this layer, rank and capacity".into()); }
            if cache.pending.is_some() { return Err("finish QSA attention before detaching its weights".into()); }
            cache } else {
            let lengths = [(capacity * keys * DIM).max(1), (capacity * keys * DIM).max(1), capacity * 128];
            let mut v = if kv_reserve() { b.kv_arena(&lengths)? } else { b.vector_arena(&lengths)? }.into_iter();
            QsaState { layer, rank: w.plan.rank, keys: v.next().unwrap(), values: v.next().unwrap(), raw_index: v.next().unwrap(), capacity, past: 0, pending: None } };
        let [qg, k, v, index, out] = proj;
        Ok(Self { next: false, draft_mx8: false, layer, plan: w.plan.clone(), qg, k, v, index, out,
            native: None, keys: cache.keys, values: cache.values, raw_index: cache.raw_index,
            capacity, past: cache.past, pending: cache.pending,
            weight_bytes: w.prefix_bytes(&format!("model.language_model.layers.{layer}.self_attn.")),
            _weights: PhantomData })
    }
    pub(super) fn load_next_rows(b: &Backend<'a>, w: &'w Weights<'_>, layer: usize, capacity: usize,
        proj: [u64; 5], state: Option<QsaState<'a>>) -> Result<Self> {
        let mut q = Self::load_next(b, w, layer, capacity, proj, state)?; q.next = true; Ok(q)
    }
    /// VADE_FLASHNEXT_DRAFT_MX8_LAYER=1: every projection matrix must already have its twin.
    pub(super) fn use_draft_mx8(&mut self) -> Result<()> {
        let heads = (self.plan.query_heads.end - self.plan.query_heads.start) as usize;
        let keys = (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize;
        let ok = super::blocks::mx8_has(self.qg, heads * 512, HIDDEN) && super::blocks::mx8_has(self.index, INDEX, HIDDEN)
            && super::blocks::mx8_has(self.k, keys * DIM, HIDDEN) && super::blocks::mx8_has(self.v, keys * DIM, HIDDEN)
            && super::blocks::mx8_has(self.out, HIDDEN, heads * DIM);
        if !ok || self.native.is_some() || self.next { return Err("MXFP8 drafter QSA needs a twin for every BF16 projection".into()); }
        self.draft_mx8 = true;
        Ok(())
    }
    pub fn use_native(&mut self, matrices: [super::next_native::Matrix;5]) -> Result<()> {
        if matrices.map(|m| [m.rows,m.cols]) != [[2 * (self.plan.query_heads.end-self.plan.query_heads.start) as u32 * 256,2560],[(self.plan.kv_heads.end-self.plan.kv_heads.start) as u32 * 256,2560],[(self.plan.kv_heads.end-self.plan.kv_heads.start) as u32 * 256,2560],[640,2560],[2560,(self.plan.query_heads.end-self.plan.query_heads.start) as u32 * 256]] {
            return Err("native QSA geometry does not match Flash-Next".into());
        }
        self.native=Some(matrices); Ok(())
    }
    fn output_projection(&self, b: &Backend<'_>, s: &QsaScratch<'_>, tokens: usize) -> Result<()> {
        if let Some(m) = self.native { super::next_native::project(b,&[(m[4],s.gated.device_address(),s.output.device_address())],tokens,false,false) }
        else if self.next || (self.draft_mx8 && tokens <= 4) { super::blocks::next_rows(b,self.out,s.gated.device_address(),s.output.device_address(),HIDDEN,s.heads*DIM,tokens) }
        else { projection(b,self.out,s.gated.device_address(),s.output.device_address(),HIDDEN,s.heads*DIM,tokens) }
    }
    pub fn into_state(self) -> QsaState<'a> {
        QsaState { layer: self.layer, rank: self.plan.rank, keys: self.keys, values: self.values,
            raw_index: self.raw_index, capacity: self.capacity, past: self.past, pending: self.pending }
    }
    pub fn reset(&mut self) { self.past = 0; self.pending = None; }
    pub(super) fn rewind(&mut self, position: usize) { self.past = position; self.pending = None; }
    /// Bytes `snapshot_prefix(rows)` allocates.
    pub(super) fn prefix_bytes(&self, rows: usize) -> usize {
        let width = (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize * DIM;
        (2 * (rows.max(1) * width).max(1) + rows.max(1) * 128) * 4 + 3 * 256
    }
    pub(super) fn snapshot_prefix(&self, b: &Backend<'a>, past: usize) -> Result<QsaState<'a>> {
        if past > self.past { return Err("MTP snapshot exceeds committed attention history".into()); }
        self.snapshot_rows(b, past.max(1), past)
    }
    fn snapshot_capacity(&self, b: &Backend<'a>, capacity: usize) -> Result<QsaState<'a>> {
        self.snapshot_rows(b, capacity, self.past)
    }
    fn snapshot_rows(&self, b: &Backend<'a>, capacity: usize, past: usize) -> Result<QsaState<'a>> {
        if self.pending.is_some() || capacity < past { return Err("invalid MTP attention snapshot capacity".into()); }
        let width = (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize * DIM;
        let mut v = b.vector_arena(&[(capacity * width).max(1), (capacity * width).max(1), capacity * 128])?.into_iter();
        let mut state = QsaState { layer: self.layer, rank: self.plan.rank, keys: v.next().unwrap(), values: v.next().unwrap(),
            raw_index: v.next().unwrap(), capacity, past, pending: None };
        if width != 0 && past != 0 {
            b.copy_range(&self.keys, &mut state.keys, 0, 0, past * width)?;
            b.copy_range(&self.values, &mut state.values, 0, 0, past * width)?;
        }
        if past != 0 { b.copy_range(&self.raw_index, &mut state.raw_index, 0, 0, past * 128)?; }
        Ok(state)
    }
    /// Prefix cache: FP32 lengths of `snapshot_prefix_into(past)`'s vectors.
    pub(super) fn prefix_lengths(&self, past: usize) -> [usize; 3] {
        let width = (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize * DIM;
        let capacity = past.max(1);
        [(capacity * width).max(1), (capacity * width).max(1), capacity * 128]
    }
    /// `snapshot_prefix(past)` in caller-allocated vectors (one arena per prefix).
    pub(super) fn snapshot_prefix_into(&self, b: &Backend<'a>, past: usize, [keys, values, raw_index]: [DeviceVector<'a>; 3])
        -> Result<QsaState<'a>> {
        if self.pending.is_some() || past > self.past { return Err("MTP snapshot exceeds committed attention history".into()); }
        if [keys.len(), values.len(), raw_index.len()] != self.prefix_lengths(past) { return Err("MTP prefix storage shape".into()); }
        let width = (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize * DIM;
        let mut state = QsaState { layer: self.layer, rank: self.plan.rank, keys, values, raw_index, capacity: past.max(1), past, pending: None };
        if width != 0 && past != 0 {
            b.copy_range(&self.keys, &mut state.keys, 0, 0, past * width)?;
            b.copy_range(&self.values, &mut state.values, 0, 0, past * width)?;
        }
        if past != 0 { b.copy_range(&self.raw_index, &mut state.raw_index, 0, 0, past * 128)?; }
        Ok(state)
    }
    pub(super) fn reserve(&mut self, b: &Backend<'a>, capacity: usize) -> Result<()> {
        if capacity <= self.capacity { return Ok(()); }
        let state = self.snapshot_capacity(b, capacity)?;
        self.keys = state.keys;
        self.values = state.values;
        self.raw_index = state.raw_index;
        self.capacity = capacity;
        Ok(())
    }
    pub(super) fn restore(&mut self, b: &Backend<'a>, state: &QsaState<'a>) -> Result<()> {
        if state.layer != self.layer || state.rank != self.plan.rank || state.past > self.capacity || state.pending.is_some() {
            return Err("MTP attention snapshot differs in layer, rank or capacity".into());
        }
        let width = (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize * DIM;
        if width != 0 && state.past != 0 {
            b.copy_range(&state.keys, &mut self.keys, 0, 0, state.past * width)?;
            b.copy_range(&state.values, &mut self.values, 0, 0, state.past * width)?;
        }
        if state.past != 0 { b.copy_range(&state.raw_index, &mut self.raw_index, 0, 0, state.past * 128)?; }
        self.rewind(state.past);
        Ok(())
    }
    pub fn cached_tokens(&self) -> usize { self.past }
    pub fn weight_bytes(&self) -> usize { self.weight_bytes }
    pub fn kv_bytes(&self) -> usize { 2 * (self.capacity * (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize * DIM * 4).max(4).div_ceil(64) * 64 }
    pub fn index_bytes(&self) -> usize { self.capacity * 128 * 4 }
    pub(super) fn project_batch(&self, b: &Backend<'_>, batch: &mut QsaBatch<'_>, input: &DeviceVector<'_>, count: usize) -> Result<()> {
        let heads = (self.plan.query_heads.end - self.plan.query_heads.start) as usize;
        batch.index.resize_within_capacity(count * INDEX)?;
        self.project_rows(b, input.device_address(), count, heads, false,
            [batch.qg.device_address(), batch.k.device_address(), batch.v.device_address(), batch.index.device_address()])
    }
    /// Prepare independent group rows; causal lengths still come from each tile.
    pub(super) fn project_prompt(&mut self, b: &Backend<'_>, s: &QsaScratch<'_>, batch: &mut QsaBatch<'_>,
        input: &DeviceVector<'_>, count: usize, reuse: bool) -> Result<()> {
        self.session_row(s)?;
        if count == 0 || self.past.checked_add(count).is_none_or(|n| n > self.capacity)
            || input.len() != count * HIDDEN || input.is_segmented() { return Err("invalid QSA prompt input".into()); }
        batch.index.resize_within_capacity(count * INDEX)?;
        self.project_rows(b, input.device_address(), count, s.heads, reuse && super::blocks::hc_packed(count),
            [batch.qg.device_address(), batch.k.device_address(), batch.v.device_address(), batch.index.device_address()])?;
        let q = batch.sessions.device_address();
        let iq = q + (count * s.heads * DIM * 4) as u64;
        let used = count * (s.heads * DIM + 512);
        let blocks = (self.past + count) / 4;
        // Long contexts retain tile selection when the idle arena cannot hold scores.
        let ids = if used + count * blocks <= batch.sessions.len() { q + (used * 4) as u64 } else { 0 };
        if used > batch.sessions.len() { return Err("QSA prompt preparation exceeds idle arena".into()); }
        self.pending = Some(count);
        let prepared = self.prepare_cache(b, s, batch.index.device_address(),
            [batch.qg.device_address(), batch.k.device_address(), batch.v.device_address()], [q, iq]);
        self.pending = None;
        prepared?;
        if ids != 0 { self.select_index(b, iq, ids, count, blocks, 0, index_score_grid(count, blocks), blocks)?; }
        batch.prompt = Some([q, iq, ids, blocks as u64]);
        Ok(())
    }
    pub(super) fn session_row(&self, s: &QsaScratch<'_>) -> Result<QsaBatchRow> {
        if self.pending.is_some() || self.past >= self.capacity || self.capacity > s.history || !s.norm_ready
            || s.heads != (self.plan.query_heads.end - self.plan.query_heads.start) as usize
            || s.keys != (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize {
            return Err("Flash batch QSA cache/scratch mismatch".into());
        }
        Ok(QsaBatchRow { keys: self.keys.device_address(), values: self.values.device_address(),
            raw_index: self.raw_index.device_address(), query: s.q.device_address(),
            index_query: s.index_q.device_address(), pooled: self.pooled_index(), ids: s.scores.device_address(),
            scores: s.attention_scores.device_address(), denominators: s.denominators.device_address(),
            norm: s.norm_address(), past: self.past as u32, stride: s.history.min(QSA_MAX_POSITIONS) as u32 })
    }
    /// Batch the stages with the current solo score/value geometry and sums.
    /// Scores reserve grid-y for slots and put independent sessions in grid-z.
    pub(super) fn attend_batch(&self,b: &Backend<'_>,batch: &QsaBatch<'_>,count: usize) -> Result<()> {
        Self::attend_pool(b,&self.plan,batch,count)
    }
    fn attend_pool(b: &Backend<'_>,plan: &RankPlan,batch: &QsaBatch<'_>,count: usize) -> Result<()> {
        let heads = (plan.query_heads.end - plan.query_heads.start) as usize;
        let keys = (plan.kv_heads.end - plan.kv_heads.start) as usize;
        let mut params = spark_gpu_probe::kernel_params::QsaBatch {
            rows: batch.sessions.device_address(), query_gate: batch.qg.device_address(), key: batch.k.device_address(),
            value: batch.v.device_address(), index: batch.index.device_address(), output: batch.gated.device_address(),
            query_heads: heads as u32, key_heads: keys as u32, query_start: plan.query_heads.start as u32,
            key_start: plan.kv_heads.start as u32, count: count as u32, padding: 0,
        };
        let p = (&mut params as *mut spark_gpu_probe::kernel_params::QsaBatch).cast();
        let [span, blocks] = batch.shape;
        let padded = count.next_power_of_two() as u32;
        let stage = |name, grid: usize, threads| unsafe {
            b.launch_flash_rows(name, grid as u32, padded, threads, 0, &mut [p])
        };
        // the prepare rows and the five index rows in one launch (independent of index_pool).
        stage("flash_qsa_prepare2_batch", (heads + keys).div_ceil(4) + 5usize.div_ceil(4), 128)?;
        stage("flash_qsa_index_pool_batch", 1, 128)?;
        if blocks != 0 {
            if qsa_tc3().2 == 3 { stage("flash_qsa_scores_tc_batch", blocks.div_ceil(32), 128)?; }
            else { stage("flash_qsa_scores_batch", blocks.div_ceil(4), 128)?; }
            stage("flash_qsa_select_w_batch", 1, 1024)?;
        }
        if heads != 0 {
            let groups = (plan.query_heads.start as usize % QSA_SCORE_HEADS + heads).div_ceil(QSA_SCORE_HEADS);
            let (entry, planes) = score_entry(true, 1, span.div_ceil(QSA_SCORE_KEYS).div_ceil(4));
            unsafe { b.launch_flash_grid(entry, groups as u32, planes as u32, padded, 128, 0, &mut [p])?; }
            stage("flash_qsa_attention_softmax_v_batch", heads.div_ceil(4), 128)?;
            let (entry, slices, threads) = value_entry(true, 1);
            stage(entry, keys * slices, threads)?;
        }
        Ok(())
    }
    pub(super) fn commit_batch(&mut self) { self.past += 1; }
    pub(super) fn attend_batch_row(&mut self, b: &Backend<'_>, batch: &QsaBatch<'_>, row: usize,
        s: &mut QsaScratch<'_>, tokens: usize) -> Result<()> {
        self.session_row(s)?;
        if tokens == 0 || tokens > s.capacity || self.past.checked_add(tokens).is_none_or(|n| n > self.capacity)
            || row.checked_add(tokens).is_none_or(|n| n > batch.index.len() / INDEX) {
            return Err("QSA projected rows exceed group, scratch or cache".into());
        }
        let at = |v: &DeviceVector<'_>, width: usize| v.device_address() + (row * width * 4) as u64;
        let projected = [at(&batch.qg, s.heads * 512), at(&batch.k, s.keys * DIM),
            at(&batch.v, s.keys * DIM), at(&batch.gated, s.heads * DIM)];
        self.pending = Some(tokens);
        let prepared = batch.prompt.map(|[q, iq, ids, blocks]| [q + (row * s.heads * DIM * 4) as u64,
            iq + (row * 512 * 4) as u64, if ids == 0 { 0 } else { ids + row as u64 * blocks * 4 }, blocks]);
        self.attend_projected(b, s, at(&batch.index, INDEX), projected, prepared)
    }
    /// A retired slot keeps the same projection width but never updates KV or
    /// the sparse index. Its pooled output is zero before the shared O pass.
    pub(super) fn pad_batch_row(&self, b: &Backend<'_>, batch: &QsaBatch<'_>, row: usize) -> Result<()> {
        let width = (self.plan.query_heads.end - self.plan.query_heads.start) as usize * DIM;
        if width == 0 { return Ok(()); }
        b.zero_vector(&mut batch.gated.slice(row * width..(row + 1) * width)?)
    }
    pub(super) fn finish_batch(&self, b: &Backend<'_>, batch: &QsaBatch<'_>, output: &mut DeviceVector<'_>, count: usize) -> Result<()> {
        let heads = (self.plan.query_heads.end - self.plan.query_heads.start) as usize;
        if heads == 0 { return b.zero_vector(output); }
        if self.next || (self.draft_mx8 && count <= 4) { return super::blocks::next_rows(b, self.out, batch.gated.device_address(), output.device_address(), HIDDEN, heads*DIM, count); }
        projection_pair(b, batch.gated.device_address(), heads * DIM, count,
            (self.out, output.device_address(), HIDDEN), (0, 0, 0))
    }
    /// Project a batch-one causal prefill tile, or a single cached token.
    /// All 640 indexer outputs are local; the ordered stream connects project
    /// and attend without a readback or collective.
    pub fn project(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>, input: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        self.project_tile(b, s, input, tokens, false)
    }
    pub(super) fn project_cache(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>, input: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        self.project_tile(b, s, input, tokens, true)
    }
    /// Graph replay executes the projection but must also advance its host state.
    pub(super) fn replay_projected(&mut self, tokens: usize) { self.pending = Some(tokens); }
    fn project_tile(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>, input: &DeviceVector<'_>, tokens: usize, cache_only: bool) -> Result<()> {
        self.prepare_project(s, input, tokens)?;
        self.project_rows(b, input.device_address(), tokens, if cache_only { 0 } else { s.heads }, false,
            [s.qg.device_address(), s.k.device_address(), s.v.device_address(), s.index.device_address()])?;
        self.pending = Some(tokens);
        Ok(())
    }
    pub(super) fn prepare_project(&self, s: &mut QsaScratch<'_>, input: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        if self.pending.is_some() || tokens == 0 || tokens > s.capacity || self.past + tokens > self.capacity
            || self.capacity > s.history || input.len() != tokens * HIDDEN || input.is_segmented()
            || s.heads != (self.plan.query_heads.end - self.plan.query_heads.start) as usize || s.keys != (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize || !s.norm_ready {
            return Err("QSA input, cache or scratch does not match this rank".into());
        }
        s.index.resize_within_capacity(tokens * INDEX)?;
        Ok(())
    }
    // Prompt kernels read the packed planes; the flag only bypasses repacking.
    // No operation between these pairs overwrites the backend operand arena.
    // Decode widths keep their original FP32 operands and per-row conversion.
    fn project_rows(&self, b: &Backend<'_>, x: u64, tokens: usize, heads: usize, reuse: bool, output: [u64; 4]) -> Result<()> {
        if let Some(m) = self.native {
            if heads != (self.plan.query_heads.end-self.plan.query_heads.start) as usize { return Err("native QSA head ownership mismatch".into()); }
            return super::next_native::project(b,&(0..4).map(|i| (m[i],x,output[i])).collect::<Vec<_>>(),tokens,false,false);
        }
        if self.next || (self.draft_mx8 && tokens <= 4) {
            let keys = (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize;
            return super::blocks::next_group(
                b,
                x,
                HIDDEN,
                tokens,
                [
                    (self.qg, output[0], heads * 512),
                    (self.index, output[3], INDEX),
                    (self.k, output[1], keys * DIM),
                    (self.v, output[2], keys * DIM),
                ],
            );
        }
        let keys = (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize;
        let packed = tokens > spark_gpu_probe::kernel_params::BF16_ROWS_TOKENS;
        let x = if packed && reuse { b.flash_dense_input()? } else if packed { b.pack_flash_dense_input(x, HIDDEN, tokens)? } else { x };
        super::blocks::qsa_pair(b, x, tokens, (self.qg, output[0], heads * 512), (self.index, output[3], INDEX), packed)?;
        super::blocks::qsa_pair(b, x, tokens, (self.k, output[1], keys * DIM), (self.v, output[2], keys * DIM), packed)
    }
    /// Catch up a draft's committed KV without computing an unused attention
    /// result. Both paths use the same cache preparation kernels.
    pub(super) fn observe(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>) -> Result<()> {
        let index = s.index.device_address();
        let projected = [0, s.k.device_address(), s.v.device_address()];
        let tokens = self.prepare_cache(b, s, index, projected, [s.q.device_address(), s.index_q.device_address()])?;
        self.past += tokens;
        self.pending = None;
        Ok(())
    }
    fn prepare_cache(&self, b: &Backend<'_>, s: &QsaScratch<'_>, index: u64, projected: [u64; 3], [q, iq]: [u64; 2]) -> Result<usize> {
        let [qg, key, value] = projected;
        let tokens = self.pending.ok_or("QSA cache requires projected inputs")?;
        let (t, past, norm) = (tokens as u32, if s.cursor == 0 { self.past as u32 } else { 0 }, s.norm_address());
        launch(b, "flash_qsa_index_prepare", tokens * 5 * 32,
            &mut [Ptr(index), Ptr(norm + 512 * 4), Ptr(iq),
                Ptr(self.raw_index.device_address()), U32(t), U32(past), U32(0), Ptr(s.cursor)])?;
        // Pool even short/observe-only rows, so past/4 always counts valid keys.
        let groups = index_pool_range(self.past, tokens);
        // VADE_VERIFY_POOL_NODE: inside a verify segment capture the pool launches even with no
        // complete group (blocks 0, one CTA: every thread returns before any read or write).
        if !groups.is_empty() || (b.in_segment() && verify_pool_node()) {
            let offset = if s.cursor == 0 { (groups.start * 4 * 128 * 4) as u64 } else { 0 };
            let mut pool = QsaIndexPool { raw_keys: self.raw_index.device_address() + offset,
                weight: norm + 640 * 4, pooled: self.pooled_index() + offset,
                blocks: groups.len() as u32, position_offset: if s.cursor == 0 { (groups.start * 4) as u32 } else { 0 } };
            let mut cursor = s.cursor;
            unsafe { b.launch_flash("flash_qsa_index_pool", groups.len().div_ceil(4).max(1) as u32, 128,
                &mut [(&mut pool as *mut QsaIndexPool).cast(), (&mut cursor as *mut u64).cast()])?; }
        }
        let heads = if qg == 0 { 0 } else { s.heads };
        let mut p = spark_gpu_probe::kernel_params::QsaPrepare {
            query_gate: qg, key, value, query_weight: norm, key_weight: norm + 256 * 4,
            query: q, keys: self.keys.device_address(), values: self.values.device_address(),
            heads: heads as u32, kv_heads: s.keys as u32, tokens: t, past, position_offset: 0,
            frequencies: 0, precision: 0, epsilon: 1e-6, replay_position: s.cursor,
        };
        if heads + s.keys != 0 {
            unsafe { b.launch_flash_shared("flash_qsa_prepare", ((heads + s.keys) * tokens * 32).div_ceil(128) as u32, 128, 0,
                &mut [(&mut p.query_gate as *mut u64).cast(), (&mut p.key as *mut u64).cast(),
                    (&mut p.value as *mut u64).cast(), (&mut p.query_weight as *mut u64).cast(),
                    (&mut p.key_weight as *mut u64).cast(), (&mut p.query as *mut u64).cast(),
                    (&mut p.keys as *mut u64).cast(), (&mut p.values as *mut u64).cast(),
                    (&mut p.heads as *mut u32).cast(), (&mut p.kv_heads as *mut u32).cast(),
                    (&mut p.tokens as *mut u32).cast(), (&mut p.past as *mut u32).cast(),
                    (&mut p.position_offset as *mut u32).cast(), (&mut p.frequencies as *mut u64).cast(),
                    (&mut p.precision as *mut u32).cast(), (&mut p.epsilon as *mut f32).cast(),
                    (&mut p.replay_position as *mut u64).cast()])?; }
        }
        Ok(tokens)
    }
    /// Normalize local index queries, keep raw keys, then pool/normalize/
    /// rotate all visible four-token groups and select the causal QSA IDs.
    /// Result is this rank's O-projection partial, never an HC-ready output.
    /// Rank chunk prefill: one token's attention from row-major projection rows at
    /// `rows` = [qg, k, v, index] (rows of project()'s four outputs for several tokens),
    /// the gated row written to `gated`: project(.., 1) + attend() without the
    /// projections and the output projection (the caller runs both over all rows).
    pub fn attend_row(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>, rows: [u64; 4], gated: u64) -> Result<()> {
        self.attend_rows(b, s, rows, gated, 1)
    }
    /// attend_row for `tokens` < 64 consecutive rows in one pass: the cache, index, score,
    /// softmax and value entries are the one-row ones (scores_tc / softmax_v / values_tc_m,
    /// rows by token), as for verify rows; each row attends to the rows before it.
    pub fn attend_rows(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>, rows: [u64; 4], gated: u64, tokens: usize) -> Result<()> {
        if tokens == 0 || tokens >= 64 || tokens > s.capacity { return Err("QSA rows outside the one-row entries".into()); }
        if self.pending.is_some() || s.capacity == 0 || self.past + tokens > self.capacity || self.capacity > s.history
            || s.heads == 0 || s.heads != (self.plan.query_heads.end - self.plan.query_heads.start) as usize
            || s.keys != (self.plan.kv_heads.end - self.plan.kv_heads.start) as usize || !s.norm_ready {
            return Err("QSA row, cache or scratch does not match this rank".into());
        }
        s.index.resize_within_capacity(tokens * INDEX)?;
        self.pending = Some(tokens);
        self.attend_projected(b, s, rows[3], [rows[0], rows[1], rows[2], gated], None)
    }
    pub fn attend(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>) -> Result<()> {
        let tokens = self.pending.ok_or("QSA attend requires projected inputs")?;
        let index = s.index.device_address();
        let projected = [s.qg.device_address(), s.k.device_address(), s.v.device_address(), s.gated.device_address()];
        self.attend_projected(b, s, index, projected, None)?;
        if s.heads == 0 { b.zero_vector(&mut s.output) }
        else { self.output_projection(b, s, tokens) }
    }
    /// Dense twin of attend() for the FP4-Local decode path: identical cache
    /// preparation (FP32 Q/K norms, partial RoPE, FP32 KV), then dense FP32
    /// attention with the sigmoid output gate over every visible position.
    /// The sparse selection stages never run; index history still advances.
    pub fn attend_dense(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>) -> Result<()> {
        let tokens = self.prepare_cache(b, s, s.index.device_address(),
            [s.qg.device_address(), s.k.device_address(), s.v.device_address()],
            [s.q.device_address(), s.index_q.device_address()])?;
        s.output.resize_within_capacity(tokens * HIDDEN)?;
        if s.heads == 0 { b.zero_vector(&mut s.output)?; } else {
            let mut params = spark_gpu_probe::kernel_params::DenseAttention {
                query: s.q.device_address(), query_gate: s.qg.device_address(), output: s.gated.device_address(),
                rows: 0, scores: 0, replay_position: 0, keys: self.keys.device_address(),
                values: self.values.device_address(), query_heads: s.heads as u32, key_heads: s.keys as u32,
                count: tokens as u32, score_stride: 0, score_tiles: 0, past: self.past as u32 };
            unsafe { b.launch_flash_shared("flash_next_dense_attention",
                (s.heads * tokens).div_ceil(4) as u32, 128, 0,
                &mut [(&mut params as *mut spark_gpu_probe::kernel_params::DenseAttention).cast()])?; }
            self.output_projection(b, s, tokens)?;
        }
        self.past += tokens;
        self.pending = None;
        Ok(())
    }
    fn attend_projected(&mut self, b: &Backend<'_>, s: &mut QsaScratch<'_>, index: u64, projected: [u64; 4], prepared: Option<[u64; 4]>) -> Result<()> {
        let [qg, key, value, gated] = projected;
        let tokens = if prepared.is_some() { self.pending.ok_or("QSA prepared tile has no pending rows")? }
            else { self.prepare_cache(b, s, index, [qg, key, value], [s.q.device_address(), s.index_q.device_address()])? };
        s.output.resize_within_capacity(tokens * HIDDEN)?;
        let t = tokens as u32;
        let past = if s.cursor == 0 { self.past as u32 } else { 0 };
        let blocks = (self.past + tokens) / 4;
        let capacity = if s.cursor == 0 { self.past + tokens } else { s.history };
        let shape = decode_shape(self.past, tokens, capacity);
        let [q, iq, ids, stride] = prepared.unwrap_or([s.q.device_address(), s.index_q.device_address(), 0, 0]);
        if ids == 0 { self.select_index(b, iq, s.scores.device_address(), tokens, blocks, s.cursor, shape[1], capacity / 4)?; }
        // Scores share keys across prompt heads; softmax and value sums keep
        // their canonical per-head position order in the following stages.
        let mut params = QsaAttention { query: q, query_gate: qg,
            keys: self.keys.device_address(), values: self.values.device_address(), ids: if ids == 0 { s.scores.device_address() } else { ids },
            output: gated, scores: s.attention_scores.device_address(),
            query_heads: s.heads as u32, key_heads: s.keys as u32, query_start: self.plan.query_heads.start as u32,
            key_start: self.plan.kv_heads.start as u32, tokens: t, past, blocks: if ids != 0 { stride as u32 } else if s.cursor == 0 { blocks as u32 } else { 0 }, queries_per_key: 12,
            stride: s.history.min(QSA_MAX_POSITIONS) as u32, padding: 0, denominators: s.denominators.device_address() };
        let rows = s.heads * tokens;
        b.decode_qsa_read(&QsaAttention { past: self.past as u32, blocks: if s.cursor == 0 { params.blocks } else { blocks as u32 }, ..params })?;
        let p = (&mut params as *mut QsaAttention).cast();
        // Score and value stages read the same IDs before this ordered stream
        // reuses the dead index-score scratch; short rows never dereference it.
        // One kernel per stage for every query count: a decode row, MTP
        // verification rows and prompt tiles share each score and value order.
        if rows != 0 {
            let groups = (self.plan.query_heads.start as usize % QSA_SCORE_HEADS + s.heads).div_ceil(QSA_SCORE_HEADS);
            unsafe {
                if qsa_tc() == (3, 3) && tokens >= 64 {
                    // Switch-on prompt tiles: scores _tc_w, softmax and values _tc
                    // fused per token x KV head, the same bytes as the three stages
                    // (attention_scores keeps S, denominators stay unwritten).
                    b.launch_flash("flash_qsa_attention_fused_tc", (groups * tokens) as u32, 128,
                        &mut [p, (&mut s.cursor as *mut u64).cast()])?;
                } else {
                    let (entry, planes) = score_entry(false, tokens, shape[2]);
                    b.launch_flash_rows(entry, (groups * tokens) as u32,
                        planes as u32, 128, 0, &mut [p, (&mut s.cursor as *mut u64).cast()])?;
                    // Decode / verify rows: chunk probabilities via a warp tile, not 32 shuffles (same bytes).
                    let softmax = if tokens < 64 { "flash_qsa_attention_softmax_v" } else { "flash_qsa_attention_softmax" };
                    b.launch_flash(softmax, rows.div_ceil(4) as u32, 128, &mut [p, (&mut s.cursor as *mut u64).cast()])?;
                    let (entry, slices, threads) = value_entry(false, tokens);
                    if tokens == 1 && qsa_tc().1 == 3 && shape[1] != 0 {
                        // One-token TC rows with selection active (2048 selected slots): values split by k-step
                        // (tiles, then the in-order sum; same bytes). Shorter rows keep values_tc_m (the split
                        // loses at 300 / 1500 tokens). decode_shape separates the two cases for captured graphs.
                        let mut split = s.split.device_address();
                        b.launch_flash_rows("flash_qsa_attention_values_tc_k", (s.keys * 4) as u32, QSA_MAX_POSITIONS.div_ceil(64) as u32, 128, 0,
                            &mut [p, (&mut s.cursor as *mut u64).cast(), (&mut split as *mut u64).cast()])?;
                        b.launch_flash("flash_qsa_attention_values_tc_kr", (s.keys * 4) as u32, 256,
                            &mut [p, (&mut s.cursor as *mut u64).cast(), (&mut split as *mut u64).cast()])?;
                    } else {
                        b.launch_flash(entry, (s.keys * tokens * slices) as u32, threads, &mut [p, (&mut s.cursor as *mut u64).cast()])?;
                    }
                }
            }
        }
        self.past += tokens;
        self.pending = None;
        Ok(())
    }
    /// `grid` is main's scorer grid; `groups` bounds the tensor-core grid
    /// (capacity / 4 for cursor graphs, the live group count otherwise).
    fn select_index(&self, b: &Backend<'_>, query: u64, scores: u64, tokens: usize, blocks: usize, mut cursor: u64, grid: usize,
        groups: usize) -> Result<()> {
        if blocks <= 512 { return Ok(()); }
        let mut index = QsaIndex { query, pooled: self.pooled_index(), scores, tokens: tokens as u32,
            past: if cursor == 0 { self.past as u32 } else { 0 }, blocks: if cursor == 0 { blocks as u32 } else { 0 }, padding: 0 };
        let p = (&mut index as *mut QsaIndex).cast();
        unsafe {
            if qsa_tc3().2 == 3 {
                b.launch_flash_rows(if tokens >= 64 { "flash_qsa_scores_tc" } else { "flash_qsa_scores_tc_m" }, groups.div_ceil(32) as u32,
                    tokens.div_ceil(if tokens >= 64 { 64 } else { 16 }) as u32, 128, 0,
                    &mut [p, (&mut cursor as *mut u64).cast()])?;
            } else if tokens >= 64 {
                // Prompt tiles: thread-owned scores (4 queries x 4 groups), the same lane
                // tree without shuffles (same bytes).
                b.launch_flash("flash_qsa_scores_t44", (tokens.div_ceil(32) * groups.div_ceil(64)) as u32, 128,
                    &mut [p, (&mut cursor as *mut u64).cast()])?;
            } else if tokens >= 16 {
                // 16..63 rows: thread-owned scores, 16 queries x 64 groups per CTA (same bytes).
                b.launch_flash("flash_qsa_scores_t", (tokens.div_ceil(16) * groups.div_ceil(64)) as u32, 128,
                    &mut [p, (&mut cursor as *mut u64).cast()])?;
            } else if tokens >= 4 {
                // MTP verify rows: one warp per 4 queries x 32 groups (same bytes).
                b.launch_flash("flash_qsa_scores_t4", (tokens.div_ceil(4) * groups.div_ceil(32)) as u32, 32,
                    &mut [p, (&mut cursor as *mut u64).cast()])?;
            } else {
                b.launch_flash("flash_qsa_scores", grid as u32, 128, &mut [p, (&mut cursor as *mut u64).cast()])?;
            }
            // Decode and verify rows: 1024-thread selection (same IDs); prompt tiles keep 256.
            let (entry, threads) = if tokens >= 64 { ("flash_qsa_select", 256) } else { ("flash_qsa_select_w", 1024) };
            b.launch_flash(entry, tokens as u32, threads, &mut [p, (&mut cursor as *mut u64).cast()])
        }
    }
}

#[test]
fn prompt_rows_share_one_cache_consecutively() {
    let mut rows: Vec<f32> = (0..9).flat_map(|i| [16, 0, 32, 0, 600 + i, 0].map(f32::from_bits)).collect();
    assert_eq!(prompt_past(&rows), Some(600));
    rows[52] = f32::from_bits(607); assert_eq!(prompt_past(&rows), None); // Repeated past.
    rows[52] = f32::from_bits(608); rows[50] = f32::from_bits(48); assert_eq!(prompt_past(&rows), None); // Other values.
    assert_eq!(prompt_past(&rows[..6]), Some(600));
    assert_eq!(prompt_past(&[]), None);
}

#[test]
fn prompt_projection_arena_fits() {
    for world in 1..=8 { for rank in 0..world { for rows in [1, 63, 64, 65, 127, 128, 129, 4095, 4096, 8192] {
        assert!(QsaBatch::lengths(&RankPlan::new(rank, world).unwrap(), rows).iter()
            .map(|n| n.div_ceil(16) * 16).sum::<usize>() <= rows * super::moe::PARTIALS);
    } } }
}

#[test]
fn index_pool_append_and_rewind() {
    assert_eq!(decode_shape(100, 3, 32768), decode_shape(104, 3, 32768));
    assert_ne!(decode_shape(100, 3, 32768), decode_shape(101, 3, 32768)); // Pooling variant.
    assert_ne!(decode_shape(2048, 3, 32768), decode_shape(2049, 3, 32768)); // Sparse selection starts.
    assert_eq!(decode_shape(2064, 3, 32768), decode_shape(2068, 3, 32768)); // Fixed index score grid.
    assert_eq!(decode_shape(28, 3, 32768), decode_shape(32, 3, 32768)); // Fixed attention score grid.
    for (past, tokens, groups) in [(0, 3, 0..0), (0, 64, 0..16), (3, 1, 0..1), (2047, 5, 511..513), (131071, 1, 32767..32768)] { assert_eq!(index_pool_range(past, tokens), groups); }
}
