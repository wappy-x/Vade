//! Shared BF16 tensor-core projections for independent decode and prompt rows.
//! Dense verification shares Session's native BF16 accumulation order.
//! Dense and expert decode use four fixed K ranges and a balanced FP32 merge.
use super::{Mlp, Model, Result, Session};
use spark_gpu_probe::driver::{BackendStats, DecodeGraph, DeviceVector, MoeBatchScratch, MOE_ROUTING_ERROR};
use std::time::Instant;

#[derive(Debug)]
pub struct Error { pub request: Option<usize>, pub message: String }
impl Error {
    fn at(request: usize, message: impl Into<String>) -> Self { Self { request: Some(request), message: message.into() } }
}
impl From<String> for Error { fn from(message: String) -> Self { Self { request: None, message } } }
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(i) = self.request { write!(f, "request {i}: ")?; }
        f.write_str(&self.message)
    }
}
impl From<Error> for String { fn from(error: Error) -> Self { error.to_string() } }
type BatchResult<T> = std::result::Result<T, Error>;

fn input_request(requests: &[Rows<'_, '_, '_, '_>], mut row: usize) -> Option<usize> {
    for (i, r) in requests.iter().enumerate() {
        if row < r.tokens.len() { return Some(i); }
        row -= r.tokens.len();
    }
    None
}

fn output_request(requests: &[Rows<'_, '_, '_, '_>], mut row: usize) -> usize {
    for (i, r) in requests.iter().enumerate() {
        if row < r.output_rows() { return i; }
        row -= r.output_rows();
    }
    unreachable!()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output { All, Last, None }

/// One session appears once per tick; its input tokens occupy consecutive
/// positions. Use All for decode/verification, Last or None for prefill chunks.
pub struct Rows<'s, 'm, 'a, 'b> {
    pub session: &'s mut Session<'m, 'a, 'b>,
    pub tokens: &'s [u32],
    pub output: Output,
}
pub struct DraftRows<'s, 'm, 'a, 'b> {
    pub session: &'s mut Session<'m, 'a, 'b>,
    pub history: &'s [u32],
    pub next: u32,
    pub limit: usize,
}
impl Rows<'_, '_, '_, '_> {
    fn output_rows(&self) -> usize {
        match self.output { Output::All => self.tokens.len(), Output::Last => 1, Output::None => 0 }
    }
}

struct Activations<'a> {
    capacity: usize,
    ids: DeviceVector<'a>, metadata: DeviceVector<'a>, hidden: DeviceVector<'a>, input: DeviceVector<'a>, sums: DeviceVector<'a>,
    q: DeviceVector<'a>, k: DeviceVector<'a>, v: DeviceVector<'a>, attention: DeviceVector<'a>,
    projected: DeviceVector<'a>, gate: DeviceVector<'a>, up: DeviceVector<'a>,
    weight_pack: DeviceVector<'a>, input_pack: DeviceVector<'a>,
}
impl<'a> Activations<'a> {
    fn new(model: &Model<'a, '_>, capacity: usize, prefill: bool) -> Result<Self> {
        let c = &model.config;
        let (h, q, k, m) = (c.hidden_size as usize, (c.attention_heads*c.head_dim) as usize,
            model.kv_capacity.stride, if c.num_experts == 0 { c.intermediate_size as usize } else { 1 });
        let packed = capacity.div_ceil(16)*16*m.max(q).max(h)/2;
        let fast_ffn = c.qk_norm && c.num_experts == 0 && h % 128 == 0 && m % 128 == 0
            && (prefill || capacity <= 64 && h % 256 == 0 && m % 256 == 0);
        let gate = if fast_ffn { capacity.div_ceil(16)*16*m/2 } else { capacity*m };
        let up = if fast_ffn { 1 } else { capacity*m };
        // Expert prompt rows must use Session's swizzled KV and softmax order.
        // Reuse this staging arena after QKV; no weight conversion is needed.
        let kv_pack = if prefill && c.num_experts != 0 { capacity.div_ceil(64)*64*k/2 } else { 1 };
        let mut a = model.backend.vector_arena(&[capacity, capacity*16*if c.attention_heads == c.kv_heads*2 { c.layers as usize } else { 1 }, capacity*h, capacity*h, capacity*h.div_ceil(128),
            capacity*q, capacity*k, capacity*k, capacity*q, capacity*h, gate, up, kv_pack, packed])?.into_iter();
        Ok(Self { capacity, ids: a.next().unwrap(), metadata: a.next().unwrap(), hidden: a.next().unwrap(), input: a.next().unwrap(),
            sums: a.next().unwrap(), q: a.next().unwrap(), k: a.next().unwrap(), v: a.next().unwrap(),
            attention: a.next().unwrap(), projected: a.next().unwrap(), gate: a.next().unwrap(), up: a.next().unwrap(),
            weight_pack: a.next().unwrap(), input_pack: a.next().unwrap() })
    }
    fn rows(&self, rows: usize, packed_gate: bool, intermediate: usize) -> Result<Self> {
        let take = |v: &DeviceVector<'a>| v.slice(0..if rows == self.capacity { v.len() } else { v.len()/self.capacity*rows });
        Ok(Self { capacity: rows, ids: take(&self.ids)?, metadata: take(&self.metadata)?, hidden: take(&self.hidden)?, input: take(&self.input)?,
            sums: take(&self.sums)?, q: take(&self.q)?, k: take(&self.k)?, v: take(&self.v)?,
            attention: take(&self.attention)?, projected: take(&self.projected)?,
            gate: self.gate.slice(0..if packed_gate { rows.div_ceil(16)*16*intermediate/2 } else { rows*intermediate })?,
            up: if self.up.len() == 1 { self.up.slice(0..1)? } else { take(&self.up)? },
            weight_pack: self.weight_pack.slice(0..self.weight_pack.len())?, input_pack: self.input_pack.slice(0..self.input_pack.len())? })
    }
}
struct Attention<'a> { scores: DeviceVector<'a>, parts: DeviceVector<'a>, tiles: DeviceVector<'a> }
struct Head<'a> {
    capacity: usize, input: DeviceVector<'a>, logits: DeviceVector<'a>, partials: DeviceVector<'a>, result: DeviceVector<'a>,
}
struct DraftScratch<'a> { ids: DeviceVector<'a>, result: DeviceVector<'a> }

/// Reusable batch workspace. Holds no sessions or duplicate weights. Dropping a
/// session between calls releases its cache normally; results preserve row order.
pub struct Multi<'m, 'a, 'b> {
    graphs: Vec<(usize, usize, u64, DecodeGraph<'a>)>,
    model: &'m Model<'a, 'b>, max_rows: usize, activations: Option<Activations<'a>>,
    attention: Option<Attention<'a>>, head: Option<Head<'a>>, peak_bytes: usize, reserved: bool,
    moe: Option<MoeBatchScratch<'a>>, profile: bool, cohort_steps: usize, profile_position: usize, previous_step: Option<Instant>,
    profile_experts: Option<Vec<(usize, usize)>>, draft: Option<DraftScratch<'a>>,
    top_k: Option<spark_gpu_probe::driver::TopKScratch<'a>>, rejected: Vec<usize>,
}
impl<'m, 'a, 'b> Multi<'m, 'a, 'b> {
    pub fn new(model: &'m Model<'a, 'b>, max_rows: usize) -> Result<Self> {
        if !(1..=2048).contains(&max_rows) { return Err("multi row budget must be in 1..=2048".into()); }
        if model.device.embedding.is_none() || model.head.is_none() { return Err("multi requires a complete model".into()); }
        model.backend.require_batch_kernels()?;
        model.backend.require_multi_kernels()?;
        Ok(Self { graphs: Vec::new(), model, max_rows, activations: None, attention: None, head: None, peak_bytes: model.backend.allocated_bytes(), reserved: false,
            moe: None, profile: std::env::var("SPARK_PROFILE").as_deref() == Ok("1"), cohort_steps: 0, profile_position: 0, previous_step: None,
            profile_experts: None, draft: None, top_k: None, rejected: Vec::new() })
    }
    pub fn max_rows(&self) -> usize { self.max_rows }
    /// Requests of the last step or prefill whose router rows were rejected
    /// (non-finite scores, MOE_ROUTING_ERROR). They returned no rows and kept
    /// their length; every other request of that batch committed normally.
    pub fn rejected(&self) -> &[usize] { &self.rejected }
    /// Draft weights are read once per depth for the whole ready cohort.
    /// Each child session retains its own KV; target rejection truncates both.
    pub fn propose(&mut self, requests: &mut [DraftRows<'_, '_, 'a, 'b>])
        -> BatchResult<Vec<crate::draft::ModelProposal>> {
        if requests.is_empty() { return Ok(Vec::new()); }
        let catchup = std::time::Instant::now();
        for (i, r) in requests.iter_mut().enumerate() {
            if r.history.len() != r.session.len { return Err(Error::at(i, "draft prefix differs from target")); }
            let shared_limit = if r.session.config().num_experts != 0 { 256 } else { 64 };
            let draft = r.session.draft.as_mut().ok_or_else(|| Error::at(i, "missing draft model"))?;
            if !std::ptr::eq(draft.model, self.model) || r.limit == 0 || r.limit > r.session.draft_k {
                return Err(Error::at(i, "invalid shared draft model or width"));
            }
            if draft.len > r.history.len() { draft.truncate(r.history.len()).map_err(|e| Error::at(i, e))?; }
            let missing = &r.history[draft.len..];
            // Short prefixes share one prefill weight pass per 64-token chunk.
            // Only long histories need private catchup outside that workspace.
            if missing.len() > shared_limit {
                let started = Instant::now();
                draft.prefill_advance(missing).map_err(|e| Error::at(i, e))?;
                draft.model.observe_draft_catchup(missing.len(), started.elapsed().as_secs_f64()*1000.0);
                draft.device.as_mut().unwrap().release_batch();
            }
        }
        // Short prefixes share prefill. Short bonus catchup is fused
        // with the first proposal, saving a full draft weight pass.
        loop {
            let mut rows: Vec<_> = requests.iter_mut().filter_map(|r| {
                let draft = r.session.draft.as_mut().unwrap();
                let missing = &r.history[draft.len..];
                (missing.len() > 8).then(|| Rows { session: &mut **draft,
                    tokens: &missing[..missing.len().min(64)], output: Output::None })
            }).collect();
            if rows.is_empty() { break; }
            self.prefill(&mut rows)?;
            if !self.rejected.is_empty() { return Err(MOE_ROUTING_ERROR.to_owned().into()); }
        }
        let catchup_ms = catchup.elapsed().as_secs_f64()*1000.0;
        let started = std::time::Instant::now();
        let mut proposals: Vec<Vec<u32>> = requests.iter().map(|r| Vec::with_capacity(r.limit)).collect();
        // Greedy proposals avoid a host sampling boundary at every depth.
        // Verification still draws with the request's exact target sampler;
        // a proposal distribution never changes the committed token sequence.
        let b = self.model.backend;
        let phase_before = b.stats();
        let total: usize = requests.iter().map(|r| r.limit).sum();
        if self.draft.as_ref().is_none_or(|s| s.ids.len() < requests.len() || s.result.len() < total*3) {
            self.graphs.clear();
            let mut buffers = b.vector_arena(&[requests.len().next_power_of_two(), (total*3).next_power_of_two()])?.into_iter();
            self.draft = Some(DraftScratch { ids: buffers.next().unwrap(), result: buffers.next().unwrap() });
        }
        // Allocate and reserve before queueing the chain: neither depth changes
        // nor KV growth may insert a synchronization between selected IDs.
        for r in requests.iter_mut() {
            let draft = r.session.draft.as_mut().unwrap();
            draft.prepare_device()?;
            draft.device.as_mut().unwrap().reserve_cache(self.model, draft.len, r.history.len()+r.limit)?;
        }
        let c = &self.model.config;
        let needed = b.multi_attention_scratch(requests.iter().flat_map(|r| {
            let draft = r.session.draft.as_ref().unwrap();
            std::iter::repeat_n((r.history.len()+r.limit, draft.kv_format()), r.history.len()-draft.len+1)
        }), c.attention_heads as usize, c.kv_heads as usize, c.head_dim as usize);
        if self.attention.as_ref().is_none_or(|s| s.scores.len() < needed[0] || s.parts.len() < needed[1] || s.tiles.len() < needed[2]) {
            // Growing score storage must retain the reservation for later
            // prompt catchup and wider cohorts in the other two buffers.
            let retained = self.attention.as_ref().map_or([0; 3], |s|
                [s.scores.len(), s.parts.len(), s.tiles.len()]);
            let needed = std::array::from_fn::<_, 3, _>(|i| needed[i].max(retained[i]).next_power_of_two());
            self.graphs.clear();
            self.attention = None;
            let mut buffers = b.vector_arena(&needed)?.into_iter();
            self.attention = Some(Attention { scores: buffers.next().unwrap(), parts: buffers.next().unwrap(), tiles: buffers.next().unwrap() });
        }
        let scratch = self.draft.as_ref().unwrap();
        let pending = scratch.ids.slice(0..requests.len())?;
        let result = scratch.result.slice(0..total*3)?;
        let mut previous_indices = Vec::new();
        let mut previous = None;
        let mut order = Vec::with_capacity(total);
        let mut offset = 0;
        for depth in 0..requests.iter().map(|r| r.limit).max().unwrap_or(0) {
            let indices: Vec<_> = requests.iter().enumerate().filter_map(|(i, r)| (depth < r.limit).then_some(i)).collect();
            let inputs: Vec<Vec<u32>> = requests.iter().map(|r| {
                let mut tokens = if depth == 0 { r.history[r.session.draft.as_ref().unwrap().len..].to_vec() } else { Vec::new() };
                tokens.push(r.next);
                tokens
            }).collect();
            let selected = if let Some(previous) = &previous {
                let mapping: Vec<_> = indices.iter().map(|i| previous_indices.iter().position(|j| i == j).unwrap()).collect();
                let mut ids = pending.slice(0..indices.len())?;
                b.selected_ids_device(previous, &mapping, &mut ids)?;
                Some(ids)
            } else { None };
            let mut rows: Vec<_> = requests.iter_mut().zip(&inputs).filter_map(|(r, tokens)| {
                if depth >= r.limit { return None; }
                Some(Rows { session: &mut **r.session.draft.as_mut().unwrap(), tokens, output: Output::Last })
            }).collect();
            // Draft decode shares the canonical order too; a shrinking cohort
            // must not change a model proposal's arithmetic.
            let before = self.start_profile(&rows)?;
            let outputs = self.forward(&mut rows, false, true, selected.as_ref())?;
            self.finish_profile(&rows, before)?;
            for r in rows { r.session.len += r.tokens.len(); }
            let head = self.head.as_ref().unwrap();
            let v = self.model.config.vocab_size as usize;
            let mut chosen = result.slice(offset*3..(offset+outputs)*3)?;
            b.enqueue_argmax_batch_device(&head.logits.slice(0..outputs*v)?,
                &mut head.partials.slice(0..outputs*v.div_ceil(1024)*3)?, &mut chosen, outputs)?;
            offset += outputs;
            order.extend_from_slice(&indices);
            previous = Some(chosen);
            previous_indices = indices;
        }
        // Validate by row so one invalid draft does not fail the whole cohort.
        // A router-rejected draft row only shapes a proposal the target verifies.
        let values = b.read_vector_values(&result)?;
        let vocab = u32::try_from(self.model.config.vocab_size).map_err(|_| "draft vocabulary exceeds kernel ABI".to_owned())?;
        for (i, row) in order.into_iter().zip(values.chunks_exact(3)) {
            let token = spark_gpu_probe::kernel_params::id_index(row[1], vocab);
            if row[2] != 1.0 || !row[0].is_finite() || token.is_none() {
                return Err(Error::at(i, "invalid draft logits or token ID"));
            }
            proposals[i].push(token.unwrap() as u32);
        }
        let proposal_ms = started.elapsed().as_secs_f64()*1000.0;
        if self.profile { eprintln!("draft_proposal_profile chats={} wall_ms={proposal_ms:.3} stats={:?}", requests.len(), b.stats().since(phase_before)); }
        let depth = requests.iter().map(|r| r.limit).max().unwrap_or(1).max(1) as f64;
        let count = requests.len().max(1) as f64;
        Ok(requests.iter_mut().zip(proposals).map(|(r, tokens)| {
            let draft = r.session.draft.as_ref().unwrap();
            r.session.draft_kv_peak = r.session.draft_kv_peak.max(draft.cache_bytes());
            // Width selection pays for the entire cohort's next depth, not
            // 1/chats of it. Prefix setup remains separate from depth cost.
            crate::draft::ModelProposal { tokens, proposal_ms: proposal_ms*r.limit as f64/depth,
                catchup_ms: catchup_ms/count }
        }).collect())
    }
    /// Reserve a fixed serving workspace before admitting any sessions.
    pub fn reserve(&mut self, context: usize, outputs: usize, prefill: bool) -> Result<usize> {
        if self.activations.is_some() || self.attention.is_some() || self.head.is_some() {
            return Err("reserve multi workspace before its first execution".into());
        }
        if context == 0 || context > self.model.max_tokens() || outputs == 0 || outputs > self.max_rows {
            return Err("invalid multi workspace reservation".into());
        }
        let b = self.model.backend;
        let c = &self.model.config;
        let before = b.allocated_bytes();
        self.activations = Some(Activations::new(self.model, self.max_rows.next_power_of_two().max(16), prefill)?);
        self.reserve_experts(self.max_rows.next_power_of_two().max(16))?;
        // The partition count peaks just before the long-context span changes.
        let scratch = |length| b.multi_attention_scratch(std::iter::repeat_n((length, spark_gpu_probe::driver::KvFormat::Fp32), self.max_rows),
            c.attention_heads as usize, c.kv_heads as usize, c.head_dim as usize);
        let end = scratch(context);
        let boundary = scratch(context.min(16384));
        let short = scratch(context.min(4096));
        let needed = if prefill && c.head_dim == 128 { [1, 1, self.max_rows * 2] }
            else { std::array::from_fn::<_, 3, _>(|i| end[i].max(boundary[i]).max(short[i]).next_power_of_two()) };
        let mut buffers = b.vector_arena(&needed)?.into_iter();
        self.attention = Some(Attention { scores: buffers.next().unwrap(), parts: buffers.next().unwrap(), tiles: buffers.next().unwrap() });
        let (h, v) = (c.hidden_size as usize, c.vocab_size as usize);
        let mut buffers = b.vector_arena(&[outputs*h, outputs*v, outputs*v.div_ceil(1024)*3, outputs*3])?.into_iter();
        self.head = Some(Head { capacity: outputs, input: buffers.next().unwrap(), logits: buffers.next().unwrap(),
            partials: buffers.next().unwrap(), result: buffers.next().unwrap() });
        self.reserved = true;
        self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
        Ok(b.allocated_bytes() - before)
    }
    /// Shared short catchup and proposal summaries persist between cohorts.
    /// Measure them at startup so shrinking cohorts cannot strand uncharged
    /// workspace. Proposals have at most eight bonus catchup rows plus one.
    pub fn reserve_draft(&mut self, sessions: usize, depth: usize) -> Result<usize> {
        if !(1..=32).contains(&sessions) || !(1..=8).contains(&depth) {
            return Err("invalid draft workspace dimensions".into());
        }
        let b = self.model.backend;
        let c = &self.model.config;
        let before = b.allocated_bytes();
        // Shared prompt catchup is at most 64 rows. Full-context BF16
        // proposals use streamed attention, not the FP32 score matrix.
        self.reserve(self.model.max_tokens().min(64), sessions, true)?;
        // A proposal can include bonus catchup rows, exceeding sixteen even
        // with sixteen sessions. Exact decode then needs both FFN outputs;
        // the prefill-only arena omits up when its fused GEMM handles it.
        self.activations = None;
        self.activations = Some(Activations::new(self.model, self.max_rows.next_power_of_two().max(16), false)?);
        self.attention = None;
        let needed = b.multi_attention_scratch(std::iter::repeat_n(
            (self.model.max_tokens(), spark_gpu_probe::driver::KvFormat::Bf16), sessions * 9),
            c.attention_heads as usize, c.kv_heads as usize, c.head_dim as usize);
        let mut buffers = b.vector_arena(&needed.map(usize::next_power_of_two))?.into_iter();
        self.attention = Some(Attention { scores: buffers.next().unwrap(),
            parts: buffers.next().unwrap(), tiles: buffers.next().unwrap() });
        let mut buffers = b.vector_arena(&[sessions.next_power_of_two(),
            (sessions * depth * 3).next_power_of_two()])?.into_iter();
        self.draft = Some(DraftScratch { ids: buffers.next().unwrap(), result: buffers.next().unwrap() });
        self.peak_bytes = self.peak_bytes.max(b.allocated_bytes());
        Ok(b.allocated_bytes() - before)
    }
    /// Peak tracked device allocations, including model, all live sessions and
    /// transient copy-on-write KV growth; excludes driver/host allocations.
    pub fn device_peak_bytes(&self) -> usize { self.peak_bytes }

    fn reserve_experts(&mut self, rows: usize) -> Result<()> {
        self.moe = None;
        self.moe = self.model.layers.iter().find_map(|layer| match &layer.mlp {
            Mlp::Experts { experts, .. } => Some(self.model.backend.moe_batch_scratch(
                experts, self.model.config.num_experts_per_tok as usize, rows)),
            _ => None,
        }).transpose()?;
        Ok(())
    }

    /// Sample early and late cohort steps as their expert routes diverge.
    fn start_profile(&mut self, requests: &[Rows<'_, '_, 'a, 'b>]) -> Result<Option<(Instant, f64, BackendStats)>> {
        let gap_ms = self.previous_step.map_or(0.0, |at| at.elapsed().as_secs_f64()*1000.0);
        if !self.profile || !matches!(requests.len(), 4 | 8 | 16) { return Ok(None); }
        let position = requests.iter().map(|r| r.session.len).min().unwrap();
        if position < self.profile_position || position > self.profile_position + self.max_rows {
            self.cohort_steps = 0;
        }
        self.profile_position = position;
        self.cohort_steps += 1;
        if !matches!(self.cohort_steps, 8 | 64 | 120) { return Ok(None); }
        let b = self.model.backend;
        b.synchronize()?;
        b.set_profiling(true);
        if self.model.config.num_experts != 0 { self.profile_experts = Some(Vec::with_capacity(self.model.layers.len())); }
        Ok(Some((Instant::now(), gap_ms, b.stats())))
    }

    fn finish_profile(&mut self, requests: &[Rows<'_, '_, 'a, 'b>], before: Option<(Instant, f64, BackendStats)>) -> Result<()> {
        if self.profile { self.previous_step = Some(Instant::now()); }
        let Some((start, gap_ms, before)) = before else { return Ok(()); };
        let b = self.model.backend;
        b.synchronize()?;
        b.set_profiling(false);
        let routing = self.profile_experts.take().unwrap_or_default();
        let plain_experts: Vec<_> = routing.iter().map(|r| r.0).collect();
        let experts: Vec<_> = routing.iter().map(|r| r.1).collect();
        let c = &self.model.config;
        let expert_bytes = experts.iter().sum::<usize>() as u64 * 6 * c.hidden_size * c.moe_intermediate_size;
        let plain_expert_bytes = plain_experts.iter().sum::<usize>() as u64 * 6 * c.hidden_size * c.moe_intermediate_size;
        let kv_bytes = requests.iter().map(|r| r.session.len as u64).sum::<u64>() * c.layers * c.kv_heads * c.head_dim * 4;
        eprintln!("multi_decode_profile layers={} hidden={} chats={} rows={} position={} wall_ms={:.3} previous_gap_ms={gap_ms:.3} kv_read_bytes={kv_bytes} distinct_experts_per_layer={experts:?} expert_weight_bytes={expert_bytes} plain_experts_per_layer={plain_experts:?} plain_expert_weight_bytes={plain_expert_bytes} added_expert_weight_bytes={} device_bytes={} peak_device_bytes={} pinned_host_bytes={} stats={:?}",
            c.layers, c.hidden_size, requests.len(), requests.iter().map(|r| r.tokens.len()).sum::<usize>(), requests[0].session.len, start.elapsed().as_secs_f64()*1000.0,
            expert_bytes - plain_expert_bytes, b.allocated_bytes(), self.peak_bytes, b.pinned_host_bytes(), b.stats().since(before));
        Ok(())
    }

    /// Results preserve request order, then token order. All lengths commit only
    /// after the complete batch succeeds. On error, uncommitted KV is overwritten
    /// by a retry. For speculative rejection, truncate each session separately.
    pub fn step(&mut self, requests: &mut [Rows<'_, '_, 'a, 'b>]) -> BatchResult<Vec<Vec<Vec<f32>>>> {
        let before = self.start_profile(requests)?;
        let result = self.run(requests, false);
        self.finish_profile(requests, before)?;
        result
    }
    /// Independent prompts share tensor-core weight passes. Only prompt-end
    /// logits are projected; attention retains each request's own causal cache.
    /// Unreserved callers release prompt scratch after the pass.
    pub fn prefill(&mut self, requests: &mut [Rows<'_, '_, 'a, 'b>]) -> BatchResult<Vec<Vec<Vec<f32>>>> {
        for (i, request) in requests.iter().enumerate() {
            if request.output == Output::All { return Err(Error::at(i, "prefill accepts Last or None output")); }
        }
        if !self.reserved { self.graphs.clear(); self.activations = None; }
        let start = self.profile.then(Instant::now);
        let b = self.model.backend;
        let before = if self.profile { b.synchronize()?; b.set_profiling(true); Some(b.stats()) } else { None };
        let result = self.run(requests, true);
        if before.is_some() { b.set_profiling(false); }
        if let Some(start) = start {
            eprintln!("multi_prefill_profile chats={} rows={} prefix_tokens={} wall_ms={:.3} device_bytes={} stats={:?}",
                requests.len(), requests.iter().map(|r| r.tokens.len()).sum::<usize>(),
                requests.iter().map(|r| r.session.len.saturating_sub(r.tokens.len())).min().unwrap_or(0),
                start.elapsed().as_secs_f64() * 1000.0, b.allocated_bytes(), b.stats().since(before.unwrap()));
        }
        if !self.reserved { self.activations = None; self.attention = None; self.moe = None; }
        result
    }
    fn run(&mut self, requests: &mut [Rows<'_, '_, 'a, 'b>], prefill: bool) -> BatchResult<Vec<Vec<Vec<f32>>>> {
        let outputs = self.forward(requests, prefill, !prefill, None)?;
        let values = if outputs == 0 { self.model.backend.synchronize()?; Vec::new() } else {
            self.model.backend.read_vector_values(&self.head.as_ref().unwrap().logits.slice(0..outputs*self.model.config.vocab_size as usize)?)?
        };
        let rejected = self.routing(requests)?;
        let v = self.model.config.vocab_size as usize;
        let rows: Vec<_> = values.chunks_exact(v).enumerate().filter(|&(i, _)| !rejected[output_request(requests, i)]).collect();
        if let Some(&(i, _)) = rows.iter().find(|(_, row)| row.iter().any(|v| !v.is_finite())) {
            return Err(Error::at(output_request(requests, i), "nonfinite device output"));
        }
        let mut rows = rows.into_iter().map(|(_, row)| row.to_vec());
        Ok(self.commit(requests, &rejected, |count| rows.by_ref().take(count).collect()))
    }
    /// Greedy decode/verify downloads only the chosen tokens and their scores.
    /// Use step plus Sampler for token filters, penalties or nonzero temperature.
    pub fn step_greedy(&mut self, requests: &mut [Rows<'_, '_, 'a, 'b>]) -> BatchResult<Vec<Vec<(u32, f32)>>> {
        let before = self.start_profile(requests)?;
        let result = self.run_greedy(requests);
        self.finish_profile(requests, before)?;
        result
    }
    fn run_greedy(&mut self, requests: &mut [Rows<'_, '_, 'a, 'b>]) -> BatchResult<Vec<Vec<(u32, f32)>>> {
        let outputs = self.forward(requests, false, true, None)?;
        let values = if outputs == 0 { self.model.backend.synchronize()?; Vec::new() } else {
            let head = self.head.as_ref().unwrap();
            let v = self.model.config.vocab_size as usize;
            self.model.backend.argmax_batch_device(&head.logits.slice(0..outputs*v)?,
                &mut head.partials.slice(0..outputs*v.div_ceil(1024)*3)?, &mut head.result.slice(0..outputs*3)?, outputs)?
        };
        let rejected = self.routing(requests)?;
        let values = values.into_iter().enumerate().filter(|&(i, _)| !rejected[output_request(requests, i)])
            .map(|(i, value)| value.map_err(|e| Error::at(output_request(requests, i), e))).collect::<BatchResult<Vec<_>>>()?;
        let mut rows = values.into_iter();
        Ok(self.commit(requests, &rejected, |count| rows.by_ref().take(count).collect()))
    }
    /// Exact top-k selection leaves all probability math in the host sampler.
    pub fn step_top_k(&mut self, requests: &mut [Rows<'_, '_, 'a, 'b>], k: usize)
        -> BatchResult<Vec<Vec<Vec<(u32, f32)>>>> {
        let before = self.start_profile(requests)?;
        let result = (|| {
            let outputs = self.forward(requests, false, true, None)?;
            if outputs == 0 { return Err("top-k requires output rows".to_owned().into()); }
            let v = self.model.config.vocab_size as usize;
            let values = self.model.backend.top_k_device(&self.head.as_ref().unwrap().logits.slice(0..outputs*v)?,
                &mut self.top_k, outputs, k)?;
            let rejected = self.routing(requests)?;
            self.peak_bytes = self.peak_bytes.max(self.model.backend.allocated_bytes());
            let values = values.into_iter().enumerate().filter(|&(i, _)| !rejected[output_request(requests, i)])
                .map(|(i, value)| value.map_err(|e| Error::at(output_request(requests, i), e))).collect::<BatchResult<Vec<_>>>()?;
            let mut rows = values.into_iter();
            Ok(self.commit(requests, &rejected, |count| rows.by_ref().take(count).collect()))
        })();
        self.finish_profile(requests, before)?;
        result
    }
    /// After a forward's output read: requests owning a router row rejected for
    /// non-finite scores. Rows are independent, so the others stay exact.
    fn routing(&mut self, requests: &[Rows<'_, '_, 'a, 'b>]) -> BatchResult<Vec<bool>> {
        let mut rejected = vec![false; requests.len()];
        let rows: usize = requests.iter().map(|r| r.tokens.len()).sum();
        if let Some(moe) = &self.moe {
            for row in self.model.backend.moe_batch_rejected(moe, rows)? {
                rejected[input_request(requests, row).ok_or_else(|| "router status row outside the batch".to_owned())?] = true;
            }
        }
        self.rejected = (0..requests.len()).filter(|&i| rejected[i]).collect();
        Ok(rejected)
    }
    /// Advance and return each surviving request; a rejected one keeps its
    /// length (its uncommitted KV is overwritten later) and returns no rows.
    fn commit<T>(&self, requests: &mut [Rows<'_, '_, 'a, 'b>], rejected: &[bool],
        mut take: impl FnMut(usize) -> Vec<T>) -> Vec<Vec<T>> {
        requests.iter_mut().zip(rejected).map(|(r, &rejected)| {
            if rejected { return Vec::new(); }
            r.session.len += r.tokens.len();
            take(r.output_rows())
        }).collect()
    }
    fn forward(&mut self, requests: &mut [Rows<'_, '_, 'a, 'b>], prefill: bool, exact: bool,
        selected_ids: Option<&DeviceVector<'a>>) -> BatchResult<usize> {
        let b = self.model.backend;
        b.generic_profile_begin(prefill, || (requests.first().map_or(0, |r| r.session.len), requests.iter().map(|r| r.tokens.len()).sum(),
            engine_model::plan::profile::ProfilePlan::qwen(&self.model.config).classes(), 1))?;
        let result = self.forward_profiled(requests, prefill, exact, selected_ids);
        b.generic_profile_end(prefill, result.is_ok())?; result
    }
    fn forward_profiled(&mut self, requests: &mut [Rows<'_, '_, 'a, 'b>], prefill: bool, exact: bool,
        selected_ids: Option<&DeviceVector<'a>>) -> BatchResult<usize> {
        if requests.is_empty() { return Ok(0); }
        let _order = self.model.backend.prompt_order(prefill);
        let mut rows = 0;
        let mut outputs = 0;
        for (i, r) in requests.iter().enumerate() {
            if !std::ptr::eq(r.session.model, self.model) { return Err(Error::at(i, "sessions must belong to the batch model")); }
            r.session.check_batch(r.tokens).map_err(|e| Error::at(i, e))?;
            rows += r.tokens.len();
            outputs += r.output_rows();
            if rows > self.max_rows { return Err(Error::at(i, "multi row budget exceeded")); }
        }
        let m = self.model;
        let b = m.backend;
        let c = &m.config;
        // Every decode row, including draft chains and experts, has one MMA order.
        let exact = !prefill && (exact || c.num_experts != 0);
        let (h, v) = (c.hidden_size as usize, c.vocab_size as usize);
        let heads = c.attention_heads as usize;
        let eps = c.norm_epsilon();
        for (i, r) in requests.iter_mut().enumerate() {
            r.session.prepare_device().map_err(|e| Error::at(i, e))?;
            let state = r.session.device.as_mut().unwrap();
            // A committed long-prompt tail can join this shared prompt pass.
            // Its old chunk workspace has no live progress or useful rows.
            if prefill { state.release_batch(); }
            state.reserve_multi_cache(m, r.session.len, r.session.len+r.tokens.len()).map_err(|e| Error::at(i, e))?;
            self.peak_bytes = self.peak_bytes.max(state.peak_bytes);
        }
        if self.activations.as_ref().is_none_or(|s| s.capacity < rows || !exact && s.input_pack.len() == 1) {
            if self.reserved { return Err("reserved multi activation capacity exceeded".to_owned().into()); }
            self.graphs.clear();
            self.activations = None;
            let capacity = if prefill { rows.next_power_of_two().max(16) } else { rows.next_power_of_two().min(self.max_rows) };
            self.activations = Some(Activations::new(m, capacity, prefill)?);
            self.reserve_experts(capacity)?;
        }
        // BF16 prompt attention reads KV directly and needs no generic score
        // or merge workspace, including short suffixes of long cached prompts.
        let direct_prefill = prefill && c.head_dim == 128
            && requests.iter().all(|r| r.session.kv_bf16());
        // A speculative cohort occupies several 16-row tiles. Reuse the
        // canonical fused decode projections across all of those tiles.
        let packed_decode = c.qk_norm && rows <= 64 && c.hidden_size % 256 == 0 && c.intermediate_size % 256 == 0;
        let fused_norms = c.qk_norm && (direct_prefill || exact && packed_decode) && c.num_experts == 0
            && c.hidden_size % 128 == 0 && c.intermediate_size % 128 == 0;
        let lengths = requests.iter().flat_map(|r| (r.session.len+1..=r.session.len+r.tokens.len()).map(|n| (n, r.session.kv_format())));
        let needed = if direct_prefill { [1, 1, rows * 2] }
            else { b.multi_attention_scratch(lengths, heads, c.kv_heads as usize, c.head_dim as usize) };
        if self.attention.as_ref().is_none_or(|s| s.scores.len() < needed[0] || s.parts.len() < needed[1] || s.tiles.len() < needed[2]) {
            if self.reserved { return Err("reserved multi attention capacity exceeded".to_owned().into()); }
            self.graphs.clear();
            self.attention = None;
            let mut buffers = b.vector_arena(&needed.map(usize::next_power_of_two))?.into_iter();
            self.attention = Some(Attention { scores: buffers.next().unwrap(), parts: buffers.next().unwrap(), tiles: buffers.next().unwrap() });
        }
        if outputs > 0 && self.head.as_ref().is_none_or(|s| s.capacity < outputs) {
            if self.reserved { return Err("reserved multi output capacity exceeded".to_owned().into()); }
            self.graphs.clear();
            self.head = None;
            let capacity = outputs.next_power_of_two().min(self.max_rows);
            let mut buffers = b.vector_arena(&[capacity*h, capacity*v, capacity*v.div_ceil(1024)*3, capacity*3])?.into_iter();
            self.head = Some(Head { capacity, input: buffers.next().unwrap(), logits: buffers.next().unwrap(),
                partials: buffers.next().unwrap(), result: buffers.next().unwrap() });
        }
        let peak = b.allocated_bytes();
        self.peak_bytes = self.peak_bytes.max(peak);
        for r in requests.iter_mut() {
            let state = r.session.device.as_mut().unwrap();
            state.peak_bytes = state.peak_bytes.max(peak);
        }
        let expert_norms = prefill || packed_decode;
        let mut s = self.activations.as_ref().unwrap().rows(rows,
            c.qk_norm && (!exact || packed_decode) && c.num_experts == 0 && c.hidden_size % 128 == 0 && c.intermediate_size % 128 == 0,
            if c.num_experts == 0 { c.intermediate_size as usize } else { 1 })?;
        let a = self.attention.as_ref().unwrap();
        let ids = if let Some(ids) = selected_ids { ids.slice(0..rows)? } else {
            let ids: Vec<_> = requests.iter().flat_map(|r| r.tokens.iter().map(|&id| id as f32)).collect();
            b.write_vector(&mut s.ids, &ids)?;
            s.ids.slice(0..rows)?
        };
        // Stable draft depths, including one session, replay the ordinary kernels.
        // Positions and cache pointers are uploaded before capture, never baked in.
        let captured_draft = !prefill && exact && packed_decode && !b.is_profiling()
            && c.num_experts == 0
            && (1..=64).contains(&rows) && outputs > 0
            && heads == c.kv_heads as usize * 2 && c.head_dim == 128
            && requests.iter().all(|r| r.session.kv_bf16() && r.session.len+r.tokens.len() <= 512);
        if captured_draft {
            for i in 0..m.layers.len() {
                let caches = requests.iter_mut().map(|r| r.session.device.as_mut().unwrap()
                    .cache_rows(i, r.session.len, r.tokens.len())).collect::<Result<Vec<_>>>()?;
                b.prepare_draft_metadata(&caches, &mut s.metadata.slice(i*rows*16..(i+1)*rows*16)?, heads, c.kv_heads as usize)?;
            }
            let tiles: Vec<_> = (0..rows).flat_map(|r| [f32::from_bits(r as u32), 0.0]).collect();
            b.write_vector(&mut a.tiles.slice(0..rows*2)?, &tiles)?;
        }
        if captured_draft && outputs != rows {
            let (mut source, mut indices) = (0, Vec::with_capacity(outputs));
            for r in requests.iter() {
                let start = source + if r.output == Output::Last { r.tokens.len()-1 } else { 0 };
                indices.extend((start..start+r.output_rows()).map(|i| i as f32));
                source += r.tokens.len();
            }
            b.write_vector(&mut self.head.as_ref().unwrap().result.slice(0..outputs)?, &indices)?;
        }
        let first_experts = m.layers.iter().position(|layer| matches!(layer.mlp, Mlp::Experts { .. }));
        let mut enqueue = || -> Result<()> {
        b.batch_embedding_device(m.device.embedding.as_ref().unwrap(), &ids, &mut s.hidden)?;
        for (i, layer) in m.layers.iter().enumerate() {
            m.profile_layer(0, spark_gpu_probe::driver::KvFormat::Bf16);
            b.generic_read_bytes(|| (2, requests.iter().map(|r| 2 * r.session.kv_format().storage_len(
                (r.session.len + r.tokens.len()) * m.kv_capacity.stride) * 4).sum()));
            let (input_norm, post_norm) = m.device.layer_norms(i);
            if c.qk_norm && (!exact || packed_decode) && c.hidden_size % 128 == 0 && layer.q.rows() % 128 == 0 && layer.k.rows() % 128 == 0 {
                if i > 0 && (fused_norms || expert_norms && matches!(m.layers[i - 1].mlp, Mlp::Experts { .. })) {
                    b.rms_pack_apply_device(&s.hidden, input_norm, &mut s.input_pack, &s.sums, eps, false)?;
                } else {
                    b.rms_pack_device(&s.hidden, input_norm, &mut s.input_pack, &mut s.sums, eps, false)?;
                }
                b.profile_phase(14, || b.prefill_qkv_device(&layer.q, &layer.k, &layer.v, &s.input_pack,
                    &mut s.q, &mut s.k, &mut s.v, rows, false))?;
            } else {
                b.rms_batch_device(&s.hidden, input_norm, &mut s.input, &mut s.sums, eps)?;
                b.profile_phase(14, || {
                    b.matmul_device(&layer.q, &s.input, &mut s.q, &mut s.weight_pack, &mut s.input_pack, rows, exact)?;
                    b.matmul_reusing_input(&layer.k, &s.input, &mut s.k, &mut s.weight_pack, &mut s.input_pack, rows, exact, true)?;
                    b.matmul_reusing_input(&layer.v, &s.input, &mut s.v, &mut s.weight_pack, &mut s.input_pack, rows, exact, true)
                })?;
            }
            let packed_attention = prefill && c.num_experts != 0 && c.head_dim == 128
                && heads == c.kv_heads as usize * 8 && requests.len() == 1 && rows >= 512
                && !requests[0].session.device.as_ref().unwrap().split_cache()
                && requests[0].session.kv_format() == spark_gpu_probe::driver::KvFormat::Bf16
                && (requests[0].session.len + rows).div_ceil(64) * 64 * c.kv_heads as usize * 64
                    <= s.weight_pack.len().min(s.input_pack.len());
            if captured_draft {
                let (q_norm,k_norm,rope) = m.device.attention_inputs(i);
                b.draft_attention_prepared_device(&s.q, &s.k, &s.v, q_norm, k_norm, rope,
                    &s.metadata.slice(i*rows*16..(i+1)*rows*16)?, &a.tiles,
                    &s.attention, heads, c.kv_heads as usize, rows, eps)?;
            } else {
                let mut caches = requests.iter_mut().map(|r| r.session.device.as_mut().unwrap()
                    .cache_rows(i, r.session.len, r.tokens.len())).collect::<Result<Vec<_>>>()?;
                let (q_norm,k_norm,rope) = m.device.attention_inputs(i);
                if prefill && c.head_dim == 128
                    && caches.iter().all(|cache| cache.format == spark_gpu_probe::driver::KvFormat::Bf16) {
                    // Keep prompt QK/PV at checkpoint precision, as Session::prefill does.
                    let mut row = 0;
                    for cache in &mut caches {
                        let end = row + cache.rows;
                        let mut query = s.q.slice(row*heads*128..end*heads*128)?;
                        b.qk_batch_cached_device(&mut query, &s.k.slice(row*c.kv_heads as usize*128..end*c.kv_heads as usize*128)?,
                            &s.v.slice(row*c.kv_heads as usize*128..end*c.kv_heads as usize*128)?,
                            q_norm, k_norm, rope, cache.keys, cache.values, cache.position, c.qk_epsilon(), cache.rows, cache.format,
                            cache.prefix.map_or(0, |(_, _, base)| base))?;
                        // Packed expert output keeps swizzled KV. Other rows
                        // share direct KV with their model's softmax arithmetic.
                        if c.num_experts != 0 && packed_attention {
                            let mut attention = s.attention.slice(row*heads*128..end*heads*128)?;
                            b.prefill_attention_output_packed_device(&query, cache.keys, cache.values,
                                &mut attention, &mut s.weight_pack, &mut s.input_pack, heads, c.kv_heads as usize,
                                cache.position, cache.rows)?;
                        }
                        row = end;
                    }
                    if !packed_attention {
                        b.prefill_attention_multi_device(&s.q, &caches, &mut s.metadata.slice(0..rows*16)?, &mut a.tiles.slice(0..needed[2])?,
                            &mut s.attention, heads, c.kv_heads as usize)?;
                    }
                } else {
                    b.attention_multi_device(&mut s.q, &s.k, &s.v, q_norm, k_norm, rope, &mut caches, &mut s.metadata.slice(0..rows*16)?, &mut a.tiles.slice(0..needed[2])?,
                        &mut a.scores.slice(0..needed[0])?, &mut a.parts.slice(0..needed[1])?,
                        &mut s.attention, heads, c.kv_heads as usize, c.head_dim as usize, c.qk_epsilon())?;
                }
            }
            b.profile_phase(15, || {
                if packed_attention { b.matmul_packed_device(&layer.o, &s.attention, &mut s.projected, rows) }
                else { b.matmul_device(&layer.o, &s.attention, &mut s.projected, &mut s.weight_pack, &mut s.input_pack, rows, exact) }
            })?;
            if let Mlp::Experts { router, experts } = &layer.mlp {
                b.residual_rms_batch_device(&mut s.hidden, &s.projected, post_norm, &mut s.input, &mut s.sums, eps)?;
                b.moe_batch_device(router, experts, &s.input, if expert_norms { &mut s.hidden } else { &mut s.projected },
                    self.moe.as_mut().ok_or_else(|| "missing multi expert scratch".to_owned())?,
                    &mut s.weight_pack, &mut s.input_pack, !prefill, c.norm_topk_prob, first_experts == Some(i),
                    if expert_norms { Some(&mut s.sums) } else { None })?;
                if let Some(counts) = &mut self.profile_experts {
                    let routing = b.moe_batch_routing(self.moe.as_ref().unwrap(), rows)?;
                    let selected = c.num_experts_per_tok as usize;
                    let plain = requests.iter().scan(0, |offset, r| {
                        let start = *offset;
                        *offset += r.tokens.len() * selected;
                        Some(&routing[start..start + selected])
                    }).flatten().map(|r| r.0).collect::<std::collections::BTreeSet<_>>().len();
                    counts.push((plain, routing.iter().map(|r| r.0).collect::<std::collections::BTreeSet<_>>().len()));
                }
            } else if c.qk_norm && (!exact || packed_decode) && c.hidden_size % 128 == 0 && c.intermediate_size % 128 == 0 {
                if fused_norms {
                    b.residual_rms_partials_device(&mut s.hidden, &s.projected, &mut s.sums, h)?;
                } else { b.residual_checkpoint_device(&mut s.hidden, &s.projected, !c.qk_norm)?; }
                let (gate, up, down) = layer.mlp.dense()?;
                if fused_norms {
                    b.rms_pack_apply_device(&s.hidden, post_norm, &mut s.input_pack, &s.sums, eps, false)?;
                } else { b.rms_pack_device(&s.hidden, post_norm, &mut s.input_pack, &mut s.sums, eps, false)?; }
                b.profile_phase(16, || b.prefill_gate_up_packed_device(gate, up, &s.input_pack, &mut s.gate, rows))?;
                b.profile_phase(17, || b.matmul_packed_device(down, &s.gate, &mut s.projected, rows))?;
            } else {
                let (gate, up, down) = layer.mlp.dense()?;
                b.residual_rms_batch_device(&mut s.hidden, &s.projected, post_norm, &mut s.input, &mut s.sums, eps)?;
                b.profile_phase(16, || {
                    b.matmul_device(gate, &s.input, &mut s.gate, &mut s.weight_pack, &mut s.input_pack, rows, exact)?;
                    b.matmul_reusing_input(up, &s.input, &mut s.up, &mut s.weight_pack, &mut s.input_pack, rows, exact, true)
                })?;
                b.swiglu_precision_device(&mut s.gate, &s.up, !c.qk_norm)?;
                b.profile_phase(17, || b.matmul_device(down, &s.gate, &mut s.projected, &mut s.weight_pack, &mut s.input_pack, rows, exact))?;
            }
            if fused_norms && i + 1 < m.layers.len() {
                b.residual_rms_partials_device(&mut s.hidden, &s.projected, &mut s.sums, h)?;
            } else if !expert_norms || matches!(layer.mlp, Mlp::Dense { .. }) {
                b.residual_checkpoint_device(&mut s.hidden, &s.projected, !c.qk_norm)?;
            }
        }
        if outputs > 0 {
            m.profile_head();
            b.rms_batch_device(&s.hidden, m.device.final_norm()?, &mut s.input, &mut s.sums, eps)?;
            let head = self.head.as_ref().unwrap();
            let input = if outputs == rows { s.input.slice(0..rows*h)? } else if captured_draft {
                let input = head.input.slice(0..outputs*h)?;
                b.gather_output_rows_device(&s.input, &head.result.slice(0..outputs)?, &input, h, outputs)?;
                input
            } else {
                let mut input = head.input.slice(0..outputs*h)?;
                let (mut source, mut destination) = (0, 0);
                for r in requests.iter() {
                    let count = r.output_rows();
                    if count > 0 {
                        let row = source + if r.output == Output::Last { r.tokens.len()-1 } else { 0 };
                        b.copy_strided(&s.input, &mut input, row*h, destination*h, h, h, h, count)?;
                        destination += count;
                    }
                    source += r.tokens.len();
                }
                input
            };
            // Session::project uses prompt accumulation for dense prefill and
            // the canonical decode reduction for the expert vocabulary head.
            b.profile_output_head(|| b.matmul_device(m.head.as_ref().unwrap(), &input, &mut head.logits.slice(0..outputs*v)?,
                &mut s.weight_pack, &mut s.input_pack, outputs, exact || prefill && c.num_experts != 0))?;
            b.round_logits_bf16(&mut head.logits.slice(0..outputs*v)?)?;
        }
        Ok(())
        };
        if captured_draft {
            let key = ids.device_address();
            let index = if let Some(index) = self.graphs.iter().position(|(count, out, input, _)| *count == rows && *out == outputs && *input == key) { index }
                else {
                    // CUDA retires an in-flight exec after completion; its
                    // captured scratch remains owned by this Multi.
                    if self.graphs.len() == 32 { self.graphs.remove(0); }
                    // Scratch lives with Multi; invalidate before any replacement.
                    let graph = unsafe { b.capture_batch(m.layers.len(), rows, &mut enqueue)? };
                    self.graphs.push((rows, outputs, key, graph));
                    self.graphs.len()-1
                };
            unsafe { b.replay_decode(&self.graphs[index].3)?; }
        } else { enqueue()?; }
        Ok(outputs)
    }
}
