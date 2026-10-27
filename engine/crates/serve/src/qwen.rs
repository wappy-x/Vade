//! Actual owned-engine adapter. No fake generation backend is shipped.
use spark_seqpar::coordinator::{Coordinator, Endpoint, Peer, Topology};
use std::{cell::RefCell, sync::Weak};

// Admission can exceed this cohort; prefill and draft reuse bounded scratch.
const WORKSPACE_SESSIONS: usize = 32;

/// Ordered persistent workers, followed by this server as the final path rank.
pub struct Seqpar {
    coordinator: Coordinator,
    peers: Vec<Peer>,
    threshold: usize,
    busy: RefCell<Weak<()>>,
}
impl Seqpar {
    pub fn new(listen: std::net::SocketAddr, peers: &[(std::net::SocketAddr, std::net::SocketAddr)],
        threshold: usize) -> Result<Self, String> {
        if peers.is_empty() || threshold == 0 {
            return Err("seqpar requires peers and a positive threshold".into());
        }
        let peers = peers.iter().enumerate().map(|(rank, &(control, data))| Peer {
            control,
            endpoint: Endpoint { listen: data, next: peers.get(rank + 1).map_or(listen, |p| p.1) },
        }).collect();
        Ok(Self { coordinator: Coordinator::new(Endpoint { listen, next: listen }, Topology::Path),
            peers, threshold, busy: RefCell::new(Weak::new()) })
    }
}

struct CoordinatedPrefill {
    pending: spark_seqpar::coordinator::Pending,
    peers: usize,
    tokens: usize,
    started: std::time::Instant,
}
impl crate::LongPrefill<spark_engine::qwen::Session<'_, '_, '_>> for CoordinatedPrefill {
    fn allows_fallback(&self, error: &str) -> bool {
        !error.starts_with("seqpar authentication failed:")
    }
    fn advance(&mut self, session: &mut spark_engine::qwen::Session<'_, '_, '_>, budget: std::time::Duration)
        -> Result<Option<Vec<f32>>, String> {
        let output = match self.pending.advance(session, budget) {
            Ok(Some(output)) => output,
            Ok(None) => return Ok(None),
            Err(error) => {
                // No ordinary prompt tokens have committed yet. Release the
                // incomplete layer work, retaining reusable KV allocations.
                let bytes = session.cache_bytes();
                session.reset();
                return Err(format!("{error}; elapsed_ms={:.3} partial_kv_bytes={bytes}",
                    self.started.elapsed().as_secs_f64() * 1000.0));
            }
        };
        eprintln!("seqpar_prefill rank={} world={} tokens={} elapsed_ms={:.3} kv_bytes={} peak_device_bytes={} host_exchange_bytes={}",
            self.peers, self.peers + 1, self.tokens, output.elapsed.as_secs_f64() * 1000.0,
            session.cache_bytes(), session.device_peak_bytes(), output.stats.host_exchange_bytes);
        output.stats.report("helper", self.peers);
        for peer in output.peers {
            eprintln!("seqpar_peer rank={} elapsed_ms={:.3} kv_bytes={} peak_device_bytes={}",
                peer.rank, peer.elapsed.as_secs_f64() * 1000.0, peer.cache_bytes, peer.peak_device_bytes);
        }
        Ok(Some(output.logits))
    }
}

pub struct Qwen<'m, 'a, 'b> {
    model: &'m spark_engine::qwen::Model<'a, 'b>,
    draft_model: Option<(&'m spark_engine::qwen::Model<'a, 'b>, usize)>,
    context_limit: usize,
    profile: bool,
    previous_decode: std::cell::Cell<Option<std::time::Instant>>,
    eos_tokens: [u32; 3],
    multi: std::cell::RefCell<Option<spark_engine::qwen::multi::Multi<'m, 'a, 'b>>>,
    draft_multi: std::cell::RefCell<Option<spark_engine::qwen::multi::Multi<'m, 'a, 'b>>>,
    seqpar: Option<Seqpar>,
    prefill: std::cell::RefCell<Option<spark_engine::qwen::multi::Multi<'m, 'a, 'b>>>,
}
impl<'m, 'a, 'b> Qwen<'m, 'a, 'b> {
    /// A frontend limit may be smaller than, but never exceed, loaded capacity.
    pub fn new(model: &'m spark_engine::qwen::Model<'a, 'b>, context_limit: usize) -> Self {
        Self {
            model,
            draft_model: None,
            profile: std::env::var("SPARK_PROFILE").as_deref() == Ok("1"),
            previous_decode: std::cell::Cell::new(None),
            context_limit: context_limit.min(model.max_tokens()),
            eos_tokens: if model.config().llama_rope.is_some() { [128001, 128008, 128009] } else {
                [model.config().eos_token as u32, if model.config().qk_norm { model.config().bos_token as u32 } else { model.config().eos_token as u32 }, model.config().eos_token as u32] },
            multi: std::cell::RefCell::new(None),
            draft_multi: std::cell::RefCell::new(None),
            seqpar: None,
            prefill: std::cell::RefCell::new(None),
        }
    }
    pub fn with_seqpar(mut self, seqpar: Seqpar) -> Self {
        self.seqpar = Some(seqpar);
        self
    }
    pub fn with_draft_model(mut self, model: &'m spark_engine::qwen::Model<'a, 'b>, k: usize) -> Result<Self, String> {
        if !(1..=8).contains(&k) || model.config().vocab_size != self.model.config().vocab_size
            || model.config().bos_token != self.model.config().bos_token || model.config().eos_token != self.model.config().eos_token {
            return Err("draft model requires matching vocabulary/special tokens and K in 1..8".into());
        }
        self.draft_model = Some((model, k));
        Ok(self)
    }
    /// Allocate shared decode and short-prompt scratch before serving requests.
    pub fn reserve_workspace(&self, rows: usize, sessions: usize) -> Result<usize, String> {
        use spark_engine::qwen::multi::Multi;
        let sessions = sessions.min(WORKSPACE_SESSIONS);
        let mut decode = Multi::new(self.model, rows)?;
        let decode_bytes = decode.reserve(self.context_limit, rows, false)?;
        let prompt = self.context_limit.min(self.prompt_limit());
        let prefill_bytes = if prompt > 0 {
            let mut prefill = Multi::new(self.model, (sessions * prompt).min(2048))?;
            let bytes = prefill.reserve(prompt, sessions, true)?;
            *self.prefill.borrow_mut() = Some(prefill);
            bytes
        } else { 0 };
        *self.multi.borrow_mut() = Some(decode);
        let draft_bytes = if let Some((model, depth)) = self.draft_model {
            let mut draft = Multi::new(model, (sessions * 64).min(2048))?;
            let bytes = draft.reserve_draft(sessions, depth)?;
            *self.draft_multi.borrow_mut() = Some(draft);
            bytes
        } else {0};
        Ok(decode_bytes + prefill_bytes + draft_bytes)
    }
    pub fn set_context_limit(&mut self, tokens: usize) {
        self.context_limit = tokens.min(self.model.max_tokens());
    }
    fn prompt_limit(&self) -> usize {
        // Generic heads retain full-row scores. A cached suffix can exceed
        // the bounded pooled prompt arena; use Session prefill for this shape.
        if self.model.config().head_dim != 128 { return 0; }
        // A follow-up rereads generated text plus the new turn. Keep ordinary
        // 128-token answers on the shared suffix path without restoring KV.
        256
    }
}
/// Router-rejected requests fail alone; the rest of the batch committed.
fn rejected(multi: &spark_engine::qwen::multi::Multi<'_, '_, '_>, mut predictions: Vec<crate::Predictions>) -> Vec<crate::Predictions> {
    for &i in multi.rejected() {
        predictions[i] = crate::Predictions::Rejected(spark_gpu_probe::driver::MOE_ROUTING_ERROR.into());
    }
    predictions
}
impl crate::Session for spark_engine::qwen::Session<'_, '_, '_> {
    fn disable_model_draft(&mut self) {
        spark_engine::qwen::Session::disable_model_draft(self);
    }
    fn reserve_model_draft(&mut self, tokens: usize) {
        spark_engine::qwen::Session::reserve_model_draft(self, tokens);
    }
    fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> {
        Some(self)
    }
    fn draft_model_memory_bytes(&self) -> Option<(usize, usize)> {
        (self.draft_model_width() > 0).then(|| self.draft_model_kv_bytes())
    }
    fn memory_bytes(&self) -> Option<(usize, usize)> { Some((self.cache_bytes(), self.device_peak_bytes())) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        spark_engine::qwen::Session::step(self, token)
    }
    fn advance(&mut self, token: u32) -> Result<(), String> {
        spark_engine::qwen::Session::advance(self, token)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        spark_engine::qwen::Session::prefill(self, tokens)
    }
    fn prefill_advance(&mut self, tokens: &[u32]) -> Result<(), String> {
        spark_engine::qwen::Session::prefill_advance(self, tokens)
    }
    fn prefill_chunk_size(&self, prompt_end: usize) -> usize {
        spark_engine::qwen::Session::prefill_chunk_size(self, prompt_end)
    }
    fn prefill_slice(&mut self, tokens: &[u32], final_chunk: bool, budget: Option<std::time::Duration>, prompt_end: usize)
        -> Result<Option<Vec<f32>>, String> {
        spark_engine::qwen::Session::prefill_slice(self, tokens, final_chunk, budget, prompt_end)
    }
    fn fork(&self) -> Result<Self, String> {
        spark_engine::qwen::Session::fork(self)
    }
    fn retain_draft_prefix(&mut self, source: &Self) -> Result<bool, String> {
        spark_engine::qwen::Session::retain_draft_prefix(self, source)
    }
    fn has_draft_prefix(&self) -> bool { self.draft_model_kv_bytes().0 != 0 }
    fn retained_bytes(&self) -> Option<usize> {
        Some(self.cache_allocations().iter().map(|&(_, bytes)| bytes).sum())
    }
    fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(self.cache_allocations()) }
    fn retirement_bytes(&self) -> Option<usize> { Some(self.resident_bytes()) }
    fn prefix_allocations(&self, tokens: usize) -> Option<Vec<(usize, usize)>> {
        Some(spark_engine::qwen::Session::prefix_allocations(self, tokens))
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> {
        spark_engine::qwen::Session::truncate(self, position)
    }
}
impl<'m, 'a, 'b> crate::Engine for Qwen<'m, 'a, 'b> {
    fn kv_branch_handles(&self) -> bool { true }
    type Session<'s>
        = spark_engine::qwen::Session<'s, 'a, 'b>
    where
        Self: 's;
    fn session(&self) -> Result<Self::Session<'_>, String> {
        let session = self.model.session();
        Ok(match self.draft_model { Some((model, k)) => session.with_draft_model(model, k), None => session })
    }
    fn prefill_long<'s>(&'s self, session: &Self::Session<'s>, tokens: &[u32])
        -> Result<Option<Box<dyn crate::LongPrefill<Self::Session<'s>> + 's>>, String> {
        let Some(seqpar) = &self.seqpar else { return Ok(None); };
        if seqpar.busy.borrow().strong_count() > 0 || tokens.len() < seqpar.threshold || !session.is_empty()
            || session.kv_format() != spark_gpu_probe::driver::KvFormat::Fp32
            || tokens.len() / spark_engine::qwen::PREFILL_CHUNK < seqpar.peers.len() + 1 {
            return Ok(None);
        }
        let pending = match seqpar.coordinator.start(session, tokens, &seqpar.peers) {
            Ok(pending) => pending,
            Err(error) => {
                engine_model::events::note_line("seqpar_fallback", &format!("seqpar_fallback phase=start tokens={} reason={error}", tokens.len()));
                return Ok(None);
            }
        };
        *seqpar.busy.borrow_mut() = pending.lifetime();
        Ok(Some(Box::new(CoordinatedPrefill { pending,
            peers: seqpar.peers.len(), tokens: tokens.len(), started: std::time::Instant::now() })))
    }
    fn shared_batch(&self) -> bool { true }
    fn shared_prompt_limit(&self) -> usize { self.prompt_limit() }
    fn shared_draft(&self) -> bool { self.draft_model.is_some() }
    fn restore_model_draft<'s>(&'s self, session: &mut Self::Session<'s>, prompt: usize) -> Result<(), String> {
        // Recover draft16's short-prefix policy; long prompts keep lookup
        // rather than starting an expensive proposal-model catchup.
        if prompt <= 4096 {
            if let Some((model, k)) = self.draft_model { session.restore_model_draft(model, k); }
        }
        Ok(())
    }
    fn automatic_draft_allowed(&self, chats: usize) -> bool {
        // Four or more ready decoders amortize shallow expert verification.
        // A draining cohort suspends drafts before scalar traffic dominates.
        self.model.config().num_experts == 0 || chats >= 4
    }
    fn shared_draft_costs(&self, chats: usize) -> Option<spark_engine::draft::CostCurve> {
        // GB10 cohort traffic grows with distinct experts, not chat depth.
        // Shared timings refine this prior; the scalar prior remains separate.
        (self.model.config().num_experts != 0 && chats >= 8)
            .then_some(spark_engine::draft::CostCurve([107.0, 144.0, 190.0, 241.0, 290.0]))
    }
    fn propose_batch<'s>(&'s self, rows: &mut [crate::DraftRows<'_, Self::Session<'s>>])
        -> Result<Vec<spark_engine::draft::ModelProposal>, crate::BatchError> {
        use spark_engine::qwen::multi::{Multi, DraftRows};
        let (model, _) = self.draft_model.ok_or_else(|| crate::BatchError {
            request: None, message: "missing shared draft model".into() })?;
        let mut multi = self.draft_multi.borrow_mut();
        if multi.is_none() { *multi = Some(Multi::new(model, 2048)
            .map_err(|message| crate::BatchError { request: None, message })?); }
        let mut proposals = Vec::with_capacity(rows.len());
        for (chunk, rows) in rows.chunks_mut(WORKSPACE_SESSIONS).enumerate() {
            let mut requests: Vec<_> = rows.iter_mut().map(|r| DraftRows {
                session: &mut *r.session, history: r.history, next: r.next, limit: r.limit }).collect();
            proposals.extend(multi.as_mut().unwrap().propose(&mut requests)
                .map_err(|e| crate::BatchError {
                    request: e.request.map(|i| chunk * WORKSPACE_SESSIONS + i), message: e.message })?);
        }
        Ok(proposals)
    }
    fn memory_bytes(&self) -> Option<(usize, usize, usize)> { Some(self.model.memory_bytes()) }
    fn fatal_error(&self) -> Option<String> { self.model.fatal_error() }
    fn maintain_memory(&self, idle: bool) -> Result<(), String> { self.model.maintain_kv_reserve(idle) }
    fn prefill_batch<'s>(&'s self, rows: &mut [crate::BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<crate::Predictions>, crate::BatchError> {
        use spark_engine::qwen::multi::{Multi, Rows, Output};
        let mut multi = self.prefill.borrow_mut();
        if multi.is_none() { *multi = Some(Multi::new(self.model, 2048)
            .map_err(|message| crate::BatchError { request: None, message })?); }
        let mut predictions = Vec::with_capacity(rows.len());
        for (chunk, rows) in rows.chunks_mut(WORKSPACE_SESSIONS).enumerate() {
            let mut requests: Vec<_> = rows.iter_mut().map(|r| Rows { session: &mut *r.session, tokens: r.tokens,
                output: match r.output { crate::BatchOutput::All => Output::All, crate::BatchOutput::Last => Output::Last,
                    crate::BatchOutput::None => Output::None } }).collect();
            let multi = multi.as_mut().unwrap();
            predictions.extend(multi.prefill(&mut requests)
                .map(|rows| rejected(multi, rows.into_iter().map(crate::Predictions::Logits).collect()))
                .map_err(|e| crate::BatchError {
                    request: e.request.map(|i| chunk * WORKSPACE_SESSIONS + i), message: e.message })?);
        }
        Ok(predictions)
    }
    fn step_batch<'s>(&'s self, rows: &mut [crate::BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<crate::Predictions>, crate::BatchError> {
        use spark_engine::qwen::multi::{Multi, Rows, Output};
        if let [row] = rows {
            if row.tokens.len() == 1 && matches!(row.output, crate::BatchOutput::All)
                && std::ptr::eq(row.session.config(), self.model.config()) {
                // A lone decoder uses the fused scalar kernels; wider work shares weights.
                let b = self.model.backend();
                let position = row.session.len();
                let gap_ms = if self.profile { self.previous_decode.get().map_or(0.0, |at| at.elapsed().as_secs_f64()*1000.0) } else { 0.0 };
                let before = self.profile.then(|| b.stats());
                let started = self.profile.then(std::time::Instant::now);
                // Sample kernel timings separately: CUDA event recording disables
                // graph capture and must never contaminate the default gate.
                let kernels = self.profile && position % 64 == 0;
                if kernels { b.set_profiling(true); }
                let result = if row.greedy {
                    row.session.step_greedy(row.tokens[0]).map(|id| crate::Predictions::Greedy(vec![id]))
                } else if row.top_k > 0 {
                    row.session.step_top_k(row.tokens[0], row.top_k).map(|values| crate::Predictions::TopK(vec![values]))
                } else {
                    row.session.step(row.tokens[0]).map(|logits| crate::Predictions::Logits(vec![logits]))
                };
                if kernels { b.set_profiling(false); }
                if let (Some(started), Some(before)) = (started, before) {
                    eprintln!("single_decode_profile position={position} greedy={} wall_ms={:.3} previous_gap_ms={gap_ms:.3} device_bytes={} stats={:?}",
                        row.greedy, started.elapsed().as_secs_f64()*1000.0, b.allocated_bytes(), b.stats().since(before));
                    self.previous_decode.set(Some(std::time::Instant::now()));
                }
                return result.map(|prediction| vec![prediction])
                    .map_err(|message| crate::BatchError { request: Some(0), message });
            }
        }
        let mut multi = self.multi.borrow_mut();
        if multi.is_none() { *multi = Some(Multi::new(self.model, 2048)
            .map_err(|message| crate::BatchError { request: None, message })?); }
        let greedy = rows.iter().all(|r| r.greedy && matches!(r.output, crate::BatchOutput::All));
        let top_k = if rows.iter().all(|r| r.top_k > 0 && matches!(r.output, crate::BatchOutput::All)) {
            rows.iter().map(|r| r.top_k).max().unwrap_or(0)
        } else { 0 };
        let mut requests: Vec<_> = rows.iter_mut().map(|r| Rows { session: &mut *r.session, tokens: r.tokens,
            output: match r.output { crate::BatchOutput::All => Output::All, crate::BatchOutput::Last => Output::Last,
                crate::BatchOutput::None => Output::None } }).collect();
        let multi = multi.as_mut().unwrap();
        if greedy { multi.step_greedy(&mut requests)
            .map(|rows| rows.into_iter().map(|r| crate::Predictions::Greedy(r.into_iter().map(|v| v.0).collect())).collect()) }
        else if top_k > 0 { multi.step_top_k(&mut requests, top_k)
            .map(|rows| rows.into_iter().map(crate::Predictions::TopK).collect()) }
        else { multi.step(&mut requests)
            .map(|rows| rows.into_iter().map(crate::Predictions::Logits).collect()) }
            .map(|predictions| rejected(multi, predictions))
            .map_err(|e| crate::BatchError { request: e.request, message: e.message })
    }
    fn context_limit(&self) -> usize {
        self.context_limit
    }
    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size as usize
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos_tokens
    }
    fn request_reservation_bytes(&self, tokens: usize) -> Option<usize> {
        let target = self.request_reservation_bytes_plain(tokens)?;
        let draft = match self.draft_model {
            Some((model, _)) => {
                // Fund the child's KV growth and private catchup scratch at
                // this reservation, including later cohort activation.
                let draft_tokens = tokens.min(model.max_tokens());
                let per_token = model.max_cache_bytes().checked_div(model.max_tokens())?;
                crate::budget::backend_reservation(model.config(),
                    per_token.checked_mul(draft_tokens)?, model.max_tokens())?
            },
            None => 0,
        };
        target.checked_add(draft)
    }
    fn request_reservation_bytes_plain(&self, tokens: usize) -> Option<usize> {
        self.request_reservation_bytes_plain_cached(tokens, 0)
    }
    fn request_reservation_bytes_plain_cached(&self, tokens: usize, cached: usize) -> Option<usize> {
        let per_token = self.model.max_cache_bytes().checked_div(self.model.max_tokens())?;
        crate::budget::backend_reservation_cached(self.model.config(), per_token.checked_mul(tokens)?, self.model.max_tokens(), cached)
    }
    fn session_reservation_bytes(&self) -> Option<usize> {
        // Include retained capacity and dimension-sized decode scratch.
        self.request_reservation_bytes(self.context_limit)
    }
}
