//! Plan-driven hybrid cartridge adapter for the existing cooperative scheduler.
use crate::Engine;
pub struct Hybrid<'m, 'a, 'b> {
    model: &'m spark_engine::hybrid::Model<'a, 'b>,
    draft: Option<&'m spark_engine::hybrid::Model<'a, 'b>>,
    depth: usize,
    eos: [u32; 2],
    decode: std::cell::RefCell<Option<spark_engine::hybrid::decode::Decode<'m, 'a, 'b>>>,
    drafts: std::cell::RefCell<Option<spark_engine::hybrid::mtp::BatchDraft<'m, 'a, 'b>>>,
    prefill: std::cell::RefCell<Option<spark_engine::hybrid::prefill::Prefill<'m, 'a, 'b>>>,
    prompt: Option<spark_engine::hybrid::decode::PromptWorkspace<'m, 'a, 'b>>,
}
impl<'m, 'a, 'b> Hybrid<'m, 'a, 'b> {
    pub fn new(model: &'m spark_engine::hybrid::Model<'a, 'b>, chat_eos: u32) -> Self {
        Self { model, draft: None, depth: model.default_draft_depth(), eos: [model.config().eos_token as u32, chat_eos], decode: std::cell::RefCell::new(None), drafts: std::cell::RefCell::new(None), prefill: std::cell::RefCell::new(None), prompt: None }
    }
    pub fn with_shared_prompt(mut self) -> Result<Self, String> {
        if !self.model.fixed_rows() { return Err("shared prompt requires typed model".into()); }
        self.prompt=Some(std::rc::Rc::new(std::cell::RefCell::new(None))); Ok(self)
    }
    pub fn with_draft_model(mut self, model: &'m spark_engine::hybrid::Model<'a, 'b>, depth: usize) -> Result<Self, String> {
        let (a, b) = (self.model.config(), model.config());
        if a.vocab != b.vocab || a.eos_token != b.eos_token || a.bos_token != b.bos_token || depth > 8 {
            return Err("hybrid draft vocabulary, special tokens or depth mismatch".into());
        }
        self.draft = Some(model); self.depth = depth;
        Ok(self)
    }
    pub fn with_draft_depth(mut self, depth: usize) -> Result<Self, String> {
        if depth > 4 { return Err("hybrid draft depth must be 0 (adaptive) or 1..=4".into()); }
        self.depth = depth; Ok(self)
    }
    fn reservation_bytes(&self, tokens: usize, retain: bool) -> Option<usize> {
        let target=if self.model.fixed_rows() { spark_engine::hybrid::Model::fixed_request_rows(self.model.config(),tokens,
            self.model.max_tokens(),self.drafting(),spark_engine::hybrid::memory::private_rows(self.prompt.is_some()),retain)? }
            else { self.model.request_reservation_bytes(tokens) };
        if target==usize::MAX { return None; }
        let draft=self.draft.map(|m| (if m.fixed_rows() { spark_engine::hybrid::Model::fixed_request_rows(m.config(),tokens,
            m.max_tokens(),m.mtp_enabled(),spark_engine::hybrid::memory::private_rows(false),false) }
            else { Some(m.request_reservation_bytes(tokens)) })?.checked_add((if self.model.fixed_rows() { 16usize } else { 4 }).checked_mul(m.config().state_bytes() as usize)?))
            .unwrap_or(Some(0))?;
        target.checked_add(draft)?.checked_add(if self.draft.is_some() && !self.model.mtp_enabled() && !self.model.fixed_rows() {
            spark_engine::hybrid::mtp::reservation_bytes(self.model.config(),tokens,spark_engine::hybrid_draft::Options::startup())
        } else { 0 })
    }
    fn drafting(&self) -> bool { self.model.mtp_enabled() || self.draft.is_some() }
    pub fn reserve_workspace(&self) -> Result<usize, String> {
        let before = self.model.backend().allocated_bytes();
        *self.decode.borrow_mut() = Some(spark_engine::hybrid::decode::Decode::with_rows(self.model,
            if self.drafting() { spark_engine::hybrid::decode::MAX_VERIFY_ROWS } else { spark_engine::hybrid::decode::MAX_CHAT_ROWS })?);
        if self.shared_model_drafts() { *self.drafts.borrow_mut() = Some(spark_engine::hybrid::mtp::BatchDraft::new(self.model)?); }
        if let Some(prompt)=&self.prompt { *prompt.borrow_mut()=Some(Box::new(spark_engine::hybrid::decode::Decode::with_rows(self.model,spark_engine::hybrid::schedule::ROUTED_GROUP)?)); }
        Ok(self.model.backend().allocated_bytes() - before)
    }
    fn serial(&self, rows: &mut [crate::BatchRows<'_, spark_engine::hybrid::Session<'m, 'a, 'b>>])
        -> Result<Vec<crate::Predictions>, crate::BatchError> {
        for row in rows.iter() { row.session.require_cache_identity(self.model)
            .map_err(|message| crate::BatchError { request: None, message })?; }
        rows.iter_mut().map(|row| {
            let result = match row.output {
                crate::BatchOutput::All if row.greedy && (row.tokens.len() > 1 || spark_engine::draft::Target::model_draft_width(row.session) > 0) => row.session.verify_mtp(row.tokens).and_then(|rows| rows.iter().map(|v| spark_engine::greedy_token(v)).collect()).map(crate::Predictions::Greedy),
                crate::BatchOutput::All if row.tokens.len() > 1 => row.session.verify_mtp(row.tokens).map(crate::Predictions::Logits),
                crate::BatchOutput::All if row.greedy && self.model.fixed_rows() => row.session.step_greedy(row.tokens[0]).map(|id| crate::Predictions::Greedy(vec![id])),
                crate::BatchOutput::All => row.session.step(row.tokens[0]).map(|v| crate::Predictions::Logits(vec![v])),
                crate::BatchOutput::Last => row.session.prefill(row.tokens).map(|v| crate::Predictions::Logits(vec![v])),
                crate::BatchOutput::None => row.session.prefill_advance(row.tokens).map(|_| crate::Predictions::Logits(Vec::new())),
            };
            // Rows run one session at a time: a router rejection fails only its own.
            match result {
                Err(message) if message == spark_gpu_probe::driver::MOE_ROUTING_ERROR => Ok(crate::Predictions::Rejected(message)),
                result => result.map_err(|message| crate::BatchError { request: None, message }),
            }
        }).collect()
    }
}
/// Router-rejected rows fail alone; their batch mates committed.
fn rejected(rows: &[usize], predictions: &mut [crate::Predictions]) {
    for &i in rows { predictions[i] = crate::Predictions::Rejected(spark_gpu_probe::driver::MOE_ROUTING_ERROR.into()); }
}
impl crate::Session for spark_engine::hybrid::Session<'_, '_, '_> {
    fn set_cache_retention(&mut self, retain: bool) { self.set_prefix_retention(retain); }
    fn cached_prompt_tokens(&self, _prompt_tokens: usize) -> Option<usize> { self.cached_prompt_tokens() }
    fn disable_model_draft(&mut self) { self.disable_model_draft(); }
    fn automatic_draft(&mut self, prompt: usize, remaining: usize) { self.automatic_draft(prompt, remaining); }
    fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> {
        (spark_engine::draft::Target::model_draft_width(self) != 0).then_some(self)
    }
    fn draft_model_memory_bytes(&self) -> Option<(usize, usize)> {
        Some((self.mtp_cache_bytes(), self.device_peak_bytes()))
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> { spark_engine::draft::Target::truncate(self, position) }
    fn memory_bytes(&self) -> Option<(usize, usize)> { Some((self.cache_bytes(), self.device_peak_bytes())) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> { spark_engine::hybrid::Session::step(self, token) }
    fn decode_logits(&mut self, rows: usize) -> Result<Option<Vec<Vec<f32>>>, String> { spark_engine::hybrid::Session::decode_logits(self, rows) }
    fn advance(&mut self, token: u32) -> Result<(), String> { spark_engine::hybrid::Session::advance(self, token) }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> { spark_engine::hybrid::Session::prefill(self, tokens) }
    fn prefill_advance(&mut self, tokens: &[u32]) -> Result<(), String> { spark_engine::hybrid::Session::prefill_advance(self, tokens) }
    // Typed sessions group canonical tiles internally and yield within the prompt.
    fn prefill_chunk_size(&self, end: usize) -> usize {
        let tile = spark_engine::hybrid::Session::prefill_chunk_size(self);
        if tile <= spark_engine::hybrid::schedule::PROMPT_ROWS { end - self.len() }
        else { tile - self.len() % tile }
    }
    fn prefill_chunk_min(&self) -> usize {
        let tile = spark_engine::hybrid::Session::prefill_chunk_size(self);
        tile - self.len() % tile
    }
    fn prefill_slice(&mut self, tokens: &[u32], final_chunk: bool, budget: Option<std::time::Duration>, _end: usize)
        -> Result<Option<Vec<f32>>, String> {
        spark_engine::hybrid::Session::prefill_slice(self, tokens, final_chunk, budget)
    }
    fn can_fork(&self) -> bool { self.can_fork() }
    fn can_fork_prefix(&self, position: usize) -> bool {
        self.can_fork() && self.prefix_reusable_tokens(position).is_some_and(|at|
            position >= self.checkpoint_reusable_tokens() || position == at)
    }
    fn fork(&self) -> Result<Self, String> { self.fork() }
    fn fork_prefix(&self, position: usize) -> Result<Self, String> {
        let mut session = self.fork()?;
        session.truncate_prefix(position)?;
        Ok(session)
    }
    fn retained_bytes(&self) -> Option<usize> { Some(self.fork_bytes()) }
    fn retained_private_bytes(&self) -> usize { self.fork_private_bytes() }
    fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(self.kv_allocations()) }
    fn prefix_allocations(&self, tokens: usize) -> Option<Vec<(usize, usize)>> { Some(self.prefix_allocations(tokens)) }
    fn checkpoint_reusable_tokens(&self, tokens: usize) -> usize {
        self.prefix_reusable_tokens(tokens).unwrap_or(usize::MAX)
    }
}
impl<'m, 'a, 'b> crate::Engine for Hybrid<'m, 'a, 'b> {
    fn validate_request(&self, _prompt: &[u32], request: &crate::protocol::Request) -> Result<(), String> {
        if self.model.fixed_rows() && self.model.mtp_enabled() { request.draft.unwrap_or_default().require_typed_mtp()?; }
        Ok(())
    }
    fn kv_branch_handles(&self) -> bool { true }
    type Session<'s> = spark_engine::hybrid::Session<'m, 'a, 'b> where Self: 's;
    fn session(&self) -> Result<Self::Session<'_>, String> {
        let mut session = self.model.session()?;
        session.set_draft_depth(self.depth)?;
        if let Some(draft) = self.draft { session = session.with_draft_model(draft, self.depth)?; }
        if self.shared_model_drafts() { session.enable_batch_drafts(); }
        if let Some(prompt)=&self.prompt { session.share_prompt_workspace(prompt.clone())?; }
        Ok(session)
    }
    fn restore_model_draft<'s>(&'s self, session: &mut Self::Session<'s>, _prompt: usize) -> Result<(), String> {
        session.require_cache_identity(self.model)?;
        if let Some(draft) = self.draft { session.restore_model_draft(draft, self.depth)?; }
        Ok(())
    }
    fn fatal_error(&self) -> Option<String> { self.model.backend().fatal_error() }
    fn shared_batch(&self) -> bool { true }
    fn automatic_draft_limit(&self) -> usize { 1 }
    fn automatic_draft_min_output(&self) -> usize { if self.shared_model_drafts() { 32 } else { 0 } }
    fn native_draft_policy(&self) -> bool { true }
    fn shared_model_drafts(&self) -> bool { self.model.mtp_enabled() && self.draft.is_none() && !self.model.fixed_rows() }
    fn decode_row_limit(&self, chats: usize) -> usize {
        if self.drafting() { (chats * 5).min(spark_engine::hybrid::decode::MAX_VERIFY_ROWS) }
        else { chats.div_ceil(16) * 16 }
    }
    fn prepare_drafts<'s>(&'s self, rows: &mut [crate::BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<Option<Vec<u32>>>, crate::BatchError> {
        if !self.shared_model_drafts() { return Ok(rows.iter().map(|_| None).collect()); }
        let mut work = self.drafts.borrow_mut();
        if work.is_none() { *work = Some(spark_engine::hybrid::mtp::BatchDraft::new(self.model)
            .map_err(|message| crate::BatchError { request: None, message })?); }
        let mut output = Vec::with_capacity(rows.len());
        for chunk in rows.chunks_mut(spark_engine::hybrid::decode::MAX_ROWS) {
            let mut requests: Vec<_> = chunk.iter_mut().map(|r| spark_engine::hybrid::mtp::DraftRow { session: &mut *r.session, token: r.tokens[0] }).collect();
            output.extend(work.as_mut().unwrap().propose(&mut requests).map_err(|message| crate::BatchError { request: None, message })?);
        }
        Ok(output)
    }
    fn shared_prefill_limit(&self) -> usize { 512 }
    fn shared_prefill_rows(&self) -> usize { if self.prompt.is_some() { spark_engine::hybrid::schedule::PROMPT_GROUP } else { self.model.prefill_chunk_size() } }
    fn shared_prefill_chunk_rows(&self) -> usize { if self.prompt.is_some() { self.model.prefill_chunk_size() } else { 0 } }
    fn reserve_foreground_slot(&self) -> bool { false }
    fn prefill_batch<'s>(&'s self, rows: &mut [crate::BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<crate::Predictions>, crate::BatchError> {
        if let Some(prompt) = &self.prompt {
            let tile = self.model.prefill_chunk_size();
            let count = rows.iter().try_fold(0usize, |n,r| n.checked_add(r.tokens.len()));
            if self.model.rank_group().is_ok_and(|g| !g.distributed()) && !rows.is_empty()
                && count.is_some_and(|n| n <= spark_engine::hybrid::schedule::ROUTED_GROUP)
                && rows.iter().all(|r| !matches!(r.output, crate::BatchOutput::All)
                    && r.session.can_fork() && r.session.checkpoint_reusable_tokens() == r.session.len()
                    && r.session.len() % tile == 0 && r.session.require_cache_identity(self.model).is_ok()
                    && !r.tokens.is_empty() && r.tokens.len() <= tile
                    && r.session.len().checked_add(r.tokens.len()).is_some_and(|n| n <= self.model.max_tokens())
                    && r.tokens.iter().all(|&t| (t as u64) < self.model.config().vocab)) {
                let mut workspace = prompt.borrow_mut();
                let Some(executor) = workspace.as_mut() else { drop(workspace); return self.serial(rows); };
                let mut requests: Vec<_> = rows.iter_mut().map(|r| spark_engine::hybrid::prefill::Row {
                    session: &mut *r.session, tokens: r.tokens, logits: matches!(r.output, crate::BatchOutput::Last) }).collect();
                let mut output = executor.prompt_batch(&mut requests).map_err(|message| crate::BatchError { request: None, message })?
                    .into_iter().map(|v| crate::Predictions::Logits(if v.is_empty() { Vec::new() } else { vec![v] })).collect::<Vec<_>>();
                rejected(executor.rejected(), &mut output);
                return Ok(output);
            }
            return self.serial(rows);
        }
        if self.model.fixed_rows() || rows.len() <= 1 || rows.iter().any(|r| matches!(r.output, crate::BatchOutput::All)) {
            return self.serial(rows);
        }
        let mut prefill = self.prefill.borrow_mut();
        if prefill.is_none() { *prefill = Some(spark_engine::hybrid::prefill::Prefill::new(self.model)
            .map_err(|message| crate::BatchError { request: None, message })?); }
        let mut output = Vec::with_capacity(rows.len());
        for chunk in rows.chunks_mut(spark_engine::hybrid::decode::MAX_ROWS) {
            let greedy = chunk.iter().all(|r| r.greedy && matches!(r.output, crate::BatchOutput::Last));
            let mut requests: Vec<_> = chunk.iter_mut().map(|r| spark_engine::hybrid::prefill::Row {
                session: &mut *r.session, tokens: r.tokens, logits: !matches!(r.output, crate::BatchOutput::None) }).collect();
            let start = output.len();
            if greedy {
                output.extend(prefill.as_mut().unwrap().run_greedy(&mut requests)
                    .map_err(|message| crate::BatchError { request: None, message })?
                    .into_iter().map(|id| crate::Predictions::Greedy(vec![id])));
            } else { output.extend(prefill.as_mut().unwrap().run(&mut requests)
                .map_err(|message| crate::BatchError { request: None, message })?
                .into_iter().map(|v| crate::Predictions::Logits(if v.is_empty() { Vec::new() } else { vec![v] }))); }
            rejected(prefill.as_ref().unwrap().rejected(), &mut output[start..]);
        }
        Ok(output)
    }
    fn step_batch<'s>(&'s self, rows: &mut [crate::BatchRows<'_, Self::Session<'s>>])
        -> Result<Vec<crate::Predictions>, crate::BatchError> {
        // Use the native Speculator/propose_model/verify_mtp path for typed MTP.
        if self.model.fixed_rows() && self.model.mtp_enabled() {
            if rows.len() != 1 { return Err(crate::BatchError { request: None, message: "typed hybrid MoE MTP requires one active chat".into() }); }
            return self.serial(rows);
        }
        if rows.iter().any(|r| spark_engine::draft::Target::model_draft_width(r.session) > 0) {
            if rows.iter().any(|r| r.session.is_empty() || !matches!(r.output, crate::BatchOutput::All))
                || (self.model.config().fp8 && rows.iter().map(|r| r.tokens.len()).sum::<usize>() > spark_engine::hybrid::decode::MAX_ROWS) {
                return self.serial(rows);
            }
            let mut decode = self.decode.borrow_mut();
            if decode.is_none() { *decode = Some(spark_engine::hybrid::decode::Decode::with_rows(self.model, spark_engine::hybrid::decode::MAX_VERIFY_ROWS)
                .map_err(|message| crate::BatchError { request: None, message })?); }
            let greedy = rows.iter().all(|r| r.greedy);
            let mut requests: Vec<_> = rows.iter_mut().map(|r| spark_engine::hybrid::decode::VerifyRow { session: &mut *r.session, tokens: r.tokens }).collect();
            let decode = decode.as_mut().unwrap();
            let mut output: Vec<_> = match decode.verify_batch(&mut requests, greedy).map_err(|message| crate::BatchError { request: None, message })? {
                spark_engine::hybrid::decode::VerifyOutput::Greedy(rows) => rows.into_iter().map(crate::Predictions::Greedy).collect(),
                spark_engine::hybrid::decode::VerifyOutput::Logits(rows) => rows.into_iter().map(crate::Predictions::Logits).collect(),
            };
            rejected(decode.rejected(), &mut output);
            return Ok(output);
        }
        if rows.len() <= 1 || rows.iter().any(|r| r.tokens.len() != 1 || r.session.is_empty()) {
            return self.serial(rows);
        }
        let mut decode = self.decode.borrow_mut();
        if decode.is_none() { *decode = Some(spark_engine::hybrid::decode::Decode::new(self.model)
            .map_err(|message| crate::BatchError { request: None, message })?); }
        let mut output = Vec::with_capacity(rows.len());
        // Independent chats share one pass through the canonical operators.
        for chunk in rows.chunks_mut(spark_engine::hybrid::decode::MAX_CHAT_ROWS) {
            let b = self.model.backend();
            let position = chunk[0].session.len();
            let profile = std::env::var("SPARK_PROFILE").as_deref() == Ok("1") && position % 64 == 0;
            let before = profile.then(|| b.stats());
            let started = std::time::Instant::now();
            b.set_profiling(profile);
            let greedy = chunk.iter().all(|r| r.greedy && !matches!(r.output, crate::BatchOutput::None));
            let mut requests: Vec<_> = chunk.iter_mut().map(|r| spark_engine::hybrid::decode::Row {
                session: &mut *r.session, token: r.tokens[0], logits: !matches!(r.output, crate::BatchOutput::None) }).collect();
            let result = if greedy {
                decode.as_mut().unwrap().step_greedy(&mut requests)
                    .map(|ids| ids.into_iter().map(|id| crate::Predictions::Greedy(vec![id])).collect::<Vec<_>>())
            } else {
                decode.as_mut().unwrap().step(&mut requests)
                    .map(|rows| rows.into_iter().map(|v| crate::Predictions::Logits(if v.is_empty() { Vec::new() } else { vec![v] })).collect())
            };
            b.set_profiling(false);
            let start = output.len();
            output.extend(result.map_err(|message| crate::BatchError { request: None, message })?);
            rejected(decode.as_ref().unwrap().rejected(), &mut output[start..]);
            if let Some(before) = before {
                eprintln!("hybrid_decode_profile rows={} position={position} wall_ms={:.3} device_bytes={} peak_device_bytes={} stats={:?}",
                    requests.len(), started.elapsed().as_secs_f64()*1000.0, b.allocated_bytes(), b.peak_allocated_bytes(), b.stats().since(before));
            }
        }
        Ok(output)
    }
    fn context_limit(&self) -> usize { self.model.max_tokens() }
    fn vocab_size(&self) -> usize { self.model.config().vocab as usize }
    fn eos_token_ids(&self) -> &[u32] { &self.eos }
    fn session_reservation_bytes(&self) -> Option<usize> {
        self.request_reservation_bytes(self.model.max_tokens())
    }
    fn request_reservation_bytes(&self, tokens: usize) -> Option<usize> { self.reservation_bytes(tokens,true) }
    fn request_reservation_bytes_with_retention(&self, tokens: usize, _plain: bool, _cached: Option<usize>, retain: bool) -> Option<usize> {
        self.reservation_bytes(tokens,retain)
    }
    fn memory_bytes(&self) -> Option<(usize, usize, usize)> {
        let b = self.model.backend();
        Some((b.allocated_bytes(), b.peak_allocated_bytes(), b.pinned_host_bytes()))
    }
}
