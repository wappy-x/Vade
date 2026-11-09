use crate::{Engine, Limits, Session};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub timing: Option<crate::usage::Timing>,
    pub kv: Option<crate::usage::Kv>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Prefill,
    Token(u32),
    Finished(FinishReason, Usage),
}

/// All checks preceding session allocation are shared by HTTP and in-process use.
pub fn validate<E: Engine>(
    engine: &E,
    prompt: &[u32],
    max_tokens: usize,
    limits: &Limits,
) -> Result<(), String> {
    validate_bounds(engine.context_limit(), engine.vocab_size(), engine.eos_token_ids(), prompt, max_tokens, limits)
}

pub(crate) fn validate_bounds(context: usize, vocab: usize, eos: &[u32],
    prompt: &[u32], max_tokens: usize, limits: &Limits) -> Result<(), String> {
    limits.validate()?;
    if prompt.is_empty() {
        return Err("prompt must contain at least one token".into());
    }
    if prompt.len() > limits.input_tokens {
        return Err(crate::budget::token_refusal(prompt.len() as u128, limits.input_tokens, prompt.len(), max_tokens, "--input-tokens"));
    }
    if prompt.len() >= limits.context_tokens.min(context) {
        return Err(crate::budget::token_refusal(prompt.len() as u128 + 1, limits.context_tokens.min(context), prompt.len(), max_tokens, "the lower of engine context and --context; one output token needs room"));
    }
    if max_tokens == 0 || max_tokens > limits.output_tokens {
        return Err(crate::budget::token_refusal(max_tokens as u128, limits.output_tokens, prompt.len(), max_tokens, "--output-tokens (max_tokens must be positive)"));
    }
    let total = prompt
        .len()
        .checked_add(max_tokens)
        .ok_or("context length overflow")?;
    if total > limits.context_tokens.min(context) {
        return Err(crate::budget::token_refusal(total as u128, limits.context_tokens.min(context), prompt.len(), max_tokens, "the lower of engine context and --context"));
    }
    if vocab == 0 || vocab > u32::MAX as usize {
        return Err("invalid engine vocabulary".into());
    }
    if prompt.iter().any(|&t| t as usize >= vocab) {
        return Err("prompt token outside vocabulary".into());
    }
    if eos
        .iter()
        .any(|&t| t as usize >= vocab)
    {
        return Err("invalid engine stop token".into());
    }
    Ok(())
}

/// One shared incremental path for scheduled and sequential generation.
/// The last emitted token is pending until another decode step is needed.
pub(crate) struct Generation<S> {
    pub state: crate::cache::CachedSession<S>,
    pub usage: Usage,
    pending: Option<u32>,
    finished: Option<FinishReason>,
    sampler: spark_engine::sample::Sampler,
    draft_options: Option<(spark_engine::draft::Mode, spark_engine::sample::Sampling)>,
    draft: Option<spark_engine::draft::Speculator>,
    draft_seed: Vec<u32>,
    shared_prefix_plain: bool,
    automatic_draft: bool,
    pub ignore_eos: bool,
    pub mtp_depth: Option<usize>,
    /// Target row bound for lookup; independent of the MTP fallback depth.
    pub verify_rows: Option<usize>,
    pub draft_used: bool,
    pub model_draft_used: bool,
}
pub(crate) enum Work {
    Prefill { tokens: Vec<u32>, last: bool },
    Decode([u32; 1]),
    Draft(spark_engine::draft::Proposal),
}
impl Work {
    pub fn model_limit(&self) -> usize { match self { Self::Draft(p) => p.model_limit(), _ => 0 } }
    pub fn row_budget(&self) -> usize { match self { Self::Draft(p) => p.row_budget(), _ => self.tokens().len() } }
    pub fn replace_model_draft(&mut self, tokens: Vec<u32>, ms: f64, stop: impl Fn(u32) -> bool,
        allowed: impl Fn(u32) -> bool) -> Result<(), String> {
        if let Self::Draft(plan) = self { plan.replace_model_draft(tokens, ms, stop, allowed) }
        else { Err("model proposal requires a draft plan".into()) }
    }
    pub fn tokens(&self) -> &[u32] { match self { Self::Prefill { tokens, .. } => tokens,
        Self::Decode(tokens) => tokens, Self::Draft(p) => &p.input } }
    pub fn output(&self) -> crate::BatchOutput { match self { Self::Prefill { last: true, .. } => crate::BatchOutput::Last,
        Self::Prefill { last: false, .. } => crate::BatchOutput::None, _ => crate::BatchOutput::All } }
    pub fn record_target_ms(&mut self, ms: f64) {
        if let Self::Draft(plan) = self { plan.record_target_ms(ms); }
    }
}
impl<S: Session> Generation<S> {
    pub fn new(
        state: crate::cache::CachedSession<S>,
        prompt_tokens: usize,
        sampler: spark_engine::sample::Sampler,
    ) -> Self {
        let cached_tokens = state.tokens.len();
        Self {
            state,
            usage: Usage { prompt_tokens, cached_tokens, ..Usage::default() },
            pending: None,
            finished: None,
            sampler,
            draft_options: None,
            draft: None,
            draft_seed: Vec::new(),
            shared_prefix_plain: false,
            automatic_draft: false,
            ignore_eos: false,
            mtp_depth: None,
            verify_rows: None,
            draft_used: false,
            model_draft_used: false,
        }
    }
    fn verification_cap(&self, cap: usize) -> usize {
        cap.min(
            self.verify_rows
                .unwrap_or_else(|| self.mtp_depth.map_or(cap, |depth| depth + 1)),
        )
    }
    pub fn enable_draft(&mut self, mode: crate::protocol::DraftMode, sampling: spark_engine::sample::Sampling) -> crate::protocol::DraftMode {
        use crate::protocol::DraftMode;
        let mode = if self.mtp_depth == Some(0) || mode == DraftMode::Auto && self.shared_prefix_plain { DraftMode::Plain } else { mode };
        // Cohort policy suspends proposals during prefill and draining, then
        // resumes them. Only the permanent shared-prefix choice discards KV.
        if mode == DraftMode::Plain && self.shared_prefix_plain { self.state.session.disable_model_draft(); }
        self.automatic_draft = mode == DraftMode::Auto;
        self.draft_options = match mode {
            DraftMode::Plain => None,
            DraftMode::Auto | DraftMode::Strict => {
                Some((spark_engine::draft::Mode::Strict, sampling))
            }
            DraftMode::Fast => Some((spark_engine::draft::Mode::Fast, sampling)),
            // Unbounded supports and history penalties use the canonical Strict
            // sampler, avoiding full-vocabulary speculative distributions.
            DraftMode::Sampled => Some((if sampling.compact_top_k() == 0 { spark_engine::draft::Mode::Strict }
                else { spark_engine::draft::Mode::Sampled }, sampling)),
        };
        mode
    }
    pub fn prefer_plain_shared_prefix(&mut self) { self.shared_prefix_plain = true; }
    pub fn draft_stats(&self) -> Option<&spark_engine::draft::Stats> { self.draft.as_ref().map(|s| &s.stats) }
    pub(crate) fn cohort_cost_counters(&self) -> [usize; 3] {
        let (proposed, accepted) = self.draft_stats().map_or((0, 0), |s| (s.proposed, s.accepted));
        [self.usage.completion_tokens, proposed, accepted]
    }
    pub(crate) fn cohort_pending_token(&self) -> Option<u32> {
        match &self.draft {
            Some(spec) => spec.canonical_next(),
            None => self.pending,
        }
    }
    /// VADE_MTP_SAMPLED=2: the speculator, for cohort-parallel row draws before complete().
    pub fn speculator_mut(&mut self) -> Option<&mut spark_engine::draft::Speculator> { self.draft.as_mut() }
    pub fn seed_draft(&mut self, tokens: &[u32]) { self.draft_seed.extend_from_slice(tokens); }
    pub fn draft_memory_bytes(&self) -> usize {
        self.draft_seed.capacity()*4 + self.draft.as_ref().map_or(0, |s| s.memory_bytes())
    }
    pub fn prefill_slice(&mut self, prompt: &[u32], chunk: usize, budget: Option<std::time::Duration>)
        -> Result<Option<Event>, String> {
        let pos = self.state.tokens.len();
        let end = (pos + chunk).min(prompt.len());
        // This chunk and the next two: their PLE pages load while earlier chunks run.
        self.state.session.read_ahead(&prompt[pos..(pos + 3 * chunk).min(prompt.len())]);
        let Some(logits) = self.state.session.prefill_slice(&prompt[pos..end], end == prompt.len(), budget, prompt.len())?
            else { return Ok(None); };
        self.state.next_logits = logits;
        self.state.tokens.extend_from_slice(&prompt[pos..end]);
        Ok(Some(Event::Prefill))
    }
    pub fn prepare<E: Engine>(&mut self, engine: &E, prompt: &[u32], max_tokens: usize,
        row_cap: usize, chats: usize, allowed: impl Fn(u32) -> bool) -> Result<(Option<Work>, Vec<Event>), String> {
        let pos = self.state.tokens.len();
        if pos < prompt.len() {
            let end = (pos+row_cap).min(prompt.len());
            return Ok((Some(Work::Prefill { tokens: prompt[pos..end].to_vec(), last: end == prompt.len() }), Vec::new()));
        }
        // Cached prompt logits produce the first visible token without replay.
        if self.finished.is_some() || self.usage.completion_tokens == 0 {
            return self.next(engine, prompt, max_tokens, 1, allowed).map(|e| (None, vec![e]));
        }
        if self.draft_options.is_some() || self.draft.is_some() {
            if self.state.session.draft_target().is_some() {
                if self.draft.is_none() {
                    let (mode, sampling) = self.draft_options.unwrap();
                    let mut config = spark_engine::draft::Config::strict(max_tokens);
                    config.max_rows = self.verification_cap(config.max_rows);
                    let target = self.state.session.draft_target().unwrap();
                    // Native policies recheck the row cap on every proposal.
                    // A busy shared drafter must not pin this whole turn to one row.
                    if !target.native_draft_policy() {
                        config.max_rows = config.max_rows.min(target.max_verify_rows());
                    }
                    config.mode = mode;
                    config.automatic = self.automatic_draft;
                    config.costs = target.verify_costs(prompt.len()).unwrap_or_else(|| spark_engine::draft::CostCurve::at_context(spark_engine::draft::Mode::Strict, prompt.len()));
                    self.draft = Some(spark_engine::draft::Speculator::from_selected(&self.state.tokens,
                        prompt.len(), self.pending.take().ok_or("missing draft input")?, engine.vocab_size(), config, sampling)?);
                    self.draft.as_mut().unwrap().seed(std::mem::take(&mut self.draft_seed));
                }
                // A suspended speculator commits one canonical row and retains
                // its next token; resuming never rereads stale prompt logits.
                let row_cap = if self.draft_options.is_some() {
                    self.verification_cap(row_cap)
                } else { 1 };
                let target = self.state.session.draft_target().unwrap();
                let spec = self.draft.as_mut().unwrap();
                let stop = |t| !self.ignore_eos && engine.eos_token_ids().contains(&t);
                let plan = if engine.shared_draft() {
                    spec.prepare_model(target, row_cap, engine.shared_draft_costs(chats), stop, &allowed)?
                } else { spec.prepare_with_target(target, row_cap, stop, &allowed)? };
                return Ok((Some(Work::Draft(plan)), Vec::new()));
            }
        }
        Ok((Some(Work::Decode([self.pending.ok_or("missing decode input")?])), Vec::new()))
    }
    pub fn share_model_depth<E: Engine>(&mut self, engine: &E, work: &mut Work,
        row_cap: usize, depth: usize) -> usize {
        let Work::Draft(plan) = work else { return 0; };
        if self.draft_options.is_none() { return 0; }
        let row_cap = self.verification_cap(row_cap);
        let Some(target) = self.state.session.draft_target() else { return 0; };
        self.draft.as_ref().map_or(0, |spec| {
            spec.share_model_depth(
                plan,
                target.model_draft_width(),
                row_cap,
                depth,
                target.sampled_model_draft(),
                |t| !self.ignore_eos && engine.eos_token_ids().contains(&t),
            )
        })
    }
    pub fn complete_model<E: Engine>(&mut self, engine: &E, work: &mut Work,
        proposal: spark_engine::draft::ModelProposal, allowed: impl Fn(u32) -> bool) -> Result<(), String> {
        let Work::Draft(plan) = work else { return Err("missing shared draft plan".into()); };
        self.draft.as_mut().ok_or("missing speculator")?.complete_model(plan, proposal,
            |t| !self.ignore_eos && engine.eos_token_ids().contains(&t), allowed)
    }
    pub fn complete<E: Engine>(&mut self, engine: &E, prompt: &[u32], max_tokens: usize,
        work: Work, mut prediction: crate::Predictions, allowed: impl Fn(u32) -> bool) -> Result<Vec<Event>, String> {
        use crate::Predictions;
        let selected = if let Predictions::TopK(rows) = &prediction {
            if rows.len() != work.tokens().len() || matches!(work, Work::Prefill { .. }) {
                return Err("invalid compact decode result".into());
            }
            rows.iter().enumerate().map(|(i, row)| {
                    self.sampler.sample_top_k(row,
                self.state.tokens.len()+i+1, &allowed)
                }).collect::<Option<Vec<_>>>()
        } else { None };
        if matches!(&prediction, Predictions::Greedy(ids) if ids.iter().any(|&id| !allowed(id)))
            || matches!(&prediction, Predictions::TopK(_)) && selected.is_none() {
            // Flash retains canonical verifier rows but cannot rewind below the
            // first verified token. Read those rows without replaying the target;
            // targets without retained logits keep the original rewind path.
            let rows = if let Some(rows) = self.state.session.decode_logits(work.tokens().len())? { rows }
            else {
                self.state.session.truncate(self.state.tokens.len())?;
                work.tokens().iter().map(|&id| self.state.session.step(id)).collect::<Result<_, _>>()?
            };
            prediction = Predictions::Logits(rows);
        }
        match work {
            Work::Prefill { tokens, last } => {
                self.state.tokens.extend(tokens);
                let mut events = vec![Event::Prefill];
                match prediction {
                    Predictions::Logits(mut rows) => {
                        self.state.next_logits = if last { rows.pop().ok_or("missing prompt logits")? } else { Vec::new() };
                        if last { events.push(self.next(engine, prompt, max_tokens, 1, allowed)?); }
                    }
                    Predictions::Greedy(ids) => {
                        if !last || ids.len() != 1 { return Err("invalid greedy prompt result".into()); }
                        self.state.next_logits.clear();
                        events.push(self.accept(engine, ids[0], max_tokens));
                    }
                    Predictions::TopK(_) => return Err("compact prompt result unsupported".into()),
                    Predictions::Rejected(error) => return Err(error),
                }
                Ok(events)
            }
            Work::Decode([token]) => {
                self.pending = None;
                self.state.tokens.push(token);
                match prediction {
                    Predictions::Logits(mut rows) => {
                        self.state.next_logits = rows.pop().ok_or("missing decode logits")?;
                        self.next(engine, prompt, max_tokens, 1, allowed).map(|e| vec![e])
                    }
                    Predictions::Greedy(ids) => {
                        self.state.next_logits.clear();
                        Ok(vec![self.accept(engine, *ids.first().ok_or("missing greedy result")?, max_tokens)])
                    }
                    Predictions::TopK(_) => {
                        self.state.next_logits.clear();
                        Ok(vec![self.accept(engine, selected.unwrap()[0], max_tokens)])
                    }
                    Predictions::Rejected(error) => Err(error),
                }
            }
            Work::Draft(plan) => {
                self.draft_used |= plan.input.len() > 1;
                self.model_draft_used |= plan.model_draft_used();
                let spec = self.draft.as_mut().unwrap();
                let skip = self.usage.completion_tokens.saturating_sub(spec.tokens().len());
                let target = self.state.session.draft_target().unwrap();
                let stop = |t| !self.ignore_eos && engine.eos_token_ids().contains(&t);
                let tokens = match prediction {
                    Predictions::Logits(rows) => {
                        spec.finish_exact(target, plan, &rows, stop, allowed)?
                    }
                    Predictions::Greedy(ids) => spec.finish_greedy(target, plan, &ids, stop)?,
                    Predictions::TopK(rows) => {
                        if spec.sampled_verifier() {
                        spec.finish_speculative_top_k(target, plan, &rows, stop, allowed)?
                    } else { spec.finish_sampled(target, plan, &selected.unwrap(),
                        &rows.iter().map(|r| r[0].0).collect::<Vec<_>>(), stop)? }
                    }
                    Predictions::Rejected(error) => return Err(error),
                };
                self.state.tokens.extend_from_slice(tokens);
                self.state.next_logits.clear();
                self.pending = None;
                let mut events = Vec::new();
                for &token in tokens.iter().skip(skip) {
                    self.usage.completion_tokens += 1;
                    if !self.ignore_eos && engine.eos_token_ids().contains(&token) { self.finished = Some(FinishReason::Stop); break; }
                    events.push(Event::Token(token));
                }
                if self.finished.is_none() && self.usage.completion_tokens == max_tokens { self.finished = Some(FinishReason::Length); }
                if let Some(reason) = self.finished { events.push(Event::Finished(reason, self.usage)); }
                Ok(events)
            }
        }
    }
    pub fn next<E: Engine>(
        &mut self,
        engine: &E,
        prompt: &[u32],
        max_tokens: usize,
        prefill_chunk: usize,
        allowed: impl Fn(u32) -> bool,
    ) -> Result<Event, String> {
        if let Some(reason) = self.finished {
            return Ok(Event::Finished(reason, self.usage));
        }
        let state = &mut self.state;
        let pos = state.tokens.len();
        if pos < prompt.len() {
            let end = (pos + prefill_chunk).min(prompt.len());
            if end < prompt.len() {
                state.session.prefill_advance(&prompt[pos..end])?;
                state.next_logits.clear();
            } else { state.next_logits = state.session.prefill(&prompt[pos..end])?; }
            state.tokens.extend_from_slice(&prompt[pos..end]);
            return Ok(Event::Prefill);
        }
        if let Some(token) = self.pending.take() {
            state.next_logits = state.session.step(token)?;
            state.tokens.push(token);
        }
        if state.next_logits.len() != engine.vocab_size() {
            return Err("engine returned invalid logits".into());
        }
        let first = self.usage.completion_tokens == 0;
        let started = std::time::Instant::now();
        let token = self.sampler.sample(
            &state.next_logits,
            &state.tokens,
            prompt.len(),
            prompt.len() + self.usage.completion_tokens,
            allowed,
        )?;
        if first && std::env::var("SPARK_PROFILE").as_deref() == Ok("1") {
            eprintln!("request_first_decode_profile sample_ms={:.3} device_step=false", started.elapsed().as_secs_f64()*1000.0);
        }
        Ok(self.accept(engine, token, max_tokens))
    }
    fn accept<E: Engine>(&mut self, engine: &E, token: u32, max_tokens: usize) -> Event {
        self.usage.completion_tokens += 1;
        if !self.ignore_eos && engine.eos_token_ids().contains(&token) {
            self.finished = Some(FinishReason::Stop);
            return Event::Finished(FinishReason::Stop, self.usage);
        }
        if self.usage.completion_tokens == max_tokens {
            self.finished = Some(FinishReason::Length);
        } else {
            self.pending = Some(token);
        }
        Event::Token(token)
    }
}

pub fn generate<E: Engine>(
    engine: &E,
    prompt: &[u32],
    max_tokens: usize,
    limits: &Limits,
    check: impl FnMut() -> Result<(), String>,
    emit: impl FnMut(Event) -> Result<(), String>,
) -> Result<Usage, String> {
    generate_sampled(
        engine,
        prompt,
        max_tokens,
        limits,
        &spark_engine::sample::Sampling::greedy(),
        |_| true,
        check,
        emit,
    )
}

pub fn generate_sampled<E: Engine>(
    engine: &E,
    prompt: &[u32],
    max_tokens: usize,
    limits: &Limits,
    sampling: &spark_engine::sample::Sampling,
    allowed_token: impl Fn(u32) -> bool,
    mut check: impl FnMut() -> Result<(), String>,
    mut emit: impl FnMut(Event) -> Result<(), String>,
) -> Result<Usage, String> {
    validate(engine, prompt, max_tokens, limits)?;
    let sampler = spark_engine::sample::Sampler::new(*sampling)?;
    check()?;
    let mut session = engine.session()?;
    session.set_cache_retention(false);
    let mut generation = Generation::new(
        crate::cache::CachedSession {
            checkpoint_lease: None,
            session,
            tokens: Vec::new(),
            next_logits: Vec::new(),
            reserved_session_bytes: 0,
        },
        prompt.len(),
        sampler,
    );
    loop {
        check()?;
        if generation.state.tokens.len() < prompt.len() {
            // Let a failed transport stop prefill before doing backend work.
            emit(Event::Prefill)?;
        }
        let event = generation.next(engine, prompt, max_tokens, 1, &allowed_token)?;
        if let Event::Finished(_, usage) = event {
            generation.state.session.finish()?;
            emit(event)?;
            return Ok(usage);
        }
        if event != Event::Prefill {
            emit(event)?;
        }
    }
}

#[cfg(test)]
mod bounded_fallback_tests {
    use super::*;
    use std::{cell::Cell,rc::Rc};
    struct Retained { reads:Rc<Cell<usize>> }
    impl Session for Retained {
        fn step(&mut self,_:u32)->Result<Vec<f32>,String> { panic!("canonical head fallback must not replay target layers") }
        fn decode_logits(&mut self,rows:usize)->Result<Option<Vec<Vec<f32>>>,String> {
            assert_eq!(rows,1);self.reads.set(self.reads.get()+1);
            Ok(Some(vec![vec![-1.0,8.0,9.0]]))
        }
    }
    struct Model { reads:Rc<Cell<usize>> }
    impl Engine for Model {
        type Session<'a>=Retained;
        fn session(&self)->Result<Retained,String> {Ok(Retained{reads:self.reads.clone()})}
        fn context_limit(&self)->usize {128}
        fn vocab_size(&self)->usize {3}
        fn eos_token_ids(&self)->&[u32] {&[]}
    }
    #[test]
    fn cohort_coverage_reads_existing_speculators_after_pending_is_consumed() {
        use spark_engine::{draft::{Config, Speculator}, sample::{Sampler, Sampling}};
        let engine=Model{reads:Rc::new(Cell::new(0))};
        let mut generation=Generation::new(crate::cache::CachedSession {
            checkpoint_lease:None,session:engine.session().unwrap(),tokens:vec![0],
            next_logits:Vec::new(),reserved_session_bytes:0,
        },1,Sampler::new(Sampling::greedy()).unwrap());
        generation.pending=Some(70_000);
        assert_eq!(generation.cohort_pending_token(),Some(70_000));
        generation.draft=Some(Speculator::from_selected(&[0],1,generation.pending.take().unwrap(),
            80_000,Config::strict(8),Sampling::greedy()).unwrap());
        assert_eq!(generation.pending,None);
        // Membership changes must still see this existing speculator's miss.
        assert_eq!(generation.cohort_pending_token(),Some(70_000));
        assert_eq!(generation.cohort_pending_token(),Some(70_000));
    }
    #[test]
    fn rejected_greedy_winner_reads_canonical_rows_without_replaying_target() {
        let reads=Rc::new(Cell::new(0));let engine=Model{reads:reads.clone()};
        let mut generation=Generation::new(crate::cache::CachedSession {
            checkpoint_lease:None,session:engine.session().unwrap(),tokens:vec![0],
            next_logits:Vec::new(),reserved_session_bytes:0,
        },1,spark_engine::sample::Sampler::new(spark_engine::sample::Sampling::greedy()).unwrap());
        let events=generation.complete(&engine,&[0],2,Work::Decode([0]),crate::Predictions::Greedy(vec![2]),|id|id!=2).unwrap();
        assert_eq!(events,vec![Event::Token(1)]);assert_eq!(reads.get(),1);
    }
}
