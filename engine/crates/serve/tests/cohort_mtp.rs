//! Exercise temporary cohort suspension through the real HTTP scheduler.
use spark_engine::draft::{ModelProposal, Target};
use spark_serve::{
    Engine, Limits, Session, cache::CacheLimits, mtp_batch::{Mode, Policy}, protocol::DraftMode, scheduler,
};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

const SESSION_BYTES: usize = 16384;
const ACTIVE_BYTES: usize = 16 * SESSION_BYTES;

struct Admission {
    expected: usize,
    prepared: usize,
    lookup_check: bool,
    started: Instant,
}

#[derive(Default)]
struct Gate {
    state: Mutex<(bool, bool)>,
    wake: Condvar,
    armed: AtomicBool,
    delayed: AtomicBool,
    verifies: Trace,
    prefills: Trace,
    echoes: Mutex<Vec<usize>>,
    rejected: AtomicUsize,
    accepted: AtomicUsize,
    configured: Mutex<Vec<usize>>,
    /// (live sessions, verify rows) per target pass.
    tiles: Trace,
    /// (live sessions, drafting chats) per shared propose_batch call.
    batches: Trace,
    policies: Mutex<Vec<(usize, Policy)>>,
    cost_control: AtomicBool,
    choices: Mutex<Vec<Option<bool>>>,
    head_limit: AtomicUsize,
    admission: Mutex<Option<Admission>>,
}
impl Gate {
    fn wait_for_verify(&self) {
        let (state, timeout) = self
            .wake
            .wait_timeout_while(self.state.lock().unwrap(), Duration::from_secs(10), |s| {
                !s.0
            })
            .unwrap();
        assert!(
            !timeout.timed_out() && state.0,
            "verification never started"
        );
    }
    fn release(&self) {
        self.state.lock().unwrap().1 = true;
        self.wake.notify_all();
    }
}
type Trace = Arc<Mutex<Vec<(usize, usize)>>>;
type ScenarioResult = (Vec<Vec<u32>>, Vec<(usize, usize)>, Arc<Gate>);
struct Model {
    live: Arc<AtomicUsize>,
    proposals: Trace,
    gate: Arc<Gate>,
    batch: Mode,
    cohort: Arc<AtomicUsize>,
    /// Chats leave their drafts to propose_batch (FlashNext under VADE_MTP_BATCH=1).
    defer: bool,
    /// FlashNext VADE_COHORT_EVEN_ROWS.
    even: bool,
}
struct Chat {
    history: Vec<u32>,
    cycles: usize,
    verified: usize,
    verify_rows: usize,
    depth: usize,
    live: Arc<AtomicUsize>,
    proposals: Trace,
    gate: Arc<Gate>,
    defer: bool,
    batch: Mode,
    cohort: Arc<AtomicUsize>,
}
impl Drop for Chat {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}
fn logits(history: &[u32]) -> Vec<f32> {
    let state = history
        .iter()
        .fold(7usize, |s, &t| s.wrapping_mul(31) ^ t as usize);
    let mut row = vec![0.; 8];
    row[state % 6 + 1] = 1.;
    row
}
impl Session for Chat {
    fn configure_request(
        &mut self,
        request: &spark_serve::protocol::Request,
    ) -> Result<(), String> {
        self.depth = request.mtp_depth.unwrap_or(3);
        self.gate
            .configured
            .lock()
            .map_err(|e| e.to_string())?
            .push(self.depth);
        self.verify_rows =
            if request.prompt_lookup && request.cache_salt.as_deref() != Some("mtp-oracle") {
                request.prompt_lookup_max_draft + 1
            } else {
                self.depth + 1
            };
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.history.push(token);
        Ok(logits(&self.history))
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> {
        if self.verified > 1 {
            if position < self.history.len() {
                self.gate.rejected.fetch_add(1, Ordering::Relaxed);
            } else {
                self.gate.accepted.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.verified = 0;
        self.history.truncate(position);
        Ok(())
    }
    fn prefill_chunk_size(&self, _: usize) -> usize {
        2048
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.gate
            .prefills
            .lock()
            .unwrap()
            .push((self.live.load(Ordering::Relaxed), tokens.len()));
        let mut row = logits(&self.history);
        for &t in tokens {
            row = Session::step(self, t)?;
        }
        Ok(row)
    }
    fn prefill_advance(&mut self, tokens: &[u32]) -> Result<(), String> {
        self.prefill(tokens).map(|_| ())
    }
    fn can_fork(&self) -> bool {
        false
    }
    fn draft_target(&mut self) -> Option<&mut dyn Target> {
        Some(self)
    }
}
impl Target for Chat {
    fn checkpoint(&self) -> usize {
        self.history.len()
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> {
        Session::truncate(self, position)
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        Session::step(self, token)
    }
    fn max_verify_rows(&self) -> usize {
        self.verify_rows
    }
    fn model_draft_width(&self) -> usize {
        self.depth
    }
    fn native_draft_policy(&self) -> bool {
        true
    }
    fn defer_model_draft(&self, limit: usize) -> Option<usize> {
        (self.defer && self.batch.enabled(self.cohort.load(Ordering::Relaxed))).then(|| limit.min(self.depth))
    }
    fn propose_model(
        &mut self,
        _: &[u32],
        next: u32,
        limit: usize,
    ) -> Result<Option<ModelProposal>, String> {
        self.proposals
            .lock()
            .map_err(|e| e.to_string())?
            .push((self.live.load(Ordering::Relaxed), limit));
        Ok(Some(self.chain(next, limit)))
    }
}
impl Chat {
    /// One chat's draft chain, whichever path asks for it.
    fn chain(&mut self, next: u32, limit: usize) -> ModelProposal {
        let mut history = self.history.clone();
        let mut token = next;
        let mut tokens = Vec::new();
        for _ in 0..limit {
            history.push(token);
            token = spark_engine::greedy_token(&logits(&history)).unwrap();
            tokens.push(token);
        }
        self.cycles += 1;
        if self.cycles.is_multiple_of(2)
            && let Some(last) = tokens.last_mut()
        {
            *last = (*last + 1) % 8;
        }
        ModelProposal {
            tokens,
            proposal_ms: 0.,
            catchup_ms: 0.,
        }
    }
}
impl Engine for Model {
    type Session<'a> = Chat;
    fn session(&self) -> Result<Chat, String> {
        if let Some(admission) = self.gate.admission.lock().unwrap().as_ref() {
            assert_eq!(admission.prepared, admission.expected, "session admitted before cohort was ready");
        }
        self.live.fetch_add(1, Ordering::Relaxed);
        Ok(Chat {
            history: Vec::new(),
            cycles: 0,
            verified: 0,
            verify_rows: 4,
            depth: 3,
            gate: self.gate.clone(),
            live: self.live.clone(),
            proposals: self.proposals.clone(),
            defer: self.defer,
            batch: self.batch,
            cohort: self.cohort.clone(),
        })
    }
    fn propose_batch<'s>(
        &'s self,
        rows: &mut [spark_serve::DraftRows<'_, Chat>],
    ) -> Result<Vec<ModelProposal>, spark_serve::BatchError> {
        self.gate
            .batches
            .lock()
            .unwrap()
            .push((self.live.load(Ordering::Relaxed), rows.len()));
        Ok(rows
            .iter_mut()
            .map(|r| {
                assert!(r.limit <= r.session.depth && r.next < 8);
                r.session.chain(r.next, r.limit)
            })
            .collect())
    }
    fn step_batch(
        &self,
        rows: &mut [spark_serve::BatchRows<'_, Chat>],
    ) -> Result<Vec<spark_serve::Predictions>, spark_serve::BatchError> {
        if self.gate.delayed.load(Ordering::Relaxed)
            && rows
                .iter()
                .any(|r| matches!(r.output, spark_serve::BatchOutput::All))
        {
            // A bounded target-call cost lets the real intake/preparation workers
            // run before the synthetic 128-token reply can finish. The verify
            // barrier, route trace and output assertions establish the ordering.
            std::thread::sleep(Duration::from_millis(1));
        }
        let verify: usize = rows
            .iter()
            .filter(|r| matches!(r.output, spark_serve::BatchOutput::All))
            .map(|r| r.tokens.len())
            .sum();
        if verify > 0 {
            self.gate
                .tiles
                .lock()
                .unwrap()
                .push((self.live.load(Ordering::Relaxed), verify));
        }
        for row in rows.iter() {
            if matches!(row.output, spark_serve::BatchOutput::All) {
                self.gate
                    .verifies
                    .lock()
                    .unwrap()
                    .push((self.live.load(Ordering::Relaxed), row.tokens.len()));
                if row.tokens.len() > 1 && self.gate.armed.swap(false, Ordering::AcqRel) {
                    let mut state = self.gate.state.lock().unwrap();
                    state.0 = true;
                    self.gate.wake.notify_all();
                    let (state, timeout) = self
                        .gate
                        .wake
                        .wait_timeout_while(state, Duration::from_secs(10), |s| !s.1)
                        .unwrap();
                    assert!(!timeout.timed_out() && state.1, "verify gate not released");
                }
            }
        }
        rows.iter_mut()
            .map(|row| {
                row.session.verified = if matches!(row.output, spark_serve::BatchOutput::All) {
                    row.tokens.len()
                } else {
                    0
                };
                let result = match row.output {
                    spark_serve::BatchOutput::All => row
                        .tokens
                        .iter()
                        .map(|&t| Session::step(row.session, t))
                        .collect(),
                    spark_serve::BatchOutput::Last => {
                        row.session.prefill(row.tokens).map(|v| vec![v])
                    }
                    spark_serve::BatchOutput::None => {
                        row.session.prefill_advance(row.tokens).map(|_| Vec::new())
                    }
                };
                result
                    .map(spark_serve::Predictions::Logits)
                    .map_err(|message| spark_serve::BatchError {
                        request: None,
                        message,
                    })
            })
            .collect()
    }
    fn vocab_size(&self) -> usize {
        8
    }
    fn context_limit(&self) -> usize {
        8192
    }
    fn eos_token_ids(&self) -> &[u32] {
        &[]
    }
    fn active_cohort(&self, chats: usize) {
        self.cohort.store(chats, Ordering::Relaxed);
        self.gate.policies.lock().unwrap().push((chats, Policy {
            draft_chats: self.draft_chat_limit(), automatic_chats: self.automatic_draft_limit(),
            tick_rows: self.tick_row_limit(),
        }));
    }
    fn shared_batch(&self) -> bool {
        true
    }
    fn cohort_cost_control(&self) -> bool {
        self.gate.cost_control.load(Ordering::Relaxed)
    }
    fn cohort_draft_head_limit(&self) -> Option<usize> {
        let rows = self.gate.head_limit.load(Ordering::Relaxed);
        (rows > 0).then_some(rows)
    }
    fn set_cohort_choice(&self, draft: Option<bool>) {
        self.gate.choices.lock().unwrap().push(draft);
    }
    fn shared_draft(&self) -> bool {
        true
    }
    fn shared_model_drafts(&self) -> bool {
        true
    }
    fn single_chat_draft(&self) -> bool {
        true
    }
    fn draft_chat_limit(&self) -> usize {
        self.batch.policy(self.cohort.load(Ordering::Relaxed)).draft_chats
    }
    fn automatic_draft_limit(&self) -> usize {
        self.batch.policy(self.cohort.load(Ordering::Relaxed)).automatic_chats
    }
    fn tick_row_limit(&self) -> Option<usize> {
        self.batch.policy(self.cohort.load(Ordering::Relaxed)).tick_rows
    }
    fn even_cohort_rows(&self) -> bool {
        self.even && self.batch.enabled(self.cohort.load(Ordering::Relaxed))
    }
    fn greedy_draft_only(&self) -> bool {
        true
    }
    fn reserve_foreground_slot(&self) -> bool {
        false
    }
    fn validate_request(&self, _: &[u32], request: &spark_serve::protocol::Request) -> Result<(), String> {
        if let Some(admission) = self.gate.admission.lock().unwrap().as_mut() {
            admission.prepared += 1;
            admission.lookup_check = request.prompt_lookup;
            assert!(admission.prepared <= admission.expected);
        }
        Ok(())
    }
    fn request_reservation_bytes(&self, _: usize) -> Option<usize> {
        if let Some(admission) = self.gate.admission.lock().unwrap().as_mut() {
            // The scheduler validates lookup's capacity immediately after
            // validate_request. That probe uses the real bound; only later
            // admission probes wait for the complete cohort.
            if std::mem::take(&mut admission.lookup_check) { return Some(SESSION_BYTES); }
            if admission.prepared < admission.expected {
                assert!(admission.started.elapsed() < Duration::from_secs(30),
                    "cohort preparation stalled: {}/{} ready", admission.prepared, admission.expected);
                // Keep intake running; blocking the scheduler thread would
                // prevent the rest of the cohort from reaching this barrier.
                return Some(ACTIVE_BYTES + 1);
            }
        }
        Some(SESSION_BYTES)
    }
    fn session_reservation_bytes(&self) -> Option<usize> {
        Some(SESSION_BYTES)
    }
}

fn run(outputs: &[usize], depth: usize) -> (Vec<Vec<u32>>, Vec<(usize, usize)>) {
    let (rows, trace, _) = scenario(outputs, depth, false, false);
    (rows, trace)
}
fn scenario(outputs: &[usize], depth: usize, late: bool, cancel: bool) -> ScenarioResult {
    scenario_lookup(outputs, depth, late, cancel, false)
}
fn scenario_lookup(
    outputs: &[usize],
    depth: usize,
    late: bool,
    cancel: bool,
    lookup: bool,
) -> ScenarioResult {
    scenario_with_defaults(
        outputs,
        Some(depth),
        3,
        late,
        cancel,
        lookup,
        Mode::Off,
        false,
    )
}
fn batched(outputs: &[usize], depth: usize, on: bool) -> ScenarioResult {
    scenario_with_defaults(
        outputs,
        Some(depth),
        3,
        false,
        false,
        false,
        if on { Mode::On } else { Mode::Off },
        false,
    )
}
/// FlashNext's VADE_MTP_BATCH table plus its deferred, cohort-wide drafts.
fn shared(outputs: &[usize], depth: usize, on: bool) -> ScenarioResult {
    scenario_with_defaults(
        outputs,
        Some(depth),
        3,
        false,
        false,
        false,
        if on { Mode::On } else { Mode::Off },
        on,
    )
}
fn scenario_with_defaults(
    outputs: &[usize],
    depth: Option<usize>,
    startup_depth: usize,
    late: bool,
    cancel: bool,
    lookup: bool,
    batch: Mode,
    defer: bool,
) -> ScenarioResult {
    scenario_rows(outputs, depth, startup_depth, late, cancel, lookup, batch, defer, false)
}
#[allow(clippy::too_many_arguments)]
fn scenario_rows(
    outputs: &[usize],
    depth: Option<usize>,
    startup_depth: usize,
    late: bool,
    cancel: bool,
    lookup: bool,
    batch: Mode,
    defer: bool,
    even: bool,
) -> ScenarioResult {
    scenario_control(outputs, depth, startup_depth, late, cancel, lookup, batch, defer, even, false, DraftMode::Strict)
}
#[allow(clippy::too_many_arguments)]
fn scenario_control(
    outputs: &[usize], depth: Option<usize>, startup_depth: usize,
    late: bool, cancel: bool, lookup: bool, batch: Mode, defer: bool, even: bool,
    control: bool, request_mode: DraftMode,
) -> ScenarioResult {
    scenario_request_mode(outputs, depth, startup_depth, late, cancel, lookup, batch,
        defer, even, control, request_mode, Some(request_mode), None)
}
#[allow(clippy::too_many_arguments)]
fn scenario_request_mode(
    outputs: &[usize], depth: Option<usize>, startup_depth: usize,
    late: bool, cancel: bool, lookup: bool, batch: Mode, defer: bool, even: bool,
    control: bool, default_mode: DraftMode, request_mode: Option<DraftMode>, head_limit: Option<usize>,
) -> ScenarioResult {
    scenario_request_controls(outputs, depth, startup_depth, late, cancel, lookup, batch,
        defer, even, control, default_mode, request_mode, head_limit, None, None, false)
}
#[allow(clippy::too_many_arguments)]
fn scenario_request_controls(
    outputs: &[usize], depth: Option<usize>, startup_depth: usize,
    late: bool, cancel: bool, lookup: bool, batch: Mode, defer: bool, even: bool,
    control: bool, default_mode: DraftMode, request_mode: Option<DraftMode>, head_limit: Option<usize>,
    depths: Option<&[usize]>, prompts: Option<&[&[u32]]>, admit_together: bool,
) -> ScenarioResult {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let start = Arc::new(AtomicBool::new(false));
    let shutdown = Arc::new(AtomicBool::new(false));
    let proposals = Trace::default();
    let gate = Arc::new(Gate::default());
    if admit_together {
        assert!(!late, "late arrivals must not wait for the initial cohort");
        *gate.admission.lock().unwrap() = Some(Admission {
            expected: outputs.len(), prepared: 0, lookup_check: false, started: Instant::now(),
        });
    }
    gate.armed.store(late, Ordering::Release);
    gate.delayed.store(late || control, Ordering::Release);
    gate.cost_control.store(control, Ordering::Release);
    gate.head_limit.store(head_limit.unwrap_or(0), Ordering::Release);
    let model = Model {
        gate: gate.clone(),
        live: Arc::new(AtomicUsize::new(0)),
        proposals: proposals.clone(),
        batch,
        cohort: Arc::new(AtomicUsize::new(1)),
        defer,
        even,
    };
    let (go, stop, requests) = (
        start.clone(),
        shutdown.clone(),
        outputs.len() + usize::from(cancel),
    );
    let worker = std::thread::spawn(move || {
        while !go.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        scheduler::serve_requests::<Model, engine_tokenizer::Tokenizer>(
            listener,
            &model,
            None,
            &Limits {
                next_mtp_depth: Some(3),
                input_tokens: 8192,
                context_tokens: 8192,
                ..Default::default()
            },
            &CacheLimits {
                entries: 1,
                retained_bytes: 0,
                tokens_per_entry: 1024,
                logits_per_entry: 8,
                idle_ttl: Duration::from_secs(1),
            },
            &scheduler::Config {
                active: 16,
                connections: 32,
                prefill_chunk: 2048,
                foreground_rows: 64,
                background_rows: 64,
                active_bytes: ACTIVE_BYTES,
                session_budget_bytes: 2 << 20,
                draft: default_mode,
                mtp_depth: startup_depth,
                draft_global_bank: false,
                prefix_cache: false,
                sharing: Default::default(),
            },
            &stop,
            "cohort-test",
            requests,
        )
        .unwrap();
    });
    let mut streams = Vec::new();
    for (index, output) in outputs.iter().enumerate() {
        if late && index == 1 {
            start.store(true, Ordering::Release);
            gate.wait_for_verify();
            if cancel {
                let mut control = TcpStream::connect(address).unwrap();
                let body = r#"{"cancel":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#;
                write!(control, "POST /v1/cancel HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
                // Keep the capability request queued while verification owns the session.
                streams.push(control);
            }
        }
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(if admit_together { 45 } else { 10 })))
            .unwrap();
        let body = format!(
            r#"{{"prompt_token_ids":{prompt},"cancel":"{key}","max_tokens":{output},"stream":true,"cache":false,"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":42{draft},"cache_salt":"{oracle}"{depth}}}"#,
            draft = request_mode.map_or_else(String::new, |mode| format!(",\"draft\":\"{}\"",
                match mode { DraftMode::Auto => "auto", DraftMode::Plain => "plain", _ => "strict" })),
            depth = depths.map(|ds| ds[index]).or(depth)
                .map_or_else(String::new, |n| format!(",\"mtp_depth\":{n}")),
            oracle = if lookup { "lookup" } else { "mtp-oracle" },
            prompt = if let Some(prompts) = prompts {
                format!("{:?}", prompts[index])
            } else if outputs == [129, 8] && index > 0 {
                format!(
                    "[{}]",
                    (0..4096)
                        .map(|i| (i % 6 + 1).to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            } else if outputs == [128, 8] && index > 0 {
                "[1,2,5]".into()
            } else {
                "[1,2,3]".into()
            },
            key = if index == 0 {
                "a".repeat(64)
            } else {
                format!("{index:064x}")
            }
        );
        write!(stream, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        streams.push(stream);
        if late && index == 1 {
            gate.release();
        }
    }
    start.store(true, Ordering::Release);
    let mut results = Vec::new();
    for mut stream in streams {
        let mut reply = String::new();
        stream.read_to_string(&mut reply).unwrap();
        assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
        gate.echoes
            .lock()
            .unwrap()
            .extend(reply.split("\"mtp_depth\":").skip(1).map(|s| {
                s.split(|c: char| !c.is_ascii_digit())
                    .next()
                    .unwrap()
                    .parse::<usize>()
                    .unwrap()
            }));
        assert!(
            reply.ends_with("data: [DONE]\n\n") || cancel && reply.contains("cancelled"),
            "{reply}"
        );
        results.push(
            reply
                .split("\"token_id\":")
                .skip(1)
                .map(|value| {
                    value
                        .split(|c: char| !c.is_ascii_digit())
                        .next()
                        .unwrap()
                        .parse()
                        .unwrap()
                })
                .collect(),
        );
    }
    shutdown.store(true, Ordering::Relaxed);
    worker.join().unwrap();
    let trace = proposals.lock().unwrap().clone();
    (results, trace, gate)
}

#[test]
fn strict_mtp_cohort_suspends_and_resumes_without_changing_bytes() {
    let (plain, trace) = run(&[64], 0);
    assert!(trace.is_empty());
    let (solo, trace) = run(&[64], 2);
    assert_eq!(solo, plain);
    assert!(
        trace.iter().any(|&(live, depth)| live == 1 && depth == 2),
        "{trace:?}"
    );
    for count in [8, 16] {
        let (cohort, trace) = run(&vec![64; count], 2);
        assert!(cohort.iter().all(|row| row == &plain[0]));
        assert!(
            trace.iter().all(|&(live, _)| live == 1),
            "C{count}: {trace:?}"
        );
    }
    let (drain, trace) = run(&[8, 64], 2);
    assert_eq!(drain[0], plain[0][..8]);
    assert_eq!(drain[1], plain[0]);
    assert!(
        trace.iter().any(|&(live, depth)| live == 1 && depth == 2),
        "{trace:?}"
    );
    assert!(trace.iter().all(|&(live, _)| live == 1), "{trace:?}");
}

#[test]
fn late_arrival_and_cancel_during_verify_match_plain_and_reuse_slot() {
    let (plain, _, _) = scenario(&[128, 8], 0, false, false);
    for depth in 1..=3 {
        let (actual, trace, gate) = scenario(&[128, 8], depth, true, false);
        assert_eq!(actual, plain, "late arrival k{depth}");
        assert!(trace.iter().all(|&(live, _)| live == 1));
        let verifies = gate.verifies.lock().unwrap();
        let suspended = verifies
            .iter()
            .position(|&(live, width)| live > 1 && width == 1)
            .unwrap();
        assert!(verifies[..suspended].iter().any(|&(_, width)| width > 1));
        assert!(
            verifies[suspended + 1..]
                .iter()
                .any(|&(live, width)| live == 1 && width > 1)
        );
        assert!(gate.rejected.load(Ordering::Relaxed) > 0);
        assert!(gate.accepted.load(Ordering::Relaxed) > 0);
        drop(verifies);
        let (cancelled, _, _) = scenario(&[128, 8], depth, true, true);
        assert_eq!(cancelled[2], plain[1], "post-cancel slot reuse k{depth}");
        assert_eq!(cancelled[0], plain[0][..cancelled[0].len()]);
        assert!(cancelled[0].len() < 128);
    }
}

#[test]
fn configured_2048_prefill_survives_decode_contention() {
    let (plain, _, _) = scenario(&[129, 8], 0, false, false);
    let (actual, _, gate) = scenario(&[129, 8], 2, true, false);
    assert_eq!(actual, plain);
    assert!(
        gate.prefills
            .lock()
            .unwrap()
            .iter()
            .any(|&(live, rows)| live > 1 && rows == 2048)
    );
}

#[test]
fn lookup_verifier_width_does_not_replace_echoed_mtp_depth() {
    let (plain, _) = run(&[64], 0);
    for depth in 1..=3 {
        let (actual, _, gate) = scenario_lookup(&[64], depth, false, false, true);
        assert_eq!(actual, plain);
        assert!(
            gate.verifies
                .lock()
                .unwrap()
                .iter()
                .any(|&(_, width)| width == 16)
        );
        let echoes = gate.echoes.lock().unwrap();
        assert!(!echoes.is_empty() && echoes.iter().all(|&echo| echo == depth));
    }
}

#[test]
fn omitted_request_depth_reaches_adapter_as_startup_default() {
    let (plain, _) = run(&[64], 0);
    for startup in 0..=3 {
        let (actual, trace, gate) =
            scenario_with_defaults(&[64], None, startup, false, false, true, Mode::Off, false);
        assert_eq!(actual, plain);
        assert_eq!(*gate.configured.lock().unwrap(), [startup]);
        let echoes = gate.echoes.lock().unwrap();
        assert!(!echoes.is_empty() && echoes.iter().all(|&n| n == startup));
        if startup == 0 {
            assert!(trace.is_empty());
        } else {
            assert!(
                gate.verifies
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|&(_, rows)| rows == 16)
            );
        }
    }
}

#[test]
fn mtp_batch_off_keeps_single_chat_decision_table() {
    let (plain, _) = run(&[64], 0);
    for count in [2, 5, 8, 9, 16] {
        let (cohort, trace, gate) = batched(&vec![64; count], 3, false);
        assert!(cohort.iter().all(|row| row == &plain[0]), "C{count}");
        assert!(
            trace.iter().all(|&(live, _)| live == 1),
            "C{count}: {trace:?}"
        );
        let tiles = gate.tiles.lock().unwrap();
        assert!(
            tiles.iter().all(|&(live, rows)| live == 1 || rows <= live),
            "C{count}: {tiles:?}"
        );
    }
}

#[test]
fn mtp_batch_on_drafts_cohorts_up_to_eight_within_sixteen_rows() {
    let (plain, _) = run(&[64], 0);
    for depth in [2, 3] {
        let (solo, _, _) = batched(&[64], depth, true);
        assert_eq!(solo, plain, "alone k{depth}");
        for count in 2..=16 {
            let (cohort, trace, gate) = batched(&vec![64; count], depth, true);
            assert!(
                cohort.iter().all(|row| row == &plain[0]),
                "C{count} k{depth}"
            );
            let tiles = gate.tiles.lock().unwrap();
            assert!(
                tiles.iter().all(|&(_, rows)| rows <= 16),
                "C{count} k{depth}: {tiles:?}"
            );
            if count <= 8 {
                // Admission timing decides how many chats overlap in this
                // millisecond-scale fake; any multi-chat draft tile shows the lift.
                assert!(
                    trace.iter().any(|&(live, _)| live >= 2),
                    "C{count} k{depth}: {trace:?}"
                );
                assert!(
                    tiles.iter().any(|&(live, rows)| live >= 2 && rows > live),
                    "C{count} k{depth}: {tiles:?}"
                );
            } else {
                assert!(
                    trace.iter().all(|&(live, _)| live <= 8),
                    "C{count} k{depth}: {trace:?}"
                );
                assert!(
                    tiles.iter().all(|&(live, rows)| live <= 8 || rows <= live),
                    "C{count} k{depth}: {tiles:?}"
                );
            }
            assert!(
                gate.rejected.load(Ordering::Relaxed) + gate.accepted.load(Ordering::Relaxed) > 0
                    || count > 8
            );
        }
    }
}

#[test]
fn mtp_batch_on_drafts_each_tick_in_one_shared_pass() {
    let (plain, _) = run(&[64], 0);
    for depth in [2, 3] {
        for count in [1, 2, 5, 8] {
            let (cohort, inline, gate) = shared(&vec![64; count], depth, true);
            assert!(
                cohort.iter().all(|row| row == &plain[0]),
                "C{count} k{depth}"
            );
            // Every chain went through propose_batch, none through the per-chat path.
            assert!(inline.is_empty(), "C{count} k{depth}: {inline:?}");
            let batches = gate.batches.lock().unwrap();
            assert!(
                batches.iter().all(|&(_, rows)| (1..=8).contains(&rows)),
                "C{count} k{depth}: {batches:?}"
            );
            if count > 1 {
                assert!(
                    batches.iter().any(|&(_, rows)| rows >= 2),
                    "C{count} k{depth}: {batches:?}"
                );
            }
            let tiles = gate.tiles.lock().unwrap();
            assert!(tiles.iter().all(|&(_, rows)| rows <= 16), "C{count} k{depth}: {tiles:?}");
            assert!(gate.rejected.load(Ordering::Relaxed) > 0 && gate.accepted.load(Ordering::Relaxed) > 0);
        }
    }
    // Switch off: FlashNext never defers, so no shared pass and main's table.
    let (cohort, trace, gate) = shared(&[64; 5], 2, false);
    assert!(cohort.iter().all(|row| row == &plain[0]));
    assert!(gate.batches.lock().unwrap().is_empty());
    assert!(trace.iter().all(|&(live, _)| live == 1), "{trace:?}");
}

#[test]
fn auto_tracks_active_cohort_for_scheduler_tables_and_shared_drafts() {
    let (plain, _) = run(&[64], 0);
    for count in [1, 2, 5, 6, 8] {
        let mut outputs = vec![64; count];
        if count > 5 { outputs[..count - 1].fill(8); }
        let (cohort, inline, gate) = scenario_with_defaults(
            &outputs, Some(3), 3, false, false, false, Mode::Auto, true);
        assert!(cohort.iter().zip(outputs).all(|(row, n)| row == &plain[0][..n]), "C{count}");
        assert!(inline.is_empty(), "C{count}: {inline:?}");
        let policies = gate.policies.lock().unwrap();
        assert!(policies.iter().any(|&(n, _)| n == count), "C{count}: {policies:?}");
        assert!(policies.iter().all(|&(n, p)| p == Policy::new((1..=5).contains(&n))));
        let batches = gate.batches.lock().unwrap();
        assert!(batches.iter().all(|&(live, _)| live <= 5), "C{count}: {batches:?}");
        assert!(!batches.is_empty(), "C{count}");
        let tiles = gate.tiles.lock().unwrap();
        assert!(tiles.iter().all(|&(live, rows)| if live <= 5 { rows <= 16 } else { rows <= live }),
            "C{count}: {tiles:?}");
    }
}

#[test]
fn even_cohort_rows_keep_c5_depth3_inside_the_even_share() {
    let (plain, _) = run(&[64], 0);
    for even in [false, true] {
        let (cohort, inline, gate) =
            scenario_rows(&[64; 5], Some(3), 3, false, false, false, Mode::On, true, even);
        assert!(cohort.iter().all(|row| row == &plain[0]), "even={even}");
        assert!(inline.is_empty(), "even={even}: {inline:?}");
        let tiles = gate.tiles.lock().unwrap();
        let five: Vec<_> = tiles.iter().filter(|&&(live, _)| live == 5).map(|&(_, rows)| rows).collect();
        assert!(!five.is_empty(), "even={even}: {tiles:?}");
        if even {
            // 16 rows over five decoders: three each, the spare row buys no drafter step.
            assert!(five.iter().all(|&rows| rows <= 15), "{five:?}");
            assert!(five.contains(&15), "{five:?}");
        } else {
            assert!(five.contains(&16), "{five:?}");
        }
    }
}

#[test]
fn cohort_cost_controller_keeps_http_ids_and_explicit_modes_through_draining() {
    let (plain, _) = run(&[192], 0);
    for count in [1, 5, 6, 7, 8] {
        let outputs = vec![96; count];
        let (cohort, _, gate) = scenario_control(&outputs, Some(3), 3, false, false, false,
            Mode::Auto8, true, true, true, DraftMode::Auto);
        assert!(cohort.iter().all(|row| row == &plain[0][..96]), "C{count}");
        let tiles = gate.tiles.lock().unwrap();
        if count >= 6 {
            let full: Vec<_> = tiles.iter().filter(|&&(live, _)| live == count).map(|&(_, rows)| rows).collect();
            assert!(full.len() > 4, "C{count}: {full:?}");
            let choices: Vec<_> = gate.choices.lock().unwrap().iter().flatten().copied().collect();
            assert!(choices.len() >= 4, "C{count}: {choices:?}");
            // Incomplete initial frames do not count as cost samples, so a
            // probe may need more than two actual scheduler selections.
            let first_draft = choices.iter().position(|&draft| draft).unwrap();
            let first_plain = first_draft + choices[first_draft..].iter().position(|&draft| !draft).unwrap();
            assert!(first_plain - first_draft >= 2, "C{count}: {choices:?}");
            assert!(choices[first_plain..].iter().take_while(|&&draft| !draft).count() >= 2);
            assert!(full.contains(&count), "C{count}: no plain target pass");
            assert!(full.iter().any(|&rows| rows > count), "C{count}: no draft probe");
            assert!(full.iter().all(|&rows| rows <= 2 * count));
        }
    }
    let outputs = [65, 64, 63, 130, 129, 128, 96, 95];
    let (cohort, _, _) = scenario_control(&outputs, Some(3), 3, false, false, false,
        Mode::Auto8, true, true, true, DraftMode::Auto);
    assert!(cohort.iter().zip(outputs).all(|(row, n)| row == &plain[0][..n]));
    for mode in [DraftMode::Strict, DraftMode::Plain] {
        let (cohort, _, gate) = scenario_control(&[96; 8], Some(3), 3, false, false, false,
            Mode::Auto8, true, true, true, mode);
        assert!(cohort.iter().all(|row| row == &plain[0][..96]));
        let tiles = gate.tiles.lock().unwrap();
        let full: Vec<_> = tiles.iter().filter(|&&(live, _)| live == 8)
            .map(|&(_, rows)| rows).collect();
        assert!(!full.is_empty());
        assert!(full.iter().all(|&rows| rows == 8), "ineligible modes keep main's C8 policy: {full:?}");
        if mode == DraftMode::Plain { assert!(tiles.iter().all(|&(live, rows)| rows <= live)); }
        let choices = gate.choices.lock().unwrap();
        assert!(choices.contains(&Some(false)));
        assert!(!choices.contains(&Some(true)), "explicit modes must not be cost-probed");
    }
}

#[test]
fn cohort_cost_controller_probes_unspecified_requests_with_production_strict_default() {
    let (plain, _) = run(&[96], 0);
    for count in [6, 8] {
        let (rows, _, gate) = scenario_request_mode(&vec![96; count], None, 3,
            false, false, false, Mode::Auto8, true, true, true, DraftMode::Strict, None, None);
        assert!(rows.iter().all(|row| row == &plain[0]));
        let tiles = gate.tiles.lock().unwrap();
        let full: Vec<_> = tiles.iter().filter(|&&(live, _)| live == count)
            .map(|&(_, rows)| rows).collect();
        assert!(full.len() > 4, "C{count}: {full:?}");
        let selected: Vec<_> = gate.choices.lock().unwrap().iter().flatten().copied().collect();
        assert!(selected.len() >= 4, "C{count}: {selected:?}");
        let first_draft = selected.iter().position(|&draft| draft).unwrap();
        let first_plain = first_draft + selected[first_draft..].iter().position(|&draft| !draft).unwrap();
        assert!(first_plain - first_draft >= 2, "C{count}: {selected:?}");
        assert!(selected[first_plain..].iter().take_while(|&&draft| !draft).count() >= 2);
        assert!(full.contains(&count), "missing draft must also run plain target passes");
        assert!(full.iter().any(|&rows| rows > count), "missing draft must run draft probes");
        let choices = gate.choices.lock().unwrap();
        assert!(choices.contains(&Some(false)) && choices.contains(&Some(true)));
    }
}

#[test]
fn cohort_head_coverage_hint_keeps_uncovered_targets_plain_and_exact() {
    let (plain, _) = run(&[96], 0);
    for count in [6, 8] {
        // The oracle emits IDs 1..6; a one-row prefix head covers only ID0.
        let (rows, _, gate) = scenario_request_mode(&vec![96; count], None, 3,
            false, false, false, Mode::Auto8, true, true, true,
            DraftMode::Strict, None, Some(1));
        assert!(rows.iter().all(|row| row == &plain[0]));
        let choices = gate.choices.lock().unwrap();
        assert!(choices.contains(&Some(false)));
        assert!(!choices.contains(&Some(true)));
        let tiles = gate.tiles.lock().unwrap();
        let full: Vec<_> = tiles.iter().filter(|&&(live, _)| live == count)
            .map(|&(_, rows)| rows).collect();
        assert!(!full.is_empty());
        assert!(full.iter().all(|&rows| rows == count), "C{count}: {full:?}");
    }
}

#[test]
fn partially_uncovered_cohort_keeps_all_target_rows_plain() {
    let prompts: [&[u32]; 8] = [&[1, 2, 1], &[1, 2, 2], &[1, 2, 3], &[1, 2, 4],
        &[1, 2, 5], &[1, 2, 6], &[1, 2, 1], &[1, 2, 2]];
    // TCP writes and CPU preparation can straggle past the scheduler's
    // bounded collection window. Admit all eight before the first prefill
    // so every target pass exercises the intended complete cohort.
    let (rows, _, gate) = scenario_request_controls(&[96; 8], None, 3,
        false, false, false, Mode::Auto8, true, true, true,
        DraftMode::Strict, None, Some(4), None, Some(&prompts), true);
    let first: Vec<_> = rows.iter().map(|r| r[0]).collect();
    assert!(first.iter().any(|&id| id < 4) && first.iter().any(|&id| id >= 4), "{first:?}");
    for (row, prompt) in rows.iter().zip(prompts) {
        let mut history = prompt.to_vec();
        for &actual in row {
            let expected = logits(&history).iter().position(|&v| v == 1.).unwrap() as u32;
            assert_eq!(actual, expected);
            history.push(expected);
        }
    }
    let choices = gate.choices.lock().unwrap();
    assert!(choices.contains(&Some(false)));
    assert!(!choices.contains(&Some(true)), "{choices:?}");
    let tiles = gate.tiles.lock().unwrap();
    let full: Vec<_> = tiles.iter().filter(|&&(live, _)| live == 8)
        .map(|&(_, rows)| rows).collect();
    assert!(!full.is_empty());
    assert!(full.iter().all(|&rows| rows == 8), "{full:?}");
}

#[test]
fn unmeasurable_mixed_depths_keep_automatic_c8_plain_and_exact() {
    let (plain, _) = run(&[96], 0);
    for mode in [None, Some(DraftMode::Auto)] {
        let (rows, _, gate) = scenario_request_controls(&[96; 8], None, 3,
            false, false, false, Mode::Auto8, true, true, true,
            DraftMode::Strict, mode, None, Some(&[0, 1, 2, 3, 0, 1, 2, 3]), None, false);
        assert!(rows.iter().all(|row| row == &plain[0]));
        let choices = gate.choices.lock().unwrap();
        assert!(choices.contains(&Some(false)));
        assert!(!choices.contains(&Some(true)), "{choices:?}");
        let tiles = gate.tiles.lock().unwrap();
        let full: Vec<_> = tiles.iter().filter(|&&(live, _)| live == 8)
            .map(|&(_, rows)| rows).collect();
        assert!(!full.is_empty());
        assert!(full.iter().all(|&rows| rows == 8), "{full:?}");
        let mut configured = gate.configured.lock().unwrap().clone();
        configured.sort_unstable();
        assert_eq!(configured, [0, 0, 1, 1, 2, 2, 3, 3]);
        // All requested depths reached the sessions, but the resolved plain
        // policy reports zero effective depth in every response.
        assert_eq!(*gate.echoes.lock().unwrap(), [0; 8]);
    }
}

#[test]
fn cohort_cost_controller_handles_late_arrivals_and_cancellation() {
    let (plain, _) = run(&[128], 0);
    let outputs = [128, 96, 95, 94, 65, 64, 63, 62];
    let (rows, _, _) = scenario_control(&outputs, Some(3), 3, true, false, false,
        Mode::Auto8, true, true, true, DraftMode::Auto);
    assert!(rows.iter().zip(outputs).all(|(row, n)| row == &plain[0][..n]));
    let (rows, _, _) = scenario_control(&outputs, Some(3), 3, true, true, false,
        Mode::Auto8, true, true, true, DraftMode::Auto);
    assert!(rows[0].len() < 128);
    assert_eq!(rows[0], plain[0][..rows[0].len()]);
    assert!(rows[1].is_empty()); // cancellation control response
    assert!(rows[2..].iter().zip(&outputs[1..]).all(|(row, &n)| row == &plain[0][..n]));
}
