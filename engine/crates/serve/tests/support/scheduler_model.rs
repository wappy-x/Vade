//! Excludes intake, preparation, cache, drafting, errors, real devices, and deadline transitions.
//! Dispatch, prompt cohorts, generation, response writes and retirement are real.
use super::*;
use std::{cell::RefCell, sync::{Arc, Mutex, atomic::AtomicUsize}};

include!("scheduler_engine.rs");

const SLICE: usize = 4;
const WORK_BOUND: usize = 4;
const TURN_BOUND: usize = 24;
const WAIT_BOUND: usize = 8;
const CANCEL_LAST: usize = 8;
const FUNDED_BYTES: usize = 2 * 16384;
thread_local! { pub(super) static CLOCK: RefCell<Option<Instant>> = const { RefCell::new(None) }; }
pub(super) fn clock_now() -> Instant { CLOCK.with_borrow(|c| c.unwrap_or_else(Instant::now)) }
pub(super) fn clock_elapsed(start: &Instant) -> Duration { clock_now().saturating_duration_since(*start) }
pub(super) struct ClockGuard;
impl ClockGuard {
    pub(super) fn new() -> Self { CLOCK.with_borrow_mut(|c| *c = Some(Instant::now())); Self }
}
impl Drop for ClockGuard { fn drop(&mut self) { CLOCK.with_borrow_mut(|c| *c = None); } }

#[derive(Clone, Copy, Debug)]
struct Spec { arrival: usize, prompt: usize, output: usize }
impl Spec {
    fn work(self) -> usize { self.prompt.div_ceil(SLICE) + self.output - 1 }
}
#[derive(Clone)]
struct Case { requests: Vec<Spec>, cancel: Option<(usize, usize)> }
impl std::fmt::Debug for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows: Vec<_> = self.requests.iter().map(|s| (s.arrival, s.prompt, s.output)).collect();
        write!(f, "rows(arrival,prompt,output)={rows:?} cancel(request,turn)={:?}", self.cancel)
    }
}
fn site(name: &str, needle: &str) -> String {
    let source = match name {
        "dispatch.rs" => include_str!("../../src/scheduler/dispatch.rs"),
        _ => include_str!("../../src/scheduler.rs"),
    };
    format!("{name}:{}", source.lines().position(|line| line.contains(needle)).unwrap() + 1)
}

// Adapt the existing fake, without its deliberate per-token sleep. Keep its
// history-dependent logits so a wrong row, order or prompt slice changes IDs.
struct Counted(Model, Arc<Mutex<Coverage>>);
struct CountedKv(Kv, bool, bool);
impl Drop for CountedKv {
    fn drop(&mut self) {
        let fault = DRIVER.with_borrow(|d| d.as_ref().is_some_and(|d| d.fault.is_some()));
        if !std::thread::panicking() && !fault { assert!(self.1 || !self.2, "completed session dropped without retirement"); }
    }
}
impl Session for CountedKv {
    fn retire(&mut self) { assert!(!self.1, "session retired twice"); self.1 = true; }
    fn advance(&mut self, token: u32) -> Result<(), String> {
        self.0.calls.fetch_add(1, Ordering::Relaxed);
        self.0.history.push(token); Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.advance(token)?;
        self.0.heads.fetch_add(1, Ordering::Relaxed);
        Ok(self.0.logits())
    }
    fn prefill_chunk_size(&self, _: usize) -> usize { SLICE }
    fn prefill_chunk_min(&self) -> usize { SLICE }
}
#[derive(Clone, Copy, Default, Debug)]
struct Coverage { prompt_group: bool, decode_pair: bool, sliced: bool, prompt_cancel: bool, decode_cancel: bool, queued_cancel: bool }
impl Coverage {
    fn merge(&mut self, other: Self) {
        self.prompt_group |= other.prompt_group; self.decode_pair |= other.decode_pair;
        self.sliced |= other.sliced; self.prompt_cancel |= other.prompt_cancel;
        self.decode_cancel |= other.decode_cancel; self.queued_cancel |= other.queued_cancel;
    }
}
impl Engine for Counted {
    type Session<'a> = CountedKv;
    fn session(&self) -> Result<CountedKv, String> {
        assert!((self.0.live.load(Ordering::Relaxed) + 1) * 16384 <= FUNDED_BYTES, "unfunded session creation");
        let normal = DRIVER.with_borrow(|d| d.as_ref().unwrap().case.cancel.is_none());
        self.0.session().map(|kv| CountedKv(kv, false, normal))
    }
    fn shared_prefill_batch(&self) -> bool { true }
    fn shared_prefill_limit(&self) -> usize { SLICE }
    fn shared_prefill_rows(&self) -> usize { 4 * SLICE }
    fn shared_prefill_max_tokens(&self) -> usize { 9 }
    fn shared_decode_rows(&self) -> usize { 4 }
    fn shared_decode_min(&self) -> usize { 2 }
    fn prefill_batch(&self, rows: &mut [crate::BatchRows<'_, CountedKv>]) -> Result<Vec<crate::Predictions>, crate::BatchError> {
        self.1.lock().unwrap().prompt_group |= rows.len() > 1;
        self.step_batch(rows)
    }
    fn step_batch(&self, rows: &mut [crate::BatchRows<'_, CountedKv>]) -> Result<Vec<crate::Predictions>, crate::BatchError> {
        if rows.len() == 2 && rows.iter().all(|r| matches!(r.output, crate::BatchOutput::All)) {
            self.1.lock().unwrap().decode_pair = true;
        }
        // The same default batch executor as Engine uses, through the reused fake.
        rows.iter_mut().map(|r| {
            let result = match r.output {
                crate::BatchOutput::All => r.tokens.iter().map(|&t| r.session.step(t)).collect(),
                crate::BatchOutput::Last => r.session.prefill(r.tokens).map(|v| vec![v]),
                crate::BatchOutput::None => r.session.prefill_advance(r.tokens).map(|()| Vec::new()),
            };
            result.map(crate::Predictions::Logits).map_err(|message| crate::BatchError { request: None, message })
        }).collect()
    }
    fn context_limit(&self) -> usize { self.0.context_limit() }
    fn vocab_size(&self) -> usize { self.0.vocab_size() }
    fn eos_token_ids(&self) -> &[u32] { self.0.eos_token_ids() }
    fn session_reservation_bytes(&self) -> Option<usize> { self.0.session_reservation_bytes() }
}

struct Peer {
    socket: UnixStream, reply: String, key: SessionKey, expected: Vec<u32>,
    terminal: Option<usize>, cancelled: bool, progress: (usize, usize), wait: usize,
}
struct Driver {
    case: Case, pending: Vec<Option<Ready>>, peers: Vec<Peer>, turn: usize,
    fault: Option<String>, coverage: Coverage, max_wait: usize, max_reserved: usize,
}
thread_local! { static DRIVER: RefCell<Option<Driver>> = const { RefCell::new(None) }; }
pub(super) fn driving() -> bool { DRIVER.with(|d| d.borrow().is_some()) }

// Called only in test builds. No driver means no changes for other host tests.
pub(super) fn age_prompts<E: Engine, T: Tokenizer>(active: &mut [Active<'_, '_, E, T>]) {
    if driving() {
        for job in active { job.admitted = job.ready.enqueued; }
    }
}
pub(super) fn turn<E: Engine, T: Tokenizer>(queued: &mut Vec<Ready>, cancels: &mut Vec<SessionKey>,
    active: &[Active<'_, '_, E, T>], retired: &[Retired<E::Session<'_>>], reading: usize, cache_bytes: usize) -> bool {
    DRIVER.with_borrow_mut(|driver| {
        let Some(driver) = driver else { return false; };
        match driver.tick(queued, cancels, active, retired, reading, cache_bytes) {
            Ok(done) => done,
            Err(fault) => { driver.fault = Some(fault); true }
        }
    })
}
fn ids(reply: &str) -> Vec<u32> {
    reply.lines().filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| line.contains("\"token_id\":"))
        .map(|line| line.split("\"token_id\":").nth(1).unwrap()
            .split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap()).collect()
}
impl Driver {
    fn tick<E: Engine, T: Tokenizer>(&mut self, queued: &mut Vec<Ready>, cancels: &mut Vec<SessionKey>,
        active: &[Active<'_, '_, E, T>], retired: &[Retired<E::Session<'_>>], reading: usize, cache_bytes: usize) -> Result<bool, String> {
        if self.turn > TURN_BOUND { return Err(format!("loop bound {TURN_BOUND}; {}", site("dispatch.rs", "while !shutdown"))); }
        let reserved = active.iter().map(|j| j.generation.state.reserved_session_bytes).sum::<usize>()
            + retired.iter().map(|j| j.state.reserved_session_bytes).sum::<usize>() + cache_bytes;
        self.max_reserved = self.max_reserved.max(reserved);
        if reserved > FUNDED_BYTES { return Err(format!("reservation {reserved} exceeds pool {FUNDED_BYTES}")); }
        if active.iter().any(|j| j.generation.state.reserved_session_bytes != 16384)
            || retired.iter().any(|j| j.state.reserved_session_bytes != 16384) {
            return Err("session reservation differs from the fake allocation".into());
        }
        let mut states = Vec::new();
        let mut others_progress = false;
        for (i, peer) in self.peers.iter_mut().enumerate() {
            let mut bytes = [0; 8192];
            loop {
                match peer.socket.read(&mut bytes) {
                    Ok(0) => break,
                    Ok(n) => peer.reply.push_str(std::str::from_utf8(&bytes[..n]).unwrap()),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(e) => return Err(format!("response read request {i}: {e}")),
                }
            }
            let tokens = ids(&peer.reply);
            if peer.terminal.is_some() && tokens.len() != peer.progress.1 {
                return Err(format!("tokens after finish request {i}; {}", site("scheduler.rs", "fn finish(&mut self, reason")));
            }
            if !peer.expected.starts_with(&tokens) {
                return Err(format!("token order or solo equality request {i}: {tokens:?}, expected {:?}; {}", peer.expected, site("dispatch.rs", "complete_batch(engine.step_batch")));
            }
            let terminals = peer.reply.matches("data: [DONE]\n\n").count();
            if terminals > 1 { return Err(format!("duplicate finish request {i}; {}", site("scheduler.rs", "fn finish(&mut self, reason"))); }
            if terminals == 1 && peer.terminal.is_none() {
                peer.terminal = Some(self.turn);
                if peer.reply.contains("\"type\":\"client_cancelled\"") != peer.cancelled {
                    return Err(format!("wrong terminal reason request {i}; {}", site("dispatch.rs", "let mut job = active.remove(i)")));
                }
                if !peer.cancelled && tokens != peer.expected {
                    return Err(format!("token count request {i}: {tokens:?}, expected {:?}; {}", peer.expected, site("scheduler.rs", "fn finish(&mut self, reason")));
                }
                if !peer.cancelled && !peer.reply.contains("\"finish_reason\":\"length\"") {
                    return Err(format!("missing length finish request {i}; {}", site("scheduler.rs", "fn finish(&mut self, reason")));
                }
            }
            let job = active.iter().find(|j| j.ready.cancel == peer.key);
            let position = job.map_or(peer.progress.0, |j| j.generation.state.tokens.len());
            let progress = (position, tokens.len());
            others_progress |= progress != peer.progress;
            states.push(progress);
        }
        for (i, (peer, progress)) in self.peers.iter_mut().zip(states).enumerate() {
            if progress != peer.progress { peer.wait = 0; }
            else if others_progress && peer.terminal.is_none() && self.case.requests[i].arrival < self.turn { peer.wait += 1; }
            self.max_wait = self.max_wait.max(peer.wait);
            if peer.wait > WAIT_BOUND { return Err(format!("wait bound {WAIT_BOUND} request {i}; {}", site("dispatch.rs", "let aged = jobs"))); }
            if let Some(job) = active.iter().find(|j| j.ready.cancel == peer.key) {
                self.coverage.sliced |= progress.0 > 0 && progress.0 < job.ready.prompt.len();
            }
            peer.progress = progress;
        }
        for (i, pending) in self.pending.iter_mut().enumerate() {
            if self.case.requests[i].arrival == self.turn {
                queued.push(pending.take().unwrap());
            }
        }
        if let Some((i, at)) = self.case.cancel && at == self.turn {
            let peer = &mut self.peers[i];
            if queued.iter().any(|j| j.cancel == peer.key) {
                peer.cancelled = true; self.coverage.queued_cancel = true;
            }
            if let Some(job) = active.iter().find(|j| j.ready.cancel == peer.key) {
                peer.cancelled = true;
                if job.generation.state.tokens.len() < job.ready.prompt.len() { self.coverage.prompt_cancel = true; }
                else { self.coverage.decode_cancel = true; }
            }
            cancels.push(peer.key);
        }
        let quiet = self.pending.iter().all(Option::is_none) && queued.is_empty() && active.is_empty() && retired.is_empty() && reading == 0;
        if quiet {
            if reserved != 0 { return Err(format!("funded memory {reserved} at quiescence; {}", site("dispatch.rs", "state.session.retire()"))); }
            if self.peers.iter().any(|p| p.terminal.is_none()) { return Err(format!("missing finish; {}", site("scheduler.rs", "fn terminal_events"))); }
            return Ok(true);
        }
        self.turn += 1;
        CLOCK.with_borrow_mut(|c| *c = c.map(|now| now + Duration::from_millis(1)));
        Ok(false)
    }
}

// Never print response bodies: they contain runtime cancellation capabilities.
fn run(listener: &TcpListener, case: Case, alone: &[Vec<Vec<u32>>]) -> Result<Driver, String> {
    let _clock = ClockGuard::new();
    let limits = Limits { intake_burst: Duration::ZERO, generation_timeout: Duration::from_secs(3600),
        write_timeout: Duration::from_secs(3600), ..Limits::default() };
    let mut driver = Driver { case: case.clone(), pending: Vec::new(), peers: Vec::new(), turn: 0,
        fault: None, coverage: Coverage::default(), max_wait: 0, max_reserved: 0 };
    for (i, spec) in case.requests.iter().enumerate() {
        let (server, client) = UnixStream::pair().unwrap(); client.set_nonblocking(true).unwrap();
        let prompt = vec![1; spec.prompt];
        let body = format!(r#"{{"prompt_token_ids":{prompt:?},"max_tokens":{},"stream":true,"cache":false,"prefix_cache":false,"draft":"plain","ignore_eos":true,"temperature":0,"top_k":1,"top_p":1,"seed":0}}"#, spec.output);
        let input = http::HttpRequest { path: "/v1/completions".into(), body: body.into_bytes(), authorization: None };
        let mut wire = Wire::new(Socket::Admin(server)).unwrap(); wire.received -= Duration::from_secs(1);
        let prepared = prepare::<NoCodec>(input, wire, None, &limits, "model-test", 1024, 8, &[7])
            .map_err(|(_, error)| format!("prepare request {i}: {error}"))?;
        let Prepared::Generation(ready) = prepared else { unreachable!() };
        let expected = alone[match spec.prompt { 1 => 0, 4 => 1, 9 => 2, _ => unreachable!() }][spec.output - 1].clone();
        driver.peers.push(Peer { socket: client, reply: String::new(), key: ready.cancel, expected,
            terminal: None, cancelled: false, progress: (0, 0), wait: 0 });
        driver.pending.push(Some(ready));
    }
    let live = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let coverage = Arc::new(Mutex::new(Coverage::default()));
    let engine = Counted(Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(),
        heads: Arc::new(AtomicUsize::new(0)), order: Default::default(), opt_in: false, retention_extra: 0 }, coverage.clone());
    let config = Config { active: 4, connections: 4, prefill_chunk: SLICE, active_bytes: FUNDED_BYTES,
        session_budget_bytes: FUNDED_BYTES, prefix_cache: false, mtp_depth: 0, foreground_rows: 4, background_rows: 4,
        draft: protocol::DraftMode::Plain, draft_global_bank: false, sharing: Sharing::default() };
    let cache = CacheLimits { entries: 1, retained_bytes: 0, tokens_per_entry: 1024, logits_per_entry: 8, idle_ttl: Duration::from_secs(1) };
    let (tx, _rx) = mpsc::sync_channel(1); let (_tx, rx) = mpsc::sync_channel(1);
    DRIVER.with_borrow_mut(|d| *d = Some(driver));
    let result = serve_loop::<_, NoCodec>(listener, &engine, None, &limits, &cache, &config,
        &AtomicBool::new(false), None, "model-test", 0, &tx, &rx);
    let mut driver = DRIVER.with_borrow_mut(|d| d.take().unwrap());
    result.map_err(|e| format!("serve loop: {e}"))?;
    if let Some(fault) = driver.fault.take() { return Err(format!("{case:?}: turn {}: {fault}", driver.turn)); }
    if live.load(Ordering::Relaxed) != 0 { return Err(format!("{case:?}: session funding remains after serve_loop")); }
    if calls.load(Ordering::Relaxed) > case.requests.iter().map(|s| s.prompt + s.output - 1).sum() {
        return Err(format!("{case:?}: repeated engine work; dispatch.rs plans"));
    }
    driver.coverage.merge(*coverage.lock().unwrap());
    Ok(driver)
}
struct NoCodec;
impl Decoder for NoCodec {
    fn push(&mut self, _: u32) -> Result<String, String> { unreachable!() }
    fn finish(&mut self) -> Result<String, String> { unreachable!() }
}
impl Tokenizer for NoCodec {
    type Decoder<'a> = Self;
    fn encode(&self, _: &str) -> Result<Vec<u32>, String> { unreachable!() }
    fn decoder(&self) -> Self { unreachable!() }
    fn has_token(&self, _: u32) -> bool { true }
    fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { unreachable!() }
}

fn oracle() -> Vec<Vec<Vec<u32>>> {
    [1, 4, 9].into_iter().map(|prompt| (1..=3).map(|output| {
        let mut history = vec![1; prompt];
        (0..output).map(|_| {
            let token = history.iter().fold(0, |h, &t| (h * 17 + t + 1) % 7);
            history.push(token); token
        }).collect()
    }).collect()).collect()
}

#[test]
fn scheduler_model_funding_and_release() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let case = Case { requests: vec![Spec { arrival: 0, prompt: 1, output: 1 }; 4], cancel: None };
    let result = run(&listener, case, &oracle()).unwrap();
    assert_eq!(result.max_reserved, FUNDED_BYTES, "pool pressure was not exercised");
}

#[test]
fn scheduler_model_cancel_terminal_type() {
    // Minimal case from the first run. The raw completion route reports the
    // cancellation in error.type. Requiring error.code was a wrong invariant.
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let case = Case { requests: vec![Spec { arrival: 0, prompt: 1, output: 1 }], cancel: Some((0, 0)) };
    let result = run(&listener, case, &oracle()).unwrap();
    assert_eq!(ids(&result.peers[0].reply), Vec::<u32>::new());
    assert!(result.peers[0].reply.contains("\"type\":\"client_cancelled\""));
    assert!(!result.peers[0].reply.contains("\"code\":\"client_cancelled\""));
    assert_eq!(result.peers[0].reply.matches("data: [DONE]\n\n").count(), 1);
}

#[test]
fn scheduler_model_small_cases() {
    let start = Instant::now();
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let mut alone = oracle();
    // Measure each solo reference through dispatch, then use its actual IDs.
    for (p, prompt) in [1, 4, 9].into_iter().enumerate() { for output in 1..=3 {
        let case = Case { requests: vec![Spec { arrival: 0, prompt, output }], cancel: None };
        let result = run(&listener, case, &alone).unwrap(); alone[p][output - 1] = ids(&result.peers[0].reply);
    } }
    let mut types = Vec::new();
    for arrival in [0, 1, 3] { for prompt in [1, 4, 9] { for output in 1..=3 { types.push(Spec { arrival, prompt, output }); } } }
    let (mut scenarios, mut bases, mut coverage, mut max_wait, mut max_turns) = (0, 0, Coverage::default(), 0, 0);
    let mut failures = Vec::new();
    fn enumerate(types: &[Spec], requests: &mut Vec<Spec>, work: usize, visit: &mut impl FnMut(&[Spec])) {
        if !requests.is_empty() { visit(requests); }
        if requests.len() == 4 { return; }
        for &spec in types {
            if requests.last().is_some_and(|last| spec.arrival < last.arrival) { continue; }
            let next = work + spec.work(); if next > WORK_BOUND { continue; }
            requests.push(spec); enumerate(types, requests, next, visit); requests.pop();
        }
    }
    enumerate(&types, &mut Vec::new(), 0, &mut |requests| {
        let case = Case { requests: requests.to_vec(), cancel: None };
        bases += 1; scenarios += 1;
        let result = match run(&listener, case.clone(), &alone) {
            Ok(result) => result,
            Err(error) => { failures.push(error); return; }
        };
        coverage.merge(result.coverage); max_wait = max_wait.max(result.max_wait); max_turns = max_turns.max(result.turn);
        // Input enumeration must not depend on the observed terminal turn.
        for (i, spec) in requests.iter().enumerate() { for at in spec.arrival..=CANCEL_LAST {
            let mut case = case.clone(); case.cancel = Some((i, at)); scenarios += 1;
            match run(&listener, case, &alone) {
                Ok(result) => { coverage.merge(result.coverage); max_wait = max_wait.max(result.max_wait); max_turns = max_turns.max(result.turn); }
                Err(error) => failures.push(error),
            }
        } }
    });
    eprintln!("scheduler_model: bases={bases} scenarios={scenarios} failures={} elapsed={:?} work_bound={WORK_BOUND} turn_bound={TURN_BOUND} wait_bound={WAIT_BOUND} max_turns={max_turns} max_wait={max_wait} coverage={coverage:?}", failures.len(), start.elapsed());
    for failure in &failures { eprintln!("finding: {failure}"); }
    assert!(failures.is_empty(), "scheduler_model: {} failing scenarios", failures.len());
    assert_eq!(bases, 752, "request enumeration changed");
    assert_eq!(scenarios, 18232, "cancellation enumeration changed");
    assert!(coverage.prompt_group && coverage.decode_pair && coverage.sliced && coverage.prompt_cancel && coverage.decode_cancel && coverage.queued_cancel, "missing path coverage: {coverage:?}");
}
