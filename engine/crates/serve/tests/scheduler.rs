use spark_serve::{
    Decoder, Engine, Limits, Session, Tokenizer, cache::CacheLimits, protocol, scheduler,
};
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

struct Model {
    ab: Option<Arc<Mutex<Vec<bool>>>>,
    /// VADE_AB_COHORTS: arm of the live sessions and the most sessions seen live at once.
    cohort: Option<Arc<Mutex<(Option<bool>, usize)>>>,
    live: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    heads: Arc<AtomicUsize>,
    /// Session::settle_prefix and Session::fork, in the order they ran.
    order: Arc<Mutex<Vec<&'static str>>>,
    /// Engine::prefix_cache_opt_in (Flash-Next): only asking requests use the prefix cache.
    opt_in: bool,
    retention_extra: usize,
}
struct Kv {
    history: Vec<u32>,
    live: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    heads: Arc<AtomicUsize>,
    order: Arc<Mutex<Vec<&'static str>>>,
}
impl Drop for Kv {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
    }
}
impl Session for Kv {
    fn advance(&mut self, token: u32) -> Result<(), String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(2));
        self.history.push(token);
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.advance(token)?;
        self.heads.fetch_add(1, Ordering::Relaxed);
        let hash = self
            .history
            .iter()
            .fold(0usize, |h, &t| (h * 17 + t as usize + 1) % 7);
        let mut logits = vec![0.; 8];
        logits[hash] = 1.;
        Ok(logits)
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> {
        self.history.truncate(position);
        Ok(())
    }
    fn settle_prefix(&mut self) -> Result<(), String> {
        self.order.lock().unwrap().push("settle");
        Ok(())
    }
    fn fork(&self) -> Result<Self, String> {
        self.live.fetch_add(1, Ordering::Relaxed);
        self.order.lock().unwrap().push("fork");
        Ok(Self {
            history: self.history.clone(),
            live: self.live.clone(),
            calls: self.calls.clone(),
            heads: self.heads.clone(),
            order: self.order.clone(),
        })
    }
}
impl Engine for Model {
    fn kv_branch_handles(&self) -> bool { true }
    fn prefix_cache_opt_in(&self) -> bool { self.opt_in }
    fn automatic_draft_allowed(&self, _: usize) -> bool { false }
    type Session<'a> = Kv;
    fn session_for_request(&self, request: &protocol::Request) -> Result<Kv, String> {
        if let Some(arms) = &self.ab {
            let live = self.live.load(Ordering::Relaxed);
            if let Some(cohort) = &self.cohort {
                let mut cohort = cohort.lock().unwrap();
                assert!(live == 0 || cohort.0 == Some(request.arm_x), "A/B cohort mixed arms");
                *cohort = (Some(request.arm_x), cohort.1.max(live + 1));
            } else {
                assert_eq!(live, 0, "A/B requests overlapped or reused state");
            }
            arms.lock().unwrap().push(request.arm_x);
        }
        self.session()
    }
    fn session(&self) -> Result<Kv, String> {
        self.live.fetch_add(1, Ordering::Relaxed);
        Ok(Kv {
            history: Vec::new(),
            live: self.live.clone(),
            calls: self.calls.clone(),
            heads: self.heads.clone(),
            order: self.order.clone(),
        })
    }
    fn vocab_size(&self) -> usize {
        8
    }
    fn context_limit(&self) -> usize {
        1024
    }
    fn eos_token_ids(&self) -> &[u32] {
        &[7]
    }
    fn session_reservation_bytes(&self) -> Option<usize> {
        Some(16384)
    }
    fn request_reservation_bytes_with_retention(&self, _: usize, _: bool, _: Option<usize>, retain: bool) -> Option<usize> {
        16384usize.checked_add(if retain { self.retention_extra } else { 0 })
    }
}
// Typed hybrid scheduling contract, without GPU arithmetic or per-token sleeps.
type PrefillTrace = Arc<Mutex<Vec<Vec<usize>>>>;
/// The model, its prompt pass trace and the start of its last loop turn.
type LastTurn = Arc<Mutex<Option<Instant>>>;
struct Tiled(Model, Option<PrefillTrace>, LastTurn);
struct TileKv(Kv);
struct TileLong(Vec<u32>);
impl spark_serve::LongPrefill<TileKv> for TileLong {
    fn advance(&mut self, session: &mut TileKv, _: Duration) -> Result<Option<Vec<f32>>, String> {
        session.prefill(&self.0).map(Some)
    }
}
impl Session for TileKv {
    fn advance(&mut self, token: u32) -> Result<(), String> { self.0.history.push(token); Ok(()) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> { self.0.step(token) }
    fn prefill_chunk_size(&self, _: usize) -> usize { 256 - self.0.history.len() % 256 }
    fn prefill_chunk_min(&self) -> usize { self.prefill_chunk_size(0) }
    fn truncate(&mut self, position: usize) -> Result<(), String> { self.0.truncate(position) }
    fn fork(&self) -> Result<Self, String> { self.0.fork().map(Self) }
}
impl Engine for Tiled {
    type Session<'a> = TileKv;
    fn session(&self) -> Result<TileKv, String> { self.0.session().map(TileKv) }
    fn shared_batch(&self) -> bool { true }
    fn native_draft_policy(&self) -> bool { true }
    fn shared_prefill_limit(&self) -> usize { 512 }
    fn shared_prefill_rows(&self) -> usize { if self.1.is_some() { 1024 } else { 256 } }
    fn shared_prefill_chunk_rows(&self) -> usize { if self.1.is_some() { 256 } else { 0 } }
    fn profile_tick_begin(&self, _: u64) { *self.2.lock().unwrap() = Some(Instant::now()); }
    fn prefill_long<'s>(&'s self, _: &TileKv, tokens: &[u32])
        -> Result<Option<Box<dyn spark_serve::LongPrefill<TileKv> + 's>>, String> {
        if let Some(trace) = &self.1 {
            trace.lock().unwrap().push(vec![0, tokens.len()]);
            return Ok(Some(Box::new(TileLong(tokens.to_vec()))));
        }
        Ok(None)
    }
    fn prefill_batch<'s>(&'s self, rows: &mut [spark_serve::BatchRows<'_, TileKv>])
        -> Result<Vec<spark_serve::Predictions>, spark_serve::BatchError> {
        if let Some(trace) = &self.1 { trace.lock().unwrap().push(rows.iter().map(|r| r.tokens.len()).collect()); }
        self.step_batch(rows)
    }
    fn context_limit(&self) -> usize { 8192 }
    fn vocab_size(&self) -> usize { 8 }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { Some(65536) }
}
#[allow(dead_code)]
#[path = "../../gpu/src/reserve_policy.rs"]
mod reserve_policy;
/// A funded reserve under host pressure, driven by the serving loop's memory hooks. Each
/// session construction counts a pressure poll there that would release funded spares if
/// the idle period had lasted IDLE_RELEASE by then.
struct ReserveProbe { policy: Mutex<reserve_policy::ReservePolicy>, idle_passes: Mutex<usize>, idle_pass: Condvar, idle_releases: AtomicUsize, checks: AtomicUsize }
impl ReserveProbe {
    /// Wait, at most 10 s, until the scheduler counts more than `seen` idle passes.
    fn idle_pass_after(&self, seen: usize) {
        let (_passes, wait) = self.idle_pass.wait_timeout_while(self.idle_passes.lock().unwrap(), Duration::from_secs(10), |n| *n <= seen).unwrap();
        assert!(!wait.timed_out(), "no idle pass within 10 s");
    }
    fn check(&self) {
        let policy = self.policy.lock().unwrap();
        assert!(policy.pressure && policy.funded(0));
        self.checks.fetch_add(1, Ordering::Relaxed);
        if policy.sheds(false, 0, policy.idle_long(Instant::now() + reserve_policy::IDLE_RELEASE)) {
            self.idle_releases.fetch_add(1, Ordering::Relaxed);
        }
    }
}
struct Reserved(Model, Arc<ReserveProbe>);
impl Engine for Reserved {
    type Session<'a> = Kv;
    fn session(&self) -> Result<Kv, String> { self.1.check(); self.0.session() }
    fn maintain_memory(&self, idle: bool) -> Result<(), String> {
        if idle { *self.1.idle_passes.lock().unwrap() += 1; self.1.idle_pass.notify_all(); }
        self.1.policy.lock().unwrap().idle(idle, Instant::now());
        Ok(())
    }
    fn admission_started(&self) { self.1.policy.lock().unwrap().idle(false, Instant::now()); }
    fn context_limit(&self) -> usize { self.0.context_limit() }
    fn vocab_size(&self) -> usize { 8 }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { self.0.session_reservation_bytes() }
}
/// Host memory pressure leaves no physical room; `.1` bytes are the startup cache room.
struct Pressured(Model, usize);
impl Engine for Pressured {
    type Session<'a> = Kv;
    fn session(&self) -> Result<Kv, String> { self.0.session() }
    fn prefix_cache_opt_in(&self) -> bool { true }
    fn context_limit(&self) -> usize { self.0.context_limit() }
    fn vocab_size(&self) -> usize { 8 }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { self.0.session_reservation_bytes() }
    fn cache_free_physical_bytes(&self, _: usize) -> Result<Option<(i128, usize)>, String> { Ok(Some((-1, 1 << 20))) }
    fn cache_room_bytes(&self) -> usize { self.1 }
}
struct Codec(usize, bool);
impl Decoder for Codec {
    fn push(&mut self, _: u32) -> Result<String, String> {
        Ok("x".repeat(self.0))
    }
    fn finish(&mut self) -> Result<String, String> {
        Ok(String::new())
    }
}
impl Tokenizer for Codec {
    type Decoder<'a> = Codec;
    fn encode(&self, _: &str) -> Result<Vec<u32>, String> {
        Ok(vec![1])
    }
    fn decoder(&self) -> Codec {
        Codec(self.0, self.1)
    }
    fn gemma_chat(&self) -> bool { self.1 }
    fn has_token(&self, id: u32) -> bool {
        id < 8
    }
    fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> {
        Ok(vec![1])
    }
}
struct Server {
    ab: Option<Arc<Mutex<Vec<bool>>>>,
    cohort: Option<Arc<Mutex<(Option<bool>, usize)>>>,
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    live: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    heads: Arc<AtomicUsize>,
    order: Arc<Mutex<Vec<&'static str>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl Server {
    fn start() -> Self {
        Self::with_token_bytes(1)
    }
    fn with_token_bytes(bytes: usize) -> Self {
        Self::with_prefix_cache(bytes, false)
    }
    fn with_prefix_cache(bytes: usize, prefix_cache: bool) -> Self {
        Self::with_mtp_depth(bytes, prefix_cache, 0)
    }
    fn with_mtp_depth(bytes: usize, prefix_cache: bool, mtp_depth: usize) -> Self {
        Self::with_limits(Codec(bytes, false), prefix_cache, mtp_depth, Limits { output_bytes: 16 * 1024 * 1024, ..Limits::default() })
    }
    fn with_limits<T: Tokenizer + Send + 'static>(codec: T, prefix_cache: bool, mtp_depth: usize, limits: Limits) -> Self {
        Self::with_model(codec, prefix_cache, mtp_depth, limits, false)
    }
    fn with_opt_in() -> Self {
        Self::with_model(Codec(1, false), true, 0, Limits { output_bytes: 16 * 1024 * 1024, ..Limits::default() }, true)
    }
    fn with_model<T: Tokenizer + Send + 'static>(codec: T, prefix_cache: bool, mtp_depth: usize, limits: Limits, opt_in: bool) -> Self {
        Self::with_retention_bytes(codec, prefix_cache, mtp_depth, limits, opt_in, 0)
    }
    fn with_retention_bytes<T: Tokenizer + Send + 'static>(codec: T, prefix_cache: bool, mtp_depth: usize, limits: Limits, opt_in: bool, retention_extra: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let live = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let heads = Arc::new(AtomicUsize::new(0));
        let order: Arc<Mutex<Vec<&'static str>>> = Default::default();
        let ab = (limits.ab_ptx || retention_extra > 0).then(|| Arc::new(Mutex::new(Vec::new())));
        let cohort = limits.ab_cohorts.then(|| Arc::new(Mutex::new((None, 0))));
        let model = Model {
            ab: ab.clone(),
            cohort: cohort.clone(),
            live: live.clone(),
            calls: calls.clone(),
            heads: heads.clone(),
            order: order.clone(),
            opt_in,
            retention_extra,
        };
        let shutdown = stop.clone();
        let worker = std::thread::spawn(move || {
            scheduler::serve(
                listener,
                &model,
                Some(&codec),
                &limits,
                &CacheLimits {
                    entries: 32,
                    retained_bytes: 1 << 20,
                    tokens_per_entry: 1024,
                    logits_per_entry: 8,
                    idle_ttl: Duration::from_secs(60),
                },
                &scheduler::Config {
                    active: 3,
                    connections: 16,
                    prefill_chunk: 1,
                    foreground_rows: 8,
                    background_rows: 16,
                    active_bytes: 32768,
                    session_budget_bytes: 2 << 20,
                    draft: spark_serve::protocol::DraftMode::Strict, mtp_depth, draft_global_bank: false,
                    prefix_cache, sharing: scheduler::Sharing { report_prefix_reuse: true, ..Default::default() },
                },
                &shutdown,
            )
            .unwrap();
        });
        Self {
            ab,
            cohort,
            address,
            stop,
            live,
            calls,
            heads,
            order,
            worker: Some(worker),
        }
    }
    fn tiled(requests: usize) -> Self {
        Self::tiled_observed(requests, None)
    }
    fn tiled_observed(requests: usize, observation: Option<(PrefillTrace, Arc<AtomicBool>)>) -> Self {
        Self::tiled_burst(requests, observation, Limits::default().intake_burst, LastTurn::default())
    }
    fn tiled_burst(requests: usize, observation: Option<(PrefillTrace, Arc<AtomicBool>)>, intake_burst: Duration, turn: LastTurn) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (live, calls, heads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let model = Tiled(Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(), heads: heads.clone(), order: Default::default(), opt_in: false, retention_extra: 0 }, observation.as_ref().map(|o| o.0.clone()), turn);
        let shutdown = stop.clone();
        let worker = std::thread::spawn(move || {
            if let Some((_, start)) = observation {
                while !start.load(Ordering::Acquire) { std::thread::sleep(Duration::from_millis(1)); }
            }
            scheduler::serve_requests(listener, &model, Some(&Codec(1, false)),
            &Limits { input_tokens: 8192, context_tokens: 8192, generation_timeout: Duration::from_secs(2), intake_burst, ..Limits::default() },
            &CacheLimits { entries: 32, retained_bytes: 1 << 20, tokens_per_entry: 8192,
                logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
            &scheduler::Config { active: 8, connections: 16, prefill_chunk: 256,
                foreground_rows: 8, background_rows: 16, active_bytes: 8 * 65536,
                session_budget_bytes: 2 << 20, draft: protocol::DraftMode::Strict, mtp_depth: 0,
                draft_global_bank: false, prefix_cache: true,
                sharing: scheduler::Sharing { report_prefix_reuse: true, client_priority_cap: 1, ..Default::default() } },
            &shutdown, "tiled", requests).unwrap()
        });
        Self { ab: None, cohort: None, address, stop, live, calls, heads, order: Default::default(), worker: Some(worker) }
    }
    /// Production sets the active and session budgets equal (next_batched_service.rs): both are
    /// `admission`, the pool less the `room`. `retained`: the retained-bytes ceiling.
    fn pressured(room: usize, admission: usize, retained: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (live, calls, heads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let model = Pressured(Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(), heads: heads.clone(), order: Default::default(), opt_in: true, retention_extra: 0 }, room);
        let shutdown = stop.clone();
        let worker = std::thread::spawn(move || {
            scheduler::serve(listener, &model, Some(&Codec(1, false)),
            &Limits { output_bytes: 16 * 1024 * 1024, ..Limits::default() },
            &CacheLimits { entries: 32, retained_bytes: retained, tokens_per_entry: 1024,
                logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
            &scheduler::Config { active: 3, connections: 16, prefill_chunk: 1,
                foreground_rows: 8, background_rows: 16, active_bytes: admission,
                session_budget_bytes: admission, draft: protocol::DraftMode::Strict, mtp_depth: 0,
                draft_global_bank: false, prefix_cache: true,
                sharing: scheduler::Sharing { report_prefix_reuse: true, ..Default::default() } },
            &shutdown).unwrap()
        });
        Self { ab: None, cohort: None, address, stop, live, calls, heads, order: Default::default(), worker: Some(worker) }
    }
    fn reserved(requests: usize, probe: Arc<ReserveProbe>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (live, calls, heads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let model = Reserved(Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(), heads: heads.clone(), order: Default::default(), opt_in: false, retention_extra: 0 }, probe);
        let shutdown = stop.clone();
        let worker = std::thread::spawn(move || {
            scheduler::serve_requests(listener, &model, Some(&Codec(1, false)),
            &Limits { output_bytes: 16 * 1024 * 1024, ..Limits::default() },
            &CacheLimits { entries: 32, retained_bytes: 1 << 20, tokens_per_entry: 1024,
                logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
            &scheduler::Config { active: 3, connections: 16, prefill_chunk: 1,
                foreground_rows: 8, background_rows: 16, active_bytes: 32768,
                session_budget_bytes: 2 << 20, draft: protocol::DraftMode::Strict, mtp_depth: 0,
                draft_global_bank: false, prefix_cache: true,
                sharing: scheduler::Sharing { report_prefix_reuse: true, ..Default::default() } },
            &shutdown, "reserved", requests).unwrap()
        });
        Self { ab: None, cohort: None, address, stop, live, calls, heads, order: Default::default(), worker: Some(worker) }
    }
    fn send(&self, tokens: &[u32], max: usize, extra: &str) -> TcpStream {
        let body = format!(
            "{{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":{tokens:?},\"max_tokens\":{max}{extra}}}"
        );
        self.send_body(&body)
    }
    fn send_body(&self, body: &str) -> TcpStream {
        self.send_authorized(body, "")
    }
    fn send_authorized(&self, body: &str, key: &str) -> TcpStream {
        self.send_route("POST", "/v1/completions", body, key)
    }
    fn send_route(&self, method: &str, path: &str, body: &str, key: &str) -> TcpStream {
        let mut socket = TcpStream::connect(self.address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write!(
            socket,
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            self.address,
            body.len()
        )
        .unwrap();
        socket
    }
    fn run(&self, tokens: &[u32], extra: &str) -> String {
        let mut output = String::new();
        self.send(tokens, 3, extra)
            .read_to_string(&mut output)
            .unwrap();
        assert!(output.contains("data: [DONE]"), "{output}");
        assert!(!output.contains("generation_error"), "{output}");
        output
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
        assert_eq!(self.live.load(Ordering::Relaxed), 0);
    }
}
fn header<'a>(response: &'a str, name: &str) -> &'a str {
    response
        .lines()
        .find_map(|line| line.strip_prefix(name))
        .unwrap()
        .trim()
}
fn tokens(response: &str) -> Vec<u32> {
    response
        .split("\"token_id\":")
        .skip(1)
        .map(|s| {
            s.split(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect()
}

#[test]
fn oversized_retention_refuses_before_allocation_and_does_not_block_followers() {
    for extra in [16385, usize::MAX] {
        let server = Server::with_retention_bytes(Codec(1, false), true, 0,
            Limits { generation_timeout: Duration::from_secs(30), ..Limits::default() }, true, extra);
        let mut response = String::new();
        let mut socket = server.send(&[1, 2, 3], 1, ",\"cache\":true,\"prompt_lookup\":false");
        socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        socket.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503 ") && response.contains("capacity_exhausted"), "{response}");
        assert!(server.ab.as_ref().unwrap().lock().unwrap().is_empty(), "refused request allocated a session");
        let mut active = server.send(&[1, 2, 4], 32, ",\"cache\":false,\"ignore_eos\":true,\"prompt_lookup\":false");
        let wait = Instant::now();
        while server.live.load(Ordering::Relaxed) == 0 {
            assert!(wait.elapsed() < Duration::from_secs(2));
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut blocked = server.send(&[1, 2, 3], 1, ",\"cache\":true,\"prompt_lookup\":false");
        response.clear(); active.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 ") && tokens(&response).len() == 32, "{response}");
        response.clear(); blocked.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503 ") && response.contains("capacity_exhausted"), "{response}");
        assert_eq!(server.ab.as_ref().unwrap().lock().unwrap().len(), 1);
        let response = server.run(&[1, 2, 4], ",\"cache\":false,\"prompt_lookup\":false");
        assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
        assert_eq!(server.ab.as_ref().unwrap().lock().unwrap().len(), 2);
    }
}

#[test]
fn typed_prefill_cohort_requires_prompt_peers() {
    let run = |lengths: &[usize]| {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let start = Arc::new(AtomicBool::new(false));
        let server = Server::tiled_observed(lengths.len(), Some((trace.clone(), start.clone())));
        // Queue complete requests before starting intake so cohort assertions
        // do not depend on client thread scheduling or a timing sleep.
        let sockets: Vec<_> = lengths.iter().map(|&n| server.send(&vec![1; n], 3,
            ",\"cache\":false,\"ignore_eos\":true,\"top_k\":1,\"top_p\":1,\"seed\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0")).collect();
        start.store(true, Ordering::Release);
        let output: Vec<_> = sockets.into_iter().map(|mut socket| {
            let mut reply = String::new();
            socket.read_to_string(&mut reply).unwrap();
            assert!(reply.ends_with("data: [DONE]\n\n") && !reply.contains("generation_error"), "{reply}");
            let ids = tokens(&reply);
            assert_eq!(ids.len(), 3);
            ids
        }).collect();
        let events = trace.lock().unwrap().clone();
        (output, events)
    };
    let (long, solo) = run(&[2048]);
    assert_eq!(solo, [vec![0, 2048]], "lone long prompt must enter the long path");
    let (pair, cohort) = run(&[2048, 2048]);
    assert_eq!(cohort.first(), Some(&vec![256, 256]), "{cohort:?}");
    assert!(cohort.iter().all(|rows| rows[0] != 0), "{cohort:?}");
    assert_eq!(pair, [long[0].clone(), long[0].clone()]);
    let (short, solo) = run(&[32]);
    assert_eq!(solo, [vec![32]], "short prompts remain eligible alone");
    for lengths in [[32, 2048], [2048, 32], [256, 2048], [2048, 256]] {
        let (mixed, cohort) = run(&lengths);
        assert!(cohort[0].contains(&lengths[0].min(256)) && cohort[0].contains(&lengths[1].min(256)), "{cohort:?}");
        for (ids, length) in mixed.iter().zip(lengths) {
            let expected = if length == 32 { short[0].clone() } else { run(&[length]).0[0].clone() };
            assert_eq!(ids, &expected);
        }
    }
}

#[test]
fn burst_intake_late_request_bytes_join_the_first_prompt_pass() {
    // The split pattern of the two-Spark cards: every connection of a burst is
    // accepted at once, but some request bytes come after the first prompt
    // pass could start (main starts a shared pass at a 2 ms prompt age, and a
    // lone prompt above the shared prefill limit on the serial path at once).
    // Probes order the steps: the late bytes go out only after the loop
    // admitted the early chats (their sessions are live) and, under the hold,
    // after a loop turn that began 3 ms later, with no prompt pass yet.
    assert!(std::env::var_os("VADE_INTAKE_BURST_MS").is_none(), "unset VADE_INTAKE_BURST_MS for this test");
    let greedy = ",\"cache\":false,\"ignore_eos\":true,\"top_k\":1,\"top_p\":1,\"seed\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0";
    let until = |done: &dyn Fn() -> bool| {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() { assert!(Instant::now() < deadline, "probe timed out"); std::thread::sleep(Duration::from_micros(100)); }
    };
    // Returns the token IDs, the first prompt pass and whether the run can
    // judge the hold: a host stall past the window ends the hold by design.
    let burst = |window: Duration, prompt_tokens: usize, early: usize| {
        let body = format!("{{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":{:?},\"max_tokens\":3{greedy}}}", vec![1; prompt_tokens]);
        let (trace, turn) = (Arc::new(Mutex::new(Vec::new())), LastTurn::default());
        let server = Server::tiled_burst(4, Some((trace.clone(), Arc::new(AtomicBool::new(true)))), window, turn.clone());
        let request = format!("POST /v1/completions HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            server.address, body.len());
        let connected = Instant::now();
        let mut sockets: Vec<_> = (0..4).map(|_| TcpStream::connect(server.address).unwrap()).collect();
        for socket in &mut sockets[..early] { socket.write_all(request.as_bytes()).unwrap(); }
        until(&|| server.live.load(Ordering::Relaxed) == early);
        let admitted = Instant::now();
        if window.is_zero() {
            // No hold: the first pass starts without the late chats.
            until(&|| !trace.lock().unwrap().is_empty());
        } else {
            until(&|| turn.lock().unwrap().is_some_and(|at| at >= admitted + Duration::from_millis(3)));
        }
        let held = trace.lock().unwrap().is_empty();
        // Late bytes sent in the first half of the window reach the loop inside it.
        let judged = window.is_zero() || connected.elapsed() < window / 2;
        assert!(held || window.is_zero() || !judged, "a prompt pass started before the late bytes");
        for socket in &mut sockets[early..] { socket.write_all(request.as_bytes()).unwrap(); }
        let ids: Vec<_> = sockets.into_iter().map(|mut socket| {
            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut reply = String::new();
            socket.read_to_string(&mut reply).unwrap();
            assert!(reply.ends_with("data: [DONE]\n\n") && !reply.contains("generation_error"), "{reply}");
            tokens(&reply)
        }).collect();
        let first = trace.lock().unwrap()[0].clone();
        (ids, first, judged)
    };
    // Two early short chats share a pass; one early 600-token chat alone is
    // above the shared prefill limit (512) and would run on the serial path.
    for (prompt_tokens, early, pass) in [(32, 2, [32; 4]), (600, 1, [256; 4])] {
        let (main_ids, main_first, _) = burst(Duration::ZERO, prompt_tokens, early);
        assert!(main_first.len() < 4, "without the hold the late chats miss the first pass: {main_first:?}");
        // Three attempts: a run that stalled past half the 100 ms window cannot judge the hold.
        let (ids, first, _) = (0..3).map(|_| burst(Duration::from_millis(100), prompt_tokens, early)).find(|run| run.2)
            .expect("every run stalled past the window");
        assert_eq!(first, pass, "the four chats share one first prompt pass");
        // Batch membership does not change token IDs.
        assert!(ids.iter().chain(&main_ids).all(|i| *i == main_ids[0]), "{ids:?} {main_ids:?}");
    }
}

#[test]
fn typed_prefill_progress_grid() {
    let server = Server::tiled(2 + 7 * (1 + 2 * (1 + 2 + 8) + 2));
    let read = |mut socket: TcpStream| {
        let mut reply = String::new();
        socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        socket.read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n") && !reply.contains("generation_error"), "{reply}");
        assert_eq!(tokens(&reply).len(), 3, "{reply}");
        reply
    };
    let greedy = ",\"top_k\":1,\"top_p\":1,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,\"seed\":0";
    let cold = format!("{greedy},\"cache\":false");
    // The original stall: two background chats reach a 384-token remainder.
    let stalled: Vec<_> = (0..2).map(|_| server.send(&vec![1; 1152], 3, &cold)).collect();
    for socket in stalled { read(socket); }
    for length in [1, 256, 257, 512, 513, 1152, 4096] {
        let prompt = vec![1; length];
        let solo = tokens(&server.run(&prompt, &cold));
        for priority in [0, 1] { for chats in [1, 2, 8] {
            let sockets: Vec<_> = (0..chats).map(|i| server.send(&prompt, 3,
                &format!("{cold},\"priority\":{}", if i == 0 { priority } else { 0 }))).collect();
            for socket in sockets { assert_eq!(tokens(&read(socket)), solo, "length={length}, chats={chats}, priority={priority}"); }
        } }
        // Default caching stays off through 512, but longer prompts reuse a snapshot.
        for repeat in 0..2 {
            let reply = server.run(&prompt, greedy);
            assert_eq!(tokens(&reply), solo);
            if length <= 512 || repeat == 1 {
                assert_eq!(header(&reply, "X-Prefix-Reused-Tokens:"), if length <= 512 { "0".into() } else { length.to_string() });
            }
        }
    }
}

#[test]
fn forced_tool_deadline_returns_deadline_error() {
    // Silent Gemma output reaches the deadline without completing the forced call.
    let server = Server::with_limits(Codec(0, true), false, 0, Limits { generation_timeout: Duration::from_millis(50), ..Limits::default() });
    for stream in [false, true] {
        let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"f","parameters":{{"type":"object"}}}}}}],"tool_choice":"required","stream":{stream},"max_tokens":64}}"#);
        let mut reply = String::new();
        server.send_route("POST", "/v1/chat/completions", &body, "").read_to_string(&mut reply).unwrap();
        assert!(reply.starts_with(if stream { "HTTP/1.1 200" } else { "HTTP/1.1 504" }), "{reply}");
        assert_eq!(reply.matches("\"type\":\"generation_deadline\"").count(), 1, "{reply}");
        if !stream { assert_eq!(header(&reply, "x-should-retry:"), "false"); }
        assert!(!reply.contains("\"finish_reason\":\"length\"") && !reply.contains("invalid_model_output"), "{reply}");
        if stream { assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}"); }
    }
}

#[test]
fn next_mtp_depth_is_validated_on_every_http_endpoint() {
    for resident in [0, 3] {
        let server = Server::with_limits(
            Codec(1, false),
            false,
            2,
            Limits {
                next_mtp_depth: Some(resident),
                ..Limits::default()
            },
        );
        for (path, input) in [
            ("/v1/completions", r#""prompt_token_ids":[1]"#),
            (
                "/v1/chat/completions",
                r#""messages":[{"role":"user","content":"hi"}]"#,
            ),
            ("/v1/responses", r#""input":"hi""#),
            (
                "/v1/messages",
                r#""messages":[{"role":"user","content":"hi"}]"#,
            ),
        ] {
            for raw in ["0", "1", "2", "3", "4", "-1", "1.5", "true", "\"2\""] {
                let body = format!(
                    r#"{{{input},"max_tokens":1,"max_output_tokens":1,"cache":false,"temperature":0,"top_k":1,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":42,"mtp_depth":{raw}}}"#
                );
                let before = server.calls.load(Ordering::Relaxed);
                let mut reply = String::new();
                server
                    .send_route("POST", path, &body, "")
                    .read_to_string(&mut reply)
                    .unwrap();
                let valid = raw.parse::<usize>().is_ok_and(|k| k <= resident);
                assert!(
                    reply.starts_with(if valid {
                        "HTTP/1.1 200"
                    } else {
                        "HTTP/1.1 400"
                    }),
                    "{path} {raw}: {reply}"
                );
                if !valid {
                    assert!(reply.contains("mtp_depth"), "{reply}");
                    assert_eq!(server.calls.load(Ordering::Relaxed), before);
                }
            }
        }
    }
}
#[test]
fn other_engines_ignore_next_mtp_option_on_every_http_api() {
    let server = Server::with_mtp_depth(1, false, 2);
    for (path, input) in [
        ("/v1/completions", r#""prompt_token_ids":[1]"#),
        (
            "/v1/chat/completions",
            r#""messages":[{"role":"user","content":"hi"}]"#,
        ),
        ("/v1/responses", r#""input":"hi""#),
        (
            "/v1/messages",
            r#""messages":[{"role":"user","content":"hi"}]"#,
        ),
    ] {
        let mut expected = None;
        for raw in [
            None,
            Some("0"),
            Some("1"),
            Some("2"),
            Some("3"),
            Some("4"),
            Some("-1"),
            Some("1.5"),
            Some("true"),
            Some("\"2\""),
        ] {
            let extra = raw.map_or(String::new(), |v| format!(",\"mtp_depth\":{v}"));
            let body = format!(
                r#"{{{input},"max_tokens":1,"max_output_tokens":1,"cache":false,"temperature":0,"top_k":1,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":42{extra}}}"#
            );
            let mut reply = String::new();
            server
                .send_route("POST", path, &body, "")
                .read_to_string(&mut reply)
                .unwrap();
            assert!(reply.starts_with("HTTP/1.1 200"), "{path} {raw:?}: {reply}");
            let value = engine_model::json::parse(
                reply.split_once("\r\n\r\n").unwrap().1.as_bytes(),
                Default::default(),
            )
            .unwrap();
            let vade = value
                .object()
                .unwrap()
                .get("vade")
                .unwrap()
                .object()
                .unwrap();
            let policy = [
                "sampling",
                "draft",
                "mtp_depth",
                "draft_used",
                "model_draft_used",
            ]
            .map(|key| vade.get(key).unwrap().clone());
            assert_eq!(
                expected.get_or_insert_with(|| policy.clone()),
                &policy,
                "{path} {raw:?}"
            );
            println!("FP8_POLICY {path} {raw:?} {policy:?}");
        }
    }
}

#[test]
fn events_route_and_health_report_the_best_effort_counts() {
    let server = Server::with_limits(Codec(1, false), false, 0, Limits::default());
    engine_model::events::note("events_route_test", format_args!("tokens={}", 12));
    let read = |path: &str| { let mut reply = String::new(); server.send_route("GET", path, "", "").read_to_string(&mut reply).unwrap(); reply };
    let reply = read("/v1/events");
    assert!(reply.starts_with("HTTP/1.1 200 "), "{reply}");
    let value = engine_model::json::parse(reply.split_once("\r\n\r\n").unwrap().1.as_bytes(), Default::default()).unwrap();
    let object = value.object().unwrap();
    assert!(object["counts"].object().unwrap()["events_route_test"].u64().unwrap() >= 1, "{reply}");
    // Other tests in this binary share the ring: the marker is there unless 64 newer lines pushed it out.
    let recent = object["recent"].array().unwrap();
    assert!(recent.iter().all(|e| { let e = e.object().unwrap(); e["unix_ms"].u64().is_ok() && e["name"].string().is_ok() && e["line"].string().is_ok() }), "{reply}");
    assert!(recent.len() == engine_model::events::RECENT || recent.iter().any(|e| e.object().unwrap()["line"].string().unwrap() == "events_route_test tokens=12"), "{reply}");
    let reply = read("/health");
    let value = engine_model::json::parse(reply.split_once("\r\n\r\n").unwrap().1.as_bytes(), Default::default()).unwrap();
    assert!(value.object().unwrap()["events"].object().unwrap()["events_route_test"].u64().unwrap() >= 1, "{reply}");
    let mut reply = String::new();
    server.send_route("POST", "/v1/events", "{}", "").read_to_string(&mut reply).unwrap();
    assert!(!reply.starts_with("HTTP/1.1 200 "), "events is GET only: {reply}");
}

#[test]
fn sampling_echo_on_scheduler_responses() {
    let root = std::env::temp_dir().join(format!("vade-echo-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap(); std::fs::write(root.join("model.safetensors"), b"a").unwrap();
    let metadata = spark_serve::metadata::Metadata::prepare(&root, c"ptx", 3, 1, "local", std::time::UNIX_EPOCH).unwrap();
    let metadata_fields = &metadata.json()[1..metadata.json().len() - 1];
    let server = Server::with_limits(Codec(1, false), false, 3, Limits { output_bytes: 16 * 1024 * 1024, metadata: Some(metadata.clone()), ..Limits::default() });
    let sampling = r#""temperature":0.7,"top_k":3,"top_p":0.8,"repetition_penalty":1.2,"presence_penalty":0.4,"frequency_penalty":0.1,"seed":912"#;
    for chat in [false, true] { for stream in [false, true] {
        let (path, input) = if chat { ("/v1/chat/completions", r#""messages":[{"role":"user","content":"hi"}],"enable_thinking":false"#) } else { ("/v1/completions", r#""prompt_token_ids":[1,2,3]"#) };
        let body = format!(r#"{{{input},{sampling},"stream":{stream},"draft":"auto","max_tokens":3}}"#);
        let mut reply = String::new(); server.send_route("POST", path, &body, "").read_to_string(&mut reply).unwrap();
        assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
        assert_eq!(reply.matches("\"vade\":").count(), 1, "{reply}");
        let data = if stream { reply.lines().find_map(|line| line.strip_prefix("data: ").filter(|s| s.contains("\"vade\":"))).unwrap() } else { reply.split_once("\r\n\r\n").unwrap().1 };
        let value = engine_model::json::parse(data.as_bytes(), Default::default()).unwrap();
        let vade = value.object().unwrap().get("vade").unwrap().object().unwrap();
        let rows = if chat { "[1]" } else { "[1,1,1]" };
        let number = |key| match vade.get(key).unwrap() { engine_model::json::Value::Number(n) => n.parse::<f64>().unwrap(), v => panic!("{key}: {v:?}") };
        assert!(number("prefill_ms") > 0.0 && number("admission_to_first_logits_ms") >= number("prefill_ms"), "{reply}");
        let count = if chat { 1 } else { 3 };
        let effort = if chat { "\"reasoning_effort\":\"none\"," } else { "" };
        let mut expected = engine_model::json::parse(format!(r#"{{{effort}"sampling":{{{sampling}}},"draft":"plain","mtp_depth":0,"thinking":false,"draft_used":false,"model_draft_used":false,"prefill_ms":{:.3},"prefill_prompt_tokens":{count},"prefill_chunks":{count},"prefill_rows_per_chunk":{rows},"prefill_rows_per_chunk_omitted":0,"prefill_cache_restored_tokens":0,"prefill_cache_reuse_pct":0.0,"prefill_cache_restore_ms":null,"admission_to_first_logits_ms":{:.3},"prefill_estimated_ms":{:.1},"prefill_cold_estimated_ms":{:.1},{metadata_fields}}}"#, number("prefill_ms"), number("admission_to_first_logits_ms"), number("prefill_estimated_ms"), number("prefill_cold_estimated_ms")).as_bytes(), Default::default()).unwrap();
        let engine_model::json::Value::Object(fields) = &mut expected else { unreachable!() };
        let engine_model::json::Value::Object(numerics) = fields.get_mut("numerics").unwrap() else { unreachable!() };
        numerics.insert("draft".into(), engine_model::json::Value::String("plain".into()));
        numerics.insert("mtp_depth".into(), engine_model::json::Value::Number("0".into()));
        // Event counts are process-wide (other tests in this binary add to them): a name -> count object.
        let events = vade.get("events").unwrap();
        assert!(events.object().unwrap().values().all(|v| matches!(v, engine_model::json::Value::Number(_))), "{reply}");
        fields.insert("events".into(), events.clone());
        assert_eq!(value.object().unwrap().get("vade"), Some(&expected), "{reply}");
    } }
    drop(server); std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_second_turn_hits_under_host_pressure_with_a_startup_room() {
    // In a two-Spark soak with no physical room, every snapshot was skipped (no_room). The
    // startup room keeps the first turn, so the second turn reuses it; /health counts both.
    for (room, reused, counts) in [(0, "0", r#""prefix_cache":{"hits":0,"misses":2,"reused_tokens":0,"snapshots":0,"#),
        (1 << 20, "3", r#""prefix_cache":{"hits":1,"misses":1,"reused_tokens":3,"snapshots":1,"#)] {
        let server = Server::pressured(room, 2 << 20, 1 << 20);
        let send = || {
            let body = r#"{"stream":true,"temperature":0,"prompt_token_ids":[1,2,3],"max_tokens":1,"prompt_cache_key":"k"}"#;
            let mut reply = String::new();
            server.send_body(body).read_to_string(&mut reply).unwrap();
            assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
            header(&reply, "X-Prefix-Reused-Tokens:").to_owned()
        };
        assert_eq!(send(), "0");
        assert_eq!(send(), reused);
        let mut health = String::new();
        server.send_route("GET", "/health", "", "").read_to_string(&mut health).unwrap();
        assert!(health.contains(counts) && health.contains(&format!("\"room_bytes\":{room},")), "{health}");
        let skipped = if room == 0 { r#""skipped":{"no_room":2,"# } else { r#""skipped":{"no_room":0,"# };
        assert!(health.contains(skipped), "{health}");
    }
}

#[test]
fn a_room_entry_beside_admitted_requests_keeps_the_server_serving() {
    // The startup room is outside admission, so live requests may use every admission byte
    // beside a prefix the room keeps. Retained ceilings: the admission bytes, then the whole
    // pool (adoption). Each turn has its own prompt, so the first entry is unrelated.
    const CHAT: usize = 16384;
    for retained in [2 * CHAT, 4 * CHAT] {
        let server = Server::pressured(2 * CHAT, 2 * CHAT, retained);
        for (prompt, key) in [("1,2,3", "k"), ("4,5,6", "j")] {
            let body = format!(r#"{{"stream":true,"temperature":0,"prompt_token_ids":[{prompt}],"max_tokens":1,"prompt_cache_key":"{key}"}}"#);
            let mut reply = String::new();
            server.send_body(&body).read_to_string(&mut reply).unwrap();
            assert!(reply.ends_with("data: [DONE]\n\n"), "retained {retained}: {reply}");
            assert_eq!(header(&reply, "X-Prefix-Reused-Tokens:"), "0");
        }
        let mut health = String::new();
        server.send_route("GET", "/health", "", "").read_to_string(&mut health).unwrap();
        assert!(health.contains(r#""status":"ready""#) && health.contains(r#""prefix_cache":{"hits":0,"misses":2,"#)
            && health.contains(&format!("\"room_bytes\":{},", 2 * CHAT)), "{health}");
    }
}

#[test]
fn opt_in_prefix_cache_serves_only_requests_that_ask() {
    // Engine::prefix_cache_opt_in: a request without cache:true or a prompt_cache_key runs as
    // cache:false (main's no-cache path), even after an identical keyed request was retained.
    let server = Server::with_opt_in();
    let send = |extra: &str| {
        let body = format!(r#"{{"stream":true,"temperature":0,"prompt_token_ids":[1,2,3],"max_tokens":1{extra}}}"#);
        let mut reply = String::new();
        server.send_body(&body).read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
        header(&reply, "X-Prefix-Reused-Tokens:").to_owned()
    };
    assert_eq!(send(r#","prompt_cache_key":"k""#), "0");
    assert_eq!(send(""), "0");
    assert_eq!(send(""), "0");
    assert_eq!(send(r#","prompt_cache_key":"k""#), "3");
    assert_eq!(send(r#","cache":true"#), "3");
}

#[test]
fn automatic_prefix_cache_is_isolated_per_api_key() {
    let server = Server::with_prefix_cache(1, true);
    let keys = std::array::from_fn::<_, 2, _>(|_| spark_serve::cache::SessionKey::generate().unwrap().expose());
    let body = r#"{"stream":true,"temperature":0,"prompt_token_ids":[1,2,3],"max_tokens":1,"prompt_cache_key":"shared-hint"}"#;
    for (key, reused) in [(&keys[0], "0"), (&keys[1], "0"), (&keys[0], "3")] {
        let mut reply = String::new();
        server.send_authorized(body, key).read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
        assert_eq!(header(&reply, "X-Prefix-Reused-Tokens:"), reused);
        assert!(reply.contains(&format!("\"prompt_tokens_details\":{{\"cached_tokens\":{reused}}}")));
        assert!(reply.contains(&format!("\"prefill_cache_restored_tokens\":{reused}")) && (reused == "0" || reply.contains("\"prefill_ms\":0.000,\"prefill_prompt_tokens\":3,\"prefill_chunks\":0,\"prefill_rows_per_chunk\":[]")), "{reply}");
        let pct = if reused == "0" { "0.0" } else { "100.0" };
        assert!(reply.contains(&format!("\"prefill_cache_reuse_pct\":{pct},\"prefill_cache_restore_ms\":")), "{reply}");
        assert_eq!(reply.contains("\"prefill_cache_restore_ms\":null"), reused == "0", "{reply}");
    }
}

#[test]
fn stream_head_carries_the_warm_cache_estimate() {
    let server = Server::with_prefix_cache(1, true);
    let body = r#"{"stream":true,"temperature":0,"prompt_token_ids":[1,2,3],"max_tokens":1,"prompt_cache_key":"estimate"}"#;
    for cached in ["0", "3"] {
        let mut reply = String::new();
        server.send_body(body).read_to_string(&mut reply).unwrap();
        let (head, events) = reply.split_once("\r\n\r\n").unwrap();
        assert_eq!((header(head, "X-Vade-Prompt-Tokens:"), header(head, "X-Vade-Cached-Tokens:")), ("3", cached), "{head}");
        let warm: f64 = header(head, "X-Vade-Estimated-First-Token-Ms:").parse().unwrap();
        let cold: f64 = header(head, "X-Vade-Estimated-Cold-Ms:").parse().unwrap();
        assert!(warm >= 0.0 && cold > 0.0 && (cached == "3" || warm >= cold - 1.0), "{head}");
        assert!(!head.contains("data:") && events.contains(",\"prefill_estimated_ms\":") && events.contains(",\"prefill_cold_estimated_ms\":"), "{reply}");
    }
}

/// A sequence split's prefix owes other ranks its state until settle_prefix runs (Flash-Next), so
/// the prompt's snapshot settles first, and the next turn restores it.
#[test]
fn a_prompt_settles_before_its_snapshot_forks() {
    let server = Server::with_opt_in();
    let send = || {
        let body = r#"{"stream":true,"temperature":0,"prompt_token_ids":[1,2,3],"max_tokens":1,"prompt_cache_key":"k"}"#;
        let mut reply = String::new();
        server.send_body(body).read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
        header(&reply, "X-Prefix-Reused-Tokens:").to_owned()
    };
    assert_eq!(send(), "0");
    assert_eq!(send(), "3");
    let order = server.order.lock().unwrap();
    assert_eq!(order.first(), Some(&"settle"), "{order:?}");
    assert!(order.contains(&"fork"), "{order:?}");
}

#[test]
fn branches_reuse_longest_prefix_and_match_fresh_without_mutating_parent() {
    let server = Server::start();
    let first = server.run(&[1, 2, 7, 3, 4, 5], ",\"cache\":true");
    let key = header(&first, "X-Session-Capability:");
    assert_eq!(server.heads.load(Ordering::Relaxed), 3); // one prefill head, two decode heads
    let branch = [1, 2, 7, 3, 6, 1];
    let extra = format!(",\"cache\":true,\"session\":\"{key}\",\"fork\":true");
    let second = server.run(&branch, &extra);
    assert_eq!(header(&second, "X-Prefix-Reused-Tokens:"), "4");
    assert_ne!(header(&second, "X-Session-Capability:"), key);
    assert_eq!(tokens(&second), tokens(&server.run(&branch, "")));
    let original = server.run(&[1, 2, 7, 3, 4, 5], &extra);
    assert_eq!(header(&original, "X-Prefix-Reused-Tokens:"), "6"); // prompt-end snapshot: nothing is replayed
    assert_eq!(tokens(&original), tokens(&first));
    let shorter = server.run(&[1, 2, 7], &extra);
    assert_eq!(tokens(&shorter), tokens(&server.run(&[1, 2, 7], "")));
    let mut left = server.send(&branch, 3, &extra);
    let right_prompt = [1, 2, 7, 3, 4, 0];
    let mut right = server.send(&right_prompt, 3, &extra);
    let mut left_wire = String::new();
    let mut right_wire = String::new();
    left.read_to_string(&mut left_wire).unwrap();
    right.read_to_string(&mut right_wire).unwrap();
    assert_eq!(tokens(&left_wire), tokens(&server.run(&branch, "")));
    assert_eq!(tokens(&right_wire), tokens(&server.run(&right_prompt, "")));
}

#[test]
fn sampled_branches_match_full_history_draws_and_sequential_generation() {
    use spark_engine::sample::{Sampler, Sampling};
    use spark_serve::generation::{self, Event};

    let server = Server::start();
    let parent = server.run(&[1, 2, 3], ",\"cache\":true,\"kv_checkpoint\":true");
    let key = header(&parent, "X-Session-Capability:");
    let prompt = [1, 2, 3, 5, 6];
    let model = Model {
        ab: None,
        cohort: None,
        live: Arc::new(AtomicUsize::new(0)),
        calls: Arc::new(AtomicUsize::new(0)),
        heads: Arc::new(AtomicUsize::new(0)),
        order: Default::default(),
        opt_in: false,
        retention_extra: 0,
    };
    for (temperature, top_k, top_p) in [(0.7, 0, 0.95), (1.3, 3, 0.8), (0.0, 0, 1.0)] {
        let sampling = Sampling {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            seed: 912,
            repetition_penalty: 1.2,
            presence_penalty: 0.4,
            frequency_penalty: 0.1,
        };
        // Full logical history is independent of the pending-token KV path.
        let mut history = prompt.to_vec();
        let mut sampler = Sampler::new(sampling).unwrap();
        let mut expected = Vec::new();
        for _ in 0..12 {
            let hash = history
                .iter()
                .fold(0usize, |h, &t| (h * 17 + t as usize + 1) % 7);
            let mut logits = vec![0.; 8];
            logits[hash] = 1.;
            let token = sampler
                .sample(&logits, &history, prompt.len(), history.len(), |_| true)
                .unwrap();
            if token == 7 {
                break;
            }
            history.push(token);
            expected.push(token);
        }
        assert!(expected.len() > 1, "exercise pending-token penalties");
        let mut sequential = Vec::new();
        generation::generate_sampled(
            &model,
            &prompt,
            12,
            &Limits::default(),
            &sampling,
            |_| true,
            || Ok(()),
            |event| {
                if let Event::Token(token) = event {
                    sequential.push(token);
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(sequential, expected);
        let fields = format!(
            "\"temperature\":{temperature},\"top_k\":{top_k},\"top_p\":{top_p},\"seed\":912,\"repetition_penalty\":1.2,\"presence_penalty\":0.4,\"frequency_penalty\":0.1"
        );
        for cached in [false, true] {
            let cache = if cached {
                format!(",\"cache\":true,\"session\":\"{key}\",\"fork\":true,\"kv_handle\":\"{}\"", parent.split("\"handle\":\"").nth(1).unwrap().split('"').next().unwrap())
            } else {
                String::new()
            };
            let body = format!(
                "{{\"stream\":true,\"priority\":1,\"prompt_token_ids\":{prompt:?},\"max_tokens\":12,{fields}{cache}}}"
            );
            let mut reply = String::new();
            server.send_body(&body).read_to_string(&mut reply).unwrap();
            assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
            assert!(!reply.contains("generation_error"), "{reply}");
            assert_eq!(header(&reply, "X-Sampling-Seed:"), "912");
            assert_eq!(tokens(&reply), expected);
            if cached {
                assert_eq!(header(&reply, "X-Prefix-Reused-Tokens:"), "3");
                assert_ne!(header(&reply, "X-Session-Capability:"), key);
            }
        }
    }
}

#[test]
fn stalled_high_priority_reader_does_not_block_other_chats() {
    let server = Server::with_token_bytes(128 * 1024);
    let slow = server.send(&[1], 512, ",\"priority\":255");
    // Let the OS socket buffer fill, then verify model work stops for this peer.
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut previous = 0;
    loop {
        std::thread::sleep(Duration::from_millis(30));
        let calls = server.calls.load(Ordering::Relaxed);
        if calls > 0 && calls == previous {
            break;
        }
        assert!(Instant::now() < deadline);
        previous = calls;
    }
    assert!(
        previous < 100,
        "unbounded generation while socket is stalled"
    );
    let main = server.run(&[2], ",\"priority\":1");
    assert_eq!(tokens(&main).len(), 3);
    drop(slow);
    for (path, input, max, end) in [("/v1/completions", r#""prompt_token_ids":[2]"#, "max_tokens", "data: [DONE]"), ("/v1/chat/completions", r#""messages":[{"role":"user","content":"hi"}]"#, "max_tokens", "data: [DONE]"), ("/v1/responses", r#""input":"hi""#, "max_output_tokens", "event: response.incomplete"), ("/v1/messages", r#""messages":[{"role":"user","content":"hi"}]"#, "max_tokens", "event: message_stop")] { let mut long = String::new(); server.send_route("POST", path, &format!(r#"{{{input},"{max}":32,"stream":true,"temperature":0}}"#), "").read_to_string(&mut long).unwrap(); let usage = if path == "/v1/completions" || path == "/v1/chat/completions" { "completion_tokens" } else { "output_tokens" }; assert!(long.starts_with("HTTP/1.1 200") && long.len() > 1024 * 1024 && long.ends_with("\n\n") && long.matches(end).count() == 1 && long.contains(&format!("\"{usage}\":32")) && !long.contains("generation_error"), "{path}: {} bytes, missing complete terminal response", long.len()); }
}

#[test]
fn half_closed_requests_receive_complete_responses_without_cache_metrics() {
    let server = Server::start();
    let mut client = server.send(&[1, 2, 3], 3, "");
    client.shutdown(std::net::Shutdown::Write).unwrap();
    let mut reply = String::new();
    client.read_to_string(&mut reply).unwrap();
    assert_eq!(tokens(&reply), tokens(&server.run(&[1, 2, 3], "")));
    assert!(reply.ends_with("data: [DONE]\n\n"));
    let mut health = TcpStream::connect(server.address).unwrap();
    health
        .write_all(format!("GET /health HTTP/1.1\r\nHost: {}\r\n\r\n", server.address).as_bytes())
        .unwrap();
    reply.clear();
    health.read_to_string(&mut reply).unwrap();
    assert!(reply.contains("\"status\":\"ready\""));
    assert!(!reply.contains("cached_bytes"));
}

#[test]
fn finished_unread_responses_release_all_model_sessions() {
    let server = Server::with_token_bytes(512 * 1024);
    let clients: Vec<_> = (0..3)
        .map(|_| server.send(&[1], 1, ",\"priority\":255"))
        .collect();
    let deadline = Instant::now() + Duration::from_secs(3);
    while server.calls.load(Ordering::Relaxed) < 3 || server.live.load(Ordering::Relaxed) != 0 {
        assert!(
            Instant::now() < deadline,
            "finished requests retained active KV"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        server
            .run(&[2], ",\"priority\":1")
            .ends_with("data: [DONE]\n\n")
    );
    for mut client in clients {
        let mut reply = String::new();
        client.read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n"));
    }
}

#[test]
fn engine_contract_controls_and_usage() {
    let server = Server::with_prefix_cache(1, true);
    let read = |method, path, body: &str, key| { let mut reply = String::new(); server.send_route(method, &format!("{path}?beta=true"), body, key).read_to_string(&mut reply).unwrap(); assert!(reply.starts_with("HTTP/1.1 200"), "{reply}"); reply };
    let caps = read("GET", "/v1/capabilities", "", "a");
    assert!(caps.contains(spark_serve::contract::VERSION) && caps.contains("\"context_limit\":1024") && caps.contains("/v1/messages/count_tokens"));
    assert_eq!(caps, read("GET", "/v1/capabilities", "", "a"));
    assert_ne!(caps, read("GET", "/v1/capabilities", "", "b"));
    let calls = server.calls.load(Ordering::Relaxed);
    assert!(read("POST", "/v1/messages/count_tokens", r#"{"messages":[{"role":"user","content":"hi"}]}"#, "a").contains("\"input_tokens\":1"));
    assert!(read("POST", "/tokenize", r#"{"prompt":"hi","add_special_tokens":false}"#, "a").ends_with(r#"{"count":1,"max_model_len":1024,"tokens":[1]}"#));
    assert_eq!(server.calls.load(Ordering::Relaxed), calls);
    let expected = tokens(&server.run(&[1], "")); for prompt in [r#""prompt":"hi""#, r#""prompt":[1]"#, r#""prompt_token_ids":[1]"#] { assert_eq!(tokens(&read("POST", "/v1/completions", &format!(r#"{{{prompt},"stream":true,"temperature":0,"max_tokens":3}}"#), "a")), expected); }
    for (endpoint, input, max) in [("/v1/chat/completions", r#""messages":[{"role":"user","content":"é"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]"#, "max_tokens"), ("/v1/responses", r#""input":"é","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]"#, "max_output_tokens"), ("/v1/messages", r#""messages":[{"role":"user","content":"é"}],"tools":[{"name":"f","input_schema":{"type":"object"}}]"#, "max_tokens")] {
        let calls = server.calls.load(Ordering::Relaxed);
        for output in [1, 1024] {
            let body = format!("{{\"endpoint\":{endpoint:?},\"request\":{{{input}}},\"output_tokens\":{output}}}");
            let reply = read("POST", "/v1/admission", &body, "a");
            assert!(reply.contains("\"prompt_tokens\":1") && reply.contains(&format!("\"admitted\":{}", output == 1)), "{reply}");
        }
        assert_eq!(server.calls.load(Ordering::Relaxed), calls);
        read("POST", endpoint, &format!("{{{input},\"{max}\":1}}"), "a");
        let salted = read("POST", endpoint, &format!(r#"{{{input},"{max}":1,"stream":true,"cache":true,"kv_checkpoint":true,"cache_salt":{endpoint:?}}}"#), "salt-owner"); let handle = salted.split("\"handle\":\"").nth(1).unwrap().split('"').next().unwrap(); let session = header(&salted, "X-Session-Capability:");
        for (salt, reused, reason) in [(format!("{endpoint}-other"), "0", "unavailable"), (endpoint.to_owned(), "1", "hit")] { let reply = read("POST", endpoint, &format!(r#"{{{input},"{max}":1,"stream":true,"cache":true,"cache_salt":{salt:?},"session":{session:?},"kv_handle":{handle:?}}}"#), "salt-owner"); assert_eq!(header(&reply, "X-Prefix-Reused-Tokens:"), reused); assert!(reply.contains(&format!(r#""reuse":"{reason}""#)), "{reply}"); }
        for salt in [r#""""#, "7"] { let mut reply = String::new(); server.send_route("POST", endpoint, &format!(r#"{{{input},"{max}":1,"cache_salt":{salt}}}"#), "a").read_to_string(&mut reply).unwrap(); assert!(reply.starts_with("HTTP/1.1 400") && reply.contains("cache_salt"), "{reply}"); }
        for ignore in [false, true] { for stream in [false, true] { let reply = read("POST", endpoint, &format!(r#"{{{input},"{max}":3,"stream":{stream},"temperature":1000,"top_k":0,"top_p":1,"seed":31,"ignore_eos":{ignore}}}"#), "a"); let key = if endpoint == "/v1/chat/completions" { "completion_tokens" } else { "output_tokens" }; assert!(reply.contains(&format!("\"{key}\":{}", if ignore { 3 } else { 1 })), "{reply}"); } }
        if endpoint == "/v1/messages" { assert!(read("POST", endpoint, &format!("{{{input},\"max_tokens\":1,\"stop_sequences\":[\"x\"]}}"), "a").contains("\"stop_reason\":\"stop_sequence\",\"stop_sequence\":\"x\"")); }
        for (stream, cache) in [(false, true), (true, true), (true, false), (false, false)] {
            let body = format!("{{{input},\"{max}\":1,\"stream\":{stream},\"cache\":{cache},\"kv_checkpoint\":true,\"kv_handle\":\"invalid\",\"include_usage_timing\":true,\"prompt_cache_key\":\"new-hint\"}}");
            let reply = read("POST", endpoint, &body, "a");
            let cached = if endpoint == "/v1/messages" { format!("\"cache_read_input_tokens\":{}", usize::from(cache)) } else { format!("\"prompt_tokens_details\":{{\"cached_tokens\":{}}}", usize::from(cache)) };
            assert!(reply.contains(&cached) && reply.contains("\"prefill_ms\":") && reply.contains("\"decode_ms\":"), "{reply}");
            if cache {
                let handle = reply.split("\"handle\":\"").nth(1).unwrap().split('"').next().unwrap();
                assert!(read("POST", endpoint, &format!("{{{input},\"{max}\":1,\"stream\":{stream},\"kv_handle\":\"{handle}\"}}"), "a").contains("\"reuse\":\"hit\""));
                read("POST", "/v1/kv/release", &format!("{{\"kv_handle\":\"{handle}\"}}"), "a");
            }
        }
    }
}

#[test]
fn image_refusals_are_coded_on_every_http_api() {
    let server = Server::start();
    for (path, input) in [
        ("/v1/chat/completions", r#""messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]"#),
        ("/v1/responses", r#""input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]"#),
        ("/v1/responses", r#""input":[{"type":"function_call_output","call_id":"prior","output":[{"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]"#),
        ("/v1/messages", r#""messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]"#),
        ("/v1/messages", r#""messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"prior","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]}]"#),
        ("/v1/messages/count_tokens", r#""messages":[{"role":"user","content":[{"type":"image"}]}]"#),
    ] {
        for stream in [false, true] {
            let body = format!(r#"{{{input},"max_tokens":1,"stream":{stream}}}"#);
            let mut reply = String::new();
            server.send_route("POST", &format!("{path}?beta=true"), &body, "").read_to_string(&mut reply).unwrap();
            assert!(reply.starts_with("HTTP/1.1 400"), "{reply}");
            assert!(reply.contains(r#""type":"invalid_request_error""#), "{reply}");
            assert!(reply.contains(r#""code":"image_input_unsupported""#), "{reply}");
            assert!(reply.contains("text-only engine"), "{reply}");
            assert!(!reply.contains("event:"), "{reply}");
        }
    }
    assert_eq!(server.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn repeated_http_requests_keep_tool_ids_across_admissions() {
    struct ToolCodec;
    impl Decoder for ToolCodec {
        fn push(&mut self, _: u32) -> Result<String, String> { Ok(r#"</think><tool_call>{"name":"f","arguments":{"x":1}}</tool_call><tool_call>{"name":"f","arguments":{"x":2}}</tool_call>"#.into()) }
        fn finish(&mut self) -> Result<String, String> { Ok(String::new()) }
    }
    impl Tokenizer for ToolCodec {
        type Decoder<'a> = Self;
        fn encode(&self, _: &str) -> Result<Vec<u32>, String> { Ok(vec![1]) }
        fn decoder(&self) -> Self { Self }
        fn has_token(&self, id: u32) -> bool { id < 8 }
        fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { Ok(vec![1]) }
    }
    let server = Server::with_limits(ToolCodec, false, 0, Limits::default());
    for (path, fields) in [
        ("/v1/chat/completions", r#""messages":[{"role":"user","content":"go"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]"#),
        ("/v1/responses", r#""input":"go","tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]"#),
        ("/v1/messages", r#""messages":[{"role":"user","content":"go"}],"tools":[{"name":"f","input_schema":{"type":"object"}}]"#),
    ] {
        for stream in [false, true] {
            let body = format!(r#"{{{fields},"max_tokens":1,"max_output_tokens":1,"stream":{stream},"temperature":0,"seed":42}}"#);
            let mut previous = None;
            for _ in 0..2 {
                let mut reply = String::new();
                server.send_route("POST", path, &body, "").read_to_string(&mut reply).unwrap();
                assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
                let ids: Vec<_> = reply.match_indices("call-").map(|(offset, _)| reply[offset..].split('"').next().unwrap().to_owned()).collect();
                let hash = body.bytes().fold(0xcbf29ce484222325u64, |h, byte| (h ^ u64::from(byte)).wrapping_mul(0x100000001b3));
                for index in 0..2 { assert!(ids.contains(&format!("call-{hash}-{index}")), "{reply}"); }
                if let Some(previous) = previous { assert_eq!(ids, previous, "{path}, stream={stream}"); }
                previous = Some(ids);
            }
        }
    }
}

/// Exercise the production intake and incremental JSON writer with three times
/// the connection ceiling and immediate replacement of every completed request.
#[test]
fn overload_reconnects_preserve_complete_503_and_200_bodies_for_60_seconds() {
    let server = Server::with_prefix_cache(4096, false);
    let end = Instant::now() + Duration::from_secs(60);
    let start = Arc::new(std::sync::Barrier::new(48));
    let address = server.address;
    let counts = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..48).map(|_| {
            let start = start.clone();
            scope.spawn(move || -> Result<[usize; 2], String> {
                start.wait();
                let mut counts = [0, 0];
                while Instant::now() < end {
                    let mut socket = TcpStream::connect_timeout(&address, Duration::from_secs(10)).map_err(|e| e.to_string())?;
                    socket.set_read_timeout(Some(Duration::from_secs(10))).map_err(|e| e.to_string())?;
                    socket.set_write_timeout(Some(Duration::from_secs(10))).map_err(|e| e.to_string())?;
                    let body = r#"{"prompt_token_ids":[1,2,3],"max_tokens":16,"stream":false,"cache":false,"ignore_eos":true,"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":0}"#;
                    write!(socket, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).map_err(|e| e.to_string())?;
                    let mut reply = String::new();
                    socket.read_to_string(&mut reply).map_err(|e| format!("incomplete response: {e}; {} bytes", reply.len()))?;
                    let (head, body) = reply.split_once("\r\n\r\n").ok_or("missing HTTP status/headers")?;
                    let length: usize = head.lines().find_map(|line| line.strip_prefix("Content-Length: "))
                        .ok_or("missing content length")?.parse::<usize>().map_err(|e| e.to_string())?;
                    if body.len() != length { return Err(format!("truncated body: {} of {length}", body.len())); }
                    let value = engine_model::json::parse(body.as_bytes(), Default::default()).map_err(|e| e.to_string())?;
                    let object = value.object().map_err(|e| e.to_string())?;
                    if head.starts_with("HTTP/1.1 200 ") {
                        if body.matches('x').count() < 16 * 4096 || !object.contains_key("choices") || !body.contains("\"completion_tokens\":16") {
                            return Err("incomplete admitted generation".into());
                        }
                        counts[0] += 1;
                    } else if head.starts_with("HTTP/1.1 503 ") && object.contains_key("error") {
                        counts[1] += 1;
                    } else { return Err(format!("unexpected response: {reply}")); }
                }
                Ok(counts)
            })
        }).collect();
        let mut total = [0, 0];
        for worker in workers {
            let counts = worker
                .join()
                .expect("load client panicked")
                .expect("HTTP integrity failure");
            for (total, n) in total.iter_mut().zip(counts) {
                *total += n;
            }
        }
        total
    });
    assert!(
        counts.iter().all(|&n| n > 0),
        "missing success or capacity coverage: {counts:?}"
    );
    eprintln!(
        "60s overload: clients=48 connection_cap=16 complete200={} complete503={} resets=0 truncated=0",
        counts[0], counts[1]
    );
}

/// Browser cancellation must remain available through the bounded control reserve.
#[test]
fn full_table_cancel_preflight_preserves_response_and_control_admission()
-> Result<(), Box<dyn std::error::Error>> {
    let server = Server::start();
    let preflight = |path: &str| -> Result<String, Box<dyn std::error::Error>> {
        let mut socket = TcpStream::connect(server.address)?;
        socket.set_read_timeout(Some(Duration::from_millis(500)))?;
        write!(
            socket,
            "OPTIONS {path} HTTP/1.1\r\nHost: {}\r\nOrigin: http://localhost:{}\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: content-type\r\n\r\n",
            server.address,
            server.address.port()
        )?;
        let mut reply = String::new();
        socket.read_to_string(&mut reply)?;
        Ok(reply)
    };
    let empty = preflight("/v1/cancel")?;
    assert!(empty.starts_with("HTTP/1.1 204 "), "{empty}");
    assert!(empty.ends_with("\r\n\r\n"), "{empty}");
    let idle = (0..17)
        .map(|_| TcpStream::connect(server.address))
        .collect::<Result<Vec<_>, _>>()?;
    std::thread::sleep(Duration::from_millis(100));
    let full = preflight("/v1/cancel")?;
    assert_eq!(
        full, empty,
        "full-table preflight must preserve every header"
    );
    let refused = engine_model::events::count("admission_refused");
    for path in ["/v1/completions", "/v1/kv/release", "/v1/admission"] {
        let reply = preflight(path)?;
        let (head, body) = reply.split_once("\r\n\r\n").ok_or("missing headers")?;
        assert!(head.starts_with("HTTP/1.1 503 "), "{reply}");
        let length: usize = head
            .lines()
            .find_map(|line| line.strip_prefix("Content-Length: "))
            .ok_or("missing Content-Length")?
            .parse()?;
        assert_eq!(body.len(), length, "truncated overflow response");
        assert_eq!(
            body,
            r#"{"error":{"type":"overloaded","message":"connection capacity reached"}}"#
        );
    }
    assert!(engine_model::events::count("admission_refused") >= refused + 3, "each refusal is one event");
    for (method, path, body) in [
        ("GET", "/health", String::new()),
        (
            "POST",
            "/v1/cancel",
            format!(r#"{{"cancel":"{}"}}"#, "c".repeat(64)),
        ),
    ] {
        let start = Instant::now();
        let mut reply = String::new();
        let mut socket = server.send_route(method, path, &body, "");
        socket.set_read_timeout(Some(Duration::from_millis(500)))?;
        socket.read_to_string(&mut reply)?;
        assert!(reply.starts_with("HTTP/1.1 200 "), "{reply}");
        assert!(start.elapsed() < Duration::from_millis(500));
        eprintln!("full-table {path}: {:.6}s", start.elapsed().as_secs_f64());
    }
    eprintln!("M10 empty204 == full204 byte-for-byte; non-control complete503; cancel POST200");
    drop(idle);
    Ok(())
}

#[test]
fn ab_cohorts_admit_one_arm_at_a_time_in_arrival_order() {
    let limits = Limits { ab_ptx: true, ab_cohorts: true, ..Default::default() };
    let server = Server::with_limits(Codec(1, false), true, 0, limits);
    let arms = ["X", "X", "X", "A", "A", "X"];
    let replies: Vec<_> = arms.iter()
        .map(|arm| server.send(&[1, 2, 3], 24, &format!(",\"arm\":\"{arm}\",\"cache\":true")))
        .collect();
    for mut reply in replies {
        let mut body = String::new(); reply.read_to_string(&mut body).unwrap();
        assert!(body.contains("data: [DONE]"), "{body}");
        assert!(!body.contains("engine_error"), "{body}");
        assert!(body.contains("\"cached_tokens\":0"), "{body}");
    }
    // Intake order may differ from send order; the mock asserts no session met a live one of the other arm.
    let mut seen = server.ab.as_ref().unwrap().lock().unwrap().clone();
    seen.sort_unstable();
    assert_eq!(seen, [false, false, true, true, true, true]);
    assert!(server.cohort.as_ref().unwrap().lock().unwrap().1 >= 2, "no same-arm cohort ran concurrently");
}

#[test]
fn ab_mode_serializes_and_forces_cold_requests() {
    let server = Server::with_limits(Codec(1, false), true, 0, Limits { ab_ptx: true, ..Default::default() });
    // Observe admission order while responses remain unread and requests may queue.
    let replies: Vec<_> = ["A", "X", "X", "A"].iter().enumerate()
        .map(|(i, arm)| {
            let reply = server.send(&[1, 2, 3], 3, &format!(",\"arm\":\"{arm}\",\"cache\":true"));
            let start = Instant::now();
            while server.ab.as_ref().unwrap().lock().unwrap().len() <= i {
                assert!(start.elapsed() < Duration::from_secs(5), "A/B admission timed out");
                std::thread::yield_now();
            }
            reply
        }).collect();
    for mut reply in replies {
        let mut body = String::new(); reply.read_to_string(&mut body).unwrap();
        assert!(body.contains("data: [DONE]"), "{body}");
        assert!(!body.contains("engine_error"), "{body}");
        assert!(body.contains("\"cached_tokens\":0"), "{body}");
    }
    assert_eq!(*server.ab.as_ref().unwrap().lock().unwrap(), [false, true, true, false]);
}

#[test]
fn admission_after_an_idle_pass_never_takes_the_idle_release() {
    // A funded plan of eight sessions under host pressure, as the C8 card runs.
    let mut policy = reserve_policy::ReservePolicy::default();
    policy.configure(1000);
    policy.conversation_ended(16384);
    policy.fund(8, 0);
    policy.poll(0, 0);
    let probe = Arc::new(ReserveProbe { policy: Mutex::new(policy), idle_passes: Mutex::new(0),
        idle_pass: Condvar::new(), idle_releases: AtomicUsize::new(0), checks: AtomicUsize::new(0) });
    let server = Server::reserved(4, probe.clone());
    // Each request arrives after a new idle pass: the scheduler marks idle, then drains it.
    let mut seen = 0;
    for prompt in [[1, 2], [3, 4], [5, 6], [2, 1]] {
        probe.idle_pass_after(seen);
        server.run(&prompt, "");
        seen = *probe.idle_passes.lock().unwrap();
    }
    drop(server);
    assert_eq!(probe.checks.load(Ordering::Relaxed), 4);
    assert_eq!(probe.idle_releases.load(Ordering::Relaxed), 0);
}
