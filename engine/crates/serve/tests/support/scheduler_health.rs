//! Excludes real devices, arbitrary socket arrivals, and sustained control-route load.
use super::*;

#[test]
fn health_without_the_key_holds_the_state_only() {
    let path = std::env::temp_dir().join(format!("vade-health-key-{}", std::process::id()));
    std::fs::write(&path, "0123456789abcdef\n").unwrap();
    std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
    let server = Server::with_limits(Codec(1, false), false, 0, Limits { api_key: spark_serve::auth::ApiKey::load(Some(&path), false).unwrap(), ..Limits::default() });
    let health = |key: &str| { let mut reply = String::new(); server.send_route("GET", "/health", "", key).read_to_string(&mut reply).unwrap(); reply };
    let (open, keyed) = (health(""), health("0123456789abcdef"));
    assert!(open.starts_with("HTTP/1.1 200 ") && open.ends_with("\r\n\r\n{\"status\":\"ready\"}"), "{open}");
    // The head gives nothing away either: the body length is the same for every model name.
    assert!(open.contains("\r\nContent-Length: 18\r\n"), "{open}");
    assert!(keyed.starts_with("HTTP/1.1 200 ") && keyed.contains("\"model\":") && keyed.contains("\"active\":0") && keyed.contains("\"caps\":{"), "{keyed}");
    std::fs::remove_file(path).unwrap();
}

#[derive(Clone)]
struct PrepareGate(Arc<(Mutex<(bool, bool)>, Condvar)>);
impl PrepareGate {
    fn new() -> Self { Self(Default::default()) }
    fn started(&self) {
        let (state, wake) = &*self.0;
        let (state, wait) = wake.wait_timeout_while(state.lock().unwrap(), Duration::from_secs(120), |state| !state.0).unwrap();
        assert!(state.0 && !wait.timed_out(), "request preparation did not start");
    }
    fn release(&self) {
        let (state, wake) = &*self.0;
        state.lock().unwrap().1 = true;
        wake.notify_all();
    }
    fn block(&self) {
        let (state, wake) = &*self.0;
        let mut state = state.lock().unwrap();
        state.0 = true;
        wake.notify_all();
        while !state.1 { state = wake.wait(state).unwrap(); }
    }
}
struct BlockingCodec(PrepareGate);
impl Decoder for BlockingCodec {
    fn push(&mut self, _: u32) -> Result<String, String> { Ok("x".into()) }
    fn finish(&mut self) -> Result<String, String> { Ok(String::new()) }
}
impl Tokenizer for BlockingCodec {
    type Decoder<'a> = BlockingCodec;
    fn encode(&self, _: &str) -> Result<Vec<u32>, String> { self.0.block(); Ok(vec![1]) }
    fn decoder(&self) -> BlockingCodec { BlockingCodec(self.0.clone()) }
    fn has_token(&self, id: u32) -> bool { id < 8 }
    fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { self.0.block(); Ok(vec![1]) }
}
struct PrepareRelease(PrepareGate);
impl Drop for PrepareRelease { fn drop(&mut self) { self.0.release(); } }

#[derive(Default)]
struct HoldState { held: usize, opened: usize, release: bool }
#[derive(Default)]
struct HoldGate { state: Mutex<HoldState>, ready: Condvar }
impl HoldGate {
    fn hold(&self, announced: &mut bool) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.release { return false; }
        if !*announced {
            *announced = true;
            state.held += 1;
            self.ready.notify_all();
        }
        true
    }
    fn wait(&self, count: usize) {
        let (state, wait) = self.ready.wait_timeout_while(self.state.lock().unwrap(), Duration::from_secs(120), |state| state.held < count).unwrap();
        assert_eq!(state.held, count, "held-prefill hand-off timed out after {} sessions", state.held);
        assert!(!wait.timed_out());
    }
    fn release(&self) {
        self.state.lock().unwrap().release = true;
        self.ready.notify_all();
    }
    fn note_opened(&self) {
        self.state.lock().unwrap().opened += 1;
        self.ready.notify_all();
    }
    fn opened(&self) -> usize { self.state.lock().unwrap().opened }
}
struct HoldRelease(Arc<HoldGate>);
impl Drop for HoldRelease { fn drop(&mut self) { self.0.release(); } }
struct HoldEngine { inner: Model, gate: Arc<HoldGate> }
struct HoldSession { inner: Kv, gate: Arc<HoldGate>, announced: bool }
impl Session for HoldSession {
    fn advance(&mut self, token: u32) -> Result<(), String> { self.inner.advance(token) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> { self.inner.step(token) }
    fn prefill_slice(&mut self, tokens: &[u32], last: bool, _: Option<Duration>, _: usize) -> Result<Option<Vec<f32>>, String> {
        if self.gate.hold(&mut self.announced) { return Ok(None); }
        if last { self.inner.prefill(tokens).map(Some) }
        else { self.inner.prefill_advance(tokens).map(|()| Some(Vec::new())) }
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> { self.inner.truncate(position) }
}
impl Engine for HoldEngine {
    type Session<'a> = HoldSession;
    fn session(&self) -> Result<HoldSession, String> {
        let inner = self.inner.session()?;
        self.gate.note_opened();
        Ok(HoldSession { inner, gate: self.gate.clone(), announced: false })
    }
    fn context_limit(&self) -> usize { self.inner.context_limit() }
    fn vocab_size(&self) -> usize { self.inner.vocab_size() }
    fn eos_token_ids(&self) -> &[u32] { self.inner.eos_token_ids() }
    fn shared_prompt_limit(&self) -> usize { 0 }
    fn session_reservation_bytes(&self) -> Option<usize> { self.inner.session_reservation_bytes() }
}
fn held_server(gate: Arc<HoldGate>) -> Server {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let (live, calls, heads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let order = Arc::new(Mutex::new(Vec::new()));
    let engine = HoldEngine { inner: Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(), heads: heads.clone(), order: order.clone(), opt_in: false, retention_extra: 0 }, gate };
    let shutdown = stop.clone();
    let worker = std::thread::spawn(move || scheduler::serve(listener, &engine, Some(&Codec(1, false)),
        // Client hang guards fail in 120 seconds. These server deadlines cannot
        // release a held session before a successful test sends the release signal.
        &Limits { read_timeout: Duration::from_secs(3600), generation_timeout: Limits::MAX_GENERATION_TIMEOUT, ..Limits::default() },
        &CacheLimits { entries: 32, retained_bytes: 1 << 20, tokens_per_entry: 1024, logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
        &scheduler::Config { active: 2, connections: 16, prefill_chunk: 1, foreground_rows: 8, background_rows: 16,
            active_bytes: 32768, session_budget_bytes: 2 << 20, draft: protocol::DraftMode::Plain, mtp_depth: 0,
            draft_global_bank: false, prefix_cache: false, sharing: scheduler::Sharing::default() }, &shutdown).unwrap());
    Server { ab: None, cohort: None, address, stop, live, calls, heads, order, worker: Some(worker) }
}
fn read_with_guard(mut socket: TcpStream) -> Result<String, std::io::Error> {
    socket.set_read_timeout(Some(Duration::from_secs(120)))?;
    let mut reply = String::new();
    socket.read_to_string(&mut reply)?;
    Ok(reply)
}
fn health(server: &Server) -> String { read_with_guard(server.send_route("GET", "/health", "", "")).unwrap() }

#[test]
fn health_counts_requests_in_cpu_preparation_as_queued() {
    let gate = PrepareGate::new();
    let server = Server::with_limits(BlockingCodec(gate.clone()), false, 0, Limits::default());
    let release = PrepareRelease(gate.clone());
    let pending = server.send_route("POST", "/v1/responses", r#"{"input":"hi","max_output_tokens":1}"#, "");
    gate.started();
    let health = health(&server);
    drop(release);
    read_with_guard(pending).unwrap();
    assert!(health.contains(r#""queued":1,"preparing":1"#), "{health}");
}

#[test]
fn queued_responses_cancellation_reports_prompt_usage() {
    let gate = Arc::new(HoldGate::default());
    let server = held_server(gate.clone());
    let release = HoldRelease(gate.clone());
    let active: Vec<_> = (0..2).map(|_| server.send(&[1], 1, ",\"cache\":false")).collect();
    gate.wait(2);
    let cancel = format!("{:064x}", 1);
    let body = format!(r#"{{"input":"hi","max_output_tokens":1,"cancel":"{cancel}"}}"#);
    let pending = server.send_route("POST", "/v1/responses", &body, "");
    let mut last = String::new();
    for _ in 0..64 {
        last = health(&server);
        if last.contains(r#""queued":1,"preparing":0"#) { break; }
    }
    assert!(last.contains(r#""queued":1,"preparing":0"#), "request did not enter the READY queue: {last}");
    let cancellation = read_with_guard(server.send_route("POST", "/v1/cancel", &format!(r#"{{"cancel":"{cancel}"}}"#), "")).unwrap();
    let reply = read_with_guard(pending).unwrap();
    assert_eq!(gate.opened(), 2, "cancelled request was admitted instead of cancelled in the READY queue");
    // The two admitted sessions yield before prompt commit until the queued
    // cancellation response is complete. They cannot free an admission slot early.
    drop(release);
    for socket in active { read_with_guard(socket).unwrap(); }
    assert!(cancellation.starts_with("HTTP/1.1 200 "), "{cancellation}");
    assert!(reply.contains(r#""input_tokens":1,"output_tokens":0,"total_tokens":1"#), "{reply}");
}

/// The intake part of `full_table_cancel_preflight_preserves_response_and_control_admission`. It is in this file so
/// that scheduler.rs stays within the ratchet's 1,500 lines. A socket holds a slot only when its head passed the
/// checks: 26 sockets that send nothing or a bad head refuse nothing. Then 17 good heads with no body fill the table.
/// 26 is one more than the 17 slots and the 8 reserve places together. Not more: this file's tests run in parallel
/// with a 60 second test that holds about 100 sockets, and the file must pass at the default limit of 256 open files.
pub(super) fn fill_the_table(address: SocketAddr, preflight: &dyn Fn(&str) -> Result<String, Box<dyn std::error::Error>>, empty: &str) -> Result<Vec<TcpStream>, Box<dyn std::error::Error>> {
    let silent = (0..26).map(|n| { let mut socket = TcpStream::connect(address)?; if n % 2 == 0 { socket.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\n\r\n")?; } Ok(socket) }).collect::<std::io::Result<Vec<_>>>()?;
    assert_eq!(preflight("/v1/completions")?, empty, "a socket with no good head refused a request");
    drop(silent);
    let idle = (0..17).map(|_| { let mut socket = TcpStream::connect(address)?;
        write!(socket, "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: 8\r\n\r\n")?;
        Ok(socket) }).collect::<std::io::Result<Vec<_>>>()?;
    let started = Instant::now();
    while !preflight("/v1/completions")?.starts_with("HTTP/1.1 503 ") {
        assert!(started.elapsed() < HANG_GUARD, "table did not reach capacity before the hang guard");
        std::thread::yield_now();
    }
    Ok(idle)
}
