//! A host engine with explicit scheduler passes. Product clocks still apply.
use spark_serve::{cache::CacheLimits, protocol, scheduler, Decoder, Engine, Limits, Session, Tokenizer};
use std::{collections::{BTreeMap, BTreeSet}, io::{Read, Write}, net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}, mpsc}, thread::JoinHandle, time::Duration};

#[derive(Clone, Default, Debug)]
pub struct Ledger {
    pub opened: usize, pub dropped: usize, pub attempts: BTreeMap<u32, usize>,
    pub steps: BTreeMap<u32, usize>, pub batches: Vec<usize>, pub restores: Vec<usize>,
    pub validated: BTreeMap<u32, usize>,
    pub rejected_proposals: usize,
    pub retired: BTreeMap<usize, usize>, pub dropped_ids: BTreeSet<usize>, pub modes: BTreeMap<usize, u32>,
}
pub struct Model {
    ledger: Arc<Mutex<Ledger>>, passes: mpsc::Sender<()>, permits: mpsc::Receiver<()>,
    pub hold: Arc<AtomicBool>, pub fault: Arc<AtomicBool>, pub malformed: Arc<AtomicBool>,
}
pub struct Row { id: usize, history: Vec<u32>, ledger: Arc<Mutex<Ledger>>, fault: Arc<AtomicBool>, hold: Arc<AtomicBool>, prompt: usize, restored: usize }
impl Row {
    fn new(model: &Model) -> Self {
        let id = { let mut ledger = model.ledger.lock().unwrap(); ledger.opened += 1; ledger.opened };
        Self { id, history: Vec::new(), ledger: model.ledger.clone(), fault: model.fault.clone(), hold: model.hold.clone(), prompt: 2, restored: 0 }
    }
}
impl Drop for Row { fn drop(&mut self) { let mut ledger = self.ledger.lock().unwrap(); ledger.dropped += 1; ledger.dropped_ids.insert(self.id); } }
impl Session for Row {
    fn retire(&mut self) { *self.ledger.lock().unwrap().retired.entry(self.id).or_default() += 1; }
    fn prefill_slice(&mut self, tokens: &[u32], last: bool, _: Option<Duration>, _: usize) -> Result<Option<Vec<f32>>, String> {
        if self.hold.load(Ordering::Relaxed) { return Ok(None); }
        if last { self.prefill(tokens).map(Some) } else { self.prefill_advance(tokens).map(|_| Some(Vec::new())) }
    }
    fn configure_request(&mut self, request: &protocol::Request) -> Result<(), String> {
        if let protocol::Prompt::Tokens(tokens) = &request.prompt {
            self.prompt = tokens.len(); self.ledger.lock().unwrap().modes.insert(self.id, tokens[0]);
            if tokens[0] == 22 { return Err("injected admission refusal".into()); }
        }
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.history.push(token);
        let mode = self.history[0];
        *self.ledger.lock().unwrap().steps.entry(mode).or_default() += 1;
        let pos = self.history.len().saturating_sub(self.prompt);
        if mode == 16 && pos >= 2 { return Err("injected step failure".into()); }
        let mut logits = vec![-100.0; 256];
        if (18..=20).contains(&mode) && pos >= 2 && self.fault.load(Ordering::Relaxed) {
            return Ok(match mode { 18 => vec![f32::NAN; 256], 19 => vec![f32::INFINITY; 256], _ => vec![0.0; 255] });
        }
        let script: Option<&[u8]> = match mode {
            11 => Some(b"ok"), 12 => Some(b"okSTOPtail"), 13 => Some("€!".as_bytes()),
            14 => Some(b"<tool_call>{\"name\":\"search\",\"arguments\":{\"x\":1}}</tool_call>"),
            15 => Some(b"<think>why</think>ok"), 21 => Some(b"ok<tool_call>{\"name\":\"search\",\"arguments\":{\"x\":1}}"), _ => None,
        };
        let next = script.map_or_else(|| b'A' + (self.history.iter().fold(0u32, |a, b| (a * 17 + b) % 26) as u8),
            |bytes| bytes.get(pos).copied().unwrap_or(255));
        logits[next as usize] = 100.0;
        Ok(logits)
    }
    fn fork(&self) -> Result<Self, String> {
        let id = { let mut ledger = self.ledger.lock().unwrap(); ledger.opened += 1; ledger.opened };
        Ok(Self { id, history: self.history.clone(), ledger: self.ledger.clone(), fault: self.fault.clone(), hold: self.hold.clone(), prompt: self.prompt, restored: self.history.len() })
    }
    fn truncate(&mut self, n: usize) -> Result<(), String> { self.history.truncate(n); self.restored = self.restored.min(n); Ok(()) }
    fn retained_bytes(&self) -> Option<usize> { Some(4096) }
    fn cached_prompt_tokens(&self, n: usize) -> Option<usize> {
        let restored = n.min(self.restored);
        self.ledger.lock().unwrap().restores.push(restored);
        Some(restored)
    }
    fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> { Some(self) }
}
impl spark_engine::draft::Target for Row {
    fn checkpoint(&self) -> usize { self.history.len() }
    fn truncate(&mut self, n: usize) -> Result<(), String> { Session::truncate(self, n) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> { Session::step(self, token) }
    fn model_draft_width(&self) -> usize { 2 }
    fn propose_model(&mut self, _: &[u32], _: u32, limit: usize) -> Result<Option<spark_engine::draft::ModelProposal>, String> {
        let n = limit.min(2);
        self.ledger.lock().unwrap().rejected_proposals += n;
        // Every canonical script emits nonzero tokens. These proposals must be rejected.
        Ok(Some(spark_engine::draft::ModelProposal { tokens: vec![0; n], proposal_ms: 0.0, catchup_ms: 0.0 }))
    }
}
impl Engine for Model {
    type Session<'a> = Row;
    fn session(&self) -> Result<Row, String> { Ok(Row::new(self)) }
    fn session_for_admission(&self, request: &protocol::Request, _: usize) -> Result<Row, String> {
        let mode = match &request.prompt { protocol::Prompt::Tokens(t) => t[0], _ => 10 };
        *self.ledger.lock().unwrap().attempts.entry(mode).or_default() += 1;
        self.session()
    }
    fn context_limit(&self) -> usize { 4096 }
    fn vocab_size(&self) -> usize { 256 }
    fn eos_token_ids(&self) -> &[u32] { &[255] }
    fn session_reservation_bytes(&self) -> Option<usize> { Some(4096) }
    fn request_option_bytes(&self, request: &protocol::Request) -> usize {
        usize::from(request.cache_salt.as_deref() == Some("cannot-fit")) * (1 << 22)
    }
    fn kv_branch_handles(&self) -> bool { true }
    fn validate_request(&self, prompt: &[u32], _: &protocol::Request) -> Result<(), String> {
        *self.ledger.lock().unwrap().validated.entry(prompt[0]).or_default() += 1; Ok(())
    }
    fn reserve_foreground_slot(&self) -> bool { false }
    fn shared_batch(&self) -> bool { true }
    fn shared_prompt_limit(&self) -> usize { 0 }
    fn shared_prefill_batch(&self) -> bool { false }
    fn maintain_memory(&self, _: bool) -> Result<(), String> {
        self.passes.send(()).map_err(|_| "test driver closed")?;
        self.permits.recv().map_err(|_| "test driver closed".into())
    }
    fn step_batch(&self, rows: &mut [spark_serve::BatchRows<'_, Row>])
        -> Result<Vec<spark_serve::Predictions>, spark_serve::BatchError> {
        use spark_serve::{BatchError, BatchOutput, Predictions};
        self.ledger.lock().unwrap().batches.push(rows.len());
        let mut result = Vec::new();
        for row in rows {
            let values = match row.output {
                BatchOutput::All => row.tokens.iter().map(|&t| row.session.step(t)).collect(),
                BatchOutput::Last => row.session.prefill(row.tokens).map(|v| vec![v]),
                BatchOutput::None => row.session.prefill_advance(row.tokens).map(|_| Vec::new()),
            }.map_err(|message| BatchError { request: None, message })?;
            result.push(Predictions::Logits(values));
        }
        if self.malformed.swap(false, Ordering::Relaxed) { result.pop(); }
        Ok(result)
    }
}

pub struct Codec;
pub struct Bytes(Vec<u8>);
impl Decoder for Bytes {
    fn push(&mut self, token: u32) -> Result<String, String> {
        self.0.push(token as u8);
        match std::str::from_utf8(&self.0) {
            Ok(s) => { let out = s.to_owned(); self.0.clear(); Ok(out) }
            Err(e) if e.error_len().is_none() => Ok(String::new()),
            Err(_) => Err("invalid test UTF-8".into()),
        }
    }
    fn finish(&mut self) -> Result<String, String> { String::from_utf8(std::mem::take(&mut self.0)).map_err(|_| "incomplete test UTF-8".into()) }
}
impl Tokenizer for Codec {
    type Decoder<'a> = Bytes;
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        // Chat input selects a script. Raw completion requests use explicit IDs.
        let mode = text.parse().unwrap_or(10);
        Ok(vec![mode, 1])
    }
    fn encode_chat(&self, input: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> {
        let value = engine_model::json::parse(input.messages.last().unwrap().as_bytes(), Default::default()).unwrap();
        self.encode(value.object().unwrap()["content"].string().unwrap())
    }
    fn decoder(&self) -> Bytes { Bytes(Vec::new()) }
    fn has_token(&self, token: u32) -> bool { token < 256 }
}

pub struct Client { socket: TcpStream, pub wire: String, pending: Vec<u8>, done: bool }
impl Client {
    pub fn append(&mut self, bytes: &[u8]) { self.socket.write_all(bytes).unwrap(); }
    pub fn poll(&mut self) {
        let mut buf = [0; 8192];
        loop {
            match self.socket.read(&mut buf) {
                Ok(0) => { assert!(self.pending.is_empty(), "truncated UTF-8 on wire"); self.done = true; return; }
                Ok(n) => {
                    self.pending.extend_from_slice(&buf[..n]);
                    let valid = match std::str::from_utf8(&self.pending) {
                        Ok(s) => s.len(),
                        Err(e) if e.error_len().is_none() => e.valid_up_to(),
                        Err(_) => panic!("invalid UTF-8 on wire"),
                    };
                    self.wire.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    self.pending.drain(..valid);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return,
                Err(e) => panic!("client read: {e}"),
            }
        }
    }
}
pub struct Server {
    address: SocketAddr, permits: mpsc::Sender<()>, passes: mpsc::Receiver<()>, stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<(), String>>>, pub ledger: Arc<Mutex<Ledger>>,
    pub hold: Arc<AtomicBool>, pub fault: Arc<AtomicBool>, pub malformed: Arc<AtomicBool>,
}
impl Server {
    pub fn keyed() -> (Self, String) {
        use std::os::unix::fs::OpenOptionsExt;
        let owner = spark_serve::cache::SessionKey::generate().unwrap();
        let path = std::env::temp_dir().join(format!("vade-client-contract-fake-key-{}", std::process::id()));
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).unwrap();
        file.write_all(b"obviously-fake-alice-scope").unwrap(); drop(file);
        let key = spark_serve::auth::ApiKey::load(Some(&path), false).unwrap().unwrap().with_owner(Some(owner));
        std::fs::remove_file(path).unwrap();
        (Self::start(8, 16, Limits { api_key: Some(key), intake_burst: Duration::ZERO, ..Default::default() }), owner.expose())
    }
    pub fn start(active: usize, connections: usize, limits: Limits) -> Self { Self::with_budget(active, connections, limits, 1 << 20) }
    pub fn budgeted() -> Self { Self::with_budget(2, 8, Limits { intake_burst: Duration::ZERO, ..Default::default() }, 4096) }
    fn with_budget(active: usize, connections: usize, mut limits: Limits, active_bytes: usize) -> Self {
        // No elapsed time is an oracle. Use the API's largest bounds for host descheduling.
        limits.read_timeout = Duration::from_secs(3600); limits.write_timeout = Duration::from_secs(3600);
        limits.generation_timeout = Limits::MAX_GENERATION_TIMEOUT;
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let (permits, permit_rx) = mpsc::channel();
        let (pass_tx, passes) = mpsc::channel();
        let ledger = Arc::new(Mutex::new(Ledger::default()));
        let hold = Arc::new(AtomicBool::new(false));
        let fault = Arc::new(AtomicBool::new(false));
        let malformed = Arc::new(AtomicBool::new(false));
        let model = Model { ledger: ledger.clone(), passes: pass_tx, permits: permit_rx, hold: hold.clone(), fault: fault.clone(), malformed: malformed.clone() };
        let stop = Arc::new(AtomicBool::new(false));
        let shutdown = stop.clone();
        let worker = std::thread::spawn(move || scheduler::serve(listener, &model, Some(&Codec), &limits,
            &CacheLimits { entries: 32, retained_bytes: 1 << 20, tokens_per_entry: 4096, logits_per_entry: 256, idle_ttl: Duration::from_secs(86400) },
            &scheduler::Config { active, connections, prefill_chunk: 512, active_bytes, session_budget_bytes: 1 << 21,
                draft: protocol::DraftMode::Plain, mtp_depth: 0, draft_global_bank: false, prefix_cache: true,
                sharing: scheduler::Sharing { report_prefix_reuse: true, private_counts: true, ..Default::default() }, foreground_rows: 8, background_rows: 8 }, &shutdown));
        passes.recv().unwrap();
        Self { address, permits, passes, stop, worker: Some(worker), ledger, hold, fault, malformed }
    }
    pub fn new() -> Self { Self::start(8, 16, Limits { intake_burst: Duration::ZERO, ..Default::default() }) }
    pub fn pass(&self) { self.permits.send(()).unwrap(); self.passes.recv().unwrap(); std::thread::yield_now(); }
    pub fn until(&self, mut done: impl FnMut() -> bool) {
        for _ in 0..10000 { if done() { return; } self.pass(); }
        panic!("scheduler pass budget exhausted");
    }
    pub fn raw(&self, path: &str, body: &str, headers: &str) -> Client {
        let mut socket = TcpStream::connect(self.address).unwrap();
        let head = format!("POST {path} HTTP/1.1\r\nHost: {}\r\n{headers}\r\n", self.address);
        // At capacity the product has a short control read bound. Buffer the complete
        // request before the first scheduler pass, so no partial head waits on a permit.
        socket.write_all(format!("{head}{body}").as_bytes()).unwrap();
        socket.set_nonblocking(true).unwrap();
        Client { socket, wire: String::new(), pending: Vec::new(), done: false }
    }
    /// Only use with normal intake room. Full-table tests send complete requests.
    pub fn fragmented(&self, path: &str, body: &str, headers: &str) -> Client {
        let mut socket = TcpStream::connect(self.address).unwrap();
        let head = format!("POST {path} HTTP/1.1\r\nHost: {}\r\n{headers}\r\n", self.address);
        let split = head.len() / 2;
        socket.write_all(&head.as_bytes()[..split]).unwrap();
        self.pass();
        socket.write_all(&head.as_bytes()[split..]).unwrap();
        socket.write_all(body.as_bytes()).unwrap();
        socket.set_nonblocking(true).unwrap();
        Client { socket, wire: String::new(), pending: Vec::new(), done: false }
    }
    pub fn send(&self, path: &str, body: &str, key: &str) -> Client {
        let auth = if key.is_empty() { String::new() } else { format!("Authorization: Bearer {key}\r\n") };
        self.raw(path, body, &format!("Content-Type: application/json\r\nContent-Length: {}\r\n{auth}", body.len()))
    }
    pub fn get(&self, path: &str, key: &str) -> Client {
        let mut socket = TcpStream::connect(self.address).unwrap();
        let auth = if key.is_empty() { String::new() } else { format!("Authorization: Bearer {key}\r\n") };
        write!(socket, "GET {path} HTTP/1.1\r\nHost: {}\r\n{auth}\r\n", self.address).unwrap();
        socket.shutdown(Shutdown::Write).unwrap(); socket.set_nonblocking(true).unwrap();
        Client { socket, wire: String::new(), pending: Vec::new(), done: false }
    }
    pub fn finish(&self, client: &mut Client) -> String {
        self.until(|| { client.poll(); client.done }); client.wire.clone()
    }
    pub fn run(&self, path: &str, body: &str, key: &str) -> String { self.finish(&mut self.send(path, body, key)) }
    pub fn stop(mut self) -> Ledger {
        self.shutdown();
        let ledger = self.ledger.lock().unwrap().clone();
        assert_eq!(ledger.opened, ledger.dropped, "session ownership must balance"); ledger
    }
    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.permits.send(());
        if let Some(worker) = self.worker.take() { worker.join().unwrap().unwrap(); }
    }
}
impl Drop for Server { fn drop(&mut self) { self.shutdown(); } }
