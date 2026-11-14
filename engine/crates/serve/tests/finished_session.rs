//! A finished session is asked nothing more by the scheduler.
//!
//! The multi-Spark hybrid service (`tp-serve`, `spark_tp::service::typed`) closes a session by
//! removing it from the service's session table (`CLOSE` in `hybrid_service.rs`), and its
//! `observe` helpers read that table by session id. In `scheduler.rs` (`Active::finish`), the
//! scheduler calls `Session::finish` whenever `self.key.is_none()`, which happens in three ways:
//!   (a) the request ends on a stop string (or parser stop), clearing an existing cache key;
//!   (b) prefix caching is off (`"cache":false` on the request or `Config::prefix_cache == false`);
//!   (c) the engine uses a native draft policy and `prompt.len() <= shared_prompt_limit` (`512`).
//! Later in the same tick (`dispatch.rs`), completion logging asks for `memory_bytes` and
//! `draft_model_memory_bytes` (and a post-finish deadline asks `cached_prompt_tokens`). Indexing
//! the removed session panics with `no entry found for key`, and `panic = "abort"` ends `tp-serve`.
//!
//! `Table` enforces that strict contract on the host: finishing removes the row, and any later
//! observation on that row records the question and indexes the missing key.
use spark_serve::{
    BatchError, BatchOutput, BatchRows, Decoder, Engine, Limits, Predictions, Session, Tokenizer,
    cache::CacheLimits, protocol, scheduler,
};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    sync::{Arc, Barrier, Mutex, atomic::{AtomicBool, Ordering}},
    thread::JoinHandle,
    time::Duration,
};

/// What the engine saw, shared with the test thread.
#[derive(Default)]
struct Ledger {
    opened: usize,
    dropped: usize,
    forks: usize,
    multi_row_batches: usize,
    /// `Session::finish` calls that closed a live row, by session id.
    closes: BTreeMap<u64, usize>,
    /// Questions asked of a row after it closed.
    asked_after_close: Vec<&'static str>,
}

/// The hybrid service's session table, without GPU work.
struct Table {
    rows: RefCell<BTreeMap<u64, Vec<u32>>>,
    next: Cell<u64>,
    ledger: Arc<Mutex<Ledger>>,
}
struct Row<'e> {
    table: &'e Table,
    id: u64,
    closed: bool,
}
impl Row<'_> {
    /// Deliberately indexes `rows[&self.id]` like unguarded `hybrid_service.rs`: any scheduler
    /// question after `Session::finish` both records `question` and panics on the missing key.
    fn observe<T>(&self, question: &'static str, f: impl FnOnce(&Vec<u32>) -> T) -> T {
        if self.closed { self.table.ledger.lock().unwrap().asked_after_close.push(question); }
        f(&self.table.rows.borrow()[&self.id])
    }
    fn history(&mut self) -> Result<std::cell::RefMut<'_, Vec<u32>>, String> {
        if self.closed { return Err("session closed".into()); }
        Ok(std::cell::RefMut::map(self.table.rows.borrow_mut(), |rows| rows.get_mut(&self.id).unwrap()))
    }
}
impl Drop for Row<'_> {
    fn drop(&mut self) {
        let _ = Session::finish(self);
        self.table.ledger.lock().unwrap().dropped += 1;
    }
}
impl Session for Row<'_> {
    fn advance(&mut self, token: u32) -> Result<(), String> {
        self.history()?.push(token);
        Ok(())
    }
    /// Prompt `[0, ..]` never reaches EOS; `[6, ..]` fails immediately; `[4, 4, ..]` fails during
    /// decode with a per-row routing rejection; every other prompt steps `token + 1` up to EOS 7.
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        let (first, second, len) = {
            let mut history = self.history()?;
            history.push(token);
            (history[0], history.get(1).copied(), history.len())
        };
        let next = match first {
            0 => { std::thread::sleep(Duration::from_millis(2)); 1 + len as u32 % 5 }
            6 => return Err("injected generation error".into()),
            4 if second == Some(4) && len >= 4 => return Err("MoE routing rejected".into()),
            _ => { std::thread::sleep(Duration::from_millis(1)); (token + 1) % 8 }
        };
        let mut logits = vec![0.; 8];
        logits[next as usize] = 1.;
        Ok(logits)
    }
    fn prefill_chunk_size(&self, _: usize) -> usize { self.observe("prefill_chunk_size", |_| 512) }
    fn prefill_chunk_min(&self) -> usize { self.observe("prefill_chunk_min", |_| 1) }
    fn cached_prompt_tokens(&self, _: usize) -> Option<usize> {
        self.observe("cached_prompt_tokens", |h| Some(usize::from(!h.is_empty())))
    }
    fn can_fork(&self) -> bool { !self.closed && self.observe("can_fork", |_| true) }
    fn can_fork_prefix(&self, p: usize) -> bool {
        !self.closed && self.observe("can_fork_prefix", |h| p <= h.len())
    }
    fn fork(&self) -> Result<Self, String> {
        if self.closed { return Err("session closed".into()); }
        let cloned = self.observe("fork", Clone::clone);
        let id = self.table.next.get() + 1;
        self.table.next.set(id);
        self.table.rows.borrow_mut().insert(id, cloned);
        let mut ledger = self.table.ledger.lock().unwrap();
        ledger.opened += 1;
        ledger.forks += 1;
        Ok(Row { table: self.table, id, closed: false })
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> {
        self.history()?.truncate(position);
        Ok(())
    }
    fn retained_bytes(&self) -> Option<usize> { Some(16384) }
    fn memory_bytes(&self) -> Option<(usize, usize)> {
        self.observe("memory_bytes", |h| Some((h.len() * 4, h.len() * 4)))
    }
    fn draft_model_memory_bytes(&self) -> Option<(usize, usize)> {
        self.observe("draft_model_memory_bytes", |h| Some((h.len() * 2, h.len() * 2)))
    }
    fn checkpoint_reusable_tokens(&self, tokens: usize) -> usize {
        self.observe("checkpoint_reusable_tokens", |_| tokens)
    }
    fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> { Some(self) }
    fn finish(&mut self) -> Result<(), String> {
        if self.closed { return Ok(()); }
        let slow = self.table.rows.borrow().get(&self.id).is_some_and(|h| h.starts_with(&[3, 3]));
        drop(self.table.rows.borrow_mut().remove(&self.id));
        self.closed = true;
        *self.table.ledger.lock().unwrap().closes.entry(self.id).or_default() += 1;
        if slow { std::thread::sleep(Duration::from_millis(50)); }
        Ok(())
    }
}
impl spark_engine::draft::Target for Row<'_> {
    fn checkpoint(&self) -> usize { self.observe("checkpoint", Vec::len) }
    fn truncate(&mut self, position: usize) -> Result<(), String> { Session::truncate(self, position) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> { Session::step(self, token) }
    fn kv_bf16(&self) -> bool { true }
    fn max_verify_rows(&self) -> usize { 5 }
    fn native_draft_policy(&self) -> bool { true }
}
impl Engine for Table {
    type Session<'a> = Row<'a> where Self: 'a;
    fn session(&self) -> Result<Row<'_>, String> {
        let id = self.next.get() + 1;
        self.next.set(id);
        self.rows.borrow_mut().insert(id, Vec::new());
        self.ledger.lock().unwrap().opened += 1;
        Ok(Row { table: self, id, closed: false })
    }
    fn context_limit(&self) -> usize { 2048 }
    fn vocab_size(&self) -> usize { 8 }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { Some(16384) }
    fn request_reservation_bytes(&self, _: usize) -> Option<usize> { Some(16384) }
    // The hybrid service's scheduling shape (`hybrid_service.rs`, `hybrid.rs`): shared batches,
    // native draft policy and a 512-token shared prompt limit, so short prompts keep no cache key.
    fn shared_batch(&self) -> bool { true }
    fn native_draft_policy(&self) -> bool { true }
    fn shared_prefill_limit(&self) -> usize { 512 }
    fn shared_prefill_rows(&self) -> usize { 256 }
    fn reserve_foreground_slot(&self) -> bool { false }
    fn step_batch<'s>(&'s self, rows: &mut [BatchRows<'_, Self::Session<'s>>]) -> Result<Vec<Predictions>, BatchError> {
        let mut seen = BTreeSet::new();
        for (i, r) in rows.iter().enumerate() {
            if r.session.closed || !seen.insert(r.session.id) || r.tokens.is_empty()
                || r.session.observe("validate_rows", |h| r.tokens.len() > self.context_limit().saturating_sub(h.len()))
            {
                return Err(BatchError { request: Some(i), message: "invalid hybrid batch row".into() });
            }
        }
        if rows.len() > 1 { self.ledger.lock().unwrap().multi_row_batches += 1; }
        rows.iter_mut().map(|r| {
            let result = match r.output {
                BatchOutput::All => r.tokens.iter().map(|&t| Session::step(r.session, t)).collect::<Result<Vec<_>, _>>(),
                BatchOutput::Last => r.session.prefill(r.tokens).map(|v| vec![v]),
                BatchOutput::None => r.session.prefill_advance(r.tokens).map(|()| Vec::new()),
            };
            match result {
                Err(e) if e == "MoE routing rejected" => Ok(Predictions::Rejected(e)),
                other => other.map(Predictions::Logits).map_err(|message| BatchError { request: None, message }),
            }
        }).collect()
    }
}

/// Token `t` decodes to the letter `'a' + t`: token 2 is `"c"`.
struct Letters;
impl Decoder for Letters {
    fn push(&mut self, token: u32) -> Result<String, String> { Ok(char::from(b'a' + token as u8).to_string()) }
    fn finish(&mut self) -> Result<String, String> { Ok(String::new()) }
}
impl Tokenizer for Letters {
    type Decoder<'a> = Letters;
    fn encode(&self, _: &str) -> Result<Vec<u32>, String> { Ok(vec![1]) }
    fn decoder(&self) -> Letters { Letters }
    fn has_token(&self, id: u32) -> bool { id < 8 }
    fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { Ok(vec![1]) }
}

struct Server {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    ledger: Arc<Mutex<Ledger>>,
    worker: Option<JoinHandle<Result<(), String>>>,
}
impl Server {
    fn start() -> Self { Self::with_options(true, Duration::from_secs(10)) }
    fn with_options(prefix_cache: bool, generation_timeout: Duration) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let ledger = Arc::new(Mutex::new(Ledger::default()));
        let (shutdown, shared) = (stop.clone(), ledger.clone());
        let worker = std::thread::spawn(move || {
            let table = Table { rows: RefCell::new(BTreeMap::new()), next: Cell::new(0), ledger: shared };
            let result = scheduler::serve(listener, &table, Some(&Letters),
                &Limits { output_bytes: 1 << 20, generation_timeout, ..Limits::default() },
                &CacheLimits { entries: 4, retained_bytes: 1 << 20, tokens_per_entry: 2048,
                    logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
                &scheduler::Config { active: 4, connections: 16, prefill_chunk: 512,
                    foreground_rows: 8, background_rows: 16, active_bytes: 1 << 20,
                    session_budget_bytes: 1 << 20, draft: protocol::DraftMode::Plain, mtp_depth: 0,
                    draft_global_bank: true, prefix_cache, sharing: scheduler::Sharing::default() },
                &shutdown);
            // Every session, retained or not, is gone once the server returns.
            assert!(table.rows.borrow().is_empty(), "sessions left in the table");
            result
        });
        Self { address, stop, ledger, worker: Some(worker) }
    }
    fn connect(&self, body: &str) -> TcpStream {
        let mut socket = TcpStream::connect(self.address).unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let address = self.address;
        write!(socket, "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()).unwrap();
        socket
    }
    /// One streamed completion read to its end.
    fn complete(&self, prompt: &[u32], max_tokens: usize, extra: &str) -> String {
        let body = format!("{{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":{prompt:?},\"max_tokens\":{max_tokens}{extra}}}");
        let mut output = String::new();
        self.connect(&body).read_to_string(&mut output).unwrap();
        assert!(output.contains("data: [DONE]"), "{output}");
        output
    }
    /// Stop serving; the scheduler thread must return normally, and each session must have
    /// closed once and been asked nothing after it closed.
    fn stop(mut self) -> Ledger {
        self.stop.store(true, Ordering::Relaxed);
        let joined = self.worker.take().unwrap().join();
        assert!(joined.is_ok(), "the scheduler thread panicked");
        joined.unwrap().unwrap();
        let ledger = std::mem::take(&mut *self.ledger.lock().unwrap());
        assert_eq!(ledger.asked_after_close, Vec::<&str>::new(), "a closed session was asked");
        assert_eq!(ledger.dropped, ledger.opened, "every session dropped once");
        assert_eq!(ledger.closes.len(), ledger.opened, "every session closed");
        assert!(ledger.closes.values().all(|&n| n == 1), "a session closed twice: {:?}", ledger.closes);
        ledger
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
    }
}

fn finish_reason(output: &str) -> &str {
    output.rsplit("\"finish_reason\":\"").next().and_then(|s| s.split('"').next()).unwrap_or("")
}

/// Trigger (a): an existing cache key (`cache:true` or automatic long-prompt key) is cleared when
/// generation stops on a stop string (`Active::finish`), closing the session before memory logging.
#[test]
fn stop_string_clears_cache_key_and_asks_closed_session_nothing() {
    let server = Server::start();
    let output = server.complete(&[1], 8, ",\"stop\":\"d\",\"cache\":true");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    // Long prompt (> 512) has an automatic cache key; speculative drafting exercises `draft_model`.
    let mut long = vec![1u32; 519];
    long.extend_from_slice(&[2, 3, 4, 1]);
    let output = server.complete(&long, 8, ",\"stop\":\"e\",\"draft\":\"strict\"");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    let output = server.complete(&[1], 8, ",\"stop\":[\"c\"]");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    let ledger = server.stop();
    assert!(ledger.forks >= 2, "prompt snapshots were forked before stop cleared the active keys");
}

/// Trigger (b): a long prompt (> 512) normally gets `automatic_cache = true`, but turning prefix
/// caching off (`"cache":false` on the request or `Config::prefix_cache == false`) leaves `key = None`.
#[test]
fn prefix_caching_off_on_long_prompt_asks_closed_session_nothing() {
    let mut long = vec![1u32; 519];
    long.push(5); // continues 6, 7 (EOS) without any stop string.

    // Request-level opt-out (`protocol.rs`: `"cache":false` sets `request.prefix_cache = false`).
    let server = Server::start();
    let output = server.complete(&long, 8, ",\"cache\":false");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    let output = server.complete(&long, 1, ",\"cache\":false,\"draft\":\"strict\"");
    assert_eq!(finish_reason(&output), "length", "{output}");
    let ledger = server.stop();
    assert_eq!(ledger.forks, 0, "cache:false must not fork snapshots");

    // Server-level prefix caching disabled (`Config::prefix_cache == false`).
    let disabled = Server::with_options(false, Duration::from_secs(10));
    let output = disabled.complete(&long, 8, "");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    let ledger = disabled.stop();
    assert_eq!(ledger.forks, 0, "disabled prefix_cache must not fork snapshots");
}

/// Trigger (c): on a native-draft-policy engine (`tp-serve`), any plain prompt with
/// `prompt.len() <= shared_prompt_limit` (`512`) and default caching gets `automatic_cache = false`
/// and `key = None`, closing the session even on ordinary EOS or `max_tokens` completion.
#[test]
fn plain_short_prompt_under_shared_limit_asks_closed_session_nothing() {
    let server = Server::start();
    // 1-token plain prompt ending on EOS.
    let output = server.complete(&[5], 8, "");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    // 512-token boundary prompt (`== shared_prompt_limit`) ending on EOS and on `max_tokens`.
    let mut boundary = vec![1u32; 511];
    boundary.push(5);
    let output = server.complete(&boundary, 8, "");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    let output = server.complete(&[1, 2, 3, 4, 1], 3, ",\"draft\":\"strict\"");
    assert_eq!(finish_reason(&output), "length", "{output}");
    let ledger = server.stop();
    assert_eq!(ledger.forks, 0, "short default prompts keep no cache key");
}

#[test]
fn sibling_endings_error_disconnect_retained_and_post_finish_deadline() {
    let server = Server::start();
    // Retained path: `cache:true` keeps a cache key at completion; the cache closes its fork.
    let output = server.complete(&[5], 8, ",\"cache\":true");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    // Immediate generation error.
    let output = server.complete(&[6], 8, "");
    assert!(output.contains("generation"), "{output}");
    // Client disconnect mid-stream.
    let mut socket = server.connect("{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":[0],\"max_tokens\":800}");
    let mut first = [0u8; 1];
    socket.read_exact(&mut first).unwrap();
    drop(socket);
    // Scheduler remains healthy afterwards.
    let output = server.complete(&[5], 8, "");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    server.stop();

    // Post-finish deadline: `Session::finish` closes the row and crosses `generation_timeout`, so
    // `Active::cancel` -> `Active::error` -> `update_usage` runs after the row is already gone.
    let tight = Server::with_options(true, Duration::from_millis(40));
    let mut reply = String::new();
    tight.connect("{\"stream\":false,\"temperature\":0,\"prompt_token_ids\":[3,3],\"max_tokens\":1}")
        .read_to_string(&mut reply).unwrap();
    assert!(reply.contains("generation_deadline"), "{reply}");
    tight.stop();
}

#[test]
fn several_sessions_ending_every_way_in_one_cohort() {
    let server = Arc::new(Server::start());
    let barrier = Arc::new(Barrier::new(4));
    let mut clients = Vec::new();

    // 1. Stop string (with cache snapshot + speculative draft).
    {
        let (server, barrier) = (server.clone(), barrier.clone());
        clients.push(std::thread::spawn(move || {
            barrier.wait();
            let out = server.complete(&[1, 2, 3, 4, 1], 8, ",\"stop\":\"f\",\"cache\":true,\"draft\":\"strict\"");
            assert_eq!(finish_reason(&out), "stop", "{out}");
        }));
    }
    // 2. Max tokens (short prompt, no cache key).
    {
        let (server, barrier) = (server.clone(), barrier.clone());
        clients.push(std::thread::spawn(move || {
            barrier.wait();
            let out = server.complete(&[1], 5, "");
            assert_eq!(finish_reason(&out), "length", "{out}");
        }));
    }
    // 3. Client disconnect mid-stream.
    {
        let (server, barrier) = (server.clone(), barrier.clone());
        clients.push(std::thread::spawn(move || {
            barrier.wait();
            let mut socket = server.connect("{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":[0],\"max_tokens\":800}");
            let mut first = [0u8; 1];
            socket.read_exact(&mut first).unwrap();
        }));
    }
    // 4. Mid-decode per-row routing rejection alongside live cohort peers.
    {
        let (server, barrier) = (server.clone(), barrier.clone());
        clients.push(std::thread::spawn(move || {
            barrier.wait();
            let out = server.complete(&[4, 4], 8, "");
            assert!(out.contains("generation_error"), "{out}");
        }));
    }
    for client in clients { client.join().unwrap(); }

    let server = Arc::try_unwrap(server).ok().unwrap();
    let output = server.complete(&[5], 8, "");
    assert_eq!(finish_reason(&output), "stop", "{output}");
    let ledger = server.stop();
    assert!(ledger.multi_row_batches > 0, "cohort never shared a batch");
}
