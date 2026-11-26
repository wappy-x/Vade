//! Late prompts beside a live decoder, on an engine whose serial prompt slice is the shared
//! limit (`prompt_cohort::late_join`). The serial-slice engines are in
//! `tests/support/scheduler_late_arrivals.rs`.
//! Excludes cancellation, errors, cache, drafting, slow readers and real devices.
use super::*;
use crate as spark_serve;
use crate::scheduler;
use std::{net::SocketAddr, sync::{Arc, Mutex, Condvar, atomic::AtomicUsize}};
include!("../../tests/support/cohort_engine.rs");
use super::model_tests::{CLOCK, ClockGuard};

/// (chat, rows) of each decode step (rows 0) and of each serial prompt slice.
type Turns = Arc<Mutex<Vec<(usize, usize)>>>;
/// (chat, rows) of each row of each shared prompt pass.
type Passes = Arc<Mutex<Vec<Vec<(usize, usize)>>>>;
struct Codec;
impl Decoder for Codec {
    fn push(&mut self, _: u32) -> Result<String,String> { Ok("x".into()) }
    fn finish(&mut self) -> Result<String,String> { Ok(String::new()) }
}
struct Server { address: SocketAddr, stop: Arc<AtomicBool>, live: Arc<AtomicUsize>, worker: Option<std::thread::JoinHandle<()>> }
impl Server {
    fn send(&self, path: &str, body: &str) -> TcpStream {
        let mut socket = TcpStream::connect(self.address).unwrap();
        write!(socket,"POST {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",self.address,body.len()).unwrap();
        socket
    }
    fn chat(&self, tokens: &[u32], max: usize) -> TcpStream {
        self.send("/v1/completions", &format!("{{\"stream\":true,\"temperature\":0,\"seed\":0,\"prompt_token_ids\":{tokens:?},\"max_tokens\":{max},\"cache\":false,\"prefix_cache\":false,\"draft\":\"plain\",\"ignore_eos\":true}}"))
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed); self.worker.take().unwrap().join().unwrap();
        assert_eq!(self.live.load(Ordering::Relaxed), 0, "session ownership remains");
    }
}
fn tokens(reply: &str) -> Vec<u32> {
    reply.split("\"token_id\":").skip(1).map(|s| s.split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap()).collect()
}
fn oracle(prompt: usize, count: usize) -> Vec<u32> {
    let mut history = vec![1; prompt];
    (0..count).map(|_| { let token = history.iter().fold(0, |hash, &t| (hash * 17 + t + 1) % 7); history.push(token); token }).collect()
}
/// What the client and the server did so far. The gate makes every late chat live in one turn.
#[derive(Default)]
struct Arrival { older_steps: usize, late_sent: bool, late_prepared: usize, late_queued: usize, released: bool }
#[derive(Default)]
struct Gate { state: Mutex<Arrival>, ready: Condvar }
impl Gate {
    fn wait(&self, mut done: impl FnMut(&Arrival) -> bool) {
        let _state = self.ready.wait_while(self.state.lock().unwrap(), |s| !done(s)).unwrap();
    }
    fn update(&self, f: impl FnOnce(&mut Arrival)) {
        f(&mut self.state.lock().unwrap());
        self.ready.notify_all();
    }
}
struct GateCodec { gate: Arc<Gate>, expected: usize }
impl Tokenizer for GateCodec {
    type Decoder<'a> = Codec;
    // The sentinel request. It releases late admission and the older decode.
    fn encode(&self, _: &str) -> Result<Vec<u32>, String> {
        self.gate.update(|s| {
            assert_eq!(s.late_prepared, self.expected, "sentinel prepared before all late arrivals");
            s.released = true;
        });
        Ok(vec![1])
    }
    fn decoder(&self) -> Codec { Codec }
    fn has_token(&self, id: u32) -> bool { id < 8 }
    fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { Ok(vec![1]) }
    fn checkpoint_eos_token_ids(&self) -> Result<Option<&[u32]>, String> {
        if self.gate.state.lock().unwrap().older_steps >= 1 { self.gate.update(|s| s.late_prepared += 1); }
        Ok(None)
    }
}
struct JoinEngine { model: Model, next: AtomicUsize, turns: Turns, passes: Passes, gate: Arc<Gate>, clock_due: std::cell::Cell<bool> }
struct JoinSession { inner: Kv, id: usize, turns: Turns, gate: Arc<Gate> }
impl Session for JoinSession {
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.turns.lock().unwrap().push((self.id, 0));
        if self.id == 0 {
            let step = {
                let mut s = self.gate.state.lock().unwrap();
                s.older_steps += 1;
                self.gate.ready.notify_all();
                s.older_steps
            };
            if step == 1 { self.gate.wait(|s| s.late_sent); }
        }
        self.inner.history.push(token); Ok(self.inner.logits())
    }
    fn advance(&mut self, token: u32) -> Result<(), String> { self.inner.history.push(token); Ok(()) }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.inner.history.extend_from_slice(tokens); Ok(self.inner.logits())
    }
    fn prefill_chunk_min(&self) -> usize { 64 }
    fn prefill_slice(&mut self, tokens: &[u32], last: bool, _: Option<Duration>, _: usize) -> Result<Option<Vec<f32>>, String> {
        self.turns.lock().unwrap().push((self.id, tokens.len()));
        if last { self.prefill(tokens).map(Some) } else { self.prefill_advance(tokens).map(|()| Some(Vec::new())) }
    }
}
impl Engine for JoinEngine {
    type Session<'a> = JoinSession;
    fn session(&self) -> Result<JoinSession, String> {
        self.clock_due.set(true);
        Ok(JoinSession { inner: self.model.session()?, id: self.next.fetch_add(1, Ordering::Relaxed), turns: self.turns.clone(), gate: self.gate.clone() })
    }
    fn active_cohort(&self, _: usize) {
        if self.clock_due.replace(false) { CLOCK.with_borrow_mut(|c| *c = c.map(|now| now + Duration::from_millis(10))); }
    }
    fn shared_prompt_limit(&self) -> usize { 0 }
    fn shared_prefill_batch(&self) -> bool { true }
    fn shared_prefill_limit(&self) -> usize { 256 }
    fn shared_prefill_rows(&self) -> usize { 2048 }
    fn shared_prefill_max_tokens(&self) -> usize { 256 }
    fn shared_decode_rows(&self) -> usize { 8 }
    fn shared_decode_min(&self) -> usize { 2 }
    fn validate_request(&self, prompt: &[u32], _: &protocol::Request) -> Result<(), String> {
        if prompt.len() > 1 { self.gate.update(|s| s.late_queued += 1); }
        Ok(())
    }
    // Keep intake running while the client waits for every late preparation.
    // Hold late admission and older decode until the sentinel releases both.
    fn request_reservation_bytes_for_prompt(&self, prompt: usize, _: usize, _: bool, _: Option<usize>, _: bool) -> Option<usize> {
        Some(if prompt > 1 && !self.gate.state.lock().unwrap().released { 8 * 16384 } else { 16384 })
    }
    fn tick_row_limit(&self) -> Option<usize> {
        let s = self.gate.state.lock().unwrap();
        (s.older_steps >= 2 && !s.released).then_some(0)
    }
    fn prefill_batch(&self, rows: &mut [spark_serve::BatchRows<'_, JoinSession>]) -> Result<Vec<spark_serve::Predictions>, spark_serve::BatchError> {
        self.passes.lock().unwrap().push(rows.iter().map(|r| (r.session.id, r.tokens.len())).collect()); self.step_batch(rows)
    }
    fn context_limit(&self) -> usize { 1024 }
    fn vocab_size(&self) -> usize { 8 }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { Some(16384) }
}

/// One older chat decodes. `count` chats of `length` prompt rows arrive beside it and become
/// live in one turn. `chunk` is `Config::prefill_chunk`; the shared limit of the engine is 256.
/// Returns the shared passes that held a late chat, and the serial slices of the late chats.
fn arrivals(count: usize, length: usize, chunk: usize) -> (Vec<Vec<(usize, usize)>>, Vec<(usize, usize)>) {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap(); let address = listener.local_addr().unwrap();
    let (live, calls, heads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let order = Arc::new(Mutex::new(Vec::new())); let stop = Arc::new(AtomicBool::new(false));
    let (turns, passes, gate) = (Turns::default(), Passes::default(), Arc::new(Gate::default()));
    let model = JoinEngine { model: Model { ab: None, cohort: None, live: live.clone(), calls, heads, order, opt_in: false, retention_extra: 0 }, next: AtomicUsize::new(0), turns: turns.clone(), passes: passes.clone(), gate: gate.clone(), clock_due: std::cell::Cell::new(false) };
    let codec = GateCodec { gate: gate.clone(), expected: count };
    let shutdown = stop.clone();
    let worker = std::thread::spawn(move || {
        let _clock = ClockGuard::new();
        scheduler::serve_requests(listener, &model, Some(&codec),
        &Limits { intake_burst: Duration::ZERO, ..Limits::default() },
        &CacheLimits { entries: 32, retained_bytes: 1 << 20, tokens_per_entry: 1024, logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
        &scheduler::Config { active: 8, connections: 16, prefill_chunk: chunk, active_bytes: 8 * 16384,
            session_budget_bytes: 2 << 20, prefix_cache: false, mtp_depth: 0, foreground_rows: 8, background_rows: 16,
            draft: protocol::DraftMode::Plain, draft_global_bank: false, sharing: scheduler::Sharing::default() }, &shutdown, "late-join", count + 2).unwrap() });
    let server = Server { address, stop, live, worker: Some(worker) };
    let older = server.chat(&[1], 256);
    gate.wait(|s| s.older_steps >= 1);
    let late: Vec<_> = (0..count).map(|_| server.chat(&vec![1; length], 6)).collect();
    gate.update(|s| s.late_sent = true);
    gate.wait(|s| s.older_steps >= 2);
    // Completed client writes do not establish the server's intake order.
    gate.wait(|s| s.late_queued == count);
    let mut sentinel = server.send("/tokenize", r#"{"prompt":"x"}"#);
    let mut sentinel_reply = String::new();
    sentinel.read_to_string(&mut sentinel_reply).unwrap();
    assert!(sentinel_reply.starts_with("HTTP/1.1 200 OK"), "tokenize did not return success");
    for (index, mut socket) in std::iter::once(older).chain(late).enumerate() {
        let mut reply = String::new(); socket.read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n") && !reply.contains("\"error\""), "no clean end of stream");
        let (prompt, output) = if index == 0 { (1, 256) } else { (length, 6) };
        assert_eq!(tokens(&reply), oracle(prompt, output), "wrong output token IDs");
    }
    drop(server);
    let late_passes = passes.lock().unwrap().iter().filter(|pass| pass.iter().any(|&(id, _)| id != 0)).cloned().collect();
    let slices = turns.lock().unwrap().iter().copied().filter(|&(id, rows)| id != 0 && rows != 0).collect();
    (late_passes, slices)
}

#[test]
fn late_prompts_beside_a_decoder_are_read_whole_in_one_shared_pass() {
    // The serial slice (256 rows) is the shared limit: late chats take no serial turn.
    for count in [1, 3, 7] {
        let (passes, slices) = arrivals(count, 133, 256);
        assert_eq!(passes, [(1..=count).map(|id| (id, 133)).collect::<Vec<_>>()], "late arrivals={count}");
        assert!(slices.is_empty(), "a late chat took a serial turn; late arrivals={count}: {slices:?}");
    }
}

#[test]
fn a_smaller_serial_slice_keeps_the_serial_turns() {
    // The serial slice (64 rows) cuts a 133-row prompt: each late chat keeps its own slices.
    let (passes, slices) = arrivals(3, 133, 64);
    assert!(passes.is_empty(), "a late chat joined a shared pass: {passes:?}");
    assert_eq!(slices.iter().filter(|&&(id, _)| id == 2).map(|&(_, rows)| rows).collect::<Vec<_>>(), [64, 64, 5]);
    assert_eq!(slices.len(), 9);
}
