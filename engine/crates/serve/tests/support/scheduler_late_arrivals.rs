//! Excludes cancellation, errors, cache, drafting, slow readers, multiple late cohorts, and real devices.
use super::*;
use crate as spark_serve;
use crate::scheduler;
use std::{net::SocketAddr, sync::{Arc, Mutex, Condvar, atomic::AtomicUsize}};
include!("cohort_engine.rs");
type PrefillTrace = Arc<Mutex<Vec<Vec<usize>>>>;
use super::model_tests::{CLOCK, ClockGuard};
struct Codec;
impl Decoder for Codec {
    fn push(&mut self, _: u32) -> Result<String,String> { Ok("x".into()) }
    fn finish(&mut self) -> Result<String,String> { Ok(String::new()) }
}
struct Server { address: SocketAddr, stop: Arc<AtomicBool>, live: Arc<AtomicUsize>, worker: Option<std::thread::JoinHandle<()>> }
impl Server {
    fn send(&self, tokens: &[u32], max: usize, extra: &str) -> TcpStream {
        let body = format!("{{\"stream\":true,\"temperature\":0,\"seed\":0,\"prompt_token_ids\":{tokens:?},\"max_tokens\":{max}{extra}}}");
        self.send_route("POST", "/v1/completions", &body, "")
    }
    fn send_route(&self, method: &str, path: &str, body: &str, _: &str) -> TcpStream {
        let mut socket = TcpStream::connect(self.address).unwrap();
        write!(socket,"{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",self.address,body.len()).unwrap();
        socket
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
type Turns = Arc<Mutex<Vec<(usize, usize)>>>;
#[derive(Default)]
struct ArrivalState {
    older_steps: usize,
    late_sent: bool,
    late_prepared: usize,
    late_queued: usize,
    control_prepared: bool,
}
#[derive(Default)]
struct ArrivalGate { state: Mutex<ArrivalState>, ready: Condvar }
impl ArrivalGate {
    fn wait(&self, mut done: impl FnMut(&ArrivalState) -> bool) {
        let _state = self.ready.wait_while(self.state.lock().unwrap(), |s| !done(s)).unwrap();
    }
    fn update(&self, f: impl FnOnce(&mut ArrivalState)) {
        f(&mut self.state.lock().unwrap());
        self.ready.notify_all();
    }
}
struct ArrivalCodec { gate: Arc<ArrivalGate>, expected: usize }
impl Tokenizer for ArrivalCodec {
    type Decoder<'a> = Codec;
    fn encode(&self, _: &str) -> Result<Vec<u32>, String> {
        self.gate.update(|s| {
            assert_eq!(s.late_prepared, self.expected, "sentinel prepared before all late arrivals");
            s.control_prepared = true;
        });
        Ok(vec![1])
    }
    fn decoder(&self) -> Codec { Codec }
    fn has_token(&self, id: u32) -> bool { id < 8 }
    fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { Ok(vec![1]) }
    fn checkpoint_eos_token_ids(&self) -> Result<Option<&[u32]>, String> {
        if self.gate.state.lock().unwrap().older_steps >= 1 {
            self.gate.update(|s| s.late_prepared += 1);
        }
        Ok(None)
    }
}
struct ArrivalEngine { model: Model, next: AtomicUsize, turns: Turns, groups: PrefillTrace, gate: Arc<ArrivalGate>, grouped: bool, clock_due: std::cell::Cell<bool> }
struct ArrivalSession { inner: Kv, id: usize, turns: Turns, gate: Arc<ArrivalGate> }
impl Session for ArrivalSession {
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
        let (&last, prefix) = tokens.split_last().unwrap();
        self.inner.history.extend_from_slice(prefix); self.inner.history.push(last); Ok(self.inner.logits())
    }
    fn prefill_chunk_min(&self) -> usize { 64 }
    fn prefill_slice(&mut self, tokens: &[u32], last: bool, _: Option<Duration>, _: usize) -> Result<Option<Vec<f32>>, String> {
        self.turns.lock().unwrap().push((self.id, tokens.len()));
        if last { self.prefill(tokens).map(Some) } else { self.prefill_advance(tokens).map(|()| Some(Vec::new())) }
    }
}
impl Engine for ArrivalEngine {
    type Session<'a> = ArrivalSession;
    fn session(&self) -> Result<ArrivalSession, String> {
        self.clock_due.set(true);
        Ok(ArrivalSession { inner: self.model.session()?, id: self.next.fetch_add(1, Ordering::Relaxed), turns: self.turns.clone(), gate: self.gate.clone() })
    }
    fn active_cohort(&self, _: usize) {
        if self.clock_due.replace(false) { CLOCK.with_borrow_mut(|c| *c = c.map(|now| now + Duration::from_millis(10))); }
    }
    fn shared_prompt_limit(&self) -> usize { 0 }
    fn shared_prefill_batch(&self) -> bool { self.grouped }
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
        Some(if prompt > 1 && !self.gate.state.lock().unwrap().control_prepared { 8 * 16384 } else { 16384 })
    }
    fn tick_row_limit(&self) -> Option<usize> {
        let s = self.gate.state.lock().unwrap();
        (s.older_steps >= 2 && !s.control_prepared).then_some(0)
    }
    fn prefill_batch(&self, rows: &mut [spark_serve::BatchRows<'_, ArrivalSession>]) -> Result<Vec<spark_serve::Predictions>, spark_serve::BatchError> {
        self.groups.lock().unwrap().push(rows.iter().map(|r| r.tokens.len()).collect()); self.step_batch(rows)
    }
    fn context_limit(&self) -> usize { 1024 }
    fn vocab_size(&self) -> usize { 8 }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { Some(16384) }
}

fn late_arrivals(count: usize, grouped: bool) -> (Vec<Vec<u32>>, Vec<usize>) {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap(); let address = listener.local_addr().unwrap();
    let (live, calls, heads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let order = Arc::new(Mutex::new(Vec::new())); let stop = Arc::new(AtomicBool::new(false));
    let (turns, groups, gate) = (Turns::default(), PrefillTrace::default(), Arc::new(ArrivalGate::default()));
    let (trace, group_trace, engine_gate, codec_gate) = (turns.clone(), groups.clone(), gate.clone(), gate.clone());
    let model = ArrivalEngine { model: Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(), heads: heads.clone(), order: order.clone(), opt_in: false, retention_extra: 0 }, next: AtomicUsize::new(0), turns, groups, gate: engine_gate, grouped, clock_due: std::cell::Cell::new(false) };
    let codec = ArrivalCodec { gate: codec_gate, expected: count };
    let shutdown = stop.clone();
    let worker = std::thread::spawn(move || {
        let _clock = ClockGuard::new();
        scheduler::serve_requests(listener, &model, Some(&codec),
        &Limits { intake_burst: Duration::ZERO, ..Limits::default() },
        &CacheLimits { entries: 32, retained_bytes: 1 << 20, tokens_per_entry: 1024, logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
        &scheduler::Config { active: 8, connections: 16, prefill_chunk: 64, active_bytes: 8 * 16384,
            session_budget_bytes: 2 << 20, prefix_cache: false, mtp_depth: 0, foreground_rows: 8, background_rows: 16,
            draft: protocol::DraftMode::Plain, draft_global_bank: false, sharing: scheduler::Sharing::default() }, &shutdown, "arrival", count + 2).unwrap() });
    let server = Server { address, stop, live, worker: Some(worker) };
    let extra = ",\"cache\":false,\"prefix_cache\":false,\"draft\":\"plain\",\"ignore_eos\":true";
    let older = server.send(&[1], 256, extra);
    gate.wait(|s| s.older_steps >= 1);
    let late: Vec<_> = (0..count).map(|_| server.send(&[1; 133], 6, extra)).collect();
    gate.update(|s| s.late_sent = true);
    gate.wait(|s| s.older_steps >= 2);
    // Completed client writes do not establish the server's intake order.
    gate.wait(|s| s.late_queued == count);
    let mut sentinel = server.send_route("POST", "/tokenize", r#"{"prompt":"x"}"#, "");
    let mut sentinel_reply = String::new();
    sentinel.read_to_string(&mut sentinel_reply).unwrap();
    assert!(sentinel_reply.starts_with("HTTP/1.1 200 OK"), "tokenize did not return success");
    let ids = std::iter::once(older).chain(late).enumerate().map(|(index, mut socket)| {
        let mut reply = String::new(); socket.read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n"), "missing terminal record");
        assert_eq!(reply.matches("data: [DONE]\n\n").count(), 1, "duplicate terminal record");
        assert!(!reply.contains("\"error\""), "response error");
        assert!(reply.contains("\"finish_reason\":\"length\""), "wrong finish reason");
        let ids = tokens(&reply); let (prompt, output) = if index == 0 { (1, 256) } else { (133, 6) };
        assert_eq!(ids.len(), output, "missing output tokens");
        assert_eq!(ids, oracle(prompt, output), "wrong output token IDs"); ids
    }).collect();
    let events = trace.lock().unwrap().clone();
    assert!(group_trace.lock().unwrap().iter().all(|g| g == &[1]), "late arrival used a grouped pass");
    for id in 1..=count {
        assert_eq!(events.iter().filter(|(who, rows)| *who == id && *rows != 0).map(|(_, rows)| *rows).collect::<Vec<_>>(), [64, 64, 5]);
    }
    let prompt_indices: Vec<_> = events.iter().enumerate().filter(|(_, (id, rows))| *id != 0 && *rows != 0).map(|(i, _)| i).collect();
    for pair in prompt_indices.windows(2) {
        assert!(events[pair[0] + 1..pair[1]].contains(&(0, 0)), "older decode missing between prompt slices: {events:?}");
    }
    // Collapse surplus older-only turns while CPU preparation runs. Every GPU
    // prompt turn must still alternate with an older decode, as before PR 79.
    let mut order = Vec::new();
    for &(id, rows) in &events[prompt_indices[0]..=*prompt_indices.last().unwrap()] {
        if rows != 0 { order.push(rows); } else if id == 0 && order.last() != Some(&0) { order.push(0); }
    }
    (ids, order)
}

#[test]
fn late_arrivals_keep_the_pre_grouping_turn_order_even_when_alone() {
    for count in [1, 3, 7] {
        let old = late_arrivals(count, false); let fresh_only = late_arrivals(count, true);
        assert_eq!(fresh_only, old, "late arrivals={count}");
    }
}
