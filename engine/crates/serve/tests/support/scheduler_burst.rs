//! Excludes socket acceptance, preparation workers, faults, cache, drafting, and device arithmetic.
use super::*;
use super::model_tests::{CLOCK, ClockGuard};
use std::{cell::RefCell, sync::{Arc, Mutex}};

type Trace = Arc<Mutex<Vec<Vec<usize>>>>;
struct Fake(Trace);
struct Kv(Vec<u32>);
struct Long(Vec<u32>);
impl crate::LongPrefill<Kv> for Long {
    fn advance(&mut self, session: &mut Kv, _: Duration) -> Result<Option<Vec<f32>>, String> {
        session.prefill(&self.0).map(Some)
    }
}
impl Session for Kv {
    fn advance(&mut self, token: u32) -> Result<(), String> { self.0.push(token); Ok(()) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.advance(token)?;
        let id = self.0.iter().fold(0usize, |h, &t| (h * 17 + t as usize + 1) % 7);
        let mut logits = vec![0.; 8]; logits[id] = 1.; Ok(logits)
    }
    fn prefill_chunk_size(&self, _: usize) -> usize { 256 - self.0.len() % 256 }
    fn prefill_chunk_min(&self) -> usize { self.prefill_chunk_size(0) }
}
impl Engine for Fake {
    type Session<'a> = Kv;
    fn session(&self) -> Result<Kv, String> { Ok(Kv(Vec::new())) }
    fn shared_batch(&self) -> bool { true }
    fn shared_prefill_batch(&self) -> bool { true }
    fn native_draft_policy(&self) -> bool { true }
    fn shared_decode_rows(&self) -> usize { 8 }
    fn shared_decode_min(&self) -> usize { 3 }
    fn shared_prefill_limit(&self) -> usize { 512 }
    fn shared_prefill_rows(&self) -> usize { 1024 }
    fn shared_prefill_chunk_rows(&self) -> usize { 256 }
    fn shared_prefill_max_tokens(&self) -> usize { usize::MAX }
    fn prefill_long<'s>(&'s self, _: &Kv, tokens: &[u32]) -> Result<Option<Box<dyn crate::LongPrefill<Kv> + 's>>, String> {
        self.0.lock().unwrap().push(vec![0, tokens.len()]);
        Ok(Some(Box::new(Long(tokens.to_vec()))))
    }
    fn prefill_batch(&self, rows: &mut [crate::BatchRows<'_, Kv>]) -> Result<Vec<crate::Predictions>, crate::BatchError> {
        self.0.lock().unwrap().push(rows.iter().map(|r| r.tokens.len()).collect()); self.step_batch(rows)
    }
    fn context_limit(&self) -> usize { 8192 }
    fn vocab_size(&self) -> usize { 8 }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { Some(65536) }
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
struct Driver { turn: usize, hold: bool, early: usize, pending: Vec<Ready>, silent: Vec<Reading>, trace: Trace }
thread_local! { static DRIVER: RefCell<Option<Driver>> = const { RefCell::new(None) }; }
struct Reset;
impl Drop for Reset { fn drop(&mut self) { DRIVER.with_borrow_mut(|d| *d = None); } }

pub(super) fn turn(queued: &mut Vec<Ready>, reading: &mut Vec<Reading>, active: usize) -> bool {
    DRIVER.with_borrow_mut(|driver| {
        let Some(d) = driver else { return false; };
        assert!(d.turn < 64, "burst fixture did not finish within its work bound");
        match d.turn {
            0 => { queued.extend(d.pending.drain(..d.early)); reading.append(&mut d.silent); }
            3 => {
                assert_eq!(d.trace.lock().unwrap().is_empty(), d.hold, "first pass before late arrivals");
                assert_eq!(reading.len(), 4 - d.early, "silent arrivals remain visible to intake");
                reading.clear(); queued.append(&mut d.pending);
            }
            5 => CLOCK.with_borrow_mut(|c| *c = c.map(|now| now + Duration::from_millis(100))),
            _ => {}
        }
        let finished = d.turn > 5 && active == 0 && queued.is_empty() && reading.is_empty();
        CLOCK.with_borrow_mut(|c| *c = c.map(|now| now + Duration::from_millis(3)));
        d.turn += 1;
        finished
    })
}

fn burst(prompt: usize, early: usize, hold: bool) -> (Vec<Vec<u32>>, Vec<usize>) {
    let _clock = ClockGuard::new(); let _reset = Reset;
    let limits = Limits { intake_burst: if hold { Duration::from_millis(100) } else { Duration::ZERO }, ..Limits::default() };
    let trace = Trace::default(); let engine = Fake(trace.clone());
    let mut pending = Vec::new(); let mut peers = Vec::new(); let mut silent = Vec::new(); let mut silent_peers = Vec::new();
    for i in 0..4 {
        let (server, client) = UnixStream::pair().unwrap(); peers.push(client);
        let body = format!(r#"{{"prompt_token_ids":{:?},"max_tokens":3,"stream":true,"cache":false,"prefix_cache":false,"draft":"plain","ignore_eos":true,"temperature":0,"top_k":1,"top_p":1,"seed":0}}"#, vec![1; prompt]);
        let input = http::HttpRequest { path: "/v1/completions".into(), body: body.into_bytes(), authorization: None };
        let prepared = prepare::<NoCodec>(input, Wire::new(Socket::Admin(server)).unwrap(), None, &limits, "burst-test", 8192, 8, &[7])
            .unwrap_or_else(|(_, error)| panic!("prepare: {error}"));
        let Prepared::Generation(ready) = prepared else { unreachable!() }; pending.push(ready);
        if i >= early {
            let (server, client) = UnixStream::pair().unwrap(); silent_peers.push(client);
            silent.push(Reading::new(Socket::Admin(server), &limits).unwrap());
        }
    }
    DRIVER.with_borrow_mut(|d| *d = Some(Driver { turn: 0, hold, early, pending, silent, trace: trace.clone() }));
    let config = Config { active: 8, connections: 16, prefill_chunk: 256, active_bytes: 8 * 65536,
        session_budget_bytes: 2 << 20, prefix_cache: false, mtp_depth: 0, foreground_rows: 8, background_rows: 16,
        draft: protocol::DraftMode::Plain, draft_global_bank: false, sharing: Sharing::default() };
    let cache = CacheLimits { entries: 1, retained_bytes: 0, tokens_per_entry: 8192, logits_per_entry: 8, idle_ttl: Duration::from_secs(60) };
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let (tx, _rx) = mpsc::sync_channel(1); let (_tx, rx) = mpsc::sync_channel(1);
    serve_loop::<_, NoCodec>(&listener, &engine, None, &limits, &cache, &config, &AtomicBool::new(false), None, "burst-test", 0, &tx, &rx).unwrap();
    let mut history = vec![1; prompt];
    let expected: Vec<u32> = (0..3).map(|_| { let t = history.iter().fold(0, |h, &v| (h * 17 + v + 1) % 7); history.push(t); t }).collect();
    let ids = peers.into_iter().map(|mut peer| {
        let mut reply = String::new(); peer.read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n"));
        assert_eq!(reply.matches("data: [DONE]\n\n").count(), 1);
        assert!(!reply.contains("\"error\"")); assert!(reply.contains("\"finish_reason\":\"length\""));
        let ids: Vec<u32> = reply.split("\"token_id\":").skip(1).map(|s| s.split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap()).collect();
        assert_eq!(ids, expected, "complete independent output oracle"); ids
    }).collect();
    let first = trace.lock().unwrap().first().expect("a prompt pass must run").clone();
    (ids, first)
}

#[test]
fn burst_intake_late_request_bytes_join_the_first_prompt_pass() {
    for (prompt, early, pass) in [(32, 2, vec![32; 4]), (600, 1, vec![256; 4])] {
        let (alone, first) = burst(prompt, early, false);
        assert!(first.len() < 4, "without a hold, late requests miss the first pass");
        let (held, first) = burst(prompt, early, true);
        assert_eq!(first, pass, "the complete burst shares its first prompt pass");
        assert_eq!(held, alone, "batch membership preserves output IDs");
    }
}
