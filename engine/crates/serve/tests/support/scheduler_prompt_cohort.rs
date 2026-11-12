use super::*;

fn run(lengths: &[usize], shared_decode: bool) -> (Vec<Vec<u32>>, Vec<Vec<usize>>) {
    let trace = PrefillTrace::default(); let start = Arc::new(AtomicBool::new(false));
    let server = Server::tiled_mode(lengths.len(), Some((trace.clone(), start.clone())),
        Limits::default().intake_burst, LastTurn::default(), shared_decode);
    let sockets: Vec<_> = lengths.iter().map(|&n| server.send(&vec![1; n], 6,
        ",\"cache\":false,\"prefix_cache\":false,\"draft\":\"plain\",\"ignore_eos\":true,\"top_k\":1,\"top_p\":1,\"seed\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0")).collect();
    start.store(true, Ordering::Release);
    let output = sockets.into_iter().map(|mut socket| {
        let mut reply = String::new(); socket.read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n") && !reply.contains("generation_error"), "{reply}");
        let ids = tokens(&reply); assert_eq!(ids.len(), 6); ids
    }).collect();
    let events = trace.lock().unwrap().clone(); (output, events)
}

#[test]
fn independent_prefill_batches_do_not_need_shared_decode() {
    for count in [4, 8] {
        let (ids, calls) = run(&vec![127; count], false);
        assert_eq!(calls, vec![vec![127; count]], "{calls:?}");
        assert_eq!(ids, run(&vec![127; count], true).0);
        assert_eq!(ids, vec![run(&[127], false).0[0].clone(); count]);
    }
}

#[test]
fn a_prompt_cohort_shares_the_funded_row_bound_and_keeps_each_tail() {
    let lengths = [128, 129, 130, 131, 132, 133, 134, 135];
    let (ids, calls) = run(&lengths, false);
    assert_eq!(calls.first(), Some(&vec![128; 8]), "{calls:?}");
    let mut tails = calls[1].clone(); tails.sort_unstable(); assert_eq!(tails, (1..=7).collect::<Vec<_>>(), "{calls:?}");
    assert_eq!(calls.len(), 2, "{calls:?}");
    assert!(calls.iter().all(|rows| rows.iter().sum::<usize>() <= 1024));
    for (&length, ids) in lengths.iter().zip(ids) { assert_eq!(ids, run(&[length], false).0[0]); }
}

#[test]
fn a_long_prompt_retains_its_serial_path_beside_a_short_cohort() {
    let (ids, calls) = run(&[133, 2048, 134], false);
    assert!(calls.contains(&vec![0, 2048]), "{calls:?}");
    // If the long prompt finishes before the short prompts become due, it is
    // already a decoder. Those short prompts must then keep the serial path.
    if calls.first() == Some(&vec![0, 2048]) {
        assert!(calls.contains(&vec![0, 133]) && calls.contains(&vec![0, 134]), "{calls:?}");
    } else { assert!(calls.contains(&vec![133, 134]), "{calls:?}"); }
    for (&length, ids) in [133, 2048, 134].iter().zip(ids) { assert_eq!(ids, run(&[length], false).0[0]); }
}

type Turns = Arc<Mutex<Vec<(usize, usize)>>>;
struct ArrivalEngine { model: Model, next: AtomicUsize, turns: Turns, groups: PrefillTrace, grouped: bool }
struct ArrivalSession { inner: Kv, id: usize, turns: Turns }
impl Session for ArrivalSession {
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.turns.lock().unwrap().push((self.id, 0)); self.inner.step(token)
    }
    fn advance(&mut self, token: u32) -> Result<(), String> { self.inner.history.push(token); Ok(()) }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let (&last, prefix) = tokens.split_last().unwrap();
        self.inner.history.extend_from_slice(prefix); self.inner.step(last)
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
        Ok(ArrivalSession { inner: self.model.session()?, id: self.next.fetch_add(1, Ordering::Relaxed), turns: self.turns.clone() })
    }
    fn shared_prompt_limit(&self) -> usize { 0 }
    fn shared_prefill_batch(&self) -> bool { self.grouped }
    fn shared_prefill_limit(&self) -> usize { 256 }
    fn shared_prefill_rows(&self) -> usize { 2048 }
    fn shared_prefill_max_tokens(&self) -> usize { 256 }
    fn shared_decode_rows(&self) -> usize { 8 }
    fn shared_decode_min(&self) -> usize { 2 }
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
    let (turns, groups) = (Turns::default(), PrefillTrace::default()); let trace = turns.clone(); let group_trace = groups.clone();
    let model = ArrivalEngine { model: Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(), heads: heads.clone(), order: order.clone(), opt_in: false, retention_extra: 0 }, next: AtomicUsize::new(0), turns, groups, grouped };
    let shutdown = stop.clone();
    let worker = std::thread::spawn(move || scheduler::serve_requests(listener, &model, Some(&Codec(1, false)),
        &Limits { generation_timeout: Duration::from_secs(10), ..Limits::default() },
        &CacheLimits { entries: 32, retained_bytes: 1 << 20, tokens_per_entry: 1024, logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
        &scheduler::Config { active: 8, connections: 16, prefill_chunk: 64, active_bytes: 8 * 16384,
            session_budget_bytes: 2 << 20, prefix_cache: false, mtp_depth: 0, foreground_rows: 8, background_rows: 16,
            draft: protocol::DraftMode::Plain, draft_global_bank: false, sharing: scheduler::Sharing::default() }, &shutdown, "arrival", count + 1).unwrap());
    let server = Server { ab: None, cohort: None, address, stop, live, calls, heads, order, worker: Some(worker) };
    let extra = ",\"cache\":false,\"prefix_cache\":false,\"draft\":\"plain\",\"ignore_eos\":true";
    let older = server.send(&[1], 256, extra);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !trace.lock().unwrap().contains(&(0, 0)) {
        assert!(Instant::now() < deadline); std::thread::sleep(Duration::from_millis(1));
    }
    let late: Vec<_> = (0..count).map(|_| server.send(&[1; 133], 6, extra)).collect();
    let ids = std::iter::once(older).chain(late).map(|mut socket| {
        let mut reply = String::new(); socket.read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n") && !reply.contains("generation_error"), "{reply}"); tokens(&reply)
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
