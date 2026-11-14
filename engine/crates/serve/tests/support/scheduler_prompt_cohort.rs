//! Cohort integration cases exclude real devices, transport, and arbitrary arrival schedules.
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

#[test]
fn draft_over_canonical_token_stop_drops_decode_plan() {
    struct OverEngine(Arc<AtomicUsize>);
    struct OverSession { pos: usize, strict: bool, over: bool, verified: usize, plain_steps: Arc<AtomicUsize> }
    impl spark_engine::draft::Target for OverSession {
        fn checkpoint(&self) -> usize { self.pos }
        fn truncate(&mut self, pos: usize) -> Result<(), String> { self.pos = pos; Ok(()) }
        fn step(&mut self, t: u32) -> Result<Vec<f32>, String> { Session::step(self, t) }
        fn max_verify_rows(&self) -> usize { 4 }
        fn model_draft_width(&self) -> usize { 1 }
        fn propose_model(&mut self, _: &[u32], _: u32, _: usize) -> Result<Option<spark_engine::draft::ModelProposal>, String> {
            Ok(Some(spark_engine::draft::ModelProposal { tokens: vec![2], proposal_ms: 0.0, catchup_ms: 0.0 }))
        }
    }
    impl Session for OverSession {
        fn draft_target(&mut self) -> Option<&mut dyn spark_engine::draft::Target> { if self.strict { Some(self) } else { None } }
        fn draft_over(&mut self) -> bool { if self.over && self.strict { self.strict = false; true } else { false } }
        fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
            self.pos += tokens.len();
            Ok(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        fn step(&mut self, _: u32) -> Result<Vec<f32>, String> {
            self.pos += 1;
            if self.strict {
                self.verified += 1;
                if self.verified == 2 { self.over = true; }
                let mut row = vec![0.0; 8];
                row[if self.verified == 1 { 3 } else { 4 }] = 1.0;
                return Ok(row);
            }
            self.plain_steps.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        }
    }
    impl Engine for OverEngine {
        type Session<'a> = OverSession;
        fn session(&self) -> Result<OverSession, String> { Ok(OverSession { pos: 0, strict: true, over: false, verified: 0, plain_steps: self.0.clone() }) }
        fn context_limit(&self) -> usize { 64 }
        fn vocab_size(&self) -> usize { 8 }
        fn eos_token_ids(&self) -> &[u32] { &[7] }
        fn session_reservation_bytes(&self) -> Option<usize> { Some(1024) }
        fn single_chat_draft(&self) -> bool { true }
    }
    let plain_steps = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let engine = OverEngine(plain_steps.clone());
    let handle = std::thread::spawn(move || {
        scheduler::serve_requests(
            listener, &engine, Some(&Codec(1, false)), &Limits::default(),
            &CacheLimits { entries: 1, retained_bytes: 0, tokens_per_entry: 64, logits_per_entry: 8, idle_ttl: Duration::from_secs(1) },
            &scheduler::Config {
                active: 4, connections: 8, prefill_chunk: 64, foreground_rows: 4, background_rows: 4,
                active_bytes: 1 << 20, session_budget_bytes: 1 << 20, draft: protocol::DraftMode::Strict,
                mtp_depth: 1, draft_global_bank: false, prefix_cache: false, sharing: Default::default(),
            },
            &AtomicBool::new(false), "over-stop", 1,
        ).unwrap();
    });
    let body = r#"{"prompt":"go","stream":true,"max_tokens":8,"temperature":0,"stop":["xx"],"cache":false}"#;
    let mut socket = TcpStream::connect(address).unwrap();
    socket.set_read_timeout(Some(HANG_GUARD)).unwrap();
    write!(socket, "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut reply = String::new();
    socket.read_to_string(&mut reply).unwrap();
    handle.join().unwrap();
    assert!(reply.starts_with("HTTP/1.1 200") && reply.contains("\"finish_reason\":\"stop\""), "{reply}");
    assert_eq!(plain_steps.load(Ordering::SeqCst), 0, "stop on draft_over canonical token must drop the plain decode plan");
    assert_eq!(reply.matches("data: [DONE]").count(), 1, "{reply}");
}
