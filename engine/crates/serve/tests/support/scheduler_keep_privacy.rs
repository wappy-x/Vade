//! Excludes device memory and arbitrary tenant schedules; expiry uses the real cache clock.
use super::*;

#[test]
fn private_cache_events_do_not_reach_another_scope() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let live = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let heads = Arc::new(AtomicUsize::new(0));
    let order = Arc::new(Mutex::new(Vec::new()));
    let model = Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(),
        heads: heads.clone(), order: order.clone(), opt_in: false, retention_extra: 0 };
    let shutdown = stop.clone();
    let worker = std::thread::spawn(move || {
        scheduler::serve(listener, &model, Some(&Codec(1, false)), &Limits::default(),
            &CacheLimits { entries: 1, retained_bytes: 1 << 20, tokens_per_entry: 1024,
                logits_per_entry: 8, idle_ttl: Duration::from_secs(300) },
            &scheduler::Config { active: 3, connections: 16, prefill_chunk: 1,
                foreground_rows: 8, background_rows: 16, active_bytes: 32768,
                session_budget_bytes: 2 << 20, draft: protocol::DraftMode::Plain,
                mtp_depth: 0, draft_global_bank: false, prefix_cache: true,
                sharing: scheduler::Sharing { private_counts: true, report_prefix_reuse: true, ..Default::default() } },
            &shutdown).unwrap();
    });
    let server = Server { ab: None, cohort: None, address, stop, live, calls, heads, order, worker: Some(worker) };
    let keys = std::array::from_fn::<_, 2, _>(|_| spark_serve::cache::SessionKey::generate().unwrap().expose());
    for (prompt, reused) in [("[1,2,3]", "0"), ("[1,2,3]", "3"), ("[4,5,6]", "0"), ("[1,2,3]", "0")] {
        let body = format!(r#"{{"stream":true,"temperature":0,"prompt_token_ids":{prompt},"max_tokens":1}}"#);
        let mut reply = String::new();
        server.send_authorized(&body, &keys[0]).read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
        assert_eq!(header(&reply, "X-Prefix-Reused-Tokens:"), reused);
        for route in ["/health", "/v1/events"] {
            let mut reply = String::new();
            server.send_route("GET", route, "", &keys[1]).read_to_string(&mut reply).unwrap();
            assert!(reply.starts_with("HTTP/1.1 200 "), "{reply}");
            assert!(!reply.contains("prefix_cache_"), "{route} leaked another scope's cache events: {reply}");
            let body = reply.split_once("\r\n\r\n").unwrap().1;
            let value = engine_model::json::parse(body.as_bytes(), Default::default()).unwrap();
            if route == "/health" {
                assert_eq!(value.object().unwrap()["prefix_cache"].object().unwrap().len(), 1);
            }
        }
    }
    // Completion receipts also carry process event counts. The second scope must
    // not receive the first scope's cache activity through that copy either.
    let mut reply = String::new();
    server.send_authorized(r#"{"stream":true,"temperature":0,"prompt_token_ids":[6,5],"max_tokens":1,"cache":false}"#,
        &keys[1]).read_to_string(&mut reply).unwrap();
    assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
    assert!(!reply.contains("prefix_cache_"), "completion leaked another scope's cache events: {reply}");
}

/// AN IDLE SERVER DROPS A PAST-DUE PROMPT BY ITSELF. The serve loop balances the cache on every pass, with or without
/// requests, and that pass checks the idle times at most once a second. One prompt is kept with an idle time of one
/// second. Then NO request touches the cache: the test only sleeps and reads `/health`, which reads the counters and
/// changes nothing. `/health` must come to show the prompt expired (not evicted) and no entry left.
#[test]
fn an_idle_server_drops_a_past_due_prompt_with_no_request() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let (live, calls, heads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let order = Arc::new(Mutex::new(Vec::new()));
    let model = Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(),
        heads: heads.clone(), order: order.clone(), opt_in: false, retention_extra: 0 };
    let shutdown = stop.clone();
    let worker = std::thread::spawn(move || {
        scheduler::serve(listener, &model, Some(&Codec(1, false)), &Limits::default(),
            &CacheLimits { entries: 4, retained_bytes: 1 << 20, tokens_per_entry: 1024,
                logits_per_entry: 8, idle_ttl: Duration::from_secs(1) },
            &scheduler::Config { active: 3, connections: 16, prefill_chunk: 1,
                foreground_rows: 8, background_rows: 16, active_bytes: 32768,
                session_budget_bytes: 2 << 20, draft: protocol::DraftMode::Plain,
                mtp_depth: 0, draft_global_bank: false, prefix_cache: true,
                sharing: scheduler::Sharing::default() },
            &shutdown).unwrap();
    });
    let server = Server { ab: None, cohort: None, address, stop, live, calls, heads, order, worker: Some(worker) };
    let key = spark_serve::cache::SessionKey::generate().unwrap().expose();
    let mut reply = String::new();
    server.send_authorized(r#"{"stream":true,"temperature":0,"prompt_token_ids":[1,2,3],"max_tokens":1}"#, &key).read_to_string(&mut reply).unwrap();
    assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
    // The counters of /health: (snapshots, entries, expired, evicted).
    let counters = || {
        let mut reply = String::new();
        server.send_route("GET", "/health", "", &key).read_to_string(&mut reply).unwrap();
        let count = |name: &str| reply.split(&format!("\"{name}\":")).nth(1).unwrap_or_else(|| panic!("/health has no {name}: {reply}"))
            .chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap();
        (count("snapshots"), count("entries"), count("expired"), count("evicted"))
    };
    // Health reads do not touch the cache. Observe expiry instead of assuming a wake time.
    let started = Instant::now();
    let mut seen = counters();
    while seen.2 == 0 {
        assert!(started.elapsed() < HANG_GUARD, "cache expiry hang guard expired");
        std::thread::sleep(Duration::from_millis(250)); seen = counters();
    }
    assert_eq!(seen, (1, 0, 1, 0), "one prompt was kept, and the idle server dropped it by its idle time with no request");
}

/// WHERE PRIVATE COUNTS MEET THE SHORT KEYLESS /health: a key file is configured and private counts are on. A prompt is
/// kept and then reused, by a caller that shows the key. `/health` WITHOUT the key, or with a wrong key, is the short
/// body, the state only (18 bytes): the short body wins, and nothing of the cache is in it. `/health` WITH the key is
/// the full body, and private counts still apply to it: the cache is only "enabled", the device bytes are null, and
/// no `prefix_cache_` event name is counted. `/v1/events` with the key holds no cache line either.
#[test]
fn with_a_key_file_and_private_counts_health_is_short_without_the_key_and_hides_the_cache_with_it() {
    let path = std::env::temp_dir().join(format!("vade-private-health-key-{}", std::process::id()));
    std::fs::write(&path, "0123456789abcdef\n").unwrap();
    std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
    let limits = Limits { api_key: spark_serve::auth::ApiKey::load(Some(&path), false).unwrap(), ..Limits::default() };
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let (live, calls, heads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let order = Arc::new(Mutex::new(Vec::new()));
    let model = Model { ab: None, cohort: None, live: live.clone(), calls: calls.clone(),
        heads: heads.clone(), order: order.clone(), opt_in: false, retention_extra: 0 };
    let shutdown = stop.clone();
    let worker = std::thread::spawn(move || {
        scheduler::serve(listener, &model, Some(&Codec(1, false)), &limits,
            &CacheLimits { entries: 4, retained_bytes: 1 << 20, tokens_per_entry: 1024,
                logits_per_entry: 8, idle_ttl: Duration::from_secs(300) },
            &scheduler::Config { active: 3, connections: 16, prefill_chunk: 1,
                foreground_rows: 8, background_rows: 16, active_bytes: 32768,
                session_budget_bytes: 2 << 20, draft: protocol::DraftMode::Plain,
                mtp_depth: 0, draft_global_bank: false, prefix_cache: true,
                sharing: scheduler::Sharing { private_counts: true, report_prefix_reuse: true, ..Default::default() } },
            &shutdown).unwrap();
    });
    let server = Server { ab: None, cohort: None, address, stop, live, calls, heads, order, worker: Some(worker) };
    let key = "0123456789abcdef";
    // A miss that keeps the prompt, then a hit: the cache has something to hide.
    for reused in ["0", "3"] {
        let mut reply = String::new();
        server.send_authorized(r#"{"stream":true,"temperature":0,"prompt_token_ids":[1,2,3],"max_tokens":1}"#, key).read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("data: [DONE]\n\n"), "{reply}");
        assert_eq!(header(&reply, "X-Prefix-Reused-Tokens:"), reused);
    }
    let get = |route: &str, key: &str| { let mut reply = String::new(); server.send_route("GET", route, "", key).read_to_string(&mut reply).unwrap(); reply };
    // No key, or a wrong one: the state only.
    for shown in ["", "fedcba9876543210"] {
        let short = get("/health", shown);
        assert!(short.starts_with("HTTP/1.1 200 ") && short.ends_with("\r\n\r\n{\"status\":\"ready\"}") && short.contains("\r\nContent-Length: 18\r\n"), "key {shown:?}: {short}");
    }
    // The key: the full body, with the cache's counters, events and device bytes hidden.
    let full = get("/health", key);
    assert!(full.starts_with("HTTP/1.1 200 ") && full.contains("\"model\":") && full.contains("\"active\":0") && full.contains("\"caps\":{"), "{full}");
    assert!(full.contains("\"prefix_cache\":{\"enabled\":true}") && full.contains("\"device_bytes\":null") && !full.contains("prefix_cache_"), "{full}");
    for name in ["\"entries\":", "\"hits\":", "\"misses\":", "\"reused_tokens\":", "\"retained_bytes\":"] { assert!(!full.contains(name), "/health with the key shows {name}: {full}"); }
    let events = get("/v1/events", key);
    assert!(events.starts_with("HTTP/1.1 200 ") && !events.contains("prefix_cache_"), "{events}");
    std::fs::remove_file(path).unwrap();
}
