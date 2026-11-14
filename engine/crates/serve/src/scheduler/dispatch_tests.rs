use super::{fork_failed, prompt_cohort_candidate, prompt_uses_cohort};

// Hybrid pooled geometry: 512-row short limit, 256-row canonical tile.
const LIMIT: usize = 512;
const CHUNK: usize = 256;

#[test]
fn lone_long_prompt_takes_long_path() {
    // One 8192-token tile-aligned prompt alone: not cohort, so the
    // dispatch long-prefill branch (which requires !shared_prompt) runs it.
    assert!(!prompt_uses_cohort(8192, 0, LIMIT, CHUNK, 1));
    // The same prompt with another prompt-ready job shares cohort chunks.
    assert!(prompt_uses_cohort(8192, 0, LIMIT, CHUNK, 2));
    assert!(prompt_uses_cohort(8192, 0, LIMIT, CHUNK, 4));
    // It is a cohort candidate either way; only the count decides.
    assert!(prompt_cohort_candidate(8192, 0, LIMIT, CHUNK));
    // Mid-prompt at a tile boundary behaves the same.
    assert!(!prompt_uses_cohort(4096, 4096, LIMIT, CHUNK, 1));
    assert!(prompt_uses_cohort(4096, 4096, LIMIT, CHUNK, 2));
}

#[test]
fn short_and_unaligned_prompts_keep_existing_paths() {
    // Short prompts share even when alone (existing behavior).
    assert!(prompt_uses_cohort(LIMIT, 0, LIMIT, CHUNK, 1));
    assert!(prompt_uses_cohort(1, 8191, LIMIT, CHUNK, 1));
    // Long prompts off a tile boundary never join the cohort.
    assert!(!prompt_uses_cohort(8192, 100, LIMIT, CHUNK, 2));
    // Engines without chunked sharing never admit long prompts.
    assert!(!prompt_uses_cohort(8192, 0, LIMIT, 0, 2));
    assert!(prompt_uses_cohort(LIMIT, 0, LIMIT, 0, 1));
}

#[cfg(target_os = "linux")]
#[test]
fn tile_constants_match_engine_schedule() {
    assert_eq!(spark_engine::hybrid::schedule::PROMPT_ROWS, CHUNK);
    assert_eq!(spark_engine::hybrid::schedule::ROUTED_GROUP, 4096);
}

#[test]
fn cohort_and_long_paths_visit_same_tiles_in_order() {
    // Engine schedule geometry (hybrid/schedule.rs; Linux-gated there).
    const PROMPT_ROWS: usize = CHUNK;
    const ROUTED_GROUP: usize = 4096;
    // Cohort: one canonical tile per tick. Long path: ROUTED_GROUP
    // passes whose attention visits PROMPT_ROWS tiles in scan order
    // (hybrid_decode.rs prompt tiling). Both must partition the same
    // per-session tile sequence, so switching paths at a tile boundary
    // cannot reorder any token's arithmetic.
    for total in [8192usize, 32768] {
        let cohort: Vec<(usize, usize)> = (0..total).step_by(PROMPT_ROWS)
            .map(|first| (first, PROMPT_ROWS.min(total - first))).collect();
        let long: Vec<(usize, usize)> = (0..total).step_by(ROUTED_GROUP)
            .flat_map(|group| {
                let rows = ROUTED_GROUP.min(total - group);
                (0..rows).step_by(PROMPT_ROWS)
                    .map(move |first| (group + first, PROMPT_ROWS.min(rows - first)))
            }).collect();
        assert_eq!(cohort, long, "tile order differs at {total} tokens");
    }
}

#[test]
fn each_passed_fork_error_is_one_event() {
    // Counts only grow, so other tests that note events at the same time cannot break this check.
    let before = engine_model::events::count("prefix_cache_fork_failed");
    fork_failed("automatic", 7, &["fork failed: a".into(), "truncate failed: b".into()]);
    fork_failed("session", 7, &[]);
    assert!(engine_model::events::count("prefix_cache_fork_failed") >= before + 2);
}

/// The tests below fill a pool of 64 readers, so each holds up to about 140 sockets (both ends are in
/// this process). The default limit of open files on macOS is 256, and tests run in parallel: this
/// lock lets one of them run at a time, and each is sized to stay well under that limit alone.
fn sockets() -> std::sync::MutexGuard<'static, ()> {
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ONE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
/// A loopback listener and `count` client sockets that have connected to it.
fn connected(count: usize) -> (super::TcpListener, Vec<std::net::TcpStream>) {
    let listener = super::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let clients = (0..count).map(|_| std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap()).collect();
    (listener, clients)
}
/// One pass of the dispatcher over the readers, as `serve_loop` does it: a reader that ends with
/// an error gets its refusal and stays as a response until that drains, and a response that has
/// drained leaves. A complete request leaves the table here; it is returned.
fn pass(reading: &mut Vec<super::Reading>, limits: &super::Limits, access: &super::http::AccessPolicy) -> Vec<Result<super::http::HttpRequest, String>> {
    let (mut seen, mut at) = (Vec::new(), 0);
    while at < reading.len() {
        if reading[at].responding {
            if reading[at].wire.flush().is_err() || (reading[at].wire.drained() && reading[at].replied(limits)) { reading.swap_remove(at); } else { at += 1; }
            continue;
        }
        match reading[at].read(limits, access) {
            Ok(None) => at += 1,
            Ok(Some(request)) => { reading.swap_remove(at); seen.push(Ok(request)); }
            Err(error) => {
                let mut input = reading.swap_remove(at);
                let (status, body) = crate::auth::request_error(&error);
                if input.wire.response(status, &body, limits).is_ok() { input.responding = true; reading.push(input); }
                seen.push(Err(error));
            }
        }
    }
    seen
}
/// Passes over the readers until `done` holds, for at most 30 seconds: a test waits for the state
/// that it needs, not for a fixed time, so a loaded machine cannot fail it.
fn pass_until(reading: &mut Vec<super::Reading>, limits: &super::Limits, access: &super::http::AccessPolicy,
    done: impl Fn(&[super::Reading], &[Result<super::http::HttpRequest, String>]) -> bool) -> Vec<Result<super::http::HttpRequest, String>> {
    let (mut seen, until) = (Vec::new(), super::Instant::now() + super::Duration::from_secs(30));
    loop {
        seen.extend(pass(reading, limits, access));
        if done(reading, &seen) || super::Instant::now() >= until { return seen; }
        std::thread::sleep(super::Duration::from_millis(2));
    }
}
/// Accept until no connection waits, as passes of the scheduler do. The table has 17 slots.
fn accept_all(listener: &super::TcpListener, reading: &mut Vec<super::Reading>, limits: &super::Limits) -> usize {
    std::thread::sleep(super::Duration::from_millis(20));
    let mut total = 0;
    loop {
        let access = super::http::AccessPolicy::new(listener.local_addr().unwrap(), limits).unwrap();
        match super::accept_connections(listener, reading, limits, &access, 17, 32) { 0 => return total, count => total += count }
    }
}
fn request(address: std::net::SocketAddr, head: &str) -> std::net::TcpStream {
    use std::io::Write;
    let mut client = std::net::TcpStream::connect(address).unwrap();
    client.write_all(head.replace("{host}", &address.to_string()).as_bytes()).unwrap();
    client
}
const HEAD: &str = "POST /v1/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n";
/// The limits of an engine that has the API key `0123456789abcdef`.
fn keyed(name: &str) -> super::Limits {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::temp_dir().join(format!("vade-{name}-key-{}", std::process::id()));
    std::fs::write(&path, "0123456789abcdef\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let limits = super::Limits { api_key: crate::auth::ApiKey::load(Some(&path), false).unwrap(), ..Default::default() };
    std::fs::remove_file(&path).unwrap();
    limits
}

#[test]
fn silent_sockets_hold_no_slot_and_never_stop_the_accept_loop() {
    let _sockets = sockets();
    let limits = super::Limits::default();
    // A table of 17 slots, and 40 peers that connect and send nothing.
    let (listener, silent) = connected(40);
    let address = listener.local_addr().unwrap();
    let access = super::http::AccessPolicy::new(address, &limits).unwrap();
    let mut reading = Vec::new();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 40);
    assert!(reading.iter().all(|input| !input.wire.control_only), "a silent socket never makes the table full");
    assert!(reading.iter().all(|input| input.deadline > super::Instant::now() + super::Duration::from_secs(5)), "with room, a silent socket keeps the normal deadline");
    // 30 more make 70: the pool of 64 is full, and the last 6 each take the place of the oldest one. Each one is accepted.
    // (30, not more: with the server's ends this test holds about 135 sockets; see `sockets`.)
    let first = reading[0].wire.received;
    let more: Vec<_> = (0..30).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 30);
    assert_eq!(reading.len(), 64);
    assert!(reading.iter().all(|input| !input.wire.control_only && input.wire.received > first), "the oldest readers left");
    assert!(pass(&mut reading, &limits, &access).is_empty());
    // A peer that sends a request is accepted behind them and is read in the same pass: it is not refused.
    let _real = request(address, &format!("{HEAD}\r\n{{}}"));
    assert_eq!(accept_all(&listener, &mut reading, &limits), 1);
    assert!(reading.iter().all(|input| !input.wire.control_only));
    let seen = pass(&mut reading, &limits, &access);
    assert_eq!(seen.len(), 1);
    let request = seen[0].as_ref().unwrap();
    assert_eq!((request.path.as_str(), request.body.as_slice()), ("/v1/completions", &b"{}"[..]));
    assert!(engine_model::events::count("intake_evicted") >= 1);
    drop((silent, more));
}

#[test]
fn a_refusal_for_a_peer_with_no_key_holds_no_slot() {
    let _sockets = sockets();
    let limits = keyed("intake");
    let (listener, _) = connected(0);
    let address = listener.local_addr().unwrap();
    let access = super::http::AccessPolicy::new(address, &limits).unwrap();
    // 33 peers send a complete head with no key (or a bad head) and keep the socket open.
    let keyless: Vec<_> = (0..33).map(|peer| request(address, &if peer % 2 == 0 { format!("{HEAD}\r\n") } else { "POST /v1/completions HTTP/1.1\r\nHost: x\r\n\r\n".into() })).collect();
    let mut reading = Vec::new();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 33);
    let seen = pass(&mut reading, &limits, &access);
    assert_eq!(seen.iter().filter(|result| result.as_ref().is_err_and(|error| error == crate::auth::UNAUTHORIZED)).count(), 33);
    // Each one got its 401 and now waits for the peer to close (up to one second). None holds a slot.
    assert_eq!(reading.iter().filter(|input| input.responding && input.wire.headless && input.head.is_none()).count(), 33);
    assert!(pass(&mut reading, &limits, &access).is_empty());
    assert_eq!(reading.len(), 33, "the refusals still wait for their peers");
    // A peer with the key is a normal reader, and its request is read.
    let _keyed = request(address, &format!("{HEAD}Authorization: Bearer 0123456789abcdef\r\n\r\n{{}}"));
    assert_eq!(accept_all(&listener, &mut reading, &limits), 1);
    assert!(!reading.last().unwrap().wire.control_only, "refusals for peers with no key filled the table");
    assert!(pass(&mut reading, &limits, &access).iter().any(|result| result.as_ref().is_ok_and(|request| request.path == "/v1/completions")));
    // The health request that needs no key holds no slot either, also while its short reply drains.
    let probes: Vec<_> = (0..20).map(|_| request(address, "GET /health HTTP/1.1\r\nHost: {host}\r\n\r\n")).collect();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 20);
    let mut index = 0;
    while index < reading.len() {
        if reading[index].responding { index += 1; continue; }
        let probe = reading[index].read(&limits, &access).unwrap().unwrap();
        assert_eq!(probe.path, super::http::HEALTH_LIVE);
        reading[index].wire.response(200, "{\"status\":\"ready\",\"model\":\"m\"}", &limits).unwrap();
        reading[index].responding = true;
    }
    let _late = request(address, &format!("{HEAD}Authorization: Bearer 0123456789abcdef\r\n\r\n{{}}"));
    assert_eq!(accept_all(&listener, &mut reading, &limits), 1);
    assert!(!reading.last().unwrap().wire.control_only, "health probes with no key filled the table");
    drop((keyless, probes));
}

#[test]
fn requests_that_passed_the_check_hold_the_slots() {
    let _sockets = sockets();
    use std::io::Write;
    let limits = super::Limits::default();
    let (listener, mut clients) = connected(20);
    let address = listener.local_addr().unwrap();
    let access = super::http::AccessPolicy::new(address, &limits).unwrap();
    // 17 peers send a complete head and no body: each holds a slot until its deadline.
    for client in &mut clients[..17] {
        write!(client, "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: 8\r\n\r\n").unwrap();
    }
    let mut reading = Vec::new();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 20);
    assert!(pass(&mut reading, &limits, &access).is_empty());
    assert_eq!(reading.iter().filter(|input| input.head.is_some()).count(), 17);
    // The table is full: the next peers are overflow readers with the short deadline.
    let (late, _held) = (std::net::TcpStream::connect(address).unwrap(), std::net::TcpStream::connect(address).unwrap());
    assert_eq!(accept_all(&listener, &mut reading, &limits), 2);
    let overflow: Vec<_> = reading.iter().filter(|input| input.wire.control_only).collect();
    assert_eq!(overflow.len(), 2);
    assert!(overflow.iter().all(|input| input.deadline <= super::Instant::now() + super::Duration::from_millis(100)));
    // Peers that send nothing take no place in the reserve of 8: they wait in the pool, and none waits in the listen queue.
    let more: Vec<_> = (0..10).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 10);
    assert_eq!(reading.iter().filter(|input| input.wire.control_only).count(), 12);
    assert_eq!(reading.iter().filter(|input| input.head.is_some()).count(), 17, "a request that passed the check lost its slot");
    drop((late, more, clients));
    // A response on a wire that the dispatcher handed back has no head, and it holds its slot.
    let (listener, clients) = connected(18);
    let mut reading = Vec::new();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 18);
    for input in &mut reading[..17] { input.responding = true; }
    let _next = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 1);
    assert!(reading.last().unwrap().wire.control_only);
    drop(clients);
}

/// With the table full, sockets that send nothing must not take the control reserve and keep a
/// cancel or a health request out for their read time. Readers that hold a control request do
/// take it: with eight of them the accept loop stops.
#[test]
fn silent_sockets_do_not_take_the_control_reserve() {
    let _sockets = sockets();
    use std::io::Write;
    let limits = super::Limits::default();
    let full_table = || {
        let (listener, mut clients) = connected(17);
        let address = listener.local_addr().unwrap();
        for client in &mut clients {
            write!(client, "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: 8\r\n\r\n").unwrap();
        }
        let access = super::http::AccessPolicy::new(address, &limits).unwrap();
        let mut reading = Vec::new();
        assert_eq!(accept_all(&listener, &mut reading, &limits), 17);
        assert!(pass(&mut reading, &limits, &access).is_empty());
        (listener, clients, address, access, reading)
    };
    // The table is full. Eight silent peers arrive: as many as the reserve has places.
    let (listener, clients, address, access, mut reading) = full_table();
    let silent: Vec<_> = (0..8).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 8);
    assert_eq!(reading.iter().filter(|input| input.wire.control_only).count(), 8);
    // A health request arrives. It is accepted at once and read; no silent peer had to leave for it.
    let _probe = request(address, "GET /health HTTP/1.1\r\nHost: {host}\r\n\r\n");
    assert_eq!(accept_all(&listener, &mut reading, &limits), 1, "silent sockets kept a control request out of the reserve");
    let seen = pass(&mut reading, &limits, &access);
    assert_eq!(seen.iter().filter_map(|request| request.as_ref().ok()).map(|request| request.path.as_str()).collect::<Vec<_>>(), ["/health"]);
    assert_eq!(reading.len(), 25);
    // A flood of silent peers is bounded by the pool of 64, and no request that passed the check loses its slot.
    // 58 more make 66 silent peers, 2 over the pool (not more: about 165 sockets are open here; see `sockets`).
    let flood: Vec<_> = (0..58).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 58);
    assert_eq!(reading.len(), 17 + 64);
    assert_eq!(reading.iter().filter(|input| input.head.is_some()).count(), 17, "a request that passed the check lost its slot");
    drop((silent, flood, clients));
    // Eight peers that each sent a cancel head and wait with its body hold the reserve: the next peer waits, and none is pushed out.
    let (listener, clients, address, access, mut reading) = full_table();
    let cancels: Vec<_> = (0..8).map(|_| request(address, "POST /v1/cancel HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: 8\r\n\r\n")).collect();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 8);
    // This part is about the places, not about the short read time of a reserve reader (that time is
    // asserted in `requests_that_passed_the_check_hold_the_slots`). Give the readers a long time, and
    // wait until each holds its head, so a stall of the test thread cannot end them first.
    for input in reading.iter_mut().filter(|input| input.wire.control_only) { input.deadline = super::Instant::now() + super::Duration::from_secs(120); }
    let held = |reading: &[super::Reading]| reading.iter().filter(|input| input.wire.control_only && input.head.is_some()).count();
    assert!(pass_until(&mut reading, &limits, &access, |reading, _| held(reading) == 8).is_empty());
    assert_eq!(held(&reading), 8);
    let _late = std::net::TcpStream::connect(address).unwrap();
    assert_eq!(accept_all(&listener, &mut reading, &limits), 0, "the reserve was full of control requests and one more reader got in");
    assert_eq!(reading.len(), 25);
    drop((cancels, clients));
}

/// A client with the key whose head arrives in two parts. The first read gets a part; the rest,
/// with the key, then waits in the socket. A peer with no key opens 64 newer sockets, and the
/// scheduler takes them in two accept batches with no reader pass between them (the end and the
/// start of a turn). The keyed readers are the oldest in the pool. They must not be pushed out:
/// before a member leaves, the intake reads what has arrived, and the complete heads pass the
/// key check. Eight more sockets then find the pool full and push out only readers with no key.
#[test]
fn a_keyed_head_that_has_arrived_is_read_before_a_full_pool_pushes_anyone_out() {
    let _sockets = sockets();
    use std::io::Write;
    let limits = keyed("flood");
    let (listener, _) = connected(0);
    let address = listener.local_addr().unwrap();
    let access = super::http::AccessPolicy::new(address, &limits).unwrap();
    // Accept calls of at most 32, as the scheduler makes them, until `count` connections are in; no reader pass
    // runs between them. A connection can reach the listen queue a moment after `connect` returns, so this waits
    // for the count (30 second bound), not for a time.
    let accept = |reading: &mut Vec<super::Reading>, count: usize| {
        let (mut total, until) = (0, super::Instant::now() + super::Duration::from_secs(30));
        while total < count && super::Instant::now() < until {
            match super::accept_connections(&listener, reading, &limits, &access, 17, 32) { 0 => std::thread::sleep(super::Duration::from_millis(1)), taken => total += taken }
        }
        total
    };
    let key = "Authorization: Bearer 0123456789abcdef\r\n";
    let mut health = request(address, "GET /health HTTP/1.1\r\nHost: {host}\r\n");
    let mut cancel = request(address, "POST /v1/cancel HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n");
    let mut reading = Vec::new();
    assert_eq!(accept(&mut reading, 2), 2);
    // The first read of each gets the part that was sent: no head yet.
    assert!(pass_until(&mut reading, &limits, &access, |reading, _| reading.iter().all(|input| !input.data.is_empty())).is_empty());
    assert!(reading.len() == 2 && reading.iter().all(|input| input.head.is_none()));
    let rest = [format!("{key}\r\n"), format!("{key}\r\n{{}}")];
    health.write_all(rest[0].as_bytes()).unwrap();
    cancel.write_all(rest[1].as_bytes()).unwrap();
    // The flood starts only when the rest of each head, with its key, is readable on the server's side of its socket.
    // The test looks (a peek takes nothing out) until both are there, for at most 30 seconds.
    let waiting = |reading: &[super::Reading]| {
        let mut sizes: Vec<_> = reading.iter().map(|input| { let mut bytes = [0; 256]; match &input.wire.socket { super::super::Socket::Tcp(socket) => socket.peek(&mut bytes).unwrap_or(0), _ => 0 } }).collect();
        sizes.sort_unstable();
        sizes
    };
    let until = super::Instant::now() + super::Duration::from_secs(30);
    while waiting(&reading) != [rest[0].len(), rest[1].len()] && super::Instant::now() < until { std::thread::sleep(super::Duration::from_millis(1)); }
    assert_eq!(waiting(&reading), [rest[0].len(), rest[1].len()], "the rest of the two heads did not reach the server's sockets");
    // 64 newer sockets with no key, taken with no reader pass between the accept calls.
    let flood: Vec<_> = (0..64).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
    assert_eq!(accept(&mut reading, 64), 64);
    assert_eq!(reading.iter().filter(|input| input.head.is_some()).count(), 2, "a full pool pushed out a reader whose head with the key had arrived");
    // 8 more: the pool is full, and they push out its 8 oldest members with no key. The keyed readers stay.
    // (8, not a whole batch: about 140 sockets are open here; see `sockets`.)
    let more: Vec<_> = (0..8).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
    assert_eq!(accept(&mut reading, 8), 8);
    assert_eq!((reading.len(), reading.iter().filter(|input| input.head.is_some()).count()), (66, 2));
    // The next reader pass gives both requests to the dispatcher, each with its key accepted.
    let seen = pass_until(&mut reading, &limits, &access, |_, seen| seen.len() == 2);
    let mut paths: Vec<_> = seen.iter().map(|request| request.as_ref().map(|request| (request.path.as_str(), request.authorization.is_some())).unwrap()).collect();
    paths.sort_unstable();
    assert_eq!(paths, [("/health", true), ("/v1/cancel", true)]);
    assert_eq!(reading.len(), 64);
    drop((flood, more, health, cancel));
}

/// The pool of 64 fills on a real serve loop: the loop that answers while an engine loads, with its
/// own accept calls, reads and replies. 68 sockets that send nothing stay open: 4 more than the pool,
/// so the loop must push readers out. Each request with the key still gets its whole reply.
/// 68, not more: with the server's ends about 135 sockets are open, and a test program must pass at
/// the default limit of 256 open files (see `sockets`). The scheduler test file has no such case: its
/// 60 second overload test holds about 100 sockets the whole time, which leaves no room there.
#[test]
fn a_full_pool_of_silent_sockets_keeps_no_keyed_request_out() {
    use std::io::Read;
    let _sockets = sockets();
    let limits = keyed("pool");
    let (listener, _) = connected(0);
    let address = listener.local_addr().unwrap();
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let server = scope.spawn(|| super::super::unavailable(&listener, &limits, "m", &stop, None, "loading", None, None).map(|_| ()));
        let silent: Vec<_> = (0..68).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
        for round in 0..20 {
            let mut client = request(address, "GET /health HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer 0123456789abcdef\r\n\r\n");
            client.set_read_timeout(Some(super::Duration::from_secs(30))).unwrap();
            let mut reply = String::new();
            client.read_to_string(&mut reply).unwrap();
            assert!(reply.starts_with("HTTP/1.1 503 ") && reply.contains("{\"status\":\"loading\",\"model\":\"m\",\"active\":0,") && reply.ends_with('}'), "round {round}: {reply}");
        }
        // The pool was full: the loop closed at least the 4 silent sockets over 64. Wait for that state (30 second bound).
        for socket in &silent { socket.set_nonblocking(true).unwrap(); }
        let pushed_out = |silent: &[std::net::TcpStream]| silent.iter().filter(|socket| { let mut socket: &std::net::TcpStream = socket; match socket.read(&mut [0]) { Ok(0) => true, Err(error) => error.kind() != std::io::ErrorKind::WouldBlock, Ok(_) => false } }).count();
        let until = super::Instant::now() + super::Duration::from_secs(30);
        while pushed_out(&silent) < 4 && super::Instant::now() < until { std::thread::sleep(super::Duration::from_millis(2)); }
        assert!(pushed_out(&silent) >= 4, "68 silent sockets did not fill the pool of 64");
        stop.store(true, std::sync::atomic::Ordering::Release);
        server.join().unwrap().unwrap();
    });
}

fn poisoned_bad_request_reply() -> Vec<u8> {
    // Keep the recorded reply. API compatibility adds only these nullable fields.
    let (head, body) = include_str!("../../tests/fixtures/poisoned-keyed-bad-request.main-1f5ee64f55.http").split_once("\r\n\r\n").unwrap();
    let updated = format!("{},\"param\":null,\"code\":null}}}}", body.strip_suffix("}}").unwrap());
    let head = head.replace(&format!("Content-Length: {}", body.len()), &format!("Content-Length: {}", updated.len()));
    format!("{head}\r\n\r\n{updated}").into_bytes()
}

/// A request that shows the key and then fails a later check of its head (here: no Content-Type)
/// has no head, but it passed the key check. A full pool must not push it out: it gets its
/// reply, and on a poisoned engine that reply adds only nullable error details to main's.
#[test]
fn a_keyed_request_with_a_bad_head_keeps_its_place_in_a_full_pool_and_gets_mains_reply() {
    use std::io::{Read, Write};
    let _sockets = sockets();
    let limits = keyed("badhead");
    let parts = ["POST /v1/completions HTTP/1.1\r\nHost: {host}\r\nContent-Length: 2\r\n", "Authorization: Bearer 0123456789abcdef\r\n\r\n{}"];
    // Part 1, with the accept calls and the reader passes in the test's hands: this part always fails without the fix.
    {
        let (listener, _) = connected(0);
        let address = listener.local_addr().unwrap();
        let access = super::http::AccessPolicy::new(address, &limits).unwrap();
        let accept = |reading: &mut Vec<super::Reading>, count: usize| {
            let (mut total, until) = (0, super::Instant::now() + super::Duration::from_secs(30));
            while total < count && super::Instant::now() < until {
                match super::accept_connections(&listener, reading, &limits, &access, 17, 32) { 0 => std::thread::sleep(super::Duration::from_millis(1)), taken => total += taken }
            }
            total
        };
        // The request comes in two parts. The first reader pass gets the first one: no head, no key yet.
        let mut client = request(address, parts[0]);
        let mut reading = Vec::new();
        assert_eq!(accept(&mut reading, 1), 1);
        assert!(pass_until(&mut reading, &limits, &access, |reading, _| !reading[0].data.is_empty()).is_empty());
        client.write_all(parts[1].as_bytes()).unwrap();
        // The second part, with the key, is readable on the server's side before the flood starts (a peek, 30 second bound).
        let arrived = |reading: &[super::Reading]| { let mut bytes = [0; 256]; matches!(&reading[0].wire.socket, super::super::Socket::Tcp(socket) if socket.peek(&mut bytes).unwrap_or(0) == parts[1].len()) };
        let until = super::Instant::now() + super::Duration::from_secs(30);
        while !arrived(&reading) && super::Instant::now() < until { std::thread::sleep(super::Duration::from_millis(1)); }
        assert!(arrived(&reading), "the second part did not reach the server's socket");
        // 64 newer silent sockets fill the pool with no reader pass between the accept calls; 8 more push members out.
        let flood: Vec<_> = (0..72).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
        assert_eq!(accept(&mut reading, 72), 72);
        // The keyed request is still there, outside the pool, with its error kept for the dispatcher.
        assert_eq!(reading.iter().filter(|input| input.wire.keyed).count(), 1, "a full pool pushed out a request that had shown the key");
        assert_eq!(reading.len(), 65);
        let seen = pass_until(&mut reading, &limits, &access, |_, seen| !seen.is_empty());
        assert_eq!(seen.iter().map(|request| request.as_ref().map(|_| ()).map_err(String::as_str)).collect::<Vec<_>>(), [Err("Content-Type must be application/json")]);
        assert_eq!(reading.iter().filter(|input| input.wire.keyed && input.responding).count(), 1);
        // 8 more silent sockets: its refusal drains, and still only members with no key are pushed out.
        let more: Vec<_> = (0..8).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
        assert_eq!(accept(&mut reading, 8), 8);
        assert_eq!(reading.iter().filter(|input| input.wire.keyed && input.responding).count(), 1);
        drop((flood, more, client));
    }
    // Part 2, on the real loop of a poisoned engine: the same request in two parts, the oldest member of a full pool of
    // 64, and newer sockets right behind its second part. The reply is main's with nullable error details added
    // (the fixture records main `1f5ee64f55`). The 100 ms wait only lets the loop take the 64 in first, which makes the
    // case sharper; what is asserted does not depend on it.
    let (listener, _) = connected(0);
    let address = listener.local_addr().unwrap();
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let server = scope.spawn(|| super::super::unavailable(&listener, &limits, "m", &stop, None, "poisoned", None, Some("GPU fault in /private/models/m")).map(|_| ()));
        let mut client = request(address, parts[0]);
        let silent: Vec<_> = (0..63).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
        std::thread::sleep(super::Duration::from_millis(100));
        client.write_all(parts[1].as_bytes()).unwrap();
        let newer: Vec<_> = (0..8).map(|_| std::net::TcpStream::connect(address).unwrap()).collect();
        client.set_read_timeout(Some(super::Duration::from_secs(30))).unwrap();
        let mut reply = Vec::new();
        let _ = client.read_to_end(&mut reply);
        stop.store(true, std::sync::atomic::Ordering::Release);
        server.join().unwrap().unwrap();
        assert_eq!(reply, poisoned_bad_request_reply(), "{}", String::from_utf8_lossy(&reply));
        drop((silent, newer));
    });
}

#[test]
fn the_private_admin_socket_is_not_in_the_pool() {
    let _sockets = sockets();
    let limits = super::Limits::default();
    let (listener, silent) = connected(72);
    let (admin, _peer) = std::os::unix::net::UnixStream::pair().unwrap();
    let mut reading = vec![super::Reading::new(super::super::Socket::Admin(admin), &limits).unwrap()];
    // 72 silent peers, 8 over the pool of 64, push each other out. The admin reader, the oldest one, stays.
    assert_eq!(accept_all(&listener, &mut reading, &limits), 72);
    assert_eq!(reading.len(), 65);
    assert_eq!(reading.iter().filter(|input| input.wire.socket.is_admin()).count(), 1);
    drop(silent);
}


#[test]
fn a_failed_engine_gives_its_failure_reason_only_to_a_request_with_the_key() {
    let _sockets = sockets();
    use std::io::Read;
    let limits = keyed("failure");
    let (listener, _) = connected(0);
    let address = listener.local_addr().unwrap();
    let stop = std::sync::atomic::AtomicBool::new(false);
    // The server closes its side after each reply, so each read ends.
    let reply = |head: String| {
        let mut client = request(address, &head);
        client.set_read_timeout(Some(super::Duration::from_secs(10))).unwrap();
        let mut text = String::new();
        let _ = client.read_to_string(&mut text);
        text
    };
    let key = "Authorization: Bearer 0123456789abcdef\r\n";
    let (open, wrong, bad, old, health, keyed) = std::thread::scope(|scope| {
        let server = scope.spawn(|| super::super::unavailable(&listener, &limits, "m", &stop, None, "poisoned", None, Some("GPU fault in /private/models/m")).map(|_| ()));
        let replies = (
            reply(format!("{HEAD}\r\n{{}}")),
            reply(format!("{HEAD}Authorization: Bearer 0123456789abcdeX\r\n\r\n{{}}")),
            // A request with the key that ends with an error (here: no Content-Type) gets main's answer, reason and all.
            reply(format!("POST /v1/completions HTTP/1.1\r\nHost: {{host}}\r\nContent-Length: 2\r\n{key}\r\n{{}}")),
            // A request with no key that ends with an error before the key check gets no reason.
            reply("POST /v1/completions HTTP/1.0\r\nHost: {host}\r\nContent-Length: 2\r\n\r\n{}".into()),
            reply("GET /health HTTP/1.1\r\nHost: {host}\r\n\r\n".into()),
            reply(format!("{HEAD}{key}\r\n{{}}")),
        );
        stop.store(true, std::sync::atomic::Ordering::Release);
        server.join().unwrap().unwrap();
        replies
    });
    // A client with the key gets main's recorded answer plus the nullable error details.
    assert_eq!(bad.as_bytes(), poisoned_bad_request_reply(), "{bad}");
    for (name, reply, status) in [("no key", &open, "401"), ("wrong key", &wrong, "401"), ("bad request with no key", &old, "400"), ("health with no key", &health, "503")] {
        assert!(reply.starts_with(&format!("HTTP/1.1 {status} ")), "{name}: {reply}");
        assert!(!reply.contains("reason") && !reply.contains("GPU fault"), "{name}: {reply}");
    }
    assert!(health.ends_with("\r\n\r\n{\"status\":\"poisoned\"}"), "{health}");
    // A request that showed the key reads why the engine does not serve.
    assert!(keyed.starts_with("HTTP/1.1 503 ") && keyed.contains("\"reason\":\"GPU fault in /private/models/m\""), "{keyed}");
}
