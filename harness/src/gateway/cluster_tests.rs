//! Three independent, first-party loopback endpoints with mock prefix caches.
use super::*;
use std::{
    collections::{HashSet, VecDeque},
    sync::{Mutex, mpsc},
    time::Instant,
};
const CAPS: &str = include_str!("../../tests/fixtures/engine-capabilities-v1.json");
const CHAT: &str = "/v1/chat/completions";

enum Reply {
    Normal,
    Status(u16),
    Drop,
    Truncated,
    Raw(&'static str, Vec<u8>),
    Wait(mpsc::Sender<()>, mpsc::Receiver<()>),
}
struct Seen {
    path: String,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
}
struct MockState {
    capabilities: Option<String>,
    replies: VecDeque<Reply>,
    requests: Vec<Seen>,
    prefixes: HashSet<String>,
    probes: usize,
}
struct Mock {
    url: String,
    state: Arc<Mutex<MockState>>,
    cancel: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn start(index: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(MockState {
            capabilities: Some(CAPS.replace("mock-load-epoch", &format!("mock-epoch-{index}"))),
            replies: VecDeque::new(),
            requests: Vec::new(),
            prefixes: HashSet::new(),
            probes: 0,
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let running = cancel.clone();
        let shared = state.clone();
        let worker = thread::spawn(move || {
            let mut clients = Vec::new();
            while !running.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((socket, _)) => {
                        let state = shared.clone();
                        let cancel = running.clone();
                        clients.push(thread::spawn(move || {
                            let _ = mock_request(socket, &state, &cancel);
                        }));
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1))
                    }
                    Err(e) => panic!("mock accept: {e}"),
                }
            }
            for client in clients {
                client.join().unwrap();
            }
        });
        Self {
            url,
            state,
            cancel,
            worker: Some(worker),
        }
    }
    fn reply(&self, reply: Reply) {
        self.state.lock().unwrap().replies.push_back(reply);
    }
    fn count(&self) -> usize {
        self.state.lock().unwrap().requests.len()
    }
    fn stop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.stop();
    }
}
fn mock_request(
    socket: TcpStream,
    shared: &Mutex<MockState>,
    cancel: &AtomicBool,
) -> io::Result<()> {
    let mut reader = BufReader::new(DeadlineStream::new(socket, Some(cancel))?);
    let head = transport::read_head(&mut reader)?;
    let path = head.first.split(' ').nth(1).unwrap_or("").to_owned();
    let mut incoming = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES)?;
    let mut body = Vec::new();
    incoming.read_to_end(&mut body)?;
    let writer = incoming.reader.get_mut();
    if head.first.starts_with("GET ") {
        assert!(path.ends_with("/capabilities"));
        assert!(
            head.get("authorization").is_none(),
            "background probes carry no client credentials"
        );
        let caps = {
            let mut state = shared.lock().unwrap();
            state.probes += 1;
            state.capabilities.clone()
        };
        return respond(
            writer,
            if caps.is_some() { 200 } else { 503 },
            "application/json",
            caps.as_deref().unwrap_or("{}").as_bytes(),
        );
    }
    let request = json::view(std::str::from_utf8(&body).unwrap()).unwrap();
    let key = request
        .get("prompt_cache_key")
        .and_then(json::View::as_str)
        .map(|s| s.into_owned())
        .unwrap_or_else(|| {
            request
                .get("messages")
                .or_else(|| request.get("input"))
                .and_then(json::View::array)
                .and_then(|mut a| {
                    a.find(|v| {
                        v.get("role")
                            .and_then(json::View::as_str)
                            .is_some_and(|r| r == "user")
                    })
                })
                .map(|v| v.raw().to_owned())
                .unwrap_or_else(|| "mock-empty-prefix".into())
        });
    let (reply, cached) = {
        let mut state = shared.lock().unwrap();
        let cached = if state.prefixes.contains(&key) { 80 } else { 0 };
        let reply = state.replies.pop_front().unwrap_or(Reply::Normal);
        if matches!(reply, Reply::Normal | Reply::Wait(..)) {
            state.prefixes.insert(key);
        }
        state.requests.push(Seen {
            path: path.clone(),
            body: body.clone(),
            headers: head.fields,
        });
        (reply, cached)
    };
    match reply {
        Reply::Drop => return Ok(()),
        Reply::Status(code) => {
            return respond(
                writer,
                code,
                "application/json",
                b"{ \"error\": \"mock overload\", \"unknown\": [1,2] }",
            );
        }
        Reply::Truncated => {
            writer.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{\"usage\":{\"prompt_tokens\":100,\"prompt_tokens_details\":{\"cached_tokens\":99}}}")?;
            return writer.flush();
        }
        Reply::Raw(media, bytes) => return respond(writer, 200, media, &bytes),
        Reply::Wait(started, resume) => {
            started.send(()).unwrap();
            resume.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        Reply::Normal => {}
    }
    let responses = path.starts_with("/v1/responses") || path.starts_with("/responses");
    let usage = if responses {
        format!(
            r#"{{"input_tokens":100,"output_tokens":1,"input_tokens_details":{{"cached_tokens":{cached}}}}}"#
        )
    } else {
        format!(
            r#"{{"prompt_tokens":100,"completion_tokens":1,"prompt_tokens_details":{{"cached_tokens":{cached}}}}}"#
        )
    };
    let stream = request.get("stream").is_some_and(|v| v.raw() == "true");
    if stream {
        let bytes = if responses {
            format!(
                "event: response.completed\r\ndata: {{\"type\":\"response.completed\",\"response\":{{\"usage\":{usage}}}}}\r\n\r\n"
            )
        } else {
            format!("data: {{\"choices\":[],\"usage\":{usage}}}\n\ndata: [DONE]\n\n")
        };
        // Deliberately split UTF-8/event lines across HTTP chunks.
        writer.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
        for chunk in bytes.as_bytes().chunks(3) {
            write!(writer, "{:x}\r\n", chunk.len())?;
            writer.write_all(chunk)?;
            writer.write_all(b"\r\n")?;
        }
        writer.write_all(b"0\r\n\r\n")?;
        writer.flush()
    } else {
        respond(
            writer,
            200,
            "application/json",
            format!("{{\"usage\":{usage}}}").as_bytes(),
        )
    }
}
fn respond(writer: &mut impl Write, status: u16, media: &str, bytes: &[u8]) -> io::Result<()> {
    write!(
        writer,
        "HTTP/1.1 {status} Mock\r\nContent-Type: {media}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    )?;
    writer.write_all(bytes)?;
    writer.flush()
}
fn setup(config: RoutingConfig) -> ([Mock; 3], Cluster) {
    let mocks = [Mock::start(0), Mock::start(1), Mock::start(2)];
    let urls = mocks.iter().map(|m| m.url.clone()).collect::<Vec<_>>();
    let cluster = Cluster::new(&urls, config).unwrap();
    for i in 0..3 {
        cluster.refresh(i, &AtomicBool::new(false));
    }
    assert!(cluster.stats().endpoints.iter().all(|e| e.healthy));
    (mocks, cluster)
}
struct Exchange {
    status: u16,
    body: Vec<u8>,
    complete: bool,
}
fn exchange_at(address: std::net::SocketAddr, path: &str, body: &[u8]) -> Exchange {
    let socket = TcpStream::connect(address).unwrap();
    let mut socket = DeadlineStream::new(socket, None).unwrap();
    let method = if body.is_empty() { "GET" } else { "POST" };
    write!(socket, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nX-Mock-Forwarded: exact\r\n\r\n", body.len()).unwrap();
    socket.write_all(body).unwrap();
    socket.flush().unwrap();
    let mut reader = BufReader::new(socket);
    let head = transport::read_head(&mut reader).unwrap();
    let status = head.first.split(' ').nth(1).unwrap().parse().unwrap();
    let mut incoming = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
    let mut body = Vec::new();
    let complete = incoming.read_to_end(&mut body).is_ok();
    Exchange {
        status,
        body,
        complete,
    }
}
fn exchange(cluster: &Cluster, path: &str, body: &[u8]) -> Exchange {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let cluster = cluster.clone();
    let worker = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        handle_target(
            socket,
            &Upstream::Cluster(cluster),
            None,
            &ConnectionPool::default(),
            &Access::default(),
        )
    });
    let response = exchange_at(address, path, body);
    let _ = worker.join().unwrap(); // Truncation intentionally yields a relay error.
    response
}
fn request(key: Option<&str>, conversation: &str, later: bool, stream: bool) -> Vec<u8> {
    format!(r#"{{ "temperature":0, "seed":42, "top_p":1, "model":"mock", "tools":[{{"type":"function","function":{{"name":"read"}}}}], {} "messages":[{{"role":"system","content":"fixed rules"}},{{"role":"user","content":{}}}{}], "stream":{}, "vendor":{{"number":1.00e+0}} }}"#,
        key.map_or(String::new(), |k| format!("\"prompt_cache_key\":{},", json::quote(k))), json::quote(conversation),
        if later { ",{\"role\":\"assistant\",\"content\":\"reply\"},{\"role\":\"user\",\"content\":\"continue\"}" } else { "" }, stream).into_bytes()
}
fn recipient(mocks: &[Mock; 3], body: &[u8]) -> usize {
    mocks
        .iter()
        .position(|m| {
            m.state
                .lock()
                .unwrap()
                .requests
                .iter()
                .any(|r| r.body == body)
        })
        .unwrap()
}

#[test]
fn three_cache_endpoints_preserve_bytes_and_reuse_keyed_and_inferred_prefixes() {
    let (mocks, cluster) = setup(RoutingConfig::default());
    for keyed in [true, false] {
        for n in 0..3 {
            let conversation = format!("mock-conversation-{keyed}-{n}");
            let key = keyed.then_some(conversation.as_str());
            let first = request(key, &conversation, false, n % 2 == 0);
            let next = request(key, &conversation, true, n % 2 == 0);
            let a = exchange(&cluster, CHAT, &first);
            let b = exchange(&cluster, CHAT, &next);
            assert_eq!(
                (a.status, b.status, a.complete, b.complete),
                (200, 200, true, true)
            );
            assert!(
                String::from_utf8(a.body)
                    .unwrap()
                    .contains("\"cached_tokens\":0")
            );
            assert!(
                String::from_utf8(b.body)
                    .unwrap()
                    .contains("\"cached_tokens\":80")
            );
            assert_eq!(recipient(&mocks, &first), recipient(&mocks, &next));
        }
    }
    for mock in &mocks {
        assert_eq!(mock.count(), 4);
        for seen in &mock.state.lock().unwrap().requests {
            assert_eq!(seen.path, CHAT);
            assert!(
                seen.headers
                    .iter()
                    .any(|(k, v)| k == "x-mock-forwarded" && v == "exact")
            );
        }
    }
    for stats in cluster.stats().endpoints {
        assert_eq!(
            (stats.requests, stats.failures, stats.usage_responses),
            (4, 0, 4)
        );
        assert_eq!(
            (
                stats.prompt_tokens,
                stats.cached_tokens,
                stats.cached_ratio()
            ),
            (400, 160, Some(0.4))
        );
    }
    let stats = exchange(&cluster, STATS_PATH, &[]);
    assert_eq!(stats.status, 200);
    let stats = std::str::from_utf8(&stats.body).unwrap();
    json::view(stats).unwrap();
    assert!(
        !stats.contains("conversation")
            && !stats.contains("127.0.0.1")
            && !stats.contains("fixed rules")
    );
    assert_eq!(mocks.iter().map(Mock::count).sum::<usize>(), 12);
}

#[test]
fn early_overload_retries_once_on_a_compatible_endpoint_with_identical_bytes() {
    for status in [429, 503, 529] {
        let (mocks, cluster) = setup(RoutingConfig::default());
        let first = request(Some("hot"), "first", false, false);
        assert_eq!(exchange(&cluster, CHAT, &first).status, 200);
        let owner = recipient(&mocks, &first);
        mocks[owner].reply(Reply::Status(status));
        let next = request(Some("hot"), "second", true, false);
        let response = exchange(&cluster, CHAT, &next);
        assert_eq!(response.status, 200);
        assert!(response.complete);
        assert!(
            !response
                .body
                .windows(b"mock overload".len())
                .any(|w| w == b"mock overload")
        );
        assert_eq!(
            mocks.iter().map(Mock::count).sum::<usize>(),
            3,
            "one seed request, then refusal and one fallback"
        );
        let recipients = mocks
            .iter()
            .filter(|m| {
                m.state
                    .lock()
                    .unwrap()
                    .requests
                    .iter()
                    .any(|r| r.body == next)
            })
            .count();
        assert_eq!(recipients, 2);
        let explicit = request(Some("hot"), "third", true, false);
        assert_eq!(exchange(&cluster, CHAT, &explicit).status, 200);
        assert_ne!(recipient(&mocks, &explicit), owner);
        assert_eq!(cluster.stats().endpoints[owner].failures, 1);
        assert_eq!(cluster.stats().endpoints[owner].overloads, 1);
    }
}

#[test]
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn fresh_connect_failure_retries_without_sending_to_the_failed_endpoint() {
    use std::os::fd::{AsRawFd, FromRawFd};
    unsafe extern "C" {
        fn socket(domain: i32, kind: i32, protocol: i32) -> i32;
        fn bind(fd: i32, address: *const u8, length: u32) -> i32;
    }
    // Own an ephemeral TCP port without listening; it cannot be reused by another test.
    let fd = unsafe { socket(2, 1, 0) }; // AF_INET, SOCK_STREAM on both supported hosts.
    assert!(fd >= 0, "{}", io::Error::last_os_error());
    // SAFETY: socket returned a fresh, owned TCP descriptor; the guard closes it.
    let refused = unsafe { TcpStream::from_raw_fd(fd) };
    let mut address = [0u8; 16]; // sockaddr_in; port zero asks the OS to allocate.
    #[cfg(target_os = "macos")]
    {
        address[0] = 16;
        address[1] = 2;
    }
    #[cfg(target_os = "linux")]
    address[..2].copy_from_slice(&2u16.to_ne_bytes());
    address[4..8].copy_from_slice(&std::net::Ipv4Addr::LOCALHOST.octets());
    // SAFETY: the live descriptor and complete sockaddr_in remain valid during bind.
    assert_eq!(unsafe { bind(refused.as_raw_fd(), address.as_ptr(), 16) }, 0);
    let address = refused.local_addr().unwrap();
    assert_eq!(
        TcpListener::bind(address).unwrap_err().kind(),
        io::ErrorKind::AddrInUse
    );
    let (mocks, mut cluster) = setup(RoutingConfig::default());
    cluster.replace_endpoint_for_test(0, Endpoint::parse(&format!("http://{address}")).unwrap());
    let body = request(Some("connect-refusal"), "first", false, true);
    let response = exchange(&cluster, CHAT, &body);
    assert_eq!(response.status, 200);
    assert!(response.complete);
    assert_eq!(mocks.iter().map(Mock::count).sum::<usize>(), 1);
    assert_ne!(recipient(&mocks, &body), 0);
    let failed = &cluster.stats().endpoints[0];
    assert_eq!(
        (failed.requests, failed.failures, failed.in_flight),
        (1, 1, 0)
    );
}

#[test]
fn second_refusal_is_forwarded_and_no_third_endpoint_is_tried() {
    for status in [429, 503, 529] {
        let (mocks, cluster) = setup(RoutingConfig::default());
        for mock in &mocks {
            mock.reply(Reply::Status(status));
        }
        let response = exchange(
            &cluster,
            CHAT,
            &request(Some("two-refusals"), "first", false, true),
        );
        assert_eq!(response.status, status);
        assert_eq!(
            response.body,
            b"{ \"error\": \"mock overload\", \"unknown\": [1,2] }"
        );
        assert_eq!(mocks.iter().map(Mock::count).sum::<usize>(), 2);
        assert!(cluster.stats().endpoints.iter().all(|e| e.in_flight == 0));
    }
    let mock = Mock::start(0);
    let cluster = Cluster::new(&[mock.url.clone()], RoutingConfig::default()).unwrap();
    cluster.refresh(0, &AtomicBool::new(false));
    mock.reply(Reply::Status(503));
    assert_eq!(
        exchange(&cluster, CHAT, &request(None, "alone", false, false)).status,
        503
    );
    assert_eq!(mock.count(), 1);
}

#[test]
fn uncertain_or_truncated_request_is_never_sent_to_another_endpoint() {
    for reply in [Reply::Drop, Reply::Truncated] {
        let (mocks, cluster) = setup(RoutingConfig::default());
        mocks[0].reply(reply);
        let first = request(Some("uncertain"), "first", false, false);
        let response = exchange(&cluster, CHAT, &first);
        assert!(response.status == 502 || !response.complete);
        assert_eq!(mocks.iter().map(Mock::count).sum::<usize>(), 1);
        assert_eq!(cluster.stats().endpoints[0].usage_responses, 0);
        assert_eq!(cluster.stats().endpoints[0].failures, 1);
        let next = request(Some("uncertain"), "next explicit prompt", true, false);
        assert_eq!(exchange(&cluster, CHAT, &next).status, 200);
        assert_ne!(recipient(&mocks, &next), 0);
        assert_eq!(cluster.stats().endpoints[0].in_flight, 0);
    }
}

#[test]
fn active_affine_endpoint_at_limit_falls_back_without_waiting_for_slow_stream() {
    let (mocks, cluster) = setup(RoutingConfig {
        endpoint_concurrency: 1,
        ..RoutingConfig::default()
    });
    let (started, ready) = mpsc::channel();
    let (resume, resumed) = mpsc::channel();
    mocks[0].reply(Reply::Wait(started, resumed));
    let slow = request(Some("concurrent"), "first", false, true);
    let owner = cluster.clone();
    let worker = thread::spawn(move || exchange(&owner, CHAT, &slow));
    ready.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(cluster.stats().endpoints[0].in_flight, 1);
    let next = request(Some("concurrent"), "second", true, true);
    let response = exchange(&cluster, CHAT, &next);
    assert_eq!(response.status, 200);
    assert_eq!(recipient(&mocks, &next), 1);
    resume.send(()).unwrap();
    assert!(worker.join().unwrap().complete);
    assert!(cluster.stats().endpoints.iter().all(|s| s.in_flight == 0));
}

#[test]
fn failed_discovery_and_changed_model_are_quarantined_and_recovery_reprobes() {
    let (mocks, cluster) = setup(RoutingConfig::default());
    let first = request(Some("sticky"), "first", false, false);
    assert_eq!(exchange(&cluster, CHAT, &first).status, 200);
    mocks[0].state.lock().unwrap().capabilities = None;
    cluster.refresh(0, &AtomicBool::new(false));
    let next = request(Some("sticky"), "next", true, false);
    assert_eq!(exchange(&cluster, CHAT, &next).status, 200);
    assert_ne!(recipient(&mocks, &next), 0);
    mocks[0].state.lock().unwrap().capabilities = Some(CAPS.to_owned());
    cluster.refresh(0, &AtomicBool::new(false));
    assert!(cluster.stats().endpoints[0].healthy);
    mocks[1].state.lock().unwrap().capabilities =
        Some(CAPS.replace("mock-weights", "changed-weights"));
    cluster.refresh(1, &AtomicBool::new(false));
    assert!(!cluster.stats().endpoints[1].healthy);
    mocks[2].state.lock().unwrap().capabilities =
        Some(CAPS.replace("vade.engine.v1", "vade.engine.v2"));
    cluster.refresh(2, &AtomicBool::new(false));
    assert!(!cluster.stats().endpoints[2].healthy);
    let last = request(Some("sticky"), "last", true, false);
    assert_eq!(exchange(&cluster, CHAT, &last).status, 200);
    assert_eq!(recipient(&mocks, &last), 0);
}

#[test]
fn responses_admission_and_opaque_unparseable_payloads_keep_original_bytes() {
    let (mocks, cluster) = setup(RoutingConfig::default());
    let body = br#"{ "model":"mock", "prompt_cache_key":"response-key", "instructions":"fixed", "input":[{"role":"user","content":"hello"}], "previous_response_id":"mock-id", "stream":true }"#;
    for _ in 0..2 {
        let response = exchange(&cluster, "/responses", body);
        assert_eq!(response.status, 200);
        assert!(response.complete);
    }
    assert_eq!(mocks[0].count(), 2);
    assert_eq!(cluster.stats().endpoints[0].cached_tokens, 80);
    let admission = format!(
        "{{\"contract_version\":\"vade.engine.v1\",\"endpoint\":\"/v1/responses\",\"output_tokens\":32,\"request\":{}}}",
        std::str::from_utf8(body).unwrap()
    );
    assert_eq!(
        exchange(&cluster, "/v1/admission", admission.as_bytes()).status,
        200
    );
    assert_eq!(mocks[0].count(), 3);
    assert_eq!(
        cluster.stats().endpoints[0].usage_responses,
        2,
        "admission never counts generation usage"
    );
    let opaque = b"event: future\ndata: not-json\n\n: comment\n\n\xff\0\n";
    mocks[0].reply(Reply::Raw("text/event-stream", opaque.to_vec()));
    let response = exchange(&cluster, "/v1/responses", body);
    assert_eq!(response.body, opaque);
    assert!(response.complete);
    assert_eq!(cluster.stats().endpoints[0].usage_responses, 2);
    assert_eq!(cluster.stats().endpoints[0].failures, 0);
}

#[test]
fn public_server_discovers_all_three_and_stops_its_probe_workers() {
    let mocks = [Mock::start(0), Mock::start(1), Mock::start(2)];
    let cluster = Cluster::new(
        &mocks.iter().map(|m| m.url.clone()).collect::<Vec<_>>(),
        RoutingConfig::default(),
    )
    .unwrap();
    // Reserve an ephemeral port; only this test's loopback mock can be contacted.
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let stop = Arc::new(AtomicBool::new(false));
    let running = stop.clone();
    let owner = cluster.clone();
    let worker = thread::spawn(move || run_cluster_until(&address.to_string(), owner, &running));
    let deadline = Instant::now() + Duration::from_secs(3);
    while !cluster.stats().endpoints.iter().all(|e| e.healthy) {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    let response = exchange_at(
        address,
        CHAT,
        &request(Some("server"), "first", false, false),
    );
    assert_eq!(response.status, 200);
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap().unwrap();
    assert_eq!(
        mocks
            .iter()
            .map(|m| m.state.lock().unwrap().probes)
            .sum::<usize>(),
        3
    );
}

#[test]
fn cancellation_joins_discovery_workers_while_upstream_has_sent_no_headers() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (accepted, ready) = mpsc::channel();
    let upstream = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut reader = BufReader::new(socket);
        let head = transport::read_head(&mut reader).unwrap();
        assert!(head.first.starts_with("GET /v1/capabilities "));
        accepted.send(()).unwrap();
        assert_eq!(reader.read(&mut [0; 1]).unwrap(), 0);
    });
    let cluster = Cluster::new(&[url], RoutingConfig::default()).unwrap();
    let cancellation = Arc::new(AtomicBool::new(false));
    let probes = ProbeWorkers::start(&cluster, &cancellation).unwrap();
    ready.recv_timeout(Duration::from_secs(3)).unwrap();
    let start = Instant::now();
    drop(probes);
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(cancellation.load(Ordering::Relaxed));
    assert_eq!(cluster.stats().endpoints[0].probe_failures, 1);
    upstream.join().unwrap();
}

#[test]
fn foreign_initial_endpoint_never_receives_requests_and_healthy_replicas_still_work() {
    let mocks = [Mock::start(0), Mock::start(1), Mock::start(2)];
    mocks[0].state.lock().unwrap().capabilities =
        Some(CAPS.replace("vade.engine.v1", "foreign.v1"));
    let cluster = Cluster::new(
        &mocks.iter().map(|m| m.url.clone()).collect::<Vec<_>>(),
        RoutingConfig::default(),
    )
    .unwrap();
    for i in 0..3 {
        cluster.refresh(i, &AtomicBool::new(false));
    }
    for n in 0..4 {
        let body = request(Some(&format!("client-{n}")), "first", false, false);
        assert_eq!(exchange(&cluster, CHAT, &body).status, 200);
    }
    assert_eq!(mocks[0].count(), 0);
    assert_eq!(mocks[1].count(), 2);
    assert_eq!(mocks[2].count(), 2);
    assert_eq!(cluster.stats().endpoints[0].group, None);
    assert_eq!(cluster.stats().endpoints[0].probe_failures, 1);
}
