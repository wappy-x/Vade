//! Local, first-party gateway conformance fixtures.
use super::*;
use std::sync::{Mutex, mpsc};

fn receive(reader: &mut BufReader<DeadlineStream<'_>>) -> (transport::Head, Vec<u8>) {
    let head = transport::read_head(reader).unwrap();
    let mut body = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
    let mut bytes = Vec::new();
    body.read_to_end(&mut bytes).unwrap();
    (head, bytes)
}

// Exercise the plain relay, standalone relay, and opt-in standalone policies.
fn call_gateway(
    endpoint: Endpoint, lane: u8, method: &str, path: &str, payload: &[u8], record_dir: Option<String>,
) -> (transport::Head, Vec<u8>) {
    let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = gateway.local_addr().unwrap();
    let relay = thread::spawn(move || {
        let socket = gateway.accept().unwrap().0;
        let result = if lane == 0 { handle(socket, &endpoint) } else {
            invisible::handle_conformance(socket, &format!("http://{}", endpoint.authority), lane == 2, record_dir.as_deref())
        };
        match result {
            Ok(()) => {},
            Err(e) if e.kind() == io::ErrorKind::ConnectionAborted => {},
            Err(e) => panic!("relay failed: {e}"),
        }
    });
    let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
    write!(socket, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", payload.len()).unwrap();
    socket.write_all(payload).unwrap();
    let mut reader = BufReader::new(socket);
    let head = transport::read_head(&mut reader).unwrap();
    let mut body = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
    let mut bytes = Vec::new();
    body.read_to_end(&mut bytes).unwrap();
    relay.join().unwrap();
    (head, bytes)
}

#[test]
fn identical_unsampled_requests_forward_byte_identical_bodies_on_both_paths() {
    for lane in 0..3 {
        for path in ["/v1/completions", "/completions", "/v1/chat/completions", "/chat/completions", "/v1/responses?client=1", "/responses", "/v1/messages"] {
            let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
            let server = thread::spawn(move || {
                let mut bodies = Vec::new();
                for _ in 0..2 {
                    let mut reader = BufReader::new(DeadlineStream::new(upstream.accept().unwrap().0, None).unwrap());
                    bodies.push(receive(&mut reader).1);
                    reader.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
                }
                assert_eq!(bodies[0], bodies[1]);
                let body = json::view(std::str::from_utf8(&bodies[0]).unwrap()).unwrap();
                assert_eq!(body.get("temperature").unwrap().raw(), "0");
                assert_eq!(body.get("seed").unwrap().raw(), "20260905");
                assert_eq!(body.get("top_p").unwrap().raw(), "1.0");
                assert_eq!(body.get("messages").unwrap().raw(), r#"[{"role":"user", "content":"exact\u0020雪"}]"#);
                assert_eq!(body.get("extension").unwrap().raw(), "1.20e0");
            });
            let payload = r#"{ "messages":[{"role":"user", "content":"exact\u0020雪"}], "extension":1.20e0 }"#.as_bytes();
            for _ in 0..2 {
                assert!(call_gateway(endpoint.clone(), lane, "POST", path, payload, None).0.first.starts_with("HTTP/1.1 200 "));
            }
            server.join().unwrap();
        }
    }
}

#[test]
fn capabilities_advertise_exact_sampling_defaults_on_both_paths() {
    for lane in 0..3 {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
        let payload = include_bytes!("../tests/fixtures/engine-capabilities-v1.json");
        let server = thread::spawn(move || {
            let mut reader = BufReader::new(DeadlineStream::new(upstream.accept().unwrap().0, None).unwrap());
            assert!(receive(&mut reader).1.is_empty());
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", payload.len()).unwrap();
            reader.get_mut().write_all(payload).unwrap();
        });
        let (head, bytes) = call_gateway(endpoint, lane, "GET", "/v1/capabilities?client=1", b"", None);
        assert_eq!(head.get("x-vade-sampling-defaults"), Some(sampling::DEFAULTS_JSON));
        assert_eq!(head.get("x-vade-sampling-defaults-when"), Some("no sampling fields present; otherwise unchanged (engine defaults)"));
        assert_eq!(head.get("x-vade-sampling-fields"), Some(sampling::SAMPLING_FIELDS));
        assert_eq!(bytes, payload);
        server.join().unwrap();
    }
}

#[test]
fn recorded_defaults_replay_the_effective_upstream_body() {
    let dir = std::env::temp_dir().join(format!("vade-replay-{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", upstream.local_addr().unwrap());
    let endpoint = Endpoint::parse(&url).unwrap();
    let server = thread::spawn(move || {
        let mut bodies = Vec::new();
        for _ in 0..2 {
            let mut reader = BufReader::new(DeadlineStream::new(upstream.accept().unwrap().0, None).unwrap());
            bodies.push(receive(&mut reader).1);
            reader.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        }
        assert_eq!(bodies[0], bodies[1]);
        bodies.remove(0)
    });
    let original = br#"{"input":"exact"}"#;
    call_gateway(endpoint, 2, "POST", "/v1/responses", original, Some(dir.to_str().unwrap().into()));
    let folder = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
    assert_eq!(std::fs::read(folder.join("request.body")).unwrap(), original);
    let effective = std::fs::read(folder.join("upstream-request.body")).unwrap();
    let value = json::view(std::str::from_utf8(&effective).unwrap()).unwrap();
    assert_eq!(value.get("seed").unwrap().raw(), "20260905");
    assert_eq!(value.get("temperature").unwrap().raw(), "0");
    assert_eq!(value.get("top_p").unwrap().raw(), "1.0");
    invisible::cli(vec!["replay".into(), dir.to_str().unwrap().into(), "--upstream".into(), url].into_iter()).unwrap();
    assert_eq!(server.join().unwrap(), effective);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn body_limits_reject_declared_and_cumulative_overflow_without_truncation() {
    for limit in [json::MAX_INPUT_BYTES, transport::MAX_BODY] {
        let head = transport::Head { first: String::new(), fields: vec![("content-length".into(), limit.to_string())] };
        assert!(BodyReader::new(io::Cursor::new(b""), &head, true, limit).is_ok());
        let over = transport::Head { fields: vec![("content-length".into(), (limit + 1).to_string())], ..head };
        let error = BodyReader::new(io::Cursor::new(b""), &over, true, limit).err().unwrap();
        assert!(transport::body_limit_exceeded(&error));
        assert_eq!(request_body_error(&error).0, 413);
        let overflow = transport::Head { first: String::new(), fields: vec![("content-length".into(), "9".repeat(100))] };
        assert!(transport::body_limit_exceeded(&BodyReader::new(io::Cursor::new(b""), &overflow, true, limit).err().unwrap()));
        let head = transport::Head { first: String::new(), fields: vec![("transfer-encoding".into(), "chunked".into())] };
        for wire in [format!("{:x}\r\n", limit + 1), format!("1\r\nx\r\n{limit:x}\r\n"), format!("{}\r\n", "f".repeat(100))] {
            let mut body = BodyReader::new(io::Cursor::new(wire), &head, true, limit).unwrap();
            let mut bytes = Vec::new();
            assert!(transport::body_limit_exceeded(&body.read_to_end(&mut bytes).unwrap_err()));
            assert!(bytes.len() <= 1);
        }
    }
    assert_eq!(json::MAX_INPUT_BYTES, 8 * 1024 * 1024);
    assert_eq!(transport::MAX_BODY, 64 * 1024 * 1024);
}

#[test]
fn oversized_bodies_return_413_on_every_route_without_upstream_dispatch() {
    for lane in 0..4 {
        let mut routes = vec![
            ("POST", "/v1/completions"), ("POST", "/completions"),
            ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"),
            ("POST", "/v1/responses"), ("POST", "/responses"),
            ("POST", "/v1/messages?beta=true"), ("POST", "/v1/messages/count_tokens"),
            ("POST", "/v1/admission"), ("GET", "/v1/capabilities"),
        ];
        if matches!(lane, 1 | 2) { routes.extend([("POST", "/custom"), ("POST", "/v1/models"), ("POST", "/messages")]); }
        if lane == 3 { routes.push(("GET", STATS_PATH)); }
        for (method, path) in routes {
            for framing in [
                format!("Content-Length: {}\r\n\r\n", json::MAX_INPUT_BYTES + 1),
                format!("Transfer-Encoding: chunked\r\n\r\n{:x}\r\n", json::MAX_INPUT_BYTES + 1),
                format!("Transfer-Encoding: chunked\r\n\r\n1\r\nx\r\n{:x}\r\n", json::MAX_INPUT_BYTES),
            ] {
                let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
                let url = format!("http://{}", upstream.local_addr().unwrap());
                let endpoint = Endpoint::parse(&url).unwrap();
                let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
                let address = gateway.local_addr().unwrap();
                let relay = thread::spawn(move || {
                    let socket = gateway.accept().unwrap().0;
                    if lane == 0 { handle(socket, &endpoint).unwrap() } else if lane == 3 {
                        let cluster = Cluster::new(&[url], RoutingConfig::default()).unwrap();
                        handle_target(socket, &Upstream::Cluster(cluster), None, &ConnectionPool::default(), &Access::default()).unwrap()
                    } else {
                        // vade-gateway reports its own 413 as an error, so the stats row is not a completion.
                        assert_eq!(invisible::handle_conformance(socket, &url, lane == 2, None).unwrap_err().to_string(), "answered locally");
                    }
                });
                let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
                write!(socket, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n{framing}").unwrap();
                let mut reader = BufReader::new(socket);
                let head = transport::read_head(&mut reader).unwrap();
                assert!(head.first.starts_with("HTTP/1.1 413 "), "{lane} {path}: {}", head.first);
                let mut body = BodyReader::new(reader, &head, false, 8192).unwrap();
                let mut bytes = String::new();
                body.read_to_string(&mut bytes).unwrap();
                assert!(bytes.contains("8388608 byte limit"));
                let value = json::view(&bytes).unwrap();
                assert!(value.get("error").unwrap().get("message").is_some());
                if path.starts_with("/v1/messages") || path == "/messages" {
                    assert_eq!(value.get("type").unwrap().as_str().as_deref(), Some("error"));
                    assert_eq!(value.get("error").unwrap().get("type").unwrap().as_str().as_deref(), Some("request_too_large"));
                }
                relay.join().unwrap();
                upstream.set_nonblocking(true).unwrap();
                assert_eq!(upstream.accept().unwrap_err().kind(), io::ErrorKind::WouldBlock);
            }
        }
    }
}

#[test]
fn full_size_requests_still_forward_all_bytes_and_sampling_defaults() {
    for lane in 0..3 {
        let mut original = br#"{"input":""#.to_vec();
        original.resize(json::MAX_INPUT_BYTES - 2, b'x');
        original.extend_from_slice(b"\"}");
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
        let expected = original.clone();
        let server = thread::spawn(move || {
            let mut reader = BufReader::new(DeadlineStream::new(upstream.accept().unwrap().0, None).unwrap());
            let head = transport::read_head(&mut reader).unwrap();
            let mut input = BodyReader::new(reader, &head, true, transport::MAX_BODY).unwrap();
            let mut bytes = Vec::new();
            input.read_to_end(&mut bytes).unwrap();
            assert!(bytes.starts_with(&expected[..expected.len() - 1]));
            let suffix = &bytes[expected.len() - 1..];
            if lane == 2 {
                let object = format!("{{{}", std::str::from_utf8(&suffix[1..]).unwrap());
                let fields = json::view(&object).unwrap();
                assert!(fields.get("prompt_cache_key").unwrap().as_str().unwrap().starts_with("vade-"));
                assert_eq!(fields.get("cache").unwrap().raw(), "true");
                assert_eq!(fields.get("temperature").unwrap().raw(), "0");
                assert_eq!(fields.get("seed").unwrap().raw(), "20260905");
                assert_eq!(fields.get("top_p").unwrap().raw(), "1.0");
            } else {
                assert_eq!(suffix, br#","temperature":0,"seed":20260905,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}"#);
            }
            input.reader.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        });
        assert!(call_gateway(endpoint, lane, "POST", "/v1/responses", &original, None).0.first.starts_with("HTTP/1.1 200 "));
        server.join().unwrap();
    }
}

#[test]
fn backpressure_stops_upstream_reads_and_failed_write_never_completes_stream() {
    struct Body {
        data: Vec<u8>,
        at: usize,
        reads: Arc<AtomicUsize>,
        consumed: Arc<AtomicUsize>,
    }
    impl Read for Body {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let bytes = self.fill_buf()?;
            let n = out.len().min(bytes.len());
            out[..n].copy_from_slice(&bytes[..n]);
            self.consume(n);
            Ok(n)
        }
    }
    impl BufRead for Body {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(&self.data[self.at..(self.at + 8192).min(self.data.len())])
        }
        fn consume(&mut self, n: usize) {
            self.at += n;
            self.consumed.fetch_add(n, Ordering::SeqCst);
        }
    }
    struct Writer {
        bytes: Arc<Mutex<Vec<u8>>>,
        blocked: mpsc::Sender<()>,
        resume: mpsc::Receiver<bool>,
    }
    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.blocked.send(()).unwrap();
            if self.resume.recv_timeout(Duration::from_secs(3)).unwrap() {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "mock downstream disconnected",
                ))
            }
        }
    }
    let reads = Arc::new(AtomicUsize::new(0));
    let consumed = Arc::new(AtomicUsize::new(0));
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let (blocked, waiting) = mpsc::channel();
    let (resume, resumed) = mpsc::channel();
    let mut body = Body {
        data: vec![b'x'; 1024 * 1024],
        at: 0,
        reads: reads.clone(),
        consumed: consumed.clone(),
    };
    let mut writer = Writer {
        bytes: bytes.clone(),
        blocked,
        resume: resumed,
    };
    let relay = thread::spawn(move || relay_body(&mut body, &mut writer));
    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    assert_eq!(consumed.load(Ordering::SeqCst), 0);
    resume.send(true).unwrap();
    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(reads.load(Ordering::SeqCst), 2);
    assert_eq!(consumed.load(Ordering::SeqCst), 8192);
    resume.send(false).unwrap();
    assert_eq!(
        relay.join().unwrap().unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    assert_eq!(reads.load(Ordering::SeqCst), 2);
    assert!(!bytes.lock().unwrap().ends_with(b"0\r\n\r\n"));
}

#[test]
fn opaque_contract_routes_unknown_fields_and_wrong_dialect_errors() {
    let cases: Vec<(&str, &str, &[u8], u16, &str, &[u8])> = vec![
        ("POST", "/v1/chat/completions", br#"{"temperature":0,"seed":42,"top_p":1}"#, 200, "application/json", b"{\"choices\":[],\"complete\":true}"),
        ("GET", "/v1/capabilities", b"", 200, "application/json", include_bytes!("../tests/fixtures/engine-capabilities-v1.json")),
        // Model-listing clients (Open WebUI, Continue, OpenAI SDKs) read this first.
        ("GET", "/v1/models?limit=1000", b"", 200, "application/json", br#"{"object":"list","data":[{"id":"mock","object":"model","context_length":32768}]}"#),
        ("GET", "/models", b"", 200, "application/json", br#"{"object":"list","data":[]}"#),
        ("POST", "/v1/admission", br#"{ "contract_version":"vade.engine.v1", "endpoint":"/v1/responses", "output_tokens":17, "request":{"input":[], "future":9.00100e-3, "prompt_cache_key":"caller-affinity"} }"#, 200, "application/json", br#"{ "object":"engine.admission", "admitted":false, "future_metadata": [null, "exact\u0020"] }"#),
        ("POST", "/v1/chat/completions", br#"{ "temperature":0,"seed":42,"top_p":1,"model":"unchanged", "input":[{"type":"future"}], "future":9007199254740993, "prompt_cache_key":"caller-only" }"#, 422, "application/json", br#"{ "error":{"message":"Responses body on chat endpoint", "vendor":42}, "future":1.00 }"#),
        ("POST", "/responses", br#"{"temperature":0,"seed":42,"top_p":1,"model":"unchanged","messages":[],"tools":null,"opaque":{"x":-0.00e9}}"#, 400, "application/octet-stream", b"not JSON\x00\xff\nprovider refusal\r\n"),
        // The relay does not reinterpret even an unfamiliar SSE dialect.
        ("POST", "/chat/completions", br#"{"temperature":0,"seed":42,"top_p":1,"messages":[],"stream":true,"reasoning":{"unknown":true}}"#, 200, "text/event-stream", b": heartbeat\r\nid: opaque-7\r\nevent: vendor.unknown\r\ndata: {\"error\":true}\r\ndata: non-json payload\r\n\r\nretry: 250\n\n"),
    ];
    for (method, path, payload, status, kind, output) in cases {
        let payload = payload.to_vec();
        let output = output.to_vec();
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint::parse(&format!(
            "http://{}/v1/responses",
            upstream.local_addr().unwrap()
        ))
        .unwrap();
        let expected = payload.clone();
        let reply = output.clone();
        let server = thread::spawn(move || {
            let (socket, _) = upstream.accept().unwrap();
            let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
            let (head, bytes) = receive(&mut reader);
            let route = if path.starts_with("/v1/") {
                path.to_owned()
            } else {
                format!("/v1{path}")
            };
            assert_eq!(head.first, format!("{method} {route} HTTP/1.1"));
            assert_eq!(
                head.get("authorization"),
                Some("Bearer mock-client-supplied")
            );
            assert_eq!(head.get("x-provider-extension"), Some("opaque"));
            assert_eq!(bytes, expected);
            let socket = reader.get_mut();
            write!(socket, "HTTP/1.1 {status} Mock\r\nContent-Type: {kind}\r\nTransfer-Encoding: chunked\r\nX-Provider-Error: opaque\r\n\r\n").unwrap();
            // Every byte can cross a transfer chunk boundary.
            for byte in reply {
                socket
                    .write_all(&[b'1', b'\r', b'\n', byte, b'\r', b'\n'])
                    .unwrap();
            }
            socket.write_all(b"0\r\n\r\n").unwrap();
        });
        let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = gateway.local_addr().unwrap();
        let relay = thread::spawn(move || {
            let (socket, _) = gateway.accept().unwrap();
            handle(socket, &endpoint).unwrap();
        });
        let sending = TcpStream::connect(address).unwrap();
        let mut socket = DeadlineStream::new(sending.try_clone().unwrap(), None).unwrap();
        write!(socket, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\nAuthorization: Bearer mock-client-supplied\r\nX-Provider-Extension: opaque\r\n\r\n", payload.len()).unwrap();
        socket.write_all(&payload).unwrap();
        let mut reader = BufReader::new(socket);
        let head = transport::read_head(&mut reader).unwrap();
        assert!(head.first.starts_with(&format!("HTTP/1.1 {status} ")), "{method} {path}: {}", head.first);
        assert_eq!(head.get("x-provider-error"), Some("opaque"));
        let mut body = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
        let mut actual = Vec::new();
        body.read_to_end(&mut actual).unwrap();
        assert_eq!(actual, output);
        relay.join().unwrap();
        server.join().unwrap();
    }
}

#[test]
fn truncated_upstream_is_observable_as_incomplete_downstream_framing() {
    for wire in [
        b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\npartial".as_slice(),
        b"HTTP/1.1 503 Error\r\nTransfer-Encoding: chunked\r\n\r\n7\r\npartial\r\n".as_slice(),
    ] {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint =
            Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
        let server = thread::spawn(move || {
            let (socket, _) = upstream.accept().unwrap();
            let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
            receive(&mut reader);
            reader.get_mut().write_all(wire).unwrap();
        });
        let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = gateway.local_addr().unwrap();
        let relay = thread::spawn(move || {
            let (socket, _) = gateway.accept().unwrap();
            assert!(handle(socket, &endpoint).is_err());
        });
        let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
        socket
            .write_all(
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
            )
            .unwrap();
        let mut reader = BufReader::new(socket);
        let head = transport::read_head(&mut reader).unwrap();
        let mut body = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
        let mut actual = Vec::new();
        assert_eq!(
            body.read_to_end(&mut actual).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(actual, b"partial");
        relay.join().unwrap();
        server.join().unwrap();
    }
}

#[test]
fn uncertain_post_on_reused_connection_is_never_replayed() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
    let server = thread::spawn(move || {
        let (socket, _) = upstream.accept().unwrap();
        let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
        for i in 0..2 {
            let (_, bytes) = receive(&mut reader);
            assert_eq!(bytes, br#"{"temperature":0,"seed":20260905,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}"#);
            if i == 0 {
                reader
                    .get_mut()
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                    .unwrap();
            }
            // Second POST was completely accepted, then its outcome is lost.
        }
        upstream
    });
    let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = gateway.local_addr().unwrap();
    let relay = thread::spawn(move || {
        let pool = ConnectionPool::default();
        for _ in 0..2 {
            let (socket, _) = gateway.accept().unwrap();
            // The client closes as soon as it has its complete reply; on a slow host that close can
            // reach the relay before it returns, which is not a relay failure (flaky on CI otherwise).
            match handle_pooled(socket, &endpoint, None, &pool) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::ConnectionAborted => {}
                Err(error) => panic!("relay failed: {error}"),
            }
        }
    });
    for status in [200, 502] {
        let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
        socket
            .write_all(
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
            )
            .unwrap();
        let mut reader = BufReader::new(socket);
        let head = transport::read_head(&mut reader).unwrap();
        assert!(head.first.starts_with(&format!("HTTP/1.1 {status} ")));
        let mut body = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
        let mut bytes = Vec::new();
        body.read_to_end(&mut bytes).unwrap();
        if status == 502 {
            assert!(
                String::from_utf8(bytes)
                    .unwrap()
                    .contains("outcome may be unknown; request was not replayed")
            );
        }
    }
    relay.join().unwrap();
    let listener = server.join().unwrap();
    listener.set_nonblocking(true).unwrap();
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
}

#[test]
fn downstream_disconnect_closes_active_upstream_response() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
    let (release, released) = mpsc::channel();
    let server = thread::spawn(move || {
        let (socket, _) = upstream.accept().unwrap();
        let mut peer = socket.try_clone().unwrap();
        let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
        receive(&mut reader);
        // Configure before disconnect: Darwin can reject socket options after
        // a reset. Observe closure with recv, without DeadlineStream's options.
        peer.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let socket = reader.get_mut();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 67108864\r\n\r\n")
            .unwrap();
        socket.write_all(&[b'x'; 65536]).unwrap();
        released.recv_timeout(Duration::from_secs(30)).unwrap();
        // A gateway that saw the hang-up first sends one blank line, then closes.
        let mut after = Vec::new();
        if let Err(e) = peer.read_to_end(&mut after) {
            assert_eq!(e.kind(), io::ErrorKind::ConnectionReset);
        }
        assert!(after.is_empty() || after == b"\r\n", "{after:?}");
    });
    let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = gateway.local_addr().unwrap();
    let relay = thread::spawn(move || {
        let (socket, _) = gateway.accept().unwrap();
        assert!(handle(socket, &endpoint).is_err());
    });
    let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
    socket
        .write_all(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}")
        .unwrap();
    let mut reader = BufReader::new(socket);
    transport::read_head(&mut reader).unwrap();
    reader.get_mut().write_all(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}").unwrap();
    reader.fill_buf().unwrap();
    drop(reader);
    release.send(()).unwrap();
    relay.join().unwrap();
    server.join().unwrap();
}

#[test]
fn gateway_errors_carry_a_type_in_both_dialects() {
    for (status, openai, anthropic) in [
        (400, "invalid_request_error", "invalid_request_error"), (401, "authentication_error", "authentication_error"),
        (403, "permission_error", "permission_error"), (404, "not_found_error", "not_found_error"),
        (413, "request_too_large", "request_too_large"), (429, "rate_limit_error", "rate_limit_error"),
        (501, "invalid_request_error", "invalid_request_error"), (502, "server_error", "api_error"), (503, "overloaded", "overloaded_error"),
    ] {
        let code = match status { 401 => "\"invalid_api_key\"", 501 => "\"unsupported_endpoint\"", _ => "null" };
        assert_eq!(error_body(false, status, "m\"x"), format!(r#"{{"error":{{"type":"{openai}","message":"m\"x","param":null,"code":{code}}}}}"#));
        assert_eq!(error_body(true, status, "m"), format!(r#"{{"type":"error","error":{{"type":"{anthropic}","message":"m"}}}}"#));
    }
}

#[test]
fn unsupported_routes_methods_encodings_and_ambiguous_framing_are_explicit() {
    for (request, status) in [
        (b"GET /v1/responses HTTP/1.1\r\nHost: localhost\r\n\r\n".as_slice(), 405),
        (b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 501),
        (b"POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 404),
        (b"POST /v1/capabilities HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 405),
        (b"POST /v1/models HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 405),
        (b"GET /v1/models/mock HTTP/1.1\r\nHost: localhost\r\n\r\n".as_slice(), 404),
        (b"GET /v1/capabilities HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 400),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Encoding: gzip\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 415),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Encoding: identity\r\nContent-Encoding: gzip\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 400),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}".as_slice(), 400),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 400),
        (b"POST /v1/responses HTTP/1.1\r\nHost: hostile.invalid\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 421),
        (b"POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nOrigin: http://hostile.invalid\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 403),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nOrigin: null\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 403),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nOrigin: http://localhost\r\nOrigin: http://localhost\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 400),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 415),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 415),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 9\r\n\r\n{\"a\":1}{}".as_slice(), 400),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"a\":1,\"a\":2}".as_slice(), 400),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost:1\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}".as_slice(), 421),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n[]".as_slice(), 400),
        (b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 1\r\n\r\n\xff".as_slice(), 400),
    ] {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
        let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = gateway.local_addr().unwrap();
        let relay = thread::spawn(move || { let (socket, _) = gateway.accept().unwrap(); handle(socket, &endpoint).unwrap(); });
        let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
        socket.write_all(request).unwrap();
        let mut reader = BufReader::new(socket);
        let head = transport::read_head(&mut reader).unwrap();
        assert!(head.first.starts_with(&format!("HTTP/1.1 {status} ")));
        relay.join().unwrap();
        upstream.set_nonblocking(true).unwrap();
        assert_eq!(upstream.accept().unwrap_err().kind(), io::ErrorKind::WouldBlock);
    }
}

#[test]
fn explicit_sampling_bodies_are_byte_identical_in_plain_dispatch_preparation() {
    let upstream = Upstream::Single(Endpoint::parse("http://127.0.0.1:1").unwrap());
    let head = transport::Head {
        first: String::new(),
        fields: Vec::new(),
    };
    for path in [
        "/v1/completions",
        "/completions",
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/responses",
        "/responses",
        "/v1/messages",
    ] {
        for &(field, explicit) in sampling::EXPLICIT_SAMPLING_EXAMPLES {
            let original = format!(r#" {{ "{field}":{explicit}, "extension":1.20e0 }} "#);
            let (lease, effective) =
                prepare_dispatch(&upstream, "POST", path, None, &head, original.as_bytes())
                    .ok()
                    .unwrap();
            assert!(lease.is_none());
            assert_eq!(effective.as_ref(), original.as_bytes(), "{path}: {field}");
        }
        let (_, effective) = prepare_dispatch(&upstream, "POST", path, None, &head, b"{}")
            .ok()
            .unwrap();
        assert_eq!(effective.as_ref(), sampling::DEFAULTS_JSON.as_bytes());
    }
}

#[test]
fn individually_explicit_sampling_bodies_forward_unchanged_on_both_paths() {
    for lane in 0..3 {
        for path in ["/v1/completions", "/completions", "/v1/chat/completions", "/v1/responses", "/v1/messages"] {
            for &(field, explicit) in sampling::EXPLICIT_SAMPLING_EXAMPLES {
                let payload = format!(r#" {{ "{field}":{explicit}, "prompt_cache_key":"client", "cache":false, "extension":1.20e0 }} "#).into_bytes();
                let expected = payload.clone();
                let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
                let endpoint =
                    Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
                let server = thread::spawn(move || {
                    let mut reader = BufReader::new(
                        DeadlineStream::new(upstream.accept().unwrap().0, None).unwrap(),
                    );
                    assert_eq!(receive(&mut reader).1, expected);
                    reader
                        .get_mut()
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                        )
                        .unwrap();
                });
                assert!(
                    call_gateway(endpoint, lane, "POST", path, &payload, None)
                        .0
                        .first
                        .starts_with("HTTP/1.1 200 ")
                );
                server.join().unwrap();
            }
        }
    }
}

// These responses were captured from the real direct engine route. The
// recording pins build/checkpoint/date and the independently compared live IDs.
// This host test replays those bytes; it does not run model arithmetic.
fn completions_recording() -> json::Value {
    let value = json::parse(include_str!("../tests/fixtures/completions-engine-2026-10-23.json")).unwrap();
    assert_eq!(value.get("schema").unwrap().as_str(), Some("vade.completions.recording.v1"));
    assert_eq!(value.get("engine_sha").unwrap().as_str().unwrap().len(), 40);
    assert_eq!(value.get("checkpoint").unwrap().get("hash_state").unwrap().as_str(), Some("ready"));
    value
}
fn completion_ids(body: &str) -> Vec<u32> {
    body.lines().filter_map(|line| line.strip_prefix("data: ")).filter(|line| *line != "[DONE]")
        .flat_map(|line| {
            let value = json::parse(line).unwrap();
            value.get("choices").unwrap().as_array().unwrap().iter()
                .filter_map(|choice| choice.get("token_id"))
                .map(|id| json::stringify(id).unwrap().parse().unwrap()).collect::<Vec<_>>()
        }).collect()
}

#[test]
fn recorded_engine_completions_pass_through_both_gateways_exactly() {
    let recording = completions_recording();
    for case in recording.get("cases").unwrap().as_array().unwrap() {
        let ids: Vec<u32> = case.get("direct_ids").unwrap().as_array().unwrap().iter()
            .map(|id| json::stringify(id).unwrap().parse().unwrap()).collect();
        assert_eq!(ids.len(), 32);
        assert_eq!(completion_ids(case.get("direct_sse").unwrap().as_str().unwrap()), ids);
        for name in ["direct-repeat_ids", "gateway_ids", "gateway-alias_ids"] {
            assert_eq!(case.get("live_comparisons").unwrap().get(name), case.get("direct_ids"));
        }
        for lane in 0..3 {
            for path in ["/v1/completions?client=1", "/completions?client=1"] {
                for stream in [false, true] {
                    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
                    let endpoint = Endpoint::parse(&format!("http://{}/v1/chat/completions", upstream.local_addr().unwrap())).unwrap();
                    let payload = case.get(if stream { "stream_request" } else { "nonstream_request" }).unwrap().as_str().unwrap().as_bytes().to_vec();
                    let expected = case.get(if stream { "direct_sse" } else { "direct_json" }).unwrap().as_str().unwrap().as_bytes().to_vec();
                    let original = payload.clone(); let reply = expected.clone();
                    let defaults = case.get("name").unwrap().as_str() == Some("no-fields");
                    let server = thread::spawn(move || {
                        let mut reader = BufReader::new(DeadlineStream::new(upstream.accept().unwrap().0, None).unwrap());
                        let (head, body) = receive(&mut reader);
                        assert_eq!(head.first, "POST /v1/completions?client=1 HTTP/1.1");
                        if defaults {
                            let original = std::str::from_utf8(&original).unwrap();
                            let end = original.rfind('}').unwrap();
                            let effective = format!("{},{}{}", &original[..end],
                                &sampling::DEFAULTS_JSON[1..sampling::DEFAULTS_JSON.len()-1], &original[end..]);
                            assert_eq!(body, effective.as_bytes());
                        } else { assert_eq!(body, original); }
                        write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            if stream { "text/event-stream" } else { "application/json" }, reply.len()).unwrap();
                        reader.get_mut().write_all(&reply).unwrap();
                    });
                    let (head, forwarded) = call_gateway(endpoint, lane, "POST", path, &payload, None);
                    assert!(head.first.starts_with("HTTP/1.1 200 "));
                    assert_eq!(forwarded, expected, "lane={lane} path={path} stream={stream}");
                    if stream { assert_eq!(completion_ids(std::str::from_utf8(&forwarded).unwrap()), ids); }
                    server.join().unwrap();
                }
            }
        }
    }
}

#[test]
fn opaque_completions_sse_survives_bytewise_http_chunks_on_both_gateways() {
    // This intentionally includes unfamiliar events, multiline non-JSON data,
    // CRLF, Unicode and an opaque token ID. Relays must not parse or repair it.
    let recording = completions_recording();
    let case = &recording.get("cases").unwrap().as_array().unwrap()[0];
    let sse = format!(": heartbeat\r\nid: completion-fixture\r\nevent: vendor.unknown\r\ndata: {{\"note\":\"雪 café\",\"future\":1.00}}\r\ndata: opaque non-JSON continuation\r\n\r\nretry: 250\n\n{}",
        case.get("direct_sse").unwrap().as_str().unwrap());
    for lane in 0..3 {
        for path in ["/v1/completions?client=1", "/completions?client=1"] {
            let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = Endpoint::parse(&format!("http://{}/v1/responses", upstream.local_addr().unwrap())).unwrap();
            let payload = br#"{ "model":"mock", "prompt":"exact\u0020", "stream":true, "temperature":0, "cache":false, "prompt_cache_key":"caller", "future":9007199254740993 }"#;
            let reply = sse.clone();
            let server = thread::spawn(move || {
                let mut reader = BufReader::new(DeadlineStream::new(upstream.accept().unwrap().0, None).unwrap());
                let (head, body) = receive(&mut reader);
                let routed = "/v1/completions?client=1";
                assert_eq!(head.first, format!("POST {routed} HTTP/1.1"));
                assert_eq!(body, payload);
                reader.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nX-Provider-Extension: opaque\r\nConnection: close\r\n\r\n").unwrap();
                for byte in reply.as_bytes() {
                    // HTTP chunks split every multibyte character and frame.
                    reader.get_mut().write_all(&[b'1', b'\r', b'\n', *byte, b'\r', b'\n']).unwrap();
                }
                reader.get_mut().write_all(b"0\r\n\r\n").unwrap();
            });
            let (head, body) = call_gateway(endpoint, lane, "POST", path, payload, None);
            assert!(head.first.starts_with("HTTP/1.1 200 "));
            assert_eq!(head.get("x-provider-extension"), Some("opaque"));
            assert_eq!(head.get("content-type"), Some("text/event-stream"));
            assert_eq!(body, sse.as_bytes(), "lane={lane} path={path}");
            server.join().unwrap();
        }
    }
}

// Requests from the clients people point at Vade, replayed through every gateway
// lane. Each case says whether its request was captured from the real client or
// written from memory; the engine replies follow the engine's own wire format.
// A case passes when the gateway forwards what the engine expects and the client
// can read everything it needs from the relayed reply.
fn client_cases() -> Vec<json::Value> {
    let value = json::parse(include_str!("../tests/fixtures/openai-clients.json")).unwrap();
    assert_eq!(value.get("schema").unwrap().as_str(), Some("vade.openai-clients.v1"));
    value.get("cases").unwrap().as_array().unwrap().to_vec()
}
fn text(value: &json::Value, key: &str) -> String { value.get(key).and_then(json::Value::as_str).unwrap_or_else(|| panic!("{key}")).into() }
fn number(value: Option<&json::Value>) -> u64 { json::stringify(value.unwrap()).unwrap().parse().unwrap() }

fn client_call(endpoint: Endpoint, lane: u8, access: Access, case: &json::Value, extra: &str) -> (transport::Head, Vec<u8>) {
    let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = gateway.local_addr().unwrap();
    let relay = thread::spawn(move || {
        let socket = gateway.accept().unwrap().0;
        let result = if lane == 0 { handle_target(socket, &Upstream::Single(endpoint.clone()), None, &ConnectionPool::default(), &access) } else {
            invisible::handle_conformance(socket, &format!("http://{}", endpoint.authority), lane == 2, None)
        };
        if let Err(e) = result { assert_eq!(e.kind(), io::ErrorKind::ConnectionAborted, "relay failed: {e}"); }
    });
    let payload = text(case, "request");
    let mut request = format!("{} {} HTTP/1.1\r\nHost: localhost:{}\r\n", text(case, "method"), text(case, "path"), address.port());
    for pair in case.get("headers").unwrap().as_array().unwrap() {
        let pair = pair.as_array().unwrap();
        if extra.is_empty() || pair[0].as_str() != Some("authorization") {
            request.push_str(&format!("{}: {}\r\n", pair[0].as_str().unwrap(), pair[1].as_str().unwrap()));
        }
    }
    if !payload.is_empty() { request.push_str(&format!("Content-Length: {}\r\n", payload.len())); }
    request.push_str(extra);
    let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
    write!(socket, "{request}\r\n{payload}").unwrap();
    let mut reader = BufReader::new(socket);
    let head = transport::read_head(&mut reader).unwrap();
    let mut body = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
    let mut bytes = Vec::new();
    body.read_to_end(&mut bytes).unwrap();
    relay.join().unwrap();
    (head, bytes)
}

/// What a client reads from a reply: text, tool calls rebuilt from deltas the way
/// the OpenAI SDKs rebuild them, the finish reason, usage, model ids and errors.
#[derive(Debug, Default, PartialEq)]
struct ClientView { content: String, reasoning: String, text: String, finish: String,
    calls: Vec<(String, String, String)>, usage: Vec<u64>, models: Vec<String>, error: Vec<String> }

fn client_view(bytes: &[u8], stream: bool) -> ClientView {
    let body = std::str::from_utf8(bytes).unwrap();
    let mut view = ClientView::default();
    let read = |value: &json::Value, view: &mut ClientView| {
        if let Some(usage) = value.get("usage").filter(|u| **u != json::Value::Null) {
            view.usage = vec![number(usage.get("prompt_tokens")), number(usage.get("completion_tokens")), number(usage.get("total_tokens")),
                number(usage.get("prompt_tokens_details").and_then(|d| d.get("cached_tokens")))];
            assert_eq!(view.usage[0] + view.usage[1], view.usage[2]);
        }
        let object = text(value, "object");
        assert!(value.get("id").and_then(json::Value::as_str).is_some() && value.get("created").is_some() && value.get("model").is_some());
        for choice in value.get("choices").unwrap().as_array().unwrap() {
            assert_eq!(number(choice.get("index")), 0);
            if let Some(reason) = choice.get("finish_reason").and_then(json::Value::as_str) {
                assert!(["stop", "length", "tool_calls"].contains(&reason), "{reason}");
                view.finish = reason.into();
            }
            if object == "text_completion" { view.text.push_str(&text(choice, "text")); continue; }
            let message = if stream { assert_eq!(object, "chat.completion.chunk"); choice.get("delta").unwrap() } else {
                assert_eq!(object, "chat.completion");
                let message = choice.get("message").unwrap();
                assert_eq!(message.get("role").and_then(json::Value::as_str), Some("assistant"));
                message
            };
            view.content.push_str(message.get("content").and_then(json::Value::as_str).unwrap_or(""));
            view.reasoning.push_str(message.get("reasoning_content").and_then(json::Value::as_str).unwrap_or(""));
            for (position, call) in message.get("tool_calls").and_then(json::Value::as_array).unwrap_or(&[]).iter().enumerate() {
                let index = if stream { number(call.get("index")) as usize } else { position };
                let function = call.get("function").unwrap();
                if let Some(id) = call.get("id").and_then(json::Value::as_str) {
                    // A new call opens with its id, type and name, as the SDKs require.
                    assert_eq!((index, text(call, "type")), (view.calls.len(), "function".into()));
                    view.calls.push((id.into(), text(function, "name"), String::new()));
                }
                view.calls[index].2.push_str(&text(function, "arguments"));
            }
        }
    };
    if stream {
        let events: Vec<&str> = body.split("\n\n").filter(|event| !event.is_empty()).collect();
        assert_eq!(events.last(), Some(&"data: [DONE]"), "a stream ends with [DONE]");
        for event in &events[..events.len() - 1] {
            read(&json::parse(event.strip_prefix("data: ").unwrap()).unwrap(), &mut view);
        }
        for call in &view.calls { assert!(json::parse(&call.2).is_ok_and(|args| matches!(args, json::Value::Object(_))), "{}", call.2); }
        return view;
    }
    let value = json::parse(body).unwrap();
    if let Some(error) = value.get("error") {
        view.error = vec![text(error, "type"), text(error, "message")];
        if let Some(code) = error.get("code").and_then(json::Value::as_str) { view.error.push(code.into()); }
    } else if value.get("object").and_then(json::Value::as_str) == Some("list") {
        view.models = value.get("data").unwrap().as_array().unwrap().iter().map(|model| text(model, "id")).collect();
    } else { read(&value, &mut view); }
    view
}

fn expected_view(expect: &json::Value) -> ClientView {
    let field = |key: &str| expect.get(key).and_then(json::Value::as_str).unwrap_or("").to_owned();
    let list = |key: &str| expect.get(key).and_then(json::Value::as_array).unwrap_or(&[]).to_vec();
    let error = ["error_type", "error_message", "error_code"].iter().map(|key| field(key)).filter(|s| !s.is_empty()).collect();
    ClientView { content: field("content"), reasoning: field("reasoning"), text: field("text"), finish: field("finish_reason"),
        calls: list("tool_calls").iter().map(|call| { let call = call.as_array().unwrap();
            (call[0].as_str().unwrap().into(), call[1].as_str().unwrap().into(), call[2].as_str().unwrap().into()) }).collect(),
        usage: list("usage").iter().map(|n| number(Some(n))).collect(),
        models: list("models").iter().map(|m| m.as_str().unwrap().into()).collect(), error }
}

fn effective_request(case: &json::Value) -> Option<Vec<u8>> {
    let mut body = text(case, "request");
    if let Some(fold) = case.get("fold").and_then(json::Value::as_array) {
        body = body.replace(fold[0].as_str().unwrap(), fold[1].as_str().unwrap());
    }
    match text(case, "upstream").as_str() {
        "none" => None,
        "same" => Some(body.into_bytes()),
        "defaults" => {
            let end = body.rfind('}').unwrap();
            Some(format!("{},{}{}", &body[..end], &sampling::DEFAULTS_JSON[1..sampling::DEFAULTS_JSON.len() - 1], &body[end..]).into_bytes())
        }
        other => panic!("unknown upstream expectation {other}"),
    }
}

/// One mock engine answering one request with the case's reply; returns what it received.
fn engine_for(case: &json::Value) -> (Endpoint, thread::JoinHandle<Option<(transport::Head, Vec<u8>)>>) {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = Endpoint::parse(&format!("http://{}/v1", upstream.local_addr().unwrap())).unwrap();
    let reply = case.get("reply").map(|reply| (number(case.get("reply_status")), text(case, "reply_type"), reply.as_str().unwrap().to_owned()));
    let server = thread::spawn(move || {
        let (status, kind, reply) = reply?;
        let mut reader = BufReader::new(DeadlineStream::new(upstream.accept().unwrap().0, None).unwrap());
        let received = receive(&mut reader);
        write!(reader.get_mut(), "HTTP/1.1 {status} Engine\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).unwrap();
        Some(received)
    });
    (endpoint, server)
}

#[test]
fn recorded_client_requests_reach_the_engine_and_clients_read_the_replies() {
    let cases = client_cases();
    assert!(cases.len() >= 15);
    for client in ["openai Python client", "Continue", "Open WebUI", "curl"] {
        assert!(cases.iter().any(|case| text(case, "client").starts_with(client)), "{client}");
    }
    for case in &cases {
        let name = text(case, "name");
        let provenance = text(case, "provenance");
        assert!(["verified:", "from memory:", "from docs:"].iter().any(|p| provenance.starts_with(p)), "{name}: {provenance}");
        let expected = effective_request(case);
        // The standalone gateway passes every path through, so only the core answers 501 itself.
        // Its opt-in policies (lane 2) add cache fields on purpose and are covered above.
        let lanes = if expected.is_none() { 0..1 } else { 0..2 };
        for lane in lanes {
            let (endpoint, server) = engine_for(case);
            let (head, bytes) = client_call(endpoint, lane, Access::default(), case, "");
            let received = server.join().unwrap();
            let expect = case.get("expect").unwrap();
            assert!(head.first.starts_with(&format!("HTTP/1.1 {} ", number(expect.get("status")))), "{name} lane {lane}: {}", head.first);
            match (&expected, received) {
                (Some(body), Some((upstream, forwarded))) => {
                    assert_eq!(upstream.first, format!("{} {} HTTP/1.1", text(case, "method"), text(case, "path")), "{name} lane {lane}");
                    assert_eq!(String::from_utf8(forwarded).unwrap(), String::from_utf8(body.clone()).unwrap(), "{name} lane {lane}");
                    assert_eq!(bytes, text(case, "reply").as_bytes(), "{name} lane {lane}: the reply is relayed byte for byte");
                }
                (None, None) => {}
                _ => panic!("{name} lane {lane}: upstream expectation mismatch"),
            }
            let stream = expect.get("stream").is_some_and(|s| *s == json::Value::Bool(true));
            assert_eq!(client_view(&bytes, stream), expected_view(expect), "{name} lane {lane}");
        }
    }
}

#[test]
fn optional_bearer_token_gates_client_requests_and_stays_local() {
    let case = client_cases().into_iter().find(|case| text(case, "name") == "openai-python-chat").unwrap();
    let mut access = Access::default();
    access.require_token("sk-local").unwrap();
    // The SDK's own Authorization header carries the configured key.
    let (endpoint, server) = engine_for(&case);
    let (head, _) = client_call(endpoint, 0, access.clone(), &case, "");
    assert!(head.first.starts_with("HTTP/1.1 200 "));
    let (upstream, _) = server.join().unwrap().unwrap();
    assert_eq!(upstream.get("authorization"), None, "the gateway token is not forwarded");
    for wrong in ["Authorization: Bearer other\r\n", "Authorization: Basic c2stbG9jYWw=\r\n", "X-Unrelated: 1\r\n"] {
        let (endpoint, _) = engine_for(&json::Value::Object(Vec::new()));
        let (head, bytes) = client_call(endpoint, 0, access.clone(), &case, wrong);
        assert!(head.first.starts_with("HTTP/1.1 401 "), "{wrong}");
        let view = client_view(&bytes, false);
        assert_eq!((view.error[0].as_str(), view.error.last().unwrap().as_str()), ("authentication_error", "invalid_api_key"));
    }
    // Off by default: any client key, or none, passes through to the engine unchanged.
    let (endpoint, server) = engine_for(&case);
    assert!(client_call(endpoint, 0, Access::default(), &case, "").0.first.starts_with("HTTP/1.1 200 "));
    assert_eq!(server.join().unwrap().unwrap().0.get("authorization"), Some("Bearer sk-local"));
}

#[test]
fn client_hang_up_before_the_reply_stops_the_queued_engine_request() {
    // A queued engine stream survives a closed socket but is dropped when bytes
    // follow the request (engine serve crate, scheduler.rs, Wire::connected).
    for lane in 0..3 {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
        let (queued, hold) = mpsc::channel();
        let server = thread::spawn(move || {
            let (socket, _) = upstream.accept().unwrap();
            let mut peer = socket.try_clone().unwrap();
            peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
            receive(&mut reader);
            queued.send(()).unwrap();
            // No reply: the request waits in the engine queue until the client leaves.
            let mut after = Vec::new();
            peer.read_to_end(&mut after).unwrap();
            after
        });
        let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = gateway.local_addr().unwrap();
        let relay = thread::spawn(move || {
            let socket = gateway.accept().unwrap().0;
            let _ = if lane == 0 { handle(socket, &endpoint) } else {
                invisible::handle_conformance(socket, &format!("http://{}", endpoint.authority), lane == 2, None)
            };
        });
        let payload = r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let mut client = TcpStream::connect(address).unwrap();
        write!(client, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}", payload.len()).unwrap();
        hold.recv_timeout(Duration::from_secs(10)).unwrap();
        drop(client);
        relay.join().unwrap();
        assert_eq!(server.join().unwrap(), b"\r\n", "lane {lane}: the engine sees a byte after the request, not only a close");
    }
}
