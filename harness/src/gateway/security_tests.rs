use super::*;

fn listener() -> TcpListener {
    TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap()
}

#[test]
fn connect_timeout_returns_a_502_with_the_engine_reason() {
    let endpoint = Endpoint::parse("http://engine.invalid:80").unwrap();
    let gateway = listener();
    let address = gateway.local_addr().unwrap();
    let worker = thread::spawn(move || transport::resolve::with_lookup(Box::new(|_, deadline| {
        assert!(deadline.saturating_duration_since(Instant::now()) <= Duration::from_secs(5));
        Err(transport::resolve::lookup_timeout("engine.invalid", false))
    }), || handle_target_with_timeout(gateway.accept().unwrap().0, &Upstream::Single(endpoint), None,
        &ConnectionPool::default(), &Access::default(), Duration::from_secs(30))));
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    client.write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
    let mut reply = String::new();
    client.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.1 502 ") && reply.contains("the engine could not be reached"), "{reply}");
    worker.join().unwrap().unwrap();
}

/// The slow-bytes tests: three bytes, one a step, against one total deadline. The last byte
/// leaves at two steps, 600 ms before the deadline.
const SLOW_BYTES_DEADLINE: Duration = Duration::from_millis(900);
const SLOW_BYTE_STEP: Duration = Duration::from_millis(150);

#[test]
fn slow_request_bytes_hit_one_total_receive_deadline() {
    let upstream = listener();
    let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
    let gateway = listener();
    let address = gateway.local_addr().unwrap();
    let worker = thread::spawn(move || {
        let socket = gateway.accept().unwrap().0;
        handle_target_with_timeout(
            socket,
            &Upstream::Single(endpoint),
            None,
            &ConnectionPool::default(),
            &Access::default(),
            SLOW_BYTES_DEADLINE,
        )
    });
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    // Every byte is sent well before the deadline. A byte that arrives after the server gave up
    // makes it reset the connection, and the client then loses the 408 (seen on a hosted Mac
    // with 50 ms steps against a 120 ms deadline).
    for byte in b"GET" {
        let _ = client.write_all(&[*byte]);
        thread::sleep(SLOW_BYTE_STEP);
    }
    let mut reply = String::new();
    client.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.1 408 "), "{reply}");
    assert!(reply.contains("Retry-After: 1\r\n") && reply.contains("send the request again"), "{reply}");
    worker.join().unwrap().unwrap();
    upstream.set_nonblocking(true).unwrap();
    assert_eq!(upstream.accept().unwrap_err().kind(), io::ErrorKind::WouldBlock);
}

#[test]
fn slow_request_body_uses_the_same_total_receive_deadline() {
    let upstream = listener();
    let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
    let gateway = listener();
    let address = gateway.local_addr().unwrap();
    let worker = thread::spawn(move || {
        let socket = gateway.accept().unwrap().0;
        handle_target_with_timeout(socket, &Upstream::Single(endpoint), None, &ConnectionPool::default(), &Access::default(), Duration::from_millis(120))
    });
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: 3\r\n\r\na").as_bytes()).unwrap();
    thread::sleep(Duration::from_millis(180));
    let mut reply = String::new();
    let _ = client.read_to_string(&mut reply);
    assert!(reply.starts_with("HTTP/1.1 408 ") && reply.contains("send the request again"), "{reply}");
    worker.join().unwrap().unwrap();
    upstream.set_nonblocking(true).unwrap();
    assert_eq!(upstream.accept().unwrap_err().kind(), io::ErrorKind::WouldBlock);
}

#[test]
fn slow_connections_cannot_take_more_than_the_listener_limit() {
    let upstream = listener();
    let endpoint = Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
    let gateway = listener();
    let address = gateway.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let server_stop = Arc::clone(&stop);
    let worker = thread::spawn(move || {
        serve_listener(
            gateway,
            Upstream::Single(endpoint),
            Some(&server_stop),
            Access::default(),
            3,
        )
    });
    let mut slow = Vec::new();
    for _ in 0..3 {
        let mut socket = TcpStream::connect(address).unwrap();
        socket.write_all(b"G").unwrap();
        slow.push(socket);
        thread::sleep(Duration::from_millis(25));
    }
    let mut extra = TcpStream::connect(address).unwrap();
    extra.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    extra.write_all(format!("GET /v1/models HTTP/1.1\r\nHost: {address}\r\n\r\n").as_bytes()).unwrap();
    let mut reply = String::new();
    extra.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.1 503 ") && reply.contains("Retry-After: 1\r\n"), "{reply}");
    drop(slow);
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap().unwrap();
}
