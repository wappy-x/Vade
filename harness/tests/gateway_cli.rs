//! Actual repeated-upstream CLI wiring; all peers are first-party loopback mocks.
#![cfg(target_os = "macos")]
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

struct Gateway(Child);
impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct Mock {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    posts: Arc<AtomicUsize>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let running = stop.clone();
        let posts = Arc::new(AtomicUsize::new(0));
        let count = posts.clone();
        let worker = thread::spawn(move || {
            while !running.load(Ordering::Relaxed) {
                let (stream, _) = match listener.accept() {
                    Ok(client) => client,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("mock accept failed: {e}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut reader = BufReader::new(stream);
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    assert!(reader.read_line(&mut line).unwrap() > 0, "mock header EOF");
                    if line == "\r\n" {
                        break;
                    }
                    if let Some((key, value)) = line.split_once(':') {
                        if key.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse::<usize>().unwrap();
                        }
                    }
                }
                assert!(length < 4096);
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let response = if first.starts_with("GET /v1/capabilities ") {
                    include_str!("fixtures/engine-capabilities-v1.json").to_string()
                } else {
                    assert!(first.starts_with("POST /v1/chat/completions "));
                    assert_eq!(body, br#"{ "temperature":0, "seed":42, "top_p":1, "model":"mock", "prompt_cache_key":"cli", "messages":[], "unknown":1.00 }"#);
                    let cached = if count.fetch_add(1, Ordering::Relaxed) > 0 {
                        80
                    } else {
                        0
                    };
                    format!(
                        "{{\"usage\":{{\"prompt_tokens\":100,\"prompt_tokens_details\":{{\"cached_tokens\":{cached}}}}}}}"
                    )
                };
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).unwrap();
            }
        });
        Self {
            address,
            stop,
            posts,
            worker: Some(worker),
        }
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let result = self.worker.take().unwrap().join();
        if !thread::panicking() {
            result.unwrap();
        }
    }
}
fn call(address: SocketAddr, path: &str, body: &[u8]) -> std::io::Result<String> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(100))?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    let method = if body.is_empty() { "GET" } else { "POST" };
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}
#[test]
fn cli_routes_repeated_upstreams_and_exposes_local_stats_without_logging_payloads() {
    let mocks = [Mock::start(), Mock::start(), Mock::start()];
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
    command.args(["gateway", "--bind", &address.to_string()]);
    // An inherited client token would answer these unauthenticated calls with 401.
    command.env_remove("VADE_GATEWAY_TOKEN");
    for mock in &mocks {
        command.args(["--upstream", &format!("http://{}", mock.address)]);
    }
    let mut gateway = Gateway(
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        assert!(Instant::now() < deadline, "gateway discovery timed out");
        assert!(
            gateway.0.try_wait().unwrap().is_none(),
            "gateway exited before discovery"
        );
        if call(address, "/_vade/gateway/stats", &[])
            .is_ok_and(|s| s.matches("\"healthy\":true").count() == 3)
        {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let body = br#"{ "temperature":0, "seed":42, "top_p":1, "model":"mock", "prompt_cache_key":"cli", "messages":[], "unknown":1.00 }"#;
    for cached in [0, 80] {
        let response = call(address, "/v1/chat/completions", body).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.contains(&format!("\"cached_tokens\":{cached}")));
    }
    assert_eq!(
        mocks
            .iter()
            .map(|m| m.posts.load(Ordering::Relaxed))
            .sum::<usize>(),
        2
    );
    assert_eq!(
        mocks
            .iter()
            .filter(|m| m.posts.load(Ordering::Relaxed) == 2)
            .count(),
        1
    );
    let stats = call(address, "/_vade/gateway/stats", &[]).unwrap();
    assert!(stats.contains("\"cached_tokens_ratio\":0.40000000"));
    assert!(!stats.contains("prompt_cache_key") && !stats.contains("127.0.0.1"));
    gateway.0.kill().unwrap();
    gateway.0.wait().unwrap();
    let mut stdout = String::new();
    let mut stderr = String::new();
    gateway
        .0
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    gateway
        .0
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(stdout.is_empty() && stderr.is_empty());
}
