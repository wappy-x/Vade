//! Paired direct and default gateway times against a fixed local mock engine.
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    process::{Child, Command, Stdio},
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    thread,
    time::{Duration, Instant},
};

const ROUNDS: usize = 3;
const WARMUP_PAIRS: usize = 20;
const PAIRS: usize = 200;
const TIMEOUT: Duration = Duration::from_secs(5);

struct Case {
    name: &'static str,
    method: &'static str,
    path: &'static str,
    body: String,
    reply: &'static str,
    sse: bool,
}

fn cases() -> Vec<Case> {
    const CHAT: &str = r#"{"id":"chatcmpl-bench","object":"chat.completion","model":"mock","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":1,"total_tokens":9}}"#;
    const SSE: &str = "data: {\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"model\":\"mock\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-bench\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let chat = |stream, content: &str| format!(r#"{{"model":"mock","messages":[{{"role":"user","content":"{content}"}}],"temperature":0,"seed":42,"max_tokens":16,"stream":{stream}}}"#);
    vec![
        Case { name: "health", method: "GET", path: "/health", body: String::new(), reply: r#"{"status":"ok"}"#, sse: false },
        Case { name: "models", method: "GET", path: "/v1/models", body: String::new(), reply: r#"{"object":"list","data":[{"id":"mock","object":"model","owned_by":"vade"}]}"#, sse: false },
        Case { name: "chat_json", method: "POST", path: "/v1/chat/completions", body: chat(false, "Say OK."), reply: CHAT, sse: false },
        Case { name: "chat_sse", method: "POST", path: "/v1/chat/completions", body: chat(true, "Say OK."), reply: SSE, sse: true },
        Case { name: "responses", method: "POST", path: "/v1/responses", body: r#"{"model":"mock","input":"Say OK.","temperature":0,"seed":42,"max_output_tokens":16,"stream":false}"#.into(), reply: r#"{"id":"resp-bench","object":"response","status":"completed","model":"mock","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"OK","annotations":[]}]}]}"#, sse: false },
        Case { name: "chat_long", method: "POST", path: "/v1/chat/completions", body: chat(false, &"a".repeat(16 * 1024)), reply: CHAT, sse: false },
        Case { name: "chat_tool", method: "POST", path: "/v1/chat/completions", body: r#"{"model":"mock","messages":[{"role":"user","content":"Call echo with OK."}],"tools":[{"type":"function","function":{"name":"echo","description":"Return text.","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}],"tool_choice":"auto","temperature":0,"seed":42,"max_tokens":64,"stream":false}"#.into(), reply: r#"{"id":"chatcmpl-bench","object":"chat.completion","model":"mock","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-bench","type":"function","function":{"name":"echo","arguments":"{\"text\":\"OK\"}"}}]},"finish_reason":"tool_calls"}]}"#, sse: false },
    ]
}

fn listener() -> io::Result<TcpListener> {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
}

struct Mock {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<usize>>,
}
impl Mock {
    fn start() -> io::Result<Self> {
        let listener = listener()?;
        let address = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let worker = thread::spawn(move || {
            let cases = cases();
            let mut served = 0;
            loop {
                let (socket, _) = listener.accept().expect("mock accept failed");
                if stopped.load(Ordering::Relaxed) { return served; }
                socket.set_nodelay(true).unwrap();
                socket.set_read_timeout(Some(TIMEOUT)).unwrap();
                socket.set_write_timeout(Some(TIMEOUT)).unwrap();
                let mut reader = BufReader::new(socket);
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut length = None;
                let mut index = None;
                let mut bytes = first.len();
                loop {
                    let mut line = String::new();
                    assert!(reader.read_line(&mut line).unwrap() > 0, "mock header ended early");
                    bytes += line.len();
                    assert!(bytes <= 64 * 1024, "mock header is too large");
                    if line == "\r\n" { break; }
                    let (key, value) = line.split_once(':').expect("invalid mock header");
                    if key.eq_ignore_ascii_case("content-length") { length = Some(value.trim().parse::<usize>().unwrap()); }
                    if key.eq_ignore_ascii_case("x-bench-case") { index = Some(value.trim().parse::<usize>().unwrap()); }
                }
                let case = &cases[index.expect("missing case index")];
                assert_eq!(first, format!("{} {} HTTP/1.1\r\n", case.method, case.path));
                assert_eq!(length, Some(case.body.len()), "request length changed");
                let mut body = vec![0; case.body.len()];
                reader.read_exact(&mut body).unwrap();
                assert_eq!(body, case.body.as_bytes(), "request bytes changed");
                // The server closes after each reply. JSON has a length. SSE ends at EOF.
                // See engine/crates/serve/src/http.rs and scheduler/dispatch.rs.
                let framing = if case.sse { String::new() } else { format!("Content-Length: {}\r\n", case.reply.len()) };
                let content_type = if case.sse { "text/event-stream" } else { "application/json" };
                let head = format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n{framing}Cache-Control: no-store\r\nConnection: close\r\n\r\n");
                reader.get_mut().write_all(head.as_bytes()).unwrap();
                reader.get_mut().write_all(case.reply.as_bytes()).unwrap();
                served += 1;
            }
        });
        Ok(Self { address, stop, worker: Some(worker) })
    }
    fn finish(&mut self) -> thread::Result<usize> {
        self.stop.store(true, Ordering::Relaxed);
        let _ = TcpStream::connect_timeout(&self.address, TIMEOUT);
        self.worker.take().unwrap().join()
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        if self.worker.is_some() { let _ = self.finish(); }
    }
}

struct Gateway(Child);
impl Gateway {
    fn start(upstream: SocketAddr) -> io::Result<(Self, SocketAddr)> {
        let reservation = listener()?;
        let address = reservation.local_addr()?;
        drop(reservation);
        let child = Command::new(std::env::current_exe()?)
            .args(["--gateway-child", &address.to_string(), &format!("http://{upstream}")])
            .env_clear()
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
            .spawn()?;
        let mut gateway = Self(child);
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if gateway.0.try_wait()?.is_some() { return Err(io::Error::other("gateway exited before startup")); }
            if TcpStream::connect_timeout(&address, Duration::from_millis(50)).is_ok() { return Ok((gateway, address)); }
            if Instant::now() >= deadline { return Err(io::Error::new(io::ErrorKind::TimedOut, "gateway startup timed out")); }
            thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[derive(Clone, Copy)]
struct Timing { first_ns: i128, full_ns: i128 }
struct Pair { round: usize, case: usize, index: usize, direct: Timing, gateway: Timing }

fn packet(case: &Case, index: usize, address: SocketAddr) -> Vec<u8> {
    format!("{} {} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nX-Bench-Case: {index}\r\nConnection: close\r\n\r\n{}", case.method, case.path, case.body.len(), case.body).into_bytes()
}

fn check_response(response: &[u8], case: &Case) {
    let end = response.windows(4).position(|window| window == b"\r\n\r\n").expect("missing response head") + 4;
    assert!(response.starts_with(b"HTTP/1.1 200 OK\r\n"), "unexpected status");
    let head = std::str::from_utf8(&response[..end]).unwrap();
    let chunked = head.lines().filter_map(|line| line.split_once(':'))
        .any(|(name, value)| name.eq_ignore_ascii_case("transfer-encoding") && value.trim().eq_ignore_ascii_case("chunked"));
    let mut body = &response[end..];
    if chunked {
        let mut decoded = Vec::new();
        loop {
            let line_end = body.windows(2).position(|window| window == b"\r\n").expect("missing chunk size");
            let size = usize::from_str_radix(std::str::from_utf8(&body[..line_end]).unwrap(), 16).unwrap();
            body = &body[line_end + 2..];
            if size == 0 { assert_eq!(body, b"\r\n", "invalid final chunk"); break; }
            decoded.extend_from_slice(&body[..size]);
            assert_eq!(&body[size..size + 2], b"\r\n", "invalid chunk end");
            body = &body[size + 2..];
        }
        assert_eq!(decoded, case.reply.as_bytes(), "response bytes changed");
    } else {
        assert_eq!(body, case.reply.as_bytes(), "response bytes changed");
    }
}

fn call(address: SocketAddr, packet: &[u8], case: &Case) -> io::Result<Timing> {
    let mut response = Vec::with_capacity(case.reply.len() + 1024);
    let started = Instant::now();
    let mut socket = TcpStream::connect_timeout(&address, TIMEOUT)?;
    socket.set_nodelay(true)?;
    socket.set_read_timeout(Some(TIMEOUT))?;
    socket.set_write_timeout(Some(TIMEOUT))?;
    socket.write_all(packet)?;
    let mut first = [0];
    socket.read_exact(&mut first)?;
    let first_ns = started.elapsed().as_nanos() as i128;
    response.push(first[0]);
    socket.read_to_end(&mut response)?;
    let full_ns = started.elapsed().as_nanos() as i128;
    // Validate after the timer. Never print a result for an incomplete reply.
    check_response(&response, case);
    Ok(Timing { first_ns, full_ns })
}

fn pair(round: usize, index: usize, case_index: usize, case: &Case, addresses: [SocketAddr; 2], packets: &[Vec<u8>; 2]) -> io::Result<Pair> {
    // Reverse the order each pair and each round to limit time-order bias.
    let (direct, gateway) = if (round + index + case_index).is_multiple_of(2) {
        (call(addresses[0], &packets[0], case)?, call(addresses[1], &packets[1], case)?)
    } else {
        let gateway = call(addresses[1], &packets[1], case)?;
        (call(addresses[0], &packets[0], case)?, gateway)
    };
    Ok(Pair { round, case: case_index, index, direct, gateway })
}

fn percentile(values: &mut [i128], percent: usize) -> i128 {
    values.sort_unstable();
    values[(values.len() * percent).div_ceil(100) - 1]
}

fn summary(writer: &mut impl Write, round: &str, name: &str, rows: &[&Pair]) -> io::Result<()> {
    let collect = |field: fn(&Pair) -> i128, percent| percentile(&mut rows.iter().map(|row| field(row)).collect::<Vec<_>>(), percent);
    writeln!(writer, "summary,{round},{name},{},{},{},{},{},{},{}", rows.len(),
        collect(|row| row.direct.full_ns, 50), collect(|row| row.gateway.full_ns, 50),
        collect(|row| row.gateway.full_ns - row.direct.full_ns, 50),
        collect(|row| row.gateway.full_ns - row.direct.full_ns, 95),
        collect(|row| row.gateway.first_ns - row.direct.first_ns, 50),
        collect(|row| row.gateway.full_ns, 50) - collect(|row| row.direct.full_ns, 50))
}

fn bench() -> io::Result<()> {
    let cases = cases();
    let mut mock = Mock::start()?;
    let (gateway, address) = Gateway::start(mock.address)?;
    let addresses = [mock.address, address];
    let packets: Vec<_> = cases.iter().enumerate().map(|(index, case)|
        [packet(case, index, addresses[0]), packet(case, index, addresses[1])]).collect();
    let mut rows = Vec::with_capacity(ROUNDS * PAIRS * cases.len());
    for round in 0..ROUNDS {
        for (case_index, case) in cases.iter().enumerate() {
            for index in 0..WARMUP_PAIRS { pair(round, index, case_index, case, addresses, &packets[case_index])?; }
            for index in 0..PAIRS { rows.push(pair(round, index, case_index, case, addresses, &packets[case_index])?); }
        }
    }
    drop(gateway);
    let served = mock.finish().expect("mock worker failed");
    assert_eq!(served, ROUNDS * cases.len() * (WARMUP_PAIRS + PAIRS) * 2);
    // Write all receipts after measurement. The child has no inherited environment.
    let mut writer = io::BufWriter::new(io::stdout().lock());
    writeln!(writer, "gateway_passthrough_v1 os={} arch={} debug_assertions={} parallelism={}", std::env::consts::OS, std::env::consts::ARCH, cfg!(debug_assertions), thread::available_parallelism()?.get())?;
    writeln!(writer, "settings rounds={ROUNDS} pairs_per_case_round={PAIRS} warmup_pairs_per_case_round={WARMUP_PAIRS} concurrency=1 mock_requests={served} checkpoint=none")?;
    writeln!(writer, "method=fresh_tcp; timer=before_connect_through_eof; first=first_http_byte; added=paired_gateway_minus_direct; percentile=nearest_rank; sse=no_delay; gateway=default_no_stats_no_record")?;
    for case in &cases { writeln!(writer, "case,{},request_body_bytes={},reply_body_bytes={},sse={}", case.name, case.body.len(), case.reply.len(), case.sse)?; }
    writeln!(writer, "type,round,case,pairs,direct_p50_ns,gateway_p50_ns,added_p50_ns,added_p95_ns,first_added_p50_ns,difference_of_p50_ns")?;
    for round in 0..ROUNDS {
        let selected: Vec<_> = rows.iter().filter(|row| row.round == round).collect();
        summary(&mut writer, &round.to_string(), "all", &selected)?;
    }
    for (index, case) in cases.iter().enumerate() {
        let selected: Vec<_> = rows.iter().filter(|row| row.case == index).collect();
        summary(&mut writer, "all", case.name, &selected)?;
    }
    summary(&mut writer, "all", "all", &rows.iter().collect::<Vec<_>>())?;
    writeln!(writer, "type,round,case,pair,direct_first_ns,direct_full_ns,gateway_first_ns,gateway_full_ns,added_full_ns")?;
    for row in &rows {
        writeln!(writer, "sample,{},{},{},{},{},{},{},{}", row.round, cases[row.case].name, row.index,
            row.direct.first_ns, row.direct.full_ns, row.gateway.first_ns, row.gateway.full_ns, row.gateway.full_ns - row.direct.full_ns)?;
    }
    writer.flush()
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--gateway-child") {
        assert_eq!(args.len(), 3, "invalid child arguments");
        return vade_harness::gateway::invisible::cli(["--bind".into(), args[1].clone(), "--upstream".into(), args[2].clone()].into_iter());
    }
    if args.iter().any(|arg| arg == "--bench") { return bench(); }
    println!("Run cargo bench --offline --locked -p vade-harness --bench gateway_passthrough to measure the relay.");
    Ok(())
}
