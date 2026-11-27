use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpListener,
    process::Command,
    sync::{Arc, atomic::Ordering},
    thread,
    time::Duration,
};

/// Exercise the product workload CLI against the same first-party v1
/// capabilities fixture as the workload_mock tests. Counts are invented fixture tokens,
/// not an Engine/tokenizer/performance measurement. Never opens a live URL.
pub fn run(
    binary: &std::path::Path,
    destination: &str,
    deadline: std::time::Instant,
) -> io::Result<()> {
    use std::sync::atomic::AtomicBool;
    use vade_harness::json;
    if std::path::Path::new(destination).try_exists()? {
        return Err(io::Error::other("choose a new mock JSON file"));
    }
    let listener = TcpListener::bind("127.0.0.1:0")?;
    listener.set_nonblocking(true)?;
    let endpoint = format!("http://{}", listener.local_addr()?);
    let stop = Arc::new(AtomicBool::new(false));
    let server_stop = Arc::clone(&stop);
    let server = thread::spawn(move || -> io::Result<()> {
        let mut workers = Vec::new();
        while workers.len() < 6 && !server_stop.load(Ordering::Relaxed) {
            let socket = match listener.accept() {
                Ok((socket, _)) => socket,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                }
                Err(e) => return Err(e),
            };
            workers.push(thread::spawn(move || -> io::Result<()> {
                socket.set_nonblocking(false)?;
                socket.set_nodelay(true)?;
                socket.set_read_timeout(Some(Duration::from_secs(5)))?;
                socket.set_write_timeout(Some(Duration::from_secs(5)))?;
                let mut reader = BufReader::new(socket);
                let mut turn = 0;
                let mut prompt = 0;
                let mut admitted_body = String::new();
                loop {
                    let mut first = String::new();
                    if reader.read_line(&mut first)? == 0 { break; }
                    let mut length = 0;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line)? == 0 {
                            return Err(io::Error::other("mock incomplete headers"));
                        }
                        if line == "\r\n" { break; }
                        if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                            length = n.trim().parse::<usize>().map_err(io::Error::other)?;
                        }
                    }
                    if length > 5 * 1024 * 1024 { return Err(io::Error::other("mock request limit")); }
                    let mut body = vec![0; length];
                    reader.read_exact(&mut body)?;
                    let body = String::from_utf8(body).map_err(io::Error::other)?;
                    let (kind, reply) = if first.starts_with("GET /v1/capabilities ") {
                        ("application/json", include_str!("../../tests/fixtures/engine-capabilities-v1.json").to_owned())
                    } else if first.starts_with("POST /v1/admission ") {
                        let envelope = json::view(&body)?;
                        admitted_body = envelope.get("request").ok_or_else(|| io::Error::other("mock missing request"))?.raw().into();
                        let v = json::view(&admitted_body)?;
                        let system = v.get("messages").and_then(json::View::array)
                            .and_then(|mut a| a.next()).and_then(|m| m.get("content"))
                            .and_then(json::View::as_str).ok_or_else(|| io::Error::other("mock missing system"))?;
                        let closed = v.get("messages").and_then(json::View::array)
                            .and_then(|a| a.last()).and_then(|m| m.get("role"))
                            .and_then(json::View::as_str).as_deref() == Some("assistant");
                        let counted = if closed { prompt + 3 } else {
                            prompt = (if system.contains("role main.") { 1000 } else { 100 }) + turn * 100;
                            prompt
                        };
                        ("application/json", format!(r#"{{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","prompt_tokens":{counted},"output_tokens":17,"context_limit":4096,"input_limit":4000,"output_limit":32,"admitted":true,"admission_kind":"context_fit","reserves_resources":false}}"#))
                    } else if first.starts_with("POST /v1/chat/completions ") {
                        if body != admitted_body || turn >= 3 { return Err(io::Error::other("mock generation differs from admission")); }
                        let cached = if turn == 0 { 0 } else { prompt - 100 + 3 };
                        turn += 1;
                        thread::sleep(Duration::from_millis(5));
                        ("text/event-stream", format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"synthetic answer\"}},\"finish_reason\":\"stop\"}}]}}\n\ndata: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{prompt},\"completion_tokens\":3,\"prompt_tokens_details\":{{\"cached_tokens\":{cached}}}}}}}\n\ndata: [DONE]\n\n"))
                    } else { return Err(io::Error::other("unexpected mock route")); };
                    write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\r\n{reply}", reply.len())?;
                }
                Ok(())
            }));
        }
        for worker in workers {
            worker
                .join()
                .map_err(|_| io::Error::other("mock worker failed"))??;
        }
        Ok(())
    });
    eprintln!("first-party mock only: invented token counts, 6 sessions × 3 turns");
    let mut command = Command::new(binary);
    command.args([
        "workload",
        "--endpoint",
        &endpoint,
        "--model",
        "mock",
        "--sessions",
        "6",
        "--turns",
        "3",
        "--output-tokens",
        "17",
        "--json",
        destination,
    ]);
    let result =
        super::process::OwnedChild::spawn(&mut command).and_then(|child| child.wait(deadline));
    stop.store(true, Ordering::Relaxed);
    server
        .join()
        .map_err(|_| io::Error::other("mock server failed"))??;
    result.map(|_| ())
}
