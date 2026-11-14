//! Plain HTTP transport and SSE response parsing for the benchmark protocols.
//!
//! Requests use a deadline and cancellation flag. Replies retain token IDs,
//! timing, usage and effective server settings for receipt validation.
use engine_model::json::{self, Value};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream, ToSocketAddrs},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
    },
    time::{Duration, Instant},
};

#[derive(Clone)]
pub struct Endpoint {
    address: SocketAddr,
    host: String,
    path: String,
}
impl Endpoint {
    pub fn parse(url: &str) -> Result<Self, String> {
        let rest = url
            .strip_prefix("http://")
            .ok_or("use an http:// URL (TLS via a local tunnel)")?;
        let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
        if host.is_empty() || host.contains(['\r', '\n', '@']) {
            return Err("invalid HTTP authority".into());
        }
        let authority = if host.starts_with('[') {
            if host.ends_with(']') {
                format!("{host}:80")
            } else {
                host.to_owned()
            }
        } else if host.contains(':') {
            host.to_owned()
        } else {
            format!("{host}:80")
        };
        let address = authority
            .to_socket_addrs()
            .map_err(|e| e.to_string())?
            .next()
            .ok_or("no server address")?;
        let path = match path.trim_end_matches('/') {
            "" => "/v1/chat/completions".into(),
            "v1" => "/v1/chat/completions".into(),
            p => format!("/{p}"),
        };
        if path.contains(['\r', '\n', ' ']) {
            return Err("invalid HTTP path".into());
        }
        Ok(Self {
            address,
            host: host.to_owned(),
            path,
        })
    }
}

#[derive(Clone, Default)]
pub struct Reply {
    pub status: String,
    pub error: Option<String>,
    pub accepted_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub elapsed_ms: f64,
    pub decode_tps: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub content_events: u64,
    pub token_ids: Vec<u32>,
    pub token_elapsed_ms: Vec<f64>,
    pub text: String,
    pub session: Option<String>,
    pub reused_tokens: Option<u64>,
    pub finish_reason: Option<String>,
    pub response_model: Option<String>,
    /// Effective server settings, retained verbatim in benchmark receipts.
    pub vade: Option<Value>,
}

impl Reply {
    pub fn verify_greedy(&self) -> Result<(), String> {
        let sampling = self.vade.as_ref().and_then(|v| get(v, "sampling"))
            .ok_or("missing vade.sampling echo")?;
        for (name, expected) in [("temperature", 0.0), ("top_p", 1.0),
            ("repetition_penalty", 1.0), ("presence_penalty", 0.0), ("frequency_penalty", 0.0)] {
            let actual = get(sampling, name).and_then(|v| v.f64().ok());
            if actual != Some(expected) {
                return Err(format!("non-greedy sampling echo: {name}={actual:?}, expected {expected}"));
            }
        }
        if !matches!(get(sampling, "top_k").and_then(|v| v.u64().ok()), Some(0 | 1)) {
            return Err("non-greedy sampling echo: top_k must be 0 or 1".into());
        }
        Ok(())
    }
}

struct TimedSocket {
    socket: TcpStream,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    complete: bool,
}
impl Drop for TimedSocket {
    fn drop(&mut self) {
        if !self.complete {
            // Vade permits write-half-close, so FIN alone cannot cancel prefill.
            // An extra body byte triggers its no-pipelining check on the next
            // scheduler tick, releasing the request before the next case.
            let _ = self
                .socket
                .set_write_timeout(Some(Duration::from_millis(200)));
            let _ = self.socket.write_all(b"\n");
        }
    }
}
impl Read for TimedSocket {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(io::Error::other("cancelled"));
            }
            let left = self.deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "request deadline"));
            }
            self.socket
                .set_read_timeout(Some(left.min(Duration::from_millis(200))))?;
            match self.socket.read(buf) {
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    continue;
                }
                result => return result,
            }
        }
    }
}

fn line<R: BufRead>(reader: &mut R) -> io::Result<Option<String>> {
    let mut bytes = Vec::new();
    reader.take(1024 * 1024 + 1).read_until(b'\n', &mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err(io::Error::other("HTTP/SSE line exceeds 1 MiB"));
    }
    if bytes.is_empty() {
        return Ok(None);
    }
    String::from_utf8(bytes).map(Some).map_err(io::Error::other)
}

// HTTP chunk boundaries need not coincide with SSE lines or UTF-8 characters.
struct Body {
    reader: BufReader<TimedSocket>,
    chunked: bool,
    remaining: usize,
    separator: bool,
    done: bool,
}
impl Read for Body {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if !self.chunked {
            return self.reader.read(out);
        }
        if self.done || out.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            if self.separator {
                let mut crlf = [0; 2];
                self.reader.read_exact(&mut crlf)?;
                if crlf != *b"\r\n" {
                    return Err(io::Error::other("invalid chunk separator"));
                }
            }
            let size =
                line(&mut self.reader)?.ok_or_else(|| io::Error::other("missing chunk size"))?;
            self.remaining = usize::from_str_radix(size.trim().split(';').next().unwrap_or(""), 16)
                .map_err(io::Error::other)?;
            if self.remaining == 0 {
                self.done = true;
                return Ok(0);
            }
            self.separator = true;
        }
        let cap = out.len().min(self.remaining);
        let n = self.reader.read(&mut out[..cap])?;
        if n == 0 {
            return Err(io::Error::other("truncated HTTP chunk"));
        }
        self.remaining -= n;
        Ok(n)
    }
}

pub fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.object().ok()?.get(key)
}
fn string(v: Option<&Value>) -> Option<String> {
    v?.string().ok().map(str::to_owned)
}
fn integer(v: Option<&Value>) -> Option<u64> {
    v?.u64().ok()
}

pub fn request(
    endpoint: &Endpoint,
    body: &str,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    accepted: Option<Sender<()>>,
) -> Reply {
    let start = Instant::now();
    let mut reply = Reply::default();
    let result = receive(
        endpoint,
        body,
        start,
        deadline,
        cancel.clone(),
        accepted,
        &mut reply,
    );
    reply.elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
    match result {
        Ok(()) => reply.status = "ok".into(),
        Err(e) => {
            reply.status = if cancel.load(Ordering::Relaxed) {
                "cancelled"
            } else if Instant::now() >= deadline {
                "timeout"
            } else {
                "error"
            }
            .into();
            reply.error = Some(e.to_string());
            reply.decode_tps = None;
        }
    }
    reply
}

fn receive(
    endpoint: &Endpoint,
    body: &str,
    start: Instant,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    accepted: Option<Sender<()>>,
    reply: &mut Reply,
) -> io::Result<()> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(io::Error::new(io::ErrorKind::TimedOut, "request deadline"));
    }
    let mut socket =
        TcpStream::connect_timeout(&endpoint.address, left.min(Duration::from_secs(5)))?;
    socket.set_nodelay(true)?;
    socket.set_write_timeout(Some(left))?;
    write!(
        socket,
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
        endpoint.path,
        endpoint.host,
        body.len(),
        body
    )?;
    let mut reader = BufReader::new(TimedSocket {
        socket,
        deadline,
        cancel,
        complete: false,
    });
    let status = line(&mut reader)?.ok_or_else(|| io::Error::other("missing HTTP status"))?;
    let code = status.split_whitespace().nth(1).unwrap_or("").to_owned();
    let mut chunked = false;
    let mut header_bytes = status.len();
    loop {
        let l = line(&mut reader)?.ok_or_else(|| io::Error::other("truncated headers"))?;
        header_bytes += l.len();
        if header_bytes > 65536 {
            return Err(io::Error::other("headers exceed 64 KiB"));
        }
        if l.trim().is_empty() {
            break;
        }
        if let Some((key, value)) = l.split_once(':') {
            let value = value.trim();
            match key.to_ascii_lowercase().as_str() {
                "transfer-encoding" => {
                    if !value.eq_ignore_ascii_case("chunked") {
                        return Err(io::Error::other("unsupported transfer encoding"));
                    }
                    chunked = true;
                }
                "x-session-capability" => reply.session = Some(value.into()),
                "x-prefix-reused-tokens" => reply.reused_tokens = value.parse().ok(),
                _ => (),
            }
        }
    }
    let mut reader = BufReader::new(Body {
        reader,
        chunked,
        remaining: 0,
        separator: false,
        done: false,
    });
    if code != "200" {
        let mut error = String::new();
        reader.take(4096).read_to_string(&mut error)?;
        return Err(io::Error::other(format!("HTTP {code}: {error}")));
    }
    if let Some(tx) = accepted {
        let _ = tx.send(());
    }
    reply.accepted_ms = Some(start.elapsed().as_secs_f64() * 1000.);
    let mut data = String::new();
    let mut finished_ms = None;
    let mut first_content_tokens = 0;
    loop {
        let l = line(&mut reader)?.ok_or_else(|| io::Error::other("SSE ended without [DONE]"))?;
        if let Some(value) = l.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(
                value
                    .strip_prefix(' ')
                    .unwrap_or(value)
                    .trim_end_matches(['\r', '\n']),
            );
            if data.len() > 1024 * 1024 {
                return Err(io::Error::other("SSE event exceeds 1 MiB"));
            }
        }
        if !l.trim().is_empty() || data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            reader.get_mut().reader.get_mut().complete = true;
            break;
        }
        let v = json::parse(data.as_bytes(), json::Limits::default()).map_err(io::Error::other)?;
        data.clear();
        let now = start.elapsed().as_secs_f64() * 1000.;
        if let Some(error) = get(&v, "error") {
            return Err(io::Error::other(format!("SSE error: {error:?}")));
        }
        if let Some(model) = string(get(&v, "model")) {
            reply.response_model = Some(model);
        }
        if let Some(choices) = get(&v, "choices").and_then(|v| v.array().ok()) {
            for choice in choices {
                if let Some(id) = integer(get(choice, "token_id")) {
                    reply.token_elapsed_ms.push(now);
                    reply
                        .token_ids
                        .push(u32::try_from(id).map_err(io::Error::other)?);
                }
                let content = get(choice, "delta")
                    .and_then(|d| get(d, "content"))
                    .and_then(|v| v.string().ok())
                    .or_else(|| get(choice, "text").and_then(|v| v.string().ok()));
                if let Some(content) = content.filter(|s| !s.is_empty()) {
                    reply.content_events += 1;
                    if reply.ttft_ms.is_none() {
                        reply.ttft_ms = Some(now);
                        first_content_tokens = reply.token_ids.len().max(1) as u64;
                    }
                    reply.text.push_str(content);
                    if reply.text.len() > 4 * 1024 * 1024 {
                        return Err(io::Error::other("completion exceeds 4 MiB"));
                    }
                }
                if let Some(reason) = string(get(choice, "finish_reason")) {
                    reply.finish_reason = Some(reason);
                    finished_ms = Some(now);
                }
            }
        }
        if let Some(usage) = get(&v, "usage").filter(|v| **v != Value::Null) {
            reply.vade = get(&v, "vade").cloned();
            reply.prompt_tokens = integer(get(usage, "prompt_tokens"));
            reply.completion_tokens = integer(get(usage, "completion_tokens"));
            reply.total_tokens = integer(get(usage, "total_tokens"));
            reply.cached_tokens = get(usage, "prompt_tokens_details")
                .and_then(|v| integer(get(v, "cached_tokens")));
            if finished_ms.is_none() {
                finished_ms = Some(now);
            }
        }
    }
    let tokens = reply.completion_tokens.ok_or_else(|| {
        io::Error::other("missing completion usage; SSE chunks are not token counts")
    })?;
    let first = reply
        .ttft_ms
        .ok_or_else(|| io::Error::other("no nonempty content delta"))?;
    let last = finished_ms.ok_or_else(|| io::Error::other("missing finish event"))?;
    if reply.vade.is_some() {
        reply.verify_greedy().map_err(io::Error::other)?;
    }
    if tokens > first_content_tokens && last > first {
        reply.decode_tps = Some((tokens - first_content_tokens) as f64 * 1000. / (last - first));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::TcpListener, thread};
    const GREEDY_ECHO: &str = r#"{"sampling":{"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0}}"#;

    #[test]
    fn verify_greedy_rejects_missing_and_non_greedy_echo() {
        assert_eq!(Reply::default().verify_greedy().unwrap_err(), "missing vade.sampling echo");
        let reply = |s: &str| Reply { vade: Some(json::parse(s.as_bytes(), Default::default()).unwrap()), ..Default::default() };
        assert!(reply(GREEDY_ECHO).verify_greedy().is_ok());
        assert!(reply(&GREEDY_ECHO.replace("\"top_k\":0", "\"top_k\":1")).verify_greedy().is_ok());
        for (old, new) in [("\"temperature\":0", "\"temperature\":1"), ("\"top_k\":0", "\"top_k\":20"), ("\"top_p\":1", "\"top_p\":0.95"), ("\"repetition_penalty\":1", "\"repetition_penalty\":1.2"), ("\"presence_penalty\":0", "\"presence_penalty\":0.4"), ("\"frequency_penalty\":0", "\"frequency_penalty\":0.1"), ("\"temperature\":0,", ""), ("\"sampling\"", "\"other\"")] {
            assert!(reply(&GREEDY_ECHO.replace(old, new)).verify_greedy().is_err(), "{new}");
        }
    }

    #[test]
    fn streamed_content_usage_chunking_and_deadline() {
        for (chunked, echo) in [(false, true), (true, true), (false, false)] {
            let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
            let endpoint =
                Endpoint::parse(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
            let worker = thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                let mut request = BufReader::new(socket.try_clone().unwrap());
                while !line(&mut request).unwrap().unwrap().trim().is_empty() {}
                let mut body = [0; 2];
                request.read_exact(&mut body).unwrap();
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\n{}X-Prefix-Reused-Tokens: 12\r\n\r\n",
                    if chunked {
                        "Transfer-Encoding: chunked\r\n"
                    } else {
                        ""
                    }
                )
                .unwrap();
                let finish = format!("data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":20,\"completion_tokens\":4}}{}}}\n\n", if echo { format!(",\"vade\":{GREEDY_ECHO}") } else { String::new() });
                let events = [
                    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
                    "data: {\"model\":\"qwen3-8b\",\"choices\":[{\"delta\":{\"content\":\"café\"},\"token_id\":10}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{},\"token_id\":11}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"!\"},\"token_id\":12}]}\n\n",
                    &finish,
                    "data: [DONE]\n\n",
                ];
                for event in events {
                    thread::sleep(Duration::from_millis(12));
                    if chunked {
                        for piece in event.as_bytes().chunks(7) {
                            write!(socket, "{:x}\r\n", piece.len()).unwrap();
                            socket.write_all(piece).unwrap();
                            socket.write_all(b"\r\n").unwrap();
                        }
                    } else {
                        socket.write_all(event.as_bytes()).unwrap();
                    }
                }
                if chunked {
                    let _ = socket.write_all(b"0\r\n\r\n"); // the client stops at [DONE] and may already have closed
                }
            });
            let reply = request(
                &endpoint,
                "{}",
                Instant::now() + Duration::from_secs(3),
                Arc::new(AtomicBool::new(false)),
                None,
            );
            worker.join().unwrap();
            assert_eq!(reply.vade.is_some(), echo);
            assert_eq!(reply.status, "ok", "{:?}", reply.error);
            assert_eq!(reply.text, "café!");
            assert_eq!(reply.completion_tokens, Some(4));
            assert_eq!(reply.content_events, 2);
            assert_eq!(reply.token_ids, [10, 11, 12]);
            assert_eq!(reply.reused_tokens, Some(12));
            assert!(reply.ttft_ms.unwrap() >= 20.);
            assert!(reply.decode_tps.unwrap() > 0.);
            // Four usage tokens (including EOS), two content events: using
            // event count as tokens would put this product below two.
            assert!(
                reply.decode_tps.unwrap() * (reply.elapsed_ms - reply.ttft_ms.unwrap()) / 1000.
                    >= 3.
            );
            let expired = request(
                &endpoint,
                "{}",
                Instant::now(),
                Arc::new(AtomicBool::new(false)),
                None,
            );
            assert_eq!(expired.status, "timeout");
            assert!(expired.decode_tps.is_none());
        }
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let endpoint =
            Endpoint::parse(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
        let worker = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            while !line(&mut reader).unwrap().unwrap().trim().is_empty() {}
            reader.read_exact(&mut [0; 2]).unwrap();
            socket.write_all(b"HTTP/1.1 200 OK\r\n\r\n").unwrap();
            let mut cancel = [0];
            reader.read_exact(&mut cancel).unwrap();
            assert_eq!(cancel, *b"\n");
        });
        let reply = request(
            &endpoint,
            "{}",
            Instant::now() + Duration::from_millis(100),
            Arc::new(AtomicBool::new(false)),
            None,
        );
        worker.join().unwrap();
        assert_eq!(reply.status, "timeout");
        assert!(reply.accepted_ms.is_some());
        assert!(reply.ttft_ms.is_none());
    }
}
