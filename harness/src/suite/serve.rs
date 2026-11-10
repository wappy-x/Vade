//! Loopback model endpoint for one task: a recording proxy to a real server, or
//! a replay of a transcript. One request per connection; every reply says
//! `Connection: close`, so the harness never reuses a socket we closed.
use super::{fnv, n, obj, s, u64_of, write};
use crate::json::{self, Value};
use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const MAX_BODY: usize = 64 << 20;
/// Tool calls in one reply; a higher index is not a real call.
const MAX_CALLS: usize = 256;
const RECORD_CAP: usize = 16 << 20;

#[derive(Clone)]
pub struct Reply { pub status: u16, pub content_type: String, pub body: String }

pub enum Source { Upstream(String), Replay(Mutex<VecDeque<Reply>>) }

/// One chat-completions exchange, as the proxy saw it.
#[derive(Clone, Debug, Default)]
pub struct Exchange {
    pub step: usize,
    pub call: usize,
    pub status: u16,
    pub ms: u128,
    pub held: bool,
    pub client_closed: bool,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    /// The streamed `token_id` values of the reply, in order.
    pub token_ids: Vec<u64>,
    /// A digest of the request without the fields that change on every run
    /// (see `request_digest`).
    pub request: u64,
    /// The tool calls of the reply: name and raw arguments.
    pub calls: Vec<(String, String)>,
    /// The tool results at the end of the request, in order: the results of
    /// the calls that the previous reply of the same step made.
    pub results: Vec<String>,
}

pub struct Shared {
    pub source: Source,
    pub step: AtomicUsize,
    pub step_calls: AtomicUsize,
    pub hold_at: Mutex<Option<(usize, Sender<()>)>>,
    pub exchanges: Mutex<Vec<Exchange>>,
    pub record: Mutex<Option<std::fs::File>>,
    pub stop: AtomicBool,
    /// Expand script tags and replay as an SSE stream, like the engine.
    pub sse: AtomicBool,
    /// The task workspace, for `{{edit_tag PATH}}` in replayed replies.
    pub work: std::path::PathBuf,
    /// Connection handlers that have not ended. A handler records its exchange
    /// after the client has the reply, so a client can exit first.
    pub busy: AtomicUsize,
    /// Set when `exchanges` stopped waiting for a handler: the list may lack a call.
    pub wait_expired: AtomicBool,
}

/// Ends one count of `Shared::busy` when the handler ends, also by a panic.
struct Busy(Arc<Shared>);

impl Drop for Busy {
    fn drop(&mut self) { self.0.busy.fetch_sub(1, Ordering::AcqRel); }
}

/// Runs `f` on its own thread and counts it in `busy` until it ends.
fn track(s: &Arc<Shared>, f: impl FnOnce(&Shared) + Send + 'static) {
    s.busy.fetch_add(1, Ordering::AcqRel);
    let guard = Busy(s.clone());
    std::thread::spawn(move || f(&guard.0));
}

/// How long `exchanges` waits for the handlers that are still running.
const HANDLER_WAIT: Duration = Duration::from_secs(5);

pub struct Server { pub url: String, pub shared: Arc<Shared>, thread: Option<std::thread::JoinHandle<()>> }

impl Server {
    pub fn start(source: Source, record: Option<std::fs::File>, sse: bool, work: &std::path::Path) -> io::Result<Server> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}/v1/chat/completions", listener.local_addr()?);
        listener.set_nonblocking(true)?;
        let shared = Arc::new(Shared {
            source, step: AtomicUsize::new(0), step_calls: AtomicUsize::new(0), hold_at: Mutex::new(None),
            exchanges: Mutex::new(Vec::new()), record: Mutex::new(record), stop: AtomicBool::new(false), sse: AtomicBool::new(sse), work: work.to_path_buf(),
            busy: AtomicUsize::new(0), wait_expired: AtomicBool::new(false),
        });
        let s = shared.clone();
        let thread = std::thread::spawn(move || {
            while !s.stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => track(&s, move |s| { let _ = handle(s, stream); }),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(5)),
                    Err(_) => std::thread::sleep(Duration::from_millis(50)),
                }
            }
        });
        Ok(Server { url, shared, thread: Some(thread) })
    }
    pub fn begin_step(&self, step: usize, hold: Option<(usize, Sender<()>)>) {
        self.shared.step.store(step, Ordering::Release);
        self.shared.step_calls.store(0, Ordering::Release);
        *self.shared.hold_at.lock().unwrap() = hold;
    }
    /// The exchanges so far, after the running handlers end (at most
    /// HANDLER_WAIT), so that the last call of an exited client is in the list.
    pub fn exchanges(&self) -> Vec<Exchange> { self.exchanges_within(HANDLER_WAIT) }
    fn exchanges_within(&self, wait: Duration) -> Vec<Exchange> {
        let start = Instant::now();
        while self.shared.busy.load(Ordering::Acquire) > 0 {
            if start.elapsed() >= wait { self.shared.wait_expired.store(true, Ordering::Release); break; }
            std::thread::sleep(Duration::from_millis(2));
        }
        self.shared.exchanges.lock().unwrap().clone()
    }
    /// True when a call to `exchanges` stopped waiting for a running handler.
    pub fn wait_expired(&self) -> bool { self.shared.wait_expired.load(Ordering::Acquire) }
    pub fn remaining_replies(&self) -> Option<usize> {
        match &self.shared.source { Source::Replay(q) => Some(q.lock().unwrap().len()), Source::Upstream(_) => None }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() { let _ = t.join(); }
    }
}

struct Request { method: String, path: String, headers: Vec<(String, String)>, body: Vec<u8> }

fn read_request(r: &mut BufReader<TcpStream>) -> io::Result<Request> {
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    if method.is_empty() { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "empty request")); }
    let headers = read_headers(r)?;
    let body = read_body(r, &headers, true)?;
    Ok(Request { method, path, headers, body })
}

fn read_headers(r: &mut impl BufRead) -> io::Result<Vec<(String, String)>> {
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "headers cut off")); }
        let h = h.trim_end();
        if h.is_empty() { return Ok(headers); }
        if let Some((k, v)) = h.split_once(':') { headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string())); }
    }
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
}

/// Reads a body framed by Content-Length or chunks. A request with neither has no body.
fn read_body(r: &mut impl BufRead, headers: &[(String, String)], request: bool) -> io::Result<Vec<u8>> {
    if let Some(n) = header(headers, "content-length") {
        let n: usize = n.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad length"))?;
        if n > MAX_BODY { return Err(io::Error::new(io::ErrorKind::InvalidData, "body too large")); }
        let mut b = vec![0; n];
        r.read_exact(&mut b)?;
        return Ok(b);
    }
    if header(headers, "transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked")) {
        let mut out = Vec::new();
        while let Some(chunk) = read_chunk(r)? {
            out.extend_from_slice(&chunk);
            if out.len() > MAX_BODY { return Err(io::Error::new(io::ErrorKind::InvalidData, "body too large")); }
        }
        return Ok(out);
    }
    let mut out = Vec::new();
    if !request { r.take(MAX_BODY as u64).read_to_end(&mut out)?; }
    Ok(out)
}

fn read_chunk(r: &mut impl BufRead) -> io::Result<Option<Vec<u8>>> {
    let mut size = String::new();
    if r.read_line(&mut size)? == 0 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "chunk cut off")); }
    let n = usize::from_str_radix(size.trim().split(';').next().unwrap_or(""), 16)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad chunk size"))?;
    if n == 0 {
        loop { let mut t = String::new(); if r.read_line(&mut t)? == 0 || t.trim().is_empty() { break; } }
        return Ok(None);
    }
    if n > MAX_BODY { return Err(io::Error::new(io::ErrorKind::InvalidData, "chunk too large")); }
    let mut b = vec![0; n + 2];
    r.read_exact(&mut b)?;
    b.truncate(n);
    Ok(Some(b))
}

fn respond(out: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
    let reason = match status { 200 => "OK", 404 => "Not Found", _ => "Error" };
    write!(out, "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())?;
    out.write_all(body)?;
    out.flush()
}

fn handle(s: &Shared, stream: TcpStream) -> io::Result<()> {
    // On macOS an accepted socket inherits the listener's nonblocking flag.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(600)))?;
    let mut out = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    let req = read_request(&mut reader)?;
    let chat = req.method == "POST" && req.path.ends_with("/chat/completions");
    let started = Instant::now();
    let mut ex = Exchange { step: s.step.load(Ordering::Acquire), ..Default::default() };
    if chat {
        ex.call = s.step_calls.fetch_add(1, Ordering::AcqRel) + 1;
        let hold = s.hold_at.lock().unwrap().as_ref().filter(|(at, _)| *at == ex.call).map(|(_, tx)| tx.clone());
        if let Some(tx) = hold {
            // Hold the reply and tell the runner; the harness cancels by closing the socket.
            let _ = tx.send(());
            ex.held = true;
            ex.results = tool_results(&req.body);
            ex.client_closed = wait_closed(&mut reader, Duration::from_secs(180));
            ex.ms = started.elapsed().as_millis();
            record(s, &req, &ex, "", "");
            s.exchanges.lock().unwrap().push(ex);
            return Ok(());
        }
    }
    let (status, ctype, body) = match &s.source {
        Source::Replay(queue) => {
            if !chat {
                (404, "application/json".to_string(), b"{}".to_vec())
            } else if let Some(mut reply) = queue.lock().unwrap().pop_front() {
                let scripted = s.sse.load(Ordering::Acquire);
                if scripted { reply.body = fill_tags(&reply.body, &s.work); }
                let streamed = !String::from_utf8_lossy(&req.body).contains("\"stream\":false");
                if scripted && streamed && reply.content_type == "application/json" {
                    // Stream like the engine: SSE, Connection: close, the body ends at EOF.
                    let events = to_sse(&reply.body, req.body.len());
                    let _ = out.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
                    for e in &events {
                        if out.write_all(e.as_bytes()).and_then(|_| out.flush()).is_err() { ex.client_closed = true; break; }
                        std::thread::sleep(Duration::from_millis(3));
                    }
                    ex.status = 200;
                    ex.ms = started.elapsed().as_millis();
                    let text = events.concat();
                    inspect(&mut ex, &req.body, &text);
                    record(s, &req, &ex, "text/event-stream", &text);
                    s.exchanges.lock().unwrap().push(ex);
                    let _ = out.shutdown(Shutdown::Both);
                    return Ok(());
                }
                (reply.status, reply.content_type, reply.body.into_bytes())
            } else {
                (500, "application/json".to_string(), br#"{"error":{"message":"replay transcript has no more replies"}}"#.to_vec())
            }
        }
        Source::Upstream(addr) => match relay(addr, &req, &mut out) {
            Ok((status, ctype, body, closed)) => {
                ex.client_closed = closed;
                ex.status = status;
                ex.ms = started.elapsed().as_millis();
                let text = String::from_utf8_lossy(&body).to_string();
                if chat {
                    inspect(&mut ex, &req.body, &text);
                    record(s, &req, &ex, &ctype, &text);
                    s.exchanges.lock().unwrap().push(ex);
                }
                let _ = out.shutdown(Shutdown::Both);
                return Ok(());
            }
            Err(e) => (502, "application/json".to_string(), format!("{{\"error\":{{\"message\":{}}}}}", json::quote(&format!("suite proxy: {e}"))).into_bytes()),
        },
    };
    ex.status = status;
    ex.client_closed = respond(&mut out, status, &ctype, &body).is_err();
    ex.ms = started.elapsed().as_millis();
    if chat {
        let text = String::from_utf8_lossy(&body).to_string();
        inspect(&mut ex, &req.body, &text);
        record(s, &req, &ex, &ctype, &text);
        s.exchanges.lock().unwrap().push(ex);
    }
    let _ = out.shutdown(Shutdown::Both);
    Ok(())
}

fn wait_closed(r: &mut BufReader<TcpStream>, limit: Duration) -> bool {
    let start = Instant::now();
    let _ = r.get_ref().set_read_timeout(Some(Duration::from_millis(200)));
    let mut b = [0; 512];
    while start.elapsed() < limit {
        match r.read(&mut b) {
            Ok(0) => return true,
            Ok(_) => {}
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
            Err(_) => return true,
        }
    }
    false
}

/// Forwards one request upstream and streams the reply back, re-framed for a
/// closing connection. Returns status, content type, the decoded body (capped)
/// and whether the client went away first.
fn relay(addr: &str, req: &Request, client: &mut TcpStream) -> io::Result<(u16, String, Vec<u8>, bool)> {
    let mut up = TcpStream::connect(addr)?;
    up.set_read_timeout(Some(Duration::from_secs(900)))?;
    let mut head = format!("{} {} HTTP/1.1\r\nHost: {addr}\r\n", req.method, req.path);
    for (k, v) in &req.headers {
        if !matches!(k.as_str(), "host" | "connection" | "content-length" | "transfer-encoding" | "keep-alive") {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
    }
    head.push_str(&format!("Connection: close\r\nContent-Length: {}\r\n\r\n", req.body.len()));
    up.write_all(head.as_bytes())?;
    up.write_all(&req.body)?;
    let mut r = BufReader::new(up);
    let mut status_line = String::new();
    r.read_line(&mut status_line)?;
    let status: u16 = status_line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let headers = read_headers(&mut r)?;
    let ctype = header(&headers, "content-type").unwrap_or("").to_string();
    let chunked = header(&headers, "transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
    let length = header(&headers, "content-length").and_then(|v| v.parse::<usize>().ok());
    let mut out = format!("{}\r\n", status_line.trim_end());
    for (k, v) in &headers {
        if !matches!(k.as_str(), "connection" | "content-length" | "transfer-encoding" | "keep-alive") {
            out.push_str(&format!("{k}: {v}\r\n"));
        }
    }
    out.push_str("Connection: close\r\n");
    if chunked { out.push_str("Transfer-Encoding: chunked\r\n"); } else if let Some(n) = length { out.push_str(&format!("Content-Length: {n}\r\n")); }
    out.push_str("\r\n");
    let mut body = Vec::new();
    let mut closed = client.write_all(out.as_bytes()).is_err();
    let keep = |piece: &[u8], body: &mut Vec<u8>| if body.len() < RECORD_CAP { body.extend_from_slice(&piece[..piece.len().min(RECORD_CAP - body.len())]); };
    if chunked {
        while !closed {
            match read_chunk(&mut r)? {
                Some(c) => {
                    keep(&c, &mut body);
                    closed = client.write_all(format!("{:x}\r\n", c.len()).as_bytes()).and_then(|_| client.write_all(&c)).and_then(|_| client.write_all(b"\r\n")).and_then(|_| client.flush()).is_err();
                }
                None => { closed = client.write_all(b"0\r\n\r\n").is_err(); break; }
            }
        }
    } else {
        let mut left = length.unwrap_or(usize::MAX);
        let mut buf = vec![0; 64 << 10];
        while left > 0 && !closed {
            let n = r.read(&mut buf[..left.min(64 << 10)])?;
            if n == 0 { break; }
            left = left.saturating_sub(n);
            keep(&buf[..n], &mut body);
            closed = client.write_all(&buf[..n]).and_then(|_| client.flush()).is_err();
        }
    }
    Ok((status, ctype, body, closed))
}

/// The last reported usage numbers in a JSON or SSE body.
pub fn usage(body: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    let Some(at) = body.rfind("\"usage\"") else { return (None, None, None) };
    let tail = &body[at..];
    let num = |key: &str| {
        let i = tail.find(&format!("\"{key}\""))?;
        let rest = tail[i + key.len() + 2..].trim_start().strip_prefix(':')?.trim_start();
        rest.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
    };
    (num("prompt_tokens"), num("completion_tokens"), num("cached_tokens"))
}

fn record(shared: &Shared, req: &Request, ex: &Exchange, ctype: &str, response: &str) {
    let Ok(mut guard) = shared.record.lock() else { return };
    let Some(file) = guard.as_mut() else { return };
    let line = obj(vec![
        ("step", n(ex.step)), ("call", n(ex.call)), ("method", s(&req.method)), ("path", s(&req.path)),
        ("status", n(ex.status)), ("ms", n(ex.ms)), ("held", Value::Bool(ex.held)), ("client_closed", Value::Bool(ex.client_closed)),
        ("content_type", s(ctype)), ("request", s(&String::from_utf8_lossy(&req.body))), ("response", s(response)),
    ]);
    let _ = writeln!(file, "{}", write(&line));
}

/// Fills the usage, token IDs, tool calls and tool results of one exchange.
fn inspect(ex: &mut Exchange, request: &[u8], response: &str) {
    (ex.prompt_tokens, ex.completion_tokens, ex.cached_tokens) = usage(response);
    ex.token_ids = token_ids(response);
    ex.request = request_digest(request);
    ex.calls = tool_calls(response);
    ex.results = tool_results(request);
}

/// A digest of a chat request without its per-run protocol fields: the
/// `prompt_cache_key` and `cancel` tokens, and the tool call IDs that the
/// server makes up for each reply. Two runs that send the same digest send
/// the model the same conversation.
pub fn request_digest(request: &[u8]) -> u64 {
    fn clean(v: &mut Value) {
        match v {
            Value::Object(m) => {
                m.retain(|(k, _)| k != "prompt_cache_key" && k != "cancel");
                for (k, x) in m.iter_mut() {
                    if k == "tool_call_id" || (k == "id" && x.as_str().is_some_and(|s| s.starts_with("call"))) { *x = Value::Null; } else { clean(x); }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(clean),
            _ => {}
        }
    }
    let Ok(mut body) = json::parse_bytes(request) else { return fnv(request.iter().map(|b| u64::from(*b))) };
    clean(&mut body);
    fnv(write(&body).bytes().map(u64::from))
}

/// The text of the `tool` messages at the end of a chat request.
pub fn tool_results(request: &[u8]) -> Vec<String> {
    let Ok(body) = json::parse_bytes(request) else { return Vec::new() };
    let messages = body.get("messages").and_then(Value::as_array).unwrap_or(&[]);
    let tail = messages.iter().rev().take_while(|m| m.get("role").and_then(Value::as_str) == Some("tool")).collect::<Vec<_>>();
    tail.into_iter().rev().map(|m| match m.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }).collect()
}

/// The tool calls of a reply, from a JSON body or from SSE deltas joined by index.
pub fn tool_calls(response: &str) -> Vec<(String, String)> {
    let mut calls: Vec<(String, String)> = Vec::new();
    let mut add = |list: &[Value], delta: bool| for (k, call) in list.iter().enumerate() {
        let i = call.get("index").and_then(u64_of).map_or(k, |i| usize::try_from(i).unwrap_or(usize::MAX));
        if i >= MAX_CALLS { continue; }
        if calls.len() <= i { calls.resize(i + 1, (String::new(), String::new())); }
        let f = call.get("function");
        let (name, args) = (f.and_then(|f| f.get("name")).and_then(Value::as_str).unwrap_or(""), f.and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or(""));
        if delta { calls[i].0.push_str(name); calls[i].1.push_str(args); } else { calls[i] = (name.into(), args.into()); }
    };
    if let Ok(v) = json::parse(response.trim()) {
        for choice in v.get("choices").and_then(Value::as_array).unwrap_or(&[]) {
            add(choice.get("message").and_then(|m| m.get("tool_calls")).and_then(Value::as_array).unwrap_or(&[]), false);
        }
        return calls;
    }
    for line in response.lines() {
        let Some(Ok(event)) = line.strip_prefix("data:").map(|d| json::parse(d.trim())) else { continue };
        for choice in event.get("choices").and_then(Value::as_array).unwrap_or(&[]) {
            add(choice.get("delta").and_then(|d| d.get("tool_calls")).and_then(Value::as_array).unwrap_or(&[]), true);
        }
    }
    calls
}

/// Replaces each `{{edit_tag PATH}}` in a replayed reply with the edit tag
/// that `workspace_read_lines` gives for the file as it is now. A scripted
/// model copies the tag from the read result, as a real model does.
pub fn fill_tags(body: &str, work: &std::path::Path) -> String {
    let mut out = String::new();
    let mut rest = body;
    while let Some(i) = rest.find("{{edit_tag ") {
        let Some(len) = rest[i..].find("}}") else { break };
        let path = rest[i + 11..i + len].trim();
        out.push_str(&rest[..i]);
        out.push_str(&crate::line_edit::tag(path, &std::fs::read_to_string(work.join(path)).unwrap_or_default()));
        rest = &rest[i + len + 2..];
    }
    out.push_str(rest);
    out
}

/// The `token_id` values of every streamed choice in an SSE body, in order.
/// The engine sends one per chunk; a plain JSON reply has none.
pub fn token_ids(body: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
        let Ok(event) = json::parse(data) else { continue };
        for choice in event.get("choices").and_then(Value::as_array).unwrap_or(&[]) {
            if let Some(id) = choice.get("token_id").and_then(u64_of) { out.push(id); }
        }
    }
    out
}

/// Splits one chat completion into SSE chunk events, ending with a usage event
/// and `[DONE]`, as the engine streams. Stub usage numbers are byte counts
/// divided by 4, and each stub `token_id` is the byte length of its piece:
/// fixed numbers, not tokens.
pub fn to_sse(body: &str, request_bytes: usize) -> Vec<String> {
    let v = json::parse(body).unwrap_or(Value::Null);
    let choice = v.get("choices").and_then(Value::as_array).and_then(|c| c.first()).cloned().unwrap_or(Value::Null);
    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    let finish = choice.get("finish_reason").cloned().unwrap_or(Value::Null);
    let chunk = |delta: Value, finish: Value, token: usize| format!("data: {}\n\n", write(&obj(vec![
        ("id", s("stub")), ("object", s("chat.completion.chunk")), ("model", s("replay")),
        ("choices", Value::Array(vec![obj(vec![("index", n(0)), ("delta", delta), ("finish_reason", finish), ("token_id", n(token))])])),
    ])));
    let halves = |t: &str| { let mut i = t.len() / 2; while !t.is_char_boundary(i) { i += 1; } (t[..i].to_string(), t[i..].to_string()) };
    let mut out = vec![chunk(obj(vec![("role", s("assistant"))]), Value::Null, 0)];
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        let (a, b) = halves(text);
        for part in [a, b] { out.push(chunk(obj(vec![("content", s(&part))]), Value::Null, part.len())); }
    }
    for (i, call) in message.get("tool_calls").and_then(Value::as_array).unwrap_or(&[]).iter().enumerate() {
        let f = call.get("function").cloned().unwrap_or(Value::Null);
        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
        let (a, b) = halves(f.get("arguments").and_then(Value::as_str).unwrap_or(""));
        out.push(chunk(obj(vec![("tool_calls", Value::Array(vec![obj(vec![("index", n(i)),
            ("id", call.get("id").cloned().unwrap_or(Value::Null)), ("type", s("function")),
            ("function", obj(vec![("name", s(name)), ("arguments", s(&a))]))])]))]), Value::Null, a.len()));
        out.push(chunk(obj(vec![("tool_calls", Value::Array(vec![obj(vec![("index", n(i)),
            ("function", obj(vec![("arguments", s(&b))]))])]))]), Value::Null, b.len()));
    }
    out.push(chunk(obj(vec![]), finish, 0));
    let (p, c) = (request_bytes / 4, body.len() / 4);
    out.push(format!("data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{p},\"completion_tokens\":{c},\"total_tokens\":{}}}}}\n\n", p + c));
    out.push("data: [DONE]\n\n".into());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn usage_reads_last_object() {
        let sse = "data: {\"choices\":[]}\n\ndata: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3}}\n\ndata: {\"usage\":{\"prompt_tokens\": 812,\"completion_tokens\":41,\"prompt_tokens_details\":{\"cached_tokens\":768}}}\n\ndata: [DONE]\n";
        assert_eq!(usage(sse), (Some(812), Some(41), Some(768)));
        assert_eq!(usage("{}"), (None, None, None));
    }
    #[test]
    fn stub_stream_carries_token_ids() {
        let body = super::super::task::answer_reply(1, "abcd");
        let events = to_sse(&body, 40).concat();
        assert_eq!(token_ids(&events), vec![0, 2, 2, 0]);
        assert_eq!(usage(&events), (Some(10), Some(body.len() as u64 / 4), None));
        assert!(token_ids(&body).is_empty());
    }
    #[test]
    fn calls_and_results_from_both_reply_forms() {
        let body = super::super::task::parse_script(">>> call workspace_read\n{\"path\":\"a.rs\"}\n").unwrap().remove(0).body;
        let want = vec![("workspace_read".to_string(), r#"{"path":"a.rs"}"#.to_string())];
        assert_eq!(tool_calls(&body), want);
        assert_eq!(tool_calls(&to_sse(&body, 10).concat()), want);
        assert!(tool_calls(&to_sse(&super::super::task::answer_reply(1, "x"), 10).concat()).is_empty());
        let req = br#"{"messages":[{"role":"user","content":"u"},{"role":"tool","content":"one"},{"role":"tool","content":[{"type":"text","text":"two"}]}]}"#;
        assert_eq!(tool_results(req), vec!["one", "two"]);
        let a = br#"{"prompt_cache_key":"k1","cancel":"c1","messages":[{"role":"assistant","tool_calls":[{"id":"call-1-0"}]},{"role":"tool","tool_call_id":"call-1-0","content":"x"}]}"#;
        let b = br#"{"prompt_cache_key":"k2","cancel":"c2","messages":[{"role":"assistant","tool_calls":[{"id":"call-9-0"}]},{"role":"tool","tool_call_id":"call-9-0","content":"x"}]}"#;
        assert_eq!(request_digest(a), request_digest(b));
        assert_ne!(request_digest(a), request_digest(&String::from_utf8_lossy(b).replace("\"x\"", "\"y\"").into_bytes()));
        assert!(tool_results(br#"{"messages":[{"role":"tool","content":"x"},{"role":"user","content":"u"}]}"#).is_empty());
    }
    #[test]
    fn a_panicking_handler_ends_its_count() {
        let dir = std::env::temp_dir();
        let server = Server::start(Source::Replay(Mutex::new(VecDeque::new())), None, false, &dir).unwrap();
        track(&server.shared, |_| panic!("handler"));
        let start = Instant::now();
        while server.shared.busy.load(Ordering::Acquire) > 0 && start.elapsed() < Duration::from_secs(2) { std::thread::sleep(Duration::from_millis(2)); }
        assert_eq!(server.shared.busy.load(Ordering::Acquire), 0);
        assert!(server.exchanges_within(Duration::from_millis(50)).is_empty());
        assert!(!server.wait_expired());
    }
    #[test]
    fn exchanges_say_when_the_wait_expires() {
        let dir = std::env::temp_dir();
        let server = Server::start(Source::Replay(Mutex::new(VecDeque::new())), None, false, &dir).unwrap();
        server.shared.busy.fetch_add(1, Ordering::AcqRel);
        let _ = server.exchanges_within(Duration::from_millis(20));
        assert!(server.wait_expired());
        server.shared.busy.fetch_sub(1, Ordering::AcqRel);
    }
    #[test]
    fn edit_tags_come_from_the_workspace() {
        let dir = std::env::temp_dir().join(format!("vade-suite-tag-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "x\n").unwrap();
        let got = fill_tags(r#"{"edit_tag":"{{edit_tag a.rs}}","n":"{{edit_tag a.rs}}"}"#, &dir);
        let tag = crate::line_edit::tag("a.rs", "x\n");
        assert_eq!(got, format!(r#"{{"edit_tag":"{tag}","n":"{tag}"}}"#));
        assert_eq!(fill_tags("no tags {{edit_tag", &dir), "no tags {{edit_tag");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
