//! One streamed chat reply: SSE parsing and the bounded request.
//! Token ids and usage counts parse strictly as integers. A fractional,
//! negative or out-of-range value is never converted; see
//! `reply_evidence_failure` for the one complete-evidence check.
use super::*;

/// One completed streamed reply, with the receipt fields the cards print.
#[derive(Clone, Debug, Default)]
pub struct Reply {
    pub text: String,
    pub reasoning: String,
    pub token_ids: Vec<u32>,
    /// A streamed `token_id` was not an exact unsigned integer.
    pub invalid_token_id: bool,
    /// The stream ended with the terminal `[DONE]` event.
    pub saw_done: bool,
    pub finish_reason: Option<String>,
    pub completion_id: Option<u64>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub draft: Option<String>,
    pub draft_used: Option<bool>,
    pub model_draft_used: Option<bool>,
    /// The model name the stream itself reports, when it reports one.
    pub served_model: Option<String>,
    /// The last `vade` receipt object as streamed, kept for the receipt.
    pub vade_receipt: Option<String>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub ttft_s: Option<f64>,
    pub total_s: f64,
}
impl Reply {
    /// Output tokens per second over the whole request, when counted.
    pub fn end_to_end_tok_s(&self) -> Option<f64> {
        let tokens = self.completion_tokens?;
        (self.total_s > 0.0).then(|| tokens as f64 / self.total_s)
    }
    /// Decode estimate: tokens after the first over the post-first-token span.
    pub fn decode_tok_s(&self) -> Option<f64> {
        let tokens = self.completion_tokens?;
        let ttft = self.ttft_s?;
        let span = self.total_s - ttft;
        (tokens > 1 && span > 0.0).then(|| (tokens - 1) as f64 / span)
    }
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(text)) => text.parse().ok(),
        _ => None,
    }
}
/// A usage count, strict: the raw JSON number must spell one `u64`.
/// Fractional, negative and exponent forms yield None, never a cast.
fn count(value: Option<&Value>) -> Option<u64> {
    match value {
        Some(Value::Number(text)) => text.parse::<u64>().ok(),
        _ => None,
    }
}
/// A token id, strict: the raw JSON number must spell one `u32`.
fn token_id(value: Option<&Value>) -> Option<u32> {
    match value {
        Some(Value::Number(text)) => text.parse::<u32>().ok(),
        _ => None,
    }
}
fn text_of(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_owned)
}
fn bool_of(value: Option<&Value>) -> Option<bool> {
    match value {
        Some(Value::Bool(flag)) => Some(*flag),
        _ => None,
    }
}

/// Apply one SSE `data:` payload to a reply. Returns true on `[DONE]`.
pub fn apply_sse_data(reply: &mut Reply, data: &str, elapsed_s: f64) -> io::Result<bool> {
    let data = data.trim();
    if data == "[DONE]" {
        reply.saw_done = true;
        return Ok(true);
    }
    let event = json::parse(data)?;
    if let Some(id) = text_of(event.get("id")) {
        if let Some(raw) = id.strip_prefix("completion-") {
            reply.completion_id = raw.parse().ok().or(reply.completion_id);
        }
    }
    if let Some(model) = text_of(event.get("model")) {
        reply.served_model = Some(model);
    }
    if event.get("error").is_some_and(|e| *e != Value::Null) {
        return Err(json::invalid("upstream stream returned an error"));
    }
    if let Some(choice) = event
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    {
        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    reply.ttft_s.get_or_insert(elapsed_s);
                }
                reply.text.push_str(content);
            }
            if let Some(reasoning) = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(Value::as_str)
            {
                if !reasoning.is_empty() {
                    reply.ttft_s.get_or_insert(elapsed_s);
                }
                reply.reasoning.push_str(reasoning);
            }
        }
        if let Some(content) = choice.get("text").and_then(Value::as_str) {
            if !content.is_empty() {
                reply.ttft_s.get_or_insert(elapsed_s);
            }
            reply.text.push_str(content);
        }
        if let Some(raw) = choice.get("token_id") {
            match token_id(Some(raw)) {
                Some(token) => {
                    reply.ttft_s.get_or_insert(elapsed_s);
                    reply.token_ids.push(token);
                }
                None => reply.invalid_token_id = true,
            }
        }
        if let Some(finish) = text_of(choice.get("finish_reason")) {
            reply.finish_reason = Some(finish);
        }
    }
    if let Some(usage) = event.get("usage") {
        reply.prompt_tokens = count(usage.get("prompt_tokens")).or(reply.prompt_tokens);
        reply.completion_tokens =
            count(usage.get("completion_tokens")).or(reply.completion_tokens);
        reply.cached_tokens = usage
            .get("prompt_tokens_details")
            .and_then(|details| count(details.get("cached_tokens")))
            .or(reply.cached_tokens);
        if let Some(timing) = usage.get("vade") {
            reply.prefill_ms = number(timing.get("prefill_ms")).or(reply.prefill_ms);
            reply.decode_ms = number(timing.get("decode_ms")).or(reply.decode_ms);
        }
    }
    if let Some(vade) = event.get("vade") {
        if let Ok(text) = json::stringify(vade) {
            reply.vade_receipt = Some(text);
        }
        if let Some(draft) = text_of(vade.get("draft")) {
            reply.draft = Some(draft);
        }
        reply.draft_used = bool_of(vade.get("draft_used")).or(reply.draft_used);
        reply.model_draft_used =
            bool_of(vade.get("model_draft_used")).or(reply.model_draft_used);
        reply.prefill_ms = number(vade.get("prefill_ms")).or(reply.prefill_ms);
    }
    Ok(false)
}

/// One deadline-bounded buffered line reader over the reply socket. Every
/// fill waits at most the time left to the request deadline, so a peer that
/// trickles bytes without ending a line still hits the total deadline.
struct BoundedReader {
    stream: TcpStream,
    buffer: Vec<u8>,
    start: usize,
    started: Instant,
    deadline: Duration,
}
impl BoundedReader {
    fn remaining(&self) -> io::Result<Duration> {
        remaining(self.started, self.deadline)
    }
    fn fill(&mut self) -> io::Result<bool> {
        if self.start > 0 {
            self.buffer.drain(..self.start);
            self.start = 0;
        }
        self.stream.set_read_timeout(Some(self.remaining()?))?;
        let mut chunk = [0u8; 8192];
        let read = self.stream.read(&mut chunk)?;
        if read == 0 {
            return Ok(false);
        }
        self.buffer.extend_from_slice(&chunk[..read]);
        Ok(true)
    }
    /// Read one line (with its ending) into `out`; 0 means end of stream.
    fn read_line(&mut self, out: &mut String, max: usize) -> io::Result<usize> {
        loop {
            if let Some(at) = self.buffer[self.start..].iter().position(|b| *b == b'\n') {
                let len = at + 1;
                if len > max {
                    return Err(json::invalid("upstream reply line is too long"));
                }
                let text = std::str::from_utf8(&self.buffer[self.start..self.start + len])
                    .map_err(|_| json::invalid("upstream reply is not UTF-8"))?
                    .to_owned();
                out.push_str(&text);
                self.start += len;
                return Ok(len);
            }
            if self.buffer.len() - self.start > max {
                return Err(json::invalid("upstream reply line is too long"));
            }
            if !self.fill()? {
                let len = self.buffer.len() - self.start;
                if len > 0 {
                    let text = std::str::from_utf8(&self.buffer[self.start..])
                        .map_err(|_| json::invalid("upstream reply is not UTF-8"))?
                        .to_owned();
                    out.push_str(&text);
                    self.start += len;
                }
                return Ok(len);
            }
        }
    }
}

fn remaining(started: Instant, deadline: Duration) -> io::Result<Duration> {
    deadline
        .checked_sub(started.elapsed())
        .filter(|left| !left.is_zero())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::TimedOut, "quality-card request deadline passed")
        })
}

/// Resolve one host inside the same elapsed budget as the whole request.
/// Name resolution has no socket timeout, so it runs on its own thread and
/// the caller stops waiting at the deadline.
fn resolve(endpoint: &Endpoint, started: Instant, deadline: Duration) -> io::Result<std::net::SocketAddr> {
    let target = format!("{}:{}", endpoint.host, endpoint.port);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let found = target
            .to_socket_addrs()
            .map(|mut addrs| addrs.next())
            .map_err(|error| error.kind());
        let _ = sender.send(found);
    });
    match receiver.recv_timeout(remaining(started, deadline)?) {
        Ok(Ok(Some(address))) => Ok(address),
        Ok(Ok(None)) => Err(json::invalid("endpoint host did not resolve")),
        Ok(Err(kind)) => Err(io::Error::from(kind)),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "quality-card request deadline passed",
        )),
    }
}

/// Send one streamed chat request and collect the whole reply. The Vade
/// server streams with `Connection: close` and no chunked framing; a reply
/// that uses chunked framing is refused rather than misread. Reading stops
/// at the `[DONE]` event. Resolution, the write sequence and every read
/// share the one `deadline` budget from the start of the request.
pub fn send(
    endpoint: &Endpoint,
    body: &str,
    auth_header: Option<&str>,
    deadline: Duration,
) -> io::Result<Reply> {
    let started = Instant::now();
    let address = resolve(endpoint, started, deadline)?;
    let connect = Duration::from_secs(10).min(remaining(started, deadline)?);
    let mut stream = TcpStream::connect_timeout(&address, connect)?;
    let auth = auth_header.unwrap_or("");
    let request = format!(
        "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccept: text/event-stream\r\nConnection: close\r\n{auth}\r\n{body}",
        endpoint.path,
        endpoint.host,
        endpoint.port,
        body.len()
    );
    let bytes = request.as_bytes();
    let mut written = 0;
    while written < bytes.len() {
        stream.set_write_timeout(Some(remaining(started, deadline)?))?;
        written += stream.write(&bytes[written..])?;
    }
    stream.flush()?;
    let mut reader = BoundedReader { stream, buffer: Vec::new(), start: 0, started, deadline };
    let mut status = String::new();
    if reader.read_line(&mut status, 16 * 1024)? == 0 {
        return Err(json::invalid("upstream closed before response headers"));
    }
    if !status.starts_with("HTTP/1.1 200") && !status.starts_with("HTTP/1.0 200") {
        return Err(io::Error::other(format!(
            "upstream HTTP status line: {}",
            status.trim()
        )));
    }
    let mut content_type = String::new();
    let mut headers = 0usize;
    loop {
        headers += 1;
        if headers > 128 {
            return Err(json::invalid("upstream sent too many response headers"));
        }
        let mut line = String::new();
        if reader.read_line(&mut line, 16 * 1024)? == 0 {
            return Err(json::invalid("upstream closed before response headers"));
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            if name.eq_ignore_ascii_case("transfer-encoding")
                && value.to_ascii_lowercase().contains("chunked")
            {
                return Err(json::invalid("chunked upstream replies are unsupported"));
            }
            if name.eq_ignore_ascii_case("content-type") {
                content_type = value.trim().to_ascii_lowercase();
            }
        }
    }
    if !content_type.starts_with("text/event-stream") {
        return Err(json::invalid("upstream reply is not an SSE stream"));
    }
    let mut reply = Reply::default();
    let mut done = false;
    let mut line = String::new();
    while !done {
        line.clear();
        if reader.read_line(&mut line, 1024 * 1024)? == 0 {
            break;
        }
        let Some(data) = line.trim_end().strip_prefix("data:") else {
            continue;
        };
        done = apply_sse_data(&mut reply, data.trim(), started.elapsed().as_secs_f64())?;
    }
    reply.total_s = started.elapsed().as_secs_f64();
    if !done {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "incomplete chat SSE stream",
        ));
    }
    Ok(reply)
}
