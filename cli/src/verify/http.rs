//! Bounded HTTP/SSE transport for exact token comparisons. No subprocesses.
use crate::evals::json::{self, Value};
use std::{
    collections::BTreeMap,
    io::{self, BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream, ToSocketAddrs},
    time::{Duration, Instant},
};

const MAX_BODY: usize = 8 * 1024 * 1024;
const MAX_LINE: usize = 1024 * 1024;

pub struct Endpoint {
    authority: String,
    address: String,
}

#[derive(Debug)]
pub struct Reply {
    pub token_ids: Vec<u32>,
    pub token_texts: Vec<String>,
    pub model: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub draft_used: bool,
    pub model_draft_used: bool,
    pub identity: Value,
}

impl Endpoint {
    pub fn parse(url: &str) -> Result<Self, String> {
        let rest = url
            .strip_prefix("http://")
            .ok_or("verify needs an http:// engine URL")?;
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        if authority.is_empty()
            || authority
                .bytes()
                .any(|c| c.is_ascii_control() || c.is_ascii_whitespace())
            || authority.contains(['@', '?', '#', '\\'])
            || !matches!(path.trim_end_matches('/'), "" | "v1" | "v1/completions")
        {
            return Err("verify needs an engine root URL without credentials or a query".into());
        }
        let address = if authority.starts_with('[') {
            let end = authority
                .find(']')
                .ok_or("invalid bracketed engine address")?;
            let host = authority[1..end]
                .parse::<std::net::Ipv6Addr>()
                .map_err(|_| "invalid IPv6 engine address")?;
            let suffix = &authority[end + 1..];
            let port = if suffix.is_empty() {
                80
            } else {
                valid_port(suffix.strip_prefix(':').ok_or("invalid engine port")?)?
            };
            format!("[{host}]:{port}")
        } else if let Some((host, port)) = authority.split_once(':') {
            if host.is_empty() || host.contains(['[', ']']) {
                return Err("invalid engine host".into());
            }
            let port = valid_port(port)?;
            format!("{}:{port}", host.to_ascii_lowercase())
        } else {
            if authority.contains(['[', ']']) {
                return Err("invalid engine host".into());
            }
            format!("{}:80", authority.to_ascii_lowercase())
        };
        Ok(Self {
            authority: authority.into(),
            address,
        })
    }

    /// Syntactic endpoint identity only: DNS aliases need not be separate machines.
    pub fn same_address(&self, other: &Self) -> bool {
        self.address == other.address
    }

    pub fn completion(
        &self,
        prompt: &str,
        draft_on: bool,
        max_tokens: usize,
        timeout: Duration,
    ) -> Result<Reply, String> {
        if max_tokens == 0 || timeout.is_zero() {
            return Err("verify needs positive output and timeout limits".into());
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or("verify timeout is too large")?;
        let mut connected = None;
        let addresses = if let Ok(address) = self.address.parse::<SocketAddr>() {
            vec![address]
        } else {
            let address = self.address.clone();
            resolve_before(deadline, move || {
                address.to_socket_addrs().map(Iterator::collect)
            })?
        };
        for address in addresses {
            let left = remaining(deadline).map_err(|e| e.to_string())?;
            if let Ok(socket) = TcpStream::connect_timeout(&address, left) {
                connected = Some(socket);
                break;
            }
        }
        let socket = connected.ok_or("cannot connect to the running engine")?;
        socket.set_nodelay(true).map_err(|e| e.to_string())?;
        let mut timed = TimedSocket {
            socket,
            deadline,
            complete: false,
        };
        let body = request_body(prompt, draft_on, max_tokens);
        let authorization = crate::auth::bearer_key()?
            .map(|key| format!("Authorization: Bearer {key}\r\n"))
            .unwrap_or_default();
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: {}\r\n{authorization}Content-Type: application/json\r\nAccept: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
            self.authority,
            body.len()
        );
        timed
            .write_all(request.as_bytes())
            .map_err(|e| format!("cannot send verify request: {e}"))?;
        let mut reader = BufReader::new(timed);
        let result =
            receive(&mut reader).and_then(|body| parse_stream(&body, draft_on, max_tokens));
        reader.get_mut().complete = result.is_ok();
        result
    }
}

fn valid_port(value: &str) -> Result<u16, String> {
    if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
        return Err("invalid engine port".into());
    }
    value
        .parse::<u16>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| "invalid engine port".into())
}

fn resolve_before(
    deadline: Instant,
    lookup: impl FnOnce() -> io::Result<Vec<SocketAddr>> + Send + 'static,
) -> Result<Vec<SocketAddr>, String> {
    remaining(deadline).map_err(|e| e.to_string())?;
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    // System DNS has no portable timeout. A late resolver can only send its
    // result to this channel; it never opens an engine connection or request.
    std::thread::Builder::new()
        .name("vade-verify-resolve".into())
        .spawn(move || {
            let _ = send.send(lookup());
        })
        .map_err(|_| "cannot start engine address lookup")?;
    receive
        .recv_timeout(remaining(deadline).map_err(|e| e.to_string())?)
        .map_err(|error| match error {
            std::sync::mpsc::RecvTimeoutError::Timeout => {
                "verify request deadline exceeded during engine address lookup"
            }
            std::sync::mpsc::RecvTimeoutError::Disconnected => "engine address lookup failed",
        })?
        .map_err(|e| format!("cannot resolve engine: {e}"))
}

fn request_body(prompt: &str, draft_on: bool, max_tokens: usize) -> String {
    let prompt = json::packed(&Value::Str(prompt.into()));
    let draft = if draft_on { "strict" } else { "plain" };
    format!(
        r#"{{"prompt":{prompt},"stream":true,"stream_options":{{"include_usage":true}},"max_tokens":{max_tokens},"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":0,"enable_thinking":false,"cache":false,"ignore_eos":true,"draft":"{draft}"}}"#
    )
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "verify request deadline exceeded"))
}

struct TimedSocket {
    socket: TcpStream,
    deadline: Instant,
    complete: bool,
}
impl Read for TimedSocket {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.socket
            .set_read_timeout(Some(remaining(self.deadline)?))?;
        self.socket.read(bytes)
    }
}
impl Write for TimedSocket {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.socket
            .set_write_timeout(Some(remaining(self.deadline)?))?;
        self.socket.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.socket.flush()
    }
}
impl Drop for TimedSocket {
    fn drop(&mut self) {
        if !self.complete {
            // The engine accepts write-half-close; a trailing byte explicitly
            // cancels an unfinished request through its no-pipelining check.
            let _ = self
                .socket
                .set_write_timeout(Some(Duration::from_millis(100)));
            let _ = self.socket.write_all(b"\n");
        }
    }
}

fn read_line(reader: &mut impl BufRead, limit: usize) -> Result<String, String> {
    let mut bytes = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > limit || !bytes.ends_with(b"\n") {
        return Err("oversized or truncated HTTP line".into());
    }
    String::from_utf8(bytes).map_err(|_| "HTTP line is not UTF-8".into())
}

fn receive(reader: &mut impl BufRead) -> Result<String, String> {
    let status = read_line(reader, 8192)?;
    let status: Vec<_> = status.split_whitespace().collect();
    if status.len() < 2 || !matches!(status[0], "HTTP/1.0" | "HTTP/1.1") || status[1] != "200" {
        let code = status.get(1).and_then(|v| v.parse::<u16>().ok());
        return Err(format!(
            "engine HTTP request failed (status {})",
            code.map_or_else(|| "invalid".into(), |v| v.to_string())
        ));
    }
    let mut headers = BTreeMap::new();
    let mut header_bytes = 0;
    loop {
        let line = read_line(reader, 8192)?;
        header_bytes += line.len();
        if header_bytes > 64 * 1024 {
            return Err("engine HTTP headers exceed 64 KiB".into());
        }
        if matches!(line.as_str(), "\r\n" | "\n") {
            break;
        }
        let (key, value) = line
            .trim_end_matches(['\r', '\n'])
            .split_once(':')
            .ok_or("malformed HTTP header")?;
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err("invalid HTTP header name".into());
        }
        let key = key.to_ascii_lowercase();
        let value = value.trim().to_owned();
        if headers.insert(key.clone(), value).is_some()
            && matches!(
                key.as_str(),
                "content-length" | "transfer-encoding" | "content-type"
            )
        {
            return Err("duplicate HTTP framing header".into());
        }
    }
    if !headers.get("content-type").is_some_and(|v| {
        v.split(';')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case("text/event-stream")
    }) {
        return Err("engine did not return a token SSE stream".into());
    }
    if headers
        .get("content-encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return Err("compressed verify responses are unsupported".into());
    }
    let mut bytes = Vec::new();
    if let Some(encoding) = headers.get("transfer-encoding") {
        if !encoding.eq_ignore_ascii_case("chunked") || headers.contains_key("content-length") {
            return Err("unsupported or ambiguous HTTP framing".into());
        }
        loop {
            let line = read_line(reader, 8192)?;
            let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16)
                .map_err(|_| "invalid HTTP chunk size")?;
            if size == 0 {
                let mut trailer_bytes = 0;
                loop {
                    let trailer = read_line(reader, 8192)?;
                    trailer_bytes += trailer.len();
                    if trailer_bytes > 64 * 1024 {
                        return Err("oversized HTTP trailers".into());
                    }
                    if matches!(trailer.as_str(), "\r\n" | "\n") {
                        break;
                    }
                    if !trailer.contains(':') {
                        return Err("malformed HTTP trailer".into());
                    }
                }
                break;
            }
            append_exact(reader, &mut bytes, size)?;
            let mut crlf = [0; 2];
            reader.read_exact(&mut crlf).map_err(|e| e.to_string())?;
            if crlf != *b"\r\n" {
                return Err("invalid HTTP chunk separator".into());
            }
        }
    } else if let Some(length) = headers.get("content-length") {
        if length.is_empty() || !length.bytes().all(|c| c.is_ascii_digit()) {
            return Err("invalid HTTP content length".into());
        }
        let length = length
            .parse::<usize>()
            .map_err(|_| "invalid HTTP content length")?;
        append_exact(reader, &mut bytes, length)?;
    } else {
        reader
            .take(MAX_BODY as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > MAX_BODY {
            return Err("verify response exceeds 8 MiB".into());
        }
    }
    String::from_utf8(bytes).map_err(|_| "verify response is not UTF-8".into())
}

fn append_exact(reader: &mut impl Read, bytes: &mut Vec<u8>, size: usize) -> Result<(), String> {
    let length = bytes
        .len()
        .checked_add(size)
        .filter(|n| *n <= MAX_BODY)
        .ok_or("verify response exceeds 8 MiB")?;
    let start = bytes.len();
    bytes.resize(length, 0);
    reader
        .read_exact(&mut bytes[start..])
        .map_err(|e| format!("truncated verify response: {e}"))
}

fn integer(value: Option<&Value>, field: &str) -> Result<usize, String> {
    value
        .and_then(Value::as_i64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("missing or invalid {field}"))
}
fn boolean(value: Option<&Value>, field: &str) -> Result<bool, String> {
    match value {
        Some(Value::Bool(v)) => Ok(*v),
        _ => Err(format!("missing or invalid {field}")),
    }
}

fn parse_stream(body: &str, draft_on: bool, max_tokens: usize) -> Result<Reply, String> {
    let mut reply = Reply {
        token_ids: Vec::new(),
        token_texts: Vec::new(),
        model: String::new(),
        prompt_tokens: 0,
        completion_tokens: 0,
        finish_reason: String::new(),
        draft_used: false,
        model_draft_used: false,
        identity: Value::Null,
    };
    let (mut data, mut done, mut finish) = (String::new(), false, false);
    for line in body.split_inclusive('\n') {
        if line.len() > MAX_LINE {
            return Err("oversized SSE line".into());
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if !line.is_empty() {
            if let Some(value) = line.strip_prefix("data:") {
                if done {
                    return Err("SSE data follows [DONE]".into());
                }
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value.strip_prefix(' ').unwrap_or(value));
                if data.len() > MAX_LINE {
                    return Err("oversized SSE event".into());
                }
            } else if !line.starts_with(':')
                && !line.starts_with("event:")
                && !line.starts_with("id:")
                && !line.starts_with("retry:")
            {
                return Err("malformed SSE field".into());
            }
            continue;
        }
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            if !finish {
                return Err("SSE ended before final usage and settings".into());
            }
            done = true;
            data.clear();
            continue;
        }
        // This parser validates the entire object and rejects duplicate keys,
        // invalid JSON numbers and excessive nesting before the typed read.
        let model = crate::json::string(&data, "model")
            .filter(|s| !s.is_empty())
            .ok_or("invalid SSE JSON or missing response model")?;
        if !reply.model.is_empty() && reply.model != model {
            return Err("response model changed within stream".into());
        }
        reply.model = model;
        let event = json::parse(&data)?;
        data.clear();
        if event.get("error").is_some() {
            return Err("engine reported a streaming error".into());
        }
        if finish {
            return Err("SSE event follows the final completion".into());
        }
        let choices = event
            .get("choices")
            .and_then(Value::as_array)
            .ok_or("missing SSE choices")?;
        if choices.len() != 1 {
            return Err("verify requires exactly one completion choice".into());
        }
        let choice = &choices[0];
        if integer(choice.get("index"), "choice index")? != 0 {
            return Err("unexpected completion choice index".into());
        }
        let text = choice
            .get("text")
            .and_then(Value::as_str)
            .ok_or("missing token text")?;
        if let Some(id) = choice.get("token_id") {
            let id = u32::try_from(integer(Some(id), "token id")?)
                .map_err(|_| "token id exceeds u32")?;
            if reply.token_ids.len() >= max_tokens {
                return Err("too many streamed token ids".into());
            }
            reply.token_ids.push(id);
            reply.token_texts.push(text.into());
        } else if !text.is_empty() {
            // A decoder may flush a final partial UTF-8 sequence separately.
            // Only permit this after every requested token has arrived.
            if reply.token_ids.len() != max_tokens {
                return Err("completion text lacks a token id".into());
            }
            reply
                .token_texts
                .last_mut()
                .ok_or("text without generated tokens")?
                .push_str(text);
        }
        match choice.get("finish_reason") {
            Some(Value::Null) => {
                if event.get("usage").is_some() || event.get("vade").is_some() {
                    return Err("settings or usage arrived without a final completion".into());
                }
            }
            Some(Value::Str(reason)) if reason == "length" => {
                reply.finish_reason = reason.clone();
                validate_final(&event, &mut reply, draft_on, max_tokens)?;
                finish = true;
            }
            _ => {
                return Err(
                    "missing or unexpected finish reason for fixed-length verification".into(),
                );
            }
        }
    }
    if !data.is_empty() || !done {
        return Err("incomplete SSE response: missing [DONE] or event terminator".into());
    }
    Ok(reply)
}

fn validate_final(
    event: &Value,
    reply: &mut Reply,
    draft_on: bool,
    max_tokens: usize,
) -> Result<(), String> {
    let usage = event.get("usage").ok_or("missing final usage")?;
    reply.prompt_tokens = integer(usage.get("prompt_tokens"), "prompt_tokens")?;
    reply.completion_tokens = integer(usage.get("completion_tokens"), "completion_tokens")?;
    let total = integer(usage.get("total_tokens"), "total_tokens")?;
    if reply.prompt_tokens == 0
        || reply.completion_tokens != max_tokens
        || reply.token_ids.len() != max_tokens
        || reply.prompt_tokens.checked_add(reply.completion_tokens) != Some(total)
    {
        return Err("incomplete token ids or inconsistent completion usage".into());
    }
    if let Some(details) = usage.get("prompt_tokens_details") {
        if integer(details.get("cached_tokens"), "cached_tokens")? != 0 {
            return Err("cache reuse is disabled during verification".into());
        }
    }
    let echo = event.get("vade").ok_or("missing effective settings echo")?;
    let sampling = echo
        .get("sampling")
        .ok_or("missing effective sampling echo")?;
    for (name, expected) in [
        ("temperature", 0.),
        ("top_p", 1.),
        ("top_k", 0.),
        ("repetition_penalty", 1.),
        ("presence_penalty", 0.),
        ("frequency_penalty", 0.),
        ("seed", 0.),
    ] {
        if sampling.get(name).and_then(Value::as_f64) != Some(expected) {
            return Err(format!("effective sampling differs: {name}"));
        }
    }
    let expected = if draft_on { "strict" } else { "plain" };
    if echo.get("draft").and_then(Value::as_str) != Some(expected) {
        return Err("effective draft mode differs from verify request".into());
    }
    if boolean(echo.get("thinking"), "thinking")? {
        return Err("thinking must be off during verification".into());
    }
    integer(echo.get("mtp_depth"), "mtp_depth")?;
    reply.draft_used = boolean(echo.get("draft_used"), "draft_used")?;
    reply.model_draft_used = boolean(echo.get("model_draft_used"), "model_draft_used")?;
    if !draft_on && (reply.draft_used || reply.model_draft_used) {
        return Err("engine used drafting during the OFF run".into());
    }
    reply.identity = identity(echo);
    Ok(())
}

// Optional evidence only. Never retain arbitrary server objects, model paths,
// source locations, or unbounded values in the small public receipt.
fn identity(echo: &Value) -> Value {
    let string = |value: Option<&Value>| match value {
        Some(Value::Str(s)) if s.len() <= 128 => Value::Str(s.clone()),
        _ => Value::Null,
    };
    let build = echo.get("build");
    let checkpoint = echo.get("checkpoint");
    json::object(vec![
        (
            "build",
            json::object(vec![
                ("git_sha", string(build.and_then(|v| v.get("git_sha")))),
                (
                    "dirty",
                    match build.and_then(|v| v.get("dirty")) {
                        Some(Value::Bool(dirty)) => Value::Bool(*dirty),
                        _ => Value::Null,
                    },
                ),
            ]),
        ),
        ("ptx_sha256", string(echo.get("ptx_sha256"))),
        ("layer_ptx_sha256", string(echo.get("layer_ptx_sha256"))),
        (
            "checkpoint",
            json::object(vec![
                ("sha256", string(checkpoint.and_then(|v| v.get("sha256")))),
                (
                    "hash_state",
                    string(checkpoint.and_then(|v| v.get("hash_state"))),
                ),
            ]),
        ),
        (
            "server_start_unix_ms",
            match echo.get("server_start_unix_ms") {
                Some(Value::Int(value)) if *value >= 0 => Value::Int(*value),
                _ => Value::Null,
            },
        ),
    ])
}

#[cfg(test)]
mod tests;
