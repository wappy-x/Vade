//! The timed_load client contract: server usage counts, ordered request bytes,
//! and client send-to-DONE clocks. Transport follows the existing bench HTTP reader.
use super::{
    Config,
    value::{PYTHON_INT_DIGITS, Value, b, digest, encode, n, o, parse, s},
};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

const MAX_LINE_BYTES: usize = 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_HEALTH_BYTES: usize = 4 * 1024 * 1024;
const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;
const CONNECT_POLL: Duration = Duration::from_millis(200);

/// One cohort uses one gate; a timeout breaks it and wakes every waiting chat.
pub struct Gate {
    expected: usize,
    state: Mutex<(usize, bool)>,
    wake: Condvar,
}
impl Gate {
    pub fn new(expected: usize) -> Self {
        Self {
            expected,
            state: Mutex::new((0, false)),
            wake: Condvar::new(),
        }
    }
    pub fn wait(&self, timeout: f64) -> Result<(), String> {
        let duration = Duration::try_from_secs_f64(timeout).map_err(|e| e.to_string())?;
        let deadline = Instant::now()
            .checked_add(duration)
            .ok_or("cohort gate timeout is too large")?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| "cohort start gate poisoned")?;
        if state.1 {
            return Err("cohort start gate broken".into());
        }
        state.0 += 1;
        if state.0 == self.expected {
            self.wake.notify_all();
            return Ok(());
        }
        while state.0 < self.expected && !state.1 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                state.1 = true;
                self.wake.notify_all();
                return Err("cohort start gate timeout".into());
            }
            let (next, _) = self
                .wake
                .wait_timeout(state, left)
                .map_err(|_| "cohort start gate poisoned")?;
            state = next;
        }
        if state.1 {
            Err("cohort start gate broken".into())
        } else {
            Ok(())
        }
    }
}

pub fn sampling() -> Value {
    o([
        ("temperature", n(0)),
        ("top_p", n(1)),
        ("top_k", n(0)),
        ("repetition_penalty", n("1.0")),
        ("presence_penalty", n(0)),
        ("frequency_penalty", n(0)),
    ])
}

/// GLM's chat template has no thinking-off form and no `/no_think` switch, and a GLM server refuses a
/// thinking-off field. The name rule is the card tool's (`is_glm_model` in `cli/src/card.rs`).
pub fn glm_model(model: &str) -> bool { model.to_ascii_lowercase().contains("glm") }

pub fn request_body(template: &Value, args: &Config, max_tokens: usize) -> Value {
    let mut body = template.clone();
    for key in [
        "seed",
        "draft",
        "mtp_depth",
        "prompt_lookup",
        "session",
        "kv_handle",
        "kv_checkpoint",
        "cancel",
        "fork",
        "stop",
        "max_completion_tokens",
        "prompt_cache_key",
    ] {
        body.del(key);
    }
    if let Value::Object(values) = sampling() {
        for (key, value) in values {
            body.put(&key, value);
        }
    }
    for (key, value) in [
        ("model", s(&args.model)),
        ("stream", b(true)),
        ("stream_options", o([("include_usage", b(true))])),
        ("max_tokens", n(max_tokens)),
        ("cache", b(false)),
        ("ignore_eos", b(true)),
    ] {
        body.put(key, value);
    }
    // Thinking as low as the template allows: off, or low effort for GLM (the cards ask the same).
    if glm_model(&args.model) {
        for key in ["enable_thinking", "chat_template_kwargs"] {
            body.del(key);
        }
        body.put("reasoning_effort", s("low"));
    } else {
        body.put("enable_thinking", b(false));
        body.put("chat_template_kwargs", o([("enable_thinking", b(false))]));
    }
    if let Some(draft) = &args.draft {
        body.put("draft", s(draft));
    }
    body
}

fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(v) => if *v { "True" } else { "False" }.into(),
        Value::String(v) => {
            let quote = if v.contains('\'') && !v.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut output = String::new();
            output.push(quote);
            for ch in v.chars() {
                match ch {
                    '\\' => output.push_str("\\\\"),
                    '\n' => output.push_str("\\n"),
                    '\r' => output.push_str("\\r"),
                    '\t' => output.push_str("\\t"),
                    ch if ch == quote => {
                        output.push('\\');
                        output.push(ch);
                    }
                    '\0'..='\u{1f}' | '\u{7f}' => output.push_str(&format!("\\x{:02x}", ch as u32)),
                    _ => output.push(ch),
                }
            }
            output.push(quote);
            output
        }
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(key, value)| format!("{}: {}", py_repr(&s(key)), py_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Number(v) => v.clone(),
    }
}
pub fn validate_settings(value: &Value, draft: Option<&str>) -> Result<(), String> {
    super::drafting::validate_echo(value, draft)?;
    fn sampling_fields(value: &Value) -> Result<(), String> {
        if let Value::Object(items) = value {
            for (key, value) in items {
                if matches!(value, Value::Object(_)) {
                    sampling_fields(value)?;
                } else {
                    let expected = sampling().get(key).clone();
                    if expected != Value::Null && value != &expected {
                        return Err(format!("sampling echo disagrees: {key}={}, requested {}",
                            py_repr(value), py_repr(&expected)));
                    }
                }
            }
        }
        Ok(())
    }
    sampling_fields(value)
}

pub fn memory_bytes(value: &Value) -> Value {
    fn visit(value: &Value, prefix: &str, output: &mut Value) {
        if let Value::Object(items) = value {
            for (key, value) in items {
                let name = format!("{prefix}{key}");
                if key.ends_with("_bytes") && value.nonnegative_integer().is_some() {
                    output.put(&name, value.clone());
                } else if matches!(value, Value::Object(_)) {
                    visit(value, &format!("{name}."), output);
                }
            }
        }
    }
    let mut output = o([]);
    visit(value, "", &mut output);
    output
}

static EMPTY_OBJECT: Value = Value::Object(Vec::new());
static EMPTY_ARRAY: Value = Value::Array(Vec::new());
static NULL: Value = Value::Null;

// Python dict.get defaults only absent keys; a supplied null must survive
// until the next accessed operation rejects it. Keep these lookups fallible.
fn field<'a>(value: &'a Value, key: &str, default: &'a Value) -> Result<&'a Value, String> {
    let Value::Object(items) = value else {
        return Err(format!("SSE object required when accessing {key}"));
    };
    Ok(items
        .iter()
        .find(|(name, _)| name == key)
        .map_or(default, |(_, value)| value))
}

fn iterable<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], String> {
    match value {
        Value::Array(items) => Ok(items),
        // Python iterates dict keys/string characters. Empty ones do nothing;
        // nonempty ones fail at the first item's dict.get before any mutation.
        Value::Object(items) if items.is_empty() => Ok(&[]),
        Value::String(text) if text.is_empty() => Ok(&[]),
        _ => Err(format!("SSE {name} must be an array or an empty iterable")),
    }
}

fn function_has_token(function: &Value) -> Result<bool, String> {
    Ok(field(function, "name", &NULL)?.yes() || field(function, "arguments", &NULL)?.yes())
}

fn prefix_tokens(text: &str) -> Result<Value, String> {
    // Python int(string) accepts decimal Unicode digits, separators and signs.
    // Decimal-zero code points follow Python's Unicode 16.0 Nd data; every
    // decimal alphabet is ten contiguous digits, including mathematical sets.
    const ZEROS: &[u32] = &[
        0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
        0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
        0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0,
        0xff10, 0x104a0, 0x10d30, 0x10d40, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450,
        0x114d0, 0x11650, 0x116c0, 0x116d0, 0x116da, 0x11730, 0x118e0, 0x11950, 0x11bf0, 0x11c50,
        0x11d50, 0x11da0, 0x11f50, 0x16130, 0x16a60, 0x16ac0, 0x16b50, 0x16d70, 0x1ccf0, 0x1d7ce,
        0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e5f1, 0x1e950, 0x1fbf0,
    ];
    let trimmed = text.trim();
    let (negative, digits) = if let Some(rest) = trimmed.strip_prefix('-') {
        (true, rest)
    } else {
        (false, trimmed.strip_prefix('+').unwrap_or(trimmed))
    };
    let mut number = String::new();
    let mut last_digit = false;
    for ch in digits.chars() {
        if ch == '_' && last_digit {
            last_digit = false;
            continue;
        }
        let point = ch as u32;
        let Some(digit) = ZEROS
            .iter()
            .find_map(|zero| point.checked_sub(*zero).filter(|digit| *digit < 10))
        else {
            return Err(format!("invalid prefix reused token count: {text}"));
        };
        number.push((b'0' + digit as u8) as char);
        last_digit = true;
    }
    if !last_digit || number.len() > PYTHON_INT_DIGITS {
        return Err(format!("invalid prefix reused token count: {text}"));
    }
    let number = number.trim_start_matches('0');
    Ok(n(if number.is_empty() {
        "0".into()
    } else if negative {
        format!("-{number}")
    } else {
        number.to_owned()
    }))
}

pub fn has_token(delta: &Value) -> Result<bool, String> {
    for key in ["content", "reasoning_content", "reasoning"] {
        if field(delta, key, &NULL)?
            .str()
            .is_some_and(|s| !s.is_empty())
        {
            return Ok(true);
        }
    }
    for call in iterable(field(delta, "tool_calls", &EMPTY_ARRAY)?, "tool_calls")? {
        if function_has_token(field(call, "function", &EMPTY_OBJECT)?)? {
            return Ok(true);
        }
    }
    function_has_token(field(delta, "function_call", &EMPTY_OBJECT)?)
}

struct Endpoint {
    host: String,
    authority: String,
    target: String,
}
fn endpoint(url: &str, path: &str) -> Result<Endpoint, String> {
    if url.starts_with("https://") {
        return Err(
            "HTTPS is unavailable in the std-only client; use an http:// local tunnel".into(),
        );
    }
    let rest = url
        .trim_end_matches('/')
        .strip_prefix("http://")
        .ok_or("--url must be an HTTP(S) base URL without credentials, query or fragment")?;
    let (authority, base) = rest.split_once('/').unwrap_or((rest, ""));
    if authority.is_empty() || rest.contains(['@', '?', '#', '\r', '\n']) || authority.contains(' ')
    {
        return Err(
            "--url must be an HTTP(S) base URL without credentials, query or fragment".into(),
        );
    }
    let address = if authority.starts_with('[') {
        if authority.ends_with(']') {
            format!("{authority}:80")
        } else {
            authority.to_owned()
        }
    } else if authority.contains(':') {
        authority.to_owned()
    } else {
        format!("{authority}:80")
    };
    let base = if base.is_empty() {
        String::new()
    } else {
        format!("/{}", base.trim_end_matches('/'))
    };
    let base = base.strip_suffix("/v1").unwrap_or(&base);
    let target = format!("{base}{path}");
    if target.contains(['\r', '\n', ' ']) {
        return Err("invalid HTTP path".into());
    }
    Ok(Endpoint {
        host: authority.to_owned(),
        authority: address,
        target,
    })
}

fn line<R: BufRead>(reader: &mut R) -> Result<Option<String>, String> {
    let mut bytes = Vec::new();
    let read = reader
        .take((MAX_LINE_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > MAX_LINE_BYTES {
        return Err(format!("HTTP/SSE line exceeds {MAX_LINE_BYTES} bytes"));
    }
    if read == 0 {
        return Ok(None);
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|e| e.to_string())
}

struct PollSocket {
    socket: TcpStream,
    timeout: Duration,
    interrupted: fn() -> bool,
}
impl PollSocket {
    fn deadline(&self) -> io::Result<Instant> {
        Instant::now()
            .checked_add(self.timeout)
            .ok_or_else(|| io::Error::other("HTTP I/O timeout is too large"))
    }
    fn left(&self, deadline: Instant) -> io::Result<Duration> {
        if (self.interrupted)() {
            // Read/Write convenience methods retry Interrupted automatically.
            // A cancellation must escape them so owned launch cleanup can run.
            return Err(io::Error::other("benchmark interrupted"));
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "timed out"));
        }
        Ok(left.min(Duration::from_millis(200)))
    }
}
impl Read for PollSocket {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        let deadline = self.deadline()?;
        loop {
            self.socket.set_read_timeout(Some(self.left(deadline)?))?;
            match self.socket.read(output) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) =>
                {
                    continue;
                }
                result => return result,
            }
        }
    }
}
impl Write for PollSocket {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let deadline = self.deadline()?;
        loop {
            self.socket.set_write_timeout(Some(self.left(deadline)?))?;
            match self.socket.write(bytes) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) =>
                {
                    continue;
                }
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.socket.flush()
    }
}

struct Body {
    reader: BufReader<PollSocket>,
    chunked: bool,
    remaining: usize,
    separator: bool,
    done: bool,
    length: Option<u64>,
}
impl Read for Body {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.done || output.is_empty() || self.length == Some(0) {
            return Ok(0);
        }
        if !self.chunked {
            let cap = self
                .length
                .map_or(output.len(), |v| output.len().min(v as usize));
            let read = self.reader.read(&mut output[..cap])?;
            if let Some(left) = &mut self.length {
                *left -= read as u64;
            }
            return Ok(read);
        }
        if self.remaining == 0 {
            if self.separator {
                let mut crlf = [0; 2];
                self.reader.read_exact(&mut crlf)?;
                if crlf != *b"\r\n" {
                    return Err(io::Error::other("invalid chunk separator"));
                }
            }
            let size = line(&mut self.reader)
                .map_err(io::Error::other)?
                .ok_or_else(|| io::Error::other("missing chunk size"))?;
            self.remaining = usize::from_str_radix(size.trim().split(';').next().unwrap_or(""), 16)
                .map_err(io::Error::other)?;
            if self.remaining == 0 {
                self.done = true;
                return Ok(0);
            }
            self.separator = true;
        }
        let cap = output.len().min(self.remaining);
        let read = self.reader.read(&mut output[..cap])?;
        if read == 0 {
            return Err(io::Error::other("truncated HTTP chunk"));
        }
        self.remaining -= read;
        Ok(read)
    }
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn exchange(
    endpoint: &Endpoint,
    method: &str,
    bytes: &[u8],
    timeout: f64,
    authorization: Option<&str>,
    interrupted: fn() -> bool,
) -> Result<(u16, Vec<(String, String)>, BufReader<Body>), String> {
    let duration = super::support::timeout_duration(timeout, "HTTP timeout")?;
    let deadline = super::support::timeout_deadline(timeout, "HTTP connection timeout")?;
    if interrupted() {
        return Err("benchmark interrupted".into());
    }
    let mut addresses: Vec<_> = endpoint
        .authority
        .to_socket_addrs()
        .map_err(|e| e.to_string())?
        .collect();
    let mut last_error = "no server address".to_owned();
    // Each connection attempt yields to cancellation, and all addresses share
    // one deadline. Retry only transient failures; refused ports fail promptly.
    let socket = 'connect: loop {
        if addresses.is_empty() {
            return Err(last_error);
        }
        let mut pending = Vec::new();
        for address in addresses {
            if interrupted() {
                return Err("benchmark interrupted".into());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(format!("HTTP connection timed out: {last_error}"));
            }
            match TcpStream::connect_timeout(&address, left.min(CONNECT_POLL)) {
                Ok(socket) => break 'connect socket,
                Err(error) => {
                    if matches!(
                        error.kind(),
                        io::ErrorKind::TimedOut
                            | io::ErrorKind::WouldBlock
                            | io::ErrorKind::Interrupted
                    ) {
                        pending.push(address);
                    }
                    last_error = error.to_string();
                }
            }
        }
        addresses = pending;
    };
    socket.set_nodelay(true).map_err(|e| e.to_string())?;
    let mut socket = PollSocket {
        socket,
        timeout: duration,
        interrupted,
    };
    write!(
        socket,
        "{method} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
        endpoint.target, endpoint.host
    )
    .map_err(|e| e.to_string())?;
    if method == "POST" {
        write!(
            socket,
            "Content-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\n",
            bytes.len()
        )
        .map_err(|e| e.to_string())?;
    }
    // A GET sends the key too: an engine with a key gives a health request without it only the state.
    if let Some(key) = authorization {
        if key.contains(['\r', '\n']) {
            return Err("invalid authorization header".into());
        }
        write!(socket, "Authorization: Bearer {key}\r\n").map_err(|e| e.to_string())?;
    }
    socket
        .write_all(b"\r\n")
        .and_then(|_| socket.write_all(bytes))
        .map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(socket);
    let mut header_bytes = 0;
    let (status, headers) = loop {
        let status = line(&mut reader)?.ok_or("missing HTTP status")?;
        header_bytes += status.len();
        if header_bytes > MAX_HEADER_BYTES {
            return Err(format!("HTTP headers exceed {MAX_HEADER_BYTES} bytes"));
        }
        let status: u16 = status
            .split_whitespace()
            .nth(1)
            .ok_or("invalid HTTP status")?
            .parse()
            .map_err(|_| "invalid HTTP status")?;
        let mut headers = Vec::new();
        loop {
            let text = line(&mut reader)?.ok_or("truncated HTTP headers")?;
            header_bytes += text.len();
            if header_bytes > MAX_HEADER_BYTES {
                return Err(format!("HTTP headers exceed {MAX_HEADER_BYTES} bytes"));
            }
            if text.trim_end_matches(['\r', '\n']).is_empty() {
                break;
            }
            if let Some((key, value)) = text.split_once(':') {
                headers.push((key.to_owned(), value.trim().to_owned()));
            } else {
                return Err("invalid HTTP header".into());
            }
        }
        if status != 100 {
            break (status, headers);
        }
    };
    let chunked =
        header(&headers, "transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked"));
    let length = if chunked {
        None
    } else {
        header(&headers, "content-length").and_then(|v| v.parse().ok())
    };
    Ok((
        status,
        headers,
        BufReader::new(Body {
            reader,
            chunked,
            remaining: 0,
            separator: false,
            done: false,
            length,
        }),
    ))
}

pub fn get_health(url: &str, timeout: f64) -> Result<Value, String> {
    get_health_with_cancel(url, timeout, None, || false)
}

/// `authorization` is the bearer key, when the caller has one. An engine that has a key gives a
/// health request without it only `status`: no model name, build, checkpoint or drafting facts.
pub fn get_health_with_cancel(
    url: &str,
    timeout: f64,
    authorization: Option<&str>,
    interrupted: fn() -> bool,
) -> Result<Value, String> {
    let (status, _, reader) = exchange(
        &endpoint(url, "/health")?,
        "GET",
        &[],
        timeout,
        authorization,
        interrupted,
    )?;
    let mut bytes = Vec::new();
    reader
        .take((MAX_HEALTH_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > MAX_HEALTH_BYTES {
        return Err(format!("/health body exceeds {MAX_HEALTH_BYTES} bytes"));
    }
    let value = parse(&bytes)?;
    if status != 200
        || !matches!(&value, Value::Object(_))
        || value.get("status").str() != Some("ready")
    {
        return Err(format!("/health is not ready (HTTP {status})"));
    }
    if let Some(state) = value.path("vade.checkpoint.hash_state").str()
        && ["pending", "failed", "failed-stale", "stale"].contains(&state)
    {
        return Err(format!("/health checkpoint hash {state}"));
    }
    Ok(value)
}

/// Occupied launch ports are refused even if they serve invalid or unready health JSON.
pub fn responds(url: &str, timeout: f64) -> Result<bool, String> {
    match exchange(
        &endpoint(url, "/health")?,
        "GET",
        &[],
        timeout,
        None,
        || false,
    ) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

fn record(body: &Value) -> Value {
    let encoded = encode(body);
    let mut request_sampling = sampling();
    request_sampling.put("draft", body.get("draft").clone());
    request_sampling.put("draft_source", s(if body.get("draft") == &Value::Null { "server" } else { "request" }));
    request_sampling.put("ignore_eos", body.get("ignore_eos").clone());
    request_sampling.put("cache", b(false));
    o([
        ("usage", Value::Null),
        ("sampling_source", s("request")),
        ("requested_draft_mode", s(body.get("draft").str().unwrap_or("server"))),
        ("request_sampling", request_sampling),
        ("sampling_echo", o([])),
        ("finish_reason", Value::Null),
        ("ttft_ms", Value::Null),
        ("error", Value::Null),
        ("http_status", Value::Null),
        ("expected_tokens", body.get("max_tokens").clone()),
        ("request_sha256", s(digest(encoded.as_bytes()))),
        ("request_bytes", n(encoded.len())),
        ("output_text_bytes", n(0)),
    ])
}

fn event<R: BufRead>(reader: &mut R) -> Result<Option<String>, String> {
    let mut lines = Vec::new();
    let mut event_bytes = 0;
    loop {
        let Some(raw) = line(reader)? else {
            return Ok(if lines.is_empty() {
                None
            } else {
                Some(lines.join("\n"))
            });
        };
        event_bytes += raw.len();
        if event_bytes > MAX_EVENT_BYTES {
            return Err(format!("SSE event exceeds {MAX_EVENT_BYTES} bytes"));
        }
        let text = raw.trim_end_matches(['\r', '\n']);
        if text.is_empty() {
            if !lines.is_empty() {
                return Ok(Some(lines.join("\n")));
            }
            event_bytes = 0;
        } else if let Some(text) = text.strip_prefix("data:") {
            lines.push(text.strip_prefix(' ').unwrap_or(text).to_owned());
        }
    }
}

fn consume<R: BufRead, F: FnMut() -> f64>(
    reader: &mut R,
    record: &mut Value,
    body: &Value,
    args: &Config,
    first: &mut Option<f64>,
    first_content: &mut Option<f64>,
    clock: &mut F,
) -> Result<(), String> {
    const STANDARD: [&str; 8] = [
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
        "error",
    ];
    let mut done = false;
    while let Some(event) = event(reader)? {
        let now = clock();
        if event.trim() == "[DONE]" {
            done = true;
            break;
        }
        let chunk = parse(event.as_bytes())?;
        let Value::Object(fields) = &chunk else {
            return Err("SSE data must be a JSON object".into());
        };
        if chunk.get("error").yes() {
            return Err(format!(
                "stream error: {}",
                chunk
                    .get("error")
                    .str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| py_repr(chunk.get("error")))
            ));
        }
        if chunk.get("model") != &Value::Null
            && chunk.get("model").str() != Some(args.model.as_str())
        {
            return Err("response model differs from --model".into());
        }
        if matches!(chunk.get("usage"), Value::Object(_)) {
            record.put("usage", chunk.get("usage").clone());
        }
        for (key, value) in fields {
            if !STANDARD.contains(&key.as_str()) && matches!(value, Value::Object(_)) {
                let mut echoes = record.get("sampling_echo").clone();
                echoes.put(key, value.clone());
                if encode(&echoes).len() > MAX_EVENT_BYTES {
                    return Err(format!("sampling metadata exceeds {MAX_EVENT_BYTES} bytes"));
                }
                record.put("sampling_echo", echoes);
                record.put("sampling_source", s("server_echo"));
                validate_settings(value, args.draft.as_deref())?;
            }
        }
        // Missing choices and empty iterables are valid usage-only events in
        // the retained Python client. Supplied null or nonempty non-arrays fail.
        let choices = iterable(field(&chunk, "choices", &EMPTY_ARRAY)?, "choices")?;
        for choice in choices {
            let Value::Object(items) = choice else {
                return Err("SSE choice must be a JSON object".into());
            };
            let owned;
            // Completion text overrides delta, just as in Python; an absent
            // delta defaults to {}, while an explicitly supplied null fails.
            let delta = if items.iter().any(|(key, _)| key == "text") {
                owned = o([("content", choice.get("text").clone())]);
                &owned
            } else {
                match items.iter().find(|(key, _)| key == "delta") {
                    Some((_, delta @ Value::Object(_))) => delta,
                    None => {
                        owned = o([]);
                        &owned
                    }
                    _ => return Err("SSE delta must be a JSON object".into()),
                }
            };
            if first.is_none() && has_token(delta)? {
                *first = Some(now);
            }
            if first_content.is_none()
                && delta
                    .get("content")
                    .str()
                    .is_some_and(|text| !text.is_empty())
            {
                *first_content = Some(now);
            }
            if let Value::Object(values) = delta {
                let bytes = values
                    .iter()
                    .filter_map(|(_, value)| value.str())
                    .map(str::len)
                    .sum::<usize>();
                record.put(
                    "output_text_bytes",
                    n(record.get("output_text_bytes").uint().unwrap_or(0) + bytes as u64),
                );
            }
            if choice.get("finish_reason") != &Value::Null {
                record.put("finish_reason", choice.get("finish_reason").clone());
            }
        }
    }
    record.put("complete_clock_s", n(clock()));
    if !done || record.get("finish_reason") == &Value::Null {
        return Err("incomplete SSE stream: missing [DONE] or finish_reason".into());
    }
    let usage = record.get("usage");
    if usage.get("prompt_tokens").nonnegative_integer().is_none() {
        return Err("missing/invalid server usage; chunk counts are never token counts".into());
    }
    let tokens = usage.get("completion_tokens");
    if tokens.nonnegative_integer().is_none() {
        return Err("missing/invalid server usage; chunk counts are never token counts".into());
    }
    if first.is_none() || tokens == &n(0) {
        return Err("no observable generated token".into());
    }
    if tokens != body.get("max_tokens") {
        return Err(format!(
            "generated {} instead of {} tokens (early EOS/stop is ineligible)",
            encode(tokens),
            encode(body.get("max_tokens"))
        ));
    }
    let reused = record.get("prefix_reused_tokens");
    // Preserve Python's left-to-right cache checks, including dict.get on a
    // supplied prompt_tokens_details before the top-level cache/header fields.
    let zero = n(0);
    if field(
        field(usage, "prompt_tokens_details", &EMPTY_OBJECT)?,
        "cached_tokens",
        &zero,
    )? != &zero
        || field(usage, "cached_tokens", &zero)? != &zero
        || (reused != &Value::Null && reused != &zero)
    {
        return Err("cache=false response reports reused prompt tokens".into());
    }
    if record.get("sampling_echo").yes() {
        record.put("sampling_source", s("server_echo"));
        validate_settings(record.get("sampling_echo"), args.draft.as_deref())?;
    }
    Ok(())
}

fn response<R: BufRead, F: FnMut() -> f64>(
    status: u16,
    headers: &[(String, String)],
    reader: &mut R,
    record: &mut Value,
    body: &Value,
    args: &Config,
    first: &mut Option<f64>,
    first_content: &mut Option<f64>,
    clock: &mut F,
) -> Result<(), String> {
    record.put("http_status", n(status));
    if status != 200 {
        let mut bytes = Vec::new();
        reader
            .take(4096)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        return Err(format!(
            "HTTP {status}: {}",
            String::from_utf8_lossy(&bytes)
        ));
    }
    if !header(headers, "content-type").is_some_and(|text| text.contains("text/event-stream")) {
        return Err("server did not return an SSE stream".into());
    }
    let reused = match header(headers, "x-prefix-reused-tokens") {
        Some(text) => prefix_tokens(text)?,
        None => Value::Null,
    };
    record.put("prefix_reused_tokens", reused);
    consume(reader, record, body, args, first, first_content, clock)
}

fn finish<F: FnMut() -> f64>(
    mut record: Value,
    result: Result<(), String>,
    sent: f64,
    first: Option<f64>,
    first_content: Option<f64>,
    clock: &mut F,
) -> Value {
    if let Err(error) = result {
        record.put("error", s(error));
    }
    if record.get("complete_clock_s") == &Value::Null {
        record.put("complete_clock_s", n(clock()));
    }
    let complete = record.get("complete_clock_s").num().unwrap();
    let elapsed = (complete - sent) * 1000.0;
    record.put("elapsed_ms", n(elapsed));
    record.put(
        "ttft_ms",
        first.map_or(Value::Null, |v| n((v - sent) * 1000.0)),
    );
    let tokens = record.path("usage.completion_tokens").uint();
    let valid = !record.get("error").yes();
    record.put(
        "ok",
        b(record.get("http_status").uint() == Some(200)
            && valid
            && tokens == record.get("expected_tokens").uint()),
    );
    record.put(
        "tokens_per_s",
        if valid && elapsed > 0.0 {
            tokens.map_or(Value::Null, |v| n(v as f64 * 1000.0 / elapsed))
        } else {
            Value::Null
        },
    );
    let mut bytes = memory_bytes(record.get("usage"));
    if let Value::Object(items) = memory_bytes(record.get("sampling_echo")) {
        for (key, value) in items {
            bytes.put(&key, value);
        }
    }
    record.put("server_bytes", bytes);
    record.put(
        "first_content_ms",
        first_content.map_or(Value::Null, |v| n((v - sent) * 1000.0)),
    );
    record.put(
        "writing_tokens_per_s",
        // Preserve the canary estimator from Python receipt fields exactly:
        // all completion tokens minus one, after the first observable output
        // (including reasoning/tools), through client stream completion.
        match (tokens, record.get("ttft_ms").num()) {
            (Some(tokens), Some(ttft)) if valid && tokens > 0 && elapsed > ttft => {
                n((tokens - 1) as f64 * 1000.0 / (elapsed - ttft))
            }
            _ => Value::Null,
        },
    );
    // Standard usage does not count tokens through first answer content.
    // Neither completion totals nor event/byte counts give that numerator.
    record.put("answer_content_tokens_per_s", Value::Null);
    record
}

pub fn stream_chat(url: &str, body: Value, args: &Config, gate: Option<Arc<Gate>>) -> Value {
    let mut record = record(&body);
    let bytes = encode(&body);
    // Resolve/open after the barrier: cohort wall time includes TCP connection.
    let path = if let Value::Object(items) = &body {
        if items.iter().any(|(k, _)| k == "messages") {
            "/v1/chat/completions"
        } else {
            "/v1/completions"
        }
    } else {
        "/v1/completions"
    };
    let endpoint = endpoint(url, path);
    let gated = gate.map_or(Ok(()), |gate| gate.wait(args.timeout));
    let mut clock = || args.clock.elapsed().as_secs_f64();
    let sent = clock();
    record.put("send_clock_s", n(sent));
    let mut first = None;
    let mut first_content = None;
    let result = gated.and(endpoint).and_then(|endpoint| {
        let (status, headers, mut reader) = exchange(
            &endpoint,
            "POST",
            bytes.as_bytes(),
            args.timeout,
            args.api_key.as_deref(),
            args.interrupted,
        )?;
        response(
            status,
            &headers,
            &mut reader,
            &mut record,
            &body,
            args,
            &mut first,
            &mut first_content,
            &mut clock,
        )
    });
    finish(record, result, sent, first, first_content, &mut clock)
}

/// Fixed-clock replay exercises the same SSE parsing and metric finalization as TCP.
/// Clocks are send, each yielded event (including DONE), then completion/error.
pub fn replay_chat(
    body: Value,
    args: &Config,
    events: &[u8],
    status: u16,
    headers: &[(&str, &str)],
    clocks: &[f64],
) -> Value {
    let mut index = 0;
    let mut clock = || {
        let value = *clocks.get(index).expect("replay clock schedule exhausted");
        index += 1;
        value
    };
    let sent = clock();
    let mut record = record(&body);
    record.put("send_clock_s", n(sent));
    let headers = headers
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect::<Vec<_>>();
    let mut reader = BufReader::new(events);
    let mut first = None;
    let mut first_content = None;
    let result = response(
        status,
        &headers,
        &mut reader,
        &mut record,
        &body,
        args,
        &mut first,
        &mut first_content,
        &mut clock,
    );
    finish(record, result, sent, first, first_content, &mut clock)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_requests_drop_template_seeds() {
        let args = Config::parse(&["--self-test".into()]).unwrap();
        for seed in [n(42), Value::Null, n(20260905)] {
            let template = o([("seed", seed.clone()), ("temperature", n("0.8")), ("prompt", s("frozen prompt"))]);
            let body = request_body(&template, &args, 256);
            let Value::Object(fields) = &body else { panic!("request object") };
            assert!(fields.iter().all(|(key, _)| key != "seed"));
            assert_eq!(body.get("temperature"), &n(0));
            assert_eq!(body.get("prompt"), template.get("prompt"));
            assert_eq!(template.get("seed"), &seed);
        }
    }
}
