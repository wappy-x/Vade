//! Standard-library HTTP/1.1 transport for text chat and typed native agents.
//! HTTPS is rejected: this project does not implement cryptography.
//! Protocol references: RFC 9112 (HTTP/1.1) and the WHATWG HTML SSE specification.
//! Implementation is original; no external transport or decoder code is used.
use crate::{
    agent::{Model, ModelRequest},
    agent_wire::{AssistantTurn, ChatAccumulator},
    json,
    protocol::{self, Completion, Message},
};
use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

mod kv;
pub use kv::ReleaseSummary;
mod native;
mod chat;
mod sse;
pub use native::{NativeModel, NativePhase, ThinkingUpdate};
mod cancel;
mod resolve;
mod request;
pub(crate) use request::{
    ControlFanout, request_gateway_pooled, request_method_pooled, request_method_pooled_deadline,
    request_native_pooled, request_native_pooled_deadline, request_pooled,
};
use request::{RequestPolicy, gateway_response_idle, request_with_policy};
pub use chat::{ChatClient, stream_chat, stream_chat_cancellable};
pub(crate) use sse::SseDecoder;

pub(crate) const MAX_HEADERS: usize = 32 * 1024;
pub(crate) const MAX_BODY: usize = 64 * 1024 * 1024;
const MAX_LINE: usize = 8 * 1024;
const MAX_EVENT: usize = 1024 * 1024;
const READ_BUFFER_BYTES: usize = 8 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
// Generation reads have their own long bound; ordinary I/O stays responsive.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const LIFETIME: Duration = Duration::from_secs(3600);
const POLL_TIMEOUT: Duration = Duration::from_millis(250);
const GATEWAY_POLL_TIMEOUT: Duration = Duration::from_millis(50);

pub(crate) use crate::json::invalid;

/// A trusted status code, separate from untrusted upstream error text.
#[derive(Debug)]
struct HttpStatus(u16);
impl std::fmt::Display for HttpStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "upstream HTTP status {}", self.0)
    }
}
impl std::error::Error for HttpStatus {}

/// Only errors created from a parsed HTTP response expose a status here.
/// An upstream body that happens to contain status-like text cannot spoof it.
pub fn http_status(error: &io::Error) -> Option<u16> {
    error.get_ref()?.downcast_ref::<HttpStatus>().map(|status| status.0)
}

#[derive(Debug)]
struct BodyLimitExceeded;
impl std::fmt::Display for BodyLimitExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HTTP body exceeds limit")
    }
}
impl std::error::Error for BodyLimitExceeded {}
fn body_limit_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, BodyLimitExceeded)
}
pub(crate) fn body_limit_exceeded(error: &io::Error) -> bool {
    error.get_ref().is_some_and(|e| e.is::<BodyLimitExceeded>())
}

#[derive(Clone, Debug)]
pub(crate) struct Endpoint {
    pub host: String,
    pub port: u16,
    pub authority: String,
    pub path: String,
}

impl Endpoint {
    /// Choose a sibling API route for known endpoints. Custom exact
    /// upstream paths remain exact; model/cache fields are never interpreted.
    pub(crate) fn gateway_route(&self, suffix: &str, query: Option<&str>) -> io::Result<Self> {
        let mut endpoint = self.clone();
        let (path, configured_query) = self
            .path
            .split_once('?')
            .map_or((self.path.as_str(), None), |(a, b)| (a, Some(b)));
        if !matches!(
            suffix,
            "/responses"
                | "/completions"
                | "/chat/completions"
                | "/messages"
                | "/messages/count_tokens"
                | "/capabilities"
                | "/models"
                | "/admission"
                | "/cancel"
                | "/kv/release"
                | "/health"
        ) {
            return Err(invalid("unsupported gateway route"));
        }
        if suffix == "/health" {
            // Engines answer health at the server root, outside the API base.
            endpoint.path = suffix.into();
            return Ok(endpoint);
        }
        endpoint.path = if let Some(base) = path
            .strip_suffix("/chat/completions")
            .or(match path {
                "/completions" => Some(""),
                "/v1/completions" => Some("/v1"),
                _ => None, // A custom exact endpoint is not an API suffix.
            })
            .or_else(|| path.strip_suffix("/responses"))
            .or_else(|| path.strip_suffix("/messages/count_tokens"))
            .or_else(|| path.strip_suffix("/messages"))
            .or_else(|| path.strip_suffix("/admission"))
            .or_else(|| path.strip_suffix("/capabilities"))
        {
            format!("{base}{suffix}")
        } else if path.trim_end_matches('/').ends_with("/v1") {
            format!("{}{suffix}", path.trim_end_matches('/'))
        } else {
            path.to_owned()
        };
        if let Some(configured) = configured_query {
            endpoint.path.push('?');
            endpoint.path.push_str(configured);
        }
        if let Some(query) = query {
            endpoint.path.push(if endpoint.path.contains('?') {
                '&'
            } else {
                '?'
            });
            endpoint.path.push_str(query);
        }
        if endpoint.path.len() > MAX_LINE {
            return Err(invalid("forwarded HTTP path exceeds limit"));
        }
        Ok(endpoint)
    }

    pub(crate) fn parse(endpoint: &str) -> io::Result<Self> {
        if endpoint.starts_with("https://") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "HTTPS is unsupported; configure an HTTP endpoint",
            ));
        }
        if endpoint.len() > MAX_LINE {
            return Err(invalid("HTTP endpoint exceeds size limit"));
        }
        let rest = endpoint
            .strip_prefix("http://")
            .ok_or_else(|| invalid("endpoint must use http://"))?;
        if !rest.is_ascii()
            || rest
                .bytes()
                .any(|b| b <= 0x20 || b == 0x7f || b == b'#' || b == b'\\')
        {
            return Err(invalid("invalid HTTP endpoint"));
        }
        let split = rest.find(['/', '?']).unwrap_or(rest.len());
        let authority = &rest[..split];
        if authority.is_empty() || authority.contains('@') {
            return Err(invalid(
                "endpoint credentials and empty hosts are unsupported",
            ));
        }
        let (host, port) = if let Some(ipv6) = authority.strip_prefix('[') {
            let close = ipv6
                .find(']')
                .ok_or_else(|| invalid("invalid IPv6 endpoint"))?;
            let host = &ipv6[..close];
            host.parse::<std::net::Ipv6Addr>()
                .map_err(|_| invalid("invalid IPv6 endpoint"))?;
            let tail = &ipv6[close + 1..];
            let port = if tail.is_empty() {
                80
            } else {
                parse_port(
                    tail.strip_prefix(':')
                        .ok_or_else(|| invalid("invalid endpoint port"))?,
                )?
            };
            (host.to_owned(), port)
        } else {
            let (host, port) = match authority.split_once(':') {
                Some((host, port)) => (host, parse_port(port)?),
                None => (authority, 80),
            };
            if host.is_empty()
                || !host
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-'))
            {
                return Err(invalid("invalid HTTP host"));
            }
            (host.to_owned(), port)
        };
        let raw_path = &rest[split..];
        let (path, query) = raw_path
            .split_once('?')
            .map_or((raw_path, None), |(path, query)| (path, Some(query)));
        let mut path = match path {
            "" | "/" => "/v1/chat/completions".to_owned(),
            "/v1" | "/v1/" => "/v1/chat/completions".to_owned(),
            path => path.to_owned(),
        };
        if let Some(query) = query {
            path.push('?');
            path.push_str(query);
        }
        if path.len() > MAX_LINE {
            return Err(invalid("endpoint path exceeds limit"));
        }
        Ok(Self {
            host,
            port,
            authority: authority.to_owned(),
            path,
        })
    }
}

fn parse_port(port: &str) -> io::Result<u16> {
    if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid("invalid endpoint port"));
    }
    let port = port
        .parse::<u16>()
        .map_err(|_| invalid("invalid endpoint port"))?;
    if port == 0 {
        return Err(invalid("endpoint port must be nonzero"));
    }
    Ok(port)
}

/// All socket I/O checks an absolute lifetime, an idle timeout, and cancellation.
/// DNS resolution is provided by the OS and is not cancellable through std.
pub(crate) struct DeadlineStream<'a> {
    stream: TcpStream,
    start: Instant,
    progress: Instant,
    idle_timeout: Duration,
    cancel: Option<&'a AtomicBool>,
    deadline: Option<Instant>,
    downstream: Option<Arc<Mutex<Downstream>>>,
    buffered_input: Option<Arc<Mutex<Downstream>>>,
    #[cfg(test)]
    on_read_wait: Option<Box<dyn FnOnce() + Send>>,
}

impl<'a> DeadlineStream<'a> {
    pub(crate) fn new(stream: TcpStream, cancel: Option<&'a AtomicBool>) -> io::Result<Self> {
        // Accepted sockets can inherit listener flags on some platforms.
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(POLL_TIMEOUT))?;
        stream.set_write_timeout(Some(POLL_TIMEOUT))?;
        stream.set_nodelay(true)?;
        Ok(Self {
            stream,
            start: Instant::now(),
            progress: Instant::now(),
            idle_timeout: IDLE_TIMEOUT,
            cancel,
            deadline: None,
            downstream: None,
            buffered_input: None,
            #[cfg(test)]
            on_read_wait: None,
        })
    }

    pub(crate) fn check(&self) -> io::Result<()> {
        if self.cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "request cancelled",
            ));
        }
        if self.start.elapsed() >= LIFETIME
            || self.progress.elapsed() >= self.idle_timeout
            || self.deadline.is_some_and(|at| Instant::now() >= at)
        {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "HTTP connection deadline exceeded",
            ));
        }
        for downstream in [self.downstream.as_ref(), self.buffered_input.as_ref()]
            .into_iter()
            .flatten()
        {
            downstream
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .check()?;
        }
        Ok(())
    }
    pub(crate) fn gateway_response(&mut self, method: &str, path: &str) {
        self.idle_timeout = gateway_response_idle(method, path);
    }
    pub(crate) fn watch_downstream(&mut self) -> io::Result<()> {
        self.buffered_input = Some(Arc::new(Mutex::new(Downstream {
            socket: self.stream.try_clone()?,
            deadline: self.request_deadline(),
            buffered: VecDeque::new(),
            failure: None,
        })));
        self.check()
    }
    pub(crate) fn watch_peer(&mut self, peer: &Self) {
        self.downstream = peer.buffered_input.clone();
        self.bound_deadline(peer.request_deadline());
    }
    pub(crate) fn finish_framing(&mut self) {
        self.downstream = None;
        self.bound_deadline(self.request_deadline().min(Instant::now() + Duration::from_secs(1)));
    }
    pub(crate) fn request_deadline(&self) -> Instant {
        self.deadline.unwrap_or(self.start + LIFETIME)
    }
    pub(crate) fn bound_deadline(&mut self, deadline: Instant) {
        self.deadline = Some(deadline);
    }
    fn poll_timeout(&self) -> Duration {
        let now = Instant::now();
        let mut remaining = (self.start + LIFETIME)
            .saturating_duration_since(now)
            .min((self.progress + self.idle_timeout).saturating_duration_since(now));
        if let Some(deadline) = self.deadline {
            remaining = remaining.min(deadline.saturating_duration_since(now));
        }
        let interval = if self.downstream.is_some() || self.buffered_input.is_some() {
            GATEWAY_POLL_TIMEOUT
        } else {
            POLL_TIMEOUT
        };
        interval.min(remaining).max(Duration::from_millis(1))
    }
}

// After a complete request, downstream EOF or a socket error cancels upstream;
// write-half-closed callers are unsupported. Read and buffer pipelined bytes so
// they cannot hide EOF or errors, retaining the original absolute deadline.
// Dispatch, polling and relay share one worker, so temporary nonblocking reads
// cannot overlap downstream writes. Buffer at most one maximum-sized request.
struct Downstream {
    socket: TcpStream,
    deadline: Instant,
    buffered: VecDeque<u8>,
    failure: Option<io::ErrorKind>,
}

impl Downstream {
    fn check(&mut self) -> io::Result<()> {
        if let Some(kind) = self.failure {
            return Err(io::Error::new(kind, "downstream unavailable"));
        }
        let result = self.poll();
        if let Err(error) = &result {
            self.failure = Some(error.kind());
        }
        result
    }

    fn poll(&mut self) -> io::Result<()> {
        if let Some(error) = self.socket.take_error()? {
            return Err(error);
        }
        self.socket.set_nonblocking(true)?;
        let result = self.buffer_available();
        let restored = self.socket.set_nonblocking(false);
        result.and(restored)
    }

    fn buffer_available(&mut self) -> io::Result<()> {
        let mut bytes = [0; READ_BUFFER_BYTES];
        let limit = MAX_HEADERS + json::MAX_INPUT_BYTES;
        loop {
            if Instant::now() >= self.deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "HTTP request deadline exceeded",
                ));
            }
            let remaining = limit.saturating_sub(self.buffered.len());
            let take = bytes.len().min(remaining + 1);
            match self.socket.read(&mut bytes[..take]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::ConnectionAborted,
                        "downstream closed",
                    ));
                }
                Ok(n) => {
                    if n > remaining {
                        return Err(invalid("pipelined HTTP input exceeds limit"));
                    }
                    if self.buffered.capacity() - self.buffered.len() < n {
                        let capacity = (self.buffered.capacity() * 2)
                            .max(READ_BUFFER_BYTES)
                            .min(limit);
                        self.buffered.reserve_exact(capacity - self.buffered.len());
                    }
                    self.buffered.extend(&bytes[..n]);
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }

    fn read_buffered(&mut self, out: &mut [u8]) -> Option<usize> {
        if !self.buffered.is_empty() {
            let n = out.len().min(self.buffered.len());
            for (target, byte) in out[..n].iter_mut().zip(self.buffered.drain(..n)) {
                *target = byte;
            }
            Some(n)
        } else {
            None
        }
    }
}

impl Read for DeadlineStream<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            self.check()?;
            if let Some(input) = &self.buffered_input {
                let n = input
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .read_buffered(buffer);
                if let Some(n) = n {
                    if n > 0 {
                        self.progress = Instant::now();
                    }
                    return Ok(n);
                }
            }
            self.stream.set_read_timeout(Some(self.poll_timeout()))?;
            match self.stream.read(buffer) {
                Ok(n) => {
                    if n > 0 {
                        self.progress = Instant::now();
                    }
                    return Ok(n);
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) =>
                {
                    #[cfg(test)]
                    if e.kind() != io::ErrorKind::Interrupted {
                        if let Some(on_wait) = self.on_read_wait.take() { on_wait(); }
                    }
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl Write for DeadlineStream<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        loop {
            self.check()?;
            self.stream.set_write_timeout(Some(self.poll_timeout()))?;
            match self.stream.write(buffer) {
                Ok(n) => {
                    if n > 0 {
                        self.progress = Instant::now();
                    }
                    return Ok(n);
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) =>
                {
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.check()?;
        self.stream.flush()
    }
}

#[derive(Clone)]
pub(crate) struct Head {
    pub first: String,
    pub fields: Vec<(String, String)>,
}

impl Head {
    pub(crate) fn connection_close(&self) -> bool {
        self.fields
            .iter()
            .filter(|(key, _)| key == "connection")
            .any(|(_, value)| {
                value
                    .split(',')
                    .any(|part| part.trim().eq_ignore_ascii_case("close"))
            })
    }

    pub(crate) fn get(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// Read one CRLF-terminated framing line without unbounded read_line allocation.
pub(crate) fn line<R: BufRead>(reader: &mut R, limit: usize) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete HTTP framing",
            ));
        }
        let take = bytes
            .iter()
            .position(|&b| b == b'\n')
            .map_or(bytes.len(), |i| i + 1);
        if out.len().saturating_add(take) > limit {
            return Err(invalid("HTTP line exceeds limit"));
        }
        out.extend_from_slice(&bytes[..take]);
        reader.consume(take);
        if out.last() == Some(&b'\n') {
            if out.len() < 2 || out[out.len() - 2] != b'\r' {
                return Err(invalid("HTTP framing requires CRLF"));
            }
            out.truncate(out.len() - 2);
            return Ok(out);
        }
    }
}

pub(crate) fn read_head<R: BufRead>(reader: &mut R) -> io::Result<Head> {
    read_head_with_first(reader, |_| {})
}

pub(crate) fn read_head_with_first<R: BufRead>(
    reader: &mut R,
    mut on_first: impl FnMut(&str),
) -> io::Result<Head> {
    let first = line(reader, MAX_LINE)?;
    let mut count = first.len() + 2;
    let first = String::from_utf8(first).map_err(|_| invalid("invalid HTTP start line"))?;
    if !first.is_ascii() || first.bytes().any(|b| b < b' ' || b == 0x7f) {
        return Err(invalid("invalid HTTP start line"));
    }
    on_first(&first);
    let mut fields = Vec::new();
    loop {
        let bytes = line(reader, MAX_LINE.min(MAX_HEADERS.saturating_sub(count)))?;
        count += bytes.len() + 2;
        if count > MAX_HEADERS {
            return Err(invalid("HTTP headers exceed limit"));
        }
        if bytes.is_empty() {
            break;
        }
        if fields.len() >= 100 {
            return Err(invalid("too many HTTP headers"));
        }
        let field = String::from_utf8(bytes).map_err(|_| invalid("invalid HTTP header"))?;
        let (name, value) = field
            .split_once(':')
            .ok_or_else(|| invalid("invalid HTTP header"))?;
        if name.is_empty() || !name.bytes().all(is_token) {
            return Err(invalid("invalid HTTP header name"));
        }
        if value.bytes().any(|b| (b < 0x20 && b != b'\t') || b == 0x7f) {
            return Err(invalid("invalid HTTP header value"));
        }
        fields.push((
            name.to_ascii_lowercase(),
            value.trim_matches([' ', '\t']).to_owned(),
        ));
    }
    let head = Head { first, fields };
    // Ambiguous framing is rejected even when duplicate lengths happen to agree.
    for name in [
        "content-length",
        "transfer-encoding",
        "host",
        "content-encoding",
        "content-type",
    ] {
        if head.fields.iter().filter(|(key, _)| key == name).count() > 1 {
            return Err(invalid("duplicate HTTP framing or representation header"));
        }
    }
    if head.get("content-length").is_some() && head.get("transfer-encoding").is_some() {
        return Err(invalid("conflicting HTTP body framing"));
    }
    Ok(head)
}

fn is_token(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || matches!(
            b,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

#[derive(Debug)]
enum Framing {
    Length(usize),
    Chunked {
        remaining: usize,
        needs_crlf: bool,
        done: bool,
    },
    Close,
}

pub(crate) struct BodyReader<R> {
    pub reader: R,
    framing: Framing,
    total: usize,
    limit: usize,
    chunks: usize,
}

impl<R: BufRead> BodyReader<R> {
    fn empty(reader: R) -> Self {
        Self {
            reader,
            framing: Framing::Length(0),
            total: 0,
            limit: 0,
            chunks: 0,
        }
    }
    pub(crate) fn new(reader: R, head: &Head, request: bool, limit: usize) -> io::Result<Self> {
        if head
            .get("content-encoding")
            .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
        {
            return Err(invalid("compressed HTTP bodies are unsupported"));
        }
        let framing = if let Some(encoding) = head.get("transfer-encoding") {
            if !encoding.eq_ignore_ascii_case("chunked") {
                return Err(invalid("unsupported HTTP transfer encoding"));
            }
            Framing::Chunked {
                remaining: 0,
                needs_crlf: false,
                done: false,
            }
        } else if let Some(length) = head.get("content-length") {
            if length.is_empty() || !length.bytes().all(|b| b.is_ascii_digit()) {
                return Err(invalid("invalid HTTP content length"));
            }
            let length = length
                .parse::<usize>()
                .map_err(|_| body_limit_error())?;
            if length > limit {
                return Err(body_limit_error());
            }
            Framing::Length(length)
        } else if request {
            Framing::Length(0)
        } else {
            Framing::Close
        };
        Ok(Self {
            reader,
            framing,
            total: 0,
            limit,
            chunks: 0,
        })
    }
}

impl<R: BufRead> BufRead for BodyReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if matches!(self.framing, Framing::Length(0)) {
            return Ok(&[]);
        }
        if let Framing::Chunked {
            remaining,
            needs_crlf,
            done,
        } = &mut self.framing
        {
            if *done {
                return Ok(&[]);
            }
            if *remaining == 0 {
                if *needs_crlf {
                    let mut crlf = [0; 2];
                    self.reader.read_exact(&mut crlf)?;
                    if crlf != *b"\r\n" {
                        return Err(invalid("invalid HTTP chunk terminator"));
                    }
                }
                self.chunks += 1;
                if self.chunks > 1_000_000 {
                    return Err(invalid("HTTP chunk count exceeds limit"));
                }
                let size = line(&mut self.reader, MAX_LINE)?;
                let size =
                    std::str::from_utf8(&size).map_err(|_| invalid("invalid HTTP chunk size"))?;
                let (number, extension) = size
                    .split_once(';')
                    .map_or((size, None), |(a, b)| (a, Some(b)));
                if number.is_empty()
                    || !number.bytes().all(|b| b.is_ascii_hexdigit())
                    || extension.is_some_and(|e| e.bytes().any(|b| b < 0x20 || b == 0x7f))
                {
                    return Err(invalid("invalid HTTP chunk size"));
                }
                *remaining = usize::from_str_radix(number, 16)
                    .map_err(|_| body_limit_error())?;
                if remaining.saturating_add(self.total) > self.limit {
                    return Err(body_limit_error());
                }
                if *remaining == 0 {
                    let mut trailer_bytes = 0;
                    loop {
                        let trailer = line(&mut self.reader, MAX_LINE)?;
                        trailer_bytes += trailer.len() + 2;
                        if trailer_bytes > MAX_HEADERS {
                            return Err(invalid("HTTP trailers exceed limit"));
                        }
                        if trailer.is_empty() {
                            break;
                        }
                        let (name, value) = trailer.split_at(
                            trailer
                                .iter()
                                .position(|&b| b == b':')
                                .ok_or_else(|| invalid("invalid HTTP trailer"))?,
                        );
                        if name.is_empty()
                            || !name.iter().copied().all(is_token)
                            || value.iter().any(|&b| b < 0x20 && b != b'\t' || b == 0x7f)
                        {
                            return Err(invalid("invalid HTTP trailer"));
                        }
                    }
                    *done = true;
                    return Ok(&[]);
                }
                *needs_crlf = true;
            }
        }
        let bytes = self.reader.fill_buf()?;
        let take = match self.framing {
            Framing::Length(0) => return Ok(&[]),
            Framing::Length(n) => n.min(bytes.len()),
            Framing::Chunked { remaining, .. } => remaining.min(bytes.len()),
            Framing::Close => bytes
                .len()
                .min(self.limit.saturating_sub(self.total).saturating_add(1)),
        };
        if take == 0 && !matches!(self.framing, Framing::Close) {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated HTTP body",
            ));
        }
        if take > self.limit.saturating_sub(self.total) {
            return Err(body_limit_error());
        }
        Ok(&bytes[..take])
    }
    fn consume(&mut self, n: usize) {
        self.reader.consume(n);
        self.total += n;
        match &mut self.framing {
            Framing::Length(left) => *left -= n,
            Framing::Chunked { remaining, .. } => *remaining -= n,
            Framing::Close => {}
        }
    }
}
impl<R: BufRead> Read for BodyReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let bytes = self.fill_buf()?;
        let n = buffer.len().min(bytes.len());
        buffer[..n].copy_from_slice(&bytes[..n]);
        self.consume(n);
        Ok(n)
    }
}
impl<R: BufRead> BodyReader<R> {
    fn complete(&self) -> bool {
        matches!(
            self.framing,
            Framing::Length(0) | Framing::Chunked { done: true, .. }
        )
    }
}

/// Idle sockets only; in-flight requests never hold this mutex. No POST replay.
#[derive(Clone)]
pub(crate) struct ConnectionPool(Arc<Mutex<Vec<(String, Instant, TcpStream)>>>);

impl Default for ConnectionPool {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}

impl ConnectionPool {
    fn take(&self, endpoint: &Endpoint) -> Option<TcpStream> {
        let mut idle = self.0.lock().unwrap_or_else(|e| e.into_inner());
        // Conservative idle expiry. A peer can still close after the probe:
        // that request fails explicitly; non-idempotent POSTs are never retried.
        idle.retain(|(_, at, _)| at.elapsed() < Duration::from_secs(15));
        while let Some(index) = idle
            .iter()
            .position(|(key, _, _)| key == &endpoint.authority)
        {
            let (_, _, socket) = idle.swap_remove(index);
            if socket.set_nonblocking(true).is_err() {
                continue;
            }
            let alive =
                matches!(socket.peek(&mut [0; 1]), Err(e) if e.kind() == io::ErrorKind::WouldBlock);
            if socket.set_nonblocking(false).is_ok() && alive {
                return Some(socket);
            }
        }
        None
    }
    fn put(&self, authority: String, socket: TcpStream) {
        let mut idle = self.0.lock().unwrap_or_else(|e| e.into_inner());
        idle.retain(|(_, at, _)| at.elapsed() < Duration::from_secs(15));
        if idle.len() < 16 {
            idle.push((authority, Instant::now(), socket));
        }
    }
}

pub(crate) struct Response<'a> {
    pub status: u16,
    pub head: Head,
    pub body: BodyReader<BufReader<DeadlineStream<'a>>>,
    reuse: Option<(ConnectionPool, String)>,
}

impl Response<'_> {
    /// Cancellation/deadline replies can arrive before SSE headers. Bound their
    /// JSON envelope and expose only a recognized code; all other HTTP failures
    /// keep the status-only diagnostic used for untrusted upstream bodies.
    fn failure(&mut self) -> io::Error {
        if matches!(self.status, 409 | 504) {
            const ERROR_BYTES: u64 = 8192;
            let mut body = Vec::new();
            if self.body.by_ref().take(ERROR_BYTES + 1).read_to_end(&mut body).is_ok()
                && body.len() <= ERROR_BYTES as usize
            {
                if let Ok(value) = json::parse_bytes(&body) {
                    if let Some(error) = value.get("error") {
                        let error = protocol::response_error(error, "upstream returned an error");
                        if matches!(error.kind(), io::ErrorKind::Interrupted | io::ErrorKind::TimedOut) {
                            return error;
                        }
                    }
                }
            }
        }
        io::Error::other(HttpStatus(self.status))
    }

    /// Called only after successful consumption and validation. Incomplete,
    /// canceled and failed responses drop their connection instead.
    pub(crate) fn recycle(self) {
        if self.body.complete() && self.body.reader.buffer().is_empty() {
            if let Some((pool, authority)) = self.reuse {
                pool.put(authority, self.body.reader.into_inner().stream);
            }
        }
    }
}

pub(crate) fn hop_header(name: &str, head: &Head) -> bool {
    matches!(
        name,
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "host"
            | "content-length"
    ) || head
        .fields
        .iter()
        .filter(|(key, _)| key == "connection")
        .any(|(_, value)| {
            value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case(name))
        })
}

#[cfg(test)]
mod tests;
