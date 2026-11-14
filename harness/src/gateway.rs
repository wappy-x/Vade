//! Bounded loopback-only chat and Responses relay. JSON bodies are validated, then forwarded
//! with exact sampling defaults; explicit fields retain their original bytes.
//! Response bodies are relayed incrementally, including upstream SSE bytes.
use crate::{
    json,
    transport::{self, BodyReader, ConnectionPool, DeadlineStream, Endpoint},
};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

mod access;
mod routing;
mod sampling;
mod system_turns;
mod usage;
pub mod invisible;
pub use access::{Access, TOKEN_ENV};
pub use routing::{Cluster, EndpointStats, RoutingConfig, RoutingStats};

const MAX_CONNECTIONS: usize = 64;
const DEFAULT_BIND: &str = "127.0.0.1:8787";
const STATS_PATH: &str = "/_vade/gateway/stats";
/// The shared field-presence sampling policy (engine-model `sampling_policy.rs`),
/// exposed so public help and documentation tests can state it exactly.
pub const SAMPLING_DEFAULTS_JSON: &str = sampling::DEFAULTS_JSON;
pub const SAMPLING_FIELDS: &str = sampling::SAMPLING_FIELDS;
/// Every path the core relay accepts; `vade-harness gateway --help` lists them.
pub const ROUTES: &[&str] = &[
    "/v1/completions",
    "/completions",
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/responses",
    "/responses",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/capabilities",
    "/v1/models",
    "/models",
    "/v1/admission",
    "/v1/cancel",
    "/v1/kv/release",
    "/health",
];
/// OpenAI routes this text-only engine does not serve. The gateway answers them
/// itself with 501 and a plain reason, so a client does not read a bare 404 as
/// a wrong base URL.
const UNSERVED_ROUTES: &[&str] = &["/v1/embeddings", "/embeddings"];
const UNSERVED_MESSAGE: &str = "embeddings are not implemented: Vade serves text generation only; use a separate embedding server";
/// The read-only routes; every other accepted route takes POST.
const GET_ROUTES: &[&str] = &["/v1/capabilities", "/v1/models", "/models", "/health"];
/// `vade-gateway --help`; `vade-harness gateway --help` states the same rules.
const BROWSER_GUARD_HELP: &str = "Browser guard: Host must be 127.0.0.1, localhost or [::1] (with the listener port) or an --allow-host value, else 421.\nAny request with an Origin header gets 403 unless --allow-origin names that exact origin (repeatable); allowed origins get CORS headers.\n--token TOKEN (or VADE_GATEWAY_TOKEN) requires Authorization: Bearer TOKEN, else 401; that header is not forwarded upstream.\nPrefer VADE_GATEWAY_TOKEN: other local users can read a --token value in the process list.\n";
/// How long one request may wait out upstream 503 refusals before the gateway
/// passes the refusal on, and how often the wait checks for a client hang-up.
const ADMISSION_PATIENCE: Duration = Duration::from_secs(if cfg!(test) { 2 } else { 30 });
const ADMISSION_POLL: Duration = Duration::from_millis(50);
/// The largest refusal body held while waiting; larger refusals pass on at once.
const HELD_REFUSAL_BYTES: usize = 4096;

/// The pause before re-sending after an upstream 503: 50 ms doubling to 800 ms,
/// or one second when Retry-After says 0 or 1. A longer Retry-After is the
/// upstream's advice to the client, so that refusal is passed on unchanged.
fn admission_delay(retry_after: Option<&str>, waits: u32) -> Option<Duration> {
    match retry_after.map(|s| s.trim().parse::<u64>()) {
        None => Some(Duration::from_millis(50 << waits.min(4))),
        Some(Ok(0 | 1)) => Some(Duration::from_secs(1)),
        Some(_) => None,
    }
}

/// The engine's refusal of a request it has read but not admitted (all its
/// connections are busy). Other 503s, such as capacity running out after
/// tokens were generated, a model swap or a shutdown, may follow work.
fn admission_refusal(body: &[u8]) -> bool {
    json::parse_bytes(body).ok().and_then(|value| value.get("error").map(|error| {
        error.get("type").and_then(json::Value::as_str) == Some("overloaded")
            && error.get("message").and_then(json::Value::as_str) == Some("connection capacity reached")
    })).unwrap_or(false)
}
/// The bind refusal; the hint table matches this exact text.
pub(crate) const LOOPBACK_ONLY: &str = "gateway bind must resolve only to loopback addresses";
pub(crate) const MARKER_HEADER: &str = "x-vade-gateway";
pub(crate) const MARKER_VALUE: &str = "v1";

#[derive(Clone)]
enum Upstream {
    Single(Endpoint),
    Cluster(Cluster),
}

fn prepare_dispatch<'a>(
    upstream: &Upstream,
    method: &str,
    path: &str,
    query: Option<&str>,
    head: &transport::Head,
    client_body: &'a [u8],
) -> Result<(Option<routing::Lease>, std::borrow::Cow<'a, [u8]>), routing::RouteError> {
    let lease = match upstream {
        Upstream::Single(_) => None,
        Upstream::Cluster(cluster) => Some(cluster.select(path, query, client_body, head)?),
    };
    let body = system_turns::fold(method, path, std::borrow::Cow::Borrowed(client_body));
    Ok((lease, sampling::apply_to(method, path, head, client_body, body).0))
}

/// Serve the core gateway on `bind` in front of one upstream until the process
/// ends. See [`run_with_access`] for browser and token checks.
pub fn run(bind: &str, upstream: &str) -> io::Result<()> {
    run_with_access(bind, upstream, Access::default())
}

/// [`run`] with browser allow-lists and an optional client token; see [`Access`].
pub fn run_with_access(bind: &str, upstream: &str, access: Access) -> io::Result<()> {
    serve(bind, Upstream::Single(Endpoint::parse(upstream)?), None, access)
}

/// Stops accepting connections and cancels in-flight socket I/O when the owner
/// sets the flag. Requests remain bounded by idle / absolute I/O deadlines.
pub fn run_until(bind: &str, upstream: &str, stop: &AtomicBool) -> io::Result<()> {
    serve(
        bind,
        Upstream::Single(Endpoint::parse(upstream)?),
        Some(stop),
        Access::default(),
    )
}

/// Opt-in, capability-verified routing. One dispatch per caller request.
pub fn run_cluster(bind: &str, cluster: Cluster) -> io::Result<()> {
    run_cluster_with_access(bind, cluster, Access::default())
}

/// [`run_cluster`] with browser allow-lists and an optional client token.
pub fn run_cluster_with_access(bind: &str, cluster: Cluster, access: Access) -> io::Result<()> {
    serve(bind, Upstream::Cluster(cluster), None, access)
}

/// [`run_cluster`] until the owner sets `stop`, like [`run_until`].
pub fn run_cluster_until(bind: &str, cluster: Cluster, stop: &AtomicBool) -> io::Result<()> {
    serve(bind, Upstream::Cluster(cluster), Some(stop), Access::default())
}

fn serve(bind: &str, upstream: Upstream, stop: Option<&AtomicBool>, access: Access) -> io::Result<()> {
    let listener = bind_loopback(bind)?;
    let access = Arc::new(access);
    // The standalone gateway can sleep in accept instead of adding a polling
    // interval to every fresh connection. Embedded owners still poll for stop.
    listener.set_nonblocking(stop.is_some())?;
    let pool = ConnectionPool::default();
    let active = Arc::new(AtomicUsize::new(0));
    let cancellation = Arc::new(AtomicBool::new(false));
    let _stop_guard = CancelOnDrop(Arc::clone(&cancellation));
    let _probes = match &upstream {
        Upstream::Cluster(cluster) => Some(ProbeWorkers::start(cluster, &cancellation)?),
        Upstream::Single(_) => None,
    };
    loop {
        if stop.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Ok(());
        }
        match listener.accept() {
            Ok((socket, peer)) => {
                if !peer.ip().is_loopback() {
                    continue;
                }
                if active.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
                    // A busy peer must not be allowed to hold the accept loop.
                    let _ = reject_busy(socket);
                    continue;
                }
                active.fetch_add(1, Ordering::Relaxed);
                let worker_active = Arc::clone(&active);
                let upstream = upstream.clone();
                let pool = pool.clone();
                let cancellation = Arc::clone(&cancellation);
                let access = Arc::clone(&access);
                let spawned =
                    thread::Builder::new()
                        .name("gateway-request".into())
                        .spawn(move || {
                            let _guard = ActiveGuard(worker_active);
                            // No request headers, body, credentials, or upstream URL are logged.
                            let _ = handle_target(socket, &upstream, Some(&cancellation), &pool, &access);
                        });
                if spawned.is_err() {
                    // A failed spawn drops the closure; its guard wasn't constructed.
                    active_decrement(&active);
                    return Err(io::Error::other("unable to start gateway request worker"));
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(25))
            }
            Err(e) => accept_failed(&e),
        }
    }
}

struct ProbeWorkers {
    cancellation: Arc<AtomicBool>,
    workers: Vec<thread::JoinHandle<()>>,
}
impl ProbeWorkers {
    fn start(cluster: &Cluster, cancellation: &Arc<AtomicBool>) -> io::Result<Self> {
        let mut result = Self {
            cancellation: cancellation.clone(),
            workers: Vec::new(),
        };
        for index in 0..cluster.len() {
            let cluster = cluster.clone();
            let cancel = cancellation.clone();
            result.workers.push(
                thread::Builder::new()
                    .name("gateway-capabilities".into())
                    .spawn(move || {
                        while !cancel.load(Ordering::Relaxed) {
                            cluster.refresh(index, &cancel);
                            if cancel.load(Ordering::Relaxed) {
                                break;
                            }
                            thread::park_timeout(cluster.probe_interval());
                        }
                    })?,
            );
        }
        Ok(result)
    }
}
impl Drop for ProbeWorkers {
    fn drop(&mut self) {
        self.cancellation.store(true, Ordering::Relaxed);
        for worker in &self.workers {
            worker.thread().unpark();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// A failed accept belongs to one connection (an aborted handshake, a full
/// descriptor table), not to the listener: log it, back off briefly, keep serving.
fn accept_failed(error: &io::Error) {
    if error.kind() != io::ErrorKind::Interrupted {
        crate::diagnostics::stderr_line(format_args!("vade-gateway: accept failed ({}); still listening.", error.kind()));
        thread::sleep(Duration::from_millis(50));
    }
}

fn active_decrement(active: &AtomicUsize) {
    active.fetch_sub(1, Ordering::Relaxed);
}
struct ActiveGuard(Arc<AtomicUsize>);
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        active_decrement(&self.0);
    }
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn bind_loopback(bind: &str) -> io::Result<TcpListener> {
    let bind = if bind.is_empty() { DEFAULT_BIND } else { bind };
    let addresses: Vec<_> = bind.to_socket_addrs()?.take(17).collect();
    if addresses.is_empty()
        || addresses.len() > 16
        || addresses.iter().any(|address| !address.ip().is_loopback())
    {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, LOOPBACK_ONLY));
    }
    TcpListener::bind(addresses.as_slice())
}

fn messages_route(first: &str) -> bool {
    let target = first.split(' ').nth(1).unwrap_or("");
    let path = target.split('?').next().unwrap_or(target);
    matches!(
        path,
        "/v1/messages" | "/v1/messages/count_tokens" | "/messages" | "/messages/count_tokens"
    )
}

fn reject_busy(socket: TcpStream) -> io::Result<()> {
    let mut stream = DeadlineStream::new(socket.try_clone()?, None)?;
    stream.bound_deadline(Instant::now() + Duration::from_millis(100));
    socket.set_read_timeout(Some(Duration::from_millis(100)))?;
    socket.set_write_timeout(Some(Duration::from_millis(100)))?;
    let mut reader = BufReader::new(stream);
    let mut messages = false;
    let _ = transport::read_head_with_first(&mut reader, |first| messages = messages_route(first));
    write_error(&mut &socket, messages, 503, "gateway is busy")
}

/// The body of an error the gateway answers itself. Messages routes get the
/// Anthropic shape; every other route gets the OpenAI shape with a `type`, as
/// the engine's own errors carry, so SDKs can tell throttling from failure.
/// The OpenAI shape also carries `param` and `code`, as the OpenAI schema does.
pub(crate) fn error_body(messages: bool, status: u16, message: &str) -> String {
    let kind = match status {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        501 => "invalid_request_error",
        503 | 529 if messages => "overloaded_error",
        503 | 529 => "overloaded",
        500..=599 if messages => "api_error",
        500..=599 => "server_error",
        _ => "invalid_request_error",
    };
    let code = match status {
        401 => "\"invalid_api_key\"",
        501 => "\"unsupported_endpoint\"",
        _ => "null",
    };
    if messages {
        format!("{{\"type\":\"error\",\"error\":{{\"type\":\"{kind}\",\"message\":{}}}}}", json::quote(message))
    } else {
        format!("{{\"error\":{{\"type\":\"{kind}\",\"message\":{},\"param\":null,\"code\":{code}}}}}", json::quote(message))
    }
}

/// Like every whole reply below, this ends without a flush: the downstream
/// socket is unbuffered, and a client may close on the last byte, which a
/// checked flush would report as a failed delivery (see `relay_observed`).
fn write_error(
    writer: &mut dyn Write,
    messages: bool,
    status: u16,
    message: &str,
) -> io::Result<()> {
    let body = error_body(messages, status, message);
    write!(
        writer,
        "HTTP/1.1 {status} Error\r\nX-Vade-Gateway: v1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn reject(
    reader: &mut BufReader<DeadlineStream<'_>>,
    messages: bool,
    status: u16,
    message: &str,
) -> io::Result<()> {
    write_error(reader.get_mut(), messages, status, message)
}

fn request_body_error(error: &io::Error) -> (u16, String) {
    if transport::body_limit_exceeded(error) {
        (413, format!("HTTP request body exceeds {} byte limit (8 MiB)", json::MAX_INPUT_BYTES))
    } else {
        (400, "invalid HTTP request framing or body".into())
    }
}

#[cfg(test)]
fn handle(socket: TcpStream, endpoint: &Endpoint) -> io::Result<()> {
    handle_cancellable(socket, endpoint, None)
}

#[cfg(test)]
fn handle_cancellable(
    socket: TcpStream,
    endpoint: &Endpoint,
    cancel: Option<&AtomicBool>,
) -> io::Result<()> {
    handle_pooled(socket, endpoint, cancel, &ConnectionPool::default())
}

#[cfg(test)]
fn handle_pooled(
    socket: TcpStream,
    endpoint: &Endpoint,
    cancel: Option<&AtomicBool>,
    pool: &ConnectionPool,
) -> io::Result<()> {
    handle_target(socket, &Upstream::Single(endpoint.clone()), cancel, pool, &Access::default())
}

fn handle_target(
    socket: TcpStream,
    upstream: &Upstream,
    cancel: Option<&AtomicBool>,
    pool: &ConnectionPool,
    access: &Access,
) -> io::Result<()> {
    let local = socket.local_addr()?;
    let mut reader = BufReader::new(DeadlineStream::new(socket, cancel)?);
    let mut messages = false;
    let mut head = match transport::read_head_with_first(&mut reader, |first| {
        messages = messages_route(first)
    }) {
        Ok(head) => head,
        Err(_) => {
            return reject(
                &mut reader,
                messages,
                400,
                "invalid or oversized HTTP headers",
            );
        }
    };
    let parts: Vec<_> = head.first.split(' ').collect();
    if parts.len() != 3 || !matches!(parts[2], "HTTP/1.1" | "HTTP/1.0") {
        return reject(&mut reader, messages, 400, "invalid HTTP request line");
    }
    let method = parts[0];
    let origin = match access.check(&head, local) {
        Ok(origin) => origin,
        Err(access::Refusal::Status(status, message)) => {
            return reject(&mut reader, messages, status, message);
        }
        Err(access::Refusal::Preflight(headers)) => {
            return access::write_preflight(reader.get_mut(), "X-Vade-Gateway: v1\r\n", &headers);
        }
    };
    if access.consumes_authorization() {
        head.fields.retain(|(key, _)| key != "authorization");
    }
    let (path, query) = parts[1]
        .split_once('?')
        .map_or((parts[1], None), |(a, b)| (a, Some(b)));
    if parts[1].contains(['#', '\\']) {
        return reject(&mut reader, messages, 400, "invalid API request target");
    }
    let local_stats = path == STATS_PATH && matches!(upstream, Upstream::Cluster(_));
    // One engine's /health is its own; a cluster reports per endpoint in its stats.
    let cluster_health = path == "/health" && matches!(upstream, Upstream::Cluster(_));
    if UNSERVED_ROUTES.contains(&path) {
        return reject(&mut reader, messages, 501, UNSERVED_MESSAGE);
    }
    if !local_stats
        && (!ROUTES.contains(&path) || cluster_health)
    {
        return reject(
            &mut reader,
            messages,
            404,
            "gateway supports completions, chat, Responses, Messages, token count, capabilities, models, admission, cancel and checkpoint release routes",
        );
    }
    if method
        != if GET_ROUTES.contains(&path) || local_stats {
            "GET"
        } else {
            "POST"
        }
    {
        return reject(
            &mut reader,
            messages,
            405,
            "unsupported method for API route",
        );
    }
    if head.get("expect").is_some() {
        return reject(&mut reader, messages, 417, "Expect is unsupported");
    }
    if head.get("upgrade").is_some() {
        return reject(
            &mut reader,
            messages,
            400,
            "protocol upgrades are unsupported",
        );
    }
    if head
        .get("content-encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return reject(
            &mut reader,
            messages,
            415,
            "compressed request bodies are unsupported",
        );
    }
    if method == "POST"
        && head.get("content-type").is_none_or(|value| {
            !value
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("application/json")
        })
    {
        return reject(&mut reader, messages, 415, "API requests must contain JSON");
    }
    // Validate framing before moving the reader so malformed input receives a response.
    if let Err(error) = BodyReader::new(io::Cursor::new(&[]), &head, true, json::MAX_INPUT_BYTES) {
        let (status, message) = request_body_error(&error);
        return reject(&mut reader, messages, status, &message);
    }
    let mut body_reader = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES)?;
    let mut body = Vec::new();
    if let Err(error) = body_reader.read_to_end(&mut body) {
        let (status, message) = request_body_error(&error);
        return reject(&mut body_reader.reader, messages, status, &message);
    }
    let mut reader = body_reader.reader;
    if method == "GET" {
        if !body.is_empty() {
            return reject(&mut reader, messages, 400, "GET body is unsupported");
        }
    } else {
        let object = std::str::from_utf8(&body)
            .ok()
            .and_then(|s| json::view(s).ok())
            .is_some_and(|v| v.fields().is_some());
        if !object {
            return reject(
                &mut reader,
                messages,
                400,
                "API body must be a valid JSON object",
            );
        }
    }
    if local_stats {
        if let Upstream::Cluster(cluster) = upstream {
            let body = cluster.stats().to_json();
            let writer = reader.get_mut();
            write!(
                writer,
                "HTTP/1.1 200 OK\r\nX-Vade-Gateway: v1\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )?;
            return writer.flush();
        }
    }
    let client_body = body;
    // The downstream socket also waits silently during a non-stream cold prefill.
    reader.get_mut().gateway_response(method, path);
    reader.get_mut().watch_downstream()?;
    if let (Upstream::Cluster(cluster), Some(field)) = (upstream, broadcast_field(path)) {
        let request = Request { method, path, query, head: &head, body: &client_body, origin: origin.as_deref() };
        return broadcast(&mut reader, cluster, request, field, cancel, pool, messages);
    }
    let (mut lease, body) = match prepare_dispatch(upstream, method, path, query, &head, &client_body) {
        Ok(prepared) => prepared,
        Err(routing::RouteError(status, message)) => {
            return reject(&mut reader, messages, status, &message);
        }
    };
    let mut retried = false;
    let admission = Instant::now() + ADMISSION_PATIENCE;
    let mut waits = 0u32;
    let mut held = None;
    let mut response = loop {
        let endpoint = match upstream {
            Upstream::Single(endpoint) => endpoint,
            Upstream::Cluster(_) => lease.as_ref().unwrap().endpoint(),
        };
        let endpoint = match endpoint.gateway_route(path.strip_prefix("/v1").unwrap_or(path), query)
        {
            Ok(endpoint) => endpoint,
            Err(_) => {
                return reject(
                    &mut reader,
                    messages,
                    414,
                    "forwarded API path exceeds limit",
                );
            }
        };
        if let Some(lease) = &mut lease {
            lease.start();
        }
        let attempt = transport::request_gateway_pooled(
            method,
            &endpoint,
            &body,
            &head,
            cancel,
            pool,
            reader.get_ref(),
        );
        if let (Some(lease), Ok(response)) = (&mut lease, &attempt) {
            lease.status(response.status, response.head.get("retry-after"));
        }
        let refused = match &attempt {
            Ok(response) => matches!(response.status, 429 | 503 | 529),
            Err(failure) => failure.connect_failed,
        };
        reader.get_ref().check()?;
        if refused && !retried && !cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            if let (Upstream::Cluster(cluster), Some(old)) = (upstream, lease.as_mut()) {
                let excluded = old.index();
                // Retain the original response and lease when no alternate is
                // available, so forwarding/framing failures still quarantine it.
                if let Ok(next) =
                    cluster.select_excluding(path, query, &client_body, &head, Some(excluded))
                {
                    if let Ok(response) = &attempt {
                        old.finish(response.status, None);
                    }
                    lease = Some(next);
                    retried = true;
                    continue;
                }
            }
        }
        // The engine refuses a request it cannot admit before any work starts
        // (it reads the whole request first), so a lone upstream may take the
        // request once a finishing stream frees its slot. Wait briefly instead
        // of passing a momentary refusal to the client. Every other 503 may
        // follow work and reaches the client once. Clusters keep their single
        // fallback above.
        let attempt = match (attempt, upstream) {
            (Ok(mut refusal), Upstream::Single(_)) if refusal.status == 503
                && refusal.head.get("content-length").and_then(|n| n.parse::<usize>().ok()).is_some_and(|n| n <= HELD_REFUSAL_BYTES)
                && !cancel.is_some_and(|c| c.load(Ordering::Relaxed)) => {
                // The patience covers reading the refusal as well as the pauses.
                let limit = admission.min(reader.get_ref().request_deadline());
                refusal.body.reader.get_mut().bound_deadline(limit);
                let mut bytes = Vec::new();
                if refusal.body.read_to_end(&mut bytes).is_err() {
                    return reject(&mut reader, messages, 502, "HTTP upstream refusal was incomplete; request was not replayed");
                }
                let delay = admission_delay(refusal.head.get("retry-after"), waits)
                    .filter(|delay| admission_refusal(&bytes) && Instant::now() + *delay < limit);
                let Some(delay) = delay else {
                    return relay_whole(reader.get_mut(), messages, (refusal.status, &refusal.head, &bytes), method, path, origin.as_deref());
                };
                let until = Instant::now() + delay;
                while Instant::now() < until && !cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                    reader.get_ref().check()?;
                    thread::sleep(until.saturating_duration_since(Instant::now()).min(ADMISSION_POLL));
                }
                if Instant::now() >= limit {
                    return relay_whole(reader.get_mut(), messages, (refusal.status, &refusal.head, &bytes), method, path, origin.as_deref());
                }
                // Keep the refusal: if the upstream cannot be reached again, the
                // client still gets this answer rather than a gateway error.
                held = Some((refusal.status, refusal.head, bytes));
                waits = waits.saturating_add(1);
                continue;
            }
            (attempt, _) => attempt,
        };
        // Only a failed fresh connection proves the retry sent nothing; any
        // later failure leaves its outcome unknown, reported below as such.
        if let (Err(failure), Some((status, refusal, bytes))) = (&attempt, held.take()) {
            if failure.connect_failed {
                return relay_whole(reader.get_mut(), messages, (status, &refusal, &bytes), method, path, origin.as_deref());
            }
        }
        match attempt {
            Ok(response) => break response,
            Err(failure) => {
                if matches!(
                    failure.error.kind(),
                    io::ErrorKind::ConnectionAborted | io::ErrorKind::TimedOut
                ) {
                    return Err(failure.error);
                }
                return reject(
                    &mut reader,
                    messages,
                    502,
                    if retried {
                        "HTTP compatible fallback failed; outcome may be unknown; no further replay"
                    } else {
                        "HTTP upstream request failed; outcome may be unknown; request was not replayed"
                    },
                );
            }
        }
    };
    let writer = reader.get_mut();
    let no_body = matches!(response.status, 204 | 304);
    let framing = if no_body { "" } else { "Transfer-Encoding: chunked\r\n" };
    let mut headers = response_head(response.status, &response.head, method, path, origin.as_deref(), framing);
    if headers.len() + 2 > transport::MAX_HEADERS {
        return write_error(
            writer,
            messages,
            502,
            "upstream response headers exceed limit",
        );
    }
    headers.push_str("\r\n");
    writer.write_all(headers.as_bytes())?;
    if no_body {
        if let Some(lease) = &mut lease {
            lease.finish(response.status, None);
        }
        response.recycle();
        return Ok(());
    }
    writer.flush()?;
    let mut usage = usage::UsageTap::new(
        response.head.get("content-type"),
        lease.is_some()
            && (200..300).contains(&response.status)
            && (path.ends_with("/responses")
                || path.ends_with("/completions")
                || path.ends_with("/messages")),
    );
    relay_observed(&mut response.body, writer, |bytes| usage.feed(bytes))?;
    if let Some(lease) = &mut lease {
        lease.finish(response.status, usage.finish());
    }
    response.recycle();
    Ok(())
}

/// Forward an upstream reply already read in full, with its own length.
fn relay_whole(writer: &mut dyn Write, messages: bool, (status, head, bytes): (u16, &transport::Head, &[u8]),
    method: &str, path: &str, origin: Option<&str>) -> io::Result<()> {
    let mut headers = response_head(status, head, method, path, origin, &format!("Content-Length: {}\r\n", bytes.len()));
    if headers.len() + 2 > transport::MAX_HEADERS {
        return write_error(writer, messages, 502, "upstream response headers exceed limit");
    }
    headers.push_str("\r\n");
    writer.write_all(headers.as_bytes())?;
    writer.write_all(bytes)
}

/// The gateway's status line and headers for an upstream reply, without the
/// blank line that ends them. Hop-by-hop and marker headers stay upstream, and
/// an allowed browser gets exactly one CORS answer: the gateway's.
fn response_head(status: u16, upstream: &transport::Head, method: &str, path: &str, origin: Option<&str>, framing: &str) -> String {
    let mut headers = format!("HTTP/1.1 {status} Upstream\r\nX-Vade-Gateway: v1\r\nConnection: close\r\n");
    headers.push_str(&sampling::capabilities_header(method, path));
    headers.push_str(&access::cors(origin));
    headers.push_str(framing);
    // Tell clients when to try again, even if the upstream left it out.
    if matches!(status, 429 | 503) && upstream.get("retry-after").is_none() {
        headers.push_str("Retry-After: 1\r\n");
    }
    for (name, value) in &upstream.fields {
        if !transport::hop_header(name, upstream) && !name.eq_ignore_ascii_case(MARKER_HEADER)
            && !(origin.is_some() && name.starts_with("access-control-"))
        {
            headers.push_str(name);
            headers.push_str(": ");
            headers.push_str(value);
            headers.push_str("\r\n");
        }
    }
    headers
}

/// Cancel and checkpoint-release keys belong to the endpoint that ran the
/// request, which cluster routing does not record. With several upstreams
/// these calls go to every verified endpoint; keys are random capabilities, so
/// an unknown key changes nothing there. Returns the JSON field that reports
/// success.
fn broadcast_field(path: &str) -> Option<&'static str> {
    match path {
        "/v1/cancel" => Some("cancelled"),
        "/v1/kv/release" => Some("released"),
        _ => None,
    }
}

/// One validated client request, as the broadcast forwards it.
struct Request<'a> {
    method: &'a str,
    path: &'a str,
    query: Option<&'a str>,
    head: &'a transport::Head,
    body: &'a [u8],
    origin: Option<&'a str>,
}

/// How long a cancel or checkpoint release may take across all endpoints. The
/// native cancel client waits two seconds in total, so one stalled endpoint must
/// not hold the answer past that.
const CONTROL_BUDGET: Duration = Duration::from_millis(1500);

/// One endpoint's control reply: status, head, body, and whether it reported success.
type ControlReply = (u16, transport::Head, Vec<u8>, bool);

/// Send one request to every verified endpoint at once, reading each bounded
/// reply within [`CONTROL_BUDGET`], and answer once: the first 2xx reply (in
/// endpoint order) whose `field` is true, otherwise the first reply. Endpoints
/// that fail or stall are skipped; nothing is replayed, and a foreign endpoint
/// never sees the key.
fn broadcast(
    reader: &mut BufReader<DeadlineStream<'_>>,
    cluster: &Cluster,
    request: Request<'_>,
    field: &str,
    cancel: Option<&AtomicBool>,
    pool: &ConnectionPool,
    messages: bool,
) -> io::Result<()> {
    let suffix = request.path.strip_prefix("/v1").unwrap_or(request.path);
    let fanout = transport::ControlFanout::new(reader.get_ref(), CONTROL_BUDGET)?;
    let endpoints = cluster.verified_endpoints();
    let replies: Vec<Option<ControlReply>> = thread::scope(|scope| {
        let calls: Vec<_> = endpoints
            .iter()
            .map(|endpoint| {
                let (fanout, request) = (&fanout, &request);
                scope.spawn(move || {
                    let target = endpoint.gateway_route(suffix, request.query).ok()?;
                    let mut response = fanout
                        .request(request.method, &target, request.body, request.head, cancel, pool)
                        .ok()?;
                    let mut body = Vec::new();
                    response.body.read_to_end(&mut body).ok()?;
                    let succeeded = (200..300).contains(&response.status)
                        && std::str::from_utf8(&body).ok().and_then(|text| json::view(text).ok())
                            .and_then(|value| value.get(field)).is_some_and(|value| value.raw() == "true");
                    let (status, head) = (response.status, response.head.clone());
                    response.recycle();
                    Some((status, head, body, succeeded))
                })
            })
            .collect();
        calls.into_iter().map(|call| call.join().ok().flatten()).collect()
    });
    let mut chosen: Option<ControlReply> = None;
    for (status, head, body, succeeded) in replies.into_iter().flatten() {
        if chosen.as_ref().is_none_or(|(_, _, _, done)| succeeded && !done) {
            chosen = Some((status, head, body, succeeded));
        }
    }
    reader.get_ref().check()?;
    let Some((status, head, body, _)) = chosen else {
        return reject(reader, messages, 502, "no upstream answered; request was not replayed");
    };
    let no_body = matches!(status, 204 | 304);
    let framing = if no_body { String::new() } else { format!("Content-Length: {}\r\n", body.len()) };
    let mut headers = response_head(status, &head, request.method, request.path, request.origin, &framing);
    let writer = reader.get_mut();
    if headers.len() + 2 > transport::MAX_HEADERS {
        return write_error(writer, messages, 502, "upstream response headers exceed limit");
    }
    headers.push_str("\r\n");
    writer.write_all(headers.as_bytes())?;
    if !no_body {
        writer.write_all(&body)?;
    }
    Ok(())
}

/// No read-ahead queue: downstream write/flush completes before the next body
/// read. The transport's BufReader bounds in-process response data to 8 KiB.
/// Socket send/receive buffers add finite OS buffering outside this bound.
#[cfg(test)]
fn relay_body(body: &mut impl BufRead, writer: &mut impl Write) -> io::Result<()> {
    relay_observed(body, writer, |_| {})
}

fn relay_observed(
    body: &mut impl BufRead,
    writer: &mut impl Write,
    mut observe: impl FnMut(&[u8]),
) -> io::Result<()> {
    loop {
        // The previous chunk is flushed before waiting for the next upstream read.
        let bytes = body.fill_buf()?;
        if bytes.is_empty() {
            break;
        }
        let n = bytes.len();
        write!(writer, "{n:x}\r\n")?;
        writer.write_all(bytes)?;
        writer.write_all(b"\r\n")?;
        writer.flush()?;
        observe(bytes);
        body.consume(n);
    }
    // An upstream truncation leaves this terminator unsent, exposing the error
    // to downstream clients rather than pretending the response completed.
    writer.write_all(b"0\r\n\r\n")?;
    // The downstream socket is unbuffered. Once the terminator is written,
    // the caller may close; a checked flush here would turn a completed
    // response into cancellation and discard its reusable upstream socket.
    Ok(())
}

#[cfg(test)]
#[path = "gateway/cluster_tests.rs"]
mod cluster_tests;

#[cfg(test)]
#[path = "gateway_conformance.rs"]
mod conformance;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn stream_head_forwards_the_warm_cache_estimate() {
        let fields = ["x-vade-prompt-tokens: 8000", "x-vade-cached-tokens: 7812",
            "x-vade-estimated-first-token-ms: 812", "x-vade-estimated-cold-ms: 3770"];
        let head = transport::Head { first: "HTTP/1.1 200 OK".into(), fields: fields.iter()
            .map(|f| { let (k, v) = f.split_once(": ").unwrap(); (k.into(), v.into()) }).collect() };
        let headers = response_head(200, &head, "POST", "/v1/chat/completions", None, "");
        assert!(fields.iter().all(|f| headers.contains(&format!("{f}\r\n"))), "{headers}");
    }

    #[test]
    fn refuses_public_bind_and_https_upstream() {
        assert!(bind_loopback("0.0.0.0:0").is_err());
        assert!(bind_loopback("[::]:0").is_err());
        assert!(bind_loopback("127.0.0.1:0").is_ok());
        assert_eq!(
            run("127.0.0.1:0", "https://127.0.0.1:1")
                .unwrap_err()
                .kind(),
            io::ErrorKind::Unsupported
        );
    }

    #[test]
    fn exact_request_and_live_stream_relay_with_mock_upstream() {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint::parse(&format!(
            "http://{}/v1/chat/completions?configured=1",
            upstream.local_addr().unwrap()
        ))
        .unwrap();
        let body = br#"{ "model":"mock-model", "messages":[{"role":"user","content":" exact\n"}], "temperature":0.125, "top_p":0.9, "seed":42, "tools":[{"type":"function","function":{"name":"demo"}}], "tool_choice":"auto", "stream":true, "vendor_field":{"x":1} }"#;
        let expected = body.to_vec();
        let (release_tx, release_rx) = mpsc::channel();
        let upstream_worker = thread::spawn(move || {
            let (socket, _) = upstream.accept().unwrap();
            let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
            let head = transport::read_head(&mut reader).unwrap();
            assert_eq!(
                head.first,
                "POST /v1/chat/completions?configured=1&client=2 HTTP/1.1"
            );
            assert_eq!(head.get("x-test"), Some("preserved"));
            assert!(head.get("x-hop").is_none());
            // The gateway token is consumed by the gateway, never forwarded.
            assert!(head.get("authorization").is_none());
            let mut request = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
            let mut bytes = Vec::new();
            request.read_to_end(&mut bytes).unwrap();
            assert_eq!(bytes, expected);
            let socket = request.reader.get_mut();
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nX-Upstream: yes\r\nAccess-Control-Allow-Origin: *\r\n\r\n").unwrap();
            let first = b"data: {\"choices\":[]}\n\n";
            write!(socket, "{:x}\r\n", first.len()).unwrap();
            socket.write_all(first).unwrap();
            socket.write_all(b"\r\n").unwrap();
            socket.flush().unwrap();
            // The client must receive this chunk before the upstream finishes.
            release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
            socket
                .write_all(b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")
                .unwrap();
        });
        let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = gateway.local_addr().unwrap();
        // Even a same-origin page needs the explicit allow-list entry.
        let mut access = Access::default();
        access.allow_origin(&format!("http://localhost:{}", address.port())).unwrap();
        access.require_token("relay-token-0123456").unwrap();
        let gateway_worker = thread::spawn(move || {
            let (socket, _) = gateway.accept().unwrap();
            handle_target(socket, &Upstream::Single(endpoint), None, &ConnectionPool::default(), &access).unwrap();
        });
        let socket = TcpStream::connect(address).unwrap();
        let mut socket = DeadlineStream::new(socket, None).unwrap();
        write!(socket, "POST /v1/chat/completions?client=2 HTTP/1.1\r\nHost: localhost:{}\r\nOrigin: http://localhost:{}\r\nAuthorization: Bearer relay-token-0123456\r\nContent-Type: application/json\r\nContent-Length: {}\r\nX-Test: preserved\r\nConnection: X-Hop\r\nX-Hop: stripped\r\n\r\n", address.port(), address.port(), body.len()).unwrap();
        socket.write_all(body).unwrap();
        let mut reader = BufReader::new(socket);
        let head = transport::read_head(&mut reader).unwrap();
        assert!(head.first.starts_with("HTTP/1.1 200"));
        assert_eq!(head.get("x-upstream"), Some("yes"));
        let allowed: Vec<_> = head.fields.iter().filter(|(k, _)| k == "access-control-allow-origin").map(|(_, v)| v.clone()).collect();
        assert_eq!(allowed, [format!("http://localhost:{}", address.port())]);
        assert_eq!(head.get("vary"), Some("Origin"));
        let mut response = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
        let first = b"data: {\"choices\":[]}\n\n";
        let mut bytes = vec![0; first.len()];
        response.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes, first);
        release_tx.send(()).unwrap();
        let mut tail = Vec::new();
        response.read_to_end(&mut tail).unwrap();
        assert_eq!(tail, b"data: [DONE]\n\n");
        gateway_worker.join().unwrap();
        upstream_worker.join().unwrap();
    }

    #[test]
    fn token_and_preflight_are_answered_locally() {
        let mut access = Access::default();
        access.allow_origin("https://app.example").unwrap();
        access.require_token("t0k-0123456789abcdef").unwrap();
        for (request, status) in [
            ("POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}", "401"),
            ("POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer nope\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}", "401"),
            ("POST /v1/responses HTTP/1.1\r\nHost: evil.example\r\nAuthorization: Bearer t0k-0123456789abcdef\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}", "421"),
            ("POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nOrigin: https://evil.example\r\nAuthorization: Bearer t0k-0123456789abcdef\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}", "403"),
            ("OPTIONS /v1/responses HTTP/1.1\r\nHost: localhost\r\nOrigin: https://app.example\r\nAccess-Control-Request-Method: POST\r\n\r\n", "204"),
            // Passing every check, the closed upstream is the only failure.
            ("POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nOrigin: https://app.example\r\nAuthorization: Bearer t0k-0123456789abcdef\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}", "502"),
        ] {
            let endpoint = Endpoint::parse("http://127.0.0.1:1").unwrap();
            let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = gateway.local_addr().unwrap();
            let access = access.clone();
            let worker = thread::spawn(move || {
                let (socket, _) = gateway.accept().unwrap();
                handle_target(socket, &Upstream::Single(endpoint), None, &ConnectionPool::default(), &access).unwrap();
            });
            let mut socket = TcpStream::connect(address).unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            socket.write_all(request.as_bytes()).unwrap();
            let mut response = String::new();
            socket.read_to_string(&mut response).unwrap();
            assert!(response.starts_with(&format!("HTTP/1.1 {status} ")), "{request:?} -> {response}");
            if status == "204" {
                assert!(response.contains("X-Vade-Gateway: v1\r\nAccess-Control-Allow-Origin: https://app.example\r\n"));
            }
            if status == "401" && request.contains("/v1/messages") {
                assert!(response.contains("authentication_error"));
            }
            worker.join().unwrap();
        }
    }

    /// One mock upstream reply: `declared` body bytes, of which `body` is sent
    /// after `pause`, with `head` as extra header lines. Status 0 reads the
    /// request, then closes without a reply.
    struct MockReply { status: u16, head: &'static str, body: &'static [u8], declared: usize, pause: Duration }
    fn reply(status: u16, body: &'static [u8]) -> MockReply { MockReply { status, head: "", body, declared: body.len(), pause: Duration::ZERO } }

    /// Send one chat request through a lone-upstream relay; return the client's
    /// status line, Retry-After, reply body, every request body the upstream
    /// read, and how long the client waited for the status line.
    fn lone_upstream_exchange(replies: Vec<MockReply>) -> (String, Option<String>, Vec<u8>, Vec<Vec<u8>>, Duration) {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let upstream_address = upstream.local_addr().unwrap();
        let endpoint = Endpoint::parse(&format!("http://{upstream_address}/v1")).unwrap();
        let upstream_worker = thread::spawn(move || {
            let mut bodies = Vec::new();
            for reply in replies {
                let (socket, _) = upstream.accept().unwrap();
                let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
                // The test's own empty connection: the relay sent nothing more.
                let Ok(head) = transport::read_head(&mut reader) else { break };
                let mut body = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
                let mut bytes = Vec::new(); body.read_to_end(&mut bytes).unwrap();
                bodies.push(bytes);
                if reply.status == 0 { continue; }
                let socket = body.reader.get_mut();
                write!(socket, "HTTP/1.1 {} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}\r\n", reply.status, reply.declared, reply.head).unwrap();
                socket.flush().unwrap();
                thread::sleep(reply.pause);
                let _ = socket.write_all(reply.body);
            }
            bodies
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let relay = thread::spawn(move || { let (socket, _) = listener.accept().unwrap(); handle(socket, &endpoint).unwrap(); });
        let request = br#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}"#;
        let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
        write!(socket, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", request.len()).unwrap();
        socket.write_all(request).unwrap();
        let sent = Instant::now();
        let mut reader = BufReader::new(socket);
        let head = transport::read_head(&mut reader).unwrap();
        let waited = sent.elapsed();
        let mut body = Vec::new();
        BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap().read_to_end(&mut body).unwrap();
        relay.join().unwrap();
        // Release a mock still waiting for a request the relay rightly never sent.
        drop(TcpStream::connect(upstream_address));
        let bodies = upstream_worker.join().unwrap();
        assert!(bodies.iter().all(|body| body == &bodies[0]), "every attempt sends the same bytes");
        (head.first.clone(), head.get("retry-after").map(str::to_owned), body, bodies, waited)
    }

    #[test]
    fn lone_upstream_admission_refusals_are_waited_out_and_refusals_say_when_to_retry() {
        // The engine's full-capacity reply, as sent between two chat rounds.
        let busy = br#"{"error":{"type":"overloaded","message":"connection capacity reached"}}"#;
        for (replies, status, retry_after, sent) in [
            (vec![reply(503, busy), reply(503, busy), reply(200, br#"{"ok":true}"#)], 200, None, 3),
            (vec![reply(429, br#"{"error":{"type":"rate_limit"}}"#)], 429, Some("1"), 1),
            // The upstream then stops listening: its refusal still reaches the client.
            (vec![reply(503, busy)], 503, Some("1"), 1),
        ] {
            let (first, header, body, bodies, _) = lone_upstream_exchange(replies);
            assert!(first.starts_with(&format!("HTTP/1.1 {status} ")), "{first}");
            assert_eq!(header.as_deref(), retry_after);
            assert!(!body.is_empty());
            assert_eq!(bodies.len(), sent);
        }
    }

    #[test]
    fn lone_upstream_503s_that_may_follow_work_are_forwarded_once() {
        for terminal in [
            br#"{"error":{"type":"capacity_exhausted","message":"request capacity exhausted"}}"#.as_slice(),
            br#"{"error":{"type":"model_swapping","message":"request cancelled for model swap"}}"#,
            br#"{"error":{"type":"server_shutdown","message":"server is shutting down; request cancelled"}}"#,
            br#"{"error":{"type":"overloaded","message":"other"}}"#,
            b"not json",
        ] {
            let (first, header, body, bodies, _) = lone_upstream_exchange(vec![reply(503, terminal), reply(200, b"{}")]);
            assert!(first.starts_with("HTTP/1.1 503 "), "{first}");
            assert_eq!((header.as_deref(), body.as_slice(), bodies.len()), (Some("1"), terminal, 1));
        }
    }

    #[test]
    fn a_retry_that_may_have_run_is_not_answered_with_the_earlier_refusal() {
        let busy = br#"{"error":{"type":"overloaded","message":"connection capacity reached"}}"#;
        // The retry's connection takes the request, then closes without a reply.
        let (first, _, body, bodies, _) = lone_upstream_exchange(vec![reply(503, busy), reply(0, b"")]);
        assert!(first.starts_with("HTTP/1.1 502 "), "{first}");
        assert!(String::from_utf8_lossy(&body).contains("outcome may be unknown"));
        assert_eq!(bodies.len(), 2);
    }

    /// The downstream once its client has read a whole reply and closed: the
    /// written bytes were delivered, and a checked flush reports the close.
    struct ClosedOnLastByte(Vec<u8>);
    impl Write for ClosedOnLastByte {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::new(io::ErrorKind::ConnectionAborted, "downstream closed")) }
    }

    #[test]
    fn a_client_closing_on_the_last_byte_got_a_whole_reply_not_a_failure() {
        let mut client = ClosedOnLastByte(Vec::new());
        write_error(&mut client, false, 502, "HTTP upstream request failed").unwrap();
        assert!(client.0.starts_with(b"HTTP/1.1 502 ") && client.0.ends_with(b"}"));
        let head = transport::Head { first: "HTTP/1.1 503 Mock".into(), fields: Vec::new() };
        let mut client = ClosedOnLastByte(Vec::new());
        relay_whole(&mut client, false, (503, &head, b"busy"), "POST", "/v1/chat/completions", None).unwrap();
        assert!(client.0.starts_with(b"HTTP/1.1 503 ") && client.0.ends_with(b"\r\n\r\nbusy"));
    }

    #[test]
    fn holding_a_refusal_stays_within_the_admission_budget() {
        let busy = br#"{"error":{"type":"overloaded","message":"connection capacity reached"}}"#;
        // A refusal that declares a body and sends it only well after the budget.
        let stalled = MockReply { status: 503, head: "", body: b"", declared: busy.len(), pause: ADMISSION_PATIENCE + Duration::from_secs(2) };
        let (first, _, body, bodies, waited) = lone_upstream_exchange(vec![stalled]);
        assert!(first.starts_with("HTTP/1.1 502 "), "{first}");
        assert!(waited < ADMISSION_PATIENCE + Duration::from_secs(1), "{waited:?}");
        assert!(String::from_utf8_lossy(&body).contains("refusal was incomplete"));
        assert_eq!(bodies.len(), 1);
        // A refusal whose pause would end past the budget is passed on, not replayed.
        let late = MockReply { status: 503, head: "Retry-After: 1\r\n", body: busy, declared: busy.len(), pause: ADMISSION_PATIENCE / 2 + Duration::from_millis(200) };
        let (first, _, body, bodies, _) = lone_upstream_exchange(vec![late, reply(200, b"{}")]);
        assert!(first.starts_with("HTTP/1.1 503 "), "{first}");
        assert_eq!((body.as_slice(), bodies.len()), (busy.as_slice(), 1));
    }

    #[test]
    fn health_reaches_the_lone_engine_at_its_server_root() {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = Endpoint::parse(&format!("http://{}/v1", upstream.local_addr().unwrap())).unwrap();
        let upstream_worker = thread::spawn(move || {
            let (mut socket, _) = upstream.accept().unwrap();
            let head = transport::read_head(&mut BufReader::new(socket.try_clone().unwrap())).unwrap();
            let body = br#"{"status":"ready","model":"mock"}"#;
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len()).unwrap();
            socket.write_all(body).unwrap();
            head.first
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let relay = thread::spawn(move || { let (socket, _) = listener.accept().unwrap(); handle(socket, &endpoint).unwrap(); });
        let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
        socket.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut reader = BufReader::new(socket);
        let head = transport::read_head(&mut reader).unwrap();
        assert!(head.first.starts_with("HTTP/1.1 200 "), "{}", head.first);
        let mut reply = Vec::new();
        BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap().read_to_end(&mut reply).unwrap();
        assert_eq!(reply, br#"{"status":"ready","model":"mock"}"#);
        relay.join().unwrap();
        assert_eq!(upstream_worker.join().unwrap(), "GET /health HTTP/1.1");
    }

    #[test]
    fn malformed_request_is_rejected_without_upstream_connection() {
        let endpoint = Endpoint::parse("http://127.0.0.1:1").unwrap();
        let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = gateway.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (socket, _) = gateway.accept().unwrap();
            handle(socket, &endpoint).unwrap();
        });
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        socket.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 400"));
        worker.join().unwrap();
    }

    #[test]
    fn responses_route_preserves_raw_state_tools_cache_identity_and_errors() {
        let request_body = br#"{ "model":"mock-model", "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":" exact\n"}]},{"type":"function_call_output","call_id":"mock-call","output":"tool state"}], "tools":[{"type":"function","name":"demo","parameters":{"type":"object"}}], "reasoning":{"effort":"low","summary":"auto"}, "previous_response_id":"mock-response", "store":false, "temperature":0.125, "seed":42, "top_p":0.9, "max_output_tokens":32, "prompt_cache_key":"independent-cache-identity", "stream":true }"#;
        for (status, content_type, response_body) in [
            (200, "text/event-stream", "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hé🦀\"}\n\nevent: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"mock failure\"}}}\n\n".as_bytes()),
            (429, "application/json", br#"{ "error": { "type":"rate_limit", "message":"mock provider refusal" } }"#.as_slice()),
        ] {
            let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = Endpoint::parse(&format!("http://{}/v1?configured=1", upstream.local_addr().unwrap())).unwrap();
            let expected_response = response_body.to_vec();
            let upstream_worker = thread::spawn(move || {
                let (socket, _) = upstream.accept().unwrap();
                let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
                let head = transport::read_head(&mut reader).unwrap();
                assert_eq!(head.first, "POST /v1/responses?configured=1&client=2 HTTP/1.1");
                let mut body = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
                let mut bytes = Vec::new(); body.read_to_end(&mut bytes).unwrap();
                assert_eq!(bytes, request_body);
                let socket = body.reader.get_mut();
                write!(socket, "HTTP/1.1 {status} Mock\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nRetry-After: 7\r\n\r\n").unwrap();
                for chunk in response_body.chunks(3) { write!(socket, "{:x}\r\n", chunk.len()).unwrap(); socket.write_all(chunk).unwrap(); socket.write_all(b"\r\n").unwrap(); }
                socket.write_all(b"0\r\n\r\n").unwrap();
            });
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let relay = thread::spawn(move || { let (socket, _) = listener.accept().unwrap(); handle(socket, &endpoint).unwrap(); });
            let mut socket = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap();
            socket.write_all(b"POST /v1/responses?client=2 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
            for chunk in request_body.chunks(11) { write!(socket, "{:x}\r\n", chunk.len()).unwrap(); socket.write_all(chunk).unwrap(); socket.write_all(b"\r\n").unwrap(); }
            socket.write_all(b"0\r\n\r\n").unwrap();
            let mut reader = BufReader::new(socket);
            let head = transport::read_head(&mut reader).unwrap();
            assert!(head.first.starts_with(&format!("HTTP/1.1 {status} ")));
            assert_eq!(head.get("retry-after"), Some("7"));
            let mut body = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
            let mut bytes = Vec::new(); body.read_to_end(&mut bytes).unwrap();
            assert_eq!(bytes, expected_response);
            relay.join().unwrap(); upstream_worker.join().unwrap();
        }
    }

    #[test]
    fn cancellation_closes_pending_mock_upstream_without_success_response() {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint =
            Endpoint::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap();
        let (ready, waiting) = mpsc::channel();
        let upstream_worker = thread::spawn(move || {
            let (socket, _) = upstream.accept().unwrap();
            let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
            let head = transport::read_head(&mut reader).unwrap();
            let mut body = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
            body.read_to_end(&mut Vec::new()).unwrap();
            ready.send(()).unwrap();
            assert_eq!(body.reader.read(&mut [0; 1]).unwrap(), 0);
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let relay = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            assert_eq!(
                handle_cancellable(socket, &endpoint, Some(&flag))
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::ConnectionAborted
            );
        });
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        socket
            .write_all(
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
            )
            .unwrap();
        waiting.recv_timeout(Duration::from_secs(3)).unwrap();
        cancel.store(true, Ordering::Relaxed);
        let mut response = Vec::new();
        socket.read_to_end(&mut response).unwrap();
        assert!(response.is_empty());
        relay.join().unwrap();
        upstream_worker.join().unwrap();
    }

    #[test]
    fn responses_multiturn_tool_prefix_is_opaque_and_upstream_is_reused() {
        let records = [
            r#"{"role":"user","content":[{"type":"input_text","text":" exact\n🦀"}]}"#,
            r#"{"type":"function_call","id":"item-1","call_id":"call-1","name":"lookup","arguments":"{}"}"#,
            r#"{"type":"function_call_output","call_id":"call-1","output":" exact result\n"}"#,
            r#"{"role":"assistant","content":[{"type":"output_text","text":"answer"}]}"#,
            r#"{"role":"user","content":[{"type":"input_text","text":"next"}]}"#,
        ];
        let fixed = r#"{ "temperature":0,"seed":42,"top_p":1, "model":"mock", "instructions":" exact system\n", "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}], "reasoning":{"effort":"low"}, "prompt_cache_key":"mock-stable", "stream":true, "input":["#;
        let bodies: Vec<Vec<u8>> = [1, 3, 5]
            .into_iter()
            .map(|len| format!("{fixed}{}]}}", records[..len].join(",")).into_bytes())
            .collect();
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint =
            Endpoint::parse(&format!("http://{}/v1", upstream.local_addr().unwrap())).unwrap();
        let expected = bodies.clone();
        let server = thread::spawn(move || {
            let (socket, _) = upstream.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut reader = BufReader::new(socket);
            let mut previous = Vec::new();
            for expected in expected {
                let head = transport::read_head(&mut reader).unwrap();
                assert_eq!(head.first, "POST /v1/responses HTTP/1.1");
                assert_eq!(head.get("connection"), Some("keep-alive"));
                let mut body =
                    BodyReader::new(&mut reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
                let mut bytes = Vec::new();
                body.read_to_end(&mut bytes).unwrap();
                assert_eq!(bytes, expected); // exact opaque bytes, including whitespace
                assert!(bytes.starts_with(&previous));
                previous = bytes[..bytes.len() - 2].to_vec();
                let payload = b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\"}\n\n";
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n", payload.len()).unwrap();
                reader.get_mut().write_all(payload).unwrap();
            }
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let gateway = thread::spawn(move || {
            let pool = ConnectionPool::default();
            for _ in 0..3 {
                let (socket, _) = listener.accept().unwrap();
                handle_pooled(socket, &endpoint, None, &pool).unwrap();
            }
        });
        for body in bodies {
            let mut socket = TcpStream::connect(address).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            write!(socket, "POST /responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len()).unwrap();
            socket.write_all(&body).unwrap();
            let mut reader = BufReader::new(socket);
            let head = transport::read_head(&mut reader).unwrap();
            assert!(head.first.contains("200"));
            let mut response = BodyReader::new(reader, &head, false, transport::MAX_BODY).unwrap();
            let mut bytes = Vec::new();
            response.read_to_end(&mut bytes).unwrap();
            assert!(bytes.ends_with(b"data: {\"type\":\"response.completed\"}\n\n"));
        }
        gateway.join().unwrap();
        server.join().unwrap();
    }
}
