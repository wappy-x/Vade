//! Standalone opaque HTTP relay. Observers and optional policies never own replies.
use crate::transport::{self, BodyReader, DeadlineStream, Endpoint, Head};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

#[path = "invisible_stats.rs"]
mod stats;
use stats::{Metrics, StatsLog};
#[path = "invisible_record.rs"]
mod record;
use record::{Recorder, Recording};
#[path = "invisible_policy.rs"]
mod policy;
use policy::{Flags, Runtime};
use super::access::{self, Access, Refusal};
#[path = "invisible_route.rs"]
mod route;
use route::{Lease, Router};

#[derive(Clone)]
struct Origin {
    endpoint: Endpoint,
    prefix: String,
}
impl Origin {
    fn parse(url: &str) -> io::Result<Self> {
        let endpoint = Endpoint::parse(url)?;
        let prefix = url.strip_prefix("http://").unwrap()[endpoint.authority.len()..]
            .trim_end_matches('/')
            .to_owned();
        if prefix.contains('?') {
            return Err(transport::invalid("upstream base URL cannot have a query"));
        }
        Ok(Self { endpoint, prefix })
    }
    fn target(&self, path: &str) -> String {
        // The public completions alias selects the same engine route. Other
        // paths stay opaque, including custom names ending in /completions.
        let alias = path.split_once('?').map_or(path, |(route, _)| route) == "/completions";
        let canonical;
        let path = if alias {
            canonical = format!("/v1{path}");
            canonical.as_str()
        } else { path };
        if self.prefix.is_empty() || self.prefix == "/v1" {
            return path.to_owned();
        }
        let suffix = if self.prefix.ends_with("/v1") {
            path.strip_prefix("/v1/")
                .map(|p| format!("/{p}"))
                .unwrap_or_else(|| path.to_owned())
        } else {
            path.to_owned()
        };
        format!("{}{suffix}", self.prefix)
    }
    fn connect(&self) -> io::Result<DeadlineStream<'static>> {
        let started = Instant::now();
        for address in (self.endpoint.host.as_str(), self.endpoint.port)
            .to_socket_addrs()?
            .take(16)
        {
            let left = Duration::from_secs(5).saturating_sub(started.elapsed());
            if left.is_zero() {
                break;
            }
            if let Ok(stream) = TcpStream::connect_timeout(&address, left) {
                return DeadlineStream::new(stream, None);
            }
        }
        Err(io::Error::new(
            io::ErrorKind::ConnectionRefused,
            "upstream connection failed",
        ))
    }
}

/// Run the relay after credential options were configured, reporting failures.
pub fn exit(args: impl Iterator<Item = String>) -> std::process::ExitCode {
    match cli(args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) if crate::auth::already_reported(&error) => std::process::ExitCode::FAILURE,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade-gateway: {}.{}",
                crate::terminal::sanitize(&error.to_string()),
                if matches!(error.kind(), io::ErrorKind::InvalidInput | io::ErrorKind::InvalidData) {
                    " Check vade-gateway --help for options and examples."
                } else {
                    crate::diagnostics::error_hint(&error)
                }));
            std::process::ExitCode::FAILURE
        }
    }
}

pub fn cli(args: impl Iterator<Item = String>) -> io::Result<()> {
    let args: Vec<_> = args.collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return print_help(args.first().is_some_and(|arg| arg == "replay"));
    }
    let mut args = args.into_iter().peekable();
    if args.peek().is_some_and(|v| v == "replay") {
        args.next();
        return record::replay(args);
    }
    let mut bind = super::DEFAULT_BIND.to_owned();
    let mut upstreams = Vec::new();
    let mut flags = Flags::default();
    let mut stats_path = None;
    let mut record_dir = None;
    let mut access = Access::default();
    let refused = |message: &str| io::Error::new(io::ErrorKind::InvalidInput, message.to_owned());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--upstream" => upstreams.push(argument(&mut args, "--upstream")?),
            "--stable-prefix" => flags.stable = true,
            "--affinity" => flags.affinity = true,
            "--no-sticky" => flags.no_sticky = true,
            "--clamp-limits" => flags.clamp = true,
            "--greedy-when-unset" => flags.greedy = true,
            "--bind" => bind = argument(&mut args, "--bind")?,
            "--record" => record_dir = Some(argument(&mut args, "--record")?),
            "--stats" => stats_path = Some(argument(&mut args, "--stats")?),
            "--allow-origin" => access.allow_origin(&argument(&mut args, "--allow-origin")?).map_err(refused)?,
            "--allow-host" => access.allow_host(&argument(&mut args, "--allow-host")?).map_err(refused)?,
            "--token" => access.require_token(&argument(&mut args, "--token")?).map_err(refused)?,
            url if url.starts_with("http") => upstreams.push(url.to_owned()),
            _ => return Err(io::Error::new(io::ErrorKind::InvalidInput,
                crate::diagnostics::option_name(&arg).map_or_else(
                    || "unknown gateway argument".to_owned(),
                    |name| format!("unknown gateway option {name}")))),
        }
    }
    if upstreams.is_empty()
        && let Ok(url) = std::env::var("VADE_GATEWAY_UPSTREAM")
    {
        upstreams.push(url);
    }
    access.token_from_env().map_err(refused)?;
    crate::auth::preload()?;
    // Validate startup before creating user output. In particular, a typo,
    // stale credential, unusable listener or invalid upstream leaves no files.
    let listener = super::bind_loopback(&bind)?;
    let runtime = Arc::new(Runtime::new(&upstreams, flags)?);
    let recorder = record_dir.as_deref().map(Recorder::open).transpose()?.map(Arc::new);
    let stats = stats_path.as_deref().map(StatsLog::open).transpose()?.map(Arc::new);
    crate::diagnostics::stderr_line(format_args!(
        "Vade gateway: listening on http://{}/v1 ({} upstream{}). Ctrl-C stops the gateway.",
        listener.local_addr()?, upstreams.len(), if upstreams.len() == 1 { "" } else { "s" }
    ));
    serve(listener, runtime, stats, recorder, Arc::new(access))
}

fn print_help(replay: bool) -> io::Result<()> {
    let help = if replay {
        "Vade gateway recording replay\n\nUsage: vade-gateway replay DIR --upstream HTTP_BASE_URL [OPTIONS]\n\n  DIR               Directory containing complete recorded requests\n  --upstream URL    HTTP engine origin or API base (or VADE_GATEWAY_UPSTREAM)\n  --concurrency N   Concurrent sessions, 1..64 (default 1)\n  --stats FILE      Append per-request statistics to this file\n  -h, --help        Show help without opening files or contacting an engine\n\nRequests within each session retain their recorded order. Replaying sends\nthe recorded request bodies to the selected engine; it does not execute tools.\nOnly complete recordings are replayed. Failed sessions cause a nonzero exit.\n".to_owned()
    } else {
        format!(
            "vade-gateway --upstream HTTP_BASE_URL [--bind LOOPBACK:PORT] [--stats FILE] [--record DIR] [--stable-prefix] [--affinity] [--no-sticky] [--clamp-limits] [--greedy-when-unset] [--allow-origin ORIGIN]... [--allow-host HOST]... [--token TOKEN]\nvade-gateway replay DIR --upstream HTTP_BASE_URL [--concurrency N] [--stats FILE]\nEnvironment: VADE_GATEWAY_UPSTREAM, VADE_GATEWAY_TOKEN\n{}Generation defaults: {}, only when none of these sampling fields is present (including null): {}.\nAny one suppresses all defaults; sampling forwards the body byte for byte and engine defaults apply.\nRouting, affinity, input limits and policy clamps use original client bytes, before defaults.\nGET /v1/capabilities advertises X-Vade-Sampling-Defaults, its condition in X-Vade-Sampling-Defaults-When, and recognized fields in X-Vade-Sampling-Fields.\nAll HTTP paths pass through; TLS and upgrades are unsupported.\nRepeat --upstream for identical replicas: each request goes to the upstream with the fewest\nin-flight gateway requests, ties rotate. Turns of one conversation (same first {} bytes of\nmessages through the last user turn) prefer their previous upstream while it is among\nthe least busy; --no-sticky disables that.\nA request is retried once per other upstream only before any reply byte reaches the client.\nWith several upstreams GET /health is answered here: per-upstream status, error if model ids differ.\nWith several upstreams POST /v1/cancel and /v1/kv/release go to every upstream; the reply is the first that reports success.\n",
            super::BROWSER_GUARD_HELP, super::sampling::DEFAULTS_JSON, super::sampling::SAMPLING_FIELDS, route::STICKY_PREFIX_BYTES
        )
    };
    crate::diagnostics::write_stdout(&(help + "Credentials: --api-key-file PATH or VADE_API_KEY_FILE; the flag overrides the environment.\nA configured key replaces incoming Authorization and X-Api-Key headers only on upstream requests.\nWithout a configured key, client credential headers pass through unchanged, except Authorization under --token.\n"))
}

fn argument(args: &mut impl Iterator<Item = String>, option: &str) -> io::Result<String> {
    args.next().filter(|value| !value.is_empty())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput,
            format!("{option} needs a nonempty value")))
}

fn serve(
    listener: TcpListener,
    runtime: Arc<Runtime>,
    stats: Option<Arc<StatsLog>>,
    recorder: Option<Arc<Recorder>>,
    access: Arc<Access>,
) -> io::Result<()> {
    let active = Arc::new(AtomicUsize::new(0));
    for socket in listener.incoming() {
        let socket = socket?;
        if !socket.peer_addr()?.ip().is_loopback() {
            continue;
        }
        if active.load(Ordering::Relaxed) >= super::MAX_CONNECTIONS {
            let _ = busy(socket, stats.as_deref());
            continue;
        }
        active.fetch_add(1, Ordering::Relaxed);
        let guard = super::ActiveGuard(active.clone());
        let runtime = runtime.clone();
        let stats = stats.clone();
        let recorder = recorder.clone();
        let access = access.clone();
        thread::Builder::new()
            .name("invisible-relay".into())
            .spawn(move || {
                let _guard = guard;
                let _ = handle_guarded(socket, &runtime, stats.as_deref(), recorder.as_deref(), &access);
            })?;
    }
    Ok(())
}

fn busy(socket: TcpStream, log: Option<&StatsLog>) -> io::Result<()> {
    let mut metrics = Metrics::new(log.is_some());
    let mut reader = BufReader::new(DeadlineStream::new(socket, None)?);
    reader
        .get_mut()
        .bound_deadline(Instant::now() + Duration::from_millis(100));
    if let Ok(head) = WireHead::read(&mut reader) {
        metrics.path = head.parsed.first.split(' ').nth(1).unwrap_or("").to_owned();
    }
    metrics.response(
        &Head {
            first: "HTTP/1.1 503 Gateway Error".into(),
            fields: Vec::new(),
        },
        503,
    );
    let result = local_error(reader.get_mut(), 503, "gateway busy");
    metrics.finish(result.is_ok());
    if let Some(log) = log {
        log.write(&metrics.json())?;
    }
    result
}

struct WireHead {
    parsed: Head,
    lines: Vec<Vec<u8>>,
}
impl WireHead {
    fn read(reader: &mut impl BufRead) -> io::Result<Self> {
        let mut wire = Vec::new();
        let mut lines = Vec::new();
        loop {
            let line = transport::line(reader, transport::MAX_HEADERS.saturating_sub(wire.len()))?;
            wire.extend_from_slice(&line);
            wire.extend_from_slice(b"\r\n");
            if wire.len() > transport::MAX_HEADERS {
                return Err(transport::invalid("headers exceed limit"));
            }
            if line.is_empty() {
                break;
            }
            lines.push(line);
        }
        let parsed = transport::read_head(&mut io::Cursor::new(&wire))?;
        Ok(Self { parsed, lines })
    }
    fn forward(&self, writer: &mut impl Write, first: &str, owned: &str) -> io::Result<()> {
        self.forward_headers(writer, first, owned, None)
    }
    fn forward_upstream(&self, writer: &mut impl Write, first: &str, owned: &str) -> io::Result<()> {
        let key = crate::auth::ApiKey::configured()?;
        let authorization = key.as_ref().map(crate::auth::ApiKey::bearer_header);
        self.forward_headers(writer, first, owned, authorization.as_deref())
    }
    fn forward_headers(&self, writer: &mut impl Write, first: &str, owned: &str, authorization: Option<&str>) -> io::Result<()> {
        write!(writer, "{first}\r\n")?;
        for line in self.lines.iter().skip(1) {
            let name = std::str::from_utf8(line)
                .unwrap()
                .split_once(':')
                .unwrap()
                .0
                .to_ascii_lowercase();
            if !transport::hop_header(&name, &self.parsed)
                && !(authorization.is_some() && matches!(name.as_str(), "authorization" | "x-api-key")) {
                writer.write_all(line)?;
                writer.write_all(b"\r\n")?;
            }
        }
        writer.write_all(owned.as_bytes())?;
        if let Some(authorization) = authorization {
            writer.write_all(authorization.as_bytes())?;
        }
        writer.write_all(b"\r\n")?;
        writer.flush()
    }
    fn body_head(&self) -> Head {
        // Content codings are opaque payload bytes; framing remains validated.
        Head {
            first: self.parsed.first.clone(),
            fields: self
                .parsed
                .fields
                .iter()
                .filter(|(k, _)| k != "content-encoding")
                .cloned()
                .collect(),
        }
    }
}

#[cfg(test)]
fn handle(
    socket: TcpStream,
    runtime: &Runtime,
    log: Option<&StatsLog>,
    recorder: Option<&Recorder>,
) -> io::Result<()> {
    handle_guarded(socket, runtime, log, recorder, &Access::default())
}

fn handle_guarded(
    socket: TcpStream,
    runtime: &Runtime,
    log: Option<&StatsLog>,
    recorder: Option<&Recorder>,
    access: &Access,
) -> io::Result<()> {
    let mut metrics = Metrics::new(log.is_some() || recorder.is_some());
    let mut recording = recorder.and_then(|r| match r.start() {
        Ok(recording) => Some(recording),
        Err(_) => {
            crate::diagnostics::stderr_line(format_args!("vade-gateway: recorder start failed"));
            None
        }
    });
    let result = forward(socket, runtime, access, &mut metrics, &mut recording);
    metrics.finish(result.is_ok());
    let line = metrics.json();
    if let Some(log) = log
        && log.write(&line).is_err()
    {
        crate::diagnostics::stderr_line(format_args!("vade-gateway: stats write failed"));
    }
    if let Some(recording) = recording
        && recording.finish(&line).is_err()
    {
        crate::diagnostics::stderr_line(format_args!("vade-gateway: recorder finish failed"));
    }
    result
}

fn forward(
    socket: TcpStream,
    runtime: &Runtime,
    access: &Access,
    metrics: &mut Metrics,
    recording: &mut Option<Recording>,
) -> io::Result<()> {
    let local = socket.local_addr()?;
    let mut client = BufReader::new(DeadlineStream::new(socket, None)?);
    let mut request = WireHead::read(&mut client)?;
    let parts: Vec<_> = request.parsed.first.split(' ').collect();
    if parts.len() != 3
        || parts[2] != "HTTP/1.1"
        || !parts[1].starts_with('/')
        || parts[1].contains(['#', '\\'])
    {
        return local_error(client.get_mut(), 400, "invalid HTTP request target");
    }
    let cors = match access.check(&request.parsed, local) {
        Ok(origin) => access::cors(origin.as_deref()),
        Err(Refusal::Status(code, message)) => return local_error(client.get_mut(), code, message),
        Err(Refusal::Preflight(headers)) => return access::write_preflight(client.get_mut(), "", &headers),
    };
    if access.consumes_authorization() {
        // The gateway token stays here; --api-key-file supplies upstream credentials.
        request.parsed.fields.retain(|(key, _)| key != "authorization");
        let mut first = true;
        request.lines.retain(|line| {
            std::mem::take(&mut first)
                || !line.split(|&b| b == b':').next().is_some_and(|key| key.eq_ignore_ascii_case(b"authorization"))
        });
    }
    if request.parsed.get("upgrade").is_some() {
        return local_error(client.get_mut(), 400, "protocol upgrades unsupported");
    }
    let body_head = request.body_head();
    let messages = super::messages_route(&request.parsed.first);
    if let Err(error) = BodyReader::new(io::Cursor::new(&[]), &body_head, true, crate::json::MAX_INPUT_BYTES) {
        let (status, message) = super::request_body_error(&error);
        return super::write_error(client.get_mut(), messages, status, &message);
    }
    if request
        .parsed
        .get("expect")
        .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    {
        client
            .get_mut()
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }
    let (method, path) = (parts[0], parts[1]);
    metrics.path = path.to_owned();
    let mut input = BodyReader::new(
        client,
        &body_head,
        true,
        crate::json::MAX_INPUT_BYTES,
    )?;
    let mut body = Vec::new();
    if let Err(error) = input.read_to_end(&mut body) {
        let (status, message) = super::request_body_error(&error);
        return super::write_error(input.reader.get_mut(), messages, status, &message);
    }
    metrics.request(&body);
    if method == "GET" && path.split('?').next() == Some("/health") && runtime.origins().len() > 1 {
        // Replicas answer as one engine; nothing is forwarded or replayed.
        let (code, document) = route::health(runtime.origins());
        let writer = input.reader.get_mut();
        write!(writer, "HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{cors}\r\n{document}",
            if code == 200 { "OK" } else { "Gateway Error" }, document.len())?;
        return writer.flush();
    }
    record::observe(recording, |r| {
        r.request(method, path, &request.parsed, &body)
    });
    let prepared = runtime.prepare(method, path, &request.parsed, &body);
    metrics.policy = prepared.report;
    if prepared.body.as_ref() != body {
        record::observe(recording, |r| r.effective(&prepared.body));
    }
    let mut lease = prepared.lease;
    let (mut index, body) = (prepared.index, prepared.body);
    let mut client = input.reader;
    client.get_mut().gateway_response(method, path);
    client.get_mut().watch_downstream()?;
    if let Some(field) = broadcast_field(method, path).filter(|_| runtime.origins().len() > 1) {
        drop(lease);
        return fan_out(&mut client, runtime, Dispatch { origin: runtime.origin(0), request: &request, body: &body, cors: &cors, retry: false }, field, metrics, recording);
    }
    let mut failed = Vec::new();
    loop {
        // An untried upstream permits one more attempt while the client has seen nothing.
        failed.push(index);
        let alternate = || runtime.route(path, &request.parsed, &body, &failed);
        let origin = runtime.origin(index);
        let mut upstream = match origin.connect() {
            Ok(socket) => socket,
            Err(_) => {
                if let Some(next) = alternate() {
                    (index, lease) = (next.index, Some(next));
                    continue;
                }
                return local_error(
                    client.get_mut(),
                    502,
                    "upstream connection failed; request not replayed",
                );
            }
        };
        upstream.watch_peer(client.get_ref());
        let retry = failed.len() < runtime.origins().len();
        let mut sent = Sent { inner: client.get_mut(), bytes: 0 };
        let delivered = exchange(
            upstream,
            Dispatch {
                origin,
                request: &request,
                body: &body,
                cors: &cors,
                retry,
            },
            &mut sent,
            metrics,
            recording,
        );
        let untouched = sent.bytes == 0 && client.get_ref().check().is_ok();
        match delivered {
            Ok(false) => {}
            Err(error) if retry && untouched && error.kind() != io::ErrorKind::TimedOut => {}
            result => return result.map(drop),
        }
        drop(lease.take());
        match alternate() {
            Some(next) => (index, lease) = (next.index, Some(next)),
            None => return local_error(client.get_mut(), 502, "upstream failed before replying"),
        }
    }
}

/// Cancel and checkpoint-release keys belong to the replica that ran the request,
/// which routing does not record. With several upstreams these calls go to every
/// replica: keys are random capabilities, so an unknown key changes nothing there.
/// Returns the JSON field that reports success.
fn broadcast_field(method: &str, path: &str) -> Option<&'static str> {
    match (method, path.split('?').next().unwrap_or(path)) {
        ("POST", "/v1/cancel") => Some("cancelled"),
        ("POST", "/v1/kv/release") => Some("released"),
        _ => None,
    }
}

/// Send one request to every upstream in order, buffering each reply, and answer
/// once: the first 2xx reply whose `field` is true, otherwise the first reply.
fn fan_out(
    client: &mut BufReader<DeadlineStream<'_>>,
    runtime: &Runtime,
    dispatch: Dispatch<'_>,
    field: &str,
    metrics: &mut Metrics,
    recording: &mut Option<Recording>,
) -> io::Result<()> {
    let mut chosen: Option<(Vec<u8>, Metrics, bool)> = None;
    for origin in runtime.origins() {
        let Ok(mut upstream) = origin.connect() else { continue };
        upstream.watch_peer(client.get_ref());
        let (mut reply, mut attempt) = (Vec::new(), metrics.attempt());
        let dispatch = Dispatch { origin, ..dispatch };
        if exchange(upstream, dispatch, &mut reply, &mut attempt, &mut None).is_err() {
            continue;
        }
        let succeeded = buffered(&reply).is_some_and(|(code, body)| {
            (200..300).contains(&code)
                && std::str::from_utf8(&body).ok().and_then(|text| crate::json::view(text).ok())
                    .and_then(|value| value.get(field)).is_some_and(|value| value.raw() == "true")
        });
        if chosen.as_ref().is_none_or(|(_, _, done)| succeeded && !done) {
            chosen = Some((reply, attempt, succeeded));
        }
    }
    let Some((reply, attempt, _)) = chosen else {
        return local_error(client.get_mut(), 502, "no upstream answered; request not replayed");
    };
    *metrics = attempt;
    if let Some((_, body)) = buffered(&reply) {
        record::observe(recording, |r| r.payload(&body));
    }
    let writer = client.get_mut();
    writer.write_all(&reply)?;
    writer.flush()
}

/// Final status and decoded body of a reply that `exchange` wrote to a buffer.
fn buffered(reply: &[u8]) -> Option<(u16, Vec<u8>)> {
    let mut reader = io::Cursor::new(reply);
    let head = loop {
        let head = WireHead::read(&mut reader).ok()?;
        if status(&head.parsed).ok()? >= 200 { break head; }
    };
    let mut body = Vec::new();
    BodyReader::new(reader, &head.body_head(), false, transport::MAX_BODY).ok()?.read_to_end(&mut body).ok()?;
    Some((status(&head.parsed).ok()?, body))
}

/// Counts bytes delivered downstream; a retry is allowed only while this is zero.
struct Sent<'w, W> {
    inner: &'w mut W,
    bytes: u64,
}
impl<W: Write> Write for Sent<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(bytes)?;
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Clone, Copy)]
struct Dispatch<'a> {
    origin: &'a Origin,
    request: &'a WireHead,
    body: &'a [u8],
    /// CORS lines for an allow-listed browser Origin; empty otherwise.
    cors: &'a str,
    /// Another upstream may take this request if this one refuses before replying.
    retry: bool,
}

fn exchange(
    mut upstream: DeadlineStream<'_>,
    dispatch: Dispatch<'_>,
    writer: &mut impl Write,
    metrics: &mut Metrics,
    recording: &mut Option<Recording>,
) -> io::Result<bool> {
    let Dispatch {
        origin,
        request,
        body,
        cors,
        retry,
    } = dispatch;
    let parts: Vec<_> = request.parsed.first.split(' ').collect();
    let (method, path) = (parts[0], parts[1]);
    let first = format!("{method} {} HTTP/1.1", origin.target(path));
    request.forward_upstream(
        &mut upstream,
        &first,
        &format!(
            "Host: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
            origin.endpoint.authority,
            body.len()
        ),
    )?;
    upstream.write_all(body)?;
    upstream.flush()?;
    upstream.gateway_response(method, path);
    let mut upstream = BufReader::new(upstream);
    if !upstream.fill_buf()?.is_empty() {
        metrics.first_byte();
    }
    let mut response = loop {
        let response = WireHead::read(&mut upstream)?;
        let status = status(&response.parsed)?;
        if (100..200).contains(&status) && status != 101 {
            response.forward(writer, &response.parsed.first, "")?;
        } else {
            break response;
        }
    };
    let code = status(&response.parsed)?;
    if retry && code == 503 {
        // Not ready or overloaded: nothing has been relayed, so another replica may answer.
        return Ok(false);
    }
    metrics.response(&response.parsed, code);
    if code == 101 {
        return local_error(writer, 502, "upstream upgrade unsupported").map(|_| true);
    }
    let no_body = method == "HEAD" || matches!(code, 204 | 304);
    let chunked = response.parsed.first.starts_with("HTTP/1.1 ");
    let mut owned = super::sampling::capabilities_header(method, path);
    if !cors.is_empty() {
        // An allowed browser gets exactly one CORS answer: the gateway's.
        let mut first = true;
        response.lines.retain(|line| {
            std::mem::take(&mut first)
                || !line.get(..15).is_some_and(|name| name.eq_ignore_ascii_case(b"access-control-"))
        });
        owned.push_str(cors);
    }
    owned.push_str(if no_body || !chunked {
        "Connection: close\r\n"
    } else {
        "Transfer-Encoding: chunked\r\nConnection: close\r\n"
    });
    response.forward(
        writer,
        &response.parsed.first,
        &owned,
    )?;
    if !no_body {
        let mut body =
            BodyReader::new(upstream, &response.body_head(), false, transport::MAX_BODY)?;
        relay(&mut body, writer, metrics, recording, chunked)?;
    }
    Ok(true)
}

fn relay(
    body: &mut BodyReader<BufReader<DeadlineStream<'_>>>,
    writer: &mut impl Write,
    metrics: &mut Metrics,
    recording: &mut Option<Recording>,
    chunked: bool,
) -> io::Result<()> {
    loop {
        let bytes = body.fill_buf()?;
        if bytes.is_empty() {
            break;
        }
        let n = bytes.len();
        let started = Instant::now();
        if chunked {
            write!(writer, "{n:x}\r\n")?;
        }
        writer.write_all(bytes)?;
        let flushed = if chunked {
            writer.write_all(b"\r\n").and_then(|_| writer.flush())
        } else {
            writer.flush()
        };
        metrics.payload(bytes, started.elapsed());
        record::observe(recording, |r| r.payload(bytes));
        body.consume(n);
        if metrics.terminal() {
            // A client may stop reading as soon as the final model event arrives.
            // Validate the remaining upstream HTTP framing within a short bound.
            body.reader.get_mut().finish_framing();
        } else {
            flushed?;
        }
    }
    metrics.upstream_complete = true;
    if chunked {
        writer.write_all(b"0\r\n\r\n")
    } else {
        Ok(())
    }
}
fn status(head: &Head) -> io::Result<u16> {
    head.first
        .split(' ')
        .nth(1)
        .and_then(|v| v.parse::<u16>().ok())
        .filter(|n| (100..600).contains(n))
        .ok_or_else(|| transport::invalid("invalid upstream status"))
}
fn local_error(writer: &mut impl Write, code: u16, message: &str) -> io::Result<()> {
    let body = format!(
        "{{\"error\":{{\"message\":{}}}}}",
        crate::json::quote(message)
    );
    write!(
        writer,
        "HTTP/1.1 {code} Gateway Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    writer.flush()
}

#[cfg(test)]
#[path = "invisible_tests.rs"]
mod tests;

#[cfg(test)]
pub(super) fn handle_conformance(
    socket: TcpStream,
    url: &str,
    opt_in: bool,
    record_dir: Option<&str>,
) -> io::Result<()> {
    let runtime = Runtime::new(&[url.into()], Flags {
        stable: opt_in, affinity: opt_in, greedy: opt_in, ..Flags::default()
    })?;
    let recorder = record_dir.map(Recorder::open).transpose()?;
    handle(socket, &runtime, None, recorder.as_ref())
}
