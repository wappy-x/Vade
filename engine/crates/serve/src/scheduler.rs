//! One CUDA thread, bounded nonblocking sockets, priority between model steps.
mod dispatch;
#[cfg(test)]
#[path = "metadata_test.rs"]
mod metadata_tests;
#[cfg(test)]
#[path = "scheduler_retirement_tests.rs"]
mod retirement_tests;
use dispatch::serve_loop;

use crate::{
    Decoder, Engine, Limits, Session, Tokenizer,
    cache::{CacheLimits, CachedSession, SessionCache, SessionKey, shared_prefix},
    chat_output,
    generation::{Event, FinishReason, Generation},
    http,
    protocol::{self, Request},
};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    os::unix::{fs::{DirBuilderExt, PermissionsExt}, net::{UnixListener, UnixStream}},
    path::{Path, PathBuf},
    sync::{atomic::{AtomicBool, Ordering}, mpsc},
    time::{Duration, Instant},
};

/// Only canonical checkpoint directories under this root may be loaded.
pub struct ModelAdmin {
    pub root: PathBuf,
    pub model: String,
    socket: Option<AdminSocket>,
}
struct AdminSocket { listener: UnixListener, path: PathBuf }
impl AdminSocket {
    fn bind(path: &Path) -> Result<Self, String> {
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        match std::fs::DirBuilder::new().mode(0o700).create(parent) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.to_string()),
        }
        let metadata = std::fs::symlink_metadata(parent).map_err(|e| e.to_string())?;
        if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
            return Err("admin socket requires a private parent directory (0700)".into());
        }
        // The private directory closes the bind-to-chmod window. Never unlink
        // an existing socket: it may belong to a live server.
        let listener = UnixListener::bind(path).map_err(|e| e.to_string())?;
        let socket = Self { listener, path: path.to_owned() };
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|e| e.to_string())?;
        socket.listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        Ok(socket)
    }
}
impl Drop for AdminSocket {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.path); }
}
pub struct Swap {
    pub path: std::path::PathBuf,
    pub started: Instant,
    wire: Wire,
}
impl Swap {
    pub fn finish(mut self, status: u16, body: &str, limits: &Limits) -> Result<(), String> {
        self.wire.response(status, body, limits)?;
        while !self.wire.drained() {
            self.wire.flush()?;
            if !self.wire.drained() { std::thread::sleep(Duration::from_millis(1)); }
        }
        Ok(())
    }
}
impl ModelAdmin {
    pub fn new(root: PathBuf, model: String, socket: Option<&Path>) -> Result<Self, String> {
        Ok(Self { root, model, socket: socket.map(AdminSocket::bind).transpose()? })
    }
    fn accept(&self, reading: &mut Vec<Reading>, limits: &Limits) {
        let Some(socket) = &self.socket else { return; };
        while reading.iter().filter(|r| r.wire.socket.is_admin()).count() < 4 {
            match socket.listener.accept() {
                Ok((stream, _)) => match Reading::new(Socket::Admin(stream), limits) {
                    Ok(input) => reading.push(input),
                    Err(error) => eprintln!("admin connection: {error}"),
                },
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => { eprintln!("admin accept: {e}"); break; }
            }
        }
    }
    fn path(&self, body: &[u8], wire: &Wire) -> Result<std::path::PathBuf, String> {
        if !wire.socket.is_admin() {
            return Err("model administration requires the private Unix socket".into());
        }
        let value = engine_model::json::parse(body, engine_model::json::Limits::default())
            .map_err(|e| e.to_string())?;
        let object = value.object().map_err(|e| e.to_string())?;
        if object.len() != 1 { return Err("expected only checkpoint path".into()); }
        let path = engine_model::json::field(object, "path").and_then(|v| v.string())
            .map_err(|e| e.to_string())?;
        let path = std::path::Path::new(path);
        let path = if path.is_absolute() { path.to_owned() } else { self.root.join(path) };
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        if !path.starts_with(&self.root) { return Err("checkpoint is outside models root".into()); }
        // Reject incomplete weights before draining sessions or unloading the model.
        engine_model::qwen3::Checkpoint::inspect(&path).map_err(|e| e.to_string())?;
        Ok(path)
    }
}
fn model_response(path: &str, state: &str, model: &str, active: usize, queued: usize, device_bytes: Option<usize>, limits: &Limits, context: usize, drafting: Option<(protocol::DraftMode, usize)>) -> Option<String> {
    let model = protocol::quote(model);
    match path {
        "/cors" => Some(String::new()),
        "/v1/models" if state == "failed" || state == "poisoned" => Some(format!("{{\"object\":\"list\",\"status\":\"{state}\",\"data\":[]}}")),
        "/health" => Some(format!("{{\"status\":\"{state}\",\"model\":{model},\"active\":{active},\"queued\":{queued},\"device_bytes\":{},\"vade\":{},\"events\":{},\"caps\":{{\"context_limit\":{},\"input_limit\":{},\"output_limit\":{}}}}}",
            device_bytes.map_or_else(|| "null".to_owned(), |n| n.to_string()),
            limits.metadata.as_ref().map_or_else(|| "null".into(), |m| drafting
                .map_or_else(|| m.json().to_owned(), |(mode, depth)| m.json_with_drafting(mode, depth))),
            engine_model::events::counts_json(), context.min(limits.context_tokens), limits.input_tokens, limits.output_limit(context))),
        // This process's best-effort event counts and last lines (engine_base::events).
        "/v1/events" => Some(engine_model::events::json()),
        "/v1/models" => Some(format!("{{\"object\":\"list\",\"status\":\"{state}\",\"data\":[{{\"id\":{model},\"object\":\"model\",\"owned_by\":\"vade\",\"context_length\":{0},\"max_model_len\":{0},\"max_output_tokens\":{1},\"input_modalities\":[\"text\"]}}]}}", context.min(limits.context_tokens), limits.output_limit(context))),
        _ => None,
    }
}

#[cfg(test)]
#[test]
fn model_listing_reports_effective_limits() {
    use engine_model::json::{self, Value};
    for (configured, engine, output) in [(32768, 65536, 16384), (65536, 32768, 65536)] {
        let limits = Limits { context_tokens: configured, output_tokens: output, ..Limits::default() };
        for state in ["ready", "loading"] {
            let body = model_response("/v1/models", state, "model", 0, 0, None, &limits, engine, None).unwrap();
            let value = json::parse(body.as_bytes(), Default::default()).unwrap();
            let entry = value.object().unwrap()["data"].array().unwrap()[0].object().unwrap();
            for key in ["context_length", "max_model_len"] {
                assert_eq!(entry[key], Value::Number(configured.min(engine).to_string()));
            }
            assert_eq!(entry["max_output_tokens"], Value::Number(limits.output_limit(engine).to_string()));
            assert_eq!(entry["input_modalities"], Value::Array(vec![Value::String("text".into())]));
        }
        for state in ["failed", "poisoned"] {
            assert!(model_response("/v1/models", state, "model", 0, 0, None, &limits, engine, None).unwrap().contains("\"data\":[]"));
        }
    }
}

/// Keep health and explicit rejection responses available while CUDA loads on
/// its owning thread. This thread never borrows the model or backend.
pub fn serve_loading(listener: &TcpListener, limits: &Limits, model: &str, stop: &AtomicBool, admin: &ModelAdmin) -> Result<(), String> {
    unavailable(listener, limits, model, stop, Some(admin), "loading", None).map(|_| ())
}
pub fn wait_for_model(listener: &TcpListener, limits: &Limits, admin: &ModelAdmin, stop: &AtomicBool) -> Result<Option<Swap>, String> {
    unavailable(listener, limits, &admin.model, stop, Some(admin), "failed", None)
}
fn unavailable(listener: &TcpListener, limits: &Limits, model: &str, stop: &AtomicBool, admin: Option<&ModelAdmin>, state: &str, deadline: Option<Instant>) -> Result<Option<Swap>, String> {
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let access = http::AccessPolicy::new(listener.local_addr().map_err(|e| e.to_string())?, limits)?;
    let mut reading: Vec<Reading> = Vec::new();
    let mut requested = None;
    let mut stop_deadline = None;
    loop {
        if deadline.is_some_and(|d| Instant::now() >= d) { return Ok(None); }
        if stop.load(Ordering::Acquire) || requested.is_some() {
            if stop.load(Ordering::Acquire) {
                let deadline = stop_deadline.get_or_insert_with(|| Instant::now() + Duration::from_millis(500));
                if Instant::now() >= *deadline { return Ok(None); }
            }
            // An incomplete request already accepted during loading receives an
            // explicit retry response instead of being silently disconnected.
            for input in &mut reading {
                if !input.responding {
                    let _ = input.wire.response(503, &protocol::error_json("model_loading", "retry when ready"), limits);
                    input.responding = true;
                }
            }
            if reading.is_empty() { return Ok(requested); }
        } else {
            if let Some(admin) = admin { admin.accept(&mut reading, limits); }
            accept_connections(listener, &mut reading, limits, 33, 32);
        }
        let mut i = 0;
        while i < reading.len() {
            if reading[i].responding {
                if reading[i].wire.flush().is_err() || reading[i].wire.drained() { reading.swap_remove(i); }
                else { i += 1; }
                continue;
            }
            let result = reading[i].read(limits, &access);
            if matches!(result, Ok(None)) { i += 1; continue; }
            let mut input = reading.swap_remove(i);
            let (code, body) = match result {
                Ok(Some(http)) => {
                    if http.path == "/v1/admin/swap" && requested.is_none() && state == "failed" {
                        if let Some(admin) = admin {
                            match admin.path(&http.body, &input.wire) {
                                Ok(path) => { requested = Some(Swap { path, started: Instant::now(), wire: input.wire }); continue; }
                                Err(error) => {
                                    if input.wire.response(400, &protocol::error_json("invalid_request_error", &error), limits).is_ok() {
                                        input.responding = true; reading.push(input);
                                    }
                                    continue;
                                }
                            }
                        }
                    }
                    (if http.path == "/cors" { 204 } else { 503 }, model_response(&http.path, state, model, 0, 0, None, limits, limits.context_tokens, None)
                        .unwrap_or_else(|| protocol::error_json("model_unavailable", "retry when ready")))
                }
                Err(error) => crate::auth::request_error(&error),
                Ok(None) => unreachable!(),
            };
            if input.wire.response(code, &body, limits).is_ok() { input.responding = true; reading.push(input); }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[derive(Clone, Debug)]
pub struct Sharing {
    pub prefix_cache_per_api_key: bool,
    pub report_prefix_reuse: bool,
    /// Public requests cannot claim the owner's priority (255).
    pub client_priority_cap: u8,
    pub owner_key: Option<SessionKey>,
}
impl Default for Sharing {
    fn default() -> Self {
        Self { prefix_cache_per_api_key: true, report_prefix_reuse: false,
            client_priority_cap: 0, owner_key: None }
    }
}
impl Sharing {
    pub fn load_owner_key(&mut self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o777 != 0o600 {
            return Err("owner key file must be a regular file with permissions 0600".into());
        }
        let mut text = String::new();
        file.take(66).read_to_string(&mut text).map_err(|_| "invalid owner key file")?;
        if metadata.len() > 65 { return Err("owner key file must contain 64 lowercase hex digits".into()); }
        self.owner_key = Some(SessionKey::parse(text.trim_end())
            .map_err(|_| "owner key file must contain 64 lowercase hex digits")?);
        Ok(())
    }
    pub(crate) fn priority(&self, authorization: Option<&str>, requested: u8, authenticated: bool) -> u8 {
        let owner = if authenticated { authorization == Some("authenticated-owner") && self.owner_key.is_some() }
            else { authorization.and_then(|a| a.strip_prefix("Bearer "))
            .and_then(|token| SessionKey::parse(token).ok())
            .is_some_and(|key| self.owner_key.is_some_and(|owner| owner.matches(&key))) };
        if owner { 255 } else { requested.min(self.client_priority_cap) }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub active: usize,
    pub connections: usize,
    pub prefill_chunk: usize,
    /// Aggregate pool for request-sized active reservations.
    pub active_bytes: usize,
    pub session_budget_bytes: usize,
    pub draft: protocol::DraftMode,
    /// Default Next request depth, also used by the adapter and sampling echo.
    pub mtp_depth: usize,
    pub draft_global_bank: bool,
    pub prefix_cache: bool,
    pub sharing: Sharing,
    pub foreground_rows: usize,
    pub background_rows: usize,
}
impl Config {
    fn validate(&self, cache: &CacheLimits) -> Result<(), String> {
        if self.active == 0 || self.active == usize::MAX
            || self.sharing.client_priority_cap == 255
            || self.connections < self.active
            || self.connections > self.active.max(128)
            // The prompt tile: 2048 by default, 4096 or 8192 via VADE_PROMPT_TILE.
            || !(1..=8192).contains(&self.prefill_chunk)
            || !(1..=self.active.max(512)).contains(&self.foreground_rows)
            || !(1..=self.active.max(512)).contains(&self.background_rows)
            || self.active_bytes == 0
            || self.active_bytes > self.session_budget_bytes
            || cache.retained_bytes > self.session_budget_bytes
        {
            return Err("invalid scheduler capacity or aggregate session budget".into());
        }
        Ok(())
    }
}

/// Hybrid automatic drafting is reserved for a single active chat.
struct DraftPolicy { strict: bool, quiet_since: Option<Instant> }
impl DraftPolicy {
    fn new() -> Self { Self { strict: true, quiet_since: None } }
    fn update(&mut self, chats: usize, now: Instant, report: bool, chat_limit: usize) {
        let previous = self.strict;
        if chat_limit == 1 { self.strict = chats <= 1; self.quiet_since = None; }
        else if chats > chat_limit {
            self.strict = false;
            self.quiet_since = None;
        } else if !self.strict {
            let since = *self.quiet_since.get_or_insert(now);
            if now.duration_since(since) >= Duration::from_millis(500) { self.strict = true; }
        }
        if report && previous != self.strict {
            engine_model::events::note_line("draft_policy", &format!("draft_policy active_chats={chats} mode={}", if self.strict { "strict" } else { "plain" }));
        }
    }
    fn resolve(&self, mode: protocol::DraftMode) -> protocol::DraftMode {
        if mode == protocol::DraftMode::Auto {
            if self.strict { protocol::DraftMode::Strict } else { protocol::DraftMode::Plain }
        } else { mode }
    }
}

/// Recent committed KV histories supply drafts, never cached model outputs.
/// One bounded host copy; model swaps drop the bank with this scheduler.
struct DraftEntry {
    credential: Option<String>,
    session: Option<SessionKey>,
    prompt_len: usize,
    history: Vec<u32>,
}
#[derive(Default)]
struct DraftBank { entries: Vec<DraftEntry>, tokens: usize }
impl DraftBank {
    const TOKENS: usize = 65_536;
    fn propose(&self, credential: Option<&str>, session: Option<SessionKey>, global: bool, prompt: &[u32]) -> &[u32] {
        self.entries.iter().rev().find(|entry| entry.credential.as_deref() == credential
            && (global || session.is_some() && entry.session == session)
            && entry.prompt_len == prompt.len() && entry.history.starts_with(prompt))
            .map_or(&[], |entry| &entry.history[entry.prompt_len..])
    }
    fn record(&mut self, credential: Option<&str>, session: Option<SessionKey>, global: bool, prompt_len: usize, history: &[u32]) {
        if (!global && session.is_none()) || history.len() <= prompt_len || history.len() > Self::TOKENS { return; }
        if let Some(i) = self.entries.iter().position(|entry| entry.credential.as_deref() == credential
            && entry.session == session && entry.prompt_len == prompt_len && entry.history.starts_with(&history[..prompt_len])) {
            self.tokens -= self.entries.remove(i).history.len();
        }
        while self.entries.len() >= 16 || self.tokens + history.len() > Self::TOKENS {
            self.tokens -= self.entries.remove(0).history.len();
        }
        self.tokens += history.len();
        self.entries.push(DraftEntry { credential: credential.map(str::to_owned), session, prompt_len, history: history.to_vec() });
    }
    fn memory_bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<DraftEntry>()
            + self.entries.iter().map(|entry| entry.history.capacity() * 4
                + entry.credential.as_ref().map_or(0, String::capacity)).sum::<usize>()
    }
}

/// Bound background work between foreground turns without starving prefill
/// behind a continuous decode stream. The engine yields between layers.
struct PrefillBudget { max_slice: Duration }
impl PrefillBudget {
    fn new() -> Self { Self { max_slice: Duration::ZERO } }
    fn slice(&self, contended: bool) -> Duration {
        Duration::from_millis(if contended { 100 } else { 15 })
    }
}

// Output is drained before any further model work for this connection.
const PENDING_BYTES: usize = 1024 * 1024;
const HEARTBEAT: Duration = Duration::from_millis(250);
enum Socket { Tcp(TcpStream), Admin(UnixStream) }
impl Socket {
    fn is_admin(&self) -> bool { matches!(self, Self::Admin(_)) }
    fn configure(&self) -> std::io::Result<()> {
        match self {
            Self::Tcp(socket) => { socket.set_nonblocking(true)?; socket.set_nodelay(true) }
            Self::Admin(socket) => socket.set_nonblocking(true),
        }
    }
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        match self { Self::Tcp(s) => s.read(bytes), Self::Admin(s) => s.read(bytes) }
    }
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        match self { Self::Tcp(s) => s.write(bytes), Self::Admin(s) => s.write(bytes) }
    }
    fn input_pending(&self) -> bool {
        match self {
            Self::Tcp(s) => !matches!(s.peek(&mut [0; 1]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock),
            Self::Admin(_) => true,
        }
    }
}
struct Wire {
    socket: Socket,
    bytes: Vec<u8>,
    sent: usize,
    total: usize,
    streaming: bool,
    write_deadline: Instant,
    last_write: Instant,
    received: Instant,
    cors: String,
    anthropic: bool,
    control: Option<ControlWrite>,
    control_only: bool,
}
struct ControlWrite {
    body: Option<http::ControlBody>,
    cursor: usize,
    timeout: Duration,
}
impl Wire {
    fn new(socket: Socket) -> Result<Self, String> {
        socket.configure().map_err(|e| e.to_string())?;
        Ok(Self {
            socket,
            bytes: Vec::new(),
            sent: 0,
            total: 0,
            streaming: false,
            write_deadline: Instant::now(),
            last_write: Instant::now(),
            received: Instant::now(),
            cors: String::new(),
            anthropic: false,
            control: None,
            control_only: false,
        })
    }
    fn push(&mut self, text: &str, limits: &Limits, terminal: bool) -> Result<(), String> {
        let ceiling = limits
            .output_bytes
            .saturating_sub(if terminal { 0 } else { 1024 });
        if (!self.streaming && self.total.saturating_add(text.len()) > ceiling)
            || self.bytes.len().saturating_add(text.len()) > PENDING_BYTES.saturating_sub(if terminal { 0 } else { 1024 })
        {
            return Err("output byte limit exceeded".into());
        }
        if self.bytes.is_empty() {
            self.write_deadline = Instant::now() + limits.write_timeout;
        }
        self.bytes.extend_from_slice(text.as_bytes());
        self.total = self.total.saturating_add(text.len());
        Ok(())
    }
    fn flush(&mut self) -> Result<(), String> {
        if self.bytes.is_empty() {
            if let Some(control) = &mut self.control {
                if control.body.as_ref().is_some_and(|body| body.fill(&mut control.cursor, &mut self.bytes)) {
                    // Release the input IDs as soon as the final chunk is
                    // staged. Its bytes and deadline stay until fully sent.
                    control.body = None;
                }
            }
        }
        if self.bytes.is_empty() {
            if self.control.as_ref().is_some_and(|control| control.body.is_none()) {
                self.control = None;
            }
            return Ok(());
        }
        if Instant::now() >= self.write_deadline {
            return Err("response write deadline exceeded".into());
        }
        // One bounded write per scheduler visit; never wait for a reader.
        match self.socket.write(&self.bytes[self.sent..]) {
            Ok(0) => return Err("client disconnected".into()),
            Ok(n) => {
                self.sent += n;
                self.last_write = Instant::now();
                // A long control response may span many quanta. Only a stalled
                // reader times out; successful writes renew its drain deadline.
                if let Some(control) = &self.control {
                    self.write_deadline = self.last_write + control.timeout;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.to_string()),
        }
        if self.sent == self.bytes.len() {
            self.bytes.clear();
            self.sent = 0;
            if self.control.as_ref().is_some_and(|control| control.body.is_none()) {
                self.control = None;
            }
        }
        Ok(())
    }
    fn heartbeat(&mut self, limits: &Limits) -> Result<(), String> {
        if self.bytes.is_empty() && self.last_write.elapsed() >= HEARTBEAT {
            let ping = if self.anthropic { "event: ping\ndata: {\"type\":\"ping\"}\n\n" } else { ": heartbeat\n\n" };
            self.push(ping, limits, false)?;
        }
        self.flush()
    }
    fn connected(&self, streaming: bool) -> Result<(), String> {
        let Socket::Tcp(socket) = &self.socket else { return Ok(()); };
        match socket.peek(&mut [0; 1]) {
            // SSE writes/heartbeats detect a closed reader while preserving
            // streaming half-close. JSON has no writes until completion, so
            // EOF must cancel its work rather than run to the output limit.
            Ok(0) if streaming => Ok(()),
            Ok(0) => Err("client disconnected".into()),
            Ok(_) => Err("unexpected pipelined bytes".into()),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
    fn response(&mut self, status: u16, body: &str, limits: &Limits) -> Result<(), String> {
        if self.anthropic && status >= 400 && status != 401 {
            let (status, body) = http::anthropic_error(status, body);
            return self.json_response(status, &body, limits);
        }
        self.json_response(status, body, limits)
    }

    fn json_response(&mut self, status: u16, body: &str, limits: &Limits) -> Result<(), String> {
        let head = http::response_head(status, body.len(), &self.cors);
        let length = if status == 204 { 0 } else { body.len() };
        // Preserve the generation output budget without making the socket
        // queue impose an additional, smaller lifetime response ceiling.
        if self.total.saturating_add(head.len()).saturating_add(length) > limits.output_bytes {
            return Err("output byte limit exceeded".into());
        }
        self.begin_response(head, http::ControlBody::Json(if status == 204 {
            String::new()
        } else {
            body.to_owned()
        }), limits)
    }

    fn control_response(&mut self, body: http::ControlBody, limits: &Limits) -> Result<(), String> {
        // Never enqueue a complete large response atomically. The HTTP length
        // is known before sending, and fill() stages at most 64 KiB per visit.
        let head = http::response_head(200, body.len(), &self.cors);
        self.begin_response(head, body, limits)
    }

    fn terminal_events(&mut self, events: String, limits: &Limits) -> Result<(), String> {
        // Responses includes the whole completed output in its final SSE
        // event. Its decoded content is already bounded by the output parser;
        // let framing drain in pieces rather than applying the pending limit
        // to the entire event. Existing queued deltas always drain first.
        self.begin_response(String::new(), http::ControlBody::Json(events), limits)
    }

    fn begin_response(&mut self, head: String, body: http::ControlBody,
        limits: &Limits) -> Result<(), String> {
        if self.control.is_some() {
            return Err("response already started".into());
        }
        self.push(&head, limits, true)?;
        self.total = self.total.saturating_add(body.len());
        self.control = Some(ControlWrite { body: Some(body), cursor: 0, timeout: limits.write_timeout });
        Ok(())
    }

    fn drained(&self) -> bool {
        self.bytes.is_empty() && self.control.is_none()
    }
}

#[cfg(test)] #[test] fn control_http_responses_resume_with_exact_lengths() {
    for tokens in [vec![], vec![0, 9, 10, 999999, u32::MAX], vec![123456; 131071], vec![123456; 262144]] { let expected = format!("{{\"count\":{},\"max_model_len\":262144,\"tokens\":[{}]}}", tokens.len(), tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(",")); let body = http::ControlBody::tokens(tokens, 262144); assert_eq!(body.len(), expected.len());
        for (status, body, expected) in [(200, body, expected), (204, http::ControlBody::Json(String::new()), String::new())] { let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap(); let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap(); client.set_read_timeout(Some(Duration::from_secs(5))).unwrap(); let reader = std::thread::spawn(move || { std::thread::sleep(Duration::from_millis(20)); let mut reply = String::new(); client.read_to_string(&mut reply).unwrap(); reply }); let mut wire = Wire::new(Socket::Tcp(listener.accept().unwrap().0)).unwrap(); if status == 200 { wire.control_response(body, &Limits::default()).unwrap(); } else { wire.response(status, "", &Limits::default()).unwrap(); }
            while !wire.drained() { assert!(wire.bytes.len() <= http::ControlBody::CHUNK_BYTES); wire.flush().unwrap(); std::thread::yield_now(); } drop(wire); let reply = reader.join().unwrap(); let (head, actual) = reply.split_once("\r\n\r\n").unwrap(); assert!(head.starts_with(&format!("HTTP/1.1 {status}"))); if status == 200 { assert!(head.contains(&format!("Content-Length: {}\r\n", expected.len()))); } assert_eq!(actual, expected);
        } }
}

struct Reading {
    wire: Wire,
    data: Vec<u8>,
    head: Option<(usize, http::Head)>,
    deadline: Instant,
    responding: bool,
}
impl Reading {
    fn new(socket: Socket, limits: &Limits) -> Result<Self, String> {
        Ok(Self {
            wire: Wire::new(socket)?,
            data: Vec::new(),
            head: None,
            deadline: Instant::now() + limits.read_timeout,
            responding: false,
        })
    }
    fn read(&mut self, limits: &Limits, access: &http::AccessPolicy) -> Result<Option<http::HttpRequest>, String> {
        if !self.wire.bytes.is_empty() {
            self.wire.flush()?;
            if !self.wire.bytes.is_empty() { return Ok(None); }
        }
        let mut buf = [0; 8192];
        loop {
            if let Some((end, head)) = &self.head {
                if self.data.len() > end + head.length {
                    return Err("unexpected trailing request bytes".into());
                }
                if self.data.len() == end + head.length {
                    // Overflow admission reads the whole request before replying,
                    // then uses the normal response drain without evicting owners.
                    if self.wire.control_only && !head.control_target {
                        engine_model::events::note("admission_refused", format_args!("reason=connection_table_full body_bytes={}", head.length));
                        self.wire.response(
                            503,
                            &protocol::error_json("overloaded", "connection capacity reached"),
                            limits,
                        )?;
                        self.responding = true;
                        return Ok(None);
                    }
                    return Ok(Some(http::HttpRequest {
                        path: head.path.clone(),
                        body: self.data[*end..].to_vec(),
                        authorization: head.authorization.clone(),
                    }));
                }
            }
            match self.wire.socket.read(&mut buf) {
                Ok(0) => return Err("client disconnected".into()),
                Ok(n) => self.data.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    return if Instant::now() >= self.deadline {
                        Err("request read deadline exceeded".into())
                    } else { Ok(None) };
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.to_string()),
            }
            if self.head.is_none() {
                // Preserve the requested error format even when the header/body
                // is rejected before it can become an HttpRequest.
                if let Some(line) = self.data.windows(2).position(|w| w == b"\r\n") {
                    self.wire.anthropic = std::str::from_utf8(&self.data[..line]).ok()
                        .and_then(|s| s.split(' ').nth(1)).is_some_and(http::anthropic_route);
                }
                if let Some(index) = self.data.windows(4).position(|w| w == b"\r\n\r\n") {
                    if index + 4 > limits.header_bytes {
                        return Err("header byte limit exceeded".into());
                    }
                    let head = http::parse_head(&self.data[..index], access, self.wire.socket.is_admin())?;
                    if head.length > limits.body_bytes {
                        return Err("body byte limit exceeded".into());
                    }
                    self.wire.cors = head.cors.clone();
                    let continue_body = head.continue_body;
                    self.head = Some((index + 4, head));
                    if continue_body {
                        self.wire.push("HTTP/1.1 100 Continue\r\n\r\n", limits, false)?;
                        self.wire.flush()?;
                        if !self.wire.bytes.is_empty() { return Ok(None); }
                    }
                } else if self.data.len() >= limits.header_bytes {
                    return Err("header byte limit exceeded".into());
                }
            }
        }
    }
}

/// Never evict an accepted reader or a draining response. At capacity, reserve
/// eight bounded readers for health/cancel; other complete requests drain a 503.
/// Overflow readers have a short deadline, so silent peers cannot hold the reserve.
fn accept_connections(
    listener: &TcpListener,
    reading: &mut Vec<Reading>,
    limits: &Limits,
    slots: usize,
    attempts: usize,
) -> usize {
    const CONTROL_SLOTS: usize = 8;
    const CONTROL_READ_TIMEOUT: Duration = Duration::from_millis(100);
    let mut controls = reading
        .iter()
        .filter(|input| input.wire.control_only)
        .count();
    let mut accepted = 0;
    for _ in 0..attempts {
        let control_only = reading.len() >= slots;
        if control_only && controls >= CONTROL_SLOTS {
            break;
        }
        let socket = match listener.accept() {
            Ok((socket, _)) => socket,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                eprintln!("accept: {e}");
                break;
            }
        };
        accepted += 1;
        match Reading::new(Socket::Tcp(socket), limits) {
            Ok(mut input) => {
                if control_only {
                    input.wire.control_only = true;
                    input.deadline = input.deadline.min(Instant::now() + CONTROL_READ_TIMEOUT);
                    controls += 1;
                }
                reading.push(input);
            }
            Err(error) => eprintln!("connection setup: {error}"),
        }
    }
    accepted
}

enum Prepared { Generation(Ready), Control(Wire, http::ControlBody) }

struct Ready {
    model: String,
    tool_hash: u64,
    wire: Wire,
    request: Request,
    prompt: Vec<u32>,
    parser: Option<chat_output::Parser>,
    output: Option<http::Output>,
    enqueued: Instant,
    deadline: Instant,
    cancel: SessionKey,
    authorization: Option<String>,
    scope: Option<String>,
    handle_scope: Option<String>,
    affinity_scope: Option<String>,
}
impl Ready {
    fn reservation_tokens(&self, output: usize) -> usize {
        let output = if self.request.max_tokens_explicit { self.request.max_tokens }
            else { output.max(1).div_ceil(256).saturating_mul(256).min(self.request.max_tokens) };
        self.prompt.len() + output
    }
    /// A queued cancellation is a failure, never an exhausted output budget.
    fn cancel(&mut self, id: u64, error: RequestError, limits: &Limits, domain: &crate::contract::Domain, mtp_depth: usize) -> Result<(), String> {
        let mut echo = protocol::SamplingEcho::new(&self.request, protocol::DraftMode::Plain, mtp_depth).with_metadata(limits).with_events();
        echo.prefill = Some(crate::usage::Prefill::new(self.prompt.len(), 0, self.enqueued));
        echo.log(id);
        eprintln!("request_cancelled id={id} code={} vade={}", error.code(), echo.json());
        let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_secs();
        let mut usage = crate::generation::Usage { timing: self.request.include_usage_timing.then(Default::default),
            ..Default::default() };
        if self.request.kv_checkpoint || self.request.kv_handle.is_some() {
            usage.kv = Some(crate::usage::Kv {
                cache_domain: domain.key(self.authorization.as_deref()),
                reuse: if self.request.kv_handle.is_some() { "not_attempted" } else { "not_requested" },
                reused_tokens: 0, checkpoint: None,
                retention: if self.request.kv_checkpoint { "prompt_incomplete" } else { "not_requested" },
            });
        }
        if let Some(output) = &mut self.output {
            if self.request.stream {
                self.wire.push(&format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n{}\r\n", self.wire.cors), limits, false)?;
                self.wire.push(&output.start(&self.model, id, created, usage), limits, false)?;
                return self.wire.terminal_events(output.failed(&self.model, id, created, usage,
                    error.status(), error.code(), error.cancellation_message(), true, &echo), limits);
            }
            return self.wire.response(error.status(), &output.failed(&self.model, id, created, usage,
                error.status(), error.code(), error.cancellation_message(), false, &echo), limits);
        }
        let body = echo.attach(protocol::error_json(error.code(), error.cancellation_message()));
        if self.request.stream {
            self.wire.push(&format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n{}\r\n", self.wire.cors), limits, false)?;
            self.wire.terminal_events(format!("data: {body}\n\ndata: {{\"choices\":[],\"usage\":{}}}\n\ndata: [DONE]\n\n", usage.json(false)), limits)
        } else {
            self.wire.response(error.status(), &body, limits)
        }
    }
}
fn prepare<T: Tokenizer>(
    http: http::HttpRequest,
    wire: Wire,
    tokenizer: Option<&T>,
    limits: &Limits,
    model: &str,
    context: usize,
    vocab: usize,
    eos: &[u32],
) -> Result<Prepared, (Wire, String)> {
    if http.path == "/v1/admission" {
        return match crate::contract::admission(&http.body, context, vocab, eos, tokenizer, limits, model) {
            Ok(body) => Ok(Prepared::Control(wire, http::ControlBody::Json(body))),
            Err(error) => Err((wire, error)),
        };
    }
    if http.path == "/v1/messages/count_tokens" {
        return match http::count_tokens(&http, context, vocab, tokenizer, limits, model) {
            Ok(body) => Ok(Prepared::Control(wire, http::ControlBody::Json(body))),
            Err(error) => Err((wire, error)),
        };
    }
    if http.path == "/tokenize" {
        return match http::tokenize(&http, context, vocab, tokenizer, limits, model) {
            Ok(body) => Ok(Prepared::Control(wire, body)),
            Err(error) => Err((wire, error)),
        };
    }
    match http::prepare_request(&http, context, vocab, eos, tokenizer, limits, model) {
        Ok((request, prompt, parser, output)) => {
            let mut wire = wire;
            wire.streaming = request.stream;
            let cancel = match request.cancel.map(Ok).unwrap_or_else(SessionKey::generate) {
                Ok(key) => key,
                Err(error) => return Err((wire, error)),
            };
            let enqueued = wire.received;
            let affinity_scope = if request.prompt_cache_key.is_some() { http.authorization.clone() } else { None };
            let scope = crate::cache::namespace(http.authorization.as_deref(), request.cache_salt.as_deref());
            Ok(Prepared::Generation(Ready {
                model: model.to_owned(),
                tool_hash: http::request_hash(&http.body),
                wire,
                request,
                prompt,
                parser,
                output,
                enqueued,
                deadline: enqueued + limits.generation_timeout,
                cancel,
                affinity_scope,
                handle_scope: scope.clone(),
                scope,
                authorization: http.authorization,
            }))
        }
        Err(error) => Err((wire, error)),
    }
}

struct Active<'e, 't, E: Engine + 'e, T: Tokenizer + 't> {
    ready: Ready,
    generation: Generation<E::Session<'e>>,
    long_prefill: Option<Box<dyn crate::LongPrefill<E::Session<'e>> + 'e>>,
    prefill_started: bool,
    key: Option<SessionKey>,
    automatic_cache: bool,
    decoder: Option<T::Decoder<'t>>,
    id: u64,
    created: u64,
    response: http::Completion,
    content_bytes: usize,
    echo: protocol::SamplingEcho,
    stop: protocol::StopBuffer,
    tool_index: usize,
    output_failed: bool,
    done: bool,
    terminal_logged: bool,
    last_tick: u64,
    prefill_rows: usize,
    admitted: Instant,
    usage_clock: Option<crate::usage::Clock>,
    memory_reported_tokens: usize,
    profile: bool,
    first_token_sent: bool,
    first_token_generated: bool,
    reserved_tokens: usize,
    reservation_plain: bool,
}
struct Retired<S: Session> {
    state: CachedSession<S>,
    finished: Instant,
    id: u64,
}
impl<S: Session> Retired<S> {
    fn release(self) {
        let started = Instant::now();
        let deferred_ms = self.finished.elapsed().as_secs_f64() * 1000.0;
        let released = self.state.reserved_session_bytes;
        drop(self.state);
        eprintln!("request_retired id={} released_reserved_bytes={released} wall_ms={:.3} deferred_ms={deferred_ms:.3}",
            self.id, started.elapsed().as_secs_f64() * 1000.0);
    }
}
fn retirement_pressure_refresh(value: Result<String, std::env::VarError>) -> Result<bool, String> {
    match value.as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("1") => Ok(true),
        Ok("0") => Ok(false),
        _ => Err("VADE_RETIRE_PRESSURE_REFRESH must be 0 or 1".into()),
    }
}
// The disabled path installs no callback and uses main's unchanged timed loop.
fn retirement_refresh_hook(enabled: bool, mut refresh: impl FnMut() -> Result<(), String>)
    -> Option<impl FnMut()> {
    enabled.then(|| move || {
        if let Err(error) = refresh() {
            engine_model::events::note_line("retirement_pressure_refresh_skipped", &format!("retirement_pressure_refresh skipped: {error}"));
        }
    })
}
fn reclaim_retired<S: Session>(retired: &mut Vec<Retired<S>>, budget: Duration) {
    let started = Instant::now();
    while !retired.is_empty() && started.elapsed() < budget {
        retired.remove(0).release();
    }
}
fn reclaim_retired_refresh<S: Session>(retired: &mut Vec<Retired<S>>, budget: Duration,
    mut released: impl FnMut()) {
    let started = Instant::now();
    while !retired.is_empty() && started.elapsed() < budget {
        retired.remove(0).release();
        released();
    }
}

#[cfg(test)]
#[test]
fn retirement_refresh_defaults_on_and_explicit_zero_installs_no_callback() {
    let cases = [(Err(std::env::VarError::NotPresent), true),
        (Ok("0".to_owned()), false), (Ok("1".to_owned()), true)];
    for (value, expected) in cases {
        let enabled = retirement_pressure_refresh(value).unwrap();
        let mut calls = 0;
        {
            let mut hook = retirement_refresh_hook(enabled, || { calls += 1; Ok(()) });
            assert_eq!(hook.is_some(), expected);
            if let Some(refresh) = hook.as_mut() { refresh(); }
        }
        assert_eq!(calls, usize::from(expected));
    }
    for value in ["", "2", "true", " 1"] {
        assert!(retirement_pressure_refresh(Ok(value.to_owned())).is_err());
    }
    assert!(retirement_pressure_refresh(Err(std::env::VarError::NotUnicode("invalid".into()))).is_err());
}

#[cfg(test)]
#[test]
fn retirement_refresh_off_installs_no_callback() {
    let mut calls = 0;
    {
        let hook = retirement_refresh_hook(false, || { calls += 1; Ok(()) });
        assert!(hook.is_none());
    }
    assert_eq!(calls, 0);
}

#[cfg(test)]
#[test]
fn retirement_refresh_runs_after_drop_before_the_next_release() {
    use std::{cell::Cell, rc::Rc};
    struct Ended(Rc<Cell<usize>>);
    impl Session for Ended {
        fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
    }
    impl Drop for Ended {
        fn drop(&mut self) { self.0.set(self.0.get() + 1); }
    }
    let dropped = Rc::new(Cell::new(0));
    let mut retired = (0..3).map(|id| Retired {
        state: CachedSession { checkpoint_lease: None, session: Ended(dropped.clone()),
            tokens: vec![], next_logits: vec![], reserved_session_bytes: 1024 },
        finished: Instant::now(), id,
    }).collect::<Vec<_>>();
    let mut refreshed = 0;
    // Failed pressure reads must neither interrupt remaining destruction nor
    // escape the serving loop before its terminal-response drain.
    {
        let mut hook = retirement_refresh_hook(true, || {
            refreshed += 1;
            assert_eq!(dropped.get(), refreshed);
            if refreshed == 1 { Err("pressure read failed".into()) } else { Ok(()) }
        }).unwrap();
        reclaim_retired_refresh(&mut retired, Duration::from_secs(1), &mut hook);
    }
    assert_eq!((retired.len(), refreshed), (0, 3));
    reclaim_retired_refresh(&mut retired, Duration::from_secs(1), || panic!("no release"));
    assert_eq!(retirement_pressure_refresh(Err(std::env::VarError::NotPresent)), Ok(true));
    assert_eq!(retirement_pressure_refresh(Ok("0".into())), Ok(false));
    assert_eq!(retirement_pressure_refresh(Ok("1".into())), Ok(true));
    for bad in ["", "true", "01", "2", " 1"] {
        assert!(retirement_pressure_refresh(Ok(bad.into())).is_err());
    }
}

enum RequestError { Generation, Output, Capacity, Swapping, Shutdown, Cancelled, Deadline }
fn shared_plain<'a>(source: &Ready, config: &Config, others: impl Iterator<Item = &'a Ready>) -> bool {
    config.prefix_cache && source.request.prefix_cache
        && source.request.draft.unwrap_or(config.draft) == protocol::DraftMode::Auto
        && source.prompt.get(..1024).is_some_and(|prefix| others.into_iter().any(|other|
            !std::ptr::eq(source, other) && other.request.prefix_cache && other.scope == source.scope
                && other.prompt.starts_with(prefix)))
}
impl RequestError {
    fn code(&self) -> &'static str {
        match self {
            Self::Generation => "generation_error",
            Self::Output => "invalid_model_output",
            Self::Capacity => "capacity_exhausted",
            Self::Swapping => "model_swapping",
            Self::Shutdown => "server_shutdown",
            Self::Cancelled => "client_cancelled",
            Self::Deadline => "generation_deadline",
        }
    }
    fn status(&self) -> u16 {
        match self { Self::Cancelled => 409, Self::Deadline => 504, Self::Output => 422,
            Self::Generation => 500, Self::Capacity | Self::Swapping | Self::Shutdown => 503 }
    }
    fn cancellation_message(&self) -> &'static str {
        match self { Self::Deadline => "generation deadline exceeded", _ => "request cancelled by client" }
    }
}
fn complete_batch<T>(result: Result<Vec<T>, crate::BatchError>, rows: usize) -> Result<Vec<T>, crate::BatchError> {
    result.and_then(|output| if output.len() == rows { Ok(output) }
        else { Err(crate::BatchError { request: None, message: "incomplete batch result".into() }) })
}

fn reservations<'s, S: Session + 's>(states: impl Iterator<Item = &'s CachedSession<S>>)
    -> (usize, std::collections::BTreeSet<usize>) {
    let mut pages = std::collections::BTreeSet::new();
    let (mut bytes, mut duplicate) = (0usize, 0usize);
    for state in states {
        bytes += state.reserved_session_bytes;
        for (id, size) in state.session.prefix_allocations(state.tokens.len()).unwrap_or_default() {
            if !pages.insert(id) { duplicate += size; }
        }
        pages.extend(state.session.funded_snapshot_allocations(None).into_iter().map(|(id, _)| id));
    }
    // Reservations charge physical BF16 pages once. Fork tail/growth slack
    // remains private to each request even when immutable pages are shared.
    (bytes.saturating_sub(duplicate), pages)
}
/// `hold`: device bytes outside the reservations that unfunded retention must leave free
/// (Engine::cache_hold_bytes plus each live request's Engine::unfunded_request_bytes).
/// `free`: Engine::cache_free_physical_bytes for the live reservations.
/// Admission itself keeps main's bound: only the cache's share shrinks.
fn balance_cache<'s, S: Session + 's>(cache: &mut SessionCache<S>, budget: usize,
    states: impl Iterator<Item = &'s CachedSession<S>>, hold: usize,
    free: impl FnOnce(usize) -> Result<Option<(i128, usize)>, String>) -> Result<usize, String> {
    let started = Instant::now();
    let (reserved, pages) = reservations(states);
    let accounting_ms = started.elapsed().as_secs_f64()*1000.0;
    let available = budget.checked_sub(reserved).ok_or_else(|| crate::budget::session_refusal(reserved as u128, budget, "live request reservations; CONTEXT and session count not supplied to cache balancing", "the scheduler session pool after weights and shared workspace"))?;
    cache.rebalance_within(available.saturating_sub(hold), free(reserved)?, pages, Instant::now());
    if reserved.checked_add(cache.unfunded_bytes()).is_none_or(|bytes| bytes > budget) {
        return Err(crate::budget::session_refusal(reserved as u128 + cache.unfunded_bytes() as u128, budget, &format!("live request reservations {reserved} bytes + unfunded retained cache {} bytes; CONTEXT not supplied to cache balancing", cache.unfunded_bytes()), "the scheduler session pool after weights and shared workspace"));
    }
    if std::env::var("SPARK_PROFILE").as_deref() == Ok("1") && started.elapsed().as_micros() >= 100 {
        eprintln!("cache_balance_profile accounting_ms={accounting_ms:.3} total_ms={:.3}", started.elapsed().as_secs_f64()*1000.0);
    }
    Ok(reserved)
}
impl<'e, 't, E: Engine, T: Tokenizer> Active<'e, 't, E, T> {
    fn sampling_echo(&self) -> protocol::SamplingEcho {
        protocol::SamplingEcho { draft_used: self.generation.draft_used,
            model_draft_used: self.generation.model_draft_used, ..self.echo.clone() }.with_events()
    }
    fn chunk(
        &mut self,
        text: &str,
        token: Option<u32>,
        finish: Option<&str>,
        limits: &Limits,
    ) -> Result<(), String> {
        if !self.ready.request.stream || self.ready.output.is_some() { return Ok(()); }
        self.ready.wire.push(
            &http::sse_chunk(
                &self.ready.model,
                self.ready.request.chat,
                self.id,
                self.created,
                text,
                token,
                finish,
                finish.map(|_| self.generation.usage),
                finish.map(|_| self.sampling_echo()).as_ref(),
            ),
            limits,
            finish.is_some(),
        )
    }

    fn emit(&mut self, event: chat_output::Event, token: Option<u32>, limits: &Limits) -> Result<(), String> {
        if let Some(output) = &mut self.ready.output {
            let events = output.push(event, self.id, &mut self.tool_index, limits).map_err(|error| {
                if matches!(output, http::Output::Anthropic(_)) { self.output_failed = true; }
                error
            })?;
            return if self.ready.request.stream { self.ready.wire.push(&events, limits, false) } else { Ok(()) };
        }
        if self.ready.request.stream {
            let delta = http::chat_delta(event, self.ready.tool_hash, &mut self.tool_index);
            self.chunk(&delta, token, None, limits)
        } else {
            self.response.push(event, self.ready.tool_hash, &mut self.tool_index, limits)
        }
    }
    fn text(&mut self, text: &str, token: Option<u32>, limits: &Limits) -> Result<(), String> {
        if self.ready.request.stop.is_empty() { return self.output(text, token, limits); }
        let text = self.stop.push(text, &self.ready.request.stop);
        self.output(&text, token, limits)
    }
    fn output(&mut self, text: &str, token: Option<u32>, limits: &Limits) -> Result<(), String> {
        if let Some(parser) = &mut self.ready.parser {
            let events = parser.push(text.as_bytes()).map_err(|error| {
                self.output_failed = true;
                error
            })?;
            if events.is_empty() {
                return self.chunk("{}", token, None, limits);
            }
            for (index, event) in events.into_iter().enumerate() {
                self.emit(event, if index == 0 { token } else { None }, limits)?;
            }
            Ok(())
        } else if self.ready.request.stream {
            self.content_bytes = self.content_bytes.checked_add(text.len()).ok_or("output byte count overflow")?;
            if self.content_bytes > limits.output_bytes.saturating_sub(1024) {
                return Err("output byte limit exceeded".into());
            }
            self.chunk(text, token, None, limits)
        } else {
            self.response.push(chat_output::Event::Content(text.to_owned()), self.id, &mut self.tool_index, limits)
        }
    }
    fn update_usage(&mut self) {
        if let Some(cached) = self.generation.state.session.cached_prompt_tokens(self.ready.prompt.len()) {
            self.generation.usage.cached_tokens = cached.min(self.ready.prompt.len());
        }
        if let Some(prefill) = &mut self.echo.prefill {
            prefill.cache_restored_tokens = self.generation.usage.cached_tokens;
        }
        if let Some(kv) = &mut self.generation.usage.kv {
            if matches!(kv.reuse, "hit" | "replayed") {
                kv.reused_tokens = self.generation.usage.cached_tokens;
                kv.reuse = if kv.reused_tokens == 0 { "replayed" } else { "hit" };
            }
        }
        self.generation.usage.timing = self.usage_clock.as_ref().map(crate::usage::Clock::elapsed);
    }
    fn checkpoint(&mut self, cache: &mut SessionCache<E::Session<'e>>) {
        let Some(kv) = &mut self.generation.usage.kv else { return; };
        if kv.retention != "pending" || self.generation.state.tokens.len() != self.ready.prompt.len() { return; }
        let state = &self.generation.state;
        cache.fund_pages(state.session.prefix_allocations(state.tokens.len()).unwrap_or_default());
        match cache.checkpoint(self.ready.handle_scope.as_deref(), self.ready.authorization.as_deref(), state, Instant::now()) {
            Ok(checkpoint) => { kv.checkpoint = Some(checkpoint); kv.retention = "retained"; }
            Err(reason) => kv.retention = reason,
        }
    }
    fn finish(&mut self, reason: FinishReason, limits: &Limits) -> Result<(), String> {
        if Instant::now() >= self.ready.deadline {
            return self.cancel(RequestError::Deadline, limits);
        }
        if let Some(kv) = &mut self.generation.usage.kv { kv.finish(); }
        self.update_usage();
        let tail = match &mut self.decoder {
            Some(d) => d.finish()?,
            None => String::new(),
        };
        if !tail.is_empty() {
            self.text(&tail, None, limits)?;
        }
        let pending = std::mem::take(&mut self.stop.pending);
        if !pending.is_empty() { self.output(&pending, None, limits)?; }
        if let Some(parser) = &mut self.ready.parser {
            let events = match parser.finish() {
                Ok(events) => events,
                Err(error) => { self.output_failed = true; return Err(error); }
            };
            for event in events {
                self.emit(event, None, limits)?;
            }
        }
        // A speculative batch can contain tokens beyond the stop; never retain that suffix.
        if self.stop.matched || self.ready.parser.as_ref().is_some_and(|p| p.stopped()) { self.key = None; }
        if self.key.is_none() {
            self.generation.state.session.finish()?;
        }
        // Final synchronization can cross the request deadline too. An output
        // limit reached by that work must not authorize client continuation.
        if Instant::now() >= self.ready.deadline {
            return self.cancel(RequestError::Deadline, limits);
        }
        let reason = http::finish_reason(if self.stop.matched { FinishReason::Stop } else { reason }, self.tool_index);
        let echo = self.sampling_echo();
        if let Some(output) = &mut self.ready.output {
            if self.ready.request.stream {
                let events = output.finish(&self.ready.model, self.id, self.created, reason,
                    self.generation.usage, self.stop.matched_sequence.as_deref(), &echo);
                self.ready.wire.terminal_events(events, limits)?;
            } else {
                let body = output.json(&self.ready.model, self.id, self.created, reason, self.generation.usage, self.stop.matched_sequence.as_deref(), &echo);
                self.ready.wire.response(200, &body, limits)?;
            }
        } else if self.ready.request.stream {
            self.chunk(if self.ready.request.chat { "{}" } else { "" }, None, Some(reason), limits)?;
            self.ready.wire.push("data: [DONE]\n\n", limits, true)?;
        } else {
            let body = self.response.json(&self.ready.model, self.ready.request.chat, self.id,
                self.created, reason, self.generation.usage, &echo);
            self.ready.wire.response(200, &body, limits)?;
        }
        self.done = true;
        if !self.terminal_logged {
            self.terminal_logged = true;
            eprintln!("request_completed id={} wall_ms={:.3} admitted_ms={:.3} prompt_tokens={} output_tokens={} cached_tokens={} vade={}",
                self.id, self.ready.wire.received.elapsed().as_secs_f64() * 1000.0,
                self.admitted.elapsed().as_secs_f64() * 1000.0, self.generation.usage.prompt_tokens,
                self.generation.usage.completion_tokens, self.generation.usage.cached_tokens, echo.json());
        }
        Ok(())
    }
    fn cancelled(&mut self, code: &'static str) {
        if self.terminal_logged { return; }
        self.terminal_logged = true;
        self.update_usage();
        eprintln!("request_cancelled id={} code={code} vade={}", self.id, self.sampling_echo().json());
    }
    fn cancel(&mut self, error: RequestError, limits: &Limits) -> Result<(), String> {
        // Keep emitted/buffered deltas, but do not finish the decoder or parser:
        // interrupted tool proposals must never become successful completions.
        self.key = None;
        let message = error.cancellation_message();
        let result = self.error(error, message, limits);
        self.done = true;
        result?;
        Ok(())
    }
    fn flush(&mut self, limits: &Limits) -> Result<(), String> {
        if self.ready.request.stream { self.ready.wire.heartbeat(limits) }
        else { self.ready.wire.flush() }
    }
    fn error(&mut self, error: RequestError, message: &str, limits: &Limits) -> Result<(), String> {
        // A terminal result may already be buffered when its final flush fails.
        if self.done { return Ok(()); }
        if let Some(kv) = &mut self.generation.usage.kv { kv.finish(); }
        self.update_usage();
        let echo = self.sampling_echo();
        let event = if matches!(error, RequestError::Swapping | RequestError::Shutdown | RequestError::Cancelled | RequestError::Deadline) {
            "request_cancelled"
        } else { "request_failed" };
        if !self.terminal_logged {
            self.terminal_logged = true;
            eprintln!("{event} id={} code={} vade={}", self.id, error.code(), echo.json());
        }
        if let Some(output) = &mut self.ready.output {
            return if self.ready.request.stream {
                let events = output.failed(&self.ready.model, self.id, self.created,
                    self.generation.usage, error.status(), error.code(), message, true, &echo);
                self.ready.wire.terminal_events(events, limits)
            } else {
                self.ready.wire.response(error.status(), &output.failed(&self.ready.model, self.id, self.created, self.generation.usage, error.status(), error.code(), message, false, &echo), limits)
            };
        }
        let body = echo.attach(protocol::error_json(error.code(), message));
        if self.ready.request.stream {
            let usage = self.generation.usage.json(false);
            self.ready.wire.terminal_events(format!("data: {body}\n\ndata: {{\"choices\":[],\"usage\":{usage}}}\n\ndata: [DONE]\n\n"), limits)
        } else { self.ready.wire.response(error.status(), &body, limits) }
    }
    fn events(
        &mut self,
        engine: &'e E,
        limits: &Limits,
        cache: &mut SessionCache<E::Session<'e>>,
        events: Vec<Event>,
    ) -> Result<(), String> {
        // A model step can expire while CUDA/peers are completing. Check again
        // before publishing its events, including a final length completion.
        if Instant::now() >= self.ready.deadline {
            return self.cancel(RequestError::Deadline, limits);
        }
        if self.generation.state.tokens.len() >= self.ready.prompt.len() {
            if let Some(clock) = &mut self.usage_clock { clock.prefilled(); }
        }
        self.update_usage();
        let tokens = self.generation.usage.completion_tokens;
        if tokens / 1024 > self.memory_reported_tokens / 1024 {
            self.memory_reported_tokens = tokens;
            if let (Some((current, peak, pinned)), Some((kv, _))) =
                (engine.memory_bytes(), self.generation.state.session.memory_bytes())
            {
                eprintln!("request_progress completion_tokens={tokens} elapsed_secs={:.3} kv_bytes={kv} device_bytes={current} peak_device_bytes={peak} pinned_host_bytes={pinned}",
                    self.admitted.elapsed().as_secs_f64());
            }
        }
        {
            let mut prefill = false;
            for (index, event) in events.iter().copied().enumerate() { match event {
                Event::Prefill => {
                    prefill = true;
                    self.checkpoint(cache);
                    let state = &self.generation.state;
                    if let Some(key) = self.key {
                        if engine
                            .eos_token_ids()
                            .contains(state.tokens.last().unwrap())
                            || state.tokens.len() == self.ready.prompt.len()
                        {
                            cache.fund_pages(state.session.prefix_allocations(state.tokens.len()).unwrap_or_default());
                            cache.snapshot_scoped(key, self.ready.scope.as_deref(), state, Instant::now(),
                                state.tokens.len() == self.ready.prompt.len());
                        }
                    }
                }
                Event::Token(token) => {
                    if !self.first_token_generated {
                        self.first_token_generated = true;
                        if self.profile {
                            eprintln!("request_first_generated_profile id={} ready_ms={:.3}",
                                self.id, self.ready.wire.received.elapsed().as_secs_f64()*1000.0);
                        }
                    }
                    let text = match &mut self.decoder {
                        Some(d) => d.push(token)?,
                        None => String::new(),
                    };
                    self.text(&text, Some(token), limits)?;
                    if self.ready.request.stream && !self.first_token_sent {
                        let written = self.ready.wire.last_write;
                        self.ready.wire.flush().inspect_err(|_| self.cancelled("write_failed"))?;
                        self.first_token_sent = self.ready.wire.last_write > written;
                        if self.profile && self.first_token_sent {
                            eprintln!("request_first_token_profile id={} ready_ms={:.3} first_byte_ms={:.3} device_bytes={}",
                                self.id, self.ready.wire.received.elapsed().as_secs_f64()*1000.0,
                                self.ready.wire.last_write.saturating_duration_since(self.ready.wire.received).as_secs_f64()*1000.0,
                                engine.memory_bytes().map_or(0, |m| m.0));
                        }
                    }
                    if self.stop.matched || self.ready.parser.as_ref().is_some_and(|p| p.stopped()) {
                        let extra = events[index + 1..].iter().filter(|e|
                            matches!(e, Event::Token(_) | Event::Finished(FinishReason::Stop, _))).count();
                        self.generation.usage.completion_tokens -= extra;
                        self.finish(FinishReason::Stop, limits)?;
                        return self.ready.wire.flush();
                    }
                }
                Event::Finished(reason, _) => {
                    self.finish(reason, limits)?;
                    return self.ready.wire.flush();
                }
            } }
            if !prefill { return self.ready.wire.flush().inspect_err(|_| self.cancelled("write_failed")); }
        }
        if self.ready.request.stream { self.ready.wire.push(": prefill\n\n", limits, false)?; }
        self.ready.wire.flush().inspect_err(|_| self.cancelled("write_failed"))
    }
}

/// Higher priority preempts at the next token/prefill slice; equal priorities rotate.
/// At least three slots and an owner key enable one reserved owner slot.
pub fn serve<'e, E: Engine, T: Tokenizer>(
    listener: TcpListener,
    engine: &'e E,
    tokenizer: Option<&T>,
    limits: &Limits,
    cache_limits: &CacheLimits,
    config: &Config,
    shutdown: &AtomicBool,
) -> Result<(), String> {
    serve_inner(&listener, engine, tokenizer, limits, cache_limits, config, shutdown, None, crate::MODEL_ID, 0).map(|_| ())
}
pub fn serve_model<'e, E: Engine, T: Tokenizer>(
    listener: &TcpListener,
    engine: &'e E,
    tokenizer: Option<&T>,
    limits: &Limits,
    cache_limits: &CacheLimits,
    config: &Config,
    shutdown: &AtomicBool,
    admin: &ModelAdmin,
) -> Result<Option<Swap>, String> {
    serve_inner(listener, engine, tokenizer, limits, cache_limits, config, shutdown, Some(admin), &admin.model, 0)
}
/// Drain after accepting this many connections; zero serves continuously.
pub fn serve_requests<'e, E: Engine, T: Tokenizer>(
    listener: TcpListener, engine: &'e E, tokenizer: Option<&T>, limits: &Limits,
    cache_limits: &CacheLimits, config: &Config, shutdown: &AtomicBool, model: &str, requests: usize,
) -> Result<(), String> {
    serve_inner(&listener, engine, tokenizer, limits, cache_limits, config, shutdown, None, model, requests).map(|_| ())
}
fn serve_inner<'e, E: Engine, T: Tokenizer>(
    listener: &TcpListener,
    engine: &'e E,
    tokenizer: Option<&T>,
    limits: &Limits,
    cache_limits: &CacheLimits,
    config: &Config,
    shutdown: &AtomicBool,
    admin: Option<&ModelAdmin>,
    model: &str,
    requests: usize,
) -> Result<Option<Swap>, String> {
    let mut ab_config;
    let mut ab_cache;
    let (config, cache_limits) = if limits.ab_ptx {
        ab_config = config.clone(); ab_config.prefix_cache = false; ab_config.draft_global_bank = false;
        // Cohorts keep the pool width; admission below runs one arm at a time.
        if !limits.ab_cohorts { ab_config.active = 1; }
        ab_cache = cache_limits.clone(); ab_cache.retained_bytes = 0;
        (&ab_config, &ab_cache)
    } else { (config, cache_limits) };
    protocol::SamplingEcho::log_defaults(limits, config.draft, config.mtp_depth);
    engine_model::events::note("scheduler_pool", format_args!("active={} connections={} active_bytes={} session_budget_bytes={} retained_cache_bytes={} prefix_cache={} draft={} mtp_depth={}",
        config.active, config.connections, config.active_bytes, config.session_budget_bytes, cache_limits.retained_bytes, config.prefix_cache, config.draft.name(), config.mtp_depth));
    let (prepare_tx, input_rx) = mpsc::sync_channel(config.connections.max(1));
    let (output_tx, prepare_rx) = mpsc::channel();
    let (context, vocab, eos) = (engine.context_limit(), engine.vocab_size(), engine.eos_token_ids());
    let scheduler_thread = std::thread::current();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            // Parsing and tokenizing a long prompt sits on its first-token path.
            #[cfg(target_os = "linux")]
            if let Err(error) = spark_gpu_probe::driver::prefer_fast_cores() { eprintln!("request preparation: {error}"); }
            while let Ok((request, wire)) = input_rx.recv() {
                let ready = prepare(request, wire, tokenizer, limits, model, context, vocab, eos);
                if output_tx.send(ready).is_err() { break; }
                scheduler_thread.unpark();
            }
        });
        // Inference may wait for the next packed prompt slice while its peers
        // enter a collective. Keep that work on its own persistent worker: HTTP
        // parsing/tokenization must never queue ahead of it. Only immutable CPU
        // packing moves here; GPU work and collectives stay on the engine thread.
        let result = spark_gpu_probe::host_prep::with_worker(||
            serve_loop(listener, engine, tokenizer, limits, cache_limits, config,
            shutdown, admin, model, requests, &prepare_tx, &prepare_rx));
        drop(prepare_tx);
        result?
    })
}
#[cfg(test)] #[test] fn cache_yields_held_device_bytes_without_refusing_admission() {
    #[derive(Clone)] struct Page(std::rc::Rc<[u8; 1024]>);
    impl Session for Page {
        fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
        fn fork(&self) -> Result<Self, String> { Ok(self.clone()) }
        fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(vec![(std::rc::Rc::as_ptr(&self.0) as usize, 1024)]) }
    }
    let now = Instant::now();
    let mut cache = SessionCache::new(CacheLimits { entries: 4, retained_bytes: 1 << 20, tokens_per_entry: 16,
        logits_per_entry: 16, idle_ttl: Duration::from_secs(60) }).unwrap();
    let state = |reserved| CachedSession { checkpoint_lease: None, session: Page(std::rc::Rc::new([0; 1024])),
        tokens: vec![1, 2], next_logits: vec![1., 2.], reserved_session_bytes: reserved };
    cache.snapshot_scoped(SessionKey::generate().unwrap(), None, &state(4096), now, true);
    let retained = cache.retained_bytes();
    assert!(cache.len() == 1 && retained >= 1024);
    let live = state(4096);
    // Without a hold the unfunded entry fits beside the live reservation.
    assert_eq!(balance_cache(&mut cache, 4096 + retained, std::iter::once(&live), 0, |_| Ok(None)).unwrap(), 4096);
    assert_eq!(cache.len(), 1);
    // Held bytes (elastic captures, unreserved live state) evict it; admission still holds.
    assert_eq!(balance_cache(&mut cache, 4096 + retained, std::iter::once(&live), 1, |_| Ok(None)).unwrap(), 4096);
    assert_eq!((cache.len(), cache.retained_bytes()), (0, 0));
    // A hold above the whole slack never turns into a refusal.
    assert_eq!(balance_cache(&mut cache, 4096, std::iter::once(&live), usize::MAX, |_| Ok(None)).unwrap(), 4096);
    // Variant (b): the engine's physical room bounds retention below the pool's share.
    balance_cache(&mut cache, 4096 + (1 << 19), std::iter::once(&live), 0, |_| Ok(None)).unwrap();
    cache.snapshot_scoped(SessionKey::generate().unwrap(), None, &state(4096), now, true);
    let retained = cache.retained_bytes();
    assert_eq!(balance_cache(&mut cache, 4096 + (1 << 19), std::iter::once(&live), 0, |reserved| {
        assert_eq!(reserved, 4096); Ok(Some((0, 0))) }).unwrap(), 4096);
    assert_eq!((cache.len(), cache.retained_bytes()), (1, retained));
    // Inside the growth slack nothing yields (hysteresis), but nothing more is retained.
    assert_eq!(balance_cache(&mut cache, 4096 + (1 << 19), std::iter::once(&live), 0, |_| Ok(Some((0, 1 << 30)))).unwrap(), 4096);
    assert_eq!((cache.len(), cache.retained_bytes()), (1, retained));
    cache.snapshot_scoped(SessionKey::generate().unwrap(), None, &state(4096), now, true);
    assert_eq!((cache.len(), cache.retained_bytes()), (1, retained));
    // One byte short of the spare pages' room: the entry yields; admission is unchanged.
    assert_eq!(balance_cache(&mut cache, 4096 + (1 << 19), std::iter::once(&live), 0, |_| Ok(Some((-1, 0)))).unwrap(), 4096);
    assert_eq!((cache.len(), cache.retained_bytes()), (0, 0));
}
