//! One MCP server process: JSON-RPC 2.0, one message per line on stdin and stdout.
use super::config::ServerSpec;
use crate::json::{self, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The protocol revision this client asks for.
pub const PROTOCOL_VERSION: &str = "2025-06-18";
/// Earlier revisions whose tool calls have the same shape.
pub const ACCEPTED_VERSIONS: [&str; 3] = [PROTOCOL_VERSION, "2025-03-26", "2024-11-05"];
/// One incoming message may not exceed the JSON parser limit (8 MiB).
pub const MAX_MESSAGE_BYTES: usize = json::MAX_INPUT_BYTES;
/// Progress notifications extend a call to at most this many timeouts.
pub const PROGRESS_EXTENSION: u32 = 10;
const STDERR_TAIL_BYTES: usize = 2048;
const MAX_LIST_PAGES: usize = 16;
const POLL: Duration = Duration::from_millis(25);
const EXIT_GRACE: Duration = Duration::from_millis(500);
/// Messages that wait for the writer thread. A server that stops reading its
/// input fills this queue; the next send then stops the server.
const WRITE_QUEUE: usize = 16;
/// Messages read from the server and not yet handled. With 8 MiB per message,
/// at most 9 messages (72 MiB) wait for a server that writes without a request.
const READ_QUEUE: usize = 8;

/// Why a request produced no result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallError {
    Spawn(String),
    Timeout,
    Cancelled,
    /// The server closed stdout or exited; the text is the end of its stderr.
    Exited(String),
    /// The server stopped reading its input, so the client stopped it.
    Stalled,
    Malformed(String),
    Rpc(i64, String),
    Protocol(String),
}
impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(error) => write!(f, "cannot start the server: {error}"),
            Self::Timeout => f.write_str("the server did not answer in time"),
            Self::Cancelled => f.write_str("canceled"),
            Self::Exited(tail) if tail.is_empty() => f.write_str("the server stopped"),
            Self::Exited(tail) => write!(f, "the server stopped; its last error output: {tail}"),
            Self::Stalled => f.write_str("the server stopped reading its input, so Vade stopped it"),
            Self::Malformed(why) => write!(f, "the server sent malformed JSON-RPC: {why}"),
            Self::Rpc(code, message) => write!(f, "the server returned error {code}: {message}"),
            Self::Protocol(why) => f.write_str(why),
        }
    }
}

/// What `initialize` returned.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub protocol: String,
    pub tools: bool,
    pub resources: bool,
    pub prompts: bool,
}

/// One `notifications/progress` for the running call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Progress {
    pub progress: String,
    pub total: Option<String>,
    pub message: Option<String>,
}

enum Incoming {
    Line(Vec<u8>),
    TooLong,
    Closed,
}

/// A started and initialized server. Dropping it closes stdin, waits a short
/// time for stdout to close and then stops the server's process group. A
/// writer thread owns stdin, so a server that stops reading cannot block the
/// timeout and the cancel flag. The direct child is reaped only after its
/// group was stopped: until then its pid stays the group id, so a wrapper
/// that exits first cannot leave its server child running.
pub struct Connection {
    child: Child,
    writer: Option<SyncSender<Vec<u8>>>,
    /// Messages given to the writer thread, and messages it wrote.
    queued: Arc<AtomicU64>,
    written: Arc<AtomicU64>,
    incoming: Receiver<Incoming>,
    /// Set when stdout closes: every process that held it is gone.
    closed: Arc<AtomicBool>,
    /// Recent stderr with configured env values already replaced.
    stderr: Arc<Mutex<Redactor>>,
    next_id: u64,
    dead: bool,
    /// Vade stopped the server at a call's time limit; that is not a crash.
    pub timed_out: bool,
    pub info: ServerInfo,
}

impl Connection {
    /// Spawn the server and run `initialize` / `notifications/initialized`.
    pub fn start(spec: &ServerSpec, root: &Path, cancel: &AtomicBool) -> Result<Self, CallError> {
        // Only ordinary runtime names pass from the chat; a server gets a
        // token only through its own `env` entry.
        const KEEP: &[&str] = &["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM"];
        // The folder must resolve inside the workspace, also through links.
        let inside = |dir: &Path| Some(std::fs::canonicalize(dir).ok()?.starts_with(std::fs::canonicalize(root).ok()?));
        let directory = spec.directory(root);
        if inside(&directory) != Some(true) {
            return Err(CallError::Spawn("cwd is not a folder inside the workspace".into()));
        }
        let mut command = Command::new(&spec.command);
        command
            .args(&spec.args)
            .env_clear()
            .envs(std::env::vars_os().filter(|(name, _)| name.to_str().is_some_and(|name| KEEP.contains(&name))))
            .envs(spec.env.iter().map(|(k, v)| (k, v)))
            .current_dir(directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            // Own process group: a Ctrl-C in the chat terminal does not reach the server.
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command.spawn().map_err(|e| CallError::Spawn(e.kind().to_string()))?;
        let stdout = child.stdout.take().expect("piped stdout");
        let stdin = child.stdin.take().expect("piped stdin");
        let (writer, lines) = mpsc::sync_channel(WRITE_QUEUE);
        let written = Arc::new(AtomicU64::new(0));
        let count = Arc::clone(&written);
        std::thread::spawn(move || write_lines(stdin, &lines, &count));
        let stderr_pipe = child.stderr.take().expect("piped stderr");
        // Bounded: a server that writes while nobody reads blocks on its stdout.
        let (tx, incoming) = mpsc::sync_channel(READ_QUEUE);
        let closed = Arc::new(AtomicBool::new(false));
        let eof = Arc::clone(&closed);
        std::thread::spawn(move || read_lines(stdout, tx, &eof));
        let secrets = spec.env.iter().filter(|(_, value)| value.chars().count() >= 4).map(|(_, value)| value.as_bytes().to_vec()).collect();
        let stderr = Arc::new(Mutex::new(Redactor::new(secrets)));
        let tail = Arc::clone(&stderr);
        std::thread::spawn(move || keep_tail(stderr_pipe, &tail));
        let queued = Arc::new(AtomicU64::new(0));
        let mut connection = Self { writer: Some(writer), queued, written, child, incoming, closed, stderr, next_id: 1, dead: false, timed_out: false, info: ServerInfo::default() };
        let params = json::parse(&format!(
            r#"{{"protocolVersion":"{PROTOCOL_VERSION}","capabilities":{{}},"clientInfo":{{"name":"vade","version":"{}"}}}}"#,
            env!("CARGO_PKG_VERSION")
        ))
        .expect("static initialize params");
        let result = connection.request("initialize", params, spec.timeout, cancel, None)?;
        let protocol = result.get("protocolVersion").and_then(Value::as_str).unwrap_or_default();
        if !ACCEPTED_VERSIONS.contains(&protocol) {
            return Err(CallError::Protocol(format!(
                "the server speaks MCP {protocol:?}; this client supports {}",
                ACCEPTED_VERSIONS.join(", ")
            )));
        }
        let text = |value: Option<&Value>, key| value.and_then(|v| v.get(key)).and_then(Value::as_str).unwrap_or_default().to_owned();
        let capabilities = result.get("capabilities");
        let has = |key| capabilities.and_then(|c| c.get(key)).is_some();
        connection.info = ServerInfo {
            name: text(result.get("serverInfo"), "name"),
            version: text(result.get("serverInfo"), "version"),
            protocol: protocol.into(),
            tools: has("tools"),
            resources: has("resources"),
            prompts: has("prompts"),
        };
        connection.notify("notifications/initialized", None)?;
        Ok(connection)
    }

    /// False once stdout closed, a message broke the framing, or Vade
    /// stopped the server. It never reaps the process (see [`Connection`]).
    pub fn alive(&mut self) -> bool {
        !self.dead && !self.closed.load(Ordering::Acquire)
    }

    /// The last bytes the server wrote to stderr, as one line.
    pub fn stderr_tail(&self) -> String {
        self.stderr.lock().map(|tail| tail.text()).unwrap_or_default()
    }

    /// Messages handed to the writer thread so far. Tests use it to see that
    /// a request was sent.
    pub fn queued(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.queued)
    }

    /// Every page of a `*/list` method, capped at 16 pages and `max` items.
    pub fn list(&mut self, method: &str, key: &str, max: usize, timeout: Duration, cancel: &AtomicBool) -> Result<Vec<Value>, CallError> {
        let mut items = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let params = match &cursor {
                Some(cursor) => Value::Object(vec![("cursor".into(), Value::String(cursor.clone()))]),
                None => Value::Null,
            };
            let page = self.request(method, params, timeout, cancel, None)?;
            let Some(Value::Array(found)) = page.get(key) else {
                return Err(CallError::Malformed(format!("{method} result has no {key} array")));
            };
            items.extend(found.iter().take(max.saturating_sub(items.len())).cloned());
            cursor = page.get("nextCursor").and_then(Value::as_str).map(str::to_owned);
            if cursor.is_none() || items.len() >= max {
                break;
            }
        }
        Ok(items)
    }

    /// `tools/call` with a progress token. Progress resets the timeout, up to
    /// [`PROGRESS_EXTENSION`] timeouts in total.
    pub fn call_tool(&mut self, tool: &str, arguments: &Value, timeout: Duration, cancel: &AtomicBool, progress: &mut dyn FnMut(Progress)) -> Result<Value, CallError> {
        let token = format!("vade-{}", self.next_id);
        let params = Value::Object(vec![
            ("name".into(), Value::String(tool.into())),
            ("arguments".into(), arguments.clone()),
            ("_meta".into(), Value::Object(vec![("progressToken".into(), Value::String(token))])),
        ]);
        self.request("tools/call", params, timeout, cancel, Some(progress))
    }

    /// Send one request and wait for its response. Server requests and
    /// notifications that arrive first are handled in place. Only a request
    /// with `progress` (a tool call) lets progress notes extend its time.
    pub fn request(&mut self, method: &str, params: Value, timeout: Duration, cancel: &AtomicBool, mut progress: Option<&mut dyn FnMut(Progress)>) -> Result<Value, CallError> {
        let id = self.next_id;
        self.next_id += 1;
        let mut fields = vec![
            ("jsonrpc".into(), Value::String("2.0".into())),
            ("id".into(), Value::Number(id.to_string())),
            ("method".into(), Value::String(method.into())),
        ];
        if params != Value::Null {
            fields.push(("params".into(), params));
        }
        self.send(&Value::Object(fields))?;
        let token = format!("vade-{id}");
        let started = Instant::now();
        let hard = started + timeout.saturating_mul(PROGRESS_EXTENSION);
        let mut deadline = started + timeout;
        loop {
            if cancel.load(Ordering::Relaxed) {
                self.abandon(id, "canceled by the user");
                return Err(CallError::Cancelled);
            }
            let now = Instant::now();
            if now >= deadline {
                // Stop the server so the call does not run on past its time
                // limit; the next call starts it again.
                self.abandon(id, "timed out");
                self.stop();
                self.timed_out = true;
                return Err(CallError::Timeout);
            }
            let line = match self.incoming.recv_timeout((deadline - now).min(POLL)) {
                Err(RecvTimeoutError::Timeout) => continue,
                Ok(Incoming::Line(line)) => line,
                Ok(Incoming::TooLong) => {
                    self.dead = true;
                    return Err(CallError::Malformed(format!("a message is larger than {MAX_MESSAGE_BYTES} bytes")));
                }
                Ok(Incoming::Closed) | Err(RecvTimeoutError::Disconnected) => {
                    self.dead = true;
                    // Let the stderr reader catch up before quoting it.
                    std::thread::sleep(POLL);
                    return Err(CallError::Exited(self.stderr_tail()));
                }
            };
            // Some servers print a banner on stdout. A line that cannot start
            // a JSON object is not a message; skip it, whatever its encoding.
            if line.trim_ascii_start().first() != Some(&b'{') {
                continue;
            }
            let Ok(line) = std::str::from_utf8(&line) else {
                return Err(CallError::Malformed("a line is not UTF-8".into()));
            };
            let line = line.trim();
            let message = match json::parse(line) {
                Ok(message @ Value::Object(_)) => message,
                _ => return Err(CallError::Malformed("a line is not a JSON object".into())),
            };
            match (message.get("method").and_then(Value::as_str), message.get("id")) {
                (Some(method), Some(request)) => self.answer(method, request.clone())?,
                (Some("notifications/progress"), None) => {
                    let params = message.get("params");
                    let field = |key| params.and_then(|p| p.get(key));
                    if let Some(progress) = progress.as_mut()
                        && field("progressToken").and_then(Value::as_str) == Some(&token)
                    {
                        deadline = (Instant::now() + timeout).min(hard);
                        let number = |key| match field(key) { Some(Value::Number(n)) => Some(n.clone()), _ => None };
                        progress(Progress {
                            progress: number("progress").unwrap_or_default(),
                            total: number("total"),
                            message: field("message").and_then(Value::as_str).map(str::to_owned),
                        });
                    }
                }
                (Some(_), None) => {} // logging and list-changed notifications
                (None, Some(Value::Number(n))) if *n == id.to_string() => {
                    if let Some(result) = message.get("result") {
                        return Ok(result.clone());
                    }
                    let error = message.get("error");
                    let code = match error.and_then(|e| e.get("code")) { Some(Value::Number(n)) => n.parse().unwrap_or(0), _ => 0 };
                    let text = error.and_then(|e| e.get("message")).and_then(Value::as_str).unwrap_or("no message");
                    return Err(CallError::Rpc(code, text.chars().take(512).collect()));
                }
                (None, Some(_)) => {} // a late answer to an earlier, abandoned request
                (None, None) => return Err(CallError::Malformed("a message has neither method nor id".into())),
            }
        }
    }

    /// Answer a server request: `ping` succeeds, everything else is unknown
    /// because this client offers no sampling, roots or elicitation.
    fn answer(&mut self, method: &str, id: Value) -> Result<(), CallError> {
        let mut fields = vec![("jsonrpc".into(), Value::String("2.0".into())), ("id".into(), id)];
        if method == "ping" {
            fields.push(("result".into(), Value::Object(Vec::new())));
        } else {
            fields.push(("error".into(), Value::Object(vec![
                ("code".into(), Value::Number("-32601".into())),
                ("message".into(), Value::String("Method not found".into())),
            ])));
        }
        self.send(&Value::Object(fields))
    }

    /// Tell the server to drop request `id`. When the writer thread still
    /// holds unwritten messages, the server does not read its input: stop it,
    /// so that the next call starts it again.
    fn abandon(&mut self, id: u64, reason: &str) {
        if self.written.load(Ordering::Acquire) < self.queued.load(Ordering::Relaxed) {
            self.stop();
            return;
        }
        let params = Value::Object(vec![
            ("requestId".into(), Value::Number(id.to_string())),
            ("reason".into(), Value::String(reason.into())),
        ]);
        let _ = self.notify("notifications/cancelled", Some(params));
    }

    fn notify(&mut self, method: &str, params: Option<Value>) -> Result<(), CallError> {
        let mut fields = vec![("jsonrpc".into(), Value::String("2.0".into())), ("method".into(), Value::String(method.into()))];
        fields.extend(params.map(|params| ("params".into(), params)));
        self.send(&Value::Object(fields))
    }

    fn send(&mut self, message: &Value) -> Result<(), CallError> {
        let mut line = json::stringify(message).map_err(|e| CallError::Protocol(format!("cannot encode the request: {e}")))?;
        line.push('\n');
        let sent = match &self.writer {
            Some(writer) => writer.try_send(line.into_bytes()),
            None => Err(TrySendError::Disconnected(Vec::new())),
        };
        match sent {
            Ok(()) => {
                self.queued.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                self.stop();
                Err(CallError::Stalled)
            }
            // The writer thread ends when a write fails: the server closed its input.
            Err(TrySendError::Disconnected(_)) => {
                self.dead = true;
                Err(CallError::Exited(self.stderr_tail()))
            }
        }
    }

    /// Kill the server now. A blocked write then fails and the writer thread ends.
    fn stop(&mut self) {
        self.dead = true;
        self.kill();
    }

    /// Stop the server's process group, so a wrapper's child that holds the
    /// pipes stops too. The direct child is not reaped yet, so its pid is
    /// still the group id.
    fn kill(&mut self) {
        crate::process_group::kill_group(&mut self.child);
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // The stdio shutdown order: close input, wait, then stop the process.
        // The writer thread closes stdin after the messages it still holds.
        drop(self.writer.take());
        let end = Instant::now() + EXIT_GRACE;
        while Instant::now() < end && !self.closed.load(Ordering::Acquire) {
            // Read what is left, so the stdout reader is not blocked on a full queue.
            // A reader that already ended (an oversized message) leaves nothing to wait for.
            if let Err(RecvTimeoutError::Disconnected) = self.incoming.recv_timeout(Duration::from_millis(10)) {
                break;
            }
        }
        // Also when stdout closed: a process of the group may live without it.
        self.kill();
        let _ = self.child.wait();
    }
}

fn write_lines(mut stdin: ChildStdin, lines: &Receiver<Vec<u8>>, written: &AtomicU64) {
    for line in lines {
        if stdin.write_all(&line).and_then(|()| stdin.flush()).is_err() {
            return;
        }
        written.fetch_add(1, Ordering::Release);
    }
}

fn read_lines(stdout: impl Read, tx: SyncSender<Incoming>, closed: &AtomicBool) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut line = Vec::new();
        match (&mut reader).take(MAX_MESSAGE_BYTES as u64 + 1).read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) if line.len() > MAX_MESSAGE_BYTES => {
                let _ = tx.send(Incoming::TooLong);
                return;
            }
            Ok(_) => {
                if tx.send(Incoming::Line(line)).is_err() {
                    return;
                }
            }
        }
    }
    closed.store(true, Ordering::Release);
    let _ = tx.send(Incoming::Closed);
}

fn keep_tail(mut stderr: impl Read, tail: &Mutex<Redactor>) {
    let mut buffer = [0; 1024];
    while let Ok(read @ 1..) = stderr.read(&mut buffer) {
        let Ok(mut tail) = tail.lock() else { return };
        tail.push(&buffer[..read]);
    }
}

/// Recent stderr with configured env values hidden as the bytes arrive.
/// Vade finds every match over the original bytes, also a value that starts
/// inside another match, and shows one marker for each run of bytes that a
/// match covers. A byte is shown only when no match that may cover it is
/// still open, so neither a cut nor a read boundary can split a value, and
/// undecided bytes are never shown.
pub(crate) struct Redactor {
    /// Values of 4 or more characters.
    secrets: Vec<Vec<u8>>,
    /// Shown bytes, the last [`STDERR_TAIL_BYTES`] of them.
    shown: Vec<u8>,
    /// Original bytes: the last shown ones (one less than the longest value,
    /// to find matches that started there), then the undecided ones.
    raw: Vec<u8>,
    /// How many bytes at the start of `raw` are already shown.
    decided: usize,
    /// The last shown byte was inside a match, so its marker is out.
    in_match: bool,
    cut: bool,
}
impl Redactor {
    pub(crate) fn new(secrets: Vec<Vec<u8>>) -> Self {
        Self { secrets, shown: Vec::new(), raw: Vec::new(), decided: 0, in_match: false, cut: false }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.raw.extend_from_slice(bytes);
        let raw = &self.raw;
        // Bytes from the first match start that the next bytes may still
        // complete wait. Shown bytes never have one (see `decided`).
        let open = (0..raw.len())
            .find(|&at| self.secrets.iter().any(|secret| raw.len() - at < secret.len() && secret.starts_with(&raw[at..])))
            .unwrap_or(raw.len());
        let mut covered = vec![false; open];
        for at in 0..open {
            for secret in self.secrets.iter().filter(|secret| raw[at..].starts_with(secret)) {
                covered[at..(at + secret.len()).min(open)].fill(true);
            }
        }
        for at in self.decided..open {
            if !covered[at] {
                self.shown.push(raw[at]);
            } else if !self.in_match {
                self.shown.extend_from_slice(b"[env value]");
            }
            self.in_match = covered[at];
        }
        let context = self.secrets.iter().map(Vec::len).max().unwrap_or(1) - 1;
        let drop = open.saturating_sub(context);
        self.raw.drain(..drop);
        self.decided = open - drop;
        let excess = self.shown.len().saturating_sub(STDERR_TAIL_BYTES);
        if excess > 0 {
            self.shown.drain(..excess);
            self.cut = true;
        }
    }

    /// The decided bytes as one line.
    pub(crate) fn text(&self) -> String {
        let text = String::from_utf8_lossy(&self.shown);
        // A cut tail starts inside older output; drop that first, partial word.
        text.split_whitespace().skip(usize::from(self.cut)).collect::<Vec<_>>().join(" ")
    }
}
