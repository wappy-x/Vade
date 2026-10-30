//! Optional installed Codex app-server adapter, implemented with Rust std only.
//! Protocol source and pinned subset: docs/bridge-contract.md. No installed
//! agent or schema generator is needed for verification; fixtures are first-party.
//! The harness reads no credential files and never approves server tool requests.
mod child;
#[cfg(unix)]
mod connection;
mod contract;
pub mod hermes;
use crate::json::{self, Value, invalid};
use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver},
};
use std::time::{Duration, Instant};

const MAX_LINE: usize = 1024 * 1024;
const MAX_PENDING: usize = 32;
const MAX_PENDING_BYTES: usize = 4 * 1024 * 1024;
const MAX_TURN_BYTES: usize = 64 * 1024 * 1024;
const MAX_DELTA_BYTES: usize = 8 * 1024 * 1024;
const POLL: Duration = Duration::from_millis(50);

/// The sandbox preset the Codex app-server thread runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermissionPolicy {
    ReadOnly,
    WorkspaceWrite,
}

/// How to start the installed Codex app-server: executable, workspace, an
/// optional thread to resume, model override and permission preset.
#[derive(Clone, Debug)]
pub struct Options {
    pub executable: PathBuf,
    pub workspace: PathBuf,
    pub thread_id: Option<String>,
    pub model: Option<String>,
    pub permission: PermissionPolicy,
}

/// What the adapter reports while a turn runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// Provisional display only. Never use these bytes as completed history.
    AgentMessageDelta(String),
    AgentMessageCompleted {
        item_id: String,
        text: String,
    },
    Phase(Phase),
    Steering {
        request_id: u64,
        status: SteeringStatus,
    },
    Notice(String),
}

/// A request the host sends into a running turn.
#[derive(Clone, Debug)]
pub enum Control {
    Steer { request_id: u64, text: String },
}

/// What the app-server said about one steering request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteeringStatus {
    Accepted,
    Rejected,
    Unconfirmed,
}

/// The visible activity of a running turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Generating,
    ToolRunning,
    WaitingForApproval,
    Compacting,
    Cancelling,
}

/// How a turn ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurnOutcome {
    Completed,
    Interrupted,
    /// Cancelled before any turn request was sent. No remote stop is implied.
    NotStarted,
    Failed(String),
}

/// Whether shutdown reaped the child process and joined its I/O workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cleanup {
    pub child_reaped: bool,
    pub io_workers_joined: bool,
}

#[derive(Clone, Copy)]
struct Limits {
    rpc: Duration,
    turn: Duration,
    idle: Duration,
    interrupt: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            rpc: Duration::from_secs(20),
            turn: Duration::from_secs(300),
            idle: Duration::from_secs(60),
            interrupt: Duration::from_secs(5),
        }
    }
}

/// A connection to one installed Codex app-server child and its thread.
pub struct Client {
    wire: child::Connection,
    thread_id: String,
    workspace: String,
    permission: PermissionPolicy,
    model: Option<String>,
    next_id: u64,
    limits: Limits,
    completed_text: Option<String>,
}

fn timed_out(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, message)
}
fn object(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(
        fields
            .into_iter()
            .map(|(name, value)| (name.into(), value))
            .collect(),
    )
}
fn string(text: &str) -> Value {
    Value::String(text.to_owned())
}
fn id_value(id: u64) -> Value {
    Value::Number(id.to_string())
}
fn matches_id(value: &Value, id: u64) -> bool {
    matches!(value.get("id"), Some(Value::Number(number)) if number == &id.to_string())
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 512 && !id.chars().any(|c| c.is_control() || c.is_whitespace())
}
fn error_message(value: &Value) -> String {
    // Only the public error message is surfaced, never raw stderr, error data,
    // request bodies, authentication state, or additional provider details.
    value
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Codex app-server reported an error")
        .chars()
        .take(4096)
        .collect()
}

impl Client {
    pub fn connect(options: Options) -> io::Result<Self> {
        Self::connect_cancellable(options, &AtomicBool::new(false))
    }

    pub fn connect_cancellable(options: Options, cancel: &AtomicBool) -> io::Result<Self> {
        #[cfg(unix)]
        {
            Self::connect_unix(options, cancel, Limits::default())
        }
        #[cfg(not(unix))]
        {
            let _ = (options, cancel);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "bounded Codex stdio adapter currently requires Unix",
            ))
        }
    }

    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// Only populated after a validated successful terminal event. Item
    /// completion, never concatenated display deltas, supplies this answer.
    pub fn completed_text(&self) -> Option<&str> {
        self.completed_text.as_deref()
    }

    pub fn cleanup(&self) -> Option<Cleanup> {
        self.wire.cleanup()
    }

    fn validate_policy(&self, result: &Value) -> io::Result<()> {
        if result.get("cwd").and_then(Value::as_str) != Some(self.workspace.as_str())
            || result.get("approvalPolicy").and_then(Value::as_str) != Some("never")
        {
            return Err(invalid(
                "app-server did not confirm the requested workspace and approval policy",
            ));
        }
        let sandbox = result
            .get("sandbox")
            .ok_or_else(|| invalid("app-server did not confirm a sandbox policy"))?;
        let expected = match self.permission {
            PermissionPolicy::ReadOnly => "readOnly",
            PermissionPolicy::WorkspaceWrite => "workspaceWrite",
        };
        let Value::Object(fields) = sandbox else {
            return Err(invalid("app-server sandbox is not an object"));
        };
        let allowed: &[&str] = match self.permission {
            PermissionPolicy::ReadOnly => &["type", "access", "networkAccess"],
            PermissionPolicy::WorkspaceWrite => &[
                "type",
                "readOnlyAccess",
                "networkAccess",
                "writableRoots",
                "excludeSlashTmp",
                "excludeTmpdirEnvVar",
            ],
        };
        if fields
            .iter()
            .any(|(name, _)| !allowed.contains(&name.as_str()))
        {
            return Err(invalid("app-server returned unknown sandbox policy fields"));
        }
        if sandbox.get("type").and_then(Value::as_str) != Some(expected)
            || sandbox
                .get("networkAccess")
                .is_some_and(|access| access != &Value::Bool(false))
        {
            return Err(invalid(
                "app-server returned a broader or unsupported sandbox policy",
            ));
        }
        if self.permission == PermissionPolicy::WorkspaceWrite {
            if sandbox.get("networkAccess") != Some(&Value::Bool(false))
                || sandbox.get("excludeSlashTmp") != Some(&Value::Bool(true))
                || sandbox.get("excludeTmpdirEnvVar") != Some(&Value::Bool(true))
            {
                return Err(invalid(
                    "app-server did not restrict temporary-directory write access",
                ));
            }
            let roots = sandbox
                .get("writableRoots")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("app-server writable roots are unconfirmed"))?;
            if roots.len() > 32 {
                return Err(invalid("app-server writable root count exceeds limit"));
            }
            for root in roots {
                let root = root
                    .as_str()
                    .ok_or_else(|| invalid("invalid app-server writable root"))?;
                let root = PathBuf::from(root);
                if !root.is_absolute() {
                    return Err(invalid("app-server writable roots must be absolute"));
                }
                let root = root.canonicalize()?;
                if !root.starts_with(&self.workspace) {
                    return Err(invalid(
                        "app-server writable roots exceed the selected workspace",
                    ));
                }
            }
        }
        Ok(())
    }

    fn send_cancellable(
        &mut self,
        value: Value,
        deadline: Instant,
        cancel: Option<&AtomicBool>,
    ) -> io::Result<()> {
        self.wire.send(value, deadline, cancel)
    }
    fn receive(&mut self, wait: Duration) -> io::Result<Option<(Value, usize)>> {
        if let Some((value, size)) = self.wire.receive(wait)? {
            contract::envelope(&value)?;
            Ok(Some((value, size)))
        } else {
            Ok(None)
        }
    }

    fn request(&mut self, method: &str, params: Value, deadline: Instant) -> io::Result<u64> {
        self.request_cancellable(method, params, deadline, None)
    }

    fn request_cancellable(
        &mut self,
        method: &str,
        params: Value,
        deadline: Instant,
        cancel: Option<&AtomicBool>,
    ) -> io::Result<u64> {
        contract::outbound(method)?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| invalid("app-server request id exhausted"))?;
        self.send_cancellable(
            object(vec![
                ("id", id_value(id)),
                ("method", string(method)),
                ("params", params),
            ]),
            deadline,
            cancel,
        )?;
        Ok(id)
    }

    fn rpc(
        &mut self,
        method: &str,
        params: Value,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> io::Result<Value> {
        let id = self.request_cancellable(method, params, deadline, Some(cancel))?;
        let mut bytes = 0usize;
        let mut events = 0;
        loop {
            check_cancel(cancel)?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(timed_out("app-server handshake deadline exceeded"));
            }
            let Some((value, size)) = self.receive(POLL.min(remaining))? else {
                continue;
            };
            bytes += size;
            events += 1;
            if bytes > MAX_TURN_BYTES || events > contract::MAX_EVENTS {
                return Err(invalid("app-server handshake output exceeds limit"));
            }
            if self.deny_request(&value, deadline, Some(cancel))? {
                continue;
            }
            if !matches_id(&value, id) {
                if let Some(method) = value.get("method").and_then(Value::as_str) {
                    if contract::notification(method)? == contract::Notification::Informational {
                        continue;
                    }
                }
                return Err(invalid("unexpected app-server handshake message"));
            }
            if let Some(error) = value.get("error") {
                return Err(io::Error::other(error_message(error)));
            }
            return value
                .get("result")
                .cloned()
                .ok_or_else(|| invalid("app-server RPC response has no result or error"));
        }
    }

    fn deny_request(
        &mut self,
        value: &Value,
        deadline: Instant,
        cancel: Option<&AtomicBool>,
    ) -> io::Result<bool> {
        let (Some(method), Some(id)) =
            (value.get("method").and_then(Value::as_str), value.get("id"))
        else {
            return Ok(false);
        };
        if !matches!(id, Value::Number(_) | Value::String(_)) {
            return Err(invalid("invalid app-server request id"));
        }
        let result = match method {
            "item/commandExecution/requestApproval" | "item/fileChange/requestApproval" => {
                Some(object(vec![("decision", string("decline"))]))
            }
            "item/permissions/requestApproval" => Some(object(vec![
                ("permissions", object(vec![])),
                ("scope", string("turn")),
            ])),
            "mcpServer/elicitation/request" => Some(object(vec![
                ("action", string("decline")),
                ("content", Value::Null),
            ])),
            _ => None,
        };
        let reply = if let Some(result) = result {
            object(vec![("id", id.clone()), ("result", result)])
        } else {
            object(vec![
                ("id", id.clone()),
                (
                    "error",
                    object(vec![
                        ("code", Value::Number("-32601".into())),
                        (
                            "message",
                            string(
                                "Unsupported server request; no tool execution or permissions granted",
                            ),
                        ),
                    ]),
                ),
            ])
        };
        self.send_cancellable(reply, deadline, cancel)?;
        Ok(true)
    }

    fn sandbox_policy(&self) -> Value {
        match self.permission {
            PermissionPolicy::ReadOnly => object(vec![
                ("type", string("readOnly")),
                ("networkAccess", Value::Bool(false)),
            ]),
            PermissionPolicy::WorkspaceWrite => object(vec![
                ("type", string("workspaceWrite")),
                ("writableRoots", Value::Array(vec![string(&self.workspace)])),
                ("networkAccess", Value::Bool(false)),
                ("excludeTmpdirEnvVar", Value::Bool(true)),
                ("excludeSlashTmp", Value::Bool(true)),
            ]),
        }
    }

    pub fn turn(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(Event),
    ) -> io::Result<TurnOutcome> {
        let (_, controls) = mpsc::sync_channel(0);
        self.turn_with_control(text, cancel, &controls, on_event)
    }

    /// The UI owns a bounded control channel. Only one steering RPC is in
    /// flight; interruption is checked before new steering is dispatched.
    pub fn turn_with_control(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        controls: &Receiver<Control>,
        on_event: &mut dyn FnMut(Event),
    ) -> io::Result<TurnOutcome> {
        let result = self.turn_inner(text, cancel, controls, on_event);
        if result.is_err() {
            self.shutdown();
            if self.cleanup()
                != Some(Cleanup {
                    child_reaped: true,
                    io_workers_joined: true,
                })
            {
                return Err(io::Error::other(
                    "app-server failed and local cleanup is unconfirmed; remote outcome unknown",
                ));
            }
        }
        result
    }

    fn turn_inner(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        controls: &Receiver<Control>,
        on_event: &mut dyn FnMut(Event),
    ) -> io::Result<TurnOutcome> {
        self.completed_text = None;
        if self.wire.is_closed() {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "app-server client is closed",
            ));
        }
        if cancel.load(Ordering::Relaxed) {
            return Ok(TurnOutcome::NotStarted);
        }
        if json::quoted_len(text)? > MAX_LINE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Codex text input exceeds JSONL limit",
            ));
        }
        let start = Instant::now();
        let mut last_progress = start;
        let rpc_deadline = start + self.limits.rpc;
        let mut params = vec![
            ("threadId", string(&self.thread_id)),
            (
                "input",
                Value::Array(vec![object(vec![
                    ("type", string("text")),
                    ("text", string(text)),
                ])]),
            ),
            ("cwd", string(&self.workspace)),
            ("approvalPolicy", string("never")),
            ("sandboxPolicy", self.sandbox_policy()),
        ];
        if let Some(model) = &self.model {
            params.push(("model", string(model)));
        }
        let start_id =
            self.request_cancellable("turn/start", object(params), rpc_deadline, Some(cancel))?;
        let mut turn_id: Option<String> = None;
        let mut interrupt_id = None;
        let mut interrupt_deadline = None;
        let mut pending = VecDeque::new();
        let mut pending_bytes = 0usize;
        let mut output_bytes = 0usize;
        let mut delta_bytes = 0usize;
        let mut failure = None;
        let mut initial_terminal = None;
        let mut initial_items = None;
        let mut items = contract::Items::default();
        let mut events = 0;
        let mut steering: Option<(u64, u64, Instant)> = None;
        let mut steering_count = 0;
        let mut steering_bytes = 0usize;
        loop {
            let now = Instant::now();
            if cancel.load(Ordering::Relaxed) && interrupt_deadline.is_none() {
                interrupt_deadline = Some(now + self.limits.interrupt);
                on_event(Event::Phase(Phase::Cancelling));
            }
            if turn_id.is_some()
                && pending.is_empty()
                && let Some(outcome) = initial_terminal.take()
            {
                if let Some(turn) = initial_items.take() {
                    items.terminal_items(&turn)?;
                }
                return self.finish_turn(outcome, &items);
            }
            if let Some(deadline) = interrupt_deadline {
                if now >= deadline {
                    return Err(timed_out(
                        "app-server interruption did not reach terminal status",
                    ));
                }
                if interrupt_id.is_none()
                    && let Some(id) = &turn_id
                {
                    interrupt_id = Some(self.request(
                        "turn/interrupt",
                        object(vec![
                            ("threadId", string(&self.thread_id)),
                            ("turnId", string(id)),
                        ]),
                        deadline,
                    )?);
                    on_event(Event::Notice("Codex turn interruption requested".into()));
                }
            }
            if let Some((_, local_id, deadline)) = steering {
                if now >= deadline {
                    on_event(Event::Steering {
                        request_id: local_id,
                        status: SteeringStatus::Unconfirmed,
                    });
                    return Err(timed_out(
                        "app-server steering acknowledgment deadline exceeded; outcome unconfirmed",
                    ));
                }
            } else if interrupt_deadline.is_none()
                && let Some(turn_id) = &turn_id
            {
                if let Ok(Control::Steer { request_id, text }) = controls.try_recv() {
                    if text.trim().is_empty()
                        || text.len() > (64 * 1024usize).saturating_sub(steering_bytes)
                        || steering_count >= 32
                    {
                        on_event(Event::Steering {
                            request_id,
                            status: SteeringStatus::Rejected,
                        });
                    } else {
                        steering_count += 1;
                        steering_bytes += text.len();
                        let deadline = now + self.limits.rpc;
                        let rpc_id = self.request_cancellable(
                            "turn/steer",
                            object(vec![
                                ("threadId", string(&self.thread_id)),
                                ("expectedTurnId", string(turn_id)),
                                (
                                    "input",
                                    Value::Array(vec![object(vec![
                                        ("type", string("text")),
                                        ("text", string(&text)),
                                    ])]),
                                ),
                            ]),
                            deadline,
                            Some(cancel),
                        )?;
                        steering = Some((rpc_id, request_id, deadline));
                    }
                }
            }
            if now.duration_since(start) >= self.limits.turn
                || now.duration_since(last_progress) >= self.limits.idle
            {
                return Err(timed_out("app-server turn deadline exceeded"));
            }
            if turn_id.is_none() && now >= rpc_deadline {
                return Err(timed_out("app-server turn/start deadline exceeded"));
            }
            let deferred = turn_id.is_some() && !pending.is_empty();
            let next = if deferred {
                pending.pop_front()
            } else {
                self.receive(POLL)?
            };
            let Some((value, size)) = next else {
                continue;
            };
            last_progress = Instant::now();
            if !deferred {
                output_bytes += size;
                events += 1;
                if output_bytes > MAX_TURN_BYTES || events > contract::MAX_EVENTS {
                    return Err(invalid("app-server turn output exceeds limit"));
                }
            }
            let write_deadline =
                interrupt_deadline.unwrap_or_else(|| Instant::now() + self.limits.rpc);
            if value.get("method").is_some() && value.get("id").is_some() {
                on_event(Event::Phase(Phase::WaitingForApproval));
            }
            if self.deny_request(
                &value,
                write_deadline,
                if interrupt_deadline.is_some() {
                    None
                } else {
                    Some(cancel)
                },
            )? {
                on_event(Event::Notice(
                    "Codex permission or tool request denied by harness policy".into(),
                ));
                on_event(Event::Phase(if interrupt_deadline.is_some() {
                    Phase::Cancelling
                } else {
                    Phase::Generating
                }));
                continue;
            }
            if matches_id(&value, start_id) && turn_id.is_none() {
                if let Some(error) = value.get("error") {
                    return Ok(TurnOutcome::Failed(error_message(error)));
                }
                let turn = value
                    .get("result")
                    .and_then(|result| result.get("turn"))
                    .ok_or_else(|| invalid("app-server turn/start response has no turn"))?;
                let id = turn
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| valid_id(id))
                    .ok_or_else(|| invalid("app-server turn/start response has no valid id"))?;
                initial_terminal = terminal(turn, &failure)?;
                if initial_terminal.is_some() {
                    initial_items = Some(turn.clone());
                }
                turn_id = Some(id.to_owned());
                on_event(Event::Phase(Phase::Generating));
                if pending.is_empty()
                    && let Some(outcome) = initial_terminal.take()
                {
                    items.terminal_items(turn)?;
                    return self.finish_turn(outcome, &items);
                }
                continue;
            }
            if interrupt_id.is_some_and(|id| matches_id(&value, id)) {
                if value.get("error").is_some() {
                    return Err(io::Error::other(
                        "Codex interruption rejected; remote outcome unconfirmed",
                    ));
                }
                continue;
            }
            if let Some((rpc_id, local_id, _)) = steering
                && matches_id(&value, rpc_id)
            {
                let status = if value.get("error").is_some() {
                    SteeringStatus::Rejected
                } else {
                    if value
                        .get("result")
                        .and_then(|v| v.get("turnId"))
                        .and_then(Value::as_str)
                        != turn_id.as_deref()
                    {
                        return Err(invalid("app-server steering acknowledged a different turn"));
                    }
                    SteeringStatus::Accepted
                };
                on_event(Event::Steering {
                    request_id: local_id,
                    status,
                });
                steering = None;
                continue;
            }
            let Some(method) = value.get("method").and_then(Value::as_str) else {
                // An interrupt acknowledgment can trail its terminal event.
                if value.get("id").is_some_and(|v| matches!(v, Value::Number(n) if n.parse::<u64>().is_ok_and(|id| id < start_id))) { continue; }
                return Err(invalid("unexpected app-server RPC response"));
            };
            let kind = contract::notification(method)?;
            if kind == contract::Notification::Informational {
                continue;
            }
            let Some(params) = value.get("params") else {
                return Err(invalid("app-server notification has no params"));
            };
            if params.get("threadId").and_then(Value::as_str).is_none() {
                return Err(invalid("app-server notification has no thread id"));
            }
            if params.get("threadId").and_then(Value::as_str) != Some(self.thread_id.as_str()) {
                continue;
            }
            if turn_id.is_none() {
                if matches!(
                    method,
                    "turn/started"
                        | "turn/completed"
                        | "item/started"
                        | "item/completed"
                        | "item/agentMessage/delta"
                        | "error"
                ) {
                    pending_bytes += size;
                    if pending.len() >= MAX_PENDING || pending_bytes > MAX_PENDING_BYTES {
                        return Err(invalid(
                            "app-server pre-response notification limit exceeded",
                        ));
                    }
                    pending.push_back((value, size));
                }
                continue;
            }
            let expected = turn_id.as_deref().unwrap();
            let event_turn = params.get("turnId").and_then(Value::as_str).or_else(|| {
                params
                    .get("turn")
                    .and_then(|turn| turn.get("id"))
                    .and_then(Value::as_str)
            });
            if event_turn.is_none() {
                return Err(invalid("app-server notification has no turn id"));
            }
            if event_turn != Some(expected) {
                continue;
            }
            match method {
                "item/agentMessage/delta" => {
                    let delta = items.delta(params)?;
                    delta_bytes += delta.len();
                    if delta_bytes > MAX_DELTA_BYTES {
                        return Err(invalid("app-server agent text exceeds limit"));
                    }
                    on_event(Event::AgentMessageDelta(delta.to_owned()));
                }
                "item/started" | "item/completed" => {
                    let item = params
                        .get("item")
                        .ok_or_else(|| invalid("app-server item event has no item"))?;
                    if let Some(event) = items.update(item, method == "item/completed")? {
                        on_event(event);
                    }
                }
                "turn/started" => {
                    if params
                        .get("turn")
                        .and_then(|t| t.get("status"))
                        .and_then(Value::as_str)
                        != Some("inProgress")
                    {
                        return Err(invalid("app-server started event is not in progress"));
                    }
                }
                "error" => {
                    if params.get("willRetry") == Some(&Value::Bool(true)) {
                        on_event(Event::Notice("Codex is retrying a provider error".into()));
                    } else {
                        failure = Some(
                            params
                                .get("error")
                                .map(error_message)
                                .unwrap_or_else(|| "Codex reported a turn error".into()),
                        );
                    }
                }
                "turn/completed" => {
                    let turn = params
                        .get("turn")
                        .ok_or_else(|| invalid("app-server terminal event has no turn"))?;
                    let outcome = terminal(turn, &failure)?
                        .ok_or_else(|| invalid("app-server completed event is not terminal"))?;
                    if initial_terminal
                        .as_ref()
                        .is_some_and(|initial| initial != &outcome)
                    {
                        return Err(invalid(
                            "app-server supplied contradictory terminal statuses",
                        ));
                    }
                    if let Some(initial) = initial_items.take() {
                        items.terminal_items(&initial)?;
                    }
                    items.terminal_items(turn)?;
                    if let Some((_, local_id, _)) = steering.take() {
                        on_event(Event::Steering {
                            request_id: local_id,
                            status: SteeringStatus::Unconfirmed,
                        });
                    }
                    return self.finish_turn(outcome, &items);
                }
                _ => {}
            }
            if pending.is_empty()
                && let Some(outcome) = initial_terminal.take()
            {
                if let Some(turn) = initial_items.take() {
                    items.terminal_items(&turn)?;
                }
                return self.finish_turn(outcome, &items);
            }
        }
    }

    fn finish_turn(
        &mut self,
        outcome: TurnOutcome,
        items: &contract::Items,
    ) -> io::Result<TurnOutcome> {
        if outcome == TurnOutcome::Completed {
            self.completed_text = Some(items.finish()?);
        }
        Ok(outcome)
    }

    fn shutdown(&mut self) {
        self.wire.shutdown();
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn terminal(turn: &Value, failure: &Option<String>) -> io::Result<Option<TurnOutcome>> {
    let status = turn
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("app-server turn has no valid status"))?;
    let error = turn
        .get("error")
        .filter(|error| !matches!(error, Value::Null))
        .map(error_message);
    match status {
        "inProgress" => Ok(None),
        "completed" => Ok(Some(match error.or_else(|| failure.clone()) {
            Some(message) => TurnOutcome::Failed(message),
            None => TurnOutcome::Completed,
        })),
        "interrupted" => Ok(Some(TurnOutcome::Interrupted)),
        "failed" => Ok(Some(TurnOutcome::Failed(
            error
                .or_else(|| failure.clone())
                .unwrap_or_else(|| "Codex turn failed".into()),
        ))),
        _ => Err(invalid("app-server turn has an unknown status")),
    }
}

fn check_cancel(cancel: &AtomicBool) -> io::Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(io::Error::new(
            io::ErrorKind::ConnectionAborted,
            "delegated operation canceled; remote outcome unconfirmed",
        ))
    } else {
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod tests;
