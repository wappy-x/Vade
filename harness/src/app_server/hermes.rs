//! Optional Hermes ACP v1 delegation. Interface sources and the intentionally
//! narrow compatibility contract are recorded in docs/bridge-contract.md.
//! No native loop, host tools, credentials, or external implementation code.
use super::{
    Cleanup, Event, Limits, MAX_DELTA_BYTES, MAX_PENDING, MAX_PENDING_BYTES, MAX_TURN_BYTES, POLL,
    Phase, TurnOutcome, check_cancel, child::Connection, contract, id_value, invalid, matches_id,
    object, string, timed_out, valid_id,
};
use crate::json::Value;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const MAX_ITEMS: usize = 256;
const MAX_UPDATE_BYTES: usize = 8 * 1024 * 1024;
const MAX_UPDATES: usize = 4096;

#[derive(Clone, Debug)]
pub struct Options {
    /// A selected executable accepting `acp`, never a shell command string.
    pub executable: PathBuf,
    pub workspace: PathBuf,
    pub session_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    pub load_session: bool,
    pub resume_session: bool,
    pub close_session: bool,
    pub fork_session: bool,
    pub list_sessions: bool,
}

pub struct Client {
    wire: Connection,
    session_id: String,
    capabilities: Capabilities,
    provider_data: Value,
    updates: Vec<Value>,
    last_response: Option<Value>,
    completed_text: Option<String>,
    next_id: u64,
    limits: Limits,
}

impl Client {
    pub fn connect(options: Options) -> io::Result<Self> {
        Self::connect_cancellable(options, &AtomicBool::new(false))
    }

    pub fn connect_cancellable(options: Options, cancel: &AtomicBool) -> io::Result<Self> {
        Self::connect_with_limits(options, cancel, Limits::default())
    }

    fn connect_with_limits(
        options: Options,
        cancel: &AtomicBool,
        limits: Limits,
    ) -> io::Result<Self> {
        check_cancel(cancel)?;
        let workspace = options.workspace.canonicalize()?;
        let cwd = workspace
            .to_str()
            .filter(|s| s.len() <= 8192 && !s.chars().any(char::is_control))
            .ok_or_else(|| invalid("ACP workspace must be bounded UTF-8"))?;
        if !workspace.is_dir()
            || options
                .session_id
                .as_deref()
                .is_some_and(|id| !valid_id(id))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid ACP workspace or session identifier",
            ));
        }
        let executable =
            if options.executable.is_relative() && options.executable.components().count() > 1 {
                options.executable.canonicalize()?
            } else {
                options.executable
            };
        let mut command = Command::new(executable);
        command.arg("acp").current_dir(&workspace);
        let mut client = Self {
            wire: Connection::spawn(command)?,
            session_id: String::new(),
            capabilities: Capabilities::default(),
            provider_data: Value::Null,
            updates: Vec::new(),
            last_response: None,
            completed_text: None,
            next_id: 1,
            limits,
        };
        let deadline = Instant::now() + limits.rpc;
        let (init, updates) = client.rpc(
            "initialize",
            object(vec![
                ("protocolVersion", id_value(1)),
                (
                    "clientInfo",
                    object(vec![
                        ("name", string("vade_harness")),
                        ("version", string("0.1.0")),
                    ]),
                ),
                (
                    "clientCapabilities",
                    object(vec![
                        (
                            "fs",
                            object(vec![
                                ("readTextFile", Value::Bool(false)),
                                ("writeTextFile", Value::Bool(false)),
                            ]),
                        ),
                        ("terminal", Value::Bool(false)),
                        ("auth", object(vec![("terminal", Value::Bool(false))])),
                    ]),
                ),
            ]),
            deadline,
            cancel,
        )?;
        if !updates.is_empty() {
            return Err(invalid("ACP updates arrived before session setup"));
        }
        if init.get("protocolVersion") != Some(&id_value(1)) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Hermes adapter supports ACP major version 1 only",
            ));
        }
        client.capabilities = capabilities(&init)?;
        let mut params = vec![("cwd", string(cwd)), ("mcpServers", Value::Array(vec![]))];
        let method = if let Some(id) = &options.session_id {
            params.push(("sessionId", string(id)));
            client.session_id = id.clone();
            if client.capabilities.resume_session {
                "session/resume"
            } else if client.capabilities.load_session {
                "session/load"
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "ACP peer did not advertise session load or resume",
                ));
            }
        } else {
            "session/new"
        };
        let (session, replay) = client.rpc(method, object(params), deadline, cancel)?;
        if options.session_id.is_none() {
            client.session_id = session
                .get("sessionId")
                .and_then(Value::as_str)
                .filter(|id| valid_id(id))
                .ok_or_else(|| invalid("ACP new-session response has no valid identifier"))?
                .to_owned();
        } else if session
            .get("sessionId")
            .is_some_and(|id| id.as_str() != Some(client.session_id.as_str()))
        {
            return Err(invalid("ACP peer returned a different session"));
        }
        let mut discarded_replay = Projection::default();
        for value in &replay {
            // Retain bounded opaque replay in the provider domain, without
            // importing or displaying it as native history or new prompt output.
            discarded_replay.update(value, &client.session_id, &mut |_| {})?;
        }
        client.provider_data = object(vec![
            ("initialize", init),
            ("session", session),
            ("replay", Value::Array(replay)),
        ]);
        Ok(client)
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }
    /// Provider-scoped opaque setup data. Never native context or a KV handle.
    pub fn provider_data(&self) -> &Value {
        &self.provider_data
    }
    /// Bounded opaque updates for this attempt, including a failed partial one.
    /// Only completed_text() is a successfully completed answer.
    pub fn updates(&self) -> &[Value] {
        &self.updates
    }
    /// Opaque terminal response for this attempt, including a provider RPC error.
    pub fn last_response(&self) -> Option<&Value> {
        self.last_response.as_ref()
    }
    pub fn completed_text(&self) -> Option<&str> {
        self.completed_text.as_deref()
    }
    pub fn cleanup(&self) -> Option<Cleanup> {
        self.wire.cleanup()
    }
    /// Local shutdown, not an unnegotiated ACP shutdown/session-close RPC.
    pub fn shutdown(&mut self) -> Cleanup {
        self.wire.shutdown();
        self.wire.cleanup().expect("shutdown records its outcome")
    }

    fn send(
        &mut self,
        fields: Vec<(&str, Value)>,
        deadline: Instant,
        cancel: Option<&AtomicBool>,
    ) -> io::Result<()> {
        let mut fields = fields;
        fields.push(("jsonrpc", string("2.0")));
        self.wire.send(object(fields), deadline, cancel)
    }
    fn request(
        &mut self,
        method: &str,
        params: Value,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> io::Result<u64> {
        if !matches!(
            method,
            "initialize" | "session/new" | "session/load" | "session/resume" | "session/prompt"
        ) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "method outside the Hermes ACP contract",
            ));
        }
        let id = self.next_id;
        self.next_id = id
            .checked_add(1)
            .ok_or_else(|| invalid("ACP request identifier exhausted"))?;
        self.send(
            vec![
                ("id", id_value(id)),
                ("method", string(method)),
                ("params", params),
            ],
            deadline,
            Some(cancel),
        )?;
        Ok(id)
    }
    fn receive(&mut self, wait: std::time::Duration) -> io::Result<Option<(Value, usize)>> {
        if let Some((value, bytes)) = self.wire.receive(wait)? {
            contract::envelope(&value)?;
            if value.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
                return Err(invalid("ACP requires JSON-RPC 2.0 framing"));
            }
            Ok(Some((value, bytes)))
        } else {
            Ok(None)
        }
    }

    fn deny_callback(
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
        let reply = if method == "session/request_permission" {
            // Never select an allow option. A missing/malformed reject option
            // becomes cancellation of the permission request, still no grant.
            let reject = if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                None
            } else {
                value
                    .get("params")
                    .and_then(|p| p.get("options"))
                    .and_then(Value::as_array)
                    .filter(|options| options.len() <= 32)
                    .and_then(|options| {
                        options.iter().find(|option| {
                            option.get("kind").and_then(Value::as_str) == Some("reject_once")
                                && option
                                    .get("optionId")
                                    .and_then(Value::as_str)
                                    .is_some_and(valid_id)
                        })
                    })
                    .and_then(|option| option.get("optionId"))
                    .cloned()
            };
            let outcome = if let Some(id) = reject {
                object(vec![("outcome", string("selected")), ("optionId", id)])
            } else {
                object(vec![("outcome", string("cancelled"))])
            };
            vec![
                ("id", id.clone()),
                ("result", object(vec![("outcome", outcome)])),
            ]
        } else {
            vec![
                ("id", id.clone()),
                (
                    "error",
                    object(vec![
                        ("code", Value::Number("-32601".into())),
                        (
                            "message",
                            string("Unsupported host callback; no execution or permission granted"),
                        ),
                    ]),
                ),
            ]
        };
        // During cancellation the denial must still be written; the caller
        // supplies the shorter interruption deadline for bounded cleanup.
        self.send(
            reply,
            deadline,
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                None
            } else {
                cancel
            },
        )?;
        Ok(true)
    }

    fn rpc(
        &mut self,
        method: &str,
        params: Value,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> io::Result<(Value, Vec<Value>)> {
        let id = self.request(method, params, deadline, cancel)?;
        let mut replay = Vec::new();
        let mut replay_bytes = 0;
        let mut bytes = 0;
        let mut events = 0;
        loop {
            check_cancel(cancel)?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(timed_out("ACP setup deadline exceeded"));
            }
            let Some((value, size)) = self.receive(POLL.min(remaining))? else {
                continue;
            };
            bytes += size;
            events += 1;
            if bytes > MAX_TURN_BYTES || events > contract::MAX_EVENTS {
                return Err(invalid("ACP setup output exceeds limit"));
            }
            if self.deny_callback(&value, deadline, Some(cancel))? {
                continue;
            }
            if matches_id(&value, id) {
                if value.get("error").is_some() {
                    return Err(io::Error::other(
                        "ACP setup rejected; external authentication/configuration may be required; no login attempted",
                    ));
                }
                return Ok((
                    value
                        .get("result")
                        .filter(|v| matches!(v, Value::Object(_)))
                        .ok_or_else(|| invalid("ACP setup result is not an object"))?
                        .clone(),
                    replay,
                ));
            }
            if value.get("method").and_then(Value::as_str) != Some("session/update") {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "unexpected ACP setup message",
                ));
            }
            replay_bytes += size;
            if replay.len() >= MAX_PENDING || replay_bytes > MAX_PENDING_BYTES {
                return Err(invalid("ACP setup replay exceeds limit"));
            }
            replay.push(value);
        }
    }

    pub fn turn(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(Event),
    ) -> io::Result<TurnOutcome> {
        self.completed_text = None;
        self.updates.clear();
        self.last_response = None;
        let result = self.turn_inner(text, cancel, on_event);
        if result.is_err() {
            let cleanup = self.shutdown();
            if !cleanup.child_reaped || !cleanup.io_workers_joined {
                return Err(io::Error::other(
                    "ACP failed and local cleanup is unconfirmed; remote outcome unknown",
                ));
            }
        }
        result
    }
    fn turn_inner(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(Event),
    ) -> io::Result<TurnOutcome> {
        if self.wire.is_closed() {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "ACP client is closed",
            ));
        }
        if cancel.load(Ordering::Relaxed) {
            return Ok(TurnOutcome::NotStarted);
        }
        if text.len() > 64 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ACP prompt exceeds 64 KiB",
            ));
        }
        let started = Instant::now();
        let mut progress = started;
        let id = self.request(
            "session/prompt",
            object(vec![
                ("sessionId", string(&self.session_id)),
                (
                    "prompt",
                    Value::Array(vec![object(vec![
                        ("type", string("text")),
                        ("text", string(text)),
                    ])]),
                ),
            ]),
            started + self.limits.rpc,
            cancel,
        )?;
        on_event(Event::Phase(Phase::Generating));
        let mut interrupt = None;
        let mut projection = Projection::default();
        let mut bytes = 0;
        let mut update_bytes = 0;
        let mut events = 0;
        loop {
            let now = Instant::now();
            if cancel.load(Ordering::Relaxed) && interrupt.is_none() {
                let deadline = now + self.limits.interrupt;
                interrupt = Some(deadline);
                on_event(Event::Phase(Phase::Cancelling));
                self.send(
                    vec![
                        ("method", string("session/cancel")),
                        (
                            "params",
                            object(vec![("sessionId", string(&self.session_id))]),
                        ),
                    ],
                    deadline,
                    None,
                )?;
            }
            if interrupt.is_some_and(|deadline| now >= deadline) {
                return Err(timed_out(
                    "ACP cancellation was not acknowledged; remote outcome unknown",
                ));
            }
            if now.duration_since(started) >= self.limits.turn
                || now.duration_since(progress) >= self.limits.idle
            {
                return Err(timed_out(
                    "ACP prompt deadline exceeded; remote outcome unknown",
                ));
            }
            let Some((value, size)) = self.receive(POLL)? else {
                continue;
            };
            progress = Instant::now();
            bytes += size;
            events += 1;
            if bytes > MAX_TURN_BYTES || events > contract::MAX_EVENTS {
                return Err(invalid("ACP prompt output exceeds limit"));
            }
            if value.get("id").is_some() && value.get("method").is_some() {
                on_event(Event::Phase(Phase::WaitingForApproval));
            }
            if self.deny_callback(
                &value,
                interrupt.unwrap_or_else(|| Instant::now() + self.limits.rpc),
                Some(cancel),
            )? {
                on_event(Event::Notice(
                    "Hermes permission or host-tool callback denied".into(),
                ));
                on_event(Event::Phase(if interrupt.is_some() {
                    Phase::Cancelling
                } else {
                    projection.phase()
                }));
                continue;
            }
            if matches_id(&value, id) {
                self.last_response = Some(value.clone());
                if value.get("error").is_some() {
                    return Err(io::Error::other(
                        "ACP prompt RPC failed; remote outcome unconfirmed; provider details suppressed",
                    ));
                }
                let reason = value
                    .get("result")
                    .and_then(|v| v.get("stopReason"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("ACP prompt response has no stop reason"))?;
                return match reason {
                    "end_turn" => {
                        self.completed_text = Some(projection.finish()?);
                        Ok(TurnOutcome::Completed)
                    }
                    "cancelled" => Ok(TurnOutcome::Interrupted),
                    "max_tokens" | "max_turn_requests" | "refusal" => {
                        Ok(TurnOutcome::Failed(format!("ACP turn ended with {reason}")))
                    }
                    _ => Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "unknown ACP stop reason",
                    )),
                };
            }
            if value.get("method").and_then(Value::as_str) != Some("session/update") {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "unknown ACP notification or response",
                ));
            }
            update_bytes += size;
            if update_bytes > MAX_UPDATE_BYTES || self.updates.len() >= MAX_UPDATES {
                return Err(invalid("ACP retained update budget exceeded"));
            }
            projection.update(&value, &self.session_id, on_event)?;
            self.updates.push(value);
        }
    }
}

fn capabilities(result: &Value) -> io::Result<Capabilities> {
    let Some(value) = result.get("agentCapabilities") else {
        return Ok(Capabilities::default());
    };
    if !matches!(value, Value::Object(_)) {
        return Err(invalid("ACP agent capabilities must be an object"));
    }
    let load_session = match value.get("loadSession") {
        None | Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) => true,
        _ => return Err(invalid("invalid ACP load-session capability")),
    };
    let sessions = value.get("sessionCapabilities");
    if sessions.is_some_and(|v| !matches!(v, Value::Object(_) | Value::Null)) {
        return Err(invalid("invalid ACP session capabilities"));
    }
    let supported = |name| -> io::Result<bool> {
        match sessions.and_then(|v| v.get(name)) {
            None | Some(Value::Null) => Ok(false),
            Some(Value::Object(_)) => Ok(true),
            _ => Err(invalid("invalid ACP optional session capability")),
        }
    };
    Ok(Capabilities {
        load_session,
        resume_session: supported("resume")?,
        close_session: supported("close")?,
        fork_session: supported("fork")?,
        list_sessions: supported("list")?,
    })
}

#[derive(Default)]
struct Projection {
    messages: Vec<(Option<String>, String)>,
    text_bytes: usize,
    tools: Vec<(String, String)>,
}
impl Projection {
    fn update(
        &mut self,
        value: &Value,
        session: &str,
        on_event: &mut dyn FnMut(Event),
    ) -> io::Result<()> {
        let params = value
            .get("params")
            .ok_or_else(|| invalid("ACP update has no params"))?;
        if params.get("sessionId").and_then(Value::as_str) != Some(session) {
            return Err(invalid("ACP update belongs to an unexpected session"));
        }
        let update = params
            .get("update")
            .ok_or_else(|| invalid("ACP update has no body"))?;
        let kind = update
            .get("sessionUpdate")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("ACP update has no variant"))?;
        match kind {
            "agent_message_chunk" => {
                let content = update
                    .get("content")
                    .ok_or_else(|| invalid("ACP agent chunk has no content"))?;
                if content.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "ACP adapter supports text output only",
                    ));
                }
                let text = content
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("ACP text chunk is not text"))?;
                self.text_bytes += text.len();
                if self.text_bytes > MAX_DELTA_BYTES {
                    return Err(invalid("ACP answer text exceeds limit"));
                }
                let id = update
                    .get("messageId")
                    .map(|id| {
                        id.as_str()
                            .filter(|id| valid_id(id))
                            .map(str::to_owned)
                            .ok_or_else(|| invalid("invalid ACP message identifier"))
                    })
                    .transpose()?;
                let index = if let Some(id) = &id {
                    self.messages
                        .iter()
                        .position(|(previous, _)| previous.as_ref() == Some(id))
                } else {
                    self.messages
                        .len()
                        .checked_sub(1)
                        .filter(|&i| self.messages[i].0.is_none())
                };
                let index = if let Some(index) = index {
                    index
                } else {
                    if self.messages.len() >= MAX_ITEMS {
                        return Err(invalid("ACP message count exceeds limit"));
                    }
                    self.messages.push((id, String::new()));
                    self.messages.len() - 1
                };
                self.messages[index].1.push_str(text);
                on_event(Event::AgentMessageDelta(text.into()));
            }
            "tool_call" | "tool_call_update" => {
                let id = update
                    .get("toolCallId")
                    .and_then(Value::as_str)
                    .filter(|id| valid_id(id))
                    .ok_or_else(|| invalid("ACP tool call has no valid id"))?;
                let previous = self.tools.iter().position(|(previous, _)| previous == id);
                let status = update
                    .get("status")
                    .map(|v| {
                        v.as_str()
                            .ok_or_else(|| invalid("ACP tool status is not text"))
                    })
                    .transpose()?;
                if status.is_some_and(|s| {
                    !matches!(s, "pending" | "in_progress" | "completed" | "failed")
                }) {
                    return Err(invalid("unknown ACP tool status"));
                }
                let index = if kind == "tool_call" {
                    if previous.is_some() || self.tools.len() >= MAX_ITEMS {
                        return Err(invalid("duplicate or excessive ACP tool calls"));
                    }
                    self.tools
                        .push((id.into(), status.unwrap_or("pending").into()));
                    self.tools.len() - 1
                } else {
                    previous.ok_or_else(|| invalid("ACP update refers to an unknown tool call"))?
                };
                if let Some(status) = status {
                    let previous = &self.tools[index].1;
                    if matches!(previous.as_str(), "completed" | "failed") && previous != status {
                        return Err(invalid("ACP changed a terminal tool status"));
                    }
                    self.tools[index].1 = status.into();
                }
                on_event(Event::Phase(
                    if self
                        .tools
                        .iter()
                        .any(|(_, s)| matches!(s.as_str(), "pending" | "in_progress"))
                    {
                        Phase::ToolRunning
                    } else {
                        Phase::Generating
                    },
                ));
            }
            "user_message_chunk"
            | "agent_thought_chunk"
            | "plan"
            | "available_commands_update"
            | "current_mode_update"
            | "config_option_update"
            | "session_info_update"
            | "usage_update" => (),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "unknown ACP session update; contract update required",
                ));
            }
        }
        Ok(())
    }
    fn phase(&self) -> Phase {
        if self
            .tools
            .iter()
            .any(|(_, s)| matches!(s.as_str(), "pending" | "in_progress"))
        {
            Phase::ToolRunning
        } else {
            Phase::Generating
        }
    }
    fn finish(&self) -> io::Result<String> {
        if self
            .tools
            .iter()
            .any(|(_, status)| !matches!(status.as_str(), "completed" | "failed"))
        {
            return Err(invalid("ACP completed a turn with unfinished tool calls"));
        }
        Ok(self
            .messages
            .iter()
            .map(|(_, text)| text.as_str())
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::json;
    use std::{
        fs,
        path::Path,
        sync::{OnceLock, atomic::AtomicUsize},
        thread,
        time::Duration,
    };

    fn mock() -> &'static Path {
        static BINARY: OnceLock<PathBuf> = OnceLock::new();
        BINARY
            .get_or_init(|| {
                let root = Path::new(env!("CARGO_MANIFEST_DIR"));
                let directory = root
                    .join("target")
                    .join(format!("hermes-mock-{}", std::process::id()));
                fs::create_dir_all(&directory).unwrap();
                let source = directory.join("mock.rs");
                fs::write(
                    &source,
                    format!(
                        "#[path = {:?}] mod json;\n{}",
                        root.join("src/json.rs"),
                        include_str!("../../tests/fixtures/hermes_child.rs")
                    ),
                )
                .unwrap();
                let binary = directory.join("mock-child");
                let compiler = std::env::var_os("VADE_TEST_RUSTC")
                    .or_else(|| std::env::var_os("RUSTC"))
                    .unwrap_or_else(|| "rustc".into());
                assert!(
                    Command::new(compiler)
                        .args([
                            "--edition=2024",
                            "--crate-name=vade_hermes_fixture",
                            "-Adead_code"
                        ])
                        .arg(source)
                        .arg("-o")
                        .arg(&binary)
                        .status()
                        .unwrap()
                        .success()
                );
                binary
            })
            .as_path()
    }
    fn options(mode: &str, session: Option<&str>) -> Options {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let workspace = mock().parent().unwrap().join(format!(
            "workspace-{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&workspace).unwrap();
        fs::write(workspace.join("fixture-mode"), mode).unwrap();
        Options {
            executable: mock().into(),
            workspace,
            session_id: session.map(str::to_owned),
        }
    }
    fn client() -> Client {
        Client::connect(options("", None)).unwrap()
    }
    fn assert_closed(client: &Client) {
        assert!(client.wire.stopped());
        assert_eq!(
            client.cleanup(),
            Some(Cleanup {
                child_reaped: true,
                io_workers_joined: true
            })
        );
        assert!(client.completed_text().is_none());
    }
    #[test]
    fn new_load_resume_exact_prompt_and_opaque_data() {
        let exact = "  exact\n\t🦀 \"text\"  ";
        for (mode, session, expected) in [
            ("", None, "session/new"),
            ("", Some("saved-session"), "session/load"),
            ("resume", Some("saved-session"), "session/resume"),
        ] {
            let mut client = Client::connect(options(mode, session)).unwrap();
            assert_eq!(client.session_id(), session.unwrap_or("mock-session"));
            assert!(client.capabilities().load_session);
            assert_eq!(client.capabilities().resume_session, mode == "resume");
            assert!(
                client.capabilities().fork_session
                    && client.capabilities().close_session
                    && client.capabilities().list_sessions
            );
            let provider = json::stringify(client.provider_data()).unwrap();
            assert!(provider.contains("opaque setup") && provider.contains(expected));
            let mut displayed = String::new();
            assert_eq!(
                client
                    .turn(exact, &AtomicBool::new(false), &mut |event| {
                        if let Event::AgentMessageDelta(text) = event {
                            displayed.push_str(&text);
                        }
                    })
                    .unwrap(),
                TurnOutcome::Completed
            );
            assert_eq!(displayed, exact); // loaded history was not shown or forwarded
            assert_eq!(client.completed_text(), Some(exact));
            assert_eq!(client.updates().len(), 1);
            assert!(
                json::stringify(client.last_response().unwrap())
                    .unwrap()
                    .contains("retained externally")
            );
            assert_eq!(
                client
                    .turn("next exact prompt", &AtomicBool::new(false), &mut |_| {})
                    .unwrap(),
                TurnOutcome::Completed
            );
            assert_eq!(client.completed_text(), Some("next exact prompt"));
        }
    }
    #[test]
    fn interleaved_messages_and_tools_commit_only_at_terminal_response() {
        let mut client = client();
        let mut provisional = String::new();
        let mut tool_seen = false;
        assert_eq!(
            client
                .turn(
                    "interleaved",
                    &AtomicBool::new(false),
                    &mut |event| match event {
                        Event::AgentMessageDelta(text) => provisional.push_str(&text),
                        Event::Phase(Phase::ToolRunning) => tool_seen = true,
                        _ => (),
                    }
                )
                .unwrap(),
            TurnOutcome::Completed
        );
        assert!(tool_seen);
        assert_eq!(provisional, "first secondmessage 🦀");
        assert_eq!(client.completed_text(), Some("first message 🦀\nsecond"));
        assert_eq!(client.updates().len(), 6);
        assert!(
            client
                .updates()
                .iter()
                .any(|v| json::stringify(v).unwrap().contains("provider field"))
        );
    }
    #[test]
    fn all_permissions_and_known_or_unknown_host_callbacks_are_denied() {
        let mut client = client();
        let mut denials = 0;
        assert_eq!(
            client
                .turn(
                    "callbacks",
                    &AtomicBool::new(false),
                    &mut |event| if matches!(event, Event::Notice(_)) {
                        denials += 1;
                    }
                )
                .unwrap(),
            TurnOutcome::Completed
        );
        assert_eq!(denials, 9);
        assert_eq!(client.completed_text(), Some("all callbacks denied"));
    }
    #[test]
    fn cancel_during_tool_or_permission_waits_for_actual_prompt_acknowledgment() {
        for prompt in ["cancel-tool", "cancel-permission", "complete-race"] {
            let mut client = client();
            let cancel = AtomicBool::new(false);
            let began = Instant::now();
            let result = client
                .turn(prompt, &cancel, &mut |event| {
                    if matches!(event, Event::Phase(Phase::ToolRunning)) {
                        cancel.store(true, Ordering::Relaxed);
                    }
                })
                .unwrap();
            assert!(began.elapsed() >= Duration::from_millis(120));
            assert!(began.elapsed() < Duration::from_secs(2));
            if prompt == "complete-race" {
                assert_eq!(result, TurnOutcome::Completed);
                assert_eq!(
                    client.completed_text(),
                    Some("completed before cancellation")
                );
            } else {
                assert_eq!(result, TurnOutcome::Interrupted);
                assert!(client.completed_text().is_none());
            }
            cancel.store(false, Ordering::Relaxed);
            assert_eq!(
                client
                    .turn("explicit follow-up", &cancel, &mut |_| {})
                    .unwrap(),
                TurnOutcome::Completed
            );
        }
    }
    #[test]
    fn unknown_malformed_or_partial_outcomes_close_and_reap() {
        for prompt in [
            "crash",
            "partial",
            "unknown-event",
            "wrong-session",
            "unknown-response",
            "unknown-notification",
            "unknown-stop",
            "rpc-error",
            "unknown-tool",
            "unfinished-tool",
            "reopened-tool",
            "unknown-tool-status",
            "oversized",
        ] {
            let mut client = client();
            assert!(
                client
                    .turn(prompt, &AtomicBool::new(false), &mut |_| {})
                    .is_err(),
                "{prompt}"
            );
            assert_closed(&client);
            assert!(
                client
                    .turn(
                        "must not restart automatically",
                        &AtomicBool::new(false),
                        &mut |_| {}
                    )
                    .is_err()
            );
        }
    }
    #[test]
    fn explicit_non_success_stops_never_commit_provisional_text() {
        let mut client = client();
        for prompt in ["max_tokens", "max_turn_requests", "refusal"] {
            assert!(matches!(
                client
                    .turn(prompt, &AtomicBool::new(false), &mut |_| {})
                    .unwrap(),
                TurnOutcome::Failed(_)
            ));
            assert!(client.completed_text().is_none());
        }
        assert_eq!(
            client
                .turn("explicit recovery", &AtomicBool::new(false), &mut |_| {})
                .unwrap(),
            TurnOutcome::Completed
        );
    }
    #[test]
    fn negotiated_session_operations_and_version_must_match() {
        for (mode, session) in [
            ("v2", None),
            ("no-load", Some("saved")),
            ("init-error", None),
            ("wrong-setup-session", Some("saved")),
        ] {
            assert!(Client::connect(options(mode, session)).is_err(), "{mode}");
        }
        let mut client = client();
        for method in [
            "session/fork",
            "session/list",
            "session/close",
            "authenticate",
            "terminal/create",
            "_future/run",
        ] {
            assert_eq!(
                client
                    .request(
                        method,
                        object(vec![]),
                        Instant::now() + Duration::from_secs(1),
                        &AtomicBool::new(false)
                    )
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::Unsupported
            );
        }
        assert_eq!(
            client
                .turn("still usable", &AtomicBool::new(false), &mut |_| {})
                .unwrap(),
            TurnOutcome::Completed
        );
    }
    #[test]
    fn hung_setup_or_turn_and_missing_cancel_ack_have_bounded_local_cleanup() {
        let limits = Limits {
            rpc: Duration::from_millis(120),
            ..Limits::default()
        };
        let options = options("init-hang", None);
        let start = Instant::now();
        assert!(Client::connect_with_limits(options, &AtomicBool::new(false), limits).is_err());
        assert!(start.elapsed() < Duration::from_secs(2));
        let mut client = client();
        client.limits.idle = Duration::from_millis(120);
        let start = Instant::now();
        assert_eq!(
            client
                .turn("hang", &AtomicBool::new(false), &mut |_| {})
                .unwrap_err()
                .kind(),
            io::ErrorKind::TimedOut
        );
        assert!(start.elapsed() < Duration::from_secs(2));
        assert_closed(&client);
        let mut client = self::client();
        client.limits.interrupt = Duration::from_millis(120);
        let cancel = AtomicBool::new(false);
        let start = Instant::now();
        assert_eq!(
            client
                .turn("ignore-cancel", &cancel, &mut |event| if matches!(
                    event,
                    Event::Phase(Phase::Generating)
                ) {
                    cancel.store(true, Ordering::Relaxed);
                })
                .unwrap_err()
                .kind(),
            io::ErrorKind::TimedOut
        );
        assert!(start.elapsed() < Duration::from_secs(2));
        assert_closed(&client);
    }
    #[test]
    fn setup_cancellation_and_pre_dispatch_cancellation_do_not_claim_remote_stop() {
        let options = options("init-hang", None);
        let cancel = AtomicBool::new(false);
        thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(80));
                cancel.store(true, Ordering::Relaxed);
            });
            assert!(Client::connect_cancellable(options, &cancel).is_err());
        });
        let mut client = client();
        assert_eq!(
            client
                .turn("not sent", &AtomicBool::new(true), &mut |_| panic!(
                    "no events before dispatch"
                ))
                .unwrap(),
            TurnOutcome::NotStarted
        );
        assert_eq!(
            client
                .turn("sent once", &AtomicBool::new(false), &mut |_| {})
                .unwrap(),
            TurnOutcome::Completed
        );
    }
    #[test]
    fn restart_is_an_explicit_new_connection_with_negotiated_load() {
        let options = options("", None);
        let mut first = Client::connect(options.clone()).unwrap();
        assert!(
            first
                .turn("partial", &AtomicBool::new(false), &mut |_| {})
                .is_err()
        );
        assert_closed(&first);
        let session = first.session_id().to_owned();
        let mut second = Client::connect(Options {
            session_id: Some(session),
            ..options
        })
        .unwrap();
        assert_eq!(
            second
                .turn("explicit new input", &AtomicBool::new(false), &mut |_| {})
                .unwrap(),
            TurnOutcome::Completed
        );
        assert_eq!(second.completed_text(), Some("explicit new input"));
    }
    #[test]
    fn projection_and_capability_resource_limits_fail_explicitly() {
        assert!(
            capabilities(&json::parse(r#"{"agentCapabilities":{"loadSession":"yes"}}"#).unwrap())
                .is_err()
        );
        assert!(
            capabilities(
                &json::parse(r#"{"agentCapabilities":{"sessionCapabilities":{"resume":true}}}"#)
                    .unwrap()
            )
            .is_err()
        );
        let mut projection = Projection::default();
        for id in 0..=MAX_ITEMS {
            let value = object(vec![(
                "params",
                object(vec![
                    ("sessionId", string("session")),
                    (
                        "update",
                        object(vec![
                            ("sessionUpdate", string("tool_call")),
                            ("toolCallId", string(&format!("tool-{id}"))),
                            ("status", string("completed")),
                        ]),
                    ),
                ]),
            )]);
            assert_eq!(
                projection.update(&value, "session", &mut |_| {}).is_ok(),
                id < MAX_ITEMS
            );
        }
        let mut projection = Projection {
            text_bytes: MAX_DELTA_BYTES,
            ..Projection::default()
        };
        let value = json::parse(r#"{"params":{"sessionId":"s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"x"}}}}"#).unwrap();
        assert!(projection.update(&value, "s", &mut |_| {}).is_err());
    }
}
