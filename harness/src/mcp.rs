//! A std-only Model Context Protocol client for stdio servers.
//!
//! Servers come from `.vade/mcp.json` in the project and `mcp.json` in the
//! Vade home. Their tools reach the model as `<server>__<tool>` next to the
//! native tools, which stay unchanged. Resources and prompts are listed only.
pub mod cli;
pub mod client;
pub mod config;

use crate::agent::{ToolError, ToolExecutor, ToolInvocation};
use crate::coding_tools::Preview;
use crate::json::{self, Value};
use crate::shell::ApprovalPolicy;
use client::{CallError, Connection, ServerInfo};
use config::ServerSpec;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Separates the server name from the tool name in a model-facing tool name.
pub const SEPARATOR: &str = "__";
/// Model-facing tool names follow the common function-name rule.
pub const MAX_TOOL_NAME: usize = 64;
pub const MAX_TOOLS_PER_SERVER: usize = 128;
pub const MAX_LISTED: usize = 256;
/// One tool's description and input schema, serialized.
pub const MAX_SCHEMA_BYTES: usize = 16 * 1024;
pub const MAX_DESCRIPTION_CHARS: usize = 1024;
/// A tool result is cut to this size (or the run's smaller per-call limit).
pub const MAX_RESULT_BYTES: usize = 32 * 1024;
/// One tool definition may nest this deep; a chat request allows 48 levels.
pub const MAX_SCHEMA_DEPTH: usize = 32;
/// JSON values in all MCP tool definitions together. The chat history shares
/// a limit of 32768 values with the tool list, so MCP tools take at most a quarter.
pub const MAX_DEFINITION_VALUES: usize = 8192;
/// Bytes of all MCP tool definitions together, as JSON. The chat history
/// shares a limit of 4 MiB with the tool list, so MCP tools take at most an eighth.
pub const MAX_DEFINITION_BYTES: usize = 512 * 1024;

/// One server tool as the model sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    /// `<server>__<tool>`, with characters outside `A-Z a-z 0-9 _ -` replaced by `_`.
    pub exposed: String,
    /// The name the server knows.
    pub name: String,
    pub description: String,
    pub schema: Value,
}
impl Tool {
    /// The chat-completions function definition.
    pub fn definition(&self, server: &str) -> Value {
        let mut description = format!("MCP server {server}, tool {}.", self.name);
        if !self.description.is_empty() {
            description.push(' ');
            description.extend(self.description.chars().take(MAX_DESCRIPTION_CHARS));
        }
        Value::Object(vec![
            ("type".into(), Value::String("function".into())),
            ("function".into(), Value::Object(vec![
                ("name".into(), Value::String(self.exposed.clone())),
                ("description".into(), Value::String(description)),
                ("parameters".into(), self.schema.clone()),
            ])),
        ])
    }
}

/// The model-facing name, or `None` when it would exceed [`MAX_TOOL_NAME`].
pub fn exposed_name(server: &str, tool: &str) -> Option<String> {
    let tool: String = tool.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    let name = format!("{server}{SEPARATOR}{tool}");
    (!tool.is_empty() && name.len() <= MAX_TOOL_NAME).then_some(name)
}

/// Turn a `tools/list` page into model tools. Skipped tools become notes.
/// Large descriptions and schemas are measured before anything is copied.
pub fn parse_tools(server: &str, listed: &[Value], notes: &mut Vec<String>) -> Vec<Tool> {
    let mut tools: Vec<Tool> = Vec::new();
    for item in listed {
        if tools.len() == MAX_TOOLS_PER_SERVER {
            notes.push(format!("MCP {server}: kept the first {MAX_TOOLS_PER_SERVER} tools"));
            break;
        }
        let Some(name) = item.get("name").and_then(Value::as_str) else {
            notes.push(format!("MCP {server}: skipped a tool with no name"));
            continue;
        };
        let shown: String = crate::terminal::sanitize(name).chars().take(MAX_TOOL_NAME).collect();
        let Some(exposed) = exposed_name(server, name) else {
            notes.push(format!("MCP {server}: skipped tool {shown}: the name is longer than {MAX_TOOL_NAME} characters with the server prefix"));
            continue;
        };
        if tools.iter().any(|tool| tool.exposed == exposed) {
            notes.push(format!("MCP {server}: skipped tool {shown}: another tool has the same name {exposed}"));
            continue;
        }
        let too_large = || format!("MCP {server}: skipped tool {shown}: its definition is larger than 16 KiB");
        let schema = match item.get("inputSchema") {
            Some(schema @ Value::Object(_)) if schema.get("type").and_then(Value::as_str) == Some("object") => {
                if json::stringify(schema).map_or(true, |text| text.len() > MAX_SCHEMA_BYTES) {
                    notes.push(too_large());
                    continue;
                }
                schema.clone()
            }
            None => json::parse(r#"{"type":"object"}"#).expect("static schema"),
            _ => {
                notes.push(format!("MCP {server}: skipped tool {shown}: inputSchema is not an object schema"));
                continue;
            }
        };
        let description = item.get("description").and_then(Value::as_str).unwrap_or_default().chars().take(MAX_DESCRIPTION_CHARS).collect();
        let tool = Tool { exposed, name: name.into(), description, schema };
        if json::stringify(&tool.definition(server)).map_or(true, |text| text.len() > MAX_SCHEMA_BYTES) {
            notes.push(too_large());
            continue;
        }
        if shape(&tool.definition(server)).0 > MAX_SCHEMA_DEPTH {
            notes.push(format!("MCP {server}: skipped tool {shown}: its schema nests deeper than {MAX_SCHEMA_DEPTH} levels"));
            continue;
        }
        tools.push(tool);
    }
    tools
}

/// The nesting depth and the number of values of `value`.
fn shape(value: &Value) -> (usize, usize) {
    match value {
        Value::Array(items) => items.iter().map(shape).fold((1, 1), |(depth, values), (d, v)| (depth.max(d + 1), values + v)),
        Value::Object(fields) => fields.iter().map(|(_, item)| shape(item)).fold((1, 1), |(depth, values), (d, v)| (depth.max(d + 1), values + v)),
        _ => (1, 1),
    }
}

/// Keep tools, in config order, while all definitions together stay within
/// [`MAX_DEFINITION_VALUES`] and [`MAX_DEFINITION_BYTES`]; the rest become
/// one note per server.
fn fit_budget(servers: &mut [Server], notes: &mut Vec<String>) {
    let (mut values, mut bytes) = (0, 0);
    for server in servers {
        let before = server.tools.len();
        server.tools.retain(|tool| {
            let definition = tool.definition(&server.spec.name);
            let (more_values, more_bytes) = (shape(&definition).1, json::stringify(&definition).map_or(usize::MAX, |text| text.len()));
            let fits = values + more_values <= MAX_DEFINITION_VALUES && bytes + more_bytes <= MAX_DEFINITION_BYTES;
            if fits {
                (values, bytes) = (values + more_values, bytes + more_bytes);
            }
            fits
        });
        let left = before - server.tools.len();
        if left > 0 {
            notes.push(format!(
                "MCP {}: left out {left} tools: all MCP tool definitions together are limited to {MAX_DEFINITION_VALUES} JSON values and {} KiB",
                server.spec.name,
                MAX_DEFINITION_BYTES / 1024
            ));
        }
    }
}

/// The text the model gets for a `tools/call` result, cut to `cap` bytes.
/// Text parts are kept as is; images, audio and blobs become one line each.
/// At most `cap + 1` bytes are copied, however large the result is.
pub fn render_result(result: &Value, cap: usize) -> String {
    let mut text = String::new();
    let mut total = 0;
    let mut push = |part: &str| {
        if total > 0 {
            total += 1;
            if text.len() <= cap {
                text.push('\n');
            }
        }
        total += part.len();
        let mut end = part.len().min((cap + 1).saturating_sub(text.len()));
        while !part.is_char_boundary(end) {
            end -= 1;
        }
        text.push_str(&part[..end]);
    };
    for item in result.get("content").and_then(Value::as_array).unwrap_or_default() {
        let field = |key| item.get(key).and_then(Value::as_str).unwrap_or_default();
        match field("type") {
            "text" => push(field("text")),
            kind @ ("image" | "audio") => push(&format!("[{kind} {}, {} base64 bytes not shown]", field("mimeType"), field("data").len())),
            "resource_link" => push(&format!("[resource link {} {}]", field("uri"), field("name"))),
            "resource" => {
                let resource = item.get("resource");
                let uri = resource.and_then(|r| r.get("uri")).and_then(Value::as_str).unwrap_or_default();
                match resource.and_then(|r| r.get("text")).and_then(Value::as_str) {
                    Some(body) => {
                        push(&format!("[resource {uri}]"));
                        push(body);
                    }
                    None => push(&format!("[resource {uri}, binary not shown]")),
                }
            }
            other => push(&format!("[{} content not shown]", crate::terminal::sanitize(other))),
        }
    }
    if result.get("content").and_then(Value::as_array).is_none_or(<[_]>::is_empty)
        && let Some(structured) = result.get("structuredContent")
    {
        push(&json::stringify(structured).unwrap_or_default());
    }
    if result.get("isError") == Some(&Value::Bool(true)) {
        return error_json("mcp_tool_error", &text, total, cap);
    }
    cut(&text, total, cap)
}

/// Cut at a character boundary and say how much was kept.
pub fn truncate(text: String, cap: usize) -> String {
    if text.len() <= cap { text } else { cut(&text, text.len(), cap) }
}

/// `text` is the start of a `total`-byte output. Keep what fits in `cap`
/// with a note; below the size of the note itself, keep only text.
fn cut(text: &str, total: usize, cap: usize) -> String {
    if total <= cap && text.len() == total {
        return text.to_owned();
    }
    let note = |kept: usize| format!("\n[output cut: kept {kept} of {total} bytes]");
    let room = cap.checked_sub(note(cap).len());
    let mut kept = room.unwrap_or(cap).min(text.len());
    while !text.is_char_boundary(kept) {
        kept -= 1;
    }
    match room {
        Some(_) => format!("{}{}", &text[..kept], note(kept)),
        None => text[..kept].to_owned(),
    }
}

/// `{"ok":false,"error":..,"message":..}` in at most `cap` bytes. The message
/// is cut before it is quoted, with room for the escapes, so the JSON stays
/// closed. Only a `cap` below the size of the empty wrapper cuts the JSON.
fn error_json(error: &str, text: &str, total: usize, cap: usize) -> String {
    let wrap = |message: &str| format!("{{\"ok\":false,\"error\":\"{error}\",\"message\":{}}}", json::quote(message));
    let mut room = cap.saturating_sub(wrap("").len());
    loop {
        let wrapped = wrap(&cut(text, total, room));
        if wrapped.len() <= cap || room == 0 {
            return truncate(wrapped, cap);
        }
        room -= (wrapped.len() - cap).min(room);
    }
}

fn failure(error: &str, message: &str, cap: usize) -> String {
    error_json(error, message, message.len(), cap)
}

/// One configured server and its connection, if any.
pub struct Server {
    pub spec: ServerSpec,
    pub connection: Option<Connection>,
    pub tools: Vec<Tool>,
    pub resources: Vec<Value>,
    pub prompts: Vec<Value>,
    pub error: Option<String>,
    restarted: bool,
}
impl Server {
    pub fn info(&self) -> Option<&ServerInfo> {
        self.connection.as_ref().map(|connection| &connection.info)
    }
}

/// The running servers of one chat. Every server starts at once; a server
/// that fails to start is reported and left out.
pub struct Hub {
    root: PathBuf,
    pub servers: Vec<Server>,
    pub notes: Vec<String>,
}

/// Start one server and list what it offers. `cancel` stops each step.
fn connect(spec: &ServerSpec, root: &Path, cancel: &AtomicBool, notes: &mut Vec<String>) -> Result<(Connection, Vec<Tool>, Vec<Value>, Vec<Value>), CallError> {
    let mut connection = Connection::start(spec, root, cancel)?;
    let info = connection.info.clone();
    let tools = if info.tools {
        let listed = connection.list("tools/list", "tools", MAX_LISTED, spec.timeout, cancel)?;
        parse_tools(&spec.name, &listed, notes)
    } else {
        Vec::new()
    };
    // Resources and prompts are listed only; reading them comes later.
    let mut optional = |enabled: bool, method: &str, key: &str| {
        if !enabled {
            return Ok(Vec::new());
        }
        match connection.list(method, key, MAX_LISTED, spec.timeout, cancel) {
            Err(CallError::Cancelled) => Err(CallError::Cancelled),
            Err(error) => {
                notes.push(format!("MCP {}: {method} failed: {error}", spec.name));
                Ok(Vec::new())
            }
            listed => listed,
        }
    };
    let resources = optional(info.resources, "resources/list", "resources")?;
    let prompts = optional(info.prompts, "prompts/list", "prompts")?;
    Ok((connection, tools, resources, prompts))
}

/// Whether a failed call had already reached the server.
enum Dispatch {
    Sent(ToolError),
    NotSent(ToolError),
}
use Dispatch::{NotSent, Sent};

impl Hub {
    /// Start every enabled, trusted server in `specs` in parallel.
    pub fn start(specs: Vec<ServerSpec>, root: &Path) -> Self {
        let mut notes = Vec::new();
        // Startup runs before the chat has a run to cancel.
        let never = AtomicBool::new(false);
        let never = &never;
        let started: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = specs
                .iter()
                .map(|spec| {
                    let runnable = spec.enabled && spec.trusted;
                    runnable.then(|| scope.spawn(move || {
                        let mut notes = Vec::new();
                        (connect(spec, root, never, &mut notes), notes)
                    }))
                })
                .collect();
            handles.into_iter().map(|handle| handle.map(|h| h.join().expect("MCP start thread"))).collect()
        });
        let mut servers = Vec::new();
        for (spec, started) in specs.into_iter().zip(started) {
            let mut server = Server { spec, connection: None, tools: Vec::new(), resources: Vec::new(), prompts: Vec::new(), error: None, restarted: false };
            match started {
                None if !server.spec.enabled => {}
                None => notes.push(format!(
                    "MCP: project server {} is not trusted yet, so it did not start; run vade mcp trust {} after you check its command",
                    server.spec.name, server.spec.name
                )),
                Some((Ok((connection, tools, resources, prompts)), found)) => {
                    notes.extend(found);
                    (server.connection, server.tools, server.resources, server.prompts) = (Some(connection), tools, resources, prompts);
                }
                Some((Err(error), found)) => {
                    notes.extend(found);
                    notes.push(format!("MCP: server {} did not start: {error}", server.spec.name));
                    server.error = Some(error.to_string());
                }
            }
            servers.push(server);
        }
        fit_budget(&mut servers, &mut notes);
        Self { root: root.to_path_buf(), servers, notes }
    }

    /// Load the config for `root` and start its servers. `VADE_MCP=off` starts
    /// none. `home` is the chat's Vade home, if any; `starting` gets the
    /// number of servers before they start.
    pub fn from_config(root: &Path, home: Option<&Path>, starting: impl FnOnce(usize)) -> Self {
        if std::env::var_os("VADE_MCP").is_some_and(|value| value == "off") {
            return Self { root: root.to_path_buf(), servers: Vec::new(), notes: Vec::new() };
        }
        let loaded = config::load(root, home);
        let runnable = loaded.servers.iter().filter(|spec| spec.enabled && spec.trusted).count();
        if runnable > 0 {
            starting(runnable);
        }
        let mut hub = Self::start(loaded.servers, root);
        hub.notes.splice(0..0, loaded.notes);
        hub
    }

    /// Function definitions for every listed tool, in config order.
    pub fn definitions(&self) -> Vec<Value> {
        self.servers.iter().flat_map(|server| server.tools.iter().map(|tool| tool.definition(&server.spec.name))).collect()
    }

    /// Run one `<server>__<tool>` call. `Ok` holds the model-facing text,
    /// including tool errors. The bool is true when the outcome is unknown:
    /// a timeout, or a cancel or server stop after the request was sent. It
    /// goes with `Err(Cancelled)` too, because a cancel notice does not prove
    /// that the server stopped its work.
    pub fn call(&mut self, exposed: &str, arguments: &Value, cap: usize, cancel: &AtomicBool) -> (Result<String, ToolError>, bool) {
        match self.call_inner(exposed, arguments, cap, cancel) {
            Ok((text, uncertain)) => (Ok(text), uncertain),
            Err(Sent(error)) => (Err(error), true),
            Err(NotSent(error)) => (Err(error), false),
        }
    }

    fn call_inner(&mut self, exposed: &str, arguments: &Value, cap: usize, cancel: &AtomicBool) -> Result<(String, bool), Dispatch> {
        let cap = cap.min(MAX_RESULT_BYTES);
        let unavailable = |message: String| Ok((failure("mcp_unavailable", &message, cap), false));
        let Some((server_name, _)) = exposed.split_once(SEPARATOR) else { return Err(NotSent(ToolError::UnknownTool)) };
        let Some(server) = self.servers.iter_mut().find(|server| server.spec.name == server_name) else {
            return unavailable(format!("MCP server {server_name} is not configured in this chat"));
        };
        if !(server.spec.enabled && server.spec.trusted) {
            return unavailable(format!("MCP server {server_name} is disabled or not trusted in this chat"));
        }
        // The chat's tool list is fixed when the chat starts, so a server that
        // did not start then has no tool to call; starting it now adds none.
        if !server.tools.iter().any(|tool| tool.exposed == exposed) {
            return unavailable(if server.error.is_some() {
                format!("MCP server {server_name} did not start when this chat began, so this chat has none of its tools; restart the chat to start it again")
            } else {
                format!("MCP server {server_name} does not offer {exposed} in this chat")
            });
        }
        // A server that stopped gets one more start per chat.
        if !server.connection.as_mut().is_some_and(Connection::alive) {
            if server.restarted {
                return unavailable(format!("MCP server {server_name} stopped again after one restart; restart the chat to start it again"));
            }
            server.restarted = true;
            server.connection = None;
            let mut notes = Vec::new();
            match connect(&server.spec, &self.root, cancel, &mut notes) {
                Ok((connection, tools, _, _)) => {
                    // The chat's tool list is fixed: a restart adds no tool and
                    // changes no definition; it only drops a tool the server no
                    // longer lists. The kept definitions passed the budgets at
                    // the start, so a fresh conversation still fits.
                    server.tools.retain(|old| tools.iter().any(|tool| tool.exposed == old.exposed));
                    (server.connection, server.error) = (Some(connection), None);
                }
                Err(CallError::Cancelled) => {
                    // A canceled start does not use the one restart.
                    server.restarted = false;
                    return Err(NotSent(ToolError::Cancelled));
                }
                Err(error) => return unavailable(format!("MCP server {server_name} stopped and did not restart: {error}")),
            }
        }
        let Some(tool) = server.tools.iter().find(|tool| tool.exposed == exposed) else {
            return unavailable(format!("MCP server {server_name} does not offer {exposed} now"));
        };
        let name = tool.name.clone();
        if cancel.load(Ordering::Relaxed) {
            return Err(NotSent(ToolError::Cancelled));
        }
        let connection = server.connection.as_mut().expect("connected above");
        match connection.call_tool(&name, arguments, server.spec.timeout, cancel, &mut |_| {}) {
            Ok(result) => Ok((render_result(&result, cap), false)),
            // The request is already in the server's input.
            Err(CallError::Cancelled) => Err(Sent(ToolError::Cancelled)),
            Err(CallError::Timeout) => Ok((
                failure("mcp_timeout", &format!("MCP tool {name} did not answer within {} ms; the call was canceled and its outcome is unknown", server.spec.timeout.as_millis()), cap),
                true,
            )),
            Err(error @ (CallError::Exited(_) | CallError::Stalled | CallError::Malformed(_))) => {
                Ok((failure("mcp_server_failed", &format!("MCP server {server_name} failed during {name}: {error}; the outcome is unknown"), cap), true))
            }
            Err(error) => Ok((failure("mcp_call_failed", &format!("MCP tool {name}: {error}"), cap), false)),
        }
    }
}

/// A tool executor that sends `<server>__<tool>` calls to the hub and every
/// other call to `inner`, unchanged. MCP calls follow the approval policy:
/// `ask` asks before each call, `auto` runs it, `never` denies it.
pub struct McpTools<'a, E, F> {
    pub inner: E,
    hub: Option<&'a Mutex<Hub>>,
    policy: ApprovalPolicy,
    preview: F,
    mcp_last: bool,
    uncertain: bool,
}
impl<'a, E, F> McpTools<'a, E, F>
where
    F: FnMut(Preview<'_>, &AtomicBool) -> bool,
{
    pub fn new(inner: E, hub: Option<&'a Mutex<Hub>>, policy: ApprovalPolicy, preview: F) -> Self {
        Self { inner, hub, policy, preview, mcp_last: false, uncertain: false }
    }
}
impl<E: ToolExecutor, F> ToolExecutor for McpTools<'_, E, F>
where
    F: FnMut(Preview<'_>, &AtomicBool) -> bool,
{
    fn missing_deliverables(&self, task: &str) -> Vec<String> {
        self.inner.missing_deliverables(task)
    }
    fn execute(&mut self, invocation: ToolInvocation<'_>, cancel: &AtomicBool) -> Result<String, ToolError> {
        let name = invocation.call.name.as_str();
        self.mcp_last = name.contains(SEPARATOR);
        if !self.mcp_last {
            return self.inner.execute(invocation, cancel);
        }
        self.uncertain = false;
        let Some(hub) = self.hub else {
            return Ok(failure("mcp_unavailable", "MCP servers are off in this chat", invocation.max_output_bytes));
        };
        if self.policy == ApprovalPolicy::Never {
            return Err(ToolError::Denied);
        }
        let arguments = json::stringify(invocation.arguments).map_err(|_| ToolError::InvalidArguments("arguments must be JSON"))?;
        let (server, tool) = name.split_once(SEPARATOR).unwrap_or((name, ""));
        // Bound the preview like the native tools do; the call itself gets every byte.
        let detail = format!("MCP server {server}, tool {tool}\nArguments: {}", truncate(arguments, 4096));
        if !(self.preview)(Preview { title: "MCP tool", detail: &detail, approval_required: self.policy == ApprovalPolicy::Ask }, cancel) {
            return Err(ToolError::Denied);
        }
        let mut hub = hub.lock().map_err(|_| ToolError::Failed)?;
        let (result, uncertain) = hub.call(name, invocation.arguments, invocation.max_output_bytes, cancel);
        self.uncertain = uncertain;
        result
    }
    fn last_side_effect_uncertain(&self) -> bool {
        if self.mcp_last { self.uncertain } else { self.inner.last_side_effect_uncertain() }
    }
    fn take_pending_check(&mut self) -> Option<Value> {
        self.inner.take_pending_check()
    }
    fn take_completion_check(&mut self) -> Option<Value> {
        self.inner.take_completion_check()
    }
    fn check_failing(&self) -> bool {
        self.inner.check_failing()
    }
    fn check_enabled(&self) -> bool {
        self.inner.check_enabled()
    }
    fn check_not_run(&self) -> bool {
        self.inner.check_not_run()
    }
}

#[cfg(test)]
mod tests;
