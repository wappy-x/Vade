//! Native, bounded tool-aware orchestration. No transport, filesystem or shell
//! access is performed here. Adapters own I/O deadlines and workspace policy.
//!
//! A model turn is staged before history changes. Every committed call gets a
//! result, including skipped/cancelled calls. Failed streams are discarded;
//! completed calls are never replayed when `run` is explicitly resumed.
use crate::agent_wire::{
    AssistantTurn, FinishReason, MAX_HISTORY_NODES, Message, ToolCall, WireLimits, json_nodes,
    required_string,
};
use crate::json::{self, Value, invalid};
use std::collections::HashSet;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

mod execution;
mod recovery;
pub use execution::{run, run_observed, run_observed_with_budget};
pub use recovery::has_visible_answer;

/// Shared with the interactive `/continue` command. Keep recorded user messages
/// identical whether the host or the user requests continuation. Plain chat
/// without tools continues prose; tool sessions resume work instead of planning.
pub const CONTINUATION_PROMPT: &str = "Continue your previous answer from where it was cut off. Avoid repeating the opening; finish the remaining explanation.";
pub const TOOL_CONTINUATION_PROMPT: &str = "Continue. If you were still planning, stop planning and make the next concrete edit or tool call now, in small parts.";
pub const REPLY_BUDGET_PROMPT: &str = "The local reply budget ended. No incomplete tool proposal was executed. Continue in smaller steps: make the next concrete edit or tool call in small parts; avoid repeating the discarded draft.";
pub const MAX_AUTO_CONTINUES: usize = 32;
/// Escalating truncated-call recoveries per tool/target hint per task,
/// derived from journaled history so the bound survives resume.
pub const MAX_TRUNCATED_RETRIES: usize = 3;
/// A finished turn with no visible text and no tool call (for example hidden
/// reasoning only) is never accepted silently: the host nudges at most twice.
pub const EMPTY_REPLY_PROMPT: &str = "Your last reply had no visible answer and no tool call. Continue the task now: make the next tool call, or give your final answer as visible text.";
pub const MAX_EMPTY_NUDGES: usize = 2;
/// A final answer while the user's automatic check still fails gets at most two reminders.
pub const CHECK_FAILING_PROMPT: &str = "The automatic check still fails (see the latest vade-check result). Fix the reported errors and let the check pass before finishing, or state exactly why it cannot pass.";
pub const MAX_CHECK_NUDGES: usize = 2;
/// One completion review per conversation, retained as an exact user message
/// through journal resume and compaction. Passing checks can miss requirements.
pub const SELF_REVIEW_PROMPT: &str = "Re-read the task requirements and check each one against the current code. List any requirement not covered and fix it; otherwise say DONE. A passing native check alone does not establish that every requirement is covered. This is the single automatic self-review for this task; keep it brief and use the remaining budget.";
/// One recovery opportunity, persisted as user history across resume/compaction.
pub const PATCH_RECOVERY_PROMPT: &str = "The same apply_patch argument bytes were refused three times. Stop resending that proposal. Use a different tool now: workspace_read_lines to inspect an existing target, or workspace_list for a new target's parent. On macOS, workspace_shell can write a complete new file with explicit argv inside the usual sandbox and approval policy; otherwise send a different small canonical patch using the previous receipt's example. Do not run an external patch utility. Checks and all remaining quotas still apply. This is the single automatic patch recovery; another repeated refusal ends this activation.";
/// Call IDs of harness-authored automatic check calls.
pub const CHECK_CALL_PREFIX: &str = "vade-check-";
/// One reminder per no-edit discovery stretch, retained across resume.
pub const DISCOVERY_RECOVERY_PROMPT: &str = "The same discovery call succeeded three times in a row. Use its existing results instead of repeating it. Read the relevant lines or narrow the query, then make the requested change or create the requested output file. For a read-only question, give the answer once the evidence is sufficient. Keep the existing time and tool budgets. This is the single automatic discovery recovery; another repeated discovery loop ends this activation.";

/// Budgets for one user request: model and tool turns, tool calls, active
/// work time, output and history bytes, context tokens and wire bounds.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_model_turns: usize,
    pub max_tool_turns: usize,
    pub max_tool_calls: usize,
    /// Active work for one user request; `None` or zero disables the deadline.
    /// Hosts pause only after presenting approval and while awaiting a decision.
    pub max_run_time: Option<Duration>,
    pub max_tool_output_bytes: usize,
    /// Includes generated error results as well as successful output.
    pub max_total_output_bytes: usize,
    /// Full serialized request, including tools/options/system messages.
    pub max_history_bytes: usize,
    pub max_context_tokens: usize,
    pub output_reserve_tokens: usize,
    /// Continue length-limited prose within the same request's existing budget.
    pub auto_continue: bool,
    pub wire: WireLimits,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_model_turns: 61,
            max_tool_turns: 60,
            max_tool_calls: 240,
            max_run_time: Some(Duration::from_secs(20 * 60)),
            max_tool_output_bytes: 64 * 1024,
            max_total_output_bytes: 256 * 1024,
            max_history_bytes: 4 * 1024 * 1024,
            max_context_tokens: crate::context::DEFAULT_CONTEXT_TOKENS,
            output_reserve_tokens: crate::context::DEFAULT_RESERVE,
            auto_continue: true,
            wire: WireLimits::default(),
        }
    }
}

/// A single request's active-work clock. Clones share the same clock, allowing
/// parent and child previews to share a pause after the UI presents an approval.
/// Queueing, presentation and automatic denials remain charged to the request.
/// Human response time is excluded. Create a fresh budget for continuation.
#[derive(Clone)]
pub struct RunBudget {
    state: Arc<Mutex<BudgetState>>,
}

struct BudgetState {
    limit: Option<Duration>,
    started: Instant,
    paused_since: Option<Instant>,
    paused_total: Duration,
    pauses: usize,
    expired: bool,
}

impl RunBudget {
    pub fn new(limit: Option<Duration>) -> Self {
        Self {
            state: Arc::new(Mutex::new(BudgetState {
                limit: limit.filter(|duration| !duration.is_zero()),
                started: Instant::now(),
                paused_since: None,
                paused_total: Duration::ZERO,
                pauses: 0,
                expired: false,
            })),
        }
    }

    /// Hold only while waiting for a user's approval. Nested guards exclude
    /// overlapping approval waits once; dropping the final guard resumes work.
    /// Cancellation remains active while paused, so users can still stop a run.
    pub fn pause(&self) -> RunPause {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.pauses == 0 {
            state.paused_since = Some(Instant::now());
        }
        state.pauses += 1;
        RunPause {
            budget: self.clone(),
        }
    }

    fn enabled(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .limit
            .is_some()
    }

    fn remaining(&self) -> Option<Duration> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let limit = state.limit?;
        let now = state.paused_since.unwrap_or_else(Instant::now);
        let elapsed = now
            .saturating_duration_since(state.started)
            .saturating_sub(state.paused_total);
        let remaining = limit.saturating_sub(elapsed);
        if remaining.is_zero() {
            state.expired = true;
        }
        Some(remaining)
    }

    /// Unlimited requests do not opt into extra completion work. Use the
    /// host-owned active clock, including its excluded approval waits.
    fn quarter_remaining(&self) -> bool {
        let limit = self.state.lock().unwrap_or_else(|e| e.into_inner()).limit;
        limit.is_some_and(|limit| self.remaining().is_some_and(|left| {
            left >= limit / 4 && left >= Duration::from_secs(10)
        }))
    }

    /// Past the halfway point with at least a quarter provably left: late
    /// enough that waiting for a final answer risks losing the review to
    /// the deadline, early enough that the review can still act.
    fn review_window(&self) -> bool {
        let limit = self.state.lock().unwrap_or_else(|e| e.into_inner()).limit;
        limit.is_some_and(|limit| {
            self.remaining().is_some_and(|left| {
                left < limit / 2 && left >= limit / 4 && left >= Duration::from_secs(10)
            })
        })
    }

    pub fn expired(&self) -> bool {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).expired
    }

    /// The deadline's active-work clock, excluding presented approval waits.
    pub fn active_elapsed(&self) -> Duration {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state
            .paused_since
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(state.started)
            .saturating_sub(state.paused_total)
    }

    /// Apply the same clock to admission/counting and other work surrounding
    /// generation. Reusing the budget across phases preserves one request limit.
    /// The scoped watchdog is always joined before returning or unwinding.
    pub fn with_cancellation<T>(&self, cancel: &AtomicBool, work: impl FnOnce() -> T) -> T {
        if !self.enabled() {
            return work();
        }
        self.signal_expiry(cancel);
        std::thread::scope(|scope| {
            let (stop, stopped) = mpsc::channel();
            scope.spawn(move || {
                loop {
                    if cancelled(cancel) {
                        break;
                    }
                    let remaining = self.remaining().unwrap_or(Duration::from_millis(25));
                    if remaining.is_zero() {
                        cancel.store(true, Ordering::Release);
                        break;
                    }
                    // Short checks also observe cancellation and newly paused
                    // approval clocks; shutdown wakes immediately on return.
                    match stopped.recv_timeout(remaining.min(Duration::from_millis(25))) {
                        Err(mpsc::RecvTimeoutError::Timeout) => (),
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            });
            let result = work();
            self.signal_expiry(cancel);
            let _ = stop.send(());
            result
        })
    }

    fn signal_expiry(&self, cancel: &AtomicBool) {
        if !cancelled(cancel) && self.remaining().is_some_and(|left| left.is_zero()) {
            cancel.store(true, Ordering::Release);
        }
    }
}

/// Approval pause guard. The shared budget keeps running after the last guard
/// drops, including when an approval callback exits early or unwinds.
pub struct RunPause {
    budget: RunBudget,
}

impl Drop for RunPause {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock().unwrap_or_else(|e| e.into_inner());
        state.pauses -= 1;
        if state.pauses == 0 {
            if let Some(started) = state.paused_since.take() {
                state.paused_total += started.elapsed();
            }
        }
    }
}

impl Limits {
    fn validate(self) -> io::Result<()> {
        if self.max_context_tokens < 1024
            || self.output_reserve_tokens >= self.max_context_tokens
            || self.max_model_turns > 1024
            || self.max_tool_turns > 1024
            || self.max_tool_calls > 4096
            || self.wire.max_calls > 256
            || self.wire.max_argument_bytes > json::MAX_INPUT_BYTES
            || self.wire.max_stream_bytes > 64 * 1024 * 1024
            || self.wire.max_events > 1_000_000
            || !(1024..=json::MAX_INPUT_BYTES).contains(&self.wire.max_message_bytes)
            || !(128..=json::MAX_INPUT_BYTES).contains(&self.max_tool_output_bytes)
            || self.max_total_output_bytes > json::MAX_INPUT_BYTES
            || self.max_history_bytes > json::MAX_INPUT_BYTES
        {
            return Err(invalid("invalid agent limits"));
        }
        Ok(())
    }
}

/// Immutable options/tools and append-only full-fidelity messages. Construction
/// accepts a complete chat request; unfamiliar root/message fields are kept.
/// The caller supplies model, tools, stream settings and sampling options.
#[derive(Clone, Debug)]
pub struct Conversation {
    options: Value,
    options_json: String,
    tool_bytes: usize,
    messages: Vec<Message>,
    bytes: usize,
    nodes: usize,
    prefix_len: usize,
    tool_names: HashSet<String>,
    call_ids: HashSet<String>,
    // Private, optional acceleration metadata. Never part of the model prompt.
    pub(crate) kv_hints: Vec<crate::kv::Hint>,
}

impl Conversation {
    pub fn from_request(request: Value, limits: Limits) -> io::Result<Self> {
        limits.validate()?;
        // Validate duplicate fields, JSON complexity, and the whole input before
        // splitting it into separately retained options and messages.
        if json::stringify(&request)?.len() > limits.max_history_bytes {
            return Err(invalid("agent request exceeds history byte limit"));
        }
        let Value::Object(mut fields) = request else {
            return Err(invalid("agent request must be an object"));
        };
        let index = fields
            .iter()
            .position(|(key, _)| key == "messages")
            .ok_or_else(|| invalid("agent request requires messages"))?;
        let Value::Array(raw_messages) = fields.remove(index).1 else {
            return Err(invalid("messages must be an array"));
        };
        let options = Value::Object(fields);
        if let Some(n) = options.get("n") {
            if n != &Value::Number("1".into()) {
                return Err(invalid("agent mode requires n=1"));
            }
        }
        let tool_names = advertised_tools(&options)?;
        let messages = raw_messages
            .into_iter()
            .map(|value| Message::from_value(value, limits.wire))
            .collect::<io::Result<Vec<_>>>()?;
        let call_ids = validate_history(&messages)?;
        let prefix_len = messages
            .iter()
            .take_while(|message| matches!(message.role(), "system" | "developer"))
            .count();
        let options_json = json::stringify(&options)?;
        let tool_bytes = json::view(&options_json)?
            .get("tools")
            .map_or(0, |v| v.raw().len());
        let nodes =
            json_nodes(&options)? + 1 + messages.iter().map(|message| message.nodes).sum::<usize>();
        if nodes > MAX_HISTORY_NODES {
            return Err(invalid("agent history exceeds JSON complexity limit"));
        }
        let mut result = Self {
            options,
            options_json,
            tool_bytes,
            messages,
            bytes: 0,
            nodes,
            prefix_len,
            tool_names,
            call_ids,
            kv_hints: Vec::new(),
        };
        result.bytes = result.request(limits.wire).to_json()?.len();
        if result.bytes > limits.max_history_bytes {
            return Err(invalid("agent request exceeds history byte limit"));
        }
        Ok(result)
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
    /// Drop only the final prose answer. Completed tool groups remain exact.
    pub fn without_last_answer(&self, limits: Limits) -> io::Result<Self> {
        let last = self
            .messages
            .last()
            .ok_or_else(|| invalid("no answer to retry"))?;
        if last.role() != "assistant" || !last.tool_calls().is_empty() {
            return Err(invalid("last message is not a completed prose answer"));
        }
        self.with_messages(self.messages[..self.messages.len() - 1].to_vec(), limits)
    }
    /// Deliberate context rewrite with already validated immutable messages.
    /// Retain their cached wire bytes instead of rebuilding old JSON trees.
    pub(crate) fn with_messages(&self, messages: Vec<Message>, limits: Limits) -> io::Result<Self> {
        limits.validate()?;
        for message in &messages {
            if message.serialized_bytes() > limits.wire.max_message_bytes
                || message.tool_calls().len() > limits.wire.max_calls
                || message
                    .tool_calls()
                    .iter()
                    .map(|c| c.arguments.len())
                    .sum::<usize>()
                    > limits.wire.max_argument_bytes
            {
                return Err(invalid("retained message exceeds new wire limits"));
            }
        }
        let call_ids = validate_history(&messages)?;
        let nodes =
            json_nodes(&self.options)? + 1 + messages.iter().map(|m| m.nodes).sum::<usize>();
        if nodes > MAX_HISTORY_NODES {
            return Err(invalid("agent history exceeds JSON complexity limit"));
        }
        let prefix_len = messages
            .iter()
            .take_while(|m| matches!(m.role(), "system" | "developer"))
            .count();
        let mut result = Self {
            options: self.options.clone(),
            options_json: self.options_json.clone(),
            tool_bytes: self.tool_bytes,
            messages,
            bytes: 0,
            nodes,
            prefix_len,
            tool_names: self.tool_names.clone(),
            call_ids,
            kv_hints: Vec::new(), // a rewrite invalidates later causal KV
        };
        result.bytes = result.request(limits.wire).to_json()?.len();
        if result.bytes > limits.max_history_bytes {
            return Err(invalid("agent request exceeds history byte limit"));
        }
        Ok(result)
    }
    pub fn checkpoint_hints(&self) -> &[crate::kv::Hint] {
        &self.kv_hints
    }
    /// Transfer only checkpoints whose full submitted message prefix is shared.
    /// Engine still verifies rendered tokens. A prefix-only child cannot inherit
    /// a parent checkpoint that includes a task it did not receive.
    pub fn inherit_checkpoint_hints(&mut self, parent: &Conversation) {
        self.kv_hints.clear();
        if self.options_json != parent.options_json {
            return;
        }
        let shared = self
            .messages
            .iter()
            .zip(&parent.messages)
            .take_while(|(a, b)| a.serialized_json() == b.serialized_json())
            .count();
        self.kv_hints.extend(
            parent
                .kv_hints
                .iter()
                .filter(|h| h.messages <= shared)
                .cloned(),
        );
    }
    pub fn options(&self) -> &Value {
        &self.options
    }
    pub(crate) fn serialized_tool_bytes(&self) -> usize {
        self.tool_bytes
    }
    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }
    pub fn serialized_bytes(&self) -> usize {
        self.bytes
    }
    pub fn request(&self, wire: WireLimits) -> ModelRequest<'_> {
        ModelRequest {
            options: &self.options,
            options_json: &self.options_json,
            messages: &self.messages,
            wire,
            kv_hints: &self.kv_hints,
        }
    }

    /// A new user message may include content parts and arbitrary extensions.
    /// It cannot smuggle tool results or modify the stable system/tools prefix.
    pub fn push_user(&mut self, value: Value, limits: Limits) -> io::Result<()> {
        limits.validate()?;
        let message = Message::from_value(value, limits.wire)?;
        if message.role() != "user" {
            return Err(invalid("push_user requires a user message"));
        }
        if !self.fits(std::slice::from_ref(&message), limits) {
            return Err(invalid("agent request exceeds history byte limit"));
        }
        self.commit(vec![message]);
        Ok(())
    }

    /// Keep only visible prose from an interrupted generation. No tool call is
    /// reconstructed or executed; completed tool receipts already remain closed.
    /// Reserve the serialized marker before fitting a UTF-8 prefix of the prose.
    /// Returns whether prose was truncated; a refusal leaves history unchanged.
    pub fn push_interrupted_answer(&mut self, text: &str, limits: Limits) -> io::Result<bool> {
        limits.validate()?;
        let answer = |text: &str| Message::from_value(Value::Object(vec![
            ("role".into(), Value::String("assistant".into())),
            ("content".into(), Value::String(format!("{text}\n[interrupted]"))),
        ]), limits.wire);
        let marker = answer("")?;
        if !self.fits(std::slice::from_ref(&marker), limits) {
            return Err(invalid("agent history has no room for an interruption marker"));
        }
        let mut room = (limits.max_history_bytes - self.bytes_after(std::slice::from_ref(&marker)))
            .min(limits.wire.max_message_bytes - marker.serialized_bytes());
        let mut end = 0;
        for ch in text.chars() {
            let mut utf8 = [0; 4];
            let bytes = json::quoted_len(ch.encode_utf8(&mut utf8))? - 2;
            if bytes > room { break; }
            room -= bytes;
            end += ch.len_utf8();
        }
        let mut messages = self.messages.clone();
        messages.push(answer(&text[..end])?);
        *self = self.with_messages(messages, limits)?;
        Ok(end < text.len())
    }

    fn bytes_after(&self, appended: &[Message]) -> usize {
        appended
            .iter()
            .enumerate()
            .fold(self.bytes, |bytes, (index, message)| {
                bytes
                    .saturating_add(message.serialized_bytes())
                    .saturating_add(usize::from(!self.messages.is_empty() || index > 0))
            })
    }

    fn fits(&self, appended: &[Message], limits: Limits) -> bool {
        self.bytes_after(appended) <= limits.max_history_bytes
            && self.nodes + appended.iter().map(|message| message.nodes).sum::<usize>()
                <= MAX_HISTORY_NODES
    }

    /// Maximum unescaped receipt bytes whose serialized envelope is guaranteed
    /// to fit both history and wire limits. Reserve against the closed batch,
    /// including results for every later call, before allowing any side effect.
    fn receipt_capacity(&self, staged: &[Message], index: usize, limits: Limits) -> usize {
        let old = &staged[index + 1];
        let Ok(empty) = Message::tool_result(
            old.tool_call_id().expect("staged tool result"),
            String::new(),
            limits.wire,
        ) else {
            return 0;
        };
        let history_base = self
            .bytes_after(staged)
            .saturating_sub(old.serialized_bytes())
            .saturating_add(empty.serialized_bytes());
        let history_room = limits.max_history_bytes.saturating_sub(history_base);
        let wire_room = limits
            .wire
            .max_message_bytes
            .saturating_sub(empty.serialized_bytes());
        // JSON can expand each byte to six bytes (a control character becomes
        // \u00XX). UTF-8, quotes and backslashes expand by less. Content is a
        // string, so replacing it does not increase the message's AST nodes.
        // This also reserves wrapped uncertainty markers within the same cap.
        history_room.min(wire_room) / 6
    }

    fn commit(&mut self, appended: Vec<Message>) {
        self.bytes = self.bytes_after(&appended);
        self.nodes += appended.iter().map(|message| message.nodes).sum::<usize>();
        for message in &appended {
            self.call_ids
                .extend(message.tool_calls().iter().map(|call| call.id.clone()));
        }
        self.messages.extend(appended);
    }
}

fn advertised_tools(options: &Value) -> io::Result<HashSet<String>> {
    let mut names = HashSet::new();
    if let Some(tools) = options.get("tools") {
        let Value::Array(tools) = tools else {
            return Err(invalid("tools must be an array"));
        };
        for tool in tools {
            if required_string(tool, "type")? != "function" {
                return Err(invalid("unsupported advertised tool type"));
            }
            let function = tool
                .get("function")
                .ok_or_else(|| invalid("tool definition has no function"))?;
            let name = required_string(function, "name")?;
            if name.is_empty() || !names.insert(name.to_owned()) {
                return Err(invalid("empty or duplicate tool definition"));
            }
        }
    }
    Ok(names)
}

fn validate_history(messages: &[Message]) -> io::Result<HashSet<String>> {
    let mut seen = HashSet::new();
    let mut pending = HashSet::new();
    for message in messages {
        if let Some(id) = message.tool_call_id() {
            if !pending.remove(id) {
                return Err(invalid("tool result has no pending call ID"));
            }
            continue;
        }
        if !pending.is_empty() {
            return Err(invalid(
                "tool calls require results before the next message",
            ));
        }
        for call in message.tool_calls() {
            if !seen.insert(call.id.clone()) {
                return Err(invalid("reused tool call ID in history"));
            }
            pending.insert(call.id.clone());
        }
    }
    if !pending.is_empty() {
        return Err(invalid("history contains unresolved tool calls"));
    }
    Ok(seen)
}

/// What a [`Model`] sends for one turn: request options, the message history
/// and the wire bounds its response must respect.
#[derive(Clone, Copy)]
pub struct ModelRequest<'a> {
    pub options: &'a Value,
    pub messages: &'a [Message],
    pub wire: WireLimits,
    options_json: &'a str,
    pub(crate) kv_hints: &'a [crate::kv::Hint],
}

impl ModelRequest<'_> {
    pub fn serialized_options(&self) -> &str {
        self.options_json
    }

    /// Serialize borrowed messages without cloning the full conversation tree.
    /// The options/tools prefix is serialized once when the conversation starts.
    pub fn to_json(&self) -> io::Result<String> {
        let mut out = self.options_json[..self.options_json.len() - 1].to_owned();
        if out.len() > 1 {
            out.push(',');
        }
        out.push_str("\"messages\":[");
        for (index, message) in self.messages.iter().enumerate() {
            let needed = out
                .len()
                .saturating_add(message.serialized_bytes())
                .saturating_add(3);
            if needed > json::MAX_INPUT_BYTES {
                return Err(invalid("agent request exceeds JSON byte limit"));
            }
            if index > 0 {
                out.push(',');
            }
            out.push_str(message.serialized_json());
        }
        out.push_str("]}");
        if out.len() > json::MAX_INPUT_BYTES {
            return Err(invalid("agent request exceeds JSON byte limit"));
        }
        Ok(out)
    }
}

/// Implement with a bounded local HTTP adapter (or a mock). For SSE use
/// `agent_wire::ChatAccumulator`; JSON fallback uses `AssistantTurn::from_response`.
/// Check cancellation/deadlines while waiting for I/O and propagate callback
/// errors immediately. `std` cannot forcibly interrupt an arbitrary callback.
/// Text callbacks are provisional UI output; only the returned, validated turn
/// can become history. Do not persist the displayed deltas as a second message.
pub trait Model {
    /// Remaining active task time, refreshed before each generation.
    fn set_work_remaining(&mut self, _: Option<Duration>) {}
    /// Latest validated engine occupancy after a completed response.
    fn context_usage(&self) -> Option<(u64, u64)> {
        None
    }
    fn prompt_rewritten(&mut self) {}
    /// Latest reservation selected by an adaptive adapter during token counting.
    /// Other adapters retain the caller's fixed reservation.
    fn output_reserve_tokens(&self, fallback: usize) -> usize {
        fallback
    }
    /// Actual generation cap, which may be below a compaction reservation.
    fn reply_limit_tokens(&self, fallback: usize) -> usize {
        self.output_reserve_tokens(fallback)
    }
    /// Optional renderer/tokenizer count without inference. A context refusal
    /// still returns its validated count so the host can compact first.
    fn count_input(
        &mut self,
        _: ModelRequest<'_>,
        _: &AtomicBool,
    ) -> io::Result<crate::contract::InputCount> {
        Ok(crate::contract::InputCount::Unknown)
    }
    /// Optional cache warm after a persisted rewrite. Never commit the reply or
    /// execute tools from it; ordinary adapters need no warm request.
    fn warm_compacted(&mut self, _: ModelRequest<'_>, _: &AtomicBool) -> io::Result<()> {
        Ok(())
    }
    /// Optional validated checkpoint metadata from the last completed response.
    /// Default keeps non-Engine adapters and existing mocks unchanged.
    fn checkpoint_hints(&self) -> Option<&[crate::kv::Hint]> {
        None
    }
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn>;
}

/// One validated tool call handed to a [`ToolExecutor`].
pub struct ToolInvocation<'a> {
    pub call: &'a ToolCall,
    /// Parsed JSON object. Invalid/oversized arguments never reach an executor.
    pub arguments: &'a Value,
    pub max_output_bytes: usize,
}

/// Why a tool call produced no result; each kind becomes a tool message the
/// model sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolError {
    UnknownTool,
    Denied,
    InvalidArguments(&'static str),
    Cancelled,
    Failed,
    OutputLimit,
}

/// Executors must enforce workspace boundaries/allowlists and honor cancellation
/// and the output bound. This module grants no shell or filesystem capability.
pub trait ToolExecutor {
    /// Read-only inspection of explicit task outputs before accepting completion.
    fn missing_deliverables(&self, _task: &str) -> Vec<String> { Vec::new() }

    fn execute(
        &mut self,
        invocation: ToolInvocation<'_>,
        cancel: &AtomicBool,
    ) -> Result<String, ToolError>;

    /// The last call returned a receipt but may have stopped after partial side
    /// effects. Keep that receipt while requiring reconciliation before retry.
    /// Executors reset this for each invocation; generic errors are separately
    /// treated as potentially uncertain by the scheduler.
    fn last_side_effect_uncertain(&self) -> bool {
        false
    }

    /// A user-configured check owed after a committed round changed files:
    /// `workspace_shell` arguments, returned once per change. The scheduler runs
    /// it as a journaled harness call under the same approval and sandbox.
    fn take_pending_check(&mut self) -> Option<Value> {
        None
    }

    /// First configured check before completion, even if no file changed.
    /// Uses the same journal, sandbox, approval and output quotas as edit checks.
    fn take_completion_check(&mut self) -> Option<Value> {
        None
    }

    /// The latest owed check failed, did not finish, or was not run.
    fn check_failing(&self) -> bool {
        false
    }
    /// A configured check permits reminders from retained check receipts.
    /// Hosts that disable checks must not revive historical repair priorities.
    fn check_enabled(&self) -> bool {
        false
    }
    fn check_not_run(&self) -> bool {
        false
    }
}

/// Which budget in [`Limits`] ended a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitKind {
    ModelTurns,
    ToolTurns,
    ToolCalls,
    ToolOutput,
    History,
    Context,
    Time,
    AutoContinues,
    /// Replies stayed empty (no visible text, no tool call) after every nudge.
    EmptyReplies,
}

/// Why a run ended. Only error kinds escape, never upstream text or paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    Finished(FinishReason),
    Cancelled,
    Limit(LimitKind),
    /// Only error kinds escape the adapter: no private upstream body/path text.
    ModelError(io::ErrorKind),
    InvalidResponse,
    /// A length cutoff split a tool proposal; no proposed call was executed.
    TruncatedToolCall,
    InvalidConfiguration,
    ToolFailed,
    IncompleteSubmission,
    JournalError,
}

/// What one run did: why it stopped, turn, tool and output counts, and
/// whether the last answer can be continued.
#[derive(Clone, Debug)]
pub struct RunReport {
    pub reason: StopReason,
    /// The last committed answer ended at the output limit and can be continued.
    /// Independent of the run's stopping reason: turn/time/context limits or a
    /// failed continuation must not hide a saved answer. Truncated tool calls
    /// are never continuable; a later committed tool round clears this state.
    pub answer_cut_off: bool,
    pub model_turns: usize,
    pub auto_continues: usize,
    /// Harness reminders for empty replies, checks, cutoffs and completion review.
    pub nudges: usize,
    /// Automatic user-configured checks run after file changes.
    pub auto_checks: usize,
    pub tool_turns: usize,
    /// Actual executor invocations (not rejected or skipped proposals).
    pub tool_calls: usize,
    pub output_bytes: usize,
}

impl RunReport {
    fn stopped(mut self, reason: StopReason) -> Self {
        self.reason = reason;
        self
    }
}

/// Run one bounded activation. A caller may explicitly resume with new budgets
/// and a fresh cancellation flag; this function never automatically retries a
/// failed response. Tool outputs and the associated calls commit together.
/// Persistence hook: a closed base plus a replaceable, closed staged tail.
/// Called before each tool (with an unknown-outcome marker), after each tool,
/// and before committing final answers. Errors stop execution immediately.
pub trait RunObserver {
    /// Presentation only: loop-owned rounds and authoritative token counts.
    fn progress(
        &mut self,
        _rounds: usize,
        _limits: Limits,
        _context: Option<(u64, u64)>,
        _cancel: &AtomicBool,
    ) {}
    /// Presentation only. The continuation user message has already been
    /// durably checkpointed and committed; never add this notice to the prompt.
    fn auto_continued(&mut self, _count: usize) {}
    /// Native hosts decide using real counts and count the candidate before
    /// persistence. Legacy observers keep the original no-I/O hook.
    fn before_counted_request(
        &mut self,
        c: &mut Conversation,
        limits: Limits,
        cancel: &AtomicBool,
        _: &crate::context::Usage,
        _: &mut dyn FnMut(&Conversation) -> io::Result<crate::context::Usage>,
    ) -> io::Result<bool> {
        self.before_request(c, limits, cancel)
    }
    /// A host may durably compact closed history between requests. Return true
    /// only after the replacement is persisted; stable options/prefix stay exact.
    fn before_request(
        &mut self,
        _: &mut Conversation,
        _: Limits,
        _: &AtomicBool,
    ) -> io::Result<bool> {
        Ok(false)
    }
    fn checkpoint(&mut self, base: &Conversation, staged: &[Message]) -> io::Result<()>;
    /// Persist completion metadata separately from the assistant wire message.
    fn checkpoint_reply(
        &mut self,
        base: &Conversation,
        message: &Message,
        _finish: &FinishReason,
    ) -> io::Result<()> {
        self.checkpoint(base, std::slice::from_ref(message))
    }
}
struct NoObserver;
impl RunObserver for NoObserver {
    fn checkpoint(&mut self, _: &Conversation, _: &[Message]) -> io::Result<()> {
        Ok(())
    }
}
fn cancelled(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Relaxed)
}

fn uncertain_content(receipt: &str, cap: usize) -> String {
    // Structured receipts survive unchanged whenever they already fit. The
    // executor flag also protects prose/artifact excerpts and post-effect errors.
    if receipt.len() <= cap && crate::context::has_unknown_outcome_content(receipt) {
        return receipt.to_owned();
    }
    let mut end = receipt.len().min(cap);
    loop {
        while !receipt.is_char_boundary(end) {
            end -= 1;
        }
        let output = format!(
            "{{\"ok\":false,\"error\":\"unknown_outcome\",\"uncertain_writes\":true,\"receipt\":{},\"receipt_truncated\":{}}}",
            json::quote(&receipt[..end]),
            end < receipt.len()
        );
        if output.len() <= cap {
            return output;
        }
        if end == 0 {
            return error_content("unknown_outcome");
        }
        end = end.saturating_sub(output.len() - cap);
    }
}

fn error_content(code: &str) -> String {
    // Constant-size fallback envelopes fit the reservation even when no detailed
    // receipt fits, including after earlier tools have executed.
    let mut content = if code == "invalid_arguments" {
        r#"{"ok":false,"error":"invalid_arguments","retry":"JSON object"}"#.to_owned()
    } else {
        format!("{{\"ok\":false,\"error\":\"{code}\"}}")
    };
    content.extend(std::iter::repeat_n(' ', 64 - content.len()));
    content
}

/// Shared validators still identify the individual field without copying input.
pub(crate) fn invalid_integer(key: &str) -> ToolError {
    ToolError::InvalidArguments(match key {
        "offset" => "offset must be a nonnegative integer",
        "limit" => "limit must be a nonnegative integer",
        "length" => "length must be a nonnegative integer",
        "start_line" => "start_line must be a nonnegative integer",
        "end_line" => "end_line must be a nonnegative integer",
        _ => "integer field must be a nonnegative integer",
    })
}

/// Detailed validation errors use only the already reserved receipt capacity.
fn invalid_arguments_content(reason: &'static str, cap: usize) -> String {
    let fallback = error_content("invalid_arguments");
    if reason.len() > 160 { return fallback; }
    let content = format!("{{\"ok\":false,\"error\":\"invalid_arguments\",\"message\":{}}}", json::quote(reason));
    if content.len() <= cap { content } else { fallback }
}

/// A small adapter for the UI worker's existing workspace-bound read/list API:
/// `WorkspaceTools::new(|path, offset, limit| workspace.read_range(path, offset, limit), |path| workspace.list(path))`.
/// The supplied callbacks MUST already enforce workspace/credential boundaries.
pub struct WorkspaceTools<R, L> {
    read: R,
    list: L,
}

impl<R, L> WorkspaceTools<R, L> {
    pub fn new(read: R, list: L) -> Self {
        Self { read, list }
    }

    pub fn definitions(&self) -> Value {
        json::parse(r#"[
            {"type":"function","function":{"name":"workspace_read","description":"Read a UTF-8 byte range inside the approved workspace. Source ≤4 MiB; offset is zero-based (default 0), limit 1–65536 bytes (default bounded by tool output). Partial results state actual bytes and next offset. Offset must be a UTF-8 boundary.","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer","minimum":0},"limit":{"type":"integer","minimum":1,"maximum":65536}},"required":["path"],"additionalProperties":false}}},
            {"type":"function","function":{"name":"workspace_list","description":"List a bounded directory inside the approved workspace.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}
        ]"#).expect("static tool definitions are valid JSON")
    }
}

impl<R, L> ToolExecutor for WorkspaceTools<R, L>
where
    R: FnMut(&str, usize, usize) -> io::Result<String>,
    L: FnMut(&str) -> io::Result<Vec<String>>,
{
    fn execute(
        &mut self,
        invocation: ToolInvocation<'_>,
        cancel: &AtomicBool,
    ) -> Result<String, ToolError> {
        if cancelled(cancel) {
            return Err(ToolError::Cancelled);
        }
        let Value::Object(fields) = invocation.arguments else {
            return Err(ToolError::InvalidArguments("arguments must be a JSON object"));
        };
        if fields
            .iter()
            .any(|(key, _)| !matches!(key.as_str(), "path" | "offset" | "limit"))
            || invocation.call.name != "workspace_read" && fields.len() != 1
        {
            return Err(ToolError::InvalidArguments("workspace_read/list: unexpected fields; list requires only path"));
        }
        let path = invocation
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .ok_or(ToolError::InvalidArguments("path must be a string"))?;
        let result = match invocation.call.name.as_str() {
            "workspace_read" => {
                let integer = |key: &str, default| match invocation.arguments.get(key) {
                    None => Ok(default),
                    Some(Value::Number(n)) => {
                        n.parse::<usize>().map_err(|_| invalid_integer(key))
                    }
                    _ => Err(invalid_integer(key)),
                };
                let offset = integer("offset", 0)?;
                let limit = integer("limit", crate::tools::MAX_READ_BYTES)?;
                let limit = if limit <= crate::tools::MAX_READ_BYTES {
                    limit.min(invocation.max_output_bytes.saturating_sub(256).max(1))
                } else {
                    limit
                };
                (self.read)(path, offset, limit).map_err(|_| ToolError::Denied)
            }
            "workspace_list" => {
                let names = (self.list)(path).map_err(|_| ToolError::Denied)?;
                let size = names
                    .iter()
                    .fold(names.len().saturating_sub(1), |sum, name| {
                        sum.saturating_add(name.len())
                    });
                if size > invocation.max_output_bytes {
                    return Err(ToolError::OutputLimit);
                }
                Ok(names.join("\n"))
            }
            _ => Err(ToolError::UnknownTool),
        }?;
        if cancelled(cancel) {
            return Err(ToolError::Cancelled);
        }
        if result.len() > invocation.max_output_bytes {
            return Err(ToolError::OutputLimit);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
