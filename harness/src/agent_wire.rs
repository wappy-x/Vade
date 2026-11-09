//! Lossless message storage and bounded chat-completion tool-call assembly.
//!
//! Wire conventions follow the OpenAI chat-completions message/delta format.
//! This is an original implementation. HTTP and SSE *framing* belong to the
//! transport: pass complete SSE data fields here, including `[DONE]`.
use crate::json::{self, Value, invalid};
use std::collections::{BTreeMap, HashSet};
use std::io;

/// Bounds on one streamed model response: tool calls, argument and message
/// bytes, total stream bytes and event count.
#[derive(Clone, Copy, Debug)]
pub struct WireLimits {
    pub max_calls: usize,
    /// Sum of unescaped argument bytes across all calls in one response.
    pub max_argument_bytes: usize,
    /// Serialized assistant message, including unfamiliar fields.
    pub max_message_bytes: usize,
    /// All event payloads, including usage and other metadata.
    pub max_stream_bytes: usize,
    pub max_events: usize,
}

impl Default for WireLimits {
    fn default() -> Self {
        Self {
            max_calls: 16,
            // Sized for one reply that uses the whole 64k-token default output cap (and
            // more): the server streams one event per token, plus JSON envelopes.
            max_argument_bytes: 256 * 1024,
            max_message_bytes: 4 * 1024 * 1024,
            max_stream_bytes: 64 * 1024 * 1024,
            max_events: 262_144,
        }
    }
}

pub const MAX_CALL_ID_BYTES: usize = 256;
pub const MAX_TOOL_NAME_BYTES: usize = 128;
// Independent, conservative AST budgets leave room for request envelopes inside
// the core JSON parser's depth/value limits. Byte bounds alone do not bound the
// overhead of many tiny extension values.
pub(crate) const MAX_HISTORY_NODES: usize = 32_768;
const MAX_MESSAGE_DEPTH: usize = 48;

pub(crate) fn json_nodes(value: &Value) -> io::Result<usize> {
    fn visit(value: &Value, depth: usize, nodes: &mut usize) -> io::Result<()> {
        *nodes += 1;
        if depth > MAX_MESSAGE_DEPTH || *nodes > MAX_HISTORY_NODES {
            return Err(invalid("agent JSON complexity limit exceeded"));
        }
        match value {
            Value::Array(items) => {
                for item in items {
                    visit(item, depth + 1, nodes)?;
                }
            }
            Value::Object(fields) => {
                for (_, item) in fields {
                    visit(item, depth + 1, nodes)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    let mut nodes = 0;
    visit(value, 0, &mut nodes)?;
    Ok(nodes)
}

/// A typed projection; the original call object remains in `Message::value()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Full JSON is authoritative: content arrays, refusal/reasoning fields and
/// provider extensions are retained, never reduced to role/text pairs. Reserved
/// harness metadata is excluded from wire messages, including legacy markers.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    value: Value,
    calls: Vec<ToolCall>,
    result_id: Option<String>,
    serialized: String,
    pub(crate) nodes: usize,
}

impl Message {
    pub fn from_value(mut value: Value, limits: WireLimits) -> io::Result<Self> {
        // Older local journals stored this private marker inside messages.
        // Remove it at ingestion so every cached/replayed wire prefix is clean.
        if let Value::Object(fields) = &mut value {
            fields.retain(|(key, _)| key != "vade_truncated");
        }
        let nodes = json_nodes(&value)?;
        let serialized = json::stringify(&value)?;
        if serialized.len() > limits.max_message_bytes {
            return Err(invalid("agent message exceeds byte limit"));
        }
        let role = required_string(&value, "role")?;
        if role.is_empty() {
            return Err(invalid("message role is empty"));
        }
        if value
            .get("function_call")
            .is_some_and(|v| *v != Value::Null)
        {
            return Err(invalid(
                "legacy function_call is unsupported; use tool_calls",
            ));
        }
        let mut calls = Vec::new();
        if let Some(raw_calls) = value.get("tool_calls") {
            match raw_calls {
                Value::Null => (),
                Value::Array(items) => {
                    if !items.is_empty() && role != "assistant" {
                        return Err(invalid("only assistants may request tools"));
                    }
                    if items.len() > limits.max_calls {
                        return Err(invalid("too many tool calls in response"));
                    }
                    let mut ids = HashSet::new();
                    let mut argument_bytes = 0usize;
                    for item in items {
                        let id = required_string(item, "id")?;
                        identifier(id, MAX_CALL_ID_BYTES)?;
                        if !ids.insert(id) {
                            return Err(invalid("duplicate tool call ID"));
                        }
                        if required_string(item, "type")? != "function" {
                            return Err(invalid("unsupported tool call type"));
                        }
                        let function = item
                            .get("function")
                            .ok_or_else(|| invalid("missing function"))?;
                        let name = required_string(function, "name")?;
                        identifier(name, MAX_TOOL_NAME_BYTES)?;
                        let arguments = required_string(function, "arguments")?;
                        argument_bytes = argument_bytes.saturating_add(arguments.len());
                        if argument_bytes > limits.max_argument_bytes {
                            return Err(invalid("tool arguments exceed byte limit"));
                        }
                        calls.push(ToolCall {
                            id: id.to_owned(),
                            name: name.to_owned(),
                            arguments: arguments.to_owned(),
                        });
                    }
                }
                _ => return Err(invalid("tool_calls must be an array or null")),
            }
        }
        let result_id = if role == "tool" {
            let id = required_string(&value, "tool_call_id")?;
            identifier(id, MAX_CALL_ID_BYTES)?;
            Some(id.to_owned())
        } else {
            if value.get("tool_call_id").is_some() {
                return Err(invalid("tool_call_id requires a tool message"));
            }
            None
        };
        Ok(Self {
            value,
            calls,
            result_id,
            serialized,
            nodes,
        })
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
    pub fn into_value(self) -> Value {
        self.value
    }
    pub fn role(&self) -> &str {
        self.value.get("role").and_then(Value::as_str).unwrap()
    }
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.calls
    }
    pub fn tool_call_id(&self) -> Option<&str> {
        self.result_id.as_deref()
    }
    pub fn serialized_bytes(&self) -> usize {
        self.serialized.len()
    }
    /// Validated messages are immutable; serialize once at creation, then reuse
    /// these exact bytes even for tool, reasoning and provider extension fields.
    pub fn serialized_json(&self) -> &str {
        &self.serialized
    }

    /// Keep failed calls structurally complete without replaying malformed JSON
    /// to upstreams that parse argument strings while rendering chat history.
    /// The caller must validate/execute the original calls, never placeholders.
    pub(crate) fn with_valid_argument_json(self, limits: WireLimits) -> io::Result<Self> {
        let invalid_arguments: Vec<_> = self
            .calls
            .iter()
            .enumerate()
            .filter_map(|(index, call)| json::view(&call.arguments).is_err().then_some(index))
            .collect();
        if invalid_arguments.is_empty() {
            return Ok(self);
        }
        let mut value = self.into_value();
        let Value::Object(fields) = &mut value else {
            unreachable!("validated message object")
        };
        let Value::Array(calls) = &mut fields
            .iter_mut()
            .find(|(key, _)| key == "tool_calls")
            .expect("validated tool calls")
            .1
        else {
            unreachable!("validated tool call array")
        };
        for index in invalid_arguments {
            let Value::Object(function) = object_slot(&mut calls[index], "function")? else {
                unreachable!("validated tool function object")
            };
            function
                .iter_mut()
                .find(|(key, _)| key == "arguments")
                .expect("validated tool arguments")
                .1 = Value::String("{}".into());
        }
        Self::from_value(value, limits)
    }

    pub fn tool_result(id: &str, content: String, limits: WireLimits) -> io::Result<Self> {
        Self::from_value(
            Value::Object(vec![
                ("role".into(), Value::String("tool".into())),
                ("tool_call_id".into(), Value::String(id.into())),
                ("content".into(), Value::String(content)),
            ]),
            limits,
        )
    }

    pub(crate) fn truncated(self, limits: WireLimits) -> io::Result<Self> {
        response_message(self.into_value(), &FinishReason::Length, limits)
    }
}

fn identifier(value: &str, limit: usize) -> io::Result<()> {
    if value.is_empty() || value.len() > limit || value.chars().any(char::is_control) {
        Err(invalid("invalid tool identifier"))
    } else {
        Ok(())
    }
}

pub(crate) fn required_string<'a>(value: &'a Value, key: &str) -> io::Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("missing or non-string required field"))
}

/// Why the model stopped one response, as reported in `finish_reason`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    /// Kept exactly and treated as terminal, never as permission to run tools.
    Other(String),
}

impl FinishReason {
    pub fn from_wire(value: &str) -> Self {
        match value {
            "stop" => Self::Stop,
            "tool_calls" => Self::ToolCalls,
            "length" => Self::Length,
            "content_filter" => Self::ContentFilter,
            value => Self::Other(value.to_owned()),
        }
    }
}

/// One completed assistant response: its message, finish reason and local
/// truncation metadata that never enters history.
#[derive(Clone, Debug)]
pub struct AssistantTurn {
    message: Message,
    finish: FinishReason,
    // Local completion metadata survives normalization; never enters history.
    truncated_tool_call: bool,
    // Bounded description only; discarded arguments never become executable.
    truncated_tool_hint: Option<String>,
    // Streamed content + argument bytes discarded with the truncated group.
    discarded_bytes: usize,
    reply_budget_exhausted: bool,
}

impl AssistantTurn {
    pub fn new(message: Message, finish: FinishReason) -> io::Result<Self> {
        if message.role() != "assistant" {
            return Err(invalid("model returned a non-assistant message"));
        }
        if finish == FinishReason::ToolCalls && message.calls.is_empty() {
            return Err(invalid("tool_calls finish has no calls"));
        }
        if finish == FinishReason::Stop && !message.calls.is_empty() {
            return Err(invalid("stop finish contains tool calls"));
        }
        let truncated_tool_call = finish == FinishReason::Length && !message.calls.is_empty();
        let truncated_tool_hint = truncated_tool_call.then(|| tool_hint(message.value())).flatten();
        let discarded_bytes = if truncated_tool_call { discarded_bytes_of(message.value()) } else { 0 };
        Ok(Self {
            message,
            finish,
            truncated_tool_call,
            truncated_tool_hint,
            discarded_bytes,
            reply_budget_exhausted: false,
        })
    }
    pub(crate) fn with_reply_budget_cutoff(mut self) -> Self {
        self.reply_budget_exhausted = true;
        self
    }
    pub(crate) fn reply_budget_exhausted(&self) -> bool {
        self.reply_budget_exhausted
    }
    pub fn message(&self) -> &Message {
        &self.message
    }
    pub fn finish_reason(&self) -> &FinishReason {
        &self.finish
    }
    /// A length cutoff occurred while proposing tools. The normalized message
    /// deliberately drops those proposals; continuing prose cannot finish them.
    pub fn truncated_tool_call(&self) -> bool {
        self.truncated_tool_call
    }
    pub fn truncated_tool_hint(&self) -> Option<&str> {
        self.truncated_tool_hint.as_deref()
    }
    /// Bytes discarded with a truncated tool group (0 when not truncated).
    pub fn discarded_bytes(&self) -> usize {
        self.discarded_bytes
    }
    pub fn into_parts(self) -> (Message, FinishReason) {
        (self.message, self.finish)
    }

    /// Non-streaming fallback retains the assistant fields, supplying empty
    /// content for a textless answer before the message is frozen as history.
    pub fn from_response(value: &Value, limits: WireLimits) -> io::Result<Self> {
        if json::stringify(value)?.len() > limits.max_stream_bytes {
            return Err(invalid("response exceeds byte limit"));
        }
        let choice = choice(value)?.ok_or_else(|| invalid("response has no choice"))?;
        let message = choice
            .get("message")
            .ok_or_else(|| invalid("response has no message"))?;
        let finish = required_string(choice, "finish_reason")?;
        if finish.is_empty() {
            return Err(invalid("invalid finish reason"));
        }
        let mut turn = Self::new(
            response_message(message.clone(), &FinishReason::from_wire(finish), limits)?,
            FinishReason::from_wire(finish),
        )?;
        turn.truncated_tool_call = finish == "length"
            && (message.get("tool_calls").is_some_and(|calls| match calls {
                Value::Null => false,
                Value::Array(calls) => !calls.is_empty(),
                _ => true,
            }) || message
                .get("function_call")
                .is_some_and(|call| *call != Value::Null));
        if turn.truncated_tool_call {
            turn.truncated_tool_hint = tool_hint(message);
            turn.discarded_bytes = discarded_bytes_of(message);
        }
        Ok(turn)
    }
}

/// Content + argument bytes of a raw truncated reply, before normalization
/// drops its tool proposals. Sizes only; the bytes never become executable.
fn discarded_bytes_of(message: &Value) -> usize {
    let arguments = |call: &Value| call.get("function").unwrap_or(call).get("arguments").and_then(Value::as_str).map_or(0, str::len);
    let calls = message.get("tool_calls").and_then(Value::as_array).map_or(0, |calls| calls.iter().map(arguments).sum());
    message.get("content").and_then(Value::as_str).map_or(0, str::len)
        .saturating_add(calls).saturating_add(message.get("function_call").map_or(0, arguments))
}

/// Extract only known tool/path text, including a completed patch header in
/// incomplete JSON. It is quoted as untrusted data in the repair message.
fn tool_hint(message: &Value) -> Option<String> {
    let call = message.get("tool_calls").and_then(Value::as_array).and_then(|c| c.last())
        .or_else(|| message.get("function_call")).unwrap_or(message);
    let function = call.get("function").unwrap_or(call);
    let name = function.get("name")?.as_str()?;
    let args = function.get("arguments").and_then(Value::as_str).unwrap_or("");
    let path = args.split_once("\"path\"").and_then(|(_, rest)| {
        let text = rest.trim_start().strip_prefix(':')?.trim_start();
        let mut escaped = false;
        for (i, ch) in text.strip_prefix('"')?.char_indices() {
            if ch == '"' && !escaped {
                return json::parse(&text[..i + 2]).ok();
            }
            escaped = ch == '\\' && !escaped;
        }
        None
    }).or_else(|| {
        ["*** Add File: ", "*** Update File: ", "*** Delete File: "].iter().find_map(|marker| {
            let path = args.split_once(marker)?.1.split_once("\\n")?.0;
            json::parse(&format!("\"{path}\"")).ok()
        })
    });
    let bounded = |text: &str| json::quote(&text.chars().take(256).collect::<String>());
    Some(match path.as_ref().and_then(Value::as_str) {
        Some(path) => format!("{} targeting {}", bounded(name), bounded(path)),
        None => bounded(name),
    })
}

// Engine's chat contract requires content on assistants without tool calls.
// Reasoning/refusal-only and length-limited replies still need a legal history
// entry. Normalize once, at response ingestion, never while serializing an old
// request: reasoning and provider fields stay in their original fields.
// Assistant tool-call messages may omit content; retain their representation.
fn response_message(
    mut value: Value,
    finish: &FinishReason,
    limits: WireLimits,
) -> io::Result<Message> {
    if *finish == FinishReason::Length {
        if let Value::Object(fields) = &mut value {
            // A cutoff can split a tool ID, name, or arguments. Retain prose
            // and reasoning, but never persist executable/dangling proposals.
            fields.retain(|(key, _)| !matches!(key.as_str(), "tool_calls" | "function_call"));
        }
    }
    if value.get("role").and_then(Value::as_str) == Some("assistant")
        && matches!(value.get("content"), None | Some(Value::Null))
        && !value
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty())
    {
        let Value::Object(fields) = &mut value else {
            unreachable!()
        };
        if let Some((_, content)) = fields.iter_mut().find(|(key, _)| key == "content") {
            *content = Value::String(String::new());
        } else {
            fields.push(("content".into(), Value::String(String::new())));
        }
    }
    Message::from_value(value, limits)
}

fn choice(value: &Value) -> io::Result<Option<&Value>> {
    if let Some(error) = value.get("error").filter(|v| **v != Value::Null) {
        // Do not echo upstream bodies: they can include private request data.
        return Err(crate::protocol::response_error(error, "upstream returned an agent error"));
    }
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("response has no choices array"))?;
    if choices.len() > 1 {
        return Err(invalid("agent mode requires exactly one choice"));
    }
    let Some(choice) = choices.first() else {
        return Ok(None);
    };
    if !matches!(choice, Value::Object(_)) {
        return Err(invalid("choice must be an object"));
    }
    if let Some(index) = choice.get("index") {
        if index != &Value::Number("0".into()) {
            return Err(invalid("agent mode requires choice index zero"));
        }
    }
    Ok(Some(choice))
}

/// An error poisons the accumulator. Neither a partial response nor a stream
/// missing its finish reason / `[DONE]` can be committed or executed.
pub struct ChatAccumulator {
    limits: WireLimits,
    message: Value,
    calls: BTreeMap<usize, Value>,
    finish: Option<FinishReason>,
    done: bool,
    failed: bool,
    bytes: usize,
    events: usize,
    argument_bytes: usize,
}

impl ChatAccumulator {
    pub fn new(limits: WireLimits) -> Self {
        Self {
            limits,
            message: Value::Object(vec![("role".into(), Value::String("assistant".into()))]),
            calls: BTreeMap::new(),
            finish: None,
            done: false,
            failed: false,
            bytes: 0,
            events: 0,
            argument_bytes: 0,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done && !self.failed
    }

    /// Roll back to before the entire uncommitted tool group. Never manufacture
    /// closing arguments or execute a prefix, even if earlier calls look whole.
    /// Ordinary EOF, malformed framing and upstream cutoffs retain their guards.
    pub(crate) fn cap_reply(&mut self, tokens: usize, expired: bool) -> bool {
        if self.failed || self.finish.is_some() { return false; }
        let bytes = self.message.get("content").and_then(Value::as_str).map_or(0,str::len);
        // A soft byte estimate must not repeatedly discard a nearly finished
        // tool call. Give tool groups at most 2x headroom; the absolute reply
        // deadline, wire limits and upstream output limit still apply.
        let headroom = if self.calls.is_empty() { 3 } else { 6 };
        if !expired && bytes.saturating_add(self.argument_bytes) < tokens.saturating_mul(headroom) { return false; }
        self.finish = Some(FinishReason::Length);
        self.done = true;
        true
    }

    pub fn push_sse_data(
        &mut self,
        data: &str,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<()> {
        let result = self.push_inner(data, on_text);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn push_inner(
        &mut self,
        data: &str,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.failed || self.done {
            return Err(invalid("agent stream is already closed"));
        }
        self.bytes = self.bytes.saturating_add(data.len());
        self.events = self.events.saturating_add(1);
        if self.bytes > self.limits.max_stream_bytes || self.events > self.limits.max_events {
            return Err(invalid("agent stream exceeds limit"));
        }
        if data.trim() == "[DONE]" {
            if self.finish.is_none() {
                return Err(invalid("stream ended without a finish reason"));
            }
            self.done = true;
            return Ok(());
        }
        // The frequent content-only delta borrows JSON/SSE input all the way
        // to the display callback. Tool/extension deltas retain the full parser.
        if self.push_text_view(json::view(data)?, on_text)? {
            return Ok(());
        }
        let value = json::parse(data)?;
        let Some(choice) = choice(&value)? else {
            return Ok(());
        }; // usage-only event
        if self.finish.is_some() {
            return Err(invalid("choice received after finish"));
        }
        let delta = choice
            .get("delta")
            .ok_or_else(|| invalid("stream choice has no delta"))?;
        let Value::Object(fields) = delta else {
            return Err(invalid("delta must be an object"));
        };
        for (key, value) in fields {
            match key.as_str() {
                "tool_calls" => self.merge_calls(value)?,
                "role" => {
                    if value != &Value::String("assistant".into()) {
                        return Err(invalid("stream role must be assistant"));
                    }
                }
                "function_call" if *value != Value::Null => {
                    return Err(invalid(
                        "legacy function_call is unsupported; use tool_calls",
                    ));
                }
                "content" | "refusal" | "reasoning_content" | "reasoning" => {
                    merge_field(
                        &mut self.message,
                        key,
                        value,
                        true,
                        self.limits.max_message_bytes,
                    )?;
                }
                _ => merge_field(
                    &mut self.message,
                    key,
                    value,
                    false,
                    self.limits.max_message_bytes,
                )?,
            }
        }
        if let Some(reason) = choice.get("finish_reason") {
            match reason {
                Value::Null => (),
                Value::String(reason) if !reason.is_empty() => {
                    self.finish = Some(FinishReason::from_wire(reason))
                }
                _ => return Err(invalid("invalid finish reason")),
            }
        }
        if let Some(Value::String(text)) = delta.get("content") {
            on_text(text)?;
        }
        Ok(())
    }

    fn push_text_view(
        &mut self,
        value: json::View<'_>,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<bool> {
        if let Some(error) = value.get("error").filter(|error| error.raw() != "null") {
            return Err(crate::protocol::response_error_view(error, "upstream returned an agent error"));
        }
        let Some(mut choices) = value.get("choices").and_then(json::View::array) else {
            return Ok(false);
        };
        let Some(choice) = choices.next() else {
            return Ok(true);
        };
        if choices.next().is_some() {
            return Err(invalid("agent mode requires exactly one choice"));
        }
        if choice.get("index").is_some_and(|index| index.raw() != "0") {
            return Ok(false);
        }
        let Some(mut fields) = choice.get("delta").and_then(json::View::fields) else {
            return Ok(false);
        };
        let Some((name, text)) = fields.next() else {
            return Ok(false);
        };
        if name != "content" || fields.next().is_some() {
            return Ok(false);
        }
        let Some(text) = text.as_str() else {
            return Ok(false);
        };
        if self.finish.is_some() {
            return Err(invalid("choice received after finish"));
        }
        let finish = match choice.get("finish_reason") {
            None => None,
            Some(value) if value.raw() == "null" => None,
            Some(value) => {
                let reason = value
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| invalid("invalid finish reason"))?;
                Some(FinishReason::from_wire(&reason))
            }
        };
        let Value::Object(fields) = &mut self.message else {
            unreachable!()
        };
        let index = fields
            .iter()
            .position(|(key, _)| key == "content")
            .unwrap_or_else(|| {
                fields.push(("content".into(), Value::Null));
                fields.len() - 1
            });
        let content = &mut fields[index].1;
        if *content == Value::Null {
            *content = Value::String(String::new());
        }
        let Value::String(content) = content else {
            return Err(invalid("unsupported conflicting content delta"));
        };
        if content.len().saturating_add(text.len()) > self.limits.max_message_bytes {
            return Err(invalid("delta field exceeds byte limit"));
        }
        content.push_str(&text);
        self.finish = finish;
        on_text(&text)?;
        Ok(true)
    }

    fn merge_calls(&mut self, value: &Value) -> io::Result<()> {
        if self.message.get("tool_calls").is_none() {
            let Value::Object(fields) = &mut self.message else {
                unreachable!()
            };
            fields.push((
                "tool_calls".into(),
                if *value == Value::Null {
                    Value::Null
                } else {
                    Value::Array(Vec::new())
                },
            ));
        }
        if *value == Value::Null {
            return Ok(());
        }
        let Value::Array(calls) = value else {
            return Err(invalid("tool_calls delta must be an array"));
        };
        for call in calls {
            let index = match call.get("index") {
                Some(Value::Number(text)) if text.bytes().all(|b| b.is_ascii_digit()) => text
                    .parse::<usize>()
                    .map_err(|_| invalid("tool index exceeds limit"))?,
                _ => return Err(invalid("tool call delta requires an integer index")),
            };
            if index >= self.limits.max_calls {
                return Err(invalid("tool index exceeds limit"));
            }
            let Value::Object(fields) = call else {
                return Err(invalid("tool call must be an object"));
            };
            let target = self
                .calls
                .entry(index)
                .or_insert_with(|| Value::Object(Vec::new()));
            for (key, value) in fields {
                match key.as_str() {
                    "index" => (), // stream routing metadata, not a message field
                    "id" => merge_field(target, key, value, true, MAX_CALL_ID_BYTES)?,
                    "function" => {
                        let Value::Object(function) = value else {
                            return Err(invalid("function delta must be an object"));
                        };
                        let function_target = object_slot(target, "function")?;
                        for (key, value) in function {
                            let limit = match key.as_str() {
                                "arguments" => {
                                    let Value::String(fragment) = value else {
                                        return Err(invalid("arguments delta must be a string"));
                                    };
                                    self.argument_bytes =
                                        self.argument_bytes.saturating_add(fragment.len());
                                    if self.argument_bytes > self.limits.max_argument_bytes {
                                        return Err(invalid("tool arguments exceed byte limit"));
                                    }
                                    self.limits.max_argument_bytes
                                }
                                "name" => MAX_TOOL_NAME_BYTES,
                                _ => self.limits.max_message_bytes,
                            };
                            merge_field(
                                function_target,
                                key,
                                value,
                                matches!(key.as_str(), "name" | "arguments"),
                                limit,
                            )?;
                        }
                    }
                    _ => merge_field(target, key, value, false, self.limits.max_message_bytes)?,
                }
            }
        }
        Ok(())
    }

    pub fn into_turn(mut self) -> io::Result<AssistantTurn> {
        if self.failed {
            return Err(invalid("cannot finish a failed agent stream"));
        }
        if !self.done {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "agent stream ended without [DONE]",
            ));
        }
        let truncated_tool_call =
            !self.calls.is_empty() && self.finish == Some(FinishReason::Length);
        let truncated_tool_hint = if truncated_tool_call {
            self.calls.values().next_back().and_then(tool_hint)
        } else {
            None
        };
        if !self.calls.is_empty() && !truncated_tool_call {
            if self.calls.keys().copied().ne(0..self.calls.len()) {
                return Err(invalid("tool call indices contain gaps"));
            }
            let Value::Object(fields) = &mut self.message else {
                unreachable!()
            };
            fields
                .iter_mut()
                .find(|(key, _)| key == "tool_calls")
                .unwrap()
                .1 = Value::Array(self.calls.into_values().collect());
        }
        let discarded_bytes = if truncated_tool_call {
            self.argument_bytes.saturating_add(self.message.get("content").and_then(Value::as_str).map_or(0, str::len))
        } else { 0 };
        let mut turn = AssistantTurn::new(
            response_message(self.message, self.finish.as_ref().unwrap(), self.limits)?,
            self.finish.unwrap(),
        )?;
        turn.truncated_tool_call = truncated_tool_call;
        turn.truncated_tool_hint = truncated_tool_hint;
        turn.discarded_bytes = discarded_bytes;
        Ok(turn)
    }
}

fn object_slot<'a>(object: &'a mut Value, key: &str) -> io::Result<&'a mut Value> {
    let Value::Object(fields) = object else {
        return Err(invalid("expected object"));
    };
    let index = match fields.iter().position(|(name, _)| name == key) {
        Some(index) => index,
        None => {
            fields.push((key.to_owned(), Value::Object(Vec::new())));
            fields.len() - 1
        }
    };
    if !matches!(fields[index].1, Value::Object(_)) {
        return Err(invalid("conflicting object delta"));
    }
    Ok(&mut fields[index].1)
}

// Unknown fields are kept verbatim. Disjoint object patches can be combined;
// conflicting unfamiliar scalars/arrays fail explicitly, since guessing their
// delta semantics could corrupt provider-specific state needed on the next turn.
fn merge_field(
    object: &mut Value,
    key: &str,
    value: &Value,
    append: bool,
    limit: usize,
) -> io::Result<()> {
    if matches!(value, Value::String(text) if text.len() > limit) {
        return Err(invalid("delta field exceeds byte limit"));
    }
    let Value::Object(fields) = object else {
        return Err(invalid("expected object"));
    };
    if let Some((_, old)) = fields.iter_mut().find(|(name, _)| name == key) {
        if *value == Value::Null {
            if append || *old == Value::Null {
                return Ok(());
            }
            return Err(invalid("unsupported conflicting extension delta"));
        }
        if append {
            if let (Value::String(old), Value::String(fragment)) = (&mut *old, value) {
                if old.len().saturating_add(fragment.len()) > limit {
                    return Err(invalid("delta field exceeds byte limit"));
                }
                old.push_str(fragment);
                return Ok(());
            }
        }
        if *old == Value::Null {
            *old = value.clone();
        } else if let (Value::Object(_), Value::Object(patch)) = (&*old, value) {
            for (name, next) in patch {
                merge_field(old, name, next, false, limit)?;
            }
        } else if old != value {
            return Err(invalid("unsupported conflicting extension delta"));
        }
    } else {
        if matches!(value, Value::String(text) if text.len() > limit) {
            return Err(invalid("delta field exceeds byte limit"));
        }
        fields.push((key.to_owned(), value.clone()));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
