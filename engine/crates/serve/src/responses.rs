//! Stateless Responses wire translation. Inference and chat markup stay shared.
use crate::{Limits, chat_output::Event, generation::Usage, protocol::{self, present, quote, raw_array, raw_member}};
use engine_model::json::{self, Value};
use std::collections::{BTreeMap, BTreeSet};

type Object = BTreeMap<String, Value>;
fn field<'a>(object: &'a Object, key: &str) -> Result<&'a Value, String> {
    object.get(key).ok_or_else(|| format!("missing {key}"))
}
fn string<'a>(object: &'a Object, key: &str) -> Result<&'a str, String> {
    field(object, key)?.string().map_err(|e| e.to_string())
}
fn parse_json(body: &[u8], limits: &Limits) -> Result<Value, String> {
    json::parse(body, json::Limits { bytes: limits.body_bytes, depth: 24,
        nodes: limits.input_tokens.saturating_add(65536), string_bytes: limits.body_bytes })
        .map_err(|e| e.to_string())
}
fn text(value: &Value) -> Result<String, String> {
    if let Value::String(s) = value { return Ok(s.clone()); }
    let mut result = String::new();
    for part in value.array().map_err(|e| e.to_string())? {
        let part = part.object().map_err(|e| e.to_string())?;
        if string(part, "type")? == "input_image" { return Err(protocol::IMAGE_INPUT_UNSUPPORTED.into()); }
        if !matches!(string(part, "type")?, "input_text" | "output_text" | "summary_text" | "text") {
            return Err("only text Responses content is supported".into());
        }
        result.push_str(string(part, "text")?);
    }
    Ok(result)
}
fn message(role: &str, content: &str) -> String {
    format!("{{\"role\":{},\"content\":{}}}", quote(role), quote(content))
}
fn function_name(object: &Object) -> Result<String, String> {
    let name = string(object, "name")?;
    match object.get("namespace").filter(|v| **v != Value::Null) {
        Some(ns) => Ok(format!("{}.{name}", ns.string().map_err(|e| e.to_string())?)),
        None => Ok(name.to_owned()),
    }
}
#[derive(Default)]
struct Assistant {
    content: String,
    reasoning: String,
    calls: Vec<String>,
    present: bool,
}
impl Assistant {
    fn flush(&mut self, messages: &mut Vec<String>) {
        if !self.present { return; }
        let reasoning = if self.reasoning.is_empty() { String::new() }
            else { format!(",\"reasoning_content\":{}", quote(&self.reasoning)) };
        let calls = if self.calls.is_empty() { String::new() }
            else { format!(",\"tool_calls\":[{}]", self.calls.join(",")) };
        messages.push(format!("{{\"role\":\"assistant\",\"content\":{}{reasoning}{calls}}}", quote(&self.content)));
        *self = Self::default();
    }
}
/// A custom (freeform) tool reaches the model as a function with one required string, `input`.
/// Its description is the tool's own text, then how to pass that text and, when given, the grammar
/// it must match. The grammar informs the model; decoding is not constrained by it.
fn custom_tool(object: &Object) -> Result<[String; 2], String> {
    let mut description = present(object, "description").map(Value::string).transpose()
        .map_err(|e| e.to_string())?.unwrap_or("").to_owned();
    if !description.is_empty() { description.push_str("\n\n"); }
    description.push_str("Pass the tool's text as the `input` string.");
    if let Some(format) = present(object, "format") {
        let format = format.object().map_err(|e| e.to_string())?;
        match string(format, "type")? {
            "text" => {}
            "grammar" => {
                let syntax = match string(format, "syntax")? {
                    "lark" => "Lark grammar",
                    "regex" => "regular expression",
                    _ => return Err("custom tool grammar syntax must be lark or regex".into()),
                };
                description.push_str(&format!(" It must match this {syntax}:\n{}", string(format, "definition")?));
            }
            _ => return Err("custom tool format must be text or grammar".into()),
        }
    }
    Ok([format!("\"description\":{}", quote(&description)),
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"input\":{\"type\":\"string\"}},\"required\":[\"input\"],\"additionalProperties\":false}".into()])
}
/// The arguments a custom call stands for, written as the template's own `tojson` would write
/// them. Custom history therefore renders exactly as the equivalent function call does.
fn custom_arguments(input: &str) -> String {
    format!("{{\"input\": {}}}", engine_tokenizer::chat::quote(input))
}
fn tools(source: &str, namespace: Option<&str>, limits: &Limits, output: &mut Output,
    result: &mut Vec<String>) -> Result<(), String> {
    for raw in raw_array(source)? {
        let value = parse_json(raw.as_bytes(), limits)?;
        let object = value.object().map_err(|e| e.to_string())?;
        match string(object, "type")? {
            "namespace" if namespace.is_none() => {
                let ns = string(object, "name")?;
                if ns.is_empty() || ns.contains('.') { return Err("invalid tool namespace".into()); }
                tools(raw_member(&raw, "tools")?.ok_or("missing namespace tools")?, Some(ns), limits, output, result)?;
            }
            kind @ ("function" | "custom") => {
                if result.len() >= 64 { return Err("too many tools".into()); }
                let name = string(object, "name")?;
                let full = namespace.map_or_else(|| name.to_owned(), |ns| format!("{ns}.{name}"));
                if let Some(ns) = namespace { output.namespaces.insert(full.clone(), (ns.to_owned(), name.to_owned())); }
                let mut fields = vec![format!("\"name\":{}", quote(&full))];
                if kind == "custom" {
                    fields.extend(custom_tool(object)?);
                    output.custom.insert(full);
                } else {
                    // Keep schema key order, just as the chat endpoint does.
                    for key in ["description", "parameters", "strict"] {
                        if let Some(raw) = raw_member(&raw, key)?.filter(|s| s.trim() != "null") {
                            fields.push(format!("{}:{raw}", quote(key)));
                        }
                    }
                }
                result.push(format!("{{\"type\":\"function\",\"function\":{{{}}}}}", fields.join(",")));
            }
            _ => return Err("only function, custom and namespace tools are supported".into()),
        }
    }
    Ok(())
}

pub(crate) fn parse(body: &[u8], limits: &Limits, model: &str,
    reservation: Option<usize>) -> Result<(protocol::Request, Output), String> {
    let value = parse_json(body, limits)?;
    let object = value.object().map_err(|e| e.to_string())?;
    let source = std::str::from_utf8(body).map_err(|_| "invalid UTF-8")?;
    if object.get("background").is_some_and(|v| *v != Value::Null && *v != Value::Bool(false)) {
        return Err("background Responses are unsupported".into());
    }
    if let Some(previous) = object.get("previous_response_id").filter(|v| **v != Value::Null) {
        previous.string().map_err(|e| e.to_string())?;
        // The client supplies the full history. IDs never look up another request's state.
    }
    let choice = present(object, "tool_choice");
    let chat_choice = match choice {
        None => quote("auto"),
        Some(Value::String(s)) if matches!(s.as_str(), "auto" | "none" | "required") => quote(s),
        Some(Value::Object(o)) if matches!(string(o, "type")?, "function" | "custom") => {
            let name = string(o, "name")?;
            let namespace = present(o, "namespace").map(Value::string).transpose().map_err(|e| e.to_string())?;
            let name = namespace.map_or_else(|| name.to_owned(), |ns| format!("{ns}.{name}"));
            format!("{{\"type\":\"function\",\"function\":{{\"name\":{}}}}}", quote(&name))
        }
        _ => return Err("tool_choice must be auto, none, required or a named function/custom tool".into()),
    };
    if let Some(format) = present(object, "text")
        .map(Value::object).transpose().map_err(|e| e.to_string())?
        .and_then(|o| present(o, "format")) {
        if string(format.object().map_err(|e| e.to_string())?, "type")? != "text" {
            return Err("structured Responses output is unsupported".into());
        }
    }
    let mut messages = Vec::new();
    if let Some(instructions) = object.get("instructions").filter(|v| **v != Value::Null) {
        messages.push(message("system", instructions.string().map_err(|e| e.to_string())?));
    }
    let instruction_messages = messages.len();
    match field(object, "input")? {
        Value::String(s) => messages.push(message("user", s)),
        Value::Array(items) => {
            if items.len() > 4096 { return Err("too many Responses input items".into()); }
            let mut assistant = Assistant::default();
            for item in items {
                let item = item.object().map_err(|e| e.to_string())?;
                let kind = present(item, "type").map(Value::string).transpose().map_err(|e| e.to_string())?.unwrap_or("message");
                match kind {
                    "message" => {
                        let role = string(item, "role")?;
                        if !matches!(role, "system" | "developer" | "user" | "assistant") {
                            return Err("unsupported Responses message role".into());
                        }
                        let content = match item.get("content") {
                            // Match Chat's empty assistant turn after reasoning-only output.
                            None | Some(Value::Null) if role == "assistant" => String::new(),
                            _ => text(field(item, "content")?)?,
                        };
                        if role == "assistant" {
                            assistant.present = true;
                            assistant.content.push_str(&content);
                        } else {
                            assistant.flush(&mut messages);
                            messages.push(message(role, &content));
                        }
                    }
                    "function_call" | "custom_tool_call" => {
                        let call_id = string(item, "call_id")?;
                        if call_id.is_empty() { return Err("empty function call ID".into()); }
                        let arguments = if kind == "function_call" { string(item, "arguments")?.to_owned() }
                            else { custom_arguments(string(item, "input")?) };
                        assistant.present = true;
                        assistant.calls.push(format!("{{\"id\":{},\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":{}}}}}",
                            quote(call_id), quote(&function_name(item)?), quote(&arguments)));
                    }
                    "function_call_output" | "custom_tool_call_output" => {
                        let call_id = string(item, "call_id")?;
                        if call_id.is_empty() { return Err("empty function call ID".into()); }
                        assistant.flush(&mut messages);
                        messages.push(format!("{{\"role\":\"tool\",\"tool_call_id\":{},\"content\":{}}}",
                            quote(call_id), quote(&text(field(item, "output")?)?)));
                    }
                    "reasoning" => {
                        // Opaque encrypted state has no meaning to a local model.
                        if let Some(summary) = present(item, "summary") {
                            let summary = text(summary)?;
                            if !summary.is_empty() {
                                assistant.present = true;
                                assistant.reasoning.push_str(&summary);
                            }
                        }
                    }
                    _ => return Err("unsupported Responses input item; send full text history".into()),
                }
            }
            assistant.flush(&mut messages);
        }
        _ => return Err("input must be a string or an array of items".into()),
    }
    if messages.len() == instruction_messages { return Err("Responses requires full input history".into()); }
    let mut output = Output { tool_hash: crate::http::request_hash(body), ..Output::default() };
    let mut functions = Vec::new();
    if !matches!(choice, Some(Value::String(s)) if s == "none") {
        if let Some(raw) = raw_member(source, "tools")?.filter(|s| s.trim() != "null") {
            tools(raw, None, limits, &mut output, &mut functions)?;
        }
    }
    let mut translated = vec![format!("\"messages\":[{}]", messages.join(",")),
        format!("\"tools\":[{}]", functions.join(",")), format!("\"tool_choice\":{chat_choice}")];
    for key in ["model", "stream", "temperature", "top_p", "top_k", "min_p", "seed", "stop",
        "repetition_penalty", "presence_penalty", "frequency_penalty", "chat_template_kwargs",
        "enable_thinking", "cache", "cache_salt", "ignore_eos", "session", "cancel", "priority", "fork", "draft", "mtp_depth",
        "prompt_cache_key", "include_usage_timing", "kv_checkpoint", "kv_handle", "parallel_tool_calls"] {
        // Ignored fields must not consume the translated body's byte budget.
        if key == "mtp_depth" && limits.next_mtp_depth.is_none() {
            continue;
        }
        if let Some(raw) = raw_member(source, key)?
            .filter(|s| s.trim() != "null" || protocol::SUPPORTED_SAMPLING_FIELDS.contains(&key)) {
            translated.push(format!("{}:{raw}", quote(key)));
        }
    }
    // These compatibility controls have no value-level implementation here.
    // Preserve their named presence without copying ignored payloads into the
    // translated body's byte budget.
    for key in engine_model::sampling_policy::FIELDS.split(',')
        .filter(|key| !protocol::SUPPORTED_SAMPLING_FIELDS.contains(key)) {
        if object.contains_key(key) { translated.push(format!("{}:null", quote(key))); }
    }
    if let Some(raw) = raw_member(source, "max_output_tokens")?.filter(|s| s.trim() != "null") {
        translated.push(format!("\"max_tokens\":{raw}"));
    }
    // Chat parsing maps the effort to thinking, so both endpoints agree.
    if let Some(effort) = object.get("reasoning").filter(|v| **v != Value::Null)
        .map(Value::object).transpose().map_err(|e| e.to_string())?
        .and_then(|o| o.get("effort")).filter(|v| **v != Value::Null) {
        translated.push(format!("\"reasoning_effort\":{}", protocol::stringify(effort)));
    }
    if let Some(raw) = raw_member(source, "arm")? { translated.push(format!("\"arm\":{raw}")); }
    let body = format!("{{{}}}", translated.join(","));
    Ok((protocol::parse_with_reservation(body.as_bytes(), true, limits, model, reservation)?, output))
}

enum Item {
    Message(String),
    Reasoning(String),
    /// A function call, or a custom tool call when it decodes freeform input from its arguments.
    Call { name: String, namespace: Option<String>, arguments: String, call: usize, freeform: Option<Freeform> },
}
/// A custom call's input, decoded from the arguments the model writes for it: the `input` string of
/// `{"input": ...}`, or a bare JSON string. Text after that string is ignored. Arguments that do not
/// open either way are delivered exactly as written. Decoding is incremental and shared by both
/// modes, so streamed deltas always concatenate to the final input.
#[derive(Default)]
struct Freeform {
    input: String,
    stage: Stage,
    /// Opening text not yet decided, or an incomplete escape inside the string.
    held: String,
    /// A UTF-16 high surrogate awaiting its low half.
    high: Option<u32>,
}
#[derive(Default, Clone, Copy)]
enum Stage { #[default] Opening, Text, Done, Raw }
impl Freeform {
    /// Returns the input text decoded from this argument chunk.
    fn push(&mut self, arguments: &str) -> String {
        let before = self.input.len();
        for ch in arguments.chars() {
            match self.stage {
                Stage::Opening => {
                    self.held.push(ch);
                    // Whitespace cannot settle the opening; skipping the rescan keeps this linear.
                    if matches!(ch, ' ' | '\t' | '\r' | '\n') { continue; }
                    match opening(&self.held) {
                        Some(true) => { self.held.clear(); self.stage = Stage::Text; }
                        Some(false) => {}
                        None => { self.input.push_str(&std::mem::take(&mut self.held)); self.stage = Stage::Raw; }
                    }
                }
                Stage::Text => self.text(ch),
                Stage::Done => {}
                Stage::Raw => self.input.push(ch),
            }
        }
        self.input[before..].to_owned()
    }
    /// At the end of the call, undecided or incomplete text is delivered as written.
    fn finish(&mut self) -> String {
        let before = self.input.len();
        if matches!(self.stage, Stage::Text) { self.unpaired(); }
        if matches!(self.stage, Stage::Opening | Stage::Text) { self.input.push_str(&self.held); }
        self.held.clear();
        self.stage = Stage::Done;
        self.input[before..].to_owned()
    }
    fn text(&mut self, ch: char) {
        if self.held.is_empty() {
            match ch {
                '"' => { self.unpaired(); self.stage = Stage::Done; }
                '\\' => self.held.push(ch),
                _ => { self.unpaired(); self.input.push(ch); }
            }
            return;
        }
        self.held.push(ch);
        if self.held == "\\u" { return; }
        let unit = match self.held.strip_prefix("\\u") {
            Some(_) if !ch.is_ascii_hexdigit() => None,
            Some(hex) if hex.len() < 4 => return,
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => match ch {
                '"' | '\\' | '/' => Some(ch as u32),
                'b' => Some(8), 'f' => Some(12), 'n' => Some(10), 'r' => Some(13), 't' => Some(9),
                _ => None,
            },
        };
        if let Some(unit) = unit {
            self.held.clear();
            self.unit(unit);
        } else {
            // An invalid escape stays literal; its last character is read again as ordinary text.
            self.held.pop();
            self.unpaired();
            self.input.push_str(&std::mem::take(&mut self.held));
            self.text(ch);
        }
    }
    fn unit(&mut self, unit: u32) {
        if let Some(high) = self.high.take() {
            if (0xDC00..0xE000).contains(&unit) {
                self.input.push(char::from_u32(0x10000 + ((high - 0xD800) << 10) + (unit - 0xDC00)).unwrap());
                return;
            }
            self.input.push(char::REPLACEMENT_CHARACTER);
        }
        if (0xD800..0xDC00).contains(&unit) { self.high = Some(unit); }
        else { self.input.push(char::from_u32(unit).unwrap_or(char::REPLACEMENT_CHARACTER)); }
    }
    fn unpaired(&mut self) {
        if self.high.take().is_some() { self.input.push(char::REPLACEMENT_CHARACTER); }
    }
}
/// Whether arguments open the input string: Some(true) once they have, Some(false) while they still
/// may, None when they cannot.
fn opening(text: &str) -> Option<bool> {
    let space = |c: char| matches!(c, ' ' | '\t' | '\r' | '\n');
    let tokens: &[&str] = if text.trim_start_matches(space).starts_with('"') { &["\""] }
        else { &["{", "\"input\"", ":", "\""] };
    let mut rest = text;
    for token in tokens {
        rest = rest.trim_start_matches(space);
        if !rest.starts_with(token) { return token.starts_with(rest).then_some(false); }
        rest = &rest[token.len()..];
    }
    Some(true)
}
#[derive(Default)]
pub(crate) struct Output {
    tool_hash: u64,
    items: Vec<Item>,
    open: Option<usize>,
    sequence: usize,
    bytes: usize,
    namespaces: BTreeMap<String, (String, String)>,
    custom: BTreeSet<String>,
}
impl Output {
    fn event(&mut self, kind: &str, fields: &str) -> String {
        let sequence = self.sequence;
        self.sequence += 1;
        format!("event: {kind}\ndata: {{\"type\":\"{kind}\",\"sequence_number\":{sequence},{fields}}}\n\n")
    }
    fn item_id(&self, id: u64, index: usize) -> String {
        format!("item-{id}-{index}")
    }
    fn part(&self, index: usize, empty: bool) -> String {
        let (kind, text) = match &self.items[index] {
            Item::Message(s) => ("output_text", s),
            Item::Reasoning(s) => ("summary_text", s),
            Item::Call { .. } => unreachable!(),
        };
        let annotations = if kind == "output_text" { ",\"annotations\":[]" } else { "" };
        format!("{{\"type\":\"{kind}\",\"text\":{}{annotations}}}", quote(if empty { "" } else { text }))
    }
    fn item(&self, id: u64, index: usize, empty: bool) -> String {
        let item_id = quote(&self.item_id(id, index));
        let status = if empty || (self.open == Some(index) && matches!(self.items[index], Item::Call { .. })) { "in_progress" } else { "completed" };
        match &self.items[index] {
            Item::Message(_) => format!("{{\"id\":{item_id},\"type\":\"message\",\"role\":\"assistant\",\"status\":\"{status}\",\"content\":[{}]}}",
                if empty { String::new() } else { self.part(index, false) }),
            Item::Reasoning(_) => format!("{{\"id\":{item_id},\"type\":\"reasoning\",\"summary\":[{}]}}",
                if empty { String::new() } else { self.part(index, false) }),
            Item::Call { name, namespace, arguments, call, freeform } => {
                let ns = namespace.as_ref().map(|s| format!(",\"namespace\":{}", quote(s))).unwrap_or_default();
                let (kind, field, value) = match freeform {
                    Some(freeform) => ("custom_tool_call", "input", &freeform.input),
                    None => ("function_call", "arguments", arguments),
                };
                format!("{{\"id\":{item_id},\"type\":\"{kind}\",\"call_id\":\"call-{}-{call}\",\"name\":{}{ns},\"{field}\":{},\"status\":\"{status}\"}}",
                    self.tool_hash, quote(name), quote(if empty { "" } else { value }))
            }
        }
    }
    pub fn json(&self, model: &str, id: u64, created: u64, status: &str, usage: Option<Usage>, error: Option<(&str, &str)>) -> String {
        let usage = usage.map(|u| u.json(true)).unwrap_or_else(|| "null".into());
        let error = error.map(|(code, message)| format!("{{\"code\":{},\"message\":{}}}", quote(code), quote(message))).unwrap_or_else(|| "null".into());
        let incomplete = if status == "incomplete" { "{\"reason\":\"max_output_tokens\"}" } else { "null" };
        let items = (0..self.items.len()).map(|i| self.item(id, i, false)).collect::<Vec<_>>().join(",");
        format!("{{\"id\":\"resp-{id}\",\"object\":\"response\",\"created_at\":{created},\"model\":{},\"status\":{},\"output\":[{items}],\"usage\":{usage},\"error\":{error},\"incomplete_details\":{incomplete},\"store\":false}}", quote(model), quote(status))
    }
    pub fn start(&mut self, model: &str, id: u64, created: u64) -> String {
        let fields = format!("\"response\":{}", self.json(model, id, created, "in_progress", None, None));
        let mut result = self.event("response.created", &fields);
        result.push_str(&self.event("response.in_progress", &fields));
        result
    }
    fn close(&mut self, id: u64) -> String {
        let Some(index) = self.open.take() else { return String::new(); };
        let fields = format!("\"item_id\":{},\"output_index\":{index}", quote(&self.item_id(id, index)));
        let mut result = String::new();
        match &self.items[index] {
            Item::Message(s) | Item::Reasoning(s) => {
                let reasoning = matches!(self.items[index], Item::Reasoning(_));
                let (text_event, part_event, position) = if reasoning {
                    ("response.reasoning_summary_text.done", "response.reasoning_summary_part.done", "summary_index")
                } else { ("response.output_text.done", "response.content_part.done", "content_index") };
                result.push_str(&self.event(text_event, &format!("{fields},\"{position}\":0,\"text\":{}", quote(s))));
                result.push_str(&self.event(part_event, &format!("{fields},\"{position}\":0,\"part\":{}", self.part(index, false))));
            }
            Item::Call { arguments, freeform, .. } => {
                let (kind, field, value) = match freeform {
                    Some(freeform) => ("response.custom_tool_call_input.done", "input", &freeform.input),
                    None => ("response.function_call_arguments.done", "arguments", arguments),
                };
                result.push_str(&self.event(kind, &format!("{fields},\"{field}\":{}", quote(value))));
            }
        }
        result.push_str(&self.event("response.output_item.done", &format!("\"output_index\":{index},\"item\":{}", self.item(id, index, false))));
        result
    }
    fn call_delta(&mut self, id: u64, index: usize, delta: &str) -> String {
        if delta.is_empty() { return String::new(); }
        let kind = if matches!(self.items[index], Item::Call { freeform: Some(_), .. }) { "response.custom_tool_call_input.delta" }
            else { "response.function_call_arguments.delta" };
        let fields = format!("\"item_id\":{},\"output_index\":{index},\"delta\":{}", quote(&self.item_id(id, index)), quote(delta));
        self.event(kind, &fields)
    }
    pub fn push(&mut self, event: Event, id: u64, call: &mut usize, limits: &Limits) -> Result<String, String> {
        let mut result = String::new();
        let reasoning = matches!(event, Event::Reasoning(_));
        let text = match event {
            Event::Content(s) | Event::Reasoning(s) => s,
            Event::ToolStart { name } => {
                self.bytes = self.bytes.saturating_add(name.len());
                if self.bytes > limits.output_bytes.saturating_sub(1024) { return Err("output byte limit exceeded".into()); }
                result.push_str(&self.close(id));
                let index = self.items.len();
                let freeform = self.custom.contains(&name).then(Freeform::default);
                let (namespace, name) = self.namespaces.get(&name).map(|(ns, name)| (Some(ns.clone()), name.clone())).unwrap_or((None, name));
                self.items.push(Item::Call { name, namespace, arguments: String::new(), call: *call, freeform });
                self.open = Some(index);
                result.push_str(&self.event("response.output_item.added", &format!("\"output_index\":{index},\"item\":{}", self.item(id, index, true))));
                return Ok(result);
            }
            Event::ToolArguments(arguments) => {
                self.bytes = self.bytes.saturating_add(arguments.len());
                if self.bytes > limits.output_bytes.saturating_sub(1024) { return Err("output byte limit exceeded".into()); }
                let index = self.open.ok_or("tool arguments without a tool start")?;
                let Item::Call { arguments: pending, freeform, .. } = &mut self.items[index] else {
                    return Err("tool arguments without a tool start".into());
                };
                pending.push_str(&arguments);
                let delta = match freeform { Some(freeform) => freeform.push(&arguments), None => arguments };
                result.push_str(&self.call_delta(id, index, &delta));
                return Ok(result);
            }
            Event::ToolCall { arguments, .. } => {
                let index = self.open.ok_or("tool completion without a tool start")?;
                let Item::Call { arguments: pending, freeform, .. } = &mut self.items[index] else {
                    return Err("streamed tool arguments differ from completed call".into());
                };
                if *pending != arguments { return Err("streamed tool arguments differ from completed call".into()); }
                let tail = freeform.as_mut().map(Freeform::finish).unwrap_or_default();
                result.push_str(&self.call_delta(id, index, &tail));
                *call += 1;
                result.push_str(&self.close(id));
                return Ok(result);
            }
        };
        if text.is_empty() { return Ok(result); }
        self.bytes = self.bytes.saturating_add(text.len());
        if self.bytes > limits.output_bytes.saturating_sub(1024) { return Err("output byte limit exceeded".into()); }
        if !self.open.is_some_and(|i| matches!((&self.items[i], reasoning), (Item::Reasoning(_), true) | (Item::Message(_), false))) {
            result.push_str(&self.close(id));
            let index = self.items.len();
            self.items.push(if reasoning { Item::Reasoning(String::new()) } else { Item::Message(String::new()) });
            self.open = Some(index);
            result.push_str(&self.event("response.output_item.added", &format!("\"output_index\":{index},\"item\":{}", self.item(id, index, true))));
            let (kind, position) = if reasoning { ("response.reasoning_summary_part.added", "summary_index") } else { ("response.content_part.added", "content_index") };
            result.push_str(&self.event(kind, &format!("\"item_id\":{},\"output_index\":{index},\"{position}\":0,\"part\":{}", quote(&self.item_id(id, index)), self.part(index, true))));
        }
        let index = self.open.unwrap();
        match &mut self.items[index] { Item::Message(s) | Item::Reasoning(s) => s.push_str(&text), _ => unreachable!() }
        let (kind, position) = if reasoning { ("response.reasoning_summary_text.delta", "summary_index") } else { ("response.output_text.delta", "content_index") };
        result.push_str(&self.event(kind, &format!("\"item_id\":{},\"output_index\":{index},\"{position}\":0,\"delta\":{}", quote(&self.item_id(id, index)), quote(&text))));
        Ok(result)
    }
    pub fn finish(&mut self, model: &str, id: u64, created: u64, status: &str, usage: Usage) -> String {
        let mut result = self.close(id);
        let kind = if status == "incomplete" { "response.incomplete" } else { "response.completed" };
        result.push_str(&self.event(kind, &format!("\"response\":{}", self.json(model, id, created, status, Some(usage), None))));
        result
    }
    pub fn failed(&mut self, model: &str, id: u64, created: u64, usage: Usage, code: &str, message: &str) -> String {
        self.event("response.failed", &format!("\"response\":{}", self.json(model, id, created, "failed", Some(usage), Some((code, message)))))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn freeform_input_does_not_depend_on_chunking() {
        for (arguments, input) in [
            (r#"{"input": "a\n\"b\" é😀\/\x"} "#, "a\n\"b\" é😀/\\x"),
            (r#" "bare\tstring""#, "bare\tstring"),
            (r#"{"patch": "x"}"#, r#"{"patch": "x"}"#),
            (r#"{"input": "cut\u00"#, "cut\\u00"),
        ] {
            for split in (0..=arguments.len()).filter(|&i| arguments.is_char_boundary(i)) {
                let mut freeform = super::Freeform::default();
                let deltas = freeform.push(&arguments[..split]) + &freeform.push(&arguments[split..]) + &freeform.finish();
                assert_eq!((deltas.as_str(), freeform.input.as_str()), (input, input), "{arguments} split at {split}");
            }
        }
    }
}
