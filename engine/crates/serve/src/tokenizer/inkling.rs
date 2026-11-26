//! Text-only chat rendering for the checked template copy templates/inkling2-mini.jinja.
//! The bridge selects this renderer by the bytes of that template, never by a model name.
//! Line numbers in the comments and the tests name that file. This file holds its own
//! code and does not run Jinja. Transformers compiles a template with trim_blocks and
//! lstrip_blocks, and its tojson filter writes JSON with no ASCII escapes
//! (utils/chat_template_utils.py, `_compile_jinja_template`).
use crate::protocol::{ChatInput, EffortAsk, IMAGE_INPUT_UNSUPPORTED};
use engine_model::json::Value;
use engine_tokenizer::chat::quote;
use std::ops::Range;

pub const TEMPLATE: &str = include_str!("../../templates/inkling2-mini.jinja");
/// The `response_template` of the same checkpoint's tokenizer_config.json: its reply grammar.
pub(super) const RESPONSE_TEMPLATE: &str = include_str!("../../templates/inkling2-mini-response.json");
/// The end-of-turn control of this markup. A template that holds this spelling and
/// does not have the bytes of the checked copy is refused, not passed to another renderer.
pub(crate) const STOP: &str = "<|content_model_end_sampling|>";
const MODEL: &str = "<|message_model|>";
const TEXT: &str = "<|content_text|>";
const THINKING: &str = "<|content_thinking|>";
const CALL: &str = "<|content_invoke_tool_json|>";
const END: &str = "<|end_message|>";
const USER: &str = "<|message_user|>";
const SYSTEM: &str = "<|message_system|>";
const TOOL: &str = "<|message_tool|>";
const XML: &str = "<|content_xml|>";
/// Line 1 of the template: the number of each effort word.
const EFFORT_WORDS: [(&str, f64); 7] = [("none", 0.0), ("minimal", 0.1), ("low", 0.2), ("medium", 0.7), ("high", 0.9), ("xhigh", 0.99), ("max", 0.99)];
/// Line 5 of the template: the effort when the request names none.
const EFFORT_DEFAULT: f64 = 0.9;
/// Lines 15 to 17 of the template.
const EFFORT_RANGE: &str = "reasoning_effort must be in [0.0, 0.99]";
const BROKEN: &str = "chat renderer read invalid JSON";

/// Each control of the template must be one special token with the same spelling.
/// Returns the ID of the end-of-turn token.
pub(super) fn validate_tokens(tokenizer: &engine_tokenizer::Tokenizer) -> Result<u32, String> {
    for control in [USER, MODEL, SYSTEM, TOOL, TEXT, THINKING, XML, CALL, END, STOP] { control_id(tokenizer, control)?; }
    control_id(tokenizer, STOP)
}
fn control_id(tokenizer: &engine_tokenizer::Tokenizer, control: &str) -> Result<u32, String> {
    let ids = tokenizer.encode(control).map_err(|e| e.to_string())?;
    if ids.len() != 1 || !tokenizer.is_special(ids[0]) || tokenizer.token_bytes(ids[0]).map_err(|e| e.to_string())? != control.as_bytes() {
        return Err(format!("chat template and tokenizer disagree on the control token {control}"));
    }
    Ok(ids[0])
}

pub(super) struct Output { pub(super) text: String, pub(super) controls: Vec<Range<usize>>, limit: usize }
impl Output {
    fn text(&mut self, s: &str) -> Result<(), String> {
        if self.text.len().checked_add(s.len()).is_none_or(|n| n > self.limit) {
            return Err("chat template byte limit exceeded".into());
        }
        self.text.push_str(s); Ok(())
    }
    fn control(&mut self, s: &str) -> Result<(), String> {
        let start = self.text.len(); self.text(s)?;
        self.controls.push(start..self.text.len()); Ok(())
    }
    /// One message: header, content kind, body, end (lines 72, 76, 80, 83).
    fn message(&mut self, role: &str, name: &str, kind: &str, body: &str) -> Result<(), String> {
        self.control(role)?; self.text(name)?; self.control(kind)?; self.text(body)?; self.control(END)
    }
}
/// The rendered prompt as text. A control spelling in caller text reads the same as
/// a control of the template here. Use `encode` for inference.
pub fn format(input: &ChatInput, max_bytes: usize) -> Result<String, String> {
    Ok(render(input, max_bytes)?.text)
}
/// Only the controls of the template become control tokens.
pub fn encode(tokenizer: &engine_tokenizer::Tokenizer, input: &ChatInput, max_bytes: usize,
    limits: engine_tokenizer::Limits) -> Result<Vec<u32>, String> {
    let out = render(input, max_bytes)?;
    if out.text.len() > limits.max_input_bytes { return Err("input byte limit exceeded".into()); }
    let (mut tokens, mut start) = (Vec::new(), 0);
    for control in &out.controls {
        plain(tokenizer, &out.text[start..control.start], &mut tokens)?;
        tokens.push(control_id(tokenizer, &out.text[control.clone()])?);
        if tokens.len() > limits.max_tokens { return Err("token count limit exceeded".into()); }
        start = control.end;
    }
    plain(tokenizer, &out.text[start..], &mut tokens)?;
    if tokens.len() > limits.max_tokens { return Err("token count limit exceeded".into()); }
    Ok(tokens)
}
/// Caller text never becomes a control token. The spelling of a special token in
/// caller text is encoded alone, as ordinary text, and the text on each side of it is
/// encoded on its own. The reference library makes the same cut when it is asked to
/// keep control tokens out of caller text (utils/chat_template_utils.py,
/// `encode_marked_chat`).
fn plain(tokenizer: &engine_tokenizer::Tokenizer, text: &str, tokens: &mut Vec<u32>) -> Result<(), String> {
    if text.is_empty() { return Ok(()); }
    for id in tokenizer.encode(text).map_err(|e| e.to_string())? {
        if tokenizer.is_special(id) {
            let spelling = std::str::from_utf8(tokenizer.token_bytes(id).map_err(|e| e.to_string())?)
                .map_err(|_| "control token spelling is not UTF-8")?;
            tokens.extend(tokenizer.encode_ordinary(spelling).map_err(|e| e.to_string())?);
        } else { tokens.push(id); }
    }
    Ok(())
}

/// Python writes a float in its shortest form, with a signed exponent of two digits or more.
fn python_float(number: f64) -> String {
    let text = format!("{number:?}");
    match text.split_once('e') {
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') { Some(digits) => ('-', digits), None => ('+', exponent) };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => text,
    }
}
/// What the request log reports for `input`: thinking (the effort number is above 0) and the effort word that
/// `effort_number` renders. A request without an effort or a thinking field gets the template default, `high`.
/// A number is reported as `number`; an unknown word is refused by the render and reported as `unknown`.
pub(crate) fn rendered_effort(input: &ChatInput) -> (bool, &'static str) {
    let word = |number: f64| EFFORT_WORDS.iter().find(|(_, n)| *n == number).map_or("number", |(name, _)| *name);
    match &input.effort_ask {
        EffortAsk::Off | EffortAsk::OffWithNumber => (false, "none"),
        EffortAsk::Word(asked) | EffortAsk::WordWithNumber(asked) => EFFORT_WORDS.iter().find(|(name, _)| *name == asked.trim()).map_or((true, "unknown"), |(name, number)| (*number > 0.0, *name)),
        EffortAsk::Number(number) => (*number > 0.0, word(*number)),
        EffortAsk::Unset => (EFFORT_DEFAULT > 0.0, word(EFFORT_DEFAULT)),
    }
}
/// Lines 4 to 21: the number that the effort message carries.
fn effort_number(input: &ChatInput) -> Result<String, String> {
    let number = match &input.effort_ask {
        // An explicit thinking-off field selects the lowest level of the template.
        EffortAsk::Off | EffortAsk::OffWithNumber => 0.0,
        EffortAsk::Word(word) | EffortAsk::WordWithNumber(word) => EFFORT_WORDS.iter().find(|(name, _)| *name == word.trim()).map(|(_, number)| *number)
            .ok_or_else(|| format!("unknown reasoning_effort {word}: this chat template takes none, minimal, low, medium, high, xhigh, max or a number from 0.0 to 0.99"))?,
        EffortAsk::Number(number) => *number,
        // The request names no effort and no thinking field: the default of the template.
        // A service default and the words of the soft switch in message text are no client ask.
        EffortAsk::Unset => EFFORT_DEFAULT,
    };
    if !(0.0..=0.99).contains(&number) { return Err(EFFORT_RANGE.into()); }
    Ok(if number == 0.0 { "0".into() } else { python_float(number) })
}

/// JSON with the member order of the source text. The shared parser checks the text first.
enum Node { Scalar(String), Array(Vec<Node>), Object(Vec<(String, Node)>) }
struct Walk<'a> { source: &'a str, at: usize }
impl Walk<'_> {
    fn peek(&mut self) -> Result<u8, String> {
        let bytes = self.source.as_bytes();
        while bytes.get(self.at).is_some_and(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n')) { self.at += 1; }
        bytes.get(self.at).copied().ok_or_else(|| BROKEN.into())
    }
    fn string(&mut self) -> Result<String, String> {
        let (bytes, start) = (self.source.as_bytes(), self.at);
        self.at += 1;
        loop {
            match bytes.get(self.at) {
                Some(b'\\') => self.at += 2,
                Some(b'"') => { self.at += 1; break; }
                Some(_) => self.at += 1,
                None => return Err(BROKEN.into()),
            }
        }
        let slice = self.source.get(start..self.at).ok_or(BROKEN)?;
        super::object(slice, slice.len())?.string().map(str::to_owned).map_err(|e| e.to_string())
    }
    fn value(&mut self) -> Result<Node, String> {
        match self.peek()? {
            b'{' => {
                self.at += 1;
                let mut members = Vec::new();
                while self.peek()? != b'}' {
                    if !members.is_empty() { self.at += 1; self.peek()?; }
                    let key = self.string()?;
                    if self.peek()? != b':' { return Err(BROKEN.into()); }
                    self.at += 1;
                    members.push((key, self.value()?));
                    if !matches!(self.peek()?, b',' | b'}') { return Err(BROKEN.into()); }
                }
                self.at += 1;
                Ok(Node::Object(members))
            }
            b'[' => {
                self.at += 1;
                let mut items = Vec::new();
                while self.peek()? != b']' {
                    if !items.is_empty() { self.at += 1; }
                    items.push(self.value()?);
                    if !matches!(self.peek()?, b',' | b']') { return Err(BROKEN.into()); }
                }
                self.at += 1;
                Ok(Node::Array(items))
            }
            b'"' => Ok(Node::Scalar(quote(&self.string()?))),
            _ => {
                let start = self.at;
                let bytes = self.source.as_bytes();
                while bytes.get(self.at).is_some_and(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'+' | b'.')) { self.at += 1; }
                let raw = self.source.get(start..self.at).filter(|raw| !raw.is_empty()).ok_or(BROKEN)?;
                Ok(Node::Scalar(match raw {
                    "null" | "true" | "false" => raw.into(),
                    // Python keeps a whole number as written and reads any other number as a float.
                    _ if raw.contains(['.', 'e', 'E']) => {
                        let number: f64 = raw.parse().map_err(|_| BROKEN)?;
                        if !number.is_finite() { return Err("a number in the request is too large for this chat template".into()); }
                        python_float(number)
                    }
                    "-0" => "0".into(),
                    _ => raw.into(),
                }))
            }
        }
    }
}
fn node(source: &str, max_bytes: usize) -> Result<Node, String> {
    super::object(source, max_bytes)?;
    Walk { source, at: 0 }.value()
}
/// JSON with no spaces. `sort` orders the members of every object by name, as Python does.
fn write(node: &Node, sort: bool, out: &mut String) {
    match node {
        Node::Scalar(text) => out.push_str(text),
        Node::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 { out.push(','); }
                write(item, sort, out);
            }
            out.push(']');
        }
        Node::Object(members) => {
            let mut members: Vec<&(String, Node)> = members.iter().collect();
            if sort { members.sort_by(|a, b| a.0.cmp(&b.0)); }
            out.push('{');
            for (index, (key, value)) in members.into_iter().enumerate() {
                if index > 0 { out.push(','); }
                out.push_str(&quote(key)); out.push(':');
                write(value, sort, out);
            }
            out.push('}');
        }
    }
}
fn raw_member<'a>(source: &'a str, name: &str) -> Result<Option<&'a str>, String> {
    Ok(super::raw_member(source, name)?.filter(|raw| raw.trim() != "null"))
}
fn text_field(source: &str, name: &str, max_bytes: usize) -> Result<Option<String>, String> {
    raw_member(source, name)?.map(|raw| super::object(raw, max_bytes)?.string().map(str::to_owned)
        .map_err(|_| format!("{name} must be a string"))).transpose()
}
/// Lines 26 to 32: one tool of the tool list. The four keys are in name order (line 36).
fn tool_spec(source: &str, max_bytes: usize) -> Result<String, String> {
    if text_field(source, "type", max_bytes)?.as_deref() != Some("function") { return Err("only function tools are supported".into()); }
    let function = raw_member(source, "function")?.ok_or("missing tool function")?;
    super::object(function, max_bytes)?.object().map_err(|e| e.to_string())?;
    let name = text_field(function, "name", max_bytes)?.filter(|name| !name.is_empty()).ok_or("empty tool name")?;
    let description = text_field(function, "description", max_bytes)?.unwrap_or_default();
    let mut out = format!("{{\"description\":{},\"name\":{},\"parameters\":", quote(&description), quote(&name));
    match raw_member(function, "parameters")? {
        Some(parameters) => {
            let parameters = node(parameters, max_bytes)?;
            if !matches!(parameters, Node::Object(_)) { return Err("tool parameters must be an object".into()); }
            write(&parameters, true, &mut out);
        }
        None => out.push_str("{}"),
    }
    out.push_str(",\"type\":\"function\"}");
    Ok(out)
}
struct Call { id: Option<String>, name: String, arguments: String }
/// Lines 99 to 120. The top-level argument names are in name order; a nested object keeps its order.
fn call(source: &str, max_bytes: usize) -> Result<Call, String> {
    super::object(source, max_bytes)?.object().map_err(|e| e.to_string())?;
    if text_field(source, "type", max_bytes)?.is_some_and(|kind| kind != "function") { return Err("only function tool calls are supported".into()); }
    let function = raw_member(source, "function")?.ok_or("missing tool function")?;
    super::object(function, max_bytes)?.object().map_err(|e| e.to_string())?;
    let name = text_field(function, "name", max_bytes)?.ok_or("tool call function name must be a string")?;
    // The template takes an object only (line 108). A client sends the arguments as JSON
    // text, so this renderer reads that text here.
    let serialized;
    let arguments = match raw_member(function, "arguments")? {
        None => "{}",
        Some(raw) if raw.trim_start().starts_with('"') => {
            serialized = super::object(raw, max_bytes)?.string().map(str::to_owned).map_err(|e| e.to_string())?;
            if serialized.trim().is_empty() { "{}" } else { serialized.as_str() }
        }
        Some(raw) => raw,
    };
    let Node::Object(mut members) = node(arguments, max_bytes).map_err(|_| "tool call arguments must be a JSON object")? else {
        return Err("tool call arguments must be a JSON object".into());
    };
    members.sort_by(|a, b| a.0.cmp(&b.0));
    let mut arguments = String::from("{");
    for (index, (key, value)) in members.iter().enumerate() {
        if index > 0 { arguments.push(','); }
        arguments.push_str(&quote(key)); arguments.push(':');
        write(value, false, &mut arguments);
    }
    arguments.push('}');
    Ok(Call { id: text_field(source, "id", max_bytes)?, name, arguments })
}
enum Content { Absent, Text(String), Parts(Vec<String>) }
struct Turn { role: &'static str, name: Option<String>, tool_call_id: Option<String>, reasoning: String, content: Content, calls: Vec<Call> }
/// Lines 78 to 94: one message for each text part. This engine is text only.
fn parts(parts: &[Value]) -> Result<Vec<String>, String> {
    parts.iter().map(|part| {
        let fields = match part {
            Value::String(text) => return Ok(text.clone()),
            Value::Object(fields) => fields,
            _ => return Err("a content part must be a string or an object".into()),
        };
        match fields.get("type") {
            None | Some(Value::String(_)) => {}
            Some(_) => return Err("unsupported content part type".into()),
        }
        match fields.get("type").and_then(|kind| kind.string().ok()) {
            None | Some("text" | "input_text") => Ok(fields.get("text").and_then(|text| text.string().ok()).unwrap_or_default().to_owned()),
            Some("image" | "input_image" | "image_url") => Err(IMAGE_INPUT_UNSUPPORTED.into()),
            Some("audio" | "input_audio" | "audio_url") => Err("audio input is not accepted by this chat template".into()),
            Some(kind) => Err(format!("unsupported content part type: {kind}")),
        }
    }).collect()
}
fn turn(source: &str, max_bytes: usize) -> Result<Turn, String> {
    let root = super::object(source, max_bytes)?;
    let fields = root.object().map_err(|e| e.to_string())?;
    let role = match text_field(source, "role", max_bytes)?.as_deref() {
        // The template refuses the developer role (line 42). Clients send their
        // instructions with it, so it renders as a system message in its place.
        Some("system" | "developer") => SYSTEM,
        Some("user") => USER,
        Some("assistant") => MODEL,
        Some("tool") => TOOL,
        _ => return Err("unsupported chat role".into()),
    };
    let content = match fields.get("content") {
        Some(Value::String(text)) => Content::Text(text.clone()),
        Some(Value::Array(items)) => Content::Parts(parts(items)?),
        // A reply that a length limit cut can hold reasoning or calls and no answer.
        Some(Value::Null) | None if role == MODEL => Content::Absent,
        _ => return Err("message content must be a string or text parts".into()),
    };
    let tool_call_id = text_field(source, "tool_call_id", max_bytes)?;
    if tool_call_id.as_ref().is_some_and(|id| role != TOOL || id.is_empty()) {
        return Err("tool_call_id requires a nonempty string on a tool message".into());
    }
    // The library copies the first of these fields that is set into reasoning_content
    // (utils/chat_template_utils.py, `_alias_reasoning_fields`).
    let mut reasoning = String::new();
    if role == MODEL {
        for name in ["reasoning_content", "thinking", "reasoning"] {
            if let Some(text) = text_field(source, name, max_bytes)? { reasoning = text; break; }
        }
    }
    let mut calls = Vec::new();
    if role == MODEL && let Some(raw) = raw_member(source, "tool_calls")? {
        let raw_calls = super::raw_array(raw)?;
        if raw_calls.len() > 64 { return Err("too many tool calls".into()); }
        for raw_call in &raw_calls { calls.push(call(raw_call, max_bytes)?); }
    }
    Ok(Turn { role, name: text_field(source, "name", max_bytes)?.filter(|name| !name.is_empty()), tool_call_id, reasoning, content, calls })
}
/// Lines 4 to 21.
fn effort(out: &mut Output, number: &str) -> Result<(), String> {
    out.message(SYSTEM, "", TEXT, &format!("Thinking effort level: {number}"))
}
pub(super) fn render(input: &ChatInput, max_bytes: usize) -> Result<Output, String> {
    // The template has no part that opens a call for the model.
    if input.forced_tool_choice() { return Err("forced tool_choice is not supported by this chat renderer".into()); }
    let number = effort_number(input)?;
    let turns = input.messages.iter().map(|source| turn(source, max_bytes)).collect::<Result<Vec<_>, _>>()?;
    let mut out = Output { text: String::new(), controls: Vec::new(), limit: max_bytes };
    // Lines 23 to 38: the tool list.
    let tools = input.callable_tools();
    if !tools.is_empty() {
        let specs = tools.iter().map(|tool| tool_spec(tool, max_bytes)).collect::<Result<Vec<_>, _>>()?;
        out.control(SYSTEM)?; out.text("tool_declare")?; out.control(XML)?;
        out.text(&format!("[{}]", specs.join(",")))?; out.control(END)?;
    }
    let mut effort_emitted = false;
    for (index, turn) in turns.iter().enumerate() {
        // Lines 45 to 48: the effort message goes before the first message that is not a system message.
        if !effort_emitted && turn.role != SYSTEM { effort(&mut out, &number)?; effort_emitted = true; }
        // Lines 50 to 68: the name that follows the role token.
        let mut name = String::new();
        if turn.role == TOOL {
            if let Some(own) = &turn.name { name.push_str(own); }
            else if let Some(id) = &turn.tool_call_id
                && let Some(call) = turns[..index].iter().rev().filter(|other| other.role == MODEL)
                    .flat_map(|other| other.calls.iter().rev()).find(|call| call.id.as_ref() == Some(id)) {
                name.push_str(&call.name);
            }
        } else if turn.role == MODEL && let Some(own) = &turn.name { name.push_str(own); }
        // Lines 71 to 73.
        if turn.role == MODEL && !turn.reasoning.is_empty() { out.message(turn.role, &name, THINKING, &turn.reasoning)?; }
        // Lines 75 to 95.
        match &turn.content {
            Content::Text(text) if !text.is_empty() || !(turn.role == MODEL && (!turn.calls.is_empty() || !turn.reasoning.is_empty())) =>
                out.message(turn.role, &name, TEXT, text)?,
            Content::Parts(parts) => for part in parts { out.message(turn.role, &name, TEXT, part)?; },
            Content::Text(_) | Content::Absent => {}
        }
        // Lines 97 to 122.
        for call in &turn.calls {
            if turn.name.as_ref().is_some_and(|own| *own != call.name) {
                return Err(format!("assistant name {} does not match its tool call {}", turn.name.as_deref().unwrap_or_default(), call.name));
            }
            out.message(MODEL, &call.name, CALL, &format!("{{\"name\":{},\"args\":{}}}", quote(&call.name), call.arguments))?;
        }
        // Lines 124 to 126.
        if turn.role == MODEL { out.control(STOP)?; }
    }
    // Lines 135 to 137. Nothing opens the turn of the model: the prompt ends here.
    if !effort_emitted { effort(&mut out, &number)?; }
    Ok(out)
}
#[cfg(test)]
#[path = "inkling_tests.rs"]
mod tests;
