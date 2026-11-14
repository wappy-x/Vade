//! Incremental Qwen and Gemma chat markup decoding. This module never invokes tools.
//! Declared names stream once complete and argument text streams as it is
//! generated: the exact JSON slice for Qwen3 calls, and a JSON object built
//! from typed XML parameters for Qwen3.5-family calls. A ToolCall completes
//! the call, including malformed or truncated arguments.
#[path = "chat_output/gemma4.rs"]
mod gemma4;
mod glm;
use engine_model::json;
use std::collections::{BTreeMap, BTreeSet};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
/// The parser state broke one of its own rules. The request fails with this
/// error. A panic would stop the server for every request: release builds
/// abort on panic.
const BROKEN: &str = "chat output parser reached an invalid state";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Content(String),
    Reasoning(String),
    ToolStart { name: String },
    ToolArguments(String),
    /// Completes a started call. For Qwen3 JSON calls, arguments is the exact generated slice, including
    /// whitespace and key order; for XML calls, the JSON object built from the typed parameters. The HTTP
    /// layer assigns request-local call IDs and indices.
    ToolCall {
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub total_bytes: usize,
    pub tool_call_bytes: usize,
    pub tool_calls: usize,
    pub name_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            total_bytes: 1024 * 1024,
            tool_call_bytes: 64 * 1024,
            tool_calls: 16,
            name_bytes: 128,
        }
    }
}
impl Limits {
    fn validate(&self) -> Result<(), String> {
        if self.total_bytes == 0
            || self.total_bytes > 16 * 1024 * 1024
            || self.tool_call_bytes == 0
            || self.tool_call_bytes > self.total_bytes
            || self.tool_calls == 0
            || self.tool_calls > 128
            || self.name_bytes == 0
            || self.name_bytes > 1024
        {
            return Err("invalid chat output parser limits".into());
        }
        Ok(())
    }
}

enum Mode {
    Content,
    Reasoning,
    Tool(ToolBuffer),
    Xml(XmlCall),
}

/// An initial think block (optionally preceded by whitespace) becomes reasoning.
/// Later think markers in normal content are literal. Markup inside reasoning
/// cannot invoke tools. With no declared tools all tool markup remains content.
///
/// Calls stream once a declared name is complete. Before that point malformed
/// markup remains literal content and disables tool recognition. After a start,
/// malformed or truncated calls complete with raw arguments: wire deltas cannot be retracted.
/// UTF-8 and total output limits still fail the parser.
/// Unterminated reasoning at a length stop is flushed by finish.
pub struct Parser {
    limits: Limits,
    declared: BTreeSet<String>,
    /// Qwen3.5-family XML calls: per tool, the parameters whose values are JSON, not strings.
    xml: Option<BTreeMap<String, BTreeSet<String>>>,
    gemma: Option<gemma4::Parser>,
    glm: Option<glm::Parser>,
    mode: Mode,
    marker: String,
    initial: bool,
    utf8_tail: Vec<u8>,
    total: usize,
    calls: usize,
    finished: bool,
    failed: bool,
    pending: Vec<Event>,
    forced: bool,
    single_call: bool,
    stopped: bool,
}
impl Parser {
    pub fn new(limits: Limits, allowed_tool_names: Vec<String>) -> Result<Self, String> {
        limits.validate()?;
        if allowed_tool_names.len() > 64 {
            return Err("too many declared tools".into());
        }
        let mut declared = BTreeSet::new();
        for name in allowed_tool_names {
            if name.is_empty()
                || name.len() > limits.name_bytes
                || name.chars().any(char::is_control)
            {
                return Err("invalid declared tool name".into());
            }
            if !declared.insert(name) {
                return Err("duplicate declared tool name".into());
            }
        }
        Ok(Self {
            limits,
            declared,
            xml: None,
            gemma: None,
            glm: None,
            mode: Mode::Content,
            marker: String::new(),
            initial: true,
            utf8_tail: Vec::new(),
            total: 0,
            calls: 0,
            finished: false,
            failed: false,
            pending: Vec::new(),
            forced: false,
            single_call: false,
            stopped: false,
        })
    }

    /// Qwen3.5-family output: XML tool calls typed by the declared schemas, and a
    /// generation prompt that already opened the think block when thinking is on.
    pub fn hybrid(mut self, thinking: bool, json_parameters: BTreeMap<String, BTreeSet<String>>) -> Self {
        self.xml = Some(json_parameters);
        if thinking { self.mode = Mode::Reasoning; self.initial = false; }
        self
    }
    /// Gemma's prompt opens a thought channel only after a tool response.
    pub fn gemma4(mut self, thinking_open: bool) -> Self {
        self.gemma = Some(gemma4::Parser::new(thinking_open));
        self
    }
    /// The pinned GLM template opens reasoning and uses named argument tags.
    /// Calls are emitted only after the complete tagged object is valid.
    pub fn glm(mut self, thinking_open: bool, json_parameters: BTreeMap<String, BTreeSet<String>>) -> Self {
        self.glm = Some(glm::Parser::new(thinking_open, json_parameters));
        self
    }
    pub fn tool_policy(mut self, prefix: &str, forced: bool, single_call: bool) -> Result<Self, String> {
        self.pending = self.push(prefix.as_bytes())?;
        self.forced = forced;
        self.single_call = single_call;
        Ok(self)
    }
    pub fn stopped(&self) -> bool { self.stopped }
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Event>, String> {
        self.live()?;
        let result = self.push_inner(bytes);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn live(&self) -> Result<(), String> {
        if self.failed {
            Err("chat output parser previously failed".into())
        } else if self.finished {
            Err("chat output parser already finished".into())
        } else {
            Ok(())
        }
    }
    fn push_inner(&mut self, bytes: &[u8]) -> Result<Vec<Event>, String> {
        if self.stopped { return Ok(std::mem::take(&mut self.pending)); }
        self.total = self
            .total
            .checked_add(bytes.len())
            .ok_or("chat output byte count overflow")?;
        if self.total > self.limits.total_bytes {
            return Err("chat output byte limit exceeded".into());
        }
        // At most three bytes are retained between calls. The caller's complete
        // chunk is validated before any events from this call can be returned.
        let mut input = std::mem::take(&mut self.utf8_tail);
        input.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&input) {
            Ok(_) => input.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => return Err("invalid UTF-8 in chat output".into()),
        };
        self.utf8_tail = input[valid..].to_vec();
        let text =
            std::str::from_utf8(&input[..valid]).map_err(|_| "invalid UTF-8 in chat output")?;
        let mut events = std::mem::take(&mut self.pending);
        for ch in text.chars() {
            let before = events.len();
            self.character(ch, &mut events)?;
            if self.single_call && events[before..].iter().any(|e| matches!(e, Event::ToolCall { .. })) {
                self.stopped = true;
                self.utf8_tail.clear();
                break;
            }
        }
        self.validate_forced(&events)?;
        Ok(events)
    }
    fn validate_forced(&self, events: &[Event]) -> Result<(), String> {
        if self.forced {
            for event in events {
                match event {
                    Event::Content(s) if !s.trim().is_empty() => return Err("model did not follow forced tool_choice".into()),
                    Event::Reasoning(_) => return Err("model did not follow forced tool_choice".into()),
                    Event::ToolCall { arguments, .. } => {
                        let value = json::parse(arguments.as_bytes(), json::Limits {
                            bytes: self.limits.tool_call_bytes, string_bytes: self.limits.tool_call_bytes,
                            ..json::Limits::default()
                        }).map_err(|_| "model emitted invalid forced tool arguments")?;
                        value.object().map_err(|_| "forced tool arguments must be a JSON object")?;
                    }
                    _ => (),
                }
            }
        }
        Ok(())
    }
    fn character(&mut self, ch: char, events: &mut Vec<Event>) -> Result<(), String> {
        if let Some(parser) = &mut self.glm {
            return parser.push(ch, &self.limits, &mut self.declared, events);
        }
        if let Some(parser) = &mut self.gemma {
            return parser.push(ch, &self.limits, &mut self.declared, events);
        }
        if let Mode::Xml(call) = &mut self.mode {
            let Some(json) = self.xml.as_ref() else { return Err(BROKEN.into()) };
            match call.push(ch, &self.declared, json, &self.limits, self.calls, events) {
                Err(_) => self.tool_content(events)?,
                Ok(true) => {
                    self.calls += 1;
                    events.push(call.call());
                    self.mode = Mode::Content;
                }
                Ok(false) => {}
            }
            return Ok(());
        }
        if let Mode::Tool(buffer) = &mut self.mode {
            let complete = buffer.push(ch, self.limits.tool_call_bytes)
                .and_then(|done| {
                    buffer.progress(&self.declared, &self.limits, self.calls, events, false)?;
                    Ok(done)
                });
            match complete {
                Err(_) => self.tool_content(events)?,
                Ok(true) => {
                    let Mode::Tool(buffer) = std::mem::replace(&mut self.mode, Mode::Content) else {
                        return Err(BROKEN.into());
                    };
                    if buffer.began {
                        self.calls += 1;
                        events.push(buffer.call()?);
                    } else {
                        self.mode = Mode::Tool(buffer);
                        self.tool_content(events)?;
                    }
                }
                Ok(false) => {}
            }
            return Ok(());
        }
        if self.marker.is_empty() && ch != '<' {
            self.literal(ch, events);
            return Ok(());
        }
        self.marker.push(ch);
        loop {
            let candidates = match self.mode {
                Mode::Reasoning => [Some(THINK_CLOSE), None],
                Mode::Content => [
                    self.initial.then_some(THINK_OPEN),
                    (!self.declared.is_empty()).then_some(TOOL_OPEN),
                ],
                Mode::Tool(_) | Mode::Xml(_) => return Err(BROKEN.into()),
            };
            if let Some(full) = candidates.into_iter().flatten().find(|m| *m == self.marker) {
                self.marker.clear();
                self.initial = false;
                self.mode = match full {
                    THINK_OPEN => Mode::Reasoning,
                    THINK_CLOSE => Mode::Content,
                    TOOL_OPEN if self.xml.is_some() => Mode::Xml(XmlCall::default()),
                    TOOL_OPEN => Mode::Tool(ToolBuffer::new()),
                    _ => return Err(BROKEN.into()),
                };
                return Ok(());
            }
            if candidates
                .into_iter()
                .flatten()
                .any(|m| m.starts_with(&self.marker))
            {
                return Ok(());
            }
            // A mismatch may itself end with '<', so flush just one scalar and
            // retry the bounded suffix rather than lose a possible next marker.
            let Some(first) = self.marker.chars().next() else { return Err(BROKEN.into()) };
            self.marker.drain(..first.len_utf8());
            self.literal(first, events);
            if self.marker.is_empty() {
                return Ok(());
            }
        }
    }
    fn literal(&mut self, ch: char, events: &mut Vec<Event>) {
        if matches!(self.mode, Mode::Reasoning) {
            if let Some(Event::Reasoning(text)) = events.last_mut() {
                text.push(ch);
            } else {
                events.push(Event::Reasoning(ch.to_string()));
            }
        } else {
            if !ch.is_whitespace() {
                self.initial = false;
            }
            if let Some(Event::Content(text)) = events.last_mut() {
                text.push(ch);
            } else {
                events.push(Event::Content(ch.to_string()));
            }
        }
    }
    fn tool_content(&mut self, events: &mut Vec<Event>) -> Result<(), String> {
        let text = match std::mem::replace(&mut self.mode, Mode::Content) {
            Mode::Tool(buffer) => buffer.text,
            Mode::Xml(call) => call.text,
            _ => return Err(BROKEN.into()),
        };
        self.declared.clear();
        for ch in TOOL_OPEN.chars().chain(text.chars()) {
            self.literal(ch, events);
        }
        Ok(())
    }
    pub fn finish(&mut self) -> Result<Vec<Event>, String> {
        self.live()?;
        let result = self.finish_inner().and_then(|events| {
            self.validate_forced(&events)?;
            if self.forced && self.calls == 0 { return Err("model did not complete a forced tool call".into()); }
            Ok(events)
        });
        if result.is_err() {
            self.failed = true;
        } else {
            self.finished = true;
        }
        result
    }
    fn finish_inner(&mut self) -> Result<Vec<Event>, String> {
        if !self.utf8_tail.is_empty() {
            return Err("incomplete UTF-8 in chat output".into());
        }
        let mut events = std::mem::take(&mut self.pending);
        if let Some(parser) = &mut self.glm {
            parser.finish(&mut events);
            return Ok(events);
        }
        if let Some(parser) = &mut self.gemma {
            parser.finish(&mut events);
            return Ok(events);
        }
        if let Mode::Tool(buffer) = &mut self.mode {
            if buffer.began {
                // At EOS/length any withheld argument bytes are now unambiguous.
                buffer.progress(&self.declared, &self.limits, self.calls, &mut events, true)?;
                events.push(buffer.call()?);
                self.calls += 1;
                self.mode = Mode::Content;
            } else { self.tool_content(&mut events)?; }
        } else if let Mode::Xml(call) = &mut self.mode {
            if call.began {
                call.finish(&mut events);
                events.push(call.call());
                self.calls += 1;
                self.mode = Mode::Content;
            } else { self.tool_content(&mut events)?; }
        }
        // A fully recognized opener changes mode; a partial opener is literal
        // text, never an executable call. A partial reasoning closer stays in
        // reasoning, as does all reasoning truncated by a token budget.
        let marker = std::mem::take(&mut self.marker);
        for ch in marker.chars() {
            self.literal(ch, &mut events);
        }
        Ok(events)
    }
}

#[derive(Clone, Copy)]
enum Field { Key, KeyString, Colon, Value, Body, After }

struct ToolBuffer {
    text: String,
    stack: Vec<char>,
    quoted: bool,
    escaped: bool,
    started: bool,
    json_end: usize,
    closing: String,
    bytes: usize,
    field: Field,
    key: String,
    value_start: usize,
    name: Option<String>,
    arguments_start: Option<usize>,
    arguments_end: Option<usize>,
    sent: usize,
    began: bool,
}
impl ToolBuffer {
    fn new() -> Self {
        Self {
            text: String::new(),
            stack: Vec::new(),
            quoted: false,
            escaped: false,
            started: false,
            json_end: 0,
            closing: String::new(),
            bytes: 0,
            field: Field::Key,
            key: String::new(),
            value_start: 0,
            name: None,
            arguments_start: None,
            arguments_end: None,
            sent: 0,
            began: false,
        }
    }
    /// Returns true at the closing tag, even if a started call has malformed JSON.
    fn push(&mut self, ch: char, limit: usize) -> Result<bool, String> {
        let depth = self.stack.len();
        let quoted = self.quoted;
        let done = self.push_json(ch, limit)?;
        if depth != 0 && self.closing.is_empty()
            && let Err(error) = self.field(ch, depth, quoted)
                && !self.began { return Err(error); }
        Ok(done)
    }
    fn call(&self) -> Result<Event, String> {
        let arguments = self.arguments_start.map(|start| {
            &self.text[start..self.arguments_end.unwrap_or(self.text.len())]
        }).unwrap_or("");
        Ok(Event::ToolCall {
            name: self.name.clone().ok_or(BROKEN)?,
            arguments: arguments.to_owned(),
        })
    }
    fn push_json(&mut self, ch: char, limit: usize) -> Result<bool, String> {
        // Preserve whitespace, closing tags and the offending character on
        // fallback without keeping a second copy of the candidate call.
        self.text.push(ch);
        self.bytes = self
            .bytes
            .checked_add(ch.len_utf8())
            .ok_or("tool call byte count overflow")?;
        if self.bytes > limit && !self.began {
            return Err("tool call byte limit exceeded".into());
        }
        // A closing tag outside a quoted argument also terminates malformed
        // JSON with missing braces. Keep only the possible marker in reserve.
        if self.began && !self.quoted && (ch == '<' || !self.closing.is_empty()) {
            if ch == '<' { self.closing.clear(); }
            self.closing.push(ch);
            if self.closing == TOOL_CLOSE {
                if self.arguments_start.is_some() && self.arguments_end.is_none() {
                    self.arguments_end = Some(self.text.len() - TOOL_CLOSE.len());
                }
                return Ok(true);
            }
            if TOOL_CLOSE.starts_with(&self.closing) { return Ok(false); }
            self.closing.clear();
        }
        if self.json_end != 0 {
            if self.closing.is_empty() && json_space(ch) {
                return Ok(false);
            }
            self.closing.push(ch);
            if !TOOL_CLOSE.starts_with(&self.closing) && !self.began {
                return Err("expected tool call closing tag".into());
            }
            return Ok(self.closing == TOOL_CLOSE);
        }
        if !self.started {
            if !json_space(ch) {
                if ch != '{' {
                    return Err("tool call must contain a JSON object".into());
                }
                self.started = true;
                self.stack.push('{');
            }
            return Ok(false);
        }
        if self.quoted {
            if self.escaped {
                self.escaped = false;
            } else if ch == '\\' {
                self.escaped = true;
            } else if ch == '"' {
                self.quoted = false;
            }
            return Ok(false);
        }
        match ch {
            '"' => self.quoted = true,
            '{' | '[' => {
                if self.stack.len() >= 24 && !self.began {
                    return Err("tool call JSON depth limit exceeded".into());
                }
                self.stack.push(ch);
            }
            '}' | ']' => {
                let opening = self.stack.pop().unwrap_or('{');
                if (opening == '{') != (ch == '}') && !self.began {
                    return Err("invalid tool call JSON nesting".into());
                }
                if self.stack.is_empty() {
                    self.json_end = self.text.len();
                }
            }
            _ => {}
        }
        Ok(false)
    }
    // Track root members without validating or rewriting argument JSON. Once a
    // declared name starts, invalid arguments belong to the delivered call.
    fn field(&mut self, ch: char, depth: usize, quoted: bool) -> Result<(), String> {
        match self.field {
            Field::Key if depth == 1 && !quoted => {
                if json_space(ch) || ch == '}' { return Ok(()); }
                if ch != '"' { return Err("invalid tool field name".into()); }
                self.value_start = self.text.len() - 1;
                self.field = Field::KeyString;
            }
            Field::KeyString if depth == 1 && quoted && !self.quoted => {
                self.key = self.string_value()?;
                self.field = Field::Colon;
                if !self.began && ((self.key != "name" && self.key != "arguments")
                    || (self.key == "name" && self.name.is_some())
                    || (self.key == "arguments" && self.arguments_start.is_some())) {
                    return Err("tool call requires exactly name and arguments".into());
                }
            }
            Field::Colon if !quoted => {
                if json_space(ch) { return Ok(()); }
                if ch != ':' { return Err("invalid tool field separator".into()); }
                self.field = Field::Value;
            }
            Field::Value if !quoted => {
                if json_space(ch) { return Ok(()); }
                self.value_start = self.text.len() - ch.len_utf8();
                if self.key == "name" {
                    if ch != '"' { return Err("tool name must be a string".into()); }
                } else if self.key == "arguments" && self.arguments_start.is_none() {
                    if ch != '{' && !self.began { return Err("tool arguments must be a JSON object".into()); }
                    self.arguments_start = Some(self.value_start);
                    self.sent = self.value_start;
                    if ch == '}' {
                        self.arguments_end = Some(self.value_start);
                    }
                }
                self.field = Field::Body;
            }
            Field::Body => {
                if self.key == "name" && depth == 1 && quoted && !self.quoted {
                    if self.name.is_none() { self.name = Some(self.string_value()?); }
                    self.field = Field::After;
                } else if self.key == "arguments" && self.arguments_end.is_none() {
                    let start = self.arguments_start.unwrap_or(self.text.len());
                    let first = self.text[start..].chars().next();
                    let container = matches!(first, Some('{') | Some('['));
                    let string = first == Some('"');
                    if (container && !quoted && self.stack.len() == 1)
                        || (string && quoted && !self.quoted) {
                        self.arguments_end = Some(self.text.len());
                        self.field = Field::After;
                    } else if !container && !string && !quoted
                        && (json_space(ch) || ch == ',' || ch == '}') {
                        self.arguments_end = Some(self.text.len() - ch.len_utf8());
                        self.field = if ch == ',' { Field::Key } else { Field::After };
                    }
                } else if !quoted && self.stack.len() == 1 {
                    self.field = Field::After;
                }
            }
            Field::After if depth == 1 && !quoted => {
                if json_space(ch) || ch == '}' { return Ok(()); }
                if ch != ',' { return Err("invalid tool field separator".into()); }
                self.field = Field::Key;
            }
            _ => {}
        }
        Ok(())
    }
    #[allow(clippy::sliced_string_as_bytes)] // the str slice checks char boundaries.
    fn string_value(&self) -> Result<String, String> {
        let value = json::parse(self.text[self.value_start..].as_bytes(), json::Limits::default())
            .map_err(|_| "invalid tool field string")?;
        value.string().map(str::to_owned).map_err(|_| "tool field must be a string".into())
    }
    fn progress(&mut self, declared: &BTreeSet<String>, limits: &Limits, calls: usize,
        events: &mut Vec<Event>, finishing: bool) -> Result<(), String> {
        if !self.began {
            let Some(name) = &self.name else { return Ok(()); };
            if name.len() > limits.name_bytes || !declared.contains(name) {
                return Err("undeclared tool name".into());
            }
            if calls >= limits.tool_calls { return Err("tool call count limit exceeded".into()); }
            events.push(Event::ToolStart { name: name.clone() });
            self.began = true;
        }
        if self.arguments_start.is_some() {
            let end = self.arguments_end.unwrap_or_else(|| {
                if !finishing && TOOL_CLOSE.starts_with(&self.closing) {
                    self.text.len() - self.closing.len()
                } else { self.text.len() }
            });
            if end > self.sent {
                let text = &self.text[self.sent..end];
                if let Some(Event::ToolArguments(pending)) = events.last_mut() {
                    pending.push_str(text);
                } else { events.push(Event::ToolArguments(text.to_owned())); }
                self.sent = end;
            }
        }
        Ok(())
    }

}
fn json_space(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\r' | '\n')
}

const FUNCTION_OPEN: &str = "<function=";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";
const FUNCTION_CLOSE: &str = "</function>";

#[derive(Default, PartialEq)]
enum Xml { #[default] Open, Name, Between, Parameter, Value, End }

/// One `<function=NAME>` call (Qwen3.5 family). Arguments are the JSON object the
/// template would render back to the same text: string parameters stream as JSON
/// strings, other parameters are emitted as JSON (or a string if invalid) at their
/// close tag. One newline after the open tag and before the close tag belongs to the
/// markup. Values end only at `</parameter>`; repeated parameter names are dropped.
/// The object closes at `</function>` (or `</tool_call>`); a call cut off earlier keeps
/// its partial, invalid arguments, identically in both response modes.
#[derive(Default)]
struct XmlCall {
    text: String,
    bytes: usize,
    state: Xml,
    marker: String,
    name: String,
    began: bool,
    arguments: String,
    seen: BTreeSet<String>,
    parameter: String,
    string: bool,
    skip: bool,
    value: String,
    fresh: bool,
}
impl XmlCall {
    fn emit(&mut self, text: &str, events: &mut Vec<Event>) {
        if text.is_empty() { return; }
        self.arguments.push_str(text);
        if let Some(Event::ToolArguments(pending)) = events.last_mut() {
            pending.push_str(text);
        } else { events.push(Event::ToolArguments(text.to_owned())); }
    }
    fn call(&self) -> Event {
        Event::ToolCall { name: self.name.clone(), arguments: self.arguments.clone() }
    }
    /// Err only before the call began: the caller restores the text as content.
    fn push(&mut self, ch: char, declared: &BTreeSet<String>, json: &BTreeMap<String, BTreeSet<String>>,
        limits: &Limits, calls: usize, events: &mut Vec<Event>) -> Result<bool, String> {
        if !self.began {
            self.text.push(ch);
            self.bytes += ch.len_utf8();
            if self.bytes > limits.tool_call_bytes { return Err("tool call byte limit exceeded".into()); }
        }
        match self.state {
            Xml::Open => {
                if self.marker.is_empty() && json_space(ch) { return Ok(false); }
                self.marker.push(ch);
                if self.marker == FUNCTION_OPEN { self.marker.clear(); self.state = Xml::Name; }
                else if !FUNCTION_OPEN.starts_with(&self.marker) { return Err("expected function tag".into()); }
            }
            Xml::Name if ch == '>' => {
                if !declared.contains(&self.name) { return Err("undeclared tool name".into()); }
                if calls >= limits.tool_calls { return Err("tool call count limit exceeded".into()); }
                events.push(Event::ToolStart { name: self.name.clone() });
                self.began = true;
                self.text = String::new();
                self.emit("{", events);
                self.state = Xml::Between;
            }
            Xml::Name => {
                if ch.is_control() || self.name.len() + ch.len_utf8() > limits.name_bytes {
                    return Err("invalid tool name".into());
                }
                self.name.push(ch);
            }
            Xml::Between | Xml::End => {
                if self.marker.is_empty() && ch != '<' { return Ok(false); }
                if ch == '<' { self.marker.clear(); }
                self.marker.push(ch);
                if self.marker == TOOL_CLOSE {
                    if self.state == Xml::Between { self.emit("}", events); }
                    return Ok(true);
                }
                if self.state == Xml::Between && self.marker == FUNCTION_CLOSE {
                    self.emit("}", events);
                    self.state = Xml::End;
                } else if self.state == Xml::Between && self.marker == PARAMETER_OPEN {
                    self.parameter.clear();
                    self.state = Xml::Parameter;
                } else if ![TOOL_CLOSE, FUNCTION_CLOSE, PARAMETER_OPEN].iter().any(|m| m.starts_with(&self.marker)) {
                    self.marker.clear();
                    return Ok(false);
                } else { return Ok(false); }
                self.marker.clear();
            }
            Xml::Parameter if ch == '>' && !self.parameter.is_empty() => {
                self.skip = !self.seen.insert(self.parameter.clone());
                self.string = !json.get(&self.name).is_some_and(|p| p.contains(&self.parameter));
                if self.string && !self.skip {
                    let key = format!("{}{}: \"", if self.seen.len() > 1 { ", " } else { "" }, quote(&self.parameter));
                    self.emit(&key, events);
                }
                self.value.clear();
                self.fresh = true;
                self.state = Xml::Value;
            }
            Xml::Parameter => {
                if ch == '<' || ch == '>' || ch.is_control() || self.parameter.len() + ch.len_utf8() > limits.name_bytes {
                    self.state = Xml::Between;
                    self.marker = if ch == '<' { "<".into() } else { String::new() };
                } else { self.parameter.push(ch); }
            }
            Xml::Value => {
                if std::mem::take(&mut self.fresh) && ch == '\n' { return Ok(false); }
                self.value.push(ch);
                if let Some(value) = self.value.strip_suffix(PARAMETER_CLOSE) {
                    let value = value.strip_suffix('\n').unwrap_or(value).to_owned();
                    if !self.skip {
                        if self.string {
                            let quoted = quote(&value);
                            self.emit(&quoted[1..], events);
                        } else {
                            let value = engine_tokenizer::chat::json_for_template(&value)
                                .unwrap_or_else(|_| quote(&value));
                            let member = format!("{}{}: {value}", if self.seen.len() > 1 { ", " } else { "" }, quote(&self.parameter));
                            self.emit(&member, events);
                        }
                    }
                    self.value.clear();
                    self.state = Xml::Between;
                } else if self.string || self.skip {
                    // Hold back only a possible "\n</parameter>" and stream the rest.
                    let held = (1..=self.value.len().min(PARAMETER_CLOSE.len() + 1)).rev()
                        .filter(|&n| self.value.is_char_boundary(self.value.len() - n))
                        .find(|&n| { let tail = &self.value[self.value.len() - n..];
                            PARAMETER_CLOSE.starts_with(tail) || tail.strip_prefix('\n').is_some_and(|t| PARAMETER_CLOSE.starts_with(t)) })
                        .unwrap_or(0);
                    let ready: String = self.value.drain(..self.value.len() - held).collect();
                    if !self.skip && !ready.is_empty() {
                        let quoted = quote(&ready);
                        self.emit(&quoted[1..quoted.len() - 1], events);
                    }
                }
            }
        }
        Ok(false)
    }
    /// At EOS or a length stop a held string tail is value text; the object stays open.
    fn finish(&mut self, events: &mut Vec<Event>) {
        if self.state == Xml::Value && self.string && !self.skip {
            let quoted = quote(&std::mem::take(&mut self.value));
            self.emit(&quoted[1..quoted.len() - 1], events);
        }
    }
}
/// JSON string literal. The single definition: `protocol::quote` re-exports it, and this module stays
/// free of crate imports so its tests can include it on its own.
pub fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\u{20}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod invariant_tests {
    use super::*;
    fn parser() -> Parser {
        Parser::new(Limits::default(), vec!["search".into()]).unwrap()
    }
    /// Each state below is one that the parser never builds. It must fail
    /// one request with an error: a panic would abort the whole server.
    #[test]
    fn a_broken_state_is_an_error_and_not_a_panic() {
        // XML call mode without the schema map of a hybrid model.
        let mut xml = parser();
        xml.mode = Mode::Xml(XmlCall::default());
        assert_eq!(xml.push(b"<function=search>").unwrap_err(), BROKEN);
        assert_eq!(xml.push(b"x").unwrap_err(), "chat output parser previously failed");
        // A started tool call that has no name, at the closing tag and at the end of output.
        let started = || { let mut buffer = ToolBuffer::new(); buffer.began = true; buffer };
        assert_eq!(started().call().unwrap_err(), BROKEN);
        let mut closing = parser();
        closing.mode = Mode::Tool(started());
        assert_eq!(closing.push(b"</tool_call>").unwrap_err(), BROKEN);
        let mut ending = parser();
        ending.mode = Mode::Tool(started());
        assert_eq!(ending.finish().unwrap_err(), BROKEN);
        assert_eq!(ending.finish().unwrap_err(), "chat output parser previously failed");
        // Tool text asked for outside a tool call.
        assert_eq!(parser().tool_content(&mut Vec::new()).unwrap_err(), BROKEN);
    }
    /// The same inputs as before give the same events.
    #[test]
    fn ordinary_output_is_unchanged() {
        let mut ordinary = parser();
        let mut events = ordinary.push(b"<think>plan</think>text <tool_call>{\"name\":\"search\",\"arguments\":{\"q\":1}}</tool_call>").unwrap();
        events.extend(ordinary.finish().unwrap());
        assert_eq!(events, [
            Event::Reasoning("plan".into()), Event::Content("text ".into()), Event::ToolStart { name: "search".into() },
            Event::ToolArguments("{\"q\":1}".into()), Event::ToolCall { name: "search".into(), arguments: "{\"q\":1}".into() },
        ]);
        // Malformed tool markup stays literal content, as before.
        let mut literal = parser();
        let mut events = literal.push(b"<tool_call>not json</tool_call>").unwrap();
        events.extend(literal.finish().unwrap());
        assert_eq!(events, [Event::Content("<tool_call>not json</tool_call>".into())]);
    }
}
