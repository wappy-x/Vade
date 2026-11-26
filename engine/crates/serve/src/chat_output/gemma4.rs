//! Google's pinned response_template: thought channels and repeated function
//! calls with raw <|"|>-delimited strings and unquoted object keys.
use super::{Event, Limits, quote};
use std::collections::BTreeSet;

const THOUGHT: &str = "<|channel>thought\n";
const END_THOUGHT: &str = "<channel|>";
const CALL: &str = "<|tool_call>";
const END_CALL: &str = "<tool_call|>";
const STRING: &str = "<|\"|>";

pub(super) struct Parser {
    reasoning: bool,
    initial: bool,
    marker: String,
    call: Option<Call>,
    calls: usize,
}
impl Parser {
    pub(super) fn new(reasoning: bool) -> Self {
        Self { reasoning, initial: true, marker: String::new(), call: None, calls: 0 }
    }
    pub(super) fn push(&mut self, ch: char, limits: &Limits, declared: &mut BTreeSet<String>, events: &mut Vec<Event>) -> Result<(), String> {
        if let Some(call) = &mut self.call {
            let done = call.push(ch, limits, declared, self.calls, events);
            match done {
                Ok(false) => {},
                Ok(true) | Err(()) => {
                    let Some(mut call) = self.call.take() else { return Err(super::BROKEN.into()) };
                    if let Some(name) = call.name.take() {
                        call.finish(name, events);
                        self.calls += 1;
                    } else {
                        self.text(CALL, events);
                        self.text(&call.raw, events);
                    }
                    if done.is_err() { declared.clear(); }
                    self.initial = true;
                }
            }
            return Ok(());
        }
        self.marker.push(ch);
        // The loop ends when the marker holds no character: no state here can panic.
        while let Some(first) = self.marker.chars().next() {
            let candidates = if self.reasoning {
                [Some(END_THOUGHT), None]
            } else {
                [self.initial.then_some(THOUGHT), (!declared.is_empty()).then_some(CALL)]
            };
            if let Some(marker) = candidates.into_iter().flatten().find(|m| *m == self.marker) {
                self.marker.clear();
                match marker {
                    THOUGHT => self.reasoning = true,
                    END_THOUGHT => { self.reasoning = false; self.initial = false; },
                    CALL => self.call = Some(Call::default()),
                    _ => return Err(super::BROKEN.into()),
                }
                return Ok(());
            }
            if candidates.into_iter().flatten().any(|m| m.starts_with(&self.marker)) { return Ok(()); }
            self.marker.drain(..first.len_utf8());
            self.text(first.encode_utf8(&mut [0; 4]), events);
        }
        Ok(())
    }
    fn text(&mut self, text: &str, events: &mut Vec<Event>) {
        if text.is_empty() { return; }
        if self.reasoning {
            if let Some(Event::Reasoning(s)) = events.last_mut() { s.push_str(text); }
            else { events.push(Event::Reasoning(text.into())); }
        } else {
            if !text.trim().is_empty() { self.initial = false; }
            if let Some(Event::Content(s)) = events.last_mut() { s.push_str(text); }
            else { events.push(Event::Content(text.into())); }
        }
    }
    pub(super) fn finish(&mut self, events: &mut Vec<Event>) {
        if let Some(mut call) = self.call.take() {
            if let Some(name) = call.name.take() { call.finish(name, events); }
            else { self.text(CALL, events); self.text(&call.raw, events); }
        }
        let marker = std::mem::take(&mut self.marker);
        self.text(&marker, events);
    }
}

#[derive(Default)]
struct Call {
    raw: String,
    bytes: usize,
    name: Option<String>,
    arguments: String,
    marker: String,
    quoted: bool,
    key: bool,
    key_text: String,
    stack: Vec<char>,
}
impl Call {
    fn emit(&mut self, text: &str, events: &mut Vec<Event>) {
        self.arguments.push_str(text);
        if let Some(Event::ToolArguments(s)) = events.last_mut() { s.push_str(text); }
        else { events.push(Event::ToolArguments(text.into())); }
    }
    fn push(&mut self, ch: char, limits: &Limits, declared: &BTreeSet<String>, calls: usize, events: &mut Vec<Event>) -> Result<bool, ()> {
        self.bytes += ch.len_utf8();
        if self.name.is_none() { self.raw.push(ch); }
        if self.bytes > limits.tool_call_bytes { return Err(()); }
        if self.name.is_none() {
            if ch != '{' {
                if self.raw.len() > limits.name_bytes + "call:".len() { return Err(()); }
                return Ok(false);
            }
            let name = self.raw.strip_prefix("call:").and_then(|s| s.strip_suffix('{')).ok_or(())?;
            if !declared.contains(name) || calls >= limits.tool_calls { return Err(()); }
            self.name = Some(name.into());
            events.push(Event::ToolStart { name: name.into() });
            self.raw.clear();
            self.stack.push('}');
            self.key = true;
            self.emit("{", events);
            return Ok(false);
        }
        self.marker.push(ch);
        while let Some(first) = self.marker.chars().next() {
            if self.marker == STRING {
                self.marker.clear();
                if !self.quoted && self.key {
                    if !self.key_text.trim().is_empty() { return Err(()); }
                    self.key_text.clear();
                    self.key = false;
                }
                self.quoted = !self.quoted;
                self.emit("\"", events);
                return Ok(false);
            }
            if !self.quoted && self.marker == END_CALL {
                self.marker.clear();
                return Ok(true);
            }
            if STRING.starts_with(&self.marker) || (!self.quoted && END_CALL.starts_with(&self.marker)) {
                return Ok(false);
            }
            self.marker.drain(..first.len_utf8());
            self.character(first, events)?;
        }
        Ok(false)
    }
    fn character(&mut self, ch: char, events: &mut Vec<Event>) -> Result<(), ()> {
        if self.quoted {
            // Gemma strings are raw; backslashes, quotes and newlines must be
            // escaped for the JSON argument string exposed on the wire.
            let escaped = quote(ch.encode_utf8(&mut [0; 4]));
            self.emit(&escaped[1..escaped.len()-1], events);
            return Ok(());
        }
        if self.key {
            if ch == ':' {
                let key = std::mem::take(&mut self.key_text);
                if key.trim().is_empty() { return Err(()); }
                self.emit(&quote(key.trim()), events);
                self.emit(":", events);
                self.key = false;
                return Ok(());
            }
            if ch != '}' || !self.key_text.trim().is_empty() {
                self.key_text.push(ch);
                return Ok(());
            }
            self.key_text.clear();
            self.key = false;
        }
        match ch {
            '{' | '[' => {
                if self.stack.len() >= 32 { return Err(()); }
                self.stack.push(if ch == '{' { '}' } else { ']' });
                self.key = ch == '{';
            },
            '}' | ']' => { if self.stack.pop() != Some(ch) { return Err(()); } },
            ',' => self.key = self.stack.last() == Some(&'}'),
            _ => {},
        }
        self.emit(ch.encode_utf8(&mut [0; 4]), events);
        Ok(())
    }
    /// `name` is the name of this call: the caller ends only a call that has one.
    fn finish(&mut self, name: String, events: &mut Vec<Event>) {
        // A token/byte limit must not invent closing braces or a complete call.
        // Flush the same partial JSON to streaming and collected responses.
        let marker = std::mem::take(&mut self.marker);
        for ch in marker.chars() { let _ = self.character(ch, events); }
        if !self.key_text.is_empty() {
            let key = std::mem::take(&mut self.key_text);
            let quoted = quote(key.trim_start());
            self.emit(&quoted[..quoted.len()-1], events);
        }
        events.push(Event::ToolCall { name, arguments: std::mem::take(&mut self.arguments) });
    }
}
