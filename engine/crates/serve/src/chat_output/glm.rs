//! Output grammar from the pinned template in templates/glm.jinja.
//! Raw string values have no escaping in that grammar. Schema types decide
//! which values need JSON parsing. Incomplete or malformed calls stay text.
use super::{Event, Limits, quote};
use engine_model::json;
use std::collections::{BTreeMap, BTreeSet};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const THINK: &str = "</think>";

pub(super) struct Parser {
    reasoning: bool, marker: String, call: Option<String>, calls: usize,
    json_parameters: BTreeMap<String, BTreeSet<String>>,
}
impl Parser {
    pub(super) fn new(reasoning: bool, json_parameters: BTreeMap<String, BTreeSet<String>>) -> Self {
        Self { reasoning, marker: String::new(), call: None, calls: 0, json_parameters }
    }
    pub(super) fn push(&mut self, ch: char, limits: &Limits, declared: &mut BTreeSet<String>, events: &mut Vec<Event>) -> Result<(), String> {
        if let Some(raw) = &mut self.call {
            raw.push(ch);
            if raw.len() > limits.tool_call_bytes { return Err("GLM tool call byte limit exceeded".into()); }
            if raw.ends_with(CLOSE) {
                let raw = self.call.take().unwrap();
                let parsed = parse_call(&raw[..raw.len()-CLOSE.len()], limits, declared, &self.json_parameters);
                if let Ok((name, arguments)) = parsed {
                    if self.calls >= limits.tool_calls { return Err("GLM tool call count limit exceeded".into()); }
                    self.calls += 1;
                    events.push(Event::ToolStart { name: name.clone() });
                    events.push(Event::ToolArguments(arguments.clone()));
                    events.push(Event::ToolCall { name, arguments });
                } else {
                    self.text(OPEN, events); self.text(&raw, events); declared.clear();
                }
            }
            return Ok(());
        }
        self.marker.push(ch);
        loop {
            let candidate = if self.reasoning { Some(THINK) } else if !declared.is_empty() { Some(OPEN) } else { None };
            if candidate == Some(self.marker.as_str()) {
                self.marker.clear();
                if self.reasoning { self.reasoning = false; } else { self.call = Some(String::new()); }
                return Ok(());
            }
            if candidate.is_some_and(|s| s.starts_with(&self.marker)) { return Ok(()); }
            let first = self.marker.chars().next().unwrap(); self.marker.drain(..first.len_utf8());
            self.text(first.encode_utf8(&mut [0;4]), events);
            if self.marker.is_empty() { return Ok(()); }
        }
    }
    fn text(&self, text: &str, events: &mut Vec<Event>) {
        if text.is_empty() { return; }
        if self.reasoning {
            if let Some(Event::Reasoning(s)) = events.last_mut() { s.push_str(text); }
            else { events.push(Event::Reasoning(text.into())); }
        } else if let Some(Event::Content(s)) = events.last_mut() { s.push_str(text); }
        else { events.push(Event::Content(text.into())); }
    }
    pub(super) fn finish(&mut self, events: &mut Vec<Event>) {
        if let Some(raw) = self.call.take() { self.text(OPEN, events); self.text(&raw, events); }
        let marker = std::mem::take(&mut self.marker); self.text(&marker, events);
    }
}
fn parse_call(raw: &str, limits: &Limits, declared: &BTreeSet<String>,
    types: &BTreeMap<String, BTreeSet<String>>) -> Result<(String, String), ()> {
    let name_end = raw.find('<').unwrap_or(raw.len());
    let name = &raw[..name_end];
    if name.len() > limits.name_bytes || !declared.contains(name) { return Err(()); }
    let mut rest = &raw[name_end..]; let mut keys = BTreeSet::new(); let mut fields = Vec::new();
    while !rest.is_empty() {
        rest = rest.strip_prefix("<arg_key>").ok_or(())?;
        let end = rest.find("</arg_key>").ok_or(())?;
        let key = &rest[..end];
        if key.contains('<') || !keys.insert(key) { return Err(()); }
        rest = rest[end+"</arg_key>".len()..].strip_prefix("<arg_value>").ok_or(())?;
        let end = rest.find("</arg_value>").ok_or(())?;
        let value = &rest[..end];
        let value = if types.get(name).is_some_and(|fields| fields.contains(key)) {
            json::parse(value.as_bytes(), json::Limits { bytes: limits.tool_call_bytes,
                string_bytes: limits.tool_call_bytes, ..Default::default() }).map_err(|_| ())?;
            value.to_owned()
        } else { quote(value) };
        fields.push(format!("{}:{value}", quote(key)));
        rest = &rest[end+"</arg_value>".len()..];
    }
    let arguments = format!("{{{}}}", fields.join(","));
    if arguments.len() > limits.tool_call_bytes {return Err(());}
    Ok((name.into(), arguments))
}
