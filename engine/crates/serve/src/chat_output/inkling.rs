//! Reply grammar of the message markup of templates/inkling2-mini.jinja, as the
//! checkpoint's `response_template` gives it (templates/inkling2-mini-response.json).
//! A reply is a list of messages and then the end-of-turn token:
//! `<|message_model|>` NAME `<|content_KIND|>` BODY `<|end_message|>`.
//! `Channels` is the state machine. It reads token IDs at the decoder, so no text
//! can pass for a control token. It gives each body to the parser as text behind a
//! mark. `Parser` reads the marks and makes the events: answer text, reasoning,
//! tool calls. A tool call body is held until its message ends. A body that is not
//! a call of a declared tool is shown as answer text.
use super::{Event, Limits};
use engine_model::json;
use std::collections::BTreeSet;

/// A mark is this noncharacter and one more. Body text that holds this
/// noncharacter is written with it twice.
const MARK: char = '\u{fdd0}';
const OPEN_TEXT: char = '\u{fdd1}';
const OPEN_REASONING: char = '\u{fdd2}';
const OPEN_CALL: char = '\u{fdd3}';
const CLOSE: char = '\u{fdd4}';

/// The control token IDs of the reply, bound from the tokenizer by their spelling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelIds {
    /// `<|message_model|>`: the header of a message. A name can follow it.
    pub message: u32,
    /// The content kinds whose body is answer text.
    pub text: Vec<u32>,
    /// `<|content_thinking|>`.
    pub reasoning: u32,
    /// `<|content_invoke_tool_json|>`.
    pub call: u32,
    /// `<|end_message|>` ends one message. It is not a stop.
    pub end_message: u32,
    /// `<|content_model_end_sampling|>` ends the turn.
    pub stop: u32,
}
impl ChannelIds {
    pub fn from_tokenizer(tokenizer: &engine_tokenizer::Tokenizer) -> Result<Self, String> {
        let id = |control: &str| {
            let ids = tokenizer.encode(control).map_err(|e| e.to_string())?;
            if ids.len() != 1 || !tokenizer.is_special(ids[0]) || tokenizer.token_bytes(ids[0]).map_err(|e| e.to_string())? != control.as_bytes() {
                return Err(format!("reply grammar and tokenizer disagree on the control token {control}"));
            }
            Ok(ids[0])
        };
        Ok(Self { message: id("<|message_model|>")?,
            text: vec![id("<|content_text|>")?, id("<|content_tool_error|>")?, id("<|content_invoke_tool_text|>")?],
            reasoning: id("<|content_thinking|>")?, call: id("<|content_invoke_tool_json|>")?,
            end_message: id("<|end_message|>")?, stop: id("<|content_model_end_sampling|>")? })
    }
    fn open(&self, token: u32) -> Option<char> {
        if self.text.contains(&token) { Some(OPEN_TEXT) }
        else if token == self.reasoning { Some(OPEN_REASONING) }
        else if token == self.call { Some(OPEN_CALL) }
        else { None }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Between messages. Tokens here are not part of any field and are not shown.
    Outside,
    /// After the message token. The tokens here are the name of the message.
    Header,
    /// After a content kind token, up to the end of the message.
    Body,
}
/// The state machine on token IDs. One per reply.
pub struct Channels { ids: ChannelIds, state: State }
impl Channels {
    pub fn new(ids: ChannelIds) -> Self { Self { ids, state: State::Outside } }
    /// Read one token. `decode(Some(token))` gives the text that an ordinary token
    /// completes, and no text for a special token. `decode(None)` gives the bytes that
    /// the decoder still holds. Returns marked text for the parser.
    pub fn push(&mut self, token: u32, mut decode: impl FnMut(Option<u32>) -> Result<String, String>) -> Result<String, String> {
        if token == self.ids.end_message || token == self.ids.stop {
            // A body ends at the end of its message. The end of the turn ends it too.
            let mut out = String::new();
            if self.state == State::Body { out = escape(&decode(None)?); out.push(MARK); out.push(CLOSE); }
            self.state = State::Outside;
            return Ok(out);
        }
        match self.state {
            // Only the end of the message closes a body. The decoder shows no special
            // token, so another control token in a body adds no text.
            State::Body => Ok(escape(&decode(Some(token))?)),
            State::Outside | State::Header => {
                if let Some(open) = self.ids.open(token) { self.state = State::Body; return Ok(String::from_iter([MARK, open])); }
                if token == self.ids.message { self.state = State::Header; }
                Ok(String::new())
            }
        }
    }
    /// The reply ended with no end token: a length limit or a stop string cut it.
    pub fn finish(&mut self, tail: String) -> String {
        if self.state == State::Body { escape(&tail) } else { String::new() }
    }
}
fn escape(text: &str) -> String {
    if text.contains(MARK) { text.replace(MARK, "\u{fdd0}\u{fdd0}") } else { text.to_owned() }
}
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Channel { #[default] Closed, Content, Reasoning, Call }
#[derive(Default)]
pub(super) struct Parser { mark: bool, channel: Channel, call: String, calls: usize }
impl Parser {
    pub(super) fn push(&mut self, ch: char, limits: &Limits, declared: &BTreeSet<String>, events: &mut Vec<Event>) -> Result<(), String> {
        if std::mem::take(&mut self.mark) {
            let open = match ch {
                MARK => return self.text(ch, limits, events),
                OPEN_TEXT => Channel::Content,
                OPEN_REASONING => Channel::Reasoning,
                // With no declared tool there is no call to make: the body is answer text, as it comes.
                OPEN_CALL => if declared.is_empty() { Channel::Content } else { Channel::Call },
                CLOSE => Channel::Closed,
                _ => return Err(super::BROKEN.into()),
            };
            // The end of its message completes a call. A body with no end is not a call.
            if self.channel == Channel::Call { self.settle(ch == CLOSE, limits, declared, events)?; }
            self.channel = open;
        } else if ch == MARK { self.mark = true; }
        else { self.text(ch, limits, events)?; }
        Ok(())
    }
    fn text(&mut self, ch: char, limits: &Limits, events: &mut Vec<Event>) -> Result<(), String> {
        match (self.channel, events.last_mut()) {
            (Channel::Content, Some(Event::Content(text))) | (Channel::Reasoning, Some(Event::Reasoning(text))) => text.push(ch),
            (Channel::Content, _) => events.push(Event::Content(ch.into())),
            (Channel::Reasoning, _) => events.push(Event::Reasoning(ch.into())),
            (Channel::Call, _) => {
                self.call.push(ch);
                if self.call.len() > limits.tool_call_bytes { return Err("tool call byte limit exceeded".into()); }
            }
            (Channel::Closed, _) => {}
        }
        Ok(())
    }
    /// The held body of a call message becomes one call, or answer text.
    fn settle(&mut self, complete: bool, limits: &Limits, declared: &BTreeSet<String>, events: &mut Vec<Event>) -> Result<(), String> {
        let body = std::mem::take(&mut self.call);
        match if complete { parse_call(&body, limits, declared) } else { None } {
            Some((name, arguments)) => {
                if self.calls >= limits.tool_calls { return Err("tool call count limit exceeded".into()); }
                self.calls += 1;
                events.push(Event::ToolStart { name: name.clone() });
                events.push(Event::ToolArguments(arguments.clone()));
                events.push(Event::ToolCall { name, arguments });
            }
            None if body.is_empty() => {}
            None => match events.last_mut() {
                Some(Event::Content(text)) => text.push_str(&body),
                _ => events.push(Event::Content(body)),
            },
        }
        Ok(())
    }
    /// The reply ended. A call body that a length limit or a stop string cut is answer text.
    pub(super) fn finish(&mut self, limits: &Limits, declared: &BTreeSet<String>, events: &mut Vec<Event>) -> Result<(), String> {
        if std::mem::replace(&mut self.channel, Channel::Closed) == Channel::Call { self.settle(false, limits, declared, events)?; }
        Ok(())
    }
}
/// A call body is one JSON object with the name of the tool and its arguments:
/// `{"name":NAME,"args":{...}}` (lines 113 to 121 of the template; the `tool_calls`
/// field of the reply grammar takes the name and the arguments from this object, and
/// does not read the name in the message header). Returns the name and the text of
/// the arguments as the model wrote it. None: the body is not a call of a declared tool.
fn parse_call(body: &str, limits: &Limits, declared: &BTreeSet<String>) -> Option<(String, String)> {
    let bounds = json::Limits { bytes: limits.tool_call_bytes, string_bytes: limits.tool_call_bytes, ..Default::default() };
    let root = json::parse(body.as_bytes(), bounds).ok()?;
    let fields = root.object().ok()?;
    let name = fields.get("name")?.string().ok()?;
    if !declared.contains(name) { return None; }
    fields.get("args")?.object().ok()?;
    Some((name.to_owned(), member(body, "args", bounds)?.to_owned()))
}
/// The text of one member of a JSON object. The caller has parsed `source`, so it is one valid object.
fn member<'a>(source: &'a str, name: &str, bounds: json::Limits) -> Option<&'a str> {
    let bytes = source.as_bytes();
    let mut at = bytes.iter().position(|byte| *byte == b'{')? + 1;
    loop {
        let key_start = at + bytes.get(at..)?.iter().position(|byte| *byte == b'"')?;
        let key_end = string_end(bytes, key_start)?;
        at = key_end + bytes.get(key_end..)?.iter().position(|byte| *byte == b':')? + 1;
        while bytes.get(at).is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n')) { at += 1; }
        let (start, mut depth) = (at, 0usize);
        loop {
            match *bytes.get(at)? {
                b'"' => { at = string_end(bytes, at)?; continue; }
                b'{' | b'[' => depth += 1,
                b'}' | b']' | b',' if depth == 0 => break,
                b'}' | b']' => depth -= 1,
                _ => {}
            }
            at += 1;
        }
        if json::parse(bytes.get(key_start..key_end)?, bounds).ok()?.string().ok()? == name {
            return Some(source.get(start..at)?.trim_end_matches([' ', '\t', '\r', '\n']));
        }
        if bytes.get(at) != Some(&b',') { return None; }
        at += 1;
    }
}
/// The index after the closing quote of the JSON string that starts at `start`.
fn string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut at = start + 1;
    loop {
        match *bytes.get(at)? { b'\\' => at += 2, b'"' => return Some(at + 1), _ => at += 1 }
    }
}
