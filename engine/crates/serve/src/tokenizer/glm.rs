//! Text-only GLM chat rendering from the pinned checkpoint template.
//! Line references in the tests refer to templates/glm.jinja.
//! Transformers compiles templates with trim_blocks and lstrip_blocks enabled.
//! See <https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py>.
use engine_tokenizer::chat::{self, Message, Role, ToolCall};
use engine_model::json::{self, Value};
use std::ops::Range;

pub const TEMPLATE: &str = include_str!("../../templates/glm.jinja");

pub(super) fn validate_tokens(tokenizer: &engine_tokenizer::Tokenizer) -> Result<(), String> {
    for control in ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
        "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>", "</tool_response>"] {
        let ids = tokenizer.encode(control).map_err(|e| e.to_string())?;
        if ids.len() != 1 || tokenizer.token_bytes(ids[0]).map_err(|e| e.to_string())? != control.as_bytes() {
            return Err("GLM template and tokenizer controls disagree".into());
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Options {
    pub reasoning_effort: String,
    pub clear_thinking: bool,
    pub add_generation_prompt: bool,
    pub tools: Vec<String>,
    pub max_bytes: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self { reasoning_effort: "max".into(), clear_thinking: false,
            add_generation_prompt: true, tools: Vec::new(), max_bytes: 1024 * 1024 }
    }
}
struct Output { text: String, controls: Vec<Range<usize>>, limit: usize }
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
}
/// Render text without changing the caller's message or argument order.
pub fn format(messages: &[Message], options: &Options) -> Result<String, String> {
    Ok(render(messages, options)?.text)
}
/// Only template controls can become added tokens. Caller text stays ordinary.
pub fn encode(tokenizer: &engine_tokenizer::Tokenizer, messages: &[Message], options: &Options,
    limits: engine_tokenizer::Limits) -> Result<Vec<u32>, String> {
    let out = render(messages, options)?;
    if out.text.len() > limits.max_input_bytes { return Err("input byte limit exceeded".into()); }
    let (mut tokens, mut start) = (Vec::new(), 0);
    for control in &out.controls {
        tokens.extend(tokenizer.encode_ordinary(&out.text[start..control.start]).map_err(|e| e.to_string())?);
        tokens.extend(tokenizer.encode(&out.text[control.clone()]).map_err(|e| e.to_string())?);
        if tokens.len() > limits.max_tokens { return Err("token count limit exceeded".into()); }
        start = control.end;
    }
    tokens.extend(tokenizer.encode_ordinary(&out.text[start..]).map_err(|e| e.to_string())?);
    if tokens.len() > limits.max_tokens { return Err("token count limit exceeded".into()); }
    Ok(tokens)
}
fn parse(source: &str, max_bytes: usize) -> Result<Value, String> {
    json::parse(source.as_bytes(), json::Limits { bytes: max_bytes, depth: 24, nodes: 65536,
        string_bytes: max_bytes }).map_err(|e| e.to_string())
}
// The shared parser checks types and duplicate keys. Raw slices keep key order.
fn members(source: &str, max_bytes: usize) -> Result<Vec<(String, String)>, String> {
    let root = parse(source, max_bytes)?;
    let object = root.object().map_err(|e| e.to_string())?;
    let mut rest = source.trim().strip_prefix('{').ok_or("expected JSON object")?;
    let mut out = Vec::new();
    while !rest.trim_start().starts_with('}') {
        rest = rest.trim_start();
        let mut end = 1;
        while let Some(&b) = rest.as_bytes().get(end) {
            if b == b'"' { break; }
            end += if b == b'\\' { 2 } else { 1 };
        }
        let name = parse(&rest[..=end], max_bytes)?.string().map_err(|e| e.to_string())?.to_owned();
        let raw = super::raw_member(source, &name)?.ok_or("missing object value")?;
        rest = rest[end+1..].trim_start().strip_prefix(':').ok_or("missing object colon")?.trim_start();
        rest = rest.strip_prefix(raw).ok_or("invalid object value slice")?.trim_start();
        out.push((name, raw.to_owned()));
        if let Some(next) = rest.strip_prefix(',') { rest = next; } else { break; }
    }
    if out.len() != object.len() { return Err("invalid JSON object members".into()); }
    Ok(out)
}
fn tool_json(source: &str, max_bytes: usize) -> Result<Option<String>, String> {
    parse(source, max_bytes)?.object().map_err(|e| e.to_string())?;
    let function = super::raw_member(source, "function")?.unwrap_or(source);
    let root = parse(function, max_bytes)?;
    let object = root.object().map_err(|e| e.to_string())?;
    if let Some(deferred) = object.get("defer_loading") {
        if deferred.boolean().map_err(|e| e.to_string())? { return Ok(None); }
    }
    let mut fields = Vec::new();
    for (key, value) in members(function, max_bytes)? {
        if key != "defer_loading" && key != "strict" {
            // The pinned template writes top-level tool keys without JSON escaping.
            fields.push(format!("\"{key}\": {}", chat::json_for_template(&value).map_err(|e| e.to_string())?));
        }
    }
    Ok(Some(format!("{{{}}}", fields.join(", "))))
}
fn call(out: &mut Output, call: &ToolCall, max_bytes: usize) -> Result<(), String> {
    out.control("<tool_call>")?; out.text(&call.name)?;
    for (key, value) in members(&call.arguments, max_bytes)? {
        let parsed = parse(&value, max_bytes)?;
        out.control("<arg_key>")?; out.text(&key)?; out.control("</arg_key>")?;
        out.control("<arg_value>")?;
        out.text(&match parsed { Value::String(s) => s,
            _ => chat::json_for_template(&value).map_err(|e| e.to_string())? })?;
        out.control("</arg_value>")?;
    }
    out.control("</tool_call>")
}
fn responses(out: &mut Output, block: &[Message], calls: &[ToolCall]) -> Result<(), String> {
    out.control("<|observation|>")?;
    let unique = |ids: Vec<Option<&str>>| ids.iter().enumerate().all(|(i, id)|
        id.is_some_and(|s| !s.is_empty()) && !ids[..i].contains(id));
    let can_sort = !calls.is_empty() && unique(calls.iter().map(|c| c.id.as_deref()).collect())
        && unique(block.iter().map(|m| m.tool_call_id.as_deref()).collect())
        && block.iter().all(|m| calls.iter().any(|c| c.id == m.tool_call_id));
    let ordered: Vec<&Message> = if can_sort {
        calls.iter().flat_map(|c| block.iter().filter(move |m| m.tool_call_id == c.id)).collect()
    } else { block.iter().collect() };
    for message in ordered {
        out.control("<tool_response>")?; out.text(&message.content)?; out.control("</tool_response>")?;
    }
    Ok(())
}
fn render(messages: &[Message], options: &Options) -> Result<Output, String> {
    let mut out = Output { text: String::new(), controls: Vec::new(), limit: options.max_bytes };
    out.control("[gMASK]")?; out.control("<sop>")?; out.control("<|system|>")?;
    out.text("Reasoning Effort: ")?;
    out.text(match options.reasoning_effort.as_str() { "low" => "Low", "high" => "High", _ => "Max" })?;
    if !options.tools.is_empty() {
        out.control("<|system|>")?;
        out.text("\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")?;
        for tool in &options.tools {
            if let Some(tool) = tool_json(tool, options.max_bytes)? { out.text(&tool)?; out.text("\n")?; }
        }
        out.text("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n")?;
        out.control("<tool_call>")?; out.text("{function-name}")?;
        for i in 1..=2 {
            out.control("<arg_key>")?; out.text(&format!("{{arg-key-{i}}}"))?; out.control("</arg_key>")?;
            out.control("<arg_value>")?; out.text(&format!("{{arg-value-{i}}}"))?; out.control("</arg_value>")?;
        }
        out.text("...")?; out.control("</tool_call>")?;
    }
    let last_user = messages.iter().rposition(|m| m.role == Role::User);
    let mut i = 0;
    while i < messages.len() {
        let message = &messages[i];
        match message.role {
            Role::System => { out.control("<|system|>")?; out.text(&message.content)?; }
            Role::Developer => {}
            Role::User => { out.control("<|user|>")?; out.text(&message.content)?; }
            Role::Assistant => {
                out.control("<|assistant|>")?;
                let (thought, content) = if let Some(thought) = message.reasoning_content.as_deref() {
                    (Some(thought), message.content.as_str())
                } else if let Some((thought, _)) = message.content.split_once("</think>") {
                    (Some(thought.rsplit("<think>").next().unwrap_or(thought)), message.content.rsplit("</think>").next().unwrap_or(""))
                } else { (None, message.content.as_str()) };
                out.control("<think>")?;
                if !options.clear_thinking || last_user.is_none_or(|last| i > last) {
                    if let Some(thought) = thought { out.text(thought)?; }
                }
                out.control("</think>")?; out.text(trim(content))?;
                for tc in &message.tool_calls { call(&mut out, tc, options.max_bytes)?; }
            }
            Role::Tool => {
                let end = i + messages[i..].iter().take_while(|m| m.role == Role::Tool).count();
                let calls = if i > 0 && messages[i-1].role == Role::Assistant { messages[i-1].tool_calls.as_slice() } else { &[] };
                responses(&mut out, &messages[i..end], calls)?; i = end; continue;
            }
        }
        i += 1;
    }
    if options.add_generation_prompt { out.control("<|assistant|>")?; out.control("<think>")?; }
    Ok(out)
}
// Python str.strip also treats the ASCII information separators as whitespace.
fn trim(s: &str) -> &str { s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn system_and_user_golden() {
        // Hand-derived from glm.jinja: prefix/effort lines 1-3; raw user text
        // lines 54-60, 139; system lines 255-256; generation prompt 259-260.
        let messages = [Message::new(Role::System, " Be brief.\n"), Message::new(Role::User, " Hello. ")];
        let expected = "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|> Be brief.\n<|user|> Hello. <|assistant|><think>";
        assert_eq!(format(&messages, &Options::default()).unwrap().as_bytes(), expected.as_bytes());
    }
    #[test]
    fn multi_turn_think_history_golden() {
        // Hand-derived from glm.jinja: last user lines 132-137; assistant
        // role 140-141; explicit thought 143-144; inline split 145-147;
        // default thought retention 4, 149-152; content strip 154-155.
        let mut explicit = Message::new(Role::Assistant, " answer ");
        explicit.reasoning_content = Some(" old\nthought ".into());
        let messages = [Message::new(Role::User, "first"), explicit,
            Message::new(Role::User, "next"), Message::new(Role::Assistant, "<think>\nnew\n</think>\n done ")];
        let expected = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>first<|assistant|><think> old\nthought </think>answer<|user|>next<|assistant|><think>\nnew\n</think>done<|assistant|><think>";
        assert_eq!(format(&messages, &Options::default()).unwrap().as_bytes(), expected.as_bytes());
        // Lines 149-152 clear thoughts before the last user only when requested.
        let expected = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>first<|assistant|><think></think>answer<|user|>next<|assistant|><think>\nnew\n</think>done<|assistant|><think>";
        assert_eq!(format(&messages, &Options { clear_thinking: true, ..Default::default() }).unwrap(), expected);
    }
    #[test]
    fn reasoning_effort_goldens() {
        // Hand-derived from glm.jinja lines 2-3: only low and high survive;
        // all other values select max. Line 260 always opens a thought.
        let messages = [Message::new(Role::User, "go")];
        for (effort, expected) in [
            ("low", "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>go<|assistant|><think>"),
            ("high", "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>go<|assistant|><think>"),
            ("max", "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>go<|assistant|><think>"),
            ("medium", "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>go<|assistant|><think>"),
            ("unknown", "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>go<|assistant|><think>"),
        ] { assert_eq!(format(&messages, &Options { reasoning_effort: effort.into(), ..Default::default() }).unwrap(), expected); }
    }
    fn tool_call() -> Message {
        let mut message = Message::new(Role::Assistant, " checking ");
        message.tool_calls.push(ToolCall { id: Some("c".into()), name: "lookup".into(),
            arguments: r#"{"z":"雪 <&>","a":true,"list":[1,null],"map":{"z":2,"a":1}}"#.into() });
        message
    }
    #[test]
    fn tool_call_golden() {
        // Hand-derived from glm.jinja: empty thought 152; content strip 154-155;
        // trimmed block line breaks 157-158, 163-164; name, ordered keys and raw string or
        // JSON values 159-163. No end-of-turn token follows a message.
        let messages = [Message::new(Role::User, "find"), tool_call()];
        let expected = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>find<|assistant|><think></think>checking<tool_call>lookup<arg_key>z</arg_key><arg_value>雪 <&></arg_value><arg_key>a</arg_key><arg_value>true</arg_value><arg_key>list</arg_key><arg_value>[1, null]</arg_value><arg_key>map</arg_key><arg_value>{\"z\": 2, \"a\": 1}</arg_value></tool_call>";
        assert_eq!(format(&messages, &Options { add_generation_prompt: false, ..Default::default() }).unwrap().as_bytes(), expected.as_bytes());
    }
    #[test]
    fn tool_result_golden() {
        // Hand-derived from glm.jinja: observation role 165-168; string result
        // wrappers 75-80; ID match 225-245; generation prompt 259-260.
        let mut result = Message::new(Role::Tool, " found\n"); result.tool_call_id = Some("c".into());
        let messages = [Message::new(Role::User, "find"), tool_call(), result];
        let expected = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>find<|assistant|><think></think>checking<tool_call>lookup<arg_key>z</arg_key><arg_value>雪 <&></arg_value><arg_key>a</arg_key><arg_value>true</arg_value><arg_key>list</arg_key><arg_value>[1, null]</arg_value><arg_key>map</arg_key><arg_value>{\"z\": 2, \"a\": 1}</arg_value></tool_call><|observation|><tool_response> found\n</tool_response><|assistant|><think>";
        assert_eq!(format(&messages, &Options::default()).unwrap().as_bytes(), expected.as_bytes());
    }
    #[test]
    fn tool_definition_golden() {
        // Hand-derived from glm.jinja: tools system text 32-50; unwrap function
        // 40-42; skip deferred definitions 43; omit strict/defer_loading 9-14;
        // retained line breaks 38-47. User and prompt follow lines 139, 260.
        let options = Options { tools: vec![r#"{"type":"function","function":{"name":"lookup","strict":true,"defer_loading":false,"parameters":{"type":"object"}}}"#.into()], ..Default::default() };
        let expected = "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"name\": \"lookup\", \"parameters\": {\"type\": \"object\"}}\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call><|user|>go<|assistant|><think>";
        assert_eq!(format(&[Message::new(Role::User, "go")], &options).unwrap().as_bytes(), expected.as_bytes());
    }
    #[test]
    fn escaped_tool_fields_follow_template_bytes() {
        // glm.jinja line 13 writes a raw top-level key. The tojson filter
        // escapes nested keys and values. Call keys and string values stay raw.
        let definition = r#"{"name":"f","a\"\\\n":{"雪":"\"\\\n"}}"#;
        assert_eq!(tool_json(definition, 8192).unwrap().unwrap(), "{\"name\": \"f\", \"a\"\\\n\": {\"雪\": \"\\\"\\\\\\n\"}}");
        let mut out = Output { text: String::new(), controls: vec![], limit: 8192 };
        call(&mut out, &ToolCall { name: "f\"\\".into(), id: None,
            arguments: r#"{"a\"\\\n":"雪\"\\\n","nested":{"q\"":"<&>"}}"#.into() }, 8192).unwrap();
        assert_eq!(out.text, "<tool_call>f\"\\<arg_key>a\"\\\n</arg_key><arg_value>雪\"\\\n</arg_value><arg_key>nested</arg_key><arg_value>{\"q\\\"\": \"<&>\"}</arg_value></tool_call>");
    }
    #[test]
    fn result_id_order_and_fallback_goldens() {
        // Hand-derived from glm.jinja: result block 166-180; unique matching
        // IDs 181-224; call order 225-248; input-order fallback 249-252.
        let mut assistant = Message::new(Role::Assistant, "");
        assistant.tool_calls = ["a", "b"].map(|id| ToolCall { id: Some(id.into()), name: id.into(), arguments: "{}".into() }).to_vec();
        let mut b = Message::new(Role::Tool, "B"); b.tool_call_id = Some("b".into());
        let mut a = Message::new(Role::Tool, "A"); a.tool_call_id = Some("a".into());
        let mut messages = [assistant, b, a];
        let expected = "[gMASK]<sop><|system|>Reasoning Effort: Max<|assistant|><think></think><tool_call>a</tool_call><tool_call>b</tool_call><|observation|><tool_response>A</tool_response><tool_response>B</tool_response><|assistant|><think>";
        assert_eq!(format(&messages, &Options::default()).unwrap(), expected);
        messages[1].tool_call_id = None;
        let expected = "[gMASK]<sop><|system|>Reasoning Effort: Max<|assistant|><think></think><tool_call>a</tool_call><tool_call>b</tool_call><|observation|><tool_response>B</tool_response><tool_response>A</tool_response><|assistant|><think>";
        assert_eq!(format(&messages, &Options::default()).unwrap(), expected);
    }
    #[test]
    fn limits_and_invalid_arguments_fail() {
        let messages = [Message::new(Role::User, "雪")];
        let full = format(&messages, &Options::default()).unwrap();
        assert!(format(&messages, &Options { max_bytes: full.len() - 1, ..Default::default() }).is_err());
        assert_eq!(format(&messages, &Options { max_bytes: full.len(), ..Default::default() }).unwrap(), full);
        for args in ["[]", "null", "{", r#"{"x":1,"x":2}"#] {
            let mut call = tool_call(); call.tool_calls[0].arguments = args.into();
            assert!(format(&[call], &Options::default()).is_err());
        }
    }
}
