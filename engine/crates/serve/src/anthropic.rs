//! Stateless Anthropic Messages translation into the shared Chat path.
use crate::{Limits, chat_output::Event, generation::Usage, protocol::{self, present, quote, raw_array, raw_member}};
use engine_model::json::{self, Value};
use std::collections::BTreeMap;

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
    for block in value.array().map_err(|e| e.to_string())? {
        let block = block.object().map_err(|e| e.to_string())?;
        if string(block, "type")? == "image" { return Err(protocol::IMAGE_INPUT_UNSUPPORTED.into()); }
        if string(block, "type")? != "text" {
            return Err("only text blocks are supported; images and documents are unsupported".into());
        }
        result.push_str(string(block, "text")?);
    }
    Ok(result)
}
fn message(role: &str, content: &str) -> String {
    format!("{{\"role\":{},\"content\":{}}}", quote(role), quote(content))
}
fn flush_text(content: &mut String, messages: &mut Vec<String>) {
    if !content.is_empty() { messages.push(message("user", &std::mem::take(content))); }
}
fn translate_message(raw: &str, limits: &Limits, messages: &mut Vec<String>) -> Result<(), String> {
    let value = parse_json(raw.as_bytes(), limits)?;
    let object = value.object().map_err(|e| e.to_string())?;
    let role = string(object, "role")?;
    if !matches!(role, "user" | "assistant") { return Err(format!("Unexpected role {}. Messages roles must be user or assistant", quote(role))); }
    let value = field(object, "content")?;
    if let Value::String(content) = value { messages.push(message(role, content)); return Ok(()); }
    value.array().map_err(|e| e.to_string())?;
    let blocks = raw_array(raw_member(raw, "content")?.ok_or("missing content")?)?;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    let before = messages.len();
    for raw in blocks {
        let value = parse_json(raw.as_bytes(), limits)?;
        let block = value.object().map_err(|e| e.to_string())?;
        match string(block, "type")? {
            "text" => {
                let text = string(block, "text")?;
                if role == "assistant" && !calls.is_empty() {
                    if text.trim().is_empty() { continue; }
                    return Err("assistant text after tool_use is unsupported by the Chat template".into());
                }
                content.push_str(text);
            }
            "thinking" if role == "assistant" => {
                if !content.is_empty() || !calls.is_empty() { return Err("assistant thinking must precede text and tool_use".into()); }
                reasoning.push_str(string(block, "thinking")?);
            }
            "tool_use" if role == "assistant" => {
                let id = string(block, "id")?;
                if id.is_empty() { return Err("empty tool use ID".into()); }
                field(block, "input")?.object().map_err(|_| "tool use input must be a JSON object")?;
                // Argument whitespace and key order are part of the canonical Chat prompt.
                let arguments = raw_member(&raw, "input")?.ok_or("missing tool use input")?;
                calls.push(format!("{{\"id\":{},\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":{}}}}}",
                    quote(id), quote(string(block, "name")?), quote(arguments)));
            }
            "tool_result" if role == "user" => {
                let id = string(block, "tool_use_id")?;
                if id.is_empty() { return Err("empty tool result ID".into()); }
                let error = present(block, "is_error").map(Value::boolean).transpose()
                    .map_err(|e| e.to_string())?.unwrap_or(false);
                let result_text = present(block, "content").map(text).transpose()?.unwrap_or_default();
                // Chat has no error flag: retain its meaning in the tool result seen by the model.
                let result_text = if error { format!("[tool error]\n{result_text}") } else { result_text };
                flush_text(&mut content, messages);
                messages.push(format!("{{\"role\":\"tool\",\"tool_call_id\":{},\"content\":{}}}", quote(id), quote(&result_text)));
            }
            "image" => return Err(protocol::IMAGE_INPUT_UNSUPPORTED.into()),
            "document" => return Err("images and documents are unsupported by this text-only engine".into()),
            "thinking" | "tool_use" => return Err("thinking and tool_use blocks require an assistant role".into()),
            "tool_result" => return Err("tool_result blocks require a user role".into()),
            _ => return Err("unsupported Messages content block".into()),
        }
    }
    if role == "assistant" {
        let reasoning = if reasoning.is_empty() { String::new() }
            else { format!(",\"reasoning_content\":{}", quote(&reasoning)) };
        let calls = if calls.is_empty() { String::new() }
            else { format!(",\"tool_calls\":[{}]", calls.join(",")) };
        messages.push(format!("{{\"role\":\"assistant\",\"content\":{}{reasoning}{calls}}}", quote(&content)));
    } else {
        flush_text(&mut content, messages);
        if before == messages.len() { messages.push(message(role, "")); }
    }
    Ok(())
}

pub(crate) fn parse(body: &[u8], limits: &Limits, model: &str,
    reservation: Option<usize>) -> Result<(protocol::Request, Output), String> {
    let value = parse_json(body, limits)?;
    let object = value.object().map_err(|e| e.to_string())?;
    let source = std::str::from_utf8(body).map_err(|_| "invalid UTF-8")?;
    for key in ["mcp_servers", "container"] {
        if present(object, key).is_some() { return Err(format!("Anthropic-hosted {key} is unsupported")); }
    }
    let choice_object = present(object, "tool_choice").map(Value::object).transpose()
        .map_err(|_| "tool_choice must be an object")?;
    let choice = choice_object.map(|o| string(o, "type")).transpose()?.unwrap_or("auto");
    let chat_choice = match choice {
        "auto" | "none" => quote(choice),
        "any" => quote("required"),
        "tool" => format!("{{\"type\":\"function\",\"function\":{{\"name\":{}}}}}", quote(string(choice_object.unwrap(), "name")?)),
        _ => return Err("tool_choice type must be auto, none, any or tool".into()),
    };
    let single_call = choice_object.and_then(|o| present(o, "disable_parallel_tool_use"))
        .map(Value::boolean).transpose().map_err(|e| e.to_string())?.unwrap_or(false);
    let mut messages = Vec::new();
    if let Some(system) = present(object, "system") { messages.push(message("system", &text(system)?)); }
    let input = field(object, "messages")?.array().map_err(|e| e.to_string())?;
    if input.is_empty() || input.len() > 4096 { return Err("messages count outside 1..4096".into()); }
    for raw in raw_array(raw_member(source, "messages")?.ok_or("missing messages")?)? {
        translate_message(&raw, limits, &mut messages)?;
    }
    let mut tools = Vec::new();
    if let Some(value) = present(object, "tools") {
        if value.array().map_err(|e| e.to_string())?.len() > 64 { return Err("too many tools".into()); }
        for raw in raw_array(raw_member(source, "tools")?.ok_or("missing tools")?)? {
            let value = parse_json(raw.as_bytes(), limits)?;
            let tool = value.object().map_err(|e| e.to_string())?;
            if present(tool, "type").is_some_and(|v| v.string().ok() != Some("custom")) {
                return Err(format!("Input tag '{}' is unsupported; send custom tools with input_schema", string(tool, "type")?));
            }
            field(tool, "input_schema")?.object().map_err(|_| "tool input_schema must be an object")?;
            let mut fields = vec![format!("\"name\":{}", quote(string(tool, "name")?)),
                format!("\"parameters\":{}", raw_member(&raw, "input_schema")?.ok_or("missing input_schema")?)];
            if let Some(description) = present(tool, "description") {
                fields.push(format!("\"description\":{}", quote(description.string().map_err(|e| e.to_string())?)));
            }
            if choice != "none" { tools.push(format!("{{\"type\":\"function\",\"function\":{{{}}}}}", fields.join(","))); }
        }
    }
    let mut translated = vec![format!("\"messages\":[{}]", messages.join(",")),
        format!("\"tools\":[{}]", tools.join(",")), format!("\"tool_choice\":{chat_choice}"),
        format!("\"parallel_tool_calls\":{}", !single_call)];
    // Messages model names are client aliases; only the loaded model performs inference.
    let requested_model = present(object, "model").map(Value::string).transpose()
        .map_err(|e| e.to_string())?.map(str::to_owned);
    for key in ["max_tokens", "stream", "temperature", "top_p", "top_k", "min_p", "seed",
        "repetition_penalty", "presence_penalty", "frequency_penalty", "chat_template_kwargs",
        "cache", "cache_salt", "ignore_eos", "session", "cancel", "priority", "fork", "draft", "mtp_depth", "prompt_cache_key",
        "include_usage_timing", "kv_checkpoint", "kv_handle"] {
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
    if let Some(raw) = raw_member(source, "stop_sequences")?.filter(|s| s.trim() != "null") {
        field(object, "stop_sequences")?.array().map_err(|_| "stop_sequences must be an array")?;
        translated.push(format!("\"stop\":{raw}"));
    }
    if let Some(thinking) = present(object, "thinking") {
        let thinking = thinking.object().map_err(|e| e.to_string())?;
        let enabled = match string(thinking, "type")? {
            "enabled" | "adaptive" => true,
            "disabled" => false,
            _ => return Err("thinking type must be enabled, disabled or adaptive".into()),
        };
        translated.push(format!("\"enable_thinking\":{enabled}"));
    }
    if let Some(raw) = raw_member(source, "arm")? { translated.push(format!("\"arm\":{raw}")); }
    let translated = format!("{{{}}}", translated.join(","));
    Ok((protocol::parse_with_reservation(translated.as_bytes(), true, limits, model, reservation)?,
        Output { requested_model, tool_hash: crate::http::request_hash(body), ..Output::default() }))
}

enum Block {
    Text(String),
    Thinking(String),
    Tool { name: String, arguments: String, call: usize, complete: bool },
}
#[derive(Default)]
pub(crate) struct Output {
    tool_hash: u64,
    requested_model: Option<String>,
    blocks: Vec<Block>,
    open: Option<usize>,
    bytes: usize,
}
fn event(kind: &str, fields: &str) -> String {
    let fields = if fields.is_empty() { String::new() } else { format!(",{fields}") };
    format!("event: {kind}\ndata: {{\"type\":\"{kind}\"{fields}}}\n\n")
}
pub(crate) fn error_json(kind: &str, message: &str) -> String {
    format!("{{\"type\":\"error\",\"error\":{{\"type\":{},\"message\":{}}}}}", quote(kind), quote(message))
}
fn usage_json(usage: Usage) -> String {
    // Keep optional Vade accounting in the shared serializer.
    let shared = usage.json(true);
    let vade = raw_member(&shared, "vade").ok().flatten()
        .map(|v| format!(",\"vade\":{v}")).unwrap_or_default();
    format!("{{\"input_tokens\":{},\"output_tokens\":{},\"cache_read_input_tokens\":{},\"cache_creation_input_tokens\":0{vade}}}",
        usage.prompt_tokens.saturating_sub(usage.cached_tokens), usage.completion_tokens, usage.cached_tokens)
}
impl Output {
    fn block(&self, _id: u64, index: usize, empty: bool) -> String {
        match &self.blocks[index] {
            Block::Text(s) => format!("{{\"type\":\"text\",\"text\":{}}}", quote(if empty { "" } else { s })),
            Block::Thinking(s) => format!("{{\"type\":\"thinking\",\"thinking\":{},\"signature\":\"\"}}", quote(if empty { "" } else { s })),
            Block::Tool { name, arguments, call, .. } => format!("{{\"type\":\"tool_use\",\"id\":\"call-{}-{call}\",\"name\":{},\"input\":{}}}",
                self.tool_hash, quote(name), if empty { "{}" } else { arguments }),
        }
    }
    fn close(&mut self) -> String {
        self.open.take().map(|i| event("content_block_stop", &format!("\"index\":{i}"))).unwrap_or_default()
    }
    fn reserve(&mut self, bytes: usize, limits: &Limits) -> Result<(), String> {
        self.bytes = self.bytes.saturating_add(bytes);
        if self.bytes > limits.output_bytes.saturating_sub(1024) { return Err("output byte limit exceeded".into()); }
        Ok(())
    }
    pub fn count_tools(&self) -> usize {
        self.blocks.iter().filter(|b| matches!(b, Block::Tool { complete: true, .. })).count()
    }
    fn stop_reason(&self, reason: &str, stop_sequence: Option<&str>) -> &'static str {
        if stop_sequence.is_some() { "stop_sequence" }
        else if reason == "length" { "max_tokens" }
        else if self.count_tools() != 0 { "tool_use" }
        else { "end_turn" }
    }
    pub fn json(&self, model: &str, id: u64, reason: &str, usage: Usage, stop_sequence: Option<&str>) -> String {
        let model = self.requested_model.as_deref().unwrap_or(model);
        let blocks = (0..self.blocks.len()).map(|i| self.block(id, i, false)).collect::<Vec<_>>().join(",");
        let stop_reason = self.stop_reason(reason, stop_sequence);
        let stop_sequence = stop_sequence.map(quote).unwrap_or_else(|| "null".into());
        format!("{{\"id\":\"msg-{id}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":{},\"content\":[{blocks}],\"stop_reason\":\"{stop_reason}\",\"stop_sequence\":{stop_sequence},\"usage\":{}}}",
            quote(model), usage_json(usage))
    }
    pub fn start(&mut self, model: &str, id: u64, usage: Usage) -> String {
        let model = self.requested_model.as_deref().unwrap_or(model);
        let mut result = event("message_start", &format!("\"message\":{{\"id\":\"msg-{id}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":{},\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{}}}",
            quote(model), usage_json(Usage { completion_tokens: 0, ..usage })));
        result.push_str(&event("ping", ""));
        result
    }
    pub fn push(&mut self, value: Event, id: u64, call: &mut usize, limits: &Limits) -> Result<String, String> {
        if matches!(value, Event::Content(_)) && self.blocks.iter().any(|b| matches!(b, Block::Tool { .. })) {
            // The model's newline between tool markers is emitted by the shared
            // parser but the Chat template already supplies that separator.
            if matches!(&value, Event::Content(s) if s.trim().is_empty()) { return Ok(String::new()); }
            return Err("model emitted text after tool_use; Chat history cannot preserve this block order".into());
        }
        if matches!(value, Event::Reasoning(_)) && self.blocks.iter().any(|b| !matches!(b, Block::Thinking(_))) {
            return Err("model emitted thinking after text or tool_use; Chat history cannot preserve this block order".into());
        }
        let mut result = String::new();
        let thinking = matches!(value, Event::Reasoning(_));
        let text = match value {
            Event::Content(s) | Event::Reasoning(s) => s,
            Event::ToolStart { name } => {
                self.reserve(name.len(), limits)?;
                result.push_str(&self.close());
                let index = self.blocks.len();
                self.blocks.push(Block::Tool { name, arguments: String::new(), call: *call, complete: false });
                self.open = Some(index);
                result.push_str(&event("content_block_start", &format!("\"index\":{index},\"content_block\":{}", self.block(id, index, true))));
                return Ok(result);
            }
            Event::ToolArguments(arguments) => {
                self.reserve(arguments.len(), limits)?;
                let index = self.open.ok_or("tool arguments without a tool start")?;
                let Block::Tool { arguments: pending, .. } = &mut self.blocks[index] else {
                    return Err("tool arguments without a tool start".into());
                };
                pending.push_str(&arguments);
                if !arguments.is_empty() { result.push_str(&event("content_block_delta", &format!("\"index\":{index},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":{}}}", quote(&arguments)))); }
                return Ok(result);
            }
            Event::ToolCall { name, arguments } => {
                let index = self.open.ok_or("tool completion without a tool start")?;
                let Block::Tool { name: pending_name, arguments: pending, complete, .. } = &mut self.blocks[index] else {
                    return Err("tool completion without a tool start".into());
                };
                if *pending_name != name || *pending != arguments { return Err("streamed tool call differs from completed call".into()); }
                json::parse(arguments.as_bytes(), json::Limits { bytes: limits.output_bytes, depth: 24,
                    nodes: limits.output_bytes, string_bytes: limits.output_bytes })
                    .and_then(|v| v.object().map(|_| ()))
                    .map_err(|_| "model emitted invalid tool input; Anthropic tool input requires a JSON object")?;
                *complete = true;
                *call += 1;
                result.push_str(&self.close());
                return Ok(result);
            }
        };
        if text.is_empty() { return Ok(result); }
        self.reserve(text.len(), limits)?;
        if !self.open.is_some_and(|i| matches!((&self.blocks[i], thinking), (Block::Thinking(_), true) | (Block::Text(_), false))) {
            result.push_str(&self.close());
            let index = self.blocks.len();
            self.blocks.push(if thinking { Block::Thinking(String::new()) } else { Block::Text(String::new()) });
            self.open = Some(index);
            result.push_str(&event("content_block_start", &format!("\"index\":{index},\"content_block\":{}", self.block(id, index, true))));
        }
        let index = self.open.unwrap();
        match &mut self.blocks[index] { Block::Text(s) | Block::Thinking(s) => s.push_str(&text), _ => unreachable!() }
        let (kind, field) = if thinking { ("thinking_delta", "thinking") } else { ("text_delta", "text") };
        result.push_str(&event("content_block_delta", &format!("\"index\":{index},\"delta\":{{\"type\":\"{kind}\",\"{field}\":{}}}", quote(&text))));
        Ok(result)
    }
    pub fn finish(&mut self, _model: &str, _id: u64, reason: &str, usage: Usage, stop_sequence: Option<&str>) -> String {
        let mut result = self.close();
        let stop_reason = self.stop_reason(reason, stop_sequence);
        let stop_sequence = stop_sequence.map(quote).unwrap_or_else(|| "null".into());
        result.push_str(&event("message_delta", &format!("\"delta\":{{\"stop_reason\":\"{stop_reason}\",\"stop_sequence\":{stop_sequence}}},\"usage\":{}", usage_json(usage))));
        result.push_str(&event("message_stop", ""));
        result
    }
    pub fn failed(&mut self, kind: &str, message: &str) -> String {
        format!("event: error\ndata: {}\n\n", error_json(kind, message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mtp_depth_translation_preserves_non_next_body_budget() -> Result<(), String> {
        // Legal wire bodies with a tiny prompt and a large ignored value
        // must retain main's admission, at both generic and Flash body limits.
        for body_bytes in [262144, 4194304] {
            for spare in [0, 128] {
                for (prefix, suffix) in [("\"", "\""), ("[\"", "\"]"), ("{\"k\":\"", "\"}")] {
                    for (input, responses) in [
                        (r#""input":"hi","max_output_tokens":1"#, true),
                        (
                            r#""messages":[{"role":"user","content":"hi"}],"max_tokens":1"#,
                            false,
                        ),
                    ] {
                        let fields = format!(
                            r#"{input},"temperature":0,"top_k":1,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":42"#
                        );
                        let wire = |padding: &str| {
                            format!(r#"{{{fields},"mtp_depth":{prefix}{padding}{suffix}}}"#)
                        };
                        let body = wire(&"x".repeat(body_bytes - spare - wire("").len()));
                        assert_eq!(body.len(), body_bytes - spare);
                        for resident in [None, Some(3)] {
                            let limits = Limits {
                                body_bytes,
                                next_mtp_depth: resident,
                                ..Limits::default()
                            };
                            let parse_api = |body: &str| {
                                if responses {
                                    crate::responses::parse(
                                        body.as_bytes(),
                                        &limits,
                                        crate::MODEL_ID,
                                        None,
                                    )
                                    .map(|v| v.0)
                                } else {
                                    parse(body.as_bytes(), &limits, crate::MODEL_ID, None)
                                        .map(|v| v.0)
                                }
                            };
                            let request = parse_api(&body);
                            if resident.is_some() {
                                assert!(request.is_err(), "Next must reject noninteger mtp_depth");
                            } else {
                                let request = request?;
                                let baseline = parse_api(&format!("{{{fields}}}"))?;
                                assert_eq!(request.mtp_depth, None);
                                assert_eq!(request.draft, baseline.draft);
                                let protocol::Prompt::Chat(actual) = request.prompt else {
                                    return Err("translated request must be chat".into());
                                };
                                let protocol::Prompt::Chat(expected) = baseline.prompt else {
                                    return Err("baseline request must be chat".into());
                                };
                                assert_eq!(actual.messages, expected.messages);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn next_mtp_depth_survives_every_api_and_other_engines_ignore_it() {
        for resident in [None, Some(0), Some(1), Some(2), Some(3)] {
            let limits = Limits {
                next_mtp_depth: resident,
                ..Limits::default()
            };
            for raw in [
                "0", "1", "2", "3", "4", "-1", "1.5", "true", "\"2\"", "null",
            ] {
                let fields = format!(r#""mtp_depth":{raw},"temperature":0"#);
                let completions = format!(r#"{{"prompt":"hi",{fields}}}"#);
                let chat = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{fields}}}"#);
                let responses = format!(r#"{{"input":"hi",{fields}}}"#);
                let messages = format!(
                    r#"{{"messages":[{{"role":"user","content":"hi"}}],"max_tokens":8,{fields}}}"#
                );
                let requests = [
                    protocol::parse(completions.as_bytes(), false, &limits),
                    protocol::parse(chat.as_bytes(), true, &limits),
                    crate::responses::parse(responses.as_bytes(), &limits, crate::MODEL_ID, None)
                        .map(|v| v.0),
                    parse(messages.as_bytes(), &limits, crate::MODEL_ID, None).map(|v| v.0),
                ];
                let depth = raw.parse::<usize>().ok();
                for request in requests {
                    if resident.is_none() || raw == "null" {
                        let request = request.unwrap();
                        assert_eq!(request.mtp_depth, None);
                        assert_eq!(request.draft, None);
                    } else if depth.is_some_and(|k| k <= resident.unwrap()) {
                        let request = request.unwrap();
                        assert_eq!(request.mtp_depth, depth);
                        assert_eq!(
                            request.draft,
                            Some(if depth == Some(0) {
                                protocol::DraftMode::Plain
                            } else {
                                protocol::DraftMode::Strict
                            })
                        );
                    } else {
                        assert!(request.unwrap_err().contains("mtp_depth"), "{raw}");
                    }
                }
            }
        }
    }
    #[test]
    fn forced_tools_share_prompt_and_parser_across_apis() {
        let limits = Limits::default();
        let parse_apis = |fields: &str| { let chat = format!(r#"{{"messages":[{{"role":"user","content":"go"}}],{fields}}}"#); let responses = format!(r#"{{"input":"go",{fields}}}"#); let messages = format!(r#"{{"max_tokens":4,"messages":[{{"role":"user","content":"go"}}],{fields}}}"#); [protocol::parse(chat.as_bytes(), true, &limits), crate::responses::parse(responses.as_bytes(), &limits, crate::MODEL_ID, None).map(|v| v.0), parse(messages.as_bytes(), &limits, crate::MODEL_ID, None).map(|v| v.0)] };
        for salt in ["tenant-a", "tenant-b"] { for request in parse_apis(&format!(r#""cache_salt":{},"ignore_eos":true"#, quote(salt))) { let request = request.unwrap(); assert_eq!(request.cache_salt.as_deref(), Some(salt)); assert!(request.ignore_eos); } }
        for (fields, error) in [(r#""cache_salt":42"#, "cache_salt"), (r#""cache_salt":"""#, "cache_salt"), (r#""ignore_eos":"true""#, "boolean")] { for result in parse_apis(fields) { assert!(result.err().unwrap().contains(error)); } }
        for named in [false, true] { for parallel in [false, true] {
            let choice = if named { r#"{"type":"function","function":{"name":"f"}}"# } else { "\"required\"" };
            let chat = format!(r#"{{"messages":[{{"role":"user","content":"go"}}],"tools":[{{"type":"function","function":{{"name":"f","parameters":{{"type":"object"}}}}}}],"tool_choice":{choice},"parallel_tool_calls":{parallel}}}"#);
            let choice = if named { r#""type":"tool","name":"f""# } else { r#""type":"any""# };
            let body = format!(r#"{{"model":"client-model-alias","max_tokens":32,"messages":[{{"role":"user","content":"go"}}],"tools":[{{"name":"f","input_schema":{{"type":"object"}}}}],"tool_choice":{{{choice},"disable_parallel_tool_use":{}}}}}"#, !parallel);
            let (request, mut output) = parse(body.as_bytes(), &limits, "loaded", None).unwrap();
            assert!(output.start("loaded", 1, Usage::default()).contains("client-model-alias")); assert!(output.json("loaded", 1, "stop", Usage::default(), None).contains("client-model-alias")); assert!(parse(body.as_bytes(), &limits, "loaded", Some(0)).is_ok());
            let choice = if named { r#"{"type":"function","name":"f"}"# } else { "\"required\"" }; let responses = format!(r#"{{"input":"go","tools":[{{"type":"function","name":"f","parameters":{{"type":"object"}}}}],"tool_choice":{choice},"parallel_tool_calls":{parallel}}}"#);
            for request in [request, protocol::parse(chat.as_bytes(), true, &limits).unwrap(), crate::responses::parse(responses.as_bytes(), &limits, "loaded", None).unwrap().0] { let protocol::Prompt::Chat(input) = &request.prompt else { panic!() }; assert!(!input.enable_thinking); assert!(crate::tokenizer::format_chat(input, limits.body_bytes).unwrap().ends_with(&input.generation_prefix(false)));
                for hybrid in [false, true] { let call = if hybrid { "<tool_call>\n<function=f>\n</function>\n</tool_call>" } else { "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>" }; let generated = format!("{}{call}", call.strip_prefix(&input.generation_prefix(hybrid)).unwrap());
                    for split in 0..=generated.len() { let mut parser = crate::http::output_parser(&request, &limits, hybrid).unwrap().unwrap(); let mut events = parser.push(&generated.as_bytes()[..split]).unwrap(); events.extend(parser.push(&generated.as_bytes()[split..]).unwrap()); events.extend(parser.finish().unwrap()); assert_eq!(events.iter().filter(|e| matches!(e, Event::ToolCall { name, arguments } if name == "f" && arguments == "{}")).count(), if parallel { 2 } else { 1 }); assert_eq!(parser.stopped(), !parallel); }
                }
            }
        } }
        assert!(protocol::parse(br#"{"messages":[{"role":"user","content":"go"}],"tool_choice":"required"}"#, true, &limits).is_err());
        for bad in ["unknown", "<bad>"] { let body = format!(r#"{{"max_tokens":4,"messages":[{{"role":"user","content":"go"}}],"tools":[{{"name":"f","input_schema":{{}}}}],"tool_choice":{{"type":"tool","name":{}}}}}"#, quote(bad)); assert!(parse(body.as_bytes(), &limits, "loaded", None).is_err()); }
        assert!(protocol::parse_for_model(br#"{"model":"client-model-alias","messages":[{"role":"user","content":"go"}]}"#, true, &limits, "loaded").is_err());
    }
    #[test]
    fn translation_and_tool_round_trip() {
        let limits = Limits::default(); let arguments = r#"{ "z":2,"a": "é" }"#;
        let body = format!(r#"{{"max_tokens":4,"system":[{{"type":"text","text":"rules","cache_control":{{"type":"ephemeral"}}}}],"thinking":{{"type":"disabled"}},"messages":[{{"role":"assistant","content":[{{"type":"thinking","thinking":"plan","signature":"ignored"}},{{"type":"tool_use","id":"c","name":"f","input":{arguments}}}]}},{{"role":"user","content":[{{"type":"text","text":"before"}},{{"type":"tool_result","tool_use_id":"c","is_error":true,"content":[{{"type":"text","text":"failed"}}]}},{{"type":"text","text":"after"}}]}}],"tools":[{{"name":"f","input_schema":{{"type":"object","properties":{{"z":{{"type":"integer"}},"a":{{"type":"string"}}}}}}}}],"metadata":{{"user_id":"ignored"}}}}"#);
        let (request, mut output) = parse(body.as_bytes(), &limits, crate::MODEL_ID, None).unwrap(); let protocol::Prompt::Chat(chat) = request.prompt else { panic!() };
        let assistant = parse_json(chat.messages[1].as_bytes(), &limits).unwrap(); assert!(!chat.enable_thinking && assistant.object().unwrap()["reasoning_content"] == Value::String("plan".into()));
        assert!(chat.messages[1].contains(&quote(arguments))); assert!(chat.messages[3].contains("[tool error]\\nfailed"));
        assert!(chat.tools[0].contains(r#""properties":{"z":{"type":"integer"},"a":{"type":"string"}}"#));
        let mut call = 0; for event in [Event::ToolStart { name: "f".into() }, Event::ToolArguments(arguments.into()), Event::ToolCall { name: "f".into(), arguments: arguments.into() }] { output.push(event, 7, &mut call, &limits).unwrap(); }
        let result = output.json(crate::MODEL_ID, 7, "stop", Usage { prompt_tokens: 12, cached_tokens: 8, completion_tokens: 3, ..Usage::default() }, None);
        assert!(result.contains(&format!("\"input\":{arguments}"))); assert!(result.contains("\"input_tokens\":4")); assert!(result.contains("\"cache_read_input_tokens\":8"));
        let history = format!(r#"{{"max_tokens":4,"messages":[{{"role":"assistant","content":{}}}]}}"#, raw_member(&result, "content").unwrap().unwrap());
        let (round_trip, _) = parse(history.as_bytes(), &limits, crate::MODEL_ID, None).unwrap(); let protocol::Prompt::Chat(chat) = round_trip.prompt else { panic!() }; assert!(chat.messages[0].contains(&quote(arguments)));
        let mut output = Output::default(); let mut call = 0;
        let mut stream = output.start(crate::MODEL_ID, 1, Usage::default());
        for event in [Event::Reasoning("plan".into()), Event::Content("answer".into()), Event::ToolStart { name: "f".into() }, Event::ToolArguments("{}".into()), Event::ToolCall { name: "f".into(), arguments: "{}".into() }, Event::Content("\n".into())] { stream.push_str(&output.push(event, 1, &mut call, &limits).unwrap()); }
        stream.push_str(&output.finish(crate::MODEL_ID, 1, "stop", Usage::default(), None));
        assert_eq!(stream.lines().filter_map(|s| s.strip_prefix("event: ")).collect::<Vec<_>>(), ["message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
        for arguments in ["{", "[]", "1"] { let mut output = Output::default(); output.push(Event::ToolStart { name: "f".into() }, 1, &mut call, &limits).unwrap(); output.push(Event::ToolArguments(arguments.into()), 1, &mut call, &limits).unwrap(); assert!(output.push(Event::ToolCall { name: "f".into(), arguments: arguments.into() }, 1, &mut call, &limits).is_err()); }
    }
}
