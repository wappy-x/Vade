//! Bridge to the authored tokenizer and exact checkpoint chat template.
use crate::protocol::{ChatInput, present, raw_array, raw_member};
use engine_model::json::{self, Value, field};
use engine_tokenizer::chat::{self, Message, Role, ToolCall};
use std::collections::BTreeMap;

impl crate::Decoder for engine_tokenizer::Decoder<'_> {
    fn push(&mut self, token: u32) -> Result<String, String> {
        engine_tokenizer::Decoder::push(self, token).map_err(|e| e.to_string())
    }
    fn finish(&mut self) -> Result<String, String> {
        Ok(engine_tokenizer::Decoder::finish(self))
    }
}
impl crate::Tokenizer for engine_tokenizer::Tokenizer {
    type Decoder<'a> = engine_tokenizer::Decoder<'a>;
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_ordinary(text).map_err(|e| e.to_string())
    }
    fn decoder(&self) -> Self::Decoder<'_> {
        self.stream_decoder(true)
    }
    fn chat_decoder(&self) -> Self::Decoder<'_> {
        self.stream_decoder(true).gemma_chat()
    }
    fn gemma_chat(&self) -> bool { self.is_gemma4() }
    fn chat_thinking_open(&self, prompt: &[u32]) -> bool {
        self.is_gemma4() && self.encode("<|channel>thought\n")
            .is_ok_and(|suffix| prompt.ends_with(&suffix))
    }
    fn has_token(&self, token: u32) -> bool {
        self.contains_id(token)
    }
    fn hybrid_chat(&self) -> bool { self.is_hybrid_text() }
    fn encode_chat(&self, input: &ChatInput, max_bytes: usize) -> Result<Vec<u32>, String> {
        let messages = parse_messages(input, max_bytes, self.is_gemma4())?;
        chat::encode(
            self,
            &messages,
            &chat::Options {
                enable_thinking: input.enable_thinking,
                reasoning_effort: input.reasoning_effort,
                tools: input.callable_tools().to_vec(),
                generation_prefix: input.generation_prefix(self.is_hybrid_text()),
                max_bytes,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())
    }
}
fn object(source: &str, max_bytes: usize) -> Result<Value, String> {
    json::parse(
        source.as_bytes(),
        json::Limits {
            bytes: max_bytes,
            depth: 24,
            nodes: 65536,
            string_bytes: max_bytes,
        },
    )
    .map_err(|e| e.to_string())
}
fn string_field<'a>(object: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a str, String> {
    field(object, key)
        .and_then(Value::string)
        .map_err(|e| e.to_string())
}
/// Tool IDs are accepted routing metadata; the Qwen template itself renders
/// ordered tool responses without IDs. Multimodal content is rejected;
/// unknown metadata is ignored. Use encode_chat for inference.
pub fn format_chat(input: &ChatInput, max_bytes: usize) -> Result<String, String> {
    let messages = parse_messages(input, max_bytes, false)?;
    chat::format(
        &messages,
        &chat::Options {
            enable_thinking: input.enable_thinking,
            reasoning_effort: input.reasoning_effort,
            tools: input.callable_tools().to_vec(),
            generation_prefix: input.generation_prefix(false),
            max_bytes,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}
fn parse_messages(input: &ChatInput, max_bytes: usize, gemma: bool) -> Result<Vec<Message>, String> {
    let mut messages = Vec::new();
    for source in &input.messages {
        let root = object(source, max_bytes)?;
        let m = root.object().map_err(|e| e.to_string())?;
        let role = match string_field(m, "role")? {
            "developer" if gemma => Role::Developer,
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => return Err("unsupported chat role".into()),
        };
        let content = match m.get("content") {
            Some(value @ (Value::String(_) | Value::Array(_))) => crate::protocol::text_content(value)?,
            // A length-limited reply can contain reasoning without any answer or calls.
            Some(Value::Null) | None if role == Role::Assistant => {
                String::new()
            }
            _ => return Err("message content must be a string or text parts".into()),
        };
        let reasoning_content = match if gemma { m.get("reasoning").filter(|v| matches!(v, Value::String(s) if !s.is_empty())).or_else(|| m.get("reasoning_content")) } else { m.get("reasoning_content") } {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.string().map_err(|e| e.to_string())?.to_owned()),
        };
        if let Some(id) = present(m, "tool_call_id") {
            if role != Role::Tool || id.string().map_err(|e| e.to_string())?.is_empty() {
                return Err("tool_call_id requires a nonempty string on a tool message".into());
            }
        }
        let mut tool_calls = Vec::new();
        if let Some(calls) = raw_member(source, "tool_calls")?.filter(|raw| raw.trim() != "null") {
            let raw_calls = raw_array(calls)?;
            if raw_calls.len() > 64 {
                return Err("too many tool calls".into());
            }
            for raw_call in raw_calls {
                let root = object(&raw_call, max_bytes)?;
                let call = root.object().map_err(|e| e.to_string())?;
                if let Some(id) = present(call, "id") {
                    id.string().map_err(|e| e.to_string())?;
                }
                if let Some(kind) = present(call, "type") {
                    if kind.string().map_err(|e| e.to_string())? != "function" {
                        return Err("only function tool calls are supported".into());
                    }
                }
                let raw_function =
                    raw_member(&raw_call, "function")?.ok_or("missing tool function")?;
                let function_value = object(raw_function, max_bytes)?;
                let function = function_value.object().map_err(|e| e.to_string())?;
                let name = string_field(function, "name")?.to_owned();
                let arguments = match field(function, "arguments").map_err(|e| e.to_string())? {
                    Value::String(s) => s.clone(),
                    Value::Object(_) => chat::json_for_template(
                        raw_member(raw_function, "arguments")?.ok_or("missing arguments")?,
                    )
                    .map_err(|e| e.to_string())?,
                    _ => {
                        return Err(
                            "tool arguments must be a JSON object or serialized JSON string".into(),
                        );
                    }
                };
                tool_calls.push(ToolCall { name, arguments, id: present(call, "id").map(|v| v.string().map(str::to_owned)).transpose().map_err(|e| e.to_string())? });
            }
        }
        messages.push(Message {
            role,
            content,
            reasoning_content,
            tool_calls,
            tool_call_id: present(m, "tool_call_id").map(|v| v.string().map(str::to_owned)).transpose().map_err(|e| e.to_string())?,
            tool_name: if gemma { present(m, "name").map(|v| v.string().map(str::to_owned)).transpose().map_err(|e| e.to_string())? } else { None },
        });
    }
    for raw_tool in &input.tools {
        let root = object(raw_tool, max_bytes)?;
        let tool = root.object().map_err(|e| e.to_string())?;
        if string_field(tool, "type")? != "function" {
            return Err("only function tools are supported".into());
        }
        let function = field(tool, "function")
            .and_then(Value::object)
            .map_err(|e| e.to_string())?;
        if string_field(function, "name")?.is_empty() {
            return Err("empty tool name".into());
        }
        if let Some(description) = present(function, "description") {
            description.string().map_err(|e| e.to_string())?;
        }
        if let Some(parameters) = present(function, "parameters") {
            parameters.object().map_err(|e| e.to_string())?;
        }
    }
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_assistant_history_round_trips_chat_and_responses() {
        let limits = crate::Limits::default();
        for thought in ["", "partial reasoning é"] {
            for followup in ["", r#",{"role":"user","content":"continue"}"#] {
                let mut expected = ChatInput { messages: vec![r#"{"role":"user","content":"go"}"#.into(),
                    format!(r#"{{"role":"assistant","content":"","reasoning_content":{thought:?}}}"#)],
                    tools: vec![], enable_thinking: true, reasoning_effort: Default::default(), tool_choice: Default::default(), parallel_tool_calls: true };
                if !followup.is_empty() { expected.messages.push(followup[1..].into()); }
                for content in [r#", "content":"""#, r#", "content":null"#, ""] {
                    for responses in [false, true] {
                        let body = if responses {
                            format!(r#"{{"input":[{{"role":"user","content":"go"}},{{"type":"reasoning","summary":[{{"type":"summary_text","text":{thought:?}}}]}},{{"role":"assistant"{content}}}{followup}]}}"#)
                        } else {
                            format!(r#"{{"messages":[{{"role":"user","content":"go"}},{{"role":"assistant","reasoning_content":{thought:?}{content}}}{followup}]}}"#)
                        };
                        let request = if responses { crate::responses::parse(body.as_bytes(), &limits, "model", None).unwrap().0 }
                            else { crate::protocol::parse(body.as_bytes(), true, &limits).unwrap() };
                        let crate::protocol::Prompt::Chat(input) = request.prompt else { panic!() };
                        let messages = parse_messages(&input, limits.body_bytes, false).unwrap();
                        assert_eq!(messages.len(), expected.messages.len());
                        assert!(messages[1].content.is_empty() && messages[1].tool_calls.is_empty());
                        assert_eq!(messages[1].reasoning_content.as_deref().unwrap_or(""), thought);
                        assert_eq!(format_chat(&input, limits.body_bytes).unwrap(), format_chat(&expected, limits.body_bytes).unwrap());
                    }
                }
            }
        }
    }
    #[test]
    fn ordered_object_and_serialized_arguments_remain_distinct() {
        let request = crate::protocol::parse(br#"{"messages":[{"role":"user","content":"go"},{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"f","arguments":{"z":1,"a":2}}},{"type":"function","function":{"name":"g","arguments":"{\"z\":1,\"a\":2}"}}]},{"role":"tool","content":"done","tool_call_id":"call-1"}],"stream":true,"enable_thinking":false}"#, true, &crate::Limits::default()).unwrap();
        let crate::protocol::Prompt::Chat(input) = request.prompt else {
            panic!()
        };
        let rendered = format_chat(&input, 8192).unwrap();
        assert!(rendered.contains("\"name\": \"f\", \"arguments\": {\"z\": 1, \"a\": 2}"));
        assert!(rendered.contains("\"name\": \"g\", \"arguments\": {\"z\":1,\"a\":2}"));
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
    #[test]
    fn unsupported_content_and_roles_fail() {
        for message in [
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}"#,
            r#"{"role":"unknown","content":"x"}"#,
            r#"{"role":"user","content":null}"#,
            r#"{"role":"assistant","content":123}"#,
        ] {
            let input = ChatInput {
                messages: vec![message.into()],
                tools: vec![],
                enable_thinking: true,
                reasoning_effort: Default::default(),
                tool_choice: Default::default(), parallel_tool_calls: true,
            };
            assert!(format_chat(&input, 8192).is_err());
        }
    }
}
