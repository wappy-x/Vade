//! Bridge to the authored tokenizer and exact checkpoint chat template.
use crate::protocol::{ChatInput, present, raw_array, raw_member};
use engine_model::json::{self, Value, field};
use engine_tokenizer::chat::{self, Message, Role, ToolCall};
use std::collections::BTreeMap;
pub mod glm;
#[cfg(test)] mod glm_external_tests;

/// Select the authored renderer from checkpoint template data.
/// Other templates retain the tokenizer crate's own validation and rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatTemplate { Checkpoint, Glm }
impl ChatTemplate {
    pub fn from_config(config: &Value, separate: Option<&str>) -> Result<Self, String> {
        let fields = config.object().map_err(|e| e.to_string())?;
        let embedded = fields.get("chat_template").and_then(|v| v.string().ok());
        let normalize = crate::contract::normalize_template;
        if embedded.into_iter().chain(separate).any(|t| t.trim_start().starts_with("[gMASK]")) {
            if fields.contains_key("chat_template") && embedded.is_none() { return Err("GLM chat_template must be a string".into()); }
            for template in embedded.into_iter().chain(separate) {
                if normalize(template) != normalize(glm::TEMPLATE) { return Err("unsupported GLM chat template configuration".into()); }
            }
            Ok(Self::Glm)
        } else { Ok(Self::Checkpoint) }
    }
}
/// Checkpoint tokenizer with a renderer selected from its template artifacts.
/// GLM tokenization stays in the tokenizer crate. This bridge owns its prompt.
pub struct ModelTokenizer { inner: engine_tokenizer::Tokenizer, template: ChatTemplate, checkpoint_eos: Option<Vec<u32>>, limits: engine_tokenizer::Limits }
impl std::ops::Deref for ModelTokenizer {
    type Target = engine_tokenizer::Tokenizer;
    fn deref(&self) -> &Self::Target { &self.inner }
}
impl ModelTokenizer {
    pub fn from_directory(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let read = |name: &str| engine_model::safetensors::read_bounded(&path.join(name), 4 << 20).map_err(|e| e.to_string());
        let config = read("tokenizer_config.json")?;
        let config = json::parse(&config, json::Limits { bytes: 4 << 20, ..Default::default() }).map_err(|e| e.to_string())?;
        let separate = match std::fs::metadata(path.join("chat_template.jinja")) {
            Ok(_) => Some(String::from_utf8(read("chat_template.jinja")?).map_err(|_| "invalid chat template UTF-8")?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.to_string()),
        };
        let template = ChatTemplate::from_config(&config, separate.as_deref())?;
        let inner = engine_tokenizer::Tokenizer::from_directory(path).map_err(|e| e.to_string())?;
        let checkpoint_eos = if template == ChatTemplate::Glm {
            glm::validate_tokens(&inner)?;
            let eos = inner.eos_token_ids().map_err(|e| e.to_string())?;
            if eos.is_empty() {
                return Err("chat checkpoint requires a generation EOS list".into());
            }
            Some(eos.to_vec())
        } else { None };
        Ok(Self { inner, template, checkpoint_eos, limits: Default::default() })
    }
    pub fn set_limits(&mut self, limits: engine_tokenizer::Limits) -> Result<(), String> {
        self.inner.set_limits(limits).map_err(|e| e.to_string())?;
        self.limits = limits; Ok(())
    }
    pub fn format_chat_input(&self, input: &ChatInput, max_bytes: usize) -> Result<String, String> {
        let messages = parse_messages_with_roles(input, max_bytes, self.inner.is_gemma4(), self.template == ChatTemplate::Glm)?;
        match self.template {
            ChatTemplate::Glm => glm::format(&messages, &glm_options(input, max_bytes)?),
            ChatTemplate::Checkpoint => self.inner.format_chat(&messages, &chat_options(input, max_bytes, self.inner.is_hybrid_text())).map_err(|e| e.to_string()),
        }
    }
}
fn chat_options(input: &ChatInput, max_bytes: usize, hybrid: bool) -> chat::Options {
    chat::Options { enable_thinking: input.enable_thinking, reasoning_effort: input.reasoning_effort.qwen(),
        tools: input.callable_tools().to_vec(), generation_prefix: input.generation_prefix(hybrid), max_bytes, ..Default::default() }
}
fn glm_options(input: &ChatInput, max_bytes: usize) -> Result<glm::Options, String> {
    if input.forced_tool_choice() { return Err("forced tool_choice is not supported by the GLM chat renderer".into()); }
    // Match the maker: GLM's chat template has no thinking-off form. It always
    // renders a Reasoning Effort line (low/high, else max) and opens the think
    // channel, so an explicit thinking-off ask can only be refused, never faked.
    // A thinking-off resolution from the server default alone is not a client ask
    // and still renders exactly as before.
    if input.thinking_off_requested {
        return Err("thinking cannot be disabled for GLM (enable_thinking:false, reasoning_effort none/minimal or /no_think): the checkpoint chat template has no thinking-off form; omit those fields or set reasoning_effort low, high or max".into());
    }
    Ok(glm::Options { reasoning_effort: input.reasoning_effort.glm().into(),
        tools: input.callable_tools().to_vec(), max_bytes, ..Default::default() })
}
impl crate::Tokenizer for ModelTokenizer {
    type Decoder<'a> = engine_tokenizer::Decoder<'a>;
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> { crate::Tokenizer::encode(&self.inner, text) }
    fn decoder(&self) -> Self::Decoder<'_> { crate::Tokenizer::decoder(&self.inner) }
    fn chat_decoder(&self) -> Self::Decoder<'_> { crate::Tokenizer::chat_decoder(&self.inner) }
    fn gemma_chat(&self) -> bool { crate::Tokenizer::gemma_chat(&self.inner) }
    fn glm_chat(&self) -> bool { self.template == ChatTemplate::Glm }
    fn hybrid_chat(&self) -> bool { crate::Tokenizer::hybrid_chat(&self.inner) }
    fn has_token(&self, token: u32) -> bool { crate::Tokenizer::has_token(&self.inner, token) }
    fn checkpoint_eos_token_ids(&self) -> Result<Option<&[u32]>, String> {
        Ok(self.checkpoint_eos.as_deref())
    }
    fn chat_thinking_open(&self, prompt: &[u32]) -> bool {
        if self.template == ChatTemplate::Glm {
            self.inner.encode("<think>").is_ok_and(|suffix| prompt.ends_with(&suffix))
        } else { crate::Tokenizer::chat_thinking_open(&self.inner, prompt) }
    }
    fn encode_chat(&self, input: &ChatInput, max_bytes: usize) -> Result<Vec<u32>, String> {
        match self.template {
            ChatTemplate::Checkpoint => crate::Tokenizer::encode_chat(&self.inner, input, max_bytes),
            ChatTemplate::Glm => glm::encode(&self.inner, &parse_messages_with_roles(input, max_bytes, false, true)?, &glm_options(input, max_bytes)?, self.limits),
        }
    }
}

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
        chat::encode(self, &messages, &chat_options(input, max_bytes, self.is_hybrid_text()))
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
            reasoning_effort: input.reasoning_effort.qwen(),
            tools: input.callable_tools().to_vec(),
            generation_prefix: input.generation_prefix(false),
            max_bytes,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}
fn parse_messages(input: &ChatInput, max_bytes: usize, gemma: bool) -> Result<Vec<Message>, String> {
    parse_messages_with_roles(input, max_bytes, gemma, false)
}
fn parse_messages_with_roles(input: &ChatInput, max_bytes: usize, gemma: bool, template_roles: bool) -> Result<Vec<Message>, String> {
    let mut messages = Vec::new();
    for source in &input.messages {
        let root = object(source, max_bytes)?;
        let m = root.object().map_err(|e| e.to_string())?;
        let role = match string_field(m, "role")? {
            "developer" if gemma || template_roles => Role::Developer,
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
        if let Some(id) = present(m, "tool_call_id")
            && (role != Role::Tool || id.string().map_err(|e| e.to_string())?.is_empty()) {
                return Err("tool_call_id requires a nonempty string on a tool message".into());
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
                if let Some(kind) = present(call, "type")
                    && kind.string().map_err(|e| e.to_string())? != "function" {
                        return Err("only function tool calls are supported".into());
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
            tool_call_id: present(m, "tool_call_id").or_else(|| {
                if template_roles && role == Role::Tool { present(m, "id") } else { None }
            }).map(|v| v.string().map(str::to_owned)).transpose().map_err(|e| e.to_string())?,
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct TestDirectory(std::path::PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            loop {
                let root = std::env::temp_dir().join(format!("vade-glm-chat-{}-{}",
                    std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
                match std::fs::create_dir(&root) {
                    Ok(()) => return Self(root),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("cannot create test folder: {e}"),
                }
            }
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let removed = std::fs::remove_dir_all(&self.0);
            if !std::thread::panicking() { removed.unwrap(); }
        }
    }
    const FIXTURE: &str = include_str!("../../tokenizer/tests/fixtures/host_edges/tokenizer.json");
    fn codec(hybrid: bool) -> engine_tokenizer::Tokenizer {
        let mut source = FIXTURE.to_owned();
        if hybrid {
            source = source.replace(r"\\p{L}+", r"[\\p{L}\\p{M}]+").replace(r"\\s\\p{L}\\p{N}", r"\\s\\p{L}\\p{M}\\p{N}");
            for (old, new) in [(151643, 248044), (151644, 248045), (151645, 248046)] {
                source = source.replace(&format!("\"id\":{old}"), &format!("\"id\":{new}"));
            }
        }
        engine_tokenizer::Tokenizer::from_json(source.as_bytes(), Default::default()).unwrap()
    }
    fn glm_fixture() -> String {
        let controls = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
            "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>", "</tool_response>"];
        let extra = controls.iter().enumerate().map(|(i, text)| format!(r#"{{"id":{},"content":{},"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":{}}}"#,
            300+i, chat::quote(text), i < 6)).collect::<Vec<_>>().join(",");
        FIXTURE.replacen(r#""added_tokens":["#, &format!(r#""added_tokens":[{extra},"#), 1)
            .replace(r#""normalizer":{"type":"NFC"}"#, r#""normalizer":null"#)
            .replace(r#""trim_offsets":false"#, r#""trim_offsets":true"#)
            .replace(r#""decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}"#,
                r#""decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}"#)
    }
    fn chat(body: &str) -> ChatInput {
        let crate::protocol::Prompt::Chat(input) = crate::protocol::parse(body.as_bytes(), true, &crate::Limits::default()).unwrap().prompt else { panic!() };
        input
    }
    #[test]
    fn template_data_selects_glm_and_rejects_edits() {
        let config = |template: &str| object(&format!(r#"{{"model_type":"other","chat_template":{}}}"#, chat::quote(template)), 4 << 20).unwrap();
        let empty = object("{}", 4 << 20).unwrap();
        assert_eq!(ChatTemplate::from_config(&config(glm::TEMPLATE), None).unwrap(), ChatTemplate::Glm);
        assert_eq!(ChatTemplate::from_config(&empty, Some(glm::TEMPLATE)).unwrap(), ChatTemplate::Glm);
        assert_eq!(ChatTemplate::from_config(&config(glm::TEMPLATE), Some(&format!("{}\r\n", glm::TEMPLATE))).unwrap(), ChatTemplate::Glm);
        assert_eq!(ChatTemplate::from_config(&config("qwen template"), None).unwrap(), ChatTemplate::Checkpoint);
        for edited in [glm::TEMPLATE.replacen("capitalize", "upper", 1), format!("{}x", glm::TEMPLATE)] {
            assert!(ChatTemplate::from_config(&config(&edited), None).is_err());
            assert!(ChatTemplate::from_config(&config(glm::TEMPLATE), Some(&edited)).is_err());
        }
        assert!(ChatTemplate::from_config(&config("other template"), Some(glm::TEMPLATE)).is_err());
    }
    #[test]
    fn qwen_hybrid_bytes_stay_equal() {
        let model = ModelTokenizer { inner: codec(true), template: ChatTemplate::Checkpoint, checkpoint_eos: None, limits: Default::default() };
        assert!(model.inner.is_hybrid_text());
        let input = chat(r#"{"messages":[{"role":"system","content":" s "},{"role":"user","content":" u "},{"role":"assistant","content":" a ","reasoning_content":" old "},{"role":"user","content":" next "}],"enable_thinking":false}"#);
        let expected = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\nold\n</think>\n\na<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(model.format_chat_input(&input, 8192).unwrap().as_bytes(), expected.as_bytes());
        assert_eq!(crate::Tokenizer::encode_chat(&model, &input, 8192).unwrap(), crate::Tokenizer::encode_chat(&model.inner, &input, 8192).unwrap());
        // Retain the old Qwen alias bytes, including high -> xhigh.
        for effort in ["low", "medium", "high", "xhigh", "max", "none"] {
            let input = chat(&format!(r#"{{"messages":[{{"role":"user","content":"go"}}],"reasoning_effort":"{effort}"}}"#));
            let messages = [Message::new(Role::User, "go")];
            let qwen = match effort { "low" => chat::ReasoningEffort::Low, "medium" => chat::ReasoningEffort::Medium, _ => chat::ReasoningEffort::XHigh };
            let options = chat::Options { enable_thinking: effort != "none", reasoning_effort: qwen, ..Default::default() };
            let expected = model.inner.format_chat(&messages, &options).unwrap();
            assert_eq!(model.format_chat_input(&input, 8192).unwrap().as_bytes(), expected.as_bytes(), "{effort}");
            assert_eq!(crate::Tokenizer::encode_chat(&model, &input, 8192).unwrap(), chat::encode(&model.inner, &messages, &options).unwrap(), "{effort}");
        }
    }
    #[test]
    fn glm_bridge_retains_high_and_template_thinking() {
        let model = ModelTokenizer { inner: codec(false), template: ChatTemplate::Glm, checkpoint_eos: None, limits: Default::default() };
        assert!(crate::Tokenizer::glm_chat(&model));
        // Hand-derived from glm.jinja lines 2-3, 139, 260. enable_thinking
        // is not a template input and cannot change its generation prompt.
        for (fields, expected) in [
            (r#""reasoning_effort":"high""#, "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>go<|assistant|><think>"),
            (r#""reasoning_effort":"max""#, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>go<|assistant|><think>"),
            (r#""reasoning_effort":"high","chat_template_kwargs":{"reasoning_effort":"low"}"#, "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>go<|assistant|><think>"),
        ] {
            let input = chat(&format!(r#"{{"messages":[{{"role":"user","content":"go"}}],{fields}}}"#));
            assert_eq!(model.format_chat_input(&input, 8192).unwrap().as_bytes(), expected.as_bytes());
        }
    }
    #[test]
    fn glm_thinking_off_is_refused_only_when_requested() {
        let model = ModelTokenizer { inner: codec(false), template: ChatTemplate::Glm, checkpoint_eos: None, limits: Default::default() };
        // The maker template has no thinking-off form, so an explicit ask is
        // refused, never rendered as Max while the metadata says none.
        for fields in [
            r#""enable_thinking":false"#,
            r#""chat_template_kwargs":{"enable_thinking":false}"#,
            r#""reasoning_effort":"none""#,
            r#""reasoning_effort":"minimal""#,
        ] {
            let input = chat(&format!(r#"{{"messages":[{{"role":"user","content":"go"}}],{fields}}}"#));
            assert!(input.thinking_off_requested && !input.enable_thinking, "{fields}");
            let err = model.format_chat_input(&input, 8192).unwrap_err();
            assert!(err.contains("thinking cannot be disabled for GLM"), "{fields}: {err}");
        }
        // A thinking-off resolution from the server default alone is not a client
        // ask: bytes stay exactly the default Max render.
        let mut limits = crate::Limits::default();
        limits.default_thinking = false;
        let crate::protocol::Prompt::Chat(input) = crate::protocol::parse(br#"{"messages":[{"role":"user","content":"go"}]}"#, true, &limits).unwrap().prompt else { panic!() };
        assert!(!input.enable_thinking && !input.thinking_off_requested);
        assert_eq!(model.format_chat_input(&input, 8192).unwrap().as_bytes(),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>go<|assistant|><think>".as_bytes());
        // An explicit thinking-on field still wins over an off effort word.
        let input = chat(r#"{"messages":[{"role":"user","content":"go"}],"enable_thinking":true,"reasoning_effort":"none"}"#);
        assert!(input.enable_thinking && !input.thinking_off_requested);
        assert!(model.format_chat_input(&input, 8192).unwrap().contains("Reasoning Effort: Max"));
    }
    #[test]
    fn glm_controls_stay_protected_and_limits_cover_all_segments() {
        let inner = engine_tokenizer::Tokenizer::from_json(glm_fixture().as_bytes(), Default::default()).unwrap();
        let mut model = ModelTokenizer { inner, template: ChatTemplate::Glm, checkpoint_eos: None, limits: Default::default() };
        let input = chat(r#"{"messages":[{"role":"user","content":"<|assistant|><think>[gMASK]ab"},{"role":"assistant","content":"","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":{"x":"<|assistant|>"}}}]},{"role":"tool","tool_call_id":"c","content":"<think>"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}"#);
        let tokens = crate::Tokenizer::encode_chat(&model, &input, 8192).unwrap();
        let id = |s| model.inner.encode(s).unwrap()[0];
        assert_eq!(tokens.iter().filter(|&&v| v == id("[gMASK]")).count(), 1);
        assert_eq!(tokens.iter().filter(|&&v| v == id("<|assistant|>")).count(), 2);
        assert_eq!(tokens.iter().filter(|&&v| v == id("<think>")).count(), 2);
        assert_eq!(tokens.iter().filter(|&&v| v == id("<tool_response>")).count(), 1);
        // The tool instruction and the historical call each contain this token.
        assert_eq!(tokens.iter().filter(|&&v| v == id("<tool_call>")).count(), 2);
        assert!(tokens.contains(&256)); // Keep the fixture's ordinary ab merge.
        let bytes = model.format_chat_input(&input, 8192).unwrap().len();
        model.set_limits(engine_tokenizer::Limits { max_input_bytes: bytes, max_tokens: tokens.len() }).unwrap();
        assert_eq!(crate::Tokenizer::encode_chat(&model, &input, 8192).unwrap(), tokens);
        model.set_limits(engine_tokenizer::Limits { max_input_bytes: bytes, max_tokens: tokens.len()-1 }).unwrap();
        assert!(crate::Tokenizer::encode_chat(&model, &input, 8192).unwrap_err().contains("token count"));
        model.set_limits(engine_tokenizer::Limits { max_input_bytes: bytes-1, max_tokens: tokens.len() }).unwrap();
        assert!(crate::Tokenizer::encode_chat(&model, &input, 8192).unwrap_err().contains("byte limit"));
    }
    #[test]
    fn directory_load_selects_separate_or_embedded_template() {
        let directory = TestDirectory::new();
        let root = directory.0.clone();
        std::fs::write(root.join("tokenizer.json"), glm_fixture()).unwrap();
        std::fs::write(root.join("generation_config.json"), r#"{"eos_token_id":[151643,303,305]}"#).unwrap();
        for embedded in [false, true] {
            let config = if embedded { format!(r#"{{"chat_template":{}}}"#, chat::quote(glm::TEMPLATE)) } else { "{}".into() };
            std::fs::write(root.join("tokenizer_config.json"), config).unwrap();
            std::fs::write(root.join("chat_template.jinja"), glm::TEMPLATE).unwrap();
            let tokenizer = ModelTokenizer::from_directory(&root).unwrap();
            assert_eq!(tokenizer.template, ChatTemplate::Glm);
            assert_eq!(crate::Tokenizer::checkpoint_eos_token_ids(&tokenizer).unwrap(), Some([151643,303,305].as_slice()));
            let input = chat(r#"{"messages":[{"role":"user","content":"go"}]}"#);
            let tokens = crate::Tokenizer::encode_chat(&tokenizer, &input, 8192).unwrap();
            assert!(crate::Tokenizer::chat_thinking_open(&tokenizer, &tokens));
        }
        std::fs::write(root.join("tokenizer.json"), FIXTURE).unwrap();
        assert!(ModelTokenizer::from_directory(&root).is_err());
        drop(directory);
        assert!(!root.exists());
    }
    #[test]
    fn checkpoint_stop_set_is_required_and_bound_to_both_request_paths() {
        let directory = TestDirectory::new();
        let root = &directory.0;
        std::fs::write(root.join("tokenizer.json"), glm_fixture()).unwrap();
        std::fs::write(root.join("tokenizer_config.json"), "{}").unwrap();
        std::fs::write(root.join("chat_template.jinja"), glm::TEMPLATE).unwrap();
        assert!(ModelTokenizer::from_directory(root).is_err());
        for config in [r#"{"eos_token_id":[]}"#, r#"{"eos_token_id":[999999]}"#, "invalid"] {
            std::fs::write(root.join("generation_config.json"), config).unwrap();
            assert!(ModelTokenizer::from_directory(root).is_err());
        }
        std::fs::write(root.join("generation_config.json"), r#"{"eos_token_id":[151643,303,305]}"#).unwrap();
        let model = ModelTokenizer::from_directory(root).unwrap();
        let limits = crate::Limits::default();
        for (path, body) in [("/v1/completions", r#"{"prompt":"go","max_tokens":1}"#),
            ("/v1/chat/completions", r#"{"messages":[{"role":"user","content":"go"}],"max_tokens":1}"#)] {
            let http = crate::http::HttpRequest { path: path.into(), body: body.as_bytes().to_vec(), authorization: None };
            let admission = format!(r#"{{"endpoint":{path:?},"output_tokens":1,"request":{body}}}"#);
            for stops in [&[][..], &[151643][..], &[151643,303][..], &[151643,303,305,1][..]] {
                let error = crate::http::prepare_request(&http, 8192, 151646, stops, Some(&model), &limits, "model").err().unwrap();
                assert!(error.contains("checkpoint EOS set"), "{error}");
                assert!(crate::contract::admission(admission.as_bytes(), 8192, 151646, stops, Some(&model), &limits, "model").unwrap_err().contains("checkpoint EOS set"));
            }
            let stops = &[305,151643,303];
            crate::http::prepare_request(&http, 8192, 151646, stops, Some(&model), &limits, "model").unwrap();
            assert!(crate::contract::admission(admission.as_bytes(), 8192, 151646, stops, Some(&model), &limits, "model").unwrap().contains("\"admitted\":true"));
        }
        let empty = crate::http::HttpRequest { path: "/v1/completions".into(), body: br#"{"prompt":"","max_tokens":1}"#.to_vec(), authorization: None };
        assert!(crate::http::prepare_request(&empty, 8192, 151646, &[151643,303,305], Some(&model), &limits, "model").err().unwrap().contains("prompt must contain at least one token"));
        let legacy = ModelTokenizer { inner: codec(true), template: ChatTemplate::Checkpoint, checkpoint_eos: None, limits: Default::default() };
        assert_eq!(crate::Tokenizer::checkpoint_eos_token_ids(&legacy).unwrap(), None);
    }
    #[test]
    fn glm_developer_role_follows_the_pinned_template() {
        let model = ModelTokenizer { inner: codec(false), template: ChatTemplate::Glm, checkpoint_eos: None, limits: Default::default() };
        let input = chat(r#"{"messages":[{"role":"user","content":"雪"},{"role":"developer","content":"omit"},{"role":"system","content":""},{"role":"assistant","content":"","reasoning_content":"é"}]}"#);
        // glm.jinja lines 139-164 and 255-260 have no developer branch.
        assert_eq!(model.format_chat_input(&input, 8192).unwrap(), "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>雪<|system|><|assistant|><think>é</think><|assistant|><think>");
    }
    #[test]
    fn glm_tool_result_id_fallback_keeps_call_order() {
        let model = ModelTokenizer { inner: codec(false), template: ChatTemplate::Glm, checkpoint_eos: None, limits: Default::default() };
        let input = chat(r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"a","arguments":{}}},{"id":"b","function":{"name":"b","arguments":{}}}]},{"role":"tool","id":"b","content":"B"},{"role":"tool","id":"a","content":"A"}]}"#);
        // glm.jinja lines 99-105 use id when tool_call_id is absent.
        // Lines 225-248 then render results in the call order.
        assert_eq!(model.format_chat_input(&input, 8192).unwrap(), "[gMASK]<sop><|system|>Reasoning Effort: Max<|assistant|><think></think><tool_call>a</tool_call><tool_call>b</tool_call><|observation|><tool_response>A</tool_response><tool_response>B</tool_response><|assistant|><think>");
    }
    #[test]
    fn empty_assistant_history_round_trips_chat_and_responses() {
        let limits = crate::Limits::default();
        for thought in ["", "partial reasoning é"] {
            for followup in ["", r#",{"role":"user","content":"continue"}"#] {
                let mut expected = ChatInput { messages: vec![r#"{"role":"user","content":"go"}"#.into(),
                    format!(r#"{{"role":"assistant","content":"","reasoning_content":{thought:?}}}"#)],
                    tools: vec![], enable_thinking: true, thinking_off_requested: false, reasoning_effort: Default::default(), tool_choice: Default::default(), parallel_tool_calls: true };
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
                enable_thinking: true, thinking_off_requested: false,
                reasoning_effort: Default::default(),
                tool_choice: Default::default(), parallel_tool_calls: true,
            };
            assert!(format_chat(&input, 8192).is_err());
        }
    }
}
