//! DERIVED BY HAND from the stored templates and the text-only API adapter.
use super::*;
fn input(messages: &str) -> ChatInput {
    let body = format!(r#"{{"messages":{messages}}}"#);
    let crate::protocol::Prompt::Chat(input) = crate::protocol::parse(body.as_bytes(), true, &crate::Limits::default()).unwrap().prompt else { panic!() };
    input
}
#[test]
fn content_shapes_cross_the_service_adapter() {
    // Hybrid qwen3.6.jinja render_content joins text parts before trim.
    // GLM visible_text also joins them. Gemma captured_content trims each.
    // Dense Qwen, Phi and Llama receive the API's joined string. Their maker
    // templates do not define text arrays like the hybrid/Gemma templates do.
    for family in ["dense", "flash-next", "hybrid", "gemma", "phi", "llama", "glm"] {
        for (raw, joined, parts) in [
            (r#"" a  b ""#, " a  b ", None),
            (r#"[{"type":"text","text":" a "},{"type":"text","text":" b "}]"#, " a  b ", Some(vec![" a ".to_owned(), " b ".to_owned()])),
            ("[]", "", Some(vec![])),
        ] {
            let source = input(&format!(r#"[{{"role":"user","content":{raw}}}]"#));
            let messages = parse_messages_with_roles(&source, 8192, family == "gemma", family == "glm").unwrap();
            assert_eq!(messages[0].content, joined, "{family} {raw}");
            assert_eq!(messages[0].text_parts, if family == "gemma" { parts.clone() } else { None });
            let payload = if family == "gemma" && parts.is_some() { if joined.is_empty() { "" } else { "ab" } }
                else if matches!(family, "hybrid" | "flash-next" | "gemma" | "llama") { joined.trim() } else { joined };
            let options = chat::Options { enable_thinking: false, ..Default::default() };
            match family {
                "gemma" => assert_eq!(chat::gemma4::format(&messages, &options).unwrap(), format!("<bos><|turn>user\n{payload}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")),
                "phi" => assert_eq!(chat::format_phi(&messages, &options).unwrap(), format!("<|im_start|>user<|im_sep|>{payload}<|im_end|><|im_start|>assistant<|im_sep|>")),
                "glm" => assert_eq!(glm::format(&messages, &Default::default()).unwrap(), format!("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{payload}<|assistant|><think>")),
                "llama" => {}, // The shared conversion is checked here; frame goldens are in tokenizer.
                _ => assert_eq!(tests::codec(family != "dense").format_chat(&messages, &options).unwrap(), format!("<|im_start|>user\n{payload}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")),
            }
        }
        // Null assistant content becomes the empty string before rendering.
        let source = input(r#"[{"role":"user","content":"U"},{"role":"assistant","content":null}]"#);
        let messages = parse_messages_with_roles(&source, 8192, family == "gemma", family == "glm").unwrap();
        assert!(messages[1].content.is_empty());
        assert!(messages[1].text_parts.is_none());
    }
    println!("SERVICE CONTENT SHAPES: 28 cases");
}

#[test]
fn gemma_parts_keep_role_specific_rules() {
    let options = chat::Options { enable_thinking: false, ..Default::default() };
    // gemma4.jinja 202: every part in the first system message gets a space.
    // 333-336: assistant strip_thinking is applied separately to each part.
    // 295-301: tool results join raw parts without per-part trim.
    for (source, expected) in [
        (r#"[{"role":"user","content":"U"},{"role":"system","content":[{"type":"text","text":" a "},{"type":"text","text":" b "}]}]"#,
            "<bos><|turn>user\nU<turn|>\n<|turn>system\nab<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
        (r#"[{"role":"user","content":"U"},{"role":"assistant","content":[{"type":"text","text":" a <|channel>thought\nx<channel|> "},{"type":"text","text":" b "}]}]"#,
            "<bos><|turn>user\nU<turn|>\n<|turn>model\nab<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
        (r#"[{"role":"user","content":"U"},{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":{}}}]},{"role":"tool","tool_call_id":"c","content":[{"type":"text","text":" a "},{"type":"text","text":" b "}]}]"#,
            "<bos><|turn>user\nU<turn|>\n<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>response:f{value:<|\"|> a  b <|\"|>}<tool_response|>"),
        (r#"[{"role":"system","content":[{"type":"text","text":" "},{"type":"text","text":" b "}]},{"role":"user","content":[]}]"#,
            "<bos><|turn>system\n b <turn|>\n<|turn>user\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
    ] {
        let messages = parse_messages(&input(source), 8192, true).unwrap();
        assert_eq!(chat::gemma4::format(&messages, &options).unwrap(), expected);
    }
}

#[test]
fn gemma_parsed_information_separators_follow_maker_trim() {
    // DERIVED BY HAND: Python trim removes the information separator in
    // content, including the final trim in strip_thinking. A tool result's
    // raw joined body does not take either trim branch.
    let options = chat::Options { enable_thinking: false, ..Default::default() };
    for (source, expected) in [
        (r#"[{"role":"assistant","content":[{"type":"text","text":"\u001c"}]}]"#,
            "<bos><|turn>model\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
        (r#"[{"role":"system","content":"\u001dX\u001e"},{"role":"user","content":"\u001fU\u001c"}]"#,
            "<bos><|turn>system\nX<turn|>\n<|turn>user\nU<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
        (r#"[{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":{}}}]},{"role":"tool","tool_call_id":"c","content":[{"type":"text","text":"\u001c"}]}]"#,
            "<bos><|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>response:f{value:<|\"|>\u{1c}<|\"|>}<tool_response|>"),
    ] {
        let messages = parse_messages(&input(source), 8192, true).unwrap();
        assert_eq!(chat::gemma4::format(&messages, &options).unwrap(), expected);
    }
}

#[test]
fn glm_product_shape_goldens() {
    // serve/templates/glm.jinja: raw system/user text; empty thought for
    // assistant history; no turn terminator; new assistant generation prompt.
    let prefix = "[gMASK]<sop><|system|>Reasoning Effort: Max";
    let suffix = "<|assistant|><think>";
    for (source, body) in [
        (r#"[{"role":"user","content":"U"}]"#, "<|user|>U"),
        (r#"[{"role":"system","content":"A"},{"role":"user","content":"U"}]"#, "<|system|>A<|user|>U"),
        (r#"[{"role":"system","content":"A"},{"role":"system","content":"B"},{"role":"user","content":"U"}]"#, "<|system|>A<|system|>B<|user|>U"),
        (r#"[{"role":"user","content":"U"},{"role":"system","content":"B"}]"#, "<|user|>U<|system|>B"),
        (r#"[{"role":"user","content":"U"},{"role":"assistant","content":" P "}]"#, "<|user|>U<|assistant|><think></think>P"),
        (r#"[{"role":"user","content":"U"},{"role":"assistant","content":null}]"#, "<|user|>U<|assistant|><think></think>"),
        (r#"[{"role":"user","content":"U"},{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":{}}}]},{"role":"tool","tool_call_id":"c","content":"R"}]"#,
            "<|user|>U<|assistant|><think></think><tool_call>f</tool_call><|observation|><tool_response>R</tool_response>"),
        (r#"[{"role":"user","content":"雪é🚀\n\t\u0000\u0007Z"}]"#, "<|user|>雪é🚀\n\t\0\u{7}Z"),
        (r#"[{"role":"user","content":"<|user|><|system|><|assistant|><think></think>"}]"#, "<|user|><|user|><|system|><|assistant|><think></think>"),
    ] {
        let messages = parse_messages_with_roles(&input(source), 8192, false, true).unwrap();
        assert_eq!(glm::format(&messages, &Default::default()).unwrap(), format!("{prefix}{body}{suffix}"));
    }
}
