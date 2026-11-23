//! The whole text path on the public tokenizer files of the checkpoint: prompt token
//! IDs equal the reference lists, and the reference replies decode to their text.
//! These tests run no model and no GPU. See fixtures/inkling-chat/README.md.
use spark_serve::{Tokenizer, chat_output::{Event, Limits, Parser}, tokenizer::ModelTokenizer};
use engine_model::json;

const PROMPTS: [&str; 2] = [include_str!("fixtures/inkling-chat/reference-prompts.jsonl"), include_str!("fixtures/inkling-chat/reference-cases.jsonl")];
/// The request reader of this engine refuses these cases and the template takes them: see the rule list in src/tokenizer/inkling_tests.rs.
const REFUSED_BY_RULE: [&str; 2] = ["effort-word-padded", "tools-bare"];
const REPLIES: &str = include_str!("fixtures/inkling-chat/reference-replies.jsonl");
const TOOL_REPLIES: &str = include_str!("fixtures/inkling-chat/reference-tool-replies.jsonl");

fn checkpoint() -> ModelTokenizer {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.local/inkling-chat");
    assert_eq!(std::fs::read_to_string(root.join("chat_template.jinja")).expect("save the public tokenizer files in .local/inkling-chat"),
        spark_serve::tokenizer::inkling::TEMPLATE);
    ModelTokenizer::from_directory(root).unwrap()
}
fn lines(text: &str) -> impl Iterator<Item = (&str, json::Value)> {
    text.lines().filter(|line| !line.trim().is_empty()).map(|line|
        (line, json::parse(line.as_bytes(), json::Limits { bytes: 1 << 22, string_bytes: 1 << 22, ..Default::default() }).unwrap()))
}
fn ids(value: &json::Value) -> Vec<u32> { value.array().unwrap().iter().map(|id| id.u64().unwrap() as u32).collect() }

#[test]
#[ignore = "needs the public tokenizer files in .local/inkling-chat"]
fn prompt_token_ids_equal_the_reference() {
    let model = checkpoint();
    let limits = spark_serve::Limits { thinking_soft_switch: false, body_bytes: 1 << 22, ..Default::default() };
    let (mut equal, mut by_rule, mut differ) = (0, Vec::new(), Vec::new());
    for (line, root) in PROMPTS.iter().flat_map(|file| lines(file)) {
        let case = root.object().unwrap();
        let Some(expected) = case.get("ids") else { continue };
        // The request is the last key of a fixture line. Its text keeps the member order of nested objects.
        let start = line.find("\"request\":").unwrap() + "\"request\":".len();
        let request = match spark_serve::protocol::parse(&line.as_bytes()[start..line.len() - 1], true, &limits) {
            Ok(request) => request,
            Err(_) if REFUSED_BY_RULE.contains(&case["name"].string().unwrap()) => { by_rule.push(case["name"].string().unwrap().to_owned()); continue; }
            Err(error) => { differ.push(format!("{}: {error}", case["name"].string().unwrap())); continue; }
        };
        let spark_serve::protocol::Prompt::Chat(input) = request.prompt else { panic!() };
        match model.encode_chat(&input, 1 << 22) {
            Ok(tokens) if tokens == ids(expected) => equal += 1,
            Ok(_) => differ.push(case["name"].string().unwrap().to_owned()),
            Err(_) if REFUSED_BY_RULE.contains(&case["name"].string().unwrap()) => by_rule.push(case["name"].string().unwrap().to_owned()),
            Err(error) => differ.push(format!("{}: {error}", case["name"].string().unwrap())),
        }
    }
    println!("inkling_prompt_ids equal={equal} refused_by_rule={} differ={}", by_rule.len(), differ.len());
    assert!(differ.is_empty(), "{differ:?}");
    assert_eq!(by_rule, REFUSED_BY_RULE);
}
#[test]
#[ignore = "needs the public tokenizer files in .local/inkling-chat"]
fn reference_replies_decode_to_their_reasoning_and_answer() {
    let model = checkpoint();
    let mut replies = 0;
    for (_, root) in lines(REPLIES) {
        let reply = root.object().unwrap();
        let mut decoder = model.chat_decoder();
        let mut parser = Parser::new(Limits::default(), Vec::new()).unwrap().token_channels();
        let mut events = Vec::new();
        for token in ids(&reply["ids"]) { events.extend(parser.push(decoder.push(token).unwrap().as_bytes()).unwrap()); }
        events.extend(parser.push(decoder.finish().as_bytes()).unwrap());
        events.extend(parser.finish().unwrap());
        let (mut reasoning, mut content) = (String::new(), String::new());
        for event in events { match event { Event::Reasoning(text) => reasoning.push_str(&text), Event::Content(text) => content.push_str(&text), other => panic!("{other:?}") } }
        assert_eq!((reasoning.as_str(), content.as_str()), (reply["reasoning"].string().unwrap(), reply["content"].string().unwrap()), "{}", reply["name"].string().unwrap());
        replies += 1;
    }
    println!("inkling_reply_text replies={replies}");
}
#[test]
#[ignore = "needs the public tokenizer files in .local/inkling-chat"]
fn reference_tool_replies_decode_to_their_calls() {
    let model = checkpoint();
    let (mut replies, mut calls) = (0, 0);
    for (_, root) in lines(TOOL_REPLIES) {
        let reply = root.object().unwrap();
        let tools = reply["tools"].array().unwrap().iter().map(|tool| tool.string().unwrap().to_owned()).collect();
        let mut decoder = model.chat_decoder();
        let mut parser = Parser::new(Limits::default(), tools).unwrap().token_channels();
        let mut events = Vec::new();
        for token in ids(&reply["ids"]) { events.extend(parser.push(decoder.push(token).unwrap().as_bytes()).unwrap()); }
        events.extend(parser.push(decoder.finish().as_bytes()).unwrap());
        events.extend(parser.finish().unwrap());
        // The same list form as the fixture: text deltas of one kind joined, and one entry for each whole call.
        let mut got: Vec<Vec<String>> = Vec::new();
        let text = |kind: &str, delta: String, got: &mut Vec<Vec<String>>| match got.last_mut() {
            Some(last) if last[0] == kind => last[1].push_str(&delta),
            _ => got.push(vec![kind.to_owned(), delta]),
        };
        for event in events {
            match event {
                Event::Reasoning(delta) => text("reasoning", delta, &mut got),
                Event::Content(delta) => text("content", delta, &mut got),
                Event::ToolCall { name, arguments } => { got.push(vec!["call".into(), name, arguments]); calls += 1; }
                Event::ToolStart { .. } | Event::ToolArguments(_) => {}
            }
        }
        let expected: Vec<Vec<String>> = reply["events"].array().unwrap().iter()
            .map(|event| event.array().unwrap().iter().map(|field| field.string().unwrap().to_owned()).collect()).collect();
        assert_eq!(got, expected, "{}", reply["name"].string().unwrap());
        replies += 1;
    }
    println!("inkling_tool_reply_text replies={replies} calls={calls}");
}
