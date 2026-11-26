//! Host checks for output accounting and terminal error frames.
use crate::{Limits, anthropic, responses, http, chat_output::Event, generation::Usage};
use engine_model::json::{self, Value};

fn limits() -> Limits { Limits { output_bytes: 1032, ..Limits::default() } }

fn cap(result: Result<String, String>) {
    assert_eq!(result.unwrap_err(), "output byte limit exceeded");
}

#[test]
fn completion_counts_content_and_reasoning_bytes_together() {
    let mut out = http::Completion::default();
    let mut call = 0;
    out.push(Event::Reasoning("éé".into()), 1, &mut call, &limits()).unwrap();
    out.push(Event::Content("éé".into()), 1, &mut call, &limits()).unwrap();
    assert_eq!(out.push(Event::Content("x".into()), 1, &mut call, &limits()).unwrap_err(), "output byte limit exceeded");
}

fn responses_text(reasoning: bool) {
    let mut out = responses::Output::default();
    let mut call = 0;
    let event = |s: &str| if reasoning { Event::Reasoning(s.into()) } else { Event::Content(s.into()) };
    out.push(event("éé"), 1, &mut call, &limits()).unwrap();
    out.push(event("éé"), 1, &mut call, &limits()).unwrap();
    cap(out.push(event("x"), 1, &mut call, &limits()));
}
#[test] fn responses_caps_content_bytes() { responses_text(false); }
#[test] fn responses_caps_reasoning_bytes() { responses_text(true); }
#[test]
fn responses_caps_tool_name_bytes() {
    let mut out = responses::Output::default();
    out.push(Event::ToolStart { name: "abcdefgh".into() }, 1, &mut 0, &limits()).unwrap();
    cap(out.push(Event::ToolStart { name: "x".into() }, 1, &mut 0, &limits()));
}
#[test]
fn responses_caps_tool_argument_bytes() {
    let mut out = responses::Output::default();
    out.push(Event::ToolStart { name: "f".into() }, 1, &mut 0, &limits()).unwrap();
    out.push(Event::ToolArguments("{\"a\":1}".into()), 1, &mut 0, &limits()).unwrap();
    cap(out.push(Event::ToolArguments(" ".into()), 1, &mut 0, &limits()));
}

fn anthropic_text(reasoning: bool) {
    let mut out = anthropic::Output::default();
    let event = |s: &str| if reasoning { Event::Reasoning(s.into()) } else { Event::Content(s.into()) };
    out.push(event("éé"), 1, &mut 0, &limits()).unwrap();
    out.push(event("éé"), 1, &mut 0, &limits()).unwrap();
    cap(out.push(event("x"), 1, &mut 0, &limits()));
}
#[test] fn anthropic_caps_content_bytes() { anthropic_text(false); }
#[test] fn anthropic_caps_reasoning_bytes() { anthropic_text(true); }
#[test]
fn anthropic_caps_tool_name_bytes() {
    let mut out = anthropic::Output::default();
    out.push(Event::ToolStart { name: "abcdefgh".into() }, 1, &mut 0, &limits()).unwrap();
    cap(out.push(Event::ToolStart { name: "x".into() }, 1, &mut 0, &limits()));
}
#[test]
fn anthropic_caps_tool_argument_bytes() {
    let mut out = anthropic::Output::default();
    out.push(Event::ToolStart { name: "f".into() }, 1, &mut 0, &limits()).unwrap();
    out.push(Event::ToolArguments("{\"a\":1}".into()), 1, &mut 0, &limits()).unwrap();
    cap(out.push(Event::ToolArguments(" ".into()), 1, &mut 0, &limits()));
}

fn frame(frame: &str, event: &str) -> Value {
    assert!(frame.starts_with(&format!("event: {event}\n")), "{frame}");
    assert!(frame.ends_with("\n\n"), "{frame}");
    let data = frame.lines().find_map(|line| line.strip_prefix("data: ")).unwrap();
    json::parse(data.as_bytes(), json::Limits::default()).unwrap()
}

#[test]
fn responses_failure_preserves_partial_output_and_error() {
    let mut out = responses::Output::default();
    out.push(Event::Content("partial".into()), 7, &mut 0, &Limits::default()).unwrap();
    let wire = out.failed("test", 7, 0, Usage::default(), "server_error", "step \"failed\"\n");
    let frames: Vec<_> = wire.split_inclusive("\n\n").collect();
    let value = frame(frames.last().unwrap(), "response.failed");
    let response = value.object().unwrap()["response"].object().unwrap();
    assert_eq!(response["status"].string().unwrap(), "failed");
    let error = response["error"].object().unwrap();
    assert_eq!(error["code"].string().unwrap(), "server_error");
    assert_eq!(error["message"].string().unwrap(), "step \"failed\"\n");
    let item = response["output"].array().unwrap()[0].object().unwrap();
    assert_eq!(item["status"].string().unwrap(), "incomplete");
    let done = frame(frames[frames.len()-2], "response.output_item.done");
    assert_eq!(done.object().unwrap()["item"].object().unwrap(), item);
    let content = item["content"].array().unwrap()[0].object().unwrap();
    assert_eq!(content["text"].string().unwrap(), "partial");
    assert!(!wire.contains("event: response.completed"));
}

#[test]
fn anthropic_failure_is_terminal_error_frame() {
    let mut out = anthropic::Output::default();
    let wire = out.failed("api_error", "step \"failed\"\n");
    let value = frame(&wire, "error");
    let object = value.object().unwrap();
    assert_eq!(object["type"].string().unwrap(), "error");
    let error = object["error"].object().unwrap();
    assert_eq!(error["type"].string().unwrap(), "api_error");
    assert_eq!(error["message"].string().unwrap(), "step \"failed\"\n");
    assert!(!wire.contains("event: message_stop"));
}
