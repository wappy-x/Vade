//! Expected markup follows the saved template, not model-generated output.
use spark_serve::chat_output::{Event, Limits, Parser};
use std::collections::{BTreeMap, BTreeSet};
fn parser(thinking: bool) -> Parser {
    let types = BTreeMap::from([("f".into(), BTreeSet::from(["count".into(),"items".into()]))]);
    Parser::new(Limits::default(), vec!["f".into()]).unwrap().glm(thinking, types)
}
fn collect(mut parser: Parser, text: &str, split: usize) -> Vec<Event> {
    let mut events = parser.push(&text.as_bytes()[..split]).unwrap();
    events.extend(parser.push(&text.as_bytes()[split..]).unwrap()); events.extend(parser.finish().unwrap());
    // Chunk boundaries can split text deltas. Compare their joined contents.
    let mut joined: Vec<Event> = Vec::new();
    for event in events {
        match (joined.last_mut(), &event) {
            (Some(Event::Content(a)), Event::Content(b)) | (Some(Event::Reasoning(a)), Event::Reasoning(b)) => a.push_str(b),
            _ => joined.push(event),
        }
    }
    joined
}
#[test]
fn template_reasoning_and_arguments_survive_every_byte_split() {
    let text = "雪<tool_call>f</tool_call></think>Answer<tool_call>f<arg_key>text</arg_key><arg_value>é\"\\\n</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>items</arg_key><arg_value>[true,null]</arg_value></tool_call>done";
    let args = "{\"text\":\"é\\\"\\\\\\n\",\"count\":2,\"items\":[true,null]}";
    let expected = vec![Event::Reasoning("雪<tool_call>f</tool_call>".into()), Event::Content("Answer".into()),
        Event::ToolStart {name:"f".into()}, Event::ToolArguments(args.into()),
        Event::ToolCall {name:"f".into(),arguments:args.into()}, Event::Content("done".into())];
    for split in 0..=text.len() { assert_eq!(collect(parser(true),text,split),expected,"split={split}"); }
}
#[test]
fn malformed_unknown_duplicate_and_partial_calls_stay_literal() {
    for text in ["<tool_call>other</tool_call>", "<tool_call>f<arg_key>x</arg_key><arg_value>a</arg_value><arg_key>x</arg_key><arg_value>b</arg_value></tool_call>",
        "<tool_call>f<arg_key>count</arg_key><arg_value>oops</arg_value></tool_call>",
        "<tool_call>f<arg_key>x</arg_key><arg_value>partial", "<tool_ca"] {
        for split in 0..=text.len() { assert_eq!(collect(parser(false),text,split),vec![Event::Content(text.into())]); }
    }
}
#[test]
fn no_arguments_and_single_call_policy() {
    let parser = parser(false).tool_policy("",false,true).unwrap();
    assert_eq!(collect(parser,"<tool_call>f</tool_call>ignored",0),vec![
        Event::ToolStart {name:"f".into()},Event::ToolArguments("{}".into()),Event::ToolCall {name:"f".into(),arguments:"{}".into()}]);
}
#[test]
fn limits_and_utf8_fail_without_emitting_a_partial_call() {
    let limits = Limits {tool_call_bytes:16,..Default::default()};
    let mut p = Parser::new(limits,vec!["f".into()]).unwrap().glm(false,Default::default());
    assert!(p.push(b"<tool_call>f<arg_key>verylong").is_err());
    assert!(p.finish().is_err());
    assert!(parser(false).push(&[0xff]).is_err());
}
#[test]
fn undeclared_tools_and_unclosed_reasoning_remain_text() {
    let p = Parser::new(Default::default(),vec![]).unwrap().glm(false,Default::default());
    assert_eq!(collect(p,"<tool_call>f</tool_call>",4),vec![Event::Content("<tool_call>f</tool_call>".into())]);
    assert_eq!(collect(parser(true),"unclosed</thi",3),vec![Event::Reasoning("unclosed</thi".into())]);
}
