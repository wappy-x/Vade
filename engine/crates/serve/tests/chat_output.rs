use spark_serve::chat_output::{Event, Limits, Parser};

fn parser() -> Parser {
    Parser::new(Limits::default(), vec!["search".into(), "lookup".into()]).unwrap()
}
fn normalize(events: Vec<Event>) -> Vec<Event> {
    let mut result = Vec::new();
    let mut arguments = String::new();
    for event in events {
        match (result.last_mut(), event) {
            (_, Event::ToolStart { .. }) => {}
            (_, Event::ToolArguments(s)) => arguments.push_str(&s),
            (_, event @ Event::ToolCall { .. }) => { if let Event::ToolCall { arguments: expected, .. } = &event { assert_eq!(&arguments, expected); } arguments.clear(); result.push(event); }
            (Some(Event::Content(a)), Event::Content(b)) => a.push_str(&b),
            (Some(Event::Reasoning(a)), Event::Reasoning(b)) => a.push_str(&b),
            (_, event) => result.push(event),
        }
    }
    result
}
fn parse_chunks(input: &[u8], sizes: &[usize]) -> Result<Vec<Event>, String> {
    let mut parser = parser();
    let mut result = Vec::new();
    let mut pos = 0;
    for &size in sizes {
        let end = (pos + size).min(input.len());
        result.extend(parser.push(&input[pos..end])?);
        pos = end;
    }
    result.extend(parser.push(&input[pos..])?);
    result.extend(parser.finish()?);
    Ok(normalize(result))
}
#[test]
fn all_byte_boundaries_preserve_utf8_json_strings_and_argument_order() {
    let text = " \n<think>étude 🚀 <tool_call>is only reasoning</think>\nAnswer <tool_call>\n{\"arguments\": { \"z\": [\"</tool_call>\", \"<think>\", \"quote: \\\" \\\\\"], \"a\": \"é🚀\" }, \"name\":\"search\"}\n</tool_call> then <tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>.";
    let expected = parse_chunks(text.as_bytes(), &[]).unwrap();
    assert_eq!(expected, vec![
        Event::Content(" \n".into()),
        Event::Reasoning("étude 🚀 <tool_call>is only reasoning".into()),
        Event::Content("\nAnswer ".into()),
        Event::ToolCall { name: "search".into(), arguments: "{ \"z\": [\"</tool_call>\", \"<think>\", \"quote: \\\" \\\\\"], \"a\": \"é🚀\" }".into() },
        Event::Content(" then ".into()),
        Event::ToolCall { name: "lookup".into(), arguments: "{}".into() },
        Event::Content(".".into()),
    ]);
    for boundary in 0..=text.len() {
        assert_eq!(
            parse_chunks(text.as_bytes(), &[boundary]).unwrap(),
            expected,
            "boundary {boundary}"
        );
    }
    assert_eq!(
        parse_chunks(text.as_bytes(), &vec![1; text.len()]).unwrap(),
        expected
    );
}
#[test]
fn name_and_arguments_stream_before_commit() {
    let mut p = parser();
    assert!(
        p.push(br#"<tool_call>{"name":"search","arguments":{"x":1}}</tool_call"#)
            .unwrap()
            == vec![Event::ToolStart { name: "search".into() }, Event::ToolArguments("{\"x\":1}".into())]
    );
    assert_eq!(
        p.push(b">").unwrap(),
        vec![Event::ToolCall {
            name: "search".into(),
            arguments: "{\"x\":1}".into()
        }]
    );
    assert!(p.finish().unwrap().is_empty());
}
#[test]
fn disabled_tools_are_literal_and_later_think_markers_are_content() {
    let text = "<tool_call>{not even JSON}</tool_call> <think>literal</think>";
    let mut p = Parser::new(Limits::default(), vec![]).unwrap();
    let mut events = Vec::new();
    for byte in text.bytes() {
        events.extend(p.push(&[byte]).unwrap());
    }
    events.extend(p.finish().unwrap());
    assert_eq!(normalize(events), vec![Event::Content(text.into())]);
}
#[test]
fn length_stop_inside_reasoning_is_valid_and_preserves_partial_marker() {
    for text in ["<think>unfinished", "<think>unfinished </thi", "<think>"] {
        let events = parse_chunks(text.as_bytes(), &vec![1; text.len()]).unwrap();
        let expected = text.strip_prefix("<think>").unwrap();
        assert_eq!(
            events,
            if expected.is_empty() {
                vec![]
            } else {
                vec![Event::Reasoning(expected.into())]
            }
        );
    }
}
#[test]
fn incomplete_tools_fallback_before_start_and_complete_after_start() {
    for text in [
        "<tool_call>",
        "<tool_call>{",
        "<tool_call>{\"name\":\"search\",\"arguments\":{}}",
        "<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_",
    ] {
        let mut p = parser();
        if !p.push(text.as_bytes()).unwrap().is_empty() { assert!(p.finish().unwrap().iter().any(|e| matches!(e, Event::ToolCall { .. }))); continue; }
        assert_eq!(p.finish().unwrap(), vec![Event::Content(text.into())]);
        assert!(p.push(b"anything").is_err());
        assert!(p.finish().is_err());
    }
}
#[test]
fn malformed_started_calls_complete_and_undeclared_calls_stay_literal() {
    for text in [
        r#"<tool_call>{"name":"undeclared","arguments":{}}</tool_call>"#,
        r#"<tool_call>{"name":"search","arguments":"{}"}</tool_call>"#,
        r#"<tool_call>{"name":"search","arguments":[]}</tool_call>"#,
        r#"<tool_call>{"name":"search","arguments":{},"extra":1}</tool_call>"#,
        r#"<tool_call>{"name":"search","name":"lookup","arguments":{}}</tool_call>"#,
        r#"<tool_call>{"name":"search","arguments":{"x":NaN}}</tool_call>"#,
        r#"<tool_call>{"name":"search","arguments":{}} garbage </tool_call>"#,
        r#"<tool_call>{"name":"search","arguments":{}}</wrong>"#,
        r#"<tool_call><tool_call>{"name":"search","arguments":{}}</tool_call>"#,
    ] {
        let mut p = parser();
        let mut events = Vec::new();
        let mut failed = false;
        for &byte in text.as_bytes() {
            match p.push(&[byte]) { Ok(chunk) => events.extend(chunk), Err(_) => { failed = true; break; } }
        }
        if failed { assert!(p.finish().is_err()); continue; }
        events.extend(p.finish().unwrap());
        if events.iter().any(|e| matches!(e, Event::ToolStart { .. })) { assert!(normalize(events).iter().any(|e| matches!(e, Event::ToolCall { .. }))); continue; }
        assert!(
            !events.iter().any(|e| matches!(e, Event::ToolCall { .. })),
            "{text}"
        );
        assert_eq!(normalize(events), vec![Event::Content(text.into())]);
    }
}
#[test]
fn incomplete_openers_remain_literal_and_cannot_execute() {
    for text in ["<", "<tool_", "<think", "<<tool_"] {
        assert_eq!(
            parse_chunks(text.as_bytes(), &[1, 2]).unwrap(),
            vec![Event::Content(text.into())]
        );
    }
}
#[test]
fn strict_utf8_and_single_finish() {
    let mut p = parser();
    assert!(p.push(&[0xf0, 0x9f]).unwrap().is_empty());
    assert!(p.finish().is_err());
    let mut p = parser();
    assert!(p.push(&[0xc0, 0x80]).is_err());
    assert!(p.push(b"ok").is_err());
    let mut p = parser();
    p.finish().unwrap();
    assert!(p.finish().is_err());
    assert!(p.push(b"").is_err());
}
#[test]
fn limits_bound_total_buffered_call_count_and_declared_names() {
    let mut p = Parser::new(
        Limits {
            total_bytes: 8,
            tool_call_bytes: 8,
            ..Limits::default()
        },
        vec![],
    )
    .unwrap();
    p.push(b"12345678").unwrap();
    assert!(p.push(b"9").is_err());
    let mut p = Parser::new(
        Limits {
            tool_call_bytes: 8,
            ..Limits::default()
        },
        vec!["search".into()],
    )
    .unwrap();
    assert_eq!(p.push(b"<tool_call>         ").unwrap(), vec![Event::Content("<tool_call>         ".into())]);
    let mut p = Parser::new(
        Limits {
            tool_calls: 1,
            ..Limits::default()
        },
        vec!["search".into()],
    )
    .unwrap();
    let call = br#"<tool_call>{"name":"search","arguments":{}}</tool_call>"#;
    assert_eq!(p.push(call).unwrap().len(), 3);
    assert_eq!(p.push(call).unwrap(), vec![Event::Content(std::str::from_utf8(call).unwrap().into())]);
    assert!(
        Parser::new(
            Limits {
                name_bytes: 3,
                ..Limits::default()
            },
            vec!["long".into()]
        )
        .is_err()
    );
    assert!(Parser::new(Limits::default(), vec!["same".into(), "same".into()]).is_err());
}
#[test]
fn unicode_escaped_argument_key_and_name_are_valid_without_rewriting_arguments() {
    let text = r#"<tool_call>{"na\u006de":"se\u0061rch","argu\u006dents":{ "é": "\u0061", "z":1e0 }}</tool_call>"#;
    assert_eq!(
        parse_chunks(text.as_bytes(), &[5, 9, 3]).unwrap(),
        vec![Event::ToolCall {
            name: "search".into(),
            arguments: "{ \"é\": \"\\u0061\", \"z\":1e0 }".into()
        }]
    );
}

#[test]
fn started_calls_preserve_arguments_beyond_json_validation_limits() {
    let depth_bomb = format!(
        "<tool_call>{{\"name\":\"search\",\"arguments\":{{\"x\":{}0{}}}}}</tool_call>",
        "[".repeat(25),
        "]".repeat(25)
    );
    assert!(parse_chunks(depth_bomb.as_bytes(), &[]).is_ok());
    let node_bomb = format!(
        "<tool_call>{{\"name\":\"search\",\"arguments\":{{\"x\":[{}]}}}}</tool_call>",
        vec!["0"; 8200].join(",")
    );
    assert!(parse_chunks(node_bomb.as_bytes(), &[]).is_ok());
}

#[test]
fn gemma_channels_and_calls_at_every_byte_boundary() {
    let input = "<|channel>thought\nétude <|tool_call>literal<channel|><|tool_call>call:search{query:<|\"|>a\n\"\\🚀 <tool_call|><|\"|>,nested:{x:[true,null,2]}}<tool_call|><|tool_call>call:lookup{}<tool_call|>done";
    let expected = vec![Event::Reasoning("étude <|tool_call>literal".into()), Event::ToolCall { name: "search".into(), arguments: r#"{"query":"a\n\"\\🚀 <tool_call|>","nested":{"x":[true,null,2]}}"#.into() }, Event::ToolCall { name: "lookup".into(), arguments: "{}".into() }, Event::Content("done".into())];
    for width in 1..=input.len() {
        let mut p = parser().gemma4(false);
        let mut events = Vec::new();
        for chunk in input.as_bytes().chunks(width) { events.extend(p.push(chunk).unwrap()); }
        events.extend(p.finish().unwrap());
        assert_eq!(normalize(events), expected, "chunk width {width}");
    }
    let mut p = Parser::new(Limits::default(), vec![]).unwrap().gemma4(false);
    assert_eq!(p.push(b"<|tool_call>call:unknown{}<tool_call|>").unwrap(), vec![Event::Content("<|tool_call>call:unknown{}<tool_call|>".into())]);
    let mut p = parser().gemma4(true);
    let mut events = p.push(b"continuation<channel|>answer").unwrap();
    events.extend(p.finish().unwrap());
    assert_eq!(normalize(events), vec![Event::Reasoning("continuation".into()), Event::Content("answer".into())]);
    let mut p = parser().gemma4(false);
    assert_eq!(p.push(b"<|tool_call>call:search{x:").unwrap(), vec![Event::ToolStart { name: "search".into() }, Event::ToolArguments("{\"x\":".into())]);
    assert_eq!(p.finish().unwrap(), vec![Event::ToolCall { name: "search".into(), arguments: "{\"x\":".into() }]);
}
