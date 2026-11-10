use super::*;

fn assemble(events: &[&str], limits: WireLimits) -> io::Result<AssistantTurn> {
    let mut stream = ChatAccumulator::new(limits);
    for event in events {
        stream.push_sse_data(event, &mut |_| Ok(()))?;
    }
    stream.into_turn()
}

#[test]
fn textless_responses_supply_content_once_and_preserve_reasoning_fields() {
    for content in ["", ",\"content\":null", ",\"content\":\"\""] {
        for calls in ["", ",\"tool_calls\":null", ",\"tool_calls\":[]"] {
            for field in ["reasoning_content", "reasoning"] {
                let raw = format!(
                    r#"{{"role":"assistant"{content}{calls},"{field}":" think\n🦀 ","provider":{{"opaque":true}}}}"#
                );
                let response = json::parse(&format!(
                    r#"{{"choices":[{{"message":{raw},"finish_reason":"length"}}]}}"#
                ))
                .unwrap();
                let json_turn =
                    AssistantTurn::from_response(&response, WireLimits::default()).unwrap();
                let event =
                    format!(r#"{{"choices":[{{"delta":{raw},"finish_reason":"length"}}]}}"#);
                let sse_turn = assemble(&[&event, "[DONE]"], WireLimits::default()).unwrap();
                for turn in [json_turn, sse_turn] {
                    assert_eq!(turn.finish_reason(), &FinishReason::Length);
                    let value = turn.message().value();
                    assert_eq!(value.get("content").and_then(Value::as_str), Some(""));
                    assert_eq!(
                        value.get(field).and_then(Value::as_str),
                        Some(" think\n🦀 ")
                    );
                    let mut expected = json::parse(&raw).unwrap();
                    let Value::Object(fields) = &mut expected else {
                        unreachable!()
                    };
                    if let Some((_, content)) =
                        fields.iter_mut().find(|(key, _)| key == "content")
                    {
                        *content = Value::String(String::new());
                    } else {
                        fields.push(("content".into(), Value::String(String::new())));
                    }
                    fields.retain(|(key, _)| key != "tool_calls");
                    assert_eq!(*value, expected);
                    let frozen =
                        Message::from_value(value.clone(), WireLimits::default()).unwrap();
                    assert_eq!(frozen.serialized_json(), turn.message().serialized_json());
                }
            }
        }
    }
}

#[test]
fn length_finish_retains_text_and_discards_incomplete_streamed_tools() {
    let turn = assemble(&[
        r#"{"choices":[{"delta":{"content":"partial 🦀","tool_calls":[{"index":3,"function":{"arguments":"{"}}]},"finish_reason":"length"}]}"#,
        "[DONE]",
    ], WireLimits::default()).unwrap();
    assert_eq!(
        turn.message()
            .value()
            .get("content")
            .and_then(Value::as_str),
        Some("partial 🦀")
    );
    assert!(turn.message().value().get("vade_truncated").is_none());
    assert!(turn.message().tool_calls().is_empty());
    assert!(turn.truncated_tool_call());
    let legacy = Message::from_value(
        json::parse(r#"{"role":"assistant","content":"partial 🦀","vade_truncated":true}"#)
            .unwrap(),
        WireLimits::default(),
    )
    .unwrap();
    assert_eq!(legacy.serialized_json(), turn.message().serialized_json());
}

#[test]
fn reasoning_fragments_and_empty_answers_are_valid_without_visible_deltas() {
    let mut stream = ChatAccumulator::new(WireLimits::default());
    for data in [
        r#"{"choices":[{"delta":{"role":"assistant","content":null}}]}"#,
        r#"{"choices":[{"delta":{"reasoning_content":"think "}}]}"#,
        r#"{"choices":[{"delta":{"reasoning_content":"more 🦀"},"finish_reason":"length"}]}"#,
        "[DONE]",
    ] {
        stream
            .push_sse_data(data, &mut |_| panic!("reasoning is not visible text"))
            .unwrap();
    }
    let turn = stream.into_turn().unwrap();
    assert_eq!(
        turn.message()
            .value()
            .get("reasoning_content")
            .and_then(Value::as_str),
        Some("think more 🦀")
    );
    assert_eq!(
        turn.message()
            .value()
            .get("content")
            .and_then(Value::as_str),
        Some("")
    );
    let empty = assemble(
        &[
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ],
        WireLimits::default(),
    )
    .unwrap();
    assert_eq!(
        empty
            .message()
            .value()
            .get("content")
            .and_then(Value::as_str),
        Some("")
    );
}

#[test]
fn supplied_content_stays_exact_and_normalized_content_counts_toward_limit() {
    for content in [r#""answer""#, r#"[{"type":"text","text":"answer"}]"#] {
        let message = json::parse(&format!(
            r#"{{"role":"assistant","content":{content},"reasoning_content":"think"}}"#
        ))
        .unwrap();
        let response = json::parse(&format!(
            r#"{{"choices":[{{"message":{},"finish_reason":"stop"}}]}}"#,
            json::stringify(&message).unwrap()
        ))
        .unwrap();
        let turn = AssistantTurn::from_response(&response, WireLimits::default()).unwrap();
        assert_eq!(turn.message().value(), &message);
    }
    let response = json::parse(
        r#"{"choices":[{"message":{"role":"assistant"},"finish_reason":"length"}]}"#,
    )
    .unwrap();
    assert!(
        AssistantTurn::from_response(
            &response,
            WireLimits {
                max_message_bytes: r#"{"role":"assistant"}"#.len(),
                ..WireLimits::default()
            }
        )
        .is_err()
    );
}

#[test]
fn interleaved_calls_assemble_by_index_and_keep_extensions() {
    let turn = assemble(&[
        r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":"Inspect ","provider":{"signature":"opaque"},"tool_calls":[{"index":1,"id":"call_","type":"function","function":{"name":"workspace_","arguments":"{\"path\":"},"provider_call":{"keep":true}},{"index":0,"id":"zero","type":"function","function":{"name":"workspace_list","arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{"reasoning_content":"files.","tool_calls":[{"index":0,"function":{"arguments":"th\":\".\"}"}},{"index":1,"id":"one","function":{"name":"read","arguments":"\"src/🦀.rs\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        r#"{"choices":[],"usage":{"prompt_tokens":42}}"#,
        "[DONE]",
    ], WireLimits::default()).unwrap();
    assert_eq!(turn.finish_reason(), &FinishReason::ToolCalls);
    let message = turn.message();
    assert_eq!(
        message.tool_calls(),
        &[
            ToolCall {
                id: "zero".into(),
                name: "workspace_list".into(),
                arguments: r#"{"path":"."}"#.into()
            },
            ToolCall {
                id: "call_one".into(),
                name: "workspace_read".into(),
                arguments: r#"{"path":"src/🦀.rs"}"#.into()
            },
        ]
    );
    assert_eq!(
        message
            .value()
            .get("reasoning_content")
            .and_then(Value::as_str),
        Some("Inspect files.")
    );
    assert_eq!(message.value().get("content"), Some(&Value::Null));
    let wire_calls = message
        .value()
        .get("tool_calls")
        .unwrap()
        .as_array()
        .unwrap();
    assert!(wire_calls[1].get("provider_call").is_some());
    assert!(wire_calls[1].get("index").is_none());
    assert_eq!(
        json::parse(&json::stringify(message.value()).unwrap()).unwrap(),
        *message.value()
    );
}

#[test]
fn complete_message_keeps_structured_content_and_future_fields() {
    let value = json::parse(r#"{"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"hi"},{"type":"future_part","payload":[1,2]}],"refusal":null,"audio":{"id":"opaque"},"annotations":[{"a":1}]},"finish_reason":"future_terminal"}]}"#).unwrap();
    let turn = AssistantTurn::from_response(&value, WireLimits::default()).unwrap();
    assert_eq!(
        turn.message().value(),
        value.get("choices").unwrap().as_array().unwrap()[0]
            .get("message")
            .unwrap()
    );
    assert_eq!(
        turn.finish_reason(),
        &FinishReason::Other("future_terminal".into())
    );
}

#[test]
fn truncated_missing_finish_and_upstream_errors_never_complete() {
    let partial = r#"{"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}"#;
    assert_eq!(
        assemble(&[partial], WireLimits::default())
            .unwrap_err()
            .kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert!(assemble(&[partial, "[DONE]"], WireLimits::default()).is_err());
    assert!(
        assemble(
            &[r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#],
            WireLimits::default()
        )
        .is_err()
    );
    let mut stream = ChatAccumulator::new(WireLimits::default());
    let error = stream
        .push_sse_data(
            r#"{"error":{"message":"private upstream details"}}"#,
            &mut |_| Ok(()),
        )
        .unwrap_err();
    assert!(!error.to_string().contains("private"));
    assert!(stream.push_sse_data("[DONE]", &mut |_| Ok(())).is_err());
    assert!(stream.into_turn().is_err());
}

#[test]
fn malformed_call_ids_indices_types_and_finish_are_rejected() {
    for event in [
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"gap","type":"function","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"same","type":"function","function":{"name":"read","arguments":"{}"}},{"index":1,"id":"same","type":"function","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x","type":"future","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x","type":"function","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"stop"}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":-1}]},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0.0}]},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":9999999999999999999999999999}]},"finish_reason":null}]}"#,
        r#"{"choices":[{"delta":{"function_call":{"name":"legacy"}},"finish_reason":"function_call"}]}"#,
        r#"{"choices":[{"index":1,"delta":{},"finish_reason":"stop"}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"stop"},{"delta":{},"finish_reason":"stop"}]}"#,
    ] {
        assert!(
            assemble(&[event, "[DONE]"], WireLimits::default()).is_err(),
            "accepted {event}"
        );
    }
}

#[test]
fn fragmented_aggregate_argument_and_event_limits_are_enforced() {
    let first = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"r","arguments":"12"}},{"index":1,"id":"b","type":"function","function":{"name":"r","arguments":"34"}}]},"finish_reason":null}]}"#;
    let last = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"5"}}]},"finish_reason":"tool_calls"}]}"#;
    let limits = WireLimits {
        max_argument_bytes: 4,
        ..WireLimits::default()
    };
    assert!(assemble(&[first, last, "[DONE]"], limits).is_err());
    assert!(
        assemble(
            &[first],
            WireLimits {
                max_calls: 1,
                ..WireLimits::default()
            }
        )
        .is_err()
    );
    let stop = r#"{"choices":[{"delta":{"content":"é🦀"},"finish_reason":"stop"}]}"#;
    assert!(
        assemble(
            &[stop, "[DONE]"],
            WireLimits {
                max_events: 1,
                ..WireLimits::default()
            }
        )
        .is_err()
    );
    assert!(
        assemble(
            &[stop],
            WireLimits {
                max_stream_bytes: stop.len() - 1,
                ..WireLimits::default()
            }
        )
        .is_err()
    );
    assert!(
        assemble(
            &[stop, "[DONE]"],
            WireLimits {
                max_message_bytes: 5,
                ..WireLimits::default()
            }
        )
        .is_err()
    );
}

#[test]
fn unknown_delta_fields_are_preserved_or_fail_explicitly() {
    let start = r#"{"choices":[{"delta":{"content":"hi","future":{"a":1},"tool_calls":null},"finish_reason":null}]}"#;
    let end = r#"{"choices":[{"delta":{"future":{"b":2}},"finish_reason":"stop"}]}"#;
    let turn = assemble(&[start, end, "[DONE]"], WireLimits::default()).unwrap();
    assert_eq!(
        turn.message().value().get("future"),
        Some(&json::parse(r#"{"a":1,"b":2}"#).unwrap())
    );
    assert_eq!(turn.message().value().get("tool_calls"), Some(&Value::Null));
    let conflict = r#"{"choices":[{"delta":{"future":{"a":2}},"finish_reason":"stop"}]}"#;
    assert!(assemble(&[start, conflict, "[DONE]"], WireLimits::default()).is_err());
}

#[test]
fn callback_failure_poison_and_post_finish_data_are_rejected() {
    let mut stream = ChatAccumulator::new(WireLimits::default());
    assert!(
        stream
            .push_sse_data(
                r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}"#,
                &mut |_| Err(io::ErrorKind::Interrupted.into())
            )
            .is_err()
    );
    assert!(!stream.is_done());
    assert!(stream.into_turn().is_err());
    let end = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
    assert!(assemble(&[end, end, "[DONE]"], WireLimits::default()).is_err());
    assert!(assemble(&[end, "[DONE]", "[DONE]"], WireLimits::default()).is_err());
}

#[test]
fn argument_syntax_is_deferred_until_execution_and_finish_kinds_are_distinct() {
    let raw = json::parse(r#"{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"not JSON"}}]}"#).unwrap();
    let message = Message::from_value(raw, WireLimits::default()).unwrap();
    assert!(AssistantTurn::new(message.clone(), FinishReason::ToolCalls).is_ok());
    assert!(AssistantTurn::new(message.clone(), FinishReason::Length).is_ok());
    assert!(AssistantTurn::new(message, FinishReason::Stop).is_err());
    assert_ne!(FinishReason::Length, FinishReason::ContentFilter);
}
