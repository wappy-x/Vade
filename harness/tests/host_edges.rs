//! Host-only error boundaries. These tests use no sockets, processes, or writable paths.
use std::io;
use vade_harness::{
    agent_wire::{AssistantTurn, ChatAccumulator, FinishReason, Message, WireLimits, MAX_CALL_ID_BYTES, MAX_TOOL_NAME_BYTES},
    json::{self, Value},
    prompt::AppendOnlyRequest,
};

const MESSAGES: &str = include_str!("fixtures/host_edges/messages.jsonl");

fn object(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(key, value)| (key.into(), value)).collect())
}

fn call_message(id: &str, name: &str) -> Value {
    object(vec![
        ("role", Value::String("assistant".into())),
        ("tool_calls", Value::Array(vec![object(vec![
            ("id", Value::String(id.into())), ("type", Value::String("function".into())),
            ("function", object(vec![("name", Value::String(name.into())), ("arguments", Value::String("{}".into()))])),
        ])])),
    ])
}

#[test]
fn incoming_message_schema_errors_are_rejected_before_history() {
    for (line, input) in include_str!("fixtures/host_edges/invalid-messages.jsonl").lines().enumerate() {
        let error = Message::from_value(json::parse(input).unwrap(), WireLimits::default()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "fixture line {}", line + 1);
    }
}

#[test]
fn tool_identifiers_use_utf8_byte_limits_and_reject_control_characters() {
    for (id, name, valid) in [
        ("i".repeat(MAX_CALL_ID_BYTES), "n".repeat(MAX_TOOL_NAME_BYTES), true),
        ("🦀".repeat(MAX_CALL_ID_BYTES / 4), "é".repeat(MAX_TOOL_NAME_BYTES / 2), true),
        (format!("{}x", "🦀".repeat(MAX_CALL_ID_BYTES / 4)), "read".into(), false),
        ("id".into(), format!("{}x", "é".repeat(MAX_TOOL_NAME_BYTES / 2)), false),
        ("id\u{85}".into(), "read".into(), false),
        ("id".into(), "read\u{7f}".into(), false),
    ] {
        let result = Message::from_value(call_message(&id, &name), WireLimits::default());
        assert_eq!(result.is_ok(), valid, "id bytes {}, name bytes {}", id.len(), name.len());
        let result = Message::tool_result(&id, "result".into(), WireLimits::default());
        assert_eq!(result.is_ok(), !id.is_empty() && id.len() <= MAX_CALL_ID_BYTES && !id.chars().any(char::is_control));
    }
}

#[test]
fn message_complexity_limits_reject_deep_and_wide_provider_fields() {
    for (depth, valid) in [(47, true), (48, false)] {
        let mut nested = Value::Null;
        for _ in 0..depth { nested = Value::Array(vec![nested]); }
        let value = object(vec![("role", Value::String("assistant".into())), ("provider", nested)]);
        assert_eq!(Message::from_value(value, WireLimits::default()).is_ok(), valid, "depth {depth}");
    }
    // Root, role, and provider array consume 3 nodes before their elements.
    for (elements, valid) in [(32_765, true), (32_766, false)] {
        let value = object(vec![("role", Value::String("assistant".into())), ("provider", Value::Array(vec![Value::Null; elements]))]);
        assert_eq!(Message::from_value(value, WireLimits::default()).is_ok(), valid, "elements {elements}");
    }
}

#[test]
fn fallback_response_schema_requires_a_terminal_finish_reason() {
    for (line, input) in include_str!("fixtures/host_edges/invalid-responses.jsonl").lines().enumerate() {
        let result = AssistantTurn::from_response(&json::parse(input).unwrap(), WireLimits::default());
        assert!(result.is_err(), "accepted fixture line {}: {input}", line + 1);
    }
    let future = json::parse(r#"{"choices":[{"message":{"role":"assistant","content":"text"},"finish_reason":"future_terminal"}]}"#).unwrap();
    let turn = AssistantTurn::from_response(&future, WireLimits::default()).unwrap();
    assert_eq!(turn.finish_reason(), &FinishReason::Other("future_terminal".into()));
}

#[test]
fn fallback_response_size_limit_includes_metadata_and_escaped_bytes() {
    let response = json::parse(r#"{"choices":[{"message":{"role":"assistant","content":"a\n🦀"},"finish_reason":"stop"}],"provider":"\u0000"}"#).unwrap();
    let exact = json::stringify(&response).unwrap().len();
    for (limit, valid) in [(exact - 1, false), (exact, true), (exact + 1, true)] {
        let limits = WireLimits { max_stream_bytes: limit, ..WireLimits::default() };
        assert_eq!(AssistantTurn::from_response(&response, limits).is_ok(), valid, "limit {limit}");
    }
}

#[test]
fn error_after_finish_and_usage_poison_the_stream_without_disclosing_details() {
    let events: Vec<_> = include_str!("fixtures/host_edges/late-error.jsonl").lines().collect();
    let mut stream = ChatAccumulator::new(WireLimits::default());
    let mut shown = String::new();
    for event in &events[..2] { stream.push_sse_data(event, &mut |s| { shown.push_str(s); Ok(()) }).unwrap(); }
    let error = stream.push_sse_data(events[2], &mut |_| panic!("error emitted content")).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(!error.to_string().contains("PRIVATE_FIXTURE_DETAIL"));
    assert_eq!(shown, "preview");
    assert!(!stream.is_done());
    assert!(stream.push_sse_data(events[3], &mut |_| Ok(())).is_err());
    assert!(stream.into_turn().is_err());
}

#[test]
fn streamed_identifier_fragments_enforce_aggregate_byte_limits() {
    for (field, limit) in [("id", MAX_CALL_ID_BYTES), ("name", MAX_TOOL_NAME_BYTES)] {
        for excess in [0, 1] {
            let mut stream = ChatAccumulator::new(WireLimits::default());
            let initial = if field == "id" {
                format!(r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":{},"type":"function","function":{{"name":"read","arguments":"{{}}"}}}}]}}}}]}}"#, json::quote(&"é".repeat((limit - 2) / 2)))
            } else {
                format!(r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"call","type":"function","function":{{"name":{},"arguments":"{{}}"}}}}]}}}}]}}"#, json::quote(&"é".repeat((limit - 2) / 2)))
            };
            stream.push_sse_data(&initial, &mut |_| Ok(())).unwrap();
            let fragment = json::quote(&format!("é{}", "x".repeat(excess)));
            let part = if field == "id" { format!(r#""id":{fragment}"#) } else { format!(r#""function":{{"name":{fragment}}}"#) };
            let last = format!(r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":0,{part}}}]}},"finish_reason":"tool_calls"}}]}}"#);
            let result = stream.push_sse_data(&last, &mut |_| Ok(()));
            assert_eq!(result.is_ok(), excess == 0, "{field}, excess {excess}");
            if excess == 0 {
                stream.push_sse_data("[DONE]", &mut |_| Ok(())).unwrap();
                let turn = stream.into_turn().unwrap();
                let call = &turn.message().tool_calls()[0];
                assert_eq!(if field == "id" { call.id.len() } else { call.name.len() }, limit);
            } else {
                assert!(stream.into_turn().is_err());
            }
        }
    }
}

#[test]
fn append_request_size_boundary_preserves_the_previous_request_on_error() {
    let mut request = AppendOnlyRequest::default();
    let initial = request.prepare("{}", ["\"\""]).unwrap().to_vec();
    let payload = format!("\"{}\"", "x".repeat(json::MAX_INPUT_BYTES - "{\"messages\":[\"\",]}".len() - 2));
    let exact = request.prepare("{}", ["\"\"", payload.as_str()]).unwrap().to_vec();
    assert_eq!(exact.len(), json::MAX_INPUT_BYTES);
    let prefix = request.prefix_bytes().to_vec();
    assert!(request.prepare("{}", ["\"\"", payload.as_str(), "null"]).is_err());
    assert_eq!(request.prefix_bytes(), prefix);
    assert!(exact.starts_with(&initial[..initial.len() - 2]));
}

#[test]
fn suffix_replacement_errors_leave_the_frozen_request_intact() {
    let messages: Vec<_> = MESSAGES.lines().collect();
    let mut request = AppendOnlyRequest::default();
    request.prepare("{\"stream\":true}", messages.iter().copied()).unwrap();
    let before = request.prefix_bytes().to_vec();
    for (options, records, stable) in [
        ("{\"stream\":true", messages.clone(), 1),
        ("{\"stream\":null}", messages.clone(), 1),
        ("{\"stream\":true}", messages.clone(), messages.len() + 1),
        ("{\"stream\":true}", vec![messages[0]], 2),
        ("{\"stream\":true}", vec![messages[1], messages[0]], 1),
    ] {
        assert_eq!(request.replace_suffix(options, records, stable).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert_eq!(request.prefix_bytes(), before);
    }
    let oversized = format!("\"{}\"", "x".repeat(json::MAX_INPUT_BYTES));
    assert!(request.replace_suffix("{\"stream\":true}", [messages[0], &oversized], 1).is_err());
    assert_eq!(request.prefix_bytes(), before);
}

#[test]
fn fixed_seed_suffix_replacements_preserve_each_validated_prefix() {
    let records: Vec<_> = MESSAGES.lines().collect();
    let mut seed = 0x1ea5_2026_1026_u64;
    for _ in 0..128 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let stable = (seed as usize) % (records.len() + 1);
        let mut request = AppendOnlyRequest::default();
        request.prepare("{}", records.iter().copied()).unwrap();
        let prefix = request.stable_prefix_bytes(stable).unwrap().to_vec();
        let replacement = format!(r#"{{"role":"user","content":"turn {seed}"}}"#);
        let mut next = records[..stable].to_vec(); next.push(&replacement);
        let body = request.replace_suffix("{}", next.iter().copied(), stable).unwrap().to_vec();
        assert!(body.starts_with(&prefix));
        let parsed = json::parse_bytes(&body).unwrap();
        assert_eq!(parsed.get("messages").unwrap().as_array().unwrap().len(), stable + 1);
        let before = request.prefix_bytes().to_vec();
        if stable > 0 {
            next[stable - 1] = "{\"role\":\"user\",\"content\":\"changed\"}";
            assert!(request.replace_suffix("{}", next, stable).is_err());
            assert_eq!(request.prefix_bytes(), before);
        }
        assert!(request.stable_prefix_bytes(stable + 2).is_none());
    }
}
