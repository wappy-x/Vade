//! Minimal OpenAI-compatible text chat wire types.
use crate::json::{self, Value, invalid};
use std::io;

/// One chat message: a role and its exact text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A finished text reply and its `finish_reason`, if the server sent one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Completion {
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Interruption errors are terminal. Preserve only the engine's known code,
/// never its untrusted message (which may contain request or provider data).
pub(crate) fn interruption_error(code: &str) -> Option<io::Error> {
    let (kind, message) = match code {
        "client_cancelled" => (io::ErrorKind::Interrupted, "upstream request cancelled"),
        "generation_deadline" => (io::ErrorKind::TimedOut, "upstream generation deadline exceeded"),
        _ => return None,
    };
    Some(io::Error::new(kind, message))
}

pub(crate) fn response_error(error: &Value, fallback: &'static str) -> io::Error {
    error.get("code").and_then(Value::as_str)
        .or_else(|| error.get("type").and_then(Value::as_str)).and_then(interruption_error)
        .unwrap_or_else(|| io::Error::new(io::ErrorKind::InvalidData, fallback))
}

pub(crate) fn response_error_view(error: json::View<'_>, fallback: &'static str) -> io::Error {
    error.get("code").and_then(json::View::as_str)
        .or_else(|| error.get("type").and_then(json::View::as_str)).and_then(|code| interruption_error(&code))
        .unwrap_or_else(|| io::Error::new(io::ErrorKind::InvalidData, fallback))
}

pub(crate) fn chat_request(model: &str, messages: &[Message]) -> io::Result<Vec<u8>> {
    fn oversized() -> io::Error {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "chat request exceeds size limit",
        )
    }
    // Preflight the exact size. Serialize borrowed messages directly, without
    // copying the complete history into an intermediate JSON value tree.
    let mut size = "{\"model\":,\"messages\":[],\"stream\":true}".len() + json::quoted_len(model)?;
    for (i, message) in messages.iter().enumerate() {
        size += usize::from(i > 0)
            + "{\"role\":,\"content\":}".len()
            + json::quoted_len(&message.role)?
            + json::quoted_len(&message.content)?;
        if size > json::MAX_INPUT_BYTES {
            return Err(oversized());
        }
    }
    if size > json::MAX_INPUT_BYTES {
        return Err(oversized());
    }
    let mut body = String::with_capacity(size);
    body.push_str("{\"stream\":true,\"model\":");
    json::write_string(model, &mut body);
    body.push_str(",\"messages\":[");
    for (i, message) in messages.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        body.push_str("{\"role\":");
        json::write_string(&message.role, &mut body);
        body.push_str(",\"content\":");
        json::write_string(&message.content, &mut body);
        body.push('}');
    }
    body.push_str("]}");
    debug_assert_eq!(size, body.len());
    Ok(body.into_bytes())
}

/// Extract the first text choice. Tool calls remain the gateway's raw wire data;
/// direct text chat fails explicitly when a completion requires tool handling.
pub(crate) fn apply_event(
    value: &Value,
    completion: &mut Completion,
    on_delta: &mut dyn FnMut(&str),
) -> io::Result<()> {
    if let Some(error) = value.get("error") {
        return Err(response_error(error, "upstream returned a chat error"));
    }
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("chat response has no choices array"))?;
    // Empty choices are legal for streaming usage metadata.
    let Some(choice) = choices.first() else {
        return Ok(());
    };
    if let Some(reason) = choice.get("finish_reason") {
        match reason {
            Value::Null => {}
            Value::String(reason) => {
                if matches!(reason.as_str(), "tool_calls" | "function_call") {
                    return Err(invalid(
                        "direct text chat does not execute upstream tool calls",
                    ));
                }
                completion.finish_reason = Some(reason.to_owned());
            }
            _ => return Err(invalid("chat finish_reason must be a string or null")),
        }
    }
    let content = choice.get("delta").or_else(|| choice.get("message"));
    if let Some(content) = content {
        if !matches!(content, Value::Object(_)) {
            return Err(invalid("chat message or delta must be an object"));
        }
        let tool_calls = content.get("tool_calls").is_some_and(|value| {
            !matches!(value, Value::Null)
                && !matches!(value, Value::Array(items) if items.is_empty())
        });
        let function_call = content
            .get("function_call")
            .is_some_and(|value| !matches!(value, Value::Null));
        if tool_calls || function_call {
            return Err(invalid(
                "direct text chat does not execute upstream tool calls",
            ));
        }
        if let Some(text) = content.get("content") {
            match text {
                Value::Null => {}
                Value::String(text) => {
                    if completion.text.len().saturating_add(text.len()) > json::MAX_INPUT_BYTES {
                        return Err(invalid("completion exceeds size limit"));
                    }
                    completion.text.push_str(text);
                    on_delta(text);
                }
                _ => return Err(invalid("direct text chat requires string content")),
            }
        }
    } else {
        return Err(invalid("chat choice has no message or delta"));
    }
    Ok(())
}

/// Return the already validated event for usage projection without reparsing.
pub(crate) fn apply_event_data<'a>(
    data: &'a str,
    completion: &mut Completion,
    on_delta: &mut dyn FnMut(&str),
) -> io::Result<json::View<'a>> {
    let value = json::view(data)?;
    if let Some(error) = value.get("error") {
        return Err(response_error_view(error, "upstream returned a chat error"));
    }
    let mut choices = value
        .get("choices")
        .and_then(json::View::array)
        .ok_or_else(|| invalid("chat response has no choices array"))?;
    // Empty choices are legal for streaming usage metadata.
    let Some(choice) = choices.next() else {
        return Ok(value);
    };
    if let Some(reason) = choice.get("finish_reason") {
        match reason.as_str() {
            None if reason.raw() == "null" => {}
            Some(reason) => {
                if matches!(reason.as_ref(), "tool_calls" | "function_call") {
                    return Err(invalid(
                        "direct text chat does not execute upstream tool calls",
                    ));
                }
                completion.finish_reason = Some(reason.into_owned());
            }
            _ => return Err(invalid("chat finish_reason must be a string or null")),
        }
    }
    let content = choice.get("delta").or_else(|| choice.get("message"));
    if let Some(content) = content {
        if content.fields().is_none() {
            return Err(invalid("chat message or delta must be an object"));
        }
        let tool_calls = content.get("tool_calls").is_some_and(|value| {
            value.raw() != "null"
                && !value
                    .array()
                    .is_some_and(|mut items| items.next().is_none())
        });
        let function_call = content
            .get("function_call")
            .is_some_and(|value| value.raw() != "null");
        if tool_calls || function_call {
            return Err(invalid(
                "direct text chat does not execute upstream tool calls",
            ));
        }
        if let Some(text) = content.get("content") {
            match text.as_str() {
                None if text.raw() == "null" => {}
                Some(text) => {
                    if completion.text.len().saturating_add(text.len()) > json::MAX_INPUT_BYTES {
                        return Err(invalid("completion exceeds size limit"));
                    }
                    completion.text.push_str(&text);
                    on_delta(&text);
                }
                _ => return Err(invalid("direct text chat requires string content")),
            }
        }
    } else {
        return Err(invalid("chat choice has no message or delta"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_serialization_keeps_text_and_checks_escaped_size() {
        let messages = [Message {
            role: "user".into(),
            content: "\0\n🦀\"\\".into(),
        }];
        let bytes = chat_request("mock", &messages).unwrap();
        let parsed = json::parse_bytes(&bytes).unwrap();
        assert_eq!(
            parsed.get("messages").unwrap().as_array().unwrap()[0]
                .get("content")
                .unwrap()
                .as_str(),
            Some(messages[0].content.as_str())
        );
        let huge = [Message {
            role: "user".into(),
            content: "\0".repeat(json::MAX_INPUT_BYTES / 6 + 1),
        }];
        assert!(chat_request("mock", &huge).is_err());
    }

    #[test]
    fn tool_calls_fail_explicitly_in_direct_mode() {
        let value =
            json::parse(r#"{"choices":[{"delta":{"tool_calls":[{}]},"finish_reason":null}]}"#)
                .unwrap();
        assert!(apply_event(&value, &mut Completion::default(), &mut |_| {}).is_err());
        assert!(
            apply_event(
                &json::parse(r#"{"choices":[{}]}"#).unwrap(),
                &mut Completion::default(),
                &mut |_| {}
            )
            .is_err()
        );
        let no_calls = json::parse(r#"{"choices":[{"message":{"content":"hello","tool_calls":null,"function_call":null},"finish_reason":"stop"}]}"#).unwrap();
        let mut completion = Completion::default();
        apply_event(&no_calls, &mut completion, &mut |_| {}).unwrap();
        assert_eq!(completion.text, "hello");
    }
    #[test]
    fn borrowed_events_agree_with_owned_events_including_rejections() {
        for input in [
            r#"{"choices":[]}"#,
            r#"{"choices":[{"delta":{"content":" exact\n🦀 ","role":"assistant"},"finish_reason":"stop"}]}"#,
            r#"{"choices":[{"delta":{"content":null},"finish_reason":null}]}"#,
            r#"{"choices":[{"message":{"content":"json fallback"},"finish_reason":"length"}]}"#,
            r#"{"choices":[{"delta":{"content":{},"tool_calls":null}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{}]}}]}"#,
            r#"{"choices":[{"delta":{"function_call":{}}}]}"#,
            r#"{"choices":[{"delta":{"content":"x"},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":42}]}"#,
            r#"{"choices":[{}]}"#,
            r#"{"error":null,"choices":[]}"#,
            r#"{"error":{"type":"client_cancelled","message":"private"}}"#,
            r#"{"error":{"type":"generation_deadline","message":"private"}}"#,
            r#"{"choices":null}"#,
            r#"{"choices":[{"delta":null}]}"#,
            r#"{"choices":[{"delta":{"content":"x","tool_calls":[],"function_call":null}}]}"#,
        ] {
            let mut old = Completion::default();
            let mut new = Completion::default();
            let mut old_text = String::new();
            let mut new_text = String::new();
            let owned = apply_event(&json::parse(input).unwrap(), &mut old, &mut |s| {
                old_text.push_str(s)
            });
            let borrowed = apply_event_data(input, &mut new, &mut |s| new_text.push_str(s));
            assert_eq!(owned.is_ok(), borrowed.is_ok(), "{input}");
            assert_eq!(owned.as_ref().err().map(io::Error::kind), borrowed.as_ref().err().map(io::Error::kind));
            assert_eq!(old, new, "{input}");
            assert_eq!(old_text, new_text);
        }
    }
}
