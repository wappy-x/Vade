//! Two client role values the engine refuses, relabeled so the request runs.
//! Claude Code sends its environment notes as `role: "system"` entries inside
//! Messages `messages`. The engine's Messages dialect takes only user and
//! assistant turns there and refuses the whole request, and Claude Code does
//! not retry. The gateway relabels just those role values as `"user"`, which
//! is the recovery Claude Code documents for such upstreams.
//! The openai Python client builds a streamed reply's message from the deltas,
//! and the engine's deltas carry no `role`. Code that appends that message to
//! the history then sends `role: null`, which the engine refuses. On chat routes
//! the gateway relabels a `null` role as `"assistant"`, the only role a reply
//! has. Only the role value's bytes change; every other byte passes through.
use crate::json;
use std::borrow::Cow;

pub(super) fn fold<'a>(method: &str, path: &str, body: Cow<'a, [u8]>) -> Cow<'a, [u8]> {
    if method != "POST" {
        return body;
    }
    let (from, to): (Option<&str>, &[u8]) = match path.split('?').next().unwrap_or(path) {
        "/v1/messages" | "/messages" | "/v1/messages/count_tokens" | "/messages/count_tokens" => (Some("system"), b"\"user\""),
        "/v1/chat/completions" | "/chat/completions" => (None, b"\"assistant\""),
        _ => return body,
    };
    // Most chat bodies hold no null at all; they skip the parse.
    if from.is_none() && !body.windows(4).any(|w| w == b"null") {
        return body;
    }
    let spans = match roles(&body, from) {
        Some(spans) if !spans.is_empty() => spans,
        _ => return body,
    };
    let mut folded = Vec::with_capacity(body.len());
    let mut copied = 0;
    for (start, end) in spans {
        folded.extend_from_slice(&body[copied..start]);
        folded.extend_from_slice(to);
        copied = end;
    }
    folded.extend_from_slice(&body[copied..]);
    Cow::Owned(folded)
}

/// Byte ranges of each `messages[i].role` value that decodes to `from`, or
/// that is JSON `null` when `from` is `None`.
fn roles(body: &[u8], from: Option<&str>) -> Option<Vec<(usize, usize)>> {
    let text = std::str::from_utf8(body).ok()?;
    let messages = json::view(text).ok()?.get("messages")?.array()?;
    let mut spans = Vec::new();
    for message in messages {
        let Some(role) = message.get("role") else { continue };
        let matched = match from {
            Some(from) => role.as_str().as_deref() == Some(from),
            None => role.raw() == "null",
        };
        if matched {
            let start = role.raw().as_ptr() as usize - text.as_ptr() as usize;
            spans.push((start, start + role.raw().len()));
        }
    }
    Some(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folded(path: &str, body: &str) -> String {
        String::from_utf8(fold("POST", path, Cow::Borrowed(body.as_bytes())).into_owned()).unwrap()
    }

    #[test]
    fn only_system_role_values_inside_messages_change() {
        let role = |value: &str| format!(r#"{{ "system":"rules", "messages":[ {{"role":"user","content":"hi"}}, {{"content":[{{"type":"text","text":"role system"}}], "role" : {value}, "cache_control":{{"type":"ephemeral"}}}}, {{"role":"assistant","content":"system"}} ], "metadata":{{"role":"system"}} }}"#);
        let escaped = format!("\"sys{}u0074em\"", '\\');
        for value in ["\"system\"", escaped.as_str()] {
            assert_eq!(folded("/v1/messages?beta=true", &role(value)), role("\"user\""), "{value}");
        }
        let two = r#"{"messages":[{"role":"system","content":"a"},{"role":"user","content":"b"},{"role":"system","content":"c"}]}"#;
        for path in ["/messages", "/v1/messages/count_tokens"] {
            assert_eq!(folded(path, two), two.replace(r#""role":"system""#, r#""role":"user""#));
        }
    }

    #[test]
    fn chat_null_roles_become_assistant_and_nothing_else_changes() {
        // The shape openai-python 3.41.0 sent for a message built by its stream helper.
        let echoed = |role: &str| format!(r#"{{"model":"m","messages":[{{"role":"user","content":"Weather?"}},{{"content":null,"refusal":null,"role":{role},"annotations":null,"tool_calls":[{{"id":"call-1-0","function":{{"arguments":"{{\"city\": \"Paris\"}}","name":"get_weather","parsed_arguments":null}},"type":"function","index":0}}],"parsed":null}},{{"role":"tool","tool_call_id":"call-1-0","content":"{{\"role\":null}}"}}],"metadata":{{"role":null}}}}"#);
        for path in ["/v1/chat/completions", "/chat/completions?x=1"] {
            assert_eq!(folded(path, &echoed("null")), echoed("\"assistant\""));
            assert_eq!(folded(path, &echoed(" null ")), echoed(" \"assistant\" "));
        }
        for path in ["/v1/responses", "/v1/completions", "/v1/messages"] {
            assert!(matches!(fold("POST", path, Cow::Borrowed(echoed("null").as_bytes())), Cow::Borrowed(_)), "{path}");
        }
    }

    #[test]
    fn other_routes_methods_and_bodies_stay_borrowed() {
        let system = r#"{"messages":[{"role":"system","content":"a"}]}"#;
        for (method, path, body) in [
            ("POST", "/v1/chat/completions", system),
            ("POST", "/v1/responses", system),
            ("GET", "/v1/messages", system),
            ("POST", "/v1/messages", r#"{"messages":[{"role":"user","content":"system"}]}"#),
            ("POST", "/v1/messages", "not json"),
            ("POST", "/v1/messages", r#"{"messages":{"role":"system"}}"#),
        ] {
            assert!(matches!(fold(method, path, Cow::Borrowed(body.as_bytes())), Cow::Borrowed(_)), "{method} {path} {body}");
        }
    }
}
