//! Claude Code sends its environment notes as `role: "system"` entries inside
//! Messages `messages`. The engine's Messages dialect takes only user and
//! assistant turns there and refuses the whole request, and Claude Code does
//! not retry. The gateway relabels just those role values as `"user"`, which
//! is the recovery Claude Code documents for such upstreams. Only the role
//! string's bytes change; every other body passes through byte for byte.
use crate::json;
use std::borrow::Cow;

pub(super) fn fold<'a>(method: &str, path: &str, body: Cow<'a, [u8]>) -> Cow<'a, [u8]> {
    if method != "POST" || !matches!(path.split('?').next().unwrap_or(path),
        "/v1/messages" | "/messages" | "/v1/messages/count_tokens" | "/messages/count_tokens") {
        return body;
    }
    let spans = match system_roles(&body) {
        Some(spans) if !spans.is_empty() => spans,
        _ => return body,
    };
    let mut folded = Vec::with_capacity(body.len());
    let mut copied = 0;
    for (start, end) in spans {
        folded.extend_from_slice(&body[copied..start]);
        folded.extend_from_slice(b"\"user\"");
        copied = end;
    }
    folded.extend_from_slice(&body[copied..]);
    Cow::Owned(folded)
}

/// Byte ranges of each `messages[i].role` value that decodes to `system`.
fn system_roles(body: &[u8]) -> Option<Vec<(usize, usize)>> {
    let text = std::str::from_utf8(body).ok()?;
    let messages = json::view(text).ok()?.get("messages")?.array()?;
    let mut spans = Vec::new();
    for message in messages {
        let Some(role) = message.get("role") else { continue };
        if role.as_str().as_deref() == Some("system") {
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
