//! Bounded, best-effort telemetry tap. Parsing cannot alter or reject relay bytes.
use crate::{json, transport::SseDecoder};

const MAX_JSON: usize = 1024 * 1024;
#[derive(Default)]
struct Counts {
    pair: Option<(u64, u64)>,
    snapshot: Option<(Option<u64>, Option<u64>)>,
    invalid: bool,
    terminal: bool,
    messages_final: bool,
}
impl Counts {
    fn event(&mut self, value: json::View<'_>) {
        let kind = value.get("type").and_then(json::View::as_str);
        if kind
            .as_deref()
            .is_some_and(|s| matches!(s, "response.failed" | "response.incomplete" | "error"))
        {
            self.invalid = true;
        }
        if matches!(kind.as_deref(), Some("response.completed" | "message_stop")) {
            self.terminal = true;
        }
        self.usage(
            value,
            matches!(kind.as_deref(), Some("message" | "message_delta")),
        );
        if let Some(response) = value.get("response") {
            self.usage(response, false);
        }
        if let Some(message) = value.get("message") {
            self.usage(message, false);
        }
    }
    fn usage(&mut self, value: json::View<'_>, message_final: bool) {
        if value.get("error").is_some_and(|v| v.raw() != "null") {
            self.invalid = true;
        }
        if value
            .get("status")
            .and_then(json::View::as_str)
            .is_some_and(|s| s == "failed" || s == "incomplete")
        {
            self.invalid = true;
        }
        let Some(usage) = value.get("usage").filter(|v| v.raw() != "null") else {
            return;
        };
        if usage.fields().is_none() {
            self.invalid = true;
            return;
        }
        // Messages input excludes both cache reads and cache writes. Delta events
        // can contain output_tokens only; retain the message_start input snapshot.
        if usage.get("cache_read_input_tokens").is_some()
            || usage.get("cache_creation_input_tokens").is_some()
        {
            let integer = |key| {
                usage.get(key).and_then(|v| {
                    v.raw()
                        .bytes()
                        .all(|b| b.is_ascii_digit())
                        .then(|| v.raw().parse::<u64>().ok())
                        .flatten()
                })
            };
            let pair = integer("input_tokens")
                .zip(integer("cache_read_input_tokens"))
                .zip(integer("cache_creation_input_tokens"))
                .and_then(|((input, read), created)| {
                    input
                        .checked_add(read)?
                        .checked_add(created)
                        .map(|total| (total, read))
                });
            if let Some((prompt, cached)) = pair {
                // Flash can confirm restored cache rows only after prefill. The
                // final Messages snapshot replaces that provisional split;
                // prompt total and repeated snapshots must remain consistent.
                if self.snapshot.is_some_and(|old| {
                    old.0 != Some(prompt)
                        || (self.messages_final || !message_final) && old.1 != Some(cached)
                }) {
                    self.invalid = true;
                }
                self.messages_final |= message_final;
                self.snapshot = Some((Some(prompt), Some(cached)));
                self.pair = Some((prompt, cached));
            } else {
                self.invalid = true;
            }
            return;
        }
        if value.get("type").and_then(json::View::as_str).as_deref() == Some("message_delta")
            && usage.get("input_tokens").is_none()
        {
            return;
        }
        // Both Chat and Responses counts are cumulative snapshots, never deltas.
        let mut observed = false;
        for (input, details) in [
            ("prompt_tokens", "prompt_tokens_details"),
            ("input_tokens", "input_tokens_details"),
        ] {
            let prompt = usage.get(input);
            if prompt.is_none() && usage.get(details).is_none() {
                continue;
            }
            observed = true;
            let detail = usage.get(details).filter(|v| v.raw() != "null");
            if detail.is_some_and(|v| v.fields().is_none()) {
                self.invalid = true;
                return;
            }
            let cached = detail.and_then(|v| v.get("cached_tokens"));
            let parse = |v: json::View<'_>| -> Option<u64> {
                if !v.raw().bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                v.raw().parse().ok()
            };
            let p = prompt.filter(|v| v.raw() != "null").and_then(parse);
            let c = cached.filter(|v| v.raw() != "null").and_then(parse);
            if prompt.is_some_and(|v| v.raw() != "null") && p.is_none()
                || cached.is_some_and(|v| v.raw() != "null") && c.is_none()
            {
                self.invalid = true;
            }
            self.snapshot(p, c);
        }
        if !observed {
            self.snapshot(None, None);
        }
    }
    fn snapshot(&mut self, prompt: Option<u64>, cached: Option<u64>) {
        let snapshot = (prompt, cached);
        if self.snapshot.is_some_and(|old| old != snapshot)
            || prompt.zip(cached).is_some_and(|(p, c)| c > p)
        {
            self.invalid = true;
        }
        self.snapshot = Some(snapshot);
        self.pair = prompt.zip(cached);
    }
}

enum Body {
    Json(Vec<u8>),
    Sse(SseDecoder),
    Unknown,
}
pub(super) struct UsageTap {
    body: Body,
    counts: Counts,
}
impl UsageTap {
    pub(super) fn new(content_type: Option<&str>, enabled: bool) -> Self {
        let media = content_type
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let body = if enabled && media.eq_ignore_ascii_case("application/json") {
            Body::Json(Vec::new())
        } else if enabled && media.eq_ignore_ascii_case("text/event-stream") {
            Body::Sse(SseDecoder::default())
        } else {
            Body::Unknown
        };
        Self {
            body,
            counts: Counts::default(),
        }
    }
    pub(super) fn feed(&mut self, bytes: &[u8]) {
        let valid = match &mut self.body {
            Body::Unknown => return,
            Body::Json(body) => {
                if body.len().saturating_add(bytes.len()) > MAX_JSON {
                    false
                } else {
                    body.extend_from_slice(bytes);
                    true
                }
            }
            Body::Sse(decoder) => {
                let counts = &mut self.counts;
                decoder
                    .feed(bytes, &mut |data| {
                        if data == "[DONE]" {
                            counts.terminal = true;
                        } else {
                            counts.event(json::view(data)?);
                        }
                        // Keep observing after DONE to notice conflicting/error events;
                        // this has no effect on the opaque downstream response.
                        Ok(false)
                    })
                    .is_ok()
            }
        };
        if !valid {
            self.body = Body::Unknown;
            self.counts.invalid = true;
        }
    }
    pub(super) fn finish(mut self) -> Option<(u64, u64)> {
        match self.body {
            Body::Json(bytes) => {
                let value = std::str::from_utf8(&bytes)
                    .ok()
                    .and_then(|v| json::view(v).ok())?;
                self.counts.event(value);
            }
            Body::Sse(decoder) => {
                if !self.counts.terminal || !decoder.at_event_boundary() {
                    return None;
                }
            }
            Body::Unknown => return None,
        }
        if self.counts.invalid {
            None
        } else {
            self.counts.pair
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sse(parts: &[&str]) -> String {
        parts.iter().map(|p| format!("data: {p}\r\n\r\n")).collect()
    }
    const USAGE: &str =
        r#"{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":80}}}"#;
    #[test]
    fn fragmented_chat_and_responses_snapshots_count_once() {
        for payload in [
            sse(&[USAGE, USAGE, "[DONE]"]),
            sse(&[
                r#"{"type":"message_start","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}"#,
                r#"{"type":"message_delta","usage":{"input_tokens":15,"cache_read_input_tokens":80,"cache_creation_input_tokens":5,"output_tokens":3}}"#,
                r#"{"type":"message_stop"}"#,
            ]),
            sse(&[
                r#"{"type":"message_start","message":{"usage":{"input_tokens":15,"cache_read_input_tokens":80,"cache_creation_input_tokens":5}}}"#,
                r#"{"type":"message_delta","usage":{"output_tokens":3}}"#,
                r#"{"type":"message_stop"}"#,
            ]),
            sse(&[
                r#"{"type":"response.completed","response":{"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":80}}}}"#,
            ]),
        ] {
            for split in 0..payload.len() {
                let mut tap = UsageTap::new(Some("text/event-stream; charset=utf-8"), true);
                tap.feed(&payload.as_bytes()[..split]);
                tap.feed(&payload.as_bytes()[split..]);
                assert_eq!(tap.finish(), Some((100, 80)), "split {split}");
            }
        }
    }
    #[test]
    fn missing_conflicting_invalid_failed_and_unterminated_usage_remains_unknown() {
        for payload in [
            sse(&[USAGE]),
            sse(&[USAGE, &USAGE.replace("100", "200"), "[DONE]"]),
            sse(&[USAGE, r#"{"usage":{"prompt_tokens":100}}"#, "[DONE]"]),
            sse(&[USAGE, r#"{"usage":{}}"#, "[DONE]"]),
            sse(&[&USAGE.replace("80", "101"), "[DONE]"]),
            sse(&[&USAGE.replace("100", "-1"), "[DONE]"]),
            sse(&[USAGE, "[DONE]", r#"{"error":"oops"}"#]),
            sse(&[USAGE, r#"{"type":"response.failed"}"#, "[DONE]"]),
            sse(&[
                USAGE,
                r#"{"type":"response.completed","response":{"error":"failed"}}"#,
            ]),
            sse(&[USAGE, "[DONE]"]) + "data: incomplete",
            sse(&["not json", "[DONE]"]),
        ] {
            let mut tap = UsageTap::new(Some("text/event-stream"), true);
            tap.feed(payload.as_bytes());
            assert_eq!(tap.finish(), None, "{payload}");
        }
        for payload in [
            r#"{"type":"message","usage":{"input_tokens":15,"cache_read_input_tokens":80,"cache_creation_input_tokens":5}}"#,
            r#"{"type":"message","usage":{"input_tokens":100,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}"#,
        ] {
            let mut tap = UsageTap::new(Some("application/json"), true);
            tap.feed(payload.as_bytes());
            assert_eq!(
                tap.finish(),
                Some((100, if payload.contains(":80") { 80 } else { 0 }))
            );
        }
        let mut tap = UsageTap::new(Some("application/json"), true);
        tap.feed(USAGE.replace("80", "0").as_bytes());
        assert_eq!(tap.finish(), Some((100, 0)));
        let mut tap = UsageTap::new(Some("application/json"), true);
        tap.feed(&vec![b' '; MAX_JSON + 1]);
        assert_eq!(tap.finish(), None);
        let mut tap = UsageTap::new(Some("text/event-stream"), true);
        tap.feed(&vec![b'x'; MAX_JSON + 1]);
        assert_eq!(tap.finish(), None);
    }
}
