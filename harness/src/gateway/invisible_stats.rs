//! Best-effort bounded observation. Never changes or rejects engine bytes.
use super::policy::integer;
use crate::{
    json::{self, View},
    transport::{Head, SseDecoder},
};
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub(super) const OPEN_FAILED: &str = "cannot open the --stats file";
pub(super) struct StatsLog(Mutex<File>);
impl StatsLog {
    pub(super) fn open(path: &str) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        // Name the flag and keep the kind; the OS text alone names no file.
        options.open(path).map(|file| Self(Mutex::new(file))).map_err(|error| io::Error::new(error.kind(), OPEN_FAILED))
    }
    pub(super) fn write(&self, line: &str) -> io::Result<()> {
        let mut file = self.0.lock().unwrap_or_else(|p| p.into_inner());
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()
    }
}

enum Tap {
    None,
    Json(Vec<u8>),
    Sse(SseDecoder),
}
#[derive(Default)]
struct Observed {
    prompt: Option<u64>,
    completion: Option<u64>,
    cached: Option<u64>,
    prefill: Option<f64>,
    logits: Option<f64>,
    first: Option<f64>,
    last: Option<f64>,
    generated_events: usize,
    terminal: bool,
    invalid: bool,
    events: usize,
}
impl Observed {
    fn event(&mut self, value: View<'_>, at: f64, streaming: bool) {
        self.events += 1;
        let kind = value.get("type").and_then(View::as_str);
        if matches!(kind.as_deref(), Some("error" | "response.failed"))
            || value.get("error").is_some_and(|v| v.raw() != "null")
        {
            self.invalid = true;
        }
        if matches!(
            kind.as_deref(),
            Some("response.completed" | "response.incomplete" | "message_stop")
        ) {
            self.terminal = true;
        }
        let response = value.get("response").unwrap_or(value);
        if let Some(usage) = response.get("usage").filter(|v| v.raw() != "null") {
            let prompt = integer(
                usage
                    .get("prompt_tokens")
                    .or_else(|| usage.get("input_tokens")),
            );
            let completion = integer(
                usage
                    .get("completion_tokens")
                    .or_else(|| usage.get("output_tokens")),
            );
            let cached = integer(
                usage
                    .get("prompt_tokens_details")
                    .or_else(|| usage.get("input_tokens_details"))
                    .and_then(|v| v.get("cached_tokens")),
            );
            let vade = response
                .get("vade")
                .or_else(|| value.get("vade"))
                .or_else(|| usage.get("vade"));
            // Messages input_tokens leaves out cache reads and writes: add them
            // back for the whole prompt, as the engine counts it. An absent part
            // counts 0; a part that is not a count, or a sum that overflows, is invalid.
            let (read, write) = (usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens"));
            let (prompt, cached) = if read.is_some() || write.is_some() {
                let part = |v: Option<View<'_>>| v.map_or(Some(0), |v| integer(Some(v)));
                let total = prompt.zip(part(read)).zip(part(write))
                    .and_then(|((input, r), w)| input.checked_add(r)?.checked_add(w));
                if total.is_none() {
                    self.invalid = true;
                }
                (total, cached.or_else(|| total.and(part(read))))
            } else {
                (prompt, cached)
            };
            let restored = integer(vade.and_then(|v| v.get("prefill_cache_restored_tokens")));
            if cached.zip(restored).is_some_and(|(a, b)| a != b) {
                self.invalid = true;
            }
            if let Some(v) = vade {
                self.prefill = number(v.get("prefill_ms"));
                self.logits = number(v.get("admission_to_first_logits_ms"));
            }
            let cached = cached.or(restored);
            if self.prompt.is_some()
                && (self.prompt != prompt || self.completion != completion || self.cached != cached)
                || cached.zip(prompt).is_some_and(|(c, p)| c > p)
            {
                self.invalid = true;
            }
            self.prompt = prompt;
            self.completion = completion;
            self.cached = cached;
        }
        let response_delta = matches!(
            kind.as_deref(),
            Some(
                "response.output_text.delta"
                    | "response.reasoning_summary_text.delta"
                    | "response.reasoning_text.delta"
                    | "response.function_call_arguments.delta"
            )
        ) && nonempty(value.get("delta"));
        let tool_start = kind.as_deref() == Some("response.output_item.added")
            && value.get("item").is_some_and(|item| {
                item.get("type").and_then(View::as_str).as_deref() == Some("function_call")
                    && nonempty(item.get("name"))
            });
        let completion_delta = value
            .get("choices")
            .and_then(View::array)
            .is_some_and(|mut choices| {
                choices.any(|choice| {
                    nonempty(choice.get("text")) || choice.get("delta").is_some_and(|delta| {
                        ["content", "reasoning_content", "reasoning"]
                            .iter()
                            .any(|key| nonempty(delta.get(key)))
                            || delta.get("tool_calls").and_then(View::array).is_some_and(
                                |mut calls| {
                                    calls.any(|call| {
                                        call.get("function").is_some_and(|f| {
                                            nonempty(f.get("name")) || nonempty(f.get("arguments"))
                                        })
                                    })
                                },
                            )
                    })
                })
            });
        // Messages: a text, thinking or tool-input delta, or a named tool_use start.
        let message_delta = match kind.as_deref() {
            Some("content_block_delta") => value.get("delta").is_some_and(|delta|
                ["text", "thinking", "partial_json"].iter().any(|key| nonempty(delta.get(key)))),
            Some("content_block_start") => value.get("content_block").is_some_and(|block|
                block.get("type").and_then(View::as_str).as_deref() == Some("tool_use") && nonempty(block.get("name"))),
            _ => false,
        };
        if streaming && (response_delta || tool_start || completion_delta || message_delta) {
            self.first.get_or_insert(at);
            self.last = Some(at);
            self.generated_events += 1;
        }
    }
}
fn number(v: Option<View<'_>>) -> Option<f64> {
    v?.raw()
        .parse::<f64>()
        .ok()
        .filter(|n| n.is_finite() && *n >= 0.0)
}
fn nonempty(v: Option<View<'_>>) -> bool {
    v.and_then(View::as_str).is_some_and(|s| !s.is_empty())
}
fn optional<T: ToString>(v: Option<T>) -> String {
    v.map_or_else(|| "null".into(), |v| v.to_string())
}

pub(super) struct Metrics {
    pub(super) upstream_complete: bool,
    downstream_complete: bool,
    pub(super) policy: String,
    enabled: bool,
    started: Instant,
    ts: u128,
    pub(super) path: String,
    model: Option<String>,
    stream: Option<bool>,
    status: Option<u16>,
    first_byte: Option<f64>,
    complete: bool,
    bytes_in: usize,
    bytes_out: usize,
    chunks: usize,
    write_ms: f64,
    tap: Tap,
    observed: Observed,
}
impl Metrics {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            upstream_complete: false,
            downstream_complete: false,
            policy: "null".into(),
            enabled,
            started: Instant::now(),
            ts: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            path: String::new(),
            model: None,
            stream: None,
            status: None,
            first_byte: None,
            complete: false,
            bytes_in: 0,
            bytes_out: 0,
            chunks: 0,
            write_ms: 0.0,
            tap: Tap::None,
            observed: Observed::default(),
        }
    }
    /// A fresh reply record for one upstream of a broadcast, keeping the request side.
    pub(super) fn attempt(&self) -> Self {
        Self {
            policy: self.policy.clone(),
            started: self.started,
            ts: self.ts,
            path: self.path.clone(),
            model: self.model.clone(),
            stream: self.stream,
            bytes_in: self.bytes_in,
            ..Self::new(self.enabled)
        }
    }
    fn ms(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }
    pub(super) fn request(&mut self, body: &[u8]) {
        self.bytes_in = body.len();
        if !self.enabled {
            return;
        }
        if let Some(v) = std::str::from_utf8(body)
            .ok()
            .and_then(|v| json::view(v).ok())
        {
            self.model = v
                .get("model")
                .and_then(View::as_str)
                .map(|s| s.into_owned());
            self.stream = v.get("stream").and_then(|v| match v.raw() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            });
            if v.get("stream").is_none() && v.fields().is_some() {
                self.stream = Some(false);
            }
        }
    }
    pub(super) fn first_byte(&mut self) {
        self.first_byte = Some(self.ms());
    }
    pub(super) fn terminal(&self) -> bool {
        self.observed.terminal && !self.observed.invalid
    }
    pub(super) fn response(&mut self, head: &Head, status: u16) {
        self.status = Some(status);
        if !self.enabled
            || head
                .get("content-encoding")
                .is_some_and(|v| v != "identity")
        {
            return;
        }
        self.tap = match head
            .get("content-type")
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
        {
            "text/event-stream" => Tap::Sse(SseDecoder::default()),
            "application/json" => Tap::Json(Vec::new()),
            _ => Tap::None,
        };
    }
    pub(super) fn payload(&mut self, bytes: &[u8], write: Duration) {
        if !self.enabled {
            return;
        }
        let at = self.ms();
        self.bytes_out += bytes.len();
        self.chunks += 1;
        self.write_ms += write.as_secs_f64() * 1000.0;
        match &mut self.tap {
            Tap::None => (),
            Tap::Json(body) => {
                if body.len().saturating_add(bytes.len()) <= 1024 * 1024 {
                    body.extend_from_slice(bytes);
                } else {
                    self.observed.invalid = true;
                    self.tap = Tap::None;
                }
            }
            Tap::Sse(decoder) => {
                let observed = &mut self.observed;
                if decoder
                    .feed(bytes, &mut |data| {
                        if data == "[DONE]" {
                            observed.terminal = true;
                        } else {
                            observed.event(json::view(data)?, at, true);
                        }
                        Ok(false)
                    })
                    .is_err()
                {
                    observed.invalid = true;
                    self.tap = Tap::None;
                }
            }
        }
    }
    pub(super) fn finish(&mut self, complete: bool) {
        self.downstream_complete = complete;
        self.complete = complete || self.upstream_complete;
        match &self.tap {
            Tap::Json(body) => match std::str::from_utf8(body)
                .ok()
                .and_then(|v| json::view(v).ok())
            {
                Some(value) => self.observed.event(value, self.ms(), false),
                None => self.observed.invalid = true,
            },
            Tap::Sse(decoder) => {
                self.observed.invalid |= !self.observed.terminal || !decoder.at_event_boundary();
            }
            Tap::None => (),
        }
        if !self.complete || !self.status.is_some_and(|s| (200..300).contains(&s)) {
            self.observed.invalid = true;
        }
    }
    pub(super) fn json(&self) -> String {
        let o = &self.observed;
        let valid = !o.invalid;
        let (prompt, completion, cached) = if valid {
            (o.prompt, o.completion, o.cached)
        } else {
            (None, None, None)
        };
        let reuse = prompt
            .zip(cached)
            .filter(|(p, _)| *p > 0)
            .map(|(p, c)| c as f64 * 100.0 / p as f64);
        let decode = if valid && o.generated_events > 1 {
            completion
                .zip(o.first.zip(o.last))
                .filter(|(n, (a, b))| *n > 1 && b > a)
                .map(|(n, (a, b))| (n - 1) as f64 * 1000.0 / (b - a))
        } else {
            None
        };
        let overhead = o
            .first
            .zip(o.logits)
            .map(|(client, engine)| client - engine);
        let per_token = completion
            .filter(|n| *n > 0)
            .map(|n| self.write_ms / n as f64);
        format!(
            "{{\"ts\":{},\"path\":{},\"model\":{},\"stream\":{},\"status\":{},\"complete\":{},\"downstream_complete\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"cached_tokens\":{},\"cache_reuse_pct\":{},\"vade\":{{\"prefill_ms\":{},\"admission_to_first_logits_ms\":{}}},\"first_byte_ms\":{},\"first_token_ms\":{},\"decode_tok_s\":{},\"added_first_token_ms\":{},\"relay_write_ms\":{},\"relay_overhead_ms_per_token\":{},\"chunks\":{},\"sse_events\":{},\"bytes_in\":{},\"bytes_out\":{},\"total_ms\":{},\"telemetry_invalid\":{},\"policy\":{}}}",
            self.ts,
            json::quote(&self.path),
            self.model
                .as_deref()
                .map_or_else(|| "null".into(), json::quote),
            optional(self.stream),
            optional(self.status),
            self.complete,
            self.downstream_complete,
            optional(prompt),
            optional(completion),
            optional(cached),
            optional(reuse),
            optional(o.prefill),
            optional(o.logits),
            optional(self.first_byte),
            optional(o.first),
            optional(decode),
            optional(overhead),
            self.write_ms,
            optional(per_token),
            self.chunks,
            o.events,
            self.bytes_in,
            self.bytes_out,
            self.ms(),
            o.invalid,
            self.policy
        )
    }
}

#[cfg(test)] mod tests { use super::*;
    /// A retained-prefix warm turn in each dialect, cut at several fragment
    /// sizes: every row reports the engine's 960 of 1000 prompt tokens.
    #[test]
    fn warm_turn_rows_match_the_engine_cache_counts_in_every_dialect() {
        let fixture = json::parse(include_str!("../../tests/fixtures/engine-warm-turn.json")).unwrap();
        for case in fixture.get("cases").unwrap().as_array().unwrap() {
            let (api, stream) = (case.get("api").unwrap().as_str().unwrap(), case.get("stream") == Some(&json::Value::Bool(true)));
            let body = case.get("body").unwrap().as_str().unwrap().as_bytes();
            for size in [1, 7, body.len()] {
                let mut m = Metrics::new(true);
                let media = if stream { "text/event-stream" } else { "application/json" };
                m.response(&Head { first: "HTTP/1.1 200 OK".into(), fields: vec![("content-type".into(), media.into())] }, 200);
                for fragment in body.chunks(size) { m.payload(fragment, Duration::ZERO); }
                m.finish(true);
                let row = json::parse(&m.json()).unwrap();
                let number = |key: &str| row.get(key).cloned();
                assert_eq!(number("telemetry_invalid"), Some(json::Value::Bool(false)), "{api} stream={stream} size={size}: {}", m.json());
                assert_eq!(number("prompt_tokens"), Some(json::Value::Number("1000".into())), "{api} stream={stream}");
                assert_eq!(number("cached_tokens"), Some(json::Value::Number("960".into())), "{api} stream={stream}");
                assert_eq!(number("completion_tokens"), Some(json::Value::Number("3".into())), "{api} stream={stream}");
                assert_eq!(number("cache_reuse_pct"), Some(json::Value::Number("96".into())), "{api} stream={stream}");
                // A stream's first generated event times the warm turn; a JSON body cannot.
                assert_eq!(m.observed.first.is_some(), stream, "{api} stream={stream} size={size}");
            }
        }
    }
    /// Messages cache parts that cannot form the engine's prompt count publish
    /// no prompt count; an absent part counts 0.
    #[test]
    fn messages_cache_parts_that_do_not_add_up_mark_the_row_invalid() {
        let row = |usage: &str| {
            let body = format!(r#"{{"type":"message","content":[{{"type":"text","text":"ok"}}],"usage":{usage}}}"#);
            let mut m = Metrics::new(true);
            m.response(&Head { first: "HTTP/1.1 200 OK".into(), fields: vec![("content-type".into(), "application/json".into())] }, 200);
            m.payload(body.as_bytes(), Duration::ZERO);
            m.finish(true);
            json::parse(&m.json()).unwrap()
        };
        for usage in [
            r#"{"input_tokens":18446744073709551615,"output_tokens":3,"cache_read_input_tokens":1,"cache_creation_input_tokens":0}"#,
            r#"{"input_tokens":40,"output_tokens":3,"cache_read_input_tokens":0,"cache_creation_input_tokens":-5}"#,
            r#"{"output_tokens":3,"cache_read_input_tokens":7}"#,
        ] {
            let r = row(usage);
            assert_eq!(r.get("telemetry_invalid"), Some(&json::Value::Bool(true)), "{usage}");
            assert_eq!(r.get("prompt_tokens"), Some(&json::Value::Null), "{usage}");
        }
        let r = row(r#"{"input_tokens":40,"output_tokens":3,"cache_creation_input_tokens":60}"#);
        assert_eq!(r.get("telemetry_invalid"), Some(&json::Value::Bool(false)));
        assert_eq!(r.get("prompt_tokens"), Some(&json::Value::Number("100".into())));
        assert_eq!(r.get("cached_tokens"), Some(&json::Value::Number("0".into())));
    }
    #[test]
    fn recorded_engine_completions_have_generated_timing_at_every_fragment_size() {
        let recording = json::parse(include_str!("../../tests/fixtures/completions-engine-2026-10-23.json")).unwrap();
        for case in recording.get("cases").unwrap().as_array().unwrap() {
            let sse = case.get("direct_sse").unwrap().as_str().unwrap().as_bytes();
            for size in [1, 3, 31, sse.len()] {
                let mut m = Metrics::new(true);
                m.response(&Head { first: "HTTP/1.1 200 OK".into(), fields: vec![("content-type".into(), "text/event-stream".into())] }, 200);
                for fragment in sse.chunks(size) { m.payload(fragment, Duration::ZERO); }
                m.finish(true);
                let row = json::parse(&m.json()).unwrap();
                assert!(!m.observed.invalid);
                assert_eq!(m.observed.generated_events, 32);
                assert_eq!(m.observed.completion, Some(32));
                assert_ne!(row.get("first_token_ms"), Some(&json::Value::Null));
                // One coalesced read cannot establish a decode interval.
                if size == sse.len() { assert_eq!(row.get("decode_tok_s"), Some(&json::Value::Null)); }
            }
        }
    }
    #[test]
    fn completion_timing_ignores_empty_finish_and_usage_events() {
        let mut o = Observed::default();
        for (at, data) in [
            (1.0, r#"{"choices":[{"text":"","finish_reason":null}]}"#),
            (2.0, r#"{"choices":[{"text":null}]}"#),
            (3.0, r#"{"choices":[{"text":17}]}"#),
            (10.0, r#"{"choices":[{"text":"雪"}]}"#),
            (25.0, r#"{"choices":[{"text":" café","finish_reason":"stop"}]}"#),
            (100.0, r#"{"choices":[{"text":"","finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2}}"#),
        ] { o.event(json::view(data).unwrap(), at, true); }
        assert_eq!((o.first, o.last, o.generated_events), (Some(10.0), Some(25.0), 2));
        assert_eq!((o.prompt, o.completion), (Some(7), Some(2)));
        let mut json_reply = Observed::default();
        json_reply.event(json::view(r#"{"choices":[{"text":"complete answer"}]}"#).unwrap(), 1.0, false);
        assert!(json_reply.first.is_none());
    }
    #[test]
fn fragmented_final_usage_and_generated_timing_exclude_role_and_trailer() { let mut m = Metrics::new(true); m.request(br#"{"model":"mock","stream":true}"#); m.response( &Head { first: "HTTP/1.1 200 OK".into(),
fields: vec![("content-type".into(), "text/event-stream".into())], }, 200, ); for data in [ "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n", "data: {\"choices\":[{\"delta\":{\"content\":\"one\"}}]}\n\n",
"data: {\"choices\":[{\"delta\":{\"content\":\"two\"}}]}\n\n",
"data: {\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":90}},\"vade\":{\"prefill_cache_restored_tokens\":90,\"prefill_ms\":3,\"admission_to_first_logits_ms\":4}}\n\ndata: [DONE]\n\n",
] { for fragment in data.as_bytes().chunks(3) { m.payload(fragment, Duration::ZERO); } } m.finish(true); let row = json::parse(&m.json()).unwrap(); assert_eq!( row.get("cache_reuse_pct"), Some(&json::Value::Number("90".into())) );
assert_ne!(row.get("decode_tok_s"), Some(&json::Value::Null)); assert_eq!(m.observed.generated_events, 2); }
    #[test]
fn conflicting_cache_aliases_and_incomplete_replies_never_publish_counts() { let mut o = Observed::default();
o.event(json::view(r#"{"usage":{"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":3}},"vade":{"prefill_cache_restored_tokens":4}}"#).unwrap(),1.0,false); assert!(o.invalid); let mut m = Metrics::new(true);
m.observed.prompt = Some(10); m.status = Some(200); m.finish(false); assert_eq!( json::parse(&m.json()).unwrap().get("prompt_tokens"), Some(&json::Value::Null) ); }
    #[test]
fn responses_wrapper_vade_and_nested_usage_timing_are_observed() { for raw in [
r#"{"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":2,"input_tokens_details":{"cached_tokens":9}}},"vade":{"prefill_ms":3,"admission_to_first_logits_ms":4,"prefill_cache_restored_tokens":9}}"#,
r#"{"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":9},"vade":{"prefill_ms":3,"admission_to_first_logits_ms":4,"prefill_cache_restored_tokens":9}}}"#, ] { let mut o = Observed::default();
o.event(json::view(raw).unwrap(), 5.0, true); assert_eq!( (o.prompt, o.cached, o.prefill, o.logits), (Some(10), Some(9), Some(3.0), Some(4.0)) ); assert!(!o.invalid); } } }
