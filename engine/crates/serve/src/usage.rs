//! Shared Chat/Responses accounting. Cached tokens are a subset of input.
use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    pub prefill: Duration,
    pub decode: Duration,
}

/// Host wall time of committed engine prefill chunks. A yielded chunk keeps
/// its start until commit; batch peers share the same engine-call clock pair.
/// No device synchronization or clock reads during decode.
#[derive(Debug, Clone)]
pub(crate) struct Prefill {
    pub prompt_tokens: usize,
    pub cache_restored_tokens: usize,
    pub elapsed: Duration,
    pub rows_per_chunk: Vec<usize>,
    pub first_logits: Option<Duration>,
    /// Wall time from cache lookup to the restored session (set only on a cache hit).
    pub cache_restore: Option<Duration>,
    chunks: usize,
    admitted: Instant,
    started: Option<Instant>,
}
impl Prefill {
    pub fn new(prompt_tokens: usize, cache_restored_tokens: usize, admitted: Instant) -> Self {
        Self { prompt_tokens, cache_restored_tokens, elapsed: Duration::ZERO,
            rows_per_chunk: Vec::new(), first_logits: None, cache_restore: None, chunks: 0, admitted, started: None }
    }
    pub fn begin(&mut self, started: Instant) { self.started.get_or_insert(started); }
    pub fn abandon(&mut self) { self.started = None; }
    pub fn commit(&mut self, rows: usize, finished: Instant, last: bool) {
        if let Some(started) = self.started.take() {
            self.elapsed += finished.duration_since(started);
            self.chunks += 1;
            if self.rows_per_chunk.len() < 64 { self.rows_per_chunk.push(rows); }
        }
        if last { self.first_logits.get_or_insert(finished.duration_since(self.admitted)); }
    }
    /// Request cohort width is log-only: keep every response field unchanged.
    pub fn commit_cohort(&mut self, id: u64, rows: usize, width: usize, finished: Instant, last: bool) {
        self.commit(rows, finished, last);
        eprintln!("request_prefill_cohort id={id} width={width} rows={rows}");
    }
    pub fn json(&self) -> String {
        let ms = |t: Option<Duration>| t.map(|t| format!("{:.3}", t.as_secs_f64() * 1000.0))
            .unwrap_or_else(|| "null".into());
        let first = ms(self.first_logits);
        // The one reuse field for clients: restored prompt tokens / prompt tokens, in percent.
        let reuse = self.cache_restored_tokens as f64 * 100.0 / self.prompt_tokens.max(1) as f64;
        format!("\"prefill_ms\":{:.3},\"prefill_prompt_tokens\":{},\"prefill_chunks\":{},\"prefill_rows_per_chunk\":{:?},\"prefill_rows_per_chunk_omitted\":{},\"prefill_cache_restored_tokens\":{},\"prefill_cache_reuse_pct\":{reuse:.1},\"prefill_cache_restore_ms\":{},\"admission_to_first_logits_ms\":{first}",
            self.elapsed.as_secs_f64() * 1000.0, self.prompt_tokens, self.chunks,
            self.rows_per_chunk, self.chunks - self.rows_per_chunk.len(), self.cache_restored_tokens,
            ms(self.cache_restore))
    }
}

#[cfg(test)] #[test] fn cohort_logging_preserves_response_bytes() {
    let started = Instant::now(); let mut solo = Prefill::new(4, 0, started); solo.begin(started); let mut shared = solo.clone(); let finished = started + Duration::from_millis(3); solo.commit(4, finished, true); shared.commit_cohort(42, 4, 2, finished, true); assert_eq!(solo.json(), shared.json());
    let request = crate::protocol::parse(br#"{"prompt_token_ids":[1],"temperature":0}"#, false, &crate::Limits::default()).unwrap(); let mut a = crate::protocol::SamplingEcho::new(&request, crate::protocol::DraftMode::Plain, 0); a.prefill = Some(solo); let mut b = a.clone(); b.prefill = Some(shared); let usage = crate::generation::Usage::default();
    for chat in [false, true] { let body = crate::http::Completion::default(); assert_eq!(body.json("model", chat, 1, 2, "length", usage, &a), body.json("model", chat, 1, 2, "length", usage, &b)); assert_eq!(crate::http::sse_chunk("model", chat, 1, 2, "text", None, Some("length"), Some(usage), Some(&a)), crate::http::sse_chunk("model", chat, 1, 2, "text", None, Some("length"), Some(usage), Some(&b))); } let mut responses = crate::http::Output::Responses(Default::default()); assert_eq!(responses.json("model", 1, 2, "length", usage, None, &a), responses.json("model", 1, 2, "length", usage, None, &b)); assert_eq!(responses.finish("model", 1, 2, "length", usage, None, &a), crate::http::Output::Responses(Default::default()).finish("model", 1, 2, "length", usage, None, &b)); }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kv {
    pub cache_domain: crate::cache::SessionKey,
    pub reuse: &'static str,
    pub reused_tokens: usize,
    pub retention: &'static str,
    pub checkpoint: Option<crate::cache::Checkpoint>,
}
impl Kv {
    pub(crate) fn finish(&mut self) {
        if self.retention == "pending" { self.retention = "prompt_incomplete"; }
    }
    fn json(self) -> String {
        let checkpoint = self.checkpoint.map(|c| format!(
            "{{\"handle\":\"{}\",\"prompt_tokens\":{},\"reusable_prefix_tokens\":{},\"expires_at_ms\":{},\"retained_bytes\":{},\"scope_retained_bytes\":{}}}",
            c.handle.expose(), c.prompt_tokens, c.reusable_prefix_tokens, c.expires_at_ms, c.retained_bytes, c.scope_retained_bytes))
            .unwrap_or_else(|| "null".into());
        format!("\"kv\":{{\"cache_domain\":\"{}\",\"reuse\":\"{}\",\"reused_tokens\":{},\"retention\":\"{}\",\"checkpoint\":{checkpoint}}}",
            self.cache_domain.expose(), self.reuse, self.reused_tokens, self.retention)
    }
}

/// Admitted wall time, split when prompt logits become available. Includes
/// scheduler waits and streaming backpressure, excludes queue/tokenization.
pub(crate) struct Clock {
    started: Instant,
    prefilled: Option<Instant>,
}
impl Clock {
    pub fn new(started: Instant) -> Self { Self { started, prefilled: None } }
    pub fn prefilled(&mut self) { self.prefilled.get_or_insert_with(Instant::now); }
    pub fn elapsed(&self) -> Timing {
        let now = Instant::now();
        let split = self.prefilled.unwrap_or(now);
        Timing { prefill: split.duration_since(self.started), decode: now.duration_since(split) }
    }
}

impl crate::generation::Usage {
    pub(crate) fn json(self, responses: bool) -> String {
        let (input, output) = if responses { ("input_tokens", "output_tokens") }
            else { ("prompt_tokens", "completion_tokens") };
        let details = format!("{{\"cached_tokens\":{}}}", self.cached_tokens);
        let alias = if responses { format!(",\"prompt_tokens_details\":{details}") } else { String::new() };
        let mut vade = Vec::new();
        if let Some(t) = self.timing { vade.push(format!(
            "\"prefill_ms\":{:.3},\"decode_ms\":{:.3},\"timing\":\"admitted_wall\"",
            t.prefill.as_secs_f64() * 1000.0, t.decode.as_secs_f64() * 1000.0)); }
        if let Some(kv) = self.kv { vade.push(kv.json()); }
        let vade = if vade.is_empty() { String::new() } else { format!(",\"vade\":{{{}}}", vade.join(",")) };
        format!("{{\"{input}\":{},\"{output}\":{},\"total_tokens\":{},\"{input}_details\":{details}{alias}{vade}}}",
            self.prompt_tokens, self.completion_tokens, self.prompt_tokens + self.completion_tokens)
    }
}
