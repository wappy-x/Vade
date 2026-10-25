//! Optional `vade.engine.v1` contract. Engine owns rendering, tokenization and KV.
//! This client compares opaque identities; it implements no tokenizer or crypto.
//! First-party mock fixtures mirror the separate Engine contract, not its code.
use crate::{
    json,
    transport::{self, ConnectionPool, Endpoint, invalid},
};
use std::{
    borrow::Cow,
    collections::VecDeque,
    io::{self, Read},
    sync::atomic::AtomicBool,
};

mod capabilities;
mod metadata;
mod usage;
pub(crate) use usage::UsageCollector;

pub const VERSION: &str = "vade.engine.v1";
/// Fallback only when the upstream does not advertise an output ceiling.
pub const DEFAULT_OUTPUT_TOKENS: u64 = 8192;
pub const AUTOMATIC_OUTPUT_TOKENS: u64 = 16_384;
pub const OUTPUT_SAFETY_TOKENS: u64 = 256;
/// Live native replies clamp to this range; guidance advertises the one
/// guaranteed session budget (2048 by default), never the automatic ceiling.
pub const NATIVE_REPLY_MIN_TOKENS: usize = 512;
pub const NATIVE_REPLY_MAX_TOKENS: usize = 4096;
/// The native reply size the contract guarantees before any capabilities are
/// known.
pub fn guaranteed_native_reply_tokens() -> usize { Contract::default().native_reply_tokens(None) }

/// Generation settings the user chose for native and direct requests. `None`
/// leaves the engine's default; no sampling field is sent unless set.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerationOptions {
    pub max_output_tokens: Option<u64>,
    /// Native reply cap (estimated tokens); zero disables the local budget.
    pub reply_tokens: Option<u64>,
    /// Local context ceiling; negotiated server limits may be smaller.
    pub context_limit: Option<u64>,
    pub thinking: Option<bool>,
    /// Qwen-style reasoning effort (low, medium, high, xhigh); implies thinking
    /// unless thinking is explicitly off. None leaves the Engine default.
    pub effort: Option<&'static str>,
    pub temperature: Option<f64>,
    pub seed: Option<u64>,
    /// Answer regeneration must not replay tools, even if advertised.
    pub disable_tools: bool,
}
const MAX_METADATA: usize = 64 * 1024;

/// Local owner-declared transcript boundary, never an Engine KV operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptBoundary {
    Compaction,
    Fork,
    Resume,
    Sampling,
    EngineBuild,
    ToolRecovery,
}
impl PromptBoundary {
    fn label(self) -> &'static str {
        match self {
            Self::Compaction => "compaction",
            Self::Fork => "transcript fork",
            Self::Resume => "resume/branch switch",
            Self::Sampling => "sampling change",
            Self::ToolRecovery => "cutoff tool recovery policy changed",
            Self::EngineBuild => "Engine build change; capabilities reconciled and prefix rebuilt",
        }
    }
}

/// The engine's reported weights, tokenizer and template fingerprints, used to
/// tell whether cached state and saved sessions still match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub weights_fingerprint: String,
    /// Engine-reported interpretation, never assumed to be a full payload hash.
    pub weights_fingerprint_kind: Option<String>,
    pub tokenizer_fingerprint: String,
    pub template_fingerprint: String,
}

/// Features are supported only when explicitly advertised by a recognized
/// contract. Unknown/null tool or thinking support is not affirmative support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub model: String,
    /// An explicit upstream build generation, distinct from cache load/domain.
    pub build_hash: Option<String>,
    pub identity: Option<Identity>,
    pub context_limit: u64,
    pub input_limit: u64,
    pub output_limit: u64,
    pub tools: bool,
    pub thinking: bool,
    pub default_thinking: bool,
    pub cache_domain: Option<String>,
    pub cache_scope: Option<String>,
    pub prefix_cache: bool,
    /// Smallest advertised reusable cache block, in rendered input tokens.
    /// Absent/null means unknown; this does not promise retention or a hit.
    pub cache_granularity_tokens: Option<u64>,
    pub prompt_cache_key: bool,
    pub exact_admission: bool,
    pub chat_completions: bool,
    pub kv_checkpoints: Option<crate::kv::Capabilities>,
}

fn number(value: json::View<'_>) -> io::Result<u64> {
    if value.raw().is_empty() || !value.raw().bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid("contract token count must be an unsigned integer"));
    }
    value
        .raw()
        .parse()
        .map_err(|_| invalid("contract token count overflow"))
}
fn count(value: json::View<'_>, name: &str) -> io::Result<u64> {
    number(
        value
            .get(name)
            .ok_or_else(|| invalid("contract token count missing"))?,
    )
}
fn equals(value: json::View<'_>, name: &str, expected: &str) -> bool {
    value
        .get(name)
        .and_then(json::View::as_str)
        .is_some_and(|s| s == expected)
}
/// Token usage one response reported. Missing values stay `None`, never zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// Actual provider-reported reuse, a subset of prompt_tokens, never a
    /// deduction from logical context occupancy. Missing differs from zero.
    pub cached_tokens: Option<u64>,
}

/// How the input size of a request is known: not at all, a rough byte
/// estimate, or exact counts from the engine's admission route.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum InputCount {
    #[default]
    Unknown,
    /// A deliberately rough wire-byte estimate with a fixed allowance. It is
    /// not an upper bound on arbitrary provider templates or an admission test.
    Estimate(u64),
    Exact {
        prompt_tokens: u64,
        output_tokens: u64,
        context_limit: u64,
        admitted: bool,
    },
}

/// Usage and client-observed timing for one model turn, numbered in order.
#[derive(Clone, Debug, Default)]
pub struct TurnUsage {
    pub sequence: u64,
    pub usage: Usage,
    pub invalid: bool,
    pub first_token_elapsed: Option<std::time::Duration>,
    pub decode_elapsed: Option<std::time::Duration>,
    pub prefill_elapsed: Option<std::time::Duration>,
    pub server_decode_elapsed: Option<std::time::Duration>,
    pub kv: Option<crate::kv::Report>,
    pub kv_invalid: bool,
}

/// Client-observed generated SSE deltas. Rates exclude the first token and
/// response trailers; missing usage or a single buffered delta has no rate.
#[derive(Clone, Debug, Default)]
pub struct SessionSpeed {
    ttft_total: std::time::Duration,
    ttft_responses: u64,
    decode_total: std::time::Duration,
    decode_tokens: u64,
    decode_responses: u64,
}
impl SessionSpeed {
    fn observe(&mut self, turn: &TurnUsage) {
        if let Some(ttft) = turn.first_token_elapsed {
            self.ttft_total = self.ttft_total.saturating_add(ttft);
            self.ttft_responses = self.ttft_responses.saturating_add(1);
        }
        if let Some((tokens, elapsed)) = turn.decode_sample() {
            self.decode_total = self.decode_total.saturating_add(elapsed);
            self.decode_tokens = self.decode_tokens.saturating_add(tokens);
            self.decode_responses = self.decode_responses.saturating_add(1);
        }
    }
    pub fn describe(&self) -> String {
        self.describe_as("Session average")
    }
    fn describe_as(&self, label: &str) -> String {
        let ttft = (self.ttft_responses > 0)
            .then(|| self.ttft_total.as_secs_f64() * 1000.0 / self.ttft_responses as f64);
        let rate = (!self.decode_total.is_zero())
            .then(|| self.decode_tokens as f64 / self.decode_total.as_secs_f64());
        format!(
            "{label}: TTFT {} · output {} (decode) · {} timed responses / {} decode samples",
            metric(ttft, "ms"),
            metric(rate, "tok/s"),
            self.ttft_responses,
            self.decode_responses
        )
    }
}

fn metric(value: Option<f64>, unit: &str) -> String {
    value.map_or_else(|| "unknown".into(), |n| format!("{n:.1} {unit}"))
}

/// Run totals are independent of the bounded per-call history, including compaction.
#[derive(Clone, Debug, Default)]
pub struct ReplySpeed {
    calls: u64,
    first: Option<std::time::Duration>,
    prefill_tokens: u64,
    prefill_seconds: f64,
    decode_tokens: u64,
    decode_seconds: f64,
    output: u64,
    missing_prefill: bool,
    missing_decode: bool,
    missing_output: bool,
    cache_prompt: u64,
    cache_restored: u64,
    missing_cache: bool,
    approximate: bool,
    pub wall: std::time::Duration,
}
impl ReplySpeed {
    fn observe(&mut self, turn: &TurnUsage) {
        if self.calls == 0 {
            self.first = turn.first_token_elapsed;
        }
        self.calls += 1;
        let usage = turn.usage;
        if let Some((prompt, restored)) = usage.prompt_tokens.zip(usage.cached_tokens).filter(|_| !turn.invalid) {
            self.cache_prompt = self.cache_prompt.saturating_add(prompt);
            self.cache_restored = self.cache_restored.saturating_add(restored);
        } else { self.missing_cache = true; }
        let prefill = usage
            .prompt_tokens
            .zip(usage.cached_tokens)
            .and_then(|(p, c)| p.checked_sub(c))
            .zip(turn.prefill_elapsed.or(turn.first_token_elapsed));
        if let Some((tokens, time)) = prefill
            .filter(|(_, t)| !t.is_zero())
            .filter(|_| !turn.invalid)
        {
            self.prefill_tokens = self.prefill_tokens.saturating_add(tokens);
            self.prefill_seconds += time.as_secs_f64();
            self.approximate |= turn.prefill_elapsed.is_none();
        } else {
            self.missing_prefill = true;
        }
        if let Some((tokens, time)) = turn.decode_sample() {
            self.decode_tokens = self.decode_tokens.saturating_add(tokens);
            self.decode_seconds += time.as_secs_f64();
        } else {
            self.missing_decode = true;
        }
        if let Some(tokens) = usage.completion_tokens.filter(|_| !turn.invalid) {
            self.output = self.output.saturating_add(tokens);
        } else {
            self.missing_output = true;
        }
    }
}

/// Cumulative request attempts and observed HTTP statuses. No retries, bodies,
/// endpoint identities or credentials are retained. Attempts include I/O errors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestCounts {
    pub capabilities: u64,
    pub admissions: u64,
    pub admitted: u64,
    pub refused: u64,
    pub generations: u64,
    pub http_429: u64,
    pub http_503: u64,
}
impl RequestCounts {
    pub(crate) fn status(&mut self, status: u16) {
        self.http_429 += u64::from(status == 429);
        self.http_503 += u64::from(status == 503);
    }
    pub fn since(self, before: Self) -> Self {
        Self {
            capabilities: self.capabilities.saturating_sub(before.capabilities),
            admissions: self.admissions.saturating_sub(before.admissions),
            admitted: self.admitted.saturating_sub(before.admitted),
            refused: self.refused.saturating_sub(before.refused),
            generations: self.generations.saturating_sub(before.generations),
            http_429: self.http_429.saturating_sub(before.http_429),
            http_503: self.http_503.saturating_sub(before.http_503),
        }
    }
}

/// Last attempted model request, not a count of an unsent next user prompt.
/// Retention is bounded to the latest 16 completed model responses per user run.
#[derive(Clone, Debug, Default)]
pub struct Context {
    /// Latest opt-in compaction warm, outside real answers/session usage totals.
    pub compaction_warm: Option<(Usage, std::time::Duration)>,
    pub capabilities: Option<Capabilities>,
    pub negotiation: &'static str,
    pub cache_domain_changes: u64,
    pub capability_reconciliations: u64,
    pub request_bytes: usize,
    pub input: InputCount,
    pub failure: Option<&'static str>,
    pub turns: VecDeque<TurnUsage>,
    pub completed_responses: u64,
    pub responses_with_cached_usage: u64,
    pub reported_cached_tokens: u64,
    pub prompt_boundary: Option<PromptBoundary>,
    pub prefix_pending: bool,
    pub requests: RequestCounts,
    /// The latest response used SSE. A JSON body cannot establish true TTFT.
    pub response_streamed: bool,
    /// Native request start to first generated SSE delta (text, reasoning or
    /// tool function data). Only meaningful after full response validation.
    pub first_token_elapsed: Option<std::time::Duration>,
    pub last_token_elapsed: Option<std::time::Duration>,
    pub session_speed: SessionSpeed,
    pub reply: ReplySpeed,
    pub last_http_status: Option<u16>,
    pub last_http_stage: &'static str,
}
impl Context {
    /// One run's footer, most useful facts first so narrow terminals keep
    /// elapsed time, tokens written and write speed. One number style.
    pub fn reply_speed(
        &self,
        elapsed: std::time::Duration,
        fallback_limit: u64,
        agent: bool,
    ) -> String {
        use crate::terminal::clock;
        let r = &self.reply;
        let rate = |tokens, seconds: f64, missing| {
            if missing || r.calls == 0 || seconds <= 0.0 {
                "—".into()
            } else {
                clock::rate(tokens, seconds)
            }
        };
        let used = self
            .turns
            .back()
            .filter(|t| !t.invalid)
            .and_then(|t| t.usage.prompt_tokens.zip(t.usage.completion_tokens))
            .map_or_else(|| "—".into(), |(p, c)| clock::count(p.saturating_add(c)));
        let limit = self
            .capabilities
            .as_ref()
            .map_or(fallback_limit, |c| c.context_limit);
        format!(
            "╰ {} · {} · {} tokens written · write {} · read {}{} · first token {}{} · ctx {used}/{} · cache reuse {}{}",
            clock::now().0,
            clock::duration(elapsed + r.wall),
            if r.missing_output || r.calls == 0 {
                "—".into()
            } else {
                clock::count(r.output)
            },
            rate(r.decode_tokens, r.decode_seconds, r.missing_decode),
            if r.approximate && !r.missing_prefill {
                "≈"
            } else {
                ""
            },
            rate(r.prefill_tokens, r.prefill_seconds, r.missing_prefill),
            r.first.map_or_else(|| "—".into(), clock::duration),
            if agent {
                format!(
                    " · {} model {}",
                    r.calls,
                    if r.calls == 1 { "call" } else { "calls" }
                )
            } else {
                String::new()
            },
            clock::count(limit),
            if r.missing_cache || r.cache_prompt == 0 { "—".into() }
            else { format!("{:.1}%", r.cache_restored as f64 * 100.0 / r.cache_prompt as f64) },
            self.capabilities.as_ref().map_or(" · prompt cache: unknown", |c| if c.prefix_cache { " · prompt cache: on" } else { " · prompt cache: off" })
        )
    }
    pub fn continue_reply(&mut self, previous: &Self, elapsed: std::time::Duration) {
        self.reply = previous.reply.clone();
        self.reply.wall += elapsed;
        for turn in &self.turns {
            self.reply.observe(turn);
        }
    }
    /// Latest validated response only. Missing measurements stay unknown;
    /// context includes that response's output, not the unsent next prompt.
    pub fn compact_status(&self, fallback_limit: u64) -> String {
        use crate::terminal::clock;
        let turn = self.turns.back();
        let first = turn.and_then(|t| t.first_token_elapsed).map_or_else(
            || "first token —".into(),
            |t| format!("first token {}", clock::duration(t)),
        );
        let rate = turn.and_then(TurnUsage::decode_sample).map_or_else(
            || "— tok/s".into(),
            |(n, t)| clock::rate(n, t.as_secs_f64()),
        );
        let limit = self
            .capabilities
            .as_ref()
            .map_or(fallback_limit, |c| c.context_limit);
        let usage = turn.filter(|t| !t.invalid).map(|t| t.usage);
        let used = usage
            .and_then(|u| u.prompt_tokens.zip(u.completion_tokens))
            .map_or_else(|| "—".into(), |(p, c)| clock::count(p.saturating_add(c)));
        let cached = usage
            .and_then(|u| u.cached_tokens.zip(u.prompt_tokens))
            .filter(|(_, p)| *p > 0)
            .map_or_else(
                || "cache —".into(),
                |(c, p)| format!("{:.0}% cached", c as f64 * 100.0 / p as f64),
            );
        format!("{first} · {rate} · ctx {used}/{} · {cached}", clock::count(limit))
    }
    pub fn run_speed(&self) -> String {
        let mut speed = SessionSpeed::default();
        for turn in &self.turns {
            speed.observe(turn);
        }
        format!(
            "{} · {} model responses",
            speed.describe_as("Run speed"),
            self.turns.len()
        )
    }
    pub fn describe(&self) -> String {
        let mut text = format!(
            "contract: {} · wire: chat completions\nlast request: {} wire bytes\n",
            if self.negotiation.is_empty() {
                "not negotiated; first prompt negotiates"
            } else {
                self.negotiation
            },
            self.request_bytes
        );
        if let Some(boundary) = self.prompt_boundary {
            text.push_str(&format!("prompt boundary: {} · {}\n", boundary.label(), if self.prefix_pending { "the next prompt resends the full conversation" } else { "the full conversation was resent; cache reuse shows in the next reply" }));
            if boundary == PromptBoundary::Compaction {
                text.push_str("Compaction invalidates later causal KV. Cached tokens may drop; reuse is only known from each new response, never from the old prefix or affinity.\n");
            }
        }
        if let Some((usage, elapsed)) = &self.compaction_warm {
            text.push_str(&format!("compaction pre-warm: {:.3} s completion wall · prompt {} · cached {} · output {} tokens · discarded reply; no tools executed\n", elapsed.as_secs_f64(), usage.prompt_tokens.map_or_else(|| "unknown".into(), |n| n.to_string()), usage.cached_tokens.map_or_else(|| "unknown".into(), |n| n.to_string()), usage.completion_tokens.map_or_else(|| "unknown".into(), |n| n.to_string())));
        }
        match self.input {
            InputCount::Unknown => text.push_str("input tokens: unknown\n"),
            InputCount::Estimate(n) => text.push_str(&format!("input tokens: estimate {n} (wire bytes + 256 allowance; not an admission guarantee)\n")),
            InputCount::Exact { prompt_tokens, output_tokens, context_limit, admitted } => text.push_str(&format!("input tokens: exact {prompt_tokens} + output reservation {output_tokens} / context limit {context_limit} · context fit: {} (no resource reservation)\n", if admitted { "admitted" } else { "refused" })),
        }
        if let Some(caps) = &self.capabilities {
            text.push_str(&format!("capabilities: tools={} thinking={} prefix_cache={} affinity={} · cache domain changes={}\n", caps.tools, caps.thinking, caps.prefix_cache, caps.prompt_cache_key, self.cache_domain_changes));
            if self.capability_reconciliations > 0 {
                text.push_str("Engine build changed; capabilities reconciled and frozen request prefix rebuilt.\n");
            }
            if let Some(kv) = &caps.kv_checkpoints {
                text.push_str(&format!("KV checkpoints: advertised · scope quota {} handles / {} bytes · TTL {} ms; complete prompts always sent\n", kv.max_handles_per_scope, kv.max_bytes_per_scope, kv.ttl_ms));
            }
            text.push_str(if caps.identity.is_some() { "model/tokenizer/template fingerprints: reported (opaque); cache identity is Engine-owned\n" } else { "model/tokenizer/template fingerprints: unknown\n" });
        }
        if let Some(failure) = self.failure {
            text.push_str(failure);
            text.push('\n');
        }
        if self.turns.is_empty() {
            text.push_str("last model response: cached tokens unknown\n");
        }
        for turn in &self.turns {
            text.push_str(&turn.describe());
            text.push('\n');
        }
        text.push_str(&format!("session reported cached tokens: {} across {}/{} completed model responses with telemetry; cached input still occupies context", self.reported_cached_tokens, self.responses_with_cached_usage, self.completed_responses));
        text
    }
}
impl TurnUsage {
    fn decode_sample(&self) -> Option<(u64, std::time::Duration)> {
        if self.invalid {
            return None;
        }
        let tokens = self.usage.completion_tokens?.checked_sub(1)?;
        let elapsed = self.server_decode_elapsed.or(self.decode_elapsed)?;
        (tokens > 0 && !elapsed.is_zero()).then_some((tokens, elapsed))
    }
    pub fn speed(&self) -> String {
        let count = |value: Option<u64>| value.map_or_else(|| "unknown".into(), |n| n.to_string());
        let rate = self
            .decode_sample()
            .map(|(tokens, elapsed)| tokens as f64 / elapsed.as_secs_f64());
        let prefill = self.usage.prompt_tokens.zip(self.usage.cached_tokens)
            .and_then(|(p, c)| p.checked_sub(c))
            .zip(self.prefill_elapsed.or(self.first_token_elapsed))
            .filter(|(_, t)| !t.is_zero() && !self.invalid)
            .map(|(n, t)| n as f64 / t.as_secs_f64());
        format!(
            "Response speed: TTFT {} · output {} (decode) · prompt {} tokens · cached {} tokens · prefill {}{}",
            metric(
                self.first_token_elapsed.map(|d| d.as_secs_f64() * 1000.0),
                "ms"
            ),
            metric(rate, "tok/s"),
            count(self.usage.prompt_tokens),
            count(self.usage.cached_tokens),
            if prefill.is_some() && self.prefill_elapsed.is_none() { "≈" } else { "" },
            prefill.map_or_else(|| "—".into(), |n| format!("{n:.1} tok/s"))
        )
    }
    pub fn describe(&self) -> String {
        let n = |v: Option<u64>| v.map_or_else(|| "unknown".into(), |v| v.to_string());
        let mut text = format!(
            "model response {}: prompt tokens {} · output tokens {} · cached tokens {}{}",
            self.sequence,
            n(self.usage.prompt_tokens),
            n(self.usage.completion_tokens),
            n(self.usage.cached_tokens),
            if self.invalid {
                " (invalid/conflicting telemetry)"
            } else {
                ""
            }
        );
        if let Some(kv) = &self.kv {
            text.push('\n');
            text.push_str(&kv.describe());
        } else if self.kv_invalid {
            text.push_str("\nKV handle: unknown (invalid/conflicting telemetry)");
        }
        text
    }
}

#[derive(Default)]
pub(crate) struct Contract {
    pub context: Context,
    pub options: GenerationOptions,
    checked: bool,
    affinity: Option<String>,
    // Freeze the advertised ceiling in the prefix; a smaller reservation is
    // applied to the outgoing generation controls after current-input admission.
    default_output: Option<u64>,
    chosen_output: Option<u64>,
    automatic_reservation: bool,
    snapshot_capabilities: bool,
    pub(crate) tool_recovery: bool,
}
impl Contract {
    pub(crate) fn native(options: GenerationOptions) -> Self {
        Self {
            options,
            snapshot_capabilities: true,
            ..Default::default()
        }
    }
    pub(crate) fn with_options(options: GenerationOptions) -> Self {
        Self {
            options,
            ..Default::default()
        }
    }
    /// Admission-only observation. Does not mutate a generation client's state,
    /// reserve resources, create KV, or dispatch inference.
    pub(crate) fn count_request(
        caps: &Capabilities,
        endpoint: &Endpoint,
        body: &[u8],
        pool: &ConnectionPool,
        cancel: &AtomicBool,
    ) -> (io::Result<InputCount>, Context) {
        let mut probe = Self::default();
        probe.context.capabilities = Some(caps.clone());
        let result = probe
            .admit(endpoint, body, pool, Some(cancel))
            .map(|()| probe.context.input.clone());
        (result, probe.context)
    }
    pub(crate) fn reset_prefix(&mut self, boundary: PromptBoundary) {
        // Preserve same-session affinity, but no old request count or cached
        // response may be presented as telemetry for the replacement prefix.
        // Native transcript rewrites preserve the session capability snapshot.
        // Direct clients retain their existing explicit renegotiation behavior.
        if !self.snapshot_capabilities {
            self.checked = false;
        }
        self.context.prompt_boundary = Some(boundary);
        self.context.prefix_pending = true;
        self.reset_attempt();
        self.context.turns.clear();
        self.context.compaction_warm = None;
    }
    pub(crate) fn begin_run(&mut self) {
        self.context.turns.clear();
        self.context.reply = ReplySpeed::default();
    }

    pub(crate) fn reset_attempt(&mut self) {
        self.context.failure = None;
        self.context.input = InputCount::Unknown;
        self.context.request_bytes = 0;
        self.context.response_streamed = false;
        self.context.first_token_elapsed = None;
        self.context.last_token_elapsed = None;
        self.context.last_http_status = None;
        self.context.last_http_stage = "none";
    }

    pub(crate) fn negotiate(
        &mut self,
        endpoint: &Endpoint,
        model: &str,
        pool: &ConnectionPool,
        cancel: Option<&AtomicBool>,
    ) -> io::Result<()> {
        self.reset_attempt();
        if self.checked && self.context.capabilities.is_none() {
            return Ok(());
        }
        let Some(discovery) = sibling(endpoint, "capabilities") else {
            self.checked = true;
            self.context.negotiation = "unavailable on custom route";
            return Ok(());
        };
        self.context.last_http_stage = "capabilities";
        let (response, bytes) = self.metadata("GET", &discovery, &[], pool, cancel)?;
        let status = response.status;
        let json_metadata = response.head.get("content-type").is_some_and(|s| {
            s.split(';').next().unwrap_or("").trim().eq_ignore_ascii_case("application/json")
        });
        let unavailable = matches!(status, 404 | 405 | 501);
        if !unavailable && status != 200 {
            self.context.failure = Some("Could not read the engine's settings; nothing was sent to the model.");
            return Err(io::Error::other("capability discovery failed"));
        }
        let mut caps = if unavailable || !json_metadata {
            None
        } else {
            Capabilities::parse(
                std::str::from_utf8(&bytes).map_err(|_| invalid("invalid capability encoding"))?,
            )?
        };
        response.recycle();
        let mut reconcile = false;
        if self.snapshot_capabilities && self.checked {
            if let (Some(old), Some(next)) = (&self.context.capabilities, &caps) {
                reconcile = old.identity != next.identity
                    || old.cache_domain != next.cache_domain
                    || old.kv_checkpoints.as_ref().map(|k| &k.cache_domain)
                        != next.kv_checkpoints.as_ref().map(|k| &k.cache_domain)
                    || (next.build_hash.is_some() && next.build_hash != old.build_hash);
                if !reconcile {
                    // Freeze decoration until the Engine reports a new identity,
                    // cache epoch, or explicit build; build_hash is optional.
                    caps = Some(old.clone());
                }
            }
        }
        if let Some(caps) = &caps {
            if self
                .options
                .max_output_tokens
                .is_some_and(|n| n == 0 || n > caps.output_limit)
            {
                self.context.failure = Some(
                    "--max-output-tokens must be between 1 and the engine's output limit; nothing was sent.",
                );
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "output token limit exceeded",
                ));
            }
            if self.options.thinking == Some(true) && !caps.thinking {
                self.context.failure =
                    Some("--thinking on needs an engine that supports thinking; nothing was sent.");
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "thinking unsupported",
                ));
            }
            if caps.model != model || !caps.chat_completions {
                self.context.failure = Some(
                    "The engine is serving a different model than the one selected; nothing was sent.",
                );
                return Err(invalid("capability model or dialect mismatch"));
            }
            if let Some(old) = &self.context.capabilities {
                if old.identity != caps.identity
                    || old.cache_domain != caps.cache_domain
                    || old.cache_scope != caps.cache_scope
                    || old.default_thinking != caps.default_thinking
                    || old.kv_checkpoints.as_ref().map(|k| &k.cache_domain)
                        != caps.kv_checkpoints.as_ref().map(|k| &k.cache_domain)
                {
                    self.context.cache_domain_changes =
                        self.context.cache_domain_changes.saturating_add(1);
                }
                if !reconcile
                    && (old.prompt_cache_key != caps.prompt_cache_key
                        || old.exact_admission != caps.exact_admission)
                {
                    self.context.failure =
                        Some("negotiated request features changed; use /new before generation");
                    return Err(invalid("negotiated features changed"));
                }
            }
            if reconcile {
                self.default_output = None;
                self.context.capability_reconciliations =
                    self.context.capability_reconciliations.saturating_add(1);
            }
            if caps.prompt_cache_key && self.affinity.is_none() {
                // OS entropy, no custom crypto; this is only a non-secret scheduling
                // hint, never authorization, durable identity, or proof of a hit.
                let mut bytes = [0u8; 16];
                std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
                let mut key = String::from("vade-");
                for byte in bytes {
                    use std::fmt::Write;
                    write!(key, "{byte:02x}").unwrap();
                }
                self.affinity = Some(key);
            }
            self.default_output
                .get_or_insert(caps.output_limit.min(AUTOMATIC_OUTPUT_TOKENS));
            self.context.negotiation = VERSION;
        } else if self.context.capabilities.is_some() {
            self.context.failure = Some("Lost the engine's settings for this chat; use /new to start over.");
            return Err(invalid("negotiated contract disappeared"));
        } else {
            self.context.negotiation = if unavailable {
                "unsupported; standard chat fallback"
            } else {
                "unrecognized metadata/version; standard chat fallback"
            };
        }
        self.context.capabilities = caps;
        self.checked = true;
        Ok(())
    }

    /// Insert negotiated fixed options without reserializing any caller bytes.
    /// Works on options-only objects and complete direct requests alike.
    pub(crate) fn decorate<'a>(&self, original: &'a str) -> io::Result<Cow<'a, str>> {
        let caps = self.context.capabilities.as_ref();
        let value = json::view(original)?;
        let mut extra = String::new();
        if value.get("temperature").is_none() {
            if let Some(temperature) = self.options.temperature {
                if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
                    return Err(invalid("temperature must be finite and between 0 and 2"));
                }
                extra.push_str(&format!("\"temperature\":{temperature},"));
            }
        }
        if value.get("seed").is_none() {
            if let Some(seed) = self.options.seed {
                extra.push_str(&format!("\"seed\":{seed},"));
            }
        }
        // Recovery changes only an omitted tool policy, never an explicit
        // caller policy or sampling controls. Plain chat/fallback stays intact.
        if self.tool_recovery && !self.options.disable_tools
            && caps.is_some_and(|caps| caps.tools)
            && value.get("tool_choice").is_none()
            && value.get("tools").and_then(|v| v.array()).is_some_and(|mut tools| tools.next().is_some())
        {
            extra.push_str("\"tool_choice\":\"required\",");
        }
        if self.options.disable_tools
            && value.get("tools").is_some()
            && value.get("tool_choice").is_none()
        {
            extra.push_str("\"tool_choice\":\"none\",");
        }
        if caps.is_some_and(|c| c.prefix_cache || c.prompt_cache_key) && value.get("cache").is_none() {
            let enabled = !std::env::var_os("VADE_HARNESS_PREFIX_CACHE").is_some_and(|v| v == "0");
            extra.push_str(&format!("\"cache\":{enabled},"));
        }
        if caps.is_some_and(|c| c.prompt_cache_key) && value.get("prompt_cache_key").is_none() {
            extra.push_str("\"prompt_cache_key\":");
            json::write_string(self.affinity.as_deref().unwrap(), &mut extra);
            extra.push(',');
        }
        if value
            .get("max_completion_tokens")
            .or_else(|| value.get("max_tokens"))
            .is_none()
        {
            let output = self
                .options
                .max_output_tokens
                .or(self.default_output.map(|n| n.min(AUTOMATIC_OUTPUT_TOKENS)))
                .unwrap_or(DEFAULT_OUTPUT_TOKENS);
            if output == 0 {
                return Err(invalid("output token limit must be positive"));
            }
            extra.push_str(&format!("\"max_tokens\":{output},"));
        }
        // Hybrid-thinking models spend the whole reply budget reasoning when the
        // engine defaults thinking on. Explicit options override the env fallback.
        // Without an effort the Engine uses the model default (xhigh for
        // Qwen3.8) whenever thinking is on; an explicit effort turns it on.
        let effort = self
            .options
            .effort
            .filter(|_| caps.is_some_and(|c| c.thinking));
        if value.get("enable_thinking").is_none() {
            if let Some(thinking) = self
                .options
                .thinking
                .or_else(|| effort.map(|_| true))
                .or_else(|| {
                    caps.filter(|c| c.thinking).map(|_| {
                        std::env::var_os("VADE_HARNESS_THINKING").is_some_and(|v| v == "1")
                    })
                })
            {
                extra.push_str(&format!("\"enable_thinking\":{thinking},"));
            }
        }
        if let Some(effort) = effort.filter(|_| value.get("reasoning_effort").is_none()) {
            extra.push_str(&format!("\"reasoning_effort\":\"{effort}\","));
        }
        if extra.is_empty() {
            return Ok(Cow::Borrowed(original));
        }
        let tail = original
            .strip_prefix('{')
            .ok_or_else(|| invalid("request options must be an object"))?;
        if tail.trim() == "}" {
            extra.pop();
        }
        if original.len() + extra.len() > json::MAX_INPUT_BYTES {
            return Err(invalid("decorated request exceeds limit"));
        }
        Ok(Cow::Owned(format!("{{{extra}{tail}")))
    }

    pub(crate) fn automatic_output(&self, original: &str) -> io::Result<bool> {
        let value = json::view(original)?;
        Ok(self.options.max_output_tokens.is_none()
            && value.get("max_tokens").is_none()
            && value.get("max_completion_tokens").is_none())
    }

    pub(crate) fn native_reply_time(
        &self,
        remaining: Option<std::time::Duration>,
    ) -> Option<std::time::Duration> {
        if self.options.reply_tokens == Some(0) {
            return None;
        }
        let ceiling = std::time::Duration::from_secs(120);
        // Keep a 10 s repair margin after this reply; inside that bound a
        // late reply still gets up to 30 s instead of collapsing as left/8.
        let reserve = std::time::Duration::from_secs(10);
        Some(remaining.map_or(ceiling, |left| (left / 8).max(std::time::Duration::from_secs(30).min(left.saturating_sub(reserve))).min(ceiling)))
    }

    pub(crate) fn native_reply_tokens(&self, remaining: Option<std::time::Duration>) -> usize {
        let Some(time) = self.native_reply_time(remaining) else {
            return usize::MAX;
        };
        let speed = &self.context.session_speed;
        let rate = if speed.decode_total.is_zero() {
            2048.0 / 120.0
        } else {
            speed.decode_tokens as f64 / speed.decode_total.as_secs_f64()
        };
        let default = (rate * time.as_secs_f64()).clamp(NATIVE_REPLY_MIN_TOKENS as f64, NATIVE_REPLY_MAX_TOKENS as f64) as u64;
        usize::try_from(self.options.reply_tokens.unwrap_or(default)).unwrap_or(usize::MAX)
    }

    pub(crate) fn reply_limit_tokens(&self, fallback: usize) -> usize {
        self.chosen_output.or(self.options.max_output_tokens)
            .map_or(fallback, |n| usize::try_from(n).unwrap_or(usize::MAX))
    }
    pub(crate) fn output_reserve_tokens(&self, fallback: usize) -> usize {
        let reserve = self.chosen_output
            .or(self.options.max_output_tokens)
            .map_or(fallback, |n| usize::try_from(n).unwrap_or(usize::MAX));
        if self.automatic_reservation {
            let ceiling = self.context.capabilities.as_ref()
                .map_or(DEFAULT_OUTPUT_TOKENS, |caps| caps.output_limit);
            let context = self.context.token_limit(self.options.context_limit);
            reserve.max(crate::context::reply_reservation(
                usize::try_from(context).unwrap_or(usize::MAX), usize::try_from(ceiling).unwrap_or(usize::MAX)))
        } else {
            reserve
        }
    }

    /// Probe the complete current request, then lower only our default output
    /// control when the exact prompt needs the remaining context. No inference
    /// is sent by either admission pass. Explicit caller output limits stay exact.
    pub(crate) fn admit_adjusted(
        &mut self,
        endpoint: &Endpoint,
        body: &[u8],
        pool: &ConnectionPool,
        cancel: Option<&AtomicBool>,
        automatic: bool,
    ) -> io::Result<Vec<u8>> {
        let mut body = body.to_vec();
        let ceiling = self
            .context
            .capabilities
            .as_ref()
            .map_or(DEFAULT_OUTPUT_TOKENS, |caps| caps.output_limit);
        let context = self.context.token_limit(self.options.context_limit);
        let exact = self
            .context
            .capabilities
            .as_ref()
            .is_some_and(|caps| caps.exact_admission);
        self.context.input = InputCount::Unknown;
        self.chosen_output = None;
        self.automatic_reservation = automatic;
        let mut result = if exact || !automatic {
            self.admit(endpoint, &body, pool, cancel)
        } else {
            Ok(())
        };
        if automatic {
            let prompt = match self.context.input {
                InputCount::Exact { prompt_tokens, .. } => prompt_tokens,
                _ if exact => return result.map(|()| body),
                _ => crate::context::estimated_request_tokens(
                    std::str::from_utf8(&body).map_err(|_| invalid("request encoding"))?,
                )? as u64,
            };
            let output = ceiling.min(AUTOMATIC_OUTPUT_TOKENS).min(
                context
                    .saturating_sub(prompt)
                    .saturating_sub(OUTPUT_SAFETY_TOKENS),
            );
            if output == 0 {
                if let InputCount::Exact { admitted, .. } = &mut self.context.input {
                    *admitted = false;
                }
                self.context.failure = Some(
                    "The chat is too long for another reply; nothing was sent. Use /compact or /new.",
                );
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "no output room in context",
                ));
            }
            self.chosen_output = Some(output);
            let request = std::str::from_utf8(&body).map_err(|_| invalid("request encoding"))?;
            let current = json::view(request)?
                .get("max_tokens")
                .ok_or_else(|| invalid("missing automatic output reservation"))?;
            if number(current)? != output {
                // View borrows the original JSON. Replace only the inserted
                // number; historical messages and tools retain their exact bytes.
                let start = current.raw().as_ptr() as usize - request.as_ptr() as usize;
                let end = start + current.raw().len();
                body.splice(start..end, output.to_string().bytes());
                result = self.admit(endpoint, &body, pool, cancel);
            } else if !exact {
                result = self.admit(endpoint, &body, pool, cancel);
            }
        } else if result.is_ok() {
            let request = std::str::from_utf8(&body).map_err(|_| invalid("request encoding"))?;
            let value = json::view(request)?;
            let output = number(
                value
                    .get("max_completion_tokens")
                    .or_else(|| value.get("max_tokens"))
                    .ok_or_else(|| invalid("missing output reservation"))?,
            )?;
            let prompt = match self.context.input {
                InputCount::Exact { prompt_tokens, .. } => prompt_tokens,
                _ => crate::context::estimated_request_tokens(request)? as u64,
            };
            self.chosen_output = Some(output);
            if prompt > context || output > context.saturating_sub(prompt) {
                if let InputCount::Exact { admitted, .. } = &mut self.context.input {
                    *admitted = false;
                }
                self.context.failure = Some(
                    "The reply length you asked for does not fit; nothing was sent. Use /compact or lower --max-output-tokens.",
                );
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "requested output exceeds context budget",
                ));
            }
        }
        result.map(|()| body)
    }

    pub(crate) fn admit(
        &mut self,
        endpoint: &Endpoint,
        body: &[u8],
        pool: &ConnectionPool,
        cancel: Option<&AtomicBool>,
    ) -> io::Result<()> {
        self.context.request_bytes = body.len();
        self.context.prefix_pending = false;
        let Some(caps) = self.context.capabilities.clone() else {
            self.context.input = InputCount::Estimate((body.len() as u64).saturating_add(256));
            return Ok(());
        };
        let request = std::str::from_utf8(body).map_err(|_| invalid("request encoding"))?;
        let value = json::view(request)?;
        let has_tools = value.get("tools").is_some_and(|v| {
            v.raw() != "null" && !v.array().is_some_and(|mut a| a.next().is_none())
        });
        if has_tools && !caps.tools {
            self.context.failure =
                Some("The selected model does not support tools; nothing was sent.");
            return Err(invalid("model tools unsupported"));
        }
        if !caps.exact_admission {
            self.context.input = InputCount::Estimate((body.len() as u64).saturating_add(256));
            return Ok(());
        }
        let output = number(
            value
                .get("max_completion_tokens")
                .or_else(|| value.get("max_tokens"))
                .ok_or_else(|| invalid("missing output reservation"))?,
        )?;
        let envelope = format!(
            "{{\"contract_version\":\"{VERSION}\",\"endpoint\":\"/v1/chat/completions\",\"output_tokens\":{output},\"request\":{request}}}"
        );
        let admission =
            sibling(endpoint, "admission").ok_or_else(|| invalid("admission route unavailable"))?;
        self.context.failure = Some("The engine could not check whether the chat fits; nothing was sent.");
        self.context.last_http_stage = "admission";
        let (response, bytes) = self.metadata("POST", &admission, envelope.as_bytes(), pool, cancel)?;
        if response.status != 200 {
            return Err(io::Error::other("exact admission HTTP failure"));
        }
        let v = json::view(
            std::str::from_utf8(&bytes).map_err(|_| invalid("invalid admission encoding"))?,
        )?;
        if !equals(v, "object", "engine.admission")
            || !equals(v, "contract_version", VERSION)
            || !equals(v, "model", &caps.model)
            || !equals(v, "admission_kind", "context_fit")
            || v.get("reserves_resources")
                .is_none_or(|v| v.raw() != "false")
            || count(v, "output_tokens")? != output
        {
            return Err(invalid("admission contract mismatch"));
        }
        let prompt = count(v, "prompt_tokens")?;
        let context = count(v, "context_limit")?;
        let input = count(v, "input_limit")?;
        let limit = count(v, "output_limit")?;
        if context != caps.context_limit || input != caps.input_limit || limit != caps.output_limit
        {
            return Err(invalid("admission limits changed since negotiation"));
        }
        let admitted = match v.get("admitted").map(json::View::raw) {
            Some("true") => true,
            Some("false") => false,
            _ => return Err(invalid("admission decision missing")),
        };
        if admitted
            && (output == 0
                || output > limit
                || prompt > input
                || prompt.checked_add(output).is_none_or(|n| n > context))
        {
            return Err(invalid("inconsistent admission decision"));
        }
        self.context.input = InputCount::Exact {
            prompt_tokens: prompt,
            output_tokens: output,
            context_limit: context,
            admitted,
        };
        response.recycle();
        if !admitted {
            self.context.requests.refused += 1;
            self.context.failure = Some(
                "The chat does not fit the model's context; nothing was sent. Use /compact or /new.",
            );
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "exact context admission refused",
            ));
        }
        self.context.requests.admitted += 1;
        self.context.failure = None;
        Ok(())
    }

    pub(crate) fn completed(&mut self, mut collector: UsageCollector) {
        let (kv, kv_invalid) = std::mem::take(&mut collector.kv).finish();
        let prefill_elapsed = collector.prefill_elapsed;
        let server_decode_elapsed = collector.server_decode_elapsed;
        let (usage, invalid) = collector.finish();
        self.context.completed_responses = self.context.completed_responses.saturating_add(1);
        if let Some(cached) = usage.cached_tokens {
            self.context.responses_with_cached_usage =
                self.context.responses_with_cached_usage.saturating_add(1);
            self.context.reported_cached_tokens =
                self.context.reported_cached_tokens.saturating_add(cached);
        }
        if self.context.turns.len() == 16 {
            self.context.turns.pop_front();
        }
        let turn = TurnUsage {
            sequence: self.context.completed_responses,
            usage,
            invalid,
            first_token_elapsed: self.context.first_token_elapsed,
            decode_elapsed: self
                .context
                .first_token_elapsed
                .zip(self.context.last_token_elapsed)
                .and_then(|(first, last)| last.checked_sub(first)),
            prefill_elapsed,
            server_decode_elapsed,
            kv,
            kv_invalid,
        };
        self.context.session_speed.observe(&turn);
        self.context.reply.observe(&turn);
        self.context.turns.push_back(turn);
    }
}

/// Only probe recognized chat routes, on the configured origin and base path.
/// Never follow a advertised URL to a different host or reinterpret a custom API.
pub(crate) fn sibling(endpoint: &Endpoint, leaf: &str) -> Option<Endpoint> {
    let (path, query) = endpoint
        .path
        .split_once('?')
        .map_or((endpoint.path.as_str(), None), |(p, q)| (p, Some(q)));
    let base = path.strip_suffix("/chat/completions")?;
    let mut result = endpoint.clone();
    result.path = format!("{base}/{leaf}");
    if let Some(q) = query {
        result.path.push('?');
        result.path.push_str(q);
    }
    Some(result)
}

#[cfg(test)]
mod tests;
