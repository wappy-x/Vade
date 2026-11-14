//! Bounded diagnostics against an explicitly selected HTTP endpoint.
//! Only synthetic prompts; no workspace, shell, config discovery, or real tools.
use crate::{
    agent::{self, Conversation, Limits, StopReason, ToolError, ToolExecutor, ToolInvocation},
    agent_wire::{FinishReason, WireLimits},
    contract::{Capabilities, Context, Contract, InputCount, Usage, VERSION},
    json::{self, Value},
    setup,
    transport::{self, ConnectionPool, Endpoint, NativeModel},
};
use std::{
    io::{self, Read},
    process::ExitCode,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

pub const HELP: &str = "Vade endpoint doctor (active HTTP probe; synthetic data only)

Usage: vade-harness doctor URL [--model NAME] [--max-seconds N]

  URL                  HTTP origin, /v1 base, or known chat/responses route
  --model NAME         Required when capabilities cannot identify the model;
                       otherwise validates the advertised model
  --max-seconds N       Whole probe budget, 1..300 (default 60)
  -h, --help           Show this help

Reads capabilities and, only for an identified Vade gateway, cluster stats.
Sizes a synthetic prefix above two advertised cache blocks (or 8192 tokens),
checking exact admission when available, then sends two bounded
Chat Completions turns in one conversation, using advertised exact admission.
No tools execute. Tool/thinking support is advertised, not behaviorally tested.
The second request retains the first prompt and full validated assistant answer.
With prefix caching advertised, PASS requires valid turn-2 cached_tokens > 0.
Explicit prefix_cache=false reports reuse as WARN engine-pending instead;
capability/admission/chat failures and invalid telemetry still fail the doctor.
Only the selected route/conversation is checked, not every cluster node or the
Responses API. Reuse percentage is cached / turn-2 prompt tokens.
Without exact admission, probe token size remains unverified.
Credentials: --api-key-file PATH or VADE_API_KEY_FILE. HTTPS fails explicitly; no TLS support.
Exit 0: checks passed (cache may WARN when explicitly disabled by the server).
Exit 1: failed/incomplete check; exit 2: invalid command. No automatic retries.
OS DNS resolution and an in-progress connect may outlast the probe budget.
";

const METADATA_LIMIT: usize = 64 * 1024;
#[derive(Debug)]
struct Config {
    base: String,
    model: Option<String>,
    seconds: u64,
}
impl Config {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        crate::diagnostics::parse_command("doctor", |option| Self::parse_options(args, option))
    }
    fn parse_options(args: impl IntoIterator<Item = String>, option: &mut Option<String>) -> Result<Self, &'static str> {
        let mut args = args.into_iter();
        let url = args.next().ok_or("doctor needs an HTTP URL; see doctor --help")?;
        // Other modes take --endpoint; doctor takes the URL as its first word.
        if url.starts_with('-') {
            return Err("doctor takes the API URL first, as in vade-harness doctor http://127.0.0.1:18080/v1 --model ID");
        }
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err("the doctor URL must begin with http://");
        }
        let base = setup::api_base(&url)?;
        let mut model = None;
        // Flash can need about 30 seconds for the cold 11k-token probe alone.
        // Leave time for metadata, exact admission, and the cached second turn.
        let mut seconds = 60;
        let mut seen = Vec::new();
        while let Some(flag) = args.next() {
            *option = crate::diagnostics::option_name(&flag).map(str::to_owned);
            if seen.contains(&flag) {
                return Err("duplicate doctor option");
            }
            let value = args.next().ok_or("doctor option needs a value")?;
            match flag.as_str() {
                "--model" if setup::valid_model(&value) => model = Some(value),
                "--model" => return Err("doctor model must be a literal model ID"),
                "--max-seconds" => {
                    seconds = value
                        .parse()
                        .map_err(|_| "doctor deadline must be 1..300 seconds")?;
                    if !(1..=300).contains(&seconds) {
                        return Err("doctor deadline must be 1..300 seconds");
                    }
                }
                _ => return Err("unknown doctor option; see doctor --help"),
            }
            seen.push(flag);
        }
        Ok(Self {
            base,
            model,
            seconds,
        })
    }
}

struct Metadata {
    status: u16,
    body: Option<String>,
    gateway: bool,
}
fn get(endpoint: &Endpoint, cancel: &AtomicBool, deadline: Instant) -> io::Result<Metadata> {
    let pool = ConnectionPool::default();
    let mut response = transport::request_method_pooled_deadline(
        "GET",
        endpoint,
        &[],
        None,
        Some(cancel),
        &pool,
        Some(deadline),
    )?;
    let status = response.status;
    let gateway =
        response.head.get(crate::gateway::MARKER_HEADER) == Some(crate::gateway::MARKER_VALUE);
    // Do not display remote error text or parse an HTML error page as metadata.
    if status != 200 {
        return Ok(Metadata {
            status,
            body: None,
            gateway,
        });
    }
    let is_json = response.head.get("content-type").is_some_and(|kind| {
        kind.split(';')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case("application/json")
    });
    if !is_json {
        return Err(transport::invalid("metadata is not JSON"));
    }
    let mut bytes = Vec::new();
    response
        .body
        .by_ref()
        .take((METADATA_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > METADATA_LIMIT {
        return Err(transport::invalid("metadata exceeds limit"));
    }
    let body = String::from_utf8(bytes).map_err(|_| transport::invalid("metadata encoding"))?;
    json::view(&body)?;
    Ok(Metadata {
        status,
        body: Some(body),
        gateway,
    })
}

#[derive(Default)]
struct Report {
    lines: String,
    caps: Option<Capabilities>,
    turns: Vec<Turn>,
    failed: bool,
    gateway: bool,
    probe_target: u64,
    cache_disabled: bool,
}
struct Turn {
    complete: bool,
    usage: Usage,
    invalid_usage: bool,
    prompt_tokens: Option<u64>,
}
impl Report {
    fn line(&mut self, line: impl AsRef<str>) {
        self.lines.push_str(line.as_ref());
        self.lines.push('\n');
    }
    fn failure(&mut self, line: impl AsRef<str>) {
        self.failed = true;
        self.line(line);
    }
    fn passed(&self) -> bool {
        !self.failed
            && self.turns.len() == 2
            && self.turns.iter().all(|t| t.complete && !t.invalid_usage)
            && (self.cache_disabled || self.turns[1].usage.cached_tokens.is_some_and(|n| n > 0))
    }
    fn finish(&mut self) {
        let reported = self
            .turns
            .iter()
            .filter(|t| t.complete && !t.invalid_usage && t.usage.cached_tokens.is_some())
            .count();
        self.line(format!(
            "Cached-token reporting: {reported}/{} attempted turns supplied valid observed counts",
            self.turns.len()
        ));
        if self.turns.len() != 2 || self.turns.iter().any(|t| !t.complete) {
            self.line("Two-turn cache check: INCOMPLETE (two validated generations required)");
        } else if self.turns.iter().any(|t| t.invalid_usage) {
            self.line("Two-turn cache check: UNKNOWN (invalid/conflicting token telemetry)");
        } else {
            match self.turns[1].usage.cached_tokens {
                Some(n) => {
                    let reuse = self.turns[1].prompt_tokens.filter(|p| *p > 0).map_or_else(
                        || "unknown (prompt count absent)".into(),
                        |p| {
                            format!(
                                "{:.2}% ({n}/{p} prompt tokens)",
                                n as f64 * 100.0 / p as f64
                            )
                        },
                    );
                    let below_target = self.turns[0]
                        .prompt_tokens
                        .is_some_and(|p| p < self.probe_target);
                    self.line(format!(
                        "Two-turn cache check: {} (turn 2 cached_tokens={n}; reuse={reuse})",
                        if self.cache_disabled {
                            "WARN engine-pending (prefix_cache=false)"
                        } else if n > 0 {
                            "PASS"
                        } else if below_target {
                            "INCONCLUSIVE (reported probe size below target)"
                        } else {
                            "NO HIT"
                        }
                    ));
                }
                None if self.cache_disabled => self.line("Two-turn cache check: WARN engine-pending (prefix_cache=false; reuse=unknown, turn 2 omitted cached_tokens)"),
                None => self.line("Two-turn cache check: UNKNOWN (turn 2 omitted cached_tokens)"),
            }
        }
        self.line("Scope: two synthetic Chat Completions turns through the selected endpoint; no tools executed.");
        self.line("Tool/thinking flags and model fingerprints are upstream advertisements, not independent verification.");
        self.line("A hit is reported reuse for this conversation, not a guarantee for other clients or every cluster node.");
        self.line(if self.passed() {
            "Overall: PASS"
        } else {
            "Overall: FAILED or incomplete (see checks above)"
        });
    }
}

fn q(value: &str) -> String {
    let mut text = String::new();
    json::write_string(value, &mut text);
    text
}
fn n(value: Option<u64>) -> String {
    value.map_or_else(|| "unknown".into(), |v| v.to_string())
}
fn feature(value: json::View<'_>, key: &str) -> &'static str {
    match value.get(key).map(|v| v.raw()) {
        Some("true") => "yes (advertised)",
        Some("false") => "no (advertised)",
        _ => "unknown",
    }
}
fn error_category(error: &io::Error) -> &'static str {
    match error.kind() {
        io::ErrorKind::PermissionDenied => {
            if error.to_string() == "HTTP 401 Unauthorized: authentication failed" {
                "HTTP 401 Unauthorized: authentication failed"
            } else {
                "permission denied"
            }
        }
        io::ErrorKind::TimedOut => "timeout",
        io::ErrorKind::ConnectionAborted => "canceled or past deadline",
        io::ErrorKind::UnexpectedEof => "incomplete HTTP response",
        io::ErrorKind::InvalidData => "invalid metadata/protocol",
        io::ErrorKind::ConnectionRefused => "connection refused",
        _ => "transport/protocol failure",
    }
}

fn capabilities(
    report: &mut Report,
    endpoint: &Endpoint,
    cancel: &AtomicBool,
    deadline: Instant,
) -> bool {
    let metadata = match get(endpoint, cancel, deadline) {
        Ok(meta) => meta,
        Err(error) => {
            report.failure(format!("Contract: FAILED ({})", error_category(&error)));
            unknown_caps(report);
            return false;
        }
    };
    report.gateway = metadata.gateway;
    if matches!(metadata.status, 404 | 405 | 501) {
        report.line(format!(
            "Contract: unavailable (HTTP {}); standard chat only",
            metadata.status
        ));
        unknown_caps(report);
        return true;
    }
    if metadata.status != 200 {
        report.failure(format!(
            "Contract: FAILED (HTTP {}; no generation sent)",
            metadata.status
        ));
        if metadata.status == 409 {
            report.line("Gateway discovery is ambiguous: configure one compatible model group per doctor endpoint.");
        }
        unknown_caps(report);
        return false;
    }
    let body = metadata.body.as_deref().unwrap();
    let caps = match Capabilities::parse(body) {
        Ok(Some(caps)) => caps,
        Ok(None) => {
            // Unknown versions are never interpreted as the known contract.
            let value = json::view(body).unwrap();
            let version = value
                .get("contract_version")
                .and_then(|v| v.as_str())
                .filter(|v| v.len() <= 512)
                .map(|v| q(&v))
                .unwrap_or_else(|| "unknown".into());
            report.line(format!(
                "Contract: unrecognized (reported version {version}); standard chat only"
            ));
            unknown_caps(report);
            return true;
        }
        Err(_) => {
            report.failure("Contract: FAILED (malformed recognized capabilities)");
            unknown_caps(report);
            return false;
        }
    };
    let value = json::view(body).unwrap();
    report.cache_disabled = value.get("prefix_cache").is_some_and(|v| v.raw() == "false");
    report.line(format!("Contract: {VERSION} (recognized)"));
    report.line(format!("Model: {} (advertised)", q(&caps.model)));
    if let Some(identity) = &caps.identity {
        report.line(format!(
            "Model identity: weights={} kind={} tokenizer={} template={} (opaque)",
            q(&identity.weights_fingerprint),
            identity
                .weights_fingerprint_kind
                .as_deref()
                .map_or_else(|| "unknown".into(), q),
            q(&identity.tokenizer_fingerprint),
            q(&identity.template_fingerprint)
        ));
    } else {
        report.line("Model identity: unknown (no fingerprints advertised)");
    }
    report.line(format!(
        "Context limit: {} tokens; input limit: {}; output limit: {} (advertised)",
        caps.context_limit, caps.input_limit, caps.output_limit
    ));
    report.line(format!(
        "Tools: {}; thinking: {}; default thinking: {}",
        feature(value, "tools"),
        feature(value, "thinking"),
        feature(value, "default_thinking")
    ));
    report.line(format!(
        "Prefix cache: {}; scheduling-hint affinity: {} (not proof of a hit)",
        feature(value, "prefix_cache"),
        if caps.prompt_cache_key {
            "advertised"
        } else {
            "not advertised"
        }
    ));
    report.line(format!(
        "Cache granularity: {} tokens (advertised when present)",
        n(caps.cache_granularity_tokens)
    ));
    report.line(format!(
        "Admission: {}",
        if caps.exact_admission {
            "context_fit advertised; will check each exact request (no resource reservation)"
        } else {
            "not advertised; no exact admission check available"
        }
    ));
    let responses = value
        .get("endpoints")
        .and_then(|v| v.array())
        .is_some_and(|mut a| a.any(|v| v.as_str().is_some_and(|s| s == "/v1/responses")));
    report.line(format!(
        "APIs advertised: chat_completions={}; responses={} (Responses not probed)",
        caps.chat_completions, responses
    ));
    let chat = caps.chat_completions;
    report.caps = Some(caps);
    if !chat {
        report.failure("Chat check: unsupported by advertised capabilities; no generation sent");
    }
    chat
}
fn unknown_caps(report: &mut Report) {
    report
        .line("Model identity: unknown; context limit: unknown; tools: unknown; thinking: unknown");
    report.line("Admission: unknown/unavailable; no exact contract check available");
}

fn integer(value: json::View<'_>, key: &str) -> io::Result<u64> {
    let raw = value
        .get(key)
        .ok_or_else(|| transport::invalid("stats missing count"))?
        .raw();
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err(transport::invalid("stats invalid count"));
    }
    raw.parse()
        .map_err(|_| transport::invalid("stats count overflow"))
}
fn boolean(value: json::View<'_>, key: &str) -> io::Result<bool> {
    match value.get(key).map(|v| v.raw()) {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        _ => Err(transport::invalid("stats missing boolean")),
    }
}
fn cluster_lines(body: &str) -> io::Result<(Vec<String>, bool)> {
    let value = json::view(body)?;
    if value.get("object").and_then(|v| v.as_str()).as_deref() != Some("vade.gateway.stats") {
        return Err(transport::invalid("unrecognized gateway stats"));
    }
    let entries = value
        .get("endpoints")
        .and_then(|v| v.array())
        .ok_or_else(|| transport::invalid("stats missing endpoints"))?;
    let mut lines = Vec::new();
    let mut ids = Vec::new();
    let mut degraded = false;
    for entry in entries {
        let id = integer(entry, "endpoint")?;
        if id >= 16 || ids.contains(&id) {
            return Err(transport::invalid("stats duplicate/out-of-bounds endpoint"));
        }
        ids.push(id);
        let healthy = boolean(entry, "healthy")?;
        let cooling = boolean(entry, "cooling_down")?;
        let group = match entry.get("group").map(|v| v.raw()) {
            Some("null") => None,
            _ => Some(integer(entry, "group")?),
        };
        if group.is_some_and(|g| g >= 16) {
            return Err(transport::invalid("stats group out of bounds"));
        }
        degraded |= !healthy || cooling || group.is_none();
        let prompt = integer(entry, "prompt_tokens")?;
        let cached = integer(entry, "cached_tokens")?;
        if cached > prompt {
            return Err(transport::invalid(
                "stats cached count exceeds prompt count",
            ));
        }
        lines.push(format!("  endpoint {id}: group={} healthy={healthy} cooling_down={cooling} in_flight={} requests={} failures={} cached_tokens={cached}/{} (cumulative, not the two-turn check)", n(group), integer(entry, "in_flight")?, integer(entry, "requests")?, integer(entry, "failures")?, prompt));
    }
    if lines.is_empty() {
        return Err(transport::invalid("empty cluster stats"));
    }
    Ok((lines, degraded))
}
fn cluster(report: &mut Report, endpoint: &Endpoint, cancel: &AtomicBool, deadline: Instant) {
    match get(endpoint, cancel, deadline) {
        Ok(meta) if matches!(meta.status, 404 | 405 | 501) => report
            .line("Gateway cluster: stats unavailable (direct endpoint or single-upstream relay)"),
        Ok(meta) if meta.status != 200 => report.failure(format!(
            "Gateway cluster: stats probe failed (HTTP {})",
            meta.status
        )),
        Ok(meta) => match cluster_lines(meta.body.as_deref().unwrap()) {
            Ok((lines, degraded)) => {
                report.line(format!(
                    "Gateway cluster: {} endpoints; {} (snapshot before generation)",
                    lines.len(),
                    if degraded { "DEGRADED" } else { "healthy" }
                ));
                report.failed |= degraded;
                for line in lines {
                    report.line(line);
                }
            }
            Err(_) => report.failure("Gateway cluster: invalid/unrecognized stats"),
        },
        Err(error) => report.failure(format!(
            "Gateway cluster: stats probe failed ({})",
            error_category(&error)
        )),
    }
}

struct NoTools;
impl ToolExecutor for NoTools {
    fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        Err(ToolError::Denied)
    }
}
fn limits() -> Limits {
    Limits {
        // Keep the diagnostic's single-response length boundary visible.
        auto_continue: false,
        max_model_turns: 1,
        max_tool_turns: 0,
        max_tool_calls: 0,
        max_history_bytes: 1024 * 1024,
        // The byte cap bounds this synthetic diagnostic. Exact upstream
        // admission enforces its context; the ordinary 128K local estimate can
        // otherwise reject an admission-verified large cache probe.
        max_context_tokens: 512 * 1024,
        output_reserve_tokens: 32,
        wire: WireLimits {
            max_calls: 0,
            max_argument_bytes: 0,
            max_message_bytes: 512 * 1024,
            max_stream_bytes: 128 * 1024,
            max_events: 2048,
        },
        ..Limits::default()
    }
}
fn outcome(reason: &StopReason) -> (&'static str, bool) {
    match reason {
        StopReason::Finished(FinishReason::Stop) => ("complete (stop)", true),
        StopReason::Finished(FinishReason::Length) => ("complete (output token limit)", true),
        StopReason::Cancelled => ("canceled or past deadline", false),
        StopReason::ModelError(io::ErrorKind::PermissionDenied) => {
            ("HTTP 401 Unauthorized: authentication failed", false)
        }
        StopReason::ModelError(io::ErrorKind::TimedOut) => ("timeout", false),
        StopReason::ModelError(io::ErrorKind::UnexpectedEof) => ("incomplete response", false),
        StopReason::ModelError(_) => ("transport/protocol/admission error", false),
        StopReason::Limit(_) => ("diagnostic limit reached", false),
        _ => ("unsupported/invalid response", false),
    }
}
fn turn_line(
    report: &mut Report,
    turn: usize,
    ctx: &Context,
    reason: &StopReason,
    output_tokens: u64,
) {
    let (label, complete) = outcome(reason);
    let usage = ctx.turns.back().map(|t| t.usage).unwrap_or_default();
    let mut invalid_usage = ctx.turns.back().is_some_and(|t| t.invalid);
    if usage.completion_tokens.is_some_and(|n| n > output_tokens) {
        invalid_usage = true;
    }
    if let InputCount::Exact { prompt_tokens, .. } = ctx.input {
        if usage.prompt_tokens.is_some_and(|n| n != prompt_tokens)
            || usage.cached_tokens.is_some_and(|n| n > prompt_tokens)
        {
            invalid_usage = true;
        }
    }
    report.line(format!(
        "Turn {turn}: {label}; {} HTTP {}",
        ctx.last_http_stage,
        ctx.last_http_status
            .map_or_else(|| "unknown".into(), |n| n.to_string())
    ));
    match ctx.input {
        InputCount::Exact { prompt_tokens, output_tokens, context_limit, admitted } =>
            report.line(format!("  Admission: {} (prompt {prompt_tokens} + output {output_tokens} / context {context_limit}; no resource reservation)", if admitted { "PASS" } else { "REFUSED" })),
        _ if ctx.requests.admissions > 0 => report.line("  Admission: FAILED (no valid exact decision for this turn)"),
        _ => report.line("  Admission: unavailable (no exact decision)"),
    }
    if complete && !invalid_usage {
        report.line(format!(
            "  Usage: prompt_tokens={} completion_tokens={} cached_tokens={}",
            n(usage.prompt_tokens),
            n(usage.completion_tokens),
            n(usage.cached_tokens)
        ));
    } else {
        report.line("  Usage: unknown (failed response or invalid/conflicting telemetry)");
    }
    report.failed |= !complete || invalid_usage;
    report.turns.push(Turn {
        complete,
        usage,
        invalid_usage,
        prompt_tokens: match ctx.input {
            InputCount::Exact { prompt_tokens, .. } => Some(prompt_tokens),
            _ => usage.prompt_tokens,
        },
    });
}

const FALLBACK_PROBE_TOKENS: u64 = 8193;
const MAX_PROBE_WORDS: usize = 65536;
fn diagnostic_request(model: &str, output: u64, words: usize) -> String {
    const WORDS: [&str; 8] = [
        "amber", "birch", "cedar", "delta", "elm", "frost", "grove", "hazel",
    ];
    let mut content = String::from("Retain this invented diagnostic sequence for the next turn. ");
    let mut state = 41u64;
    for _ in 0..words {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        content.push_str(WORDS[(state >> 61) as usize]);
        content.push(' ');
    }
    content.push_str("Reply briefly with OK. No tools are available.");
    format!(
        "{{\"model\":{},\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"max_tokens\":{output},\"messages\":[{{\"role\":\"system\",\"content\":\"This is a synthetic Vade endpoint diagnostic. No workspace or tools are available. Respond briefly with OK.\"}},{{\"role\":\"user\",\"content\":{}}}]}}",
        q(model),
        q(&content)
    )
}

fn sized_request(
    report: &mut Report,
    model: &str,
    output: u64,
    endpoint: &Endpoint,
    cancel: &AtomicBool,
) -> Option<String> {
    let target = match report
        .caps
        .as_ref()
        .and_then(|c| c.cache_granularity_tokens)
    {
        Some(g) => match g.checked_mul(2).and_then(|n| n.checked_add(1)) {
            Some(n) => n,
            None => {
                report.failure("Cache probe: advertised granularity exceeds diagnostic bounds");
                return None;
            }
        },
        None => FALLBACK_PROBE_TOKENS,
    };
    report.probe_target = target;
    report.line(format!(
        "Cache probe: target at least {target} rendered prompt tokens; {}",
        if report
            .caps
            .as_ref()
            .is_some_and(|c| c.cache_granularity_tokens.is_some())
        {
            "above two advertised cache blocks"
        } else {
            "fallback above 8192 tokens (granularity unknown)"
        }
    ));
    if target > MAX_PROBE_WORDS as u64
        || report.caps.as_ref().is_some_and(|c| {
            target > c.input_limit
                || target.saturating_add(output).saturating_add(128) > c.context_limit
        })
    {
        report.failure("Cache probe: size target cannot fit advertised context/input or diagnostic bounds; no generation sent");
        return None;
    }
    let mut words = target as usize;
    let pool = ConnectionPool::default();
    for attempt in 1..=4 {
        let request = diagnostic_request(model, output, words);
        let Some(caps) = report.caps.as_ref().filter(|c| c.exact_admission) else {
            report.line(format!("Cache probe: {words} synthetic words; token size unverified (exact admission unavailable)"));
            return Some(request);
        };
        let (result, _) =
            Contract::count_request(caps, endpoint, request.as_bytes(), &pool, cancel);
        match result {
            Ok(InputCount::Exact { prompt_tokens, .. }) if prompt_tokens >= target => {
                report.line(format!("Cache probe: size verified by exact admission ({prompt_tokens} tokens; {attempt} sizing check(s), no inference)"));
                return Some(request);
            }
            Ok(InputCount::Exact { prompt_tokens, .. }) if prompt_tokens > 0 => {
                words = ((words as u64)
                    .saturating_mul(target)
                    .div_ceil(prompt_tokens)
                    .saturating_add(32))
                .max(words as u64 + 1)
                .min(MAX_PROBE_WORDS as u64 + 1) as usize;
                if words > MAX_PROBE_WORDS {
                    break;
                }
            }
            _ => {
                report
                    .failure("Cache probe sizing admission: FAILED or REFUSED; no generation sent");
                return None;
            }
        }
    }
    report.failure("Cache probe: cannot verify size within four admission checks / bounded padding; no generation sent");
    None
}

fn inspect(config: &Config, cancel: &AtomicBool, deadline: Instant) -> Report {
    let mut report = Report::default();
    report.line("Vade doctor · HTTP / Chat Completions");
    let cap_endpoint = Endpoint::parse(&format!("{}/capabilities", config.base)).unwrap();
    let mut stats_endpoint = cap_endpoint.clone();
    let prefix = stats_endpoint
        .path
        .strip_suffix("/v1/capabilities")
        .or_else(|| stats_endpoint.path.strip_suffix("/capabilities"))
        .unwrap_or("");
    stats_endpoint.path = format!("{prefix}/_vade/gateway/stats");
    let ready = capabilities(&mut report, &cap_endpoint, cancel, deadline);
    if report.gateway {
        cluster(&mut report, &stats_endpoint, cancel, deadline);
    } else {
        report.line("Gateway cluster: not probed (endpoint did not identify as a Vade gateway)");
    }
    if !ready {
        return report;
    }
    let model = match (&config.model, &report.caps) {
        (Some(selected), Some(caps)) if selected != &caps.model => {
            report.failure("Model: selected model differs from capabilities; no generation sent");
            return report;
        }
        (_, Some(caps)) => caps.model.clone(),
        (Some(selected), None) => {
            report.line(format!(
                "Model: {} (caller-selected; identity unverified)",
                q(selected)
            ));
            selected.clone()
        }
        (None, None) => {
            report.failure("Model: unknown; supply --model NAME to run the two-turn check");
            return report;
        }
    };
    let output = report.caps.as_ref().map_or(32, |c| 32.min(c.output_limit));
    let chat_endpoint = Endpoint::parse(&format!("{}/chat/completions", config.base)).unwrap();
    let Some(request) = sized_request(&mut report, &model, output, &chat_endpoint, cancel) else {
        return report;
    };
    let setup = json::parse(&request).and_then(|value| Conversation::from_request(value, limits()));
    let mut history = match setup {
        Ok(h) => h,
        Err(_) => {
            report.failure("Chat check: cannot build diagnostic request");
            return report;
        }
    };
    let mut upstream = match NativeModel::new(&format!("{}/chat/completions", config.base)) {
        Ok(m) => m,
        Err(_) => {
            report.failure("Chat check: invalid endpoint");
            return report;
        }
    };
    if let Some(caps) = report.caps.clone() {
        upstream.expect_capabilities(caps);
    }
    for turn in 1..=2 {
        if cancel.load(Ordering::Relaxed) {
            report.failure("Chat check: deadline reached");
            break;
        }
        if turn == 2 {
            let next = Value::Object(vec![
                ("role".into(), Value::String("user".into())),
                (
                    "content".into(),
                    Value::String("Keep the same sequence. Reply with OK again.".into()),
                ),
            ]);
            if history.push_user(next, limits()).is_err() {
                report.failure("Chat check: diagnostic history limit");
                break;
            }
        }
        upstream.begin_run();
        let run = agent::run(
            &mut history,
            &mut upstream,
            &mut NoTools,
            limits(),
            cancel,
            &mut |_| {},
        );
        turn_line(&mut report, turn, upstream.context(), &run.reason, output);
        if report.caps != upstream.context().capabilities {
            report.line("Capabilities changed since initial discovery; the metadata above is the initial snapshot.");
        }
        if !report.turns.last().unwrap().complete {
            break;
        }
    }
    let changes = upstream.context().cache_domain_changes;
    if changes > 0 {
        report.line(format!("Cache identity/domain changed {changes} time(s) during the check; observed reuse does not establish stable affinity."));
    }
    report
}

fn run(config: &Config) -> Report {
    let cancel = AtomicBool::new(false);
    let (finished, done) = mpsc::channel();
    let budget = Duration::from_secs(config.seconds);
    let deadline = Instant::now() + budget;
    thread::scope(|scope| {
        let flag = &cancel;
        let watcher = scope.spawn(move || {
            if done.recv_timeout(budget).is_err() {
                flag.store(true, Ordering::Relaxed);
            }
        });
        let mut report = inspect(config, &cancel, deadline);
        let _ = finished.send(());
        let _ = watcher.join();
        if cancel.load(Ordering::Relaxed) {
            report.failure("Doctor deadline reached; remote cancellation is not confirmed.");
        }
        report.finish();
        report
    })
}

/// `vade-harness doctor`: parse its arguments, run the checks against the
/// configured engine and print the report.
pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let args: Vec<_> = args.into_iter().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return crate::diagnostics::print_stdout(HELP);
    }
    let config = match Config::parse(args) {
        Ok(c) => c,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade: {error}"));
            return ExitCode::from(2);
        }
    };
    crate::diagnostics::stderr_line(format_args!(
        "doctor: probing metadata and two synthetic turns ({}s budget)",
        config.seconds
    ));
    let report = run(&config);
    if crate::diagnostics::print_stdout(&report.lines) == ExitCode::FAILURE {
        return ExitCode::FAILURE;
    }
    if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod auth_tests {
    use super::*;
    #[test]
    fn permissions_are_not_always_http_401() {
        assert_eq!(
            error_category(&io::Error::new(
                io::ErrorKind::PermissionDenied,
                "filesystem permission denied"
            )),
            "permission denied"
        );
        assert_eq!(
            error_category(&io::Error::new(
                io::ErrorKind::PermissionDenied,
                "HTTP 401 Unauthorized: authentication failed"
            )),
            "HTTP 401 Unauthorized: authentication failed"
        );
    }
}

#[cfg(test)]
mod cache_warning_tests {
    use super::*;
    #[test]
    fn disabled_cache_is_an_explicit_warning_while_claimed_cache_and_bad_telemetry_fail() {
        let mut report = Report { cache_disabled: true, turns: (0..2).map(|_| Turn { complete: true, usage: Usage { cached_tokens: Some(0), ..Usage::default() }, invalid_usage: false, prompt_tokens: Some(100) }).collect(), ..Report::default() };
        assert!(report.passed()); report.finish(); assert!(report.lines.contains("WARN engine-pending") && report.lines.contains("reuse=0.00%"));
        report.cache_disabled = false; assert!(!report.passed()); report.cache_disabled = true; report.turns[1].invalid_usage = true; assert!(!report.passed());
    }
}
