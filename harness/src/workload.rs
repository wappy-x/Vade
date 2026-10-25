//! Product workload driver: concurrent native sessions, synthetic context only.
//! Cache criterion idea: andreasgru/ds41-flash-hybrid, METHODS.md. Original code.
//! No tokenizer, engine, workspace executor, shell, retry, or context compaction.
use crate::{
    agent::{self, Conversation, Limits, StopReason, ToolError, ToolExecutor, ToolInvocation},
    agent_wire::FinishReason,
    contract::{InputCount, RequestCounts, Usage},
    json::{self, Value},
    transport::NativeModel,
};
use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

pub const HELP: &str = "Vade native agent workload measurement (HTTP, no real tools)

Usage: vade-harness workload --endpoint URL --model NAME --json FILE [OPTIONS]

  --sessions N             Concurrent sessions, 1..32 (default 6)
  --main-sessions N        Main sessions, 1 or 2 (default 1); others subordinate
  --turns K                Turns per session, 2..1000 (default 8)
  --main-system-bytes N    Main system padding (default 16384)
  --sub-system-bytes N     Subordinate system padding (default 4096)
  --main-user-bytes N      Main new user payload per turn (default 2048)
  --sub-user-bytes N       Subordinate new user payload per turn (default 512)
  --main-tool-bytes N      Main synthetic tool result per turn (default 8192)
  --sub-tool-bytes N       Subordinate synthetic tool result per turn (default 2048)
  --output-tokens N        max_tokens reservation, 1..8192 (default 128)
  --seed N                 Fixed synthetic-content seed (default 41)
  --max-seconds N          Whole workload deadline, 1..86400 (default 600)
  --json FILE              Required new metrics file; never overwrite
  --endpoint URL           Required OpenAI chat-compatible http:// URL
  --model NAME             Required explicit model; never substituted
  -h, --help               Show this help

At most 1024 session-turns. Padding sizes are bytes, not model tokens, and may
be zero; each is bounded at 262144. Existing native message/history limits apply.
Defaults: 1 main + 5 subordinate sessions, each with independent growing history.
Each turn adds a synthetic completed tool-call/result pair and a user message.
Real model responses are preserved. No filesystem tools or child agents run.

Role metrics exclude turn 1. TTFT is request start to first generated SSE delta
(content, reasoning or tool function data),
including capability discovery and admission; JSON-only responses have no TTFT.
Output tok/s uses reported completion tokens / end-to-end elapsed seconds.
Aggregate throughput uses overlapping requests' wall window, not a sum of rates.

R = sum(min(cached_tokens, U)) / sum(U), target >= 0.98 separately per role.
Exact admission counts the retained previous prompt plus full assistant reply
in a separate admission-only request. U = min(current prompt count, retained
rendered history count). Without exact admission, previous prompt + output
counts are used. Per-turn C/U and count sources are printed.
This is an ESTIMATED token LCP:
v1 admission exposes lengths, not token IDs; rendered append-only continuity is
an assumption. Without admission, reported usage or ceil(wire bytes/4)+256 is
used and labeled. Missing telemetry or incomplete coverage cannot pass a target.
No credentials, endpoint/model identifiers, affinity keys or transcripts in JSON.
";

/// A workload run: the endpoint and model, how many main and subordinate
/// sessions and turns, synthetic prompt sizes, output and time limits, and
/// where to write the JSON report.
#[derive(Clone, Debug)]
pub struct Config {
    pub endpoint: String,
    pub model: String,
    pub json_path: PathBuf,
    pub sessions: usize,
    pub main_sessions: usize,
    pub turns: usize,
    pub main_system_bytes: usize,
    pub sub_system_bytes: usize,
    pub main_user_bytes: usize,
    pub sub_user_bytes: usize,
    pub main_tool_bytes: usize,
    pub sub_tool_bytes: usize,
    pub output_tokens: u64,
    pub seed: u64,
    pub max_seconds: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            model: String::new(),
            json_path: PathBuf::new(),
            sessions: 6,
            main_sessions: 1,
            turns: 8,
            main_system_bytes: 16384,
            sub_system_bytes: 4096,
            main_user_bytes: 2048,
            sub_user_bytes: 512,
            main_tool_bytes: 8192,
            sub_tool_bytes: 2048,
            output_tokens: 128,
            seed: 41,
            max_seconds: 600,
        }
    }
}
impl Config {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, &'static str> {
        Self::parse_options(args, &mut None)
    }
    fn parse_options(args: impl IntoIterator<Item = String>, option: &mut Option<String>) -> Result<Self, &'static str> {
        let mut config = Self::default();
        let mut args = args.into_iter();
        let mut seen = Vec::new();
        while let Some(key) = args.next() {
            *option = crate::diagnostics::option_name(&key).map(str::to_owned);
            if seen.contains(&key) {
                return Err("duplicate workload option");
            }
            seen.push(key.clone());
            let value = args
                .next()
                .ok_or("workload option needs a value; see workload --help")?;
            let number = || {
                value
                    .parse::<usize>()
                    .map_err(|_| "workload sizes must be unsigned integers")
            };
            match key.as_str() {
                "--endpoint" => config.endpoint = value,
                "--model" => config.model = value,
                "--json" => config.json_path = value.into(),
                "--sessions" => config.sessions = number()?,
                "--main-sessions" => config.main_sessions = number()?,
                "--turns" => config.turns = number()?,
                "--main-system-bytes" => config.main_system_bytes = number()?,
                "--sub-system-bytes" => config.sub_system_bytes = number()?,
                "--main-user-bytes" => config.main_user_bytes = number()?,
                "--sub-user-bytes" => config.sub_user_bytes = number()?,
                "--main-tool-bytes" => config.main_tool_bytes = number()?,
                "--sub-tool-bytes" => config.sub_tool_bytes = number()?,
                "--output-tokens" => {
                    config.output_tokens =
                        value.parse().map_err(|_| "invalid output token count")?
                }
                "--seed" => config.seed = value.parse().map_err(|_| "invalid seed")?,
                "--max-seconds" => {
                    config.max_seconds = value.parse().map_err(|_| "invalid deadline")?
                }
                _ => return Err("unknown workload option; see workload --help"),
            }
        }
        *option = None;
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<(), &'static str> {
        if self.endpoint.is_empty()
            || self.model.is_empty()
            || self.json_path.as_os_str().is_empty()
        {
            return Err("workload requires --endpoint, --model and --json");
        }
        if self.endpoint.contains('?') || self.endpoint.contains('#') {
            return Err("workload endpoint queries/fragments are unsupported");
        }
        if self.endpoint.starts_with("https://") {
            return Err("HTTPS is unsupported; use an HTTP endpoint");
        }
        NativeModel::new(&self.endpoint).map_err(|_| "invalid workload HTTP endpoint")?;
        if self.model.len() > 512 || self.model.chars().any(char::is_control) {
            return Err("invalid workload model name");
        }
        if !(1..=32).contains(&self.sessions)
            || !(1..=2).contains(&self.main_sessions)
            || self.main_sessions > self.sessions
            || !(2..=1000).contains(&self.turns)
            || self.sessions * self.turns > 1024
        {
            return Err(
                "workload needs 1..32 sessions, 1..2 mains, 2..1000 turns, at most 1024 session-turns",
            );
        }
        if [
            self.main_system_bytes,
            self.sub_system_bytes,
            self.main_user_bytes,
            self.sub_user_bytes,
            self.main_tool_bytes,
            self.sub_tool_bytes,
        ]
        .iter()
        .any(|&n| n > 262144)
        {
            return Err("workload padding exceeds 262144 bytes");
        }
        if !(1..=8192).contains(&self.output_tokens) || !(1..=86400).contains(&self.max_seconds) {
            return Err("workload needs 1..8192 output tokens and 1..86400 seconds");
        }
        Ok(())
    }
    fn sizes(&self, role: Role) -> (usize, usize, usize) {
        match role {
            Role::Main => (
                self.main_system_bytes,
                self.main_user_bytes,
                self.main_tool_bytes,
            ),
            Role::Subordinate => (
                self.sub_system_bytes,
                self.sub_user_bytes,
                self.sub_tool_bytes,
            ),
        }
    }
}

/// A session's role: a main session, or a subordinate one with its own
/// prompt sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Main,
    Subordinate,
}
impl Role {
    fn label(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Subordinate => "subordinate",
        }
    }
}
fn obj(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn s(text: impl Into<String>) -> Value {
    Value::String(text.into())
}
fn n(number: impl ToString) -> Value {
    Value::Number(number.to_string())
}
fn opt_n(number: Option<u64>) -> Value {
    number.map(n).unwrap_or(Value::Null)
}
fn opt_f(number: Option<f64>) -> Value {
    number
        .filter(|x| x.is_finite())
        .map(n)
        .unwrap_or(Value::Null)
}
fn message(role: &str, text: String) -> Value {
    obj([("role", s(role)), ("content", s(text))])
}

/// Reproducible filler, deliberately NOT a cryptographic generator. Role,
/// session, turn and channel have distinct deterministic seeds. No private data.
fn filler(seed: u64, bytes: usize) -> String {
    const WORDS: [&str; 16] = [
        " amber", " birch", " cedar", " delta", " ember", " field", " grove", " hazel", " ivory",
        " juniper", " kelp", " larch", " maple", " north", " olive", " pine",
    ];
    let mut state = seed;
    let mut result = String::with_capacity(bytes + 8);
    while result.len() < bytes {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        result.push_str(WORDS[(state >> 60) as usize]);
    }
    result.truncate(bytes); // ASCII vocabulary; exact configured padding bytes.
    result
}
fn seed(config: &Config, session: usize, turn: usize, channel: u64) -> u64 {
    config
        .seed
        .wrapping_add((session as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add((turn as u64).wrapping_mul(0xd1b54a32d192ed03))
        .wrapping_add(channel)
}
fn append_turn(
    messages: &mut Vec<Value>,
    config: &Config,
    role: Role,
    session: usize,
    turn: usize,
) {
    let (_, user_bytes, tool_bytes) = config.sizes(role);
    let id = format!("synthetic-{session}-{turn}");
    messages.push(obj([
        ("role", s("assistant")),
        ("content", Value::Null),
        (
            "tool_calls",
            Value::Array(vec![obj([
                ("id", s(&id)),
                ("type", s("function")),
                (
                    "function",
                    obj([("name", s("synthetic_observation")), ("arguments", s("{}"))]),
                ),
            ])]),
        ),
    ]));
    messages.push(obj([
        ("role", s("tool")),
        ("tool_call_id", s(id)),
        (
            "content",
            s(format!(
                "Synthetic observation for turn {turn}; no tool was executed.{}",
                filler(seed(config, session, turn, 2), tool_bytes)
            )),
        ),
    ]));
    messages.push(message("user", format!(
        "Synthetic workload turn {turn}. Continue with a short plain-text analysis of the observations. Do not call tools.{}",
        filler(seed(config, session, turn, 3), user_bytes))));
}
fn limits() -> Limits {
    Limits {
        // Measurement preserves the frozen single-response timing boundary.
        auto_continue: false,
        max_model_turns: 1,
        max_tool_turns: 0,
        max_tool_calls: 0,
        ..Limits::default()
    }
}
fn conversation(config: &Config, role: Role, session: usize) -> io::Result<Conversation> {
    let mut messages = vec![message(
        "system",
        format!(
            "Synthetic native workload session {session}, role {}. All observations and tool records are invented; no real actions occurred. Reply with plain text, without tools.{}",
            role.label(),
            filler(seed(config, session, 0, 1), config.sizes(role).0)
        ),
    )];
    append_turn(&mut messages, config, role, session, 1);
    Conversation::from_request(
        obj([
            ("model", s(&config.model)),
            ("stream", Value::Bool(true)),
            (
                "stream_options",
                obj([("include_usage", Value::Bool(true))]),
            ),
            ("max_tokens", n(config.output_tokens)),
            ("temperature", n(0)),
            ("messages", Value::Array(messages)),
        ]),
        limits(),
    )
}
fn next_turn(
    old: &Conversation,
    config: &Config,
    role: Role,
    session: usize,
    turn: usize,
) -> io::Result<Conversation> {
    // The public conversation constructor validates the closed synthetic pair.
    // Retain all authoritative assistant fields, not provisional streamed text.
    // NativeModel's persistent AppendOnlyRequest independently checks every old
    // serialized record; reconstructing the typed container cannot rewrite it.
    let Value::Object(mut fields) = old.options().clone() else {
        unreachable!()
    };
    let mut messages: Vec<_> = old.messages().iter().map(|m| m.value().clone()).collect();
    append_turn(&mut messages, config, role, session, turn);
    fields.push(("messages".into(), Value::Array(messages)));
    Conversation::from_request(Value::Object(fields), limits())
}
struct NoTools;
impl ToolExecutor for NoTools {
    fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        Err(ToolError::Denied)
    }
}

/// One measured turn: timing, outcome, input count, reported usage and wire
/// size.
#[derive(Clone, Debug)]
pub struct Sample {
    pub turn: usize,
    pub start_s: f64,
    pub end_s: f64,
    pub ttft_ms: Option<f64>,
    pub outcome: &'static str,
    pub error: Option<&'static str>,
    pub input: InputCount,
    pub usage: Usage,
    pub invalid_usage: bool,
    pub wire_bytes: usize,
    pub history_bytes: usize,
    pub requests: RequestCounts,
    pub u: Option<u64>,
    pub u_basis: &'static str,
    pub u_estimated: bool,
    pub rendered_previous_tokens: Option<u64>,
    pub prefix_count_requests: RequestCounts,
    pub prefix_count_error: Option<&'static str>,
    pub prefix_count_elapsed_ms: Option<f64>,
    pub cache_domain_changes: u64,
    pub prefix_preserved: bool,
    pub last_http_status: Option<u16>,
    pub last_http_stage: &'static str,
    output_bytes: usize,
}
impl Sample {
    fn completed(&self) -> bool {
        self.error.is_none()
    }
    fn duration(&self) -> f64 {
        (self.end_s - self.start_s).max(0.0)
    }
}
/// Every turn one session ran, and why it stopped early, if it did.
#[derive(Debug)]
pub struct SessionReport {
    pub session: usize,
    pub role: Role,
    pub samples: Vec<Sample>,
    pub stopped: Option<&'static str>,
}
/// A finished workload run: its configuration, every session and the total
/// time.
#[derive(Debug)]
pub struct Report {
    pub config: Config,
    pub sessions: Vec<SessionReport>,
    pub elapsed_s: f64,
    pub timed_out: bool,
}

fn count(sample: &Sample) -> (u64, &'static str) {
    match sample.input {
        InputCount::Exact { prompt_tokens, .. } => (prompt_tokens, "admission"),
        _ => match sample.usage.prompt_tokens {
            Some(p) => (p, "reported_usage"),
            None => (
                (sample.wire_bytes as u64).div_ceil(4).saturating_add(256),
                "wire_estimate",
            ),
        },
    }
}
fn denominator(previous: &Sample, current: &mut Sample) {
    if !previous.completed() || !current.completed() || !current.prefix_preserved {
        current.u_basis = "unavailable_failed_or_changed_prefix";
        return;
    }
    if current.cache_domain_changes != 0 {
        current.u_basis = "unavailable_cache_identity_change";
        return;
    }
    if current.prefix_count_error.is_some() {
        current.u_basis = "unavailable_rendered_history_count_failed";
        return;
    }
    if let (Some(rendered), InputCount::Exact { prompt_tokens, .. }) =
        (current.rendered_previous_tokens, &current.input)
    {
        current.u = Some(rendered.min(*prompt_tokens));
        current.u_basis = "exact_rendered_history_counts";
        return;
    }
    // Exact lengths do not reveal token identity at a template boundary. The
    // estimate assumes render(P + answer + new messages) starts with P + answer.
    // No guessed engine fields or extra count-only generations are sent.
    let (p, p_source) = count(previous);
    let (q, q_source) = count(current);
    let output = previous
        .usage
        .completion_tokens
        .unwrap_or((previous.output_bytes as u64).div_ceil(4));
    current.u = Some(q.min(p.saturating_add(output)));
    current.u_basis = if p_source == "admission"
        && q_source == "admission"
        && previous.usage.completion_tokens.is_some()
    {
        "admission_plus_reported_output"
    } else if p_source != "wire_estimate"
        && q_source != "wire_estimate"
        && previous.usage.completion_tokens.is_some()
    {
        "reported_counts"
    } else {
        "wire_bytes_estimate"
    };
}
fn outcome(reason: &StopReason) -> (&'static str, Option<&'static str>) {
    match reason {
        StopReason::Finished(FinishReason::Stop) => ("stop", None),
        // Length is normal for a bounded generation workload. The native runtime
        // commits the validated assistant message; keep its finish reason visible.
        StopReason::Finished(FinishReason::Length) => ("length", None),
        StopReason::Cancelled => ("error", Some("cancelled")),
        StopReason::ModelError(io::ErrorKind::PermissionDenied) => (
            "error",
            Some("HTTP 401 Unauthorized: authentication failed"),
        ),
        StopReason::ModelError(io::ErrorKind::TimedOut) => ("error", Some("transport_timeout")),
        StopReason::ModelError(io::ErrorKind::UnexpectedEof) => {
            ("error", Some("incomplete_response"))
        }
        StopReason::ModelError(_) => ("error", Some("model_or_transport_error")),
        StopReason::Limit(_) => ("error", Some("native_runtime_limit")),
        StopReason::Finished(_) => ("error", Some("unsupported_finish_reason")),
        _ => ("error", Some("invalid_native_response")),
    }
}
fn session_run(
    config: &Config,
    role: Role,
    session: usize,
    mut history: Conversation,
    mut model: NativeModel,
    start: Instant,
    cancel: &AtomicBool,
) -> SessionReport {
    let mut report = SessionReport {
        session,
        role,
        samples: Vec::new(),
        stopped: None,
    };
    for turn in 1..=config.turns {
        if cancel.load(Ordering::Relaxed) {
            report.stopped = Some("workload_cancelled");
            break;
        }
        let mut rendered_previous_tokens = None;
        let mut prefix_count_requests = RequestCounts::default();
        let mut prefix_count_error = None;
        let mut prefix_count_elapsed_ms = None;
        if turn > 1 {
            let next = match next_turn(&history, config, role, session, turn) {
                Ok(next) => next,
                Err(_) => {
                    report.stopped = Some("history_limit");
                    break;
                }
            };
            if model
                .context()
                .capabilities
                .as_ref()
                .is_some_and(|c| c.exact_admission)
            {
                let count_start = Instant::now();
                let (result, observation) =
                    model.count_rendered_history(history.request(limits().wire), cancel);
                prefix_count_elapsed_ms = Some(count_start.elapsed().as_secs_f64() * 1000.0);
                prefix_count_requests = observation.requests;
                match result {
                    Ok(InputCount::Exact { prompt_tokens, .. }) => {
                        rendered_previous_tokens = Some(prompt_tokens)
                    }
                    _ => prefix_count_error = Some("rendered_history_count_failed_or_refused"),
                }
            }
            history = next;
        }
        let previous_messages = history.messages().len();
        let before = model.context().requests;
        let domain_before = model.context().cache_domain_changes;
        model.begin_run();
        let request_start = Instant::now();
        let mut output_bytes = 0usize;
        let run = agent::run(
            &mut history,
            &mut model,
            &mut NoTools,
            limits(),
            cancel,
            &mut |text| {
                output_bytes = output_bytes.saturating_add(text.len());
            },
        );
        let end = Instant::now();
        let ctx = model.context();
        let (outcome, mut error) = outcome(&run.reason);
        let usage = ctx.turns.back().map(|t| t.usage).unwrap_or_default();
        let invalid_usage = ctx.turns.back().is_some_and(|t| t.invalid);
        if run.tool_calls != 0 {
            error = Some("unexpected_tool_execution");
        }
        if usage
            .completion_tokens
            .is_some_and(|n| n > config.output_tokens)
        {
            error = Some("output_usage_exceeds_reservation");
        }
        if let InputCount::Exact { prompt_tokens, .. } = ctx.input {
            if usage.prompt_tokens.is_some_and(|p| p != prompt_tokens) {
                error = Some("admission_usage_mismatch");
            }
        }
        // A committed answer comes from full native assembly. Serialized bytes
        // include opaque reasoning/tool extensions if present in the answer.
        let answer_bytes = history
            .messages()
            .get(previous_messages)
            .map(|m| m.serialized_bytes())
            .unwrap_or(output_bytes);
        let mut sample = Sample {
            turn,
            start_s: request_start.duration_since(start).as_secs_f64(),
            end_s: end.duration_since(start).as_secs_f64(),
            ttft_ms: if error.is_none() && ctx.response_streamed {
                ctx.first_token_elapsed.map(|t| t.as_secs_f64() * 1000.0)
            } else {
                None
            },
            outcome,
            error,
            input: ctx.input.clone(),
            usage,
            invalid_usage,
            wire_bytes: ctx.request_bytes,
            history_bytes: history.serialized_bytes(),
            requests: ctx.requests.since(before),
            u: None,
            u_basis: "first_turn",
            u_estimated: true,
            rendered_previous_tokens,
            prefix_count_requests,
            prefix_count_error,
            prefix_count_elapsed_ms,
            cache_domain_changes: ctx.cache_domain_changes.saturating_sub(domain_before),
            prefix_preserved: error.is_none(),
            last_http_status: ctx.last_http_status,
            last_http_stage: ctx.last_http_stage,
            output_bytes: answer_bytes,
        };
        if let Some(previous) = report.samples.last() {
            denominator(previous, &mut sample);
        }
        report.samples.push(sample);
        if error.is_some() {
            report.stopped = error;
            break;
        }
    }
    report
}

/// No network calls until every initial request and endpoint is validated.
/// Sessions run independently after one common start; no turn-by-turn barrier
/// artificially serializes fast workers behind slow ones. A failure stops that
/// session, not its siblings. Deadline cancellation joins every owned worker.
pub fn run(config: Config) -> io::Result<Report> {
    config.validate().map_err(io::Error::other)?;
    let mut initial = Vec::new();
    let mut queued = Vec::new();
    for session in 0..config.sessions {
        let role = if session < config.main_sessions {
            Role::Main
        } else {
            Role::Subordinate
        };
        let waiting = Arc::new(AtomicBool::new(false));
        queued.push(Arc::clone(&waiting));
        let mut model = NativeModel::new(&config.endpoint)?;
        model.set_progress_observer(Some(Arc::new(move |phase| {
            waiting.store(
                phase == crate::transport::NativePhase::Queued,
                Ordering::Relaxed,
            );
        })));
        initial.push((session, role, conversation(&config, role, session)?, model));
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let deadline = Duration::from_secs(config.max_seconds);
    let (result_tx, result_rx) = mpsc::channel();
    let mut sessions = thread::scope(|scope| -> io::Result<Vec<SessionReport>> {
        let mut workers = Vec::new();
        let mut starters = Vec::new();
        for (session, role, history, model) in initial {
            let cancel = Arc::clone(&cancel);
            let tx = result_tx.clone();
            let (start_tx, start_rx) = mpsc::channel();
            let config = &config;
            let worker = thread::Builder::new()
                .name(format!("workload-{session}"))
                .spawn_scoped(scope, move || {
                    if let Ok(start) = start_rx.recv() {
                        let report =
                            session_run(config, role, session, history, model, start, &cancel);
                        let _ = tx.send(report);
                    }
                });
            match worker {
                Ok(worker) => {
                    workers.push(worker);
                    starters.push(start_tx);
                }
                Err(error) => {
                    // Dropping the gates releases every already-spawned worker.
                    drop(starters);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(error);
                }
            }
        }
        drop(result_tx);
        for tx in starters {
            let _ = tx.send(start);
        }
        let mut reports = Vec::new();
        let mut last_notice = Instant::now();
        while reports.len() < config.sessions {
            if start.elapsed() >= deadline {
                cancel.store(true, Ordering::Relaxed);
            }
            match result_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(report) => reports.push(report),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if last_notice.elapsed() >= Duration::from_secs(5) {
                crate::diagnostics::stderr_line(format_args!(
                    "workload: {}/{} sessions finished, {:.0}s elapsed; {} queued / awaiting upstream{}",
                    reports.len(),
                    config.sessions,
                    start.elapsed().as_secs_f64(),
                    queued
                        .iter()
                        .filter(|waiting| waiting.load(Ordering::Relaxed))
                        .count(),
                    if cancel.load(Ordering::Relaxed) {
                        "; cancelling"
                    } else {
                        ""
                    }
                ));
                last_notice = Instant::now();
            }
        }
        for worker in workers {
            worker
                .join()
                .map_err(|_| io::Error::other("workload worker failed"))?;
        }
        Ok(reports)
    })?;
    sessions.sort_by_key(|s| s.session);
    Ok(Report {
        config,
        sessions,
        elapsed_s: start.elapsed().as_secs_f64(),
        timed_out: cancel.load(Ordering::Relaxed),
    })
}


/// `vade-harness workload`: parse its arguments, run the sessions and print
/// the report as a table or JSON.
pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let args: Vec<_> = args.into_iter().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return crate::diagnostics::print_stdout(HELP);
    }
    let config = match crate::diagnostics::parse_command("workload", |option| Config::parse_options(args, option)) {
        Ok(config) => config,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade: {error}"));
            return ExitCode::from(2);
        }
    };
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    // Validate the destination before connecting. create_new also refuses symlinks
    // and protects existing reports. Never include the actual path in an error.
    let mut file = match options.open(&config.json_path) {
        Ok(file) => file,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade: cannot create new workload JSON file. {}", crate::diagnostics::output_file_hint(&error)));
            return ExitCode::from(2);
        }
    };
    crate::diagnostics::stderr_line(format_args!(
        "workload: starting {} native sessions ({} main), {} turns each; synthetic records only",
        config.sessions, config.main_sessions, config.turns
    ));
    let report = match run(config) {
        Ok(report) => report,
        Err(_) => {
            crate::diagnostics::stderr_line(format_args!("vade: workload setup/worker failed; JSON is incomplete. Keep the partial file, check the endpoint/model with vade-harness doctor, then retry with a new --json path."));
            return ExitCode::FAILURE;
        }
    };
    let saved = report.to_json().and_then(|body| {
        file.write_all(body.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()
    });
    if let Err(error) = saved {
        crate::diagnostics::stderr_line(format_args!("vade: workload JSON write failed. {}", crate::diagnostics::output_file_hint(&error)));
        return ExitCode::FAILURE;
    }
    if crate::diagnostics::print_stdout(&report.table()) == ExitCode::FAILURE {
        return ExitCode::FAILURE;
    }
    crate::diagnostics::stderr_line(format_args!("workload: metrics saved to the requested JSON file"));
    if report.successful() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

mod report;

#[cfg(test)]
mod tests;
