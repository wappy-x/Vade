//! Typed native model sessions, prompt continuity and request progress.
use super::*;

/// Local request progress only; the HTTP contract provides no queue-position
/// receipt. Headers and role-only deltas do not prove generation has begun.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativePhase {
    Idle,
    Preparing,
    /// Request sent; awaiting output. This is not an engine queue receipt.
    Queued,
    Generating,
    Thinking,
}

pub(super) type ProgressObserver = Arc<dyn Fn(NativePhase) + Send + Sync>;

/// Presentation only: deltas are provisional, Completed follows full framing
/// validation. Neither callback changes the authoritative assistant message.
pub enum ThinkingUpdate<'a> {
    Delta(&'a str),
    Completed {
        text: &'a str,
        tokens: Option<u64>,
        limited: bool,
    },
}
pub(super) type ThinkingObserver = Arc<dyn Fn(ThinkingUpdate<'_>) + Send + Sync>;

#[derive(Default)]
pub(super) struct ThinkingProjection {
    text: String,
    limited: bool,
    tokens: Option<u64>,
    token_snapshot: bool,
    invalid_tokens: bool,
}
impl ThinkingProjection {
    pub(super) fn observe(&mut self, event: json::View<'_>, observer: &Option<ThinkingObserver>) {
        if let Some(content) = event
            .get("choices")
            .and_then(|v| v.array())
            .and_then(|mut choices| choices.next())
            .and_then(|choice| choice.get("delta").or_else(|| choice.get("message")))
            && let Some(text) = content
                .get("reasoning_content")
                .and_then(json::View::as_str)
                .filter(|text| !text.is_empty())
                .or_else(|| content.get("reasoning").and_then(json::View::as_str))
        {
            if let Some(observer) = observer {
                observer(ThinkingUpdate::Delta(&text));
            }
            // The display uses the existing terminal retention budget. Reaching
            // it never cancels generation or changes response/history limits.
            let mut end = text
                .len()
                .min(crate::terminal::MAX_TRANSCRIPT_BYTES.saturating_sub(self.text.len()));
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            self.limited |= end < text.len();
            self.text.push_str(&text[..end]);
        }
        if let Some(raw) = event
            .get("usage")
            .and_then(|usage| usage.get("completion_tokens_details"))
            .and_then(|details| details.get("reasoning_tokens"))
        {
            let tokens = if raw.raw() == "null" {
                None
            } else {
                raw.raw().parse::<u64>().ok()
            };
            self.invalid_tokens |= (raw.raw() != "null" && tokens.is_none())
                || (self.token_snapshot && self.tokens != tokens);
            self.token_snapshot = true;
            self.tokens = tokens;
        }
    }
    pub(super) fn completed(&self, observer: &Option<ThinkingObserver>, authoritative: Option<&str>) {
        if let Some(observer) = observer {
            observer(ThinkingUpdate::Completed {
                text: authoritative.unwrap_or(&self.text),
                tokens: if self.invalid_tokens {
                    None
                } else {
                    self.tokens
                },
                limited: authoritative.is_none() && self.limited,
            });
        }
    }
}

/// HTTP adapter for the typed agent loop. Requests retain all messages, tools
/// and options; the caller supplies model/stream settings and prompts.
struct CountedRequest {
    original: String,
    body: Vec<u8>,
    input: crate::contract::InputCount,
}
/// Text callbacks are provisional and their errors abort the turn. SSE requires
/// a finish reason, `[DONE]`, and complete HTTP/SSE framing before acceptance.
/// Native response reads permit queue/prefill waits up to the absolute lifetime;
/// OS DNS resolution and an in-progress connect cannot be interrupted by std.
pub struct NativeModel {
    endpoint: Endpoint,
    pool: ConnectionPool,
    prompt: crate::prompt::AppendOnlyRequest,
    contract: crate::contract::Contract,
    kv_session: Arc<kv::Session>,
    kv_hints: Vec<crate::kv::Hint>,
    kv_branch: String,
    progress: Option<ProgressObserver>,
    thinking: Option<ThinkingObserver>,
    estimate: Option<super::estimate::EstimateObserver>,
    suffix_replaced: bool,
    prewarm_compaction: bool,
    warm_pending: bool,
    frozen_prefix_messages: Option<usize>,
    counted: Option<CountedRequest>,
    work_remaining: Option<Duration>,
    reply_tokens: Option<usize>,
    server_cancellation: bool,
    cancel_confirmed: Option<bool>,
}

impl NativeModel {
    pub fn with_server_cancellation(mut self) -> Self { self.server_cancellation = true; self }
    pub fn server_cancel_confirmed(&self) -> Option<bool> { self.cancel_confirmed }

    /// Opt-in only: one bounded, discarded completion after compaction. This
    /// warms the compacted prompt without committing output or running tools.
    pub fn with_compaction_prewarm(mut self, enabled: bool) -> Self {
        self.prewarm_compaction = enabled;
        self
    }
    pub fn with_generation_options(mut self, options: crate::contract::GenerationOptions) -> Self {
        if self.contract.options != options {
            self.reestablish_prompt(crate::contract::PromptBoundary::Sampling);
        }
        self.contract.options = options;
        self
    }
    /// A bounded, nonblocking UI notification, never prompt or transcript data.
    /// Callbacks run on the model worker and must return promptly. Clear the
    /// observer when its UI activation ends; branch clients do not inherit it.
    pub fn set_progress_observer(&mut self, observer: Option<ProgressObserver>) {
        self.progress = observer;
    }
    pub fn set_thinking_observer(&mut self, observer: Option<ThinkingObserver>) {
        self.thinking = observer;
    }
    /// The engine's warm-cache estimate, once per streamed reply, before its
    /// first token. Same rules as the progress observer.
    pub fn set_estimate_observer(&mut self, observer: Option<super::estimate::EstimateObserver>) {
        self.estimate = observer;
    }

    /// Doctor has already recognized this contract while discovering the model.
    /// Preserve that expectation across the first normal per-turn refresh: a
    /// disappeared contract or changed request features must not silently fall
    /// back to ordinary chat between discovery and generation. Construction and
    /// seeding perform no I/O; negotiate still validates and decorates normally.
    pub(crate) fn expect_capabilities(&mut self, capabilities: crate::contract::Capabilities) {
        self.contract.context.capabilities = Some(capabilities);
        self.contract.context.negotiation = crate::contract::VERSION;
    }

    /// An explicit new conversation or compaction boundary. The caller must
    /// disclose that prior prompt bytes will change. Session affinity, usage
    /// counters and sockets survive; create a new model for a new session.
    pub fn reset_prompt(&mut self) {
        self.reestablish_prompt(crate::contract::PromptBoundary::Compaction);
    }

    /// Declare a replacement prefix locally; no request is sent until the next
    /// explicit prompt. New sessions should construct a new NativeModel first.
    pub fn reestablish_prompt(&mut self, boundary: crate::contract::PromptBoundary) {
        if boundary == crate::contract::PromptBoundary::Compaction {
            self.suffix_replaced = true;
            self.warm_pending = self.prewarm_compaction;
        } else {
            self.prompt = crate::prompt::AppendOnlyRequest::default();
            self.suffix_replaced = false;
            self.warm_pending = false;
            self.frozen_prefix_messages = None;
        }
        self.counted = None;
        self.contract.reset_prefix(boundary);
        // A compaction keeps the hints of its unchanged leading messages (see
        // `keep_shared_hints`). A tool-recovery policy changes no message; the
        // engine checks each hint against the rendered prompt in any case.
        if !matches!(boundary, crate::contract::PromptBoundary::Compaction | crate::contract::PromptBoundary::ToolRecovery) {
            self.kv_hints.clear();
        }
    }
    /// Before the first request after a compaction: keep only the KV hints whose
    /// messages the new request sends with the same bytes. The anchor (frozen
    /// prefix and the first task) keeps its checkpoint, so the warm turn stays
    /// warm up to there.
    fn keep_shared_hints(&mut self, options: &str, request: ModelRequest<'_>) {
        if self.suffix_replaced {
            let shared = self.prompt.shared_messages(options, request.messages.iter().map(|m| m.serialized_json()));
            self.kv_hints.retain(|h| h.messages <= shared);
        }
    }

    pub fn context(&self) -> &crate::contract::Context {
        &self.contract.context
    }
    pub fn begin_run(&mut self) {
        self.contract.begin_run();
        self.work_remaining = None;
        self.reply_tokens = None;
    }
    /// Count a closed transcript with the negotiated renderer, without inference
    /// or mutations to the generation prefix, telemetry, affinity or KV state.
    pub(crate) fn count_rendered_history(
        &self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
    ) -> (
        io::Result<crate::contract::InputCount>,
        crate::contract::Context,
    ) {
        let Some(caps) = self
            .context()
            .capabilities
            .as_ref()
            .filter(|c| c.exact_admission)
        else {
            return (
                Ok(crate::contract::InputCount::Unknown),
                crate::contract::Context::default(),
            );
        };
        let body = (|| {
            let options = self.contract.decorate(request.serialized_options())?;
            let mut prompt = crate::prompt::AppendOnlyRequest::default();
            prompt
                .prepare(
                    &options,
                    request.messages.iter().map(|m| m.serialized_json()),
                )
                .map(Vec::from)
        })();
        match body {
            Ok(body) => crate::contract::Contract::count_request(
                caps,
                &self.endpoint,
                &body,
                &self.pool,
                cancel,
            ),
            Err(error) => (Err(error), crate::contract::Context::default()),
        }
    }
    /// Two cutoff recoveries without a closed tool round must change strategy.
    /// Derive this from journaled prompts so resume does not restart the loop.
    fn refresh_tool_recovery(&mut self, request: ModelRequest<'_>) {
        let recovery = |text: &str| text.starts_with(crate::agent::TOOL_CONTINUATION_PROMPT)
            || text.starts_with(crate::agent::REPLY_BUDGET_PROMPT)
            || text.starts_with("The tool call ") && text.contains(" (untrusted tool/path text) was discarded");
        let mut recoveries = 0;
        for message in request.messages.iter().rev() {
            if message.role() == "tool" { break; }
            if !matches!(message.role(), "user" | "developer") { continue; }
            let text = message.value().get("content").and_then(json::Value::as_str).unwrap_or("");
            if recovery(text) { recoveries += 1; }
            else if message.role() == "user" && !matches!(text.trim(), "Continue" | "Continue.") { break; }
            if recoveries == 2 { break; }
        }
        let required = recoveries == 2 && !self.contract.options.disable_tools
            && request.options.get("tool_choice").is_none()
            && request.options.get("tools").and_then(json::Value::as_array).is_some_and(|tools| !tools.is_empty());
        if self.contract.tool_recovery != required {
            self.reestablish_prompt(crate::contract::PromptBoundary::ToolRecovery);
            self.contract.tool_recovery = required;
        }
    }

    /// Bind lifecycle bookkeeping to a journal name; this is never sent upstream.
    pub fn bind_branch(&mut self, name: &str) -> io::Result<()> {
        if !crate::journal::valid_name(name) {
            return Err(invalid("invalid checkpoint branch name"));
        }
        self.kv_branch = name.to_owned();
        Ok(())
    }

    /// A child or branch gets independent prompt/contract state and shares only
    /// bounded handle ownership. Each client has its own HTTP pool. Use Conversation's explicit
    /// inherit_checkpoint_hints for a child; a journal fork restores them itself.
    pub fn branch_client(&self) -> Self {
        let pool = ConnectionPool::default();
        Self {
            endpoint: self.endpoint.clone(),
            pool,
            prompt: crate::prompt::AppendOnlyRequest::default(),
            contract: crate::contract::Contract::native(self.contract.options),
            kv_session: Arc::clone(&self.kv_session),
            kv_hints: Vec::new(),
            kv_branch: crate::journal::fresh_name(),
            progress: None,
            thinking: None,
            estimate: None,
            suffix_replaced: false,
            prewarm_compaction: self.prewarm_compaction,
            warm_pending: false,
            frozen_prefix_messages: None,
            counted: None,
            work_remaining: None,
            reply_tokens: None,
            server_cancellation: self.server_cancellation,
            cancel_confirmed: None,
        }
    }
    /// Call after a journal branch was deleted. Shared parent/child references
    /// survive. Release is best effort; TTL/eviction still bound remote lifetime.
    pub fn release_branch(&self, name: &str, hints: &[crate::kv::Hint]) -> ReleaseSummary {
        self.kv_session.forget(
            name,
            hints,
            self.contract
                .context
                .capabilities
                .as_ref()
                .and_then(|c| c.kv_checkpoints.as_ref()),
        )
    }
    /// Delete one locked transcript branch and then release its unshared hints.
    /// The owner must first join that branch's worker and relinquish its history.
    pub fn delete_branch(
        &self,
        journal: crate::journal::Journal,
        workspace: &crate::tools::Workspace,
    ) -> io::Result<ReleaseSummary> {
        let name = journal.name().to_owned();
        let hints = journal.delete(workspace)?;
        Ok(self.release_branch(&name, &hints))
    }
    /// Owner shutdown hook, after child workers are joined. Also runs on last
    /// branch-client drop. A failed release never invalidates transcript work.
    pub fn release_checkpoints(&self) -> ReleaseSummary {
        self.kv_session.release_all()
    }

    /// Validate an HTTP endpoint without resolving or connecting to it.
    /// HTTPS fails explicitly; credentials come from --api-key-file or VADE_API_KEY_FILE.
    pub fn new(base_url: &str) -> io::Result<Self> {
        let endpoint = Endpoint::parse(base_url)?;
        let pool = ConnectionPool::default();
        Ok(Self {
            kv_session: Arc::new(kv::Session::new(endpoint.clone())),
            kv_hints: Vec::new(),
            kv_branch: crate::journal::fresh_name(),
            endpoint,
            pool,
            prompt: crate::prompt::AppendOnlyRequest::default(),
            contract: crate::contract::Contract::native(
                crate::contract::GenerationOptions::default(),
            ),
            progress: None,
            thinking: None,
            estimate: None,
            suffix_replaced: false,
            prewarm_compaction: false,
            warm_pending: false,
            frozen_prefix_messages: None,
            counted: None,
            work_remaining: None,
            reply_tokens: None,
            server_cancellation: false,
            cancel_confirmed: None,
        })
    }
}

impl Model for NativeModel {
    fn set_work_remaining(&mut self, remaining: Option<Duration>) {
        self.work_remaining = remaining;
    }
    fn context_usage(&self) -> Option<(u64, u64)> {
        let context = self.context();
        let (prompt, limit) = match context.input {
            crate::contract::InputCount::Exact {
                prompt_tokens,
                context_limit,
                ..
            } => {
                (Some(prompt_tokens), Some(context_limit))
            }
            _ => (None, context.capabilities.as_ref().map(|c| c.context_limit)),
        };
        let used = context
            .turns
            .back()
            .filter(|t| !t.invalid)
            .and_then(|t| t.usage.prompt_tokens.zip(t.usage.completion_tokens))
            .and_then(|(p, c)| p.checked_add(c))
            .or(prompt)?;
        Some((used, limit?))
    }
    fn reply_limit_tokens(&self, fallback: usize) -> usize {
        self.reply_tokens.unwrap_or_else(|| self.contract.reply_limit_tokens(fallback))
    }
    fn output_reserve_tokens(&self, fallback: usize) -> usize {
        self.contract.output_reserve_tokens(fallback)
    }
    fn automatic_output(&self) -> bool {
        self.contract.automatic_reservation()
    }
    fn prompt_rewritten(&mut self) {
        self.reestablish_prompt(crate::contract::PromptBoundary::Compaction);
        // Automatic rewrites already lead directly to this prompt's real
        // generation. An extra one-token request cannot improve that path.
        self.warm_pending = false;
    }
    fn checkpoint_hints(&self) -> Option<&[crate::kv::Hint]> {
        Some(&self.kv_hints)
    }
    fn count_input(
        &mut self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
    ) -> io::Result<crate::contract::InputCount> {
        self.refresh_tool_recovery(request);
        if !cancel.load(Ordering::Relaxed) {
            if let Some(counted) = &self.counted {
                if request.to_json()? == counted.original {
                    return Ok(counted.input.clone());
                }
            }
        }
        self.counted = None;
        let model = request
            .options
            .get("model")
            .and_then(json::Value::as_str)
            .ok_or_else(|| invalid("native request needs an explicit model"))?;
        let generation = self.context().capability_reconciliations;
        self.contract
            .negotiate(&self.endpoint, model, &self.pool, Some(cancel))?;
        if self.context().capability_reconciliations != generation {
            self.reestablish_prompt(crate::contract::PromptBoundary::EngineBuild);
        }
        let automatic = self
            .contract
            .automatic_output(request.serialized_options())?;
        let options = self.contract.decorate(request.serialized_options())?;
        let mut builder = crate::prompt::AppendOnlyRequest::default();
        let body = builder
            .prepare(
                &options,
                request.messages.iter().map(|m| m.serialized_json()),
            )?
            .to_vec();
        let (body, _) = super::cancel::prepare(&self.endpoint, &body,
            self.server_cancellation && self.contract.context.capabilities.is_some())?;
        let result = self.contract.admit_adjusted(
            &self.endpoint,
            &body,
            &self.pool,
            Some(cancel),
            automatic,
        );
        // Refused admission still supplies an exact count for compaction. All
        // malformed/HTTP/cancellation errors remain errors; no heuristic downgrade.
        let input = self.contract.context.input.clone();
        let body = match result {
            Ok(body) => body,
            Err(error)
                if error.kind() == io::ErrorKind::InvalidInput
                    && matches!(
                        input,
                        crate::contract::InputCount::Exact {
                            admitted: false,
                            ..
                        }
                    ) =>
            {
                body
            }
            Err(error) => return Err(error),
        };
        self.counted = Some(CountedRequest {
            original: request.to_json()?,
            body,
            input: input.clone(),
        });
        Ok(input)
    }
    fn warm_compacted(&mut self, request: ModelRequest<'_>, cancel: &AtomicBool) -> io::Result<()> {
        if !self.warm_pending {
            return Ok(());
        }
        self.warm_pending = false; // no automatic retry on warm failure
        let caps = self
            .contract
            .context
            .capabilities
            .as_ref()
            .filter(|c| c.exact_admission && c.prefix_cache)
            .ok_or_else(|| {
                invalid("compaction pre-warm requires exact admission and prefix cache")
            })?;
        if std::env::var_os("VADE_HARNESS_PREFIX_CACHE").is_some_and(|v| v == "0")
            || request
                .options
                .get("cache")
                .is_some_and(|v| *v == json::Value::Bool(false))
        {
            return Err(invalid("compaction pre-warm requires cache enabled"));
        }
        let options = self.contract.decorate(request.serialized_options())?;
        if self.suffix_replaced {
            // As `keep_shared_hints`, field by field while `caps` is borrowed.
            let shared = self.prompt.shared_messages(&options, request.messages.iter().map(|m| m.serialized_json()));
            self.kv_hints.retain(|h| h.messages <= shared);
            let prefix = request
                .messages
                .iter()
                .take_while(|m| matches!(m.role(), "system" | "developer"))
                .count();
            if self.frozen_prefix_messages.is_some_and(|n| n != prefix) {
                return Err(invalid("compaction changed the frozen system/tools prefix"));
            }
            self.prompt.replace_suffix(
                &options,
                request.messages.iter().map(|m| m.serialized_json()),
                prefix,
            )?;
            self.frozen_prefix_messages = Some(prefix);
            self.suffix_replaced = false;
        }
        // Keep tools and all rendering fields as frozen raw JSON; override only
        // generation controls. Engine has no prefill-only route and rejects 0.
        let mut warm_options = String::from(
            "{\"max_tokens\":1,\"stream\":false,\"temperature\":0,\"top_p\":1,\"top_k\":1,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0",
        );
        for (key, value) in json::view(&options)?
            .fields()
            .ok_or_else(|| invalid("invalid warm options"))?
        {
            if matches!(
                key.as_ref(),
                "max_tokens"
                    | "max_completion_tokens"
                    | "stream"
                    | "temperature"
                    | "top_p"
                    | "top_k"
                    | "repetition_penalty"
                    | "presence_penalty"
                    | "frequency_penalty"
                    | "session"
                    | "fork"
                    | "cancel"
                    | "kv_checkpoint"
                    | "kv_handle"
            ) {
                continue;
            }
            warm_options.push(',');
            json::write_string(&key, &mut warm_options);
            warm_options.push(':');
            warm_options.push_str(value.raw());
        }
        warm_options.push('}');
        let mut builder = crate::prompt::AppendOnlyRequest::default();
        let body = builder.prepare(
            &warm_options,
            request.messages.iter().map(|m| m.serialized_json()),
        )?;
        let (admission, probe) = crate::contract::Contract::count_request(
            caps,
            &self.endpoint,
            body,
            &self.pool,
            cancel,
        );
        self.contract.context.requests.admissions += probe.requests.admissions;
        self.contract.context.requests.admitted += probe.requests.admitted;
        self.contract.context.requests.refused += probe.requests.refused;
        self.contract.context.requests.http_429 += probe.requests.http_429;
        self.contract.context.requests.http_503 += probe.requests.http_503;
        admission?;
        let started = Instant::now();
        self.contract.context.failure =
            Some("compaction pre-warm failed; ordinary generation remains available");
        self.contract.context.requests.generations += 1;
        let mut response = request_native_pooled(&self.endpoint, body, cancel, &self.pool, None)?;
        self.contract.context.requests.status(response.status);
        if response.status != 200 {
            return Err(io::Error::other("compaction pre-warm HTTP failure"));
        }
        let mut bytes = Vec::new();
        response
            .body
            .by_ref()
            .take((json::MAX_INPUT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "pre-warm canceled",
            ));
        }
        // Validate framing and finish without passing any provisional output to
        // the agent. A truncated/malformed warm reply never counts as success.
        AssistantTurn::from_response(&json::parse_bytes(&bytes)?, request.wire)?;
        let mut usage = crate::contract::UsageCollector::default();
        usage.observe(json::view(
            std::str::from_utf8(&bytes).map_err(|_| invalid("warm response encoding"))?,
        )?);
        let (usage, invalid_usage) = usage.finish();
        self.contract.context.compaction_warm = Some((
            if invalid_usage {
                crate::contract::Usage::default()
            } else {
                usage
            },
            started.elapsed(),
        ));
        self.contract.context.failure = None;
        response.recycle();
        Ok(())
    }
    fn complete(
        &mut self,
        model_request: ModelRequest<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        self.cancel_confirmed = None;
        notify_progress(&self.progress, NativePhase::Preparing);
        let deadline = self.contract.native_reply_time(self.work_remaining)
            .map(|time| Instant::now() + time);
        self.reply_tokens = (self.contract.options.reply_tokens != Some(0))
            .then(|| self.contract.native_reply_tokens(self.work_remaining));
        let mut draft = None;
        let result = self.complete_inner(model_request, cancel, on_text, deadline, &mut draft);
        let result = match result {
            Err(error)
                if error.kind() == io::ErrorKind::TimedOut
                    && !cancel.load(Ordering::Relaxed)
                    && deadline.is_some_and(|at| Instant::now() >= at) =>
            {
                // Keep streamed prose/reasoning; discard only the uncommitted
                // tool group and any buffered partial SSE/JSON event.
                if self.contract.context.first_token_elapsed.is_none() {
                    let message = "Reply deadline expired before the first generated token";
                    self.contract.context.failure = Some(message);
                    Err(io::Error::new(io::ErrorKind::TimedOut, message))
                } else {
                    self.contract.context.failure = Some("Reply budget exhausted; uncommitted tool group discarded. Continue in smaller steps.");
                    let mut assembly = draft.unwrap_or_else(|| ChatAccumulator::new(model_request.wire));
                    assembly.cap_reply(0, true);
                    assembly.into_turn().map(AssistantTurn::with_reply_budget_cutoff)
                }
            }
            result => result,
        };
        notify_progress(&self.progress, NativePhase::Idle);
        result
    }
}

pub(super) fn notify_progress(observer: &Option<ProgressObserver>, phase: NativePhase) {
    if let Some(observer) = observer {
        observer(phase);
    }
}

impl NativeModel {
    fn complete_inner(
        &mut self,
        model_request: ModelRequest<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
        deadline: Option<Instant>,
        draft: &mut Option<ChatAccumulator>,
    ) -> io::Result<AssistantTurn> {
        let started = Instant::now();
        self.refresh_tool_recovery(model_request);
        let counted = self
            .counted
            .take()
            .filter(|c| model_request.to_json().is_ok_and(|body| body == c.original));
        if counted.as_ref().is_some_and(|c| {
            matches!(
                c.input,
                crate::contract::InputCount::Exact {
                    admitted: false,
                    ..
                }
            )
        }) {
            self.contract.context.failure =
                Some("The chat does not fit the model's context; nothing was sent.");
            return Err(io::Error::other("exact token admission refused"));
        }
        let progress = self.progress.clone();
        let thinking_observer = self.thinking.clone();
        let estimate_observer = self.estimate.clone();
        let mut thinking = ThinkingProjection::default();
        let mut phase = NativePhase::Queued;
        let mut prefix = crate::terminal::AnswerPrefix::default();
        let mut visible_text = |text: &str| {
            let text = prefix.visible(text);
            if text.is_empty() {
                Ok(())
            } else {
                on_text(&text)
            }
        };
        self.contract.reset_attempt();
        let check_cancel = || {
            if cancel.load(Ordering::Relaxed) {
                Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "request canceled",
                ))
            } else {
                Ok(())
            }
        };
        check_cancel()?;
        let model = model_request
            .options
            .get("model")
            .and_then(json::Value::as_str)
            .ok_or_else(|| invalid("native request needs an explicit model"))?;
        if counted.is_none() {
            let generation = self.context().capability_reconciliations;
            self.contract
                .negotiate(&self.endpoint, model, &self.pool, Some(cancel))?;
            if self.context().capability_reconciliations != generation {
                self.reestablish_prompt(crate::contract::PromptBoundary::EngineBuild);
            }
        }
        let options = self.contract.decorate(model_request.serialized_options())?;
        self.keep_shared_hints(&options, model_request);
        let caps = self
            .contract
            .context
            .capabilities
            .as_ref()
            .and_then(|c| c.kv_checkpoints.clone());
        let managed =
            kv::managed(model_request, caps.as_ref()) && self.kv_session.can_track(&self.kv_branch);
        let mut hints = if managed {
            kv::eligible(
                if model_request.kv_hints.is_empty() {
                    &self.kv_hints
                } else {
                    model_request.kv_hints
                },
                caps.as_ref().unwrap(),
                model_request.messages.len(),
            )
        } else {
            Vec::new()
        };
        let selected = hints
            .iter()
            .max_by_key(|h| h.messages)
            .map(|h| h.checkpoint.clone());
        let branch = self.kv_branch.as_str();
        if let Some(caps) = &caps {
            self.kv_session.remember(branch, &hints, caps);
        }
        let prefix_messages = model_request
            .messages
            .iter()
            .take_while(|m| matches!(m.role(), "system" | "developer"))
            .count();
        if self.suffix_replaced
            && self
                .frozen_prefix_messages
                .is_some_and(|n| n != prefix_messages)
        {
            return Err(invalid("compaction changed the frozen system/tools prefix"));
        }
        let records = model_request
            .messages
            .iter()
            .map(|message| message.serialized_json());
        let body = if self.suffix_replaced {
            self.prompt
                .replace_suffix(&options, records, prefix_messages)?
        } else {
            self.prompt.prepare(&options, records)?
        };
        self.suffix_replaced = false;
        self.frozen_prefix_messages = Some(prefix_messages);
        let decorated = if managed {
            Some(kv::decorate(body, selected.as_ref())?)
        } else {
            None
        };
        let body = decorated.as_deref().unwrap_or(body);
        let body = if let Some(counted) = counted.filter(|_| !managed) {
            // Count and dispatch share the identical bytes, with no extra round
            // trip. Admission never reserves resources; generation may reject.
            self.contract.context.input = counted.input;
            self.contract.context.request_bytes = counted.body.len();
            self.contract.context.prefix_pending = false;
            counted.body
        } else {
            let automatic = self
                .contract
                .automatic_output(model_request.serialized_options())?;
            let (body, _) = super::cancel::prepare(&self.endpoint, body,
                self.server_cancellation && self.contract.context.capabilities.is_some())?;
            self.contract.admit_adjusted(&self.endpoint, &body, &self.pool, Some(cancel), automatic)?
        };
        let (body, control) = super::cancel::prepare(&self.endpoint, &body,
            self.server_cancellation && self.contract.context.capabilities.is_some())?;
        let (result, confirmed) = super::cancel::run(control, Some(cancel), || {
        self.contract.context.requests.generations += 1;
        self.contract.context.last_http_stage = "generation";
        self.contract.context.last_http_status = None;
        let mut response = request_native_pooled_deadline(
            &self.endpoint,
            &body,
            cancel,
            &self.pool,
            Some(&|| notify_progress(&progress, NativePhase::Queued)),
            deadline,
        )?;
        self.contract.context.last_http_status = Some(response.status);
        self.contract.context.requests.status(response.status);
        if !(200..300).contains(&response.status) {
            return Err(response.failure());
        }
        if let (Some(observer), Some(estimate)) = (&estimate_observer, super::estimate::PrefillEstimate::from_head(&response.head)) {
            observer(estimate);
        }
        let content_type = response
            .head
            .get("content-type")
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let reply_tokens = self.reply_tokens.unwrap_or(usize::MAX);
        let mut capped = false;
        let mut usage = crate::contract::UsageCollector::default();
        let turn = if content_type.eq_ignore_ascii_case("text/event-stream") {
            self.contract.context.response_streamed = true;
            let assembly = draft.insert(ChatAccumulator::new(model_request.wire));
            let mut decoder = SseDecoder::default();
            let mut event = |data: &str| -> io::Result<bool> {
                check_cancel()?;
                assembly.push_sse_data(data, &mut visible_text)?;
                if data.trim() != "[DONE]" {
                    thinking.observe(json::view(data)?, &thinking_observer);
                    if let Some(next) = generated_phase(data, phase)? {
                        let elapsed = started.elapsed();
                        if self.contract.context.first_token_elapsed.is_none() {
                            self.contract.context.first_token_elapsed = Some(elapsed);
                        }
                        if next != phase {
                            phase = next;
                            notify_progress(&progress, phase);
                        }
                        self.contract.context.last_token_elapsed = Some(elapsed);
                    }
                    usage.observe_data(data)?;
                }
                check_cancel()?;
                capped = assembly.cap_reply(reply_tokens, false);
                // Except for a deliberate host rollback, validate through HTTP EOF;
                // [DONE] cannot conceal truncation or subsequent bad data.
                Ok(capped)
            };
            loop {
                check_cancel()?;
                let bytes = response.body.fill_buf()?;
                if bytes.is_empty() {
                    break;
                }
                let n = bytes.len();
                decoder.feed(bytes, &mut event)?;
                response.body.consume(n);
                if decoder.done { break; }
            }
            if !decoder.line.is_empty() || decoder.has_data {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "incomplete SSE event",
                ));
            }
            draft.take().unwrap().into_turn()?
        } else if content_type.eq_ignore_ascii_case("application/json") {
            let mut bytes = Vec::new();
            response
                .body
                .by_ref()
                .take((json::MAX_INPUT_BYTES + 1) as u64)
                .read_to_end(&mut bytes)?;
            let turn =
                AssistantTurn::from_response(&json::parse_bytes(&bytes)?, model_request.wire)?;
            usage.observe(json::view(
                std::str::from_utf8(&bytes).map_err(|_| invalid("response encoding"))?,
            )?);
            thinking.observe(
                json::view(std::str::from_utf8(&bytes).map_err(|_| invalid("response encoding"))?)?,
                &thinking_observer,
            );
            check_cancel()?;
            notify_progress(&progress, NativePhase::Generating);
            if let Some(text) = turn
                .message()
                .value()
                .get("content")
                .and_then(json::Value::as_str)
            {
                visible_text(text)?;
            }
            turn
        } else {
            return Err(invalid(
                "upstream response must be JSON or text/event-stream",
            ));
        };
        check_cancel()?;
        if capped {
            // Closing this socket requests cancellation; no partial tool call or
            // usage/KV receipt is accepted as an upstream completed response.
            drop(response);
            usage = crate::contract::UsageCollector::default();
        } else { response.recycle(); }
        self.contract.completed(usage);
        let authoritative = turn
            .message()
            .value()
            .get("reasoning_content")
            .and_then(json::Value::as_str)
            .filter(|text| !text.is_empty())
            .or_else(|| {
                turn.message()
                    .value()
                    .get("reasoning")
                    .and_then(json::Value::as_str)
            });
        thinking.completed(&thinking_observer, authoritative);
        if managed {
            let caps = caps.as_ref().unwrap();
            kv::update(
                &mut hints,
                selected.as_ref(),
                self.contract
                    .context
                    .turns
                    .back()
                    .and_then(|t| t.kv.as_ref()),
                caps,
                model_request.messages.len(),
            );
            self.kv_session.remember(branch, &hints, caps);
        }
        self.kv_hints = hints;
        Ok(if capped { turn.with_reply_budget_cutoff() } else { turn })
        });
        self.cancel_confirmed = confirmed;
        result
    }
}

// Role/usage/finish-only events are not generated tokens. Reasoning and tool
// calls may precede visible text; blank deltas retain an active thinking phase.
pub(super) fn generated_phase(data: &str, previous: NativePhase) -> io::Result<Option<NativePhase>> {
    let value = json::view(data)?;
    let Some(delta) = value
        .get("choices")
        .and_then(json::View::array)
        .and_then(|mut choices| choices.next())
        .and_then(|choice| choice.get("delta"))
    else {
        return Ok(None);
    };
    let nonempty = |value: Option<json::View<'_>>| {
        value
            .and_then(json::View::as_str)
            .is_some_and(|s| !s.is_empty())
    };
    let text = delta.get("content").and_then(json::View::as_str);
    let tools = delta
        .get("tool_calls")
        .and_then(json::View::array)
        .is_some_and(|mut calls| {
            calls.any(|call| {
                call.get("function").is_some_and(|function| {
                    nonempty(function.get("name")) || nonempty(function.get("arguments"))
                })
            })
        });
    if tools || text.as_ref().is_some_and(|text| !text.trim().is_empty()) {
        return Ok(Some(NativePhase::Generating));
    }
    if ["reasoning_content", "reasoning"]
        .iter()
        .any(|key| nonempty(delta.get(key)))
    {
        return Ok(Some(NativePhase::Thinking));
    }
    Ok(text.filter(|text| !text.is_empty()).map(|_| {
        if previous == NativePhase::Thinking {
            previous
        } else {
            NativePhase::Generating
        }
    }))
}
