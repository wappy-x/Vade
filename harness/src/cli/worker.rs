use std::io;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};
use vade_harness::{
    agent::{self, Limits, RunBudget},
    app_server, child_agents, coding_tools, context, project, prompt,
    protocol::Completion,
    session::{MAX_PROMPT_BYTES, SYSTEM_PROMPT},
    terminal, transport,
};

use super::{
    App, MAX_DISPLAY_BYTES, RunState,
    approval::ToolPreviews,
    args::Mode,
    delegated::{delegated_turn, executable_path},
    events::{Pending, TurnResult, WorkSender, WorkerResult, error_category},
    native_setup::{CompactingJournal, NativeResult, SharedJournal, native_message},
};
use crate::{delegation, interaction};

struct ActiveTools<'a, T> {
    inner: T,
    events: &'a WorkSender,
}
impl<T: agent::ToolExecutor> agent::ToolExecutor for ActiveTools<'_, T> {
    fn last_side_effect_uncertain(&self) -> bool {
        self.inner.last_side_effect_uncertain()
    }
    fn take_pending_check(&mut self) -> Option<vade_harness::json::Value> {
        self.inner.take_pending_check()
    }
    fn check_failing(&self) -> bool {
        self.inner.check_failing()
    }
    fn check_enabled(&self) -> bool {
        self.inner.check_enabled()
    }
    fn check_not_run(&self) -> bool {
        self.inner.check_not_run()
    }

    fn execute(
        &mut self,
        invocation: agent::ToolInvocation<'_>,
        cancel: &AtomicBool,
    ) -> Result<String, agent::ToolError> {
        self.events.state(RunState::ToolRunning);
        let label = super::approval::tool_label(&invocation.call.name, invocation.arguments);
        while self
            .events
            .fact(super::events::WorkerEvent::ToolCall(label.clone(), String::new(), 0))
            .is_err()
        {
            if cancel.load(Ordering::Relaxed) {
                return Err(agent::ToolError::Cancelled);
            }
            std::thread::park_timeout(Duration::from_millis(10));
        }
        let result = self.inner.execute(invocation, cancel);
        self.events.state(RunState::Generating);
        result
    }
}

impl App {
    /// Manual compaction warms while idle, on the ordinary cancellable worker.
    /// The immutable journal view remains on the UI; the client returns even on
    /// failure so no affinity, sockets or cumulative answer usage is lost.
    pub(crate) fn start_compaction_warm(&mut self) -> io::Result<()> {
        if !self.config.prewarm_compaction {
            return Ok(());
        }
        if self.pending.is_some() {
            return Err(io::Error::other("pre-warm worker already active"));
        }
        let c = self
            .native
            .as_ref()
            .ok_or_else(|| io::Error::other("native history unavailable"))?
            .clone();
        let mut model = self
            .native_model
            .take()
            .ok_or_else(|| io::Error::other("native client unavailable"))?;
        let limits = Limits {
            max_context_tokens: self.config.context_budget(&self.context),
            output_reserve_tokens: self.config.output_reserve(&self.context),
            ..Limits::default()
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let (tx, rx) = interaction::lanes();
        let (controls, _) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            use agent::Model;
            tx.state(RunState::Queued);
            let result = (|| {
                let input = model.count_input(c.request(limits.wire), &worker_cancel)?;
                if !context::Usage::for_conversation(&c)
                    .with_count(&input)?
                    .fits(limits.max_context_tokens, limits.output_reserve_tokens)
                {
                    return Err(io::Error::other("compacted context exceeds token budget"));
                }
                model.warm_compacted(c.request(limits.wire), &worker_cancel)
            })();
            tx.finish(WorkerResult {
                http: None,
                backend: None,
                result: Ok(TurnResult::Prewarmed(model, result)),
            });
        });
        self.pending = Some(Pending {
            budget: None,
            native_progress: None,
            user: String::new(),
            started: Instant::now(),
            cancel,
            rx,
            controls,
            steering_pending: Vec::new(),
            render_skipped: false,
            render_skip: 0,
            round_displayed: false,
            received: false,
            canceled: false,
            displayed: 0,
            provisional: String::new(),
            edits: 0,
            worker: Some(worker),
            join_on_drop: true,
        });
        self.run_notice(
            "Warming compacted context (one discarded token); no checkpoint is pinned. /cancel closes the local socket; Engine work may continue.",
        )
    }
    pub(crate) fn steer(&mut self, text: String) -> io::Result<()> {
        if self.config.mode != Mode::Codex {
            return self.run_notice(
                "Live steering is available in Codex mode. Use /queue for a separate follow-up.",
            );
        }
        let Some(pending) = self.pending.as_mut() else {
            return self.run_notice("No active turn to steer.");
        };
        if pending.canceled
            || pending.steering_pending.len() >= 8
            || text.trim().is_empty()
            || text.len() > MAX_PROMPT_BYTES
        {
            return self.run_notice(
                "Steering not queued: cancelling, empty/oversized input, or 8 pending requests.",
            );
        }
        let Some(request_id) = self.steering_id.checked_add(1) else {
            return self.run_notice("Steering request identifiers exhausted.");
        };
        self.steering_id = request_id;
        if pending
            .controls
            .try_send(app_server::Control::Steer { request_id, text })
            .is_err()
        {
            return self.run_notice("Steering control queue unavailable; input was not sent.");
        }
        pending.steering_pending.push(request_id);
        self.run_notice(&format!(
            "Steering #{request_id} queued; waiting for Codex acknowledgment."
        ))
    }
    pub(crate) fn start(&mut self, user: String) -> io::Result<()> {
        let draft = super::attachments::display(&user).to_owned();
        let result = self.start_turn(user);
        if self.pending.is_none() {
            let kept = self.keep_draft(&draft);
            if !kept.is_empty() { self.notice(kept.trim())?; }
        }
        result
    }
    fn start_turn(&mut self, user: String) -> io::Result<()> {
        if user.trim().is_empty() {
            self.dirty = true;
            return Ok(());
        }
        if user.len() > MAX_PROMPT_BYTES {
            self.failed = true;
            return self.notice("Prompt exceeds 64 KiB; not sent.");
        }
        if self.pending.is_some() {
            if !matches!(self.config.mode, Mode::Direct | Mode::Agent) || self.queue_input {
                return self.queue(user);
            }
            // Several edits during cancellation become one steering message;
            // keep every instruction without starting overlapping generations.
            if let Some(text) = self.steering.as_mut() {
                if text.len() + user.len() + 2 > MAX_PROMPT_BYTES {
                    let kept = self.keep_draft(&user);
                    return self.notice(&format!("Steering exceeds 64 KiB; input was not sent.{kept}"));
                }
                text.push_str("\n\n");
                text.push_str(&user);
                return Ok(());
            }
            self.steering = Some(user);
            return self.cancel();
        }
        let user =
            if self.config.mode == Mode::Agent && user.trim().eq_ignore_ascii_case("continue") {
                self.timed_out_prompt.take().unwrap_or(user)
            } else {
                self.timed_out_prompt = None;
                user
            };
        let retrying = std::mem::take(&mut self.answers.retrying);
        let mut generation = self.answer_generation();
        generation.disable_tools = retrying;
        self.last_success = false;
        if !self.quiet && !self.internal_prompt(&user) {
            self.terminal.user_message(super::attachments::display(&user))?;
        }
        if !self.quiet && !self.internal_prompt(&user) { for line in user[super::attachments::display(&user).len()..].lines().filter(|line| line.starts_with("[file: ")) { self.terminal.note(line)?; } }
        self.terminal.begin_run();
        let mode = self.config.mode;
        if mode == Mode::Direct && (self.direct_prefix_blocked || self.session.pruned_turns > 0) {
            self.failed = true;
            return self.notice("The chat no longer fits the model's context; nothing was sent. Use /new before the next message.");
        }
        let endpoint = self.config.endpoint.clone().unwrap_or_default();
        if mode == Mode::Agent {
            let Some(conversation) = self.native.as_ref() else {
                self.failed = true;
                return self.run_notice("The conversation could not be recovered after an internal error. Use /new to start over.");
            };
            if self
                .journal
                .as_ref()
                .unwrap()
                .lock()
                .map_err(|_| io::Error::other("journal lock failed"))?
                .poisoned()
            {
                self.failed = true;
                return self.run_notice(
                    "Journal unavailable. Restart and resume, or /new; no prompt sent.",
                );
            }
            // Stage the complete candidate without changing the live history or
            // durable journal. Exact admission must precede token-fit decisions.
            let mut candidate = conversation.clone();
            if !retrying
                && candidate
                    .push_user(native_message("user", &user), Limits::default())
                    .is_err()
            {
                self.failed = true;
                return self.run_notice("The conversation is full; this message was not sent. Use /new to start over.");
            }
            let model = match self.native_model.take().map(Ok).unwrap_or_else(|| {
                transport::NativeModel::new(&endpoint).map(|m| {
                    m.with_server_cancellation().with_generation_options(generation)
                        .with_compaction_prewarm(self.config.prewarm_compaction)
                })
            }) {
                Ok(model) => model.with_generation_options(generation),
                Err(error) => {
                    self.failed = true;
                    return self.run_notice(&format!(
                        "Native setup failed ({}); prompt not sent.",
                        error_category(&error)
                    ));
                }
            };
            let limits = Limits {
                max_run_time: self.config.max_run_time,
                max_context_tokens: self.config.context_budget(&self.context),
                output_reserve_tokens: self.config.output_reserve(model.context()),
                ..Limits::default()
            };
            if !self.internal_prompt(&user) {
                self.reset_recall();
                self.input.remember_async(super::attachments::display(&user));
            }
            return self.start_prompt_count(user, candidate, model, limits, retrying);
        }
        if !self.internal_prompt(&user) {
            self.reset_recall();
            self.input.remember_async(super::attachments::display(&user));
        }
        self.start_generation(user, retrying, None, None)
    }

    pub(crate) fn start_generation(
        &mut self,
        user: String,
        retrying: bool,
        native_model: Option<transport::NativeModel>,
        budget: Option<RunBudget>,
    ) -> io::Result<()> {
        let mut generation = self.answer_generation();
        generation.disable_tools = retrying;
        let mode = self.config.mode;
        let messages = if matches!(mode, Mode::Codex | Mode::Hermes | Mode::Agent) {
            Vec::new()
        } else {
            prompt::build_messages(&format!("{SYSTEM_PROMPT}\n{}", vade_harness::session::model_identity(self.config.model.as_deref().unwrap_or("unspecified"))), self.session.history(), &user)
        };
        let endpoint = self.config.endpoint.clone().unwrap_or_default();
        let model = self.config.model.clone().unwrap_or_default();
        let options = match mode {
            Mode::Codex => Some(delegation::Options::Codex(app_server::Options {
                executable: executable_path(&self.config.codex_bin)?,
                workspace: self.workspace.root().to_path_buf(),
                thread_id: self
                    .delegated_id
                    .clone()
                    .or_else(|| self.config.thread.clone()),
                model: self.config.model.clone(),
                permission: self.config.permission.unwrap().protocol(),
            })),
            Mode::Hermes => Some(delegation::Options::Hermes(app_server::hermes::Options {
                executable: executable_path(&self.config.hermes_bin)?,
                workspace: self.workspace.root().to_path_buf(),
                session_id: self
                    .delegated_id
                    .clone()
                    .or_else(|| self.config.external_session.clone()),
            })),
            _ => None,
        };
        if mode.delegated() && self.backend.is_none() {
            let policy = if mode == Mode::Hermes {
                "Hermes launch · ACP v1 · permission callbacks denied · host filesystem/terminal capabilities disabled\nExternal Hermes owns tools, authentication, and persistence. No external OS sandbox or read boundary is verified.".to_owned()
            } else {
                format!(
                    "Codex launch · permission: {} · cwd: selected workspace · approval requests denied\nCodex read roots are not confined to this workspace by the selected preset.",
                    self.config.permission.unwrap().label()
                )
            };
            self.run_notice(&policy)?;
        }
        // Direct chat labels its reply with the Answer header; no extra line.
        if !self.quiet && !matches!(mode, Mode::Agent | Mode::Direct) {
            self.terminal.heading(match mode {
                Mode::Demo => "demo",
                Mode::Codex => "codex · provisional output",
                Mode::Hermes => "hermes · provisional output",
                _ => unreachable!(),
            })?;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = interaction::lanes();
        let tx = Arc::new(tx);
        let budget = budget
            .or_else(|| {
                self.answers
                    .continuing
                    .then(|| self.answers.budget.clone())
                    .flatten()
            })
            .unwrap_or_else(|| RunBudget::new(self.config.max_run_time));
        let ui_budget = (mode == Mode::Agent).then(|| budget.clone());
        let previews = Arc::new(ToolPreviews {
            tx: Arc::clone(&tx),
            bytes: Mutex::new(0),
            budget: (mode == Mode::Agent).then(|| budget.clone()),
        });
        let child_previews = Arc::clone(&previews);
        let mut scheduler = if mode == Mode::Agent && self.config.children {
            match self.children.prepare(
                self.native.as_ref().expect("native history prepared"),
                &endpoint,
                child_agents::NativeToolHost {
                    workspace: Arc::clone(&self.workspace),
                    approval: self.config.approval,
                    project: project::Snapshot::from_conversation(self.native.as_ref().unwrap())?,
                    shell_exclusions: self.config.shell_exclusions.clone(),
                    shell_toolchain: self.config.shell_toolchain.clone(),
                    preview: Arc::new(move |id, preview, cancel| {
                        child_previews.show(Some(id), preview, cancel)
                    }),
                },
                generation,
                self.config.child_grants.clone(),
                Arc::clone(&cancel),
            ) {
                Ok(scheduler) => Some(scheduler),
                Err(error) => {
                    self.native_model = native_model;
                    self.failed = true;
                    return self.run_notice(&format!("Child scheduler setup failed ({}); no model request sent. Typed user prompt retained; no automatic retry.", error_category(&error)));
                }
            }
        } else {
            None
        };
        let native = self.native.take();
        // Recover the direct client after an unexpected worker exit; creating
        // it only validates configuration and never connects on the UI thread.
        if mode == Mode::Direct && self.http.is_none() {
            self.http =
                Some(transport::ChatClient::new(&endpoint)?.with_generation_options(generation).with_server_cancellation());
        }
        let mut http = self
            .http
            .take()
            .map(|m| m.with_server_cancellation().with_generation_options(generation));
        let workspace = Arc::clone(&self.workspace);
        let journal = self.journal.clone();
        let policy = self.config.approval;
        let check = self
            .config
            .check
            .as_ref()
            .map(|(command, _)| command.clone());
        let auto_compact = self.config.auto_compact;
        let auto_continue = self.config.auto_continue;
        let compaction_trigger = Arc::clone(&self.auto_compaction);
        let context_tokens = self.config.context_budget(&self.context);
        let max_run_time = self.config.max_run_time;
        let output_reserve_tokens = self.config.output_reserve(&self.context);
        let shell_exclusions = self.config.shell_exclusions.clone();
        let build_session = Arc::clone(&self.build_session);
        let backend = self.backend.take();
        let turns = self.session.turns();
        self.answers.begin(&user, turns);
        if mode == Mode::Direct {
            self.answers.budget = Some(budget.clone());
        }
        let bytes = user.len();
        let (controls, control_rx) = mpsc::sync_channel(8);
        let worker_cancel = Arc::clone(&cancel);
        let delegated_user = user.clone();
        let worker = std::thread::spawn(move || {
            tx.state(RunState::Connecting);
            if let Some(mut model) = native_model {
                let thinking_tx = Arc::clone(&tx);
                model.set_thinking_observer(Some(Arc::new(move |update| match update {
                    transport::ThinkingUpdate::Delta(text) => thinking_tx.thinking_delta(text),
                    transport::ThinkingUpdate::Completed {
                        text,
                        tokens,
                        limited,
                    } => {
                        if let Some(id) = thinking_tx.thinking_complete(text, tokens, limited) {
                            let _ = thinking_tx.fact(super::events::WorkerEvent::ThinkingReady(id));
                        }
                    }
                })));
                let progress_tx = Arc::clone(&tx);
                model.set_progress_observer(Some(Arc::new(move |phase| match phase {
                    transport::NativePhase::Preparing => progress_tx.state(RunState::Connecting),
                    transport::NativePhase::Queued => progress_tx.state(RunState::Queued),
                    transport::NativePhase::Generating => progress_tx.state(RunState::Generating),
                    transport::NativePhase::Thinking => progress_tx.state(RunState::Thinking),
                    transport::NativePhase::Idle => (), // terminal/tool state owns the next update
                })));
                model.begin_run();
                let mut conversation = native.expect("native worker owns prepared conversation");
                let project = project::Snapshot::from_conversation(&conversation)
                    .expect("validated frozen project context");
                let mut observer = CompactingJournal {
                    inner: SharedJournal(journal.expect("native journal")),
                    events: Arc::clone(&tx),
                    enabled: auto_compact,
                    trigger: compaction_trigger,
                    answer: String::new(),
                    tool_prose: Arc::default(),
                    cancel: Arc::clone(&worker_cancel),
                };
                if let Ok(store) = observer.inner.0.lock() {
                    let _ = model.bind_branch(store.name());
                }
                let mut artifacts = SharedJournal(Arc::clone(&observer.inner.0));
                let executor = coding_tools::CodingTools::new(
                    &workspace,
                    policy,
                    |preview: coding_tools::Preview<'_>, cancel: &AtomicBool| {
                        previews.show(None, preview, cancel)
                    },
                )
                .with_artifacts(&mut artifacts)
                .with_shell_exclusions(shell_exclusions)
                .with_build_session(&build_session)
                .with_project(&project)
                .with_check(check.as_deref());
                let mut executor = ActiveTools {
                    inner: executor,
                    events: &tx,
                };
                let mut emitted = 0;
                let prose_count = Arc::clone(&observer.tool_prose);
                let mut display_limited = false;
                let mut on_text = |text: &str| {
                    if text.len() > MAX_DISPLAY_BYTES.saturating_sub(emitted) {
                        display_limited = true;
                    }
                    emit_delta(&tx, &worker_cancel, &mut emitted, text);
                    if let Ok(mut prose) = prose_count.lock() { prose.1 = emitted; }
                    // run's callback is infallible; signal its shared cancel
                    // flag to refuse commit/tool execution after overflow.
                    if display_limited {
                        worker_cancel.store(true, Ordering::Relaxed);
                    }
                };
                let (report, children) = if let Some(scheduler) = scheduler.as_mut() {
                    let result = child_agents::run_parent_observed_with_budget(
                        &mut conversation,
                        &mut model,
                        &mut executor,
                        Limits {
                            max_run_time,
                            auto_continue,
                            max_tool_calls: if retrying {
                                0
                            } else {
                                Limits::default().max_tool_calls
                            },
                            max_context_tokens: context_tokens,
                            output_reserve_tokens,
                            ..Limits::default()
                        },
                        scheduler,
                        &mut on_text,
                        &mut observer,
                        &budget,
                    );
                    (result.report, result.children)
                } else {
                    (
                        agent::run_observed_with_budget(
                            &mut conversation,
                            &mut model,
                            &mut executor,
                            Limits {
                                max_run_time,
                                auto_continue,
                                max_tool_calls: if retrying {
                                    0
                                } else {
                                    Limits::default().max_tool_calls
                                },
                                max_context_tokens: context_tokens,
                                output_reserve_tokens,
                                ..Limits::default()
                            },
                            &worker_cancel,
                            &mut on_text,
                            &mut observer,
                            &budget,
                        ),
                        Vec::new(),
                    )
                };
                model.set_progress_observer(None);
                model.set_thinking_observer(None);
                tx.finish(WorkerResult {
                    http: None,
                    result: Ok(TurnResult::Native(NativeResult {
                        model,
                        conversation,
                        report,
                        check_failing: agent::ToolExecutor::check_failing(&executor),
                        check_not_run: agent::ToolExecutor::check_not_run(&executor),
                        display_limited,
                        children,
                        answer: observer.answer,
                    })),
                    backend: None,
                });
                return;
            }
            if let Some(options) = options {
                let result = delegated_turn(
                    backend,
                    options,
                    &delegated_user,
                    &worker_cancel,
                    &control_rx,
                    &tx,
                );
                tx.finish(result);
                return;
            }
            let mut emitted = 0usize;
            let mut delta = |text: &str| emit_delta(&tx, &worker_cancel, &mut emitted, text);
            let result = if mode == Mode::Demo {
                let text = format!(
                    "This is Vade's local demo. No network request was made.\n\nYour prompt contains {bytes} bytes; {turns} complete exchanges preceded it.\nTry /ls, /read README.md, /status, or /paste. These workspace tools run only\nwhen you invoke them, and their output is never sent to a model.\n\nDirect mode is text chat with an explicit HTTP endpoint and model. Inference\nand KV state belong to the upstream engine."
                );
                for chunk in text.split_inclusive(' ') {
                    if worker_cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    delta(chunk);
                    std::thread::sleep(Duration::from_millis(14));
                }
                Ok(Completion {
                    text,
                    finish_reason: Some("stop".into()),
                })
            } else {
                let client = http.as_mut().expect("direct client exists");
                let thinking_tx = Arc::clone(&tx);
                client.set_thinking_observer(Some(Arc::new(move |update| match update {
                    transport::ThinkingUpdate::Delta(text) => thinking_tx.thinking_delta(text),
                    transport::ThinkingUpdate::Completed {
                        text,
                        tokens,
                        limited,
                    } => {
                        if let Some(id) = thinking_tx.thinking_complete(text, tokens, limited) {
                            let _ = thinking_tx.fact(super::events::WorkerEvent::ThinkingReady(id));
                        }
                    }
                })));
                let progress_tx = Arc::clone(&tx);
                client.set_progress_observer(Some(Arc::new(move |phase| match phase {
                    transport::NativePhase::Preparing => progress_tx.state(RunState::Connecting),
                    transport::NativePhase::Queued => progress_tx.state(RunState::Queued),
                    transport::NativePhase::Thinking => progress_tx.state(RunState::Thinking),
                    transport::NativePhase::Generating => progress_tx.state(RunState::Generating),
                    transport::NativePhase::Idle => (),
                })));
                let result = budget.with_cancellation(&worker_cancel, || {
                    client.stream(&model, &messages, &mut delta, Some(&worker_cancel))
                });
                client.set_progress_observer(None);
                client.set_thinking_observer(None);
                result
            };
            tx.finish(WorkerResult {
                http,
                result: result.map(TurnResult::Completed),
                backend: None,
            });
        });
        self.pending = Some(Pending {
            budget: ui_budget,
            native_progress: None,
            user,
            started: Instant::now(),
            cancel,
            rx,
            controls,
            steering_pending: Vec::new(),
            render_skipped: false,
            render_skip: 0,
            round_displayed: false,
            received: false,
            canceled: false,
            displayed: 0,
            provisional: String::new(),
            edits: 0,
            worker: Some(worker),
            join_on_drop: mode != Mode::Demo,
        });
        self.dirty = false;
        Ok(())
    }
    pub(crate) fn cancel(&mut self) -> io::Result<()> {
        let approval = self.close_approval().is_some();
        let Some(pending) = self.pending.as_mut() else {
            return Ok(());
        };
        pending.cancel.store(true, Ordering::Relaxed);
        pending.canceled = true;
        let received = pending.received;
        let counting = pending.rx.state() == RunState::Counting;
        if approval {
            self.run_notice("Cancelling the run; the pending operation will be declined.")?;
        }
        if !received {
            self.terminal.clear_editor()?;
        } else {
            self.terminal.line("")?;
        }
        match cancel_notice(self.config.mode, counting) {
            Some(text) => self.run_notice(text),
            None => Ok(()),
        }
    }
}
/// What a cancellation reports while it waits. A prompt still being counted
/// has reached no generation, tool or server cleanup; admission reports its
/// one outcome line instead.
pub(crate) fn cancel_notice(mode: Mode, counting: bool) -> Option<&'static str> {
    if counting { return None; }
    Some(match mode {
        Mode::Demo => "Demo cancellation requested; waiting for local cleanup.",
        Mode::Hermes => "Hermes interruption requested. Local response discarded; waiting for ACP prompt acknowledgment.",
        Mode::Codex => "Codex interruption requested. Local response discarded; waiting for provider cleanup.",
        Mode::Agent => "Native cancellation requested. Completed tool results are kept; waiting for cleanup.",
        _ => "Interrupting reply; keeping the partial answer. Server cancellation requested.",
    })
}
pub(crate) fn emit_delta(tx: &WorkSender, cancel: &AtomicBool, emitted: &mut usize, text: &str) {
    let mut rest = text;
    while !rest.is_empty() && !cancel.load(Ordering::Relaxed) && *emitted < MAX_DISPLAY_BYTES {
        if terminal::interrupted() {
            cancel.store(true, Ordering::Relaxed);
            return;
        }
        let mut count = rest.len().min(4096).min(MAX_DISPLAY_BYTES - *emitted);
        while !rest.is_char_boundary(count) {
            count -= 1;
        }
        if count == 0 {
            break;
        }
        tx.state(RunState::Generating);
        if !tx.render(&rest[..count]) {
            cancel.store(true, Ordering::Relaxed);
            return;
        }
        rest = &rest[count..];
        *emitted += count;
    }
    if !rest.is_empty() {
        cancel.store(true, Ordering::Relaxed);
    }
}
