//! Worker event delivery, completion handling and result summaries.
use std::collections::VecDeque;
use std::io;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::Instant;
use vade_harness::{app_server, protocol::Completion, terminal, transport};

use super::{App, MAX_DISPLAY_BYTES, RunState, args::Mode, native_setup::NativeResult};
use crate::{delegation, interaction};

pub(crate) enum TurnResult {
    Completed(Completion),
    Native(NativeResult),
    Prewarmed(transport::NativeModel, io::Result<()>),
    Counted(super::admission::Counted),
    Interrupted,
    NotStarted,
    Failed,
}
pub(crate) struct WorkerResult {
    pub(crate) http: Option<transport::ChatClient>,
    pub(crate) result: io::Result<TurnResult>,
    pub(crate) backend: Option<delegation::Backend>,
}
pub(crate) enum WorkerEvent {
    Budget(NativeProgress),
    Delta(String),
    ThinkingReady(u64),
    ToolCall(String, String, usize),
    AutoContinued(usize),
    Notice,
    HookNotice(String),
    /// The engine's warm-cache estimate line, before the first token.
    Estimate(String),
    ThreadReady(String),
    ToolApprovalClosed,
    ToolPreview {
        title: String,
        detail: String,
        reply: Option<super::approval::ApprovalReply>,
    },
    Steering {
        request_id: u64,
        status: app_server::SteeringStatus,
    },
    Done(WorkerResult),
}
pub(crate) type WorkSender = interaction::Sender<WorkerEvent, WorkerResult>;

pub(crate) struct NativeProgress {
    pub(crate) rounds: usize,
    pub(crate) limits: vade_harness::agent::Limits,
    pub(crate) context: Option<(u64, u64)>,
}

pub(crate) struct Pending {
    pub(crate) budget: Option<vade_harness::agent::RunBudget>,
    pub(crate) native_progress: Option<NativeProgress>,
    pub(crate) user: String,
    pub(crate) started: Instant,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) rx: interaction::Inbox<WorkerEvent, WorkerResult>,
    pub(crate) controls: mpsc::SyncSender<app_server::Control>,
    pub(crate) steering_pending: Vec<u64>,
    pub(crate) render_skipped: bool,
    pub(crate) render_skip: usize,
    pub(crate) round_displayed: bool,
    pub(crate) received: bool,
    pub(crate) canceled: bool,
    pub(crate) displayed: usize,
    pub(crate) provisional: String,
    /// Applied patches and written or removed files, for the end-of-run line.
    pub(crate) edits: usize,
    pub(crate) worker: Option<std::thread::JoinHandle<()>>,
    pub(crate) join_on_drop: bool,
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        // Disconnect before joining: release a producer blocked by backpressure.
        // A returned Client in a queued Done message is dropped here as well.
        self.rx.close();
        if self.join_on_drop {
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }
}
impl Pending {
    pub(crate) fn budget_status(&self) -> String {
        let Some((progress, budget)) = self.native_progress.as_ref().zip(self.budget.as_ref())
        else {
            return String::new();
        };
        let elapsed = budget.active_elapsed().as_secs();
        let time = |s: u64| {
            if s < 60 {
                format!("{s}s")
            } else if s.is_multiple_of(60) {
                format!("{}m", s / 60)
            } else {
                format!("{}m{}s", s / 60, s % 60)
            }
        };
        let work = progress
            .limits
            .max_run_time
            .filter(|t| !t.is_zero())
            .map_or_else(
                || format!("{} work", time(elapsed)),
                |limit| format!("{}/{}", time(elapsed), time(limit.as_secs())),
            );
        // Raw counts, never a percentage; the footer badge marks 80 % use.
        // Unknown context is left out rather than shown as a dash.
        let context = progress.context.and_then(|(used, limit)| {
            let limit = limit.min(progress.limits.max_context_tokens as u64);
            (limit > 0).then(|| format!(" · ctx {}/{}", terminal::clock::count(used), terminal::clock::count(limit)))
        }).unwrap_or_default();
        format!(
            "round {}/{} · {work}{context}",
            progress.rounds, progress.limits.max_tool_turns,
        )
    }
}
impl App {
    pub(crate) fn poll_worker(&mut self) -> io::Result<()> {
        let mut events = VecDeque::new();
        if let Some(pending) = &mut self.pending {
            for _ in 0..16 {
                match pending.rx.fact() {
                    Ok(event) => events.push_back(event),
                    Err(_) => break,
                }
            }
            if pending.rx.notice() {
                events.push_back(WorkerEvent::Notice);
            }
            match pending.rx.result() {
                Ok(result) => {
                    // A fact can arrive between the first drain and the result.
                    // Receiving Done establishes that all producer facts precede
                    // it; drain again before releasing the per-turn inbox.
                    while let Ok(event) = pending.rx.fact() {
                        events.push_back(event);
                    }
                    pending.rx.discard_render();
                    events.push_back(WorkerEvent::Done(result));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    events.push_back(WorkerEvent::Done(WorkerResult {
                        http: None,
                        result: Err(io::Error::other("request worker stopped")),
                        backend: None,
                    }))
                }
                Err(mpsc::TryRecvError::Empty) => {
                    let (text, skipped) = pending.rx.render();
                    pending.render_skipped |= skipped;
                    if !text.is_empty() {
                        events.push_back(WorkerEvent::Delta(text));
                    }
                }
            }
        }
        while let Some(event) = events.pop_front() {
            match event {
                WorkerEvent::Budget(progress) => {
                    if let Some(pending) = &mut self.pending {
                        pending.native_progress = Some(progress);
                    }
                    if !self.terminal.raw() && !self.quiet {
                        let status = self.pending.as_ref()
                            .map(Pending::budget_status).unwrap_or_default();
                        if !status.is_empty() {
                            self.run_notice(&status)?;
                        }
                    }
                    self.dirty = true;
                }
                WorkerEvent::ThinkingReady(id) => {
                    if let Some(pending) = &self.pending {
                        let frames = pending.rx.completed_thinking(Some(id));
                        for frame in frames {
                            self.present_thinking(frame)?;
                        }
                    }
                }
                WorkerEvent::AutoContinued(count) => {
                    self.run_notice(&format!(
                        "Continuing automatically ({count}/{}).",
                        vade_harness::agent::MAX_AUTO_CONTINUES
                    ))?;
                }
                WorkerEvent::ToolApprovalClosed => {
                    self.close_approval();
                    self.dirty = true;
                }
                WorkerEvent::ToolCall(label, prose, emitted) => {
                    let prose = terminal::answer_text(&prose);
                    let mut shown = self.pending.as_ref().map_or(0, |p| p.provisional.get(p.render_skip..).map_or(0, |round| terminal::answer_text(round).len()).min(prose.len()));
                    while !prose.is_char_boundary(shown) { shown -= 1; }
                    if let Some(pending) = &mut self.pending {
                        pending.render_skip = pending.render_skip.max(emitted);
                        pending.received = false;
                        if label.is_empty() { pending.round_displayed = true; }
                    }
                    if !self.quiet && !terminal::answer_text(prose).is_empty() && self.terminal.raw() {
                        self.terminal.answer(prose)?;
                        // Prose before a tool closes with its clock; a final
                        // answer gets the full run footer instead, once.
                        if !label.is_empty() {
                            self.terminal.answer_status(&format!("╰ {}", terminal::clock::now().0))?;
                        }
                    } else if !self.quiet && !self.terminal.raw() && shown < prose.len() {
                        self.terminal.stream_text(&prose[shown..])?;
                    } else {
                        self.terminal.discard_preview();
                    }
                    self.terminal.flush_stream()?;
                    if label.is_empty() {
                        self.terminal.end_round();
                        self.dirty = true;
                        continue;
                    }
                    self.tool_started = Some(Instant::now());
                    if self.quiet {
                        self.run_notice(&label)?;
                    } else {
                        self.terminal.tool_call(&label)?;
                    }
                    self.dirty = true;
                }
                WorkerEvent::ToolPreview {
                    title,
                    detail,
                    reply,
                } => {
                    if reply.as_ref().is_some_and(|reply| !reply.open()) {
                        continue;
                    }
                    if self.pending.as_ref().is_none_or(|p| p.canceled) {
                        if let Some(r) = reply {
                            let _ = r.send(false);
                        }
                        continue;
                    }
                    if let Some(reply) = &reply
                        && self.terminal.interactive()
                        && self.config.prompt.is_none()
                        && !self.quiet
                        && !self.approval_input_closed
                    {
                        self.close_approval();
                        self.terminal.flush_stream()?;
                        match self.terminal.approval_preview(&title, &detail) {
                            Ok(()) => {
                                self.approval_reply = Some(reply.clone());
                                self.approval_command = title == "Command";
                                if !self.terminal.raw() {
                                    self.approval_presented();
                                }
                            }
                            Err(error) => {
                                if let Some(approvals) = &self.approvals {
                                    approvals.not_shown_for(reply.asker());
                                }
                                let _ = reply.send(false);
                                self.run_notice(&error.to_string())?;
                            }
                        }
                        self.dirty = true;
                        continue;
                    }
                    let elapsed = self
                        .tool_started
                        .map(|started| terminal::clock::duration(started.elapsed()));
                    if matches!(title.as_str(), "File written" | "File removed") {
                        if let Some(pending) = &mut self.pending {
                            pending.edits += 1;
                        }
                    }
                    if title == "Patch preview" && !self.quiet {
                        self.terminal.flush_stream()?;
                        self.terminal.note(&title)?;
                        self.terminal.diff(&detail)?;
                    } else if !self.quiet
                        && !self.config.verbose
                        && matches!(title.as_str(), "Command result" | "Patch result")
                    {
                        // One summary row under the call; full output stays
                        // under --verbose.
                        let (summary, failed, first_line) = if title == "Command result" {
                            command_summary(&detail)
                        } else {
                            patch_summary(&detail)
                        };
                        let summary = match elapsed {
                            Some(elapsed) => format!("{summary} · {elapsed}"),
                            None => summary,
                        };
                        self.terminal.flush_stream()?;
                        self.terminal.tool_result(&summary, failed)?;
                        if let Some(line) = first_line {
                            self.terminal.tool_result(&line, failed)?;
                        }
                        self.tool_started = None;
                    } else if !self.quiet
                        && matches!(title.as_str(), "File written" | "File removed")
                    {
                        self.terminal.flush_stream()?;
                        self.terminal
                            .tool_result(&format!("{} · {detail}", title.to_lowercase()), false)?;
                    } else if self.config.verbose && title == "Command" {
                        // The sandbox description repeats on every command;
                        // log it once and again only when it changes.
                        let (command, sandbox) = detail.split_once('\n').unwrap_or((&detail, ""));
                        if sandbox == self.sandbox_note {
                            self.run_notice(&format!("{title}\n{command}"))?;
                        } else {
                            self.sandbox_note = sandbox.to_owned();
                            self.run_notice(&format!("{title}\n{detail}"))?;
                        }
                    } else if self.config.verbose
                        || (title == "Command" && detail.starts_with("Child "))
                        || !matches!(
                            title.as_str(),
                            "Command" | "Command result" | "Patch result"
                        )
                    {
                        self.run_notice(&format!("{title}\n{detail}"))?;
                    }
                    if let Some(reply) = reply {
                        let _ = reply.send(false);
                        self.warn(
                            "This step needs your approval, but no one can answer here.",
                            "Vade declined it; nothing ran.",
                            "Run Vade in a terminal to approve, or use --approval auto.",
                        )?;
                    }
                    self.dirty = true;
                }
                WorkerEvent::Delta(mut text) => {
                    // Flush the first available batch immediately. Subsequent
                    // adjacent deltas share one sanitized terminal write.
                    while text.len() < 16 * 1024
                        && matches!(events.front(), Some(WorkerEvent::Delta(_)))
                    {
                        if let Some(WorkerEvent::Delta(next)) = events.pop_front() {
                            text.push_str(&next);
                        }
                    }
                    let Some(pending) = self.pending.as_mut() else {
                        continue;
                    };
                    if !pending.canceled {
                        let skip = pending.render_skip.saturating_sub(pending.provisional.len()).min(text.len());
                        pending.provisional.push_str(&text);
                        text.drain(..skip);
                        if text.is_empty() { continue; }
                        pending.round_displayed = false;
                        if matches!(self.config.mode, Mode::Codex | Mode::Hermes | Mode::Agent) {
                            // Plain/redirected output contains only the authoritative
                            // completed answer; the raw UI labels deltas provisional.
                            if self.quiet {
                                continue;
                            }
                        }
                        if !pending.received {
                            self.terminal.clear_editor()?;
                            if !self.quiet && matches!(self.config.mode, Mode::Agent | Mode::Direct) {
                                self.terminal.answer_heading()?;
                            }
                            pending.received = true;
                            // The first visible answer is a readiness event.
                            // Paint it now; only subsequent streaming frames
                            // share the normal 33 ms redraw budget.
                            if self.terminal.raw() { self.dirty = true; }
                        }
                        pending.displayed += text.len();
                        if matches!(self.config.mode, Mode::Agent | Mode::Direct)
                            && self.terminal.raw()
                        {
                            self.terminal.preview_answer(&text);
                        } else {
                            self.terminal.stream_text(&text)?;
                        }
                        self.stream_dirty = true;
                    }
                }
                WorkerEvent::HookNotice(text) => { self.run_notice(&text)?; }
                WorkerEvent::Estimate(text) => {
                    // Only before the answer; redirected output keeps the answer alone.
                    if !self.quiet && matches!(self.config.mode, Mode::Agent | Mode::Direct)
                        && self.pending.as_ref().is_some_and(|p| !p.received && !p.canceled)
                    {
                        self.notice(&text)?;
                    }
                }
                WorkerEvent::Notice => {
                    if self.quiet {
                        vade_harness::diagnostics::stderr_line(format_args!(
                            "vade: External agent reported a notice; raw details suppressed."
                        ));
                    } else {
                        if self
                            .pending
                            .as_ref()
                            .is_some_and(|p| p.received && !p.canceled)
                        {
                            self.terminal.line("")?;
                        }
                        self.notice("External agent reported a notice; raw details suppressed.")?;
                    }
                }
                WorkerEvent::ThreadReady(id) => {
                    // This is provider identity, never a model name or cache key.
                    if self.delegated_id.as_deref() != Some(id.as_str()) {
                        let text = format!(
                            "{} {}: {}",
                            self.config.mode.provider(),
                            if self.config.mode == Mode::Hermes {
                                "session"
                            } else {
                                "thread"
                            },
                            terminal::sanitize(&id)
                        );
                        if self.quiet {
                            vade_harness::diagnostics::stderr_line(format_args!("vade: {text}"));
                        } else {
                            self.notice(&text)?;
                        }
                    }
                    self.delegated_id = Some(id);
                }
                WorkerEvent::Steering { request_id, status } => {
                    if let Some(pending) = self.pending.as_mut() {
                        pending.steering_pending.retain(|id| *id != request_id);
                    }
                    self.run_notice(&format!(
                        "Steering #{request_id}: {}.",
                        match status {
                            app_server::SteeringStatus::Accepted => "accepted by Codex",
                            app_server::SteeringStatus::Rejected => "rejected; not applied",
                            app_server::SteeringStatus::Unconfirmed =>
                                "outcome unconfirmed; no automatic resend",
                        }
                    ))?;
                }
                WorkerEvent::Done(result) => {
                    self.close_approval();
                    if let Some(pending) = &self.pending {
                        // Final framing closes all producers. A full fact lane
                        // may omit a display marker, never the completed text.
                        let frames = pending.rx.completed_thinking(None);
                        for frame in frames {
                            self.present_thinking(frame)?;
                        }
                    }
                    self.terminal.discard_thinking_preview();
                    let Some(mut pending) = self.pending.take() else {
                        continue;
                    };
                    if !pending.steering_pending.is_empty() {
                        self.run_notice(&format!("{} steering message(s) were not confirmed; Vade will not resend them.", pending.steering_pending.len()))?;
                    }
                    let result = match result.result {
                        Ok(TurnResult::Counted(counted)) => {
                            self.finish_counted(&mut pending, counted)?;
                            continue;
                        }
                        Ok(TurnResult::Prewarmed(model, result)) => {
                            self.context = model.context().clone();
                            self.native_model = Some(model);
                            let success = result.is_ok() && !pending.canceled;
                            // Warming is optional acceleration, never a failed
                            // user activation or a reason to hold the queue.
                            self.queue_paused = false;
                            self.last_state = Some(if pending.canceled { RunState::LocalCancelled } else { RunState::Completed });
                            if success {
                                self.run_notice("Compacted context pre-warmed: the engine has read the shorter chat, so the next reply can start sooner. Nothing was added to the chat.")?;
                                if self.config.verbose {
                                    self.run_notice(&self.context.describe())?;
                                }
                            } else {
                                self.run_notice("Compaction retained; optional pre-warm failed or was canceled. Continuing normally; the next real request remains available.")?;
                            }
                            self.dirty = true;
                            continue;
                        }
                        outcome => WorkerResult {
                            result: outcome,
                            ..result
                        },
                    };
                    // Steering typed exactly as a direct answer ended is not
                    // an interruption: the worker returned the complete
                    // answer, so record it normally and let the steering
                    // text start as the next turn. A late server-cancel
                    // refusal for the finished generation is then moot.
                    if pending.canceled
                        && self.config.mode == Mode::Direct
                        && matches!(&result.result, Ok(TurnResult::Completed(c)) if c.finish_reason.as_deref() == Some("stop"))
                    {
                        pending.canceled = false;
                    }
                    self.last_success = !pending.canceled
                        && matches!(&result.result, Ok(TurnResult::Completed(c)) if c.text.len() <= MAX_DISPLAY_BYTES);
                    if !self.last_success {
                        self.queue_paused = true;
                    }
                    self.last_state = Some(match &result.result {
                        _ if pending.canceled && !self.config.mode.delegated() => {
                            RunState::LocalCancelled
                        }
                        Ok(TurnResult::Completed(_)) => RunState::Completed,
                        Ok(TurnResult::Interrupted) => RunState::Interrupted,
                        Ok(TurnResult::NotStarted) => RunState::NotStarted,
                        Ok(TurnResult::Failed) => RunState::Failed,
                        _ => RunState::OutcomeUnknown,
                    });
                    self.terminal.flush_stream()?;
                    self.backend = result.backend;
                    self.http = result.http;
                    if let Some(http) = &self.http {
                        let mut context = http.context().clone();
                        if self.config.mode == Mode::Direct && self.answers.auto_continues > 0 {
                            context.continue_reply(&self.context, pending.started.elapsed());
                        } else { context.reply.wall = pending.started.elapsed(); }
                        self.context = context;
                    }
                    let result = match result.result {
                        Ok(TurnResult::Native(native)) => {
                            self.finish_native(&mut pending, native)?;
                            continue;
                        }
                        result => result,
                    };
                    if pending.canceled && self.config.mode == Mode::Direct {
                        let partial = format!("{}\n[interrupted]", pending.provisional);
                        if self.terminal.raw() { self.terminal.answer(&partial)?; }
                        else { self.terminal.line("\n[interrupted]")?; }
                        self.reset_recall();
                        let kept = self.session.interrupted(std::mem::take(&mut pending.user), partial)?;
                        if !kept || self.session.pruned_turns > 0 {
                            self.direct_prefix_blocked = true;
                            self.failed = true;
                            // Never delete the unsent correction: hold it as a
                            // queued follow-up for /new then /next.
                            let held = self.steering.take();
                            self.run_notice(if held.is_some() {
                                "Interrupted reply exceeds history capacity; your correction is held. Use /new, then /next to send it."
                            } else {
                                "Interrupted reply exceeds history capacity; use /new before sending more."
                            })?;
                            if let Some(text) = held { self.queue(text)?; }
                        }
                        if self.http.as_ref().is_some_and(|http| http.server_cancel_confirmed() == Some(false)) {
                            if let Some(text) = self.steering.take() { self.queue(text)?; }
                            self.failed = true;
                            self.run_notice("Server cancellation was not acknowledged; steering held. Check the server before sending again.")?;
                        }
                        self.answers.available = kept;
                        self.answers.cut_off = false;
                        self.last_state = Some(RunState::Interrupted);
                        self.dirty = true;
                        continue;
                    }
                    if pending.canceled {
                        self.failed = true;
                        let message = if self.config.mode.delegated() {
                            match result {
                                Ok(TurnResult::Interrupted) => {
                                    "Canceled · the external agent acknowledged interruption · this turn was not saved"
                                }
                                Ok(TurnResult::NotStarted) => {
                                    "Canceled before sending · nothing was sent"
                                }
                                Ok(TurnResult::Completed(_)) => {
                                    "Canceled after the external agent had finished · its output was not saved here"
                                }
                                Ok(TurnResult::Failed) => {
                                    "Canceled · the external agent also reported a failure · this turn was not saved"
                                }
                                _ => {
                                    "Canceled · the external agent's outcome is unknown · Vade stopped its process and will not resend"
                                }
                            }
                        } else {
                            "Canceled · not added to the chat · send it again when ready"
                        };
                        self.run_notice(message)?;
                        self.dirty = true;
                        continue;
                    }
                    if !pending.received {
                        self.terminal.clear_editor()?;
                    }
                    match result {
                        Ok(
                            TurnResult::Native(_)
                            | TurnResult::Prewarmed(_, _)
                            | TurnResult::Counted(_),
                        ) => unreachable!(),
                        Ok(TurnResult::Completed(completion)) => {
                            if self.config.mode == Mode::Direct {
                                self.answers.available = completion.text.len() <= MAX_DISPLAY_BYTES;
                                self.answers.cut_off =
                                    completion.finish_reason.as_deref() == Some("length");
                                if self.answers.cut_off && !self.config.auto_continue {
                                    self.warn(
                                        "The reply hit the output length limit and was cut off.",
                                        "Vade kept the part that arrived.",
                                        "Use /continue to get the rest.",
                                    )?;
                                }
                            }
                            let display_text = if self.config.mode == Mode::Direct {
                                terminal::answer_text(&completion.text)
                            } else {
                                &completion.text
                            };
                            let corrected = pending.received
                                && (pending.render_skipped
                                    || !display_text.starts_with(&pending.provisional));
                            if corrected {
                                self.terminal.line("")?;
                                self.run_notice(
                                    "Final answer (replaces the streamed text above):",
                                )?;
                            }
                            let labeled_reply = self.config.mode == Mode::Direct
                                && (self.terminal.raw() || (!self.quiet && (!pending.received || corrected)));
                            if labeled_reply {
                                let mut limit = display_text.len().min(MAX_DISPLAY_BYTES);
                                while !display_text.is_char_boundary(limit) {
                                    limit -= 1;
                                }
                                self.terminal.answer(&display_text[..limit])?;
                            } else if !pending.received || corrected {
                                let mut limit = display_text.len().min(MAX_DISPLAY_BYTES);
                                while !display_text.is_char_boundary(limit) {
                                    limit -= 1;
                                }
                                self.terminal.text(&display_text[..limit])?;
                            } else if completion.text.len() <= MAX_DISPLAY_BYTES {
                                self.terminal
                                    .text(&display_text[pending.provisional.len()..])?;
                            }
                            if !display_text.is_empty() && !labeled_reply {
                                self.terminal.line("")?;
                            }
                            if completion.text.len() > MAX_DISPLAY_BYTES {
                                self.answers.available = false;
                                self.direct_prefix_blocked = self.config.mode == Mode::Direct;
                                self.warn(
                                    "The reply is larger than the 1 MiB display limit.",
                                    "Vade did not add it to the chat history.",
                                    "Use /new to start a fresh chat.",
                                )?;
                            } else {
                                match self.complete_session(
                                    std::mem::take(&mut pending.user),
                                    completion.text,
                                ) {
                                    Ok(false) => {
                                        self.answers.available = false;
                                        self.direct_prefix_blocked =
                                            self.config.mode == Mode::Direct;
                                        if self.config.mode.delegated() {
                                            self.notice("This exchange is too large to keep for local recall; the external agent still has it.")?;
                                        } else {
                                            self.warn(
                                                "The chat history is full.",
                                                "This exchange was not saved to the history.",
                                                "Use /new before the next message.",
                                            )?;
                                        }
                                    }
                                    Ok(true) => {
                                        if self.config.mode == Mode::Direct
                                            && self.session.pruned_turns > 0
                                        {
                                            self.direct_prefix_blocked = true;
                                            self.warn(
                                                "The chat no longer fits the model's context.",
                                                "Vade will not trim or rewrite earlier messages on its own.",
                                                "Use /new before the next message.",
                                            )?;
                                        }
                                    }
                                    Err(_) => {
                                        self.answers.available = false;
                                        self.failed = true;
                                        self.warn(
                                            "Vade could not save the session counter to disk.",
                                            "No chat text was written to disk.",
                                            "Check free disk space and folder permissions.",
                                        )?;
                                    }
                                }
                            }
                            if self.config.mode != Mode::Direct && !self.quiet {
                                let finish = match completion.finish_reason.as_deref() {
                                    Some("length") => " · upstream length limit",
                                    Some("content_filter") => " · upstream content filter",
                                    _ => "",
                                };
                                self.terminal.note(&format!(
                                    "  {:.1}s · {} {}{finish}\n",
                                    pending.started.elapsed().as_secs_f32(),
                                    self.session.turns(),
                                    if self.config.mode.delegated() {
                                        "local completed turns"
                                    } else {
                                        "turns in context"
                                    }
                                ))?;
                            }
                            if self.config.mode == Mode::Direct && self.answers.available {
                                self.auto_continue_direct()?;
                            }
                            if self.config.mode == Mode::Direct && self.pending.is_none() {
                                let speed = self.context.reply_speed(std::time::Duration::ZERO, self.config.context_budget(&self.context) as u64, false);
                                if self.quiet { self.run_notice(&speed)?; } else { self.terminal.answer_status(&speed)?; }
                            }
                        }
                        Ok(
                            outcome @ (TurnResult::Interrupted
                            | TurnResult::NotStarted
                            | TurnResult::Failed),
                        ) => {
                            if pending.received {
                                self.terminal.line("")?;
                            }
                            self.failed = true;
                            let message = if matches!(outcome, TurnResult::Interrupted) {
                                "The external agent stopped this turn; it was not saved."
                            } else if matches!(outcome, TurnResult::NotStarted) {
                                "The external turn was not sent."
                            } else {
                                "The external agent reported a failure (details hidden); this turn was not saved."
                            };
                            if self.quiet {
                                vade_harness::diagnostics::stderr_line(format_args!("vade: {message}"));
                            } else {
                                self.notice(message)?;
                            }
                        }
                        Err(error) => {
                            if pending.received {
                                self.terminal.line("")?;
                            }
                            self.failed = true;
                            if self.config.mode == Mode::Direct
                                && self
                                    .answers
                                    .budget
                                    .as_ref()
                                    .is_some_and(vade_harness::agent::RunBudget::expired)
                            {
                                self.run_notice(&super::native_setup::time_limit_notice(
                                    self.config.max_run_time,
                                ))?;
                                self.dirty = true;
                                continue;
                            }
                            // Transport errors may include URLs or server response bodies.
                            // Print a category only, never the untrusted error string.
                            let what = format!(
                                "The request to {} failed ({}).",
                                if self.config.mode == Mode::Codex {
                                    "Codex"
                                } else if self.config.mode == Mode::Hermes {
                                    "Hermes"
                                } else {
                                    "the model"
                                },
                                error_category(&error),
                            );
                            let hint = if self.config.mode.delegated() {
                                ""
                            } else {
                                error_hint(&error).trim_start()
                            };
                            self.warn(
                                &what,
                                if self.config.mode.delegated() {
                                    "Its outcome is unknown; Vade saved nothing and will not resend it on its own."
                                } else {
                                    "Nothing was saved, and Vade will not resend it on its own."
                                },
                                if hint.is_empty() { "Send the message again when ready." } else { hint },
                            )?;
                        }
                    }
                    if self.config.mode == Mode::Direct && self.config.verbose {
                        self.run_notice(&self.context.describe())?;
                    }
                    self.dirty = true;
                }
            }
        }
        if let Some(pending) = &self.pending {
            let frames = pending.rx.thinking();
            for frame in frames {
                self.present_thinking(frame)?;
            }
        }
        Ok(())
    }
    fn present_thinking(&mut self, frame: interaction::ThinkingFrame) -> io::Result<()> {
        if !self.quiet && self.pending.as_ref().is_some_and(|p| !p.canceled) {
            if frame.completed {
                self.terminal
                    .thinking(&frame.text, frame.tokens, frame.limited)?;
            } else {
                self.terminal.preview_thinking(&frame.text, frame.limited);
            }
            self.stream_dirty = true;
        }
        Ok(())
    }
}
/// `status: Exited(Some(0))\nstdout:\n…\nstderr:\n…` → one row plus, on
/// failure, the first non-empty stderr/stdout line.
pub(crate) fn command_summary(detail: &str) -> (String, bool, Option<String>) {
    let status = detail
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("status: "))
        .unwrap_or("");
    let (outcome, failed) = match status
        .strip_prefix("Exited(Some(")
        .and_then(|s| s.strip_suffix("))"))
    {
        Some(code) => (format!("exit {code}"), code != "0"),
        None if status == "Exited(None)" => ("stopped by signal".to_owned(), true),
        None => (status.to_lowercase(), true),
    };
    let rest = detail
        .find("\nstdout:\n")
        .map_or("", |at| &detail[at + "\nstdout:\n".len()..]);
    let (stdout, stderr) = if let Some(stderr) = rest.strip_prefix("stderr:\n") {
        ("", stderr)
    } else {
        match rest.find("\nstderr:\n") {
            Some(at) => (&rest[..at], &rest[at + "\nstderr:\n".len()..]),
            None => (rest, ""),
        }
    };
    let count = |text: &str| text.lines().filter(|l| !l.trim().is_empty()).count();
    let mut parts = vec![outcome];
    match count(stdout) {
        0 => (),
        1 => parts.push("1 line".into()),
        n => parts.push(format!("{n} lines")),
    }
    match count(stderr) {
        0 => (),
        n => parts.push(format!("{n} stderr")),
    }
    let first_line = failed
        .then(|| {
            stderr
                .lines()
                .chain(stdout.lines())
                .map(str::trim)
                .find(|l| !l.is_empty())
                .map(|l| l.chars().take(160).collect::<String>())
        })
        .flatten();
    (parts.join(" · "), failed, first_line)
}
pub(crate) fn patch_summary(detail: &str) -> (String, bool, Option<String>) {
    if let Some(rest) = detail.strip_prefix("Applied ") {
        (
            format!("applied {}", rest.trim_end_matches('.')),
            false,
            None,
        )
    } else if vade_harness::json::parse(detail)
        .ok()
        .is_some_and(|result| {
            result.get("ok") == Some(&vade_harness::json::Value::Bool(false))
                && result.get("uncertain_writes") == Some(&vade_harness::json::Value::Bool(false))
        })
    {
        (
            "patch refused · no files written · request a new preview".into(),
            true,
            None,
        )
    } else {
        (
            "patch stopped · outcome uncertain · inspect the affected paths".into(),
            true,
            None,
        )
    }
}
pub(crate) use vade_harness::diagnostics::{error_category, error_hint};

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::io::Read;
    use std::os::fd::FromRawFd;
    use std::process::{Command, Stdio};
    use std::time::Duration;

    #[test]
    fn app_approval_requires_complete_preview_and_pauses_without_replying() {
        const CASE: &str = "VADE_APPROVAL_APP_TEST";
        unsafe extern "C" {
            fn openpty(
                master: *mut i32,
                slave: *mut i32,
                name: *mut u8,
                term: *const std::ffi::c_void,
                size: *const std::ffi::c_void,
            ) -> i32;
            fn ioctl(fd: i32, request: usize, ...) -> i32;
            fn fcntl(fd: i32, command: i32, ...) -> i32;
        }
        let resize = |rows: u16, columns: u16| {
            let size = [rows, columns, 0u16, 0u16];
            assert_eq!(unsafe { ioctl(1, 0x80087467, &size) }, 0);
        };
        if std::env::var_os(CASE).is_none() {
            // Isolate the real terminal descriptors; no global UI test switches
            // or upstream worker are needed to exercise App's fact lane.
            let (mut master, mut slave) = (0, 0);
            assert_eq!(
                unsafe {
                    openpty(
                        &mut master,
                        &mut slave,
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        std::ptr::null(),
                    )
                },
                0
            );
            assert_eq!(unsafe { fcntl(master, 4, fcntl(master, 3) | 4) }, 0);
            let mut master = unsafe { std::fs::File::from_raw_fd(master) };
            let slave = unsafe { std::fs::File::from_raw_fd(slave) };
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "cli::events::tests::app_approval_requires_complete_preview_and_pauses_without_replying", "--nocapture"])
                .env(CASE, "1").env("TERM", "xterm-256color").env("NO_COLOR", "1")
                .stdin(slave.try_clone().unwrap()).stdout(slave).stderr(Stdio::piped())
                .spawn().unwrap();
            let start = Instant::now();
            let mut display = Vec::new();
            let mut buffer = [0; 4096];
            while child.try_wait().unwrap().is_none() {
                while let Ok(n) = master.read(&mut buffer) {
                    if n == 0 {
                        break;
                    }
                    display.extend_from_slice(&buffer[..n]);
                    assert!(display.len() <= 64 * 1024);
                }
                if start.elapsed() > Duration::from_secs(15) {
                    let _ = child.kill();
                    let output = child.wait_with_output().unwrap();
                    panic!(
                        "App approval test timed out: {} {}",
                        String::from_utf8_lossy(&display),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        resize(24, 80);
        let super::super::args::Parsed::Run(config) =
            super::super::args::Config::parse(["demo"].into_iter().map(str::to_owned)).unwrap()
        else {
            panic!("expected demo configuration")
        };
        let mut app = App::new(config).unwrap();
        assert!(app.terminal.raw() && app.terminal.interactive() && !app.quiet);
        let (tx, rx) = interaction::lanes();
        let (controls, _) = mpsc::sync_channel(1);
        app.pending = Some(Pending {
            budget: None,
            native_progress: None,
            user: String::new(),
            started: Instant::now(),
            cancel: Arc::new(AtomicBool::new(false)),
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
            worker: None,
            join_on_drop: false,
        });
        tx.render("first🦀");
        tx.fact(WorkerEvent::ToolCall(String::new(), "first🦀".into(), 9)).unwrap();
        app.poll_worker().unwrap(); assert!(app.terminal.preview_is_empty());
        tx.fact(WorkerEvent::ToolCall("Read one".into(), String::new(), 0)).unwrap();
        tx.fact(WorkerEvent::ToolCall("Read two".into(), String::new(), 0)).unwrap();
        tx.render("second"); app.poll_worker().unwrap(); assert_eq!(app.pending.as_ref().unwrap().displayed, 6);
        tx.fact(WorkerEvent::ToolCall(String::new(), "second".into(), 15)).unwrap(); app.poll_worker().unwrap(); assert!(app.terminal.preview_is_empty());
        for interactive in [false, true] {
            let budget = vade_harness::agent::RunBudget::new(Some(Duration::from_millis(200)));
            let cancel = Arc::clone(&app.pending.as_ref().unwrap().cancel);
            app.quiet = !interactive;
            std::thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    budget.with_cancellation(&cancel, || {
                        super::super::approval::request_tool_preview_with_budget(
                            &tx,
                            vade_harness::coding_tools::Preview {
                                title: "Patch preview",
                                detail: "-old\n+new",
                                approval_required: true,
                            },
                            &cancel,
                            Some(&budget),
                        )
                    })
                });
                let start = Instant::now();
                while app.pending.as_ref().unwrap().rx.state() != RunState::WaitingForApproval {
                    assert!(start.elapsed() < Duration::from_secs(2));
                    std::thread::park_timeout(Duration::from_millis(1));
                }
                if interactive {
                    app.poll_worker().unwrap();
                } // Raw presentation still awaits its frame.
                std::thread::sleep(Duration::from_millis(350));
                assert!(budget.expired() && cancel.load(Ordering::Acquire));
                app.poll_worker().unwrap();
                assert!(!worker.join().unwrap());
            });
            cancel.store(false, Ordering::Release);
        }
        app.quiet = false;
        fn preview(app: &mut App, tx: &WorkSender, detail: &str) -> mpsc::Receiver<bool> {
            let (reply, rx) = mpsc::channel();
            assert!(
                tx.fact(WorkerEvent::ToolPreview {
                    title: "Patch preview".into(),
                    detail: detail.into(),
                    reply: Some(super::super::approval::ApprovalReply::new(reply, None)),
                })
                .is_ok()
            );
            app.poll_worker().unwrap();
            rx
        }
        let diff = (0..2500)
            .map(|i| format!("-old-{i}\n+new-{i} 🦀\n"))
            .collect::<String>();
        assert_eq!(diff.lines().count(), 5000);
        let reply = preview(&mut app, &tx, &diff);
        assert!(app.approval_reply.is_some());
        app.terminal.approval_ready().unwrap();
        assert_eq!(reply.try_recv(), Err(mpsc::TryRecvError::Empty));
        app.line_input("/approve".into()).unwrap();
        assert_eq!(reply.try_recv(), Ok(true));
        assert!(app.approval_reply.is_none());

        // Both an initially short pane and an initially narrow pane retain the
        // request, decline no operation, and recover after a resize.
        for (rows, columns) in [(2, 80), (24, 2)] {
            resize(rows, columns);
            let reply = preview(&mut app, &tx, &diff);
            assert!(app.approval_reply.is_some());
            assert!(app.terminal.approval_ready().is_err());
            app.line_input("/approve".into()).unwrap();
            assert_eq!(reply.try_recv(), Err(mpsc::TryRecvError::Empty));
            assert!(app.approval_reply.is_some());
            resize(24, 80);
            app.terminal.approval_ready().unwrap();
            app.line_input("/approve".into()).unwrap();
            assert_eq!(reply.try_recv(), Ok(true));
        }
        // Genuine preview errors send false and never install an answer lane.
        // The asking tool learns that no one saw its request (10-30 review).
        let approvals = Arc::new(vade_harness::shell::permissions::Approvals::new(None, None, true).unwrap());
        app.approvals = Some(Arc::clone(&approvals));
        for detail in [
            "+hidden\u{202e}".into(),
            "+x\n".repeat(256 * 1024),
            format!("+{}", "x".repeat(192 * 1024)),
        ] {
            resize(24, 2);
            let reply = preview(&mut app, &tx, &detail);
            assert_eq!(reply.try_recv(), Ok(false));
            assert!(app.approval_reply.is_none());
            assert!(app.terminal.approval_ready().is_err());
            assert!(approvals.take_not_shown());
            assert!(!approvals.take_not_shown());
        }
    }
}

#[cfg(test)]
mod summary_tests;
