//! Terminal input loop, prompt recall and run exit status.
use std::io;
use std::process::ExitCode;
use std::sync::{atomic::Ordering, mpsc};
use std::time::{Duration, Instant};
use vade_harness::{
    child_agents,
    terminal::{self, Input, Key},
};

use super::args::Mode;
use super::help::SlashMenu;
use super::{App, RunState};
use crate::native_children;

/// A y/n typed within this time of the approval appearing is draft text.
pub(crate) const APPROVAL_SETTLE: Duration = Duration::from_millis(400);
/// Shown when a piped /exit lets the current message finish first.
pub(crate) const EXIT_AFTER_MESSAGE: &str = "/exit: the current message finishes first, then chat ends.";

/// Separate a failed configured check from generic runtime/argument failures.
pub(crate) const CHECK_FAILING_EXIT: u8 = 3;
pub(crate) fn run_exit(success: bool, check_failing: bool) -> ExitCode {
    if success && check_failing {
        ExitCode::from(CHECK_FAILING_EXIT)
    } else if success {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

impl App {
    pub(crate) fn run(&mut self) -> io::Result<ExitCode> {
        // run_loop consumes --prompt; remember its presence before starting.
        let one_shot = self.config.prompt.is_some();
        if !one_shot && self.terminal.raw() {
            if let Err(error) = self.input.load_history(&self.workspace) {
                self.notice(&format!("Input history unavailable: {error}"))?;
            }
        }
        let result = self.run_loop();
        self.input.finish_history();
        if let Some(error) = self.input.history_error() {
            self.notice(&error)?;
        }
        self.close_approval();
        if let Some(pending) = &mut self.pending {
            pending.cancel.store(true, Ordering::Release);
            pending.canceled = true;
        }
        let mut summary = if let Some(store) = &self.journal {
            store.lock().map_or_else(
                |_| "Session name unavailable (journal lock failed).".to_owned(),
                |store| {
                    format!(
                        "Session: {} · resume with --resume {}",
                        store.name(),
                        store.name()
                    )
                },
            )
        } else {
            format!(
                "Session: {}",
                self.config.session.as_deref().unwrap_or("temporary")
            )
        };
        let reply = if let Some(native) = &self.native {
            native
                .messages()
                .iter()
                .rev()
                .find(|m| m.role() == "assistant" && m.tool_calls().is_empty())
                .and_then(|m| m.value().get("content"))
                .map(super::native_setup::display_content)
                .unwrap_or_default()
        } else {
            self.session
                .history()
                .iter()
                .rev()
                .find(|m| m.role == "assistant")
                .map(|m| m.content.clone())
                .unwrap_or_default()
        };
        let reply = terminal::reply_tail(&reply);
        if !reply.is_empty() {
            summary.push_str(&format!("\nLast reply (tail):\n{reply}"));
        }
        let restored = self.terminal.finish(&summary);
        result.and_then(|success| {
            restored?;
            let failing = success && one_shot && self.check_failing;
            if failing {
                self.terminal.line(
                    if self.check_not_run {
                        "vade: check not run (configured build/test check was skipped, denied, or unsupported)"
                    } else {
                        "vade: check failing (configured build/test check did not pass)"
                    },
                )?;
            }
            Ok(run_exit(success, failing))
        })
    }
    fn run_loop(&mut self) -> io::Result<bool> {
        if !self.quiet {
            // One quiet line, no banner: what this chat is and how to get help.
            self.terminal.note(&format!("vade {} · / for commands, /help for a guide", match self.config.mode {
                Mode::Demo => "demo (scripted, offline)",
                Mode::Codex => "· Codex owns delegated context and persistence",
                Mode::Hermes => "· Hermes owns delegated context and persistence",
                Mode::Agent => "agent",
                _ => "text chat",
            }))?;
            if !self.terminal.raw() && matches!(self.config.mode, Mode::Agent | Mode::Direct) {
                let idle = self.idle_status();
                self.terminal.note(&idle)?;
            }
            if self.session.persistent() {
                self.terminal
                    .note("Completed-turn counter enabled; no chat transcript saved.")?;
            }
        }
        if self.config.mode == Mode::Agent {
            if let Some((command, source)) = &self.config.check {
                self.run_notice(&format!(
                    "Check: {command} runs after every round that changes files ({source}); --no-check disables"
                ))?;
            }
        }
        if self.config.mode == Mode::Agent && self.config.verbose {
            self.run_notice(&format!(
                "Approval: {} · /status for session details",
                self.config.approval.label()
            ))?;
            let generation = self.config.generation;
            self.run_notice(&match (generation.thinking, generation.effort) {
                (Some(false), _) => "Thinking: off (--thinking off)".to_string(),
                (_, Some(effort)) => format!("Thinking: on, effort {effort} (--effort or VADE_HARNESS_EFFORT)"),
                (Some(true), None) => "Thinking: on, Engine default effort (xhigh for Qwen3.8); --effort low|medium lowers it".to_string(),
                (None, None) if std::env::var_os("VADE_HARNESS_THINKING").is_some_and(|v| v == "1") => {
                    "Thinking: on (VADE_HARNESS_THINKING=1), Engine default effort (xhigh for Qwen3.8); --effort low|medium lowers it".to_string()
                }
                (None, None) => "Thinking: off by default when the Engine advertises thinking; --thinking on or --effort LEVEL enables it".to_string(),
            })?;
            if let Some(store) = &self.journal {
                let name = store
                    .lock()
                    .map_err(|_| io::Error::other("journal lock failed"))?
                    .name()
                    .to_owned();
                self.run_notice(&format!("Session: {name} · resume with --resume {name}"))?;
            }
            self.run_notice(&format!(
                "Shell build: {}",
                self.build_session.description()
            ))?;
            if self.config.children {
                self.run_notice("Children enabled · read-only by default · same endpoint/model · exact admission required.")?;
                self.run_notice(&native_children::describe_grants(&self.config.child_grants))?;
            }
            self.notice_cutoffs()?;
        }
        if !self.quiet {
            self.terminal.line("")?;
        }
        let one_shot = self.config.prompt.is_some();
        let input_rx = if one_shot {
            None
        } else {
            Some(terminal::input_channel(self.terminal.raw()))
        };
        if let Some(prompt) = self.config.prompt.take() {
            self.start_input(prompt)?;
        }
        let mut decoder = terminal::Decoder::default();
        let mut menu = SlashMenu::default();
        let mut last_frame = Instant::now();
        let mut width = self.terminal.width();
        let mut height = self.terminal.height();
        let mut input_done = false;
        let mut run_started = None;
        let mut notified_approval = None;
        while !self.quit || self.pending.is_some() {
            if let Some(error) = self.input.history_error() {
                self.notice(&error)?;
            }
            if !self.quit && terminal::interrupted() {
                self.cancel()?;
                self.failed = true;
                self.quit = true;
            }
            // Input/control always runs before bounded rendering. Neither lane
            // waits for the model worker or its network/child I/O.
            if let Some(rx) = &input_rx {
                for index in 0..16 {
                    if input_done || self.quit {
                        break;
                    }
                    match rx.try_recv() {
                        Ok(Input::Bytes(bytes)) => {
                            for key in decoder.feed(&bytes) {
                                self.menu_key(key, &mut menu)?;
                            }
                        }
                        Ok(Input::Line(line)) if self.piped_exit_ends_input(&line) => {
                            input_done = true;
                            if self.pending.is_some() {
                                self.run_notice(EXIT_AFTER_MESSAGE)?;
                            }
                            self.end_input()?;
                        }
                        Ok(Input::Line(line)) => self.sent_line(line)?,
                        Ok(Input::TooLong) => {
                            self.failed = true;
                            self.notice("Input is over 64 KiB, so it was discarded and not sent.")?;
                        }
                        Ok(Input::InvalidUtf8) => {
                            self.failed = true;
                            self.notice("Input is not valid UTF-8; discarded.")?;
                        }
                        Ok(Input::Eof) | Err(mpsc::TryRecvError::Disconnected) => {
                            input_done = true;
                            self.end_input()?;
                        }
                        Ok(Input::Error) => {
                            return Err(io::Error::other("cannot read terminal input"));
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                    }
                    if index == 15 {
                        // Several inputs can share one wake token. Yield to
                        // worker/render lanes, then drain the next input batch.
                        std::thread::current().unpark();
                    }
                }
            }
            if self.terminal.raw() {
                let next_width = self.terminal.width();
                let next_height = self.terminal.height();
                if next_width != width || next_height != height {
                    self.terminal.resized()?;
                    width = next_width;
                    height = next_height;
                    self.dirty = true;
                }
                if let Some(key) = decoder.timeout() {
                    self.menu_key(key, &mut menu)?;
                }
            }
            if let Some(pending) = &self.pending {
                run_started.get_or_insert(pending.started);
            }
            self.poll_worker()?;
            if let Some(started) = run_started {
                if self.pending.is_none() {
                    let message = if self.last_success {
                        "Vade · done"
                    } else {
                        "Vade · run ended"
                    };
                    self.terminal.notify_run(started.elapsed(), message)?;
                    run_started = None;
                } else if self.approval_shown.is_some() && self.approval_shown != notified_approval {
                    self.terminal
                        .notify_run(started.elapsed(), "Vade · waiting for approval")?;
                    notified_approval = self.approval_shown;
                }
            }
            if self.children.changed() {
                self.dirty = true;
            }
            if self.pending.is_none() && !self.quit {
                if let Some(text) = self.steering.take() {
                    self.start(text)?;
                }
            }
            if self.pending.is_none() && !self.quit && self.last_success && !self.queue_paused {
                if let Some(text) = self.queued.pop() {
                    self.start(text)?;
                }
            }
            if (one_shot || input_done) && self.pending.is_none() {
                if self.queued.len() > 0 {
                    self.failed = true;
                    self.run_notice(&format!(
                        "{} queued message(s) remain unsent at EOF because the run did not finish.",
                        self.queued.len()
                    ))?;
                }
                break;
            }
            menu.refresh(self.input.text(), self.input.cursor());
            let popup = if let Some(search) = self.input.search_prompt() {
                search
            } else if self.attachment_completion.0 == self.input.text() && !self.attachment_completion.1.is_empty() {
                self.attachment_completion.1.clone()
            } else if self.paste || self.multiline.is_some() || self.recall.is_some() {
                String::new()
            } else {
                menu.panel(
                    self.config.mode,
                    self.approval_reply.is_some(),
                    height.saturating_sub(6).min(10),
                    width,
                    self.terminal.color(),
                )
            };
            if let Some(pending) = &self.pending {
                // Input repaints at once; streamed text at most ~30 frames/s;
                // the spinner at 10 frames/s.
                let since = last_frame.elapsed();
                if self.dirty
                    || (self.stream_dirty && since >= Duration::from_millis(33))
                    || since >= Duration::from_millis(100)
                {
                    let state = if pending.canceled {
                        RunState::Cancelling
                    } else {
                        pending.rx.state()
                    };
                    let child_panel = if self.config.children {
                        child_agents::ui::render(
                            &self.children.snapshots(),
                            width.saturating_sub(1),
                            self.terminal.height().saturating_sub(5).min(7),
                        )
                    } else {
                        String::new()
                    };
                    if self.terminal.raw() {
                        // Prefill (nothing back yet) reads as thinking; streamed
                        // reasoning as reasoning; visible text as writing.
                        let label = match state {
                            RunState::Preparing | RunState::Connecting => "connecting",
                            RunState::Queued => "thinking",
                            RunState::Thinking => "reasoning",
                            RunState::Generating => "writing",
                            RunState::ToolRunning => "running tool",
                            RunState::Cancelling => "canceling",
                            _ => state.label(),
                        };
                        let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"]
                            [(pending.started.elapsed().as_millis() / 100 % 8) as usize];
                        let elapsed = terminal::clock::duration(pending.started.elapsed());
                        let queued = self.queued.len();
                        let mut tail = Vec::new();
                        let budgets = pending.budget_status();
                        if !budgets.is_empty() {
                            tail.push(budgets.clone());
                        }
                        if queued > 0 {
                            tail.push(format!("{queued} queued"));
                        }
                        let (status, transient) = if self.approval_reply.is_some() {
                            tail.insert(0, "Approval required · y / n".to_owned());
                            (tail.join(" · "), String::new())
                        } else if if self.config.mode == Mode::Agent {
                            self.terminal.preview_is_empty()
                        } else {
                            !pending.received
                        } {
                            // Spinner and clock sit where the answer will appear;
                            // the footer keeps the session facts.
                            let idle = if budgets.is_empty() {
                                self.idle_status()
                            } else {
                                String::new()
                            };
                            let mut footer = vec![idle];
                            footer.extend(tail);
                            footer.push("Esc cancels".into());
                            (
                                footer
                                    .into_iter()
                                    .filter(|p| !p.is_empty())
                                    .collect::<Vec<_>>()
                                    .join(" · "),
                                format!("{spinner} {label}… · {elapsed}"),
                            )
                        } else {
                            let mut footer = vec![format!("{spinner} {label} · {elapsed}")];
                            if let Some(rate) = pending.rx.estimated_rate().filter(|r| *r >= 1.0) {
                                footer.push(format!("≈{rate:.0} tok/s"));
                            }
                            footer.extend(setup_badge());
                            footer.extend(tail);
                            footer.push("Esc cancels".into());
                            (footer.join(" · "), String::new())
                        };
                        self.terminal.active_editor_with_popup(
                            self.input.text(),
                            self.input.cursor(),
                            &status,
                            &transient,
                            &child_panel,
                            &popup,
                        )?;
                        self.approval_presented();
                    } else {
                        self.terminal.active_editor_with_panel(
                            self.input.text(),
                            self.input.cursor(),
                            state.label(),
                            pending.started,
                            self.queued.len(),
                            &child_panel,
                        )?;
                    }
                    self.dirty = false;
                    self.stream_dirty = false;
                    last_frame = Instant::now();
                }
            } else if self.dirty || self.stream_dirty {
                let idle = self.idle_status();
                self.terminal.editor_with_popup(
                    self.input.text(),
                    self.input.cursor(),
                    self.multiline.is_some(),
                    &idle,
                    &popup,
                )?;
                self.dirty = false;
                self.stream_dirty = false;
            }
            // Both input and model queues unpark this thread. The timeout only
            // serves spinner, resize and signal checks; it never gates tokens.
            std::thread::park_timeout(Duration::from_millis(5));
        }
        self.close_approval();
        if let Some(pending) = self.pending.as_mut() {
            pending.cancel.store(true, Ordering::Relaxed);
            pending.canceled = true;
            self.failed = true;
        }
        if matches!(self.config.mode, Mode::Agent | Mode::Codex | Mode::Hermes) {
            while self.pending.is_some() {
                self.poll_worker()?;
                if self.pending.is_some() {
                    std::thread::park_timeout(Duration::from_millis(20));
                }
            }
        }
        self.terminal.flush_stream()?;
        if let Some(started) = run_started {
            self.terminal.notify_run(started.elapsed(), "Vade · run ended")?;
        }
        Ok(!self.failed)
    }
    /// Without a terminal, an /exit before the provider starts the user's
    /// message ends the input as EOF does: the message still runs, and chat
    /// exits when it is done. Once the provider starts the message, and
    /// during other work such as /compact, /exit cancels, as before. Inside
    /// /paste, /exit is text.
    pub(crate) fn piped_exit_ends_input(&self, line: &str) -> bool {
        self.quiet
            && self.multiline.is_none()
            && matches!(line.trim(), "/exit" | "/quit")
            && self
                .pending
                .as_ref()
                .is_none_or(|pending| !pending.user.is_empty() && !pending.rx.underway())
    }
    /// The input ended (EOF, or /exit without a terminal): no one can answer
    /// an approval now, and unfinished multiline input is dropped.
    fn end_input(&mut self) -> io::Result<()> {
        self.approval_input_closed = true;
        if let Some(approvals) = &self.approvals {
            approvals.set_interactive(false);
        }
        if self.approval_reply.is_some() {
            self.cancel()?;
        }
        if self.multiline.is_some() {
            self.notice("Unfinished multiline input discarded at EOF.")?;
            self.multiline = None;
            self.failed = true;
        }
        Ok(())
    }

    /// Footer facts at rest: model · folder · approval · ctx use · session.
    /// Only reported numbers appear; an unknown context use is omitted.
    pub(crate) fn idle_status(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        match self.config.mode {
            Mode::Demo => parts.push("demo".into()),
            Mode::Codex | Mode::Hermes => parts.push(self.config.mode.provider().to_lowercase()),
            _ => parts.extend(self.config.model.clone()),
        }
        // The numbers first: a narrow strip drops facts from the right.
        if matches!(self.config.mode, Mode::Agent | Mode::Direct) {
            // Zero context only when no exchange exists anywhere; a running
            // worker holds the conversation, and retained history counts.
            let empty = self.pending.is_none() && self.session.history().is_empty()
                && self.native.as_ref().is_none_or(|c| c.messages().iter().all(|m| m.role() == "system"));
            parts.extend(self.context.strip(self.config.context_tokens as u64, empty));
        }
        if let Some(folder) = self
            .workspace
            .root()
            .file_name()
            .and_then(|name| name.to_str())
        {
            parts.push(folder.to_owned());
        }
        if self.config.mode == Mode::Agent {
            // "ask (ask for edits and commands; …)" → "approval: ask".
            let policy = self.config.approval.label();
            let short = policy.split([' ', '(']).next().unwrap_or(policy);
            parts.push(format!("approval: {short}"));
        }
        if let Some(store) = &self.journal {
            // Never block the UI thread behind a worker checkpoint; while the
            // worker holds the lock, show the last name seen.
            if let Ok(store) = store.try_lock() {
                *self.session_name.borrow_mut() = store.name().to_owned();
            }
            let name = self.session_name.borrow();
            if !name.is_empty() {
                parts.push(name.clone());
            }
        }
        // Last, so a narrow strip keeps it (the strip drops middle facts).
        parts.extend(setup_badge());
        parts.join(" · ")
    }
    pub(crate) fn notice(&mut self, text: &str) -> io::Result<()> {
        self.terminal.flush_stream()?;
        self.terminal.note(text)?;
        self.dirty = true;
        Ok(())
    }
    pub(crate) fn run_notice(&mut self, text: &str) -> io::Result<()> {
        if self.quiet {
            vade_harness::diagnostics::stderr_line(format_args!("vade: {}", terminal::sanitize(text)));
            Ok(())
        } else {
            self.notice(text)
        }
    }
    /// A warning in three plain parts: what happened, what Vade is doing
    /// about it, what you can do. Redirected output gets one stderr line.
    pub(crate) fn warn(&mut self, what: &str, doing: &str, next: &str) -> io::Result<()> {
        if self.quiet {
            let parts = [what, doing, next].into_iter().filter(|p| !p.is_empty());
            vade_harness::diagnostics::stderr_line(format_args!("vade: warning: {}", terminal::sanitize(&parts.collect::<Vec<_>>().join(" "))));
            Ok(())
        } else {
            self.terminal.warning(what, doing, next)?;
            self.dirty = true;
            Ok(())
        }
    }
    pub(crate) fn notice_cutoffs(&mut self) -> io::Result<()> {
        let notice = self
            .journal
            .as_ref()
            .and_then(|store| store.lock().ok()?.cutoff_notice());
        if let Some(notice) = notice {
            self.run_notice(&notice)?;
        }
        Ok(())
    }
    fn menu_key(&mut self, key: Key, menu: &mut SlashMenu) -> io::Result<()> {
        if self.terminal.pager_active() || matches!(key, Key::CtrlO) {
            return self.key(key);
        }
        if self.recall.is_some() || self.input.search_prompt().is_some() || matches!(key, Key::Search) {
            return self.key(key);
        }
        // Exact approval commands are control actions, even if the slash menu
        // was previously dismissed or highlighted a different command.
        if self.approval_reply.is_some()
            && !self.paste
            && matches!(key, Key::Enter)
            && self.input.approval_command().is_some()
        {
            return self.key(key);
        }
        menu.refresh(self.input.text(), self.input.cursor());
        let approval = self.approval_reply.is_some();
        if !self.paste && self.multiline.is_none() && menu.open(self.config.mode, approval) {
            match key {
                Key::Up | Key::Down => {
                    menu.step(matches!(key, Key::Down), self.config.mode, approval)
                }
                Key::Tab => {
                    if let Some(text) = menu.complete(self.config.mode, approval) {
                        self.input.set_typed(text);
                    }
                }
                Key::Enter if menu.exact(self.config.mode, approval) => {
                    // Fully typed command: send exactly what is visible.
                    return self.key(key);
                }
                Key::Enter => {
                    // Run the highlighted command only when the user chose it
                    // (arrows) or the prefix is unambiguous; otherwise, like
                    // Tab, complete it into the editor for a second Enter. A
                    // command that takes arguments is always completed.
                    let decisive = menu.navigated() || menu.unique(self.config.mode, approval);
                    if let Some(text) = menu.complete(self.config.mode, approval) {
                        self.input.set_typed(text.clone());
                        if decisive && !text.ends_with(' ') {
                            self.dirty = true;
                            return self.key(Key::Enter);
                        }
                    }
                }
                Key::Escape => menu.close(),
                _ => return self.key(key),
            }
            self.dirty = true;
            return Ok(());
        }
        self.key(key)
    }
    pub(crate) fn key(&mut self, key: Key) -> io::Result<()> {
        if !self.paste && self.terminal.pager_key(&key)? {
            self.dirty = true;
            return Ok(());
        }
        if !self.paste && self.approval_reply.is_none() {
            if matches!(key, Key::Search) && self.input.search_prompt().is_none() {
                self.draft = self.input.text().to_owned();
            }
            if self.input.search_key(&key) {
                self.recall = (0..self.input.history_len()).rev()
                    .find(|i| self.input.history(*i) == Some(self.input.text()));
                self.dirty = true;
                return Ok(());
            }
        }
        if self.pending_approval_key(&key)? {
            return Ok(());
        }
        // With an empty single-line editor, a single y/n answers the pending
        // approval once the prompt has been on screen long enough to read.
        // Any other text, multiline input, paste, or an early key is a draft
        // and keeps the approval waiting.
        if self.approval_reply.is_some()
            && !self.paste
            && self.multiline.is_none()
            && self.input.is_empty()
            && self
                .approval_shown
                .is_some_and(|shown| shown.elapsed() >= APPROVAL_SETTLE)
        {
            if let Key::Text(text) = &key {
                let answer = match text.as_str() {
                    "y" | "Y" => Some("/approve"),
                    "n" | "N" => Some("/deny"),
                    "s" | "S" if self.approval_command => Some("/approve session"),
                    "p" | "P" if self.approval_command => Some("/approve project"),
                    _ => None,
                };
                if let Some(answer) = answer {
                    self.dirty = true;
                    return self.line_input(answer.into());
                }
            }
        }
        if self.pending.as_ref().is_some_and(|p| !p.canceled)
            && matches!(key, Key::Interrupt | Key::Escape)
        {
            return self.cancel();
        }
        match key {
            Key::PasteStart => self.paste = true,
            Key::PasteEnd => self.paste = false,
            Key::Text(text) => {
                let inserted = if self.paste {
                    self.input.insert_paste(&text)
                } else {
                    self.input.insert(&text)
                };
                if !inserted {
                    self.input_overflow = true;
                }
            }
            Key::Newline => {
                let inserted = if self.paste {
                    self.input.insert_paste("\n")
                } else {
                    self.input.insert("\n")
                };
                if !inserted {
                    self.input_overflow = true;
                }
            }
            Key::Enter if !self.paste && self.input.continue_line() => (),
            Key::Enter | Key::End if !self.paste => {
                if self.input_overflow {
                    self.input.clear();
                    self.input_overflow = false;
                    self.failed = true;
                    return self
                        .notice("Input is over 64 KiB, so it was discarded and not sent.");
                }
                // An empty-editor Ctrl+D is an exit even while an approval
                // waits, exactly like /exit there: quit and cancel, which
                // declines the approval. Nonempty input stays approval text.
                if matches!(key, Key::End) && self.input.is_empty() && self.multiline.is_none() {
                    self.quit = true;
                    self.steering = None;
                    return self.cancel();
                }
                if self.approval_reply.is_some() {
                    if let Some((command, _)) = self.input.approval_command() {
                        let command = command.to_owned();
                        self.line_input(command)?;
                        if self.approval_reply.is_none() {
                            self.input.remove_approval_command();
                        }
                    } else if self.input.text().contains('\n')
                        && matches!(self.input.text().trim(), "/approve" | "/deny" | "/approve session" | "/approve project")
                    {
                        self.notice("Multiline approval text retained as draft. Type /approve or /deny on a new line, or clear the draft and press y or n.")?;
                    } else {
                        self.line_input(self.input.text().to_owned())?;
                        if self.approval_reply.is_none() {
                            self.input.clear();
                        }
                    }
                    self.dirty = true;
                    return Ok(());
                }
                self.terminal.clear_editor()?;
                let input = self.input.take();
                self.recall = None;
                self.draft.clear();
                self.line_input(input)?;
            }
            Key::Interrupt => {
                if self.input.is_empty() && self.multiline.is_none() {
                    self.quit = true;
                } else {
                    self.input.clear();
                    self.input_overflow = false;
                    self.multiline = None;
                    self.notice("Input cleared.")?;
                }
            }
            Key::Clear => {
                self.input.clear();
                self.input_overflow = false;
                self.recall = None;
            }
            // Esc closes the popup or cancels a run (handled above); it never
            // discards a draft. Ctrl+U clears.
            Key::Escape => (),
            Key::Redraw => self.terminal.resized()?,
            Key::CtrlT => self.terminal.toggle_thinking(),
            Key::WordLeft => self.input.word_left(),
            Key::WordRight => self.input.word_right(),
            Key::DeleteWordForward => self.input.delete_word_forward(),
            Key::KillToEnd => self.input.kill_to_end(),
            Key::Tab => {
                let (suffix, candidates) = super::attachments::complete(&self.workspace, self.input.text(), self.input.cursor()).unwrap_or_else(|| ("    ".into(), String::new()));
                self.attachment_completion.1 = if self.attachment_completion.0 == self.input.text() { candidates } else { String::new() };
                if !self.input.insert(&suffix) {
                    self.input_overflow = true;
                }
                self.attachment_completion.0 = self.input.text().to_owned();
            }
            Key::PageUp | Key::PageDown => self.terminal.scroll(matches!(key, Key::PageDown)),
            Key::WheelUp | Key::WheelDown => self.terminal.wheel(matches!(key, Key::WheelDown)),
            Key::Backspace => self.input.backspace(),
            Key::DeleteWord => self.input.delete_word(),
            Key::Left => self.input.left(),
            Key::Right => self.input.right(),
            Key::Home => self.input.home(),
            Key::LineEnd => self.input.end(),
            Key::Delete => self.input.delete(),
            Key::LineUp | Key::LineDown => self.input.vertical(matches!(key, Key::LineDown)),
            Key::Up | Key::Down if self.input.move_row(matches!(key, Key::Down)) => (),
            Key::Up => {
                let turns = self.input_history_len();
                if turns > 0 {
                    let index = self
                        .recall
                        .map_or(turns - 1, |i| i.saturating_sub(1).min(turns - 1));
                    if self.recall.is_none() {
                        self.draft = self.input.text().to_owned();
                    }
                    let text = self.input_history(index).unwrap_or_default().to_owned();
                    self.input.set(text);
                    self.recall = Some(index);
                }
            }
            Key::Down => {
                if let Some(index) = self.recall {
                    if index.saturating_add(1) < self.input_history_len() {
                        self.recall = Some(index + 1);
                        let text = self.input_history(index + 1).unwrap_or_default().to_owned();
                        self.input.set(text);
                    } else {
                        self.recall = None;
                        self.input.set(std::mem::take(&mut self.draft));
                    }
                }
            }
            _ => (),
        }
        self.dirty = true;
        Ok(())
    }
    fn input_history_len(&self) -> usize {
        if self.input.history_len() > 0 {
            self.input.history_len()
        } else {
            self.session.turns()
        }
    }
    fn input_history(&self, index: usize) -> Option<&str> {
        if self.input.history_len() > 0 {
            self.input.history(index)
        } else {
            self.session
                .history()
                .get(index.checked_mul(2)?)
                .map(|message| super::attachments::display(&message.content))
        }
    }
    pub(crate) fn queue(&mut self, text: String) -> io::Result<()> {
        if text.trim().is_empty() {
            return self.notice("Use /queue TEXT for a separate follow-up prompt.");
        }
        match self.queued.push(text) {
            Ok(()) => self.run_notice(&format!(
                "Follow-up queued ({} waiting); runs after this turn succeeds, or /next.",
                self.queued.len()
            )),
            Err(text) => {
                let kept = self.keep_draft(&text);
                self.notice(&format!("Follow-up queue full (8 prompts / 64 KiB); nothing sent.{kept}"))
            }
        }
    }
    pub(crate) fn reset_recall(&mut self) {
        // Completion can evict several old turns while the user is browsing
        // recall. The editor keeps its text, but its old history index expires.
        self.recall = None;
        self.draft.clear();
    }
    pub(crate) fn complete_session(&mut self, user: String, assistant: String) -> io::Result<bool> {
        self.reset_recall();
        self.session.complete(user, assistant)
    }
}

#[cfg(test)]
mod tests;

/// What the launcher knows about the engine and the CLI cannot ask the
/// server: `VADE_CHAT_SETUP`, such as `1 Spark, drafting on`. Sanitized, at
/// most 80 bytes, shown last on the status strip.
pub(crate) fn setup_badge() -> Option<String> {
    let value = std::env::var("VADE_CHAT_SETUP").ok()?;
    let value = terminal::sanitize(value.trim());
    let mut end = value.len().min(80);
    while !value.is_char_boundary(end) { end -= 1; }
    (!value.is_empty()).then(|| value[..end].to_owned())
}
