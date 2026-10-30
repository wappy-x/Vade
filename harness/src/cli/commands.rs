//! Slash-command dispatch and session actions.
use std::io;
use std::sync::{Arc, Mutex};
use vade_harness::{
    agent::Limits, child_agents, context, journal, journal::Journal, project,
    session::MAX_PROMPT_BYTES, shell::permissions::Scope,
};

use super::{App, RunState};
use super::{
    args::Mode,
    help::{self, Command},
    native_setup::{append_resume_note, chat_conversation_with_mcp, prepare_resumed},
};
use crate::native_children;

// Shared metadata for help, completion and diagnostics. Keep new commands here.
#[rustfmt::skip]
pub(crate) const COMMANDS: &[Command] = &[
    Command::new("help", "[all|COMMAND]", "Show common commands or command details.", "Use /help all for the complete list. Help is local and does not send a prompt.", "Chat", "all", true),
    Command::new("status", "", "Inspect this session and request.", "Shows request state, queue and history. Live ≈tok/s estimates visible text; /context reports provider token telemetry.", "Chat", "all", true),
    Command::new("context", "", "Inspect context budget and cache usage.", "Reports the last admission and provider usage; estimates are identified explicitly.", "Chat", "all", true),
    Command::new("copy", "[code|N]", "Copy the last answer or its last code block.", "Exact history text: /copy N selects the Nth answer back (1 is latest); /copy code selects the last closed fenced block without its fences. Limit 64 KiB. Uses pbcopy locally or OSC 52 over SSH, with tmux passthrough. Wait for native worker cleanup.", "Chat", "all", false),
    Command::new("retry", "", "Retry the last answer without replaying tools.", "Direct and native agent modes only; wait for worker cleanup. Sampling settings and pinned seeds stay unchanged. To select a different seed, restart with a different --seed. Unspecified sampling stays unspecified, so greedy output may repeat. Native retries retain completed tool receipts and cannot execute tools.", "Chat", "answers", false),
    Command::new("continue", "", "Continue an answer cut off at its output limit.", "Direct and native agent modes only; wait for worker cleanup. Answers normally continue automatically; --no-auto-continue waits for this command. Available after an upstream length cutoff. Adds a user continuation turn, retaining prior history and the current seed.", "Chat", "answers", false),
    Command::new("cancel", "", "Cancel the active run.", "Wait for acknowledgment; upstream compute may continue. No tool is replayed automatically.", "Chat", "all", true),
    Command::new("queue", "[on|off|TEXT]", "Toggle queueing or queue one follow-up.", "Direct and native chat steer by default. /queue on queues typed messages; /queue off restores steering. /queue TEXT queues one prompt (up to eight / 64 KiB); runs after success, held after failure or cancellation.", "Chat", "all", true),
    Command::new("next", "", "Start the next held follow-up.", "Waits for active worker cleanup before starting queued work.", "Chat", "all", false),
    Command::new("queue-clear", "", "Discard queued follow-ups.", "Clears only unsent follow-ups.", "Chat", "all", false),
    Command::new("steer", "TEXT", "Send input to the active Codex turn.", "Separate receipts report acceptance, rejection or an unconfirmed outcome.", "Chat", "codex", false),
    Command::new("approve", "[session|project]", "Approve the displayed operation.", "Same as pressing y on an empty line: authorizes only the pending exact edit or command. For a command, /approve session (key s) also allows matching commands for the rest of this session, and /approve project (key p) saves a rule to .vade/permissions.json.", "Workspace", "agent", true),
    Command::new("deny", "", "Deny the pending operation.", "Same as pressing n on an empty line. Declines the displayed edit or command.", "Workspace", "agent", true),
    Command::new("ls", "[PATH]", "List local workspace entries.", "Bounded listing inside the selected workspace; no model request.", "Workspace", "all", true),
    Command::new("read", "PATH", "Preview a local UTF-8 file.", "At most 64 KiB inside the workspace. Preview text is not added to model context.", "Workspace", "all", true),
    Command::new("children", "[on|off|cancel ID|all]", "Inspect or cancel native children.", "Enable before the first prompt. Child scope and approvals remain bounded by host grants.", "Workspace", "agent", false),
    Command::new("output", "ID OFFSET LENGTH", "Inspect captured tool output.", "Local UTF-8 byte range; length 1–4096. Wait for worker cleanup.", "Workspace", "agent", false),
    Command::new("instructions", "", "Show which project instruction files loaded.", "Lists AGENTS.md, CLAUDE.md and .vade/instructions.md files from the git root down to the workspace, and ~/.vade/instructions.md, in prompt order. Vade reads them again before a turn when they change. --no-instructions turns them off.", "Workspace", "answers", false),
    Command::new("project", "", "Inspect frozen project rules and catalog.", "Rules and catalog are frozen per native session; no scripts run.", "Workspace", "agent", false),
    Command::new("project-rules", "", "Read exact frozen rules.", "Shows the native session rules snapshot locally.", "Workspace", "agent", false),
    Command::new("skill", "NAME", "Inspect a catalog skill body.", "Loads from the frozen catalog locally; no scripts run.", "Workspace", "agent", false),
    Command::new("map", "[PATH]", "Inspect bounded repository names.", "On-demand manifest inside the selected workspace.", "Workspace", "agent", false),
    Command::new("symbols", "PATH", "Inspect lexical symbol hints.", "Bounded local inspection; hints are not a compiler index.", "Workspace", "agent", false),
    Command::new("new", "", "Start a fresh conversation.", "Wait for cleanup. Native mode keeps the old journal; delegated mode starts a fresh external session.", "Sessions", "all", true),
    Command::new("clear", "", "Clear the conversation.", "Alias for /new. Keeps old native journals and waits for worker cleanup.", "Sessions", "all", false),
    Command::new("memory", "", "Show memory support.", "Saved memory is not available in this build. Use project instructions for saved rules.", "Context", "all", false),
    Command::new("quit", "", "Exit the harness.", "Active native or delegated work is cancelled and cleaned up before exit. Alias: /exit.", "Sessions", "all", true),
    Command::new("exit", "", "Exit the harness.", "Alias for /quit.", "Sessions", "all", false),
    Command::new("undo", "", "Remove the last chat exchange.", "Direct/demo only; tool and delegated histories cannot be spliced.", "Sessions", "chat", false),
    Command::new("paste", "", "Collect multiline input.", "Line mode: /end sends, /cancel discards. Raw mode: pasted text waits for Enter.", "Sessions", "all", false),
    Command::new("end", "", "Send collected multiline input.", "Available while /paste is collecting lines.", "Sessions", "all", false),
    Command::new("sessions", "", "List private native sessions.", "Newest first, with name, start, last turn, turns and model. vade sessions lists them outside chat.", "Sessions", "agent", false),
    Command::new("resume", "ID|NAME|last", "Resume a private native session.", "Wait for cleanup. A shared name picks the newest. No files restored, tools replayed or prompts sent. Alias: /switch.", "Sessions", "agent", false),
    Command::new("switch", "ID|NAME|last", "Switch to a private native session.", "Alias for /resume.", "Sessions", "agent", false),
    Command::new("fork", "NAME [TASK_LABEL]", "Fork the journal and select it.", "Copies transcript/artifacts and shares the workspace. No inference or file restore.", "Sessions", "agent", false),
    Command::new("branch", "", "Inspect ancestry and checkpoint changes.", "Local inspection after worker cleanup.", "Sessions", "agent", false),
    Command::new("compact", "[KEEP_TURNS SUMMARY]", "Compact older context.", "Without arguments the model summarizes older turns into one state block and the last two turns stay verbatim, as auto-compaction does; bounded local excerpts are the fallback. KEEP_TURNS SUMMARY supplies your own summary instead. Lossy in model context; the journal keeps originals.", "Context", "agent", false),
    Command::new("compact-preview", "[KEEP_TURNS SUMMARY]", "Preview context compaction.", "Previews the local excerpt plan, or your own summary, without applying it.", "Context", "agent", false),
    Command::new("compaction", "", "Inspect the latest compaction.", "Shows source records, summary and KV impact.", "Context", "agent", false),
    Command::new("journal", "RECORD OFFSET LENGTH", "Inspect raw journal bytes.", "Local inspection after worker cleanup.", "Context", "agent", false),
];

impl App {
    /// Session allow-list rules belong to the session that granted them, and
    /// a new session reads the allow-list files again.
    fn reset_session_rules(&mut self) -> io::Result<()> {
        if let Some(Err(error)) = self.approvals.as_ref().map(|approvals| approvals.new_session()) {
            self.notice(&format!("Allow list not read, so its commands ask: {error}"))?;
        }
        Ok(())
    }
    pub(crate) fn line_input(&mut self, line: String) -> io::Result<()> {
        if self.multiline.is_none() && self.terminal.raw() && !self.quiet && line.starts_with('/') && !line.starts_with("//") && help::lookup(line.split_whitespace().next().unwrap_or("").trim_start_matches('/')).is_some() { self.terminal.user_message(&line)?; }
        if self.multiline.is_none() && line.starts_with('/') && !line.starts_with("//") {
            self.input.remember_async(&line);
        }
        if self.approval_reply.is_some() {
            match line.trim() {
                text @ ("/approve session" | "/approve project") => {
                    let scope = if text.ends_with("session") { Scope::Session } else { Scope::Project };
                    let Some(approvals) = self.approvals.clone().filter(|_| self.approval_command) else {
                        return self.notice("Only a command can go on an allow list. Press y or n, or send /approve or /deny.");
                    };
                    if let Err(error) = self.terminal.approval_ready() {
                        return self.notice(&error.to_string());
                    }
                    // The reach goes to the tool that asked, not to a slot
                    // that another tool's approval could read first.
                    let submitted = self.close_approval().is_some_and(|reply| {
                        approvals.choose_for(reply.asker(), scope);
                        let sent = reply.send(true).is_ok();
                        if !sent {
                            approvals.withdraw_choice(reply.asker());
                        }
                        sent
                    });
                    return self.notice(match (submitted, scope) {
                        (false, _) => "Approval expired or cancelled; no operation authorized.",
                        (true, Scope::Session) => "Approved. An Allow list notice follows; it says whether a session rule was added.",
                        (true, _) => "Approved. An Allow list notice follows; it says whether a rule was saved in .vade/permissions.json.",
                    });
                }
                "/approve" => {
                    if let Err(error) = self.terminal.approval_ready() {
                        return self.notice(&error.to_string());
                    }
                    // Record the reach with every yes, so no older answer applies.
                    let approvals = self.approvals.clone();
                    let submitted = self.close_approval().is_some_and(|reply| {
                        if let Some(approvals) = &approvals {
                            approvals.choose_for(reply.asker(), Scope::Once);
                        }
                        let sent = reply.send(true).is_ok();
                        if let Some(approvals) = approvals.as_ref().filter(|_| !sent) {
                            approvals.withdraw_choice(reply.asker());
                        }
                        sent
                    });
                    return self.notice(if submitted {
                        "Approved this exact operation once."
                    } else {
                        "Approval expired or cancelled; no operation authorized."
                    });
                }
                "/deny" => {
                    if let Some(reply) = self.close_approval() {
                        let _ = reply.send(false);
                    }
                    return self.notice("Operation denied.");
                }
                "/quit" | "/exit" => {
                    self.quit = true;
                    return self.cancel();
                }
                text if text == "/children" || text.starts_with("/children ") => {
                    return self.children_command(text.strip_prefix("/children").unwrap().trim());
                }
                text if text == "/help" || text.starts_with("/help ") => {
                    return self.command_help(text.strip_prefix("/help").unwrap().trim());
                }
                "/status" => return self.status(),
                text if text == "/copy" || text.starts_with("/copy ") => {
                    return self.copy_answer(text.strip_prefix("/copy").unwrap().trim());
                }
                "/cancel" => {
                    return self.cancel();
                }
                _ => {
                    self.input.set(line);
                    return self.notice(
                        "Approval still pending. Text retained in the editor; no follow-up queued. Clear it and press y or n, or send /approve or /deny; /cancel cancels the run and declines.",
                    );
                }
            }
        }
        if self.multiline.is_some() {
            if line == "/cancel" {
                self.multiline = None;
                return self.notice("Multiline input discarded.");
            }
            if line == "/end" {
                let text = self.multiline.take().unwrap();
                return match self.prepare_input(text.clone()) {
                    Ok(sent) => { let result = self.start(sent); if self.pending.is_none() && !text.trim().is_empty() && !self.terminal.raw() { self.multiline = Some(text); } result },
                    Err(reason) => { if !self.terminal.raw() { self.multiline = Some(text); } self.notice(&reason) }
                };
            }
            let buffer = self.multiline.as_mut().unwrap();
            let extra = usize::from(self.multiline_lines > 0);
            if buffer.len() + line.len() + extra > MAX_PROMPT_BYTES {
                return self.notice(
                    "Multiline input exceeds 64 KiB; line discarded. /end sends or /cancel clears.",
                );
            }
            if self.multiline_lines > 0 {
                buffer.push('\n');
            }
            buffer.push_str(&line);
            self.multiline_lines += 1;
            self.dirty = true;
            return Ok(());
        }
        if line.starts_with("//") {
            return self.start(line[1..].into());
        }
        if super::attachments::resolves_absolute(&self.workspace, &self.config.workspace, &line) && help::lookup(line.trim().trim_start_matches('/')).is_none() {
            return self.start_input(line);
        }
        // Slash commands have one name; pasted paths and API routes stay prose.
        if line.starts_with('/') && line.split_whitespace().next().is_some_and(|token| token[1..].split_once('/').is_some_and(|(_, tail)| !tail.is_empty())) { return self.start_input(line); }
        if line.starts_with('/') {
            let (command, raw_rest) = line.split_once(char::is_whitespace).unwrap_or((&line, ""));
            let rest = raw_rest.trim();
            match command {
                "/copy" => return self.copy_answer(rest),
                "/retry" if rest.is_empty() => return self.retry_answer(),
                "/continue" if rest.is_empty() => return self.continue_answer(),
                "/cancel" if rest.is_empty() => {
                    return self.cancel();
                }
                "/steer" => {
                    return self.steer(raw_rest.to_owned());
                }
                "/queue" if matches!(rest, "" | "on" | "off") => {
                    if !matches!(self.config.mode, Mode::Direct | Mode::Agent) {
                        return self.notice("Use /queue TEXT to queue a follow-up in this mode.");
                    }
                    self.queue_input = match rest { "on" => true, "off" => false, _ => !self.queue_input };
                    return self.notice(if self.queue_input { "Queueing on: typed messages wait for the current reply." }
                        else { "Steering on: typed messages interrupt the current reply." });
                }
                "/queue" => {
                    return match self.prepare_input(raw_rest.to_owned()) {
                        Ok(sent) => self.queue(sent),
                        Err(reason) => self.notice(&reason),
                    };
                }
                "/queue-clear" if rest.is_empty() => {
                    self.queued.clear();
                    self.queue_paused = false;
                    return self.notice("Queued follow-ups cleared.");
                }
                "/next" if rest.is_empty() => {
                    if self.pending.is_some() {
                        return self
                            .notice("Wait for cleanup before starting the next queued prompt.");
                    }
                    if let Some(text) = self.queued.pop() {
                        self.queue_paused = false;
                        return self.start(text);
                    }
                    return self.notice("No follow-ups queued.");
                }
                "/quit" | "/exit" if rest.is_empty() => {
                    self.quit = true;
                    self.steering = None;
                    return self.cancel();
                }
                "/help" => {
                    self.command_help(rest)?;
                }
                "/memory" if rest.is_empty() => self.notice("Saved memory is not available in this build. Use /instructions to inspect saved project rules.")?,
                "/status" if rest.is_empty() => self.status()?,
                "/children" => self.children_command(rest)?,
                "/sessions" if rest.is_empty() && self.config.mode == Mode::Agent => {
                    let list = vade_harness::saved_sessions::list(&self.workspace)?;
                    self.terminal.line(vade_harness::saved_sessions::table(&list).trim_end())?;
                }
                "/resume" | "/switch" if !rest.is_empty() && self.config.mode == Mode::Agent => {
                    if self.pending.is_some() {
                        return self.notice("Wait for worker cleanup before resuming a session.");
                    }
                    match vade_harness::saved_sessions::resolve(&self.workspace, rest, true).and_then(|id| Journal::resume(&self.workspace, &id, Limits::default())).and_then(|(mut store, mut c, torn)| { super::native_setup::check_memory_resume(&c, self.config.memory)?; prepare_resumed(&mut store, &mut c, &self.workspace, self.config.model.as_deref().unwrap(), self.config.children, &self.config.child_grants, Limits { max_run_time: self.config.max_run_time, ..Limits::default() })?; Ok((store, c, torn)) }) {
                        Ok((store, conversation, torn)) => {
                            self.replace_native(conversation, vade_harness::contract::PromptBoundary::Resume)?;
                            self.journal = Some(Arc::new(Mutex::new(store)));
                            self.reset_session_rules()?;
                            self.resumed_answers();
                            self.session.clear(); self.native_completed = 0; self.native_report = None;
                            self.reset_recall();
                            self.notice(if torn { "Session resumed from its last complete step; an unfinished step was dropped and is not rerun." } else { "Session resumed; nothing was rerun or restored. /branch shows file changes since." })?;
                            self.notice_cutoffs()?;
                        }
                        Err(_) => self.notice("Cannot resume: missing, locked, incompatible, or invalid session. Current session retained.")?,
                    }
                }
                "/fork" if self.config.mode == Mode::Agent => {
                    if self.pending.is_some() {
                        return self.notice("Wait for worker cleanup before forking.");
                    }
                    let (name, label) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                    if !journal::valid_name(name) {
                        return self.notice("Usage: /fork NAME [TASK_LABEL]. Copies transcript/artifacts; shares workspace; no inference or file restore.");
                    }
                    let result = self
                        .journal
                        .as_ref()
                        .unwrap()
                        .lock()
                        .map_err(|_| io::Error::other("journal lock failed"))?
                        .fork(&self.workspace, name, label.trim(), Limits::default());
                    match result {
                        Ok((mut store, mut c)) => {
                            // The fork is durable, but activation must also save
                            // the revalidation note. Keep the parent on failure.
                            if append_resume_note(&mut store, &mut c).is_err() {
                                return self.notice("Fork created on disk but not selected: its revalidation note could not be saved. Current parent retained; no inference or file restore occurred.");
                            }
                            self.replace_native(c, vade_harness::contract::PromptBoundary::Fork)?;
                            self.journal = Some(Arc::new(Mutex::new(store)));
                            self.reset_session_rules()?;
                            self.resumed_answers();
                            self.native_completed = 0;
                            self.native_report = None;
                            self.session.clear();
                            self.reset_recall();
                            self.notice("Fork created and selected; the original session is kept and nothing was rerun. /branch shows where it came from; /switch NAME selects another.")?;
                            self.notice_cutoffs()?;
                        }
                        Err(e) => {
                            self.notice(&format!("Fork refused: {e}. Current session retained."))?
                        }
                    }
                }
                "/branch" if rest.is_empty() && self.config.mode == Mode::Agent => {
                    if self.pending.is_some() {
                        return self
                            .notice("Wait for worker cleanup before inspecting branch state.");
                    }
                    let result = self
                        .journal
                        .as_ref()
                        .unwrap()
                        .lock()
                        .map_err(|_| io::Error::other("journal lock failed"))?
                        .branch_status(&self.workspace);
                    match result {
                        Ok(text) => self.terminal.line(&text)?,
                        Err(e) => self.notice(&format!("Branch unavailable: {e}"))?,
                    }
                }
                "/instructions" if rest.is_empty() && matches!(self.config.mode, Mode::Direct | Mode::Agent) => {
                    if let Some(line) = self.instructions.refresh() {
                        self.notice(&line)?;
                    }
                    let text = self.instructions.inspect();
                    self.terminal.line(text.trim_end())?;
                }
                "/project" | "/project-rules" | "/skill" | "/map" | "/symbols"
                    if self.config.mode == Mode::Agent =>
                {
                    if self.pending.is_some() {
                        return self
                            .notice("Wait for worker cleanup before inspecting project context.");
                    }
                    let Some(c) = &self.native else {
                        return self.notice("Native context unavailable; use /new.");
                    };
                    let snapshot = project::Snapshot::from_conversation(c)?;
                    let result = match command {
                        "/project" if rest.is_empty() => Ok(snapshot.inspect()),
                        "/project-rules" if rest.is_empty() => Ok(snapshot.inspect_rules()),
                        "/skill" if !rest.is_empty() => snapshot.load_skill(&self.workspace, rest),
                        "/map" => project::manifest(&self.workspace, if rest.is_empty() { "." } else { rest }, 2),
                        "/symbols" if !rest.is_empty() => project::symbols(&self.workspace, rest),
                        _ => return self.notice("Usage: /project, /project-rules, /skill NAME, /map [PATH], /symbols PATH. Local inspection only; no scripts run."),
                    };
                    match result {
                        Ok(text) => self.terminal.line(&text)?,
                        Err(e) => self.notice(&format!("Project context refused: {e}"))?,
                    }
                }
                "/output" if self.config.mode == Mode::Agent => {
                    if self.pending.is_some() {
                        return self.notice("Wait for worker cleanup before inspecting artifacts.");
                    }
                    let fields: Vec<_> = rest.split_whitespace().collect();
                    if fields.len() != 3 {
                        return self.notice(
                            "Usage: /output ID OFFSET LENGTH (UTF-8 bytes; length 1–4096).",
                        );
                    }
                    let (Ok(offset), Ok(length)) =
                        (fields[1].parse::<usize>(), fields[2].parse::<usize>())
                    else {
                        return self.notice("OFFSET and LENGTH must be nonnegative integers.");
                    };
                    use vade_harness::artifacts::Access;
                    let result = self
                        .journal
                        .as_ref()
                        .unwrap()
                        .lock()
                        .map_err(|_| io::Error::other("journal lock failed"))?
                        .read(fields[0], offset, length, 8192);
                    match result {
                        Ok(text) => self.terminal.line(&text)?,
                        Err(e) => self.notice(&format!("Artifact unavailable: {e}"))?,
                    }
                }
                "/context" if rest.is_empty() && self.config.mode == Mode::Agent => {
                    if self.pending.is_some() {
                        self.notice("These numbers are from the previous request; the current one reports when it finishes.")?;
                    }
                    self.terminal.line(&self.context.describe())?;
                    for line in self.budget_lines() {
                        self.terminal.line(&line)?;
                    }
                    if self.config.children {
                        self.notice(&child_agents::ui::context(&self.children.snapshots()))?;
                    }
                    if let Some(c) = &self.native {
                        let usage = context::Usage::for_conversation(c);
                        self.terminal.line(&usage.summary(
                            self.config.context_budget(&self.context),
                            self.config.output_reserve(&self.context),
                        ))?;
                        self.terminal.line(&format!("options/schema ≈{} tokens (tools ≈{}, included); all message fields and text attachments counted; cached input is not subtracted", usage.options_estimated, usage.tools_estimated))?;
                        for m in usage.messages {
                            self.terminal.line(&format!(
                                "message {} · {} · ≈{} tokens · serialized bytes {}",
                                m.index, m.role, m.estimated, m.byte_bound
                            ))?;
                        }
                    } else {
                        self.notice("Context is owned by the worker; wait for cleanup.")?;
                    }
                }
                "/compaction" if rest.is_empty() && self.config.mode == Mode::Agent => {
                    if self.pending.is_some() {
                        return self
                            .notice("Wait for worker cleanup before inspecting compaction.");
                    }
                    let detail = self
                        .journal
                        .as_ref()
                        .unwrap()
                        .lock()
                        .map_err(|_| io::Error::other("journal lock failed"))?
                        .compaction_status();
                    self.terminal.line(&detail)?;
                }
                "/journal" if self.config.mode == Mode::Agent => {
                    if self.pending.is_some() {
                        return self
                            .notice("Wait for worker cleanup before inspecting journal records.");
                    }
                    let fields: Vec<_> = rest.split_whitespace().collect();
                    if fields.is_empty() {
                        let count = self
                            .journal
                            .as_ref()
                            .unwrap()
                            .lock()
                            .map_err(|_| io::Error::other("journal lock failed"))?
                            .record_count();
                        self.notice(&format!("{} raw records, numbered 0 through {}. /journal RECORD OFFSET LENGTH reads up to 4096 UTF-8 bytes locally.", count, count.saturating_sub(1)))?;
                    } else {
                        let numbers = fields
                            .iter()
                            .map(|s| s.parse::<usize>())
                            .collect::<Result<Vec<_>, _>>();
                        let Ok(n) = numbers else {
                            return self.notice("Usage: /journal RECORD OFFSET LENGTH (nonnegative UTF-8 byte offsets).");
                        };
                        if n.len() != 3 {
                            return self
                                .notice("Usage: /journal RECORD OFFSET LENGTH (length 1–4096).");
                        }
                        let result = self
                            .journal
                            .as_ref()
                            .unwrap()
                            .lock()
                            .map_err(|_| io::Error::other("journal lock failed"))?
                            .inspect_record(n[0], n[1], n[2]);
                        match result {
                            Ok(text) => self.terminal.line(&text)?,
                            Err(e) => self.notice(&format!("Journal inspection refused: {e}"))?,
                        }
                    }
                }
                "/compact" | "/compact-preview" if self.config.mode == Mode::Agent => {
                    if self.pending.is_some() {
                        return self.notice("Wait for worker cleanup before compaction.");
                    }
                    let Some(c) = &self.native else {
                        return self.notice("History unavailable; use /new.");
                    };
                    let limits = Limits {
                        // Planning/preview is local; applying must count the
                        // complete replacement before durable persistence.
                        // A resumed client may not have negotiated yet. Do not
                        // let a local estimate veto its exact candidate count.
                        max_context_tokens: usize::MAX,
                        output_reserve_tokens: self.config.output_reserve(&self.context),
                        ..Limits::default()
                    };
                    if rest.is_empty() && command == "/compact" {
                        // The model writes the summary on the count worker;
                        // local excerpts are its fallback.
                        let limits = Limits { max_context_tokens: self.config.context_budget(&self.context), max_run_time: self.config.max_run_time, ..limits };
                        return self.start_rollover(limits);
                    }
                    let plan = if rest.is_empty() {
                        // Same bounded-excerpt plan as automatic compaction;
                        // the user never has to write the summary.
                        context::auto_plan(c, 2, limits)
                    } else {
                        let Some((keep, summary)) = rest.split_once(char::is_whitespace) else {
                            return self.notice("Usage: /compact, or /compact KEEP_TURNS SUMMARY to supply your own summary.");
                        };
                        let Ok(keep) = keep.parse::<usize>() else {
                            return self.notice("KEEP_TURNS must be an integer of at least 1.");
                        };
                        context::plan(c, summary, keep, limits)
                    };
                    match plan {
                        Ok(plan) => {
                            self.terminal.line(&plan.preview())?;
                            if command == "/compact-preview" {
                                return Ok(());
                            }
                            let limits = Limits {
                                max_context_tokens: self.config.context_budget(&self.context),
                                ..limits
                            };
                            self.start_compaction_count(plan, limits)?;
                        }
                        Err(e) => self.notice(&format!("Compaction refused: {e}"))?,
                    }
                }
                "/context" if rest.is_empty() => {
                    if self.config.mode == Mode::Direct {
                        if self.pending.is_some() {
                            self.notice("These numbers are from the previous request; the current one reports when it finishes.")?;
                        }
                        self.notice(&self.context.describe())?;
                    } else {
                        self.notice("Context and cache numbers are not available in this mode.")?;
                    }
                }
                "/new" | "/clear" if rest.is_empty() => {
                    if self.pending.is_some() {
                        return self.notice("Wait for request cleanup before /new.");
                    }
                    self.memory = if self.config.memory { vade_harness::memory::Memory::load(&self.workspace)?.content() } else { String::new() };
                    self.timed_out_prompt = None;
                    self.reset_recall();
                    self.reset_answers();
                    self.direct_prefix_blocked = false;
                    self.context = vade_harness::contract::Context::default();
                    if let Some(http) = self.http.as_mut() {
                        http.reset_context();
                    }
                    if let Some(line) = self.instructions.refresh() {
                        self.notice(&line)?;
                    }
                    if self.config.mode == Mode::Agent {
                        let build_session = Arc::new(vade_harness::shell::BuildSession::new(
                            self.config.shell_toolchain.as_deref(),
                        )?);
                        let conversation = chat_conversation_with_mcp(
                            self.config.model.as_deref().unwrap(),
                            &self.workspace,
                            &self.config.rules_scopes,
                            self.instructions.section(),
                            self.config.memory.then_some(self.memory.as_str()),
                            self.config.web,
                            self.config.children,
                            &self.config.child_grants,
                            Limits { max_run_time: self.config.max_run_time, ..Limits::default() },
                            self.mcp_definitions(),
                        )?;
                        let store = Journal::create(
                            &self.workspace,
                            &journal::fresh_name(),
                            &conversation,
                        )?;
                        self.journal = Some(Arc::new(Mutex::new(store)));
                        self.reset_session_rules()?;
                        self.native = Some(conversation);
                        self.auto_compaction =
                            Arc::new(Mutex::new(context::AutoCompaction::default()));
                        self.budget = Default::default();
                        self.native_completed = 0;
                        self.native_report = None;
                        self.native_model = None;
                        self.build_session = build_session;
                        self.children = native_children::Children::default();
                        self.session.clear();
                        self.notice("New session started; project rules reloaded. The previous journal stays on disk.")?;
                    } else if self.config.mode.delegated() {
                        self.backend = None;
                        self.delegated_id = None;
                        self.config.thread = None;
                        self.config.external_session = None;
                        self.session.clear();
                        self.notice("The next prompt starts a new external session; workspace input history is retained.")?;
                    } else {
                        self.session.clear();
                        self.notice("Conversation cleared.")?;
                    }
                }
                "/undo" if rest.is_empty() => {
                    if self.pending.is_some() {
                        return self.notice("Wait for active request cleanup before /undo.");
                    }
                    if self.config.mode == Mode::Agent {
                        return self.notice("/undo is not available in agent mode, because tool steps cannot be removed cleanly. Use /new to start over.");
                    }
                    if self.config.mode.delegated() {
                        return self.notice("The external agent owns its history; /undo is unavailable. Use /new for a fresh external session.");
                    }
                    self.reset_recall();
                    let removed = self.session.undo();
                    self.reset_answers();
                    // Explicit rollback is an intentional prompt cache boundary.
                    self.direct_prefix_blocked = self.session.pruned_turns > 0;
                    self.notice(if removed {
                        "Last complete exchange removed."
                    } else {
                        "No complete exchange to undo."
                    })?;
                }
                "/ls" => match self
                    .workspace
                    .list(if rest.is_empty() { "." } else { rest })
                {
                    Ok(entries) => {
                        if entries.is_empty() {
                            self.terminal.note("No visible entries.")?;
                        }
                        for entry in entries {
                            self.terminal.line(&format!("  {entry}"))?;
                        }
                    }
                    Err(error) => self.notice(&format!("Workspace: {error}"))?,
                },
                "/read" if !rest.is_empty() => match self.workspace.read(rest) {
                    Ok(text) => {
                        self.terminal
                            .note("Local file preview · not sent to model")?;
                        self.terminal.stream_text(&text)?;
                        self.dirty = true;
                        if !text.ends_with('\n') {
                            self.terminal.line("")?;
                        }
                    }
                    Err(error) => self.notice(&format!("Workspace: {error}"))?,
                },
                "/paste" if rest.is_empty() => {
                    self.multiline = Some(String::new());
                    self.multiline_lines = 0;
                    self.notice("Multiline input · /end sends · /cancel discards")?;
                }
                _ => {
                    if help::lookup(command).is_some() { return self.notice(&help::unknown_command(command, self.config.mode)); }
                    match self.user_commands() {
                        Ok(commands) => {
                            if let Some(template) = commands.iter().find(|c| c.name == command[1..]) {
                                return match template.expand(raw_rest) {
                                    Ok(prompt) => self.start(prompt),
                                    Err(error) => self.notice(&error.to_string()),
                                };
                            }
                            self.notice(&format!("{}\nAvailable commands: {}", help::unknown_command(command, self.config.mode), self.command_names(&commands)))?;
                        }
                        Err(error) => self.notice(&format!("Cannot read user commands: {error}"))?,
                    }
                }
            }
            self.dirty = true;
            return Ok(());
        }
        self.start_input(line)
    }
    fn user_commands(&self) -> io::Result<Vec<vade_harness::slash_commands::Template>> {
        let home = vade_harness::slash_commands::home(self.config.state.as_ref());
        // Unit fixtures never read the developer's command directory.
        let home = home.filter(|_| !cfg!(test) || matches!(self.config.state, Some(vade_harness::tools::StateLocation::Home(_))));
        vade_harness::slash_commands::load(&self.workspace, home.as_deref())
    }
    fn command_names(&self, commands: &[vade_harness::slash_commands::Template]) -> String {
        let mut names: Vec<_> = COMMANDS.iter().filter(|c| c.available(self.config.mode)).map(|c| format!("/{}", c.name))
            .chain(commands.iter().filter(|c| help::lookup(&c.name).is_none()).map(|c| format!("/{}", c.name))).collect();
        names.sort();
        names.join(", ")
    }
    fn command_help(&mut self, query: &str) -> io::Result<()> {
        let commands = match self.user_commands() {
            Ok(commands) => commands,
            Err(error) => { self.notice(&format!("Cannot read user commands: {error}"))?; Vec::new() }
        };
        let name = query.trim_start_matches('/');
        if !query.is_empty() && query != "all" && help::lookup(name).is_none()
            && let Some(command) = commands.iter().find(|c| c.name == name)
        { return self.terminal.line(&format!("/{} [ARGUMENTS]\n{}", command.name, command.description)); }
        self.terminal.line(&help::command_help(query, self.config.mode))?;
        if query.is_empty() || query == "all" {
            for command in commands.iter().filter(|c| help::lookup(&c.name).is_none()) {
                self.terminal.line(&format!("  /{} [ARGUMENTS]  {}", command.name, command.description))?;
            }
        }
        Ok(())
    }
    fn copy_answer(&mut self, rest: &str) -> io::Result<()> {
        let code = rest == "code";
        let index = if rest.is_empty() || code {
            1
        } else if let Ok(n) = rest.parse::<usize>()
            && n > 0
        {
            n
        } else {
            return self.notice("Usage: /copy [code|N] · N starts at 1 for the last answer.");
        };
        if self.config.mode == Mode::Agent && self.pending.is_some() {
            return self.notice("Wait for native worker cleanup before /copy.");
        }
        let text = if let Some(native) = &self.native {
            native
                .messages()
                .iter()
                .rev()
                .filter(|m| m.role() == "assistant" && m.tool_calls().is_empty())
                .filter_map(|m| m.value().get("content"))
                .map(|value| match value.as_str() {
                    Some(text) => std::borrow::Cow::Borrowed(text),
                    None => std::borrow::Cow::Owned(super::native_setup::display_content(value)),
                })
                .filter(|text| !text.is_empty())
                .nth(index - 1)
        } else {
            self.session
                .history()
                .iter()
                .rev()
                .filter(|m| m.role == "assistant")
                .map(|m| std::borrow::Cow::Borrowed(m.content.as_str()))
                .nth(index - 1)
        };
        let Some(text) = text else {
            return self.notice("No answer available at that history position.");
        };
        let text = text.as_ref();
        let text = if code {
            let Some(block) = vade_harness::terminal::clipboard::last_code(text) else {
                return self.notice("No closed fenced code block in the last answer.");
            };
            block
        } else {
            text
        };
        let result = vade_harness::terminal::clipboard::copy(text, self.terminal.interactive());
        let message = match result {
            Ok(method) => format!("Copied {} chars ({method})", text.chars().count()),
            Err(error) => error.to_string(),
        };
        self.notice(&message)
    }
    pub(crate) fn status(&mut self) -> io::Result<()> {
        if matches!(self.config.mode, Mode::Agent | Mode::Direct) {
            self.run_notice(&self.context.session_speed.describe())?;
            for turn in self.context.turns.clone() {
                self.run_notice(&turn.speed())?;
            }
        }
        let state = match &self.pending {
            Some(p) if p.canceled => RunState::Cancelling.label(),
            Some(p) => p.rx.state().label(),
            None => self.last_state.map_or("ready", RunState::label),
        };
        self.run_notice(&format!(
            "Follow-ups queued: {} · steering pending: {}",
            self.queued.len(),
            self.pending
                .as_ref()
                .map_or(0, |p| p.steering_pending.len())
        ))?;
        if matches!(self.config.mode, Mode::Agent | Mode::Direct) {
            let sampling = match (self.answers.request_seed(), self.config.generation.temperature) {
                (None, None) => "Sampling: upstream defaults (no sampling fields)".into(),
                (seed, temperature) => format!(
                    "Sampling: seed {} · temperature {}",
                    seed.map_or_else(|| "unspecified".into(), |seed| seed.to_string()),
                    temperature.map_or_else(|| "maker default".into(), |t| t.to_string())
                ),
            };
            self.run_notice(&sampling)?;
        }
        if self.config.mode == Mode::Agent {
            self.run_notice(&self.context.run_speed())?;
            self.run_notice(&self.context.describe())?;
            self.run_notice(&format!(
                "Children: {} · {} tasks in current/last activation · {} retained results · /children",
                if self.config.children { "enabled (host-scoped)" } else { "disabled" },
                self.children.snapshots().len(), self.children.retained_results(),
            ))?;
            let history = self.native.as_ref().map_or_else(
                || "owned by active worker or unavailable; wait for cleanup".to_string(),
                |conversation| {
                    format!(
                        "{} typed messages / {} request bytes",
                        conversation.messages().len(),
                        conversation.serialized_bytes()
                    )
                },
            );
            let context = self
                .native
                .as_ref()
                .map(|c| {
                    context::Usage::for_conversation(c).summary(
                        self.config.context_budget(&self.context),
                        self.config.output_reserve(&self.context),
                    )
                })
                .unwrap_or_else(|| "context owned by worker".into());
            let session = self
                .journal
                .as_ref()
                .and_then(|j| j.lock().ok().map(|j| j.name().to_owned()))
                .unwrap_or_else(|| "unavailable".into());
            let web_tools = if self.config.web { "/web_fetch/web_search" } else { "" };
            return self.terminal.line(&format!("native agent · {state}\nhistory: {history} · auto-compaction: {}\n{context}\nsuccessful turns since new/resume: {}\ntools: workspace_read/workspace_list/workspace_search/workspace_read_lines/workspace_edit_lines/apply_patch{}/tool_output_read/skill_read/workspace_manifest/workspace_symbols{web_tools}\napproval: {} · {}\nsession: {session} · private durable journal\nlast run: {}", if self.config.auto_compact { "on (80%)" } else { "off" }, self.native_completed, if vade_harness::coding_tools::shell_offered() { "/workspace_shell" } else { "" }, self.config.approval.label(), if vade_harness::coding_tools::shell_offered() { "macOS sandbox · network denied" } else if vade_harness::coding_tools::SHELL_AVAILABLE { "no command tool: VADE_HARNESS_SHELL=off" } else { "no command tool here: workspace_shell needs the macOS sandbox" }, self.native_report.as_deref().unwrap_or("none")));
        }
        if self.config.mode == Mode::Codex {
            return self.terminal.line(&format!("provider: Codex app-server · {state}\npermission: {} · approval requests: denied by adapter\nread scope: Codex preset; not confined to this workspace\nthread: {}\nmodel override: {}\nlocal recall: {} completed turns / {} bytes (not sent as context)\ncontext and persistence: owned by Codex; no Vade system/history prefix\nVade metadata: completed-turn counter only, {}\ncache identity: separate from provider/model/thread; cache hits unmeasured",
                self.config.permission.unwrap().label(), self.delegated_id.as_deref().or(self.config.thread.as_deref()).unwrap_or("new thread on first prompt"),
                self.config.model.as_deref().unwrap_or("Codex configured default"), self.session.turns(), self.session.history_bytes(),
                if self.session.persistent() { "enabled" } else { "off" }));
        }
        if self.config.mode == Mode::Hermes {
            return self.terminal.line(&format!("provider: Hermes ACP v1 · {state}\npermission callbacks: denied; host filesystem and terminal disabled\nexternal tools: owned by Hermes; no OS sandbox or read boundary verified\nsession: {}\nmodel/authentication/persistence: configured externally; no login attempted\nlocal recall: {} completed turns / {} bytes (not sent as context)\nno native loop or Vade system/history prefix; cache hits unmeasured",
                self.delegated_id.as_deref().or(self.config.external_session.as_deref()).unwrap_or("new session on first prompt"),
                self.session.turns(), self.session.history_bytes()));
        }
        self.terminal.line(&format!("{} · {state}\nhistory: {} complete turns / {} bytes · {} old turns evicted\ncompleted turns: {} · metadata persistence: {}\nmodel context: stable system text + exact retained exchanges + current prompt\nprovider token/cache telemetry: /context; inference and KV remain upstream-owned",
            if self.config.mode == Mode::Demo { "demo (no network)" } else { "direct" }, self.session.turns(), self.session.history_bytes(), self.session.pruned_turns,
            self.session.metadata.completed_turns, if self.session.persistent() { "counter only" } else { "off" }))
    }
    pub(crate) fn children_command(&mut self, rest: &str) -> io::Result<()> {
        if self.config.mode != Mode::Agent {
            return self.notice("Children are available only in native agent mode.");
        }
        match rest {
            "on" | "off" => {
                if self.pending.is_some() {
                    return self.notice("Wait for request cleanup before changing child tools.");
                }
                let enabled = rest == "on";
                if enabled == self.config.children {
                    return self.notice(if enabled {
                        "Children already enabled."
                    } else {
                        "Children already disabled."
                    });
                }
                if !self
                    .native
                    .as_ref()
                    .is_some_and(|c| c.messages().len() == c.prefix_len())
                {
                    return self.notice("Child tools are frozen for this conversation. Use /new, then /children on|off before the first prompt.");
                }
                let conversation = chat_conversation_with_mcp(
                    self.config.model.as_deref().unwrap(),
                    &self.workspace,
                    &self.config.rules_scopes,
                    self.instructions.section(),
                    self.config.memory.then_some(self.memory.as_str()),
                    self.config.web,
                    enabled,
                    &self.config.child_grants,
                    Limits { max_run_time: self.config.max_run_time, ..Limits::default() },
                    self.mcp_definitions(),
                )?;
                let store =
                    Journal::create(&self.workspace, &journal::fresh_name(), &conversation)?;
                self.journal = Some(Arc::new(Mutex::new(store)));
                self.reset_session_rules()?;
                self.timed_out_prompt = None;
                self.native = Some(conversation);
                self.native_model = None;
                self.config.children = enabled;
                self.children = native_children::Children::default();
                self.notice(if enabled {
                    "Children enabled · read-only by default · same endpoint/model · exact admission required. /children cancel ID|all requests local cancellation."
                } else { "Children disabled for this fresh conversation." })
            }
            "" | "list" => {
                if !self.config.children {
                    return self.notice("Children disabled. Use --children or /children on before the first prompt.");
                }
                self.notice(&child_agents::ui::render(
                    &self.children.snapshots(),
                    240,
                    33,
                ))?;
                self.notice("/children cancel ID|all · local cancellation only; parent owns result integration")
            }
            _ => {
                let fields: Vec<_> = rest.split_whitespace().take(3).collect();
                if let ["cancel", target] = fields.as_slice() {
                    let receipt = self.children.cancel(target);
                    self.notice(&receipt)
                } else {
                    self.notice("Usage: /children [list|on|off|cancel ID|cancel all]")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
