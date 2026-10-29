use std::io;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::Instant;
use vade_harness::{
    agent::{Conversation, Limits, Model, RunBudget},
    context::{self, CompactionPlan, Usage},
    transport,
};

use super::{
    App, RunState,
    events::{Pending, TurnResult, WorkerResult, error_category},
    native_setup::count_before_save,
};
use crate::interaction;

enum Candidate {
    Prompt {
        conversation: Conversation,
        retrying: bool,
    },
    Compaction(CompactionPlan),
    /// Manual /compact: the model writes the summary on the worker, which
    /// then counts it like any other compaction plan.
    Rollover,
}

/// Counts and summaries through the chat's own native client.
struct NativeHost<'a> {
    model: &'a mut transport::NativeModel,
    limits: Limits,
    cancel: &'a AtomicBool,
}
impl context::ContextHost for NativeHost<'_> {
    fn count(&mut self, c: &Conversation) -> io::Result<Usage> {
        count_before_save(self.model, c, self.limits, self.cancel)
    }
    fn summarize(&mut self, request: &Conversation) -> io::Result<String> {
        context::summarize_with(self.model, request, self.limits.wire, self.cancel)
    }
    fn automatic_output(&self) -> bool {
        vade_harness::agent::Model::automatic_output(&*self.model)
    }
}

/// The rollover plan for manual /compact with its final count, or the local
/// excerpt plan (to be counted) when the model cannot write one. Nothing is
/// saved here.
fn rollover_candidate(
    model: &mut transport::NativeModel,
    c: &Conversation,
    limits: Limits,
    cancel: &AtomicBool,
    context_limit: Option<u64>,
) -> Result<(CompactionPlan, Option<Usage>), String> {
    let before = count_before_save(model, c, limits, cancel)
        .map_err(|error| format!("Compaction refused: count failed ({}); history and journal unchanged.", error_category(&error)))?;
    let counted = Limits {
        max_context_tokens: usize::try_from(model.context().token_limit(context_limit)).unwrap_or(usize::MAX),
        output_reserve_tokens: model.output_reserve_tokens(limits.output_reserve_tokens),
        ..limits
    };
    let mut host = NativeHost { model, limits, cancel };
    match context::rollover(c, context::KEEP_TURNS, counted, &before, &mut host) {
        Ok((plan, after)) => Ok((plan, Some(after))),
        Err(_) if cancel.load(Ordering::Relaxed) => Err("Compaction cancelled; history and journal unchanged.".into()),
        Err(error) => context::auto_plan(c, 2, Limits { max_context_tokens: usize::MAX, ..limits })
            .map(|plan| (plan, None))
            .map_err(|fallback| format!("Compaction refused: the model summary failed ({error}) and local excerpts failed ({fallback})."))
    }
}

pub(crate) struct Counted {
    model: transport::NativeModel,
    candidate: Candidate,
    result: Result<Usage, String>,
    budget: Option<RunBudget>,
}

impl App {
    pub(crate) fn start_prompt_count(
        &mut self,
        user: String,
        conversation: Conversation,
        model: transport::NativeModel,
        limits: Limits,
        retrying: bool,
    ) -> io::Result<()> {
        // A refused next input may need another in-task rewrite even while
        // optional threshold compaction is waiting for its low watermark.
        let can_compact = self.config.auto_compact;
        self.start_count(
            user,
            model,
            Candidate::Prompt {
                conversation,
                retrying,
            },
            limits,
            can_compact,
        )
    }

    pub(crate) fn start_rollover(&mut self, limits: Limits) -> io::Result<()> {
        let Some(model) = self.native_model.take() else {
            return self.notice("Compaction refused: native client unavailable.");
        };
        self.start_count(String::new(), model, Candidate::Rollover, limits, false)
    }

    pub(crate) fn start_compaction_count(
        &mut self,
        plan: CompactionPlan,
        limits: Limits,
    ) -> io::Result<()> {
        let Some(model) = self.native_model.take() else {
            return self.notice("Compaction refused: native client unavailable.");
        };
        self.start_count(
            String::new(),
            model,
            Candidate::Compaction(plan),
            limits,
            false,
        )
    }

    fn start_count(
        &mut self,
        user: String,
        mut model: transport::NativeModel,
        mut candidate: Candidate,
        limits: Limits,
        can_compact: bool,
    ) -> io::Result<()> {
        // The live history and journal stay on the UI until the worker returns
        // an admitted candidate and the UI checks cancellation before saving.
        let before = matches!(candidate, Candidate::Compaction(_) | Candidate::Rollover)
            .then(|| self.native.as_ref().expect("native count history").clone());
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        // A model summary is generation too, so the run time limit applies.
        let prompt = matches!(candidate, Candidate::Prompt { .. });
        let budget = (prompt || matches!(candidate, Candidate::Rollover))
            .then(|| RunBudget::new(limits.max_run_time));
        let ui_budget = budget.clone();
        let (tx, rx) = interaction::lanes();
        tx.state(if matches!(candidate, Candidate::Rollover) { RunState::Compacting } else { RunState::Counting });
        let (controls, _) = mpsc::sync_channel(1);
        let context_limit = self.config.generation.context_limit;
        let worker = std::thread::spawn(move || {
            // A model plan arrives already counted; an excerpt plan is counted below.
            let mut early = None;
            if matches!(candidate, Candidate::Rollover) {
                let mut roll = || rollover_candidate(&mut model, before.as_ref().unwrap(), limits, &worker_cancel, context_limit);
                let rolled = match &budget {
                    Some(budget) => budget.with_cancellation(&worker_cancel, roll),
                    None => roll(),
                };
                match rolled {
                    Ok((plan, after)) => {
                        candidate = Candidate::Compaction(plan);
                        early = after.map(Ok);
                    }
                    Err(message) => early = Some(Err(message)),
                }
            }
            let mut count = || {
                match &mut candidate {
                Candidate::Prompt { conversation, .. } => {
                    count_before_save(&mut model, conversation, limits, &worker_cancel)
                        .map_err(|error| format!(
                            "Could not check whether the message fits ({}); it was not sent. {}",
                            error_category(&error),
                            model.context().failure.unwrap_or("")
                        ))
                        .and_then(|usage| {
                            let limits = Limits {
                                max_context_tokens: usize::try_from(model.context().token_limit(context_limit)).unwrap_or(usize::MAX),
                                output_reserve_tokens: model.output_reserve_tokens(limits.output_reserve_tokens),
                                ..limits
                            };
                            let fits = usage.fits(limits.max_context_tokens, limits.output_reserve_tokens);
                            let automatic = vade_harness::agent::Model::automatic_output(&model);
                            let mut count = |candidate: &Conversation| {
                                model.count_input(candidate.request(limits.wire), &worker_cancel)
                                    .and_then(|input| Usage::for_conversation(candidate).with_count(&input))
                            };
                            // The run's observer performs the rewrite; here only
                            // check that a rollover or local excerpts can fit.
                            let room = (!fits && can_compact).then(|| {
                                context::rollover_room(conversation, context::KEEP_TURNS, limits, &usage, &mut context::CountOnly(&mut count, automatic))
                            });
                            let recoverable = match room {
                                Some(Ok(_)) => true,
                                Some(Err(_)) => context::auto_plan_fitting(conversation, 2, limits, &usage, &mut count).is_ok_and(|plan| plan.is_some()),
                                None => false,
                            };
                            // Even a compacted prompt does not fit: say so plainly. An
                            // admission issue or an unknown outcome gets the general text.
                            let window = usage.context_limit(limits.max_context_tokens);
                            match room {
                                _ if recoverable || usage.admission_issue.is_some() => (),
                                Some(Err(context::RolloverError::NoRoom { needed, window })) => return Err(context::hard_stop(needed, window)),
                                // The newest turn alone is the floor: its count is what any compaction keeps.
                                Some(Err(context::RolloverError::NotApplicable("no older turns to summarize"))) => return Err(context::hard_stop(
                                    context::needed_tokens(&usage, window, limits, automatic), window)),
                                _ => (),
                            }
                            if !fits && !recoverable {
                                return Err(format!("The message would not fit in the model's context (or its size could not be estimated); it was not sent. {}. Use /compact to shrink older turns, or /new to start over.", usage.summary(limits.max_context_tokens, limits.output_reserve_tokens)));
                            }
                            Ok(usage)
                        })
                }
                Candidate::Compaction(plan) => {
                    count_before_save(&mut model, before.as_ref().unwrap(), limits, &worker_cancel)
                        .and_then(|before| {
                            count_before_save(&mut model, plan.conversation(), limits, &worker_cancel)
                                .map(|after| (before, after))
                        })
                        .map_err(|error| format!("Compaction refused: count failed ({}); history and journal unchanged.", error_category(&error)))
                        .and_then(|(before, after)| {
                            let limits = Limits { max_context_tokens: usize::try_from(model.context().token_limit(context_limit)).unwrap_or(usize::MAX), ..limits };
                            if !after.fits(limits.max_context_tokens, model.output_reserve_tokens(limits.output_reserve_tokens)) {
                                return Err("Compaction refused: counted replacement exceeds the context budget; history and journal unchanged.".into());
                            }
                            if before.exact.zip(after.exact).is_some_and(|(a, b)| b.0 >= a.0) {
                                return Err("Compaction refused: replacement did not shrink the exact input count; history and journal unchanged.".into());
                            }
                            plan.record_counts(&before, &after);
                            Ok(after)
                        })
                }
                Candidate::Rollover => Err("Compaction refused.".into()),
            }
            };
            let result = match (early, &budget) {
                (Some(result), _) => result,
                (None, Some(budget)) => budget.with_cancellation(&worker_cancel, count),
                (None, None) => count(),
            };
            tx.finish(WorkerResult {
                http: None,
                backend: None,
                result: Ok(TurnResult::Counted(Counted {
                    model,
                    candidate,
                    result,
                    budget,
                })),
            });
        });
        self.pending = Some(Pending {
            native_progress: ui_budget.as_ref().filter(|_| prompt).map(|_| super::events::NativeProgress {
                rounds: 0, limits, context: None,
            }),
            budget: ui_budget,
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
            join_on_drop: true,
        });
        self.dirty = true;
        Ok(())
    }

    pub(crate) fn finish_counted(
        &mut self,
        pending: &mut Pending,
        counted: Counted,
    ) -> io::Result<()> {
        let Counted {
            model,
            candidate,
            result,
            budget,
        } = counted;
        if matches!(candidate, Candidate::Prompt { .. })
            && model.context().capability_reconciliations > self.context.capability_reconciliations
        {
            self.run_notice("The engine was updated; Vade will refresh its settings and resend the full conversation.")?;
        }
        self.context = model.context().clone();
        self.native_model = Some(model);
        self.dirty = true;
        if !pending.canceled && budget.as_ref().is_some_and(RunBudget::expired) {
            if !matches!(candidate, Candidate::Prompt { .. }) {
                self.last_state = Some(RunState::Failed);
                return self.notice(&format!("Compaction stopped at the {} s run time limit; history and journal unchanged.", self.config.max_run_time.unwrap_or_default().as_secs()));
            }
            self.failed = true;
            self.queue_paused = true;
            self.last_success = false;
            self.last_state = Some(RunState::Failed);
            self.timed_out_prompt = Some(std::mem::take(&mut pending.user));
            return self.run_notice(&super::native_setup::time_limit_notice(
                self.config.max_run_time,
            ));
        }
        if pending.canceled || pending.cancel.load(Ordering::Relaxed) {
            // Steering during counting: the original prompt exists only in
            // this unadmitted candidate. Recount it with the steering prompt
            // appended, so both reach generation in the order typed and the
            // journal still only saves an admitted conversation.
            if !self.quit && self.steering.is_some()
                && matches!(candidate, Candidate::Prompt { retrying: false, .. })
            {
                let steering = self.steering.take().unwrap();
                let Candidate::Prompt { mut conversation, retrying } = candidate else {
                    unreachable!("checked prompt candidate above")
                };
                let limits = pending
                    .native_progress
                    .as_ref()
                    .map_or_else(Limits::default, |progress| progress.limits);
                if conversation
                    .push_user(
                        super::native_setup::native_message("user", &steering),
                        Limits::default(),
                    )
                    .is_err()
                {
                    self.failed = true;
                    self.queue_paused = true;
                    self.last_success = false;
                    self.last_state = Some(RunState::LocalCancelled);
                    // Preserve each prompt separately: a combined draft could
                    // truncate the correction (attachment display limits, the
                    // 64 KiB prompt bound) and lose it silently.
                    let original = std::mem::take(&mut pending.user);
                    let kept = self.keep_draft(&original);
                    self.run_notice(&format!(
                        "The conversation is full; neither message was sent.{kept} The correction is held for /next after /new or /compact."
                    ))?;
                    return self.queue(steering);
                }
                if !self.internal_prompt(&steering) {
                    if !self.quiet {
                        self.terminal
                            .user_message(super::attachments::display(&steering))?;
                    }
                    self.reset_recall();
                    self.input
                        .remember_async(super::attachments::display(&steering));
                }
                self.terminal.begin_run();
                let model = self.native_model.take().unwrap();
                let combined = format!("{}\n\n{steering}", std::mem::take(&mut pending.user));
                self.queue_paused = false;
                return self.start_prompt_count(combined, conversation, model, limits, retrying);
            }
            if self.steering.is_none() && !self.quit { self.failed = true; }
            self.queue_paused = true;
            self.last_success = false;
            self.last_state = Some(RunState::LocalCancelled);
            return self.run_notice(presend_cancel_notice(self.quit, matches!(candidate, Candidate::Prompt { .. })));
        }
        let usage = match result {
            Ok(usage) => usage,
            Err(message) => {
                if matches!(candidate, Candidate::Prompt { .. }) {
                    self.failed = true;
                    self.queue_paused = true;
                    self.last_success = false;
                }
                self.last_state = Some(RunState::Failed);
                return match candidate {
                    Candidate::Prompt { .. } => { let kept = self.keep_draft(&pending.user); self.run_notice(&format!("{message}{kept}")) },
                    Candidate::Compaction(_) | Candidate::Rollover => self.notice(&message),
                };
            }
        };
        match candidate {
            Candidate::Prompt {
                conversation,
                retrying,
            } => {
                self.journal
                    .as_ref()
                    .unwrap()
                    .lock()
                    .map_err(|_| io::Error::other("journal lock failed"))?
                    .save(&conversation)?;
                self.native = Some(conversation);
                if let Some(seed) = self.answers.request_seed() {
                    self.journal
                        .as_ref()
                        .unwrap()
                        .lock()
                        .map_err(|_| io::Error::other("journal lock failed"))?
                        .set_generation_seed(seed)?;
                }
                if !self.config.auto_compact
                    && usage.near_limit(
                        self.config.context_budget(&self.context),
                        self.config.output_reserve(&self.context),
                    )
                {
                    self.run_notice(&format!(
                        "Warning: {}. Use /compact KEEP SUMMARY deliberately after this run.",
                        usage.summary(
                            self.config.context_budget(&self.context),
                            self.config.output_reserve(&self.context)
                        )
                    ))?;
                }
                self.auto_compaction
                    .lock()
                    .map_err(|_| io::Error::other("compaction lock failed"))?
                    .begin_turn();
                let model = self.native_model.take();
                self.start_generation(std::mem::take(&mut pending.user), retrying, model, budget)
            }
            Candidate::Compaction(plan) => {
                self.journal
                    .as_ref()
                    .unwrap()
                    .lock()
                    .map_err(|_| io::Error::other("journal lock failed"))?
                    .compact(&plan)?;
                let line = plan.status_line();
                self.replace_native(
                    plan.into_conversation(),
                    vade_harness::contract::PromptBoundary::Compaction,
                )?;
                match line {
                    Some(line) => self.notice(&line)?,
                    None => self.notice("Context compacted. Older turns are summarized; the full record stays on disk (/compaction shows it). The next reply rebuilds the cache, so it may start slower.")?,
                }
                self.start_compaction_warm()
            }
            Candidate::Rollover => self.notice("Compaction refused; history and journal unchanged."),
        }
    }
}

/// An exit (/exit, Ctrl+C, Ctrl+D) that arrives while a prompt is still being
/// counted says so; piped `prompt\n/exit` input reaches here before any reply.
pub(crate) fn presend_cancel_notice(quit: bool, prompt: bool) -> &'static str {
    if quit && prompt { "Exited before the message was sent; it was not sent." } else { "Canceled before sending; nothing changed." }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;
    use vade_harness::{json, terminal::Key};

    #[test]
    fn cancel_while_counting_prints_one_presend_line() {
        use super::super::{args::Mode, worker::cancel_notice};
        for mode in [Mode::Agent, Mode::Direct, Mode::Demo, Mode::Codex, Mode::Hermes] {
            assert_eq!(cancel_notice(mode, true), None);
            assert!(cancel_notice(mode, false).is_some());
        }
        assert_eq!(presend_cancel_notice(true, true), "Exited before the message was sent; it was not sent.");
        assert_eq!(presend_cancel_notice(false, true), "Canceled before sending; nothing changed.");
        assert_eq!(presend_cancel_notice(true, false), "Canceled before sending; nothing changed.");
    }

    fn request(reader: &mut BufReader<TcpStream>) -> String {
        let mut route = String::new();
        reader.read_line(&mut route).unwrap();
        let mut length = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    length = value.trim().parse::<usize>().unwrap();
                }
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        route
    }

    fn respond(reader: &mut BufReader<TcpStream>, body: &str) {
        write!(
            reader.get_mut(),
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
    }

    #[test]
    fn stalled_presave_counts_cancel_promptly_without_persistence() {
        // Prompt admission, its reducing auto-plan count, and both manual
        // compaction counts all use the same UI-controlled cancellation flag.
        // Case 5: the run time limit also stops a manual model rollover.
        for (case, compact, stall_at, interrupt) in [
            (0, false, 0, false),
            (1, false, 1, true),
            (2, true, 0, true),
            (3, true, 1, false),
            (4, false, 0, false),
            (5, true, 1, false),
        ] {
            let path = std::env::temp_dir()
                .join(format!("vade-presave-cancel-{}-{case}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            struct Cleanup(std::path::PathBuf);
            impl Drop for Cleanup {
                fn drop(&mut self) {
                    let _ = std::fs::remove_dir_all(&self.0);
                }
            }
            let _cleanup = Cleanup(path.clone());
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let (stalled, ready) = mpsc::channel();
            let upstream = std::thread::spawn(move || {
                let (socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut reader = BufReader::new(socket);
                let caps = include_str!("../../tests/fixtures/engine-capabilities-v1.json")
                    .replace("4096", "16384")
                    .replace("4000", "16000");
                for count in 0..=stall_at {
                    assert!(request(&mut reader).starts_with("GET /v1/capabilities "));
                    respond(&mut reader, &caps);
                    assert!(request(&mut reader).starts_with("POST /v1/admission "));
                    if count == stall_at {
                        stalled.send(()).unwrap();
                        if case == 4 { respond(&mut reader, "{}"); return; }
                        // Send no response at all. Cancellation must close the
                        // local socket; another count or generation is forbidden.
                        assert_eq!(reader.read(&mut [0]).unwrap(), 0);
                        listener.set_nonblocking(true).unwrap();
                        assert_eq!(
                            listener.accept().unwrap_err().kind(),
                            io::ErrorKind::WouldBlock
                        );
                    } else {
                        let tokens = if compact { 2000 } else { 9000 };
                        respond(
                            &mut reader,
                            &format!(
                                r#"{{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","admission_kind":"context_fit","reserves_resources":false,"prompt_tokens":{tokens},"output_tokens":32,"context_limit":16384,"input_limit":16000,"output_limit":32,"admitted":true}}"#
                            ),
                        );
                    }
                }
            });
            let limit = if case == 5 { &["--agent-time-limit-seconds", "1"][..] } else { &[] };
            let super::super::args::Parsed::Run(config) = super::super::args::Config::parse(
                [
                    "agent",
                    "--endpoint",
                    &endpoint,
                    "--model",
                    "mock",
                    "--workspace",
                    path.to_str().unwrap(),
                    "--session",
                    "count",
                    "--plain",
                    "--context-tokens",
                    "8192",
                    "--max-output-tokens",
                    "32",
                ]
                .into_iter()
                .chain(limit.iter().copied())
                .map(str::to_owned),
            )
            .unwrap() else {
                panic!("expected native configuration");
            };
            let mut app = App::new(config).unwrap();
            let json::Value::Object(mut fields) = json::parse(
                &app.native
                    .as_ref()
                    .unwrap()
                    .request(Limits::default().wire)
                    .to_json()
                    .unwrap(),
            )
            .unwrap() else {
                panic!("expected request object");
            };
            let json::Value::Array(messages) = &mut fields
                .iter_mut()
                .find(|(key, _)| key == "messages")
                .unwrap()
                .1
            else {
                panic!("expected messages");
            };
            for turn in 0..4 {
                messages.push(super::super::native_setup::native_message(
                    "user",
                    &format!("task {turn}"),
                ));
                messages.push(super::super::native_setup::native_message(
                    "assistant",
                    &"old result ".repeat(600),
                ));
            }
            let conversation =
                Conversation::from_request(json::Value::Object(fields), Limits::default()).unwrap();
            app.journal
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .save(&conversation)
                .unwrap();
            let history = conversation
                .request(Limits::default().wire)
                .to_json()
                .unwrap();
            app.native = Some(conversation);
            app.native_model = Some(
                transport::NativeModel::new(&endpoint)
                    .unwrap()
                    .with_generation_options(app.config.generation),
            );
            let journal_path = path.join(".vade/sessions/count.journal");
            let journal = std::fs::read(&journal_path).unwrap();
            let records = app.journal.as_ref().unwrap().lock().unwrap().record_count();
            let start = Instant::now();
            app.line_input(if compact { "/compact" } else { "new prompt" }.into())
                .unwrap();
            assert!(
                start.elapsed() < Duration::from_secs(1),
                "UI waited for admission"
            );
            ready.recv_timeout(Duration::from_secs(2)).unwrap();
            // Manual /compact asks the model for a summary: it shows as compacting.
            assert_eq!(app.pending.as_ref().unwrap().rx.state(), if compact { RunState::Compacting } else { RunState::Counting });
            assert_eq!(RunState::Counting.label(), "counting tokens");
            app.key(Key::Text("draft while counting".into())).unwrap();
            assert_eq!(app.input.text(), "draft while counting");
            let cancel_start = Instant::now();
            if interrupt {
                app.key(Key::Interrupt).unwrap();
            } else if case < 4 {
                app.line_input("/cancel".into()).unwrap();
            }
            let wait = Duration::from_secs(if case == 5 { 3 } else { 1 });
            while app.pending.is_some() && cancel_start.elapsed() < wait {
                app.poll_worker().unwrap();
                std::thread::park_timeout(Duration::from_millis(5));
            }
            assert!(
                app.pending.is_none(),
                "cancel did not clean up within a second"
            );
            assert!(cancel_start.elapsed() < wait);
            assert_eq!(app.last_state, Some(if case >= 4 { RunState::Failed } else { RunState::LocalCancelled })); if case == 4 { assert_eq!(app.input.text(), "draft while counting"); assert_eq!(app.input.history(app.input.history_len() - 1), Some("new prompt")); }
            assert_eq!(
                app.native
                    .as_ref()
                    .unwrap()
                    .request(Limits::default().wire)
                    .to_json()
                    .unwrap(),
                history
            );
            assert!(app.native_model.is_some());
            assert_eq!(
                app.journal.as_ref().unwrap().lock().unwrap().record_count(),
                records
            );
            assert_eq!(std::fs::read(&journal_path).unwrap(), journal);
            upstream.join().unwrap();
        }
    }
}
