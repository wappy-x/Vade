use super::*;
use super::recovery::*;

/// Run one user request to completion: model turns, tool calls and their
/// results, within `limits` and until `cancel` is set. [`run_observed`] with no
/// observer.
pub fn run(
    conversation: &mut Conversation,
    model: &mut dyn Model,
    executor: &mut dyn ToolExecutor,
    limits: Limits,
    cancel: &AtomicBool,
    on_text: &mut dyn FnMut(&str),
) -> RunReport {
    run_observed(
        conversation,
        model,
        executor,
        limits,
        cancel,
        on_text,
        &mut NoObserver,
    )
}

fn appended_context_stop(
    conversation: &mut Conversation,
    staged: &[Message],
    observer: &mut dyn RunObserver,
    allow_compaction: bool,
    model: &mut dyn Model,
    limits: Limits,
    cancel: &AtomicBool,
) -> Option<StopReason> {
    // Count a closed request containing all proposed calls and reserved results;
    // byte/receipt limits remain independent of the negotiated token count.
    let messages: Vec<_> = conversation
        .messages
        .iter()
        .chain(staged)
        .cloned()
        .collect();
    let request = ModelRequest {
        messages: &messages,
        ..conversation.request(limits.wire)
    };
    let count = model.count_input(request, cancel);
    let reserve = model.output_reserve_tokens(limits.output_reserve_tokens);
    let reply_cap = model.reply_limit_tokens(limits.output_reserve_tokens);
    let counted = count.and_then(|input| {
        crate::context::Usage::with_appended(conversation, staged).with_count(&input)
    });
    if allow_compaction && let Ok(usage) = &counted
        && usage.near_limit(limits.max_context_tokens, reserve)
    {
        let counted_limits = Limits { output_reserve_tokens: reserve, ..limits };
        let mut count_candidate = |c: &Conversation| {
            let messages: Vec<_> = c.messages.iter().chain(staged).cloned().collect();
            let request = ModelRequest { messages: &messages, ..c.request(limits.wire) };
            model.count_input(request, cancel).and_then(|n| crate::context::Usage::with_appended(c, staged).with_count(&n))
        };
        match observer.before_counted_request(conversation, counted_limits, cancel, usage, &mut count_candidate) {
            Ok(true) => {
                model.prompt_rewritten();
                return appended_context_stop(conversation, staged, observer, false, model, limits, cancel);
            }
            Err(_) => return Some(if cancelled(cancel) { StopReason::Cancelled } else { StopReason::JournalError }),
            Ok(false) => (),
        }
    }
    match counted {
        Ok(usage) if usage.fits_with_reply_cap(limits.max_context_tokens, reserve, reply_cap) => None,
        Ok(_) => Some(StopReason::Limit(LimitKind::Context)),
        Err(_) if cancelled(cancel) => Some(StopReason::Cancelled),
        Err(error) => Some(StopReason::ModelError(error.kind())),
    }
}

/// [`run`], reporting every committed message to `observer` (for example a
/// journal) and timing the request with its own [`RunBudget`].
pub fn run_observed(
    conversation: &mut Conversation,
    model: &mut dyn Model,
    executor: &mut dyn ToolExecutor,
    limits: Limits,
    cancel: &AtomicBool,
    on_text: &mut dyn FnMut(&str),
    observer: &mut dyn RunObserver,
) -> RunReport {
    let budget = RunBudget::new(limits.max_run_time);
    run_observed_with_budget(
        conversation,
        model,
        executor,
        limits,
        cancel,
        on_text,
        observer,
        &budget,
    )
}

/// Use a host-owned clock so approval callbacks can pause this same request.
/// The watchdog signals the adapter's existing cancellation flag; adapters must
/// return cleanly on cancellation. A tool's closed receipts are persisted before
/// timed cancellation is reported, just as with user cancellation.
pub fn run_observed_with_budget(
    conversation: &mut Conversation,
    model: &mut dyn Model,
    executor: &mut dyn ToolExecutor,
    limits: Limits,
    cancel: &AtomicBool,
    on_text: &mut dyn FnMut(&str),
    observer: &mut dyn RunObserver,
    budget: &RunBudget,
) -> RunReport {
    let report = budget.with_cancellation(cancel, || {
        run_observed_inner(
            conversation,
            model,
            executor,
            limits,
            cancel,
            on_text,
            observer,
            budget,
        )
    });
    if budget.expired()
        && matches!(
            report.reason,
            StopReason::Cancelled | StopReason::Finished(_)
        )
    {
        report.stopped(StopReason::Limit(LimitKind::Time))
    } else {
        report
    }
}

fn run_observed_inner(
    conversation: &mut Conversation,
    model: &mut dyn Model,
    executor: &mut dyn ToolExecutor,
    limits: Limits,
    cancel: &AtomicBool,
    on_text: &mut dyn FnMut(&str),
    observer: &mut dyn RunObserver,
    budget: &RunBudget,
) -> RunReport {
    let mut report = RunReport {
        reason: StopReason::InvalidConfiguration,
        answer_cut_off: false,
        model_turns: 0,
        auto_continues: 0,
        nudges: 0,
        auto_checks: 0,
        tool_turns: 0,
        tool_calls: 0,
        output_bytes: 0,
    };
    if limits.validate().is_err() {
        return report;
    }
    if conversation.bytes > limits.max_history_bytes {
        return report.stopped(StopReason::Limit(LimitKind::History));
    }
    let mut proposed_calls = 0usize;
    let mut consecutive_continues = 0usize;
    let mut empty_nudges = 0usize;
    let mut check_nudges = 0usize;
    let mut reminded_check = None;
    let mut patch_recovery_sent = conversation.messages.iter().any(|message| {
        matches!(message.role(), "user" | "developer")
            && message.value().get("content").and_then(Value::as_str) == Some(PATCH_RECOVERY_PROMPT)
    });
    // A prior activation ended on a cutoff with no tool round since its last
    // recovery prompt: restate the size ceiling before the first model call
    // so a resumed task does not re-emit the same oversized call. The attempt
    // count comes from journaled prompts, so the per-task bound holds.
    if let Some((hint, discarded)) = pending_truncated(conversation) {
        let attempt = truncated_attempts(conversation, &hint);
        if attempt < MAX_TRUNCATED_RETRIES {
            let prompt = truncated_recovery_prompt(
                &hint,
                model.reply_limit_tokens(limits.output_reserve_tokens),
                discarded,
                attempt,
            );
            if let Err(reason) =
                push_harness_message(conversation, observer, limits, "developer", &prompt)
            {
                return report.stopped(reason);
            }
            report.nudges += 1;
        }
    }
    loop {
        if cancelled(cancel) {
            return report.stopped(StopReason::Cancelled);
        }
        if report.model_turns >= limits.max_model_turns {
            return report.stopped(StopReason::Limit(LimitKind::ModelTurns));
        }
        // Leave a small repair margin; a one-second resumed activation cannot
        // complete a useful model/tool round. Unlimited requests stay unlimited.
        if budget
            .remaining()
            .is_some_and(|left| left < Duration::from_secs(10))
        {
            return report.stopped(StopReason::Limit(LimitKind::Time));
        }
        // A tool-using model may never offer a final answer. Put the repair
        // priority after the closed check, before counting the next request.
        // Derive it from journaled receipts so a fresh activation retains it.
        if executor.check_enabled()
            && let Some((id, prompt)) = check_repair_prompt(conversation, reminded_check.as_deref())
        {
            if let Err(reason) = push_harness_prompt(conversation, observer, limits, &prompt) {
                return report.stopped(reason);
            }
            reminded_check = Some(id);
            report.nudges += 1;
        }
        // Schedule the single self-review while time provably remains
        // instead of only at a final answer that may arrive too late.
        if executor.check_enabled()
            && !executor.check_failing()
            && scheduled_review_due(conversation, budget)
        {
            if let Err(reason) =
                push_harness_prompt(conversation, observer, limits, SELF_REVIEW_PROMPT)
            {
                return report.stopped(reason);
            }
            report.nudges += 1;
        }
        let mut usage = match model
            .count_input(conversation.request(limits.wire), cancel)
            .and_then(|input| {
                crate::context::Usage::for_conversation(conversation).with_count(&input)
            }) {
            Ok(usage) => usage,
            Err(error) => {
                return report.stopped(if cancelled(cancel) {
                    StopReason::Cancelled
                } else {
                    StopReason::ModelError(error.kind())
                });
            }
        };
        let counted_limits = Limits {
            output_reserve_tokens: model.output_reserve_tokens(limits.output_reserve_tokens),
            ..limits
        };
        let mut count_error = None;
        let mut count = |c: &Conversation| {
            let result = model
                .count_input(c.request(limits.wire), cancel)
                .and_then(|input| crate::context::Usage::for_conversation(c).with_count(&input));
            if let Err(error) = &result {
                count_error = Some(error.kind());
            }
            result
        };
        let rewrite = observer.before_counted_request(
            conversation,
            counted_limits,
            cancel,
            &usage,
            &mut count,
        );
        drop(count);
        let rewritten = match rewrite {
            Ok(rewritten) => rewritten,
            Err(_) => {
                return report.stopped(if cancelled(cancel) {
                    StopReason::Cancelled
                } else {
                    count_error.map_or(StopReason::JournalError, StopReason::ModelError)
                });
            }
        };
        if rewritten {
            model.prompt_rewritten();
            usage = match model
                .count_input(conversation.request(limits.wire), cancel)
                .and_then(|input| {
                    crate::context::Usage::for_conversation(conversation).with_count(&input)
                }) {
                Ok(usage) => usage,
                Err(error) => {
                    return report.stopped(if cancelled(cancel) {
                        StopReason::Cancelled
                    } else {
                        StopReason::ModelError(error.kind())
                    });
                }
            };
        }
        if !usage.fits_with_reply_cap(
            limits.max_context_tokens,
            model.output_reserve_tokens(limits.output_reserve_tokens),
            model.reply_limit_tokens(limits.output_reserve_tokens),
        ) {
            return report.stopped(StopReason::Limit(LimitKind::Context));
        }
        observer.progress(
            report.tool_turns,
            limits,
            usage.exact.map(|(p, _, c, _)| (p as u64, c as u64)),
            cancel,
        );
        // Idle, opt-in manual warming belongs to the host. Automatic compaction
        // is immediately followed by real generation of this same prompt.
        if budget.remaining().is_some_and(|left| left < Duration::from_secs(10)) {
            return report.stopped(StopReason::Limit(LimitKind::Time));
        }
        model.set_work_remaining(budget.remaining());
        report.model_turns += 1;
        let mut emitted_bytes = 0usize;
        // Remember callback failure even if a buggy adapter ignores its result.
        let mut callback_failed = false;
        let turn = model.complete(conversation.request(limits.wire), cancel, &mut |text| {
            if cancelled(cancel) {
                callback_failed = true;
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "agent cancelled",
                ));
            }
            emitted_bytes = emitted_bytes.saturating_add(text.len());
            if emitted_bytes > limits.wire.max_message_bytes || callback_failed {
                callback_failed = true;
                return Err(invalid("assistant text exceeds byte limit"));
            }
            on_text(text);
            Ok(())
        });
        if cancelled(cancel) {
            return report.stopped(StopReason::Cancelled);
        }
        if callback_failed {
            return report.stopped(StopReason::InvalidResponse);
        }
        let turn = match turn {
            Ok(turn) => turn,
            // A structured upstream cancellation is also terminal when the
            // local cancel flag is unset; it must never enter auto-continue.
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                return report.stopped(StopReason::Cancelled);
            }
            Err(error) => return report.stopped(StopReason::ModelError(error.kind())),
        };
        if let Some(context) = model.context_usage() {
            observer.progress(report.tool_turns, limits, Some(context), cancel);
        }
        let truncated_tool_call = turn.truncated_tool_call();
        let reply_budget_exhausted = turn.reply_budget_exhausted();
        let discarded_bytes = turn.discarded_bytes();
        let truncated_step = turn.truncated_tool_hint().unwrap_or("unknown tool/target").to_owned();
        let (raw_message, finish) = turn.into_parts();
        // A model implementation may have constructed the turn with larger
        // limits. Revalidate at the orchestration boundary before any tools run.
        let message = match if finish == FinishReason::Length {
            raw_message.truncated(limits.wire)
        } else {
            Message::from_value(raw_message.into_value(), limits.wire)
        } {
            Ok(message) => message,
            Err(_) => return report.stopped(StopReason::InvalidResponse),
        };
        if message
            .tool_calls()
            .iter()
            .any(|call| conversation.call_ids.contains(&call.id))
        {
            return report.stopped(StopReason::InvalidResponse);
        }
        if let Some(hints) = model.checkpoint_hints() {
            conversation.kv_hints = hints
                .iter()
                .take(crate::kv::MAX_HINTS)
                .filter(|h| h.messages <= conversation.messages.len())
                .cloned()
                .collect();
        }
        if message.tool_calls().is_empty() {
            if !conversation.fits(std::slice::from_ref(&message), limits) {
                return report.stopped(StopReason::Limit(LimitKind::History));
            }
            if observer
                .checkpoint_reply(conversation, &message, &finish)
                .is_err()
            {
                return report.stopped(StopReason::JournalError);
            }
            conversation.commit(vec![message]);
            report.answer_cut_off = finish == FinishReason::Length && !truncated_tool_call;
            if truncated_tool_call {
                // Attempts are counted from journaled recovery prompts for
                // this hint, so the per-task bound survives resume and a
                // smaller patch to the same file still escalates.
                let attempt = truncated_attempts(conversation, &truncated_step);
                if attempt >= MAX_TRUNCATED_RETRIES {
                    return report.stopped(StopReason::TruncatedToolCall);
                }
                if report.model_turns >= limits.max_model_turns {
                    return report.stopped(StopReason::Limit(LimitKind::ModelTurns));
                }
                let prompt = truncated_recovery_prompt(
                    &truncated_step,
                    model.reply_limit_tokens(limits.output_reserve_tokens),
                    discarded_bytes,
                    attempt,
                );
                if let Err(reason) =
                    push_harness_message(conversation, observer, limits, "developer", &prompt)
                {
                    return report.stopped(reason);
                }
                report.nudges += 1;
                continue;
            }
            if finish == FinishReason::Length && limits.auto_continue {
                if consecutive_continues >= MAX_AUTO_CONTINUES {
                    return report.stopped(StopReason::Limit(LimitKind::AutoContinues));
                }
                if report.model_turns >= limits.max_model_turns {
                    return report.stopped(StopReason::Limit(LimitKind::ModelTurns));
                }
                let prompt = if reply_budget_exhausted {
                    format!("{REPLY_BUDGET_PROMPT}\n{}", task_state_prompt(conversation, budget))
                } else if conversation.tool_names.is_empty() {
                    CONTINUATION_PROMPT.to_string()
                } else {
                    format!("{TOOL_CONTINUATION_PROMPT}\n{}", task_state_prompt(conversation, budget))
                };
                if let Err(reason) = push_harness_prompt(conversation, observer, limits, &prompt) {
                    return report.stopped(reason);
                }
                report.auto_continues += 1;
                consecutive_continues += 1;
                observer.auto_continued(consecutive_continues);
                // The next loop uses the same clock/counters and the existing
                // counted compaction hook before generating any continuation.
                continue;
            }
            // Never end silently, and never accept a final answer while the
            // automatic check fails without reminding the model. Both nudges use
            // the same request budget; exhausted turns keep the old outcome.
            if finish == FinishReason::Stop {
                if !conversation.messages.last().is_some_and(has_visible_answer) {
                    if empty_nudges >= MAX_EMPTY_NUDGES {
                        return report.stopped(StopReason::Limit(LimitKind::EmptyReplies));
                    }
                    if report.model_turns >= limits.max_model_turns {
                        return report.stopped(StopReason::Limit(LimitKind::ModelTurns));
                    }
                    if let Err(reason) = push_harness_prompt(conversation, observer, limits, EMPTY_REPLY_PROMPT) {
                        return report.stopped(reason);
                    }
                    empty_nudges += 1;
                    report.nudges += 1;
                    continue;
                }
                if conversation.tool_names.contains("workspace_shell")
                    && executor.check_enabled()
                    && latest_check_receipt(conversation).is_none_or(|(_, receipt)| {
                        receipt.lines().next() != Some("status: Exited(Some(0))")
                    })
                    && !crate::context::has_unknown_outcome(conversation)
                    && let Some(arguments) = executor.take_completion_check()
                {
                    if report.tool_calls >= limits.max_tool_calls {
                        return report.stopped(StopReason::Limit(LimitKind::ToolCalls));
                    }
                    if let Some(reason) = auto_check(
                        conversation,
                        model,
                        executor,
                        observer,
                        limits,
                        cancel,
                        &mut report,
                        &arguments,
                    ) {
                        return report.stopped(reason);
                    }
                    if report.model_turns == limits.max_model_turns {
                        return report.stopped(if executor.check_failing() {
                            StopReason::Limit(LimitKind::ModelTurns)
                        } else if crate::submission::prompt(conversation, executor).is_some() {
                            StopReason::IncompleteSubmission
                        } else {
                            StopReason::Finished(finish)
                        });
                    }
                    if executor.check_failing() {
                        continue;
                    }
                }
            }
            if finish == FinishReason::Stop
                && !crate::context::has_unknown_outcome(conversation)
                && let Some((prompt, reminded)) = crate::submission::prompt(conversation, executor)
            {
                if reminded || report.model_turns >= limits.max_model_turns { return report.stopped(StopReason::IncompleteSubmission); }
                if let Err(reason) = push_harness_message(conversation, observer, limits, "developer", &prompt) { return report.stopped(reason); }
                report.nudges += 1;
                continue;
            }
            if finish == FinishReason::Stop && report.model_turns < limits.max_model_turns {
                let prompt = if executor.check_failing() && check_nudges < MAX_CHECK_NUDGES {
                    check_nudges += 1;
                    Some(CHECK_FAILING_PROMPT)
                } else if executor.check_enabled()
                    && !executor.check_failing()
                    && completion_review_due(conversation, budget)
                {
                    Some(SELF_REVIEW_PROMPT)
                } else {
                    None
                };
                if let Some(prompt) = prompt {
                    if let Err(reason) = push_harness_prompt(conversation, observer, limits, prompt)
                    {
                        return report.stopped(reason);
                    }
                    report.nudges += 1;
                    continue;
                }
            }
            return report.stopped(StopReason::Finished(finish));
        }
        consecutive_continues = 0;
        empty_nudges = 0;
        let calls = message.tool_calls().to_vec();
        // Validate original argument text below, but persist only parseable
        // argument strings so a failed call cannot break the next API request.
        let message = match message.with_valid_argument_json(limits.wire) {
            Ok(message) => message,
            Err(_) => return report.stopped(StopReason::InvalidResponse),
        };
        let mut stop = if finish != FinishReason::ToolCalls {
            Some(StopReason::Finished(finish))
        } else if report.tool_turns >= limits.max_tool_turns {
            Some(StopReason::Limit(LimitKind::ToolTurns))
        } else if proposed_calls.saturating_add(calls.len()) > limits.max_tool_calls {
            Some(StopReason::Limit(LimitKind::ToolCalls))
        } else {
            None
        };
        proposed_calls = proposed_calls.saturating_add(calls.len());

        // Reserve valid small results for every ID before invoking anything.
        // This guarantees cancellation/failure cannot leave dangling calls.
        let mut staged = vec![message];
        let mut contents = Vec::new();
        let mut patch_recovery_pending = false;
        let mut tool_recovery_pending = None;
        for call in &calls {
            let content = error_content("not_executed");
            let result = match Message::tool_result(&call.id, content.clone(), limits.wire) {
                Ok(result) => result,
                Err(_) => return report.stopped(StopReason::InvalidConfiguration),
            };
            staged.push(result);
            contents.push(content);
        }
        let mut staged_output: usize = contents.iter().map(String::len).sum();
        if report.output_bytes.saturating_add(staged_output) > limits.max_total_output_bytes {
            return report.stopped(StopReason::Limit(LimitKind::ToolOutput));
        }
        if !conversation.fits(&staged, limits) {
            return report.stopped(StopReason::Limit(LimitKind::History));
        }
        if stop.is_none() {
            stop = appended_context_stop(conversation, &staged, observer, true, model, limits, cancel);
        }
        if stop.is_none() {
            report.tool_turns += 1;
            observer.progress(report.tool_turns, limits, model.context_usage(), cancel);
        }
        for (index, call) in calls.iter().enumerate() {
            if index > 0 && stop.is_none() {
                stop = appended_context_stop(conversation, &staged, observer, false, model, limits, cancel);
            }
            if cancelled(cancel) {
                stop = Some(StopReason::Cancelled);
            }
            let mut content = match &stop {
                Some(StopReason::Cancelled) => error_content("cancelled"),
                Some(_) => error_content("not_executed"),
                None => {
                    let available =
                        limits.max_total_output_bytes - report.output_bytes - staged_output
                            + contents[index].len();
                    let receipt_capacity = conversation.receipt_capacity(&staged, index, limits);
                    let output_limit = limits
                        .max_tool_output_bytes
                        .min(available)
                        .min(receipt_capacity);
                    let mut uncertain = false;
                    let result = if receipt_capacity < 128 {
                        // Even the minimum useful receipt cannot be retained.
                        // Refuse before executing rather than losing evidence
                        // after a partial write. Generated errors remain reserved.
                        stop = Some(StopReason::Limit(LimitKind::History));
                        Err(ToolError::OutputLimit)
                    } else if !conversation.tool_names.contains(&call.name) {
                        Err(ToolError::UnknownTool)
                    } else {
                        match json::parse(&call.arguments) {
                            Ok(arguments @ Value::Object(_)) => {
                                // Persist a closed history before crossing the side-effect boundary.
                                // A crash here must never imply the tool was safe to replay.
                                staged[index + 1] = Message::tool_result(
                                    &call.id,
                                    error_content("unknown_outcome"),
                                    limits.wire,
                                )
                                .expect("reserved result");
                                if observer.checkpoint(conversation, &staged).is_err() {
                                    conversation.commit(staged);
                                    report.answer_cut_off = false;
                                    return report.stopped(StopReason::JournalError);
                                }
                                report.tool_calls += 1;
                                let result = executor.execute(
                                    ToolInvocation {
                                        call,
                                        arguments: &arguments,
                                        max_output_bytes: output_limit,
                                    },
                                    cancel,
                                );
                                uncertain = executor.last_side_effect_uncertain();
                                result
                            }
                            _ => Err(ToolError::InvalidArguments("arguments must be a JSON object")),
                        }
                    };
                    let output = match result {
                        Ok(output) if output.len() <= output_limit => output,
                        Ok(_) | Err(ToolError::OutputLimit) => {
                            if stop.is_none() {
                                stop = Some(StopReason::Limit(LimitKind::ToolOutput));
                            }
                            error_content("output_limit")
                        }
                        Err(ToolError::Cancelled) => {
                            stop = Some(StopReason::Cancelled);
                            error_content("cancelled")
                        }
                        Err(ToolError::Failed) => {
                            stop = Some(StopReason::ToolFailed);
                            error_content("tool_failed")
                        }
                        Err(ToolError::UnknownTool) => error_content("unknown_tool"),
                        Err(ToolError::Denied) => error_content("denied"),
                        Err(ToolError::InvalidArguments(reason)) => invalid_arguments_content(reason, output_limit),
                    };
                    if uncertain {
                        stop = Some(StopReason::ToolFailed);
                        uncertain_content(&output, output_limit)
                    } else {
                        output
                    }
                }
            };
            let mut replacement = Message::tool_result(&call.id, content.clone(), limits.wire);
            if replacement.as_ref().map_or(true, |result| {
                conversation
                    .bytes_after(&staged)
                    .saturating_sub(staged[index + 1].serialized_bytes())
                    .saturating_add(result.serialized_bytes())
                    > limits.max_history_bytes
            }) {
                stop = Some(StopReason::Limit(LimitKind::History));
                // Even history exhaustion must preserve the effect marker.
                content = if crate::context::has_unknown_outcome_content(&content) {
                    error_content("unknown_outcome")
                } else {
                    error_content("output_limit")
                };
                replacement = Message::tool_result(&call.id, content.clone(), limits.wire);
            }
            // Detailed argument errors fit output_limit; other errors fit the reservation.
            staged_output = staged_output - contents[index].len() + content.len();
            contents[index] = content;
            staged[index + 1] =
                replacement.expect("bounded error result fits validated wire limits");
            if stop.is_none() && call.name != "apply_patch"
                && refusal_reason(&contents[index]).is_some()
            {
                let repeats = identical_tool_refusals(conversation, &staged[..index + 2], call);
                if repeats >= 3 {
                    stop = Some(StopReason::ToolFailed);
                } else if repeats == 2 {
                    tool_recovery_pending = Some(tool_recovery_prompt(call, &contents[index]));
                }
            }
            if stop.is_none()
                && call.name == "apply_patch"
                && contents[index].starts_with("Patch refused: ")
                && identical_patch_refusals(conversation, &staged, &call.arguments) >= 3
            {
                if patch_recovery_sent {
                    stop = Some(StopReason::ToolFailed);
                } else {
                    patch_recovery_pending = true;
                }
            }
            if observer.checkpoint(conversation, &staged).is_err() {
                conversation.commit(staged);
                report.answer_cut_off = false;
                return report.stopped(StopReason::JournalError);
            }
        }
        report.output_bytes += staged_output;
        if observer.checkpoint(conversation, &staged).is_err() {
            conversation.commit(staged);
            report.answer_cut_off = false;
            return report.stopped(StopReason::JournalError);
        }
        conversation.commit(staged);
        report.answer_cut_off = false;
        if cancelled(cancel) {
            stop = Some(StopReason::Cancelled);
        }
        if (stop.is_none() || stop == Some(StopReason::ToolFailed))
            && let Some(prompt) = tool_recovery_pending
        {
            if let Err(reason) = push_harness_prompt(conversation, observer, limits, &prompt) {
                return report.stopped(reason);
            }
            report.nudges += 1;
        }
        if let Some(reason) = stop {
            if reason == StopReason::Limit(LimitKind::Context) { continue; }
            return report.stopped(reason);
        }
        let (discovery_rounds, discovery_reminded) = discovery_budget(conversation);
        if discovery_rounds >= 12 || discovery_recovery_due(conversation) {
            if discovery_reminded {
                return report.stopped(StopReason::ToolFailed);
            }
            if let Err(reason) = push_harness_prompt(
                conversation, observer, limits, DISCOVERY_RECOVERY_PROMPT,
            ) {
                return report.stopped(reason);
            }
            report.nudges += 1;
        }
        if patch_recovery_pending {
            if let Err(reason) =
                push_harness_prompt(conversation, observer, limits, PATCH_RECOVERY_PROMPT)
            {
                return report.stopped(reason);
            }
            patch_recovery_sent = true;
            report.nudges += 1;
        }
        if let Some(arguments) = executor.take_pending_check() {
            if let Some(reason) = auto_check(
                conversation,
                model,
                executor,
                observer,
                limits,
                cancel,
                &mut report,
                &arguments,
            ) {
                return report.stopped(reason);
            }
        }
    }
}

/// Durably append a harness-authored continuation or nudge.
pub(super) fn push_harness_prompt(
    conversation: &mut Conversation,
    observer: &mut dyn RunObserver,
    limits: Limits,
    text: &str,
) -> Result<(), StopReason> {
    push_harness_message(conversation, observer, limits, "developer", text)
}

/// Append a harness instruction in place, preserving journal and limit checks.
fn push_harness_message(
    conversation: &mut Conversation,
    observer: &mut dyn RunObserver,
    limits: Limits,
    role: &str,
    text: &str,
) -> Result<(), StopReason> {
    let message = Message::from_value(
        Value::Object(vec![
            ("role".into(), Value::String(role.into())),
            ("content".into(), Value::String(text.into())),
        ]),
        limits.wire,
    )
    .map_err(|_| StopReason::InvalidConfiguration)?;
    if !conversation.fits(std::slice::from_ref(&message), limits) {
        return Err(StopReason::Limit(LimitKind::History));
    }
    if observer
        .checkpoint(conversation, std::slice::from_ref(&message))
        .is_err()
    {
        return Err(StopReason::JournalError);
    }
    conversation.commit(vec![message]);
    Ok(())
}

/// Run the user's check as a harness-authored `workspace_shell` call after a
/// changed round or before unverified completion. Journal it like a model call
/// (unknown outcome before the side effect), using the executor's approval and
/// sandbox. Its
/// bounded receipt reaches the model as an ordinary tool result. No room in
/// context or history stops before dispatch with the corresponding limit.
#[allow(clippy::too_many_arguments)]
fn auto_check(
    conversation: &mut Conversation,
    model: &mut dyn Model,
    executor: &mut dyn ToolExecutor,
    observer: &mut dyn RunObserver,
    limits: Limits,
    cancel: &AtomicBool,
    report: &mut RunReport,
    arguments: &Value,
) -> Option<StopReason> {
    if !conversation.tool_names.contains("workspace_shell") {
        return None;
    }
    let text = json::stringify(arguments).ok()?;
    let mut n = conversation.messages.len();
    let id = loop {
        let id = format!("{CHECK_CALL_PREFIX}{n}");
        if !conversation.call_ids.contains(&id) {
            break id;
        }
        n += 1;
    };
    let call = ToolCall {
        id: id.clone(),
        name: "workspace_shell".into(),
        arguments: text.clone(),
    };
    let string = |s: &str| Value::String(s.into());
    let request = Message::from_value(
        Value::Object(vec![
            ("role".into(), string("assistant")),
            (
                "tool_calls".into(),
                Value::Array(vec![Value::Object(vec![
                    ("id".into(), string(&id)),
                    ("type".into(), string("function")),
                    (
                        "function".into(),
                        Value::Object(vec![
                            ("name".into(), string("workspace_shell")),
                            ("arguments".into(), Value::String(text)),
                        ]),
                    ),
                ])]),
            ),
        ]),
        limits.wire,
    )
    .ok()?;
    let reserved = Message::tool_result(&id, error_content("unknown_outcome"), limits.wire).ok()?;
    let mut staged = vec![request, reserved];
    let available = limits
        .max_total_output_bytes
        .saturating_sub(report.output_bytes);
    let cap = limits
        .max_tool_output_bytes
        .min(available)
        .min(conversation.receipt_capacity(&staged, 0, limits));
    if cap < 512 {
        return Some(StopReason::Limit(LimitKind::ToolOutput));
    }
    if !conversation.fits(&staged, limits) {
        return Some(StopReason::Limit(LimitKind::History));
    }
    if let Some(reason) = appended_context_stop(conversation, &staged, observer, true, model, limits, cancel) {
        return Some(reason);
    }
    if observer.checkpoint(conversation, &staged).is_err() {
        conversation.commit(staged);
        return Some(StopReason::JournalError);
    }
    report.tool_calls += 1;
    report.auto_checks += 1;
    let result = executor.execute(
        ToolInvocation {
            call: &call,
            arguments,
            max_output_bytes: cap,
        },
        cancel,
    );
    // A timeout is an ordinary partial-check receipt; only local I/O failure
    // after spawning or cancellation stops the request.
    let mut stop = None;
    let content = match result {
        Ok(output) if output.len() <= cap => output,
        Ok(_) | Err(ToolError::OutputLimit) => error_content("output_limit"),
        Err(ToolError::Cancelled) => {
            stop = Some(StopReason::Cancelled);
            error_content("cancelled")
        }
        Err(ToolError::Failed) => {
            stop = Some(StopReason::ToolFailed);
            error_content("tool_failed")
        }
        Err(ToolError::Denied) => error_content("denied"),
        Err(_) => error_content("invalid_arguments"),
    };
    if let Ok(result) = Message::tool_result(&id, content.clone(), limits.wire) {
        report.output_bytes += content.len();
        staged[1] = result;
    }
    if observer.checkpoint(conversation, &staged).is_err() {
        conversation.commit(staged);
        return Some(StopReason::JournalError);
    }
    conversation.commit(staged);
    if cancelled(cancel) {
        stop = Some(StopReason::Cancelled);
    }
    stop
}
