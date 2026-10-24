//! Explicit context accounting without pretending to own the model tokenizer.
use crate::agent::{Conversation, Limits};
use crate::json::Value;
use std::io;

mod retention;
use retention::*;

pub const DEFAULT_CONTEXT_TOKENS: usize = 131_072;
pub const DEFAULT_RESERVE: usize = 4096;
pub const COMPACTION_OUTPUT_TOKENS: usize = 8192;

/// The reply room compaction protects: the requested output, at most
/// `COMPACTION_OUTPUT_TOKENS` and a quarter of the context. A whole-context
/// output (8192 of 8192) would otherwise leave no history room on a fresh
/// chat; a quarter keeps three quarters for history and changes nothing at
/// 32768 tokens and above.
pub fn reply_reservation(context: usize, output: usize) -> usize {
    output.min(COMPACTION_OUTPUT_TOKENS).min(context / 4)
}

/// A chat with no assistant or tool turn has nothing compaction may remove:
/// system, tools and user prompts always stay.
pub fn nothing_to_compact(c: &Conversation) -> bool {
    !c.messages().iter().any(|m| matches!(m.role(), "assistant" | "tool"))
}

/// The ordinary local heuristic applied to a complete wire request. This is
/// used only when no negotiated tokenizer admission is available.
pub(crate) fn estimated_request_tokens(request: &str) -> io::Result<usize> {
    let value = crate::json::view(request)?;
    let count = |name| {
        value
            .get(name)
            .and_then(crate::json::View::array)
            .map_or(0, Iterator::count)
    };
    let subtotal = request
        .len()
        .div_ceil(3)
        .saturating_add(256)
        .saturating_add(count("messages").saturating_mul(16))
        .saturating_add(count("tools").saturating_mul(32));
    Ok(subtotal.saturating_add(subtotal.div_ceil(8).max(256)))
}

/// One attempt per threshold crossing, shared by pre-user and in-run checks.
/// A large recent turn can keep usage high after successful compaction; rearm
/// after usage falls to 60%. A failed or cancelled uncommitted plan retries on
/// the next user turn.
#[derive(Debug)]
pub struct AutoCompaction {
    armed: bool,
    retry_next_turn: bool,
}
impl Default for AutoCompaction {
    fn default() -> Self {
        Self {
            armed: true,
            retry_next_turn: false,
        }
    }
}
impl AutoCompaction {
    /// A pre-save oversized candidate must not rely on a rewrite that the
    /// session's successful-compaction hysteresis would suppress.
    pub fn can_attempt_next_turn(&self) -> bool {
        self.armed || self.retry_next_turn
    }
    pub fn begin_turn(&mut self) {
        if self.retry_next_turn {
            self.armed = true;
            self.retry_next_turn = false;
        }
    }
    pub fn plan_failed(&mut self) {
        self.retry_next_turn = true;
    }
    pub fn observe(&mut self, usage: &Usage, limit: usize, reserve: usize) -> bool {
        let limit = usage.context_limit(limit);
        if usage.compaction_required(limit, reserve) <= limit.saturating_mul(3) / 5 {
            self.armed = true;
        }
        if self.armed && usage.near_limit(limit, reserve) {
            self.armed = false;
            return true;
        }
        false
    }
}
/// UTF-8 JSON bytes are measurable; model tokens and hidden provider framing are not.
/// These heuristics intentionally include a margin. They are never an exact count
/// or a guaranteed upper bound for an arbitrary tokenizer/template.
#[derive(Clone, Debug)]
pub struct MessageTokens {
    pub index: usize,
    pub role: String,
    pub estimated: usize,
    pub byte_bound: usize,
}
#[derive(Clone, Debug)]
pub struct Usage {
    /// Counts of the complete rendered request from negotiated token admission.
    /// Cached tokens still occupy this window; never subtract them.
    pub exact: Option<(usize, usize, usize, bool)>,
    pub messages: Vec<MessageTokens>,
    pub options_estimated: usize,
    pub tools_estimated: usize,
    pub framing_estimated: usize,
    pub safety_margin: usize,
    /// Full input estimate, including framing allowance and safety margin.
    pub estimated: usize,
    /// Serialized request bytes, not a token upper bound.
    pub byte_bound: usize,
    pub requested_output: usize,
    pub admission_issue: Option<&'static str>,
}
impl Usage {
    pub fn for_conversation(c: &Conversation) -> Self {
        Self::with_appended(c, &[])
    }
    pub fn with_appended(c: &Conversation, appended: &[crate::agent_wire::Message]) -> Self {
        let message_bytes: usize = c.messages().iter().map(|m| m.serialized_bytes()).sum();
        // Includes every root field, key, separator and array/object delimiter.
        let options_bytes = c.serialized_bytes().saturating_sub(message_bytes);
        let options_estimated = options_bytes.div_ceil(3);
        let tools_bytes = c.serialized_tool_bytes();
        let tool_count = c
            .options()
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, <[Value]>::len);
        let messages: Vec<_> = c
            .messages()
            .iter()
            .chain(appended)
            .enumerate()
            .map(|(index, m)| MessageTokens {
                index,
                role: m.role().into(),
                estimated: m.serialized_bytes().div_ceil(3),
                byte_bound: m.serialized_bytes(),
            })
            .collect();
        let separators = appended
            .len()
            .saturating_sub(usize::from(c.messages().is_empty() && !appended.is_empty()));
        let framing_estimated = 256 + messages.len() * 16 + tool_count * 32;
        let subtotal = options_estimated
            + messages.iter().map(|m| m.estimated).sum::<usize>()
            + framing_estimated
            + separators;
        let safety_margin = subtotal.div_ceil(8).max(256);
        let byte_bound = c.serialized_bytes()
            + separators
            + appended.iter().map(|m| m.serialized_bytes()).sum::<usize>();
        let mut admission_issue = None;
        for m in c.messages().iter().chain(appended) {
            if let Some(content) = m.value().get("content") {
                let text_only = match content {
                    Value::Null | Value::String(_) => true,
                    Value::Array(parts) => parts.iter().all(|p| {
                        matches!(
                            p.get("type").and_then(Value::as_str),
                            Some("text" | "input_text" | "output_text")
                        ) && p.get("text").and_then(Value::as_str).is_some()
                    }),
                    _ => false,
                };
                if !text_only {
                    admission_issue =
                        Some("non-text content requires provider token admission; unavailable");
                }
            }
            if ["audio", "attachments", "images"]
                .iter()
                .any(|k| m.value().get(k).is_some_and(|v| *v != Value::Null))
            {
                admission_issue =
                    Some("media/attachments require provider token admission; unavailable");
            }
        }
        if ["audio", "attachments", "images"]
            .iter()
            .any(|k| c.options().get(k).is_some_and(|v| *v != Value::Null))
        {
            admission_issue =
                Some("media/attachments require provider token admission; unavailable");
        }
        let mut requested_output = 0;
        for key in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
            if let Some(value) = c.options().get(key) {
                match value {
                    Value::Number(n) => match n.parse::<usize>() {
                        Ok(n) if n > 0 => requested_output = requested_output.max(n),
                        _ => admission_issue = Some("invalid provider output limit"),
                    },
                    _ => admission_issue = Some("invalid provider output limit"),
                }
            }
        }
        Self {
            exact: None,
            messages,
            options_estimated,
            tools_estimated: tools_bytes.div_ceil(3),
            framing_estimated,
            safety_margin,
            estimated: subtotal + safety_margin,
            byte_bound,
            requested_output,
            admission_issue,
        }
    }
    pub fn required(&self, reserve: usize) -> usize {
        if let Some((input, output, _, _)) = self.exact {
            return input.saturating_add(output);
        }
        self.estimated
            .saturating_add(reserve.max(self.requested_output))
    }
    pub fn near_limit(&self, limit: usize, reserve: usize) -> bool {
        self.compaction_required(limit, reserve) >= self.context_limit(limit).saturating_mul(4) / 5
    }
    pub fn compaction_required(&self, limit: usize, reserve: usize) -> usize {
        let (input, output) = self.exact.map_or(
            (self.estimated, reserve.max(self.requested_output)),
            |(input, output, _, _)| (input, output),
        );
        input.saturating_add(reply_reservation(self.context_limit(limit), output))
    }
    pub fn fits(&self, limit: usize, reserve: usize) -> bool {
        self.fits_with_reply_cap(limit, reserve, COMPACTION_OUTPUT_TOKENS)
    }
    /// Like `fits`, but the exact-admission reply floor never exceeds the
    /// adapter's actual reply cap: a native reply (at most
    /// `contract::NATIVE_REPLY_MAX_TOKENS`) must not demand an 8k floor.
    pub fn fits_with_reply_cap(&self, limit: usize, reserve: usize, reply_cap: usize) -> bool {
        if let Some((_, output, _, admitted)) = self.exact {
            let floor = reply_reservation(self.context_limit(limit), reply_cap.max(1));
            let reply_fits = reserve < COMPACTION_OUTPUT_TOKENS || output >= floor;
            return admitted && reply_fits && self.required(reserve) <= self.context_limit(limit);
        }
        self.admission_issue.is_none() && self.required(reserve) <= limit
    }
    pub fn context_limit(&self, local: usize) -> usize {
        self.exact
            .map_or(local, |(_, _, remote, _)| local.min(remote))
    }
    pub fn with_count(mut self, count: &crate::contract::InputCount) -> io::Result<Self> {
        if let crate::contract::InputCount::Exact {
            prompt_tokens,
            output_tokens,
            context_limit,
            admitted,
        } = *count
        {
            let n = |n| usize::try_from(n).map_err(|_| io::Error::other("token count overflow"));
            self.exact = Some((
                n(prompt_tokens)?,
                n(output_tokens)?,
                n(context_limit)?,
                admitted,
            ));
        }
        Ok(self)
    }
    pub fn summary(&self, limit: usize, reserve: usize) -> String {
        if let Some((input, output, _, admitted)) = self.exact {
            return format!(
                "context {input} exact input tokens + {output} output reserve / {} budget · {} request bytes · upstream tokenizer admission {}",
                self.context_limit(limit),
                self.byte_bound,
                if admitted { "accepted" } else { "refused" }
            );
        }
        format!(
            "context ≈{} input tokens (includes {} framing + {} safety margin) + {} output reserve / {} budget · {} request bytes · heuristic only; tokenizer/provider framing unknown{}",
            self.estimated,
            self.framing_estimated,
            self.safety_margin,
            reserve.max(self.requested_output),
            limit,
            self.byte_bound,
            self.admission_issue
                .map_or(String::new(), |s| format!(" · admission refused: {s}"))
        )
    }
}

pub const KV_COMPACTION_NOTICE: &str = "System/tools wire bytes stay exact; the summary starts immediately after that prefix. Changing history invalidates all later causal KV, including retained tail text. Engine-reported cached tokens determine actual reuse; byte equality alone does not prove a hit.";

/// A reviewable derived view, never a replacement for raw journal records.
/// Preserve all user text rather than guessing which sentences are constraints.
#[derive(Clone, Debug)]
pub struct CompactionPlan {
    conversation: Conversation,
    summary: String,
    before_bytes: usize,
    before_estimated: usize,
    pub(crate) source_messages: usize,
    pub(crate) source_request: String,
    pub(crate) limits: Limits,
    recent_start: usize,
    unchanged_prefix_messages: usize,
    retained_indices: Vec<usize>,
    excerpted_indices: Vec<usize>,
    removed_indices: Vec<usize>,
    automatic: bool,
    in_task: bool,
    exact_counts: Option<(usize, usize)>,
}
impl CompactionPlan {
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }
    pub fn into_conversation(self) -> Conversation {
        self.conversation
    }
    pub fn record_counts(&mut self, before: &Usage, after: &Usage) {
        self.exact_counts = before.exact.zip(after.exact).map(|(a, b)| (a.0, b.0));
    }
    pub fn auto_notice(&self) -> String {
        format!(
            "Auto-compacted older results: ≈{} → ≈{} input tokens; reply reservation fits; user rules and stable prefix kept; oversized reasoning/prose and refused edits may be excerpted. Raw journal retained.",
            self.before_estimated,
            Usage::for_conversation(&self.conversation).estimated
        )
    }
    pub fn preview(&self) -> String {
        format!(
            "Compaction preview: {} → {} request bytes; ≈{} → ≈{} input tokens. Remove {} older {} messages; retain {} exact messages, {} and recent turns starting at message {}. Unchanged prefix: {} messages.\n{}\n{} (lossy data; semantic completeness is not verified):\n{}",
            self.before_bytes,
            self.conversation.serialized_bytes(),
            self.before_estimated,
            Usage::for_conversation(&self.conversation).estimated,
            self.removed_indices.len(),
            if self.automatic {
                "results"
            } else {
                "assistant prose"
            },
            self.retained_indices.len(),
            if self.automatic {
                "original user constraints and instructions"
            } else {
                "all user constraints, instructions, tool pairs/receipts, latest answer,"
            },
            self.recent_start,
            self.unchanged_prefix_messages,
            KV_COMPACTION_NOTICE,
            if self.automatic {
                "Local excerpt summary"
            } else {
                "Exact user-supplied summary"
            },
            self.summary
        )
    }
    pub(crate) fn metadata(&self, generation: u64, source_record: u64) -> Value {
        let n = |value: usize| Value::Number(value.to_string());
        Value::Object(vec![
            (
                "exact_input_tokens".into(),
                self.exact_counts.map_or(Value::Null, |(before, after)| {
                    Value::Array(vec![n(before), n(after)])
                }),
            ),
            ("generation".into(), Value::Number(generation.to_string())),
            (
                "source_record".into(),
                Value::Number(source_record.to_string()),
            ),
            ("source_record_start".into(), Value::Number("0".into())),
            ("source_messages".into(), n(self.source_messages)),
            ("before_bytes".into(), n(self.before_bytes)),
            (
                "after_bytes".into(),
                n(self.conversation.serialized_bytes()),
            ),
            ("recent_start".into(), n(self.recent_start)),
            (
                "unchanged_prefix_messages".into(),
                n(self.unchanged_prefix_messages),
            ),
            (
                "retained_message_indices".into(),
                Value::Array(self.retained_indices.iter().copied().map(n).collect()),
            ),
            ("excerpted_message_indices".into(), Value::Array(self.excerpted_indices.iter().copied().map(n).collect())),
            (
                "removed_message_indices".into(),
                Value::Array(self.removed_indices.iter().copied().map(n).collect()),
            ),
            ("summary".into(), Value::String(self.summary.clone())),
            ("in_task".into(), Value::Bool(self.in_task)),
            (
                "summary_origin".into(),
                Value::String(
                    if self.automatic {
                        "local excerpts"
                    } else {
                        "user"
                    }
                    .into(),
                ),
            ),
            (
                "cache_status".into(),
                Value::String(KV_COMPACTION_NOTICE.into()),
            ),
        ])
    }
}

pub fn has_unknown_outcome(c: &Conversation) -> bool {
    c.messages().iter().any(unknown_tool_outcome)
}
fn unknown_tool_outcome(message: &crate::agent_wire::Message) -> bool {
    message.role() == "tool"
        && message
            .value()
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(has_unknown_outcome_content)
}
pub(crate) fn has_unknown_outcome_content(content: &str) -> bool {
    fn partial_command(text: &str) -> bool {
        [
            "status: Timeout",
            "status: Cancelled",
            "status: BuildLimit",
            "status: Exited(None)",
        ]
        .iter()
        .any(|prefix| text.starts_with(prefix))
    }
    fn uncertain(value: &Value) -> bool {
        match value {
            Value::Object(fields) => fields.iter().any(|(key, value)| {
                key == "uncertain_writes" && value == &Value::Bool(true)
                    || key == "error"
                        && matches!(value.as_str(), Some("unknown_outcome" | "tool_failed"))
                    || uncertain(value)
            }),
            Value::Array(values) => values.iter().any(uncertain),
            // Artifact excerpts can contain only the start of a JSON receipt.
            // Recognize its uncertainty markers even if the excerpt is not a
            // complete JSON object. Extra retention is safer than losing it.
            Value::String(text) => {
                partial_command(text)
                    || text.starts_with('{')
                        && [
                            "\"error\":\"unknown_outcome\"",
                            "\"error\":\"tool_failed\"",
                            "\"uncertain_writes\":true",
                        ]
                        .iter()
                        .any(|marker| text.contains(marker))
            }
            _ => false,
        }
    }
    partial_command(content)
        || crate::json::parse(if content.starts_with("Captured tool result excerpt") {
            content.split_once('\n').map_or(content, |(_, range)| range)
        } else {
            content
        })
        .ok()
        .as_ref()
        .is_some_and(uncertain)
}

fn last_answer(c: &Conversation) -> Option<usize> {
    c.messages().iter().rposition(|m| {
        m.role() == "assistant"
            && m.tool_calls().is_empty()
            && m.value().get("content").is_some_and(|v| match v {
                Value::String(text) => !text.trim().is_empty(),
                Value::Array(parts) => parts.iter().any(|p| {
                    p.get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.trim().is_empty())
                }),
                _ => false,
            })
    })
}

pub fn plan(
    c: &Conversation,
    summary: &str,
    keep_turns: usize,
    limits: Limits,
) -> io::Result<CompactionPlan> {
    if summary.trim().is_empty() || summary.len() > 16 * 1024 || keep_turns == 0 {
        return Err(io::Error::other(
            "compaction needs a nonempty summary ≤16 KiB and at least one retained turn",
        ));
    }
    if has_unknown_outcome(c) {
        return Err(io::Error::other(
            "unknown tool outcome requires inspection; keep this journal and start a reconciled session before compaction",
        ));
    }
    let starts: Vec<usize> = c
        .messages()
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role() == "user" && !is_summary(m))
        .map(|(i, _)| i)
        .collect();
    if starts.len() <= keep_turns {
        return Err(io::Error::other("no whole old turns to compact"));
    }
    let cut = starts[starts.len() - keep_turns];
    let last_answer = last_answer(c);
    let removed: Vec<usize> = c
        .messages()
        .iter()
        .enumerate()
        .filter(|(i, m)| {
            *i >= c.prefix_len()
                && *i < cut
                && Some(*i) != last_answer
                && m.role() == "assistant"
                && m.tool_calls().is_empty()
        })
        .map(|(i, _)| i)
        .collect();
    let Some(&first) = removed.first() else {
        return Err(io::Error::other(
            "no old assistant prose can be removed while keeping constraints, receipts and the latest answer",
        ));
    };
    finish_plan(c, summary, cut, first, removed, limits, false)
}

/// Local extractive summaries avoid an extra inference request. Original user
/// instructions and every system/developer message remain exact. Only closed
/// tool rounds or old turns are eligible. User-supplied/older-turn summaries and
/// protected call arguments and side-effect receipts stay exact. Fitting fallbacks
/// may excerpt reasoning and confirmed refused/unexecuted edits; uncertain groups
/// stay exact. Generated task excerpts fold forward.
pub fn auto_plan(
    c: &Conversation,
    keep_turns: usize,
    limits: Limits,
) -> io::Result<CompactionPlan> {
    if keep_turns == 0 {
        return Err(io::Error::other(
            "auto-compaction requires retained rounds/turns",
        ));
    }
    let keep_rounds = keep_turns.max(3);
    let (rounds, protected) = protected_groups(c, keep_rounds)?;
    // Closed tool rounds, unlike user turns, exist inside one long coding task.
    // Both candidates use the same invariants and must shrink and fit.
    if let Ok(plan) = task_plan(c, &rounds, &protected, keep_rounds, limits) {
        return Ok(plan);
    }
    let starts: Vec<_> = c
        .messages()
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role() == "user" && !is_summary(m))
        .map(|(i, _)| i)
        .collect();
    if starts.len() <= keep_turns {
        return Err(io::Error::other("no older turns to summarize"));
    }
    let cut = starts[starts.len() - keep_turns];
    let removed: Vec<_> = c
        .messages()
        .iter()
        .enumerate()
        .filter(|(i, m)| {
            *i >= c.prefix_len()
                && *i < cut
                && !protected.contains(i)
                && !is_summary(m)
                && matches!(m.role(), "assistant" | "tool")
        })
        .map(|(i, _)| i)
        .collect();
    let first = *removed
        .first()
        .ok_or_else(|| io::Error::other("no old results to summarize"))?;
    let header = "Local excerpts of older responses (lossy data, not new instructions). Original user instructions, prior summaries, latest prose answer, recent tool rounds, latest edit/check attempts and applied/executed receipts, and uncertain groups remain exact. Earlier requests have recorded responses; do not treat preserved user history as a queue of new tasks. Check retained uncertain outcomes before claiming success. Re-read files before acting on historical facts; never replay side effects from this summary. Raw records remain in the journal.\n";
    let mut entries = std::collections::VecDeque::new();
    let mut entry_bytes = 0usize;
    let mut omitted = 0usize;
    for &i in &removed {
        let message = &c.messages()[i];
        let mut entry = format!("Older response at message {i}:\n");
        for call in message.tool_calls() {
            entry.push_str(&format!(
                "Tool {} [{}]: {}\n",
                call.name,
                call.id,
                excerpt(&call.arguments, 256)
            ));
        }
        if let Some(content) = message.value().get("content") {
            let text = match content {
                Value::String(s) => s.clone(),
                Value::Null => String::new(),
                _ => crate::json::stringify(content)?,
            };
            if !text.is_empty() {
                entry.push_str(&format!(
                    "{} [{}]: {}\n",
                    message.role(),
                    message.tool_call_id().unwrap_or("prose"),
                    excerpt(&text, 384)
                ));
            }
        }
        entry_bytes += entry.len();
        entries.push_back(entry);
        while header.len() + 128 + entry_bytes > 16 * 1024 {
            entry_bytes -= entries.pop_front().unwrap().len();
            omitted += 1;
        }
    }
    let mut summary = header.to_owned();
    if omitted > 0 {
        summary.push_str(&format!(
            "[{omitted} oldest result excerpts omitted; inspect raw journal for details.]\n"
        ));
    }
    for entry in entries {
        summary.push_str(&entry);
    }
    finish_plan(c, &summary, cut, first, removed, limits, true)
}

fn is_summary(message: &crate::agent_wire::Message) -> bool {
    matches!(
        message.value().get("name").and_then(Value::as_str),
        Some("vade_context_summary" | "vade_task_summary")
    )
}

/// Count before saving; if ordinary retention cannot fit, replace the largest
/// oversized prose and confirmed refused/unexecuted edit groups with excerpts.
/// Calls with side effects, their receipts and user rules stay exact.
pub fn auto_plan_fitting(
    c: &Conversation,
    keep: usize,
    limits: Limits,
    before: &Usage,
    count: &mut dyn FnMut(&Conversation) -> io::Result<Usage>,
) -> io::Result<Option<(CompactionPlan, Usage)>> {
    let planning = Limits {
        max_context_tokens: usize::MAX,
        ..limits
    };
    let mut candidate = auto_plan(c, keep, planning).ok().or_else(||
        finish_plan(c, "Tool-call reasoning excerpts (lossy historical data); calls and receipts stay exact. Full reasoning remains in the raw journal.", c.messages().len(), c.prefix_len(), Vec::new(), planning, true).ok());
    let mut removed = candidate.as_ref()
        .map_or_else(Vec::new, |p| p.removed_indices.clone());
    let mut summary = candidate.as_ref()
        .map_or_else(String::new, |p| p.summary.clone());
    let mut prose: Vec<_> = c.messages().iter().enumerate()
        .filter(|(i, m)| {
            *i >= c.prefix_len()
                && !removed.contains(i)
                && m.role() == "assistant"
                && (m.tool_calls().is_empty() || refused_edit_end(c, *i).is_some())
                && m.serialized_bytes() > 4096
        })
        .map(|(i, m)| (m.serialized_bytes(), i))
        .collect();
    prose.sort_unstable();
    loop {
        if let Some(mut plan) = candidate.take() {
            let after = count(plan.conversation())?;
            if after.fits(limits.max_context_tokens, limits.output_reserve_tokens)
                && !before.exact.zip(after.exact).is_some_and(|(a, b)| b.0 >= a.0)
            {
                plan.record_counts(before, &after);
                return Ok(Some((plan, after)));
            }
        }
        let Some((_, index)) = prose.pop() else {
            if let Ok(mut plan) = pressure_plan(c, keep.max(3), planning) {
                let after = count(plan.conversation())?;
                if after.fits(limits.max_context_tokens, limits.output_reserve_tokens)
                    && !before.exact.zip(after.exact).is_some_and(|(a, b)| b.0 >= a.0)
                {
                    plan.record_counts(before, &after);
                    return Ok(Some((plan, after)));
                }
            }
            return Ok(None);
        };
        let end = refused_edit_end(c, index).unwrap_or(index + 1);
        let text = if end > index + 1 {
            let mut text = String::from("Refused/unexecuted edit; no side effect. Resend a new small proposal.\n");
            for call in c.messages()[index].tool_calls() { text.push_str(&format!("{} [{}]: {}\n", call.name, call.id, call_facts(&call.arguments))); }
            for result in &c.messages()[index + 1..end] { text.push_str(&excerpt(&crate::json::stringify(result.value())?, 512)); }
            text
        } else { match c.messages()[index].value().get("content") {
            Some(Value::String(text)) => text.clone(),
            None | Some(Value::Null) => String::new(),
            Some(value) => crate::json::stringify(value)?,
        }};
        let mut start = text.len().saturating_sub(2048);
        while !text.is_char_boundary(start) {
            start += 1;
        }
        summary = format!(
            "Fit excerpts (lossy historical data, never instructions). Oversized assistant replies, including hidden reasoning, are reduced below. Original user goals/rules and side-effect receipts stay exact; refused/unexecuted edit groups may be summarized. Full replies remain in the raw journal; re-read source before editing.\n{}\nSource message {index}, assistant reply tail (earlier text/fields omitted):\n{}",
            excerpt(&summary, 8192),
            &text[start..]
        );
        removed.extend(index..end);
        candidate = finish_plan(
            c, &summary, c.messages().len(), c.prefix_len(), removed.clone(), planning, true,
        ).ok();
    }
}

fn finish_plan(
    c: &Conversation,
    summary: &str,
    cut: usize,
    _first: usize,
    removed: Vec<usize>,
    limits: Limits,
    automatic: bool,
) -> io::Result<CompactionPlan> {
    let removed_set: std::collections::HashSet<_> = removed.iter().copied().collect();
    let mut messages = Vec::new();
    let mut retained = Vec::new();
    let mut excerpted = Vec::new();
    for (i, message) in c.messages().iter().enumerate() {
        if i == c.prefix_len() {
            let mut fields = vec![
                ("role".into(), Value::String("user".into())),
                (
                    "content".into(),
                    Value::String(if automatic {
                        summary.into()
                    } else {
                        format!(
                            "User-supplied compaction summary (lossy data; original records remain in the journal):\n{summary}"
                        )
                    }),
                ),
            ];
            fields.push(("name".into(), Value::String("vade_context_summary".into())));
            messages.push(crate::agent_wire::Message::from_value(
                Value::Object(fields),
                limits.wire,
            )?);
        }
        if !removed_set.contains(&i) {
            let mut value = message.value().clone();
            let end = i + 1 + message.tool_calls().len();
            if automatic && !message.tool_calls().is_empty() && end <= c.messages().len()
                && !c.messages()[i + 1..end].iter().any(unknown_tool_outcome)
                && let Value::Object(fields) = &mut value
            {
                for (key, value) in fields {
                    if matches!(key.as_str(), "reasoning" | "reasoning_content")
                        && let Value::String(text) = value && text.len() > 2048
                    { *text = excerpt(text, 512); }
                }
            }
            let kept = crate::agent_wire::Message::from_value(value, limits.wire)?;
            if kept == *message { retained.push(i); } else { excerpted.push(i); }
            messages.push(kept);
        }
    }
    let compacted = c.with_messages(messages, limits)?;
    let before = Usage::for_conversation(c);
    let after = Usage::for_conversation(&compacted);
    if compacted.serialized_bytes() >= c.serialized_bytes() || after.estimated >= before.estimated {
        return Err(io::Error::other(
            "summary does not reduce context while keeping constraints and receipts; shorten it or start a new session",
        ));
    }
    if !after.fits(limits.max_context_tokens, limits.output_reserve_tokens) {
        return Err(io::Error::other(
            "preserved constraints/receipts plus summary still exceed admission budget; start a narrower session",
        ));
    }
    Ok(CompactionPlan {
        conversation: compacted,
        summary: summary.into(),
        before_bytes: c.serialized_bytes(),
        before_estimated: before.estimated,
        source_messages: c.messages().len(),
        source_request: c.request(limits.wire).to_json()?,
        limits,
        recent_start: cut,
        unchanged_prefix_messages: c.prefix_len(),
        retained_indices: retained,
        excerpted_indices: excerpted,
        removed_indices: removed,
        automatic,
        in_task: false,
        exact_counts: None,
    })
}
pub fn compact(
    c: &Conversation,
    summary: &str,
    keep_turns: usize,
    limits: Limits,
) -> io::Result<Conversation> {
    Ok(plan(c, summary, keep_turns, limits)?.into_conversation())
}

/// Offline byte audit using generated text, without loading a tokenizer, opening
/// a socket or touching a workspace. Token/cost numbers are explicit estimates.
pub fn offline_compaction_demo() -> io::Result<String> {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let limits = Limits {
        max_context_tokens: 131_072,
        ..Limits::default()
    };
    let message = |role: &str, text: String| {
        Value::Object(vec![
            ("role".into(), Value::String(role.into())),
            ("content".into(), Value::String(text)),
        ])
    };
    let mut system = "Keep immutable instructions and schema bytes.\n".repeat(2000);
    system.truncate(84_000);
    let request = Value::Object(vec![
        ("model".into(), Value::String("offline-demo".into())),
        (
            "tools".into(),
            Value::Array(vec![Value::Object(vec![
                ("type".into(), Value::String("function".into())),
                (
                    "function".into(),
                    Value::Object(vec![
                        ("name".into(), Value::String("lookup".into())),
                        (
                            "parameters".into(),
                            Value::Object(vec![("type".into(), Value::String("object".into()))]),
                        ),
                    ]),
                ),
            ])]),
        ),
        (
            "messages".into(),
            Value::Array(vec![
                message("system", system),
                message(
                    "developer",
                    "Frozen scoped rules.\n  Exact whitespace 🦀".into(),
                ),
                message("user", "Keep all original data.".into()),
                message("assistant", "Resolved old analysis. ".repeat(1200)),
                message("user", "Current question.".into()),
                message("assistant", "Recent answer retained.".into()),
            ]),
        ),
    ]);
    let before = Conversation::from_request(request, limits)?;
    let plan = plan(
        &before,
        "Older analysis resolved; preserve original data.",
        1,
        limits,
    )?;
    let after = plan.conversation();
    let mut builder = crate::prompt::AppendOnlyRequest::default();
    builder.prepare(
        before.request(limits.wire).serialized_options(),
        before.messages().iter().map(|m| m.serialized_json()),
    )?;
    let frozen = builder
        .stable_prefix_bytes(before.prefix_len())
        .ok_or_else(|| io::Error::other("demo prefix absent"))?
        .to_vec();
    let hash = |bytes: &[u8]| {
        let mut h = DefaultHasher::new();
        bytes.hash(&mut h);
        h.finish()
    };
    let before_hash = hash(&frozen);
    let body = builder.replace_suffix(
        after.request(limits.wire).serialized_options(),
        after.messages().iter().map(|m| m.serialized_json()),
        after.prefix_len(),
    )?;
    let suffix = body.len() - frozen.len() - 2;
    let preserved = builder
        .stable_prefix_bytes(after.prefix_len())
        .ok_or_else(|| io::Error::other("demo prefix absent"))?;
    if frozen != preserved {
        return Err(io::Error::other("demo frozen prefix changed"));
    }
    let summary = after.messages()[after.prefix_len()].serialized_bytes();
    let summary_estimate = summary.div_ceil(3);
    let suffix_estimate = suffix.div_ceil(3);
    let baseline = 28_000usize;
    Ok(format!(
        "Offline compaction demo; no network/tokenizer/inference.\n\
         Frozen prefix: {} bytes · diagnostic DefaultHasher before {before_hash:016x} / after {:016x} · byte-identical.\n\
         Summary first at message {}: {summary} new wire bytes (≈{summary_estimate} tokens).\n\
         Fresh causal suffix including retained users/recent turns: {suffix} bytes (≈{suffix_estimate} tokens).\n\
         Estimated uncached prefill: {baseline} → {suffix_estimate} tokens (summary-only lower bound {summary_estimate}); ≈{} tokens avoided.\n\
         At the historical 60–65 s / 28k rate: ≈{:.2}–{:.2} s suffix prefill if the prefix is actually restored.\n\
         Token estimates use wire bytes/3, exclude renderer framing, and are not tokenizer counts or measured speed. Cache retention/granularity can add work; use negotiated admission and response cache usage on the pair.\n",
        frozen.len(),
        hash(preserved),
        after.prefix_len(),
        baseline.saturating_sub(suffix_estimate),
        60.0 * suffix_estimate as f64 / baseline as f64,
        65.0 * suffix_estimate as f64 / baseline as f64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;
    fn conversation() -> Conversation {
        Conversation::from_request(json::parse(r#"{"model":"mock","messages":[{"role":"system","content":"stable"},{"role":"user","content":"first long question with a great deal of repeated information"},{"role":"assistant","content":"a long answer that is intentionally large enough to benefit from a short summary. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. "},{"role":"user","content":"second"},{"role":"assistant","content":"answer"},{"role":"user","content":"third"}]}"#).unwrap(), Limits::default()).unwrap()
    }
    #[test]
    fn accounts_each_message_options_and_reserve_without_claiming_exact_tokens() {
        let c = conversation();
        let usage = Usage::for_conversation(&c);
        assert_eq!(usage.messages.len(), c.messages().len());
        assert_eq!(
            usage.estimated,
            usage.options_estimated
                + usage.messages.iter().map(|m| m.estimated).sum::<usize>()
                + usage.framing_estimated
                + usage.safety_margin
        );
        assert_eq!(
            usage.byte_bound,
            c.request(Limits::default().wire).to_json().unwrap().len()
        );
        assert!(usage.near_limit(usage.estimated + 10, 10));
        assert!(!usage.fits(usage.estimated, 1));
    }
    #[test]
    fn full_request_counts_schemas_opaque_fields_utf8_and_provider_output_limit() {
        let request = |extra: &str| {
            Conversation::from_request(crate::json::parse(&format!(
            r#"{{"model":"mock","tools":[{{"type":"function","function":{{"name":"read","description":"{}"}}}}],"max_completion_tokens":6000,"opaque":{{"reasoning":"{}"}},"messages":[{{"role":"system","content":"rules"}},{{"role":"user","content":[{{"type":"text","text":"🦀多字"}}]}}]}}"#,
            extra, extra)).unwrap(), Limits::default()).unwrap()
        };
        let small = Usage::for_conversation(&request("small"));
        let large = Usage::for_conversation(&request(&"schema ".repeat(2000)));
        assert!(large.estimated > small.estimated + 9000);
        assert!(large.tools_estimated > small.tools_estimated);
        assert!(small.fits(small.estimated + 6000, 4096));
        assert!(!small.fits(small.estimated + 5999, 4096));
        assert_eq!(
            large.byte_bound,
            request(&"schema ".repeat(2000)).serialized_bytes()
        );
        assert!(large.summary(8192, 4096).contains("heuristic only"));
        let c = Conversation::from_request(crate::json::parse(r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://invalid.example/image"}}]}]}"#).unwrap(), Limits::default()).unwrap();
        let usage = Usage::for_conversation(&c);
        assert!(usage.admission_issue.is_some());
        assert!(!usage.fits(usize::MAX, 4096));
    }
    #[test]
    fn append_admission_is_at_least_as_cautious_as_final_request() {
        let mut c = conversation();
        let m = crate::agent_wire::Message::from_value(
            crate::json::parse(
                r#"{"role":"user","content":"a\\b\n🦀","extension":{"opaque":[1,2,3]}}"#,
            )
            .unwrap(),
            Limits::default().wire,
        )
        .unwrap();
        let preview = Usage::with_appended(&c, std::slice::from_ref(&m));
        c.push_user(m.into_value(), Limits::default()).unwrap();
        let actual = Usage::for_conversation(&c);
        assert!(preview.estimated >= actual.estimated);
        assert_eq!(preview.byte_bound, actual.byte_bound);
    }
    #[test]
    fn explicit_compaction_keeps_stable_prefix_and_whole_recent_turns() {
        let c = conversation();
        let compacted = compact(&c, "First resolved.", 2, Limits::default()).unwrap();
        assert_eq!(compacted.options(), c.options());
        assert_eq!(compacted.messages()[0], c.messages()[0]);
        assert_eq!(&compacted.messages()[3..], &c.messages()[3..]);
        assert!(
            compacted.messages()[1]
                .value()
                .get("content")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("First resolved.")
        );
        assert_eq!(c.messages().len(), 6);
    }
    #[test]
    fn refuses_empty_nonreducing_or_overbroad_compaction() {
        let c = conversation();
        for (summary, keep) in [("", 1), ("short", 0), ("short", 3), ("short", usize::MAX)] {
            assert!(compact(&c, summary, keep, Limits::default()).is_err());
        }
        assert!(compact(&c, &"x".repeat(16384), 1, Limits::default()).is_err());
    }
    #[test]
    fn compaction_never_splits_tool_groups() {
        let c = Conversation::from_request(json::parse(r#"{"model":"mock","messages":[{"role":"system","content":"stable"},{"role":"user","content":"old long turn that has now been resolved and can be summarized compactly"},{"role":"assistant","content":"a detailed old answer that is now no longer needed for any further model work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. "},{"role":"user","content":"new"},{"role":"assistant","tool_calls":[{"id":"x","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"x","content":"result"},{"role":"assistant","content":"done"}]}"#).unwrap(), Limits::default()).unwrap();
        let compacted = compact(&c, "Old resolved.", 1, Limits::default()).unwrap();
        assert_eq!(&compacted.messages()[3..], &c.messages()[3..]);
    }
    #[test]
    fn plan_keeps_old_constraints_tool_receipts_and_latest_answer_exactly() {
        let original = format!(
            r#"{{"model":"mock","messages":[
            {{"role":"system","content":"system"}},
            {{"role":"user","content":"NEVER delete original data; unresolved: which file?"}},
            {{"role":"assistant","content":"{}"}},
            {{"role":"developer","content":"additional immutable constraint"}},
            {{"role":"assistant","tool_calls":[{{"id":"write","type":"function","function":{{"name":"apply_patch","arguments":"{{}}"}}}}]}},
            {{"role":"tool","tool_call_id":"write","content":"Applied file revision 2; receipt exact"}},
            {{"role":"assistant","content":"last useful answer and decision"}},
            {{"role":"user","content":"current intent"}}]}}"#,
            "verbose resolved work ".repeat(100)
        );
        let c =
            Conversation::from_request(json::parse(&original).unwrap(), Limits::default()).unwrap();
        let before = c.request(Limits::default().wire).to_json().unwrap();
        let plan = plan(
            &c,
            "Earlier analysis resolved; preserve the pending file question.",
            1,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(plan.removed_indices, [2]);
        assert_eq!(plan.unchanged_prefix_messages, c.prefix_len());
        for i in [0, 1, 3, 4, 5, 6, 7] {
            assert!(plan.conversation.messages().contains(&c.messages()[i]));
        }
        assert_eq!(c.request(Limits::default().wire).to_json().unwrap(), before);
        assert!(plan.preview().contains("invalidates all later causal KV"));
        assert!(
            plan.preview()
                .contains("semantic completeness is not verified")
        );
        let uncertain = original.replace(
            "Applied file revision 2; receipt exact",
            r#"{\"ok\":false,\"error\":\"unknown_outcome\"}"#,
        );
        let c = Conversation::from_request(json::parse(&uncertain).unwrap(), Limits::default())
            .unwrap();
        let automatic = auto_plan(&c, 1, Limits::default()).unwrap();
        for i in [4, 5] {
            assert!(automatic.retained_indices.contains(&i));
        }
        let nested = original.replace(
            "Applied file revision 2; receipt exact",
            r#"{\"results\":[{\"uncertain_writes\":true}]}"#,
        );
        let nested =
            Conversation::from_request(json::parse(&nested).unwrap(), Limits::default()).unwrap();
        let automatic = auto_plan(&nested, 1, Limits::default()).unwrap();
        for i in [4, 5] {
            assert!(automatic.retained_indices.contains(&i));
        }
        assert!(
            super::plan(&c, "short", 1, Limits::default())
                .unwrap_err()
                .to_string()
                .contains("unknown tool outcome")
        );
    }
    #[test]
    fn single_task_compaction_keeps_goal_edits_checks_and_uncertain_groups() {
        let mut rounds = Vec::new();
        for i in 0..9 {
            let name = ["apply_patch", "workspace_shell", "workspace_read_lines"][i.min(2)];
            rounds.push(format!(r#"{{"role":"assistant","tool_calls":[{{"id":"r{i}","type":"function","function":{{"name":"{name}","arguments":"{{\"path\":\"src/task.rs\",\"start_line\":13,\"end_line\":21}}"}}}}]}}"#));
            let text = if i == 2 {
                r#"{"error":"unknown_outcome"}"#.into()
            } else {
                format!("key fact {i} {}", "old source ".repeat(700))
            };
            rounds.push(format!(
                r#"{{"role":"tool","tool_call_id":"r{i}","content":{}}}"#,
                json::quote(&text)
            ));
        }
        let c = Conversation::from_request(json::parse(&format!(r#"{{"messages":[{{"role":"system","content":"stable"}},{{"role":"user","content":"fix src/task.rs; keep public API"}},{}]}}"#, rounds.join(","))).unwrap(), Limits::default()).unwrap();
        let plan = auto_plan(&c, 2, Limits::default()).unwrap();
        assert!(plan.in_task && plan.summary.contains("src/task.rs"));
        assert!(plan.summary.contains("start_line=13"));
        for i in (0..8).chain(14..20) {
            assert!(plan.retained_indices.contains(&i));
        }
        for i in 8..14 {
            assert!(plan.removed_indices.contains(&i));
        }
        assert_eq!(plan.conversation.messages()[0], c.messages()[0]);
        assert!(plan.conversation.messages().contains(&c.messages()[1]));
        assert!(is_summary(&plan.conversation.messages()[1]));
    }
    #[test]
    fn auto_compaction_preserves_prior_summaries_and_stable_wire_prefix() {
        let limits = Limits::default();
        let request = format!(
            r#"{{"model":"mock","tools":[],"messages":[{{"role":"system","content":"stable 🦀"}},{{"role":"developer","content":"exact project prefix"}},{{"role":"user","content":"first task"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"second task"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"current task"}}]}}"#,
            "resolved first ".repeat(300),
            "resolved second ".repeat(300)
        );
        let original = Conversation::from_request(json::parse(&request).unwrap(), limits).unwrap();
        let first = auto_plan(&original, 1, limits).unwrap().into_conversation();
        let summary = first
            .messages()
            .iter()
            .find(|m| m.value().get("name").and_then(Value::as_str) == Some("vade_context_summary"))
            .unwrap()
            .clone();
        // Only the prior summary is old/removable: another request must not
        // shorten it into an excerpt of itself.
        assert!(auto_plan(&first, 1, limits).is_err());
        let mut messages = first.messages().to_vec();
        messages.push(
            crate::agent_wire::Message::from_value(
                json::parse(r#"{"role":"assistant","content":"current answer completed"}"#)
                    .unwrap(),
                limits.wire,
            )
            .unwrap(),
        );
        let mut next = first.with_messages(messages, limits).unwrap();
        next.push_user(
            json::parse(r#"{"role":"user","content":"later task"}"#).unwrap(),
            limits,
        )
        .unwrap();
        let second = auto_plan(&next, 1, limits).unwrap().into_conversation();
        assert!(second.messages().contains(&summary));
        assert_eq!(
            second
                .messages()
                .iter()
                .filter(|m| m.value().get("name").and_then(Value::as_str)
                    == Some("vade_context_summary"))
                .count(),
            2
        );
        for c in [&first, &second] {
            assert_eq!(
                c.request(limits.wire).serialized_options(),
                original.request(limits.wire).serialized_options()
            );
            for i in 0..original.prefix_len() {
                assert_eq!(
                    c.messages()[i].serialized_json().as_bytes(),
                    original.messages()[i].serialized_json().as_bytes()
                );
            }
        }
    }
    #[test]
    fn auto_compaction_bounds_many_old_excerpts_without_failing() {
        let mut messages = vec![json::parse(r#"{"role":"system","content":"fixed"}"#).unwrap()];
        for i in 0..70 {
            messages
                .push(json::parse(&format!(r#"{{"role":"user","content":"task {i}"}}"#)).unwrap());
            messages.push(Value::Object(vec![
                ("role".into(), Value::String("assistant".into())),
                (
                    "content".into(),
                    Value::String(format!("result {i} {}", "resolved ".repeat(100))),
                ),
            ]));
        }
        let c = Conversation::from_request(
            Value::Object(vec![("messages".into(), Value::Array(messages))]),
            Limits::default(),
        )
        .unwrap();
        let plan = auto_plan(&c, 2, Limits::default()).unwrap();
        assert!(plan.summary.len() <= 16 * 1024);
        assert!(plan.summary.contains("oldest result excerpts omitted"));
        assert!(plan.summary.contains("result 67"));
        assert!(!plan.summary.contains("result 0 "));
        assert_eq!(
            plan.conversation
                .messages()
                .iter()
                .filter(|m| m.role() == "user" && m.value().get("name").is_none())
                .count(),
            70
        );
    }
    #[test]
    fn auto_trigger_attempts_once_until_usage_falls_to_sixty_percent() {
        let c = conversation();
        let usage = Usage::for_conversation(&c);
        let required = usage.required(0);
        let mut trigger = AutoCompaction::default();
        let high_limit = required * 5 / 4;
        assert!(trigger.observe(&usage, high_limit, 0));
        for _ in 0..8 {
            assert!(!trigger.observe(&usage, high_limit, 0));
        }
        assert!(!trigger.observe(&usage, required * 10 / 7, 0));
        assert!(!trigger.observe(&usage, required * 2, 0));
        assert!(trigger.observe(&usage, high_limit, 0));
        let mut large = usage.clone(); large.exact = Some((50_000, 65_536, 131_072, true));
        assert!(!large.near_limit(131_072, 65_536));
        large.exact = Some((122_881, 7935, 131_072, true));
        assert!(!large.fits(131_072, 16_384) && large.fits(131_072, 7935));
        let limits = Limits { max_context_tokens: 32768, output_reserve_tokens: 16384, ..Limits::default() };
        let raw = format!(r#"{{"messages":[{{"role":"system","content":"stable"}},{{"role":"user","content":"{}"}},{{"role":"assistant","content":"{} END β 🦀"}},{{"role":"user","content":"/continue"}}]}}"#, "u".repeat(36000), "reply ".repeat(5500));
        let cutoff = Conversation::from_request(json::parse(&raw).unwrap(), limits).unwrap();
        let before = Usage::for_conversation(&cutoff);
        let (plan, after) = auto_plan_fitting(&cutoff, 2, limits, &before, &mut |c| Ok(Usage::for_conversation(c))).unwrap().unwrap();
        assert!(!before.fits(32768, 16384) && after.fits(32768, 16384) && plan.summary.contains("END β 🦀") && plan.conversation.messages().contains(&cutoff.messages()[0]));
    }
    #[test]
    fn empty_recent_assistant_does_not_displace_the_last_meaningful_answer() {
        let request = format!(
            r#"{{"messages":[{{"role":"user","content":"old"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"question"}},{{"role":"assistant","content":"meaningful answer"}},{{"role":"user","content":"interrupted"}},{{"role":"assistant","content":"  "}},{{"role":"user","content":"current"}}]}}"#,
            "old resolved prose ".repeat(100)
        );
        let c =
            Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
        let plan = plan(&c, "Earlier work resolved.", 1, Limits::default()).unwrap();
        assert!(plan.retained_indices.contains(&3));
        assert!(
            auto_plan(&c, 1, Limits::default())
                .unwrap()
                .retained_indices
                .contains(&3)
        );
        assert_eq!(
            plan.conversation
                .messages()
                .iter()
                .filter(|m| m.value().get("content").and_then(Value::as_str)
                    == Some("meaningful answer"))
                .count(),
            1
        );
    }
    #[test]
    fn empty_history_append_has_exact_measured_byte_count() {
        let mut c = Conversation::from_request(
            json::parse(r#"{"messages":[]}"#).unwrap(),
            Limits::default(),
        )
        .unwrap();
        let message = crate::agent_wire::Message::from_value(
            json::parse(r#"{"role":"user","content":"first 🦀"}"#).unwrap(),
            Limits::default().wire,
        )
        .unwrap();
        let projected = Usage::with_appended(&c, std::slice::from_ref(&message));
        c.push_user(message.into_value(), Limits::default())
            .unwrap();
        assert_eq!(
            projected.byte_bound,
            c.request(Limits::default().wire).to_json().unwrap().len()
        );
        assert!(projected.estimated >= Usage::for_conversation(&c).estimated);
    }
    #[test]
    fn runtime_stops_before_a_model_call_over_budget() {
        use crate::agent::{
            self, LimitKind, Model, ModelRequest, StopReason, ToolError, ToolExecutor,
            ToolInvocation,
        };
        use crate::agent_wire::AssistantTurn;
        use std::sync::atomic::AtomicBool;
        struct NeverModel;
        impl Model for NeverModel {
            fn complete(
                &mut self,
                _: ModelRequest<'_>,
                _: &AtomicBool,
                _: &mut dyn FnMut(&str) -> io::Result<()>,
            ) -> io::Result<AssistantTurn> {
                panic!("model must not run above budget")
            }
        }
        struct NeverTools;
        impl ToolExecutor for NeverTools {
            fn execute(
                &mut self,
                _: ToolInvocation<'_>,
                _: &AtomicBool,
            ) -> Result<String, ToolError> {
                panic!("tools must not run")
            }
        }
        let mut c = conversation();
        c.push_user(
            Value::Object(vec![
                ("role".into(), Value::String("user".into())),
                ("content".into(), Value::String("x".repeat(20000))),
            ]),
            Limits::default(),
        )
        .unwrap();
        let before = c.messages().to_vec();
        let report = agent::run(
            &mut c,
            &mut NeverModel,
            &mut NeverTools,
            Limits {
                max_context_tokens: 8192,
                ..Limits::default()
            },
            &AtomicBool::new(false),
            &mut |_| {},
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::Context));
        assert_eq!(report.model_turns, 0);
        assert_eq!(before, c.messages());
        let exact = Usage::for_conversation(&c)
            .with_count(&crate::contract::InputCount::Exact {
                prompt_tokens: 100,
                output_tokens: 128,
                context_limit: 8192,
                admitted: true,
            })
            .unwrap();
        assert!(exact.fits(8192, 4096));
        assert_eq!(exact.required(4096), 228); // actual reservation, no heuristic margin
        assert!(!exact.near_limit(8192, 4096));
        let refused = exact
            .with_count(&crate::contract::InputCount::Exact {
                prompt_tokens: 9000,
                output_tokens: 128,
                context_limit: 8192,
                admitted: false,
            })
            .unwrap();
        assert!(!refused.fits(8192, 0));
    }
    #[test]
    fn pressure_compacts_old_successful_edits_but_keeps_goal_recent_and_uncertain() {
        let mut messages = vec![json::parse(r#"{"role":"system","content":"stable rules"}"#).unwrap(), json::parse(r#"{"role":"user","content":"implement all outputs"}"#).unwrap()]; for i in 0..10 { let args = json::stringify(&json::parse(&format!(r#"{{"patch":{}}}"#, json::quote(&format!("*** Begin Patch\n*** Add File: f{i}.rs\n+{}\n*** End Patch", "x".repeat(8000))))).unwrap()).unwrap(); messages.push(json::parse(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"r{i}","type":"function","function":{{"name":"apply_patch","arguments":{}}}}}]}}"#, json::quote(&args))).unwrap()); messages.push(json::parse(&format!(r#"{{"role":"tool","tool_call_id":"r{i}","content":{}}}"#, json::quote(if i == 6 { r#"{"error":"unknown_outcome"}"# } else { "Applied 1 file operations." }))).unwrap()); } let c = Conversation::from_request(Value::Object(vec![("messages".into(), Value::Array(messages))]), Limits::default()).unwrap(); let limits = Limits { max_context_tokens: 16000, output_reserve_tokens: 128, ..Limits::default() }; let plan = auto_plan_fitting(&c, 2, limits, &Usage::for_conversation(&c), &mut |c| Ok(Usage::for_conversation(c))).unwrap().unwrap().0; assert!(plan.conversation.serialized_bytes() < c.serialized_bytes()); for i in (0..2).chain(14..22) { assert!(plan.retained_indices.contains(&i)); } assert!(plan.removed_indices.contains(&2)); assert!(plan.summary.contains("f0.rs")); assert_eq!(plan.conversation.messages()[0], c.messages()[0]);
    }

}
