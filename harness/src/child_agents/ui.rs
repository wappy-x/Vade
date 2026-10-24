//! Bounded plain/TUI child list. No task prose, paths, model errors or secrets.
use super::{Outcome, StopRequest, TaskSnapshot, TaskState};

pub fn state_label(state: &TaskState) -> &'static str {
    match state {
        TaskState::Running => "running",
        TaskState::Stopping(StopRequest::WallTime) => "deadline: stopping",
        TaskState::Stopping(_) => "cancel requested",
        TaskState::Collected(Outcome::Completed) => "collected: completed",
        TaskState::Collected(Outcome::Cancelled) => "collected: cancelled",
        TaskState::Collected(Outcome::QuotaTokens) => "collected: token quota",
        TaskState::Collected(Outcome::Runtime(crate::agent::StopReason::Limit(_))) => {
            "collected: runtime quota"
        }
        TaskState::Collected(_) => "collected: failed",
        TaskState::Finished(Outcome::Completed) => "completed",
        TaskState::Finished(Outcome::Cancelled) => "cancelled locally",
        TaskState::Finished(Outcome::QuotaTokens) => "token quota",
        TaskState::Finished(Outcome::Runtime(crate::agent::StopReason::Limit(_))) => {
            "runtime quota"
        }
        TaskState::Finished(_) => "failed",
    }
}

/// Width is a cell bound because generated content here is ASCII only. Call on
/// lifecycle revision changes, or a UI's ordinary elapsed-time refresh tick.
pub fn render(children: &[TaskSnapshot], width: usize, max_rows: usize) -> String {
    let width = width.min(240);
    let rows = max_rows.min(33);
    if width == 0 || rows == 0 {
        return String::new();
    }
    let mut lines = vec![format!("Children ({})", children.len())];
    let available = rows - 1;
    let shown = children
        .len()
        .min(available.saturating_sub(usize::from(children.len() > available)));
    // Keep live work visible when older completed tasks fill a short terminal.
    let active =
        |child: &&TaskSnapshot| matches!(child.state, TaskState::Running | TaskState::Stopping(_));
    for child in children
        .iter()
        .filter(active)
        .chain(children.iter().filter(|child| !active(child)))
        .take(shown)
    {
        let request = child
            .stop_requested
            .map_or(String::new(), |r| format!(" (stop: {r:?})"));
        lines.push(format!(
            "{}  {}{}  {:.1}s  turns {}  tokens reserved {}  writes {}",
            child.id,
            state_label(&child.state),
            request,
            child.elapsed.as_secs_f64(),
            child.model_turns,
            child.tokens_reserved,
            if child.workspace_write_lease {
                "workspace (shell)".into()
            } else {
                child.writable_files.to_string()
            }
        ));
    }
    if children.len() > shown && available > shown {
        lines.push(format!("+{} more", children.len() - shown));
    }
    for line in &mut lines {
        if line.len() > width {
            line.truncate(width);
            if width >= 3 {
                line.replace_range(width - 3.., "...");
            }
        }
    }
    lines.join("\n")
}

/// Untruncated command output for the bounded current/last activation. Task
/// prose and provider error strings never enter this metadata-only view.
pub fn context(children: &[TaskSnapshot]) -> String {
    let mut lines =
        vec!["Child context (current/last activation; validated responses only)".to_owned()];
    if children.is_empty() {
        lines.push("No child tasks. Cached tokens unknown.".into());
    }
    let number = |value: Option<u64>| value.map_or_else(|| "unknown".into(), |n| n.to_string());
    for child in children {
        let usage = child.usage;
        lines.push(format!(
            "{}: {} · last response: prompt tokens {} · output tokens {} · cached tokens {}\n  reported cached tokens: {} across {}/{} completed responses with telemetry · tokens reserved {}",
            child.id, state_label(&child.state),
            number(usage.last.and_then(|u| u.prompt_tokens)),
            number(usage.last.and_then(|u| u.completion_tokens)),
            number(usage.last.and_then(|u| u.cached_tokens)),
            if usage.responses_with_cached_usage == 0 { "unknown".into() } else { usage.reported_cached_tokens.to_string() },
            usage.responses_with_cached_usage, usage.completed_responses, child.tokens_reserved,
        ));
    }
    lines.push(
        "Cached input still occupies context; reservations and affinity are not cache hits.".into(),
    );
    lines.join("\n")
}
