//! Tool labels, complete previews and serialized approval replies.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::Duration;
use vade_harness::{
    agent::{RunBudget, RunPause},
    child_agents, coding_tools,
};

use super::{
    App, RunState,
    events::{WorkSender, WorkerEvent},
};

pub(crate) fn tool_label(name: &str, args: &vade_harness::json::Value) -> String {
    use vade_harness::json::{self, Value};
    let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let label = match name {
        "workspace_read" | "workspace_read_lines" => format!("Read {path}"),
        "workspace_list" => format!("List {path}"),
        "workspace_search" => format!("Search {path}"),
        "workspace_shell" => {
            let command = args
                .get("argv")
                .and_then(Value::as_array)
                .map(|args| {
                    args.iter()
                        .map(|arg| {
                            let text = arg.as_str().unwrap_or("?");
                            if text
                                .chars()
                                .all(|c| c.is_ascii_alphanumeric() || "_./=-".contains(c))
                                && !text.is_empty()
                            {
                                text.to_owned()
                            } else {
                                json::stringify(arg).unwrap_or_else(|_| "?".into())
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_else(|| "?".into());
            format!("Run: {command}")
        }
        "apply_patch" => "Edit workspace files".into(),
        "workspace_edit_lines" => format!("Edit lines in {path}"),
        "skill_read" => format!("Read skill {path}"),
        "workspace_manifest" => "List workspace files".into(),
        "workspace_symbols" => format!("Read symbols {path}"),
        "tool_output_read" => "Read tool output".into(),
        name if name.contains(vade_harness::mcp::SEPARATOR) => {
            let (server, tool) = name.split_once(vade_harness::mcp::SEPARATOR).unwrap_or((name, ""));
            format!("MCP {server}: {tool}")
        }
        _ => name.replace('_', " "),
    };
    // Command labels may already contain JSON escapes; preserve those and show
    // remaining nonprinting Unicode without changing the quoted argument bytes.
    label
        .chars()
        .map(|c| {
            if c == '\n' || vade_harness::terminal::preview_char_needs_escape(c) {
                let mut units = [0u16; 2];
                c.encode_utf16(&mut units)
                    .iter()
                    .map(|unit| format!("\\u{unit:04x}"))
                    .collect()
            } else {
                c.to_string()
            }
        })
        .collect()
}

impl App {
    pub(crate) fn close_approval(&mut self) -> Option<ApprovalReply> {
        let reply = self.approval_reply.take();
        if let Some(reply) = &reply {
            reply.close();
        }
        self.approval_shown = None;
        self.terminal.close_approval();
        reply
    }

    /// Raw previews are buffered until the frame has been flushed successfully.
    pub(crate) fn approval_presented(&mut self) {
        if self
            .pending
            .as_ref()
            .is_none_or(|p| p.cancel.load(Ordering::Acquire))
            || self
                .approval_reply
                .as_ref()
                .is_some_and(|reply| !reply.open())
        {
            self.close_approval();
            return;
        }
        if self.approval_shown.is_none()
            && (!self.terminal.raw() || self.terminal.approval_ready().is_ok())
            && let Some(reply) = &self.approval_reply
            && reply.presented()
        {
            self.approval_shown = Some(std::time::Instant::now());
        }
    }

    pub(crate) fn pending_approval_key(
        &mut self,
        key: &vade_harness::terminal::Key,
    ) -> std::io::Result<bool> {
        if self.approval_reply.is_none()
            || !matches!(
                key,
                vade_harness::terminal::Key::Escape | vade_harness::terminal::Key::Interrupt
            )
        {
            return Ok(false);
        }
        self.notice("Approval still pending; draft retained. Press y or n on an empty line, or /approve, /deny; /cancel cancels the run and declines.")?;
        Ok(true)
    }
}

#[derive(Clone)]
pub(crate) struct ApprovalReply {
    sender: mpsc::Sender<bool>,
    clock: Arc<ApprovalClock>,
    /// The tool thread that asked: the reach of a yes goes to it alone.
    asker: std::thread::ThreadId,
}
struct ApprovalClock {
    budget: Option<RunBudget>,
    state: Mutex<ApprovalState>,
}
// Presentation and worker cancellation share one lock, so a late frame cannot
// restart a closed wait. Queued handles never acquire a pause by themselves.
impl ApprovalClock {
    fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        state.pause.take();
    }
}
#[derive(Default)]
struct ApprovalState {
    closed: bool,
    pause: Option<RunPause>,
}
impl ApprovalReply {
    pub(crate) fn new(sender: mpsc::Sender<bool>, budget: Option<RunBudget>) -> Self {
        Self {
            sender,
            clock: Arc::new(ApprovalClock {
                budget,
                state: Mutex::new(ApprovalState::default()),
            }),
            asker: std::thread::current().id(),
        }
    }

    pub(crate) fn asker(&self) -> std::thread::ThreadId {
        self.asker
    }

    pub(crate) fn open(&self) -> bool {
        !self
            .clock
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .closed
    }

    fn presented(&self) -> bool {
        let mut state = self.clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            return false;
        }
        if state.pause.is_none() {
            state.pause = self.clock.budget.as_ref().map(RunBudget::pause);
        }
        true
    }

    pub(crate) fn close(&self) {
        self.clock.close();
    }

    pub(crate) fn send(&self, approved: bool) -> Result<(), mpsc::SendError<bool>> {
        // Resume before waking the worker: response processing is active work.
        self.close();
        self.sender.send(approved)
    }
}
// The worker owns this guard independently of cloned, queued UI reply handles.
// Cancellation closes the clock even if the UI cannot drain closure facts yet.
struct ApprovalGuard(Arc<ApprovalClock>);
impl Drop for ApprovalGuard {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// One shared lane for parent and child previews. Holding this guard through
/// the answer prevents a later request from replacing the displayed approval.
/// Contention and approval waits observe each caller's cancellation flag.
pub(crate) struct ToolPreviews {
    pub(crate) tx: Arc<WorkSender>,
    pub(crate) bytes: Mutex<usize>,
    pub(crate) budget: Option<RunBudget>,
}
impl ToolPreviews {
    pub(crate) fn show(
        &self,
        child: Option<child_agents::TaskId>,
        preview: coding_tools::Preview<'_>,
        cancel: &AtomicBool,
    ) -> bool {
        let mut bytes = loop {
            if cancel.load(Ordering::Acquire) {
                return false;
            }
            match self.bytes.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::park_timeout(Duration::from_millis(10))
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    cancel.store(true, Ordering::Release);
                    return false;
                }
            }
        };
        let detail = match child {
            Some(id) => format!(
                "Child {id} · parent approval policy applies\n{}",
                preview.detail
            ),
            None => preview.detail.to_owned(),
        };
        // Each tool already bounds its preview. Lifetime display volume must
        // never silently revoke an approval or cancel a long-running task.
        *bytes = detail.len();
        request_tool_preview_with_budget(
            &self.tx,
            coding_tools::Preview {
                detail: &detail,
                ..preview
            },
            cancel,
            self.budget.as_ref(),
        )
    }
}

pub(crate) fn request_tool_preview(
    tx: &WorkSender,
    preview: coding_tools::Preview<'_>,
    cancel: &AtomicBool,
) -> bool {
    request_tool_preview_with_budget(tx, preview, cancel, None)
}

pub(crate) fn request_tool_preview_with_budget(
    tx: &WorkSender,
    preview: coding_tools::Preview<'_>,
    cancel: &AtomicBool,
    budget: Option<&RunBudget>,
) -> bool {
    let (sender, rx) = mpsc::channel();
    let reply = ApprovalReply::new(sender, budget.cloned());
    let guard = ApprovalGuard(Arc::clone(&reply.clock));
    loop {
        if cancel.load(Ordering::Acquire) {
            return false;
        }
        if tx
            .fact(WorkerEvent::ToolPreview {
                title: preview.title.into(),
                detail: preview.detail.into(),
                reply: preview.approval_required.then(|| reply.clone()),
            })
            .is_ok()
        {
            break;
        }
        // Backpressure on authoritative facts is not a denial. The UI drains
        // the lane; dropping Pending sets cancel and disconnects this waiter.
        std::thread::park_timeout(Duration::from_millis(10));
    }
    drop(reply);
    if !preview.approval_required {
        return true;
    }
    // The UI acquires the pause only after actual presentation succeeds.
    tx.state(RunState::WaitingForApproval);
    let approved = loop {
        if cancel.load(Ordering::Relaxed) {
            break false;
        }
        match rx.recv_timeout(Duration::from_millis(25)) {
            Ok(approved) => break approved,
            Err(mpsc::RecvTimeoutError::Disconnected) => break false,
            Err(mpsc::RecvTimeoutError::Timeout) => (),
        }
    };
    drop(guard);
    tx.state(RunState::ToolRunning);
    while tx.fact(WorkerEvent::ToolApprovalClosed).is_err() {
        if cancel.load(Ordering::Acquire) {
            return false;
        }
        std::thread::park_timeout(Duration::from_millis(10));
    }
    approved && !cancel.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use crate::interaction;
    use super::*;
    use vade_harness::{json, terminal};

    #[test]
    fn command_label_escapes_invisible_unicode_without_changing_json_arguments() {
        let argument = "actual\u{200b} combining\u{0301} selector\u{e0100} tab\t literal\\u200b";
        let args = json::Value::Object(vec![(
            "argv".into(),
            json::Value::Array(vec![
                json::Value::String("/bin/printf".into()),
                json::Value::String(argument.into()),
            ]),
        )]);
        let label = tool_label("workspace_shell", &args);
        assert!(!label.chars().any(terminal::preview_char_needs_escape));
        assert!(label.contains("\\u200b"));
        assert!(label.contains("\\\\u200b"));
        assert!(label.contains("\\udb40\\udd00"));
        let displayed_argument = label.strip_prefix("Run: /bin/printf ").unwrap();
        assert_eq!(
            json::parse(displayed_argument).unwrap().as_str(),
            Some(argument)
        );
    }

    #[test]
    fn path_label_keeps_newlines_and_invisible_unicode_visible() {
        let args = json::Value::Object(vec![(
            "path".into(),
            json::Value::String("line\nzero\u{200b}".into()),
        )]);
        assert_eq!(
            tool_label("workspace_read", &args),
            "Read line\\u000azero\\u200b"
        );
    }

    #[test]
    fn native_previews_survive_render_overflow_and_wait_for_fact_capacity() {
        let (tx, rx) = interaction::lanes();
        let cancel = AtomicBool::new(false);
        assert!(tx.render(&"x".repeat(128 * 1024)));
        let preview = || coding_tools::Preview {
            title: "Command",
            detail: "exact operation",
            approval_required: false,
        };
        assert!(request_tool_preview(&tx, preview(), &cancel));
        assert!(
            matches!(rx.fact(), Ok(WorkerEvent::ToolPreview { detail, .. }) if detail == "exact operation")
        );
        for _ in 0..16 {
            tx.fact(WorkerEvent::Notice).unwrap();
        }
        let worker = std::thread::spawn(move || request_tool_preview(&tx, preview(), &cancel));
        assert!(matches!(rx.fact(), Ok(WorkerEvent::Notice)));
        assert!(worker.join().unwrap());
    }
}
