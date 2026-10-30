//! Exit status, prompt recall and approval input handling.
use super::*;
use crate::cli::{approval::ApprovalReply, args::Parsed, test_support::parse};
use vade_harness::shell::permissions::{Approvals, Scope};

#[test]
fn final_check_failure_only_changes_normal_run_exit() {
    assert_eq!([(true, true), (false, true), (true, false), (false, false)].map(|(s, f)| run_exit(s, f)), [3, 1, 0, 1].map(ExitCode::from));
}

#[test]
fn completion_eviction_invalidates_recall_and_up_clamps_stale_indices() {
    let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
        panic!()
    };
    let mut app = App::new(config).unwrap();
    for i in 0..10 {
        app.complete_session(format!("prompt-{i}"), "answer".into())
            .unwrap();
    }
    app.input.set("unfinished draft".into());
    app.key(Key::Up).unwrap();
    assert_eq!(app.recall, Some(9));
    assert_eq!(app.input.text(), "prompt-9");
    assert_eq!(app.draft, "unfinished draft");

    // The worker completion path uses this same helper. One large answer
    // evicts all ten old turns while the user is browsing their recall.
    let prompt = "new prompt";
    app.complete_session(
        prompt.into(),
        "x".repeat(vade_harness::session::MAX_HISTORY_BYTES - prompt.len()),
    )
    .unwrap();
    assert_eq!(app.session.turns(), 1);
    assert_eq!(app.session.pruned_turns, 10);
    assert_eq!(app.recall, None);
    assert!(app.draft.is_empty());
    assert_eq!(
        app.input.text(),
        "prompt-9",
        "keep the active editor intact"
    );
    app.key(Key::Up).unwrap();
    assert_eq!(app.input.text(), prompt);
    assert_eq!(app.recall, Some(0));

    // Defense in depth if a future history mutation forgets invalidation.
    app.recall = Some(9);
    app.key(Key::Up).unwrap();
    assert_eq!(app.input.text(), prompt);
    assert_eq!(app.recall, Some(0));
    app.key(Key::Down).unwrap();
    assert_eq!(app.input.text(), "prompt-9");
    assert_eq!(app.recall, None);
}

#[test]
fn single_key_approval_answers_only_on_an_empty_editor() {
    let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
        panic!()
    };
    let mut app = App::new(config).unwrap();
    let settled = Instant::now() - APPROVAL_SETTLE;
    // A draft keeps y/n as text; the approval stays pending.
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    app.approval_shown = Some(settled);
    app.input.set("not yet".into());
    app.key(Key::Text("y".into())).unwrap();
    assert_eq!(app.input.text(), "not yety");
    assert!(app.approval_reply.is_some());
    assert!(rx.try_recv().is_err());
    // An empty editor answers with one key.
    app.input.clear();
    app.key(Key::Text("n".into())).unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(false));
    assert!(app.approval_reply.is_none());
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    app.approval_shown = Some(settled);
    app.key(Key::Text("Y".into())).unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(true));
    assert!(app.input.is_empty());
}

#[test]
fn single_key_approval_ignores_multiline_input_and_type_ahead() {
    let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
        panic!()
    };
    let mut app = App::new(config).unwrap();
    // Multiline (/paste) collection: a leading y is part of the line.
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    app.approval_shown = Some(Instant::now() - APPROVAL_SETTLE);
    app.multiline = Some(String::new());
    app.key(Key::Text("y".into())).unwrap();
    assert_eq!(app.input.text(), "y");
    assert!(app.approval_reply.is_some());
    assert!(rx.try_recv().is_err());
    app.multiline = None;
    app.input.clear();
    // Type-ahead: a y within the settle time of the prompt appearing is
    // draft text, not an answer; after the settle time it answers.
    app.approval_shown = Some(Instant::now());
    app.key(Key::Text("y".into())).unwrap();
    assert_eq!(app.input.text(), "y");
    assert!(app.approval_reply.is_some());
    assert!(rx.try_recv().is_err());
    app.input.clear();
    app.approval_shown = Some(Instant::now() - APPROVAL_SETTLE);
    app.key(Key::Text("y".into())).unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(true));
    assert!(app.approval_reply.is_none());
}

#[test]
fn ctrl_d_exits_through_cancellation_while_approval_is_pending() {
    let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
        panic!()
    };
    let mut app = App::new(config).unwrap();
    // Nonempty draft: Ctrl+D keeps the approval interaction as text.
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    app.approval_shown = Some(Instant::now() - APPROVAL_SETTLE);
    app.input.set("draft".into());
    app.key(Key::End).unwrap();
    assert!(app.approval_reply.is_some());
    // Empty editor: Ctrl+D is an exit, exactly like /exit during an
    // approval: quit, decline by closing the approval, and cancel.
    app.input.clear();
    app.key(Key::End).unwrap();
    assert!(app.approval_reply.is_none());
    assert!(rx.recv_timeout(Duration::from_secs(1)).is_err(),
        "approval must be declined by closing, never approved");
}

#[test]
fn enter_answers_an_approval_only_with_a_typed_one_line_key() {
    let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
        panic!()
    };
    let mut app = App::new(config).unwrap();
    // An in-memory allow list, so a scope key that got through would be
    // recorded for this thread instead of refused for a missing list.
    let approvals = std::sync::Arc::new(Approvals::new(None, None, true).unwrap());
    app.approvals = Some(std::sync::Arc::clone(&approvals));
    app.approval_command = true;
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    // Before the settle time, a key goes to the editor as a draft.
    app.approval_shown = Some(Instant::now());
    // Pasted keys, with or without a line end, and a recalled key stay
    // drafts, also for the scope keys that write an allow rule.
    for (pasted, text) in [(true, "y\n"), (true, "p"), (true, "p\n"), (false, "s")] {
        app.input.clear();
        if pasted {
            app.key(Key::PasteStart).unwrap();
            app.key(Key::Text(text.into())).unwrap();
            app.key(Key::PasteEnd).unwrap();
        } else {
            app.input.set(text.into());
        }
        app.key(Key::Enter).unwrap();
        assert!(app.approval_reply.is_some(), "{text:?}");
        assert!(rx.try_recv().is_err(), "{text:?}");
        assert_eq!(approvals.take_choice(), Scope::Once, "{text:?}");
    }
    // A typed key answers on Enter and leaves the editor empty.
    app.input.clear();
    app.key(Key::Text("y".into())).unwrap();
    assert_eq!(app.input.text(), "y");
    app.key(Key::Enter).unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(true));
    assert!(app.approval_reply.is_none());
    assert!(app.input.is_empty());
    assert_eq!(approvals.take_choice(), Scope::Once);
    // A typed scope key answers with its scope.
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    app.key(Key::Text("p".into())).unwrap();
    app.key(Key::Enter).unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(true));
    assert_eq!(approvals.take_choice(), Scope::Project);
}

#[test]
fn piped_exit_waits_only_for_a_users_message_that_has_not_started() {
    use crate::cli::events::{Pending, WorkerEvent, WorkerResult};
    let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else {
        panic!()
    };
    let mut app = App::new(config).unwrap();
    app.quiet = true;
    assert!(app.piped_exit_ends_input("/exit"));
    assert!(app.piped_exit_ends_input("/quit"));
    assert!(!app.piped_exit_ends_input("/exits"));
    // An empty user marks other work, such as /compact or a prewarm.
    for (user, underway, ends) in [("hello", false, true), ("hello", true, false), ("", false, false), ("", true, false)] {
        let (tx, rx) = crate::interaction::lanes::<WorkerEvent, WorkerResult>();
        let (controls, _) = mpsc::sync_channel(1);
        app.pending = Some(Pending {
            budget: None, native_progress: None, user: user.into(), started: Instant::now(),
            cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)), rx, controls,
            steering_pending: Vec::new(), render_skipped: false, render_skip: 0, round_displayed: false,
            received: false, canceled: false, displayed: 0, provisional: String::new(), edits: 0,
            worker: None, join_on_drop: false,
        });
        tx.state(RunState::Queued);
        if underway {
            tx.state(RunState::Generating);
            tx.state(RunState::Queued);
        }
        assert_eq!(app.piped_exit_ends_input("/exit"), ends, "{user:?} {underway}");
    }
    app.pending = None;
    app.multiline = Some(String::new());
    assert!(!app.piped_exit_ends_input("/exit"));
    app.multiline = None;
    app.quiet = false;
    assert!(!app.piped_exit_ends_input("/exit"));
}
