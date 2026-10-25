//! Exit status, prompt recall and approval input handling.
use super::*;
use crate::cli::{approval::ApprovalReply, args::Parsed, test_support::parse};

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
