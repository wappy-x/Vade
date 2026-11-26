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
    app.key(Key::Text("1".into())).unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(true));
    assert!(app.input.is_empty());
}

#[test]
fn details_key_reveals_exact_text_without_answering() {
    let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else { panic!() };
    let mut app = App::new(config).unwrap();
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    app.approval_shown = Some(Instant::now() - APPROVAL_SETTLE);
    let prompt = vade_harness::approval_prompt::Prompt {
        what: "Vade wants to run a command".into(), object: "tool".into(),
        risk: vade_harness::approval_prompt::Risk::Red, tag: "can change files".into(),
        scope: Some("this project".into()), sentences: vec!["This command can change files.".into()],
        choices: "1  Yes, one time\n2  No, do not run it\n5  Show the exact details".into(), files: Vec::new(),
    };
    app.terminal.approval_prompt("Command", &prompt, Some("argv=[\"tool\"]")).unwrap();
    for key in ["d", "D"] {
        app.key(Key::Text(key.into())).unwrap();
        app.key(Key::Enter).unwrap();
        app.sent_line(key.into()).unwrap();
        assert!(app.approval_reply.is_some() && rx.try_recv().is_err());
        app.input.clear();
    }
    app.key(Key::Text("5".into())).unwrap();
    assert!(app.approval_reply.is_some() && rx.try_recv().is_err());
    assert!(!app.terminal.show_approval_details().unwrap(), "5 consumed the retained details");
    app.terminal.approval_preview_with_details("Command", "Three plain lines", "argv=[\"other\"]").unwrap();
    app.sent_line("5".into()).unwrap();
    assert!(app.approval_reply.is_some() && !app.terminal.show_approval_details().unwrap());
}

#[test]
fn approval_keys_are_single_characters_and_have_no_other_aliases() {
    let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else { panic!() };
    let mut app = App::new(config).unwrap();
    app.approvals = Some(std::sync::Arc::new(Approvals::new(None, None, true).unwrap()));
    app.approval_command = true;
    app.approval_choices = 15;
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    app.approval_shown = Some(Instant::now() - APPROVAL_SETTLE);
    for key in ["yes", "Yes", "no", "No", "11", "1\n", "y1", "s", "p", "a", "d", "S", "P", "A", "D", "Y", "N"] {
        app.input.clear();
        app.key(Key::Text(key.into())).unwrap();
        assert_eq!(app.input.text(), key);
        assert!(app.approval_reply.is_some() && rx.try_recv().is_err(), "raw {key:?}");
        app.key(Key::Enter).unwrap();
        assert!(app.approval_reply.is_some() && rx.try_recv().is_err(), "enter {key:?}");
        app.input.clear();
        if !key.contains('\n') {
            app.sent_line(key.into()).unwrap();
            assert!(app.approval_reply.is_some() && rx.try_recv().is_err(), "line {key:?}");
        }
    }
    app.close_approval();
    for raw in [true, false] {
        for (key, approved, scope) in [("1", true, Scope::Once), ("2", false, Scope::Once), ("3", true, Scope::Session), ("4", true, Scope::Project), ("y", true, Scope::Once), ("n", false, Scope::Once)] {
            let (tx, rx) = mpsc::channel();
            app.approval_reply = Some(ApprovalReply::new(tx, None));
            app.approval_shown = Some(Instant::now() - APPROVAL_SETTLE);
            app.approval_command = true;
            app.approval_choices = 15;
            app.input.clear();
            if raw { app.key(Key::Text(key.into())).unwrap(); } else { app.sent_line(key.into()).unwrap(); }
            assert_eq!(rx.try_recv(), Ok(approved), "{raw} {key}");
            assert_eq!(app.approvals.as_ref().unwrap().take_choice(), scope);
            assert!(app.approval_reply.is_none());
        }
    }
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
    app.approval_choices = 15;
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    // Before the settle time, a key goes to the editor as a draft.
    app.approval_shown = Some(Instant::now());
    // Pasted keys, with or without a line end, and a recalled key stay
    // drafts, also for the scope keys that write an allow rule.
    for (pasted, text) in [(true, "y\n"), (true, "4"), (true, "4\n"), (false, "3")] {
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
    app.approval_choices = 15;
    let (tx, rx) = mpsc::channel();
    app.approval_reply = Some(ApprovalReply::new(tx, None));
    app.key(Key::Text("4".into())).unwrap();
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

#[test]
fn numbered_answers_keep_no_fixed_and_reject_unoffered_scopes() {
    for raw_key in [false, true] {
        for kind in ["Workspace change", "Command", "Web request", "Web redirect", "Memory note", "MCP tool", "Secret file read"] {
            let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else { panic!() };
            let mut app = App::new(config).unwrap();
            app.approval_command = kind == "Command";
            app.approval_reach = matches!(kind, "Web request" | "Web redirect" | "Memory note");
            app.approvals = Some(std::sync::Arc::new(Approvals::new(None, None, true).unwrap()));
            app.approval_choices = 3; // The displayed prompt has only 1 and 2.
            let (tx, rx) = mpsc::channel();
            app.approval_reply = Some(ApprovalReply::new(tx, None));
            app.approval_shown = Some(Instant::now() - APPROVAL_SETTLE);
            for key in ["", "3", "4"] {
                app.input.clear();
                if raw_key { app.key(Key::Text(key.into())).unwrap(); app.key(Key::Enter).unwrap(); }
                else { app.sent_line(key.into()).unwrap(); }
                assert_eq!(rx.try_recv(), Err(mpsc::TryRecvError::Empty), "{kind}: {key}");
                assert!(app.approval_reply.is_some());
            }
            app.input.clear();
            if raw_key { app.key(Key::Text("2".into())).unwrap(); } else { app.sent_line("2".into()).unwrap(); }
            assert_eq!(rx.try_recv(), Ok(false), "{kind}");
        }
    }
}

#[test]
fn number_four_uses_the_displayed_scope_in_both_input_paths() {
    for enter in [false, true] {
        for (command, scope) in [(true, Scope::Project), (false, Scope::Always)] {
            let Parsed::Run(config) = parse(&["demo", "--plain"]).unwrap() else { panic!() };
            let mut app = App::new(config).unwrap();
            let folder = std::env::temp_dir().join(format!("vade-number-{}-{enter}-{command}", std::process::id()));
            std::fs::create_dir_all(folder.join("project")).unwrap();
            let approvals = std::sync::Arc::new(Approvals::for_workspace(&folder.join("project"), Some(&folder.join("home")), true).unwrap());
            app.approvals = Some(std::sync::Arc::clone(&approvals));
            app.approval_command = command;
            app.approval_reach = !command;
            app.approval_choices = 15;
            let (tx, rx) = mpsc::channel();
            app.approval_reply = Some(ApprovalReply::new(tx, None));
            app.approval_shown = Some(Instant::now() - APPROVAL_SETTLE);
            if enter { app.input.set_typed("4".into()); app.key(Key::Enter).unwrap(); }
            else { app.key(Key::Text("4".into())).unwrap(); }
            assert_eq!(rx.try_recv(), Ok(true));
            assert_eq!(approvals.take_choice(), scope);
            let _ = std::fs::remove_dir_all(folder);
        }
    }
}
