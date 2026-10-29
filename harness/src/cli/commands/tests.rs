//! Slash-command effects on native conversation state.
use super::*;
use crate::cli::{args::Parsed, test_support::parse};

#[test]
fn user_commands_builtin_priority_escape_and_unknown_do_not_use_network() {
    let root = std::env::temp_dir().join(format!("vade-user-commands-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let Parsed::Run(config) = parse(&["demo", "--plain", "--workspace", root.to_str().unwrap()]).unwrap() else { panic!() };
    let mut app = App::new(config).unwrap();
    assert!(app.user_commands().unwrap().is_empty());
    let dir = app.workspace.state_path().join("commands");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("help.md"), "wrong builtin override").unwrap();
    std::fs::write(dir.join("clear.md"), "wrong clear override").unwrap();
    std::fs::write(dir.join("review.md"), "description: Review files\nReview $ARGUMENTS").unwrap();
    app.line_input("/help".into()).unwrap();
    assert!(app.pending.is_none());
    app.session.complete("old".into(), "answer".into()).unwrap();
    app.line_input("/clear".into()).unwrap();
    assert_eq!(app.session.turns(), 0);
    app.line_input("/memory".into()).unwrap();
    app.line_input("/unknown-command".into()).unwrap();
    assert!(app.pending.is_none());
    let commands = app.user_commands().unwrap();
    let names = app.command_names(&commands);
    assert!(names.contains("/review"));
    assert_eq!(names.matches("/help").count(), 1);
    app.line_input("/review file.rs".into()).unwrap();
    assert_eq!(app.pending.as_ref().unwrap().user, "Review file.rs");
    drop(app.pending.take());
    app.line_input("//review literal".into()).unwrap();
    assert_eq!(app.pending.as_ref().unwrap().user, "/review literal");
    drop(app);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn bare_compact_uses_the_automatic_plan_and_refuses_an_empty_history() {
    let workspace =
        std::env::temp_dir().join(format!("vade-bare-compact-{}", std::process::id()));
    std::fs::create_dir(&workspace).unwrap();
    let Parsed::Run(config) = parse(&[
        "agent",
        "--plain",
        "--endpoint",
        "http://localhost:1",
        "--model",
        "mock",
        "--workspace",
        workspace.to_str().unwrap(),
    ])
    .unwrap() else {
        panic!()
    };
    let mut app = App::new(config).unwrap();
    let before = app.native.as_ref().unwrap().messages().len();
    app.line_input("/compact".into()).unwrap();
    app.line_input("/compact-preview".into()).unwrap();
    // Nothing closed is eligible yet: history and journal stay as they were.
    assert_eq!(app.native.as_ref().unwrap().messages().len(), before);
    drop(app);
    std::fs::remove_dir_all(workspace).unwrap();
}
