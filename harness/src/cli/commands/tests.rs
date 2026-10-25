//! Slash-command effects on native conversation state.
use super::*;
use crate::cli::{args::Parsed, test_support::parse};

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
