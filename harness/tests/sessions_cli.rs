//! Saved sessions through the real CLI against a local HTTP mock: a named
//! chat, the session list, a resume by name that replays the history, the
//! Markdown export and delete.
#[path = "support/home.rs"]
mod home;
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/coding.rs"]
mod coding;
use coding::*;

fn sessions(f: &Fixture, args: &[&str]) -> std::process::Output {
    home::harness()
        .arg("sessions")
        .args(args)
        .args(["--state-in-project", "--workspace"])
        .arg(&f.0)
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn named_chat_is_listed_resumed_by_name_exported_and_deleted() {
    let f = Fixture::new();
    let (url, server) = mock(vec![answer()]);
    let first = run_bounded(f.command(&url).args(["--name", "login bug", "--plain", "--prompt", "first question"]), "");
    assert!(first.status.success(), "{}", text(&first.stderr));
    server.join().unwrap();

    let listed = sessions(&f, &[]);
    assert!(listed.status.success(), "{}", text(&listed.stderr));
    let table = text(&listed.stdout);
    let row = table.lines().nth(1).unwrap();
    assert!(row.starts_with("session-") && row.contains("login bug") && row.contains(" 1 ") && row.contains("mock"), "{table}");
    let id = row.split_whitespace().next().unwrap().to_string();

    // A resume by name sends the saved turn back to the model before the new prompt.
    let (url, server) = mock(vec![answer()]);
    let resumed = run_bounded(f.command(&url).args(["--resume", "login bug", "--plain", "--prompt", "second question"]), "");
    assert!(resumed.status.success(), "{}", text(&resumed.stderr));
    let requests = server.join().unwrap();
    let users: Vec<_> = requests[0].get("messages").unwrap().as_array().unwrap().iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .collect();
    assert_eq!(users.first(), Some(&"first question"));
    assert_eq!(users.last(), Some(&"second question"));
    assert!(requests[0].get("messages").unwrap().as_array().unwrap().iter()
        .any(|m| m.get("content").and_then(Value::as_str) == Some("mock complete")));

    let export = sessions(&f, &["export", "last", "--markdown"]);
    assert!(export.status.success(), "{}", text(&export.stderr));
    let markdown = text(&export.stdout);
    assert!(markdown.starts_with(&format!("# Session login bug\n\n- ID: `{id}`\n")), "{markdown}");
    assert!(markdown.contains("- Turns: 2\n"), "{markdown}");
    let order = ["## User\n\nfirst question", "mock complete", "*The session was resumed here.", "## User\n\nsecond question"].map(|s| markdown.find(s).unwrap_or_else(|| panic!("{s}: {markdown}")));
    assert!(order.windows(2).all(|w| w[0] < w[1]), "{markdown}");

    let deleted = sessions(&f, &["delete", &id]);
    assert!(deleted.status.success(), "{}", text(&deleted.stderr));
    assert_eq!(text(&deleted.stdout), format!("Deleted session {id}.\n"));
    assert!(text(&sessions(&f, &[]).stdout).starts_with("No saved sessions."));
    let missing = sessions(&f, &["export", &id]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(text(&missing.stderr).contains("no saved session in this project has that ID or name"));
    assert_eq!(sessions(&f, &["delete"]).status.code(), Some(2));
}
