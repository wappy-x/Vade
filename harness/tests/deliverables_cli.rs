//! A normal command-line run does NOT check that requested files exist.
//!
//! This is on purpose. The file detector in `src/submission.rs` is a word
//! heuristic. On free user text it fails correct runs: "Write a small HTTP server
//! in Node.js." asked for a file named `Node.js`, and "Add tests for the login
//! function." asked for a Rust test in a project with no Rust. So the three
//! executor wrappers of the command line do not forward `missing_deliverables`
//! (see `tests/tool_wrappers.rs`), and no run reaches the detector; the agent
//! suite starts the same command-line agent. A strict detector for normal runs
//! is a follow-up. These tests pin today's behavior: one model turn, exit 0, no
//! continuation, even when a named file is not created.
#[path = "support/home.rs"]
mod home;
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/coding.rs"]
mod coding;
use coding::*;

/// Runs `prompt` against a mock that gives exactly one final answer. A second
/// model request (a continuation turn) finds no server and fails the run.
fn one_turn(prompt: &str, children: bool) -> Fixture {
    let f = Fixture::new();
    let (url, server) = mock(vec![answer()]);
    let mut command = f.command(&url);
    command.args(["--prompt", prompt]);
    if children {
        command.arg("--children");
    }
    let output = run_bounded(&mut command, "");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{prompt}: {stderr}");
    assert!(!stderr.contains("required deliverables"), "{prompt}: {stderr}");
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 1, "{prompt}: no continuation turn");
    let messages = requests[0].get("messages").unwrap().as_array().unwrap();
    assert!(!messages.iter().any(|m| m
        .get("content")
        .and_then(Value::as_str)
        .is_some_and(|s| s.contains("Missing required deliverables"))));
    f
}

#[test]
fn a_missing_named_file_does_not_fail_a_normal_run() {
    for children in [false, true] {
        let f = one_turn("Create answers.txt.", children);
        assert!(!f.0.join("answers.txt").exists());
    }
}

#[test]
fn product_names_and_test_requests_get_no_extra_turn() {
    for prompt in ["Write a small HTTP server in Node.js.", "Add tests for the login function."] {
        for children in [false, true] {
            let f = one_turn(prompt, children);
            assert!(!f.0.join("Node.js").exists());
        }
    }
}
