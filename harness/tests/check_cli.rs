//! `--check` in real command-line agent runs, with and without children:
//! the configured check runs before the run may finish. Local HTTP mocks and
//! temporary workspaces only.
#[path = "support/home.rs"]
mod home;
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/coding.rs"]
mod coding;
use coding::*;

/// The tool results the model saw in one request whose call id starts with
/// the harness check prefix.
fn check_receipts(request: &Value) -> Vec<String> {
    request
        .get("messages")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("tool"))
        .filter(|m| {
            m.get("tool_call_id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with("vade-check-"))
        })
        .map(|m| m.get("content").and_then(Value::as_str).unwrap_or("").to_owned())
        .collect()
}

#[test]
fn check_runs_before_a_command_line_agent_run_finishes() {
    // --check needs the macOS sandbox; elsewhere the CLI refuses the flag.
    if !vade_harness::coding_tools::shell_offered() {
        return;
    }
    for children in [false, true] {
        let f = Fixture::new();
        // The model answers at once without touching a file. The completion
        // check must still run, and the model sees its receipt next turn.
        let (url, server) = mock(vec![answer(), answer()]);
        let mut command = f.command(&url);
        command.args(["--approval", "auto", "--check", "true", "--prompt", "say done"]);
        if children {
            command.arg("--children");
        }
        let output = run_bounded(&mut command, "");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "children={children}: {stderr}");
        let requests = server.join().unwrap_or_else(|_| {
            panic!("children={children}: the run ended after one request; the check was skipped: {stderr}")
        });
        assert_eq!(requests.len(), 2, "children={children}");
        assert!(check_receipts(&requests[0]).is_empty());
        let receipts = check_receipts(&requests[1]);
        assert_eq!(receipts.len(), 1, "children={children}: {receipts:?}");
        assert!(receipts[0].contains("Exited(Some(0))"), "children={children}: {receipts:?}");
    }
}
