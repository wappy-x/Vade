//! Real CLI against local HTTP mocks and temporary workspaces only: tools,
//! approvals, cutoffs, resume and child grants.
#[path = "support/home.rs"]
mod home;
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/coding.rs"]
mod coding;
use coding::*;
#[test]
fn prompt_discloses_approval_session_shell_and_child_grants_on_stderr() {
    let f = Fixture::new();
    let (url, server) = mock(vec![answer()]);
    let output = run_bounded(
        f.command(&url).args([
            "--session",
            "startup-policy",
            "--prompt",
            "check",
            "--children",
        ]),
        "",
    );
    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout), "mock complete\n");
    let stderr = String::from_utf8_lossy(&output.stderr);
    for label in [
        "Approval: ask",
        "Session: startup-policy",
        "--resume startup-policy",
        "Shell build: private session build/cache/temp area",
        "Cargo offline",
        "Children enabled",
        "No named child scopes. Unnamed children are read-only.",
    ] {
        assert!(stderr.contains(label), "missing {label:?}: {stderr}");
    }
    server.join().unwrap();
}
#[test]
fn native_cutoff_is_visible_and_resumed_without_partial_tool_calls() {
    let f = Fixture::new();
    let cutoff = r#"{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"partial answer 🦀","tool_calls":[{"function":{"arguments":"{"}}]}}],"usage":{"completion_tokens":17}}"#;
    let (url, server) = mock(vec![cutoff.into(); 4]);
    let output = run_bounded(
        f.command(&url).args([
            "--session",
            "cutoff",
            "--prompt",
            "check",
            "--max-output-tokens",
            "17",
            "--thinking",
            "off",
        ]),
        "",
    );
    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "partial answer 🦀\n\npartial answer 🦀\n\npartial answer 🦀\n\npartial answer 🦀\n"
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("Vade did not run the cut-off call."));
    let requests = server.join().unwrap();
    assert_eq!(
        requests[0].get("max_tokens"),
        Some(&Value::Number("17".into()))
    );
    assert_eq!(
        requests[0].get("enable_thinking"),
        Some(&Value::Bool(false))
    );
    let (url, server) = mock(vec![answer()]);
    let output = run_bounded(
        f.command(&url)
            .args(["--resume", "cutoff", "--prompt", "continue"]),
        "",
    );
    assert!(output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("Journal records 4 reply cutoff(s) at an upstream length limit")
    );
    let requests = server.join().unwrap();
    let messages = requests[0].get("messages").unwrap().as_array().unwrap();
    let retained = messages
        .iter()
        .find(|m| m.get("content").and_then(Value::as_str) == Some("partial answer 🦀"))
        .unwrap();
    assert_eq!(
        retained.get("content").and_then(Value::as_str),
        Some("partial answer 🦀")
    );
    assert_eq!(
        retained.get("role").and_then(Value::as_str),
        Some("assistant")
    );
    let Value::Object(fields) = retained else {
        panic!("assistant must be an object")
    };
    assert_eq!(fields.len(), 2);
    assert!(fields.iter().all(|(key, _)| !key.starts_with("vade_")));
    assert!(retained.get("function_call").is_none());
    assert!(retained.get("tool_calls").is_none());
}
#[test]
fn native_run_prints_one_speed_summary_and_status_has_response_details() {
    let f = Fixture::new();
    std::fs::write(f.0.join("input.txt"), "local test input").unwrap();
    let (url, server) = mock(vec![
        tools(vec![call(
            "read",
            "workspace_read",
            json::parse(r#"{"path":"input.txt"}"#).unwrap(),
        )]),
        answer(),
    ]);
    let output = run_bounded(f.command(&url).arg("--plain"), "read it\n/status\n/quit");
    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    let (run, status) = stderr.split_once("Session average:").unwrap();
    assert_eq!(run.matches("╰ ").count(), 1);
    assert!(run.contains("2 model calls"));
    assert!(!run.contains("Response speed:"));
    assert_eq!(status.matches("Response speed:").count(), 2);
    assert_eq!(server.join().unwrap().len(), 2);
}
#[test]
fn invalid_tool_argument_json_is_recoverable_and_safe_to_replay() {
    let f = Fixture::new();
    // Valid response JSON containing model-written invalid argument JSON, like
    // an unescaped triple quote in a patch. Admission/rendering mocks require
    // both the request envelope and every nested argument string to parse.
    let malformed =
        r#"{"patch":"*** Begin Patch\n*** Add File: rejected.txt\n+"""\n*** End Patch"}"#;
    assert!(json::parse(malformed).is_err());
    let (url, server) = mock(vec![
        tools(vec![call_with_arguments(
            "bad-edit",
            "apply_patch",
            malformed,
        )]),
        tools(vec![call("retry-edit", "apply_patch", patch())]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url).args([
            "--approval",
            "auto",
            "--session",
            "bad-arguments",
            "--prompt",
            "make the file",
        ]),
        "",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!f.0.join("rejected.txt").exists());
    assert_eq!(
        std::fs::read_to_string(f.0.join("result.txt")).unwrap(),
        "exact result\n"
    );
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 3);
    let messages = requests[1].get("messages").unwrap().as_array().unwrap();
    let result = messages
        .iter()
        .find(|message| message.get("tool_call_id").and_then(Value::as_str) == Some("bad-edit"))
        .unwrap();
    let content = json::parse(result.get("content").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(
        content.get("error").and_then(Value::as_str),
        Some("invalid_arguments")
    );
    assert_eq!(
        content.get("message").and_then(Value::as_str),
        Some("arguments must be a JSON object")
    );
    let (url, server) = mock(vec![answer()]);
    let output = run_bounded(
        f.command(&url)
            .args(["--resume", "bad-arguments", "--prompt", "continue"]),
        "",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let resumed = server.join().unwrap();
    for request in requests.iter().chain(&resumed) {
        for message in request.get("messages").unwrap().as_array().unwrap() {
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let arguments = call.get("function").unwrap().get("arguments").unwrap();
                    let text = arguments
                        .as_str()
                        .expect("arguments must remain a JSON string");
                    json::parse(text).expect("upstream must be able to parse argument JSON");
                    if call.get("id").and_then(Value::as_str) == Some("bad-edit") {
                        assert_eq!(text, "{}");
                    }
                }
            }
        }
    }
}
#[test]
fn native_cli_shows_queued_and_cancels_without_an_upstream_response() {
    let f = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let (sent, dispatched) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        let start = Instant::now();
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(start.elapsed() < Duration::from_secs(5));
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("mock accept failed: {error}"),
            }
        };
        request(&mut stream);
        sent.send(()).unwrap();
        assert_eq!(stream.read(&mut [0]).unwrap(), 0);
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    });
    let mut child = cli::Cli::start(f.command(&url).arg("--plain"));
    child.send("wait in the mock queue\n");
    dispatched.recv_timeout(Duration::from_secs(5)).unwrap();
    child.send("/status\n");
    child.until(false, "native agent · queued / awaiting upstream");
    let cancelled = Instant::now();
    child.send("/cancel\n");
    child.until(true, "Native run:");
    child.send("/status\n");
    child.until(false, "local cleanup complete · remote stop unconfirmed");
    child.send("/quit\n");
    let output = child.finish();
    server.join().unwrap();
    assert!(cancelled.elapsed() < Duration::from_secs(3));
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("mock complete"));
    // /status names only the tools this platform offers the model.
    let shell = vade_harness::coding_tools::shell_offered();
    assert!(stdout.contains(if shell { "apply_patch/workspace_shell/tool_output_read" } else { "apply_patch/tool_output_read" }), "{stdout}");
    assert!(stdout.contains(if shell {
        "macOS sandbox · network denied"
    } else if vade_harness::coding_tools::SHELL_AVAILABLE {
        "no command tool: VADE_HARNESS_SHELL=off"
    } else {
        "no command tool here: workspace_shell needs the macOS sandbox"
    }), "{stdout}");
}
#[test]
fn auto_patch_allowlisted_command_and_disk_resume_keep_exact_tool_ids() {
    let f = Fixture::new();
    let (url, handle) = mock(vec![
        tools(vec![
            call("edit-1", "apply_patch", patch()),
            call(
                "shell-1",
                "workspace_shell",
                json::parse(r#"{"argv":["/usr/bin/true"],"timeout_ms":1000}"#).unwrap(),
            ),
        ]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url).args([
            "--approval",
            "auto",
            "--session",
            "roundtrip",
            "--plain",
            "--prompt",
            "make the file",
        ]),
        "",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(f.0.join("result.txt")).unwrap(),
        "exact result\n"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("+++ result.txt"));
    assert!(stderr.contains("+exact result"));
    let requests = handle.join().unwrap();
    let messages = requests[1].get("messages").unwrap().as_array().unwrap();
    for id in ["edit-1", "shell-1"] {
        assert!(
            messages
                .iter()
                .any(|m| m.get("tool_call_id").and_then(Value::as_str) == Some(id))
        );
    }
    let (url, handle) = mock(vec![answer()]);
    let output = run_bounded(
        f.command(&url)
            .args(["--resume", "roundtrip", "--approval", "never", "--plain"]),
        "/context\n/sessions\ncontinue\n/new\n/sessions\n/quit\n",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let resumed = handle.join().unwrap();
    let messages = resumed[0].get("messages").unwrap().as_array().unwrap();
    assert!(
        messages
            .iter()
            .any(|m| m.get("tool_call_id").and_then(Value::as_str) == Some("edit-1"))
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("message 0"));
    assert!(stdout.contains("roundtrip"));
    assert_eq!(
        std::fs::read_dir(f.0.join(".vade/sessions"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|s| s == "journal"))
            .count(),
        2
    );
}
#[test]
fn noninteractive_ask_runs_reads_and_denies_edits_while_never_denies_both() {
    for policy in ["ask", "never"] {
        let f = Fixture::new();
        let (url, handle) = mock(vec![
            tools(vec![
                call("edit-1", "apply_patch", patch()),
                call(
                    "shell-1",
                    "workspace_shell",
                    json::parse(r#"{"argv":["/usr/bin/true"],"timeout_ms":1000}"#).unwrap(),
                ),
            ]),
            answer(),
        ]);
        let output = run_bounded(
            f.command(&url)
                .args(["--approval", policy, "--prompt", "try edit"]),
            "",
        );
        assert!(output.status.success());
        assert!(!f.0.join("result.txt").exists());
        let requests = handle.join().unwrap();
        let messages = requests[1].get("messages").unwrap().as_array().unwrap();
        let denied: Vec<_> = messages
            .iter()
            .filter(|m| m.get("role").and_then(Value::as_str) == Some("tool"))
            .collect();
        assert_eq!(denied.len(), 2);
        let content = |i: usize| denied[i].get("content").unwrap().as_str().unwrap();
        assert!(content(0).contains("denied"));
        // Without the shell (Linux, or VADE_HARNESS_SHELL=off) the tool is not offered.
        let expected = if !vade_harness::coding_tools::shell_offered() {
            "unknown_tool"
        } else if policy == "ask" {
            "Exited(Some(0))"
        } else {
            "denied"
        };
        assert!(content(1).contains(expected));
    }
}
#[test]
fn unreachable_upstream_uses_presave_estimate_without_poisoning_resume() {
    let f = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    drop(listener);
    let oversized = "unreachable-oversize-".to_owned() + &"x".repeat(31_980);
    let output = run_bounded(
        f.command(&url).args([
            "--plain",
            "--session",
            "unavailable",
            "--context-tokens",
            "16384",
            "--prompt",
            &oversized,
        ]),
        "",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("it was not sent"));
    let raw = std::fs::read(f.0.join(".vade/sessions/unavailable.journal")).unwrap();
    assert!(!raw.windows(20).any(|w| w == b"unreachable-oversize-"));
    let (url, server) = mock(vec![answer()]);
    let next = run_bounded(
        f.command(&url).args([
            "--plain",
            "--resume",
            "unavailable",
            "--context-tokens",
            "16384",
            "--prompt",
            "tiny follow-up",
        ]),
        "",
    );
    assert!(
        next.status.success(),
        "{}",
        String::from_utf8_lossy(&next.stderr)
    );
    assert_eq!(server.join().unwrap().len(), 1);
}
#[cfg(target_os = "macos")]
#[test]
fn interactive_tty_approval_previews_exact_diff_before_writing() {
    use std::os::fd::FromRawFd;
    unsafe extern "C" {
        fn openpty(
            master: *mut i32,
            slave: *mut i32,
            name: *mut i8,
            term: *const std::ffi::c_void,
            size: *const std::ffi::c_void,
        ) -> i32;
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    for (raw, command_tool, cancel, build) in [
        (false, false, false, false),
        (true, false, false, false),
        (true, true, false, false),
        (true, true, true, false),
        (true, true, false, true),
    ] {
        let f = Fixture::new();
        let patched = "exact result\u{200b}\t\u{202e}\njoin\u{200d}\u{fe0f}\nspace\u{00a0}\ncontrol\u{001b}\nliteral \\u{200b}\n";
        if build {
            std::fs::create_dir(f.0.join("src")).unwrap();
            std::fs::write(f.0.join("Cargo.toml"), "[package]\nname = \"cli-sandbox-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n").unwrap();
            std::fs::write(
                f.0.join("src/lib.rs"),
                "#[test] fn succeeds() { assert_eq!(2 + 3, 5); }\n",
            )
            .unwrap();
        }
        let tool = if command_tool {
            call("shell-1", "workspace_shell", Value::Object(vec![
                ("argv".into(), Value::Array(["/bin/sh", "-c", if build { "cargo build --offline && cargo test --offline" } else { r#"printf '%s' "${VADE_MOCK_SECRET-unset}"; printf shell > shell.txt"# }].into_iter().map(|s| Value::String(s.into())).collect())),
                ("timeout_ms".into(), Value::Number(if build { "120000" } else { "1000" }.into()))
            ]))
        } else {
            call(
                "edit-1",
                "apply_patch",
                Value::Object(vec![(
                    "patch".into(),
                    Value::String(format!(
                        "*** Begin Patch\n*** Add File: result.txt\n{}*** End Patch",
                        patched
                            .split_inclusive('\n')
                            .map(|line| format!("+{line}"))
                            .collect::<String>()
                    )),
                )]),
            )
        };
        let replies = if cancel {
            vec![tools(vec![tool])]
        } else {
            let mut replies = vec![tools(vec![tool]), answer()];
            if !command_tool {
                replies.push(answer());
            }
            replies
        };
        let (url, handle) = mock(replies);
        let filename = if build {
            "Cargo.lock"
        } else if command_tool {
            "shell.txt"
        } else {
            "result.txt"
        };
        let mut master = 0;
        let mut slave = 0;
        assert_eq!(
            unsafe {
                openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                )
            },
            0
        );
        let mut terminal = unsafe { std::fs::File::from_raw_fd(master) };
        let slave = unsafe { std::fs::File::from_raw_fd(slave) };
        let flags = unsafe { fcntl(master, 3) };
        assert_eq!(unsafe { fcntl(master, 4, flags | 4) }, 0);
        let mut command = f.command(&url);
        // This macOS-only test needs the shell offered whatever the caller set.
        command.env_remove("VADE_HARNESS_SHELL");
        if build {
            let cargo = std::path::Path::new(env!("CARGO")).canonicalize().unwrap();
            command
                .arg("--shell-toolchain")
                .arg(cargo.parent().unwrap().parent().unwrap());
        }
        if !raw {
            command.arg("--plain");
        }
        let child = command
            .args(["--approval", "ask"])
            .env("TERM", "xterm-256color")
            .env("VADE_MOCK_SECRET", "synthetic-inherited-marker")
            .stdin(slave.try_clone().unwrap())
            .stdout(slave.try_clone().unwrap())
            .stderr(slave)
            .spawn()
            .unwrap();
        struct ChildGuard(std::process::Child);
        impl Drop for ChildGuard {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let mut child = ChildGuard(child);
        fn until(file: &mut std::fs::File, needle: &str) -> String {
            let start = Instant::now();
            let mut text = String::new();
            let mut buf = [0u8; 4096];
            while !text.contains(needle) {
                match file.read(&mut buf) {
                    Ok(n) if n > 0 => text.push_str(&String::from_utf8_lossy(&buf[..n])),
                    Ok(_) => (),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => (),
                    Err(e) => panic!("pty: {e}; {text}"),
                }
                assert!(
                    start.elapsed() < Duration::from_secs(8),
                    "PTY deadline waiting for {needle}: {text}"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            text
        }
        until(&mut terminal, "Session:");
        let eol = if raw { "\r" } else { "\n" };
        terminal
            .write_all(format!("make the file{eol}").as_bytes())
            .unwrap();
        let preview = until(&mut terminal, "Approval required");
        assert!(preview.contains(if command_tool {
            "command argv="
        } else {
            "+++ result.txt"
        }));
        if !command_tool {
            for escaped in [
                r"\u{200b}",
                r"\t",
                r"\u{202e}",
                r"\u{200d}",
                r"\u{fe0f}",
                r"\u{a0}",
                r"\u{1b}",
                r"literal \\u{200b}",
            ] {
                assert!(preview.contains(escaped), "missing {escaped}: {preview}");
            }
            for invisible in ['\u{200b}', '\u{202e}', '\u{200d}', '\u{fe0f}', '\u{a0}'] {
                assert!(!preview.contains(invisible), "raw invisible in approval");
            }
        }
        assert!(!f.0.join(filename).exists());
        if raw {
            for key in [b"\x1b".as_slice(), b"\x03".as_slice()] {
                terminal.write_all(key).unwrap();
                let waiting = until(&mut terminal, "Approval still pending;");
                assert!(!waiting.contains("Native cancellation requested"));
                assert!(!f.0.join(filename).exists());
                assert!(child.0.try_wait().unwrap().is_none());
            }
        }
        if raw && !cancel {
            terminal
                .write_all(b"\x1b[200~pasted first line\npasted second line\x1b[201~\r")
                .unwrap();
            until(&mut terminal, "Approval still pending.");
            assert!(!f.0.join(filename).exists());
            terminal.write_all(b"\x15").unwrap();
        }
        if cancel {
            terminal.write_all(b"/cancel\r").unwrap();
        } else {
            terminal
                .write_all(format!("/approve{eol}").as_bytes())
                .unwrap();
        }
        let finished = until(&mut terminal, "Native run:");
        if cancel {
            assert!(finished.contains("pending operation will be declined"));
        }
        assert_eq!(f.0.join(filename).exists(), !cancel);
        if !command_tool && !cancel {
            assert_eq!(
                std::fs::read(f.0.join(filename)).unwrap(),
                patched.as_bytes()
            );
            let absolute = f.0.canonicalize().unwrap().join(filename);
            // The responsive terminal can wrap a long absolute path across rows.
            let mut unwrapped = String::new();
            let mut chars = finished.chars();
            while let Some(c) = chars.next() {
                if c == '\x1b' {
                    if chars.next() == Some('[') {
                        for c in chars.by_ref() {
                            if ('@'..='~').contains(&c) {
                                break;
                            }
                        }
                    }
                } else if !matches!(c, '\r' | '\n') {
                    unwrapped.push(c);
                }
            }
            assert!(unwrapped.contains(&absolute.display().to_string()));
            terminal
                .write_all(format!("/retry{eol}").as_bytes())
                .unwrap();
            until(&mut terminal, "Native run:");
        }
        terminal
            .write_all(format!("/quit{eol}").as_bytes())
            .unwrap();
        let start = Instant::now();
        while child.0.try_wait().unwrap().is_none() {
            // A full-screen redraw can still be draining after the summary marker.
            let _ = terminal.read(&mut [0; 4096]);
            assert!(
                start.elapsed() < Duration::from_secs(3),
                "exit stalled: raw={raw}, command={command_tool}, cancel={cancel}, build={build}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let requests = handle.join().unwrap();
        if !command_tool {
            assert!(requests.iter().all(|r| {
                !json::stringify(r)
                    .unwrap()
                    .contains(&f.0.display().to_string())
            }));
            if !cancel {
                assert!(requests.iter().all(|request| request.get("seed").is_none()), "default retries must keep the upstream no-field policy");
                assert_eq!(
                    requests[2].get("tool_choice"),
                    Some(&Value::String("none".into()))
                );
                assert_eq!(requests[1].get("messages"), requests[2].get("messages"));
            }
        }
        if command_tool && !cancel {
            let messages = requests[1].get("messages").unwrap().as_array().unwrap();
            let result = messages
                .iter()
                .find(|m| m.get("tool_call_id").and_then(Value::as_str) == Some("shell-1"))
                .unwrap()
                .get("content")
                .unwrap()
                .as_str()
                .unwrap();
            assert!(result.contains("status: Exited(Some(0))"), "{result}");
            assert!(
                result.contains(if build { "1 passed; 0 failed" } else { "unset" }),
                "{result}"
            );
            assert!(!result.contains("synthetic-inherited-marker"));
        }
    }
}
#[test]
fn children_opt_in_retains_parent_native_tools_approvals_and_durable_receipts() {
    for policy in ["auto", "never"] {
        let f = Fixture::new();
        let (url, server) = mock(vec![
            tools(vec![
                call("parent-edit", "apply_patch", patch()),
                call(
                    "parent-shell",
                    "workspace_shell",
                    json::parse(r#"{"argv":["/usr/bin/true"],"timeout_ms":1000}"#).unwrap(),
                ),
                call(
                    "parent-map",
                    "workspace_manifest",
                    json::parse(r#"{"path":".","depth":0}"#).unwrap(),
                ),
            ]),
            answer(),
        ]);
        let output = run_bounded(
            f.command(&url).args([
                "--children",
                "--approval",
                policy,
                "--session",
                "parent-with-children",
                "--plain",
                "--prompt",
                "Use native tools",
            ]),
            "",
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(f.0.join("result.txt").exists(), policy == "auto");
        let requests = server.join().unwrap();
        let messages = requests
            .last()
            .unwrap()
            .get("messages")
            .unwrap()
            .as_array()
            .unwrap();
        for id in ["parent-edit", "parent-shell", "parent-map"] {
            let result = messages
                .iter()
                .find(|m| m.get("tool_call_id").and_then(Value::as_str) == Some(id))
                .unwrap()
                .get("content")
                .unwrap()
                .as_str()
                .unwrap();
            // Without the shell (Linux, or VADE_HARNESS_SHELL=off) the parent
            // is not offered it, so that one call is an unknown tool.
            let unoffered_shell = id == "parent-shell" && !vade_harness::coding_tools::shell_offered();
            assert_eq!(result.contains("unknown_tool"), unoffered_shell, "{result}");
            assert!(!result.contains("unsupported_platform"), "{result}");
            assert_eq!(
                result.contains("denied"),
                policy == "never" && id != "parent-map" && !unoffered_shell
            );
        }
        let workspace = vade_harness::tools::Workspace::new(&f.0).unwrap();
        let (_, replay, _) = vade_harness::journal::Journal::resume(
            &workspace,
            "parent-with-children",
            vade_harness::agent::Limits::default(),
        )
        .unwrap();
        for id in ["parent-edit", "parent-shell", "parent-map"] {
            assert!(
                replay
                    .messages()
                    .iter()
                    .any(|message| message.tool_call_id() == Some(id))
            );
        }
    }
}
#[test]
fn resume_requires_explicit_matching_child_scopes_and_uses_current_approval_policy() {
    let f = Fixture::new();
    let (url, server) = mock(vec![answer()]);
    let output = run_bounded(
        f.command(&url).args([
            "--children",
            "--child-edit",
            "edits=result.txt",
            "--approval",
            "auto",
            "--session",
            "scoped",
            "--prompt",
            "start",
        ]),
        "",
    );
    assert!(output.status.success());
    server.join().unwrap();
    for args in [
        vec![],
        vec!["--children"],
        vec!["--children", "--child-edit", "edits=outside"],
    ] {
        let mut command = f.command("http://127.0.0.1:1");
        command
            .args(["--resume", "scoped", "--prompt", "must not send"])
            .args(args);
        let output = run_bounded(&mut command, "");
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("This session was saved with ") && stderr.contains("or start a new session."), "{stderr}");
    }
    let (url, server) = mock(vec![
        tools(vec![call("current-policy", "apply_patch", patch())]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url).args([
            "--children",
            "--child-edit",
            "edits=result.txt",
            "--approval",
            "never",
            "--resume",
            "scoped",
            "--prompt",
            "continue",
        ]),
        "",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !f.0.join("result.txt").exists(),
        "saved auto policy must not be restored"
    );
    let requests = server.join().unwrap();
    assert!(
        requests
            .last()
            .unwrap()
            .get("messages")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .any(
                |message| message.get("tool_call_id").and_then(Value::as_str)
                    == Some("current-policy")
                    && message
                        .get("content")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .contains("denied")
            )
    );
}
