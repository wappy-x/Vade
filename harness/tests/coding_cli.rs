//! Real CLI against local HTTP mocks and temporary workspaces only.
#[path = "support/cli.rs"]
mod cli;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::time::{Duration, Instant};
use vade_harness::json::{self, Value};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "vade-cli-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn command(&self, url: &str) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        c.args([
            "agent",
            "--state-in-project",
            "--verbose",
            "--endpoint",
            url,
            "--model",
            "mock",
            "--workspace",
        ])
        .arg(&self.0)
        .env("NO_COLOR", "1");
        c
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn request(stream: &mut TcpStream) -> Value {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut byte = [0];
    while !bytes.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
        assert!(bytes.len() < 64 * 1024);
    }
    let headers = std::str::from_utf8(&bytes).unwrap();
    if headers.starts_with("GET /v1/capabilities HTTP/1.1\r\n") {
        // Optional contract discovery is unsupported by these standard-chat
        // mocks. Keep the connection framed/reusable for the generation POST.
        stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n{}").unwrap();
        return request(stream);
    }
    let length: usize = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|n| n.trim().parse().unwrap())
        })
        .unwrap();
    assert!(length <= 4 * 1024 * 1024);
    let mut body = vec![0; length];
    stream.read_exact(&mut body).unwrap();
    let request = json::parse_bytes(&body).unwrap();
    for message in request.get("messages").unwrap().as_array().unwrap() {
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let arguments = call.get("function").unwrap().get("arguments").unwrap();
                json::parse(arguments.as_str().expect("arguments must be a JSON string"))
                    .expect("mock upstream must be able to render argument JSON");
            }
        }
    }
    request
}
fn response(stream: &mut TcpStream, body: &str) {
    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
}
fn answer() -> String {
    r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"mock complete"}}]}"#.into()
}
fn call(id: &str, name: &str, args: Value) -> Value {
    call_with_arguments(id, name, &json::stringify(&args).unwrap())
}
fn call_with_arguments(id: &str, name: &str, arguments: &str) -> Value {
    Value::Object(vec![
        ("id".into(), Value::String(id.into())),
        ("type".into(), Value::String("function".into())),
        (
            "function".into(),
            Value::Object(vec![
                ("name".into(), Value::String(name.into())),
                ("arguments".into(), Value::String(arguments.into())),
            ]),
        ),
    ])
}
fn tools(calls: Vec<Value>) -> String {
    json::stringify(&Value::Object(vec![(
        "choices".into(),
        Value::Array(vec![Value::Object(vec![
            ("finish_reason".into(), Value::String("tool_calls".into())),
            (
                "message".into(),
                Value::Object(vec![
                    ("role".into(), Value::String("assistant".into())),
                    ("tool_calls".into(), Value::Array(calls)),
                ]),
            ),
        ])]),
    )]))
    .unwrap()
}
fn patch() -> Value {
    Value::Object(vec![(
        "patch".into(),
        Value::String(
            "*** Begin Patch\n*** Add File: result.txt\n+exact result\n*** End Patch".into(),
        ),
    )])
}
fn mock(replies: Vec<String>) -> (String, std::thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    listener.set_nonblocking(true).unwrap();
    let handle = std::thread::spawn(move || {
        let mut requests = Vec::new();
        for reply in replies {
            let start = Instant::now();
            let mut stream = loop {
                match listener.accept() {
                    Ok((s, _)) => break s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            start.elapsed() < Duration::from_secs(8),
                            "mock request deadline"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("mock accept: {e}"),
                }
            };
            requests.push(request(&mut stream));
            response(&mut stream, &reply);
        }
        requests
    });
    (url, handle)
}
fn run_bounded(command: &mut Command, input: &str) -> std::process::Output {
    let mut child = cli::Cli::start(command);
    for line in input.lines() {
        child.send(&format!("{line}\n"));
        // Main's control lane intentionally consumes /quit and context commands
        // during work. Fixture scripts must wait for a turn receipt before
        // sending a command that depends on that turn's completed journal.
        if !line.is_empty() && !line.starts_with('/') {
            child.until(true, "Native run:");
        } else if line.starts_with("/compact ") || line == "/compact" {
            child.until(false, "Context compacted.");
        }
    }
    child.finish()
}

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
    assert!(!String::from_utf8_lossy(&output.stdout).contains("mock complete"));
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
        let expected = if !cfg!(target_os = "macos") {
            "unsupported_platform"
        } else if policy == "ask" {
            "Exited(Some(0))"
        } else {
            "denied"
        };
        assert!(content(1).contains(expected));
    }
}
#[test]
fn context_budget_refuses_prompt_without_contacting_mock() {
    let f = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/custom/rpc", listener.local_addr().unwrap()); // no negotiated tokenizer on custom routes
    let output = run_bounded(
        f.command(&url).args([
            "--context-tokens",
            "8192",
            "--plain",
            "--prompt",
            &"x".repeat(32000),
        ]),
        "",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("it was not sent"));
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
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
fn explicit_cli_compaction_preserves_journal_and_latest_whole_turn() {
    let f = Fixture::new();
    let long_answer = answer().replace(
        "mock complete",
        &format!(
            "old-assistant-marker {}",
            "verbose resolved analysis ".repeat(80)
        ),
    );
    let (url, handle) = mock(vec![long_answer, answer(), answer(), answer()]);
    let old = format!("old-marker {}", "history ".repeat(60));
    let input = format!(
        "{old}\nsecond question\nlatest question\n/compact-preview 1 Old work resolved.\n/compact 1 Old work resolved.\n/compaction\n/journal 1 0 4096\n/context\ncontinue\n/quit\n"
    );
    let output = run_bounded(
        f.command(&url)
            .args(["--plain", "--approval", "never", "--session", "compact"]),
        &input,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = handle.join().unwrap();
    let last = json::stringify(&requests[3]).unwrap();
    assert!(last.contains("old-marker")); // exact user constraints are never guessed away
    assert!(!last.contains("old-assistant-marker"));
    assert!(last.contains("Old work resolved."));
    assert!(last.contains("latest question"));
    let bytes = std::fs::read(f.0.join(".vade/sessions/compact.journal")).unwrap();
    assert!(
        bytes
            .windows(b"old-marker".len())
            .any(|w| w == b"old-marker")
    );
    let workspace = vade_harness::tools::Workspace::new(&f.0).unwrap();
    let (_, resumed, torn) = vade_harness::journal::Journal::resume(
        &workspace,
        "compact",
        vade_harness::agent::Limits::default(),
    )
    .unwrap();
    assert!(!torn);
    assert_eq!(resumed.messages().len(), 9); // system + project, all users, summary, latest old answer, new turn
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Context generation 1"));
    assert!(stdout.contains("invalidates all later causal KV"));
    assert!(stdout.contains("journal-record-1"));
}

#[test]
fn large_tool_output_is_journaled_and_retrieved_by_bounded_range_after_resume() {
    let f = Fixture::new();
    let text = "🦀 exact output\n".repeat(600);
    assert!(text.len() < 16 * 1024);
    std::fs::create_dir(f.0.join("skills")).unwrap();
    std::fs::write(
        f.0.join("skills/catalog.json"),
        r#"{"skills":[{"name":"large","description":"Fixture","path":"skills/SKILL.md"}]}"#,
    )
    .unwrap();
    std::fs::write(f.0.join("skills/SKILL.md"), &text).unwrap();
    let (url, handle) = mock(vec![
        tools(vec![call(
            "large-read",
            "skill_read",
            json::parse(r#"{"name":"large"}"#).unwrap(),
        )]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url)
            .args(["--session", "large", "--prompt", "read it"]),
        "",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = handle.join().unwrap();
    let messages = requests[1].get("messages").unwrap().as_array().unwrap();
    let result = messages
        .iter()
        .find(|m| m.get("tool_call_id").and_then(Value::as_str) == Some("large-read"))
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(result.len() <= 4096);
    assert!(result.starts_with("Captured tool result excerpt"));
    let metadata = json::parse(result.split_once('\n').unwrap().1).unwrap();
    let id = metadata.get("artifact").unwrap().as_str().unwrap();
    let prefix_bytes = metadata
        .get("text")
        .unwrap()
        .as_str()
        .unwrap()
        .find("🦀 exact output\n")
        .unwrap();
    let Value::Number(next) = metadata.get("next_offset").unwrap() else {
        panic!()
    };
    let offset: usize = next.parse().unwrap();
    assert!(offset < text.len());
    let args = Value::Object(vec![
        ("id".into(), Value::String(id.into())),
        ("offset".into(), Value::Number(offset.to_string())),
        ("length".into(), Value::Number("4096".into())),
    ]);
    let (url, handle) = mock(vec![
        tools(vec![call("chunk", "tool_output_read", args)]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url).args(["--resume", "large", "--plain"]),
        &format!("/output {id} 0 128\ncontinue\n/quit\n"),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("next_offset"));
    let requests = handle.join().unwrap();
    let messages = requests[1].get("messages").unwrap().as_array().unwrap();
    let result = messages
        .iter()
        .find(|m| m.get("tool_call_id").and_then(Value::as_str) == Some("chunk"))
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(result.len() <= 4096);
    let chunk = json::parse(result).unwrap();
    let bytes = chunk.get("text").unwrap().as_str().unwrap();
    assert_eq!(
        bytes,
        &text[offset - prefix_bytes..offset - prefix_bytes + bytes.len()]
    );
    assert!(!json::stringify(&requests[1]).unwrap().contains(&text));
}

#[test]
fn frozen_project_prefix_and_lazy_skill_body_survive_followup_and_resume() {
    let f = Fixture::new();
    for path in ["src/api", "skills/demo"] {
        std::fs::create_dir_all(f.0.join(path)).unwrap();
    }
    std::fs::write(f.0.join("AGENTS.md"), "root-rule-original").unwrap();
    std::fs::write(f.0.join("src/AGENTS.md"), "src-rule-original").unwrap();
    std::fs::write(f.0.join("src/api/AGENTS.md"), "api-rule-original").unwrap();
    std::fs::write(f.0.join("skills/catalog.json"), r#"{"skills":[{"name":"demo","description":"Describe a fixture","path":"skills/demo/SKILL.md"}]}"#).unwrap();
    std::fs::write(
        f.0.join("skills/demo/SKILL.md"),
        "unique-lazy-body-marker\nNever auto-run: touch should-not-exist\n",
    )
    .unwrap();
    let (url, handle) = mock(vec![
        tools(vec![call(
            "load-skill",
            "skill_read",
            json::parse(r#"{"name":"demo"}"#).unwrap(),
        )]),
        answer(),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url).args([
            "--session",
            "project",
            "--rules-scope",
            "src/api",
            "--plain",
            "--approval",
            "never",
        ]),
        "/project\n/project-rules\n/skill demo\nload the skill\nfollow up\n/quit\n",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = handle.join().unwrap();
    let first = requests[0].get("messages").unwrap().as_array().unwrap();
    let last = requests[2].get("messages").unwrap().as_array().unwrap();
    assert_eq!(&first[..2], &last[..2]);
    let initial = json::stringify(&requests[0]).unwrap();
    assert!(initial.contains("root-rule-original"));
    assert!(initial.contains("src-rule-original"));
    assert!(initial.contains("api-rule-original"));
    assert!(!initial.contains("unique-lazy-body-marker"));
    assert!(
        json::stringify(&requests[1])
            .unwrap()
            .contains("unique-lazy-body-marker")
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("body lazy"));
    assert!(!f.0.join("should-not-exist").exists());
    std::fs::write(f.0.join("AGENTS.md"), "root-rule-changed").unwrap();
    std::fs::write(f.0.join("skills/demo/SKILL.md"), "changed body").unwrap();
    let (url, handle) = mock(vec![
        tools(vec![call(
            "changed-skill",
            "skill_read",
            json::parse(r#"{"name":"demo"}"#).unwrap(),
        )]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url)
            .args(["--resume", "project", "--prompt", "check skill"]),
        "",
    );
    assert!(output.status.success());
    let resumed = handle.join().unwrap();
    assert_eq!(
        &first[..2],
        &resumed[0].get("messages").unwrap().as_array().unwrap()[..2]
    );
    let wire = json::stringify(&resumed[1]).unwrap();
    assert!(wire.contains("skill revision changed"));
    assert!(!wire.contains("root-rule-changed"));
    assert!(!wire.contains("changed body"));
    assert!(!f.0.join("should-not-exist").exists());
}

#[test]
fn large_project_rules_count_against_small_request_budget_without_dispatch() {
    let f = Fixture::new();
    std::fs::write(f.0.join("AGENTS.md"), "rule ".repeat(1500)).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/custom/rpc", listener.local_addr().unwrap()); // local heuristic fallback
    let output = run_bounded(
        f.command(&url)
            .args(["--context-tokens", "8192", "--max-output-tokens", "4096", "--prompt", "tiny prompt"]),
        "",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("it was not sent"));
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn transcript_forks_switch_without_replaying_tools_or_restoring_workspace_files() {
    let f = Fixture::new();
    let edit_child = Value::Object(vec![("patch".into(), Value::String("*** Begin Patch\n*** Update File: result.txt\n@@\n-exact result\n+child result\n*** End Patch".into()))]);
    let (url, handle) = mock(vec![
        tools(vec![call("parent-edit", "apply_patch", patch())]),
        answer(),
        tools(vec![call("child-edit", "apply_patch", edit_child)]),
        answer(),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url)
            .args(["--session", "parent", "--approval", "auto", "--plain"]),
        "parent task\n/fork child alternative approach\n/branch\nchild task\n/switch parent\n/branch\nparent followup\n/quit\n",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(f.0.join("result.txt")).unwrap(),
        "child result\n"
    );
    let requests = handle.join().unwrap();
    assert_eq!(requests.len(), 5); // fork/switch never send a model request
    let first = requests[0].get("messages").unwrap().as_array().unwrap();
    for index in [2, 4] {
        let messages = requests[index].get("messages").unwrap().as_array().unwrap();
        assert_eq!(&messages[..2], &first[..2]);
        assert!(
            json::stringify(&requests[index]).unwrap().contains(
                "Historical tool results and file revisions are not current file contents"
            )
        );
    }
    let parent_wire = json::stringify(&requests[4]).unwrap();
    assert!(parent_wire.contains("parent-edit"));
    assert!(!parent_wire.contains("child-edit"));
    assert!(!parent_wire.contains("child task"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Fork created and selected"));
    assert!(stdout.contains("parent parent at record"));
    assert!(stdout.contains("1 changed"));
    assert!(stdout.contains("KV reuse unknown"));
    let w = vade_harness::tools::Workspace::new(&f.0).unwrap();
    let (_, child, _) =
        vade_harness::journal::Journal::resume(&w, "child", vade_harness::agent::Limits::default())
            .unwrap();
    assert!(
        child
            .messages()
            .iter()
            .any(|m| m.tool_call_id() == Some("child-edit"))
    );
    assert!(
        !child
            .messages()
            .iter()
            .any(|m| m.value().get("content").and_then(Value::as_str) == Some("parent followup"))
    );
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
            assert!(!result.contains("unknown_tool"));
            let unsupported_shell = id == "parent-shell" && !cfg!(target_os = "macos");
            assert_eq!(result.contains("unsupported_platform"), unsupported_shell);
            assert_eq!(
                result.contains("denied"),
                policy == "never" && id != "parent-map" && !unsupported_shell
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

#[test]
fn continue_only_after_length_retains_history_and_seed_as_a_user_turn() {
    const CONTINUATION: &str = "Continue. If you were still planning, stop planning and make the next concrete edit or tool call now, in small parts.";
    for finish in ["length", "stop"] {
        let f = Fixture::new();
        let first_answer = format!("first explanation with {finish} finish");
        let first_reply = json::stringify(&Value::Object(vec![(
            "choices".into(),
            Value::Array(vec![Value::Object(vec![
                ("finish_reason".into(), Value::String(finish.into())),
                (
                    "message".into(),
                    Value::Object(vec![
                        ("role".into(), Value::String("assistant".into())),
                        ("content".into(), Value::String(first_answer.clone())),
                    ]),
                ),
            ])]),
        )]))
        .unwrap();
        let mut replies = vec![first_reply, answer()];
        if finish == "length" {
            replies.push(answer());
        }
        let (url, server) = mock(replies);
        let mut command = f.command(&url);
        command.args(["--plain", "--session", "continuation", "--seed", "713", "--no-auto-continue"]);
        let mut child = cli::Cli::start(&mut command);
        child.send("explain the result\n");
        child.until(true, "Native run:");
        child.send("/continue\n");
        if finish == "length" {
            child.until(true, "Native run:");
            // The successful continuation clears the cutoff state; another
            // /continue must not create another user turn or model request.
            child.send("/continue\n");
        }
        child.until(
            false,
            "The last reply was not cut off; no continuation sent.",
        );
        // Keep the mock listening until a normal followup, so an unexpected
        // continuation would consume a response and fail the history checks.
        child.send("normal followup\n");
        child.until(true, "Native run:");
        child.send("/quit\n");
        let output = child.finish();
        // A length cutoff remains a failed turn even after a later success.
        assert_eq!(
            output.status.success(),
            finish == "stop",
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let requests = server.join().unwrap();
        assert_eq!(requests.len(), if finish == "length" { 3 } else { 2 });
        assert!(
            requests
                .iter()
                .all(|request| request.get("seed") == Some(&Value::Number("713".into())))
        );
        let initial = requests[0].get("messages").unwrap().as_array().unwrap();
        let next = requests[1].get("messages").unwrap().as_array().unwrap();
        assert_eq!(next.len(), initial.len() + 2);
        assert_eq!(&next[..initial.len()], initial);
        assert_eq!(
            next[initial.len()],
            Value::Object(vec![
                ("role".into(), Value::String("assistant".into())),
                ("content".into(), Value::String(first_answer.clone())),
            ])
        );
        assert_eq!(
            next.last().unwrap(),
            &Value::Object(vec![
                ("role".into(), Value::String("user".into())),
                (
                    "content".into(),
                    Value::String(
                        if finish == "length" {
                            CONTINUATION
                        } else {
                            "normal followup"
                        }
                        .into(),
                    ),
                ),
            ])
        );
        let final_messages = requests
            .last()
            .unwrap()
            .get("messages")
            .unwrap()
            .as_array()
            .unwrap();
        if finish == "length" {
            assert_eq!(final_messages.len(), next.len() + 2);
            assert_eq!(&final_messages[..next.len()], next);
            assert_eq!(
                final_messages
                    .last()
                    .unwrap()
                    .get("content")
                    .and_then(Value::as_str),
                Some("normal followup")
            );
        }
        let workspace = vade_harness::tools::Workspace::new(&f.0).unwrap();
        let (journal, replay, _) = vade_harness::journal::Journal::resume(
            &workspace,
            "continuation",
            vade_harness::agent::Limits::default(),
        )
        .unwrap();
        assert_eq!(journal.generation_seed(), Some(713));
        assert_eq!(replay.messages().len(), final_messages.len() + 1);
        for (retained, sent) in replay.messages().iter().zip(final_messages) {
            assert_eq!(retained.value(), sent);
        }
        assert_eq!(
            replay
                .messages()
                .iter()
                .filter(|message| {
                    message.role() == "user"
                        && message.value().get("content").and_then(Value::as_str)
                            == Some(CONTINUATION)
                })
                .count(),
            usize::from(finish == "length")
        );
    }
}

fn compaction_cli_command(f: &Fixture, url: &str) -> Command {
    let mut command = f.command(url);
    command.args([
        "--plain",
        "--approval",
        "never",
        "--max-output-tokens",
        "128",
        "--seed",
        "902",
        "--temperature",
        "0.4",
        "--thinking",
        "off",
    ]);
    command
}

fn compaction_message(role: &str, content: &str) -> Value {
    Value::Object(vec![
        ("role".into(), Value::String(role.into())),
        ("content".into(), Value::String(content.into())),
    ])
}

fn compaction_text_to_threshold(
    conversation: &vade_harness::agent::Conversation,
    role: &str,
    marker: &str,
    percent: usize,
    budget: usize,
) -> String {
    let mut text = marker.to_owned();
    loop {
        let message = vade_harness::agent_wire::Message::from_value(
            compaction_message(role, &text),
            vade_harness::agent::Limits::default().wire,
        )
        .unwrap();
        let usage = vade_harness::context::Usage::with_appended(conversation, &[message]);
        if usage.required(128) >= budget * percent / 100 {
            assert!(usage.fits(budget, 128));
            return text;
        }
        text.push_str(" retained explanation ");
        assert!(text.len() < 60 * 1024);
    }
}

fn compaction_request_options(request: &Value) -> String {
    let Value::Object(fields) = request else {
        panic!("request must be an object")
    };
    json::stringify(&Value::Object(
        fields
            .iter()
            .filter(|(key, _)| key != "messages")
            .cloned()
            .collect(),
    ))
    .unwrap()
}

#[test]
fn automatic_cli_compaction_observes_incoming_user_and_preserves_prior_summaries() {
    use vade_harness::{
        agent::{Conversation, Limits},
        context::Usage,
    };
    for disabled in [false, true] {
        let f = Fixture::new();
        std::fs::write(
            f.0.join("AGENTS.md"),
            "Keep exact scoped rules.\n  Spacing and 🦀 matter.\n",
        )
        .unwrap();
        let old_answer = format!(
            "old-result-marker {} old-result-tail",
            "resolved ".repeat(1000)
        );
        let (url, server) = mock(vec![
            answer().replace("mock complete", &old_answer),
            answer().replace("mock complete", &old_answer),
            answer(),
        ]);
        let output = run_bounded(
            compaction_cli_command(&f, &url).args(["--session", "automatic"]),
            "old constraints\nsecond constraints\nlatest constraints\n/quit\n",
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let initial_requests = server.join().unwrap();
        let workspace = vade_harness::tools::Workspace::new(&f.0).unwrap();
        let (journal, initial, _) =
            vade_harness::journal::Journal::resume(&workspace, "automatic", Limits::default())
                .unwrap();
        assert_eq!(journal.generation(), 0);
        drop(journal);
        let budget = Usage::for_conversation(&initial)
            .required(128)
            .saturating_mul(10)
            .div_ceil(7)
            .max(8192);
        assert!(!Usage::for_conversation(&initial).near_limit(budget, 128));
        let incoming =
            compaction_text_to_threshold(&initial, "user", "incoming-long-user-marker", 81, budget);
        let raw_before = std::fs::read(f.0.join(".vade/sessions/automatic.journal")).unwrap();
        let (url, server) = mock(vec![answer(), answer()]);
        let mut command = compaction_cli_command(&f, &url);
        command.args([
            "--resume",
            "automatic",
            "--context-tokens",
            &budget.to_string(),
        ]);
        if disabled {
            command.arg("--no-auto-compact");
        }
        let output = run_bounded(
            &mut command,
            &format!("{incoming}\nsmall followup\n/quit\n"),
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let requests = server.join().unwrap();
        let notice = String::from_utf8_lossy(&output.stderr);
        assert_eq!(notice.contains("Auto-compacted older results:"), !disabled);
        assert_eq!(
            notice.matches("Auto-compacted older results:").count(),
            usize::from(!disabled)
        );
        let prefix: Vec<_> = initial.messages()[..initial.prefix_len()]
            .iter()
            .map(|m| m.value().clone())
            .collect();
        let prefix_wire = json::stringify(&Value::Array(prefix)).unwrap();
        let initial_options = compaction_request_options(&initial_requests[0]);
        for request in &requests {
            let messages = request.get("messages").unwrap().as_array().unwrap();
            assert_eq!(
                json::stringify(&Value::Array(messages[..initial.prefix_len()].to_vec())).unwrap(),
                prefix_wire
            );
            assert_eq!(compaction_request_options(request), initial_options);
            assert!(
                messages
                    .iter()
                    .any(|m| m.get("content").and_then(Value::as_str) == Some(&incoming))
            );
            for constraints in [
                "old constraints",
                "second constraints",
                "latest constraints",
            ] {
                assert!(
                    messages
                        .iter()
                        .any(|m| m.get("content").and_then(Value::as_str) == Some(constraints))
                );
            }
        }
        let first_messages = requests[0].get("messages").unwrap().as_array().unwrap();
        let summaries: Vec<_> = first_messages
            .iter()
            .filter(|m| m.get("name").and_then(Value::as_str) == Some("vade_context_summary"))
            .cloned()
            .collect();
        assert_eq!(summaries.len(), usize::from(!disabled));
        assert_eq!(
            first_messages
                .iter()
                .filter(|m| m.get("content").and_then(Value::as_str) == Some(&old_answer))
                .count(),
            if disabled { 2 } else { 0 }
        );
        let raw_after = std::fs::read(f.0.join(".vade/sessions/automatic.journal")).unwrap();
        assert!(raw_after.starts_with(&raw_before));
        assert!(
            raw_after
                .windows(old_answer.len())
                .any(|bytes| bytes == old_answer.as_bytes())
        );
        let (journal, compacted, _) =
            vade_harness::journal::Journal::resume(&workspace, "automatic", Limits::default())
                .unwrap();
        assert_eq!(journal.generation(), u64::from(!disabled));
        drop(journal);
        if disabled {
            continue;
        }
        assert!(Usage::for_conversation(&compacted).required(128) <= budget * 3 / 5);
        assert!(
            requests[1]
                .get("messages")
                .unwrap()
                .as_array()
                .unwrap()
                .contains(&summaries[0])
        );

        // Grow through another threshold crossing, then keep sending small
        // turns above 60%. Earlier summary bytes must survive a new compaction
        // and must never become the next compaction's only removable input.
        let mut sizing = compacted.clone();
        sizing
            .push_user(compaction_message("user", "grow first"), Limits::default())
            .unwrap();
        let first_growth = compaction_text_to_threshold(
            &sizing,
            "assistant",
            "new-first-result-marker",
            70,
            budget,
        );
        let mut request =
            json::parse(&sizing.request(Limits::default().wire).to_json().unwrap()).unwrap();
        let Value::Object(fields) = &mut request else {
            panic!()
        };
        let Value::Array(messages) = &mut fields
            .iter_mut()
            .find(|(key, _)| key == "messages")
            .unwrap()
            .1
        else {
            panic!()
        };
        messages.push(compaction_message("assistant", &first_growth));
        sizing = Conversation::from_request(request, Limits::default()).unwrap();
        sizing
            .push_user(compaction_message("user", "grow second"), Limits::default())
            .unwrap();
        let second_growth = compaction_text_to_threshold(
            &sizing,
            "assistant",
            "new-second-result-marker",
            90,
            budget,
        );
        let (url, server) = mock(vec![
            answer().replace("mock complete", &first_growth),
            answer().replace("mock complete", &second_growth),
            answer(),
            answer(),
        ]);
        let output = run_bounded(
            compaction_cli_command(&f, &url).args([
                "--resume",
                "automatic",
                "--context-tokens",
                &budget.to_string(),
            ]),
            "grow first\ngrow second\ntrigger second compaction\nsmall tail\n/quit\n",
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let notice = String::from_utf8_lossy(&output.stderr);
        assert_eq!(notice.matches("Auto-compacted older results:").count(), 1);
        assert!(notice.contains("later admission overflow attempts another safe rewrite"));
        let requests = server.join().unwrap();
        for request in &requests {
            assert_eq!(compaction_request_options(request), initial_options);
            let messages = request.get("messages").unwrap().as_array().unwrap();
            assert!(messages.contains(&summaries[0]));
            assert_eq!(
                json::stringify(&Value::Array(messages[..initial.prefix_len()].to_vec())).unwrap(),
                prefix_wire
            );
        }
        let last = requests
            .last()
            .unwrap()
            .get("messages")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(
            last.iter()
                .filter(|m| m.get("name").and_then(Value::as_str) == Some("vade_context_summary"))
                .count(),
            2
        );
        assert!(
            !last
                .iter()
                .any(|m| m.get("content").and_then(Value::as_str) == Some(&first_growth))
        );
        assert!(
            last.iter()
                .any(|m| m.get("content").and_then(Value::as_str) == Some(&second_growth))
        );
        let raw_final = std::fs::read(f.0.join(".vade/sessions/automatic.journal")).unwrap();
        assert!(raw_final.starts_with(&raw_after));
        let (journal, resumed, _) =
            vade_harness::journal::Journal::resume(&workspace, "automatic", Limits::default())
                .unwrap();
        assert_eq!(journal.generation(), 2);
        assert!(
            resumed
                .messages()
                .iter()
                .any(|m| m.value() == &summaries[0])
        );
    }
}
