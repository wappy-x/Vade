//! First-party loopback fixtures only. No Engine binary, model, delegate or GPU.
#[path = "support/cli.rs"]
mod cli;
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use vade_harness::{
    agent::{
        self, Conversation, Limits, Model, StopReason, ToolError, ToolExecutor, ToolInvocation,
    },
    agent_wire::FinishReason,
    contract::{Capabilities, InputCount},
    json,
    protocol::Message,
    transport::{ChatClient, NativeModel},
};

const CAPS: &str = include_str!("fixtures/engine-capabilities-v1.json");
#[cfg(target_os = "macos")]
#[path = "support/children_contract.rs"]
mod children_contract;
const DONE: &str = "data: [DONE]\n\n";
const ANSWER: &str =
    "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n";

#[test]
fn metadata_transport_break_retries_once_without_replaying_generation() {
    for admission_break in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            if admission_break { capabilities(&mut reader, CAPS); }
            let lost = request(&mut reader);
            if admission_break {
                assert_eq!(lost.0, "POST /v1/admission HTTP/1.1");
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{{").unwrap();
            } else { assert_eq!(lost.0, "GET /v1/capabilities HTTP/1.1"); }
            drop(reader);
            let mut reader = accept(&listener);
            let retried = request(&mut reader);
            assert_eq!(retried, lost, "retry must retain exact metadata request bytes");
            if admission_break {
                let v = json::view(&retried.1).unwrap();
                let output = v.get("output_tokens").unwrap().raw().parse().unwrap();
                respond(&mut reader, 200, "application/json", &admission_body(100, output, true));
            } else {
                respond(&mut reader, 200, "application/json", CAPS);
                admission(&mut reader, 100, true);
            }
            assert_eq!(request(&mut reader).0, "POST /v1/chat/completions HTTP/1.1");
            // A generation failure must propagate, without a second generation.
            drop(reader);
        });
        let mut model = NativeModel::new(&endpoint).unwrap();
        let c = native_conversation();
        let cancel = AtomicBool::new(false);
        let count = model.count_input(c.request(Limits::default().wire), &cancel);
        assert!(count.is_ok(), "read-only metadata recovery failed: {count:?}");
        assert!(model.complete(c.request(Limits::default().wire), &cancel, &mut |_| Ok(())).is_err());
        assert_eq!(model.context().requests.generations, 1);
        assert_eq!(model.context().requests.capabilities, if admission_break { 1 } else { 2 });
        assert_eq!(model.context().requests.admissions, if admission_break { 2 } else { 1 });
        server.join().unwrap();
    }
}

#[test]
fn cutoff_only_recovery_requires_a_tool_and_releases_policy_after_progress() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut generations = 0;
        while generations < 4 {
            let (line, body) = request(&mut reader);
            match line.as_str() {
                "GET /v1/capabilities HTTP/1.1" => respond(&mut reader, 200, "application/json", CAPS),
                "POST /v1/admission HTTP/1.1" => {
                    let v = json::view(&body).unwrap();
                    let output = v.get("output_tokens").unwrap().raw().parse().unwrap();
                    respond(&mut reader, 200, "application/json", &admission_body(100, output, true));
                }
                "POST /v1/chat/completions HTTP/1.1" => {
                    let v = json::view(&body).unwrap();
                    assert_eq!(v.get("tool_choice").and_then(|v| v.as_str()).as_deref(), (generations == 2).then_some("required"));
                    assert_eq!(v.get("max_tokens").unwrap().raw(), "17");
                    let reply = match generations {
                        0 | 1 => r#"{"choices":[{"message":{"role":"assistant","reasoning_content":"still planning","content":""},"finish_reason":"length"}]}"#,
                        2 => r#"{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"recovered","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
                        _ => r#"{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}"#,
                    };
                    respond(&mut reader, 200, "application/json", reply);
                    generations += 1;
                }
                _ => panic!("unexpected request {line}"),
            }
        }
    });
    let mut c = native_conversation();
    let mut model = NativeModel::new(&endpoint).unwrap();
    let limits = Limits { max_model_turns: 4, ..Limits::default() };
    let report = agent::run(&mut c, &mut model, &mut MockTools, limits, &AtomicBool::new(false), &mut |_| {});
    assert_eq!((report.reason, report.model_turns, report.tool_calls), (StopReason::Finished(FinishReason::Stop), 4, 1));
    assert_eq!(c.messages().iter().filter(|m| m.role() == "tool").count(), 1);
    server.join().unwrap();
}

#[test]
fn oversized_prompt_is_counted_before_save_and_resume_accepts_next_prompt() {
    use std::process::Command;
    let workspace = WorkspaceFixture::new();
    let (listener, endpoint) = listen();
    let oversized = "oversize-marker-".to_owned() + &"x".repeat(32_000 - 16);
    assert_eq!(oversized.len(), 32_000);
    let server = thread::spawn(move || {
        let caps = CAPS.replace("4096", "8192").replace("4000", "8160");
        let mut generations = 0;
        for _ in 0..2 {
            let mut reader = accept(&listener);
            while !reader.fill_buf().unwrap().is_empty() {
                let (line, body) = request(&mut reader);
                match line.as_str() {
                    "GET /v1/capabilities HTTP/1.1" => {
                        respond(&mut reader, 200, "application/json", &caps);
                    }
                    "POST /v1/admission HTTP/1.1" => {
                        let large = body.contains("oversize-marker-");
                        let reply = admission_body(if large { 10_000 } else { 100 }, 32, !large)
                            .replace("4096", "8192")
                            .replace("4000", "8160");
                        respond(&mut reader, 200, "application/json", &reply);
                    }
                    "POST /v1/chat/completions HTTP/1.1" => {
                        assert!(!body.contains("oversize-marker-"));
                        generations += 1;
                        respond(
                            &mut reader,
                            200,
                            "text/event-stream",
                            &format!("{ANSWER}{DONE}"),
                        );
                    }
                    _ => panic!("unexpected mock request"),
                }
            }
        }
        generations
    });
    let launch = |resume: bool, prompt: &str| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        command
            .args([
                "agent",
                "--state-in-project",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--workspace",
            ])
            .arg(&workspace.0)
            .args([
                "--plain",
                "--verbose",
                "--approval",
                "never",
                "--context-tokens",
                "8192",
                if resume { "--resume" } else { "--session" },
                "oversize",
                "--prompt",
                prompt,
            ]);
        cli::Cli::start(&mut command).finish()
    };
    let rejected = launch(false, &oversized);
    let raw = std::fs::read(workspace.0.join(".vade/sessions/oversize.journal")).unwrap();
    let clean = !raw.windows(16).any(|w| w == b"oversize-marker-");
    let next = launch(true, "tiny follow-up");
    let generations = server.join().unwrap();
    eprintln!(
        "oversize reproduction: journal_clean={clean}; next_success={}; generations={generations}",
        next.status.success()
    );
    assert!(clean, "oversized prompt was persisted before admission");
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("it was not sent"));
    assert!(
        next.status.success(),
        "{}",
        String::from_utf8_lossy(&next.stderr)
    );
    assert_eq!(generations, 1);
}

#[test]
fn presave_exact_count_accepts_a_prompt_that_the_local_estimate_would_veto() {
    use std::process::Command;
    let workspace = WorkspaceFixture::new();
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        let body = admission(&mut reader, 100, true);
        assert!(body.contains(&"x".repeat(32_000)));
        assert_eq!(
            request(&mut reader),
            ("POST /v1/chat/completions HTTP/1.1".into(), body)
        );
        respond(
            &mut reader,
            200,
            "text/event-stream",
            &format!("{ANSWER}{DONE}"),
        );
    });
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
    command
        .args([
            "agent",
            "--state-in-project",
            "--endpoint",
            &endpoint,
            "--model",
            "mock",
            "--workspace",
        ])
        .arg(&workspace.0)
        .args([
            "--plain",
            "--approval",
            "never",
            "--context-tokens",
            "8192",
            "--prompt",
            &"x".repeat(32_000),
        ]);
    let output = cli::Cli::start(&mut command).finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    server.join().unwrap();
}

#[test]
fn cached_native_refusal_cannot_be_used_to_send_generation() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        admission(&mut reader, 4090, false);
        closed_without_replay(&mut reader, &listener);
    });
    let mut model = NativeModel::new(&endpoint).unwrap();
    let c = native_conversation();
    let cancel = AtomicBool::new(false);
    for _ in 0..2 {
        assert!(matches!(
            model
                .count_input(c.request(Limits::default().wire), &cancel)
                .unwrap(),
            InputCount::Exact {
                admitted: false,
                ..
            }
        ));
    }
    assert!(
        model
            .complete(c.request(Limits::default().wire), &cancel, &mut |_| Ok(()))
            .is_err()
    );
    assert_eq!(model.context().requests.generations, 0);
    drop(model);
    server.join().unwrap();
}

// The returned slice comes from the HTTP body read off the socket, including
// every byte of options/tools and the complete leading developer record.
fn socket_head(body: &str) -> &str {
    let v = json::view(body).unwrap();
    let developer = v.get("messages").unwrap().array().unwrap().nth(1).unwrap();
    assert_eq!(
        developer.get("role").unwrap().as_str().as_deref(),
        Some("developer")
    );
    let raw = developer.raw();
    let end = raw.as_ptr() as usize - body.as_ptr() as usize + raw.len();
    &body[..end]
}

#[test]
fn socket_captured_generation_head_is_identical_after_compaction() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut bodies = Vec::new();
        for _ in 0..2 {
            capabilities(&mut reader, CAPS);
            let admitted = admission(&mut reader, 100, true);
            let (line, captured) = request(&mut reader);
            assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
            assert_eq!(captured, admitted);
            bodies.push(captured);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{DONE}"),
            );
        }
        assert_eq!(
            socket_head(&bodies[0]).as_bytes(),
            socket_head(&bodies[1]).as_bytes()
        );
        let messages: Vec<_> = json::view(&bodies[1])
            .unwrap()
            .get("messages")
            .unwrap()
            .array()
            .unwrap()
            .collect();
        assert_eq!(
            messages[2].get("name").unwrap().as_str().as_deref(),
            Some("vade_context_summary")
        );
        assert!(bodies[1].len() < bodies[0].len());
    });
    let request = format!(
        r#"{{"model":"mock","tools":[{{"type":"function","function":{{"name":"read","parameters":{{"type":"object"}}}}}}],"opaque":{{"order":[3,1,2]}},"messages":[{{"role":"system","content":"stable 🦀\\n rules"}},{{"role":"developer","content":" scoped rules  \\n keep spacing","opaque":[2,1]}},{{"role":"user","content":"old constraint"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"recent constraint"}},{{"role":"assistant","content":"recent answer"}},{{"role":"user","content":"next task"}}]}}"#,
        "discardable old results ".repeat(100)
    );
    let original =
        Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
    let mut model = NativeModel::new(&endpoint)
        .unwrap()
        .with_compaction_prewarm(true);
    let cancel = AtomicBool::new(false);
    model
        .count_input(original.request(Limits::default().wire), &cancel)
        .unwrap();
    model
        .complete(
            original.request(Limits::default().wire),
            &cancel,
            &mut |_| Ok(()),
        )
        .unwrap();
    let compacted =
        vade_harness::context::compact(&original, "Resolved older work.", 1, Limits::default())
            .unwrap();
    model.prompt_rewritten(); // automatic path: no admission/decode for a warm
    model
        .count_input(compacted.request(Limits::default().wire), &cancel)
        .unwrap();
    model
        .warm_compacted(compacted.request(Limits::default().wire), &cancel)
        .unwrap();
    model
        .complete(
            compacted.request(Limits::default().wire),
            &cancel,
            &mut |_| Ok(()),
        )
        .unwrap();
    assert!(model.context().compaction_warm.is_none());
    assert_eq!(model.context().requests.generations, 2);
    server.join().unwrap();
}

#[test]
fn native_capability_snapshot_survives_compaction_and_reconciles_one_new_build() {
    let initial = CAPS.replacen("\"object\"", "\"build_hash\":\"build-a\",\"object\"", 1);
    assert_native_capability_reconcile(initial, "build-a", "build-b");
}

#[test]
fn native_capability_snapshot_reconciles_cache_domain_once_without_build_hash() {
    assert_native_capability_reconcile(CAPS.into(), "mock-load-epoch", "mock-load-epoch-2");
}

#[test]
fn native_capability_snapshot_reconciles_each_fingerprint_once_without_build_hash() {
    for fingerprint in ["mock-weights", "mock-tokenizer", "mock-template"] {
        assert_native_capability_reconcile(CAPS.into(), fingerprint, &format!("{fingerprint}-2"));
    }
}

fn assert_native_capability_reconcile(initial: String, old_epoch: &str, new_epoch: &str) {
    let (listener, endpoint) = listen();
    let drift = initial
        .replace("\"default_thinking\": false", "\"default_thinking\": true")
        .replace(
            "\"prompt_cache_key\": \"scheduling_hint\"",
            "\"prompt_cache_key\": null",
        );
    let next = drift.replace(old_epoch, new_epoch);
    let later_drift = initial.replace(old_epoch, new_epoch);
    let expected_initial = Capabilities::parse(&initial).unwrap().unwrap();
    let expected_next = Capabilities::parse(&next).unwrap().unwrap();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut bodies = Vec::new();
        for caps in [initial, drift, next, later_drift] {
            capabilities(&mut reader, &caps);
            admission(&mut reader, 100, true);
            let (line, body) = request(&mut reader);
            assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
            bodies.push(body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{DONE}"),
            );
        }
        assert_eq!(socket_head(&bodies[0]), socket_head(&bodies[1]));
        assert_ne!(socket_head(&bodies[1]), socket_head(&bodies[2]));
        assert_eq!(socket_head(&bodies[2]), socket_head(&bodies[3]));
    });
    let mut c = Conversation::from_request(json::parse(r#"{"model":"mock","messages":[{"role":"system","content":"stable"},{"role":"developer","content":"scoped rules"},{"role":"user","content":"first"}]}"#).unwrap(), Limits::default()).unwrap();
    let mut model = NativeModel::new(&endpoint).unwrap();
    let cancel = AtomicBool::new(false);
    for turn in 0..4 {
        if turn == 1 {
            model.reset_prompt();
        }
        if turn > 0 {
            c.push_user(
                json::parse(&format!(r#"{{"role":"user","content":"turn {turn}"}}"#)).unwrap(),
                Limits::default(),
            )
            .unwrap();
        }
        model
            .count_input(c.request(Limits::default().wire), &cancel)
            .unwrap();
        model
            .complete(c.request(Limits::default().wire), &cancel, &mut |_| Ok(()))
            .unwrap();
        assert_eq!(
            model.context().capability_reconciliations,
            u64::from(turn >= 2)
        );
        assert_eq!(model.context().cache_domain_changes, u64::from(turn >= 2));
        assert_eq!(
            model.context().capabilities.as_ref(),
            Some(if turn < 2 {
                &expected_initial
            } else {
                &expected_next
            })
        );
    }
    assert!(
        model
            .context()
            .describe()
            .contains("Engine build changed; capabilities reconciled")
    );
    server.join().unwrap();
}

#[test]
fn manual_compaction_counts_before_save_and_warm_failure_allows_real_followup() {
    use std::{process::Command, sync::Arc};
    // Oversized candidate, non-shrinking exact count, and successful compaction
    // followed by an HTTP-failed opt-in warm request.
    for scenario in 0..3 {
        let workspace = WorkspaceFixture::new();
        let (listener, endpoint) = listen();
        let done = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&done);
        let server = thread::spawn(move || {
            listener.set_nonblocking(true).unwrap();
            let mut generations = 0;
            let mut failed_warms = 0;
            while !stopped.load(Ordering::Relaxed) {
                let socket = match listener.accept() {
                    Ok((socket, _)) => socket,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(socket);
                while !reader.fill_buf().unwrap().is_empty() {
                    let (line, body) = request(&mut reader);
                    match line.as_str() {
                        "GET /v1/capabilities HTTP/1.1" => {
                            respond(&mut reader, 200, "application/json", CAPS)
                        }
                        "POST /v1/admission HTTP/1.1" => {
                            let envelope = json::view(&body).unwrap();
                            let summary = envelope
                                .get("request")
                                .unwrap()
                                .get("messages")
                                .unwrap()
                                .raw()
                                .contains("vade_context_summary");
                            let output = envelope
                                .get("output_tokens")
                                .unwrap()
                                .raw()
                                .parse()
                                .unwrap();
                            let tokens = if summary {
                                [4090, 512, 100][scenario]
                            } else {
                                512
                            };
                            respond(
                                &mut reader,
                                200,
                                "application/json",
                                &admission_body(tokens, output, !(summary && scenario == 0)),
                            );
                        }
                        "POST /v1/chat/completions HTTP/1.1" => {
                            let v = json::view(&body).unwrap();
                            if v.get("max_tokens").unwrap().raw() == "1" {
                                failed_warms += 1;
                                respond(
                                    &mut reader,
                                    503,
                                    "application/json",
                                    r#"{"error":"mock warm failure"}"#,
                                );
                            } else {
                                generations += 1;
                                let text = if generations <= 2 {
                                    "discardable old prose ".repeat(150)
                                } else {
                                    "latest answer".into()
                                };
                                let event = format!(
                                    "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\"}}]}}\n\n{DONE}",
                                    json::quote(&text)
                                );
                                respond(&mut reader, 200, "text/event-stream", &event);
                            }
                        }
                        _ => panic!("unexpected mock route"),
                    }
                }
            }
            (generations, failed_warms)
        });
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        command
            .args([
                "agent",
                "--state-in-project",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--workspace",
            ])
            .arg(&workspace.0)
            .args([
                "--plain",
                "--verbose",
                "--approval",
                "never",
                "--session",
                "manual",
                "--prewarm-compaction",
            ]);
        let mut child = cli::Cli::start(&mut command);
        for prompt in ["first", "second", "latest"] {
            child.send(&format!("{prompt}\n"));
            child.until(true, "Native run:");
        }
        let path = workspace.0.join(".vade/sessions/manual.journal");
        let before = std::fs::read(&path).unwrap();
        child.send("/compact 1 Resolved old work.\n");
        if scenario < 2 {
            child.until(false, "Compaction refused:");
            assert_eq!(
                std::fs::read(&path).unwrap(),
                before,
                "refused plan must not write a journal record"
            );
        } else {
            child.until(true, "optional pre-warm failed or cancelled");
            assert_ne!(std::fs::read(&path).unwrap(), before);
        }
        child.send("tiny follow-up\n");
        child.until(true, "Native run:");
        child.send("/quit\n");
        let output = child.finish();
        done.store(true, Ordering::Relaxed);
        assert_eq!(server.join().unwrap(), (4, usize::from(scenario == 2)));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

struct WorkspaceFixture(std::path::PathBuf);
impl WorkspaceFixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "vade-contract-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for WorkspaceFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture_read<T>(mut read: impl FnMut() -> io::Result<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match read() {
            Ok(value) => return value,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                )
                    && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => panic!("mock read failed while waiting for request: {error}"),
        }
    }
}
// Cancellation is a fresh control capability, outside the model prompt.
fn without_cancel(body: &str) -> String {
    let value = json::view(body).unwrap();
    if let Some(key) = value.get("cancel") {
        let key = key.as_str().unwrap();
        assert!(key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()));
    }
    format!("{{{}}}", value.fields().unwrap().filter(|(k, _)| k != "cancel")
        .map(|(k, v)| format!("{}:{}", json::quote(&k), v.raw())).collect::<Vec<_>>().join(","))
}
fn request(reader: &mut BufReader<TcpStream>) -> (String, String) {
    let mut first = String::new();
    // Approval/UI work may leave a keepalive socket idle. WouldBlock from
    // nonblocking I/O or a Darwin read timeout does not mean the peer closed.
    fixture_read(|| reader.read_line(&mut first));
    let mut length = None;
    loop {
        let mut line = String::new();
        assert!(fixture_read(|| reader.read_line(&mut line)) > 0);
        if line == "\r\n" {
            break;
        }
        if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = Some(n.trim().parse::<usize>().unwrap());
        }
    }
    let length = length.unwrap();
    let mut body = Vec::with_capacity(length);
    // Retain partial bytes and Take's remaining length across transient errors.
    let mut limited = reader.take(length as u64);
    fixture_read(|| limited.read_to_end(&mut body));
    assert_eq!(body.len(), length, "truncated mock request body");
    (first.trim_end().into(), String::from_utf8(body).unwrap())
}
fn respond(reader: &mut BufReader<TcpStream>, status: u16, kind: &str, body: &str) {
    write!(
        reader.get_mut(),
        "HTTP/1.1 {status} Mock\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
}
fn listen() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    (listener, endpoint)
}
fn accept(listener: &TcpListener) -> BufReader<TcpStream> {
    listener.set_nonblocking(true).unwrap();
    let start = Instant::now();
    loop {
        match listener.accept() {
            Ok((socket, _)) => {
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                return BufReader::new(socket);
            }
            Err(e)
                if e.kind() == io::ErrorKind::WouldBlock
                    && start.elapsed() < Duration::from_secs(3) =>
            {
                thread::sleep(Duration::from_millis(2))
            }
            other => panic!("mock accept failed: {other:?}"),
        }
    }
}
fn capabilities(reader: &mut BufReader<TcpStream>, caps: &str) {
    assert_eq!(
        request(reader),
        ("GET /v1/capabilities HTTP/1.1".into(), String::new())
    );
    respond(reader, 200, "application/json", caps);
}
fn admission_body(prompt: u64, output: u64, admitted: bool) -> String {
    format!(
        r#"{{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","prompt_tokens":{prompt},"output_tokens":{output},"context_limit":4096,"input_limit":4000,"output_limit":32,"admitted":{admitted},"reason":null,"admission_kind":"context_fit","reserves_resources":false}}"#
    )
}
fn admission(reader: &mut BufReader<TcpStream>, prompt: u64, admitted: bool) -> String {
    let (line, envelope) = request(reader);
    assert_eq!(line, "POST /v1/admission HTTP/1.1");
    let v = json::view(&envelope).unwrap();
    assert_eq!(
        v.get("endpoint").unwrap().as_str().unwrap(),
        "/v1/chat/completions"
    );
    assert_eq!(
        v.get("contract_version").unwrap().as_str().unwrap(),
        "vade.engine.v1"
    );
    let body = v.get("request").unwrap().raw().to_owned();
    let output: u64 = v.get("output_tokens").unwrap().raw().parse().unwrap();
    let actual = json::view(&body).unwrap();
    assert_eq!(
        actual
            .get("max_completion_tokens")
            .or_else(|| actual.get("max_tokens"))
            .unwrap()
            .raw(),
        output.to_string()
    );
    respond(
        reader,
        200,
        "application/json",
        &admission_body(prompt, output, admitted),
    );
    body
}

// Separate admission-only count of the retained completed transcript. The
// fixture verifies this request never substitutes for a generation admission.
fn rendered_history_admission(reader: &mut BufReader<TcpStream>, tokens: u64) -> String {
    let body = admission(reader, tokens, true);
    let v = json::view(&body).unwrap();
    let messages: Vec<_> = v.get("messages").unwrap().array().unwrap().collect();
    assert_eq!(
        messages
            .last()
            .unwrap()
            .get("role")
            .unwrap()
            .as_str()
            .as_deref(),
        Some("assistant")
    );
    assert!(v.get("vade_kv").is_none(), "counting must not retain KV");
    body
}

fn usage(prompt: u64, cached: u64) -> String {
    format!(
        "data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{prompt},\"completion_tokens\":3,\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{cached}}}}}}}\n\n",
        prompt + 3
    )
}
fn closed_without_replay(reader: &mut BufReader<TcpStream>, listener: &TcpListener) {
    // Closing without draining a rejected HTTP body can produce TCP RST on
    // macOS instead of FIN. Both prove closure; any generation byte is failure.
    match reader.read(&mut [0]) {
        Ok(0) => (),
        Err(error) if error.kind() == io::ErrorKind::ConnectionReset => (),
        other => panic!("no generation may follow refusal: {other:?}"),
    }
    listener.set_nonblocking(true).unwrap();
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
}

#[test]
fn direct_exact_request_admission_stable_affinity_and_actual_cache_counts() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut prefix = String::new();
        let mut key = String::new();
        for (prompt, cached) in [(117, 0), (146, 109)] {
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, prompt, true);
            assert!(body.starts_with(&prefix));
            prefix = body[..body.len() - 2].to_owned();
            let v = json::view(&body).unwrap();
            let next = v
                .get("prompt_cache_key")
                .unwrap()
                .as_str()
                .unwrap()
                .into_owned();
            if key.is_empty() {
                key = next;
            } else {
                assert_eq!(key, next);
            }
            assert_eq!(
                request(&mut reader),
                ("POST /v1/chat/completions HTTP/1.1".into(), body)
            );
            // Duplicated final snapshots must count only once.
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(prompt, cached).repeat(2)),
            );
        }
    });
    let mut client = ChatClient::new(&endpoint).unwrap();
    let mut messages = vec![Message {
        role: "user".into(),
        content: " exact\n🦀".into(),
    }];
    for (prompt, cached) in [(117, 0), (146, 109)] {
        let result = client.stream("mock", &messages, &mut |_| {}, None).unwrap();
        let ctx = client.context();
        assert_eq!(
            ctx.input,
            InputCount::Exact {
                prompt_tokens: prompt,
                output_tokens: 32,
                context_limit: 4096,
                admitted: true
            }
        );
        assert_eq!(ctx.turns.back().unwrap().usage.cached_tokens, Some(cached));
        assert!(ctx.describe().contains(&format!("cached tokens {cached}")));
        messages.push(Message {
            role: "assistant".into(),
            content: result.text,
        });
        messages.push(Message {
            role: "user".into(),
            content: "next".into(),
        });
    }
    assert_eq!(client.context().reported_cached_tokens, 109);
    assert_eq!(client.context().responses_with_cached_usage, 2);
    server.join().unwrap();
}

struct MockTools;
impl ToolExecutor for MockTools {
    fn execute(&mut self, call: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        assert_eq!(call.call.name, "lookup");
        Ok(" exact result\n🦀 ".into())
    }
}
fn native_conversation() -> Conversation {
    Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"max_tokens":17,"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":" exact system\n"},{"role":"user","content":"hello 🦀"}]}"#).unwrap(), Limits::default()).unwrap()
}
#[test]
fn native_tool_loop_uses_complete_admission_and_keeps_decorated_prefix() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut prefix = String::new();
        for turn in 0..2 {
            if turn == 1 {
                capabilities(&mut reader, CAPS);
                let staged = admission(&mut reader, 130, true);
                assert!(staged.contains("not_executed"));
            }
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 120 + turn * 20, true);
            assert!(body.starts_with(&prefix));
            prefix = body[..body.len() - 2].into();
            assert_eq!(
                request(&mut reader),
                ("POST /v1/chat/completions HTTP/1.1".into(), body.clone())
            );
            let text = if turn == 0 {
                r#"{"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":"opaque reasoning","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":120,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":0}}}"#.to_owned()
            } else {
                let parsed = json::parse(&body).unwrap();
                let last = parsed
                    .get("messages")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap();
                assert_eq!(last.get("tool_call_id").unwrap().as_str(), Some("call-1"));
                assert_eq!(
                    last.get("content").unwrap().as_str(),
                    Some(" exact result\n🦀 ")
                );
                r#"{"choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":140,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":112}}}"#.to_owned()
            };
            respond(&mut reader, 200, "application/json", &text);
        }
    });
    let mut model = NativeModel::new(&endpoint).unwrap();
    let report = agent::run(
        &mut native_conversation(),
        &mut model,
        &mut MockTools,
        Limits::default(),
        &AtomicBool::new(false),
        &mut |_| {},
    );
    assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
    assert_eq!(report.tool_calls, 1);
    assert_eq!(model.context().turns.len(), 2);
    assert_eq!(model.context().turns[1].usage.cached_tokens, Some(112));
    server.join().unwrap();
}

#[test]
fn absent_and_unknown_contracts_degrade_without_custom_fields_or_fake_hits() {
    for status in [404, 405, 501, 200] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            assert_eq!(request(&mut reader).0, "GET /v1/capabilities HTTP/1.1");
            respond(
                &mut reader,
                status,
                "application/json",
                &CAPS.replace("vade.engine.v1", "vade.engine.future"),
            );
            for _ in 0..2 {
                let (line, body) = request(&mut reader);
                assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
                let value = json::view(&body).unwrap();
                assert!(value.get("prompt_cache_key").is_none());
                assert_eq!(value.get("max_tokens").unwrap().raw(), "8192");
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{ANSWER}{DONE}"),
                );
            }
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        for _ in 0..2 {
            client.stream("mock", &[], &mut |_| {}, None).unwrap();
            assert!(matches!(client.context().input, InputCount::Estimate(_)));
            assert_eq!(
                client.context().turns.back().unwrap().usage.cached_tokens,
                None
            );
            assert!(
                client
                    .context()
                    .describe()
                    .contains("cached tokens unknown")
            );
        }
        server.join().unwrap();
    }
}

#[test]
fn denied_exact_admission_never_sends_generation_or_replays() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        admission(&mut reader, 4090, false);
        closed_without_replay(&mut reader, &listener);
    });
    let mut client = ChatClient::new(&endpoint).unwrap();
    assert!(client.stream("mock", &[], &mut |_| {}, None).is_err());
    assert_eq!(
        client.context().input,
        InputCount::Exact {
            prompt_tokens: 4090,
            output_tokens: 32,
            context_limit: 4096,
            admitted: false
        }
    );
    assert!(client.context().describe().contains("nothing was sent"));
    drop(client);
    server.join().unwrap();
}

#[test]
fn malformed_admission_is_not_bypassed_with_estimates() {
    for response in [
        admission_body(4090, 32, true), // claims fit despite prompt + output overflow
        admission_body(100, 31, true),  // counted a different output reservation
        admission_body(100, 32, true).replace("vade.engine.v1", "future"),
        admission_body(100, 32, true).replace("\"context_limit\":4096", "\"context_limit\":8192"),
        admission_body(100, 32, true).replace("\"model\":\"mock\"", "\"model\":\"other\""),
        admission_body(100, 32, true).replace(
            "\"reserves_resources\":false",
            "\"reserves_resources\":true",
        ),
        admission_body(100, 32, true).replace("\"prompt_tokens\":100", "\"prompt_tokens\":1e2"),
    ] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            assert_eq!(request(&mut reader).0, "POST /v1/admission HTTP/1.1");
            respond(&mut reader, 200, "application/json", &response);
            closed_without_replay(&mut reader, &listener);
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        assert!(client.stream("mock", &[], &mut |_| {}, None).is_err());
        assert!(!matches!(client.context().input, InputCount::Estimate(_)));
        assert_eq!(client.context().completed_responses, 0);
        drop(client);
        server.join().unwrap();
    }
}

#[test]
fn unadvertised_tools_and_wrong_model_or_dialect_fail_before_dispatch() {
    for caps in [
        CAPS.replace("\"tools\": true", "\"tools\": null"),
        CAPS.replace("\"model\": \"mock\"", "\"model\": \"other\""),
        CAPS.replace("\"/v1/chat/completions\", ", ""),
    ] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, &caps);
            closed_without_replay(&mut reader, &listener);
        });
        let mut model = NativeModel::new(&endpoint).unwrap();
        assert!(
            model
                .complete(
                    native_conversation().request(Limits::default().wire),
                    &AtomicBool::new(false),
                    &mut |_| Ok(())
                )
                .is_err()
        );
        drop(model);
        server.join().unwrap();
    }
}

#[test]
fn restart_and_fingerprint_change_are_observable_and_affinity_is_not_a_hit() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut key = String::new();
        for turn in 0..3 {
            let caps = if turn == 0 {
                CAPS.into()
            } else {
                CAPS.replace("mock-load-epoch", "mock-restart")
                    .replace("mock-template", "mock-template-2")
            };
            capabilities(&mut reader, &caps);
            let body = admission(&mut reader, 130, true);
            let next = json::view(&body)
                .unwrap()
                .get("prompt_cache_key")
                .unwrap()
                .as_str()
                .unwrap()
                .into_owned();
            if turn == 0 {
                key = next;
            } else if turn == 1 {
                assert_eq!(
                    key, next,
                    "session affinity survives restart but says nothing about reuse"
                );
            } else {
                assert_ne!(key, next, "explicit new context gets a fresh hint");
            }
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!(
                    "{ANSWER}{}{DONE}",
                    usage(130, if turn == 0 { 100 } else { 0 })
                ),
            );
        }
    });
    let mut client = ChatClient::new(&endpoint).unwrap();
    client.stream("mock", &[], &mut |_| {}, None).unwrap();
    client.stream("mock", &[], &mut |_| {}, None).unwrap();
    assert_eq!(client.context().cache_domain_changes, 1);
    assert_eq!(
        client.context().turns.back().unwrap().usage.cached_tokens,
        Some(0)
    );
    client.reset_context();
    client.stream("mock", &[], &mut |_| {}, None).unwrap();
    assert_eq!(client.context().completed_responses, 1);
    server.join().unwrap();
}

#[test]
fn established_contract_cannot_disappear_or_downgrade_silently() {
    for next in [
        CAPS.replace("vade.engine.v1", "future"),
        CAPS.replace("\"admission\": \"/v1/admission\"", "\"admission\": null"),
    ] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{DONE}"),
            );
            capabilities(&mut reader, &next);
            closed_without_replay(&mut reader, &listener);
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        client.stream("mock", &[], &mut |_| {}, None).unwrap();
        assert!(client.stream("mock", &[], &mut |_| {}, None).is_err());
        assert!(client.context().describe().contains("/new"));
        drop(client);
        server.join().unwrap();
    }
}

#[test]
fn cancellation_during_admission_drops_socket_without_generation() {
    let (listener, endpoint) = listen();
    let cancel = AtomicBool::new(false);
    let (ready, waiting) = mpsc::channel();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        assert_eq!(request(&mut reader).0, "POST /v1/admission HTTP/1.1");
        ready.send(()).unwrap();
        closed_without_replay(&mut reader, &listener);
    });
    thread::scope(|scope| {
        let flag = &cancel;
        scope.spawn(move || {
            waiting.recv_timeout(Duration::from_secs(3)).unwrap();
            flag.store(true, Ordering::Relaxed);
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        let error = client
            .stream("mock", &[], &mut |_| {}, Some(&cancel))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
        assert_eq!(client.context().completed_responses, 0);
    });
    server.join().unwrap();
}

#[test]
fn advertised_admission_is_optional_and_unknown_features_do_not_enable_it() {
    let caps = CAPS.replace(
        "\"admission\": \"/v1/admission\"",
        "\"admission\": \"https://unsupported.invalid/count\"",
    );
    assert!(!Capabilities::parse(&caps).unwrap().unwrap().exact_admission);
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, &caps);
        let (line, body) = request(&mut reader);
        assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
        assert_eq!(json::view(&body).unwrap().get("max_tokens").unwrap().raw(), "32");
        respond(
            &mut reader,
            200,
            "text/event-stream",
            &format!("{ANSWER}{DONE}"),
        );
    });
    let mut client = ChatClient::new(&endpoint).unwrap();
    client.stream("mock", &[], &mut |_| {}, None).unwrap();
    assert!(matches!(client.context().input, InputCount::Estimate(_)));
    server.join().unwrap();
}

#[test]
fn cli_shows_exact_context_and_cached_tokens_and_new_resets_affinity() {
    use std::process::Command;
    for mode in ["direct", "agent"] {
        let workspace = WorkspaceFixture::new();
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            let mut first_key = String::new();
            for turn in 0..2 {
                // Native /new drops its model/pool; direct keeps framed sockets.
                if turn == 1 && mode == "agent" {
                    reader = accept(&listener);
                }
                capabilities(&mut reader, CAPS);
                let body = admission(&mut reader, 117, true);
                let v = json::view(&body).unwrap();
                let key = v
                    .get("prompt_cache_key")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .into_owned();
                if turn == 0 {
                    first_key = key;
                } else {
                    assert_ne!(first_key, key);
                }
                assert_eq!(
                    v.get("messages").unwrap().array().unwrap().count(),
                    if mode == "agent" { 3 } else { 2 }
                );
                assert_eq!(request(&mut reader).1, body);
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{ANSWER}{}{DONE}", usage(117, 87)),
                );
            }
        });
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        command.args([
            mode,
            "--state-in-project",
            "--verbose",
            "--endpoint",
            &endpoint,
            "--model",
            "mock",
            "--workspace",
            workspace.0.to_str().unwrap(),
            "--plain",
            "--max-output-tokens",
            "17",
        ]);
        let mut child = cli::Cli::start(&mut command);
        child.send("first\n");
        child.until(true, "session reported cached tokens:");
        child.send("/context\n/new\nsecond\n");
        child.until(true, "session reported cached tokens:");
        child.send("/context\n");
        let output = child.finish();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        for text in [&stdout, &stderr] {
            assert!(text.contains("exact 117 + output reservation 17 / context limit 4096"));
            assert!(text.contains("cached tokens 87"));
            assert!(text.contains("1/1 completed model responses"));
            assert!(!text.contains("sha256:mock"));
        }
        if mode == "agent" {
            assert!(stdout.contains("+ 17 output reserve / 4096 budget"));
        }
        server.join().unwrap();
    }
}

#[test]
fn cli_compaction_fork_switch_and_resume_reestablish_admitted_prefixes_and_report_cache_drop() {
    use std::process::Command;
    let workspace = WorkspaceFixture::new();
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut bodies: Vec<String> = Vec::new();
        let mut keys = Vec::new();
        // Three ordinary appends, compaction, fork, switch, fresh journal, and
        // a separate process's --resume, plus the opt-in discarded warm.
        for turn in 0..8 {
            if turn >= 4 {
                reader = accept(&listener);
            }
            if turn == 3 {
                // Manual /compact counts both the original completed history
                // and the reducing candidate before persistence or warm-up.
                capabilities(&mut reader, CAPS);
                rendered_history_admission(&mut reader, 768);
                capabilities(&mut reader, CAPS);
                rendered_history_admission(&mut reader, 512);
                capabilities(&mut reader, CAPS);
                rendered_history_admission(&mut reader, 512);
                let warm = admission(&mut reader, 512, true);
                let w = json::view(&warm).unwrap();
                assert_eq!(w.get("max_tokens").unwrap().raw(), "1");
                assert_eq!(
                    w.get("tools").unwrap().raw(),
                    json::view(&bodies[0]).unwrap().get("tools").unwrap().raw()
                );
                assert_eq!(
                    w.get("messages")
                        .unwrap()
                        .array()
                        .unwrap()
                        .nth(2)
                        .unwrap()
                        .get("name")
                        .unwrap()
                        .as_str()
                        .as_deref(),
                    Some("vade_context_summary")
                );
                assert_eq!(request(&mut reader).1, warm);
                respond(
                    &mut reader,
                    200,
                    "application/json",
                    r#"{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"discarded warm"}}],"usage":{"prompt_tokens":512,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":128}}}"#,
                );
            }
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 512, true);
            assert_eq!(
                request(&mut reader),
                ("POST /v1/chat/completions HTTP/1.1".into(), body.clone())
            );
            let v = json::view(&body).unwrap();
            let key = v
                .get("prompt_cache_key")
                .unwrap()
                .as_str()
                .unwrap()
                .into_owned();
            if turn > 0 && turn <= 3 {
                assert_eq!(
                    keys[0], key,
                    "same journal keeps affinity through compaction"
                );
            }
            if turn >= 4 {
                assert!(
                    !keys.contains(&key),
                    "a newly activated journal gets fresh affinity"
                );
            }
            if turn == 1 || turn == 2 {
                let previous = &bodies[turn - 1];
                let previous = without_cancel(previous);
                assert!(without_cancel(&body).starts_with(&previous[..previous.len() - 2]));
            }
            let messages = v.get("messages").unwrap().raw();
            if (3..=5).contains(&turn) || turn == 7 {
                assert!(messages.contains("User-supplied compaction summary"));
                assert!(!messages.contains("discardable old prose"));
                assert!(messages.contains("latest answer"));
            }
            if turn == 3 {
                assert!(!body.starts_with(&bodies[2][..bodies[2].len() - 2]));
                assert!(body.len() < bodies[2].len());
            }
            if turn == 4 {
                assert!(messages.contains("Re-read relevant files"));
            }
            if turn == 5 || turn == 7 {
                assert!(!messages.contains("child-only-task"));
            }
            if turn == 6 {
                assert_eq!(v.get("messages").unwrap().array().unwrap().count(), 3);
            }
            let text = if turn < 2 {
                "discardable old prose ".repeat(100)
            } else {
                "latest answer".into()
            };
            let event = format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\"}}]}}\n\n",
                json::quote(&text)
            );
            let cached = [0, 110, 220, 7, 0, 0, 0, 0][turn];
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{event}{}{DONE}", usage(512, cached)),
            );
            bodies.push(body);
            keys.push(key);
        }
    });
    let launch = |resume: bool, input: &str| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        command
            .args([
                "agent",
                "--state-in-project",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--workspace",
            ])
            .arg(&workspace.0)
            .args([
                "--plain",
                "--approval",
                "never",
                "--verbose",
                "--prewarm-compaction",
                if resume { "--resume" } else { "--session" },
                "parent",
            ]);
        let mut child = cli::Cli::start(&mut command);
        for line in input.lines() {
            child.send(&format!("{line}\n"));
            if !line.starts_with('/') {
                child.until(true, "Native run:");
            } else if line.starts_with("/compact ") {
                child.until(true, "Compacted context pre-warmed;");
            }
        }
        let output = child.finish();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        );
        (
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        )
    };
    let (stdout, stderr) = launch(
        false,
        "first task\nsecond task\nlatest task\n/compact 1 Old work resolved.\n/context\ncontinue compacted\n/context\n/fork child\n/context\nchild-only-task\n/switch parent\n/context\nparent followup\n/new\nfresh task\n/quit\n",
    );
    assert!(stdout.contains("the next prompt resends the full conversation"));
    assert!(stdout.contains("last model response: cached tokens unknown"));
    assert!(stdout.contains("cached tokens 7"));
    assert!(stdout.contains("compaction pre-warm:") && !stdout.contains("discarded warm"));
    assert!(stderr.contains("cached tokens 220") && stderr.contains("cached tokens 7"));
    assert!(stdout.contains("Compaction invalidates later causal KV"));
    let (stdout, _) = launch(true, "/context\nresumed followup\n/quit\n");
    assert!(stdout.contains("prompt boundary: resume/branch switch"));
    server.join().unwrap();
}

#[test]
fn truncated_usage_stream_does_not_publish_cache_hits() {
    for native in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 117, true);
            assert_eq!(request(&mut reader).1, body);
            let data = format!("{ANSWER}{}{DONE}", usage(117, 87));
            // Even a valid usage event and [DONE] cannot hide lost HTTP framing.
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{data}", data.len() + 1).unwrap();
        });
        if native {
            let mut model = NativeModel::new(&endpoint).unwrap();
            assert!(
                model
                    .complete(
                        native_conversation().request(Limits::default().wire),
                        &AtomicBool::new(false),
                        &mut |_| Ok(())
                    )
                    .is_err()
            );
            assert_eq!(model.context().reported_cached_tokens, 0);
            assert!(model.context().turns.is_empty());
        } else {
            let mut client = ChatClient::new(&endpoint).unwrap();
            assert!(client.stream("mock", &[], &mut |_| {}, None).is_err());
            assert_eq!(client.context().reported_cached_tokens, 0);
            assert!(client.context().turns.is_empty());
        }
        server.join().unwrap();
    }
}

#[test]
fn transport_failures_and_http_errors_during_admission_never_dispatch_generation() {
    for status in [0, 400, 404, 503] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            assert_eq!(request(&mut reader).0, "POST /v1/admission HTTP/1.1");
            if status != 0 {
                respond(
                    &mut reader,
                    status,
                    "application/json",
                    r#"{"error":{"message":"mock sensitive upstream detail"}}"#,
                );
                closed_without_replay(&mut reader, &listener);
            } else {
                drop(reader);
                let mut retry = accept(&listener);
                assert_eq!(request(&mut retry).0, "POST /v1/admission HTTP/1.1");
                // The second broken transport is final; generation stays unsent.
            }
            listener
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        let error = client.stream("mock", &[], &mut |_| {}, None).unwrap_err();
        assert!(!error.to_string().contains("sensitive"));
        assert!(client.context().describe().contains("nothing was sent"));
        drop(client);
        let listener = server.join().unwrap();
        listener.set_nonblocking(true).unwrap();
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
}

#[test]
fn explicit_native_prompt_reset_preserves_session_affinity_and_usage_counters() {
    native_prompt_reset(false);
    native_prompt_reset(true);
}

fn native_prompt_reset(prewarm: bool) {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut first_key = String::new();
        for turn in 0..2 {
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 117, true);
            let v = json::view(&body).unwrap();
            let key = v
                .get("prompt_cache_key")
                .unwrap()
                .as_str()
                .unwrap()
                .into_owned();
            if turn == 0 {
                first_key = key;
            } else {
                assert_eq!(first_key, key);
            }
            if turn == 1 && prewarm {
                let warm = admission(&mut reader, 117, true);
                let w = json::view(&warm).unwrap();
                assert_eq!(w.get("max_tokens").unwrap().raw(), "1");
                for field in ["tools", "messages", "prompt_cache_key"] {
                    assert_eq!(w.get(field).unwrap().raw(), v.get(field).unwrap().raw());
                }
                assert_eq!(request(&mut reader).1, warm);
                respond(
                    &mut reader,
                    200,
                    "application/json",
                    r#"{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"discarded warm"}}],"usage":{"prompt_tokens":117,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":100}}}"#,
                );
            }
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(117, 0)),
            );
        }
    });
    let mut model = NativeModel::new(&endpoint)
        .unwrap()
        .with_compaction_prewarm(prewarm);
    let first = native_conversation();
    assert!(matches!(
        model
            .count_input(
                first.request(Limits::default().wire),
                &AtomicBool::new(false)
            )
            .unwrap(),
        InputCount::Exact {
            prompt_tokens: 117,
            ..
        }
    ));
    model
        .complete(
            first.request(Limits::default().wire),
            &AtomicBool::new(false),
            &mut |_| Ok(()),
        )
        .unwrap();
    let requests = model.context().requests;
    assert!(model.context().response_streamed);
    assert!(model.context().first_token_elapsed.is_some());
    assert_eq!(model.context().last_http_status, Some(200));
    assert_eq!(model.context().last_http_stage, "generation");
    model.reset_prompt(); // owner-declared compaction; no change to session identity
    assert_eq!(model.context().requests, requests);
    assert_eq!(model.context().completed_responses, 1);
    assert!(!model.context().response_streamed);
    assert!(model.context().first_token_elapsed.is_none());
    assert_eq!(model.context().last_http_status, None);
    assert_eq!(model.context().last_http_stage, "none");
    assert!(model.context().turns.is_empty());
    assert!(model.context().prefix_pending);
    let changed = Conversation::from_request(
        json::parse(
            &first
                .request(Limits::default().wire)
                .to_json()
                .unwrap()
                .replace("hello 🦀", "explicitly rebuilt context"),
        )
        .unwrap(),
        Limits::default(),
    )
    .unwrap();
    model
        .count_input(
            changed.request(Limits::default().wire),
            &AtomicBool::new(false),
        )
        .unwrap();
    model
        .warm_compacted(
            changed.request(Limits::default().wire),
            &AtomicBool::new(false),
        )
        .unwrap();
    model
        .complete(
            changed.request(Limits::default().wire),
            &AtomicBool::new(false),
            &mut |_| Ok(()),
        )
        .unwrap();
    assert_eq!(model.context().completed_responses, 2);
    assert_eq!(model.context().compaction_warm.is_some(), prewarm);
    if prewarm {
        model.reset_prompt();
        let bad = Conversation::from_request(
            json::parse(
                &first
                    .request(Limits::default().wire)
                    .to_json()
                    .unwrap()
                    .replace(r#""role":"system""#, r#""role":"user""#),
            )
            .unwrap(),
            Limits::default(),
        )
        .unwrap();
        assert!(
            model
                .warm_compacted(bad.request(Limits::default().wire), &AtomicBool::new(false))
                .is_err()
        );
        assert_eq!(model.context().completed_responses, 2); // rejected before warm I/O
    }
    server.join().unwrap();
}

#[test]
fn foreign_html_discovery_and_custom_routes_keep_standard_chat_behavior() {
    for custom in [false, true] {
        let (listener, mut endpoint) = listen();
        if custom {
            endpoint.push_str("/custom/rpc");
        }
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            if !custom {
                assert_eq!(request(&mut reader).0, "GET /v1/capabilities HTTP/1.1");
                respond(
                    &mut reader,
                    200,
                    "text/html",
                    "<html>foreign service</html>",
                );
            }
            let (line, body) = request(&mut reader);
            assert_eq!(
                line,
                if custom {
                    "POST /custom/rpc HTTP/1.1"
                } else {
                    "POST /v1/chat/completions HTTP/1.1"
                }
            );
            assert!(json::view(&body).unwrap().get("prompt_cache_key").is_none());
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(100, 40)),
            );
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        client.stream("mock", &[], &mut |_| {}, None).unwrap();
        assert!(matches!(client.context().input, InputCount::Estimate(_)));
        assert_eq!(
            client.context().turns.back().unwrap().usage.cached_tokens,
            Some(40)
        );
        server.join().unwrap();
    }
}

// Workload mode uses this same first-party contract mock: exact admissions,
// persistent sockets and real native HTTP/SSE assembly. No live server is used.
fn workload_config(
    endpoint: &str,
    sessions: usize,
    turns: usize,
) -> vade_harness::workload::Config {
    vade_harness::workload::Config {
        endpoint: endpoint.into(),
        model: "mock".into(),
        json_path: "unused-metrics.json".into(),
        sessions,
        main_sessions: 1,
        turns,
        main_system_bytes: 128,
        sub_system_bytes: 32,
        main_user_bytes: 64,
        sub_user_bytes: 16,
        main_tool_bytes: 64,
        sub_tool_bytes: 16,
        output_tokens: 17,
        max_seconds: 10,
        ..vade_harness::workload::Config::default()
    }
}
fn member<'a>(value: &'a json::Value, path: &[&str]) -> &'a json::Value {
    path.iter().fold(value, |v, key| v.get(key).unwrap())
}
fn integer(value: &json::Value) -> u64 {
    let json::Value::Number(n) = value else {
        panic!("not a number: {value:?}")
    };
    n.parse().unwrap()
}
fn real(value: &json::Value) -> f64 {
    let json::Value::Number(n) = value else {
        panic!("not a number: {value:?}")
    };
    n.parse().unwrap()
}

#[test]
fn workload_concurrent_native_sessions_grow_and_report_per_role() {
    use std::sync::{Arc, Mutex, atomic::AtomicUsize};
    let (listener, endpoint) = listen();
    let active = Arc::new(AtomicUsize::new(0));
    let keys = Arc::new(Mutex::new(std::collections::HashSet::new()));
    let server_active = Arc::clone(&active);
    let server_keys = Arc::clone(&keys);
    let server = thread::spawn(move || {
        let mut workers = Vec::new();
        for _ in 0..6 {
            let mut reader = accept(&listener);
            let active = Arc::clone(&server_active);
            let keys = Arc::clone(&server_keys);
            workers.push(thread::spawn(move || {
                let mut prefix = String::new();
                let mut affinity = String::new();
                let mut main = false;
                for turn in 1..=3 {
                    if turn > 1 {
                        rendered_history_admission(
                            &mut reader,
                            (if main { 1000 } else { 100 }) + (turn as u64 - 2) * 100 + 3,
                        );
                    }
                    capabilities(&mut reader, CAPS);
                    let (line, envelope) = request(&mut reader);
                    assert_eq!(line, "POST /v1/admission HTTP/1.1");
                    let envelope = json::view(&envelope).unwrap();
                    let body = envelope.get("request").unwrap().raw().to_owned();
                    let v = json::view(&body).unwrap();
                    let messages: Vec<_> = v.get("messages").unwrap().array().unwrap().collect();
                    assert_eq!(messages.len(), 4 * turn);
                    assert_eq!(
                        messages[messages.len() - 2]
                            .get("role")
                            .unwrap()
                            .as_str()
                            .unwrap(),
                        "tool"
                    );
                    if turn > 1 {
                        assert_eq!(
                            messages[messages.len() - 4]
                                .get("content")
                                .unwrap()
                                .as_str()
                                .unwrap(),
                            "answer"
                        );
                    }
                    assert!(body.starts_with(&prefix), "real request prefix changed");
                    prefix = body[..body.len() - 2].into();
                    let key = v
                        .get("prompt_cache_key")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .into_owned();
                    if turn == 1 {
                        assert!(keys.lock().unwrap().insert(key.clone()));
                        affinity = key.clone();
                        main = messages[0]
                            .get("content")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .contains("role main.");
                    }
                    assert_eq!(key, affinity);
                    let prompt = (if main { 1000 } else { 100 }) + (turn as u64 - 1) * 100;
                    respond(
                        &mut reader,
                        200,
                        "application/json",
                        &admission_body(prompt, 17, true),
                    );
                    assert_eq!(
                        request(&mut reader),
                        ("POST /v1/chat/completions HTTP/1.1".into(), body)
                    );
                    if turn == 1 {
                        active.fetch_add(1, Ordering::SeqCst);
                        let start = Instant::now();
                        while active.load(Ordering::SeqCst) < 6 {
                            assert!(
                                start.elapsed() < Duration::from_secs(3),
                                "sessions were serialized"
                            );
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                    // Delay all turns so wall-window aggregate can be compared
                    // with per-session output rates without sub-ms noise.
                    thread::sleep(Duration::from_millis(10));
                    let u = prompt - 100 + 3;
                    let cached = if turn == 1 {
                        0
                    } else if main {
                        u + 2
                    } else {
                        u / 2
                    };
                    respond(
                        &mut reader,
                        200,
                        "text/event-stream",
                        &format!("{ANSWER}{}{DONE}", usage(prompt, cached)),
                    );
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let report = vade_harness::workload::run(workload_config(&endpoint, 6, 3)).unwrap();
    server.join().unwrap();
    assert!(report.successful());
    assert_eq!(keys.lock().unwrap().len(), 6);
    let body = report.to_json().unwrap();
    assert!(!body.contains(&endpoint));
    assert!(!body.contains("synthetic-0-1"));
    assert!(!body.contains("vade-"));
    let doc = json::parse(&body).unwrap();
    let main = member(&doc, &["roles", "main"]);
    let sub = member(&doc, &["roles", "subordinate"]);
    assert_eq!(integer(member(main, &["sum_u"])), 2106);
    assert_eq!(integer(member(main, &["sum_min_c_u"])), 2106); // min(C,U), not C
    assert_eq!(real(member(main, &["cache_ratio"])), 1.0);
    assert_eq!(integer(member(sub, &["sum_u"])), 1530);
    assert_eq!(integer(member(sub, &["sum_min_c_u"])), 760);
    assert!((real(member(sub, &["cache_ratio"])) - 760.0 / 1530.0).abs() < 1e-10);
    assert_eq!(member(main, &["u_estimated"]), &json::Value::Bool(true));
    assert_eq!(
        member(main, &["exact_target_verified"]),
        &json::Value::Bool(false)
    );
    assert_eq!(integer(member(main, &["admission_based_samples"])), 2);
    assert_eq!(integer(member(sub, &["ttft_samples"])), 10);
    assert_eq!(
        integer(member(&doc, &["all_turn_requests", "generations"])),
        18
    );
    assert_eq!(
        integer(member(&doc, &["aggregate", "requests", "generations"])),
        12
    );
    let aggregate = real(member(&doc, &["aggregate", "output_tok_s"]));
    let sessions = member(&doc, &["sessions"]).as_array().unwrap();
    for session in sessions {
        let rate = real(member(session, &["summary", "output_tok_s"]));
        assert!(
            aggregate > rate * 2.0,
            "aggregate must use overlapping wall time"
        );
        let turns = member(session, &["turns"]).as_array().unwrap();
        assert_eq!(member(&turns[0], &["u"]), &json::Value::Null);
        assert_eq!(member(&turns[0], &["measured"]), &json::Value::Bool(false));
        assert!(real(member(&turns[1], &["ttft_ms"])) >= 9.0);
    }
    println!("{}", report.table());
}

#[test]
fn workload_missing_telemetry_and_json_fallback_cannot_claim_a_pass_or_ttft() {
    for json_response in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            assert_eq!(request(&mut reader).0, "GET /v1/capabilities HTTP/1.1");
            respond(&mut reader, 404, "application/json", "{}");
            let mut prefix = String::new();
            for turn in 1..=3 {
                let (line, body) = request(&mut reader);
                assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
                assert!(body.starts_with(&prefix));
                prefix = body[..body.len() - 2].into();
                let v = json::view(&body).unwrap();
                assert!(v.get("prompt_cache_key").is_none());
                assert_eq!(
                    v.get("stream_options")
                        .unwrap()
                        .get("include_usage")
                        .unwrap()
                        .raw(),
                    "true"
                );
                let telemetry = if turn == 2 {
                    "".into()
                } else {
                    usage(1000, 1000)
                };
                if json_response {
                    respond(
                        &mut reader,
                        200,
                        "application/json",
                        r#"{"choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}"#,
                    );
                } else {
                    respond(
                        &mut reader,
                        200,
                        "text/event-stream",
                        &format!("{ANSWER}{telemetry}{DONE}"),
                    );
                }
            }
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 3)).unwrap();
        server.join().unwrap();
        assert!(report.successful());
        let doc = json::parse(&report.to_json().unwrap()).unwrap();
        let main = member(&doc, &["roles", "main"]);
        assert_eq!(member(main, &["cache_ratio"]), &json::Value::Null);
        assert_eq!(member(main, &["output_tok_s"]), &json::Value::Null);
        assert_eq!(
            member(main, &["target_status"]).as_str(),
            Some("insufficient_data")
        );
        assert_eq!(
            integer(member(&doc, &["all_turn_requests", "admissions"])),
            0
        );
        if json_response {
            assert_eq!(member(main, &["ttft_p50_ms"]), &json::Value::Null);
        }
        assert!(report.sessions[0].samples[1].u_estimated);
        assert_eq!(report.sessions[0].samples[1].u_basis, "wire_bytes_estimate");
    }
}

#[test]
fn workload_admission_refusal_and_http_overload_are_counted_without_replay() {
    // Status failures at both admission and generation; context-fit refusal is
    // separate from an HTTP 429/503, and each occurs after one successful turn.
    for (stage, status) in [
        ("admission", 200),
        ("admission", 429),
        ("admission", 503),
        ("generation", 429),
        ("generation", 503),
    ] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(100, 0)),
            );
            rendered_history_admission(&mut reader, 103);
            capabilities(&mut reader, CAPS);
            if stage == "admission" && status == 200 {
                admission(&mut reader, 4000, false);
            } else if stage == "admission" {
                assert_eq!(request(&mut reader).0, "POST /v1/admission HTTP/1.1");
                respond(
                    &mut reader,
                    status,
                    "application/json",
                    "{\"private\":\"must not log body\"}",
                );
            } else {
                let body = admission(&mut reader, 200, true);
                assert_eq!(request(&mut reader).1, body);
                respond(
                    &mut reader,
                    status,
                    "application/json",
                    "{\"private\":\"must not log body\"}",
                );
            }
            closed_without_replay(&mut reader, &listener);
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 3)).unwrap();
        server.join().unwrap();
        assert!(!report.successful());
        assert_eq!(report.sessions[0].samples.len(), 2);
        let raw = report.to_json().unwrap();
        assert!(!raw.contains("must not log body"));
        let doc = json::parse(&raw).unwrap();
        let main = member(&doc, &["roles", "main"]);
        assert_eq!(integer(member(main, &["errors"])), 1);
        assert_eq!(integer(member(main, &["unattempted_samples"])), 1);
        assert_eq!(integer(member(main, &["requests", "admissions"])), 1);
        assert_eq!(
            integer(member(main, &["requests", "refused"])),
            u64::from(status == 200)
        );
        assert_eq!(
            integer(member(main, &["requests", "http_429"])),
            u64::from(status == 429)
        );
        assert_eq!(
            integer(member(main, &["requests", "http_503"])),
            u64::from(status == 503)
        );
        assert_eq!(member(main, &["cache_ratio"]), &json::Value::Null);
        let sample = &report.sessions[0].samples[1];
        assert_eq!(sample.last_http_stage, stage);
        assert_eq!(sample.last_http_status, Some(status));
    }
}

#[test]
fn cli_generation_flags_override_env_and_output_limit_refuses_before_generation() {
    for mode in ["direct", "agent"] {
        for (flag, env, expected, cache_env, cache) in [
            (Some("on"), "0", true, "1", true),
            (Some("off"), "1", false, "0", false),
            (None, "1", true, "1", true),
        ] {
            let f = WorkspaceFixture::new();
            let (listener, endpoint) = listen();
            let server = thread::spawn(move || {
                let mut reader = accept(&listener);
                capabilities(&mut reader, CAPS);
                let body = admission(&mut reader, 100, true);
                let request_value = json::parse(&body).unwrap();
                assert_eq!(
                    request_value.get("max_tokens"),
                    Some(&json::Value::Number("17".into()))
                );
                assert_eq!(
                    request_value.get("enable_thinking"),
                    Some(&json::Value::Bool(expected))
                );
                assert_eq!(request_value.get("cache"), Some(&json::Value::Bool(cache)));
                if mode == "agent" {
                    let system = request_value.get("messages").unwrap().as_array().unwrap()[0]
                        .get("content")
                        .unwrap()
                        .as_str()
                        .unwrap();
                    assert!(system.contains("make that tool call in the same reply; never end a turn with only an announcement"));
                }
                assert_eq!(request(&mut reader).1, body);
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{ANSWER}{}{DONE}", usage(100, 0)),
                );
            });
            let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_vade-harness"));
            command
                .args([
                    mode,
                    "--state-in-project",
                    "--endpoint",
                    &endpoint,
                    "--model",
                    "mock",
                    "--prompt",
                    "flags",
                    "--max-output-tokens",
                    "17",
                    "--workspace",
                ])
                .arg(&f.0)
                .env("VADE_HARNESS_THINKING", env)
                .env("VADE_HARNESS_PREFIX_CACHE", cache_env);
            if let Some(flag) = flag {
                command.args(["--thinking", flag]);
            }
            let output = cli::Cli::start(&mut command).finish();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            server.join().unwrap();
        }
        let f = WorkspaceFixture::new();
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            assert_eq!(
                reader.read(&mut [0]).unwrap(),
                0,
                "no admission or generation after oversized output"
            );
        });
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        command
            .args([
                mode,
                "--state-in-project",
                "--verbose",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--prompt",
                "too much",
                "--max-output-tokens",
                "33",
                "--workspace",
            ])
            .arg(&f.0);
        let output = cli::Cli::start(&mut command).finish();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("the engine's output limit"));
        server.join().unwrap();
    }
}

#[test]
fn response_timing_in_native_and_direct_excludes_role_and_usage_trailer_waits() {
    for native in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            admission(&mut reader, 100, true);
            request(&mut reader);
            let role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
            let first = "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n";
            let tail = format!("{}{DONE}", usage(100, 0));
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{role}", role.len() + first.len() + ANSWER.len() + tail.len()).unwrap();
            thread::sleep(Duration::from_millis(40));
            reader.get_mut().write_all(first.as_bytes()).unwrap();
            thread::sleep(Duration::from_millis(80));
            reader.get_mut().write_all(ANSWER.as_bytes()).unwrap();
            thread::sleep(Duration::from_millis(160));
            reader.get_mut().write_all(tail.as_bytes()).unwrap();
        });
        let context = if native {
            let mut model = NativeModel::new(&endpoint).unwrap();
            let conversation = Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"messages":[{"role":"user","content":"timing"}]}"#).unwrap(), Limits::default()).unwrap();
            model
                .complete(
                    conversation.request(Limits::default().wire),
                    &AtomicBool::new(false),
                    &mut |_| Ok(()),
                )
                .unwrap();
            model.context().clone()
        } else {
            let mut client = ChatClient::new(&endpoint).unwrap();
            client
                .stream(
                    "mock",
                    &[Message {
                        role: "user".into(),
                        content: "timing".into(),
                    }],
                    &mut |_| {},
                    None,
                )
                .unwrap();
            client.context().clone()
        };
        server.join().unwrap();
        let turn = context.turns.back().unwrap();
        assert!(turn.first_token_elapsed.unwrap() >= Duration::from_millis(30));
        let decode = turn.decode_elapsed.unwrap();
        assert!(
            decode >= Duration::from_millis(60) && decode < Duration::from_millis(180),
            "{decode:?}"
        );
        assert!(turn.speed().contains("prompt 100 tokens · cached 0 tokens"));
    }
}

#[test]
fn workload_ttft_includes_reasoning_but_not_role_event_or_final_usage() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        for turn in 1..=2 {
            if turn > 1 {
                rendered_history_admission(&mut reader, (turn - 1) * 100 + 3);
            }
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, turn * 100, true);
            assert_eq!(request(&mut reader).1, body);
            let role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
            let reasoning =
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"considering\"}}]}\n\n";
            let tail = format!("{ANSWER}{}{DONE}", usage(turn * 100, 100));
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{role}", role.len() + reasoning.len() + tail.len()).unwrap();
            thread::sleep(Duration::from_millis(40));
            reader.get_mut().write_all(reasoning.as_bytes()).unwrap();
            thread::sleep(Duration::from_millis(100));
            reader.get_mut().write_all(tail.as_bytes()).unwrap();
        }
    });
    let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
    server.join().unwrap();
    assert!(report.successful());
    let sample = &report.sessions[0].samples[1];
    let ttft = sample.ttft_ms.unwrap();
    assert!(ttft >= 30.0, "role event must not count as a token: {ttft}");
    assert!(
        (sample.end_s - sample.start_s) * 1000.0 - ttft >= 70.0,
        "must measure reasoning before answer text"
    );
}

#[test]
fn workload_reasoning_only_replies_round_trip_eight_turns_with_exact_prefixes() {
    // One streaming session and one JSON fallback session, both length-limited
    // like the reported native failure. Validate the history at admission and
    // require those exact same bytes at generation, on the original socket.
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut workers = Vec::new();
        for streaming in [true, false] {
            let mut reader = accept(&listener);
            workers.push(thread::spawn(move || {
                let mut prefix = String::new();
                for turn in 1..=8 {
                    if turn > 1 { rendered_history_admission(&mut reader, (turn - 1) * 100 + 3); }
                    capabilities(&mut reader, CAPS);
                    let body = admission(&mut reader, turn * 100, true);
                    assert!(body.starts_with(&prefix), "old request bytes changed on turn {turn}");
                    prefix = body[..body.len() - 2].into();
                    let request_value = json::parse(&body).unwrap();
                    let messages = request_value.get("messages").unwrap().as_array().unwrap();
                    let mut reasoning_turns = 0;
                    for message in messages {
                        if message.get("role").and_then(json::Value::as_str) != Some("assistant") {
                            continue;
                        }
                        let has_calls = message.get("tool_calls").and_then(json::Value::as_array).is_some_and(|calls| !calls.is_empty());
                        // Mirrors the documented Engine shape constraint only;
                        // no engine implementation, tokenizer, or code is used.
                        assert!(has_calls || matches!(message.get("content"), Some(json::Value::String(_) | json::Value::Array(_))), "assistant lacks legal content on turn {turn}");
                        if let Some(reasoning) = message.get("reasoning_content").and_then(json::Value::as_str) {
                            reasoning_turns += 1;
                            assert_eq!(message.get("content").and_then(json::Value::as_str), Some(""));
                            assert_eq!(reasoning, format!("thinking {reasoning_turns}\n 🦀"));
                            assert_eq!(message.get("provider").unwrap().get("opaque"), Some(&json::Value::Bool(true)));
                        }
                    }
                    assert_eq!(reasoning_turns, turn - 1);
                    assert_eq!(request(&mut reader), ("POST /v1/chat/completions HTTP/1.1".into(), body));
                    let content = if turn % 2 == 0 { ",\"content\":null" } else { "" };
                    if streaming {
                        let role = format!("data: {{\"choices\":[{{\"delta\":{{\"role\":\"assistant\"{content}}}}}]}}\n\n");
                        let first = format!("data: {{\"choices\":[{{\"delta\":{{\"reasoning_content\":\"thinking {turn}\"}}}}]}}\n\n");
                        let last = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"\\n 🦀\",\"provider\":{\"opaque\":true}},\"finish_reason\":\"length\"}]}\n\n";
                        respond(&mut reader, 200, "text/event-stream", &format!("{role}{first}{last}{}{DONE}", usage(turn * 100, 0)));
                    } else {
                        respond(&mut reader, 200, "application/json", &format!(r#"{{"choices":[{{"message":{{"role":"assistant"{content},"reasoning_content":"thinking {turn}\n 🦀","provider":{{"opaque":true}}}},"finish_reason":"length"}}],"usage":{{"prompt_tokens":{},"completion_tokens":3}}}}"#, turn * 100));
                    }
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let report = vade_harness::workload::run(workload_config(&endpoint, 2, 8)).unwrap();
    server.join().unwrap();
    assert!(report.successful());
    for session in &report.sessions {
        assert_eq!(session.samples.len(), 8);
        assert!(session.stopped.is_none());
        for sample in &session.samples {
            assert_eq!(sample.outcome, "length");
            assert!(sample.error.is_none());
            assert!(sample.prefix_preserved);
            assert_eq!(sample.requests.admissions, 1);
            assert_eq!(sample.requests.generations, 1);
        }
    }
}

#[test]
fn native_queue_waits_beyond_thirty_seconds_before_headers_or_output() {
    // Exercise real socket polls and the production timeout policy. These three
    // fixtures run concurrently so the regression adds about 31s, not 93s.
    thread::scope(|scope| {
        for kind in ["headers", "role", "json"] {
            scope.spawn(move || {
                let (listener, endpoint) = listen();
                let server = thread::spawn(move || {
                    let mut reader = accept(&listener);
                    for turn in 1..=2 {
                        if turn > 1 { rendered_history_admission(&mut reader, (turn - 1) * 100 + 3); }
                    capabilities(&mut reader, CAPS);
                        let body = admission(&mut reader, turn * 100, true);
                        assert_eq!(request(&mut reader).1, body);
                        let role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
                        let answer = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"still thinking\"},\"finish_reason\":\"length\"}]}\n\n";
                        let tail = format!("{answer}{}{DONE}", usage(turn * 100, 0));
                        let json_body = format!(r#"{{"choices":[{{"message":{{"role":"assistant","reasoning_content":"still thinking"}},"finish_reason":"length"}}],"usage":{{"prompt_tokens":{},"completion_tokens":3}}}}"#, turn * 100);
                        if kind == "role" {
                            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{role}", role.len() + tail.len()).unwrap();
                        } else if kind == "json" {
                            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", json_body.len()).unwrap();
                        }
                        if turn == 1 { thread::sleep(Duration::from_secs(31)); }
                        match kind {
                            "role" => reader.get_mut().write_all(tail.as_bytes()).unwrap(),
                            "json" => reader.get_mut().write_all(json_body.as_bytes()).unwrap(),
                            _ => respond(&mut reader, 200, "text/event-stream", &tail),
                        }
                    }
                    // Two requests on this socket; no timeout reconnect/replay.
                });
                let mut config = workload_config(&endpoint, 1, 2);
                config.max_seconds = 60;
                let report = vade_harness::workload::run(config).unwrap();
                server.join().unwrap();
                assert!(report.successful(), "{kind}: {report:?}");
                assert_eq!(report.sessions[0].samples.len(), 2);
                let first = &report.sessions[0].samples[0];
                assert!(first.end_s - first.start_s >= 30.0);
                if kind != "json" { assert!(first.ttft_ms.unwrap() >= 30_000.0); }
            });
        }
    });
}

#[test]
fn model_progress_waits_for_generated_reasoning_and_clears_on_cancel_or_completion() {
    use std::sync::Arc;
    use vade_harness::transport::NativePhase;
    for (native, cancel_after_reasoning) in
        [(true, false), (true, true), (false, false), (false, true)]
    {
        let (listener, endpoint) = listen();
        let (role_tx, role_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, body);
            let role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
            let reasoning = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\",\"content\":\"\\n\\n\"}}]}\n\n";
            let done = format!(
                "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"length\"}}]}}\n\n{DONE}"
            );
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{role}", role.len() + reasoning.len() + done.len()).unwrap();
            role_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
            reader.get_mut().write_all(reasoning.as_bytes()).unwrap();
            if !cancel_after_reasoning {
                reader.get_mut().write_all(done.as_bytes()).unwrap();
            }
            closed_without_replay(&mut reader, &listener);
        });
        let (phase_tx, phase_rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let worker = thread::spawn(move || {
            let observer = Arc::new(move |phase| phase_tx.send(phase).unwrap());
            if native {
                let mut model = NativeModel::new(&endpoint).unwrap();
                model.set_progress_observer(Some(observer));
                model
                    .complete(
                        native_conversation().request(Limits::default().wire),
                        &worker_cancel,
                        &mut |_| panic!("reasoning must not be shown as answer text"),
                    )
                    .map(|turn| turn.finish_reason() == &FinishReason::Length)
            } else {
                let mut client = ChatClient::new(&endpoint).unwrap();
                client.set_progress_observer(Some(observer));
                client
                    .stream(
                        "mock",
                        &[],
                        &mut |_| panic!("reasoning must not be shown as answer text"),
                        Some(&worker_cancel),
                    )
                    .map(|turn| turn.finish_reason.as_deref() == Some("length"))
            }
        });
        let phase = || phase_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        assert_eq!(phase(), NativePhase::Preparing);
        assert_eq!(phase(), NativePhase::Queued);
        role_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        // A role event alone must not move the UI to generating.
        assert!(phase_rx.recv_timeout(Duration::from_millis(50)).is_err());
        release_tx.send(()).unwrap();
        assert_eq!(phase(), NativePhase::Thinking);
        if cancel_after_reasoning {
            cancel.store(true, Ordering::Relaxed);
        }
        assert_eq!(phase(), NativePhase::Idle);
        let result = worker.join().unwrap();
        if cancel_after_reasoning {
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
        } else {
            assert!(result.unwrap());
        }
        server.join().unwrap();
    }
}

#[test]
fn workload_cli_writes_private_json_and_one_table_without_clobbering() {
    use std::process::Command;
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        for turn in 1..=2 {
            if turn > 1 {
                rendered_history_admission(&mut reader, (turn - 1) * 100 + 3);
            }
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, turn * 100, true);
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!(
                    "{ANSWER}{}{DONE}",
                    usage(turn * 100, if turn == 1 { 0 } else { 103 })
                ),
            );
        }
    });
    let file = std::env::temp_dir().join(format!(
        "vade-workload-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let args = [
        "workload",
        "--endpoint",
        &endpoint,
        "--model",
        "mock",
        "--sessions",
        "1",
        "--turns",
        "2",
        "--output-tokens",
        "17",
        "--json",
        file.to_str().unwrap(),
    ];
    let output = Command::new(env!("CARGO_BIN_EXE_vade-harness"))
        .args(args)
        .output()
        .unwrap();
    server.join().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.matches("Scope             Done/plan").count(), 1);
    assert!(stdout.contains("~1.0000"));
    assert!(!stdout.contains("Synthetic observation"));
    let original = std::fs::read_to_string(&file).unwrap();
    json::parse(&original).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    // No server remains. Existing-file rejection must occur before any network.
    let rejected = Command::new(env!("CARGO_BIN_EXE_vade-harness"))
        .args(args)
        .output()
        .unwrap();
    assert_eq!(rejected.status.code(), Some(2));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
    std::fs::remove_file(file).unwrap();
}

#[test]
fn workload_deadline_cancels_and_joins_a_pending_native_read() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        let body = admission(&mut reader, 100, true);
        assert_eq!(request(&mut reader).1, body);
        // Leave generation pending. The workload owns and must close this socket.
        assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    });
    let mut config = workload_config(&endpoint, 1, 3);
    config.max_seconds = 1;
    let start = Instant::now();
    let report = vade_harness::workload::run(config).unwrap();
    server.join().unwrap();
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(report.timed_out);
    assert!(!report.successful());
    assert_eq!(report.sessions[0].samples.len(), 1);
    assert_eq!(report.sessions[0].samples[0].error, Some("cancelled"));
    assert_eq!(report.sessions[0].samples[0].requests.generations, 1);
    assert_eq!(report.sessions[0].samples[0].usage.cached_tokens, None);
}

#[test]
fn workload_truncation_and_unrequested_tools_never_become_success_or_real_execution() {
    for tool in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, body);
            if tool {
                respond(
                    &mut reader,
                    200,
                    "application/json",
                    r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"unexpected","type":"function","function":{"name":"workspace_read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
                );
                closed_without_replay(&mut reader, &listener);
            } else {
                let body = format!("{ANSWER}{}{DONE}", usage(100, 100));
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len() + 5).unwrap();
            }
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 3)).unwrap();
        server.join().unwrap();
        assert!(!report.successful());
        assert_eq!(report.sessions[0].samples.len(), 1);
        let sample = &report.sessions[0].samples[0];
        assert_eq!(sample.usage.cached_tokens, None);
        assert!(sample.ttft_ms.is_none());
        assert_eq!(
            sample.error,
            Some(if tool {
                "native_runtime_limit"
            } else {
                "incomplete_response"
            })
        );
    }
}

#[test]
fn workload_length_finish_is_measured_and_zero_cache_is_distinct_from_zero_denominator() {
    for zero_prompt in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            for turn in 1..=2 {
                if turn > 1 {
                    rendered_history_admission(&mut reader, if zero_prompt { 0 } else { 100 });
                }
                capabilities(&mut reader, CAPS);
                let prompt = if zero_prompt { 0 } else { 100 };
                let body = admission(&mut reader, prompt, true);
                assert_eq!(request(&mut reader).1, body);
                let answer = ANSWER.replace("stop", "length");
                let telemetry =
                    usage(prompt, 0).replace("\"completion_tokens\":3", "\"completion_tokens\":0");
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{answer}{telemetry}{DONE}"),
                );
            }
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
        server.join().unwrap();
        assert!(report.successful());
        assert_eq!(report.sessions[0].samples[1].outcome, "length");
        assert_eq!(report.sessions[0].samples[1].usage.cached_tokens, Some(0));
        let doc = json::parse(&report.to_json().unwrap()).unwrap();
        let main = member(&doc, &["roles", "main"]);
        if zero_prompt {
            assert_eq!(member(main, &["cache_ratio"]), &json::Value::Null);
            assert_eq!(
                member(main, &["target_status"]).as_str(),
                Some("insufficient_data")
            );
        } else {
            assert_eq!(real(member(main, &["cache_ratio"])), 0.0);
            assert_eq!(
                member(main, &["target_status"]).as_str(),
                Some("estimate_below_target")
            );
        }
    }
}

#[test]
fn workload_contract_identity_change_and_inconsistent_counts_are_visible() {
    for mismatch in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            for turn in 1..=2 {
                let caps = if turn == 2 && !mismatch {
                    CAPS.replace("mock-template", "mock-template-v2").replacen(
                        "\"object\"",
                        "\"build_hash\":\"build-v2\",\"object\"",
                        1,
                    )
                } else {
                    CAPS.into()
                };
                if turn > 1 {
                    rendered_history_admission(&mut reader, (turn - 1) * 100 + 3);
                }
                capabilities(&mut reader, &caps);
                let body = admission(&mut reader, 100 * turn, true);
                assert_eq!(request(&mut reader).1, body);
                let prompt = if turn == 2 && mismatch {
                    201
                } else {
                    100 * turn
                };
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{ANSWER}{}{DONE}", usage(prompt, 100)),
                );
            }
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
        server.join().unwrap();
        let sample = &report.sessions[0].samples[1];
        if mismatch {
            assert_eq!(sample.error, Some("admission_usage_mismatch"));
            assert!(!report.successful());
        } else {
            assert_eq!(sample.cache_domain_changes, 1);
            assert_eq!(sample.u, None);
            assert_eq!(sample.u_basis, "unavailable_cache_identity_change");
        }
        let doc = json::parse(&report.to_json().unwrap()).unwrap();
        assert_eq!(
            member(&doc, &["roles", "main", "cache_ratio"]),
            &json::Value::Null
        );
    }
}

#[test]
fn workload_counts_full_rendered_history_instead_of_generated_output_and_prints_each_turn() {
    for rendered in [173, 400] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let first = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, first);
            let assistant = r#"{"role":"assistant","content":"answer","reasoning_content":"retained reasoning","provider":{"opaque":true}}"#;
            // Output count absent: exact rendering must not use a byte fallback.
            respond(
                &mut reader,
                200,
                "application/json",
                &format!(
                    r#"{{"choices":[{{"message":{assistant},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":100,"prompt_tokens_details":{{"cached_tokens":0}}}}}}"#
                ),
            );
            let closed = rendered_history_admission(&mut reader, rendered);
            let v = json::view(&closed).unwrap();
            let messages: Vec<_> = v.get("messages").unwrap().array().unwrap().collect();
            assert_eq!(messages.last().unwrap().raw(), assistant);
            assert_eq!(messages.len(), 5);
            capabilities(&mut reader, CAPS);
            let next = admission(&mut reader, 300, true);
            assert!(next.starts_with(first.strip_suffix("]}").unwrap()));
            assert_eq!(request(&mut reader).1, next);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(300, 160)),
            );
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
        server.join().unwrap();
        assert!(report.successful());
        let sample = &report.sessions[0].samples[1];
        assert_eq!(sample.u, Some(rendered.min(300)));
        assert_eq!(sample.rendered_previous_tokens, Some(rendered));
        assert_eq!(sample.u_basis, "exact_rendered_history_counts");
        assert!(sample.u_estimated);
        assert_eq!(sample.requests.admissions, 1);
        assert_eq!(sample.requests.generations, 1);
        assert_eq!(sample.prefix_count_requests.admissions, 1);
        assert_eq!(sample.prefix_count_requests.generations, 0);
        assert!(matches!(
            sample.input,
            InputCount::Exact {
                prompt_tokens: 300,
                ..
            }
        ));
        let table = report.table();
        assert!(table.contains("Per-turn cache"));
        assert!(
            table.contains("first_turn / stop")
                && table.contains("exact_rendered_history_counts / stop")
        );
        assert!(table.contains(if rendered == 173 { "92.49" } else { "53.33" }));
        assert!(!table.contains("retained reasoning"));
        let doc = json::parse(&report.to_json().unwrap()).unwrap();
        assert_eq!(
            integer(member(&doc, &["prefix_count_requests", "admissions"])),
            1
        );
        assert_eq!(
            integer(member(&doc, &["all_turn_requests", "generations"])),
            2
        );
    }
}

#[test]
fn workload_failed_render_count_leaves_u_unknown_and_counts_the_probe_without_replay() {
    for mode in 0..3 {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let first = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, first);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(100, 0)),
            );
            let (line, envelope) = request(&mut reader);
            assert_eq!(line, "POST /v1/admission HTTP/1.1");
            assert_eq!(
                json::view(&envelope)
                    .unwrap()
                    .get("request")
                    .unwrap()
                    .get("messages")
                    .unwrap()
                    .array()
                    .unwrap()
                    .last()
                    .unwrap()
                    .get("role")
                    .unwrap()
                    .as_str()
                    .as_deref(),
                Some("assistant")
            );
            match mode {
                0 => respond(
                    &mut reader,
                    429,
                    "application/json",
                    "{\"private\":\"count failure sentinel\"}",
                ),
                1 => respond(
                    &mut reader,
                    200,
                    "application/json",
                    &admission_body(4000, 17, false),
                ),
                _ => respond(
                    &mut reader,
                    200,
                    "application/json",
                    &admission_body(173, 17, true)
                        .replace("\"output_tokens\":17", "\"output_tokens\":16"),
                ),
            }
            // HTTP/invalid metadata drops that observation's socket; the next
            // generation opens another, without replaying the count request.
            if mode != 1 {
                reader = accept(&listener);
            }
            capabilities(&mut reader, CAPS);
            let next = admission(&mut reader, 300, true);
            assert_eq!(request(&mut reader).1, next);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(300, 160)),
            );
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
        server.join().unwrap();
        assert!(
            report.successful(),
            "generations complete even if measurement fails"
        );
        let sample = &report.sessions[0].samples[1];
        assert_eq!(sample.u, None);
        assert_eq!(sample.u_basis, "unavailable_rendered_history_count_failed");
        assert_eq!(sample.prefix_count_requests.admissions, 1);
        assert_eq!(sample.prefix_count_requests.http_429, u64::from(mode == 0));
        assert_eq!(sample.prefix_count_requests.refused, u64::from(mode == 1));
        let raw = report.to_json().unwrap();
        assert!(!raw.contains("count failure sentinel"));
        let doc = json::parse(&raw).unwrap();
        assert_eq!(
            member(&doc, &["roles", "main", "cache_ratio"]),
            &json::Value::Null
        );
    }
}

#[test]
fn both_cli_modes_negotiate_cache_permission_and_honor_the_opt_out() {
    #[allow(dead_code)] // Only the field list is needed here.
    #[path = "../../engine/crates/model/src/sampling_policy.rs"]
    mod sampling_policy;
    use std::process::Command;
    // Retention and affinity are independent advertised capabilities. The live
    // default engine has retention=false and affinity=true; a request cannot
    // turn retention on. Each environment setting lives in its own process.
    for mode in ["direct", "agent"] {
        let mut transcript = None;
        for (retention, affinity) in [(false, true), (true, true), (false, false), (true, false)] {
            let mut normalized = None;
            for setting in [None, Some("0"), Some("1")] {
                let workspace = WorkspaceFixture::new();
                let (listener, endpoint) = listen();
                let caps = CAPS.replace("\"prefix_cache\": true", &format!("\"prefix_cache\": {retention}"))
                    .replace("\"prompt_cache_key\": \"scheduling_hint\"", if affinity {
                        "\"prompt_cache_key\": \"scheduling_hint\""
                    } else { "\"prompt_cache_key\": null" });
                let server = thread::spawn(move || {
                    let mut reader = accept(&listener);
                    let mut generations = Vec::new();
                    let mut admitted = None;
                    while !reader.fill_buf().unwrap().is_empty() {
                        let (line, body) = request(&mut reader);
                        match line.as_str() {
                            "GET /v1/capabilities HTTP/1.1" => respond(&mut reader, 200, "application/json", &caps),
                            "POST /v1/admission HTTP/1.1" => {
                                let view = json::view(&body).unwrap();
                                admitted = Some(view.get("request").unwrap().raw().to_owned());
                                let output = view.get("output_tokens").unwrap().raw().parse().unwrap();
                                respond(&mut reader, 200, "application/json", &admission_body(117, output, true));
                            }
                            "POST /v1/chat/completions HTTP/1.1" => {
                                assert_eq!(admitted.as_deref(), Some(body.as_str()), "cache decoration must precede exact admission");
                                generations.push(body);
                                respond(&mut reader, 200, "text/event-stream", &format!("{ANSWER}{}{DONE}", usage(117, 0)));
                            }
                            _ => panic!("unexpected cache-fixture request: {line}"),
                        }
                    }
                    generations
                });
                let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
                command.args([mode, "--state-in-project", "--endpoint", &endpoint, "--model", "mock", "--workspace"])
                    .arg(&workspace.0).args(["--plain", "--verbose", "--prompt", "hello", "--max-output-tokens", "17"])
                    .env_remove("VADE_API_KEY_FILE").env_remove("VADE_HARNESS_PREFIX_CACHE");
                if let Some(value) = setting { command.env("VADE_HARNESS_PREFIX_CACHE", value); }
                let output = cli::Cli::start(&mut command).finish();
                assert!(output.status.success(), "{mode} {setting:?}: {}", String::from_utf8_lossy(&output.stderr));
                assert!(String::from_utf8_lossy(&output.stdout).contains("answer"));
                assert!(String::from_utf8_lossy(&output.stderr).contains("cached tokens 0"));
                let bodies = server.join().unwrap();
                assert_eq!(bodies.len(), 1, "cache preference must not replay generation");
                let body = json::view(&bodies[0]).unwrap();
                assert_eq!(body.get("cache").map(|value| value.raw()),
                    (retention || affinity).then_some(if setting == Some("0") { "false" } else { "true" }));
                assert_eq!(body.get("prompt_cache_key").is_some(), affinity);
                let messages = body.get("messages").unwrap().raw().to_owned();
                if let Some(previous) = &transcript { assert_eq!(&messages, previous, "cache preference changed the transcript"); }
                else { transcript = Some(messages); }
                // Fresh sessions without sampling flags carry no sampling field,
                // and the cache setting changes nothing but `cache` and the
                // per-session `prompt_cache_key` (cache-defaults.md).
                assert!(!body.fields().unwrap().any(|(name, _)| sampling_policy::is_field(&name)),
                    "{mode}: cache negotiation must not add sampling fields: {}", bodies[0]);
                let json::Value::Object(fields) = json::parse(&bodies[0]).unwrap() else { panic!("object body") };
                let rest = json::stringify(&json::Value::Object(fields.into_iter()
                    .filter(|(name, _)| name != "cache" && name != "prompt_cache_key" && name != "cancel").collect())).unwrap();
                if let Some(previous) = &normalized {
                    assert_eq!(&rest, previous, "{mode} {retention}/{affinity} {setting:?}: cache setting changed another field");
                } else { normalized = Some(rest); }
            }
        }
    }
}

#[test]
fn cli_unspecified_sampling_stays_unspecified_across_retries_and_resume() {
    #[path = "../../engine/crates/model/src/sampling_policy.rs"]
    mod policy;
    use std::process::Command;
    for mode in ["direct", "agent"] {
        for (flags, seed, temperature, resumed_flags) in [
            (&[][..], None, None, &[][..]),
            (&["--seed", "713"][..], Some(713u64), None, &[][..]),
            (&["--temperature", "0.5"][..], None, Some("0.5"), &["--temperature", "0.5"][..]),
            (&["--greedy"][..], None, Some("0"), &["--greedy"][..]),
            (&["--temperature", "0.5", "--seed", "713"][..], Some(713), Some("0.5"), &["--temperature", "0.5"][..]),
        ] {
            let workspace = WorkspaceFixture::new();
            let (listener, endpoint) = listen();
            let server = thread::spawn(move || {
                let mut captured = Vec::new();
                for connection in 0..if mode == "agent" { 2 } else { 1 } {
                    let mut reader = accept(&listener);
                    let mut admitted = None;
                    let mut turn = 0;
                    while !reader.fill_buf().unwrap().is_empty() {
                        let (line, body) = request(&mut reader);
                        match line.as_str() {
                            "GET /v1/capabilities HTTP/1.1" => respond(&mut reader, 200, "application/json", CAPS),
                            "POST /v1/admission HTTP/1.1" => {
                                let view = json::view(&body).unwrap();
                                admitted = Some(view.get("request").unwrap().raw().to_owned());
                                let output = view.get("output_tokens").unwrap().raw().parse().unwrap();
                                respond(&mut reader, 200, "application/json", &admission_body(117, output, true));
                            }
                            "POST /v1/chat/completions HTTP/1.1" => {
                                assert_eq!(admitted.as_deref(), Some(body.as_str()), "sampling must match exact admission");
                                let view = json::view(&body).unwrap();
                                let expected_seed = seed;
                                assert_eq!(view.get("seed").map(|value| value.raw().parse::<u64>().unwrap()), expected_seed);
                                assert_eq!(view.get("temperature").map(|value| value.raw()), temperature);
                                let explicit = view.fields().unwrap().any(|(name, _)| policy::is_field(&name));
                                assert_eq!(explicit, seed.is_some() || temperature.is_some(), "bookkeeping must not opt into named-field policy");
                                // This fixture selects the same field-presence branch as
                                // Engine. It does not simulate token selection or claim
                                // live arithmetic parity.
                                let label = if explicit { "explicit sampling" } else {
                                    let defaults = json::view(policy::GREEDY_DEFAULTS_JSON).unwrap();
                                    assert_eq!(defaults.get("temperature").unwrap().raw(), "0");
                                    "greedy defaults"
                                };
                                captured.push(body);
                                let event = format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{label}\"}},\"finish_reason\":\"stop\"}}]}}\n\n");
                                respond(&mut reader, 200, "text/event-stream", &format!("{event}{}{DONE}", usage(117, 0)));
                                turn += 1;
                            }
                            _ => panic!("unexpected sampling fixture request: {line}"),
                        }
                    }
                    assert_eq!(turn, if connection == 0 { 3 } else { 1 });
                }
                captured
            });
            let make = |resume: bool| {
                let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
                command.args([mode, "--state-in-project", "--endpoint", &endpoint, "--model", "mock", "--workspace"])
                    .arg(&workspace.0).args(["--plain", "--verbose", "--max-output-tokens", "17", "--no-auto-continue"])
                    .args(if resume { &["--resume", "sampler"][..] } else { &["--session", "sampler"][..] })
                    .args(if resume { resumed_flags } else { flags })
                    .env_remove("VADE_API_KEY_FILE");
                command
            };
            let mut child = cli::Cli::start(&mut make(false));
            for input in ["first\n", "/retry\n", "followup\n"] {
                child.send(input);
                child.until(true, "session reported cached tokens:");
            }
            child.send("/status\n");
            let output = child.finish();
            assert!(output.status.success(), "{mode} {flags:?}: {}", String::from_utf8_lossy(&output.stderr));
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains(if seed.is_some() || temperature.is_some() { "explicit sampling" } else { "greedy defaults" }));
            if seed.is_none() && temperature.is_none() {
                assert!(String::from_utf8_lossy(&output.stderr).contains("Sampling: upstream defaults (no sampling fields)"));
                assert!(stdout.contains("a greedy answer may repeat"));
            }
            if let Some(seed) = seed {
                assert!(stdout.contains(&format!("unchanged seed {seed}; use a different --seed")));
            }
            if mode == "agent" {
                let root = vade_harness::tools::Workspace::new(&workspace.0).unwrap();
                let (journal, _, _) = vade_harness::journal::Journal::resume(&root, "sampler", Limits::default()).unwrap();
                assert_eq!(journal.generation_seed(), seed);
                drop(journal);drop(root);
                let mut command = make(true);command.args(["--prompt", "resumed followup"]);
                let output = cli::Cli::start(&mut command).finish();
                assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            }
            let captured = server.join().unwrap();
            assert_eq!(captured.len(), if mode == "agent" { 4 } else { 3 });
            let first = json::parse(&captured[0]).unwrap();let retry = json::parse(&captured[1]).unwrap();
            assert_eq!(first.get("messages"), retry.get("messages"), "retry must preserve the retained conversation");
            if mode == "agent" { assert_eq!(retry.get("tool_choice").and_then(json::Value::as_str), Some("none")); }
        }
    }
}
