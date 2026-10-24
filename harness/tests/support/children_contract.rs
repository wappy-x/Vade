//! Actual CLI/TUI parent and children against the existing first-party contract
//! mock helpers and capability fixture. Only ephemeral loopback sockets.
use super::{CAPS, accept, admission_body, cli, fixture_read, listen, request, respond};
use std::io::{BufRead, BufReader, Read};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};
use vade_harness::json::{self, Value};
#[path = "raw_ui.rs"]
mod raw_ui;

const PROMPT: &str = "Investigate three tasks; private parent input stays here.";
const FINAL: &str = "Integrated alpha finding and beta finding; third child cancelled.";
const ALL_FINAL: &str = "All three children cancelled; no findings integrated.";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scenario {
    ModelCancel,
    UiCancel,
    UiCancelAll,
    ParentCancel,
}

struct Fixture {
    endpoint: String,
    workspace: PathBuf,
    ready: mpsc::Receiver<Vec<String>>,
    gate: mpsc::Sender<()>,
    server: Option<thread::JoinHandle<()>>,
}

impl Fixture {
    fn start(scenario: Scenario) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let workspace = std::env::temp_dir().join(format!(
            "vade-children-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&workspace).unwrap();
        std::fs::write(workspace.join("note.txt"), "bounded child fixture").unwrap();
        let (listener, endpoint) = listen();
        let (ready_tx, ready) = mpsc::channel();
        let (gate, gate_rx) = mpsc::channel();
        let server = thread::spawn(move || {
            let mut parent = accept(&listener);
            let first_body = next_request(&mut parent);
            let first = json::parse(&first_body).unwrap();
            assert_eq!(
                messages(&first)
                    .last()
                    .unwrap()
                    .get("content")
                    .and_then(Value::as_str),
                Some(PROMPT)
            );
            let tools = first.get("tools").unwrap().as_array().unwrap();
            assert_eq!(tools.len(), 15);
            for name in [
                "workspace_read",
                "workspace_list",
                "apply_patch",
                "workspace_shell",
                "tool_output_read",
                "skill_read",
                "workspace_manifest",
                "workspace_symbols",
                "workspace_search",
                "workspace_read_lines",
                "workspace_edit_lines",
                "child_spawn",
                "child_list",
                "child_wait",
                "child_cancel",
            ] {
                assert!(tools.iter().any(|tool| {
                    tool.get("function")
                        .unwrap()
                        .get("name")
                        .and_then(Value::as_str)
                        == Some(name)
                }));
            }
            stream_reply(
                &mut parent,
                calls(vec![
                    call(
                        "spawn-1",
                        "child_spawn",
                        object(vec![("task", string("alpha"))]),
                    ),
                    call(
                        "spawn-2",
                        "child_spawn",
                        object(vec![("task", string("beta"))]),
                    ),
                    call(
                        "spawn-3",
                        "child_spawn",
                        object(vec![("task", string("cancel-me"))]),
                    ),
                ]),
                Some(0),
            );

            let (child_ready, child_events) = mpsc::channel();
            let mut children = Vec::new();
            for _ in 0..3 {
                tool_guard(&mut parent);
                let mut child = accept(&listener);
                let expected = first.clone();
                let ready = child_ready.clone();
                children.push(thread::spawn(move || {
                    let body = next_request(&mut child);
                    let value = json::parse(&body).unwrap();
                    assert_eq!(value.get("stream"), Some(&Value::Bool(false)));
                    assert_eq!(value.get("model"), expected.get("model"));
                    assert_eq!(value.get("tools"), expected.get("tools"));
                    assert_eq!(&messages(&value)[..2], &messages(&expected)[..2]);
                    assert_eq!(messages(&value).len(), 3);
                    assert!(!body.contains(PROMPT));
                    assert_ne!(
                        value.get("prompt_cache_key"),
                        expected.get("prompt_cache_key")
                    );
                    let key = value
                        .get("prompt_cache_key")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_owned();
                    let envelope = json::parse(
                        messages(&value)[2]
                            .get("content")
                            .unwrap()
                            .as_str()
                            .unwrap(),
                    )
                    .unwrap();
                    assert_eq!(envelope.get("child_spawning"), Some(&Value::Bool(false)));
                    assert_eq!(
                        envelope
                            .get("write_paths")
                            .unwrap()
                            .as_array()
                            .unwrap()
                            .len(),
                        0
                    );
                    assert_eq!(
                        envelope
                            .get("allowed_tools")
                            .unwrap()
                            .as_array()
                            .unwrap()
                            .len(),
                        8
                    );
                    assert!(
                        !envelope
                            .get("allowed_tools")
                            .unwrap()
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|name| matches!(
                                name.as_str(),
                                Some("apply_patch" | "workspace_shell")
                            ))
                    );
                    let task = envelope.get("task").unwrap().as_str().unwrap();
                    if task == "cancel-me"
                        || matches!(scenario, Scenario::UiCancelAll | Scenario::ParentCancel)
                    {
                        ready.send(key).unwrap();
                        assert_eq!(
                            child.read(&mut [0]).unwrap(),
                            0,
                            "cancel closes pending child HTTP"
                        );
                        return;
                    }
                    if task == "alpha" {
                        reply(
                            &mut child,
                            calls(vec![call(
                                "read-note",
                                "workspace_read",
                                object(vec![("path", string("note.txt"))]),
                            )]),
                            Some(20),
                        );
                        let followup = next_request(&mut child);
                        assert!(followup.starts_with(&body[..body.len() - 2]));
                        let parsed = json::parse(&followup).unwrap();
                        let result = messages(&parsed).last().unwrap();
                        assert_eq!(
                            result.get("tool_call_id").and_then(Value::as_str),
                            Some("read-note")
                        );
                        assert!(
                            result
                                .get("content")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .contains("bounded child fixture")
                        );
                        reply(&mut child, answer("alpha finding"), Some(30));
                    } else {
                        assert_eq!(task, "beta");
                        reply(&mut child, answer("beta finding"), None);
                    }
                    assert_eq!(
                        child.read(&mut [0]).unwrap(),
                        0,
                        "finished child drops its own connection pool"
                    );
                    ready.send(key).unwrap();
                }));
            }
            let mut keys = std::collections::HashSet::new();
            for _ in 0..3 {
                assert!(keys.insert(child_events.recv_timeout(Duration::from_secs(4)).unwrap()));
            }
            let second_body = next_request(&mut parent);
            let first_prefix = super::without_cancel(&first_body);
            assert!(super::without_cancel(&second_body).starts_with(&first_prefix[..first_prefix.len() - 2]));
            let second = json::parse(&second_body).unwrap();
            let ids: Vec<String> = (1..=3)
                .map(|n| {
                    tool_result(&second, &format!("spawn-{n}"))
                        .get("task_id")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_owned()
                })
                .collect();
            ready_tx.send(ids.clone()).unwrap();
            if scenario == Scenario::ParentCancel {
                assert_eq!(
                    parent.read(&mut [0]).unwrap(),
                    0,
                    "parent HTTP also cancelled"
                );
            } else {
                if scenario != Scenario::ModelCancel {
                    gate_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                }
                let mut controls = Vec::new();
                if scenario == Scenario::ModelCancel {
                    controls.push(call(
                        "cancel-3",
                        "child_cancel",
                        object(vec![("task_id", string(&ids[2]))]),
                    ));
                }
                if scenario == Scenario::UiCancelAll {
                    controls.push(call(
                        "closed-spawn",
                        "child_spawn",
                        object(vec![("task", string("must not run"))]),
                    ));
                }
                for (n, id) in ids.iter().enumerate() {
                    controls.push(call(
                        &format!("wait-{n}"),
                        "child_wait",
                        object(vec![
                            ("task_id", string(id)),
                            ("timeout_ms", Value::Number("1000".into())),
                        ]),
                    ));
                }
                controls.push(call("list", "child_list", object(vec![])));
                stream_reply(&mut parent, calls(controls), Some(50));
                let third_body = next_request(&mut parent);
                let third_plain = super::without_cancel(&third_body);
                let second_plain = super::without_cancel(&second_body);
                assert!(third_plain.starts_with(&second_plain[..second_plain.len() - 2]));
                let third = json::parse(&third_body).unwrap();
                if scenario == Scenario::ModelCancel {
                    let receipt = tool_result(&third, "cancel-3");
                    assert_eq!(
                        receipt.get("cancellation_requested"),
                        Some(&Value::Bool(true))
                    );
                    assert_eq!(
                        receipt.get("remote_stop_confirmed"),
                        Some(&Value::Bool(false))
                    );
                }
                if scenario == Scenario::UiCancelAll {
                    assert_eq!(
                        tool_result(&third, "closed-spawn")
                            .get("error")
                            .and_then(Value::as_str),
                        Some("Closed")
                    );
                }
                for n in 0..3 {
                    let result = tool_result(&third, &format!("wait-{n}"));
                    let cancelled = n == 2 || scenario == Scenario::UiCancelAll;
                    assert_eq!(
                        result.get("outcome").and_then(Value::as_str),
                        Some(if cancelled { "Cancelled" } else { "Completed" })
                    );
                    assert_eq!(
                        result.get("integration_owner").and_then(Value::as_str),
                        Some("parent")
                    );
                    assert_eq!(
                        result.get("remote_stop_confirmed"),
                        Some(&Value::Bool(false))
                    );
                    assert_eq!(
                        result.get("summary").and_then(Value::as_str),
                        Some(if cancelled {
                            ""
                        } else if n == 0 {
                            "alpha finding"
                        } else {
                            "beta finding"
                        })
                    );
                }
                assert!(
                    tool_result(&third, "list")
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|s| s
                            .get("state")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .starts_with("collected:"))
                );
                stream_reply(
                    &mut parent,
                    answer(if scenario == Scenario::UiCancelAll {
                        ALL_FINAL
                    } else {
                        FINAL
                    }),
                    Some(60),
                );
            }
            for child in children {
                child.join().unwrap();
            }
        });
        Self {
            endpoint,
            workspace,
            ready,
            gate,
            server: Some(server),
        }
    }
    fn command(&self, enabled: bool) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        command
            .args([
                "agent",
                "--state-in-project",
                "--verbose",
                "--endpoint",
                &self.endpoint,
                "--model",
                "mock",
                "--workspace",
            ])
            .arg(&self.workspace)
            .arg("--plain");
        if enabled {
            command.arg("--children");
        }
        command
    }
    fn ready(&self) -> Vec<String> {
        self.ready
            .recv_timeout(Duration::from_secs(5))
            .expect("parent and three children did not reach mock barrier")
    }
    fn finish(&mut self) {
        self.server.take().unwrap().join().unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Every blocking fixture operation is bounded. On a test panic do not
        // mask that failure with a second join panic; CLI drivers reap processes.
        if let Some(server) = self.server.take() {
            let _ = server.join();
        }
        let _ = std::fs::remove_dir_all(&self.workspace);
    }
}

fn counted_after_discovery(peer: &mut BufReader<TcpStream>, first: (String, String)) -> String {
    // Same v1 fixture as the transport tests, with a 1024-token output limit to
    // fit the native child default. Counts here are scripted, never tokenized.
    assert_eq!(first.0, "GET /v1/capabilities HTTP/1.1");
    respond(
        peer,
        200,
        "application/json",
        &CAPS.replace("\"output_limit\": 32", "\"output_limit\": 1024"),
    );
    let (line, envelope) = request(peer);
    assert_eq!(line, "POST /v1/admission HTTP/1.1");
    let envelope = json::view(&envelope).unwrap();
    let output = envelope
        .get("output_tokens")
        .unwrap()
        .raw()
        .parse::<u64>()
        .unwrap();
    assert_eq!(output, 1024);
    let body = envelope.get("request").unwrap().raw().to_owned();
    respond(
        peer,
        200,
        "application/json",
        &admission_body(100, output, true).replace("\"output_limit\":32", "\"output_limit\":1024"),
    );
    body
}
fn tool_guard(peer: &mut BufReader<TcpStream>) {
    let first = request(peer);
    let body = counted_after_discovery(peer, first);
    assert!(
        body.contains("tool_calls"),
        "closed tool staging must be counted before execution"
    );
}
fn next_request(peer: &mut BufReader<TcpStream>) -> String {
    next_request_or_closed(peer).expect("mock peer closed before generation")
}
fn next_request_or_closed(peer: &mut BufReader<TcpStream>) -> Option<String> {
    let read = |peer: &mut BufReader<TcpStream>| {
        if fixture_read(|| peer.fill_buf().map(|bytes| bytes.is_empty())) {
            None
        } else {
            Some(request(peer))
        }
    };
    let mut first = read(peer)?;
    loop {
        let body = counted_after_discovery(peer, first);
        first = read(peer)?;
        if first.0 == "POST /v1/chat/completions HTTP/1.1" {
            assert_eq!(first.1, body, "dispatch must equal exact admitted bytes");
            return Some(body);
        }
        // Between generations, exact guards count the closed proposed calls
        // and reserved receipts, then count each preceding actual tool result.
        assert!(body.contains("tool_calls"));
    }
}
fn string(s: &str) -> Value {
    Value::String(s.into())
}
fn object(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn answer(text: &str) -> Value {
    object(vec![
        ("role", string("assistant")),
        ("content", string(text)),
    ])
}
fn call(id: &str, name: &str, args: Value) -> Value {
    object(vec![
        ("id", string(id)),
        ("type", string("function")),
        (
            "function",
            object(vec![
                ("name", string(name)),
                ("arguments", string(&json::stringify(&args).unwrap())),
            ]),
        ),
    ])
}
fn calls(calls: Vec<Value>) -> Value {
    object(vec![
        ("role", string("assistant")),
        ("content", Value::Null),
        ("tool_calls", Value::Array(calls)),
    ])
}
fn messages(value: &Value) -> &[Value] {
    value.get("messages").unwrap().as_array().unwrap()
}
fn tool_result(request: &Value, id: &str) -> Value {
    let result = messages(request)
        .iter()
        .find(|m| m.get("tool_call_id").and_then(Value::as_str) == Some(id))
        .expect("missing original tool-call ID");
    json::parse(result.get("content").unwrap().as_str().unwrap()).unwrap()
}
fn reply(peer: &mut BufReader<TcpStream>, message: Value, cached: Option<u64>) {
    let reason = if message.get("tool_calls").is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    let usage = cached.map_or(String::new(), |c| format!(",\"usage\":{{\"prompt_tokens\":100,\"completion_tokens\":3,\"prompt_tokens_details\":{{\"cached_tokens\":{c}}}}}"));
    let body = format!(
        "{{\"choices\":[{{\"message\":{},\"finish_reason\":\"{reason}\"}}]{usage}}}",
        json::stringify(&message).unwrap()
    );
    respond(peer, 200, "application/json", &body);
}

fn stream_reply(peer: &mut BufReader<TcpStream>, mut message: Value, cached: Option<u64>) {
    let mut reason = "stop";
    if let Value::Object(fields) = &mut message {
        if let Some((_, Value::Array(calls))) =
            fields.iter_mut().find(|(key, _)| key == "tool_calls")
        {
            reason = "tool_calls";
            for (index, call) in calls.iter_mut().enumerate() {
                let Value::Object(fields) = call else {
                    unreachable!()
                };
                fields.push(("index".into(), Value::Number(index.to_string())));
            }
        }
    }
    let usage = cached.map_or(String::new(), |cached| super::usage(100, cached));
    let body = format!(
        "data: {{\"choices\":[{{\"delta\":{},\"finish_reason\":\"{reason}\"}}]}}\n\n{usage}data: [DONE]\n\n",
        json::stringify(&message).unwrap()
    );
    respond(peer, 200, "text/event-stream", &body);
}

#[test]
fn cli_parent_spawns_three_cancels_one_and_integrates_collected_results() {
    let mut fixture = Fixture::start(Scenario::ModelCancel);
    let mut child = cli::Cli::start(&mut fixture.command(true));
    child.send(&format!("{PROMPT}\n"));
    fixture.ready();
    child.until(true, "Cached input still occupies context; reservations");
    child.send("/context\n/children\n/status\n");
    let output = child.finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains(FINAL));
    assert!(text.contains("cached tokens 30"));
    assert!(text.contains("reported cached tokens: 50 across 2/2"));
    assert!(text.contains("reported cached tokens: unknown across 0/1"));
    assert!(text.contains("collected: cancelled"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("3 retained results"));
    fixture.finish();
}

#[test]
fn cli_children_command_enables_tools_and_cancels_without_waiting_for_parent_http() {
    let mut fixture = Fixture::start(Scenario::UiCancel);
    let mut child = cli::Cli::start(&mut fixture.command(false));
    child.send("/children\n/children on\n");
    child.until(false, "Children enabled");
    child.send(&format!("{PROMPT}\n"));
    let ids = fixture.ready();
    child.send("/children off\n/context\n/children\n");
    child.until(false, "/children cancel ID|all · local cancellation only");
    let start = Instant::now();
    child.send(&format!("/children cancel {}\n", ids[2]));
    child.until(false, "Waiting for local cleanup; remote stop unconfirmed.");
    assert!(start.elapsed() < Duration::from_secs(1));
    fixture.gate.send(()).unwrap();
    child.until(true, "Cached input still occupies context; reservations");
    child.send("/children off\n/new\n/children off\n/children\n");
    let output = child.finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Wait for request cleanup before changing child tools."));
    assert!(text.contains("Child tools are frozen for this conversation."));
    assert!(text.contains("Children disabled for this fresh conversation."));
    assert!(text.contains(FINAL));
    fixture.finish();
}

#[test]
fn cli_cancel_all_closes_child_admission_and_keeps_parent_live() {
    let mut fixture = Fixture::start(Scenario::UiCancelAll);
    let mut child = cli::Cli::start(&mut fixture.command(true));
    child.send(&format!("{PROMPT}\n"));
    fixture.ready();
    child.send("/children cancel all\n");
    child.until(false, "Child cancellation requested for 3 active task(s)");
    fixture.gate.send(()).unwrap();
    let output = child.finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains(ALL_FINAL));
    fixture.finish();
}

#[test]
fn cli_parent_cancel_joins_all_children_before_new_context() {
    let mut fixture = Fixture::start(Scenario::ParentCancel);
    let mut child = cli::Cli::start(&mut fixture.command(true));
    child.send(&format!("{PROMPT}\n"));
    fixture.ready();
    child.send("/cancel\n");
    child.until(true, "Canceled · ");
    child.send("/children\n/new\n/children\n");
    let output = child.finish();
    assert!(!output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches("collected: cancelled").count(), 3);
    assert!(text.contains("Children (0)"));
    fixture.finish();
}

#[test]
fn raw_tui_live_children_controls_context_and_resize_while_parent_is_stalled() {
    let mut fixture = Fixture::start(Scenario::UiCancel);
    let mut ui = raw_ui::Ui::start(
        "agent",
        &fixture.endpoint,
        fixture.workspace.to_str().unwrap(),
        true,
    );
    ui.send(format!("{PROMPT}\r").as_bytes());
    let ids = fixture.ready();
    let frame = ui.until(&format!("{}  running", ids[2]), Duration::from_secs(2));
    assert!(frame.contains("Children (3)"));
    ui.resize(9, 42);
    ui.send(b"/children\r");
    ui.until("parent owns result integration", Duration::from_secs(1));
    ui.resize(24, 110);
    ui.send(b"/context\r");
    let context = ui.until(
        "Cached input still occupies context; reservations",
        Duration::from_secs(1),
    );
    assert!(context.contains("cached tokens 30"));
    assert!(context.contains("reported cached tokens: 50 across 2/2"));
    ui.send(format!("/children cancel {}\r", ids[2]).as_bytes());
    ui.until(
        "Waiting for local cleanup; remote stop unconfirmed.",
        Duration::from_secs(1),
    );
    ui.until(
        &format!("{}  cancelled locally", ids[2]),
        Duration::from_secs(2),
    );
    fixture.gate.send(()).unwrap();
    ui.until(FINAL, Duration::from_secs(3));
    ui.send(b"/quit\r");
    fixture.finish();
}

#[test]
fn cli_without_children_flag_rejects_model_spawn_and_never_starts_a_child() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut parent = accept(&listener);
        let body = json::parse(&next_request(&mut parent)).unwrap();
        assert_eq!(
            body.get("tools"),
            Some(&vade_harness::coding_tools::definitions())
        );
        reply(
            &mut parent,
            calls(vec![call(
                "unauthorized",
                "child_spawn",
                object(vec![("task", string("not enabled"))]),
            )]),
            Some(0),
        );
        let followup = json::parse(&next_request(&mut parent)).unwrap();
        assert_eq!(
            tool_result(&followup, "unauthorized")
                .get("error")
                .and_then(Value::as_str),
            Some("unknown_tool")
        );
        stream_reply(&mut parent, answer("Children are disabled."), Some(0));
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    });
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
    command.args([
        "agent",
        "--state-in-project",
        "--verbose",
        "--endpoint",
        &endpoint,
        "--model",
        "mock",
        "--workspace",
        ".",
        "--plain",
        "--prompt",
        "No child authorization",
    ]);
    let output = cli::Cli::start(&mut command).finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Children are disabled.\n"
    );
    server.join().unwrap();
}

#[test]
fn cli_child_missing_exact_admission_fails_without_generation_or_fabricated_cache_counts() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut parent = accept(&listener);
        next_request(&mut parent);
        reply(
            &mut parent,
            calls(vec![call(
                "spawn",
                "child_spawn",
                object(vec![("task", string("unsupported child"))]),
            )]),
            Some(0),
        );
        tool_guard(&mut parent);
        let mut child = accept(&listener);
        assert_eq!(request(&mut child).0, "GET /v1/capabilities HTTP/1.1");
        respond(&mut child, 404, "application/json", "{}");
        assert_eq!(
            child.read(&mut [0]).unwrap(),
            0,
            "no child generation without exact admission"
        );
        let body = json::parse(&next_request(&mut parent)).unwrap();
        let result = tool_result(&body, "spawn");
        let id = result.get("task_id").unwrap().as_str().unwrap();
        reply(
            &mut parent,
            calls(vec![call(
                "wait",
                "child_wait",
                object(vec![("task_id", string(id))]),
            )]),
            Some(0),
        );
        let body = json::parse(&next_request(&mut parent)).unwrap();
        let result = tool_result(&body, "wait");
        assert_eq!(
            result.get("outcome").and_then(Value::as_str),
            Some("Runtime(ModelError(Unsupported))")
        );
        assert_eq!(
            result.get("tokens_reserved"),
            Some(&Value::Number("0".into()))
        );
        assert_eq!(result.get("summary").and_then(Value::as_str), Some(""));
        reply(
            &mut parent,
            answer("Child unavailable; no findings integrated."),
            Some(0),
        );
    });
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
    command.args([
        "agent",
        "--state-in-project",
        "--verbose",
        "--endpoint",
        &endpoint,
        "--model",
        "mock",
        "--workspace",
        ".",
        "--plain",
        "--children",
        "--prompt",
        "Handle missing child admission",
    ]);
    let output = cli::Cli::start(&mut command).finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(text.contains("reported cached tokens: unknown across 0/0"));
    assert!(text.contains("collected: failed"));
    server.join().unwrap();
}

#[path = "children_native.rs"]
mod native_tools;
