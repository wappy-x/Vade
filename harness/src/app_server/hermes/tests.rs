use super::*;
use crate::json;
use std::{
    fs,
    path::Path,
    sync::{OnceLock, atomic::AtomicUsize},
    thread,
    time::Duration,
};

fn mock() -> &'static Path {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY
        .get_or_init(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR"));
            let directory = root
                .join("target")
                .join(format!("hermes-mock-{}", std::process::id()));
            fs::create_dir_all(&directory).unwrap();
            let source = directory.join("mock.rs");
            fs::write(
                &source,
                format!(
                    "#[path = {:?}] mod json;\n{}",
                    root.join("src/json.rs"),
                    include_str!("../../../tests/fixtures/hermes_child.rs")
                ),
            )
            .unwrap();
            let binary = directory.join("mock-child");
            let compiler = std::env::var_os("VADE_TEST_RUSTC")
                .or_else(|| std::env::var_os("RUSTC"))
                .unwrap_or_else(|| "rustc".into());
            assert!(
                Command::new(compiler)
                    .args([
                        "--edition=2024",
                        "--crate-name=vade_hermes_fixture",
                        "-Adead_code"
                    ])
                    .arg(source)
                    .arg("-o")
                    .arg(&binary)
                    .status()
                    .unwrap()
                    .success()
            );
            binary
        })
        .as_path()
}
fn options(mode: &str, session: Option<&str>) -> Options {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let workspace = mock().parent().unwrap().join(format!(
        "workspace-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&workspace).unwrap();
    fs::write(workspace.join("fixture-mode"), mode).unwrap();
    Options {
        executable: mock().into(),
        workspace,
        session_id: session.map(str::to_owned),
    }
}
fn client() -> Client {
    Client::connect(options("", None)).unwrap()
}
fn assert_closed(client: &Client) {
    assert!(client.wire.stopped());
    assert_eq!(
        client.cleanup(),
        Some(Cleanup {
            child_reaped: true,
            io_workers_joined: true
        })
    );
    assert!(client.completed_text().is_none());
}
#[test]
fn new_load_resume_exact_prompt_and_opaque_data() {
    let exact = "  exact\n\t🦀 \"text\"  ";
    for (mode, session, expected) in [
        ("", None, "session/new"),
        ("", Some("saved-session"), "session/load"),
        ("resume", Some("saved-session"), "session/resume"),
    ] {
        let mut client = Client::connect(options(mode, session)).unwrap();
        assert_eq!(client.session_id(), session.unwrap_or("mock-session"));
        assert!(client.capabilities().load_session);
        assert_eq!(client.capabilities().resume_session, mode == "resume");
        assert!(
            client.capabilities().fork_session
                && client.capabilities().close_session
                && client.capabilities().list_sessions
        );
        let provider = json::stringify(client.provider_data()).unwrap();
        assert!(provider.contains("opaque setup") && provider.contains(expected));
        let mut displayed = String::new();
        assert_eq!(
            client
                .turn(exact, &AtomicBool::new(false), &mut |event| {
                    if let Event::AgentMessageDelta(text) = event {
                        displayed.push_str(&text);
                    }
                })
                .unwrap(),
            TurnOutcome::Completed
        );
        assert_eq!(displayed, exact); // loaded history was not shown or forwarded
        assert_eq!(client.completed_text(), Some(exact));
        assert_eq!(client.updates().len(), 1);
        assert!(
            json::stringify(client.last_response().unwrap())
                .unwrap()
                .contains("retained externally")
        );
        assert_eq!(
            client
                .turn("next exact prompt", &AtomicBool::new(false), &mut |_| {})
                .unwrap(),
            TurnOutcome::Completed
        );
        assert_eq!(client.completed_text(), Some("next exact prompt"));
    }
}
#[test]
fn interleaved_messages_and_tools_commit_only_at_terminal_response() {
    let mut client = client();
    let mut provisional = String::new();
    let mut tool_seen = false;
    assert_eq!(
        client
            .turn(
                "interleaved",
                &AtomicBool::new(false),
                &mut |event| match event {
                    Event::AgentMessageDelta(text) => provisional.push_str(&text),
                    Event::Phase(Phase::ToolRunning) => tool_seen = true,
                    _ => (),
                }
            )
            .unwrap(),
        TurnOutcome::Completed
    );
    assert!(tool_seen);
    assert_eq!(provisional, "first secondmessage 🦀");
    assert_eq!(client.completed_text(), Some("first message 🦀\nsecond"));
    assert_eq!(client.updates().len(), 6);
    assert!(
        client
            .updates()
            .iter()
            .any(|v| json::stringify(v).unwrap().contains("provider field"))
    );
}
#[test]
fn all_permissions_and_known_or_unknown_host_callbacks_are_denied() {
    let mut client = client();
    let mut denials = 0;
    assert_eq!(
        client
            .turn(
                "callbacks",
                &AtomicBool::new(false),
                &mut |event| if matches!(event, Event::Notice(_)) {
                    denials += 1;
                }
            )
            .unwrap(),
        TurnOutcome::Completed
    );
    assert_eq!(denials, 9);
    assert_eq!(client.completed_text(), Some("all callbacks denied"));
}
#[test]
fn cancel_during_tool_or_permission_waits_for_actual_prompt_acknowledgment() {
    for prompt in ["cancel-tool", "cancel-permission", "complete-race"] {
        let mut client = client();
        let cancel = AtomicBool::new(false);
        let began = Instant::now();
        let result = client
            .turn(prompt, &cancel, &mut |event| {
                if matches!(event, Event::Phase(Phase::ToolRunning)) {
                    cancel.store(true, Ordering::Relaxed);
                }
            })
            .unwrap();
        assert!(began.elapsed() >= Duration::from_millis(120));
        assert!(began.elapsed() < Duration::from_secs(2));
        if prompt == "complete-race" {
            assert_eq!(result, TurnOutcome::Completed);
            assert_eq!(
                client.completed_text(),
                Some("completed before cancellation")
            );
        } else {
            assert_eq!(result, TurnOutcome::Interrupted);
            assert!(client.completed_text().is_none());
        }
        cancel.store(false, Ordering::Relaxed);
        assert_eq!(
            client
                .turn("explicit follow-up", &cancel, &mut |_| {})
                .unwrap(),
            TurnOutcome::Completed
        );
    }
}
#[test]
fn unknown_malformed_or_partial_outcomes_close_and_reap() {
    for prompt in [
        "crash",
        "partial",
        "unknown-event",
        "wrong-session",
        "unknown-response",
        "unknown-notification",
        "unknown-stop",
        "rpc-error",
        "unknown-tool",
        "unfinished-tool",
        "reopened-tool",
        "unknown-tool-status",
        "oversized",
    ] {
        let mut client = client();
        assert!(
            client
                .turn(prompt, &AtomicBool::new(false), &mut |_| {})
                .is_err(),
            "{prompt}"
        );
        assert_closed(&client);
        assert!(
            client
                .turn(
                    "must not restart automatically",
                    &AtomicBool::new(false),
                    &mut |_| {}
                )
                .is_err()
        );
    }
}
#[test]
fn explicit_non_success_stops_never_commit_provisional_text() {
    let mut client = client();
    for prompt in ["max_tokens", "max_turn_requests", "refusal"] {
        assert!(matches!(
            client
                .turn(prompt, &AtomicBool::new(false), &mut |_| {})
                .unwrap(),
            TurnOutcome::Failed(_)
        ));
        assert!(client.completed_text().is_none());
    }
    assert_eq!(
        client
            .turn("explicit recovery", &AtomicBool::new(false), &mut |_| {})
            .unwrap(),
        TurnOutcome::Completed
    );
}
#[test]
fn negotiated_session_operations_and_version_must_match() {
    for (mode, session) in [
        ("v2", None),
        ("no-load", Some("saved")),
        ("init-error", None),
        ("wrong-setup-session", Some("saved")),
    ] {
        assert!(Client::connect(options(mode, session)).is_err(), "{mode}");
    }
    let mut client = client();
    for method in [
        "session/fork",
        "session/list",
        "session/close",
        "authenticate",
        "terminal/create",
        "_future/run",
    ] {
        assert_eq!(
            client
                .request(
                    method,
                    object(vec![]),
                    Instant::now() + Duration::from_secs(1),
                    &AtomicBool::new(false)
                )
                .unwrap_err()
                .kind(),
            io::ErrorKind::Unsupported
        );
    }
    assert_eq!(
        client
            .turn("still usable", &AtomicBool::new(false), &mut |_| {})
            .unwrap(),
        TurnOutcome::Completed
    );
}
#[test]
fn hung_setup_or_turn_and_missing_cancel_ack_have_bounded_local_cleanup() {
    let limits = Limits {
        rpc: Duration::from_millis(120),
        ..Limits::default()
    };
    let options = options("init-hang", None);
    let start = Instant::now();
    assert!(Client::connect_with_limits(options, &AtomicBool::new(false), limits).is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
    let mut client = client();
    client.limits.idle = Duration::from_millis(120);
    let start = Instant::now();
    assert_eq!(
        client
            .turn("hang", &AtomicBool::new(false), &mut |_| {})
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut
    );
    assert!(start.elapsed() < Duration::from_secs(2));
    assert_closed(&client);
    let mut client = self::client();
    client.limits.interrupt = Duration::from_millis(120);
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    assert_eq!(
        client
            .turn("ignore-cancel", &cancel, &mut |event| if matches!(
                event,
                Event::Phase(Phase::Generating)
            ) {
                cancel.store(true, Ordering::Relaxed);
            })
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut
    );
    assert!(start.elapsed() < Duration::from_secs(2));
    assert_closed(&client);
}
#[test]
fn setup_cancellation_and_pre_dispatch_cancellation_do_not_claim_remote_stop() {
    let options = options("init-hang", None);
    let cancel = AtomicBool::new(false);
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(80));
            cancel.store(true, Ordering::Relaxed);
        });
        assert!(Client::connect_cancellable(options, &cancel).is_err());
    });
    let mut client = client();
    assert_eq!(
        client
            .turn("not sent", &AtomicBool::new(true), &mut |_| panic!(
                "no events before dispatch"
            ))
            .unwrap(),
        TurnOutcome::NotStarted
    );
    assert_eq!(
        client
            .turn("sent once", &AtomicBool::new(false), &mut |_| {})
            .unwrap(),
        TurnOutcome::Completed
    );
}
#[test]
fn restart_is_an_explicit_new_connection_with_negotiated_load() {
    let options = options("", None);
    let mut first = Client::connect(options.clone()).unwrap();
    assert!(
        first
            .turn("partial", &AtomicBool::new(false), &mut |_| {})
            .is_err()
    );
    assert_closed(&first);
    let session = first.session_id().to_owned();
    let mut second = Client::connect(Options {
        session_id: Some(session),
        ..options
    })
    .unwrap();
    assert_eq!(
        second
            .turn("explicit new input", &AtomicBool::new(false), &mut |_| {})
            .unwrap(),
        TurnOutcome::Completed
    );
    assert_eq!(second.completed_text(), Some("explicit new input"));
}
#[test]
fn projection_and_capability_resource_limits_fail_explicitly() {
    assert!(
        capabilities(&json::parse(r#"{"agentCapabilities":{"loadSession":"yes"}}"#).unwrap())
            .is_err()
    );
    assert!(
        capabilities(
            &json::parse(r#"{"agentCapabilities":{"sessionCapabilities":{"resume":true}}}"#)
                .unwrap()
        )
        .is_err()
    );
    let mut projection = Projection::default();
    for id in 0..=MAX_ITEMS {
        let value = object(vec![(
            "params",
            object(vec![
                ("sessionId", string("session")),
                (
                    "update",
                    object(vec![
                        ("sessionUpdate", string("tool_call")),
                        ("toolCallId", string(&format!("tool-{id}"))),
                        ("status", string("completed")),
                    ]),
                ),
            ]),
        )]);
        assert_eq!(
            projection.update(&value, "session", &mut |_| {}).is_ok(),
            id < MAX_ITEMS
        );
    }
    let mut projection = Projection {
        text_bytes: MAX_DELTA_BYTES,
        ..Projection::default()
    };
    let value = json::parse(r#"{"params":{"sessionId":"s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"x"}}}}"#).unwrap();
    assert!(projection.update(&value, "s", &mut |_| {}).is_err());
}
