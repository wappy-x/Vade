use super::*;
use std::fs;
use std::process::{Command, Stdio};
use std::sync::{OnceLock, atomic::AtomicUsize};
use std::thread;

// A first-party Rust child fixture, compiled once locally. It implements
// only scripted JSONL protocol exchanges and never invokes Codex/models.
const MOCK: &str = include_str!("../../tests/fixtures/codex_child.rs");

fn mock_binary() -> &'static PathBuf {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY.get_or_init(|| {
        let directory = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("app-server-mock-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let json_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/json.rs");
        let source = directory.join("mock.rs");
        fs::write(
            &source,
            format!("#[path = {json_path:?}] mod json;\n{MOCK}"),
        )
        .unwrap();
        let binary = directory.join("mock-child");
        let rustc = std::env::var_os("VADE_TEST_RUSTC")
            .or_else(|| std::env::var_os("RUSTC"))
            .unwrap_or_else(|| "rustc".into());
        let status = Command::new(rustc)
            .arg("--edition=2024")
            .arg("--crate-name=vade_mock_app_server")
            .arg("-Adead_code")
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(
            status.success(),
            "first-party mock child compilation failed"
        );
        binary
    })
}

fn options(permission: PermissionPolicy, thread_id: Option<&str>, model: &str) -> Options {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let workspace = mock_binary().parent().unwrap().join(format!(
        "workspace-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&workspace).unwrap();
    fs::write(workspace.join("fixture-mode"), model).unwrap();
    Options {
        executable: mock_binary().clone(),
        workspace,
        thread_id: thread_id.map(str::to_owned),
        model: Some(model.into()),
        permission,
    }
}

fn client(permission: PermissionPolicy) -> Client {
    Client::connect(options(permission, None, "mock")).unwrap()
}

#[test]
fn mock_child_handshake_resume_stream_and_reuse() {
    let mut client = Client::connect(options(
        PermissionPolicy::ReadOnly,
        Some("saved-thread"),
        "mock",
    ))
    .unwrap();
    assert_eq!(client.thread_id(), "saved-thread");
    let cancel = AtomicBool::new(false);
    let mut output = String::new();
    let mut on_event = |event| {
        if let Event::AgentMessageDelta(text) = event {
            output.push_str(&text);
        }
    };
    assert_eq!(
        client.turn("pre-ack", &cancel, &mut on_event).unwrap(),
        TurnOutcome::Completed
    );
    assert_eq!(output, "before 🦀after é");
    let mut output = String::new();
    assert_eq!(
        client
            .turn("exact\n  🦀", &cancel, &mut |event| {
                if let Event::AgentMessageDelta(text) = event {
                    output.push_str(&text);
                }
            })
            .unwrap(),
        TurnOutcome::Completed
    );
    assert_eq!(output, "exact\n  🦀");
    assert_eq!(
        client
            .turn("initial-terminal", &cancel, &mut |_| {})
            .unwrap(),
        TurnOutcome::Completed
    );
}

#[test]
fn delegation_multiturn_exact_inputs_and_mock_owned_tool_history() {
    let mut client =
        Client::connect(options(PermissionPolicy::ReadOnly, None, "prefix-probe")).unwrap();
    let child_id = client.wire.child_id().unwrap();
    for prompt in [
        " exact first\n🦀 ",
        "quotes \" and slash \\",
        "third\n\n  whitespace",
    ] {
        let mut output = String::new();
        let result = client
            .turn(prompt, &AtomicBool::new(false), &mut |event| {
                if let Event::AgentMessageDelta(text) = event {
                    output.push_str(&text);
                }
            })
            .unwrap();
        assert_eq!(result, TurnOutcome::Completed);
        assert_eq!(output, prompt);
        assert_eq!(client.completed_text(), Some(prompt));
        assert_eq!(client.thread_id(), "mock-thread");
        assert_eq!(client.wire.child_id(), Some(child_id));
    }
}

#[test]
fn mock_child_failures_cannot_become_successful_turns() {
    let mut client = client(PermissionPolicy::ReadOnly);
    let cancel = AtomicBool::new(false);
    for input in ["failed", "contradictory"] {
        assert_eq!(
            client.turn(input, &cancel, &mut |_| {}).unwrap(),
            TurnOutcome::Failed("mock provider failed".into())
        );
    }
    assert_eq!(
        client.turn("rpc-error", &cancel, &mut |_| {}).unwrap(),
        TurnOutcome::Failed("mock start rejected".into())
    );
    let mut notices = 0;
    assert_eq!(
        client
            .turn("retry", &cancel, &mut |event| {
                if matches!(event, Event::Notice(_)) {
                    notices += 1;
                }
            })
            .unwrap(),
        TurnOutcome::Completed
    );
    assert_eq!(notices, 1);
}

#[test]
fn mock_child_permission_requests_are_denied_and_policy_is_explicit() {
    let mut client = client(PermissionPolicy::WorkspaceWrite);
    let mut notices = 0;
    assert_eq!(
        client
            .turn("approval", &AtomicBool::new(false), &mut |event| {
                if matches!(event, Event::Notice(_)) {
                    notices += 1;
                }
            })
            .unwrap(),
        TurnOutcome::Completed
    );
    assert_eq!(notices, 7);
    assert!(Client::connect(options(PermissionPolicy::ReadOnly, None, "escalate")).is_err());
    assert!(
        Client::connect(options(
            PermissionPolicy::ReadOnly,
            Some("saved"),
            "wrong-resume"
        ))
        .is_err()
    );
}

#[test]
fn mock_child_cancellation_drains_interrupted_status() {
    let mut client = client(PermissionPolicy::ReadOnly);
    let cancel = AtomicBool::new(false);
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(75));
            cancel.store(true, Ordering::Relaxed);
        });
        assert_eq!(
            client.turn("interrupt", &cancel, &mut |_| {}).unwrap(),
            TurnOutcome::Interrupted
        );
    });
    cancel.store(false, Ordering::Relaxed);
    assert_eq!(
        client.turn("still usable", &cancel, &mut |_| {}).unwrap(),
        TurnOutcome::Completed
    );
}

#[test]
fn mock_child_missing_terminal_and_exit_stop_and_reap_child() {
    let mut client = client(PermissionPolicy::ReadOnly);
    client.limits.interrupt = Duration::from_millis(150);
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(60));
            cancel.store(true, Ordering::Relaxed);
        });
        assert_eq!(
            client
                .turn("interrupt-timeout", &cancel, &mut |_| {})
                .unwrap_err()
                .kind(),
            io::ErrorKind::TimedOut
        );
    });
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(client.wire.stopped());
    let mut client = self::client(PermissionPolicy::ReadOnly);
    assert_eq!(
        client
            .turn("exit", &AtomicBool::new(false), &mut |_| {})
            .unwrap_err()
            .kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert!(client.wire.is_closed());
}

#[test]
fn mock_child_output_bounds_backpressure_and_idle_deadline() {
    let mut client = client(PermissionPolicy::ReadOnly);
    let mut bytes = 0;
    assert_eq!(
        client
            .turn("flood", &AtomicBool::new(false), &mut |event| {
                if let Event::AgentMessageDelta(text) = event {
                    bytes += text.len();
                }
            })
            .unwrap(),
        TurnOutcome::Completed
    );
    assert_eq!(bytes, 100);
    assert_eq!(
        client
            .turn("oversize", &AtomicBool::new(false), &mut |_| {})
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
    assert!(client.wire.is_closed());
    let mut client = self::client(PermissionPolicy::ReadOnly);
    client.limits.idle = Duration::from_millis(100);
    assert_eq!(
        client
            .turn("hang", &AtomicBool::new(false), &mut |_| {})
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut
    );
    assert!(client.wire.is_closed());
}

#[test]
fn cancellation_while_child_stops_reading_stdin_reaps_client() {
    let mut client =
        Client::connect(options(PermissionPolicy::ReadOnly, None, "block-input")).unwrap();
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(75));
            cancel.store(true, Ordering::Relaxed);
        });
        assert_eq!(
            client
                .turn(&"x".repeat(MAX_LINE / 2), &cancel, &mut |_| {})
                .unwrap_err()
                .kind(),
            io::ErrorKind::ConnectionAborted
        );
    });
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(client.wire.stopped());
}

#[test]
fn completed_items_supply_history_even_when_deltas_disagree() {
    let mut client = client(PermissionPolicy::ReadOnly);
    let mut provisional = String::new();
    assert_eq!(
        client
            .turn("disagreement", &AtomicBool::new(false), &mut |event| {
                if let Event::AgentMessageDelta(text) = event {
                    provisional.push_str(&text);
                }
            })
            .unwrap(),
        TurnOutcome::Completed
    );
    assert_eq!(provisional, "wrong draft");
    assert_eq!(client.completed_text(), Some("correct final"));
    assert!(
        client
            .turn("unfinished-item", &AtomicBool::new(false), &mut |_| {})
            .is_err()
    );
    assert_eq!(client.completed_text(), None);
}

#[test]
fn malformed_partial_unknown_and_conflicting_streams_reap_the_child() {
    for input in [
        "partial",
        "malformed",
        "unknown-event",
        "unknown-item",
        "unfinished-item",
        "conflicting-item",
        "wrong-response",
    ] {
        let mut client = client(PermissionPolicy::ReadOnly);
        let error = client
            .turn(input, &AtomicBool::new(false), &mut |_| {})
            .unwrap_err();
        assert!(
            matches!(
                error.kind(),
                io::ErrorKind::InvalidData | io::ErrorKind::Unsupported
            ),
            "{input}"
        );
        assert_eq!(client.completed_text(), None);
        assert_eq!(
            client.cleanup(),
            Some(Cleanup {
                child_reaped: true,
                io_workers_joined: true
            }),
            "{input}"
        );
    }
}

#[test]
fn all_returned_policy_mismatches_fail_closed() {
    for model in [
        "wrong-cwd",
        "wrong-approval",
        "missing-network",
        "network-enabled",
        "outside-root",
        "tmp-write",
        "escalate",
    ] {
        assert!(
            Client::connect(options(PermissionPolicy::WorkspaceWrite, None, model)).is_err(),
            "{model}"
        );
    }
}

#[test]
fn handshake_shape_is_pinned_without_guessing_version_capabilities() {
    for version in ["mock", "v2"] {
        let client =
            Client::connect(options(PermissionPolicy::ReadOnly, None, version)).unwrap();
        assert_eq!(client.thread_id(), "mock-thread");
    }
    assert!(Client::connect(options(PermissionPolicy::ReadOnly, None, "bad-init")).is_err());
}

#[test]
fn handshake_hang_can_be_cancelled_or_times_out_locally() {
    let limits = Limits {
        rpc: Duration::from_millis(100),
        ..Limits::default()
    };
    let start = Instant::now();
    assert_eq!(
        Client::connect_unix(
            options(PermissionPolicy::ReadOnly, None, "init-hang"),
            &AtomicBool::new(false),
            limits
        )
        .err()
        .unwrap()
        .kind(),
        io::ErrorKind::TimedOut
    );
    let cancel = AtomicBool::new(false);
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(60));
            cancel.store(true, Ordering::Relaxed);
        });
        assert_eq!(
            Client::connect_cancellable(
                options(PermissionPolicy::ReadOnly, None, "init-hang"),
                &cancel
            )
            .err()
            .unwrap()
            .kind(),
            io::ErrorKind::ConnectionAborted
        );
    });
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[test]
fn cancellation_during_tool_and_completion_race_remain_distinct() {
    for (input, expected) in [
        ("cancel-tool", TurnOutcome::Interrupted),
        ("cancel-race", TurnOutcome::Completed),
    ] {
        let mut client = client(PermissionPolicy::ReadOnly);
        let cancel = AtomicBool::new(false);
        thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(60));
                cancel.store(true, Ordering::Relaxed);
            });
            assert_eq!(client.turn(input, &cancel, &mut |_| {}).unwrap(), expected);
        });
    }
    let mut client = client(PermissionPolicy::ReadOnly);
    assert_eq!(
        client
            .turn("must not dispatch", &AtomicBool::new(true), &mut |_| {})
            .unwrap(),
        TurnOutcome::NotStarted
    );
}

#[test]
fn rejected_interrupt_is_unknown_outcome_not_turn_failure_or_stop() {
    let mut client = client(PermissionPolicy::ReadOnly);
    let cancel = AtomicBool::new(false);
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(60));
            cancel.store(true, Ordering::Relaxed);
        });
        assert!(
            client
                .turn("interrupt-rejected", &cancel, &mut |_| {})
                .is_err()
        );
    });
    assert_eq!(
        client.cleanup(),
        Some(Cleanup {
            child_reaped: true,
            io_workers_joined: true
        })
    );
}

#[test]
fn steering_has_exact_input_scoped_acknowledgments_and_no_replay() {
    for (prompt, expected) in [
        ("steer", SteeringStatus::Accepted),
        ("steer-reject", SteeringStatus::Rejected),
        ("steer-race", SteeringStatus::Unconfirmed),
    ] {
        let mut client = client(PermissionPolicy::ReadOnly);
        let (tx, rx) = mpsc::sync_channel(8);
        let input = "  preserve\n exact 🦀  ";
        tx.send(Control::Steer {
            request_id: 41,
            text: input.into(),
        })
        .unwrap();
        let mut receipts = Vec::new();
        assert_eq!(
            client
                .turn_with_control(prompt, &AtomicBool::new(false), &rx, &mut |event| {
                    if let Event::Steering { request_id, status } = event {
                        receipts.push((request_id, status));
                    }
                })
                .unwrap(),
            TurnOutcome::Completed
        );
        assert_eq!(receipts, vec![(41, expected)]);
        if prompt == "steer" {
            assert_eq!(client.completed_text(), Some(input));
        }
        assert_eq!(
            client
                .turn("next explicit prompt", &AtomicBool::new(false), &mut |_| {})
                .unwrap(),
            TurnOutcome::Completed
        );
        assert_eq!(client.completed_text(), Some("next explicit prompt"));
    }
}

#[test]
fn steering_wrong_identity_and_missing_ack_fail_and_reap() {
    for prompt in ["steer-wrong", "steer-timeout"] {
        let mut client = client(PermissionPolicy::ReadOnly);
        client.limits.rpc = Duration::from_millis(120);
        let (tx, rx) = mpsc::sync_channel(8);
        tx.send(Control::Steer {
            request_id: 1,
            text: "explicit change".into(),
        })
        .unwrap();
        assert!(
            client
                .turn_with_control(prompt, &AtomicBool::new(false), &rx, &mut |_| {})
                .is_err()
        );
        assert_eq!(
            client.cleanup(),
            Some(Cleanup {
                child_reaped: true,
                io_workers_joined: true
            })
        );
    }
}

#[test]
fn cancellation_preempts_unsent_steering_and_survives_output_burst() {
    let mut client = client(PermissionPolicy::ReadOnly);
    let (tx, rx) = mpsc::sync_channel(8);
    tx.send(Control::Steer {
        request_id: 1,
        text: "must not be sent".into(),
    })
    .unwrap();
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    let mut deltas = 0;
    // Cancellation is triggered by turn acknowledgment, before the control
    // dequeue. The fixture accepts only interruption, never steering.
    assert_eq!(
        client
            .turn_with_control("burst-control", &cancel, &rx, &mut |event| {
                if matches!(event, Event::Phase(Phase::Generating)) {
                    cancel.store(true, Ordering::Relaxed);
                }
                if matches!(event, Event::AgentMessageDelta(_)) {
                    deltas += 1;
                }
            })
            .unwrap(),
        TurnOutcome::Interrupted
    );
    assert_eq!(deltas, 2000);
    assert!(matches!(
        rx.try_recv(),
        Ok(Control::Steer { request_id: 1, .. })
    ));
    crate::diagnostics::stderr_line(format_args!(
        "child burst cancellation: 2000 x 128-byte deltas, terminal acknowledgment in {} ms (includes 120 ms fixture delay)",
        start.elapsed().as_millis()
    ));
}
