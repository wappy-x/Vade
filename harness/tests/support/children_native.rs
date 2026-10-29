//! Integrated native scopes + shared approval UI against loopback mocks only.
use super::*;

struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "vade-child-tools-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn patch_for(path: &str) -> Value {
    object(vec![(
        "patch",
        string(&format!(
            "*** Begin Patch\n*** Add File: {path}\n+child approved\n*** End Patch"
        )),
    )])
}

#[test]
#[cfg(target_os = "macos")]
fn raw_child_edits_ask_parent_once_each_and_cancel_pending_or_queued_approvals() {
    for cancel in [false, true] {
        let workspace = Workspace::new();
        let (listener, endpoint) = listen();
        let (proposed, ready) = mpsc::channel();
        let server = thread::spawn(move || {
            let mut parent = accept(&listener);
            let request = json::parse(&next_request(&mut parent)).unwrap();
            assert_eq!(request.get("seed"), Some(&Value::Number("137".into())));
            assert_eq!(
                request.get("temperature").unwrap(),
                &Value::Number("0.6".into())
            );
            reply(
                &mut parent,
                calls(vec![
                    call(
                        "spawn-a",
                        "child_spawn",
                        object(vec![("task", string("a")), ("grant", string("edit-a"))]),
                    ),
                    call(
                        "spawn-b",
                        "child_spawn",
                        object(vec![("task", string("b")), ("grant", string("edit-b"))]),
                    ),
                ]),
                Some(0),
            );
            let mut workers = Vec::new();
            for _ in 0..2 {
                tool_guard(&mut parent);
                let mut child = accept(&listener);
                let proposed = proposed.clone();
                workers.push(thread::spawn(move || {
                    let body = json::parse(&next_request(&mut child)).unwrap();
                    assert_eq!(body.get("seed"), Some(&Value::Number("137".into())));
                    assert_eq!(body.get("temperature"), Some(&Value::Number("0.6".into())));
                    let envelope = json::parse(
                        messages(&body)
                            .last()
                            .unwrap()
                            .get("content")
                            .unwrap()
                            .as_str()
                            .unwrap(),
                    )
                    .unwrap();
                    let path = envelope.get("task").unwrap().as_str().unwrap();
                    assert_eq!(
                        envelope.get("write_paths"),
                        Some(&Value::Array(vec![string(path)]))
                    );
                    assert!(
                        !envelope
                            .get("allowed_tools")
                            .unwrap()
                            .as_array()
                            .unwrap()
                            .contains(&string("workspace_shell"))
                    );
                    reply(
                        &mut child,
                        calls(vec![call("edit", "apply_patch", patch_for(path))]),
                        Some(0),
                    );
                    proposed.send(()).unwrap();
                    if cancel {
                        match child.read(&mut [0]) {
                            Ok(0) => (),
                            Err(error) if error.kind() == std::io::ErrorKind::ConnectionReset => (),
                            other => {
                                panic!("cancelled child must not ask another model turn: {other:?}")
                            }
                        }
                        return false;
                    }
                    let followup = json::parse(&next_request(&mut child)).unwrap();
                    let result = messages(&followup)
                        .last()
                        .unwrap()
                        .get("content")
                        .unwrap()
                        .as_str()
                        .unwrap();
                    let applied = result.contains("Applied 1");
                    assert!(applied || result.contains("denied"), "{result}");
                    reply(
                        &mut child,
                        answer(if applied {
                            "edit applied"
                        } else {
                            "edit denied"
                        }),
                        Some(0),
                    );
                    applied
                }));
            }
            let approved = workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .filter(|applied| *applied)
                .count();
            assert_eq!(approved, usize::from(!cancel));
            // Parent HTTP stays pending throughout child approvals; controls and
            // approval answers must remain available on their independent lane.
            let body = json::parse(&next_request(&mut parent)).unwrap();
            let ids: Vec<_> = ["spawn-a", "spawn-b"]
                .iter()
                .map(|id| {
                    tool_result(&body, id)
                        .get("task_id")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_owned()
                })
                .collect();
            reply(
                &mut parent,
                calls(
                    ids.iter()
                        .enumerate()
                        .map(|(n, id)| {
                            call(
                                &format!("wait-{n}"),
                                "child_wait",
                                object(vec![("task_id", string(id))]),
                            )
                        })
                        .collect(),
                ),
                Some(0),
            );
            // Cancellation may race tool entry. A cancelled write receipt can
            // conservatively stop the parent instead of dispatching generation.
            let Some(body) = next_request_or_closed(&mut parent) else {
                assert!(cancel);
                return false;
            };
            let body = json::parse(&body).unwrap();
            for n in 0..2 {
                let result = tool_result(&body, &format!("wait-{n}"));
                assert_eq!(
                    result.get("outcome").and_then(Value::as_str),
                    Some(if cancel { "Cancelled" } else { "Completed" })
                );
                assert!(
                    result
                        .get("journal")
                        .and_then(Value::as_str)
                        .unwrap()
                        .starts_with("child-")
                );
            }
            stream_reply(
                &mut parent,
                answer("Scoped child approvals finished."),
                Some(0),
            );
            true
        });
        let mut ui = raw_ui::Ui::start_with_args(
            "agent",
            &endpoint,
            workspace.0.to_str().unwrap(),
            &[
                "--children",
                "--seed",
                "137",
                "--temperature",
                "0.6",
                "--child-edit",
                "edit-a=a",
                "--child-edit",
                "edit-b=b",
                "--session",
                "parent",
            ],
        );
        ui.send(b"edit with two children\r");
        let display = ui.until("Approval required", Duration::from_secs(4));
        assert!(display.contains("Child child-"));
        assert!(!workspace.0.join("a").exists() && !workspace.0.join("b").exists());
        if cancel {
            for _ in 0..2 {
                ready.recv_timeout(Duration::from_secs(3)).unwrap();
            }
            ui.send(b"/children cancel all\r");
        } else {
            let next_file = if display.contains("+++ a") {
                "+++ b"
            } else {
                "+++ a"
            };
            ui.send(b"/approve\r");
            // Redrawing retained transcript rows is not a new approval.
            let mut display = ui.until(next_file, Duration::from_secs(4));
            if !display.contains("Approval required") {
                display.push_str(&ui.until("Approval required", Duration::from_secs(4)));
            }
            assert!(display.contains("Child child-"));
            ui.send(b"/deny\r");
        }
        let finished = ui.until_any(
            if cancel {
                &["Scoped child approvals finished.", "Stopped: a tool failed"]
            } else {
                &["Scoped child approvals finished."]
            },
            Duration::from_secs(4),
        );
        ui.send(b"/quit\r");
        drop(ui); // Release an idle keepalive when cancellation stopped dispatch.
        if !server.join().unwrap() {
            assert!(finished.contains("Stopped: a tool failed"));
            let w = vade_harness::tools::Workspace::new(&workspace.0).unwrap();
            let (_, c, _) = vade_harness::journal::Journal::resume(
                &w, "parent", vade_harness::agent::Limits::default()).unwrap();
            // A safety stop is valid only with an authoritative uncertainty receipt.
            assert!(c.messages().iter().any(|m| m.value()
                .get("content")
                .and_then(Value::as_str).and_then(|s| json::parse(s).ok())
                .is_some_and(|v| v.get("uncertain_writes") == Some(&Value::Bool(true)))));
        }
        let changed = ["a", "b"]
            .iter()
            .filter(|name| workspace.0.join(name).exists())
            .count();
        assert_eq!(changed, usize::from(!cancel));
    }
}

#[test]
fn cli_scope_grant_never_overrides_never_or_noninteractive_ask_approval() {
    for (approval, sampled) in ["never", "ask", "auto"].into_iter().flat_map(|approval| [false, true].map(|sampled| (approval, sampled))) {
        let workspace = Workspace::new();
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut parent = accept(&listener);
            let parent_request = json::parse(&next_request(&mut parent)).unwrap();
            assert!(parent_request.get("seed").is_none());
            assert_eq!(parent_request.get("temperature"), sampled.then_some(&Value::Number("0.6".into())));
            reply(
                &mut parent,
                calls(vec![call(
                    "spawn",
                    "child_spawn",
                    object(vec![("task", string("edit")), ("grant", string("edit"))]),
                )]),
                Some(0),
            );
            tool_guard(&mut parent);
            let mut child = accept(&listener);
            let child_request = json::parse(&next_request(&mut child)).unwrap();
            assert!(child_request.get("seed").is_none(), "native child invented a seed");
            assert_eq!(child_request.get("temperature"), parent_request.get("temperature"));
            reply(
                &mut child,
                calls(vec![call("edit", "apply_patch", patch_for("result"))]),
                Some(0),
            );
            let body = json::parse(&next_request(&mut child)).unwrap();
            assert!(body.get("seed").is_none(), "child tool continuation invented a seed");
            assert_eq!(body.get("temperature"), parent_request.get("temperature"));
            let text = messages(&body)
                .last()
                .unwrap()
                .get("content")
                .unwrap()
                .as_str()
                .unwrap();
            assert!(if approval == "auto" {
                text.contains("Applied 1")
            } else {
                text.contains("denied")
            });
            reply(&mut child, answer("child done"), Some(0));
            let body = json::parse(&next_request(&mut parent)).unwrap();
            assert!(body.get("seed").is_none(), "parent tool continuation invented a seed");
            let id = tool_result(&body, "spawn")
                .get("task_id")
                .unwrap()
                .as_str()
                .unwrap()
                .to_owned();
            reply(
                &mut parent,
                calls(vec![call(
                    "wait",
                    "child_wait",
                    object(vec![("task_id", string(&id))]),
                )]),
                Some(0),
            );
            let body = json::parse(&next_request(&mut parent)).unwrap();
            assert!(body.get("seed").is_none(), "parent tool continuation invented a seed");
            let result = tool_result(&body, "wait");
            assert_eq!(
                result.get("outcome").and_then(Value::as_str),
                Some("Completed")
            );
            let journal = result.get("journal").unwrap().as_str().unwrap().to_owned();
            reply(&mut parent, answer("parent done"), Some(0));
            journal
        });
        let mut command = crate::home::harness();
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
                "--children",
                "--child-edit",
                "edit=result",
                "--approval",
                approval,
                "--prompt",
                "host-scoped edit",
                "--session",
                "parent",
            ]);
        if sampled { command.args(["--temperature", "0.6"]); }
        let output = cli::Cli::start(&mut command).finish();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(workspace.0.join("result").exists(), approval == "auto");
        let journal = server.join().unwrap();
        let w = vade_harness::tools::Workspace::new(&workspace.0).unwrap();
        let (_, child, _) = vade_harness::journal::Journal::resume(
            &w,
            &journal,
            vade_harness::agent::Limits::default(),
        )
        .unwrap();
        assert!(
            child
                .messages()
                .iter()
                .any(|m| m.tool_call_id() == Some("edit"))
        );
        let (_, parent, _) = vade_harness::journal::Journal::resume(
            &w,
            "parent",
            vade_harness::agent::Limits::default(),
        )
        .unwrap();
        assert!(
            parent
                .messages()
                .iter()
                .any(|m| m.tool_call_id() == Some("wait"))
        );
        assert_eq!(
            workspace.0.join("result").exists(),
            approval == "auto",
            "journal replay never invokes edits"
        );
    }
}
