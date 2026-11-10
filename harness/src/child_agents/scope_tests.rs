//! Write scopes and grants: read-only and non-recursive scopes, disjoint
//! assignments, unsafe paths and cancellation in the middle of a write.
use super::*;

#[test]
fn readonly_and_nonrecursive_scopes_are_enforced_beneath_model_schemas() {
    for (name, args) in [
        ("workspace_write", val(r#"{"path":"src/a.rs"}"#)),
        ("child_spawn", val(r#"{"task":"escape"}"#)),
    ] {
        let response = proposal("denied", name, args);
        let factory: Arc<Factory> = Arc::new(move |_, _| {
            Ok(ChildWorker {
                observer: None,
                journal_name: None,
                model: Box::new(Replies {
                    replies: vec![response.clone(), answer("denied acknowledged")].into(),
                    dispatches: Arc::new(AtomicUsize::new(0)),
                    prompt_tokens: 1,
                }),
                tools: Box::new(NoTools),
            })
        });
        let mut scheduler = scheduler(write_config(), Arc::new(AtomicBool::new(false)), factory);
        let id = scheduler.spawn(Task::read_only("read only")).unwrap();
        finished(&scheduler, id);
        let result = scheduler.take_result(id).unwrap();
        assert!(
            result
                .conversation
                .unwrap()
                .messages()
                .iter()
                .any(|m| m.role() == "tool" && m.serialized_json().contains("denied"))
        );
        assert!(result.tools.is_empty());
    }
}

#[test]
fn disjoint_assignments_block_aliases_and_parent_writes_until_collection() {
    let mut scheduler = scheduler(
        write_config(),
        Arc::new(AtomicBool::new(false)),
        factory(vec![answer("done")], Arc::new(AtomicUsize::new(0))),
    );
    let a = scheduler.spawn(write_task("a")).unwrap();
    let b = scheduler.spawn(write_task("b")).unwrap();
    for id in [a, b] {
        finished(&scheduler, id);
    }
    assert_eq!(
        scheduler.spawn(write_task("case_alias")),
        Err(ScheduleError::WriteConflict)
    );
    assert_eq!(
        scheduler.spawn(write_task("ancestor")),
        Err(ScheduleError::WriteConflict)
    );
    let call = ToolCall {
        id: "parent-write".into(),
        name: "workspace_write".into(),
        arguments: "{}".into(),
    };
    let args = val(r#"{"path":"src/a.rs"}"#);
    let invocation = || ToolInvocation {
        call: &call,
        arguments: &args,
        max_output_bytes: 4096,
    };
    assert_eq!(
        scheduler.check_parent(&invocation()),
        Err(ToolError::Denied)
    );
    scheduler.take_result(a).unwrap();
    assert_eq!(scheduler.check_parent(&invocation()), Ok(()));
    scheduler.spawn(write_task("case_alias")).unwrap();
}

#[test]
fn scope_rejects_unsafe_paths_unavailable_tools_and_model_minted_grants() {
    for path in [
        "../escape",
        "/absolute",
        "a/../b",
        "a//b",
        "a/./b",
        "a/",
        ".hidden",
        "src/secret.key",
        "secrets.json",
        "src/é.rs",
        "a\\b",
    ] {
        let mut config = write_config();
        config.grants.get_mut("a").unwrap().write_paths = vec![path.into()];
        assert!(
            Scheduler::new(
                StablePrefix::from_parent(&parent()).unwrap(),
                config,
                Arc::new(AtomicBool::new(false)),
                factory(vec![], Arc::new(AtomicUsize::new(0)))
            )
            .is_err(),
            "accepted {path}"
        );
    }
    let mut config = write_config();
    config
        .grants
        .get_mut("a")
        .unwrap()
        .tools
        .push("shell".into());
    assert!(
        Scheduler::new(
            StablePrefix::from_parent(&parent()).unwrap(),
            config,
            Arc::new(AtomicBool::new(false)),
            factory(vec![], Arc::new(AtomicUsize::new(0)))
        )
        .is_err()
    );
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(vec![], Arc::new(AtomicUsize::new(0))),
    );
    assert_eq!(
        scheduler.spawn(write_task("self-authorized")),
        Err(ScheduleError::UnknownGrant)
    );
}

struct MidTool {
    entered: mpsc::SyncSender<()>,
    completed: bool,
}
impl ToolExecutor for MidTool {
    fn execute(&mut self, _: ToolInvocation<'_>, cancel: &AtomicBool) -> Result<String, ToolError> {
        self.entered.send(()).unwrap();
        while !cancel.load(Ordering::Acquire) {
            thread::park_timeout(Duration::from_millis(1));
        }
        if self.completed {
            Ok("applied_before_cancel".into())
        } else {
            Err(ToolError::Cancelled)
        }
    }
}

#[test]
fn cancellation_mid_write_preserves_real_completion_or_marks_uncertainty() {
    for completed in [true, false] {
        let (tx, rx) = mpsc::sync_channel(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let factory: Arc<Factory> = Arc::new(move |_, _| {
            Ok(ChildWorker {
                observer: None,
                journal_name: None,
                model: Box::new(Replies {
                    replies: vec![proposal(
                        "write-1",
                        "workspace_write",
                        val(r#"{"path":"src/a.rs"}"#),
                    )]
                    .into(),
                    dispatches: Arc::new(AtomicUsize::new(0)),
                    prompt_tokens: 1,
                }),
                tools: Box::new(MidTool {
                    entered: tx.clone(),
                    completed,
                }),
            })
        });
        let mut scheduler = scheduler(write_config(), Arc::clone(&cancel), factory);
        let id = scheduler.spawn(write_task("a")).unwrap();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        cancel.store(true, Ordering::Release);
        finished(&scheduler, id);
        let result = scheduler.take_result(id).unwrap();
        assert_eq!(result.outcome, Outcome::Cancelled);
        assert_eq!(result.stop_requested, Some(StopRequest::ParentCancelled));
        assert_eq!(result.uncertain_writes(), !completed);
        assert_eq!(
            result.tools[0].status,
            if completed {
                ToolStatus::Completed
            } else {
                ToolStatus::Cancelled
            }
        );
        let messages = result.conversation.unwrap();
        let receipt = messages.messages().last().unwrap();
        assert_eq!(receipt.tool_call_id(), Some("write-1"));
        assert!(receipt.serialized_json().contains(if completed {
            "applied_before_cancel"
        } else {
            "cancelled"
        }));
    }
}

#[test]
fn out_of_scope_write_is_denied_before_executor() {
    for path in ["src/b.rs", "SRC/A.RS"] {
        let factory: Arc<Factory> = Arc::new(move |_, _| {
            Ok(ChildWorker {
                observer: None,
                journal_name: None,
                model: Box::new(Replies {
                    replies: vec![
                        proposal(
                            "write",
                            "workspace_write",
                            Value::Object(vec![("path".into(), Value::String(path.into()))]),
                        ),
                        answer("denied"),
                    ]
                    .into(),
                    dispatches: Arc::new(AtomicUsize::new(0)),
                    prompt_tokens: 1,
                }),
                tools: Box::new(NoTools),
            })
        });
        let mut scheduler = scheduler(write_config(), Arc::new(AtomicBool::new(false)), factory);
        let id = scheduler.spawn(write_task("a")).unwrap();
        finished(&scheduler, id);
        let result = scheduler.take_result(id).unwrap();
        assert_eq!(result.outcome, Outcome::Completed);
        assert!(result.tools.is_empty());
    }
}
