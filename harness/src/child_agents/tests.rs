use super::*;
use crate::agent_wire::{ToolCall, WireLimits};
use crate::json::{self, Value};
use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;

fn val(text: &str) -> Value {
    json::parse(text).unwrap()
}
fn answer(text: &str) -> Value {
    Value::Object(vec![
        ("role".into(), Value::String("assistant".into())),
        ("content".into(), Value::String(text.into())),
    ])
}
fn proposal(id: &str, name: &str, arguments: Value) -> Value {
    val(&format!(
        r#"{{"role":"assistant","content":null,"tool_calls":[{{"id":"{id}","type":"function","function":{{"name":"{name}","arguments":{}}}}}]}}"#,
        json::stringify(&Value::String(json::stringify(&arguments).unwrap())).unwrap()
    ))
}
fn turn(message: Value, wire: WireLimits) -> io::Result<AssistantTurn> {
    let reason = if message.get("tool_calls").is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    AssistantTurn::from_response(
        &Value::Object(vec![(
            "choices".into(),
            Value::Array(vec![Value::Object(vec![
                ("message".into(), message),
                ("finish_reason".into(), Value::String(reason.into())),
            ])]),
        )]),
        wire,
    )
}
fn parent() -> Conversation {
    let mut request = val(
        r#"{"model":"mock","stream":true,"temperature":0,"tools":[{"type":"function","function":{"name":"workspace_read","parameters":{"type":"object"}}},{"type":"function","function":{"name":"workspace_list","parameters":{"type":"object"}}},{"type":"function","function":{"name":"workspace_write","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":" exact system\n  🦀"},{"role":"developer","content":"frozen project rules"},{"role":"user","content":"private parent history"}]}"#,
    );
    let Value::Object(fields) = &mut request else {
        unreachable!()
    };
    let Value::Array(tools) = &mut fields.iter_mut().find(|(k, _)| k == "tools").unwrap().1 else {
        unreachable!()
    };
    let Value::Array(children) = definitions() else {
        unreachable!()
    };
    tools.extend(children);
    Conversation::from_request(request, Limits::default()).unwrap()
}

struct Replies {
    replies: VecDeque<Value>,
    dispatches: Arc<AtomicUsize>,
    prompt_tokens: u64,
}
impl ChildModel for Replies {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
        meter: &mut TokenMeter,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        if cancel.load(Ordering::Acquire) {
            return Err(io::ErrorKind::Interrupted.into());
        }
        meter.reserve(self.prompt_tokens, meter.output)?;
        self.dispatches.fetch_add(1, Ordering::SeqCst);
        turn(
            self.replies.pop_front().expect("unexpected child request"),
            request.wire,
        )
    }
}
struct NoTools;
impl ToolExecutor for NoTools {
    fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        panic!("unexpected tool dispatch")
    }
}
struct ReadTools;
impl ToolExecutor for ReadTools {
    fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        Ok("bounded read".into())
    }
}
fn factory(replies: Vec<Value>, dispatches: Arc<AtomicUsize>) -> Arc<Factory> {
    Arc::new(move |_, _| {
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(Replies {
                replies: replies.clone().into(),
                dispatches: Arc::clone(&dispatches),
                prompt_tokens: 10,
            }),
            tools: Box::new(ReadTools),
        })
    })
}
fn scheduler(config: Config, cancel: Arc<AtomicBool>, factory: Arc<Factory>) -> Scheduler {
    Scheduler::new(
        StablePrefix::from_parent(&parent()).unwrap(),
        config,
        cancel,
        factory,
    )
    .unwrap()
}
fn finished(scheduler: &Scheduler, id: TaskId) -> TaskSnapshot {
    let deadline = Instant::now() + Duration::from_secs(5);
    let view = scheduler.view();
    loop {
        let revision = view.revision();
        let snapshot = view.snapshots().into_iter().find(|s| s.id == id).unwrap();
        if matches!(
            snapshot.state,
            TaskState::Finished(_) | TaskState::Collected(_)
        ) {
            return snapshot;
        }
        assert!(Instant::now() < deadline, "child cleanup did not return");
        view.wait_for_change(revision, Duration::from_millis(100));
    }
}
fn write_config() -> Config {
    let policy = ToolPolicy::new(
        [
            ("workspace_read".into(), Access::Read),
            ("workspace_list".into(), Access::Read),
            (
                "workspace_write".into(),
                Access::Write {
                    path_argument: "path".into(),
                },
            ),
        ]
        .into(),
    )
    .unwrap();
    let scope = |path: &str| Scope {
        tools: vec!["workspace_read".into(), "workspace_write".into()],
        write_paths: vec![path.into()],
    };
    Config {
        policy,
        grants: [
            ("a".into(), scope("src/a.rs")),
            ("case_alias".into(), scope("SRC/A.RS")),
            ("ancestor".into(), scope("src")),
            ("b".into(), scope("src/b.rs")),
        ]
        .into(),
        ..Config::default()
    }
}
fn write_task(grant: &str) -> Task {
    Task {
        grant: Some(grant.into()),
        ..Task::read_only("bounded edit")
    }
}

#[test]
fn child_fresh_context_preserves_exact_system_rules_and_tools() {
    let parent = parent();
    let before = parent.request(WireLimits::default()).to_json().unwrap();
    let prefix = StablePrefix::from_parent(&parent).unwrap();
    let child = prefix
        .conversation(
            &Task::read_only("investigate"),
            &ToolPolicy::default().read_scope(&prefix.advertised),
            Limits::default(),
        )
        .unwrap();
    assert_eq!(parent.options().get("tools"), child.options().get("tools"));
    for (a, b) in parent.messages()[..2].iter().zip(&child.messages()[..2]) {
        assert_eq!(a.serialized_json(), b.serialized_json());
    }
    assert_eq!(child.messages().len(), 3);
    assert!(
        !child
            .request(WireLimits::default())
            .to_json()
            .unwrap()
            .contains("private parent history")
    );
    assert_eq!(
        before,
        parent.request(WireLimits::default()).to_json().unwrap()
    );
}

#[test]
fn parent_cache_identity_is_not_inherited_and_checkpoints_fail_closed() {
    let mut request =
        json::parse(&parent().request(WireLimits::default()).to_json().unwrap()).unwrap();
    let Value::Object(fields) = &mut request else {
        unreachable!()
    };
    fields.push((
        "prompt_cache_key".into(),
        Value::String("parent-key".into()),
    ));
    let parent = Conversation::from_request(request.clone(), Limits::default()).unwrap();
    let prefix = StablePrefix::from_parent(&parent).unwrap();
    let child = prefix
        .conversation(
            &Task::read_only("fresh"),
            &Scope {
                tools: vec![],
                write_paths: vec![],
            },
            Limits::default(),
        )
        .unwrap();
    assert!(child.options().get("prompt_cache_key").is_none());
    let Value::Object(fields) = &mut request else {
        unreachable!()
    };
    fields.push((
        "checkpoint_handle".into(),
        Value::String("stale-parent-handle".into()),
    ));
    let parent = Conversation::from_request(request, Limits::default()).unwrap();
    assert!(StablePrefix::from_parent(&parent).is_err());
}

#[test]
fn result_has_one_owner_and_parent_is_unchanged_until_collection() {
    let parent = parent();
    let before = parent.request(WireLimits::default()).to_json().unwrap();
    let mut scheduler = Scheduler::new(
        StablePrefix::from_parent(&parent).unwrap(),
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(
            vec![answer("focused result")],
            Arc::new(AtomicUsize::new(0)),
        ),
    )
    .unwrap();
    let id = scheduler.spawn(Task::read_only("investigate")).unwrap();
    assert_eq!(
        finished(&scheduler, id).state,
        TaskState::Finished(Outcome::Completed)
    );
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.summary, "focused result");
    assert_eq!(result.tokens_reserved, 1034);
    assert_eq!(
        scheduler.take_result(id).err(),
        Some(ScheduleError::AlreadyCollected)
    );
    assert_eq!(
        before,
        parent.request(WireLimits::default()).to_json().unwrap()
    );
}

struct BlockingModel {
    entered: mpsc::SyncSender<()>,
    live: Arc<AtomicUsize>,
}
impl ChildModel for BlockingModel {
    fn complete(
        &mut self,
        _: ModelRequest<'_>,
        cancel: &AtomicBool,
        meter: &mut TokenMeter,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        meter.reserve(10, meter.output)?;
        self.live.fetch_add(1, Ordering::SeqCst);
        self.entered.send(()).unwrap();
        while !cancel.load(Ordering::Acquire) {
            thread::park_timeout(Duration::from_millis(1));
        }
        self.live.fetch_sub(1, Ordering::SeqCst);
        Err(io::ErrorKind::Interrupted.into())
    }
}
fn blocking_factory(entered: mpsc::SyncSender<()>, live: Arc<AtomicUsize>) -> Arc<Factory> {
    Arc::new(move |_, _| {
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(BlockingModel {
                entered: entered.clone(),
                live: Arc::clone(&live),
            }),
            tools: Box::new(NoTools),
        })
    })
}

#[test]
fn wall_deadline_cancels_model_and_reports_request_separately() {
    let (tx, rx) = mpsc::sync_channel(1);
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        blocking_factory(tx, Arc::new(AtomicUsize::new(0))),
    );
    let mut task = Task::read_only("deadline");
    task.quota.wall_time = Duration::from_millis(60);
    let id = scheduler.spawn(task).unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let snapshot = finished(&scheduler, id);
    assert_eq!(snapshot.stop_requested, Some(StopRequest::WallTime));
    assert_eq!(snapshot.state, TaskState::Finished(Outcome::Cancelled));
}

#[test]
fn token_exhaustion_prevents_generation_and_never_refunds_reserved_requests() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(
            vec![
                proposal("read", "workspace_read", val(r#"{"path":"x"}"#)),
                answer("must not dispatch"),
            ],
            Arc::clone(&calls),
        ),
    );
    let mut task = Task::read_only("budget");
    task.quota.tokens = 1500;
    let id = scheduler.spawn(task).unwrap();
    assert_eq!(
        finished(&scheduler, id).state,
        TaskState::Finished(Outcome::QuotaTokens)
    );
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.tokens_reserved, 1034);
    assert_eq!(result.report.unwrap().tool_calls, 1);
    assert_eq!(
        result
            .conversation
            .unwrap()
            .messages()
            .last()
            .unwrap()
            .role(),
        "tool"
    );
}

#[test]
fn model_turn_exhaustion_keeps_completed_tools_in_child_history() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(
            vec![proposal("read", "workspace_read", val(r#"{"path":"x"}"#))],
            Arc::clone(&calls),
        ),
    );
    let mut task = Task::read_only("one turn");
    task.quota.model_turns = 1;
    let id = scheduler.spawn(task).unwrap();
    assert_eq!(
        finished(&scheduler, id).state,
        TaskState::Finished(Outcome::Runtime(StopReason::Limit(
            agent::LimitKind::ModelTurns
        )))
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        scheduler
            .take_result(id)
            .unwrap()
            .conversation
            .unwrap()
            .messages()
            .last()
            .unwrap()
            .role(),
        "tool"
    );
}

#[test]
fn total_allocations_remain_finite_after_collection() {
    let mut scheduler = scheduler(
        Config {
            max_tasks: 1,
            max_concurrent: 1,
            ..Config::default()
        },
        Arc::new(AtomicBool::new(false)),
        factory(vec![answer("done")], Arc::new(AtomicUsize::new(0))),
    );
    let id = scheduler.spawn(Task::read_only("one")).unwrap();
    finished(&scheduler, id);
    scheduler.take_result(id).unwrap();
    assert_eq!(
        scheduler.spawn(Task::read_only("two")),
        Err(ScheduleError::Quota)
    );
}

struct ParentModel {
    step: usize,
    id: Option<String>,
}
impl Model for ParentModel {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        _: &AtomicBool,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        let response = match self.step {
            0 => proposal("spawn-id", "child_spawn", val(r#"{"task":"investigate"}"#)),
            1 => {
                let receipt = request
                    .messages
                    .last()
                    .unwrap()
                    .value()
                    .get("content")
                    .unwrap()
                    .as_str()
                    .unwrap();
                let value = json::parse(receipt).unwrap();
                self.id = Some(value.get("task_id").unwrap().as_str().unwrap().to_owned());
                proposal(
                    "wait-id",
                    "child_wait",
                    Value::Object(vec![(
                        "task_id".into(),
                        Value::String(self.id.clone().unwrap()),
                    )]),
                )
            }
            2 => {
                let receipt = request.messages.last().unwrap();
                assert_eq!(receipt.tool_call_id(), Some("wait-id"));
                let receipt =
                    json::parse(receipt.value().get("content").unwrap().as_str().unwrap()).unwrap();
                assert_eq!(
                    receipt.get("outcome").and_then(Value::as_str),
                    Some("Completed")
                );
                assert_eq!(
                    receipt.get("summary").and_then(Value::as_str),
                    Some("child finding")
                );
                answer("parent integrated finding")
            }
            _ => panic!("unexpected parent request"),
        };
        self.step += 1;
        turn(response, request.wire)
    }
}

#[test]
fn native_parent_spawns_collects_and_integrates_through_original_call_ids() {
    let mut parent = parent();
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(vec![answer("child finding")], Arc::new(AtomicUsize::new(0))),
    );
    let view = scheduler.view();
    let result = run_parent(
        &mut parent,
        &mut ParentModel { step: 0, id: None },
        &mut NoTools,
        Limits::default(),
        &mut scheduler,
        &mut |_| {},
    );
    assert_eq!(
        result.report.reason,
        StopReason::Finished(FinishReason::Stop)
    );
    assert_eq!(result.children.len(), 1);
    assert_eq!(result.children[0].summary, "child finding");
    assert!(
        parent
            .messages()
            .last()
            .unwrap()
            .serialized_json()
            .contains("parent integrated finding")
    );
    assert_eq!(
        parent
            .messages()
            .iter()
            .filter(|m| m.role() == "assistant")
            .count(),
        3
    );
    assert!(ui::render(&view.snapshots(), 90, 5).contains("collected"));
}

#[test]
fn ui_rows_are_bounded_and_do_not_contain_task_or_workspace_content() {
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(
            vec![answer("private result")],
            Arc::new(AtomicUsize::new(0)),
        ),
    );
    let id = scheduler.spawn(Task::read_only("private task")).unwrap();
    finished(&scheduler, id);
    for width in [0, 1, 3, 18, 80, 1000] {
        for rows in [0, 1, 2, 10] {
            let text = ui::render(&scheduler.view().snapshots(), width, rows);
            assert!(text.lines().count() <= rows);
            assert!(text.lines().all(|line| line.len() <= width.min(240)));
            assert!(!text.contains("private"));
        }
    }
}

#[test]
fn short_child_panel_keeps_running_tasks_visible_ahead_of_old_results() {
    let snapshots: Vec<_> = (1..=8)
        .map(|sequence| TaskSnapshot {
            id: TaskId {
                scheduler: 1,
                sequence,
            },
            state: if sequence == 8 {
                TaskState::Running
            } else {
                TaskState::Collected(Outcome::Completed)
            },
            stop_requested: None,
            writable_files: 0,
            workspace_write_lease: false,
            elapsed: Duration::ZERO,
            tokens_reserved: 0,
            model_turns: 0,
            usage: ChildUsage::default(),
        })
        .collect();
    let text = ui::render(&snapshots, 80, 3);
    assert!(text.contains("child-1-8  running"));
    assert!(text.contains("+7 more"));
    assert_eq!(text.lines().count(), 3);
}

#[test]
fn summary_is_bounded_on_utf8_boundary_with_full_transcript_available() {
    let content = "🦀".repeat(2000);
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(vec![answer(&content)], Arc::new(AtomicUsize::new(0))),
    );
    let id = scheduler.spawn(Task::read_only("long answer")).unwrap();
    finished(&scheduler, id);
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.summary.len(), MAX_SUMMARY_BYTES);
    assert!(result.summary_truncated);
    assert_eq!(
        result
            .conversation
            .unwrap()
            .messages()
            .last()
            .unwrap()
            .value()
            .get("content")
            .and_then(Value::as_str),
        Some(content.as_str())
    );
}

#[test]
fn uncertain_child_collection_stops_parent_batch_and_retains_started_write() {
    let factory: Arc<Factory> = Arc::new(|_, _| {
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(Replies {
                replies: vec![proposal(
                    "write",
                    "workspace_write",
                    val(r#"{"path":"src/a.rs"}"#),
                )]
                .into(),
                dispatches: Arc::new(AtomicUsize::new(0)),
                prompt_tokens: 1,
            }),
            tools: Box::new(NoTools),
        })
    });
    let mut scheduler = scheduler(write_config(), Arc::new(AtomicBool::new(false)), factory);
    let id = scheduler.spawn(write_task("a")).unwrap();
    assert_eq!(
        finished(&scheduler, id).state,
        TaskState::Finished(Outcome::Panicked)
    );
    struct Batch(Value);
    impl Model for Batch {
        fn complete(
            &mut self,
            r: ModelRequest<'_>,
            _: &AtomicBool,
            _: &mut dyn FnMut(&str) -> io::Result<()>,
        ) -> io::Result<AssistantTurn> {
            turn(self.0.clone(), r.wire)
        }
    }
    let mut batch = proposal(
        "collect",
        "child_wait",
        Value::Object(vec![("task_id".into(), Value::String(id.to_string()))]),
    );
    let write = proposal(
        "later-write",
        "workspace_write",
        val(r#"{"path":"src/a.rs"}"#),
    );
    if let Value::Object(fields) = &mut batch {
        if let Value::Array(calls) = &mut fields
            .iter_mut()
            .find(|(k, _)| k == "tool_calls")
            .unwrap()
            .1
        {
            calls.push(write.get("tool_calls").unwrap().as_array().unwrap()[0].clone());
        }
    }
    let mut parent = parent();
    let mut run = run_parent(
        &mut parent,
        &mut Batch(batch),
        &mut NoTools,
        Limits::default(),
        &mut scheduler,
        &mut |_| {},
    );
    assert_eq!(run.report.reason, StopReason::ToolFailed);
    assert_eq!(run.report.tool_calls, 1);
    assert!(
        parent
            .messages()
            .last()
            .unwrap()
            .serialized_json()
            .contains("not_executed")
    );
    let result = run.children.remove(0);
    assert!(result.uncertain_writes());
    assert_eq!(result.tools[0].status, ToolStatus::Started);
}

#[path = "http_tests.rs"]
mod http;
#[path = "quota_tests.rs"]
mod quota;
#[path = "scope_tests.rs"]
mod scope;
#[path = "lifecycle_tests.rs"]
mod lifecycle;

#[test]
fn child_is_not_created_if_parent_was_already_cancelled() {
    let backend: Arc<Factory> = Arc::new(|_, _| panic!("factory must not be called"));
    let mut scheduler = scheduler(Config::default(), Arc::new(AtomicBool::new(true)), backend);
    assert_eq!(
        scheduler.spawn(Task::read_only("late")),
        Err(ScheduleError::Closed)
    );
}

#[test]
fn setup_failure_has_a_terminal_outcome_and_releases_concurrent_capacity() {
    let backend: Arc<Factory> = Arc::new(|_, _| {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "private error must not escape",
        ))
    });
    let mut scheduler = scheduler(
        Config {
            max_concurrent: 1,
            ..Config::default()
        },
        Arc::new(AtomicBool::new(false)),
        backend,
    );
    for _ in 0..2 {
        let id = scheduler.spawn(Task::read_only("setup")).unwrap();
        assert_eq!(
            finished(&scheduler, id).state,
            TaskState::Finished(Outcome::SetupFailed(io::ErrorKind::PermissionDenied))
        );
        let result = scheduler.take_result(id).unwrap();
        assert!(result.summary.is_empty());
        assert!(result.conversation.is_some());
    }
}

struct ParentExit {
    entered: mpsc::Receiver<()>,
    step: usize,
    outcome: usize,
}
impl Model for ParentExit {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        _: &AtomicBool,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        if self.step == 0 {
            self.step += 1;
            return turn(
                proposal(
                    "spawn",
                    "child_spawn",
                    val(r#"{"task":"background child"}"#),
                ),
                request.wire,
            );
        }
        self.entered.recv_timeout(Duration::from_secs(2)).unwrap();
        match self.outcome {
            0 => turn(answer("parent finished"), request.wire),
            1 => Err(io::ErrorKind::ConnectionReset.into()),
            _ => panic!("mock parent panic"),
        }
    }
}

#[test]
fn parent_finish_failure_and_unwind_all_join_uncollected_children() {
    for outcome in 0..3 {
        let (tx, rx) = mpsc::sync_channel(1);
        let live = Arc::new(AtomicUsize::new(0));
        let mut scheduler = scheduler(
            Config::default(),
            Arc::new(AtomicBool::new(false)),
            blocking_factory(tx, Arc::clone(&live)),
        );
        let mut parent = parent();
        let mut model = ParentExit {
            entered: rx,
            step: 0,
            outcome,
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            run_parent(
                &mut parent,
                &mut model,
                &mut NoTools,
                Limits::default(),
                &mut scheduler,
                &mut |_| {},
            )
        }));
        assert_eq!(live.load(Ordering::SeqCst), 0);
        assert_eq!(
            scheduler.spawn(Task::read_only("after parent")),
            Err(ScheduleError::Closed)
        );
        let snapshots = scheduler.view().snapshots();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(
            snapshots[0].stop_requested,
            Some(StopRequest::ParentFinished)
        );
        if outcome < 2 {
            let result = result.unwrap();
            assert_eq!(result.children.len(), 1);
            assert_eq!(result.children[0].outcome, Outcome::Cancelled);
            // No background child transcript was injected when the parent ended.
            assert_eq!(
                parent
                    .messages()
                    .iter()
                    .filter(|m| m.role() == "tool")
                    .count(),
                1
            );
        } else {
            assert!(result.is_err());
            assert_eq!(
                scheduler.take_result(snapshots[0].id).unwrap().outcome,
                Outcome::Cancelled
            );
        }
    }
}

struct GatedAnswer {
    id: TaskId,
    entered: mpsc::SyncSender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
}
impl ChildModel for GatedAnswer {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
        meter: &mut TokenMeter,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        meter.reserve(10, meter.output)?;
        self.entered.send(()).unwrap();
        let mut released = lock(&self.release.0);
        while !*released && !cancel.load(Ordering::Acquire) {
            released = self
                .release
                .1
                .wait_timeout(released, Duration::from_millis(10))
                .unwrap()
                .0;
        }
        if cancel.load(Ordering::Acquire) {
            return Err(io::ErrorKind::Interrupted.into());
        }
        turn(answer(&self.id.to_string()), request.wire)
    }
}

#[test]
fn simultaneous_completions_keep_distinct_histories_and_live_reservation_counts() {
    let (tx, rx) = mpsc::sync_channel(2);
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let child_release = Arc::clone(&release);
    let backend: Arc<Factory> = Arc::new(move |context, _| {
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(GatedAnswer {
                id: context.id,
                entered: tx.clone(),
                release: Arc::clone(&child_release),
            }),
            tools: Box::new(NoTools),
        })
    });
    let mut scheduler = scheduler(Config::default(), Arc::new(AtomicBool::new(false)), backend);
    let a = scheduler.spawn(Task::read_only("first task")).unwrap();
    let b = scheduler.spawn(Task::read_only("second task")).unwrap();
    for _ in 0..2 {
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }
    assert!(
        scheduler
            .view()
            .snapshots()
            .iter()
            .all(|s| s.tokens_reserved == 1034 && s.model_turns == 1)
    );
    *lock(&release.0) = true;
    release.1.notify_all();
    for (id, own, other) in [
        (a, "first task", "second task"),
        (b, "second task", "first task"),
    ] {
        assert_eq!(
            finished(&scheduler, id).state,
            TaskState::Finished(Outcome::Completed)
        );
        let result = scheduler.take_result(id).unwrap();
        assert_eq!(result.summary, id.to_string());
        let transcript = result
            .conversation
            .unwrap()
            .request(WireLimits::default())
            .to_json()
            .unwrap();
        assert!(transcript.contains(own));
        assert!(!transcript.contains(other));
    }
}

#[test]
fn cancelling_a_completed_child_does_not_relabel_its_real_outcome() {
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(vec![answer("done")], Arc::new(AtomicUsize::new(0))),
    );
    let id = scheduler.spawn(Task::read_only("done")).unwrap();
    finished(&scheduler, id);
    scheduler.cancel(id).unwrap();
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.outcome, Outcome::Completed);
    assert_eq!(result.stop_requested, None);
}

#[test]
fn bounded_parent_receipt_failure_keeps_full_child_result_for_parent_cleanup() {
    let mut parent = parent();
    let content = "full child finding ".repeat(100);
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(vec![answer(&content)], Arc::new(AtomicUsize::new(0))),
    );
    let result = run_parent(
        &mut parent,
        &mut ParentModel { step: 0, id: None },
        &mut NoTools,
        Limits {
            max_tool_output_bytes: 256,
            ..Limits::default()
        },
        &mut scheduler,
        &mut |_| {},
    );
    assert_eq!(
        result.report.reason,
        StopReason::Limit(agent::LimitKind::ToolOutput)
    );
    assert_eq!(result.children.len(), 1);
    assert_eq!(result.children[0].summary, content);
    assert_eq!(result.children[0].outcome, Outcome::Completed);
    assert!(
        !parent
            .request(WireLimits::default())
            .to_json()
            .unwrap()
            .contains(&content)
    );
}

#[test]
fn child_preserves_parent_output_cap_dialect() {
    let mut request =
        json::parse(&parent().request(WireLimits::default()).to_json().unwrap()).unwrap();
    let Value::Object(fields) = &mut request else {
        unreachable!()
    };
    fields.push(("max_completion_tokens".into(), Value::Number("300".into())));
    let parent = Conversation::from_request(request, Limits::default()).unwrap();
    let prefix = StablePrefix::from_parent(&parent).unwrap();
    let child = prefix
        .conversation(
            &Task::read_only("dialect"),
            &Scope {
                tools: vec![],
                write_paths: vec![],
            },
            Limits::default(),
        )
        .unwrap();
    assert_eq!(
        child.options().get("max_completion_tokens"),
        Some(&Value::Number("1024".into()))
    );
    assert!(child.options().get("max_tokens").is_none());
}

#[cfg(target_os = "macos")]
#[test]
fn builtin_factory_retains_workspace_read_boundary_and_never_adds_writes() {
    use std::os::unix::fs::symlink;
    struct Temporary(std::path::PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);
    let base = std::env::temp_dir().join(format!(
        "vade-child-workspace-{}-{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&base).unwrap();
    let cleanup = Temporary(base);
    let root = cleanup.0.join("workspace");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("inside.txt"), "allowed mock content").unwrap();
    std::fs::write(cleanup.0.join("outside.txt"), "mock content outside root").unwrap();
    symlink(cleanup.0.join("outside.txt"), root.join("alias.txt")).unwrap();
    std::fs::hard_link(cleanup.0.join("outside.txt"), root.join("hard.txt")).unwrap();
    let workspace = Arc::new(crate::tools::Workspace::new(&root).unwrap());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    let factory = read_only_http_factory(&endpoint, workspace).unwrap();
    let context = ChildContext {
        id: TaskId {
            scheduler: 0,
            sequence: 1,
        },
        scope: Scope {
            tools: vec!["workspace_read".into(), "workspace_list".into()],
            write_paths: vec![],
        },
        quota: Quota::default(),
    };
    let cancel = AtomicBool::new(false);
    let mut worker = factory(&context, &cancel).unwrap();
    // No model call or network request is made by constructing this factory.
    for (path, allowed) in [
        ("inside.txt", true),
        ("alias.txt", false),
        ("hard.txt", false),
        ("../outside.txt", false),
    ] {
        let call = ToolCall {
            id: "read".into(),
            name: "workspace_read".into(),
            arguments: "{}".into(),
        };
        let arguments = Value::Object(vec![("path".into(), Value::String(path.into()))]);
        let result = worker.tools.execute(
            ToolInvocation {
                call: &call,
                arguments: &arguments,
                max_output_bytes: 1024,
            },
            &cancel,
        );
        if allowed {
            assert_eq!(result, Ok("allowed mock content".into()));
        } else {
            assert!(result.unwrap().starts_with("Read refused:"));
        }
    }
    let call = ToolCall {
        id: "write".into(),
        name: "workspace_write".into(),
        arguments: "{}".into(),
    };
    let arguments = val(r#"{"path":"inside.txt"}"#);
    assert_eq!(
        worker.tools.execute(
            ToolInvocation {
                call: &call,
                arguments: &arguments,
                max_output_bytes: 1024
            },
            &cancel
        ),
        Err(ToolError::UnknownTool)
    );
    assert_eq!(
        std::fs::read_to_string(root.join("inside.txt")).unwrap(),
        "allowed mock content"
    );
}

struct UnknownFinish;
impl ChildModel for UnknownFinish {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        _: &AtomicBool,
        meter: &mut TokenMeter,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        meter.reserve(1, meter.output)?;
        AssistantTurn::from_response(
            &val(
                r#"{"choices":[{"message":{"role":"assistant","content":"incomplete"},"finish_reason":"private-provider-finish-text"}]}"#,
            ),
            request.wire,
        )
    }
}

#[test]
fn opaque_finish_text_never_becomes_lifecycle_or_parent_summary_content() {
    let backend: Arc<Factory> = Arc::new(|_, _| {
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(UnknownFinish),
            tools: Box::new(NoTools),
        })
    });
    let mut scheduler = scheduler(Config::default(), Arc::new(AtomicBool::new(false)), backend);
    let id = scheduler.spawn(Task::read_only("unknown finish")).unwrap();
    let snapshot = finished(&scheduler, id);
    assert_eq!(snapshot.state, TaskState::Finished(Outcome::UnknownFinish));
    assert!(!format!("{snapshot:?}").contains("private-provider"));
    let result = scheduler.take_result(id).unwrap();
    assert!(result.summary.is_empty());
    assert!(matches!(
        result.report.unwrap().reason,
        StopReason::Finished(FinishReason::Other(_))
    ));
}

#[path = "native_tests.rs"]
mod native_tools_tests;
