//! Child resource allocation and tool quota enforcement.
use super::*;

#[test]
fn each_global_resource_allocation_is_enforced_before_start() {
    for field in 0..5 {
        let quota = Quota::default();
        let mut config = Config::default();
        match field {
            0 => config.total_model_turns = quota.model_turns - 1,
            1 => config.total_tool_calls = quota.tool_calls - 1,
            2 => config.total_tokens = quota.tokens - 1,
            3 => config.total_history_bytes = quota.history_bytes - 1,
            4 => config.total_tool_output_bytes = quota.tool_output_bytes - 1,
            _ => unreachable!(),
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut scheduler = scheduler(
            config,
            Arc::new(AtomicBool::new(false)),
            factory(vec![], Arc::clone(&calls)),
        );
        assert_eq!(
            scheduler.spawn(Task::read_only("over allocation")),
            Err(ScheduleError::Quota)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(scheduler.view().snapshots().is_empty());
    }
}

#[test]
fn zero_tool_quota_rejects_proposal_without_executing() {
    let backend: Arc<Factory> = Arc::new(|_, _| {
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(Replies {
                replies: vec![proposal("read", "workspace_read", val(r#"{"path":"x"}"#))].into(),
                dispatches: Arc::new(AtomicUsize::new(0)),
                prompt_tokens: 1,
            }),
            tools: Box::new(NoTools),
        })
    });
    let mut scheduler = scheduler(Config::default(), Arc::new(AtomicBool::new(false)), backend);
    let mut task = Task::read_only("no tools");
    task.quota.tool_calls = 0;
    let id = scheduler.spawn(task).unwrap();
    assert_eq!(
        finished(&scheduler, id).state,
        TaskState::Finished(Outcome::Runtime(StopReason::Limit(
            agent::LimitKind::ToolCalls
        )))
    );
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.report.unwrap().tool_calls, 0);
    assert!(result.tools.is_empty());
}

#[test]
fn defaults_support_8_concurrent_128k_subagents_without_arbitrary_upper_caps() {
    let config = Config::default();
    assert_eq!(config.max_concurrent, 8);
    assert_eq!(config.default_quota.tokens, 131_072);
    assert!(config.total_tokens >= 8 * 131_072);
    assert_eq!(config.default_quota.limits().max_context_tokens, 131_072);

    // Rule 17: higher concurrency, turns, history and context are accepted cleanly.
    let wide = Config {
        max_concurrent: 16,
        max_tasks: 64,
        default_quota: Quota {
            model_turns: 120,
            tool_calls: 512,
            tokens: 262_144,
            history_bytes: 6 * 1024 * 1024,
            ..Quota::default()
        },
        total_tokens: 64 * 262_144,
        ..Config::default()
    };
    let scheduler = Scheduler::new(
        StablePrefix::from_parent(&parent()).unwrap(),
        wide,
        Arc::new(AtomicBool::new(false)),
        factory(vec![], Arc::new(AtomicUsize::new(0))),
    );
    assert!(scheduler.is_ok(), "wide subagent limits must be accepted");

    // Invariant: max_concurrent must remain in 1..=max_tasks.
    let defaults = Config::default();
    let bad = Config { max_concurrent: defaults.max_tasks + 1, ..defaults };
    assert!(Scheduler::new(
        StablePrefix::from_parent(&parent()).unwrap(),
        bad,
        Arc::new(AtomicBool::new(false)),
        factory(vec![], Arc::new(AtomicUsize::new(0))),
    ).is_err());
}
