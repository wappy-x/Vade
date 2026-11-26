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
