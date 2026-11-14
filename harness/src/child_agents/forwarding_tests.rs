//! The parent and child wrappers hand every state query to the executor they
//! wrap. A query a wrapper drops gets the trait default, which reports no
//! check: `--check` would then be skipped without a word.
use super::*;
use crate::agent::{Model, ModelRequest};

/// Answers every state query with a value that no default body returns.
struct Probe;
impl ToolExecutor for Probe {
    fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        Ok("status: Exited(Some(0))".into())
    }
    fn last_side_effect_uncertain(&self) -> bool { true }
    fn take_pending_check(&mut self) -> Option<Value> { Some(Value::String("pending".into())) }
    fn take_completion_check(&mut self) -> Option<Value> { Some(Value::String("completion".into())) }
    fn check_failing(&self) -> bool { true }
    fn check_enabled(&self) -> bool { true }
    fn check_not_run(&self) -> bool { true }
}

/// A configured check that passes: offered once at completion, recorded when run.
#[derive(Default)]
struct PassingCheck {
    offered: bool,
    executed: Vec<String>,
}
impl ToolExecutor for PassingCheck {
    fn execute(&mut self, call: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        self.executed.push(call.call.name.clone());
        Ok("status: Exited(Some(0))".into())
    }
    fn take_completion_check(&mut self) -> Option<Value> {
        (!std::mem::replace(&mut self.offered, true)).then(|| json::parse(r#"{"argv":["true"]}"#).unwrap())
    }
    fn check_enabled(&self) -> bool { true }
}

/// Answers "done" every turn.
struct Done;
impl Model for Done {
    fn complete(&mut self, request: ModelRequest<'_>, _: &AtomicBool, _: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<AssistantTurn> {
        AssistantTurn::from_response(
            &json::parse(r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}"#)?,
            request.wire,
        )
    }
}

fn conversation() -> Conversation {
    let mut tools = vec![json::parse(r#"{"type":"function","function":{"name":"workspace_shell","parameters":{"type":"object"}}}"#).unwrap()];
    let Value::Array(children) = definitions() else { unreachable!() };
    tools.extend(children);
    Conversation::from_request(
        Value::Object(vec![
            ("model".into(), Value::String("mock".into())),
            ("tools".into(), Value::Array(tools)),
            ("messages".into(), json::parse(r#"[{"role":"user","content":"say done"}]"#).unwrap()),
        ]),
        Limits::default(),
    )
    .unwrap()
}

fn scheduler() -> Scheduler {
    let factory: Arc<Factory> = Arc::new(|_: &ChildContext, _: &AtomicBool| -> io::Result<ChildWorker> {
        Err(io::Error::other("this test starts no child"))
    });
    Scheduler::new(
        StablePrefix::from_parent(&conversation()).unwrap(),
        Config { policy: ToolPolicy::native_with_web(false), ..Config::default() },
        Arc::new(AtomicBool::new(false)),
        factory,
    )
    .unwrap()
}

/// The parent path the command line takes with `--children`.
#[test]
fn parent_run_takes_the_completion_check() {
    let mut conversation = conversation();
    let mut scheduler = scheduler();
    let mut tools = PassingCheck::default();
    let run = run_parent(&mut conversation, &mut Done, &mut tools, Limits::default(), &mut scheduler, &mut |_| {});
    assert_eq!(run.report.reason, StopReason::Finished(FinishReason::Stop));
    assert_eq!(run.report.auto_checks, 1, "the completion check was not taken");
    assert_eq!(tools.executed, ["workspace_shell"]);
}

#[test]
fn parent_tools_forward_every_state_query() {
    let mut scheduler = scheduler();
    let mut probe = Probe;
    let mut collected = Vec::new();
    let mut tools = ParentTools { scheduler: &mut scheduler, inner: &mut probe, collected: &mut collected, uncertain: false };
    // `last_side_effect_uncertain` is ParentTools' own: it covers child_* calls too.
    assert_eq!(tools.take_pending_check(), Some(Value::String("pending".into())));
    assert_eq!(tools.take_completion_check(), Some(Value::String("completion".into())));
    assert!(tools.check_failing());
    assert!(tools.check_enabled());
    assert!(tools.check_not_run());
}

#[test]
fn scoped_child_tools_forward_every_state_query() {
    let mut probe = Probe;
    let policy = ToolPolicy::native_with_web(false);
    let scope = Scope { tools: Vec::new(), write_paths: Vec::new() };
    let mut receipts = Vec::new();
    let mut tools = ScopedTools { inner: &mut probe, policy: &policy, scope: &scope, receipts: &mut receipts };
    // `last_side_effect_uncertain` is ScopedTools' own: it reads the child's receipts.
    assert_eq!(tools.take_pending_check(), Some(Value::String("pending".into())));
    assert_eq!(tools.take_completion_check(), Some(Value::String("completion".into())));
    assert!(tools.check_failing());
    assert!(tools.check_enabled());
    assert!(tools.check_not_run());
}

impl crate::child_agents::ChildModel for Done {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        cancel: &AtomicBool,
        meter: &mut crate::child_agents::TokenMeter,
        on_text: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        meter.reserve(10, meter.output)?;
        Model::complete(self, request, cancel, on_text)
    }
}

/// The child path wraps a worker's executor in `ScopedTools`.
#[test]
fn child_run_takes_the_completion_check() {
    use crate::child_agents::{Config, Outcome, Task, TaskState};
    use std::collections::BTreeMap;
    let factory: Arc<Factory> = Arc::new(|_, _| {
        Ok(ChildWorker {
            model: Box::new(Done),
            tools: Box::new(PassingCheck::default()),
            observer: None,
            journal_name: None,
        })
    });
    let mut grants = BTreeMap::new();
    grants.insert(
        "shell".into(),
        Scope { tools: vec!["workspace_shell".into()], write_paths: Vec::new() },
    );
    let mut scheduler = Scheduler::new(
        StablePrefix::from_parent(&conversation()).unwrap(),
        Config { policy: ToolPolicy::native_with_web(false), grants, ..Config::default() },
        Arc::new(AtomicBool::new(false)),
        factory,
    )
    .unwrap();
    let id = scheduler
        .spawn(Task {
            goal: "say done".into(),
            grant: Some("shell".into()),
            quota: scheduler.default_quota(),
        })
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let snap = scheduler.wait(id, Duration::from_millis(50)).unwrap();
        if matches!(snap.state, TaskState::Finished(_)) {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "child timed out");
    }
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.outcome, Outcome::Completed);
    assert_eq!(result.tools.len(), 1, "child completion check was not taken");
    assert_eq!(result.tools[0].tool, "workspace_shell");
}
