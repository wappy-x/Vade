//! Running a parent with children: the child_* tool schemas, their dispatch
//! and joining uncollected children.
use super::*;
use crate::json::{self, Value};

/// Append these definitions ONCE when the parent conversation is constructed,
/// before freezing StablePrefix. Children keep identical schemas but cannot
/// invoke these tools. Advertised schema is not execution authority.
pub fn definitions() -> Value {
    json::parse(r#"[
        {"type":"function","function":{"name":"child_spawn","description":"Start a bounded child task. Read-only by default; grant selects an existing host-authorized scope. No recursive children. Collect its result before integrating.","parameters":{"type":"object","properties":{"task":{"type":"string"},"grant":{"type":"string"},"model_turns":{"type":"integer"},"tokens":{"type":"integer"},"wall_ms":{"type":"integer"}},"required":["task"],"additionalProperties":false}}},
        {"type":"function","function":{"name":"child_list","description":"List child task IDs and actual lifecycle states.","parameters":{"type":"object","properties":{},"additionalProperties":false}}},
        {"type":"function","function":{"name":"child_wait","description":"Wait up to one second for a child, then collect its result once if finished. A running state is not a result. Parent owns integration.","parameters":{"type":"object","properties":{"task_id":{"type":"string"},"timeout_ms":{"type":"integer"}},"required":["task_id"],"additionalProperties":false}}},
        {"type":"function","function":{"name":"child_cancel","description":"Request child cancellation; wait for the real outcome. A request does not confirm remote compute stopped.","parameters":{"type":"object","properties":{"task_id":{"type":"string"}},"required":["task_id"],"additionalProperties":false}}}
    ]"#).expect("static child tool definitions")
}

/// A parent run's report together with every child result it produced.
pub struct ParentRun {
    pub report: RunReport,
    /// Authoritative child transcripts/receipts, owned only by the parent. They
    /// are returned even if the parent fails, cancels, or never asks to collect.
    /// Nothing is automatically appended to the parent's Conversation.
    pub children: Vec<TaskResult>,
}

/// One native parent activation. Scheduler tools return bounded receipts through
/// the original tool-call IDs; the ordinary runtime alone commits parent history.
/// Returning/aborting always cancels and joins outstanding children.
pub fn run_parent(
    conversation: &mut Conversation,
    model: &mut dyn Model,
    tools: &mut dyn ToolExecutor,
    limits: Limits,
    scheduler: &mut Scheduler,
    on_text: &mut dyn FnMut(&str),
) -> ParentRun {
    struct NoObserver;
    impl agent::RunObserver for NoObserver {
        fn checkpoint(
            &mut self,
            _: &Conversation,
            _: &[crate::agent_wire::Message],
        ) -> io::Result<()> {
            Ok(())
        }
    }
    run_parent_observed(
        conversation,
        model,
        tools,
        limits,
        scheduler,
        on_text,
        &mut NoObserver,
    )
}

/// The journal observer remains active for every parent tool, including scheduler calls.
pub fn run_parent_observed(
    conversation: &mut Conversation,
    model: &mut dyn Model,
    tools: &mut dyn ToolExecutor,
    limits: Limits,
    scheduler: &mut Scheduler,
    on_text: &mut dyn FnMut(&str),
    observer: &mut dyn agent::RunObserver,
) -> ParentRun {
    run_parent_observed_with_budget(
        conversation,
        model,
        tools,
        limits,
        scheduler,
        on_text,
        observer,
        &agent::RunBudget::new(limits.max_run_time),
    )
}

/// Share the parent's active-work budget with approval callbacks. Expiration
/// requests cancellation through the scheduler's parent flag, then joins and
/// collects every child before returning the retained parent history.
pub fn run_parent_observed_with_budget(
    conversation: &mut Conversation,
    model: &mut dyn Model,
    tools: &mut dyn ToolExecutor,
    limits: Limits,
    scheduler: &mut Scheduler,
    on_text: &mut dyn FnMut(&str),
    observer: &mut dyn agent::RunObserver,
    budget: &agent::RunBudget,
) -> ParentRun {
    let cancel = Arc::clone(&scheduler.shared.parent_cancel);
    let mut children = Vec::new();
    let report = {
        let mut executor = ParentTools {
            scheduler,
            inner: tools,
            collected: &mut children,
            uncertain: false,
        };
        agent::run_observed_with_budget(
            conversation,
            model,
            &mut executor,
            limits,
            &cancel,
            on_text,
            observer,
            budget,
        )
    };
    for snapshot in scheduler.view().snapshots() {
        if let Ok(result) = scheduler.take_result(snapshot.id) {
            children.push(result);
        }
    }
    ParentRun { report, children }
}

struct ParentTools<'a> {
    scheduler: &'a mut Scheduler,
    inner: &'a mut dyn ToolExecutor,
    collected: &'a mut Vec<TaskResult>,
    uncertain: bool,
}

// Also runs while unwinding through a parent model/tool panic. The caller may
// retain its Scheduler, but a failed activation cannot leave children running.
impl Drop for ParentTools<'_> {
    fn drop(&mut self) {
        self.scheduler.stop_all(
            if self.scheduler.shared.parent_cancel.load(Ordering::Acquire) {
                StopRequest::ParentCancelled
            } else {
                StopRequest::ParentFinished
            },
        );
        self.scheduler.shutdown();
    }
}

impl ToolExecutor for ParentTools<'_> {
    fn last_side_effect_uncertain(&self) -> bool {
        self.uncertain
    }
    fn take_pending_check(&mut self) -> Option<crate::json::Value> {
        self.inner.take_pending_check()
    }
    fn check_failing(&self) -> bool {
        self.inner.check_failing()
    }
    fn check_enabled(&self) -> bool {
        self.inner.check_enabled()
    }
    fn check_not_run(&self) -> bool {
        self.inner.check_not_run()
    }
    fn execute(
        &mut self,
        call: ToolInvocation<'_>,
        cancel: &AtomicBool,
    ) -> Result<String, ToolError> {
        self.uncertain = false;
        if cancel.load(Ordering::Acquire) {
            return Err(ToolError::Cancelled);
        }
        let output = match call.call.name.as_str() {
            "child_spawn" => {
                fields(
                    call.arguments,
                    &["task", "grant", "model_turns", "tokens", "wall_ms"],
                )?;
                let goal = string(call.arguments, "task")?.to_owned();
                let grant = call
                    .arguments
                    .get("grant")
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or(ToolError::InvalidArguments("grant must be a string"))
                    })
                    .transpose()?;
                let mut quota = self.scheduler.default_quota();
                quota.model_turns =
                    smaller(call.arguments, "model_turns", quota.model_turns as u64)? as usize;
                quota.tokens = smaller(call.arguments, "tokens", quota.tokens)?;
                quota.wall_time = Duration::from_millis(smaller(
                    call.arguments,
                    "wall_ms",
                    quota.wall_time.as_millis() as u64,
                )?);
                // Reserve sufficient output space before spawning side effects.
                if call.max_output_bytes < 256 {
                    return Err(ToolError::OutputLimit);
                }
                match self.scheduler.spawn(Task { goal, grant, quota }) {
                    Ok(id) => encode(Value::Object(vec![
                        ("task_id".into(), Value::String(id.to_string())),
                        ("state".into(), Value::String("running".into())),
                    ]))?,
                    Err(error) => schedule_error(error),
                }
            }
            "child_list" => {
                fields(call.arguments, &[])?;
                encode(Value::Array(
                    self.scheduler
                        .view()
                        .snapshots()
                        .iter()
                        .map(snapshot_value)
                        .collect(),
                ))?
            }
            "child_wait" => {
                fields(call.arguments, &["task_id", "timeout_ms"])?;
                let id = self.scheduler.find_id(string(call.arguments, "task_id")?)?;
                let timeout = Duration::from_millis(smaller(call.arguments, "timeout_ms", 1000)?);
                let snapshot = self
                    .scheduler
                    .wait(id, timeout)
                    .map_err(|_| ToolError::InvalidArguments("task_id must name an available child to wait for"))?;
                if matches!(snapshot.state, TaskState::Finished(_)) {
                    // Check actual encoded length before consuming the sole
                    // result. Failed/oversized retrieval leaves ownership intact.
                    let encoded = {
                        let state = lock(&self.scheduler.shared.state);
                        let result = state
                            .entries
                            .get(&id)
                            .and_then(|e| e.result.as_ref())
                            .ok_or(ToolError::Failed)?;
                        encode(result_value(result))?
                    };
                    if encoded.len() > call.max_output_bytes {
                        return Err(ToolError::OutputLimit);
                    }
                    let result = self
                        .scheduler
                        .take_result(id)
                        .map_err(|_| ToolError::Failed)?;
                    // Collection releases the child's lease. Its uncertainty must
                    // still stop this already-generated parent batch immediately.
                    self.uncertain = result.uncertain_writes();
                    self.collected.push(result);
                    encoded
                } else {
                    encode(snapshot_value(&snapshot))?
                }
            }
            "child_cancel" => {
                fields(call.arguments, &["task_id"])?;
                let id = self.scheduler.find_id(string(call.arguments, "task_id")?)?;
                if call.max_output_bytes < 512 {
                    return Err(ToolError::OutputLimit);
                }
                self.scheduler
                    .cancel(id)
                    .map_err(|_| ToolError::InvalidArguments("task_id must name a cancellable child"))?;
                let snapshot = self
                    .scheduler
                    .wait(id, Duration::ZERO)
                    .map_err(|_| ToolError::InvalidArguments("task_id must name an available child"))?;
                let Value::Object(mut fields) = snapshot_value(&snapshot) else {
                    unreachable!()
                };
                fields.push((
                    "cancellation_requested".into(),
                    Value::Bool(snapshot.stop_requested.is_some()),
                ));
                fields.push(("remote_stop_confirmed".into(), Value::Bool(false)));
                encode(Value::Object(fields))?
            }
            _ => {
                self.scheduler.check_parent(&call)?;
                let result = self.inner.execute(call, cancel);
                self.uncertain = self.inner.last_side_effect_uncertain();
                return result;
            }
        };
        if output.len() > call.max_output_bytes {
            Err(ToolError::OutputLimit)
        } else {
            Ok(output)
        }
    }
}

fn fields(value: &Value, allowed: &[&str]) -> Result<(), ToolError> {
    match value {
        Value::Object(fields) if fields.iter().all(|(k, _)| allowed.contains(&k.as_str())) => {
            Ok(())
        }
        _ => Err(ToolError::InvalidArguments("child tool arguments must use only the declared fields")),
    }
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, ToolError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(ToolError::InvalidArguments(if key == "task" { "task must be a string" } else { "task_id must be a string" }))
}
fn smaller(value: &Value, key: &str, cap: u64) -> Result<u64, ToolError> {
    let error = ToolError::InvalidArguments(match key {
        "model_turns" => "model_turns must be a nonnegative integer within the granted cap",
        "tokens" => "tokens must be a nonnegative integer within the granted cap",
        "wall_ms" => "wall_ms must be a nonnegative integer within the granted cap",
        _ => "timeout_ms must be a nonnegative integer within the granted cap",
    });
    match value.get(key) {
        None => Ok(cap),
        Some(Value::Number(n)) => n
            .parse::<u64>()
            .ok()
            .filter(|n| *n <= cap)
            .ok_or(error),
        _ => Err(error),
    }
}
fn encode(value: Value) -> Result<String, ToolError> {
    json::stringify(&value).map_err(|_| ToolError::OutputLimit)
}
fn schedule_error(error: ScheduleError) -> String {
    format!("{{\"ok\":false,\"error\":\"{error:?}\"}}")
}

pub(crate) fn snapshot_value(snapshot: &TaskSnapshot) -> Value {
    Value::Object(vec![
        ("task_id".into(), Value::String(snapshot.id.to_string())),
        (
            "state".into(),
            Value::String(ui::state_label(&snapshot.state).into()),
        ),
        (
            "stop_requested".into(),
            snapshot
                .stop_requested
                .map_or(Value::Null, |r| Value::String(format!("{r:?}"))),
        ),
        (
            "tokens_reserved".into(),
            Value::Number(snapshot.tokens_reserved.to_string()),
        ),
    ])
}

fn result_value(result: &TaskResult) -> Value {
    Value::Object(vec![
        ("task_id".into(), Value::String(result.id.to_string())),
        (
            "outcome".into(),
            Value::String(format!("{:?}", result.outcome)),
        ),
        (
            "stop_requested".into(),
            result
                .stop_requested
                .map_or(Value::Null, |r| Value::String(format!("{r:?}"))),
        ),
        ("summary".into(), Value::String(result.summary.clone())),
        (
            "summary_truncated".into(),
            Value::Bool(result.summary_truncated),
        ),
        (
            "tokens_reserved".into(),
            Value::Number(result.tokens_reserved.to_string()),
        ),
        (
            "uncertain_writes".into(),
            Value::Bool(result.uncertain_writes()),
        ),
        (
            "transcript_ref".into(),
            Value::String(result.id.to_string()),
        ),
        (
            "journal".into(),
            result
                .journal_name
                .as_ref()
                .map_or(Value::Null, |name| Value::String(name.clone())),
        ),
        ("integration_owner".into(), Value::String("parent".into())),
        ("remote_stop_confirmed".into(), Value::Bool(false)),
    ])
}
