//! Offline, first-party parent + children demonstration. No files, network,
//! installed agents, inference engine or GPU. All token counts are mock values.
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use vade_harness::agent::{
    Conversation, Limits, Model, ModelRequest, ToolError, ToolExecutor, ToolInvocation,
};
use vade_harness::agent_wire::{AssistantTurn, WireLimits};
use vade_harness::child_agents::{
    self, ChildModel, ChildWorker, Config, Factory, Quota, Scheduler, StablePrefix, TokenMeter,
};
use vade_harness::json::{self, Value};

fn value(text: &str) -> Value {
    json::parse(text).expect("first-party mock JSON")
}
fn turn(message: Value, wire: WireLimits) -> io::Result<AssistantTurn> {
    let finish = if message.get("tool_calls").is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    AssistantTurn::from_response(
        &Value::Object(vec![(
            "choices".into(),
            Value::Array(vec![Value::Object(vec![
                ("message".into(), message),
                ("finish_reason".into(), Value::String(finish.into())),
            ])]),
        )]),
        wire,
    )
}
fn call(id: &str, tool: &str, args: Value) -> Value {
    Value::Object(vec![
        ("id".into(), Value::String(id.into())),
        ("type".into(), Value::String("function".into())),
        (
            "function".into(),
            Value::Object(vec![
                ("name".into(), Value::String(tool.into())),
                (
                    "arguments".into(),
                    Value::String(json::stringify(&args).unwrap()),
                ),
            ]),
        ),
    ])
}
fn calls(values: Vec<Value>) -> Value {
    Value::Object(vec![
        ("role".into(), Value::String("assistant".into())),
        ("content".into(), Value::Null),
        ("tool_calls".into(), Value::Array(values)),
    ])
}

struct MockChild {
    step: usize,
    output: u64,
}
impl ChildModel for MockChild {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        _: &AtomicBool,
        meter: &mut TokenMeter,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        meter.reserve(50, self.output)?;
        let message = if self.step == 0 {
            calls(vec![call(
                "mock-read",
                "workspace_read",
                value(r#"{"path":"sample.txt"}"#),
            )])
        } else {
            value(
                r#"{"role":"assistant","content":"Mock investigation finished; parent can integrate this finding."}"#,
            )
        };
        self.step += 1;
        turn(message, request.wire)
    }
}
struct MockTools;
impl ToolExecutor for MockTools {
    fn execute(&mut self, _: ToolInvocation<'_>, cancel: &AtomicBool) -> Result<String, ToolError> {
        let until = Instant::now() + Duration::from_millis(220);
        while Instant::now() < until {
            if cancel.load(Ordering::Acquire) {
                return Err(ToolError::Cancelled);
            }
            thread::park_timeout(Duration::from_millis(5));
        }
        Ok("first-party mock file contents".into())
    }
}

struct MockParent {
    step: usize,
}
impl Model for MockParent {
    fn complete(
        &mut self,
        request: ModelRequest<'_>,
        _: &AtomicBool,
        _: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<AssistantTurn> {
        let message = match self.step {
            0 => calls(vec![
                call(
                    "spawn-ok",
                    "child_spawn",
                    value(r#"{"task":"read-only investigation"}"#),
                ),
                call(
                    "spawn-deadline",
                    "child_spawn",
                    value(r#"{"task":"deadline demonstration","wall_ms":100}"#),
                ),
                call(
                    "spawn-budget",
                    "child_spawn",
                    value(r#"{"task":"token quota demonstration","tokens":32}"#),
                ),
            ]),
            1 => {
                let mut waits = Vec::new();
                for message in request.messages.iter().filter(|m| m.role() == "tool") {
                    let text = message
                        .value()
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap();
                    let receipt = json::parse(text)?;
                    if let Some(id) = receipt.get("task_id").and_then(Value::as_str) {
                        waits.push(call(
                            &format!("wait-{}", waits.len()),
                            "child_wait",
                            Value::Object(vec![("task_id".into(), Value::String(id.into()))]),
                        ));
                    }
                }
                calls(waits)
            }
            _ => value(
                r#"{"role":"assistant","content":"Parent collected all three outcomes. Only the completed finding is available for integration."}"#,
            ),
        };
        self.step += 1;
        turn(message, request.wire)
    }
}

fn main() -> io::Result<()> {
    println!("OFFLINE MOCK: native parent + bounded children; no network or filesystem tools.");
    let Value::Array(mut tools) = child_agents::definitions() else {
        unreachable!()
    };
    tools.push(value(r#"{"type":"function","function":{"name":"workspace_read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}"#));
    let request = Value::Object(vec![
        ("model".into(), Value::String("mock".into())),
        ("tools".into(), Value::Array(tools)),
        (
            "messages".into(),
            value(
                r#"[{"role":"system","content":"Stable system, tools, and project rules. Delegate bounded tasks; parent owns integration."},{"role":"user","content":"Demonstrate child outcomes."}]"#,
            ),
        ),
    ]);
    let mut conversation = Conversation::from_request(request, Limits::default())?;
    let config = Config {
        default_quota: Quota {
            output_tokens_per_turn: 32,
            ..Quota::default()
        },
        ..Config::default()
    };
    let factory: Arc<Factory> = Arc::new(|context, _| {
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(MockChild {
                step: 0,
                output: context.quota.output_tokens_per_turn,
            }),
            tools: Box::new(MockTools),
        })
    });
    let mut scheduler = Scheduler::new(
        StablePrefix::from_parent(&conversation)?,
        config,
        Arc::new(AtomicBool::new(false)),
        factory,
    )?;
    let view = scheduler.view();
    let worker = thread::spawn(move || {
        let result = child_agents::run_parent(
            &mut conversation,
            &mut MockParent { step: 0 },
            &mut MockTools,
            Limits::default(),
            &mut scheduler,
            &mut |_| {},
        );
        (result, conversation)
    });
    let mut revision = u64::MAX;
    while !worker.is_finished() {
        let next = view.revision();
        if revision != next {
            println!("\n{}", child_agents::ui::render(&view.snapshots(), 120, 8));
            revision = next;
        }
        view.wait_for_change(revision, Duration::from_millis(100));
    }
    let (result, conversation) = worker
        .join()
        .map_err(|_| io::Error::other("mock worker failed"))?;
    println!("\n{}", child_agents::ui::render(&view.snapshots(), 120, 8));
    println!(
        "\nParent: {:?}; {} child transcripts retained by parent.",
        result.report.reason,
        result.children.len()
    );
    if let Some(text) = conversation
        .messages()
        .last()
        .and_then(|m| m.value().get("content"))
        .and_then(Value::as_str)
    {
        println!("{text}");
    }
    Ok(())
}
