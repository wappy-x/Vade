use super::*;
use crate::{coding_tools::CodingTools, memory, shell::ApprovalPolicy, test_workspace::Fixture};

fn memory_parent() -> Conversation {
    let mut request = val(&parent().request(WireLimits::default()).to_json().unwrap());
    let Value::Object(fields) = &mut request else { unreachable!() };
    memory::add_tools(&mut fields.iter_mut().find(|(name, _)| name == "tools").unwrap().1);
    Conversation::from_request(request, Limits::default()).unwrap()
}

fn memory_scheduler(parent: &Conversation, factory: Arc<Factory>) -> Scheduler {
    Scheduler::new(StablePrefix::from_parent(parent).unwrap(),
        Config { policy: ToolPolicy::native(), ..Config::default() },
        Arc::new(AtomicBool::new(false)), factory).unwrap()
}

struct MemoryModel<F>(F);
impl<F: FnMut(ModelRequest<'_>) -> Value> Model for MemoryModel<F> {
    fn complete(&mut self, request: ModelRequest<'_>, _: &AtomicBool,
        _: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<AssistantTurn> {
        let wire = request.wire;
        turn((self.0)(request), wire)
    }
}

fn receipt(request: &ModelRequest<'_>, id: &str) -> Value {
    let message = request.messages.last().unwrap();
    assert_eq!(message.tool_call_id(), Some(id));
    val(message.value().get("content").and_then(Value::as_str).unwrap())
}

#[test]
fn parent_memory_save_and_forget_work_with_child_scheduler() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    let mut parent = memory_parent();
    let mut scheduler = memory_scheduler(&parent, factory(vec![], Arc::new(AtomicUsize::new(0))));
    let mut tools = CodingTools::new(&workspace, ApprovalPolicy::Auto, |_, _| true).with_memory(true);
    let mut step = 0;
    let mut model = MemoryModel(|request: ModelRequest<'_>| {
        let response = match step {
            0 => proposal("save", "memory_save", val(r#"{"key":"style","text":"Brief."}"#)),
            1 => {
                assert_eq!(receipt(&request, "save"), val(r#"{"entries":1}"#));
                assert!(memory::Memory::load(&workspace).unwrap().content().contains("- style: Brief. ("));
                proposal("forget", "memory_forget", val(r#"{"key":"style"}"#))
            }
            2 => {
                assert_eq!(receipt(&request, "forget"), val(r#"{"entries":0}"#));
                assert_eq!(fixture.read(".vade/MEMORY.md"), "");
                answer("done")
            }
            _ => panic!("unexpected model request"),
        };
        step += 1;
        response
    });
    let run = run_parent(&mut parent, &mut model, &mut tools, Limits::default(), &mut scheduler, &mut |_| {});
    assert_eq!(run.report.reason, StopReason::Finished(FinishReason::Stop));
    assert_eq!(run.report.model_turns, 3);
}

#[test]
fn children_cannot_call_or_receive_memory_tools() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    memory::change(&workspace, "style", Some("Brief.")).unwrap();
    let before = fixture.read(".vade/MEMORY.md");
    let parent = memory_parent();
    let policy = ToolPolicy::native();
    for name in ["memory_save", "memory_forget"] {
        assert!(!Scope::native_read_only().tools.iter().any(|tool| tool == name));
        assert!(policy.validate_scope(&Scope { tools: vec![name.into()], write_paths: vec![] }).is_err());
        assert!(ToolPolicy::new([(name.into(), Access::Read)].into()).is_err());
    }
    let factory: Arc<Factory> = Arc::new(|_, _| Ok(ChildWorker {
        observer: None, journal_name: None,
        model: Box::new(Replies {
            replies: vec![
                proposal("save", "memory_save", val(r#"{"key":"style","text":"Changed."}"#)),
                proposal("forget", "memory_forget", val(r#"{"key":"style"}"#)),
                answer("denied acknowledged"),
            ].into(),
            dispatches: Arc::new(AtomicUsize::new(0)), prompt_tokens: 1,
        }),
        tools: Box::new(NoTools),
    }));
    let mut scheduler = memory_scheduler(&parent, factory);
    let id = scheduler.spawn(Task::read_only("read only")).unwrap();
    finished(&scheduler, id);
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.outcome, Outcome::Completed);
    assert!(result.tools.is_empty());
    let conversation = result.conversation.unwrap();
    for id in ["save", "forget"] {
        let message = conversation.messages().iter().find(|m| m.tool_call_id() == Some(id)).unwrap();
        let value = val(message.value().get("content").and_then(Value::as_str).unwrap());
        assert_eq!(value.get("error").and_then(Value::as_str), Some("denied"));
    }
    assert_eq!(fixture.read(".vade/MEMORY.md"), before);
}

#[test]
fn invalid_memory_arguments_allow_correction_without_changing_file() {
    let oversized = "x".repeat(memory::MAX_BYTES);
    for (name, key, text, reason) in [
        ("memory_save", "Bad", "Brief.", "memory key must be short lowercase kebab-case"),
        ("memory_forget", "Bad", "", "memory key must be short lowercase kebab-case"),
        ("memory_save", "style", "", "memory text must be a nonempty line without control characters"),
        ("memory_save", "style", "first\nsecond", "memory text must be a nonempty line without control characters"),
        ("memory_save", "style", oversized.as_str(), "memory entry exceeds the byte limit"),
    ] {
        let fixture = Fixture::new();
        let workspace = fixture.workspace();
        memory::change(&workspace, "style", Some("Original.")).unwrap();
        let before = fixture.read(".vade/MEMORY.md");
        let mut parent = memory_parent();
        let mut scheduler = memory_scheduler(&parent, factory(vec![], Arc::new(AtomicUsize::new(0))));
        let mut tools = CodingTools::new(&workspace, ApprovalPolicy::Auto, |_, _| true).with_memory(true);
        let mut step = 0;
        let mut model = MemoryModel(|request: ModelRequest<'_>| {
            let response = match step {
                0 => {
                    let mut fields = vec![("key".into(), Value::String(key.into()))];
                    if name == "memory_save" { fields.push(("text".into(), Value::String(text.into()))); }
                    proposal("invalid", name, Value::Object(fields))
                }
                1 => {
                    let value = receipt(&request, "invalid");
                    assert_eq!(value.get("error").and_then(Value::as_str), Some("invalid_arguments"));
                    assert_eq!(value.get("message").and_then(Value::as_str), Some(reason));
                    assert!(reason.len() <= 160);
                    assert_eq!(fixture.read(".vade/MEMORY.md"), before);
                    proposal("corrected", "memory_save", val(r#"{"key":"style","text":"Corrected."}"#))
                }
                2 => {
                    assert_eq!(receipt(&request, "corrected"), val(r#"{"entries":1}"#));
                    assert!(fixture.read(".vade/MEMORY.md").contains("- style: Corrected. ("));
                    answer("done")
                }
                _ => panic!("unexpected model request"),
            };
            step += 1;
            response
        });
        let run = run_parent(&mut parent, &mut model, &mut tools, Limits::default(), &mut scheduler, &mut |_| {});
        assert_eq!(run.report.reason, StopReason::Finished(FinishReason::Stop));
        assert_eq!(run.report.model_turns, 3);
    }
}

#[test]
fn memory_storage_failure_stops_parent_turn() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    memory::change(&workspace, "style", Some("Original.")).unwrap();
    let before = fixture.read(".vade/MEMORY.md");
    let dir = workspace.state_dir().unwrap();
    let _lock = dir.lock("memory.lock").unwrap();
    let mut parent = memory_parent();
    let mut scheduler = memory_scheduler(&parent, factory(vec![], Arc::new(AtomicUsize::new(0))));
    let mut tools = CodingTools::new(&workspace, ApprovalPolicy::Auto, |_, _| true).with_memory(true);
    let mut model = MemoryModel(|_: ModelRequest<'_>| proposal("save", "memory_save", val(r#"{"key":"style","text":"Changed."}"#)));
    let run = run_parent(&mut parent, &mut model, &mut tools, Limits::default(), &mut scheduler, &mut |_| {});
    assert_eq!(run.report.reason, StopReason::ToolFailed);
    assert_eq!(run.report.model_turns, 1);
    let message = parent.messages().iter().find(|m| m.tool_call_id() == Some("save")).unwrap();
    let value = val(message.value().get("content").and_then(Value::as_str).unwrap());
    assert_eq!(value.get("error").and_then(Value::as_str), Some("tool_failed"));
    assert_eq!(fixture.read(".vade/MEMORY.md"), before);
}
