use super::*;
use std::sync::atomic::AtomicBool;
use vade_harness::{
    agent::{Model, ModelRequest, ToolError, ToolExecutor, ToolInvocation},
    agent_wire::AssistantTurn,
    json,
};

fn replay_compaction(messages: &str, resume: bool) {
    struct Replay(usize);
    impl Model for Replay {
        fn complete(&mut self, request: ModelRequest<'_>, _: &AtomicBool, _: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<AssistantTurn> {
            let text = request.to_json()?;
            if text.contains("Write a compact state block") {
                return Err(io::Error::other("this mock writes no summaries; local excerpts are the fallback"));
            }
            assert!(text.contains("lossy historical data"), "must compact before generation");
            assert!(request.messages.iter().any(|m| m.value().get("content").and_then(Value::as_str) == Some("Continue")));
            self.0 += 1;
            AssistantTurn::from_response(&json::parse(r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}"#)?, request.wire)
        }
    }
    struct NoTools;
    impl ToolExecutor for NoTools {
        fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> { panic!("never replay a historical call") }
    }
    let path = std::env::temp_dir().join(format!("vade-fit-replay-{}-{resume}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let workspace = Workspace::new(&path).unwrap();
    let limits = Limits { max_context_tokens: 32768, output_reserve_tokens: 8192, ..Limits::default() };
    let raw = format!(r#"{{"model":"mock","tools":[{{"type":"function","function":{{"name":"apply_patch"}}}},{{"type":"function","function":{{"name":"workspace_shell"}}}}],"messages":[{{"role":"system","content":"stable rules"}},{{"role":"user","content":"build seed.rs"}},{messages},{{"role":"user","content":"Continue"}}]}}"#);
    let mut c = Conversation::from_request(json::parse(&raw).unwrap(), limits).unwrap();
    let original = c.clone();
    let usage = context::Usage::for_conversation(&c);
    assert!(usage.near_limit(limits.max_context_tokens, limits.output_reserve_tokens));
    assert_eq!(!usage.fits(limits.max_context_tokens, limits.output_reserve_tokens), resume);
    if resume {
        assert!(context::auto_plan_fitting(&c, 2, limits, &usage, &mut |candidate| Ok(context::Usage::for_conversation(candidate))).unwrap().is_some(), "pre-save Continue admission must be recoverable");
    }
    let journal = Arc::new(Mutex::new(Journal::create(&workspace, "replay", &c).unwrap()));
    let (tx, _rx) = crate::interaction::lanes();
    let mut observer = CompactingJournal { answer: String::new(), tool_prose: Arc::default(), cancel: Arc::default(), inner: SharedJournal(Arc::clone(&journal)), events: Arc::new(tx), enabled: true, trigger: Arc::default() };
    let mut model = Replay(0);
    let report = agent::run_observed(&mut c, &mut model, &mut NoTools, limits, &AtomicBool::new(false), &mut |_| {}, &mut observer);
    assert_eq!((report.reason, model.0, journal.lock().unwrap().generation()), (StopReason::Finished(FinishReason::Stop), 1, 1));
    assert_eq!(c.messages()[0], original.messages()[0]);
    assert!(c.messages().contains(&original.messages()[1]));
    if !resume { assert_eq!(c.messages()[3].tool_calls(), original.messages()[2].tool_calls()); assert!(c.messages().contains(&original.messages()[3])); }
    drop(observer); drop(journal);
    let (_, restored, _) = Journal::resume(&workspace, "replay", limits).unwrap();
    assert_eq!(restored.messages(), c.messages());
    std::fs::remove_dir_all(path).unwrap();
}
#[test]
fn context_stop_continue_summarizes_refused_proposals_and_reaches_model() {
    let messages: Vec<_> = (0..3).map(|i| format!(r#"{{"role":"assistant","tool_calls":[{{"id":"p{i}","type":"function","function":{{"name":"apply_patch","arguments":{}}}}}]}},{{"role":"tool","tool_call_id":"p{i}","content":{}}}"#, json::quote(&format!(r#"{{"patch":"*** Update File: seed.rs\n{}"}}"#, "x".repeat(31000))), json::quote(if i == 2 { r#"{"ok":false,"error":"not_executed"}"# } else { "Patch refused: invalid patch line prefix; no files were written" }))).collect();
    replay_compaction(&messages.join(","), true);
}
#[test]
fn resumed_sessions_accept_previous_write_guidance() {
    let path = std::env::temp_dir().join(format!("vade-legacy-guidance-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let workspace = Workspace::new(&path).unwrap();
    let limits = Limits::default();
    let grants = BTreeMap::new();
    let current = native_conversation("mock", &workspace, &[], "", false, &grants, limits).unwrap();
    assert!(native_system_prompt(false, &grants, limits).contains("~40 lines"));
    assert!(validate_resumed(&current, "mock", false, &grants, limits).is_ok());
    let legacy_prompt = native_system_prompt(false, &grants, limits).replace(
        &coding_tools::write_guidance(vade_harness::contract::guaranteed_native_reply_tokens()),
        &coding_tools::write_guidance(vade_harness::contract::AUTOMATIC_OUTPUT_TOKENS as usize),
    );
    assert!(legacy_prompt.contains("~200 lines"));
    let legacy = Conversation::from_request(
        Value::Object(vec![
            ("model".into(), Value::String("mock".into())),
            ("stream".into(), Value::Bool(true)),
            ("tools".into(), native_definitions_with(false, vade_harness::contract::AUTOMATIC_OUTPUT_TOKENS as usize, vade_harness::coding_tools::SHELL_AVAILABLE)),
            ("messages".into(), Value::Array(vec![
                native_message("system", &legacy_prompt),
                current.messages()[1].value().clone(),
            ])),
        ]),
        Limits::default(),
    ).unwrap();
    assert!(validate_resumed(&legacy, "mock", false, &grants, limits).is_ok());
    std::fs::remove_dir_all(path).unwrap();
}
/// A session saved before workspace_read named converted files (main
/// 9fbf673 wording) still resumes; another description does not.
#[test]
fn resumed_sessions_accept_previous_read_description() {
    let path = std::env::temp_dir().join(format!("vade-legacy-read-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let workspace = Workspace::new(&path).unwrap();
    let (limits, grants) = (Limits::default(), BTreeMap::new());
    let current = native_conversation("mock", &workspace, &[], "", false, &grants, limits).unwrap();
    let with_read = |text: &str| {
        let reply = vade_harness::contract::guaranteed_native_reply_tokens();
        let Value::Array(mut tools) = native_definitions_with(false, reply, vade_harness::coding_tools::SHELL_AVAILABLE) else { panic!("tool list") };
        for tool in &mut tools {
            if tool.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) == Some("workspace_read")
                && let Value::Object(fields) = tool
                && let Some((_, Value::Object(function))) = fields.iter_mut().find(|(key, _)| key == "function")
                && let Some((_, description)) = function.iter_mut().find(|(key, _)| key == "description")
            {
                *description = Value::String(text.into());
            }
        }
        Conversation::from_request(
            Value::Object(vec![
                ("model".into(), Value::String("mock".into())),
                ("stream".into(), Value::Bool(true)),
                ("tools".into(), Value::Array(tools)),
                ("messages".into(), Value::Array(current.messages().iter().map(|m| m.value().clone()).collect())),
            ]),
            Limits::default(),
        )
        .unwrap()
    };
    assert!(validate_resumed(&with_read(LEGACY_READ_DESCRIPTION), "mock", false, &grants, limits).is_ok());
    assert!(validate_resumed(&with_read("Read a file."), "mock", false, &grants, limits).is_err());
    std::fs::remove_dir_all(path).unwrap();
}
#[test]
fn no_edit_compaction_reduces_protected_command_reasoning_before_limit() {
    replay_compaction(&format!(r#"{{"role":"assistant","reasoning_content":{},"tool_calls":[{{"id":"check","type":"function","function":{{"name":"workspace_shell","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"check","content":"status: Exited(Some(0))"}}"#, json::quote(&"thinking ".repeat(6900))), false);
}

#[test]
fn fresh_chat_at_whole_context_output_never_defers_compaction() {
    // Default local launch: context 8192, output 8192, automatic reply reservation.
    let limits = Limits { max_context_tokens: 8192, output_reserve_tokens: 8192, ..Limits::default() };
    assert_eq!(context::reply_reservation(8192, 8192), 2048);
    assert_eq!(context::reply_reservation(32768, 8192), 8192);
    let path = std::env::temp_dir().join(format!("vade-fresh-compaction-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let workspace = Workspace::new(&path).unwrap();
    let request = r#"{"model":"mock","messages":[{"role":"system","content":"stable"},{"role":"user","content":"Say hello."}]}"#;
    let counted = |c: &Conversation, prompt_tokens: u64| context::Usage::for_conversation(c)
        .with_count(&vade_harness::contract::InputCount::Exact {
            prompt_tokens, output_tokens: 8192 - prompt_tokens - 64, context_limit: 8192, admitted: true,
        }).unwrap();
    // Both fit with the quarter-context reply reservation; the large fixed
    // prefix crosses 80% but a fresh chat has nothing compaction may remove.
    for prompt_tokens in [600, 6000] {
        let mut c = Conversation::from_request(json::parse(request).unwrap(), Limits::default()).unwrap();
        let usage = counted(&c, prompt_tokens);
        assert!(usage.fits(8192, 8192));
        assert_eq!(usage.near_limit(8192, 8192), prompt_tokens == 6000);
        let journal = Arc::new(Mutex::new(Journal::create(&workspace, &format!("fresh-{prompt_tokens}"), &c).unwrap()));
        let records = journal.lock().unwrap().record_count();
        let (tx, rx) = crate::interaction::lanes();
        let mut observer = CompactingJournal {
            answer: String::new(), tool_prose: Arc::default(), cancel: Arc::default(),
            inner: SharedJournal(Arc::clone(&journal)), events: Arc::new(tx), enabled: true,
            trigger: Arc::new(Mutex::new(context::AutoCompaction::default())),
        };
        assert!(!observer.before_counted_request(&mut c, limits, &AtomicBool::new(false), &usage,
            &mut |_| panic!("a fresh chat has nothing to compact")).unwrap());
        assert!(matches!(rx.fact(), Err(std::sync::mpsc::TryRecvError::Empty)));
        assert_eq!(journal.lock().unwrap().record_count(), records);
        assert!(observer.trigger.lock().unwrap().can_attempt_next_turn());
    }
    std::fs::remove_dir_all(path).unwrap();
}
