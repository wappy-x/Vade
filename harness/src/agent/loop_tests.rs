//! Loop guards that must hold when the model repeats a broken call: malformed
//! arguments, cutoffs on varied paths, and the automatic completion check.
use super::*;

fn read_call(id: &str, arguments: &str) -> Value {
    response(value(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"{id}","type":"function","function":{{"name":"read","arguments":{}}}}}]}}"#, json::quote(arguments))), "tool_calls")
}

#[test]
fn malformed_json_tool_calls_trip_identical_refusal_breaker_after_three_attempts() {
    let mut c = conversation();
    let raw = r#"{"path":"a.txt""#;
    let mut model = MockModel::new((0..4).map(|n| read_call(&format!("bad-{n}"), raw)).chain([answer()]).collect());
    let mut tools = MockTools::default();
    let r = run_mock(&mut c, &mut model, &mut tools, Limits::default());
    assert_eq!((r.reason, r.tool_turns, r.nudges, tools.ids.len()), (StopReason::ToolFailed, 3, 1, 0));
    assert_closed(&c);
    // The model sees the start of its own broken text; history keeps "{}".
    let receipt = c.messages.last().unwrap().value().get("content").and_then(Value::as_str).unwrap().to_owned();
    assert!(receipt.contains("not valid JSON") && receipt.contains(r#"They began: {\"path\":\"a.txt\""#), "{receipt}");
    assert!(c.messages.iter().filter(|m| m.role() == "assistant").all(|m| m.tool_calls().iter().all(|call| call.arguments == "{}")));
    // Different broken text on each turn is not an identical streak.
    let mut c = conversation();
    let mut model = MockModel::new((0..4).map(|n| read_call(&format!("vary-{n}"), &format!(r#"{{"path":"{n}.txt""#))).chain([answer()]).collect());
    let r = run_mock(&mut c, &mut model, &mut MockTools::default(), Limits::default());
    assert_eq!((r.reason, r.tool_turns), (StopReason::Finished(FinishReason::Stop), 4));
}

#[test]
fn truncated_tool_call_retry_limit_holds_when_model_varies_target_path() {
    let mut c = conversation();
    let cut = |path: &str| response(value(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"cut-{path}","type":"function","function":{{"name":"read","arguments":{}}}}}]}}"#, json::quote(&format!(r#"{{"path":"{path}","text":"long"#)))), "length");
    let mut model = MockModel::new(["a.rs", "b.rs", "c.rs", "d.rs", "e.rs"].iter().map(|p| cut(p)).chain([answer()]).collect());
    let mut tools = MockTools::default();
    let r = run_mock(&mut c, &mut model, &mut tools, Limits::default());
    assert_eq!((r.reason, r.model_turns, r.nudges), (StopReason::TruncatedToolCall, 4, 3));
    assert!(tools.ids.is_empty());
    // A tool round between cutoffs is progress and resets the streak.
    let mut c = conversation();
    let mut model = MockModel::new(vec![cut("a.rs"), cut("b.rs"), cut("c.rs"), read_call("ok", r#"{"path":"c.rs"}"#), cut("d.rs"), answer()]);
    let r = run_mock(&mut c, &mut model, &mut MockTools::default(), Limits::default());
    assert_eq!(r.reason, StopReason::Finished(FinishReason::Stop));
}

/// Edits arm a check after every call; check receipts come from `check`.
struct Checked {
    pending: bool,
    check: Result<String, ToolError>,
    ran: Vec<String>,
}
impl ToolExecutor for Checked {
    fn execute(&mut self, invocation: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        self.ran.push(invocation.call.id.clone());
        if invocation.call.id.starts_with(CHECK_CALL_PREFIX) {
            return self.check.clone();
        }
        self.pending = true;
        Ok("Applied 1 file operations.".into())
    }
    fn take_pending_check(&mut self) -> Option<Value> {
        std::mem::take(&mut self.pending).then(|| value(r#"{"argv":["make"],"timeout_ms":1000}"#))
    }
}

#[test]
fn post_edit_check_obeys_the_call_cap_and_stops_on_a_full_output_store() {
    let run = |check: Result<String, ToolError>, limits: Limits| {
        let mut c = conversation();
        c.tool_names.insert("workspace_shell".into());
        let mut model = MockModel::new(vec![read_call("edit", r#"{"path":"a.rs"}"#), answer()]);
        let mut tools = Checked { pending: false, check, ran: Vec::new() };
        let r = run_mock(&mut c, &mut model, &mut tools, limits);
        assert_closed(&c);
        (r.reason, r.tool_calls, tools.ran.len())
    };
    assert_eq!(run(Ok("status: Exited(Some(0))".into()), Limits::default()), (StopReason::Finished(FinishReason::Stop), 2, 2));
    assert_eq!(run(Err(ToolError::OutputLimit), Limits::default()), (StopReason::Limit(LimitKind::ToolOutput), 2, 2));
    assert_eq!(run(Ok("status: Exited(Some(0))".into()), Limits { max_tool_calls: 1, ..Limits::default() }), (StopReason::Limit(LimitKind::ToolCalls), 1, 1));
}
