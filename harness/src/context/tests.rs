use super::*;
use crate::json;
fn conversation() -> Conversation {
    Conversation::from_request(json::parse(r#"{"model":"mock","messages":[{"role":"system","content":"stable"},{"role":"user","content":"first long question with a great deal of repeated information"},{"role":"assistant","content":"a long answer that is intentionally large enough to benefit from a short summary. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. Resolved verbose analysis and intermediate observations. "},{"role":"user","content":"second"},{"role":"assistant","content":"answer"},{"role":"user","content":"third"}]}"#).unwrap(), Limits::default()).unwrap()
}
#[test]
fn accounts_each_message_options_and_reserve_without_claiming_exact_tokens() {
    let c = conversation();
    let usage = Usage::for_conversation(&c);
    assert_eq!(usage.messages.len(), c.messages().len());
    assert_eq!(
        usage.estimated,
        usage.options_estimated
            + usage.messages.iter().map(|m| m.estimated).sum::<usize>()
            + usage.framing_estimated
            + usage.safety_margin
    );
    assert_eq!(
        usage.byte_bound,
        c.request(Limits::default().wire).to_json().unwrap().len()
    );
    assert!(usage.near_limit(usage.estimated + 10, 10));
    assert!(!usage.fits(usage.estimated, 1));
}
#[test]
fn full_request_counts_schemas_opaque_fields_utf8_and_provider_output_limit() {
    let request = |extra: &str| {
        Conversation::from_request(crate::json::parse(&format!(
        r#"{{"model":"mock","tools":[{{"type":"function","function":{{"name":"read","description":"{}"}}}}],"max_completion_tokens":6000,"opaque":{{"reasoning":"{}"}},"messages":[{{"role":"system","content":"rules"}},{{"role":"user","content":[{{"type":"text","text":"🦀多字"}}]}}]}}"#,
        extra, extra)).unwrap(), Limits::default()).unwrap()
    };
    let small = Usage::for_conversation(&request("small"));
    let large = Usage::for_conversation(&request(&"schema ".repeat(2000)));
    assert!(large.estimated > small.estimated + 9000);
    assert!(large.tools_estimated > small.tools_estimated);
    assert!(small.fits(small.estimated + 6000, 4096));
    assert!(!small.fits(small.estimated + 5999, 4096));
    assert_eq!(
        large.byte_bound,
        request(&"schema ".repeat(2000)).serialized_bytes()
    );
    assert!(large.summary(8192, 4096).contains("heuristic only"));
    let c = Conversation::from_request(crate::json::parse(r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://invalid.example/image"}}]}]}"#).unwrap(), Limits::default()).unwrap();
    let usage = Usage::for_conversation(&c);
    assert!(usage.admission_issue.is_some());
    assert!(!usage.fits(usize::MAX, 4096));
}
#[test]
fn append_admission_is_at_least_as_cautious_as_final_request() {
    let mut c = conversation();
    let m = crate::agent_wire::Message::from_value(
        crate::json::parse(
            r#"{"role":"user","content":"a\\b\n🦀","extension":{"opaque":[1,2,3]}}"#,
        )
        .unwrap(),
        Limits::default().wire,
    )
    .unwrap();
    let preview = Usage::with_appended(&c, std::slice::from_ref(&m));
    c.push_user(m.into_value(), Limits::default()).unwrap();
    let actual = Usage::for_conversation(&c);
    assert!(preview.estimated >= actual.estimated);
    assert_eq!(preview.byte_bound, actual.byte_bound);
}
#[test]
fn explicit_compaction_keeps_stable_prefix_and_whole_recent_turns() {
    let c = conversation();
    let compacted = compact(&c, "First resolved.", 2, Limits::default()).unwrap();
    assert_eq!(compacted.options(), c.options());
    assert_eq!(compacted.messages()[0], c.messages()[0]);
    assert_eq!(&compacted.messages()[3..], &c.messages()[3..]);
    assert!(
        compacted.messages()[1]
            .value()
            .get("content")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("First resolved.")
    );
    assert_eq!(c.messages().len(), 6);
}
#[test]
fn refuses_empty_nonreducing_or_overbroad_compaction() {
    let c = conversation();
    for (summary, keep) in [("", 1), ("short", 0), ("short", 3), ("short", usize::MAX)] {
        assert!(compact(&c, summary, keep, Limits::default()).is_err());
    }
    assert!(compact(&c, &"x".repeat(16384), 1, Limits::default()).is_err());
}
#[test]
fn compaction_never_splits_tool_groups() {
    let c = Conversation::from_request(json::parse(r#"{"model":"mock","messages":[{"role":"system","content":"stable"},{"role":"user","content":"old long turn that has now been resolved and can be summarized compactly"},{"role":"assistant","content":"a detailed old answer that is now no longer needed for any further model work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. Discardable prose about resolved old work. "},{"role":"user","content":"new"},{"role":"assistant","tool_calls":[{"id":"x","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"x","content":"result"},{"role":"assistant","content":"done"}]}"#).unwrap(), Limits::default()).unwrap();
    let compacted = compact(&c, "Old resolved.", 1, Limits::default()).unwrap();
    assert_eq!(&compacted.messages()[3..], &c.messages()[3..]);
}
#[test]
fn plan_keeps_old_constraints_tool_receipts_and_latest_answer_exactly() {
    let original = format!(
        r#"{{"model":"mock","messages":[
            {{"role":"system","content":"system"}},
            {{"role":"user","content":"NEVER delete original data; unresolved: which file?"}},
            {{"role":"assistant","content":"{}"}},
            {{"role":"developer","content":"additional immutable constraint"}},
            {{"role":"assistant","tool_calls":[{{"id":"write","type":"function","function":{{"name":"apply_patch","arguments":"{{}}"}}}}]}},
            {{"role":"tool","tool_call_id":"write","content":"Applied file revision 2; receipt exact"}},
            {{"role":"assistant","content":"last useful answer and decision"}},
            {{"role":"user","content":"current intent"}}]}}"#,
        "verbose resolved work ".repeat(100)
    );
    let c =
        Conversation::from_request(json::parse(&original).unwrap(), Limits::default()).unwrap();
    let before = c.request(Limits::default().wire).to_json().unwrap();
    let plan = plan(
        &c,
        "Earlier analysis resolved; preserve the pending file question.",
        1,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(plan.removed_indices, [2]);
    assert_eq!(plan.unchanged_prefix_messages, c.prefix_len());
    for i in [0, 1, 3, 4, 5, 6, 7] {
        assert!(plan.conversation.messages().contains(&c.messages()[i]));
    }
    assert_eq!(c.request(Limits::default().wire).to_json().unwrap(), before);
    assert!(plan.preview().contains("invalidates all later causal KV"));
    assert!(
        plan.preview()
            .contains("semantic completeness is not verified")
    );
    let uncertain = original.replace(
        "Applied file revision 2; receipt exact",
        r#"{\"ok\":false,\"error\":\"unknown_outcome\"}"#,
    );
    let c = Conversation::from_request(json::parse(&uncertain).unwrap(), Limits::default())
        .unwrap();
    let automatic = auto_plan(&c, 1, Limits::default()).unwrap();
    for i in [4, 5] {
        assert!(automatic.retained_indices.contains(&i));
    }
    let nested = original.replace(
        "Applied file revision 2; receipt exact",
        r#"{\"results\":[{\"uncertain_writes\":true}]}"#,
    );
    let nested =
        Conversation::from_request(json::parse(&nested).unwrap(), Limits::default()).unwrap();
    let automatic = auto_plan(&nested, 1, Limits::default()).unwrap();
    for i in [4, 5] {
        assert!(automatic.retained_indices.contains(&i));
    }
    assert!(
        super::plan(&c, "short", 1, Limits::default())
            .unwrap_err()
            .to_string()
            .contains("unknown tool outcome")
    );
}
#[test]
fn single_task_compaction_keeps_goal_edits_checks_and_uncertain_groups() {
    let mut rounds = Vec::new();
    for i in 0..9 {
        let name = ["apply_patch", "workspace_shell", "workspace_read_lines"][i.min(2)];
        rounds.push(format!(r#"{{"role":"assistant","tool_calls":[{{"id":"r{i}","type":"function","function":{{"name":"{name}","arguments":"{{\"path\":\"src/task.rs\",\"start_line\":13,\"end_line\":21}}"}}}}]}}"#));
        let text = if i == 2 {
            r#"{"error":"unknown_outcome"}"#.into()
        } else {
            format!("key fact {i} {}", "old source ".repeat(700))
        };
        rounds.push(format!(
            r#"{{"role":"tool","tool_call_id":"r{i}","content":{}}}"#,
            json::quote(&text)
        ));
    }
    let c = Conversation::from_request(json::parse(&format!(r#"{{"messages":[{{"role":"system","content":"stable"}},{{"role":"user","content":"fix src/task.rs; keep public API"}},{}]}}"#, rounds.join(","))).unwrap(), Limits::default()).unwrap();
    let plan = auto_plan(&c, 2, Limits::default()).unwrap();
    assert!(plan.in_task && plan.summary.contains("src/task.rs"));
    assert!(plan.summary.contains("start_line=13"));
    for i in (0..8).chain(14..20) {
        assert!(plan.retained_indices.contains(&i));
    }
    for i in 8..14 {
        assert!(plan.removed_indices.contains(&i));
    }
    assert_eq!(plan.conversation.messages()[0], c.messages()[0]);
    assert!(plan.conversation.messages().contains(&c.messages()[1]));
    assert!(is_summary(&plan.conversation.messages()[1]));
}
#[test]
fn auto_compaction_preserves_prior_summaries_and_stable_wire_prefix() {
    let limits = Limits::default();
    let request = format!(
        r#"{{"model":"mock","tools":[],"messages":[{{"role":"system","content":"stable 🦀"}},{{"role":"developer","content":"exact project prefix"}},{{"role":"user","content":"first task"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"second task"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"current task"}}]}}"#,
        "resolved first ".repeat(300),
        "resolved second ".repeat(300)
    );
    let original = Conversation::from_request(json::parse(&request).unwrap(), limits).unwrap();
    let first = auto_plan(&original, 1, limits).unwrap().into_conversation();
    let summary = first
        .messages()
        .iter()
        .find(|m| m.value().get("name").and_then(Value::as_str) == Some("vade_context_summary"))
        .unwrap()
        .clone();
    // Only the prior summary is old/removable: another request must not
    // shorten it into an excerpt of itself.
    assert!(auto_plan(&first, 1, limits).is_err());
    let mut messages = first.messages().to_vec();
    messages.push(
        crate::agent_wire::Message::from_value(
            json::parse(r#"{"role":"assistant","content":"current answer completed"}"#)
                .unwrap(),
            limits.wire,
        )
        .unwrap(),
    );
    let mut next = first.with_messages(messages, limits).unwrap();
    next.push_user(
        json::parse(r#"{"role":"user","content":"later task"}"#).unwrap(),
        limits,
    )
    .unwrap();
    let second = auto_plan(&next, 1, limits).unwrap().into_conversation();
    assert!(second.messages().contains(&summary));
    assert_eq!(
        second
            .messages()
            .iter()
            .filter(|m| m.value().get("name").and_then(Value::as_str)
                == Some("vade_context_summary"))
            .count(),
        2
    );
    for c in [&first, &second] {
        assert_eq!(
            c.request(limits.wire).serialized_options(),
            original.request(limits.wire).serialized_options()
        );
        for i in 0..original.prefix_len() {
            assert_eq!(
                c.messages()[i].serialized_json().as_bytes(),
                original.messages()[i].serialized_json().as_bytes()
            );
        }
    }
}
#[test]
fn auto_compaction_bounds_many_old_excerpts_without_failing() {
    let mut messages = vec![json::parse(r#"{"role":"system","content":"fixed"}"#).unwrap()];
    for i in 0..70 {
        messages
            .push(json::parse(&format!(r#"{{"role":"user","content":"task {i}"}}"#)).unwrap());
        messages.push(Value::Object(vec![
            ("role".into(), Value::String("assistant".into())),
            (
                "content".into(),
                Value::String(format!("result {i} {}", "resolved ".repeat(100))),
            ),
        ]));
    }
    let c = Conversation::from_request(
        Value::Object(vec![("messages".into(), Value::Array(messages))]),
        Limits::default(),
    )
    .unwrap();
    let plan = auto_plan(&c, 2, Limits::default()).unwrap();
    assert!(plan.summary.len() <= 16 * 1024);
    assert!(plan.summary.contains("oldest result excerpts omitted"));
    assert!(plan.summary.contains("result 67"));
    assert!(!plan.summary.contains("result 0 "));
    assert_eq!(
        plan.conversation
            .messages()
            .iter()
            .filter(|m| m.role() == "user" && m.value().get("name").is_none())
            .count(),
        70
    );
}
#[test]
fn auto_trigger_attempts_once_until_usage_falls_to_sixty_percent() {
    let c = conversation();
    let usage = Usage::for_conversation(&c);
    let required = usage.required(0);
    let mut trigger = AutoCompaction::default();
    let high_limit = required * 5 / 4;
    assert!(trigger.observe(&usage, high_limit, 0));
    for _ in 0..8 {
        assert!(!trigger.observe(&usage, high_limit, 0));
    }
    assert!(!trigger.observe(&usage, required * 10 / 7, 0));
    assert!(!trigger.observe(&usage, required * 2, 0));
    assert!(trigger.observe(&usage, high_limit, 0));
    let mut large = usage.clone(); large.exact = Some((50_000, 65_536, 131_072, true));
    assert!(!large.near_limit(131_072, 65_536));
    large.exact = Some((122_881, 7935, 131_072, true));
    assert!(!large.fits(131_072, 16_384) && large.fits(131_072, 7935));
    let limits = Limits { max_context_tokens: 32768, output_reserve_tokens: 16384, ..Limits::default() };
    let raw = format!(r#"{{"messages":[{{"role":"system","content":"stable"}},{{"role":"user","content":"{}"}},{{"role":"assistant","content":"{} END β 🦀"}},{{"role":"user","content":"/continue"}}]}}"#, "u".repeat(36000), "reply ".repeat(5500));
    let cutoff = Conversation::from_request(json::parse(&raw).unwrap(), limits).unwrap();
    let before = Usage::for_conversation(&cutoff);
    let (plan, after) = auto_plan_fitting(&cutoff, 2, limits, &before, &mut |c| Ok(Usage::for_conversation(c))).unwrap().unwrap();
    assert!(!before.fits(32768, 16384) && after.fits(32768, 16384) && plan.summary.contains("END β 🦀") && plan.conversation.messages().contains(&cutoff.messages()[0]));
}
#[test]
fn empty_recent_assistant_does_not_displace_the_last_meaningful_answer() {
    let request = format!(
        r#"{{"messages":[{{"role":"user","content":"old"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"question"}},{{"role":"assistant","content":"meaningful answer"}},{{"role":"user","content":"interrupted"}},{{"role":"assistant","content":"  "}},{{"role":"user","content":"current"}}]}}"#,
        "old resolved prose ".repeat(100)
    );
    let c =
        Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
    let plan = plan(&c, "Earlier work resolved.", 1, Limits::default()).unwrap();
    assert!(plan.retained_indices.contains(&3));
    assert!(
        auto_plan(&c, 1, Limits::default())
            .unwrap()
            .retained_indices
            .contains(&3)
    );
    assert_eq!(
        plan.conversation
            .messages()
            .iter()
            .filter(|m| m.value().get("content").and_then(Value::as_str)
                == Some("meaningful answer"))
            .count(),
        1
    );
}
#[test]
fn empty_history_append_has_exact_measured_byte_count() {
    let mut c = Conversation::from_request(
        json::parse(r#"{"messages":[]}"#).unwrap(),
        Limits::default(),
    )
    .unwrap();
    let message = crate::agent_wire::Message::from_value(
        json::parse(r#"{"role":"user","content":"first 🦀"}"#).unwrap(),
        Limits::default().wire,
    )
    .unwrap();
    let projected = Usage::with_appended(&c, std::slice::from_ref(&message));
    c.push_user(message.into_value(), Limits::default())
        .unwrap();
    assert_eq!(
        projected.byte_bound,
        c.request(Limits::default().wire).to_json().unwrap().len()
    );
    assert!(projected.estimated >= Usage::for_conversation(&c).estimated);
}
#[test]
fn runtime_stops_before_a_model_call_over_budget() {
    use crate::agent::{
        self, LimitKind, Model, ModelRequest, StopReason, ToolError, ToolExecutor,
        ToolInvocation,
    };
    use crate::agent_wire::AssistantTurn;
    use std::sync::atomic::AtomicBool;
    struct NeverModel;
    impl Model for NeverModel {
        fn complete(
            &mut self,
            _: ModelRequest<'_>,
            _: &AtomicBool,
            _: &mut dyn FnMut(&str) -> io::Result<()>,
        ) -> io::Result<AssistantTurn> {
            panic!("model must not run above budget")
        }
    }
    struct NeverTools;
    impl ToolExecutor for NeverTools {
        fn execute(
            &mut self,
            _: ToolInvocation<'_>,
            _: &AtomicBool,
        ) -> Result<String, ToolError> {
            panic!("tools must not run")
        }
    }
    let mut c = conversation();
    c.push_user(
        Value::Object(vec![
            ("role".into(), Value::String("user".into())),
            ("content".into(), Value::String("x".repeat(20000))),
        ]),
        Limits::default(),
    )
    .unwrap();
    let before = c.messages().to_vec();
    let report = agent::run(
        &mut c,
        &mut NeverModel,
        &mut NeverTools,
        Limits {
            max_context_tokens: 8192,
            ..Limits::default()
        },
        &AtomicBool::new(false),
        &mut |_| {},
    );
    assert_eq!(report.reason, StopReason::Limit(LimitKind::Context));
    assert_eq!(report.model_turns, 0);
    assert_eq!(before, c.messages());
    let exact = Usage::for_conversation(&c)
        .with_count(&crate::contract::InputCount::Exact {
            prompt_tokens: 100,
            output_tokens: 128,
            context_limit: 8192,
            admitted: true,
        })
        .unwrap();
    assert!(exact.fits(8192, 4096));
    assert_eq!(exact.required(4096), 228); // actual reservation, no heuristic margin
    assert!(!exact.near_limit(8192, 4096));
    let refused = exact
        .with_count(&crate::contract::InputCount::Exact {
            prompt_tokens: 9000,
            output_tokens: 128,
            context_limit: 8192,
            admitted: false,
        })
        .unwrap();
    assert!(!refused.fits(8192, 0));
}
#[test]
fn pressure_compacts_old_successful_edits_but_keeps_goal_recent_and_uncertain() {
    let mut messages = vec![json::parse(r#"{"role":"system","content":"stable rules"}"#).unwrap(), json::parse(r#"{"role":"user","content":"implement all outputs"}"#).unwrap()]; for i in 0..10 { let args = json::stringify(&json::parse(&format!(r#"{{"patch":{}}}"#, json::quote(&format!("*** Begin Patch\n*** Add File: f{i}.rs\n+{}\n*** End Patch", "x".repeat(8000))))).unwrap()).unwrap(); messages.push(json::parse(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"r{i}","type":"function","function":{{"name":"apply_patch","arguments":{}}}}}]}}"#, json::quote(&args))).unwrap()); messages.push(json::parse(&format!(r#"{{"role":"tool","tool_call_id":"r{i}","content":{}}}"#, json::quote(if i == 6 { r#"{"error":"unknown_outcome"}"# } else { "Applied 1 file operations." }))).unwrap()); } let c = Conversation::from_request(Value::Object(vec![("messages".into(), Value::Array(messages))]), Limits::default()).unwrap(); let limits = Limits { max_context_tokens: 16000, output_reserve_tokens: 128, ..Limits::default() }; let plan = auto_plan_fitting(&c, 2, limits, &Usage::for_conversation(&c), &mut |c| Ok(Usage::for_conversation(c))).unwrap().unwrap().0; assert!(plan.conversation.serialized_bytes() < c.serialized_bytes()); for i in (0..2).chain(14..22) { assert!(plan.retained_indices.contains(&i)); } assert!(plan.removed_indices.contains(&2)); assert!(plan.summary.contains("f0.rs")); assert_eq!(plan.conversation.messages()[0], c.messages()[0]);
}
