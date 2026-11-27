    use super::*;
    use crate::agent_wire::ChatAccumulator;
    use std::collections::VecDeque;

    fn value(text: &str) -> Value {
        json::parse(text).unwrap()
    }

    fn conversation() -> Conversation {
        Conversation::from_request(value(r#"{
            "model":"mock-only","stream":true,"temperature":0,"provider_option":{"keep":true},
            "tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"},"future_schema":{"keep":true}}}],
            "messages":[{"role":"system","content":" Rules.\n  Exact 🦀\n","system_extension":{"keep":true}},
                        {"role":"user","content":[{"type":"text","text":"inspect"},{"type":"text","text":"known text","future_payload":42}],"user_extension":true}]
        }"#), Limits::default()).unwrap()
    }

    fn calls(ids: &[&str]) -> Value {
        Value::Object(vec![
            ("role".into(), Value::String("assistant".into())),
            ("content".into(), Value::Null),
            ("reasoning_content".into(), Value::String("preserve reasoning".into())),
            ("tool_calls".into(), Value::Array(ids.iter().map(|id| Value::Object(vec![
                ("id".into(), Value::String((*id).into())),
                ("type".into(), Value::String("function".into())),
                ("function".into(), value(r#"{"name":"read","arguments":"{\"path\":\"sample.rs\"}","future_function":true}"#)),
                ("future_call".into(), Value::Bool(true)),
            ])).collect())),
        ])
    }

    fn response(message: Value, finish: &str) -> Value {
        Value::Object(vec![(
            "choices".into(),
            Value::Array(vec![Value::Object(vec![
                ("message".into(), message),
                ("finish_reason".into(), Value::String(finish.into())),
            ])]),
        )])
    }

    fn answer() -> Value {
        response(
            value(r#"{"role":"assistant","content":"Done.","future_answer":{"keep":true}}"#),
            "stop",
        )
    }

    #[test]
    #[ignore = "tool dispatch overhead with zero-work executor; release, serial"]
    fn snappy_tools() {
        struct TimedTools(Duration);
        impl ToolExecutor for TimedTools {
            fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
                let start = Instant::now(); let result = Ok("ok".into()); self.0 += start.elapsed(); result
            }
        }
        for run_id in 0..5 { for name in ["read", "edit", "run"] { for durable in [false, true] {
            let mut total = Duration::ZERO;
            let repeats = if durable { 1 } else { 100 };
            for _ in 0..repeats {
                let mut c = Conversation::from_request(value(&conversation().request(Limits::default().wire).to_json().unwrap().replace("\"read\"", &format!("\"{name}\""))), Limits::default()).unwrap();
                let tool = value(&json::stringify(&calls(&["bench"])).unwrap().replace("\"read\"", &format!("\"{name}\"")));
                let mut model = MockModel::new(vec![response(tool, "tool_calls"), answer()]);
                let mut tools = TimedTools(Duration::ZERO);
                let fixture = durable.then(crate::test_workspace::Fixture::new);
                let mut journal = fixture.as_ref().map(|f| crate::journal::Journal::create(&f.workspace(), "bench", &c).unwrap());
                let start = Instant::now();
                let report = if let Some(journal) = &mut journal { run_observed(&mut c, &mut model, &mut tools, Limits::default(), &AtomicBool::new(false), &mut |_| {}, journal) } else { run(&mut c, &mut model, &mut tools, Limits::default(), &AtomicBool::new(false), &mut |_| {}) };
                assert_eq!(report.tool_calls, 1);
                total += start.elapsed().saturating_sub(tools.0);
            }
            println!("snappy run={run_id} tool={name} durable={durable} dispatch_roundtrip_ms={:.4}", total.as_secs_f64() * 1000. / repeats as f64);
        }}}
    }

    struct MockModel {
        replies: VecDeque<Value>,
        requests: Vec<Value>,
    }
    impl MockModel {
        fn new(replies: Vec<Value>) -> Self {
            Self {
                replies: replies.into(),
                requests: Vec::new(),
            }
        }
    }
    impl Model for MockModel {
        fn complete(
            &mut self,
            request: ModelRequest<'_>,
            _: &AtomicBool,
            _: &mut dyn FnMut(&str) -> io::Result<()>,
        ) -> io::Result<AssistantTurn> {
            self.requests.push(json::parse(&request.to_json()?)?);
            AssistantTurn::from_response(
                &self
                    .replies
                    .pop_front()
                    .expect("unexpected extra model turn"),
                request.wire,
            )
        }
    }

    #[derive(Default)]
    struct MockTools {
        ids: Vec<String>,
        replies: VecDeque<Result<String, ToolError>>,
        cancel_after: Option<usize>,
    }
    impl ToolExecutor for MockTools {
        fn execute(
            &mut self,
            invocation: ToolInvocation<'_>,
            cancel: &AtomicBool,
        ) -> Result<String, ToolError> {
            assert!(matches!(invocation.arguments, Value::Object(_)));
            self.ids.push(invocation.call.id.clone());
            if self.cancel_after == Some(self.ids.len()) {
                cancel.store(true, Ordering::Relaxed);
            }
            self.replies
                .pop_front()
                .unwrap_or_else(|| Ok(format!("result for {}", invocation.call.id)))
        }
    }

    #[test]
    fn chat_hooks_block_dispatch_and_save_notes_after_the_effect_checkpoint() {
        struct HookJournal {
            hooks: crate::hooks::Hooks,
            fixture: crate::test_workspace::Fixture,
            staged: Vec<Vec<Message>>,
        }
        impl RunObserver for HookJournal {
            fn before_tool(&mut self, call: &ToolCall, args: &Value, cancel: &AtomicBool) -> Option<String> {
                assert!(self.staged.last().unwrap()[1].serialized_json().contains("unknown_outcome"));
                self.hooks.before(&call.name, args, cancel, &mut |_| panic!("unexpected hook warning"))
            }
            fn after_tool(&mut self, call: &ToolCall, args: &Value, content: &mut String, cap: usize, cancel: &AtomicBool) {
                self.hooks.after(&call.name, args, content, cap, cancel, &mut |_| panic!("unexpected hook warning"));
            }
            fn checkpoint(&mut self, _: &Conversation, staged: &[Message]) -> io::Result<()> {
                self.staged.push(staged.to_vec()); Ok(())
            }
        }
        for (blocked, background) in [(false, false), (true, false), (false, true), (true, true)] {
            let fixture = crate::test_workspace::Fixture::new();
            fixture.workspace().state_dir().unwrap();
            let child = if background { "sleep 30 &\n" } else { "" };
            fixture.write("before.sh", &format!("{child}{}", if blocked { "printf 'project rule' >&2; exit 2" } else { "exit 0" }));
            fixture.write("after.sh", "printf 'test note'; touch after-ran");
            fixture.write_trusted("hooks.conf", r#"{"hooks":[{"point":"before-tool","command":"/bin/sh before.sh","timeout_ms":5000},{"point":"after-tool","command":"/bin/sh after.sh"}]}"#);
            let hooks = crate::hooks::Hooks::load(&fixture.workspace(), Some(&fixture.home())).unwrap();
            let mut observer = HookJournal { hooks, fixture, staged: Vec::new() };
            let mut c = conversation();
            let mut model = MockModel::new(vec![response(calls(&["hook-call"]), "tool_calls"), answer()]);
            let mut tools = MockTools::default();
            let report = run_observed(&mut c, &mut model, &mut tools, Limits::default(), &AtomicBool::new(false), &mut |_| {}, &mut observer);
            assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
            assert_eq!(tools.ids.is_empty(), blocked);
            assert_eq!(observer.fixture.0.join("after-ran").exists(), !blocked);
            let receipt = c.messages().iter().find(|m| m.role() == "tool").unwrap().serialized_json();
            assert!(receipt.contains(if blocked { "Hook blocked read: project rule" } else { "[after-tool hook note]" }));
            assert!(observer.staged.iter().flatten().any(|m| m.serialized_json() == receipt));
        }
    }

    #[test]
    fn generic_conformance_loop_does_not_load_project_hooks() {
        let fixture = crate::test_workspace::Fixture::new();
        fixture.workspace().state_dir().unwrap();
        fixture.write("hook.sh", "touch hook-ran; exit 2");
        fixture.write(".vade/hooks.conf", r#"{"hooks":[{"point":"before-tool","command":"/bin/sh hook.sh"},{"point":"after-tool","command":"/bin/sh hook.sh"},{"point":"on-stop","command":"/bin/sh hook.sh"}]}"#);
        let mut c = conversation();
        let mut model = MockModel::new(vec![response(calls(&["ordinary"]), "tool_calls"), answer()]);
        let mut tools = MockTools::default();
        run(&mut c, &mut model, &mut tools, Limits::default(), &AtomicBool::new(false), &mut |_| {});
        assert_eq!(tools.ids, ["ordinary"]);
        assert!(!fixture.0.join("hook-ran").exists());
    }

    #[test]
    fn exact_counts_control_both_tool_guards_despite_large_local_estimates() {
        struct ExactModel {
            inner: MockModel,
            checked: Vec<String>,
            refuse_second: bool,
        }
        impl Model for ExactModel {
            fn count_input(
                &mut self,
                request: ModelRequest<'_>,
                _: &AtomicBool,
            ) -> io::Result<crate::contract::InputCount> {
                let body = request.to_json()?;
                let refuse = self.refuse_second && body.contains("large-result-");
                self.checked.push(body);
                Ok(crate::contract::InputCount::Exact {
                    prompt_tokens: if refuse { 8200 } else { 100 },
                    output_tokens: 32,
                    context_limit: 8192,
                    admitted: !refuse,
                })
            }
            fn complete(
                &mut self,
                request: ModelRequest<'_>,
                cancel: &AtomicBool,
                text: &mut dyn FnMut(&str) -> io::Result<()>,
            ) -> io::Result<AssistantTurn> {
                self.inner.complete(request, cancel, text)
            }
            fn warm_compacted(&mut self, _: ModelRequest<'_>, _: &AtomicBool) -> io::Result<()> {
                panic!("automatic activation must never send an optional warm")
            }
        }
        for refuse_second in [false, true] {
            let mut c = conversation();
            c.push_user(
                value(&format!(
                    r#"{{"role":"user","content":"{}"}}"#,
                    "long-user ".repeat(2400)
                )),
                Limits::default(),
            )
            .unwrap();
            assert!(!crate::context::Usage::for_conversation(&c).fits(8192, 32));
            let calls = value(
                r#"{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{}"}},{"id":"b","type":"function","function":{"name":"read","arguments":"{}"}}]}"#,
            );
            let mut model = ExactModel {
                inner: MockModel::new(vec![response(calls, "tool_calls"), answer()]),
                checked: Vec::new(),
                refuse_second,
            };
            let mut tools = MockTools {
                replies: vec![Ok("large-result-".to_owned() + &"x".repeat(24000))].into(),
                ..MockTools::default()
            };
            let report = run(
                &mut c,
                &mut model,
                &mut tools,
                Limits {
                    max_context_tokens: 8192,
                    output_reserve_tokens: 32,
                    ..Limits::default()
                },
                &AtomicBool::new(false),
                &mut |_| {},
            );
            assert!(
                model
                    .checked
                    .iter()
                    .any(|b| b.contains("not_executed") && !b.contains("large-result-"))
            );
            assert!(
                model
                    .checked
                    .iter()
                    .any(|b| b.contains("large-result-") && b.contains("not_executed"))
            );
            if refuse_second {
                assert_eq!(report.reason, StopReason::Limit(LimitKind::Context));
                assert_eq!(tools.ids, ["a"]);
                assert!(
                    c.messages()
                        .last()
                        .unwrap()
                        .serialized_json()
                        .contains("not_executed")
                );
            } else {
                assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
                assert_eq!(tools.ids, ["a", "b"]);
            }
        }
    }

    #[test] fn generic_refusal_streaks_require_exact_calls_and_success_resets() {
        let refused = "Read refused: re-read with workspace_read first"; for (receipts, changed, expected, turns, nudges) in [(vec![refused; 6], 0, StopReason::ToolFailed, 3, 1), (vec![refused; 2], 0, StopReason::Limit(LimitKind::ModelTurns), 2, 1), (vec!["success"; 6], 0, StopReason::Finished(FinishReason::Stop), 7, 0), (vec![r#"{"ok":false,"error":"not_executed"}"#; 6], 0, StopReason::Finished(FinishReason::Stop), 7, 0), (vec![r#"{"ok":false,"error":"cancelled"}"#; 6], 0, StopReason::Finished(FinishReason::Stop), 7, 0), (vec![r#"{"ok":false,"error":"unknown_outcome"}"#; 6], 0, StopReason::Finished(FinishReason::Stop), 7, 0), (vec![r#"{"ok":false,"error":"output_limit"}"#; 6], 0, StopReason::Finished(FinishReason::Stop), 7, 0), (vec![r#"{"ok":false,"error":"tool_failed"}"#; 6], 0, StopReason::Finished(FinishReason::Stop), 7, 0), (vec![r#"{"ok":true,"text":"Read refused: source data"}"#; 6], 0, StopReason::Finished(FinishReason::Stop), 7, 0), (vec![refused, "success", refused, refused], 0, StopReason::Finished(FinishReason::Stop), 5, 1), (vec![refused; 2], 1, StopReason::Finished(FinishReason::Stop), 3, 0), (vec![refused; 2], 2, StopReason::Finished(FinishReason::Stop), 3, 0), (vec![r#"{"ok":false,"error":"invalid_arguments","retry":"JSON object"}"#; 6], 0, StopReason::ToolFailed, 3, 1)] { let mut c = conversation(); c.tool_names.insert("list".into()); let mut replies: Vec<_> = receipts.iter().enumerate().map(|(id, _)| { let name = if changed == 2 && id == 1 { "list" } else { "read" }; let args = if changed == 1 && id == 1 { " {}" } else { "{}" }; response(value(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"call-{id}","type":"function","function":{{"name":"{name}","arguments":{}}}}}]}}"#, json::quote(args))), "tool_calls") }).collect(); replies.push(answer()); let mut model = MockModel::new(replies); let mut tools = MockTools { replies: receipts.into_iter().map(|receipt| Ok(receipt.into())).collect(), ..MockTools::default() }; let limits = Limits { max_model_turns: if expected == StopReason::Limit(LimitKind::ModelTurns) { 2 } else { 61 }, ..Limits::default() }; let r = run_mock(&mut c, &mut model, &mut tools, limits); assert_eq!((r.reason, r.model_turns, r.nudges), (expected, turns, nudges)); assert_closed(&c); if nudges > 0 { assert!(c.messages.iter().any(|m| m.role() == "developer" && m.value().get("content").and_then(Value::as_str).is_some_and(|s| s.contains("Required next step from the refusal:") && (s.contains("workspace_read first") || s.contains("JSON object"))))); } }
        let mut c = conversation(); let mut model = MockModel::new(vec![response(calls(&["batch-1", "batch-2", "batch-3", "batch-4"]), "tool_calls")]); let mut tools = MockTools { replies: vec![Ok(refused.into()); 4].into(), ..MockTools::default() }; let r = run_mock(&mut c, &mut model, &mut tools, Limits::default()); assert_eq!((r.reason, r.tool_calls, r.nudges, tools.ids.len()), (StopReason::ToolFailed, 3, 1, 3)); assert_closed(&c); assert_eq!(result_code(&c.messages[c.messages.len() - 2]), "not_executed"); }
    #[test] fn observed_line_refusal_sequence_recovers_at_two_and_stops_at_three() {
        let f = crate::test_workspace::Fixture::new(); std::fs::create_dir(f.0.join("src")).unwrap(); f.write("src/billing.rs", &"}\n".repeat(63)); let w = f.workspace(); let mut c = Conversation::from_request(value(r#"{"tools":[{"type":"function","function":{"name":"workspace_edit_lines"}}],"messages":[{"role":"user","content":"fix-tests"}]}"#), Limits::default()).unwrap(); let raw = r#"{"path": "src/billing.rs", "start_line": 5, "end_line": 5, "edit_tag": "lines-18246f1c10de6ab7", "operation": "delete", "new_text": ""}"#; let mut model = MockModel::new((51..57).map(|id| response(value(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"call-{id}-0","type":"function","function":{{"name":"workspace_edit_lines","arguments":{}}}}}]}}"#, json::quote(raw))), "tool_calls")).collect()); let mut tools = crate::coding_tools::CodingTools::new(&w, crate::shell::ApprovalPolicy::Auto, |_: crate::coding_tools::Preview<'_>, _: &AtomicBool| true); let r = run_mock(&mut c, &mut model, &mut tools, Limits::default()); assert_eq!((r.reason, r.model_turns, r.tool_calls, r.nudges, model.replies.len()), (StopReason::ToolFailed, 3, 3, 1, 3)); let prompt = model.requests[2].get("messages").and_then(Value::as_array).unwrap().last().unwrap().get("content").and_then(Value::as_str).unwrap(); assert!(prompt.contains("re-read with workspace_read_lines in this activation") && prompt.contains(r#"workspace_read_lines {"path":"src/billing.rs","start_line":5,"end_line":5} first"#)); assert_eq!(c.messages.iter().filter(|m| m.role() == "developer" && m.value().get("content").and_then(Value::as_str) == Some(prompt)).count(), 1); assert_eq!(f.read("src/billing.rs"), "}\n".repeat(63)); assert_closed(&c);
    }
    #[test]
    fn partial_patch_receipt_survives_parent_stop_journal_resume_and_compaction() {
        // Byte-identical receipt arguments get one recovery, then a closed stop if ignored.
        let raw = r##"{"patch":"*** Update File: src/cli.rs\n+++ b/src/cli.rs\n@@ -12,6 +12,10 @@ enum Command {\n+    LowStock(i64),\n }\n\n impl Command {\n@@ -14,6 +18,15 @@ fn parse_args(args: &[String]) -> Result<Self, String> {\n+                \"--low-stock\" => {\n+                    i += 1;\n+                    let value = args.get(i).ok_or(\"low-stock value required\")?;\n+                    let threshold = value.parse::<i64>().map_err(|_| \"invalid low-stock threshold\")?;\n+                    if threshold < 0 {\n+                        return Err(\"low-stock threshold must be nonnegative\".into());\n+                    }\n+                    commands.push(Command::LowStock(threshold));\n+                }\n                 \"--warehouse\" => {\n                     i += 1;\n                     let value = args.get(i).ok_or(\"warehouse value required\")?;\n*** End Patch\n"}"##; let turn = |id: usize, args: &str, name: &str| response(value(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"repeat-{id}","type":"function","function":{{"name":"{name}","arguments":{}}}}}]}}"#, json::quote(args))), "tool_calls");
        for mode in 0..4 { let f = crate::test_workspace::Fixture::new(); std::fs::create_dir(f.0.join("src")).unwrap(); f.write("src/cli.rs", "original\n"); let w = f.workspace(); let mut c = Conversation::from_request(value(&r#"{"tools":[{"type":"function","function":{"name":"apply_patch"}},{"type":"function","function":{"name":"workspace_list"}}],"messages":[{"role":"user","content":"old"},{"role":"assistant","content":"historypadding"},{"role":"user","content":"recent"},{"role":"assistant","content":"Recent answer."},{"role":"user","content":"edit"}]}"#.replace("historypadding", &"Resolved old answer. ".repeat(500))), Limits::default()).unwrap(); let mut journal = crate::journal::Journal::create(&w, "patch-recovery", &c).unwrap(); let mut tools = crate::coding_tools::CodingTools::new(&w, crate::shell::ApprovalPolicy::Auto, |_: crate::coding_tools::Preview<'_>, _: &AtomicBool| true); let different = format!(" {raw}"); let mut replies: Vec<Value> = (0..3).map(|id| turn(id, raw, "apply_patch")).collect(); replies.push(if mode == 2 { turn(3, r#"{"path":"src"}"#, "workspace_list") } else { turn(3, if mode == 1 { &different } else { raw }, "apply_patch") }); replies.push(answer()); let limits = Limits { max_model_turns: if mode == 3 { 3 } else { 61 }, ..Limits::default() }; let mut model = MockModel::new(replies); let r = run_observed(&mut c, &mut model, &mut tools, limits, &AtomicBool::new(false), &mut |_| {}, &mut journal);
            assert_eq!(r.nudges, 1); assert_eq!(r.model_turns, if mode == 0 { 4 } else if mode == 3 { 3 } else { 5 }); assert_eq!(r.reason, if mode == 0 { StopReason::ToolFailed } else if mode == 3 { StopReason::Limit(LimitKind::ModelTurns) } else { StopReason::Finished(FinishReason::Stop) }); assert_eq!(c.messages.iter().filter(|m| m.role() == "developer" && m.value().get("content").and_then(Value::as_str) == Some(PATCH_RECOVERY_PROMPT)).count(), 1); assert_closed(&c); assert_eq!(f.read("src/cli.rs"), "original\n");
            if mode == 0 { let plan = crate::context::plan(&c, "Refused proposals; no writes.", 2, Limits::default()).unwrap(); journal.compact(&plan).unwrap(); drop(journal); let (_, mut resumed, _) = crate::journal::Journal::resume(&w, "patch-recovery", Limits::default()).unwrap(); resumed.push_user(value(r#"{"role":"user","content":"Continue."}"#), Limits::default()).unwrap(); let r = run_mock(&mut resumed, &mut MockModel::new(vec![turn(4, raw, "apply_patch")]), &mut tools, Limits::default()); assert_eq!((r.reason, r.model_turns, r.nudges), (StopReason::ToolFailed, 1, 0)); assert_closed(&resumed); }
        }
        use crate::{
            coding_tools::{CodingTools, Preview},
            journal::Journal,
            shell::ApprovalPolicy,
            test_workspace::Fixture,
        };
        let f = Fixture::new();
        let w = f.workspace();
        let patch = "*** Begin Patch\n*** Update File: a\n@@\n-old\n+new\n*** Update File: b\n@@\n-old\n+new\n*** End Patch";
        for (history, wire) in [
            (4096, 512 * 1024),
            (4 * 1024 * 1024, 1024),
            (10000, 512 * 1024),
        ] {
            let limits = Limits {
                max_history_bytes: history,
                wire: WireLimits {
                    max_message_bytes: wire,
                    ..WireLimits::default()
                },
                ..Limits::default()
            };
            std::fs::write(f.0.join("a"), "old\n").unwrap();
            std::fs::write(f.0.join("b"), "old\n").unwrap();
            let args = json::stringify(&Value::Object(vec![(
                "patch".into(),
                Value::String(patch.into()),
            )]))
            .unwrap();
            let turn = value(&format!(
                r#"{{"role":"assistant","tool_calls":[{{"id":"p","type":"function","function":{{"name":"apply_patch","arguments":{}}}}},{{"id":"skip","type":"function","function":{{"name":"apply_patch","arguments":{}}}}}]}}"#,
                json::quote(&args),
                json::quote(&args)
            ));
            let mut c = Conversation::from_request(value(r#"{"tools":[{"type":"function","function":{"name":"apply_patch"}}],"messages":[{"role":"user","content":"edit"}]}"#), Limits::default()).unwrap();
            let journal_name = format!("partial-{history}-{wire}");
            let mut store = Journal::create(&w, &journal_name, &c).unwrap();
            let mut model = MockModel::new(vec![response(turn, "tool_calls")]);
            let mut tools = CodingTools::new(
                &w,
                ApprovalPolicy::Auto,
                |p: Preview<'_>, _: &AtomicBool| {
                    if p.title == "File written" {
                        std::fs::write(f.0.join("b"), "changed\n").unwrap();
                    }
                    true
                },
            );
            let report = run_observed(
                &mut c,
                &mut model,
                &mut tools,
                limits,
                &AtomicBool::new(false),
                &mut |_| {},
                &mut store,
            );
            if history == 4096 || wire == 1024 {
                assert_eq!(w.read("a").unwrap(), "old\n");
                assert_eq!(w.read("b").unwrap(), "old\n");
                assert!(
                    c.messages()
                        .last()
                        .unwrap()
                        .serialized_json()
                        .contains("not_executed")
                );
                continue;
            }
            assert_eq!(report.reason, StopReason::ToolFailed);
            assert_eq!(report.tool_calls, 1);
            // The conflict on b rolls back a; the receipt stays uncertain.
            assert_eq!(w.read("a").unwrap(), "old\n");
            assert_eq!(w.read("b").unwrap(), "changed\n");
            let receipt = value(
                c.messages()[2]
                    .value()
                    .get("content")
                    .unwrap()
                    .as_str()
                    .unwrap(),
            );
            assert_eq!(receipt.get("uncertain_writes"), Some(&Value::Bool(true)));
            assert_eq!(
                receipt.get("committed").unwrap().as_array().unwrap().len(),
                2
            );
            assert_eq!(
                receipt
                    .get("affected_paths")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            assert!(
                c.messages()[3]
                    .value()
                    .get("content")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .contains("not_executed")
            );
            drop(store);
            let (_, resumed, _) = Journal::resume(&w, &journal_name, limits).unwrap();
            assert!(crate::context::plan(&resumed, "summary", 0, Limits::default()).is_err());
            assert!(crate::context::has_unknown_outcome(&resumed));
            assert_eq!(resumed.messages()[2], c.messages()[2]);
            assert!(crate::context::has_unknown_outcome_content(
                &uncertain_content(&"x".repeat(4096), 128)
            ));
        }
    }

    fn run_mock(
        conversation: &mut Conversation,
        model: &mut dyn Model,
        tools: &mut dyn ToolExecutor,
        limits: Limits,
    ) -> RunReport {
        run(
            conversation,
            model,
            tools,
            limits,
            &AtomicBool::new(false),
            &mut |_| {},
        )
    }

    fn assert_closed(conversation: &Conversation) {
        validate_history(conversation.messages()).unwrap();
        let request = conversation
            .request(WireLimits::default())
            .to_json()
            .unwrap();
        assert_eq!(request.len(), conversation.serialized_bytes());
        let restored = Conversation::from_request(value(&request), Limits::default()).unwrap();
        assert_eq!(restored.messages(), conversation.messages());
    }

    fn result_code(message: &Message) -> String {
        let content = message.value().get("content").unwrap().as_str().unwrap();
        required_string(&value(content), "error")
            .unwrap()
            .to_owned()
    }

    #[test]
    fn fragmented_mock_stream_parallel_ids_then_answer_keeps_stable_prefix() {
        struct StreamingMock {
            requests: Vec<Value>,
            turn: usize,
        }
        impl Model for StreamingMock {
            fn complete(
                &mut self,
                request: ModelRequest<'_>,
                _: &AtomicBool,
                on_text: &mut dyn FnMut(&str) -> io::Result<()>,
            ) -> io::Result<AssistantTurn> {
                self.requests.push(json::parse(&request.to_json()?)?);
                let events: &[&str] = if self.turn == 0 {
                    &[
                        r#"{"choices":[{"delta":{"content":null,"reasoning_content":"Inspect","tool_calls":[{"index":1,"id":"second","type":"function","function":{"name":"read","arguments":"{\"pa"}},{"index":0,"id":"first","type":"function","function":{"name":"read","arguments":"{\"path\":"},"future_call":true}]},"finish_reason":null}]}"#,
                        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.rs\"}"}},{"index":1,"function":{"arguments":"th\":\"b.rs\"}"}}]},"finish_reason":"tool_calls"}]}"#,
                        "[DONE]",
                    ]
                } else {
                    &[
                        r#"{"choices":[{"delta":{"content":"Both "},"finish_reason":null}]}"#,
                        r#"{"choices":[{"delta":{"content":"read 🦀."},"finish_reason":"stop"}]}"#,
                        "[DONE]",
                    ]
                };
                self.turn += 1;
                let mut stream = ChatAccumulator::new(request.wire);
                for data in events {
                    stream.push_sse_data(data, on_text)?;
                }
                stream.into_turn()
            }
        }
        let mut conversation = conversation();
        let original = conversation
            .request(WireLimits::default())
            .to_json()
            .unwrap();
        let prefix = conversation.messages().to_vec();
        let mut model = StreamingMock {
            requests: Vec::new(),
            turn: 0,
        };
        let mut tools = MockTools::default();
        let mut text = String::new();
        let report = run(
            &mut conversation,
            &mut model,
            &mut tools,
            Limits::default(),
            &AtomicBool::new(false),
            &mut |delta| text.push_str(delta),
        );
        assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
        assert_eq!(
            (report.model_turns, report.tool_turns, report.tool_calls),
            (2, 1, 2)
        );
        assert_eq!(tools.ids, ["first", "second"]);
        assert_eq!(text, "Both read 🦀.");
        assert_eq!(&conversation.messages()[..2], prefix);
        assert_eq!(conversation.prefix_len(), 1);
        let next = model.requests[1]
            .get("messages")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(
            next[3].get("tool_call_id").and_then(Value::as_str),
            Some("first")
        );
        assert_eq!(
            next[4].get("tool_call_id").and_then(Value::as_str),
            Some("second")
        );
        assert!(
            next[2].get("tool_calls").unwrap().as_array().unwrap()[0]
                .get("future_call")
                .is_some()
        );
        for key in ["model", "tools", "temperature", "provider_option", "stream"] {
            assert_eq!(model.requests[0].get(key), model.requests[1].get(key));
        }
        assert_eq!(model.requests[0], value(&original));
        assert_closed(&conversation);
    }

    #[test]
    fn unknown_denied_and_invalid_arguments_become_results_for_correct_ids() {
        let raw = value(
            r#"{"role":"assistant","content":null,"tool_calls":[
            {"id":"unknown","type":"function","function":{"name":"shell","arguments":"{}"}},
            {"id":"denied","type":"function","function":{"name":"read","arguments":"{\"path\":\"../outside\"}"}},
            {"id":"malformed","type":"function","function":{"name":"read","arguments":"{"}},
            {"id":"array","type":"function","function":{"name":"read","arguments":"[]"}}]}"#,
        );
        let mut model = MockModel::new(vec![response(raw, "tool_calls"), answer()]);
        let mut tools = MockTools {
            replies: [Err(ToolError::Denied)].into(),
            ..MockTools::default()
        };
        let mut conversation = conversation();
        let report = run_mock(&mut conversation, &mut model, &mut tools, Limits::default());
        assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
        assert_eq!(tools.ids, ["denied"]);
        let replayed = conversation.messages()[2].tool_calls();
        assert_eq!(replayed[1].arguments, r#"{"path":"../outside"}"#);
        assert_eq!(replayed[2].arguments, "{}");
        assert_eq!(replayed[3].arguments, "[]");
        for (message, (id, code)) in conversation.messages()[3..7].iter().zip([
            ("unknown", "unknown_tool"),
            ("denied", "denied"),
            ("malformed", "invalid_arguments"),
            ("array", "invalid_arguments"),
        ]) {
            assert_eq!(message.tool_call_id(), Some(id));
            assert_eq!(result_code(message), code);
        }
        assert_closed(&conversation);
    }

    #[test]
    fn cancellation_between_calls_commits_completed_result_and_never_replays_it() {
        let mut conversation = conversation();
        let mut model = MockModel::new(vec![response(calls(&["one", "two"]), "tool_calls")]);
        let mut tools = MockTools {
            cancel_after: Some(1),
            ..MockTools::default()
        };
        let cancel = AtomicBool::new(false);
        let report = run(
            &mut conversation,
            &mut model,
            &mut tools,
            Limits::default(),
            &cancel,
            &mut |_| {},
        );
        assert_eq!(report.reason, StopReason::Cancelled);
        assert_eq!(tools.ids, ["one"]);
        assert_eq!(
            conversation.messages()[3]
                .value()
                .get("content")
                .and_then(Value::as_str),
            Some("result for one")
        );
        assert_eq!(result_code(&conversation.messages()[4]), "cancelled");
        assert_closed(&conversation);
        let mut resumed = MockModel::new(vec![answer()]);
        let report = run_mock(
            &mut conversation,
            &mut resumed,
            &mut tools,
            Limits::default(),
        );
        assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
        assert_eq!(tools.ids, ["one"]);
    }

    #[test]
    fn cancellation_before_and_during_model_does_not_commit_partial_assistant() {
        let mut conversation = conversation();
        let before = conversation.messages().to_vec();
        let mut unused = MockModel::new(vec![]);
        let report = run(
            &mut conversation,
            &mut unused,
            &mut MockTools::default(),
            Limits::default(),
            &AtomicBool::new(true),
            &mut |_| {},
        );
        assert_eq!(report.reason, StopReason::Cancelled);
        assert_eq!(report.model_turns, 0);
        let mut unused = MockModel::new(vec![]); let stopped = run_mock(&mut conversation, &mut unused, &mut MockTools::default(), Limits { max_run_time: Some(Duration::from_secs(1)), ..Limits::default() }); assert_eq!((stopped.reason, stopped.model_turns, conversation.messages()), (StopReason::Limit(LimitKind::Time), 0, before.as_slice()));
        struct CancellingModel(RunBudget);
        impl Model for CancellingModel {
            fn complete(
                &mut self,
                _: ModelRequest<'_>,
                cancel: &AtomicBool,
                on_text: &mut dyn FnMut(&str) -> io::Result<()>,
            ) -> io::Result<AssistantTurn> {
                on_text("partial")?;
                self.0.state.lock().unwrap().started -= Duration::from_secs(21);
                self.0.signal_expiry(cancel);
                on_text("never displayed")?;
                panic!("callback should reject cancellation")
            }
        }
        let budget = RunBudget::new(Some(Duration::from_secs(20)));
        let pause = budget.pause();
        let active = budget.active_elapsed();
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(budget.active_elapsed(), active);
        drop(pause);
        let mut output = String::new();
        let report = run_observed_with_budget(
            &mut conversation,
            &mut CancellingModel(budget.clone()),
            &mut MockTools::default(),
            Limits::default(),
            &AtomicBool::new(false),
            &mut |s| output.push_str(s),
            &mut NoObserver,
            &budget,
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::Time));
        assert_eq!(output, "partial");
        assert_eq!(conversation.messages(), before);
    }

    #[test]
    fn bounded_model_tool_turns_and_call_counts_leave_closed_history() {
        let d = Limits::default();
        assert_eq!(
            (
                d.max_model_turns,
                d.max_tool_turns,
                d.max_tool_calls,
                d.max_run_time.unwrap().as_secs()
            ),
            (61, 60, 240, 1200)
        );
        for (limits, ids, expected, executed) in [
            (
                Limits {
                    max_model_turns: 1,
                    ..Limits::default()
                },
                vec!["one"],
                LimitKind::ModelTurns,
                1,
            ),
            (
                Limits {
                    max_tool_turns: 0,
                    ..Limits::default()
                },
                vec!["one"],
                LimitKind::ToolTurns,
                0,
            ),
            (
                Limits {
                    max_tool_calls: 1,
                    ..Limits::default()
                },
                vec!["one", "two"],
                LimitKind::ToolCalls,
                0,
            ),
        ] {
            let mut conversation = conversation();
            let mut model = MockModel::new(vec![response(calls(&ids), "tool_calls")]);
            let mut tools = MockTools::default();
            let report = run_mock(&mut conversation, &mut model, &mut tools, limits);
            assert_eq!(report.reason, StopReason::Limit(expected));
            assert_eq!(tools.ids.len(), executed);
            assert_closed(&conversation);
        }
        let mut conversation = conversation();
        let report = run_mock(
            &mut conversation,
            &mut MockModel::new(vec![]),
            &mut MockTools::default(),
            Limits {
                max_model_turns: 0,
                ..Limits::default()
            },
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::ModelTurns));
        assert_eq!(conversation.messages().len(), 2);
    }

    #[test]
    fn cumulative_tool_turn_limit_counts_across_rounds() {
        let mut conversation = conversation();
        let mut model = MockModel::new(vec![
            response(calls(&["one"]), "tool_calls"),
            response(calls(&["two"]), "tool_calls"),
        ]);
        let mut tools = MockTools::default();
        let report = run_mock(
            &mut conversation,
            &mut model,
            &mut tools,
            Limits {
                max_tool_turns: 1,
                ..Limits::default()
            },
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::ToolTurns));
        assert_eq!(tools.ids, ["one"]);
        assert_eq!(
            result_code(conversation.messages().last().unwrap()),
            "not_executed"
        );
        assert_closed(&conversation);
    }

    #[test]
    fn fatal_tool_failure_stops_batch_and_can_resume_without_replay() {
        let mut conversation = conversation();
        let mut model = MockModel::new(vec![response(calls(&["one", "two"]), "tool_calls")]);
        let mut tools = MockTools {
            replies: [Err(ToolError::Failed)].into(),
            ..MockTools::default()
        };
        let report = run_mock(&mut conversation, &mut model, &mut tools, Limits::default());
        assert_eq!(report.reason, StopReason::ToolFailed);
        assert_eq!(tools.ids, ["one"]);
        assert_eq!(result_code(&conversation.messages()[3]), "tool_failed");
        assert_eq!(result_code(&conversation.messages()[4]), "not_executed");
        assert_closed(&conversation);
        let report = run_mock(
            &mut conversation,
            &mut MockModel::new(vec![answer()]),
            &mut tools,
            Limits::default(),
        );
        assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
        assert_eq!(tools.ids, ["one"]);
    }

    #[test]
    fn upstream_failure_after_tools_retains_results_and_hides_error_details() {
        let mut conversation = conversation();
        let mut model = MockModel::new(vec![
            response(calls(&["one"]), "tool_calls"),
            value(r#"{"error":{"message":"sensitive private request"}}"#),
        ]);
        let mut tools = MockTools::default();
        let report = run_mock(&mut conversation, &mut model, &mut tools, Limits::default());
        assert_eq!(
            report.reason,
            StopReason::ModelError(io::ErrorKind::InvalidData)
        );
        assert_eq!(conversation.messages().len(), 4);
        assert_closed(&conversation);
        assert!(!format!("{report:?}").contains("sensitive"));
        assert_eq!(
            run_mock(
                &mut conversation,
                &mut MockModel::new(vec![answer()]),
                &mut tools,
                Limits::default()
            )
            .reason,
            StopReason::Finished(FinishReason::Stop)
        );
        assert_eq!(tools.ids, ["one"]);
    }

    #[test]
    fn output_limits_cover_per_call_total_and_generated_errors() {
        for limits in [
            Limits {
                max_tool_output_bytes: 128,
                ..Limits::default()
            },
            Limits {
                max_total_output_bytes: 128,
                ..Limits::default()
            },
        ] {
            let mut conversation = conversation();
            let mut model = MockModel::new(vec![response(calls(&["one", "two"]), "tool_calls")]);
            let mut tools = MockTools {
                replies: [Ok("x".repeat(129))].into(),
                ..MockTools::default()
            };
            let report = run_mock(&mut conversation, &mut model, &mut tools, limits);
            assert_eq!(report.reason, StopReason::Limit(LimitKind::ToolOutput));
            assert_eq!(tools.ids, ["one"]);
            assert!(report.output_bytes <= limits.max_total_output_bytes);
            assert_eq!(result_code(&conversation.messages()[3]), "output_limit");
            assert_eq!(result_code(&conversation.messages()[4]), "not_executed");
            assert_closed(&conversation);
        }
        let mut conversation = conversation();
        let mut tools = MockTools::default();
        let report = run_mock(
            &mut conversation,
            &mut MockModel::new(vec![response(calls(&["one", "two"]), "tool_calls")]),
            &mut tools,
            Limits {
                max_total_output_bytes: 127,
                ..Limits::default()
            },
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::ToolOutput));
        assert!(tools.ids.is_empty());
        assert_eq!(conversation.messages().len(), 2);
    }

    #[test]
    fn history_budget_preflights_calls_and_handles_escaped_output_growth() {
        let mut initial = conversation();
        let bytes = initial.serialized_bytes();
        let mut tools = MockTools::default();
        let report = run_mock(
            &mut initial,
            &mut MockModel::new(vec![response(calls(&["one"]), "tool_calls")]),
            &mut tools,
            Limits {
                max_history_bytes: bytes,
                ..Limits::default()
            },
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::History));
        assert!(tools.ids.is_empty());
        assert_eq!(initial.serialized_bytes(), bytes);

        let mut baseline = conversation();
        let mut denied = MockTools {
            replies: [Err(ToolError::Denied)].into(),
            ..MockTools::default()
        };
        run_mock(
            &mut baseline,
            &mut MockModel::new(vec![response(calls(&["one"]), "tool_calls")]),
            &mut denied,
            Limits {
                max_model_turns: 1,
                ..Limits::default()
            },
        );
        let budget = baseline.serialized_bytes();
        let mut conversation = conversation();
        let mut tools = MockTools {
            replies: [Ok("\0".repeat(64))].into(),
            ..MockTools::default()
        };
        let report = run_mock(
            &mut conversation,
            &mut MockModel::new(vec![response(calls(&["one"]), "tool_calls")]),
            &mut tools,
            Limits {
                max_history_bytes: budget,
                ..Limits::default()
            },
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::History));
        assert!(tools.ids.is_empty());
        assert!(conversation.serialized_bytes() <= budget);
        assert_eq!(
            result_code(conversation.messages().last().unwrap()),
            "output_limit"
        );
        assert_closed(&conversation);
    }

    #[test]
    fn terminal_finish_does_not_execute_partial_or_unrecognized_tool_calls() {
        for reason in ["length", "content_filter", "future_finish"] {
            let mut conversation = conversation();
            let mut tools = MockTools::default();
            let report = run_mock(
                &mut conversation,
                &mut MockModel::new(vec![response(calls(&["one"]), reason); 4]),
                &mut tools,
                Limits::default(),
            );
            assert_eq!(
                report.reason,
                if reason == "length" {
                    StopReason::TruncatedToolCall
                } else {
                    StopReason::Finished(FinishReason::from_wire(reason))
                }
            );
            assert!(tools.ids.is_empty());
            if reason == "length" {
                assert_eq!((report.model_turns, report.nudges), (4, 3));
                let message = conversation.messages().last().unwrap();
                assert_eq!(message.role(), "assistant");
                assert!(message.tool_calls().is_empty());
                assert!(message.value().get("vade_truncated").is_none());
                let mut repaired = self::conversation();
                let mut model = MockModel::new(vec![response(calls(&["discarded"]), "length"), response(calls(&["retry"]), "tool_calls"), answer()]);
                let report = run_mock(&mut repaired, &mut model, &mut tools, Limits::default());
                assert_eq!((report.reason, report.model_turns, report.nudges, tools.ids.clone()), (StopReason::Finished(FinishReason::Stop), 3, 1, vec!["retry".to_owned()]));
                assert!(json::stringify(&model.requests[1]).unwrap().contains("read\\\" targeting \\\"sample.rs") && json::stringify(&model.requests[1]).unwrap().contains("4096-token reply limit")); assert_closed(&repaired);
            } else {
                assert_eq!(
                    result_code(conversation.messages().last().unwrap()),
                    "not_executed"
                );
            }
            assert_closed(&conversation);
        }
    }

    #[test]
    fn duplicate_ids_and_invalid_initial_history_never_execute() {
        let mut conversation = conversation();
        let mut model = MockModel::new(vec![
            response(calls(&["same"]), "tool_calls"),
            response(calls(&["same"]), "tool_calls"),
        ]);
        let mut tools = MockTools::default();
        let report = run_mock(&mut conversation, &mut model, &mut tools, Limits::default());
        assert_eq!(report.reason, StopReason::InvalidResponse);
        assert_eq!(tools.ids, ["same"]);
        assert_closed(&conversation);
        for raw in [
            r#"{"messages":[{"role":"tool","tool_call_id":"orphan","content":"x"}]}"#,
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"x","type":"function","function":{"name":"read","arguments":"{}"}}]}]}"#,
            r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"x","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"user","content":"interrupt"}]}"#,
            r#"{"messages":[],"n":2}"#,
        ] {
            assert!(Conversation::from_request(value(raw), Limits::default()).is_err());
        }
    }

    #[test]
    fn structured_user_append_and_request_options_are_lossless() {
        let mut conversation = conversation();
        let options = conversation.options().clone();
        let before = conversation.messages().to_vec();
        let next = value(
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,mock"}}],"future_user":7}"#,
        );
        conversation
            .push_user(next.clone(), Limits::default())
            .unwrap();
        assert_eq!(conversation.options(), &options);
        assert_eq!(&conversation.messages()[..2], before);
        assert_eq!(conversation.messages()[2].value(), &next);
        assert!(
            conversation
                .push_user(
                    value(r#"{"role":"system","content":"replacement"}"#),
                    Limits::default()
                )
                .is_err()
        );
        assert_closed(&conversation);
    }

    #[test]
    fn workspace_adapter_delegates_only_read_list_and_redacts_errors() {
        let mut tools = WorkspaceTools::new(
            |path: &str, _: usize, _: usize, _: &AtomicBool| {
                if path == "safe.rs" {
                    Ok("safe text".to_owned())
                } else {
                    Err(io::Error::other("private path detail"))
                }
            },
            |path: &str| {
                if path == "." {
                    Ok(vec!["a".into(), "b".into()])
                } else {
                    Err(io::ErrorKind::PermissionDenied.into())
                }
            },
        );
        assert_eq!(tools.definitions().as_array().unwrap().len(), 2);
        let cancel = AtomicBool::new(false);
        for (name, path, expected) in [
            ("workspace_read", "safe.rs", Ok("safe text".into())),
            ("workspace_list", ".", Ok("a\nb".into())),
            ("workspace_read", "../outside", Err(ToolError::Denied)),
            ("shell", ".", Err(ToolError::UnknownTool)),
        ] {
            let call = ToolCall {
                id: "x".into(),
                name: name.into(),
                arguments: String::new(),
            };
            let arguments = Value::Object(vec![("path".into(), Value::String(path.into()))]);
            assert_eq!(
                tools.execute(
                    ToolInvocation {
                        call: &call,
                        arguments: &arguments,
                        max_output_bytes: 64
                    },
                    &cancel
                ),
                expected
            );
        }
        let call = ToolCall {
            id: "x".into(),
            name: "workspace_read".into(),
            arguments: String::new(),
        };
        let arguments = value(r#"{"path":"safe.rs","shell":"anything"}"#);
        assert_eq!(
            tools.execute(
                ToolInvocation {
                    call: &call,
                    arguments: &arguments,
                    max_output_bytes: 64
                },
                &cancel
            ),
            Err(ToolError::InvalidArguments("workspace_read/list: unexpected fields; list requires only path"))
        );
        let arguments = value(r#"{"path":"safe.rs"}"#);
        assert_eq!(
            tools.execute(
                ToolInvocation {
                    call: &call,
                    arguments: &arguments,
                    max_output_bytes: 2
                },
                &cancel
            ),
            Err(ToolError::OutputLimit)
        );
        cancel.store(true, Ordering::Relaxed);
        assert_eq!(
            tools.execute(
                ToolInvocation {
                    call: &call,
                    arguments: &arguments,
                    max_output_bytes: 64
                },
                &cancel
            ),
            Err(ToolError::Cancelled)
        );
    }

    #[test]
    fn cumulative_call_budget_counts_rejected_proposals_too() {
        let mut conversation = conversation();
        let unknown = value(
            r#"{"role":"assistant","tool_calls":[{"id":"unknown","type":"function","function":{"name":"not_advertised","arguments":"{}"}}]}"#,
        );
        let mut model = MockModel::new(vec![
            response(unknown, "tool_calls"),
            response(calls(&["next"]), "tool_calls"),
        ]);
        let mut tools = MockTools::default();
        let report = run_mock(
            &mut conversation,
            &mut model,
            &mut tools,
            Limits {
                max_tool_calls: 1,
                ..Limits::default()
            },
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::ToolCalls));
        assert!(tools.ids.is_empty());
        assert_eq!(
            result_code(conversation.messages().last().unwrap()),
            "not_executed"
        );
        assert_closed(&conversation);
    }

    #[test]
    fn invalid_configuration_and_ignored_callback_failure_cannot_execute() {
        let mut conversation = conversation();
        let mut model = MockModel::new(vec![]);
        let mut tools = MockTools::default();
        let report = run_mock(
            &mut conversation,
            &mut model,
            &mut tools,
            Limits {
                max_tool_output_bytes: 0,
                ..Limits::default()
            },
        );
        assert_eq!(report.reason, StopReason::InvalidConfiguration);
        assert!(model.requests.is_empty());
        struct IgnoringModel;
        impl Model for IgnoringModel {
            fn complete(
                &mut self,
                request: ModelRequest<'_>,
                _: &AtomicBool,
                on_text: &mut dyn FnMut(&str) -> io::Result<()>,
            ) -> io::Result<AssistantTurn> {
                let _ = on_text(&"x".repeat(request.wire.max_message_bytes + 1));
                AssistantTurn::from_response(
                    &response(calls(&["should_not_run"]), "tool_calls"),
                    request.wire,
                )
            }
        }
        let report = run_mock(
            &mut conversation,
            &mut IgnoringModel,
            &mut tools,
            Limits::default(),
        );
        assert_eq!(report.reason, StopReason::InvalidResponse);
        assert_eq!(conversation.messages().len(), 2);
        assert!(tools.ids.is_empty());
    }

    #[test]
    fn aggregate_json_complexity_cannot_make_history_unreadable() {
        let mut conversation = conversation();
        let user = Value::Object(vec![
            ("role".into(), Value::String("user".into())),
            ("content".into(), Value::String("a".into())),
            ("future".into(), Value::Array(vec![Value::Null; 20_000])),
        ]);
        conversation
            .push_user(user.clone(), Limits::default())
            .unwrap();
        let before = conversation.messages().to_vec();
        assert!(conversation.push_user(user, Limits::default()).is_err());
        assert_eq!(conversation.messages(), before);
        let large_assistant = Value::Object(vec![
            ("role".into(), Value::String("assistant".into())),
            ("content".into(), Value::String("answer".into())),
            ("future".into(), Value::Array(vec![Value::Null; 20_000])),
        ]);
        let report = run_mock(
            &mut conversation,
            &mut MockModel::new(vec![response(large_assistant, "stop")]),
            &mut MockTools::default(),
            Limits {
                max_context_tokens: 131072,
                ..Limits::default()
            },
        );
        assert_eq!(report.reason, StopReason::Limit(LimitKind::History));
        assert_eq!(conversation.messages(), before);
        assert_closed(&conversation);
        let mut nested = Value::Null;
        for _ in 0..49 {
            nested = Value::Array(vec![nested]);
        }
        assert!(
            Message::from_value(
                Value::Object(vec![
                    ("role".into(), Value::String("assistant".into())),
                    ("future".into(), nested)
                ]),
                WireLimits::default()
            )
            .is_err()
        );
    }

    #[test] fn varied_discovery_budget_resets_only_on_applied_edits_and_survives_resume() {
        let turn = |id: usize, edit: bool| { let name = if edit { "apply_patch" } else { ["workspace_search", "workspace_read", "workspace_list", "workspace_symbols", "workspace_manifest", "workspace_read_lines"][id % 6] }; response(value(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"b{id}","type":"function","function":{{"name":"{name}","arguments":"{{}}"}}}}]}}"#)), "tool_calls") }; for (first, edit, last, nudges, stopped) in [(11, false, 0, 0, false), (12, false, 0, 1, false), (24, false, 0, 1, true), (12, true, 11, 1, false), (12, true, 12, 2, false), (11, true, 11, 0, false)] { let definitions = ["apply_patch", "workspace_search", "workspace_read", "workspace_list", "workspace_symbols", "workspace_manifest", "workspace_read_lines"].map(|name| format!(r#"{{"type":"function","function":{{"name":"{name}"}}}}"#)).join(","); let mut c = Conversation::from_request(value(&format!(r#"{{"tools":[{definitions}],"messages":[{{"role":"user","content":"write findings"}}]}}"#)), Limits::default()).unwrap(); let mut replies: Vec<_> = (0..first).map(|id| turn(id, false)).collect(); let mut receipts: VecDeque<_> = (0..first).map(|id| Ok(format!("fresh evidence {id}"))).collect(); if edit { replies.push(turn(first, true)); receipts.push_back(Ok("Applied 1 file operations.".into())); } replies.extend((0..last).map(|id| turn(first + 1 + id, false))); replies.push(answer()); let mut model = MockModel::new(replies); let mut tools = MockTools { replies: receipts, ..MockTools::default() }; let r = run_mock(&mut c, &mut model, &mut tools, Limits::default()); assert_eq!((r.nudges, r.reason), (nudges, if stopped { StopReason::ToolFailed } else { StopReason::Finished(FinishReason::Stop) })); assert_closed(&c); if first == 12 && !edit { let f = crate::test_workspace::Fixture::new(); let mut journal = crate::journal::Journal::create(&f.workspace(), "budget", &c).unwrap(); journal.save(&c).unwrap(); drop(journal); let (_, mut resumed, _) = crate::journal::Journal::resume(&f.workspace(), "budget", Limits::default()).unwrap(); resumed.push_user(value(r#"{"role":"user","content":"Continue."}"#), Limits::default()).unwrap(); let mut replies: Vec<_> = (30..42).map(|id| turn(id, false)).collect(); replies.push(answer()); let r = run_mock(&mut resumed, &mut MockModel::new(replies), &mut MockTools::default(), Limits::default()); assert_eq!((r.nudges, r.tool_calls, r.reason), (0, 12, StopReason::ToolFailed)); assert_closed(&resumed); } }
    }
    #[test] fn every_harness_prompt_appends_developer_without_rewriting_request_prefix() {
        let mut c = conversation(); let original = c.request(Limits::default().wire).to_json().unwrap(); for prompt in [CONTINUATION_PROMPT, TOOL_CONTINUATION_PROMPT, REPLY_BUDGET_PROMPT, EMPTY_REPLY_PROMPT, CHECK_FAILING_PROMPT, SELF_REVIEW_PROMPT, PATCH_RECOVERY_PROMPT, DISCOVERY_RECOVERY_PROMPT, "dynamic refusal/check repair"] { let before: Vec<_> = c.messages().iter().map(|m| m.serialized_json().to_owned()).collect(); super::execution::push_harness_prompt(&mut c, &mut NoObserver, Limits::default(), prompt).unwrap(); assert_eq!(before, c.messages()[..before.len()].iter().map(|m| m.serialized_json().to_owned()).collect::<Vec<_>>()); let last = c.messages().last().unwrap(); assert_eq!(last.role(), "developer"); assert_eq!(last.value().get("content").and_then(Value::as_str), Some(prompt)); } let mut now = value(&c.request(Limits::default().wire).to_json().unwrap()); let original = value(&original); if let Value::Object(fields) = &mut now { fields.iter_mut().find(|(k, _)| k == "messages").unwrap().1 = original.get("messages").unwrap().clone(); } assert_eq!(now, original);
    }
    #[test] fn invalid_argument_reasons_reach_closed_receipts_with_bounded_fallback() {
        let f = crate::test_workspace::Fixture::new(); let w = f.workspace(); let mut tools = crate::coding_tools::CodingTools::new(&w, crate::shell::ApprovalPolicy::Auto, |_, _| panic!("invalid arguments must not reach side effects")); for (name, args, reason) in [("workspace_read", r#"{"path":7}"#, "path must be a string"), ("workspace_read", r#"{"path":"x","offset":false}"#, "offset must be a nonnegative integer"), ("workspace_read", r#"{"path":"x","limit":false}"#, "limit must be a nonnegative integer"), ("workspace_search", r#"{"path":".","literal":7}"#, "provide exactly one of literal (string) or literals (string array)"), ("workspace_read_lines", r#"{"path":".","ranges":[7]}"#, "each ranges entry must be an object"), ("workspace_manifest", r#"{"path":".","depth":false}"#, "depth must be a nonnegative integer"), ("workspace_edit_lines", r#"{"path":"x","new_text":7}"#, "new_text must be a string"), ("tool_output_read", r#"{"id":7,"offset":0,"length":1}"#, "id must be a string"), ("workspace_shell", r#"{"argv":["true"],"timeout_ms":"1"}"#, "timeout_ms must be a nonnegative integer"), ("workspace_shell", r#"{"argv":[7],"timeout_ms":1}"#, "argv must contain only strings"), ("workspace_shell", r#"{"argv":[],"timeout_ms":1}"#, "argv must be nonempty and bounded; timeout_ms must be within the command limit")] { if name == "workspace_shell" && !cfg!(target_os = "macos") { continue; } let mut c = conversation(); c.tool_names.insert(name.into()); let turn = response(value(&format!(r#"{{"role":"assistant","tool_calls":[{{"id":"bad","type":"function","function":{{"name":"{name}","arguments":{}}}}}]}}"#, json::quote(args))), "tool_calls"); let r = run_mock(&mut c, &mut MockModel::new(vec![turn, answer()]), &mut tools, Limits::default()); assert_eq!(r.reason, StopReason::Finished(FinishReason::Stop)); let receipt = c.messages().iter().find(|m| m.role() == "tool").unwrap().value().get("content").and_then(Value::as_str).unwrap(); assert_eq!(value(receipt).get("message").and_then(Value::as_str), Some(reason)); assert_closed(&c); } for (reason, cap) in [("quote \" and newline\n", 128), ("bounded reason", 64), ("x".repeat(161).leak(), 4096)] { let content = invalid_arguments_content(reason, cap); assert!(content.len() <= cap); assert_eq!(value(&content).get("error").and_then(Value::as_str), Some("invalid_arguments")); if cap == 64 || reason.len() > 160 { assert_eq!(content, error_content("invalid_arguments")); } else { assert_eq!(value(&content).get("message").and_then(Value::as_str), Some(reason)); } }
    }

    #[test]
    fn missing_submission_gets_one_journaled_continuation() {
        struct Missing; impl ToolExecutor for Missing { fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> { unreachable!() } fn missing_deliverables(&self, _: &str) -> Vec<String> { vec!["nonempty file answers.txt".into()] } }
        let mut c = conversation(); let r = run_mock(&mut c, &mut MockModel::new(vec![answer(), answer()]), &mut Missing, Limits::default()); assert_eq!((r.reason, r.model_turns, r.nudges), (StopReason::IncompleteSubmission, 2, 1)); assert!(c.messages().iter().any(|m| m.serialized_json().contains("nonempty file answers.txt"))); let r = run_mock(&mut c, &mut MockModel::new(vec![answer()]), &mut Missing, Limits::default()); assert_eq!((r.reason, r.model_turns, r.nudges), (StopReason::IncompleteSubmission, 1, 0)); c.push_user(value(r#"{"role":"user","content":"Write other.txt."}"#), Limits::default()).unwrap(); let r = run_mock(&mut c, &mut MockModel::new(vec![answer(), answer()]), &mut Missing, Limits::default()); assert_eq!((r.reason, r.model_turns, r.nudges), (StopReason::IncompleteSubmission, 2, 1));
    }

    #[test]
    fn proposed_round_compacts_before_tools_and_journals_the_rewrite() {
        struct Counted(MockModel); impl Model for Counted { fn complete(&mut self, r: ModelRequest<'_>, c: &AtomicBool, f: &mut dyn FnMut(&str)->io::Result<()>) -> io::Result<AssistantTurn> { self.0.complete(r,c,f) } fn count_input(&mut self, r: ModelRequest<'_>, _: &AtomicBool) -> io::Result<crate::contract::InputCount> { Ok(crate::contract::InputCount::Exact { prompt_tokens: r.to_json()?.len() as u64 / 4, output_tokens: 128, context_limit: 9000, admitted: true }) } }
        struct Compact(crate::journal::Journal, usize); impl RunObserver for Compact { fn checkpoint(&mut self, c: &Conversation, m: &[Message]) -> io::Result<()> { self.0.checkpoint(c,m) } fn before_counted_request(&mut self, c: &mut Conversation, l: Limits, _: &AtomicBool, u: &crate::context::Usage, count: &mut dyn FnMut(&Conversation)->io::Result<crate::context::Usage>) -> io::Result<bool> { if u.fits(l.max_context_tokens,l.output_reserve_tokens) { return Ok(false); } let (p, _) = crate::context::auto_plan_fitting(c,2,l,u,count)?.expect("fitting closed rounds"); self.0.compact(&p)?; *c=p.into_conversation(); self.1+=1; Ok(true) } }
        let mut c = conversation(); for i in 0..10 { let call = Message::from_value(calls(&[&format!("old{i}")]), WireLimits::default()).unwrap(); c.commit(vec![call, Message::tool_result(&format!("old{i}"), "x".repeat(if i<7 {4000} else {4}), WireLimits::default()).unwrap()]); } let mut big = calls(&["new"]); if let Value::Object(ref mut fields) = big { fields.iter_mut().find(|(k,_)| k=="content").unwrap().1=Value::String("p".repeat(10000)); } let f=crate::test_workspace::Fixture::new(); let w=f.workspace(); let mut observer=Compact(crate::journal::Journal::create(&w,"pressure",&c).unwrap(),0); let mut model=Counted(MockModel::new(vec![response(big,"tool_calls"),answer()])); let mut tools=MockTools::default(); let r=run_observed(&mut c,&mut model,&mut tools,Limits { max_context_tokens:9000,output_reserve_tokens:128,..Limits::default() },&AtomicBool::new(false),&mut |_|{},&mut observer); assert_eq!(r.reason,StopReason::Finished(FinishReason::Stop)); assert_eq!((observer.1,tools.ids.len()),(1,1)); assert_closed(&c); drop(observer); let (_, resumed, _)=crate::journal::Journal::resume(&w,"pressure",Limits::default()).unwrap(); assert_eq!(resumed.messages(),c.messages());
    }

    /// An automatic reply: the count gives the reply all the room the window
    /// leaves, so a smaller candidate count chooses a larger reply.
    struct Clamped { inner: MockModel, window: usize, chosen: usize }
    impl Model for Clamped {
        fn complete(&mut self, r: ModelRequest<'_>, c: &AtomicBool, f: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<AssistantTurn> { self.inner.complete(r, c, f) }
        fn count_input(&mut self, r: ModelRequest<'_>, _: &AtomicBool) -> io::Result<crate::contract::InputCount> {
            let tokens = r.to_json()?.len() / 4;
            self.chosen = self.window.saturating_sub(tokens + 256).min(16_384);
            Ok(crate::contract::InputCount::Exact { prompt_tokens: tokens as u64, output_tokens: self.chosen as u64, context_limit: self.window as u64, admitted: tokens + self.chosen <= self.window })
        }
        fn output_reserve_tokens(&self, _: usize) -> usize { self.chosen.max(crate::context::reply_reservation(self.window, 16_384)) }
        fn reply_limit_tokens(&self, _: usize) -> usize { self.chosen }
    }

    #[test]
    fn a_declined_compaction_keeps_the_reply_room_of_the_real_count() {
        // The observer counts a small candidate and then keeps the history. The
        // fit check must use the reply room of the real prompt's count, not the
        // larger one the candidate count chose.
        struct Probe; impl RunObserver for Probe { fn checkpoint(&mut self, _: &Conversation, _: &[Message]) -> io::Result<()> { Ok(()) } fn before_hosted_request(&mut self, _: &mut Conversation, _: Limits, _: &AtomicBool, _: &crate::context::Usage, host: &mut dyn crate::context::ContextHost) -> io::Result<bool> { host.count(&conversation())?; Ok(false) } }
        let mut c = conversation();
        c.push_user(value(&format!(r#"{{"role":"user","content":"{}"}}"#, "p".repeat(40_000))), Limits::default()).unwrap();
        let tokens = c.request(WireLimits::default()).to_json().unwrap().len() / 4;
        let window = tokens + 1256;
        let mut model = Clamped { inner: MockModel::new(vec![answer()]), window, chosen: 0 };
        let limits = Limits { max_context_tokens: window, output_reserve_tokens: 128, ..Limits::default() };
        let r = run_observed(&mut c, &mut model, &mut MockTools::default(), limits, &AtomicBool::new(false), &mut |_| {}, &mut Probe);
        assert_eq!(r.reason, StopReason::Finished(FinishReason::Stop));
    }

    #[test]
    fn a_failed_summary_in_the_observer_is_a_model_error() {
        // A summary that fails without a cancel is a model error, not a failed save.
        struct Summary; impl RunObserver for Summary { fn checkpoint(&mut self, _: &Conversation, _: &[Message]) -> io::Result<()> { Ok(()) } fn before_hosted_request(&mut self, c: &mut Conversation, _: Limits, _: &AtomicBool, _: &crate::context::Usage, host: &mut dyn crate::context::ContextHost) -> io::Result<bool> { host.summarize(c).map(|_| false) } }
        struct Fails; impl Model for Fails {
            fn complete(&mut self, _: ModelRequest<'_>, _: &AtomicBool, _: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<AssistantTurn> { Err(io::Error::new(io::ErrorKind::Interrupted, "time limit")) }
            fn count_input(&mut self, r: ModelRequest<'_>, _: &AtomicBool) -> io::Result<crate::contract::InputCount> { Ok(crate::contract::InputCount::Exact { prompt_tokens: r.to_json()?.len() as u64 / 4, output_tokens: 128, context_limit: 90_000, admitted: true }) }
        }
        let r = run_observed(&mut conversation(), &mut Fails, &mut MockTools::default(), Limits::default(), &AtomicBool::new(false), &mut |_| {}, &mut Summary);
        assert_eq!(r.reason, StopReason::ModelError(io::ErrorKind::Interrupted));
    }

    #[test]
    fn a_failed_summary_before_staged_tools_is_a_model_error() {
        // The first observer call keeps the history; the second one, with the
        // proposed calls staged, asks for a summary that fails.
        struct Second(usize); impl RunObserver for Second { fn checkpoint(&mut self, _: &Conversation, _: &[Message]) -> io::Result<()> { Ok(()) } fn before_hosted_request(&mut self, c: &mut Conversation, _: Limits, _: &AtomicBool, _: &crate::context::Usage, host: &mut dyn crate::context::ContextHost) -> io::Result<bool> { self.0 += 1; if self.0 == 1 { Ok(false) } else { host.summarize(c).map(|_| false) } } }
        struct Once(MockModel, usize); impl Model for Once {
            fn complete(&mut self, r: ModelRequest<'_>, c: &AtomicBool, f: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<AssistantTurn> { self.1 += 1; if self.1 == 1 { self.0.complete(r, c, f) } else { Err(io::Error::new(io::ErrorKind::Interrupted, "time limit")) } }
            // Every count is near the window and still fits.
            fn count_input(&mut self, r: ModelRequest<'_>, _: &AtomicBool) -> io::Result<crate::contract::InputCount> { let tokens = r.to_json()?.len() as u64 / 4; Ok(crate::contract::InputCount::Exact { prompt_tokens: tokens, output_tokens: 128, context_limit: tokens + 140, admitted: true }) }
        }
        let mut model = Once(MockModel::new(vec![response(calls(&["a"]), "tool_calls")]), 0);
        let mut tools = MockTools::default();
        let r = run_observed(&mut conversation(), &mut model, &mut tools, Limits { output_reserve_tokens: 128, ..Limits::default() }, &AtomicBool::new(false), &mut |_| {}, &mut Second(0));
        assert_eq!((r.reason, tools.ids.len()), (StopReason::ModelError(io::ErrorKind::Interrupted), 0));
    }

    #[test]
    fn a_handled_summary_failure_does_not_hide_a_failed_save() {
        // The summary reply is refused and local excerpts take over; the save
        // then fails. That is a journal error, not the summary's model error.
        struct Refused; impl RunObserver for Refused { fn checkpoint(&mut self, _: &Conversation, _: &[Message]) -> io::Result<()> { Ok(()) } fn before_hosted_request(&mut self, c: &mut Conversation, _: Limits, _: &AtomicBool, _: &crate::context::Usage, host: &mut dyn crate::context::ContextHost) -> io::Result<bool> { assert!(host.summarize(c).is_err()); Err(io::Error::other("disk full")) } }
        struct Fails; impl Model for Fails {
            fn complete(&mut self, _: ModelRequest<'_>, _: &AtomicBool, _: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<AssistantTurn> { Err(io::Error::other("the summary reply called a tool")) }
            fn count_input(&mut self, r: ModelRequest<'_>, _: &AtomicBool) -> io::Result<crate::contract::InputCount> { Ok(crate::contract::InputCount::Exact { prompt_tokens: r.to_json()?.len() as u64 / 4, output_tokens: 128, context_limit: 90_000, admitted: true }) }
        }
        let r = run_observed(&mut conversation(), &mut Fails, &mut MockTools::default(), Limits::default(), &AtomicBool::new(false), &mut |_| {}, &mut Refused);
        assert_eq!(r.reason, StopReason::JournalError);
    }

    #[path = "recovery_tests.rs"]
    mod recovery;
    #[path = "loop_tests.rs"]
    mod loop_guards;
