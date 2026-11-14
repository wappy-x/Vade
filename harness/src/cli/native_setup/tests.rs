use super::*;
use std::sync::atomic::AtomicBool;
use vade_harness::{
    agent::{Model, ModelRequest, ToolError, ToolExecutor, ToolInvocation},
    agent_wire::AssistantTurn,
    json,
};

#[test]
fn cancelled_candidate_counts_preserve_history_and_journal_and_retry_next_turn() {
    let path = std::env::temp_dir().join(format!(
        "vade-compaction-count-cancel-{}",
        std::process::id()
    ));
    std::fs::create_dir(&path).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    let workspace = Workspace::new(&path).unwrap();
    let limits = Limits {
        max_context_tokens: 16384,
        output_reserve_tokens: 128,
        ..Limits::default()
    };
    assert!(["apply it by the third model turn", "minimal compiling file with apply_patch", "Keep thinking available to the user", "answers in chat do not create that file", "Pure read-only questions"].iter().all(|rule| native_system_prompt(false, &BTreeMap::new(), limits).contains(rule)));
    let request = format!(
        r#"{{"model":"mock","messages":[{{"role":"system","content":"stable"}},{{"role":"user","content":"old constraint"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"recent"}},{{"role":"assistant","content":"latest answer"}},{{"role":"user","content":"current"}}]}}"#,
        "old resolved prose ".repeat(200)
    );
    let original = Conversation::from_request(json::parse(&request).unwrap(), limits).unwrap();
    let counted = |c: &Conversation, tokens, admitted| {
        context::Usage::for_conversation(c)
            .with_count(&vade_harness::contract::InputCount::Exact {
                prompt_tokens: tokens,
                output_tokens: 128,
                context_limit: 16384,
                admitted,
            })
            .unwrap()
    };
    // Cancel concurrently with a fitting count, a refused count, or a count
    // failure. Neither a compacted transcript nor a failure receipt is due.
    for case in 0..3 {
        let mut c = original.clone();
        let journal = Arc::new(Mutex::new(
            Journal::create(&workspace, &format!("cancel-{case}"), &c).unwrap(),
        ));
        let journal_path = path.join(format!(".vade/sessions/cancel-{case}.journal"));
        let bytes = std::fs::read(&journal_path).unwrap();
        let records = journal.lock().unwrap().record_count();
        let (tx, rx) = crate::interaction::lanes();
        let mut observer = CompactingJournal {
            answer: String::new(),
            tool_prose: Arc::default(),
            cancel: Arc::default(),
            inner: SharedJournal(Arc::clone(&journal)),
            events: Arc::new(tx),
            enabled: true,
            trigger: Arc::new(Mutex::new(context::AutoCompaction::default())),
        };
        let cancel = AtomicBool::new(false);
        let before = counted(&original, 14000, true);
        let result = observer.before_counted_request(
            &mut c,
            limits,
            &cancel,
            &before,
            &mut |candidate| {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                match case {
                    0 => Ok(counted(candidate, 1000, true)),
                    1 => Ok(counted(candidate, 16384, false)),
                    _ => Err(io::Error::other("mock count failure after cancellation")),
                }
            },
        );
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Interrupted);
        assert_eq!(
            c.request(limits.wire).to_json().unwrap(),
            original.request(limits.wire).to_json().unwrap()
        );
        assert_eq!(journal.lock().unwrap().generation(), 0);
        assert_eq!(journal.lock().unwrap().record_count(), records);
        assert_eq!(std::fs::read(&journal_path).unwrap(), bytes);
        assert!(matches!(
            rx.fact(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert!(observer.trigger.lock().unwrap().can_attempt_next_turn());

        // Retry eligibility lets the next prompt be admitted, but does
        // not retry an automatic attempt within the cancelled turn.
        cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        assert!(
            !observer
                .before_counted_request(&mut c, limits, &cancel, &before, &mut |_| panic!(
                    "cancelled plan must stay disarmed in the same turn"
                ))
                .unwrap()
        );
        assert_eq!(std::fs::read(&journal_path).unwrap(), bytes);

        observer.trigger.lock().unwrap().begin_turn();
        assert!(
            observer
                .before_counted_request(&mut c, limits, &cancel, &before, &mut |candidate| Ok(
                    counted(candidate, 10000, true)
                ))
                .unwrap()
        );
        assert_ne!(
            c.request(limits.wire).to_json().unwrap(),
            original.request(limits.wire).to_json().unwrap()
        );
        assert_eq!(journal.lock().unwrap().generation(), 1);
        assert_eq!(journal.lock().unwrap().record_count(), records + 1);
        assert!(
            matches!(rx.fact().unwrap(), super::super::events::WorkerEvent::ToolPreview { detail, .. } if detail.contains("Auto-compacted older results: 14000 → 10000 exact input tokens"))
        );
        assert!(matches!(
            rx.fact(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        // A successful retry above 60% restores normal hysteresis.
        assert!(!observer.trigger.lock().unwrap().can_attempt_next_turn());
    }
}

#[test]
fn failed_counted_plans_record_the_reason_and_retry_on_the_next_turn() {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "vade-compaction-retry-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir(&path).unwrap();
    let workspace = Workspace::new(&path).unwrap();
    let limits = Limits {
        max_context_tokens: 16384,
        output_reserve_tokens: 128,
        ..Limits::default()
    };
    let request = format!(
        r#"{{"model":"mock","messages":[{{"role":"system","content":"stable"}},{{"role":"user","content":"old constraint"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"recent"}},{{"role":"assistant","content":"latest answer"}},{{"role":"user","content":"current"}}]}}"#,
        "old resolved prose ".repeat(200)
    );
    let original = Conversation::from_request(json::parse(&request).unwrap(), limits).unwrap();
    let counted = |c: &Conversation, tokens, admitted| {
        context::Usage::for_conversation(c)
            .with_count(&vade_harness::contract::InputCount::Exact {
                prompt_tokens: tokens,
                output_tokens: 128,
                context_limit: 16384,
                admitted,
            })
            .unwrap()
    };
    for failure in 0..3 {
        let mut c = original.clone();
        let journal = Arc::new(Mutex::new(
            Journal::create(&workspace, &format!("retry-{failure}"), &c).unwrap(),
        ));
        let (tx, rx) = crate::interaction::lanes();
        let mut observer = CompactingJournal {
            answer: String::new(),
            tool_prose: Arc::default(),
            cancel: Arc::default(),
            inner: SharedJournal(Arc::clone(&journal)),
            events: Arc::new(tx),
            enabled: true,
            trigger: Arc::new(Mutex::new(context::AutoCompaction::default())),
        };
        let before = counted(&c, 14000, true);
        let result = observer.before_counted_request(
            &mut c,
            limits,
            &AtomicBool::new(false),
            &before,
            &mut |candidate| match failure {
                0 => Err(io::Error::other("mock count failure")),
                1 => Ok(counted(candidate, 16384, false)),
                _ => Ok(counted(candidate, 14000, true)),
            },
        );
        if failure == 0 {
            assert!(result.is_err());
        } else {
            assert!(!result.unwrap());
        }
        assert_eq!(journal.lock().unwrap().generation(), 0);
        let reason = [
            "candidate admission failed",
            "safe compaction cannot fit a useful reply reservation",
            "safe compaction cannot fit a useful reply reservation",
        ][failure];
        assert!(
            journal
                .lock()
                .unwrap()
                .inspect_record(1, 0, 4096)
                .unwrap()
                .contains(reason)
        );
        assert!(
            matches!(rx.fact().unwrap(), super::super::events::WorkerEvent::ToolPreview { detail, .. } if detail.contains(reason) && detail.contains("next turn"))
        );
        assert!(
            !observer
                .before_counted_request(
                    &mut c,
                    limits,
                    &AtomicBool::new(false),
                    &before,
                    &mut |_| panic!("same-turn failed plan must stay disarmed")
                )
                .unwrap()
        );
        c.push_user(native_message("user", "eligible next turn"), limits)
            .unwrap();
        journal.lock().unwrap().save(&c).unwrap();
        observer.trigger.lock().unwrap().begin_turn();
        let grown = counted(&c, 14500, true);
        assert!(
            observer
                .before_counted_request(
                    &mut c,
                    limits,
                    &AtomicBool::new(false),
                    &grown,
                    &mut |candidate| Ok(counted(candidate, 1000, true))
                )
                .unwrap()
        );
        assert_eq!(journal.lock().unwrap().generation(), 1);
    }
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn in_run_compaction_rewrites_once_and_keeps_the_summary_on_next_request() {
    struct ModelMock {
        requests: Vec<Value>,
        rewrites: usize,
        exact: bool,
    }
    impl Model for ModelMock {
        fn count_input(
            &mut self,
            request: ModelRequest<'_>,
            _: &AtomicBool,
        ) -> io::Result<vade_harness::contract::InputCount> {
            if !self.exact {
                return Ok(vade_harness::contract::InputCount::Unknown);
            }
            let summarized = request.messages.iter().any(|m| {
                m.value().get("name").and_then(Value::as_str) == Some("vade_context_summary")
            });
            Ok(vade_harness::contract::InputCount::Exact {
                prompt_tokens: if summarized { 1000 } else { 14000 },
                output_tokens: 128,
                context_limit: 16384,
                admitted: true,
            })
        }
        fn prompt_rewritten(&mut self) {
            self.rewrites += 1;
        }
        fn complete(
            &mut self,
            request: ModelRequest<'_>,
            _: &AtomicBool,
            _: &mut dyn FnMut(&str) -> io::Result<()>,
        ) -> io::Result<AssistantTurn> {
            self.requests.push(json::parse(&request.to_json()?)?);
            let reply = if self.requests.len() == 1 {
                r#"{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"new-read","type":"function","function":{"name":"read","arguments":"{}"}}]}}]}"#
            } else {
                r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}"#
            };
            AssistantTurn::from_response(&json::parse(reply)?, request.wire)
        }
    }
    struct ReadMock;
    impl ToolExecutor for ReadMock {
        fn execute(
            &mut self,
            _: ToolInvocation<'_>,
            _: &AtomicBool,
        ) -> Result<String, ToolError> {
            Ok("current file contents".into())
        }
    }
    let path =
        std::env::temp_dir().join(format!("vade-compacting-journal-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    let workspace = Workspace::new(&path).unwrap();
    let limits = Limits {
        max_context_tokens: 16_384,
        output_reserve_tokens: 1024,
        ..Limits::default()
    };
    let request = format!(
        r#"{{"model":"mock","tools":[{{"type":"function","function":{{"name":"read","parameters":{{"type":"object"}}}}}}],"messages":[{{"role":"system","content":"stable system"}},{{"role":"developer","content":"stable project"}},{{"role":"user","content":"old task"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"recent task"}},{{"role":"assistant","content":"recent answer"}},{{"role":"user","content":"{}"}}]}}"#,
        "old results ".repeat(900),
        "large current input ".repeat(1700)
    );
    let mut c = Conversation::from_request(json::parse(&request).unwrap(), limits).unwrap();
    let original_prefix: Vec<_> = c.messages()[..c.prefix_len()]
        .iter()
        .map(|m| m.serialized_json().to_owned())
        .collect();
    let journal = Arc::new(Mutex::new(
        Journal::create(&workspace, "worker", &c).unwrap(),
    ));
    let (tx, rx) = crate::interaction::lanes();
    let trigger = Arc::new(Mutex::new(context::AutoCompaction::default()));
    let mut observer = CompactingJournal {
        answer: String::new(),
        tool_prose: Arc::default(),
        cancel: Arc::default(),
        inner: SharedJournal(Arc::clone(&journal)),
        events: Arc::new(tx),
        enabled: true,
        trigger: Arc::clone(&trigger),
    };
    for _ in 0..16 { observer.events.fact(super::super::events::WorkerEvent::ToolCall(String::new(), String::new(), 0)).unwrap(); }
    observer.cancel.store(true, std::sync::atomic::Ordering::Relaxed); observer.publish_prose(String::new(), 0).unwrap();
    observer.cancel.store(false, std::sync::atomic::Ordering::Relaxed); while rx.fact().is_ok() {}
    let empty = vade_harness::agent_wire::Message::from_value(json::parse(r#"{"role":"assistant","tool_calls":[{"id":"no-content","type":"function","function":{"name":"workspace_list","arguments":"{}"}}]}"#).unwrap(), limits.wire).unwrap();
    let mut probe = CompactingJournal { inner: SharedJournal(Arc::new(Mutex::new(Journal::create(&workspace, "empty-content", &c).unwrap()))), events: Arc::clone(&observer.events), enabled: false, trigger: Arc::default(), answer: String::new(), tool_prose: Arc::default(), cancel: Arc::default() };
    probe.checkpoint(&c, &[empty]).unwrap(); assert!(matches!(rx.fact().unwrap(), super::super::events::WorkerEvent::ToolCall(label, prose, 0) if label.is_empty() && prose.is_empty()));
    let mut model = ModelMock {
        requests: Vec::new(),
        rewrites: 0,
        exact: false,
    };
    let report = agent::run_observed(
        &mut c,
        &mut model,
        &mut ReadMock,
        limits,
        &AtomicBool::new(false),
        &mut |_| {},
        &mut observer,
    );
    assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
    assert_eq!(model.rewrites, 1);
    assert_eq!(model.requests.len(), 2);
    let messages = |i: usize| {
        model.requests[i]
            .get("messages")
            .unwrap()
            .as_array()
            .unwrap()
    };
    let summary = messages(0)
        .iter()
        .find(|m| m.get("name").and_then(Value::as_str) == Some("vade_context_summary"))
        .unwrap();
    assert!(messages(1).contains(summary));
    for (i, original) in original_prefix.iter().enumerate() {
        assert_eq!(
            c.messages()[i].serialized_json().as_bytes(),
            original.as_bytes()
        );
    }
    assert!(
        context::Usage::for_conversation(&c)
            .near_limit(limits.max_context_tokens, limits.output_reserve_tokens)
    );
    let event = rx.fact().unwrap();
    assert!(
        matches!(event, super::super::events::WorkerEvent::ToolPreview { detail, reply: None, .. } if detail.contains("Auto-compacted older results") && detail.contains("remains above 60%"))
    );
    while let Ok(event) = rx.fact() {
        assert!(matches!(event, super::super::events::WorkerEvent::Budget(_) | super::super::events::WorkerEvent::ToolCall(_, _, _)));
    }
    // A real count overrides even a deliberately tiny local byte estimate;
    // the candidate is counted before persistence and recorded in the journal.
    let counted_request = request
        .replace(&"old results ".repeat(900), &"old results ".repeat(100))
        .replace(&"large current input ".repeat(1700), "small current input");
    let mut counted =
        Conversation::from_request(json::parse(&counted_request).unwrap(), limits).unwrap();
    let (tx, rx) = crate::interaction::lanes();
    let counted_journal = Arc::new(Mutex::new(
        Journal::create(&workspace, "counted", &counted).unwrap(),
    ));
    let mut observer = CompactingJournal {
        answer: String::new(),
        tool_prose: Arc::default(),
        cancel: Arc::default(),
        inner: SharedJournal(Arc::clone(&counted_journal)),
        events: Arc::new(tx),
        enabled: true,
        trigger: Arc::new(Mutex::new(context::AutoCompaction::default())),
    };
    let mut model = ModelMock {
        requests: Vec::new(),
        rewrites: 0,
        exact: true,
    };
    let result = agent::run_observed(
        &mut counted,
        &mut model,
        &mut ReadMock,
        limits,
        &AtomicBool::new(false),
        &mut |_| {},
        &mut observer,
    );
    assert_eq!(result.reason, StopReason::Finished(FinishReason::Stop));
    assert_eq!(model.rewrites, 1);
    assert!(
        counted_journal
            .lock()
            .unwrap()
            .compaction_status()
            .contains("[14000,1000]")
    );
    assert!(
        matches!(rx.fact().unwrap(), super::super::events::WorkerEvent::ToolPreview { detail, .. } if detail.contains("14000 → 1000 exact input tokens"))
    );
    // A rewrite below the low watermark rearms immediately, even if the
    // model finishes without another before_request check.
    let lower_request = request
        .replace(
            &"large current input ".repeat(1700),
            &"small current input ".repeat(300),
        )
        .replace(&"old results ".repeat(900), &"old results ".repeat(2600));
    let mut lower =
        Conversation::from_request(json::parse(&lower_request).unwrap(), limits).unwrap();
    let (tx, _rx) = crate::interaction::lanes();
    let mut lower_observer = CompactingJournal {
        answer: String::new(),
        tool_prose: Arc::default(),
        cancel: Arc::default(),
        inner: SharedJournal(Arc::new(Mutex::new(
            Journal::create(&workspace, "low", &lower).unwrap(),
        ))),
        events: Arc::new(tx),
        enabled: true,
        trigger: Arc::new(Mutex::new(context::AutoCompaction::default())),
    };
    assert!(
        lower_observer
            .before_request(&mut lower, limits, &AtomicBool::new(false))
            .unwrap()
    );
    assert!(
        context::Usage::for_conversation(&lower).required(limits.output_reserve_tokens)
            <= limits.max_context_tokens * 3 / 5
    );
    let future = vade_harness::agent_wire::Message::from_value(
        native_message("user", &"future user ".repeat(3000)),
        limits.wire,
    )
    .unwrap();
    let usage = context::Usage::with_appended(&lower, &[future]);
    assert!(lower_observer.trigger.lock().unwrap().observe(
        &usage,
        limits.max_context_tokens,
        limits.output_reserve_tokens
    ));

    let super::super::args::Parsed::Run(config) = super::super::args::Config::parse(
        [
            "agent",
            "--endpoint",
            "http://localhost:1",
            "--model",
            "mock",
            "--workspace",
            path.to_str().unwrap(),
            "--session",
            "reset",
        ]
        .into_iter()
        .map(str::to_owned),
    )
    .unwrap() else {
        panic!("expected run configuration");
    };
    let mut app = App::new(config).unwrap();
    let usage = context::Usage::for_conversation(app.native.as_ref().unwrap());
    assert!(app.auto_compaction.lock().unwrap().observe(&usage, 1, 0));
    assert!(!app.auto_compaction.lock().unwrap().observe(&usage, 1, 0));
    app.line_input("/new".into()).unwrap();
    assert!(app.auto_compaction.lock().unwrap().observe(&usage, 1, 0));
    // A new worker shares the same crossing state; high usage does not
    // trigger another rewrite when the run ends and a new run begins.
    let (tx, rx) = crate::interaction::lanes();
    let mut next = CompactingJournal {
        answer: String::new(),
        tool_prose: Arc::default(),
        cancel: Arc::default(),
        inner: SharedJournal(journal),
        events: Arc::new(tx),
        enabled: true,
        trigger,
    };
    let before = c.request(limits.wire).to_json().unwrap();
    assert!(
        !next
            .before_request(&mut c, limits, &AtomicBool::new(false))
            .unwrap()
    );
    assert_eq!(before, c.request(limits.wire).to_json().unwrap());
    *next.trigger.lock().unwrap() = context::AutoCompaction::default();
    // With only a prior summary eligible, warn once and retain it exactly.
    for _ in 0..3 {
        assert!(
            !next
                .before_request(&mut c, limits, &AtomicBool::new(false))
                .unwrap()
        );
        assert_eq!(before, c.request(limits.wire).to_json().unwrap());
    }
    assert!(
        matches!(rx.fact().unwrap(), super::super::events::WorkerEvent::ToolPreview { detail, reply: None, .. } if detail.contains("Auto-compaction deferred") && detail.contains("next turn"))
    );
    assert!(rx.fact().is_err());
}

fn tool_names(tools: &Value) -> Vec<String> {
    tools.as_array().unwrap().iter()
        .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str).map(str::to_owned))
        .collect()
}

#[test]
fn web_policy_preserves_saved_tool_shapes() {
    let on = native_definitions_with(false, vade_harness::contract::guaranteed_native_reply_tokens(), false);
    let off = web_definitions(on.clone(), false);
    assert!(tool_names(&on).contains(&"web_fetch".into()));
    assert!(!tool_names(&off).contains(&"web_fetch".into()));
    assert!(resumed_tools_match(Some(&on), false, &[false]));
    assert!(resumed_tools_match(Some(&off), false, &[false]));
    for mut tools in [on, off] {
        vade_harness::memory::add_tools(&mut tools);
        assert!(tool_names(&tools).contains(&"memory_save".into()));
        assert!(resumed_tools_match(Some(&tools), false, &[false]));
    }
}

#[test]
fn no_shell_edits_each_match_exactly_one_prompt_sentence() {
    for (from, _) in NO_SHELL_EDITS {
        assert_eq!(NATIVE_SYSTEM_PROMPT.matches(from).count() + BUILD_FIRST_PROMPT.matches(from).count(), 1, "{from}");
    }
}

#[test]
fn without_the_shell_the_agent_is_neither_offered_nor_told_to_use_it() {
    let (grants, limits) = (BTreeMap::new(), Limits::default());
    let reply = vade_harness::contract::guaranteed_native_reply_tokens();
    for children in [false, true] {
        let prompt = native_system_prompt_for(children, &grants, limits, false);
        for gone in ["workspace_shell", "macOS sandbox", "vade-check-", "--check", "Build or test after every edit"] {
            assert!(!prompt.contains(gone), "{gone} in the no-shell prompt");
        }
        assert!(prompt.contains("This platform has no command tool: you cannot build, test or run anything."));
        let with = native_definitions_with(children, reply, true);
        let without = native_definitions_with(children, reply, false);
        assert!(tool_names(&with).contains(&"workspace_shell".to_owned()));
        assert!(!tool_names(&without).contains(&"workspace_shell".to_owned()));
        // Every other definition keeps its exact bytes and order.
        let mut kept = with.as_array().unwrap().to_vec();
        kept.retain(|t| t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) != Some("workspace_shell"));
        assert_eq!(&Value::Array(kept), &without);
    }
}

#[test]
fn with_the_shell_the_prompt_and_tools_keep_their_bytes() {
    let (grants, limits) = (BTreeMap::new(), Limits::default());
    let frozen = format!(
        "{NATIVE_SYSTEM_PROMPT} Each request allows {} tool rounds, {} model turns, {} proposed tool calls and no active work time limit. {}\n{BUILD_FIRST_PROMPT}",
        limits.max_tool_turns, limits.max_model_turns, limits.max_tool_calls,
        coding_tools::write_guidance(vade_harness::contract::guaranteed_native_reply_tokens()),
    );
    assert_eq!(native_system_prompt_for(false, &grants, Limits { max_run_time: None, ..limits }, true), frozen);
    if coding_tools::shell_offered() {
        assert_eq!(native_system_prompt(false, &grants, Limits { max_run_time: None, ..limits }), frozen);
        assert_eq!(native_definitions(false), coding_tools::definitions_with_web(
            vade_harness::contract::guaranteed_native_reply_tokens(), true, vade_harness::web::offered()));
    } else {
        assert!(!tool_names(&native_definitions(false)).contains(&"workspace_shell".to_owned()));
    }
}

#[test]
fn sessions_saved_with_the_shell_still_resume_where_it_is_unavailable() {
    let path = std::env::temp_dir().join(format!("vade-no-shell-resume-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let workspace = Workspace::new(&path).unwrap();
    let (grants, limits) = (BTreeMap::new(), Limits::default());
    let project = native_conversation("mock", &workspace, &[], "", true, false, &grants, limits).unwrap().messages()[1].value().clone();
    let saved = |shell: bool| Conversation::from_request(
        Value::Object(vec![
            ("model".into(), Value::String("mock".into())),
            ("stream".into(), Value::Bool(true)),
            ("tools".into(), native_definitions_with(false, vade_harness::contract::guaranteed_native_reply_tokens(), shell)),
            ("messages".into(), Value::Array(vec![
                native_message("system", &format!("{}\n{}", native_system_prompt_for(false, &grants, limits, shell), vade_harness::session::model_identity("mock"))),
                project.clone(),
            ])),
        ]),
        Limits::default(),
    ).unwrap();
    // Where the shell is unavailable, both the new and the earlier sessions resume.
    for shell in [false, true] {
        assert!(validate_resumed_in(&saved(shell), "mock", false, &grants, limits, &[false, true]).is_ok(), "saved with shell={shell}");
    }
    // Where it is available, only sessions that offered it match.
    assert!(validate_resumed_in(&saved(true), "mock", false, &grants, limits, &[true]).is_ok());
    let refused = validate_resumed_in(&saved(false), "mock", false, &grants, limits, &[true]).unwrap_err().to_string();
    assert!(refused.contains("different tool set"), "{refused}");
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn automatic_compaction_is_local_and_never_asks_the_model() {
    struct Host(usize, usize);
    impl context::ContextHost for Host {
        fn count(&mut self, c: &Conversation) -> io::Result<context::Usage> {
            let tokens = (c.serialized_bytes() / 4) as u64;
            context::Usage::for_conversation(c).with_count(&vade_harness::contract::InputCount::Exact {
                prompt_tokens: tokens, output_tokens: 128, context_limit: self.0 as u64, admitted: tokens + 128 <= self.0 as u64,
            })
        }
        fn summarize(&mut self, _: &Conversation) -> io::Result<String> {
            self.1 += 1;
            Ok("Plan: ship it.\nOpen tasks: none.".into())
        }
    }
    let path = std::env::temp_dir().join(format!("vade-model-rollover-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir(&path).unwrap();
    let workspace = Workspace::new(&path).unwrap();
    let request = format!(
        r#"{{"model":"mock","messages":[{{"role":"system","content":"stable"}},{{"role":"user","content":"old ask"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"recent"}},{{"role":"assistant","content":"latest answer"}},{{"role":"user","content":"{{}}"}}]}}"#,
        "old resolved prose ".repeat(400)
    );
    for (window, current, rolled) in [(2048, "current".to_owned(), true), (1100, "current ".repeat(600), false)] {
        let request = request.replace("{}", &current);
        let limits = Limits { max_context_tokens: window, output_reserve_tokens: 128, ..Limits::default() };
        let mut c = Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
        let journal = Arc::new(Mutex::new(Journal::create(&workspace, &format!("rollover-{window}"), &c).unwrap()));
        let (tx, rx) = crate::interaction::lanes();
        let mut observer = CompactingJournal {
            answer: String::new(),
            tool_prose: Arc::default(),
            cancel: Arc::default(),
            inner: SharedJournal(Arc::clone(&journal)),
            events: Arc::new(tx),
            enabled: true,
            trigger: Arc::new(Mutex::new(context::AutoCompaction::default())),
        };
        let mut host = Host(window, 0);
        let before = context::ContextHost::count(&mut host, &c).unwrap();
        let result = observer.before_hosted_request(&mut c, limits, &AtomicBool::new(false), &before, &mut host).unwrap();
        assert_eq!(result, rolled);
        let event = rx.fact().unwrap();
        let super::super::events::WorkerEvent::ToolPreview { detail, .. } = event else { panic!("expected one context line") };
        // An automatic watermark never sends a summary request to the model.
        assert_eq!(host.1, 0);
        if rolled {
            assert!(detail.starts_with("Auto-compacted older results: "), "{detail}");
            assert_eq!(journal.lock().unwrap().generation(), 1);
            assert!(journal.lock().unwrap().compaction_status().contains("Local excerpt"));
            assert_eq!(c.messages().len(), 6, "prefix, summary, the user messages and the latest answer");
        } else {
            // No local plan fits: the hard stop, with the counted kept-turns floor.
            assert!(detail.contains("This chat cannot continue: even after compaction"), "{detail}");
            assert_eq!(journal.lock().unwrap().generation(), 0);
        }
    }
    std::fs::remove_dir_all(path).unwrap();
}
