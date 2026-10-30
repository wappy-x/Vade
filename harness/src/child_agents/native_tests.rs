//! Real native executors driven by deterministic model replies. No network.
use super::*;
use crate::{coding_tools, journal::Journal, project, shell, test_workspace::Fixture};
use shell::ApprovalPolicy;

/// Both tool shapes a native parent can have, whatever this test host is:
/// `true` offers `workspace_shell` (macOS, and a Linux session resumed from
/// before the shell was hidden); `false` is a fresh Linux session.
const SHAPES: [bool; 2] = [true, false];

fn native_parent(workspace: &crate::tools::Workspace, shell: bool) -> Conversation {
    let reply_limit = crate::contract::guaranteed_native_reply_tokens();
    let Value::Array(mut tools) = coding_tools::definitions_for(reply_limit, shell) else {
        unreachable!()
    };
    let Value::Array(extra) = definitions() else {
        unreachable!()
    };
    tools.extend(extra);
    Conversation::from_request(Value::Object(vec![
        ("model".into(), Value::String("mock".into())),
        ("tools".into(), Value::Array(tools)),
        ("messages".into(), Value::Array(vec![
            val(r#"{"role":"system","content":"Native tools follow host scopes and approvals."}"#),
            project::Snapshot::discover(workspace, &[]).unwrap().message().unwrap(),
            val(r#"{"role":"user","content":"Parent task."}"#),
        ])),
    ]), Limits::default()).unwrap()
}
fn native_factory(
    workspace: Arc<crate::tools::Workspace>,
    policy: ApprovalPolicy,
    replies: Vec<Value>,
    preview: Arc<ChildPreview>,
) -> Arc<Factory> {
    // The real HTTP adapter is constructed but replaced before invocation.
    let factory = native_http_factory(
        "http://127.0.0.1:1",
        NativeToolHost {
            project: project::Snapshot::discover(&workspace, &[]).unwrap(),
            workspace,
            approval: policy,
            web: true,
            shell_exclusions: shell::Exclusions::default(),
            shell_toolchain: None,
            preview,
        },
    )
    .unwrap();
    Arc::new(move |context, cancel| {
        let mut worker = factory(context, cancel)?;
        worker.model = Box::new(Replies {
            replies: replies.clone().into(),
            dispatches: Arc::new(AtomicUsize::new(0)),
            prompt_tokens: 10,
        });
        Ok(worker)
    })
}
/// A scheduler under a parent that offers `workspace_shell`. Grants are checked
/// against the parent's schemas, so shell grants need this shape on every host.
fn native_scheduler(
    workspace: &crate::tools::Workspace,
    grants: BTreeMap<String, Scope>,
    factory: Arc<Factory>,
) -> Scheduler {
    try_native_scheduler(workspace, grants, factory, true).unwrap()
}
fn try_native_scheduler(
    workspace: &crate::tools::Workspace,
    grants: BTreeMap<String, Scope>,
    factory: Arc<Factory>,
    shell: bool,
) -> io::Result<Scheduler> {
    Scheduler::new(
        StablePrefix::from_parent(&native_parent(workspace, shell)).unwrap(),
        Config {
            policy: ToolPolicy::native(),
            grants,
            ..Config::default()
        },
        Arc::new(AtomicBool::new(false)),
        factory,
    )
}
fn edits(paths: &[&str]) -> Scope {
    let mut scope = Scope::native_read_only();
    scope.tools.push("apply_patch".into());
    scope.write_paths = paths.iter().map(|path| (*path).into()).collect();
    scope
}
fn shell_scope() -> Scope {
    let mut scope = Scope::native_read_only();
    scope.tools.push("workspace_shell".into());
    scope
}
fn granted(name: &str) -> Task {
    Task {
        grant: Some(name.into()),
        ..Task::read_only("Scoped native task")
    }
}
fn patch(text: &str) -> Value {
    Value::Object(vec![("patch".into(), Value::String(text.into()))])
}
fn output(result: &TaskResult, id: &str) -> String {
    result
        .conversation
        .as_ref()
        .unwrap()
        .messages()
        .iter()
        .find(|m| m.value().get("tool_call_id").and_then(Value::as_str) == Some(id))
        .unwrap()
        .value()
        .get("content")
        .and_then(Value::as_str)
        .unwrap()
        .to_owned()
}
fn run_once(
    workspace: Arc<crate::tools::Workspace>,
    scope: Option<Scope>,
    policy: ApprovalPolicy,
    name: &str,
    args: Value,
    preview: Arc<ChildPreview>,
) -> TaskResult {
    let granted_scope = scope.is_some();
    let grants = scope.map_or_else(BTreeMap::new, |scope| [("worker".into(), scope)].into());
    let factory = native_factory(
        Arc::clone(&workspace),
        policy,
        vec![proposal("action", name, args), answer("done")],
        preview,
    );
    let mut scheduler = native_scheduler(&workspace, grants, factory);
    let id = scheduler
        .spawn(if granted_scope {
            granted("worker")
        } else {
            Task::read_only("read-only")
        })
        .unwrap();
    finished(&scheduler, id);
    scheduler.take_result(id).unwrap()
}

#[test]
fn native_registry_matches_schemas_and_side_effects_cannot_be_reclassified() {
    let policy = ToolPolicy::native();
    let reply_limit = crate::contract::guaranteed_native_reply_tokens();
    assert_eq!(
        coding_tools::definitions(),
        coding_tools::definitions_with_web(reply_limit, coding_tools::shell_offered(), crate::web::offered())
    );
    for name in ["web_fetch", "web_search"] {
        let scope = Scope { tools: vec![name.into()], write_paths: Vec::new() };
        assert_eq!(policy.validate_scope(&scope).is_ok(), crate::web::offered());
        assert_eq!(Scope::native_read_only().tools.contains(&name.into()), crate::web::offered());
    }
    for (shell, web) in SHAPES.into_iter().flat_map(|shell| [false, true].map(|web| (shell, web))) {
        let policy = ToolPolicy::native_with_web(web);
        let Value::Array(tools) = coding_tools::definitions_with_web(reply_limit, shell, web) else {
            unreachable!()
        };
        let names: Vec<String> = tools
            .iter()
            .map(|v| {
                v.get("function")
                    .unwrap()
                    .get("name")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .into()
            })
            .collect();
        for name in ["web_fetch", "web_search"] {
            assert_eq!(names.iter().any(|n| n == name), web);
            let scope = Scope { tools: vec![name.into()], write_paths: Vec::new() };
            assert_eq!(policy.validate_scope(&scope).is_ok(), web);
        }
        let mut all = policy.read_scope(&names);
        all.tools.extend(["apply_patch".into(), "workspace_edit_lines".into()]);
        all.write_paths.push("output.txt".into());
        assert_eq!(names.iter().any(|name| name == "workspace_shell"), shell);
        let mut with_shell = all.clone();
        with_shell.tools.push("workspace_shell".into());
        if shell {
            all = with_shell;
        } else {
            // A fresh shell-less parent cannot hand a child the tool it lacks.
            assert!(policy.validate(&with_shell, &names).is_err());
        }
        policy.validate(&all, &names).unwrap();
        assert_eq!(all.tools.len(), names.len());
    }
    for name in ["apply_patch", "workspace_shell", "workspace_edit_lines"] {
        assert!(ToolPolicy::new([(name.into(), Access::Read)].into()).is_err());
    }
    for name in ["child_spawn", "made_up"] {
        let mut scope = Scope::native_read_only();
        scope.tools.push(name.into());
        assert!(policy.validate_scope(&scope).is_err());
    }
    assert!(
        policy
            .validate_scope(&Scope {
                tools: vec!["apply_patch".into()],
                write_paths: vec![]
            })
            .is_err()
    );
}

#[test]
fn default_scope_denies_real_edits_shell_and_recursive_children_even_with_auto_approval() {
    let fixture = Fixture::new();
    let workspace = Arc::new(fixture.workspace());
    for (shell, (tool, args)) in SHAPES.into_iter().flat_map(|shell| [
        (
            "apply_patch",
            patch("*** Begin Patch\n*** Add File: output.txt\n+unauthorized\n*** End Patch"),
        ),
        (
            "workspace_shell",
            val(r#"{"argv":["/usr/bin/true"],"timeout_ms":1000}"#),
        ),
        (
            "child_spawn",
            val(r#"{"task":"escalate","grant":"worker"}"#),
        ),
    ].map(|call| (shell, call))) {
        let factory = native_factory(
            Arc::clone(&workspace),
            ApprovalPolicy::Auto,
            vec![
                proposal("action", tool, args),
                answer("denied acknowledged"),
            ],
            Arc::new(|_, _, _| panic!("scope must deny before preview")),
        );
        // Even when the host offers privileged named scopes, omitting grant
        // must not inherit one of them or the parent's native capabilities.
        let mut grants: BTreeMap<String, Scope> = [("worker".into(), edits(&["output.txt"]))].into();
        grants.insert("shell".into(), shell_scope());
        if !shell {
            // A shell-less parent has no shell scope to offer at all.
            let refused = try_native_scheduler(&workspace, grants.clone(), Arc::clone(&factory), false);
            assert_eq!(refused.err().map(|e| e.kind()), Some(io::ErrorKind::InvalidInput));
            grants.remove("shell");
        }
        let mut scheduler = try_native_scheduler(&workspace, grants, factory, shell).unwrap();
        let id = scheduler.spawn(Task::read_only("default scope")).unwrap();
        finished(&scheduler, id);
        let result = scheduler.take_result(id).unwrap();
        // Without the schema the call is not even a known tool.
        let refusal = if tool == "workspace_shell" && !shell { "unknown_tool" } else { "denied" };
        assert!(output(&result, "action").contains(refusal), "{}", output(&result, "action"));
        assert!(result.tools.is_empty());
        assert!(!fixture.0.join("output.txt").exists());
    }
}

#[test]
fn leased_native_patch_still_requires_parent_approval_and_records_child_journal() {
    for (policy, approve, expected) in [
        (ApprovalPolicy::Ask, false, false),
        (ApprovalPolicy::Ask, true, true),
        (ApprovalPolicy::Auto, true, true),
        (ApprovalPolicy::Never, true, false),
    ] {
        let fixture = Fixture::new();
        let workspace = Arc::new(fixture.workspace());
        let previews = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&previews);
        let result = run_once(
            Arc::clone(&workspace),
            Some(edits(&["output.txt"])),
            policy,
            "apply_patch",
            patch("*** Begin Patch\n*** Add File: output.txt\n+approved child edit\n*** End Patch"),
            Arc::new(move |_, preview, _| {
                if preview.title == "Patch preview" {
                    calls.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(preview.approval_required, policy == ApprovalPolicy::Ask);
                    assert!(preview.detail.contains("approved child edit"));
                }
                approve
            }),
        );
        assert_eq!(fixture.0.join("output.txt").exists(), expected);
        assert_eq!(
            previews.load(Ordering::SeqCst),
            usize::from(policy != ApprovalPolicy::Never)
        );
        assert!(result.tools[0].writes);
        assert_eq!(
            result.tools[0].status,
            if expected {
                ToolStatus::Completed
            } else {
                ToolStatus::Denied
            }
        );
        let name = result.journal_name.as_deref().unwrap();
        let raw = std::fs::read(fixture.0.join(format!(".vade/sessions/{name}.journal"))).unwrap();
        assert!(
            raw.windows(b"unknown_outcome".len())
                .any(|part| part == b"unknown_outcome"),
            "durable intent precedes the side effect"
        );
        let (_, replay, torn) = Journal::resume(&workspace, name, Limits::default()).unwrap();
        assert!(!torn);
        assert_eq!(
            replay.messages(),
            result.conversation.as_ref().unwrap().messages()
        );
        assert_eq!(
            fixture.0.join("output.txt").exists(),
            expected,
            "replay never executes a tool"
        );
    }
}

#[test]
fn every_patch_path_and_move_destination_requires_an_exact_lease() {
    for text in [
        "*** Begin Patch\n*** Add File: allowed\n+one\n*** Add File: outside\n+two\n*** End Patch",
        "*** Begin Patch\n*** Update File: source\n*** Move to: outside\n*** End Patch",
        "*** Begin Patch\n*** Delete File: outside\n-old\n*** End Patch",
        "*** Begin Patch\n*** Add File: ./allowed\n+alias\n*** End Patch",
        "*** Begin Patch\n*** Add File: ALLOWED\n+case alias\n*** End Patch",
        "*** Begin Patch\n*** Add File: allowed/child\n+directory lease is not a file lease\n*** End Patch",
    ] {
        let fixture = Fixture::new();
        fixture.write("source", "source\n");
        fixture.write("outside", "old\n");
        let result = run_once(
            Arc::new(fixture.workspace()),
            Some(edits(&["allowed", "source"])),
            ApprovalPolicy::Auto,
            "apply_patch",
            patch(text),
            Arc::new(|_, _, _| panic!("unleased effect reached approval")),
        );
        assert!(output(&result, "action").contains("denied"));
        assert_eq!(fixture.read("outside"), "old\n");
        assert_eq!(fixture.read("source"), "source\n");
        assert!(!fixture.0.join("allowed").exists());
    }
    let fixture = Fixture::new();
    fixture.write("source", "original\n");
    let result = run_once(
        Arc::new(fixture.workspace()),
        Some(edits(&["source", "destination"])),
        ApprovalPolicy::Ask,
        "apply_patch",
        patch(
            "*** Begin Patch\n*** Update File: source\n*** Move to: destination\n@@\n-original\n+moved\n*** End Patch",
        ),
        Arc::new(|_, _, _| true),
    );
    assert!(output(&result, "action").contains("Applied"));
    assert!(!fixture.0.join("source").exists());
    assert_eq!(fixture.read("destination"), "moved\n");
}

#[test]
fn native_parent_and_children_respect_all_write_leases_until_collection() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    let mut scheduler = native_scheduler(
        &workspace,
        [
            ("edit".into(), edits(&["a", "moved"])),
            ("other".into(), edits(&["b"])),
            ("shell".into(), shell_scope()),
        ]
        .into(),
        factory(vec![answer("done")], Arc::new(AtomicUsize::new(0))),
    );
    let id = scheduler.spawn(granted("edit")).unwrap();
    finished(&scheduler, id);
    assert_eq!(
        scheduler.spawn(granted("shell")),
        Err(ScheduleError::WriteConflict)
    );
    let check = |scheduler: &Scheduler, name: &str, args: Value| {
        let call = ToolCall {
            id: "parent".into(),
            name: name.into(),
            arguments: json::stringify(&args).unwrap(),
        };
        scheduler.check_parent(&ToolInvocation {
            call: &call,
            arguments: &args,
            max_output_bytes: 65536,
        })
    };
    for text in [
        "*** Begin Patch\n*** Add File: b\n+ok\n*** Add File: a\n+conflict\n*** End Patch",
        "*** Begin Patch\n*** Update File: b\n*** Move to: moved\n*** End Patch",
        "*** Begin Patch\n*** Delete File: A\n-old\n*** End Patch",
    ] {
        assert_eq!(
            check(&scheduler, "apply_patch", patch(text)),
            Err(ToolError::Denied)
        );
    }
    assert_eq!(
        check(
            &scheduler,
            "workspace_shell",
            val(r#"{"argv":["/usr/bin/true"],"timeout_ms":1000}"#)
        ),
        Err(ToolError::Denied)
    );
    assert_eq!(
        check(
            &scheduler,
            "apply_patch",
            patch("*** Begin Patch\n*** Add File: b\n+ok\n*** End Patch")
        ),
        Ok(())
    );
    scheduler.take_result(id).unwrap();
    let id = scheduler.spawn(granted("shell")).unwrap();
    finished(&scheduler, id);
    assert!(
        scheduler
            .view()
            .snapshots()
            .iter()
            .find(|s| s.id == id)
            .unwrap()
            .workspace_write_lease
    );
    assert_eq!(
        scheduler.spawn(granted("other")),
        Err(ScheduleError::WriteConflict)
    );
    assert_eq!(
        check(
            &scheduler,
            "apply_patch",
            patch("*** Begin Patch\n*** Add File: b\n+ok\n*** End Patch")
        ),
        Err(ToolError::Denied)
    );
    assert_eq!(
        check(&scheduler, "workspace_read", val(r#"{"path":"b"}"#)),
        Ok(())
    );
    scheduler.take_result(id).unwrap();
    assert_eq!(
        check(
            &scheduler,
            "workspace_shell",
            val(r#"{"argv":["/usr/bin/true"],"timeout_ms":1000}"#)
        ),
        Ok(())
    );
    scheduler.spawn(granted("other")).unwrap();
}

#[test]
fn shell_less_parent_refuses_shell_grants_and_still_enforces_edit_leases() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    let grants = || -> BTreeMap<String, Scope> {
        [("edit".into(), edits(&["a"])), ("other".into(), edits(&["b"]))].into()
    };
    let mut with_shell = grants();
    with_shell.insert("shell".into(), shell_scope());
    let factory = || factory(vec![answer("done")], Arc::new(AtomicUsize::new(0)));
    let refused = try_native_scheduler(&workspace, with_shell, factory(), false).err().unwrap();
    assert_eq!((refused.kind(), refused.to_string()), (io::ErrorKind::InvalidInput, "invalid child tool scope".into()));
    let mut scheduler = try_native_scheduler(&workspace, grants(), factory(), false).unwrap();
    let check = |scheduler: &Scheduler, text: &str| {
        let args = patch(text);
        let call = ToolCall {
            id: "parent".into(),
            name: "apply_patch".into(),
            arguments: json::stringify(&args).unwrap(),
        };
        scheduler.check_parent(&ToolInvocation {
            call: &call,
            arguments: &args,
            max_output_bytes: 65536,
        })
    };
    let add_a = "*** Begin Patch\n*** Add File: a\n+parent\n*** End Patch";
    let id = scheduler.spawn(granted("edit")).unwrap();
    finished(&scheduler, id);
    assert_eq!(check(&scheduler, add_a), Err(ToolError::Denied));
    assert_eq!(check(&scheduler, "*** Begin Patch\n*** Add File: b\n+ok\n*** End Patch"), Ok(()));
    scheduler.take_result(id).unwrap();
    assert_eq!(check(&scheduler, add_a), Ok(()));
}

#[test]
#[cfg(target_os = "macos")]
fn explicit_shell_scope_never_bypasses_parent_approval_or_the_native_sandbox() {
    for (policy, approve, allowed) in [
        (ApprovalPolicy::Never, true, false),
        (ApprovalPolicy::Ask, false, false),
        (ApprovalPolicy::Ask, true, true),
        (ApprovalPolicy::Auto, false, false),
        (ApprovalPolicy::Auto, true, true),
    ] {
        let fixture = Fixture::new();
        let workspace = Arc::new(fixture.workspace());
        let previews = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&previews);
        let result = run_once(
            workspace,
            Some(shell_scope()),
            policy,
            "workspace_shell",
            val(
                r#"{"argv":["/bin/sh","-c","printf approved > shell-output; ! printf hidden > .hidden"],"timeout_ms":2000}"#,
            ),
            Arc::new(move |_, preview, _| {
                if preview.title == "Command" {
                    assert_eq!(preview.approval_required, policy == ApprovalPolicy::Ask);
                    calls.fetch_add(1, Ordering::SeqCst);
                }
                approve
            }),
        );
        assert_eq!(fixture.0.join("shell-output").exists(), allowed);
        assert!(!fixture.0.join(".hidden").exists());
        assert_eq!(
            previews.load(Ordering::SeqCst),
            usize::from(policy != ApprovalPolicy::Never)
        );
        assert!(if allowed {
            output(&result, "action").contains("Exited(Some(0))")
        } else {
            output(&result, "action").contains("denied")
        });
    }
}

// The shell-capable parent here is a Linux session resumed from before the
// shell was hidden: the schema is present but the executor cannot run it.
#[test]
#[cfg(not(target_os = "macos"))]
fn unsupported_native_shell_keeps_child_session_alive_without_side_effects() {
    let fixture = Fixture::new();
    let result = run_once(
        Arc::new(fixture.workspace()),
        Some(shell_scope()),
        ApprovalPolicy::Ask,
        "workspace_shell",
        val(r#"{"argv":["/bin/sh","-c","touch output"],"timeout_ms":1000}"#),
        Arc::new(|_, _, _| panic!("unsupported shell must refuse before approval")),
    );
    assert_eq!(
        result.report.as_ref().unwrap().reason,
        StopReason::Finished(crate::agent_wire::FinishReason::Stop)
    );
    assert!(output(&result, "action").contains("unsupported_platform"));
    assert!(
        result
            .conversation
            .as_ref()
            .unwrap()
            .messages()
            .last()
            .unwrap()
            .value()
            .get("content")
            .and_then(Value::as_str)
            .unwrap()
            .contains("done")
    );
    assert!(!result.uncertain_writes());
    assert!(!fixture.0.join("output").exists());
}

#[test]
fn child_journal_failure_stops_before_model_or_side_effect_dispatch() {
    let fixture = Fixture::new();
    let outside = Fixture::new();
    std::os::unix::fs::symlink(&outside.0, fixture.0.join(".vade")).unwrap();
    let workspace = Arc::new(fixture.workspace());
    let factory = native_factory(
        Arc::clone(&workspace),
        ApprovalPolicy::Auto,
        vec![],
        Arc::new(|_, _, _| panic!("no executor without durable intent")),
    );
    let mut scheduler = native_scheduler(
        &workspace,
        [("edit".into(), edits(&["output"]))].into(),
        factory,
    );
    let id = scheduler.spawn(granted("edit")).unwrap();
    finished(&scheduler, id);
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.outcome, Outcome::Runtime(StopReason::JournalError));
    assert_eq!(result.tokens_reserved, 0);
    assert!(result.journal_name.is_none());
    assert!(result.tools.is_empty());
    assert_eq!(std::fs::read_dir(&outside.0).unwrap().count(), 0);
}

#[test]
fn parent_observer_cannot_be_bypassed_by_enabling_children() {
    struct FailedJournal;
    impl agent::RunObserver for FailedJournal {
        fn checkpoint(
            &mut self,
            _: &Conversation,
            staged: &[crate::agent_wire::Message],
        ) -> io::Result<()> {
            assert!(
                staged
                    .iter()
                    .any(|m| m.serialized_json().contains("unknown_outcome"))
            );
            Err(io::Error::other("mock durable sink failure"))
        }
    }
    let dispatches = Arc::new(AtomicUsize::new(0));
    let mut parent = parent();
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        factory(vec![answer("must not run")], Arc::clone(&dispatches)),
    );
    let result = run_parent_observed(
        &mut parent,
        &mut ParentModel { step: 0, id: None },
        &mut NoTools,
        Limits::default(),
        &mut scheduler,
        &mut |_| {},
        &mut FailedJournal,
    );
    assert_eq!(result.report.reason, StopReason::JournalError);
    assert!(result.children.is_empty());
    assert_eq!(dispatches.load(Ordering::SeqCst), 0);
    assert!(scheduler.view().snapshots().is_empty());
}

#[test]
fn native_preflight_failure_receipt_is_preserved_without_uncertain_writes() {
    let fixture = Fixture::new();
    fixture.write("source", "before\n");
    let path = fixture.0.join("source");
    let result = run_once(
        Arc::new(fixture.workspace()),
        Some(edits(&["source"])),
        ApprovalPolicy::Ask,
        "apply_patch",
        patch("*** Begin Patch\n*** Update File: source\n@@\n-before\n+after\n*** End Patch"),
        Arc::new(move |_, preview, _| {
            if preview.title == "Patch preview" {
                std::fs::write(&path, "concurrent change\n").unwrap();
            }
            true
        }),
    );
    assert!(output(&result, "action").contains("Patch refused before any write:"));
    assert_eq!(fixture.read("source"), "concurrent change\n");
    assert_eq!(result.tools[0].status, ToolStatus::Completed);
    assert!(!result.uncertain_writes());
}

#[test]
#[cfg(target_os = "macos")]
fn native_child_timed_out_command_preserves_receipt_and_marks_possible_writes_uncertain() {
    let fixture = Fixture::new();
    let result = run_once(
        Arc::new(fixture.workspace()),
        Some(shell_scope()),
        ApprovalPolicy::Ask,
        "workspace_shell",
        val(
            r#"{"argv":["/bin/sleep","2"],"timeout_ms":100}"#,
        ),
        Arc::new(|_, _, _| true),
    );
    assert!(output(&result, "action").contains("Timeout"));
    assert_eq!(result.tools[0].status, ToolStatus::Uncertain);
    assert!(result.uncertain_writes());
}

#[test]
fn child_timeout_receipt_preserves_in_memory_partial_writes() {
    struct PartialWrite {
        bytes: Arc<Mutex<Vec<u8>>>,
        receipt: String,
        uncertain: bool,
    }
    impl ToolExecutor for PartialWrite {
        fn execute(&mut self, call: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
            assert_eq!(call.call.name, "workspace_write");
            assert_eq!(call.arguments.get("path").and_then(Value::as_str), Some("src/a.rs"));
            self.bytes.lock().unwrap().extend_from_slice(b"partial");
            self.uncertain = true;
            Ok(self.receipt.clone())
        }
        fn last_side_effect_uncertain(&self) -> bool { self.uncertain }
    }
    // The native adapter above checks a real timeout. This executor supplies a
    // partial write before its receipt, without relying on process scheduling.
    let receipt = shell::CommandOutput {
        outcome: shell::Outcome::Timeout, stdout: "partial output".into(), stderr: String::new(),
        stdout_bytes: 14, stderr_bytes: 0, stdout_omitted: 0, stderr_omitted: 0,
        elapsed_ms: 100, exit_code: None, signal: None,
    }.render();
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let dispatches = Arc::new(AtomicUsize::new(0));
    let base = factory(vec![proposal("action", "workspace_write", val(r#"{"path":"src/a.rs"}"#))], Arc::clone(&dispatches));
    let worker_factory: Arc<Factory> = Arc::new({
        let bytes = Arc::clone(&bytes);
        let receipt = receipt.clone();
        move |context, cancel| {
            let mut worker = base(context, cancel)?;
            worker.tools = Box::new(PartialWrite { bytes: Arc::clone(&bytes), receipt: receipt.clone(), uncertain: false });
            Ok(worker)
        }
    });
    let mut scheduler = scheduler(write_config(), Arc::new(AtomicBool::new(false)), worker_factory);
    let id = scheduler.spawn(write_task("a")).unwrap();
    finished(&scheduler, id);
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(bytes.lock().unwrap().as_slice(), b"partial");
    assert!(output(&result, "action").contains(&receipt));
    assert_eq!(result.tools.len(), 1);
    assert!(result.tools[0].writes);
    assert_eq!(result.tools[0].status, ToolStatus::Uncertain);
    assert!(result.uncertain_writes());
    assert_eq!(result.outcome, Outcome::Runtime(StopReason::ToolFailed));
    assert_eq!(dispatches.load(Ordering::SeqCst), 1, "an uncertain write must not be retried");
}
