use super::*;
use crate::agent_wire::ToolCall;
use crate::artifacts::Access;
use crate::test_workspace::Fixture;
use std::io;
#[test]
fn line_edits_guard_visible_ranges_and_retain_exact_bytes() {
    for old in ["", "α\r\nβ", "one\ntwo\n", "🦀"] {
        let f = Fixture::new(); f.write("source", old); let w = f.workspace();
        let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
        let mut invoke = |name: &str, args: Value| { let call = ToolCall { id: "line-edit".into(), name: name.into(), arguments: json::stringify(&args).unwrap() }; tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap() };
        let read = || json::parse(r#"{"path":"source","start_line":1,"end_line":2}"#).unwrap();
        let output = json::parse(&invoke("workspace_read_lines", read())).unwrap(); let tag = output.get("edit_tag").and_then(Value::as_str).unwrap();
        let edit = |start, end, op, text: &str| json::parse(&format!(r#"{{"path":"source","start_line":{start},"end_line":{end},"edit_tag":{},"operation":"{op}","new_text":{}}}"#, json::quote(tag), json::quote(text))).unwrap();
        assert!(invoke("workspace_edit_lines", edit(1, 3, "replace", "x")).contains("out of bounds"));
        f.write("source", &format!("{old}changed")); let stale = invoke("workspace_edit_lines", edit(1, 1, "replace", "x\n")); assert!(stale.contains("stale edit_tag") && stale.contains("changed"));
        f.write("source", old); invoke("workspace_read_lines", read());
        let receipt = invoke("workspace_edit_lines", edit(1, old.split_inclusive('\n').count().max(1), "replace", "λ\r\nlast")); assert!(receipt.starts_with("Applied ")); assert_eq!(f.read("source"), "λ\r\nlast");
        assert!(invoke("workspace_edit_lines", edit(1, 1, "delete", "")).contains("re-read"));
    }
}
#[test]
fn check_feedback_puts_errors_first_and_flags_repeats() {
    let f = Fixture::new();
    let w = f.workspace();
    let approve = |_: Preview<'_>, _: &AtomicBool| true;
    let mut tools =
        CodingTools::new(&w, ApprovalPolicy::Auto, approve).with_check(Some("make"));
    for flags in [None, Some(""), Some("--cfg preserved -C debuginfo=1")] { let check = CodingTools::new(&w, ApprovalPolicy::Auto, approve).with_check(Some("printf '%s\\n%s' \"$RUSTFLAGS\" \"$RUSTDOCFLAGS\"")); let argv = &check.check.as_ref().unwrap().argv; let mut command = std::process::Command::new(&argv[0]); command.args(&argv[1..]).env_remove("RUSTFLAGS").env_remove("RUSTDOCFLAGS"); if let Some(flags) = flags { command.env("RUSTFLAGS", flags).env("RUSTDOCFLAGS", flags); } let output = command.output().unwrap(); let expected = format!("{}-D warnings", flags.filter(|s| !s.is_empty()).map(|s| format!("{s} ")).unwrap_or_default()); assert!(output.status.success()); assert_eq!(String::from_utf8(output.stdout).unwrap(), format!("{expected}\n{expected}")); }
    tools.check.as_mut().unwrap().pending = true;
    assert!(tools.take_pending_check().is_some() && tools.check_failing() && tools.check_not_run());
    let raw = format!(
        "status: Exited(Some(2))\nstdout:\nerror: x\n{}stderr:\nfailed build\npanicked at x\nundefined symbol\n",
        "ok\n".repeat(1800)
    );
    let outs: Vec<String> = (0..3).map(|_| tools.command_feedback(&raw, None, true, 4096)).collect();
    assert!(outs[0].len() <= CHECK_FEEDBACK_BYTES && outs[0].contains("lines):\nerror: x"));
    assert!(!outs[1].contains("hint:") && outs[2].contains("hint:") && tools.check_failing());
    let out = tools.command_feedback(&raw, Some("a-1"), false, 4096);
    assert!(out.starts_with("status: Exited(Some(2))") && ["lines):\nerror: x", "failed build", "panicked at x", "undefined symbol", "last lines:", "tool_output_read id a-1"].iter().all(|s| out.contains(s)) && !out.contains("automatic check"));
    let out = tools.command_feedback(&format!("status: Exited(Some(0))\nstdout:\n{}stderr:\nwarning: field `line` is never read\n --> src/main.rs:123:67\n{}", "ok\n".repeat(60), "Running tests\n".repeat(8)), None, true, 4096); assert!(out.len() <= CHECK_FEEDBACK_BYTES && out.starts_with("status: Exited(Some(0))") && out.contains("PASSED") && out.contains("warnings retained (2 matching lines):\nwarning: field `line` is never read\n --> src/main.rs:123:67\n") && !tools.check_failing());
    #[cfg(target_os = "macos")] {
        let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, approve).with_check(Some("printf checked > check-output"));
        for (script, changed) in [("printf edit > source", true), ("cat source", false)] {
            let args = json::parse(&format!(r#"{{"argv":["/bin/sh","-c","{script}"],"timeout_ms":1000}}"#)).unwrap(); let call = ToolCall { id: "shell-edit-read".into(), name: "workspace_shell".into(), arguments: json::stringify(&args).unwrap() };
            assert!(tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap().starts_with("status: Exited(Some(0))"));
            let pending = tools.take_pending_check(); assert_eq!(pending.is_some(), changed);
            if let Some(args) = pending { let call = ToolCall { id: "vade-check-probe".into(), name: "workspace_shell".into(), arguments: json::stringify(&args).unwrap() }; let result = tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap(); assert!(result.contains("automatic check #1:") && result.contains("PASSED") && tools.take_pending_check().is_none() && f.read("check-output") == "checked"); }
        }
    }
}

#[test]
fn diagnostic_rank_orders_failures_and_ignores_zero_counts() {
    for (line, rank) in [
        ("test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out", None),
        ("Found 0 errors and 0 warnings.", None),
        ("thread 'tests::ticks' panicked at src/cpu.rs:9:9:", Some(0)),
        ("error[E0425]: cannot find value `y` in this scope", Some(0)),
        ("error: unclosed delimiter", Some(0)),
        ("test tests::ticks ... FAILED", Some(1)),
        ("test result: FAILED. 1 passed; 1 failed; 0 ignored", Some(1)),
        ("assertion `left == right` failed", Some(2)),
        ("10 failed", Some(2)),
        ("warning: unused variable: `error`", Some(3)),
        ("  left: 3", None),
    ] {
        assert_eq!(diagnostic_rank(line), rank, "{line}");
    }
}

#[test]
fn cargo_test_feedback_keeps_panic_context_and_repeats_by_location() {
    let f = Fixture::new();
    let w = f.workspace();
    let approve = |_: Preview<'_>, _: &AtomicBool| true;
    let panic = |line: usize| {
        format!(
            "status: Exited(Some(101))\nstdout:\nrunning 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nrunning 2 tests\ntest tests::adds ... ok\ntest tests::ticks ... FAILED\nfailures:\n---- tests::ticks stdout ----\nthread 'tests::ticks' panicked at src/cpu.rs:{line}:9:\nassertion `left == right` failed\n  left: 3\n right: 4\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\nfailures:\n    tests::ticks\ntest result: FAILED. 1 passed; 1 failed; 0 ignored\nstderr:\n{}error: test failed, to rerun pass `--lib`\n",
            "warning: unused variable: `x`\n".repeat(40)
        )
    };
    let compile = |line: usize| {
        format!("status: Exited(Some(101))\nstdout:\nstderr:\nerror[E0425]: cannot find value `y` in this scope\n --> src/main.rs:{line}:13\nerror: could not compile `demo` (bin \"demo\") due to 1 previous error\n")
    };
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, approve).with_check(Some("cargo test"));
    let out = tools.command_feedback(&panic(9), None, true, 4096);
    assert!(out.contains("lines):\nthread 'tests::ticks' panicked at src/cpu.rs:9:9:\nassertion `left == right` failed\n  left: 3\n right: 4\nerror: test failed"), "{out}");
    assert!(!out.contains("0 passed; 0 failed") && !out.contains("note: run with"), "{out}");
    let cases: [(&dyn Fn(usize) -> String, bool); 4] = [(&panic, false), (&compile, false), (&panic, true), (&compile, true)];
    for (output, flagged) in cases {
        let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, approve).with_check(Some("cargo test"));
        // A new location each run is progress; the same location is a repeat.
        let lines: [usize; 3] = if flagged { [9, 9, 9] } else { [9, 10, 11] };
        let outs: Vec<String> = lines.iter().map(|line| tools.command_feedback(&output(*line), None, true, 4096)).collect();
        assert!(!outs[1].contains("hint:") && outs[2].contains("hint:") == flagged, "{}", outs[2]);
    }
}

#[test]
fn malformed_update_feedback_contains_a_short_actionable_change() {
    use crate::{agent::{self, Conversation, Limits, Model, ModelRequest}, agent_wire::AssistantTurn};
    struct Replay(usize);
    impl Model for Replay {
        fn complete(&mut self, request: ModelRequest<'_>, _: &AtomicBool, _: &mut dyn FnMut(&str) -> std::io::Result<()>) -> std::io::Result<AssistantTurn> {
            if self.0 == 1 {
                let out = request.messages.last().unwrap().value().get("content").and_then(Value::as_str).unwrap();
                assert!(out.len() <= 1024 && out.contains("First wrong line") && out.contains("use missing_prefix;"), "{out}");
                let example = json::parse(out.split("): ").last().unwrap()).unwrap();
                let patch = example.get("patch").and_then(Value::as_str).unwrap();
                assert!(patch.lines().any(|l| l.starts_with('+')) && patch.lines().any(|l| l.starts_with('-')), "{out}");
            }
            let reply = if self.0 < 2 {
                let patch = if self.0 == 0 { "*** Update File: seed.rs\n@@\nuse missing_prefix;" } else { "*** Update File: seed.rs\n@@\n-old\n+new" };
                format!(r#"{{"choices":[{{"finish_reason":"tool_calls","message":{{"role":"assistant","tool_calls":[{{"id":"edit-{}","type":"function","function":{{"name":"apply_patch","arguments":{}}}}}]}}}}]}}"#, self.0, json::quote(&json::stringify(&Value::Object(vec![("patch".into(), Value::String(patch.into()))]))?))
            } else { r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}"#.into() };
            self.0 += 1;
            AssistantTurn::from_response(&json::parse(&reply)?, request.wire)
        }
    }
    let f = Fixture::new(); f.write("seed.rs", "old\n"); let w = f.workspace();
    let mut approvals = 0;
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| { if preview.approval_required { approvals += 1; assert_eq!(f.read("seed.rs"), "old\n"); } true });
    let mut c = Conversation::from_request(json::parse(r#"{"tools":[{"type":"function","function":{"name":"apply_patch"}}],"messages":[{"role":"user","content":"edit seed.rs"}]}"#).unwrap(), Limits::default()).unwrap();
    let report = agent::run(&mut c, &mut Replay(0), &mut tools, Limits::default(), &AtomicBool::new(false), &mut |_| {});
    assert_eq!(report.reason, agent::StopReason::Finished(crate::agent_wire::FinishReason::Stop));
    assert_eq!(f.read("seed.rs"), "new\n");
    drop(tools); assert_eq!(approvals, 1);
}

#[test]
fn patch_failure_distinguishes_preflight_from_partial_writes() {
    // Exact cloud receipts: six creation/update variants and three feature variants.
    let receipts = [(r##"{"patch": "*** Begin Patch\n--- \n+++ Cargo.toml\n@@ -0,0 +1,6 @@\n+[package]\n+name = \"textstat\"\n+version = \"0.1.0\"\n+edition = \"2021\"\n+\n+[dependencies]\n", "retry": false}"##, 2), (r##"{"patch": "*** Begin Patch\n*** Add File: Cargo.toml\n--- \n+++\n@@ -0,0 +1,6 @@\n+[package]\n+name = \"textstat\"\n+version = \"0.1.0\"\n+edition = \"2021\"\n+\n+[dependencies]", "retry": false}"##, 3), (r##"{"patch": "*** Begin Patch\n*** Update File: Cargo.toml\n--- \n+++ \n@@ -1,6 +1,6 @@\n-[package]\n+[package]\n name = \"textstat\"\n version = \"0.1.0\"\n edition = \"2021\"\n-\n+[dependencies]\n", "retry": false}"##, 5), (r##"{"patch": "*** Begin Patch\n*** Update File: Cargo.toml\n--- \n+++ \n@@ -1,6 +1,6 @@\n [package]\n name = \"textstat\"\n version = \"0.1.0\"\n edition = \"2021\"\n-\n+[dependencies]\n", "retry": false}"##, 5), (r##"{"patch": "*** Begin Patch\n*** Update File: Cargo.toml\n--- \n+++ \n@@ -1,6 +1,6 @@\n- [package]\n+ [package]\n- name = \"textstat\"\n+ name = \"textstat\"\n- version = \"0.1.0\"\n+ version = \"0.1.0\"\n- edition = \"2021\"\n+ edition = \"2021\"\n- \n+ \n+[dependencies]\n", "retry": false}"##, 5), (r##"{"patch": "*** Begin Patch\n*** Update File: Cargo.toml\n--- \n+++ \n@@ -1,6 +1,6 @@\n [package]\n name = \"textstat\"\n version = \"0.1.0\"\n edition = \"2021\"\n \n+[dependencies]\n", "retry": false}"##, 5), (r##"{"patch":"*** Begin Patch\n--- a/src/cli.rs\n+++ b/src/cli.rs\n@@ -12,6 +12,10 @@ enum Command {\n     Warehouse(String),\n     Aggregate,\n     Report,\n+    LowStock(i64),\n }\n\n impl Command {\n@@ -14,6 +18,15 @@ fn parse_args(args: &[String]) -> Result<Self, String> {\n         let mut i = 1;\n         while i < args.len() {\n             match args[i].as_str() {\n+                \"--low-stock\" => {\n+                    i += 1;\n+                    let value = args.get(i).ok_or(\"low-stock value required\")?;\n+                    let threshold = value.parse::<i64>().map_err(|_| \"invalid low-stock threshold\")?;\n+                    if threshold < 0 {\n+                        return Err(\"low-stock threshold must be nonnegative\".into());\n+                    }\n+                    commands.push(Command::LowStock(threshold));\n+                }\n                 \"--warehouse\" => {\n                     i += 1;\n                     let value = args.get(i).ok_or(\"warehouse value required\")?;\n*** End Patch\n"}"##, 2), (r##"{"patch":"*** Update File: src/cli.rs\n+++ b/src/cli.rs\n@@ -12,6 +12,10 @@ enum Command {\n     Warehouse(String),\n     Aggregate,\n     Report,\n+    LowStock(i64),\n }\n\n impl Command {\n@@ -14,6 +18,15 @@ fn parse_args(args: &[String]) -> Result<Self, String> {\n         let mut i = 1;\n         while i < args.len() {\n             match args[i].as_str() {\n+                \"--low-stock\" => {\n+                    i += 1;\n+                    let value = args.get(i).ok_or(\"low-stock value required\")?;\n+                    let threshold = value.parse::<i64>().map_err(|_| \"invalid low-stock threshold\")?;\n+                    if threshold < 0 {\n+                        return Err(\"low-stock threshold must be nonnegative\".into());\n+                    }\n+                    commands.push(Command::LowStock(threshold));\n+                }\n                 \"--warehouse\" => {\n                     i += 1;\n                     let value = args.get(i).ok_or(\"warehouse value required\")?;\n*** End Patch\n"}"##, 3), (r##"{"patch":"*** Update File: src/cli.rs\n+++ b/src/cli.rs\n@@ -12,6 +12,10 @@ enum Command {\n+    LowStock(i64),\n }\n\n impl Command {\n@@ -14,6 +18,15 @@ fn parse_args(args: &[String]) -> Result<Self, String> {\n+                \"--low-stock\" => {\n+                    i += 1;\n+                    let value = args.get(i).ok_or(\"low-stock value required\")?;\n+                    let threshold = value.parse::<i64>().map_err(|_| \"invalid low-stock threshold\")?;\n+                    if threshold < 0 {\n+                        return Err(\"low-stock threshold must be nonnegative\".into());\n+                    }\n+                    commands.push(Command::LowStock(threshold));\n+                }\n                 \"--warehouse\" => {\n                     i += 1;\n                     let value = args.get(i).ok_or(\"warehouse value required\")?;\n*** End Patch\n"}"##, 3)];
    for (raw, wrong) in receipts { let f = Fixture::new(); std::fs::create_dir(f.0.join("src")).unwrap(); let args = json::parse(raw).unwrap(); let patch = args.get("patch").and_then(Value::as_str).unwrap(); if patch.contains("Update File: Cargo.toml") { f.write("Cargo.toml", "original manifest\n"); } f.write("src/cli.rs", "original source\n"); let w = f.workspace(); let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true); let call = ToolCall { id: "cloud".into(), name: "apply_patch".into(), arguments: raw.into() }; let out = tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap();
        if patch.contains("Update File:") && patch.contains("@@ -") { assert!(out.starts_with("Patch refused:") && (out.contains("old text missing") && out.contains("workspace_read_lines") || out.contains("requires nonempty exact old text")), "{out}"); assert_eq!(f.read("src/cli.rs"), "original source\n"); assert!(!f.0.join("Cargo.toml").exists() || f.read("Cargo.toml") == "original manifest\n"); continue; }
        assert!(out.starts_with("Patch refused:") && out.contains(&format!("First wrong line {wrong} ")) && out.contains(&json::quote(patch.lines().nth(wrong - 1).unwrap())) && out.contains("Expected patch string:"), "{out}"); assert_eq!(f.read("src/cli.rs"), "original source\n"); assert!(!f.0.join("Cargo.toml").exists() || f.read("Cargo.toml") == "original manifest\n"); let example = out.split("): ").last().unwrap(); let example = json::parse(example).unwrap(); let sample = example.get("patch").and_then(Value::as_str).unwrap(); assert!(!sample.contains("@@ -") && !sample.contains("--- ") && !sample.contains("+++ ")); let old: String = sample.lines().filter_map(|line| line.strip_prefix(' ').or_else(|| line.strip_prefix('-'))).map(|line| format!("{line}\n")).collect(); let path = if sample.contains("Cargo.toml") { "Cargo.toml" } else { "src/cli.rs" }; if sample.contains("Update File:") { f.write(path, &old); } assert!(crate::edit::prepare(&w, sample).is_ok(), "{sample}"); let retry = json::parse(r#"{"patch":"*** Add File: Cargo.toml","retry":true}"#).unwrap(); assert!(tools.execute(ToolInvocation { call: &call, arguments: &retry, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap().contains("invalid_arguments"));
    }
    for (existing, concurrent) in [(true, false), (false, false), (true, true), (false, true)] { let f = Fixture::new(); if existing { f.write("reuse", "old"); } let w = f.workspace(); let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
        let mut invoke = |args: Value| { let call = ToolCall { id: "retry".into(), name: "apply_patch".into(), arguments: json::stringify(&args).unwrap() }; tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap() }; let refused = invoke(json::parse(&format!(r#"{{"patch":"*** Add File: reuse\n{}first\nsecond"}}"#, if existing { "" } else { "+" })).unwrap()); assert!(refused.contains("Retained") && (!existing || f.read("reuse") == "old")); if concurrent { f.write("reuse", "concurrent\n"); }
        let header = format!("*** {} File: reuse", if existing { "Update" } else { "Add" }); let expected = format!("{{\"patch\":{},\"retry\":true}}", json::quote(&header)); for raw in [format!("{{\"patch\":{},\"retry\":true}}", json::quote(&format!("{header}\\n"))), format!("{{\"patch\":{},\"retry\":{{}}}}", json::quote(&header)), format!("{{\"patch\":{},\"retry\":{{\"patch\":\"ignored\"}}}}", json::quote(&header)), r#"{"retry":true}"#.into(), r#"{"patch":[],"retry":true}"#.into(), format!("{{\"patch\":{},\"retry\":true,\"extra\":1}}", json::quote(&header))] { let out = invoke(json::parse(&raw).unwrap()); let v = json::parse(&out).unwrap(); assert_eq!(v.get("error").and_then(Value::as_str), Some("invalid_arguments")); assert!(v.get("message").and_then(Value::as_str).unwrap().contains(&expected) && out.lines().count() == 1); assert!(!f.0.join("reuse").exists() || f.read("reuse") == if concurrent { "concurrent\n" } else { "old" }); }
        let result = invoke(json::parse(&format!("{{\"patch\":{},\"retry\":true}}", json::quote(&format!("{header}\n")))).unwrap()); assert_eq!(f.read("reuse"), if concurrent { "concurrent\n" } else { "first\nsecond\n" }); assert_eq!(result.starts_with("Applied "), !concurrent); let again = invoke(json::parse(&expected).unwrap()); assert!(again.contains("invalid_arguments") && !again.contains(&json::quote(&expected)));
    }
    for partial in [false, true] {
        let f = Fixture::new();
        f.write("a", "old\n");
        f.write("b", "old\n");
        let w = f.workspace();
        let mut tools = CodingTools::new(
            &w,
            ApprovalPolicy::Ask,
            |preview: Preview<'_>, _: &AtomicBool| {
                if (!partial && preview.title == "Patch preview")
                    || (partial && preview.title == "File written")
                {
                    f.write("b", "concurrent\n");
                }
                true
            },
        );
        let args = Value::Object(vec![("patch".into(), Value::String(
            "*** Begin Patch\n*** Update File: a\n@@\n-old\n+new\n*** Update File: b\n@@\n-old\n+new\n*** End Patch".into()
        ))]);
        let call = ToolCall {
            id: "conflict".into(),
            name: "apply_patch".into(),
            arguments: json::stringify(&args).unwrap(),
        };
        let result = tools
            .execute(
                ToolInvocation {
                    call: &call,
                    arguments: &args,
                    max_output_bytes: 4096,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        let result = json::parse(&result).unwrap();
        assert_eq!(result.get("uncertain_writes"), Some(&Value::Bool(partial)));
        assert_eq!(tools.last_side_effect_uncertain(), partial);
        assert_eq!(
            result.get("error").and_then(Value::as_str),
            Some(if partial {
                "unknown_outcome"
            } else {
                "patch_refused"
            })
        );
        assert_eq!(
            result
                .get("committed")
                .and_then(Value::as_array)
                .unwrap()
                .len(),
            usize::from(partial)
        );
        assert_eq!(f.read("a"), if partial { "new\n" } else { "old\n" });
        assert_eq!(f.read("b"), "concurrent\n");
    }
}
#[test]
#[cfg(not(target_os = "macos"))]
fn unsupported_shell_returns_clear_error_before_approval_without_uncertainty() {
    let f = Fixture::new();
    let w = f.workspace();
    let mut tools =
        CodingTools::new(&w, ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| {
            panic!("unsupported commands must not ask approval")
        });
    let args =
        json::parse(r#"{"argv":["/bin/sh","-c","touch output"],"timeout_ms":1000}"#).unwrap();
    let call = ToolCall {
        id: "unsupported-shell".into(),
        name: "workspace_shell".into(),
        arguments: json::stringify(&args).unwrap(),
    };
    let result = tools
        .execute(
            ToolInvocation {
                call: &call,
                arguments: &args,
                max_output_bytes: 4096,
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    let error = json::parse(&result).unwrap();
    assert_eq!(error.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(
        error.get("error").and_then(Value::as_str),
        Some("unsupported_platform")
    );
    assert!(
        error
            .get("message")
            .and_then(Value::as_str)
            .unwrap()
            .contains("macOS sandbox")
    );
    assert!(!tools.last_side_effect_uncertain());
    assert!(!f.0.join("output").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn workspace_shell_builds_and_tests_real_std_only_crate_offline() {
    const MARKER: &str = "VADE_BUILD_PRIVACY_MARKER";
    const VALUE: &str = "synthetic-build-parent-env-marker";
    if std::env::var_os(MARKER).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "coding_tools::tests::workspace_shell_builds_and_tests_real_std_only_crate_offline", "--nocapture"])
            .env_clear()
            .env(MARKER, VALUE)
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    // Positive control: the synthetic marker is in the unsandboxed test
    // process's argument/environment buffer. Never print that buffer.
    unsafe extern "C" {
        fn sysctl(
            name: *const i32,
            count: u32,
            out: *mut u8,
            len: *mut usize,
            input: *const u8,
            input_len: usize,
        ) -> i32;
    }
    let mut bytes = vec![0u8; 1024 * 1024];
    let mut len = bytes.len();
    let mib = [1, 49, std::process::id() as i32];
    assert_eq!(
        unsafe {
            sysctl(
                mib.as_ptr(),
                3,
                bytes.as_mut_ptr(),
                &mut len,
                std::ptr::null(),
                0,
            )
        },
        0
    );
    assert!(
        bytes[..len]
            .windows(VALUE.len())
            .any(|b| b == VALUE.as_bytes())
    );
    let f = Fixture::new();
    let outside = Fixture::new();
    outside.write("private", "synthetic-private-hardlink-marker");
    for name in ["target", "node_modules"] {
        std::fs::create_dir(f.0.join(name)).unwrap();
        std::fs::hard_link(outside.0.join("private"), f.0.join(name).join("alias")).unwrap();
    }
    std::fs::create_dir(f.0.join("src")).unwrap();
    f.write(
        "Cargo.toml",
        "[package]\nname = \"sandbox-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    );
    f.write("src/lib.rs", &format!(r#"
pub fn sum(a: u32, b: u32) -> u32 {{ a + b }}
#[test]
fn adds_and_preserves_sandbox_privacy() {{
    assert_eq!(sum(2, 3), 5);
    unsafe extern "C" {{
        fn getpagesize() -> i32;
        fn sysctlbyname(name: *const i8, out: *mut u8, len: *mut usize, input: *const u8, input_len: usize) -> i32;
        fn sysctl(name: *const i32, count: u32, out: *mut u8, len: *mut usize, input: *const u8, input_len: usize) -> i32;
    }}
    assert!(unsafe {{ getpagesize() }} > 0);
    let mut bytes = vec![0u8; 1024 * 1024];
    let mut len = bytes.len();
    assert_ne!(unsafe {{ sysctlbyname(c"kern.hostname".as_ptr(), bytes.as_mut_ptr(), &mut len, std::ptr::null(), 0) }}, 0);
    for kind in [38, 49] {{ // KERN_PROCARGS and KERN_PROCARGS2
        bytes.fill(0);
        len = bytes.len();
        let parent_args = [1, kind, {parent}]; // CTL_KERN, test harness PID
        let status = unsafe {{ sysctl(parent_args.as_ptr(), 3, bytes.as_mut_ptr(), &mut len, std::ptr::null(), 0) }};
        if status == 0 {{
            assert!(!bytes[..len].windows(b"synthetic-build-parent-env-marker".len()).any(|b| b == b"synthetic-build-parent-env-marker"), "parent environment leaked");
        }}
    }}
}}
"#, parent=std::process::id()));
    f.write("build.rs", "fn main() { let out = std::path::PathBuf::from(std::env::var_os(\"OUT_DIR\").unwrap()); std::fs::write(out.join(\"generated\"), b\"std-only build script\").unwrap(); }\n");
    // Cargo supplies its selected absolute executable when compiling tests.
    // No discovery of credentials, registry, rustup settings or live service.
    let cargo = std::path::Path::new(env!("CARGO")).canonicalize().unwrap();
    let toolchain = cargo.parent().unwrap().parent().unwrap();
    let session = shell::BuildSession::new(Some(toolchain)).unwrap();
    let workspace = f.workspace();
    let mut approvals = 0;
    let mut executor = CodingTools::new(
        &workspace,
        ApprovalPolicy::Ask,
        |preview: Preview<'_>, _: &AtomicBool| {
            if preview.title == "Command" {
                assert!(preview.approval_required);
                assert!(
                    preview
                        .detail
                        .contains("private session build/cache/temp area")
                );
                approvals += 1;
            }
            true // explicit fixture approval, never an auto-allowlisted build
        },
    )
    .with_build_session(&session);
    let mut invoke = |argv: &[&str]| {
        let arguments = Value::Object(vec![
            (
                "argv".into(),
                Value::Array(argv.iter().map(|s| Value::String((*s).into())).collect()),
            ),
            ("timeout_ms".into(), Value::Number("120000".into())),
        ]);
        let call = ToolCall {
            id: "local-build".into(),
            name: "workspace_shell".into(),
            arguments: json::stringify(&arguments).unwrap(),
        };
        executor
            .execute(
                ToolInvocation {
                    call: &call,
                    arguments: &arguments,
                    max_output_bytes: 65536,
                },
                &AtomicBool::new(false),
            )
            .unwrap()
    };
    for argv in [
        ["cargo", "build", "--offline"],
        ["cargo", "test", "--offline"],
        ["cargo", "build", "--offline"],
    ] {
        let result = invoke(&argv);
        assert!(result.starts_with("status: Exited(Some(0))"), "{result}");
        if argv[1] == "test" {
            assert!(result.contains("1 passed; 0 failed"), "{result}");
        }
    }
    let result = invoke(&[
        "/bin/sh",
        "-c",
        "test -d \"$CARGO_TARGET_DIR/debug\" && test \"$CARGO_NET_OFFLINE\" = true && test \"$HOME\" != \"$PWD\" && ! cat target/alias && ! cat node_modules/alias && ! printf changed > target/alias",
    ]);
    assert!(result.starts_with("status: Exited(Some(0))"), "{result}");
    assert!(!result.contains("synthetic-private-hardlink-marker"));
    drop(executor);
    assert_eq!(approvals, 4);
    let mut strict = CodingTools::new(&workspace, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true).with_build_session(&session).with_check(Some("cargo test --offline"));
    for warning in [true, false] { f.write("src/lib.rs", if warning { "pub fn value() { let unused = 1; }\n" } else { "pub fn value() {}\n" }); strict.mark_changed(); let arguments = strict.take_pending_check().unwrap(); let call = ToolCall { id: "vade-check-warning".into(), name: "workspace_shell".into(), arguments: json::stringify(&arguments).unwrap() };
        let result = strict.execute(ToolInvocation { call: &call, arguments: &arguments, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap(); assert_eq!(strict.check_failing(), warning, "{result}"); assert!(result.len() <= CHECK_FEEDBACK_BYTES && if warning { result.contains("FAILED (exit 101)") && result.contains("unused variable") } else { result.contains("PASSED") }, "{result}");
    }
    assert_eq!(outside.read("private"), "synthetic-private-hardlink-marker");
    assert!(!f.0.join("target/debug").exists());
    assert!(f.0.join("Cargo.lock").is_file());
}
struct Full;
impl Access for Full {
    fn check_capacity(&mut self) -> io::Result<()> {
        Err(io::Error::other("full"))
    }
    fn retain(&mut self, _: &str) -> io::Result<String> {
        panic!("must preflight")
    }
    fn read(&mut self, _: &str, _: usize, _: usize, _: usize) -> io::Result<String> {
        panic!()
    }
}
#[test]
fn artifact_capacity_refusal_precedes_patch_approval_and_side_effects() {
    let f = Fixture::new();
    let w = f.workspace();
    let mut full = Full;
    let mut tools = CodingTools::new(
        &w,
        ApprovalPolicy::Auto,
        |_: Preview<'_>, _: &AtomicBool| {
            panic!("quota must fail before preview or side effects")
        },
    )
    .with_artifacts(&mut full);
    let args = Value::Object(vec![(
        "patch".into(),
        Value::String("*** Begin Patch\n*** Add File: output\n+created\n*** End Patch".into()),
    )]);
    let call = ToolCall {
        id: "c".into(),
        name: "apply_patch".into(),
        arguments: json::stringify(&args).unwrap(),
    };
    assert_eq!(
        tools.execute(
            ToolInvocation {
                call: &call,
                arguments: &args,
                max_output_bytes: 65536
            },
            &AtomicBool::new(false)
        ),
        Err(ToolError::OutputLimit)
    );
    assert!(!f.0.join("output").exists());
}
#[test]
fn workspace_read_returns_pdf_text_through_the_tool() {
    let f = Fixture::new(); std::fs::write(f.0.join("manual.pdf"), crate::pdf::tests::sample(&["Install the tool", "Run the tool"])).unwrap(); let w = f.workspace();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
    let mut invoke = |name: &str, raw: &str| { let args = json::parse(raw).unwrap(); let call = ToolCall { id: "pdf".into(), name: name.into(), arguments: raw.into() }; tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap() };
    let whole = invoke("workspace_read", r#"{"path":"manual.pdf"}"#);
    assert!(whole.starts_with("Path: manual.pdf\nPDF text layer extracted by Vade: 2 pages.") && whole.ends_with("page 1 of 2\nInstall the tool\n\npage 2 of 2\nRun the tool\n"), "{whole}");
    let part = invoke("workspace_read", r#"{"path":"manual.pdf","offset":0,"limit":20}"#);
    assert!(part.contains("Read bytes 0..20 of ") && part.contains(" of extracted PDF text (source PDF ") && part.ends_with("next offset 20).\nPDF text layer extra"), "{part}");
    assert_eq!(invoke("workspace_read_lines", r#"{"path":"manual.pdf","start_line":1,"end_line":3}"#), "Discovery refused: source is a PDF; read its extracted text with workspace_read");
}
#[test]
fn workspace_read_lines_names_workspace_read_for_an_ascii_pdf() {
    let f = Fixture::new(); let pdf = crate::pdf::tests::sample_ascii(&["Plain"]); assert!(pdf.is_ascii()); std::fs::write(f.0.join("plain.pdf"), pdf).unwrap(); let w = f.workspace();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
    let raw = r#"{"path":"plain.pdf","start_line":1,"end_line":3}"#; let args = json::parse(raw).unwrap(); let call = ToolCall { id: "pdf".into(), name: "workspace_read_lines".into(), arguments: raw.into() };
    assert_eq!(tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap(), "Discovery refused: source is a PDF; read its extracted text with workspace_read");
}
