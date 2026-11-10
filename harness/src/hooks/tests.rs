use super::*;
use crate::test_workspace::Fixture;

fn configured(script: &str, point: &str, timeout: u64) -> (Fixture, Hooks) {
    let fixture = Fixture::new();
    fixture.write("hook.sh", script);
    let workspace = fixture.workspace();
    workspace.state_dir().unwrap();
    let config = format!(r#"{{"hooks":[{{"point":"{point}","command":"/bin/sh hook.sh","timeout_ms":{timeout}}}]}}"#);
    fixture.write(".vade/hooks.conf", &config);
    let hooks = Hooks::load(&workspace).unwrap();
    (fixture, hooks)
}

#[test]
fn allow_gets_json_environment_and_project_cwd() {
    let (fixture, hooks) = configured("cat > input.json\nprintf '%s\\n' \"$VADE_HOOK_POINT\" \"$VADE_TOOL\" \"$VADE_PROJECT\" \"$PWD\" > env.txt\n", "before-tool", 2000);
    let args = json::parse(r#"{"path":"a b","line":7}"#).unwrap();
    let mut warnings = Vec::new();
    assert!(hooks.before("workspace_read", &args, &AtomicBool::new(false), &mut |s| warnings.push(s)).is_none());
    assert!(warnings.is_empty(), "{warnings:?}");
    let input = json::parse(&fixture.read("input.json")).unwrap();
    assert_eq!(input.get("arguments"), Some(&args));
    assert_eq!(input.get("tool").and_then(Value::as_str), Some("workspace_read"));
    let root = fixture.0.canonicalize().unwrap();
    assert_eq!(fixture.read("env.txt"), format!("before-tool\nworkspace_read\n{}\n{}\n", root.display(), root.display()));
    assert_eq!(Hooks::parse(&root, r#"{"hooks":[{"point":"on-stop","command":"true"}]}"#).unwrap().entries[0].timeout, Duration::from_secs(30));
}

#[test]
fn block_uses_stderr_and_other_exits_allow_with_warning() {
    let (fixture, hooks) = configured("printf 'local rule' >&2\nexit 2\n", "before-tool", 2000);
    let mut warnings = Vec::new();
    let args = Value::Object(Vec::new());
    assert_eq!(hooks.before("apply_patch", &args, &AtomicBool::new(false), &mut |s| warnings.push(s)).unwrap(), "Hook blocked apply_patch: local rule");
    assert!(warnings.is_empty());
    fixture.write("hook.sh", "printf 'broken check' >&2\nexit 7\n");
    assert!(hooks.before("apply_patch", &args, &AtomicBool::new(false), &mut |s| warnings.push(s)).is_none());
    assert!(warnings[0].contains("exit 7. broken check"));
}

#[test]
fn timeout_does_not_require_script_readiness_or_stdin_reads() {
    let (fixture, hooks) = configured("sleep 10\n", "before-tool", 100);
    let mut warnings = Vec::new();
    assert!(hooks.before("read", &Value::Null, &AtomicBool::new(false), &mut |s| warnings.push(s)).is_none());
    assert!(warnings[0].contains("timed out"));
    // A hook need not read stdin. A full stdin pipe must not defeat its timeout.
    let entry = Entry { point: Point::BeforeTool, command: "sleep 10".into(), timeout: Duration::from_millis(50) };
    assert_eq!(run(&fixture.0, &entry, "read", &"x".repeat(256 * 1024), &AtomicBool::new(false)).err().unwrap().kind(), io::ErrorKind::TimedOut);
}

#[test]
fn cancellation_stops_the_ready_owned_shell() {
    let (fixture, hooks) = configured("touch must-not-run\n", "before-tool", 30_000);
    hooks.before("read", &Value::Null, &AtomicBool::new(true), &mut |_| panic!("already cancelled"));
    assert!(!fixture.0.join("must-not-run").exists());
    fixture.write("hook.sh", "echo $$ > hook.pid\nexec sleep 30\n");
    let entry = Entry { point: Point::BeforeTool, command: "exec /bin/sh hook.sh".into(), timeout: Duration::from_secs(30) };
    let cancel = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| run(&fixture.0, &entry, "read", "{}", &cancel));
        let start = Instant::now();
        let pid = loop {
            if let Ok(value) = std::fs::read_to_string(fixture.0.join("hook.pid"))
                && let Ok(pid) = value.trim().parse::<u32>() { break Some(pid); }
            if start.elapsed() >= Duration::from_secs(10) { break None; }
            std::thread::sleep(Duration::from_millis(5));
        };
        cancel.store(true, Ordering::Relaxed);
        let result = worker.join().unwrap();
        let pid = pid.expect("hook must signal readiness before cancellation");
        assert!(result.err().unwrap().to_string().contains("cancelled"));
        assert!(!Command::new("/bin/kill").args(["-0", &pid.to_string()]).stderr(Stdio::null()).status().unwrap().success());
    });
}

#[test]
fn completed_hooks_keep_status_and_output_with_inherited_pipes() {
    let (fixture, hooks) = configured("sleep 30 &\nprintf 'denied by policy' >&2\nexit 2\n", "before-tool", 5000);
    let cancel = AtomicBool::new(false);
    assert_eq!(hooks.before("apply_patch", &Value::Null, &cancel, &mut |s| panic!("{s}")).as_deref(),
        Some("Hook blocked apply_patch: denied by policy"));
    fixture.write("hook.sh", "sleep 30 &\nprintf 'done'\nexit 0\n");
    assert!(hooks.before("read", &Value::Null, &cancel, &mut |s| panic!("{s}")).is_none());
    let mut after = hooks.clone();
    after.entries[0].point = Point::AfterTool;
    let mut receipt = "receipt".to_owned();
    after.after("read", &Value::Null, &mut receipt, 8192, &cancel, &mut |s| panic!("{s}"));
    assert_eq!(receipt, "receipt\n[after-tool hook note]\ndone");
    after.entries[0].point = Point::OnStop;
    let mut shown = Vec::new();
    after.stop(&mut |s| shown.push(s));
    assert_eq!(shown, ["[on-stop hook]\ndone"]);
}

#[test]
fn incomplete_capture_keeps_bounded_output_and_a_label() {
    for failed in [false, true] {
        let capture = Mutex::new(Capture { bytes: vec![b'x'; CAP], done: false, failed, cut: false });
        let text = captured(&capture);
        assert!(text.len() <= CAP);
        assert!(text.ends_with("[hook output incomplete]"));
    }
}

#[test]
fn output_cap_drains_both_streams_and_after_note_keeps_receipt() {
    let (fixture, hooks) = configured("cat > after.json\ni=0; while [ \"$i\" -lt 5000 ]; do printf 'abcd'; printf 'error' >&2; i=$((i+1)); done\n", "after-tool", 5000);
    let mut result = "original receipt".to_owned();
    let mut warnings = Vec::new();
    hooks.after("read", &Value::Null, &mut result, 8192, &AtomicBool::new(false), &mut |s| warnings.push(s));
    assert!(result.starts_with("original receipt\n[after-tool hook note]\n"));
    assert!(result.len() <= "original receipt".len() + CAP);
    assert_eq!(json::parse(&fixture.read("after.json")).unwrap().get("result_summary").and_then(Value::as_str), Some("original receipt"));
    assert!(warnings.iter().any(|s| s.contains("cut")));
    fixture.write("hook.sh", "printf 'note'\n");
    let mut small = "full receipt".to_owned();
    let cap = small.len();
    hooks.after("read", &Value::Null, &mut small, cap, &AtomicBool::new(false), &mut |_| {});
    assert_eq!(small, "full receipt");
    let mut failed = r#"{"ok":false,"error":"tool_failed"}"#.to_owned();
    hooks.after("read", &Value::Null, &mut failed, 8192, &AtomicBool::new(false), &mut |_| {});
    assert!(crate::context::has_unknown_outcome_content(&failed));
    assert!(json::parse(&failed).unwrap().get("vade_hook_note").and_then(Value::as_str).unwrap().contains("note"));
}

#[test]
fn file_order_stops_at_the_first_block() {
    let fixture = Fixture::new();
    fixture.write("first.sh", "printf a >> order; exit 0");
    fixture.write("second.sh", "printf b >> order; printf stop >&2; exit 2");
    fixture.write("third.sh", "printf c >> order; exit 0");
    let hooks = Hooks::parse(&fixture.0, r#"{"hooks":[
        {"point":"before-tool","command":"/bin/sh first.sh"},
        {"point":"before-tool","command":"/bin/sh second.sh"},
        {"point":"before-tool","command":"/bin/sh third.sh"}
    ]}"#).unwrap();
    assert!(hooks.before("read", &Value::Null, &AtomicBool::new(false), &mut |_| panic!("unexpected warning")).unwrap().contains("stop"));
    assert_eq!(fixture.read("order"), "ab");
}

#[test]
fn stop_output_is_labeled_and_config_errors_reject_the_whole_file() {
    let (fixture, hooks) = configured("printf '%s:%s' \"$VADE_HOOK_POINT\" \"$VADE_TOOL\"\n", "on-stop", 2000);
    let mut shown = Vec::new();
    hooks.stop(&mut |text| shown.push(text));
    assert_eq!(shown, ["[on-stop hook]\non-stop:"]);
    for source in ["{", r#"{"hooks":[],"extra":true}"#, r#"{"hooks":[{"point":"bad","command":"true"}]}"#,
        r#"{"hooks":[{"point":"on-stop","command":"true","timeout_ms":0}]}"#,
        r#"{"hooks":[{"point":"on-stop","command":"true","command":"false"}]}"#] {
        fixture.write(".vade/hooks.conf", source);
        assert!(Hooks::load(&fixture.workspace()).is_err());
    }
}

#[test]
fn missing_hooks_are_inert_and_links_are_refused() {
    let fixture = Fixture::new();
    assert!(Hooks::load(&fixture.workspace()).unwrap().entries.is_empty());
    #[cfg(unix)]
    {
        fixture.write("external.conf", r#"{"hooks":[]}"#);
        std::os::unix::fs::symlink("../external.conf", fixture.0.join(".vade/hooks.conf")).unwrap();
        assert!(Hooks::load(&fixture.workspace()).is_err());
    }
}
