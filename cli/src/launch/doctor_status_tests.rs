//! The read-mostly local commands (doctor, status, detect, logs) and the
//! down/update guards on a fresh, installed, running and stopped launch home,
//! through the same mock runner as the launch tests.
use super::*;
use crate::evals::json::{self, Value};

fn captured(run: impl FnOnce() -> Result<(), String>) -> (Result<(), String>, String) {
    let capture = crate::output::testing::Capture::default();
    let result = output::with_writer(capture.clone(), run);
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    (result, printed)
}

fn snapshot<R: Runner, D: Detector>(launch: &mut Launcher<R, D>, dry: bool) -> Result<Value, String> {
    let mut report = crate::status::Snapshot::new("local", dry);
    launch.status_snapshot(&mut report, dry)?;
    Ok(json::parse(&report.finish(Duration::ZERO).0).unwrap())
}

#[test]
fn doctor_reports_every_check_and_counts_the_failures() {
    let f = Fixture::new();
    let mut launch = f.launcher();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    assert_eq!(result.unwrap_err(), "2 local check(s) need attention. Follow the next commands above, then retry vade doctor.");
    for line in [
        "Config: No saved local defaults.\nNext: Run vade launch to create defaults",
        "Ownership: OK.",
        "Installed engine, kernels and chat: Local artifacts are missing or have changed.\nNext: Run vade down, then vade install",
        "Model: OK.",
        "Machine and NVIDIA driver: OK.",
        "Disk: OK.",
        "Memory and launch limits: OK.",
        "Engine: stopped. Start: ",
    ] {
        assert!(printed.contains(line), "{line:?} missing from:\n{printed}");
    }
    assert!(!printed.contains("Local checks passed."), "{printed}");

    // A fresh install needs install room as well as the model volume's.
    launch.detector.identity.install_free = GIB;
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    assert!(result.unwrap_err().starts_with("3 local check(s) need attention."));
    assert!(printed.contains("Disk: Insufficient disk working room: install needs 8.0 GiB and model volume needs 1 GiB.\nNext: Free space"), "{printed}");

    launch.detector.identity = identity();
    f.installed();
    f.configured();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    result.unwrap();
    let tail = format!("Engine: stopped. Start: {}\nLocal checks passed.\n", text::command("launch"));
    assert!(printed.ends_with(&tail), "{printed}");
    assert_eq!(launch.runner.starts, 0, "doctor must not start the engine");

    captured(|| launch.execute(Command::Up(Default::default()), false)).0.unwrap();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    result.unwrap();
    assert!(printed.contains("Memory: resident engine recorded; available memory is shown above, without reserving its allocation again."), "{printed}");
    assert!(printed.contains("Owned endpoint and health: OK.\nLocal checks passed.\n"), "{printed}");
    assert_eq!(launch.runner.starts, 1);
    captured(|| launch.execute(Command::Down, false)).0.unwrap();
}

#[test]
fn status_dry_run_and_json_read_only_the_saved_records() {
    let f = Fixture::new();
    let mut launch = f.launcher();
    let (result, printed) = captured(|| launch.execute(Command::Status, true));
    result.unwrap();
    assert_eq!(printed, format!("Vade is stopped. Start: {}\n", text::command("launch")));
    let stopped = snapshot(&mut launch, false).unwrap();
    assert_eq!((stopped["phase"].as_str(), &stopped["ready"]), (Some("stopped"), &Value::Bool(false)));
    assert!(stopped["errors"].as_array().unwrap().is_empty());

    f.installed();
    captured(|| launch.execute(Command::Launch(Default::default()), false)).0.unwrap();
    let calls = launch.runner.calls.len();
    let (result, printed) = captured(|| launch.execute(Command::Status, true));
    result.unwrap();
    let config = Config::load(&f.root).unwrap().unwrap();
    assert_eq!(printed, format!(
        "Vade recorded phase: running at {}. Readiness not checked (dry run). Local config: {}\n",
        config.endpoint(), f.root.join("config").display()
    ));
    let dry = snapshot(&mut launch, true).unwrap();
    assert_eq!(dry["phase"].as_str(), Some("running"));
    assert_eq!(dry["ready"], Value::Null, "a dry run must not claim readiness");
    assert_eq!(dry["ranks"][0]["alive"], Value::Null);
    assert!(dry["checks"].as_array().unwrap().iter().all(|check| check["ok"] == Value::Null));
    assert_eq!(launch.runner.calls.len(), calls, "dry status must not probe the engine");

    // A running engine whose config went missing is an error, never 'stopped'.
    fs::rename(f.root.join("config"), f.root.join("config.moved")).unwrap();
    let expected = "An owned engine has no saved config; preserve the state folder and run vade down.";
    assert_eq!(captured(|| launch.execute(Command::Status, true)).0.unwrap_err(), expected);
    assert_eq!(snapshot(&mut launch, false).unwrap_err(), expected);
    fs::rename(f.root.join("config.moved"), f.root.join("config")).unwrap();
    captured(|| launch.execute(Command::Down, false)).0.unwrap();
}

#[test]
fn down_status_logs_and_update_name_the_next_step_in_each_state() {
    let f = Fixture::new();
    let mut launch = f.launcher();
    let run = |launch: &mut Launcher<MockRunner, MockDetector>, command| captured(|| launch.execute(command, false));
    assert_eq!(run(&mut launch, Command::Down), (Ok(()), "Vade is already stopped.\n".into()));
    assert_eq!(run(&mut launch, Command::Status), (Ok(()), format!("Vade is stopped. Start: {}\n", text::command("launch"))));
    assert_eq!(run(&mut launch, Command::Logs), (Ok(()), "No engine log yet.\n".into()));
    assert_eq!(run(&mut launch, Command::Update).0.unwrap_err(), "No local setup yet; run vade launch first.");
    let (result, printed) = run(&mut launch, Command::Detect);
    result.unwrap();
    assert!(printed.contains("Machine: Fixture OEM / GB10 test box\n") && printed.contains("Disk free: 80.0 GiB install, 60.0 GiB models\n"), "{printed}");
    assert!(!f.root.join("config").exists(), "detect must not save defaults");

    f.installed();
    run(&mut launch, Command::Launch(Default::default())).0.unwrap();
    let config = Config::load(&f.root).unwrap().unwrap();
    assert_eq!(run(&mut launch, Command::Status), (Ok(()), format!("Vade is ready at {}. Chat: {}\n", config.endpoint(), text::command("chat"))));
    assert_eq!(run(&mut launch, Command::Logs), (Ok(()), "fixture engine diagnostic".into()));
    assert_eq!(run(&mut launch, Command::Update).0.unwrap_err(), "The local engine is running; run vade down, then vade update and vade up.");
    run(&mut launch, Command::Down).0.unwrap();

    // After a stop, logs shows the retained log with terminal controls escaped.
    let saved = f.root.join("state/last-server.log");
    fs::write(&saved, "engine said \x1b[31mbye\n").unwrap();
    assert_eq!(run(&mut launch, Command::Logs), (Ok(()), "engine said \\u{1b}[31mbye\n".into()));
    fs::remove_file(&saved).unwrap();
    fs::create_dir(&saved).unwrap();
    assert_eq!(run(&mut launch, Command::Logs).0.unwrap_err(), "Cannot read the saved engine log; it must be a regular readable file.");
}
