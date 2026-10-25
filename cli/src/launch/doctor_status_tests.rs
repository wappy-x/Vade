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
    // `state  name` at fixed columns, so each row can be found by its start.
    let row = |state: &str, name: &str| format!("{state:<4}  {name:<20}  ");
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    assert_eq!(result.unwrap_err(), "2 doctor check(s) failed. Run each next: command above, then vade doctor.");
    for line in [
        format!("{}No saved local defaults.\n      next: Run vade launch to create defaults", row("FAIL", "config")),
        row("ok", "VADE_HOME writable"),
        format!("{}no engine recorded", row("ok", "ownership")),
        format!("{}Local artifacts are missing or have changed.\n      next: Run vade down, then vade install", row("FAIL", "installed binaries")),
        row("ok", "model files"),
        row("ok", "nvidia-smi"),
        row("ok", "disk"),
        row("ok", "free memory"),
        "ok    port ".into(),
        format!("{}stopped; start: ", row("skip", "engine health")),
        "FAIL  7 ok, 2 failed, 1 skipped in ".into(),
    ] {
        assert!(printed.contains(&line), "{line:?} missing from:\n{printed}");
    }

    // A fresh install needs install room as well as the model volume's.
    launch.detector.identity.install_free = GIB;
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    assert!(result.unwrap_err().starts_with("3 doctor check(s) failed."));
    assert!(printed.contains(&format!("{}Insufficient disk working room: install needs 8.0 GiB and model volume needs 1 GiB.\n      next: Free space", row("FAIL", "disk"))), "{printed}");

    launch.detector.identity = identity();
    f.installed();
    f.configured();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    result.unwrap();
    assert!(printed.contains(&format!("{}stopped; start: {}\nPASS  9 ok, 0 failed, 1 skipped in ", row("skip", "engine health"), text::command("launch"))), "{printed}");
    assert_eq!(launch.runner.starts, 0, "doctor must not start the engine");

    captured(|| launch.execute(Command::Up(Default::default()), false)).0.unwrap();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    result.unwrap();
    assert!(printed.contains("the running engine holds its share"), "{printed}");
    assert!(printed.contains("skip  port ") && printed.contains(&row("ok", "engine health")), "{printed}");
    assert!(printed.contains("\nPASS  9 ok, 0 failed, 1 skipped in "), "{printed}");
    assert_eq!(launch.runner.starts, 1);
    captured(|| launch.execute(Command::Down, false)).0.unwrap();
}

#[test]
fn status_dry_run_and_json_read_only_the_saved_records() {
    let f = Fixture::new();
    let mut launch = f.launcher();
    let (result, printed) = captured(|| launch.execute(Command::Status, true));
    result.unwrap();
    assert_eq!(printed, format!("○ Vade is stopped.\n  Start: {}\n", text::command("launch")));
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
        "○ Vade is recorded as running at {} (dry run; readiness not checked).\n  Local config: {}\n",
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
    assert_eq!(run(&mut launch, Command::Status), (Ok(()), format!("○ Vade is stopped.\n  Start: {}\n", text::command("launch"))));
    assert_eq!(run(&mut launch, Command::Logs), (Ok(()), "No engine log yet.\n".into()));
    assert_eq!(run(&mut launch, Command::Update).0.unwrap_err(), "No local setup yet; run vade launch first.");
    let (result, printed) = run(&mut launch, Command::Detect);
    result.unwrap();
    assert!(printed.contains("Machine: Fixture OEM / GB10 test box\n") && printed.contains("Disk free: 80.0 GiB install, 60.0 GiB models\n"), "{printed}");
    assert!(!f.root.join("config").exists(), "detect must not save defaults");

    f.installed();
    run(&mut launch, Command::Launch(Default::default())).0.unwrap();
    let config = Config::load(&f.root).unwrap().unwrap();
    let artifacts = install::load(&f.root).unwrap().unwrap();
    let short = |key: &str| artifacts[key].chars().take(12).collect::<String>();
    assert_eq!(run(&mut launch, Command::Status), (Ok(()), format!(
        "● Vade is ready at {}.\n  Model     checkpoint\n  Context   8192 tokens, output 8192\n  Drafting  engine default\n  Build     {} (generation {})\n  Config    {}\n\n  Chat: {}   Stop: {}\n",
        config.endpoint(), short("engine_sha256"), short("generation"), f.root.join("config").display(), text::command("chat"), text::command("down"))));
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

#[test]
fn status_config_row_escapes_terminal_controls_in_the_home_path() {
    let f = Fixture::new();
    f.configured();
    let config = Config::load(&f.root).unwrap().unwrap();
    let mut launch = f.launcher();
    launch.root = f.base.join("vade\x1b]0;title\x07\rhome");
    let table = launch.status_table(&config, &Fields::new()).unwrap();
    assert!(!table.contains(['\x1b', '\x07', '\r']), "{table:?}");
    assert!(table.contains("vade\\u{1b}]0;title\\u{7}\\rhome/config\n"), "{table:?}");
}

#[test]
fn detect_json_makes_the_same_checks_and_exit_as_detect() {
    use crate::checks::{set_json, Answer};
    let f = Fixture::new();
    let mut launch = f.launcher();
    let json_run = |launch: &mut Launcher<MockRunner, MockDetector>, schema, command: Command, dry| {
        set_json(Some(Answer { schema, mode: "local", dry }));
        let result = captured(|| launch.execute(command, dry));
        set_json(None);
        result
    };
    let (result, printed) = json_run(&mut launch, "vade.detect.v1", Command::Detect, false);
    result.unwrap();
    assert_eq!(printed.lines().count(), 1, "{printed}");
    let value = json::parse(&printed).unwrap();
    assert_eq!((value["schema"].as_str(), &value["dry_run"]), (Some("vade.detect.v1"), &Value::Bool(false)));
    assert_eq!(value["model"]["path"].as_str(), Some(f.model.to_string_lossy().as_ref()));
    assert_eq!(value["model"]["fp4_local"], Value::Bool(true));

    // Too little install room fails both forms with the same error, and
    // --json prints nothing of its own (main answers with the error object).
    launch.detector.identity.install_free = GIB;
    let plain = captured(|| launch.execute(Command::Detect, false)).0.unwrap_err();
    let (result, printed) = json_run(&mut launch, "vade.detect.v1", Command::Detect, false);
    assert_eq!(result.unwrap_err(), plain);
    assert!(printed.is_empty(), "{printed}");
    launch.detector.identity = identity();

    // An incomplete checkpoint is an error, never a successful "no model".
    fs::rename(f.model.join("config.json"), f.model.join("config.json.moved")).unwrap();
    let (result, printed) = json_run(&mut launch, "vade.detect.v1", Command::Detect, false);
    assert!(result.is_err() && printed.is_empty(), "{printed}");
    fs::rename(f.model.join("config.json.moved"), f.model.join("config.json")).unwrap();

    // Dry runs answer in the schema too, and check nothing.
    let calls = launch.detector.calls;
    let (result, printed) = json_run(&mut launch, "vade.detect.v1", Command::Detect, true);
    result.unwrap();
    assert_eq!(printed, "{\"schema\":\"vade.detect.v1\",\"dry_run\":true,\"plan\":\"Detect this Spark and existing model\"}\n");
    let (result, printed) = json_run(&mut launch, "vade.doctor.v1", Command::Doctor, true);
    result.unwrap();
    let value = json::parse(&printed).unwrap();
    assert_eq!((value["mode"].as_str(), value["result"].as_str(), value["planned"].as_i64(), value["failed"].as_i64()), (Some("local"), Some("plan"), Some(10), Some(0)), "{printed}");
    assert_eq!(launch.detector.calls, calls, "dry runs must not detect");
}
