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
    assert_eq!(result.unwrap_err(), "1 doctor check failed. Run the next: step under each FAIL row, then run vade doctor again.");
    for line in [
        format!("{}No saved config; first-run defaults apply.", row("ok", "config")),
        row("ok", "vade binary"),
        row("ok", "VADE_HOME writable"),
        format!("{}no engine recorded", row("ok", "ownership")),
        format!("{}Vade is not installed yet.\n      next: Run vade launch to install and start Vade.", row("FAIL", "installed binaries")),
        row("ok", "model files"),
        row("ok", "model folder"),
        row("ok", "config.json"),
        row("ok", "tokenizer.json"),
        row("ok", "safetensors"),
        format!("{}8192 tokens", row("ok", "default context")),
        format!("{}8192 tokens", row("ok", "default output")),
        row("ok", "default drafting"),
        row("ok", "default prefix cache"),
        row("ok", "nvidia-smi"),
        row("ok", "disk"),
        row("ok", "free memory"),
        "ok    port ".into(),
        format!("{}stopped; start: ", row("skip", "engine health")),
        "1 failed, 1 skipped in ".into(),
    ] {
        assert!(printed.contains(&line), "{line:?} missing from:\n{printed}");
    }
    assert!(!f.root.join("config").exists(), "doctor must not save first-run defaults");

    // A fresh install needs install room as well as the model volume's.
    launch.detector.identity.install_free = GIB;
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    assert!(result.unwrap_err().starts_with("2 doctor checks failed."));
    assert!(printed.contains(&format!("{}Insufficient disk working room: install needs 8.0 GiB and model volume needs 1 GiB.\n      next: Free space", row("FAIL", "disk"))), "{printed}");

    launch.detector.identity = identity();
    f.installed();
    f.configured();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    result.unwrap();
    assert!(printed.contains(&format!("{}stopped; start: {}\nPASS  ", row("skip", "engine health"), text::command("launch")))
        && printed.contains("0 failed, 1 skipped in "), "{printed}");
    assert_eq!(launch.runner.starts, 0, "doctor must not start the engine");

    captured(|| launch.execute(Command::Up(Default::default()), false)).0.unwrap();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    result.unwrap();
    assert!(printed.contains("the running engine holds its share"), "{printed}");
    assert!(printed.contains("skip  port ") && printed.contains(&row("ok", "engine health")), "{printed}");
    assert!(printed.contains("\nPASS  ") && printed.contains("0 failed, 1 skipped in "), "{printed}");
    assert_eq!(launch.runner.starts, 1);
    captured(|| launch.execute(Command::Down, false)).0.unwrap();
}

#[test]
fn doctor_keeps_file_checks_after_model_discovery_fails() {
    let f = Fixture::new();
    f.installed();
    f.configured();
    let saved = fs::read(f.root.join("config")).unwrap();
    fs::write(f.model.join("tokenizer.json"), "{").unwrap();
    let mut launch = f.launcher();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    assert!(result.is_err());
    assert!(printed.contains("FAIL  model files") && printed.contains("FAIL  tokenizer.json"), "{printed}");
    assert!(printed.contains("1 file, 32 bytes") && printed.contains("Copy a complete tokenizer.json"), "{printed}");
    assert!(printed.contains("ok    default context") && printed.contains("8192 tokens"), "{printed}");
    assert_eq!(fs::read(f.root.join("config")).unwrap(), saved, "doctor must preserve saved defaults");
    assert_eq!(launch.runner.starts, 0);
}

#[test]
fn doctor_reports_an_invalid_config_and_continues() {
    let f = Fixture::new();
    f.installed();
    fs::write(f.root.join("config"), "format=unknown\n").unwrap();
    let mut launch = f.launcher();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    assert!(result.is_err());
    assert!(printed.contains("FAIL  config") && printed.contains("unknown format"), "{printed}");
    assert!(printed.contains("ok    safetensors") && printed.contains("ok    default context"), "{printed}");
    assert_eq!(fs::read_to_string(f.root.join("config")).unwrap(), "format=unknown\n");
    assert_eq!(launch.detector.calls, 1);
    assert_eq!(launch.runner.starts, 0);
}

#[test]
fn doctor_fails_when_an_owned_engine_has_no_saved_config() {
    let f = Fixture::new();
    f.installed();
    f.configured();
    let mut launch = f.launcher();
    captured(|| launch.execute(Command::Up(Default::default()), false)).0.unwrap();
    fs::rename(f.root.join("config"), f.root.join("config.saved")).unwrap();
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    assert!(result.is_err());
    assert!(printed.contains("FAIL  saved config") && printed.contains("An owned engine has no valid saved config."), "{printed}");
    assert!(printed.contains("Preserve the state folder, then run vade down"), "{printed}");
    assert_eq!(launch.runner.starts, 1);
    fs::rename(f.root.join("config.saved"), f.root.join("config")).unwrap();
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
    // An empty event list keeps the whole status text fixed (an event line shows its age).
    launch.runner.events = Some(r#"{"counts":{},"recent":[]}"#);
    let short = |key: &str| artifacts[key].chars().take(12).collect::<String>();
    assert_eq!(run(&mut launch, Command::Status), (Ok(()), format!(
        "● Vade is ready at {}.\n  Model     checkpoint\n  Context   8192 tokens, output 8192\n  Drafting  engine default\n  Build     {} (generation {})\n  Config    {}\n  Event counts: none\n  Last events: none\n\n  Chat: {}   Stop: {}\n",
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
    let table = launch.status_table(&config, &Fields::new(), &[]).unwrap();
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

#[test]
fn local_status_reads_and_prints_the_engine_events() {
    let f = Fixture::new();
    f.installed();
    let mut launch = f.launcher();
    captured(|| launch.execute(Command::Launch(Default::default()), false)).0.unwrap();
    let (result, printed) = captured(|| launch.execute(Command::Status, false));
    result.unwrap();
    let read = launch.runner.calls.iter().rfind(|exec| exec.label == "local events").expect("status reads the events");
    assert_eq!(read.args.last().unwrap(), &format!("{}/v1/events", Config::load(&f.root).unwrap().unwrap().endpoint()));
    assert!(printed.contains("\n  Event counts: prefix_cache_miss 2, scheduler_pool 1\n  Last 1 events (oldest first):\n    00:00:00Z ("), "{printed}");
    assert!(printed.contains(") scheduler_pool active=8 connections=16\n"), "{printed}");
    // A failed read keeps the counts from /health and names the failure; it is not an older engine.
    launch.runner.events = None;
    let (result, printed) = captured(|| launch.execute(Command::Status, false));
    result.unwrap();
    assert!(printed.contains("\n  Event counts: scheduler_pool 1 (from /health)\n  Last events: not read (the request failed; run vade status again)\n"), "{printed}");
    captured(|| launch.execute(Command::Down, false)).0.unwrap();
}

#[test]
fn a_load_in_progress_is_reported_without_asking_to_stop_it() {
    let f = Fixture::new();
    f.installed();
    f.configured();
    let mut launch = f.launcher();
    captured(|| launch.execute(Command::Up(Default::default()), false)).0.unwrap();
    // A ready engine opens the agent; on Linux one line first says what it can do.
    let (result, printed) = captured(|| launch.execute(Command::Chat(Default::default()), false));
    result.unwrap();
    let linux = "On Linux, the agent reads, searches and edits files in this folder. It cannot run shell commands or checks; run builds and tests in your own terminal.\n";
    assert_eq!(printed.starts_with(linux), cfg!(target_os = "linux"), "{printed}");
    // Another terminal's vade launch has written its record and still loads
    // the model: it holds the lock for the whole load.
    let path = f.root.join("state/run.conf");
    let mut record = read_fields(&path).unwrap().unwrap();
    record.insert("phase".into(), "starting".into());
    fs::write(&path, text::encode(&record)).unwrap();
    let launcher_lock = crate::state::Lock::acquire(&f.root).unwrap();
    // Status answers (exit 0) and names the log; chat refuses (exit 1) and says to wait.
    assert_eq!(captured(|| launch.execute(Command::Status, false)), (Ok(()), format!(
        "◐ Vade is starting: the engine is still loading the model. The vade launch or vade up that started it says when it is ready.\n  Load progress: {}\n", text::command("logs"))));
    let loading = "The engine is still loading the model, so chat cannot open yet. Wait for vade launch to say it is ready, then run vade chat again. Run vade logs to follow the load.";
    assert_eq!(captured(|| launch.execute(Command::Chat(Default::default()), false)), (Err(loading.into()), String::new()));
    assert_eq!(launch.runner.chats, 1);
    // Doctor skips the health check while the load runs, with no stop advice.
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    result.unwrap();
    assert!(printed.contains("skip  engine health         still loading the model; vade launch says when it is ready\n"), "{printed}");
    assert!(!printed.contains("vade down"), "{printed}");
    // The launch died (an SSH hang-up kills it; the detached engine keeps
    // loading): the kernel drops its lock, and nothing will mark the record
    // running. Status, chat and a new launch say to clear it, not to wait.
    drop(launcher_lock);
    let unfinished = "The last engine start did not finish, and the engine is not ready. Run vade down to clear it, then vade launch.";
    for engine_gone in [false, true] {
        launch.runner.refused = engine_gone;
        assert_eq!(captured(|| launch.execute(Command::Chat(Default::default()), false)).0.unwrap_err(), unfinished);
        assert_eq!(captured(|| launch.execute(Command::Status, false)).0.unwrap_err(), unfinished);
    }
    launch.runner.refused = false;
    assert!(captured(|| launch.execute(Command::Up(Default::default()), false)).0.unwrap_err().contains(unfinished));
    assert_eq!(read_fields(&path).unwrap().unwrap(), record, "status, chat and launch keep the record");
    assert_eq!(launch.runner.chats, 1);
    captured(|| launch.execute(Command::Down, false)).0.unwrap();
    // The other chat refusals of the "what you see when" table in cli/docs/ux.md.
    assert_eq!(captured(|| launch.execute(Command::Chat(Default::default()), false)).0.unwrap_err(),
        "The engine is not running. Run vade launch to start it and open chat.");
    captured(|| launch.execute(Command::Up(Default::default()), false)).0.unwrap();
    launch.runner.health_ready = false;
    assert_eq!(captured(|| launch.execute(Command::Chat(Default::default()), false)).0.unwrap_err(),
        "The engine is not answering. Run vade logs to see why, then vade down and vade launch.");
    launch.runner.health_ready = true;
    captured(|| launch.execute(Command::Down, false)).0.unwrap();
}

#[test]
fn a_dead_engine_record_is_reported_by_status_and_doctor_and_cleared_by_up() {
    let f = Fixture::new();
    f.installed();
    f.configured();
    let mut launch = f.launcher();
    captured(|| launch.execute(Command::Up(Default::default()), false)).0.unwrap();
    // The engine crashed or was killed: its process is gone and nothing listens.
    launch.runner.refused = true;
    launch.runner.listener_owned = false;
    let record = fs::read(f.root.join("state/run.conf")).unwrap();
    assert_eq!(captured(|| launch.execute(Command::Status, false)), (Ok(()), format!(
        "○ Vade is stopped: the engine is not running (stale record found; the next start clears it).\n  Start: {}\n", text::command("launch"))));
    let (result, printed) = captured(|| launch.execute(Command::Doctor, false));
    result.unwrap();
    assert!(printed.contains(&format!("skip  engine process        not running; stale record found ({} clears it)\n", text::command("up"))), "{printed}");
    assert!(printed.contains("ok    port ") && printed.contains("free on 127.0.0.1"), "doctor checks the listener: {printed}");
    assert!(printed.contains(&format!("skip  engine health         stopped; start: {}\n", text::command("launch"))), "{printed}");
    assert!(!printed.contains("in use by the running engine") && !printed.contains("holds its share"), "{printed}");
    assert_eq!(captured(|| launch.execute(Command::Chat(Default::default()), false)).0.unwrap_err(), format!(
        "The engine is not running; it stopped or was killed. Run {} to clear its record and start it again.", text::command("up")));
    assert_eq!(fs::read(f.root.join("state/run.conf")).unwrap(), record, "status, doctor and chat keep the record");
    assert_eq!(launch.runner.starts, 1);

    // vade up clears the record in one line, then starts as usual.
    launch.runner.listener_owned = true;
    launch.runner.calls.clear();
    let (result, printed) = captured(|| launch.execute(Command::Up(Default::default()), false));
    result.unwrap();
    assert!(printed.starts_with(&format!("Cleared the record of an engine that is no longer running (it stopped or was killed). Its log: {}\n", text::command("logs"))), "{printed}");
    assert!(printed.contains("Up ready in "), "{printed}");
    assert_eq!(launch.runner.worker_ops(), ["inspect", "stop", "clean", "probe", "start", "inspect", "listener"]);
    assert_eq!(launch.runner.starts, 2);
    assert_ne!(fs::read(f.root.join("state/run.conf")).unwrap(), record, "a new record replaces the stale one");

    // A live engine whose listener is lost keeps the old error and record.
    launch.runner.listener_owned = false;
    let record = fs::read(f.root.join("state/run.conf")).unwrap();
    assert_eq!(captured(|| launch.execute(Command::Status, false)).0.unwrap_err(), "The recorded server does not own the loopback listener.");
    assert!(captured(|| launch.execute(Command::Up(Default::default()), false)).0.is_err());
    assert_eq!(fs::read(f.root.join("state/run.conf")).unwrap(), record);
    assert_eq!(launch.runner.starts, 2);
    launch.runner.listener_owned = true;
    captured(|| launch.execute(Command::Down, false)).0.unwrap();
}

#[test]
fn install_from_another_checkout_rebuilds_and_names_the_new_generation() {
    let f = Fixture::new();
    f.installed();
    f.configured();
    let mut launch = f.launcher();
    let path = f.root.join("stage/install.conf");
    let built = |path: &Path| read_fields(path).unwrap().unwrap();
    // The mock checkout is revision 40 'a'; the record says another revision, then none.
    for (recorded, from) in [(Some("b".repeat(40)), "bbbbbbbbbbbb"), (None, "an earlier install with no recorded revision")] {
        let mut record = built(&path);
        match &recorded { Some(id) => { record.insert("source_revision".into(), id.clone()); } None => { record.remove("source_revision"); } }
        text::atomic_write(&path, &text::encode(&record)).unwrap();
        let old = record["generation"].clone();
        launch.runner.calls.clear();
        let (result, printed) = captured(|| launch.execute(Command::Install, false));
        result.unwrap();
        let now = built(&path);
        assert_ne!(now["generation"], old, "the old generation must not stay installed");
        assert_eq!(now["source_revision"], "a".repeat(40));
        assert!(printed.contains(&format!(" Install this checkout is aaaaaaaaaaaa, but the installed engine was built from {from}; building this checkout\n")), "{printed}");
        assert!(printed.contains(&format!(" Install generation {} is now installed, built from aaaaaaaaaaaa\n", &now["generation"][..12])), "{printed}");
        assert!(!printed.contains("reuse installed files"), "{printed}");
        assert!(launch.runner.calls.iter().any(|exec| exec.label == "Build GPU kernels"));
    }
    // The same revision reuses the files and builds nothing.
    let record = fs::read(&path).unwrap();
    launch.runner.calls.clear();
    let (result, printed) = captured(|| launch.execute(Command::Install, false));
    result.unwrap();
    assert!(printed.contains("reuse installed files") && !printed.contains("building this checkout"), "{printed}");
    assert!(!launch.runner.calls.iter().any(|exec| exec.label.starts_with("Build ")));
    assert_eq!(fs::read(&path).unwrap(), record);
}
