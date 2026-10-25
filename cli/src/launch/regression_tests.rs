//! Review regressions use disposable children for exit status and hang bounds.
use super::*;
use std::process::{Child, Command as Process, Stdio};

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn child(mode: &str, f: &Fixture) -> ChildGuard {
    let mut command = Process::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "launch::tests::regressions::regression_child", "--nocapture"])
        .env("VADE_TEST_OWNER_MODE", mode)
        .env("VADE_TEST_OWNER_BASE", &f.base)
        .env_remove("VADE_MODELS")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if mode.starts_with("replacement") {
        command.env("VADE_MODELS", f.base.join("replacement"));
    }
    ChildGuard(command.spawn().unwrap())
}

fn bounded_status(child: &mut Child) -> std::process::ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(status) = child.try_wait().unwrap() { return status; }
        assert!(Instant::now() < deadline, "owner regression exceeded hang deadline");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn regression_child() {
    let Ok(mode) = std::env::var("VADE_TEST_OWNER_MODE") else { return; };
    let base = PathBuf::from(std::env::var_os("VADE_TEST_OWNER_BASE").unwrap());
    let home = base.join("home");
    let root = home.join(".vade");
    let mut launch = Launcher {
        runner: MockRunner::new(&root),
        detector: MockDetector { identity: identity(), calls: 0 },
        root,
        home,
    };
    match mode.as_str() {
        "incomplete-generation" => {
            let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
            assert!(error.contains("generation_config.json"), "{error}");
            assert!(launch.runner.calls.is_empty(), "no install or start may run");
        }
        "config-fifo" => assert!(Config::load(&launch.root).is_err()),
        "record-fifo" => assert!(launch.record().is_err()),
        "manifest-fifo" => assert!(launch.stage_manifest(&launch.home.join("models/checkpoint")).is_err()),
        "log-fifo" => assert!(launch.execute(Command::Logs, false).unwrap_err().contains("regular readable file")),
        "replacement" | "replacement-refused" => {
            if mode == "replacement-refused" {
                launch.detector.identity.memory_available = 1;
            }
            if let Err(error) = launch.execute(Command::Launch(Default::default()), false) {
                crate::fail(&error);
            }
            let start = launch.runner.calls.iter().find(|exec| exec.label == "local start").unwrap();
            let start = text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap();
            assert_eq!(start["model_path"], base.join("replacement").to_string_lossy());
            assert_eq!(start["arg.5"], start["model_path"]);
        }
        "resource-output" | "start-output" | "chat-output" | "stop-output" | "clean-output" => {
            let needle = match mode.as_str() {
                "resource-output" => {
                    launch.detector.identity.memory_available = 1;
                    "  Detect stopped after"
                }
                "start-output" => {
                    launch.runner.fail_start = true;
                    "      Up stopped after"
                }
                "chat-output" => {
                    launch.runner.chat_success = false;
                    "    Chat done"
                }
                _ => {
                    launch.runner.start_output = true;
                    launch.runner.fail_cleanup = Some(mode.split('-').next().unwrap().into());
                    "Fixture engine loading progress"
                }
            };
            let writer = output::testing::Capture {
                fail_on: Some((needle, std::io::ErrorKind::BrokenPipe)),
                ..Default::default()
            };
            output::with_writer(writer, || {
                let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
                // Exercise the actual public command's exit decision.
                crate::fail(&error);
            });
        }
        _ => panic!("unknown owner regression mode"),
    }
}

#[test]
fn incomplete_serving_metadata_refuses_before_saving_or_installing() {
    let f = Fixture::new();
    fs::remove_file(f.model.join("generation_config.json")).unwrap();
    let mut process = child("incomplete-generation", &f);
    assert!(bounded_status(&mut process.0).success());
    assert!(!f.root.join("config").exists(), "nothing may be saved");
    assert!(!f.root.join("state/run.conf").exists());
}

#[test]
fn rejected_saved_model_can_be_replaced_only_after_validation_and_admission() {
    for refused in [true, false] {
        let f = Fixture::new();
        f.installed();
        f.configured();
        let before = fs::read(f.root.join("config")).unwrap();
        let port = Config::load(&f.root).unwrap().unwrap().port;
        for name in ["model-manifest.sha256", "model-manifest.sha256.sizes"] {
            text::atomic_write(&f.root.join("state").join(name), "stale rejected checkpoint receipt").unwrap();
        }
        fs::write(f.model.join("config.json"), r#"{"model_type":"qwen3","max_position_embeddings":8192}"#).unwrap();
        let replacement = f.base.join("replacement");
        fs::create_dir(&replacement).unwrap();
        fs::write(replacement.join("config.json"), model::test_mixed_config(16384)).unwrap();
        model::test_write_serving_metadata(&replacement);
        fs::write(replacement.join("model.safetensors.index.json"), model::test_index()).unwrap();
        fs::write(replacement.join("model.safetensors"), [0; 64]).unwrap();
        let mut process = child(if refused { "replacement-refused" } else { "replacement" }, &f);
        assert_eq!(bounded_status(&mut process.0).success(), !refused);
        if refused {
            assert_eq!(fs::read(f.root.join("config")).unwrap(), before);
            assert!(!f.root.join("state/run.conf").exists());
            for name in ["model-manifest.sha256", "model-manifest.sha256.sizes"] {
                assert_eq!(fs::read_to_string(f.root.join("state").join(name)).unwrap(), "stale rejected checkpoint receipt");
            }
        } else {
            let config = Config::load(&f.root).unwrap().unwrap();
            assert_eq!(config.model, replacement);
            assert_eq!((config.context, config.model_bytes, config.port), (16384, 64, port));
            assert_eq!(read_fields(&f.root.join("state/run.conf")).unwrap().unwrap()["model_path"], replacement.to_string_lossy());
            for name in ["model-manifest.sha256", "model-manifest.sha256.sizes"] {
                assert!(!f.root.join("state").join(name).exists(), "replacement must not inherit {name}");
            }
        }
    }
}

#[test]
fn changed_saved_launch_limits_refuse_resident_reuse_until_down() {
    use crate::evals::json;
    for (key, flag, value, argument) in [
        ("context_tokens", "--context", "16384", "arg.6"),
        ("output_tokens", "--output", "4096", "env.FLASH_SERVE_OUTPUT_TOKENS"),
        ("mtp", "--mtp", "0", "env.MTP"),
        ("memory_bytes", "--memory-bytes", "85899345920", "env.SEQPAR_MEMORY_BYTES"),
    ] {
        let f = Fixture::new();
        f.installed();
        let mut launch = f.launcher();
        let command = crate::cli::parse(&["launch", "--mtp", "3", "--memory-bytes", "96636764160"].map(str::to_owned)).unwrap().command;
        launch.execute(command, false).unwrap();
        let record = fs::read(f.root.join("state/run.conf")).unwrap();
        let live = launch.record().unwrap().unwrap();
        let mut config = read_fields(&f.root.join("config")).unwrap().unwrap();
        config.insert(key.into(), value.into());
        text::atomic_write(&f.root.join("config"), &text::encode(&config)).unwrap();
        launch.runner.calls.clear();
        for command in [Command::Launch(Default::default()), Command::Up(Default::default()),
            crate::cli::parse(&["up", flag, value].map(str::to_owned)).unwrap().command] {
            let error = launch.execute(command, false).unwrap_err();
            assert!(error.contains("launch limits") && error.contains("vade down"), "{error}");
            assert_eq!(launch.runner.starts, 1);
            assert_eq!(launch.runner.chats, 1);
            assert!(launch.runner.calls.is_empty());
            assert_eq!(fs::read(f.root.join("state/run.conf")).unwrap(), record);
        }
        let mut report = crate::status::Snapshot::new("local", false);
        launch.status_snapshot(&mut report, false).unwrap();
        let report = json::parse(&report.finish(Duration::ZERO).0).unwrap();
        for (field, expected) in [("context_limit", 8192), ("output_limit", 8192), ("mtp", 3), ("memory_bytes_per_rank", 96636764160)] {
            assert_eq!(report["settings"][field], json::Value::Int(expected));
        }
        assert_eq!(report["settings"]["mtp_source"].as_str(), Some("explicit"));
        // A flag matching the live engine may attach despite an edited default;
        // neither that attach nor status may rewrite the requested saved config.
        let command = crate::cli::parse(&["up", flag, &live[key]].map(str::to_owned)).unwrap().command;
        launch.execute(command, false).unwrap();
        assert_eq!(read_fields(&f.root.join("config")).unwrap().unwrap(), config);
        assert_eq!(fs::read(f.root.join("state/run.conf")).unwrap(), record);
        launch.execute(Command::Down, false).unwrap();
        // Permit the newly requested context in the checkpoint metadata.
        fs::write(f.model.join("config.json"), model::test_mixed_config(32768)).unwrap();
        launch.execute(Command::Launch(Default::default()), false).unwrap();
        assert_eq!(launch.record().unwrap().unwrap()[key], value);
        let request = launch.runner.calls.iter().rev().find(|exec| exec.label == "local start").unwrap();
        let request = text::parse(std::str::from_utf8(&request.stdin).unwrap()).unwrap();
        assert_eq!(request[argument], value);
    }
}

#[test]
fn older_resident_without_effective_limits_requires_restart() {
    for key in ["context_tokens", "output_tokens", "mtp", "memory_bytes"] {
        let f = Fixture::new();
        f.installed();
        let mut launch = f.launcher();
        launch.execute(Command::Launch(Default::default()), false).unwrap();
        let mut record = launch.record().unwrap().unwrap();
        assert_eq!(record["mtp"], "engine_default");
        assert_eq!(record["memory_bytes"], "engine_default");
        record.remove(key);
        text::atomic_write(&f.root.join("state/run.conf"), &text::encode(&record)).unwrap();
        assert!(launch.execute(Command::Launch(Default::default()), false).unwrap_err().contains("vade down"));
        let mut report = crate::status::Snapshot::new("local", false);
        launch.status_snapshot(&mut report, false).unwrap();
        let report = crate::evals::json::parse(&report.finish(Duration::ZERO).0).unwrap();
        assert_eq!(report["settings"]["mtp_source"].as_str(), Some(if key == "mtp" { "unknown" } else { "engine_default" }));
        assert_eq!(report["settings"]["memory_bytes_per_rank"], crate::evals::json::Value::Null);
        launch.execute(Command::Down, false).unwrap();
        assert!(launch.record().unwrap().is_none());
    }
}

#[test]
fn operation_or_cleanup_failure_has_nonzero_exit_even_when_stdout_breaks() {
    for mode in ["resource-output", "start-output", "chat-output", "stop-output", "clean-output"] {
        let f = Fixture::new();
        f.installed();
        let mut process = child(mode, &f);
        assert_eq!(bounded_status(&mut process.0).code(), Some(1), "{mode}");
        if matches!(mode, "stop-output" | "clean-output") {
            assert!(f.root.join("state/run.conf").is_file(), "failed cleanup must retain ownership");
        }
    }
}

#[test]
fn fifo_owner_metadata_fails_without_waiting_for_a_writer() {
    let f = Fixture::new();
    fs::create_dir_all(f.root.join("state")).unwrap();
    for (mode, path) in [
        ("config-fifo", f.root.join("config")),
        ("record-fifo", f.root.join("state/run.conf")),
        ("manifest-fifo", f.model.join("manifest.sha256")),
        ("log-fifo", f.root.join("state/last-server.log")),
    ] {
        assert!(Process::new("mkfifo").arg(&path).status().unwrap().success());
        let mut process = child(mode, &f);
        assert!(bounded_status(&mut process.0).success(), "{mode}");
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn build_recovery_runs_before_install_or_stopped_down_and_failure_is_latched() {
    struct Recovery {
        calls: usize,
    }
    impl Runner for Recovery {
        fn recover_jobs(&mut self, root: &Path) -> Result<(), String> {
            assert!(crate::state::Lock::acquire(root).is_err(), "recovery must hold owner lock");
            self.calls += 1;
            Err("Fixture build recovery failed.".into())
        }
        fn run(&mut self, _: &Exec) -> Result<Output, String> { panic!("no command before recovery") }
        fn interactive(&mut self, _: &Exec) -> Result<bool, String> { unreachable!() }
        fn port_free(&mut self, _: u16) -> Result<(), String> { unreachable!() }
        fn pause(&mut self, _: Duration) { unreachable!() }
    }
    let f = Fixture::new();
    let mut launch = Launcher {
        root: f.root.clone(), home: f.home.clone(), runner: Recovery { calls: 0 },
        detector: MockDetector { identity: identity(), calls: 0 },
    };
    for command in [Command::Install, Command::Down] {
        output::with_writer(output::testing::Capture::default(), || {
            let error = launch.execute(command, false).unwrap_err();
            assert_eq!(error, "Fixture build recovery failed.");
            assert!(!output::broken_pipe_success());
        });
    }
    assert_eq!(launch.runner.calls, 2);
    assert_eq!(launch.detector.calls, 0);
    assert!(!f.root.join("config").exists());
}
