    use super::*;
    struct Fake { answer_text: Option<String>, doctor_text: Option<String>, calls: Vec<Exec>, alive: bool, absent: bool, bad_os: bool, fail_download: bool, doctor_bad: bool, reused_pid: bool, head: String, cancelled: bool, interrupt_health: bool, pauses: usize }
    impl Runner for Fake {
        fn interrupted(&self) -> bool { self.cancelled }
        fn run(&mut self, exec: &Exec) -> Result<Output, String> {
            self.calls.push(exec.clone());
            let mut success = true;
            let stdout = match exec.label.as_str() {
                "Spark prerequisites" => format!("os={}\narch=aarch64\nmissing=\ngpu=NVIDIA GB10, 580.00, N/A\nbusy=no\nram_bytes=100000000000\ndisk_bytes=100000000000\n", if self.bad_os { "Darwin" } else { "Linux" }),
                "source build prerequisites" => "missing=\nstable=ok\nnightly=ok\nrust-src\nllvm-tools\nllvm-bitcode-linker\n".into(),
                "NVPTX prerequisite" => "nvptx64-nvidia-cuda\n".into(),
                "payload SHA256" => {
                    let path = Path::new(&exec.args[0]); let filename = path.file_name().unwrap().to_str().unwrap().trim_end_matches(".part");
                    let bytes = fs::read(path).unwrap();
                    let digest = if bytes != filename.as_bytes() { "0".repeat(64) }
                        else if let Some((_, digest)) = MODEL_FILES.iter().find(|(name, _)| *name == filename) { digest.to_string() }
                        else { match filename { "spark-serve" => "1", "vade-harness" => "2", _ => "3" }.repeat(64) };
                    format!("{digest}  {}\n", path.display())
                },
                "model size" => "HTTP/2 200\ncontent-length: 100\n".into(),
                "model download" => {
                    assert!(exec.args.iter().any(|a| a == "--fail")); assert!(!exec.args.iter().any(|a| a == "--fail-with-body"));
                    let path = Path::new(&exec.args[exec.args.iter().position(|a| a == "--output").unwrap()+1]);
                    let filename = path.file_name().unwrap().to_str().unwrap().trim_end_matches(".part");
                    if self.fail_download { self.fail_download = false; fs::write(path, &filename.as_bytes()[..2]).unwrap(); success = false; }
                    else { fs::write(path, filename).unwrap(); }
                    String::new()
                },
                "local inspect" => {
                    if self.reused_pid { success = false; "error=Saved PID belongs to a different process.\n".into() }
                    else { format!("phase={}\nalive={}\n", if self.absent { "absent" } else { "running" }, self.alive) }
                },
                "safe process prerequisites" => "pidfd=ok\ncuda_driver=ok\n".into(),
                "local listener" => "listener=owned\n".into(),
                "local start" => { self.alive = true; self.absent = false; "phase=running\nalive=true\n".into() },
                "local stop" => { self.alive = false; "phase=stopped\nalive=false\n".into() },
                "local clean" => { self.absent = true; "phase=absent\nlog=owned mock log\n".into() },
                "local doctor" => { success = !self.doctor_bad; self.doctor_text.clone().unwrap_or_else(|| "doctor mock\n".into()) },
                "owned health" => { if self.interrupt_health { self.cancelled = true; return Err("Interrupted during owned health.".into()); } "{\"status\":\"ready\",\"model\":\"qwen3-8b\"}".into() },
                "first streaming answer" if self.answer_text.is_some() => format!("data: {{\"model\":\"qwen3-8b\",\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n", crate::evals::json::packed(&crate::evals::json::Value::Str(self.answer_text.clone().unwrap()))),
                "first streaming answer" => "data: {\"model\":\"qwen3-8b\",\"choices\":[{\"delta\":{\"content\":\"Ice is less dense.\"},\"finish_reason\":null}]}\n\ndata: {\"model\":\"qwen3-8b\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n".into(),
                "engine revision" | "harness revision" => self.head.clone(),
                "engine source layout" | "harness source layout" => format!("{}\n", exec.args[1]),
                "extract engine" | "extract harness" => { let destination = Path::new(exec.args.last().unwrap()); fs::write(destination.join("Cargo.toml"), "mock").unwrap(); String::new() },
                "archive engine" | "archive harness" => { let path = exec.args.iter().find_map(|a| a.strip_prefix("--output=")).unwrap(); fs::write(path, "mock").unwrap(); String::new() },
                "engine build" | "harness build" => {
                    let job = exec.job.as_ref().expect("durable job before mock spawn");
                    let intent = read_fields(&job.join("intent.conf")).unwrap().unwrap();
                    assert_eq!(intent["program"], exec.program);
                    assert_eq!(intent["cwd"], exec.cwd.as_ref().unwrap().to_string_lossy());
                    text::atomic_write(&job.join("stopped.conf"), "success=true\n").unwrap();
                    let source = exec.cwd.as_ref().unwrap(); fs::create_dir_all(source.join("target/release")).unwrap();
                    if exec.label == "engine build" {
                        fs::write(source.join("target/release/spark-serve"), "spark-serve").unwrap();
                        fs::create_dir_all(source.join("target/nvptx64-nvidia-cuda/release")).unwrap();
                        fs::write(source.join("target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx"), "kernels.ptx").unwrap();
                    } else { fs::write(source.join("target/release/vade-harness"), "vade-harness").unwrap(); }
                    for filename in ["spark-serve", "vade-harness"] { let path = source.join("target/release").join(filename); if path.exists() { fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap(); } }
                    String::new()
                },
                _ => String::new(),
            };
            Ok(Output { success, stdout })
        }
        fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.calls.push(exec.clone()); Ok(true) }
        fn port_free(&mut self, _: u16) -> Result<(), String> { Ok(()) }
        fn pause(&mut self, _: Duration) { self.pauses += 1; }
    }
    struct Fixture { directory: PathBuf, app: Local<Fake>, options: RunOptions }
    impl Fixture {
        fn new() -> Self {
            let directory = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-run-test-{}", text::random_hex(8).unwrap()));
            text::private_dir(&directory).unwrap(); let release = directory.join("release"); text::private_dir(&release).unwrap();
            let mut fields = Fields::from([("version".into(), "1".into()), ("target".into(), "linux-aarch64-gb10".into()), ("engine_revision".into(), "a".repeat(40)), ("harness_revision".into(), "b".repeat(40))]);
            for (name, filename, digit) in [("engine", "spark-serve", "1"), ("harness", "vade-harness", "2"), ("ptx", "kernels.ptx", "3")] { fs::write(release.join(filename), filename).unwrap(); if name != "ptx" { fs::set_permissions(release.join(filename), fs::Permissions::from_mode(0o700)).unwrap(); } fields.insert(format!("{name}_sha256"), digit.repeat(64)); }
            text::atomic_write(&release.join("release.conf"), &text::encode(&fields)).unwrap();
            let app = Local { root: directory.join("data"), dry: false, runner: Fake { answer_text: None, doctor_text: None, calls: Vec::new(), alive: false, absent: true, bad_os: false, fail_download: false, doctor_bad: false, reused_pid: false, head: "a".repeat(40), cancelled: false, interrupt_health: false, pauses: 0 } };
            Self { directory, app, options: RunOptions { release: Some(release), ..RunOptions::default() } }
        }
        fn launch(&mut self) -> Result<(), String> { self.app.execute(Command::Run(self.options.clone())) }
    }
    impl Drop for Fixture { fn drop(&mut self) { fs::remove_dir_all(&self.directory).unwrap(); } }
    #[test]
    fn source_path_and_copy_errors_distinguish_missing_and_wrong_type() {
        let f = Fixture::new();
        let source = f.directory.join("source");fs::write(&source, b"unchanged source").unwrap();
        let missing = f.directory.join("missing");
        let absent = absolute(&missing).unwrap_err();
        assert!(absent.contains("path not found") && absent.contains("Check the path"));
        let wrong = absolute(&source.join("child")).unwrap_err();
        assert!(wrong.contains("wrong path type") && wrong.contains("Select the expected"));
        let directory = copy(&f.directory, &missing).unwrap_err();
        assert!(directory.contains("not a regular file") && directory.contains("Restore"));
        let destination = copy(&source, &f.directory).unwrap_err();
        assert!(destination.contains("Cannot stage") && destination.contains("wrong path type"));
        assert_eq!(fs::read(source).unwrap(), b"unchanged source");assert!(!missing.exists());
    }
    #[test]
    fn local_json_status_distinguishes_stopped_unknown_and_ready_without_mutation() {
        use crate::{evals::json::{self, Value}, status::Snapshot};
        let mut f = Fixture::new();
        let path = f.app.root.join("run.conf");
        let record = Fields::from([("version".into(), "1".into()), ("phase".into(), "running".into()), ("token".into(), "a".repeat(64)), ("generation".into(), "b".repeat(24)), ("port".into(), "8080".into())]);
        for case in ["stopped", "ready", "dead", "foreign", "offline"] {
            if case != "stopped" { text::atomic_write(&path, &text::encode(&record)).unwrap(); }
            f.app.dry = case == "offline";
            f.app.runner.alive = case != "dead";
            f.app.runner.reused_pid = case == "foreign";
            f.app.runner.calls.clear();
            let before = fs::read(&path).ok();
            let mut report = Snapshot::new("local", f.app.dry);
            f.app.status_snapshot(&mut report).unwrap();
            let (encoded, code) = report.finish(Duration::ZERO); let value = json::parse(&encoded).unwrap();
            assert_eq!(code, i32::from(matches!(case, "dead" | "foreign")));
            assert_eq!(value["ready"], if case == "offline" { Value::Null } else { Value::Bool(case == "ready") });
            if matches!(case, "offline" | "stopped") { assert!(f.app.runner.calls.is_empty()); }
            if matches!(case, "dead" | "foreign") { assert!(!f.app.runner.calls.iter().any(|call| call.label == "owned health")); }
            assert!(!f.app.runner.calls.iter().any(|call| matches!(call.label.as_str(), "local start" | "local stop" | "local clean")));
            assert!(!encoded.contains(&"a".repeat(64)));
            assert_eq!(fs::read(&path).ok(), before);
        }
    }
    #[test]
    fn legacy_doctor_json_is_one_doctor_object_in_every_state() {
        use crate::{checks::{set_json, Answer}, evals::json, output::{self, testing::Capture}};
        let mut f = Fixture::new();
        let doctor = |f: &mut Fixture, dry: bool| {
            f.app.dry = dry;
            set_json(Some(Answer { schema: "vade.doctor.v1", mode: "legacy", dry }));
            let writer = Capture::default();
            let result = output::with_writer(writer.clone(), || f.app.execute(Command::Doctor));
            set_json(None);
            f.app.dry = false;
            let printed = String::from_utf8(writer.bytes.borrow().clone()).unwrap();
            assert_eq!(printed.lines().count(), 1, "{printed}");
            (result, json::parse(&printed).unwrap())
        };
        // Plain doctor on the same state, dry or not: whether it passed.
        let plain = |f: &mut Fixture, dry: bool| {
            f.app.dry = dry;
            let result = output::with_writer(Capture::default(), || f.app.execute(Command::Doctor));
            f.app.dry = false;
            result.is_ok()
        };
        // A stopped setup fails ownership in both forms, dry runs included.
        for dry in [false, true] {
            let (result, value) = doctor(&mut f, dry);
            assert_eq!(result.unwrap_err(), crate::checks::REPORTED);
            assert_eq!((value["mode"].as_str(), value["failed"].as_i64(), value["checks"][0]["name"].as_str()), (Some("legacy"), Some(1), Some("ownership")));
            assert!(!plain(&mut f, dry), "dry={dry}");
        }
        // So does a malformed record.
        fs::create_dir_all(&f.app.root).unwrap();
        fs::set_permissions(&f.app.root, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(f.app.root.join("run.conf"), "version=1\nphase=running\n").unwrap();
        for dry in [false, true] {
            let (result, value) = doctor(&mut f, dry);
            assert!(result.is_err() && value["checks"][0]["state"].as_str() == Some("fail"), "dry={dry}: {value:?}");
            assert!(!plain(&mut f, dry), "dry={dry}");
        }
        fs::remove_file(f.app.root.join("run.conf")).unwrap();
        f.launch().unwrap();
        let (result, value) = doctor(&mut f, true);
        result.unwrap();
        assert_eq!((value["result"].as_str(), value["ok"].as_i64(), value["planned"].as_i64()), (Some("plan"), Some(1), Some(2)));
        assert!(plain(&mut f, true));
        let (result, value) = doctor(&mut f, false);
        result.unwrap();
        assert_eq!((value["result"].as_str(), value["ok"].as_i64()), (Some("pass"), Some(3)));
        f.app.runner.doctor_bad = true;
        let (result, value) = doctor(&mut f, false);
        assert!(result.is_err());
        assert_eq!((value["checks"][2]["name"].as_str(), value["checks"][2]["state"].as_str()), (Some("harness doctor"), Some("fail")));
        f.app.runner.doctor_bad = false;
        f.app.execute(Command::Down).unwrap();
    }
    #[test] fn first_answer_greedy_request_omits_seed() {
        let mut f = Fixture::new();
        f.launch().unwrap();
        let request = f.app.runner.calls.iter().find(|exec| exec.label == "first streaming answer").unwrap();
        let body = crate::evals::json::parse(std::str::from_utf8(&request.stdin).unwrap()).unwrap();
        assert_eq!(body["temperature"], crate::evals::json::Value::Int(0));
        assert!(body.get("seed").is_none());
    }
    #[test] fn local_release_lifecycle_rehashes_and_reuses_owned_server() {
        let mut f = Fixture::new(); f.launch().unwrap(); let record = f.app.record().unwrap().unwrap();
        let answer = f.app.runner.calls.iter().find(|e| e.label == "first streaming answer").unwrap();
        assert!(answer.args.iter().any(|a| a == "--fail-with-body")); assert!(!answer.args.iter().any(|a| a == "--fail"));
        let call = f.app.runner.calls.iter().find(|e| e.label == "local start").unwrap(); let request = text::parse(&String::from_utf8_lossy(&call.stdin)).unwrap();
        assert_eq!(request["arg.3"], "127.0.0.1:8080"); assert_eq!(request["arg.5"], "plain");
        f.app.runner.calls.clear(); f.launch().unwrap(); assert!(!f.app.runner.calls.iter().any(|e| e.label == "local start" || e.label == "model download"));
        assert!(f.app.runner.calls.iter().filter(|e| e.label == "payload SHA256").count() >= MODEL_FILES.len());
        for command in [Command::Status, Command::Doctor, Command::Chat, Command::Down, Command::Down] { f.app.execute(command).unwrap(); }
        assert!(f.app.record().unwrap().is_none()); f.launch().unwrap(); assert_eq!(f.app.record().unwrap().unwrap()["generation"], record["generation"]);
    }
    #[test] fn interrupted_download_and_doctor_resume_without_duplicate_start() {
        let mut f = Fixture::new(); f.app.runner.fail_download = true; assert!(f.launch().is_err()); assert!(f.app.record().unwrap().is_none());
        f.app.runner.doctor_bad = true; assert!(f.launch().is_err()); assert_eq!(f.app.record().unwrap().unwrap()["phase"], "running");
        f.app.runner.calls.clear(); f.app.runner.doctor_bad = false; f.launch().unwrap(); assert!(!f.app.runner.calls.iter().any(|e| e.label == "local start"));
        assert!(f.app.runner.calls.iter().all(|e| e.label != "model download"));
    }
    #[test] fn bad_platform_payload_marker_and_reused_pid_fail_closed() {
        let mut f = Fixture::new(); f.app.runner.bad_os = true; assert!(f.launch().unwrap_err().contains("Nothing downloaded")); assert!(!f.app.root.exists());
        f.app.runner.bad_os = false; f.launch().unwrap(); f.app.runner.reused_pid = true; f.app.runner.calls.clear(); assert!(f.app.execute(Command::Down).is_err());
        assert!(!f.app.runner.calls.iter().any(|e| e.label == "local stop")); assert!(f.app.record().unwrap().is_some());
        f.app.runner.reused_pid = false; f.app.execute(Command::Down).unwrap();
        let model = f.app.root.join("models").join(MODEL_REVISION).join("Qwen3-8B"); fs::write(model.join(MODEL_FILES[0].0), "stale").unwrap(); fs::write(model.join(".ok"), "old revision success").unwrap();
        f.app.runner.calls.clear(); assert!(f.launch().unwrap_err().contains("digest mismatch")); assert!(!f.app.runner.calls.iter().any(|e| e.label == "local start"));
    }
    #[test] fn pinned_source_snapshots_build_without_mutating_supplied_sources() {
        let mut f = Fixture::new(); f.options.release = None;
        for label in ["engine", "harness"] { let source = f.directory.join(label); text::private_dir(&source).unwrap(); fs::write(source.join("Cargo.toml"), "mock").unwrap(); if label == "engine" { f.options.engine_source = Some(source); } else { f.options.harness_source = Some(source); } }
        f.launch().unwrap(); assert!(!f.options.engine_source.as_ref().unwrap().join("target").exists());
        let setup = read_fields(&f.app.root.join("setup.conf")).unwrap().unwrap(); assert_eq!(setup["engine_revision"], "a".repeat(40));
        f.app.runner.head = "b".repeat(40); assert!(f.app.source(&mut setup.clone(), "engine", "unused").unwrap_err().contains("revision changed"));
    }
    #[test] fn monorepo_snapshots_strip_component_prefixes_from_roots_and_subdirectories() {
        let _process_guard = crate::runner::process_test_lock();
        let f = Fixture::new(); let repository = f.directory.join("monorepo"); text::private_dir(&repository).unwrap();
        for label in ["engine", "harness"] { text::private_dir(&repository.join(label)).unwrap(); fs::write(repository.join(label).join("Cargo.toml"), label).unwrap(); }
        fs::write(repository.join("README.md"), "root only").unwrap();
        let git = |args: &[&str]| { let output = std::process::Command::new("git").arg("-C").arg(&repository).args(args).output().unwrap(); assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr)); String::from_utf8(output.stdout).unwrap() };
        git(&["init", "-q"]); git(&["add", "."]); git(&["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgSign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Fixture"]);
        let revision = git(&["rev-parse", "HEAD"]).trim().to_owned();
        for from_root in [true, false] { for label in ["engine", "harness"] {
            let root = f.directory.join(format!("data-{label}-{from_root}"));
            let mut local = Local { root, dry: false, runner: crate::runner::SystemRunner { dry: false } };
            let supplied = if from_root { repository.clone() } else { repository.join(label) };
            let mut setup = Fields::from([(format!("{label}_source"), supplied.to_string_lossy().into_owned())]);
            let snapshot = local.source(&mut setup, label, "unused").unwrap();
            assert_eq!(fs::read_to_string(snapshot.join("Cargo.toml")).unwrap(), label); assert!(!snapshot.join(label).exists() && !snapshot.join("README.md").exists());
            if label == "engine" { assert_eq!(fs::read_to_string(snapshot.join("REVISION")).unwrap().trim(), revision); }
            assert_eq!(setup[&format!("{label}_revision")], revision); assert!(!supplied.join("target").exists());
        } }
        assert!(git(&["status", "--porcelain"]).is_empty());
    }
    #[test] fn supplied_weights_are_read_only_and_rehashed_without_downloads() {
        let mut f = Fixture::new(); let weights = f.directory.join("Qwen3-8B"); text::private_dir(&weights).unwrap();
        for &(name, _) in MODEL_FILES { fs::write(weights.join(name), name).unwrap(); }
        f.options.weights_source = Some(weights.clone()); f.launch().unwrap(); f.app.execute(Command::Down).unwrap();
        assert!(!weights.join("verified.conf").exists() && !f.app.root.join("models").exists() && !f.app.runner.calls.iter().any(|e| e.label == "model download" || e.label == "model size"));
        fs::write(weights.join(MODEL_FILES[0].0), "altered").unwrap(); f.app.runner.calls.clear();
        assert!(f.launch().unwrap_err().contains("digest mismatch")); assert!(!f.app.runner.calls.iter().any(|e| e.label == "local start"));
    }
    #[test] fn supplied_alias_imports_only_verified_payloads_under_canonical_name() {
        let mut f = Fixture::new(); let weights = f.directory.join("snapshot"); text::private_dir(&weights).unwrap();
        for &(name, _) in MODEL_FILES { fs::write(weights.join(name), name).unwrap(); }
        f.options.weights_source = Some(weights.clone()); f.launch().unwrap();
        assert!(Path::new(&f.app.record().unwrap().unwrap()["model_path"]).ends_with("Qwen3-8B") && !weights.join("verified.conf").exists() && !f.app.runner.calls.iter().any(|e| e.label == "model download" || e.label == "model size"));
    }
    #[test] fn readiness_cancellation_stops_only_this_runs_start_without_retry_sleep() {
        let mut f = Fixture::new(); f.app.runner.interrupt_health = true;
        assert!(f.launch().unwrap_err().contains("Interrupted during readiness"));
        assert_eq!(f.app.runner.pauses, 0); assert!(!f.app.runner.alive);
        assert!(f.app.record().unwrap().is_none());
        assert_eq!(f.app.runner.calls.iter().filter(|e| e.label == "local stop").count(), 1);
        f.app.runner.interrupt_health = false; f.app.runner.cancelled = false; f.launch().unwrap();
        f.app.runner.calls.clear(); f.app.runner.interrupt_health = true;
        assert!(f.launch().unwrap_err().contains("previously running server"));
        assert!(f.app.runner.alive && f.app.record().unwrap().is_some());
        assert!(!f.app.runner.calls.iter().any(|e| e.label == "local stop"));
    }
    #[test] fn missing_persisted_sources_refuse_fetch_even_with_retained_snapshots() {
        let mut f = Fixture::new(); text::private_dir(&f.app.root).unwrap();
        for label in ["engine", "harness"] {
            let supplied = f.directory.join(format!("{label}-local")); text::private_dir(&supplied).unwrap();
            let mut setup = Fields::from([(format!("{label}_source"), supplied.to_string_lossy().into_owned())]);
            let snapshot = f.app.source(&mut setup, label, "mock-network-repository").unwrap(); assert!(snapshot.exists());
            fs::remove_dir_all(&supplied).unwrap(); f.app.runner.calls.clear();
            let mut persisted = read_fields(&f.app.root.join("setup.conf")).unwrap().unwrap();
            let error = f.app.source(&mut persisted, label, "mock-network-repository").unwrap_err();
            assert!(error.contains("Saved local") && error.contains("missing") && error.contains("Nothing fetched"));
            assert!(f.app.runner.calls.is_empty() && !supplied.exists() && snapshot.exists());
            persisted.insert(format!("{label}_source"), String::new());
            // Bare-source acquisition is still the explicitly chosen managed workflow.
            let _ = f.app.source(&mut persisted, label, "mock-network-repository");
            let fetch = f.app.runner.calls.iter().find(|e| e.label == format!("fetch {label}")).unwrap();
            assert_eq!(Path::new(fetch.args.last().unwrap()).parent().unwrap(), f.app.root.join("sources"));
        }
    }
    #[test] fn down_recovers_a_build_intent_before_any_server_record_exists() {
        let mut f = Fixture::new();
        let mut build = Exec::new("mock unspawned build", "sh", vec!["-c".into(), "exit 0".into()]);
        crate::build_job::prepare(&f.app.root, &mut build).unwrap();
        assert!(f.app.record().unwrap().is_none());
        f.app.execute(Command::Down).unwrap();
        assert!(build.job.unwrap().join("stopped.conf").exists());
        assert!(f.app.runner.calls.is_empty());
    }

    #[test]
    fn streamed_answer_and_doctor_escape_hostile_text_without_changing_request_bytes() {
        use crate::output::{self, testing::Capture};
        let mut f = Fixture::new();
        let hostile = "normal é\n\t\x1b[2J\r\x07\u{202e}hidden\u{2066}";
        f.app.runner.answer_text = Some(hostile.into());
        f.app.runner.doctor_text = Some(hostile.into());
        let writer = Capture::default();
        output::with_writer(writer.clone(), || f.launch().unwrap());
        let text = String::from_utf8(writer.bytes.borrow().clone()).unwrap();
        let escaped = crate::text::terminal_text(hostile);
        assert_eq!(text.matches(escaped.as_ref()).count(), 2, "{text:?}");
        assert!(!text.contains('\x1b') && !text.contains('\r') && !text.contains('\x07') && !text.contains('\u{202e}') && !text.contains('\u{2066}'));
        assert!(text.contains("First content token:") && text.contains("Doctor: passed"));
        let request = &f.app.runner.calls.iter().find(|e| e.label == "first streaming answer").unwrap().stdin;
        assert!(!String::from_utf8_lossy(request).contains("seed"));
        assert_eq!(f.app.runner.answer_text.as_deref(), Some(hostile));
    }

    #[test]
    fn local_output_failure_cleans_only_a_server_started_by_this_launch() {
        use crate::output::{self, testing::Capture};
        use std::io::ErrorKind;
        for existing in [false, true] {
            for marker in ["Ice is less dense.", "doctor mock"] {
                for kind in [ErrorKind::BrokenPipe, ErrorKind::StorageFull] {
                    let mut f = Fixture::new();
                    if existing { f.launch().unwrap(); f.app.runner.calls.clear(); }
                    output::with_writer(Capture { fail_on: Some((marker, kind)), ..Capture::default() }, || {
                        assert!(f.launch().unwrap_err().contains("standard output"));
                        assert_eq!(output::failure(), Some(kind));
                    });
                    assert_eq!(f.app.runner.alive, existing);
                    assert_eq!(f.app.record().unwrap().is_some(), existing);
                    assert_eq!(f.app.runner.calls.iter().filter(|e| e.label == "local stop").count(), usize::from(!existing));
                    if marker == "Ice is less dense." {
                        assert!(!f.app.runner.calls.iter().any(|e| e.label == "local doctor"));
                    }
                }
            }
        }
    }


    // Regression reproduction: fake HTTP result; no network, files or credentials.
    #[test]
    fn observed_answer_failure_exit_probe() {
        let Ok(case) = std::env::var("VADE_TEST_ANSWER_CASE") else { return; };
        struct AnswerOnly { unauthorized: bool, wrong_model: bool }
        impl Runner for AnswerOnly {
            fn run(&mut self, exec: &Exec) -> Result<Output, String> {
                assert_eq!(exec.label, "first streaming answer");
                Ok(Output { success: self.wrong_model, stdout: if self.unauthorized {
                    r#"{"error":{"code":"invalid_api_key"}}"#.into()
                } else if self.wrong_model {
                    "data: {\"model\":\"other-model\",\"choices\":[{\"delta\":{\"content\":\"Wrong model answer.\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n".into()
                } else { String::new() } })
            }
            fn interactive(&mut self, _: &Exec) -> Result<bool, String> { panic!("unexpected interactive") }
            fn port_free(&mut self, _: u16) -> Result<(), String> { panic!("unexpected port") }
            fn pause(&mut self, _: Duration) { panic!("unexpected pause") }
        }
        let wrong_model = case.starts_with("wrong_model");
        let mut local = Local { root: std::env::temp_dir().join("vade-answer-unused"), dry: true,
            runner: AnswerOnly { unauthorized: case.starts_with("unauthorized"), wrong_model } };
        let record = Fields::from([("port".into(), "8080".into())]);
        let writer = crate::output::testing::Capture {
            fail_on: case.ends_with("closed").then_some((if wrong_model { "Wrong model answer." } else { "\n" }, std::io::ErrorKind::BrokenPipe)),
            ..Default::default()
        };
        crate::output::with_writer(writer, || {
            let error = local.answer(&record).unwrap_err();
            if case == "wrong_model_closed" {
                assert_eq!(crate::output::failure(), Some(std::io::ErrorKind::BrokenPipe));
                assert!(error.contains("standard output"));
            }
            crate::fail(&error);
        });
    }
    #[test]
    fn observed_failed_answer_survives_closed_stdout() {
        for case in ["unauthorized_open", "unauthorized_closed", "incomplete_open", "incomplete_closed", "wrong_model_open", "wrong_model_closed"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "run::tests::observed_answer_failure_exit_probe", "--nocapture"])
                .env("VADE_TEST_ANSWER_CASE", case).output().unwrap();
            assert_eq!(output.status.code(), Some(1), "{case}: failed HTTP answers must stay failures when stdout closes; stderr={}", String::from_utf8_lossy(&output.stderr));
        }
    }
