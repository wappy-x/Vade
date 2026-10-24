use super::*;
use crate::runner::Output;

const GIB: u64 = 1024 * 1024 * 1024;

fn identity() -> detect::Identity {
    detect::Identity {
        vendor: "Fixture OEM".into(),
        product: "GB10 test box".into(),
        os: "Fixture DGX OS".into(),
        architecture: "aarch64".into(),
        memory_total: 120 * GIB,
        memory_available: 100 * GIB,
        install_free: 80 * GIB,
        models_free: 60 * GIB,
        volumes_shared: false,
        driver: "580.00".into(),
        cuda: "13.0".into(),
        gpu: "NVIDIA GB10".into(),
    }
}

const REFUSAL_LOG: &str = "loading\nFAIL Requested size does not fit.\nAsked: weights/rank already loaded; reservation/session 2.00 GiB x 1 sessions (graphs/publication included); CONTEXT=32768; OUTPUT included in CONTEXT; total 12.0 GiB.\nAvailable: 10.0 GiB from the world-1 session pool.\nGap: 2.0 GiB.\nNext: 1. Lower CONTEXT / OUTPUT (positional context / FLASH_SERVE_OUTPUT_TOKENS) or FLASH_SERVE_MAX_ACTIVE.\n";
// A weights-stage refusal from vade launch --memory-bytes 60000000000 on a Spark
// (engine/crates/gpu/src/driver.rs allocator): no context can help.
const WEIGHTS_LOG: &str = "nvfp4_manifest files=22 bytes=102239292690\nFAIL Requested size does not fit.\nAsked: allocation 66016133892 bytes; live 0 bytes; graphs 0 bytes; weights/rank, sessions, CONTEXT and OUTPUT not supplied to allocator; total 61.48 GiB (66016133892 bytes).\nAvailable: 55.88 GiB (60000000000 bytes) from the configured allocation ceiling (derived from Linux MemAvailable (free + reclaimable page cache), capped by CUDA total, less this process's CUDA allocations since its first query with the server reserve, or an explicit budget), including live allocations and graph storage.\nGap: 5.60 GiB (6016133892 bytes).\nNext: 1. Lower CONTEXT / OUTPUT or max sessions: spark-serve --context-tokens / --output-tokens / --max-active; Flash positional CONTEXT / FLASH_SERVE_OUTPUT_TOKENS / FLASH_SERVE_MAX_ACTIVE. 2. In containers, spark-serve supports --gpu-budget-mib N; size N by rounding up (weights + shared workspace + session reservations + retained cache + headroom) / 1048576, within verified host capacity. Flash uses SEQPAR_MEMORY_BYTES in bytes; its CUDA safety cap still applies. 3. Use more Sparks with supported tensor parallelism.\nMemory: MemAvailable 114.34 GiB (122775670784 bytes); required 61.48 GiB (66016133892 bytes); ceiling 55.88 GiB (60000000000 bytes). Fix: MemAvailable already counts page cache as free, so dropping caches does not help; stop other processes holding memory (see free -g and nvidia-smi) or lower CONTEXT or chats.\n";

struct MockDetector {
    identity: detect::Identity,
    calls: usize,
}
impl Detector for MockDetector {
    fn detect(&mut self, install: &Path, models: &Path) -> Result<detect::Identity, String> {
        assert_ne!(install, models);
        self.calls += 1;
        Ok(self.identity.clone())
    }
}

struct MockRunner {
    root: PathBuf,
    calls: Vec<Exec>,
    cleanup_ops: Vec<String>,
    health_ready: bool,
    listener_owned: bool,
    fail_start: bool,
    // The engine's memory admission refuses contexts above this many tokens.
    refuse_above: Option<u32>,
    // The engine refuses the weights themselves, at any context.
    refuse_weights: bool,
    refused: bool,
    start_output: bool,
    fail_cleanup: Option<String>,
    chat_success: bool,
    write_manifest_receipts: bool,
    starts: usize,
    chats: usize,
    port_checks: usize,
    port_busy: bool,
}

impl MockRunner {
    fn new(root: &Path) -> Self {
        Self {
            root: root.into(),
            calls: Vec::new(),
            cleanup_ops: Vec::new(),
            health_ready: true,
            listener_owned: true,
            fail_start: false,
            refuse_above: None,
            refuse_weights: false,
            refused: false,
            start_output: false,
            fail_cleanup: None,
            chat_success: true,
            write_manifest_receipts: false,
            starts: 0,
            chats: 0,
            port_checks: 0,
            port_busy: false,
        }
    }
    fn worker_ops(&self) -> Vec<String> {
        self.calls
            .iter()
            .filter(|e| e.args == ["__worker"])
            .map(|e| text::parse(std::str::from_utf8(&e.stdin).unwrap()).unwrap()["op"].clone())
            .collect()
    }
}

fn reply(stdout: &str) -> Result<Output, String> {
    Ok(Output {
        success: true,
        stdout: stdout.into(),
    })
}

fn fake_elf(path: &Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut bytes = [0u8; 64];
    bytes[..6].copy_from_slice(b"\x7fELF\x02\x01");
    bytes[18] = 183;
    fs::write(path, bytes).unwrap();
}

impl Runner for MockRunner {
    fn interrupted(&self) -> bool {
        false
    }
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        self.calls.push(exec.clone());
        if exec.args == ["__worker"] {
            let request = text::parse(std::str::from_utf8(&exec.stdin).unwrap()).unwrap();
            assert!(text::hex(&request["token"], 64));
            assert_eq!(
                request["state_dir"],
                self.root.join("state/worker").to_string_lossy()
            );
            return match request["op"].as_str() {
                "probe" => reply("phase=absent\n"),
                "start" => {
                    // Startup may execute only after durable ownership was saved.
                    let intent = read_fields(&self.root.join("state/run.conf"))
                        .unwrap()
                        .unwrap();
                    assert_eq!(intent["token"], request["token"]);
                    let allowed = [
                        "env.FLASH_SERVE_OUTPUT_TOKENS",
                        "env.TP_RUN_ID",
                        "env.SEQPAR_RDMA",
                        "env.TP_RDMA",
                        "env.XDG_CACHE_HOME",
                        "env.FLASH_NVFP4_MANIFEST",
                        "env.MTP", "env.SEQPAR_MEMORY_BYTES",
                        "env.VADE_FLASHNEXT_PREFIX_CACHE",
                    ];
                    assert!(
                        request
                            .keys()
                            .filter(|key| key.starts_with("env."))
                            .all(|key| allowed.contains(&key.as_str()))
                    );
                    assert_eq!(
                        request["env.XDG_CACHE_HOME"],
                        self.root.join("state/cache").to_string_lossy()
                    );
                    self.starts += 1;
                    self.refused = self.refuse_weights || self.refuse_above.is_some_and(|limit| request["arg.6"].parse::<u32>().unwrap() > limit);
                    if self.start_output {
                        output::line(format_args!("Fixture engine loading progress."))?;
                    }
                    if self.fail_start {
                        Err("Fixture engine failed to start.".into())
                    } else {
                        if self.write_manifest_receipts {
                            run_cpu_receipt(
                                Path::new(&request["model_path"]),
                                Path::new(&request["env.FLASH_NVFP4_MANIFEST"]),
                                Path::new(&request["env.XDG_CACHE_HOME"]),
                                "mixed",
                            );
                        }
                        reply("alive=true\nphase=running\n")
                    }
                }
                "inspect" if self.refused => reply("alive=false\nphase=stopped\n"),
                "inspect" => reply("alive=true\nphase=running\n"),
                "listener" if self.listener_owned => reply("listener=owned\n"),
                "listener" => Err("The recorded server does not own the loopback listener.".into()),
                "stop" => reply("alive=false\nphase=stopped\n"),
                "clean" => reply("phase=absent\n"),
                "logs" if self.refused => reply(&text::encode(&Fields::from([("log".into(), (if self.refuse_weights { WEIGHTS_LOG } else { REFUSAL_LOG }).into())]))),
                "logs" => reply("log=fixture engine diagnostic\n"),
                operation => panic!("Unexpected worker operation {operation}"),
            };
        }
        if exec.label == "local health" {
            assert_eq!(exec.program, "curl");
            assert!(exec.args.last().unwrap().starts_with("http://127.0.0.1:"));
            assert!(exec.args.contains(&"--noproxy".into()));
            assert!(!exec.args.iter().any(|s| s.starts_with("https:")));
            return reply(if self.health_ready {
                r#"{"status":"ready","model":"checkpoint"}"#
            } else {
                r#"{"status":"loading","model":"checkpoint"}"#
            });
        }
        if exec.program == "git" {
            assert_eq!(exec.label, "inspect local source");
            return match exec.args.first().map(String::as_str) {
                Some("rev-parse") => reply(&format!("{}\n", "a".repeat(40))),
                Some("diff") => reply(""),
                Some("ls-files") => {
                    reply("engine/Cargo.toml\0harness/Cargo.toml\0rust-toolchain.toml\0")
                }
                _ => panic!("Unexpected Git command {:?}", exec.args),
            };
        }
        // Simulate compiler outputs, never run a compiler or a GPU program.
        if exec.label.starts_with("Build ") {
            assert!(exec.unset.iter().any(|name| name == "SPARK_CUBIN"));
            assert_eq!(
                exec.env.get("CARGO_NET_OFFLINE").map(String::as_str),
                Some("true")
            );
            assert_eq!(
                exec.env.get("RUSTUP_AUTO_INSTALL").map(String::as_str),
                Some("0")
            );
            let target = Path::new(&exec.env["CARGO_TARGET_DIR"]);
            match exec.label.as_str() {
                "Build GPU kernels" => {
                    let ptx = exec
                        .cwd
                        .as_ref()
                        .unwrap()
                        .join("target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx");
                    fs::create_dir_all(ptx.parent().unwrap()).unwrap();
                    fs::write(ptx, ".version 8.0\n").unwrap();
                }
                "Build engine" => fake_elf(&target.join("release/flash-serve")),
                "Build chat" => fake_elf(&target.join("release/vade-harness")),
                _ => panic!("Unexpected build {}", exec.label),
            }
            return reply("");
        }
        panic!(
            "Unexpected external command {} {:?}",
            exec.program, exec.args
        );
    }
    fn cleanup(&mut self, exec: &Exec) -> Result<Output, String> {
        let fields = text::parse(std::str::from_utf8(&exec.stdin).unwrap()).unwrap();
        self.cleanup_ops.push(fields["op"].clone());
        if self.fail_cleanup.as_deref() == Some(&fields["op"]) {
            return Err("Fixture owned cleanup failed.".into());
        }
        self.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> {
        self.calls.push(exec.clone());
        self.chats += 1;
        assert!(matches!(exec.args[0].as_str(), "agent" | "direct"));
        assert!(
            exec.args
                .windows(2)
                .any(|pair| pair == ["--model", "checkpoint"])
        );
        assert!(!exec.env.contains_key("VADE_API_KEY_FILE"));
        // Chat must release the lock so another terminal can attach or stop.
        assert!(crate::state::Lock::acquire(&self.root).is_ok());
        Ok(self.chat_success)
    }
    fn port_free(&mut self, port: u16) -> Result<(), String> {
        assert!(port >= 1024);
        self.port_checks += 1;
        if self.port_busy { return Err("fixture port busy".into()); }
        Ok(())
    }
    fn pause(&mut self, _: Duration) {
        panic!("Fixture startup should be ready immediately");
    }
}

struct Fixture {
    base: PathBuf,
    root: PathBuf,
    home: PathBuf,
    model: PathBuf,
    _process_guard: std::sync::MutexGuard<'static, ()>,
}
impl Fixture {
    fn new() -> Self {
        // Forking subprocess tests can briefly inherit another test's lock fd.
        let process_guard = crate::runner::process_test_lock();
        let base = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("vade-launch-flow-{}", text::random_hex(8).unwrap()));
        let home = base.join("home");
        let root = home.join(".vade");
        let model = home.join("models/checkpoint");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), model::test_mixed_config(8192)).unwrap();
        model::test_write_serving_metadata(&model);
        fs::write(model.join("model.safetensors.index.json"), model::test_index()).unwrap();
        fs::write(model.join("model.safetensors"), [0; 32]).unwrap();
        Self {
            base,
            root,
            home,
            model,
            _process_guard: process_guard,
        }
    }
    fn launcher(&self) -> Launcher<MockRunner, MockDetector> {
        Launcher {
            root: self.root.clone(),
            home: self.home.clone(),
            runner: MockRunner::new(&self.root),
            detector: MockDetector {
                identity: identity(),
                calls: 0,
            },
        }
    }
    fn installed(&self) {
        let bundle = self.base.join("bundle");
        fs::create_dir_all(&bundle).unwrap();
        fake_elf(&bundle.join("flash-serve"));
        fake_elf(&bundle.join("vade-harness"));
        fs::write(bundle.join("flash.ptx"), ".version 8.0\n").unwrap();
        install::publish_test_bundle(&self.root, &bundle).unwrap();
    }
    fn configured(&self) {
        let model = model::discover(&self.home, Some(&self.model)).unwrap();
        Config::new(&self.root, &model)
            .unwrap()
            .save(&self.root)
            .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.base).unwrap();
    }
}

#[test]
fn owner_commands_are_option_free_and_have_help() {
    for (name, expected) in [("launch", Command::Launch(Default::default())), ("detect", Command::Detect)] {
        assert_eq!(crate::cli::parse(&[name.into()]).unwrap().command, expected);
        assert!(crate::cli::parse(&[name.into(), "extra".into()]).is_err());
        assert!(crate::cli::help(Some(name)).contains(name));
    }
}

#[test]
fn small_context_launch_and_saved_chat_use_direct_mode() {
    for sizing in ["checkpoint", "explicit"] {
        let fixture = Fixture::new();
        if sizing == "checkpoint" {
            // Force automatic checkpoint-derived sizing to the accepted lower bound.
            fs::write(fixture.model.join("config.json"), model::test_mixed_config(4096)).unwrap();
        }
        fixture.installed();
        let mut launcher = fixture.launcher();
        let args: Vec<String> = if sizing == "explicit" {
            ["launch", "--context", "4096", "--output", "1024"].into_iter().map(str::to_string).collect()
        } else { vec!["launch".into()] };
        launcher.execute(crate::cli::parse(&args).unwrap().command, false).unwrap();
        assert_eq!(Config::load(&fixture.root).unwrap().unwrap().context, 4096);
        launcher.execute(Command::Chat, false).unwrap();
        launcher.execute(Command::Launch(Default::default()), false).unwrap();
        assert_eq!(launcher.runner.starts, 1);
        assert_eq!(launcher.runner.chats, 3);
        for call in launcher.runner.calls.iter().filter(|e| e.label == "local chat") {
            assert_eq!(call.args[0], "direct");
            assert!(!call.args.iter().any(|a| a == "--context-tokens" || a == "--no-check"));
        }
        launcher.execute(Command::Down, false).unwrap();
    }
}

#[test]
fn fresh_launch_installs_starts_chats_and_relaunch_reuses_the_resident() {
    let _guard = crate::runner::process_test_lock();
    let mut exec = source_install_test_process("launch::tests::source_install_fixture_child");
    // Installer selection reads the process environment. Isolate this fixture
    // in a child rather than mutating the parallel test runner's environment.
    exec.unset
        .extend(["VADE_BUNDLE".into(), "VADE_MODELS".into()]);
    exec.env.insert(
        "VADE_SOURCE".into(),
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
    );
    exec.env
        .insert("VADE_TEST_SOURCE_INSTALL".into(), "1".into());
    for (value, disabled) in [(None, false), (Some("0"), false), (Some("1"), false), (Some("invalid"), false), (Some("1"), true), (Some("invalid"), true)] {
        let mut exec = exec.clone();
        exec.env.insert("VADE_TEST_CACHE_DISABLED".into(), u8::from(disabled).to_string());
        if let Some(value) = value {
            exec.env.insert("VADE_FLASHNEXT_PREFIX_CACHE".into(), value.into());
        } else {
            exec.unset.push("VADE_FLASHNEXT_PREFIX_CACHE".into());
        }
        let output = crate::runner::SystemRunner { dry: false }.run(&exec).unwrap();
        assert!(output.success, "{}", output.stdout);
        assert!(output.stdout.contains("source-install fixture complete"), "{}", output.stdout);
    }
    println!("source-install isolation complete");
}

fn source_install_test_process(test: &str) -> Exec {
    let mut exec = Exec::new(
        "source-install fixture",
        std::env::current_exe().unwrap().to_str().unwrap(),
        vec!["--exact".into(), test.into(), "--nocapture".into()],
    );
    exec.timeout = Duration::from_secs(30);
    exec.cancellable = false;
    exec
}

#[test]
fn source_install_fixture_ignores_exported_installer_selectors() {
    let f = Fixture::new();
    f.installed();
    let bundle = f.base.join("bundle");
    let installed = install::load(&f.root).unwrap().unwrap();
    let mut manifest = Fields::from([
        ("format".into(), "vade-launch-bundle-v1".into()),
        ("target".into(), "aarch64-unknown-linux-gnu".into()),
    ]);
    for key in ["engine_sha256", "ptx_sha256", "harness_sha256"] {
        manifest.insert(key.into(), installed[key].clone());
    }
    fs::write(bundle.join("launcher-bundle.conf"), text::encode(&manifest)).unwrap();
    let missing = f.base.join("missing");
    for inherited_bundle in [None, Some(missing.as_path()), Some(bundle.as_path())] {
        let mut exec = source_install_test_process(
            "launch::tests::fresh_launch_installs_starts_chats_and_relaunch_reuses_the_resident",
        );
        exec.timeout = Duration::from_secs(45);
        exec.env
            .insert("VADE_SOURCE".into(), missing.to_string_lossy().into_owned());
        if let Some(bundle) = inherited_bundle {
            exec.env
                .insert("VADE_BUNDLE".into(), bundle.to_string_lossy().into_owned());
        } else {
            exec.unset.push("VADE_BUNDLE".into());
        }
        let output = crate::runner::SystemRunner { dry: false }
            .run(&exec)
            .unwrap();
        assert!(output.success, "{}", output.stdout);
        assert!(
            output.stdout.contains("source-install isolation complete"),
            "{}",
            output.stdout
        );
    }
}

#[test]
fn source_install_fixture_child() {
    if std::env::var("VADE_TEST_SOURCE_INSTALL").as_deref() != Ok("1") {
        return;
    }
    assert!(std::env::var_os("VADE_BUNDLE").is_none());
    assert_eq!(
        PathBuf::from(std::env::var_os("VADE_SOURCE").unwrap()),
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
    );
    let f = Fixture::new();
    let mut launch = f.launcher();
    assert!(!f.root.join("config").exists());
    assert!(!f.root.join("stage/install.conf").exists());
    let disabled = std::env::var("VADE_TEST_CACHE_DISABLED").as_deref() == Ok("1");
    if !disabled && std::env::var("VADE_FLASHNEXT_PREFIX_CACHE").as_deref() == Ok("invalid") {
        let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
        assert!(error.contains("VADE_FLASHNEXT_PREFIX_CACHE must be 0 or 1"), "{error}");
        assert_eq!(launch.runner.starts, 0);
        assert!(!f.root.join("state/run.conf").exists());
        println!("source-install fixture complete");
        return;
    }
    let command = if disabled {
        crate::cli::parse(&["launch", "--cache-bytes", "0"].map(str::to_owned)).unwrap().command
    } else { Command::Launch(Default::default()) };
    launch.execute(command, false).unwrap();
    assert_eq!(launch.detector.calls, 1);
    assert_eq!(
        (
            launch.runner.starts,
            launch.runner.chats,
            launch.runner.port_checks
        ),
        (1, 1, 1)
    );
    assert_eq!(
        launch
            .runner
            .calls
            .iter()
            .filter(|e| e.label.starts_with("Build "))
            .count(),
        3
    );
    assert_eq!(
        launch.runner.worker_ops(),
        ["probe", "start", "inspect", "listener"]
    );
    let config = fs::read(f.root.join("config")).unwrap();
    let install = fs::read(f.root.join("stage/install.conf")).unwrap();
    let record = fs::read(f.root.join("state/run.conf")).unwrap();
    let start_request = launch
        .runner
        .calls
        .iter()
        .find(|e| e.label == "local start")
        .unwrap();
    let fields = text::parse(std::str::from_utf8(&start_request.stdin).unwrap()).unwrap();
    let expected_cache = if disabled { Some("0".into()) } else { std::env::var("VADE_FLASHNEXT_PREFIX_CACHE").ok() };
    assert_eq!(fields.get("env.VADE_FLASHNEXT_PREFIX_CACHE"), expected_cache.as_ref());
    let saved = Config::load(&f.root).unwrap().unwrap();
    assert_eq!(saved.prefix_cache().unwrap(), expected_cache);
    assert_eq!(saved.display().contains("cache_bytes=0"), disabled);
    assert_eq!(
        fields["env.FLASH_NVFP4_MANIFEST"],
        f.root.join("state/model-manifest.sha256").to_string_lossy()
    );
    assert!(
        !fields
            .values()
            .any(|s| matches!(s.as_str(), "--mtp" | "--indexer"))
    );

    // The resident engine already owns its loaded model. Removing discovery
    // metadata proves repeat launch does not traverse or rehash the checkpoint.
    fs::remove_file(f.model.join("config.json")).unwrap();
    launch.runner.calls.clear();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    assert_eq!(launch.detector.calls, 1);
    assert_eq!(
        (
            launch.runner.starts,
            launch.runner.chats,
            launch.runner.port_checks
        ),
        (1, 2, 1)
    );
    assert_eq!(launch.runner.worker_ops(), ["listener"]);
    assert_eq!(launch.runner.calls.len(), 3); // owned listener, loopback health, chat
    assert_eq!(fs::read(f.root.join("config")).unwrap(), config);
    assert_eq!(
        fs::read(f.root.join("stage/install.conf")).unwrap(),
        install
    );
    assert_eq!(fs::read(f.root.join("state/run.conf")).unwrap(), record);
    println!("source-install fixture complete");
}

// Run the engine's actual CPU receipt planner in its own environment. The
// fixture lock prevents fork/flock overlap; SystemRunner enforces the deadline.
fn run_cpu_receipt(model: &Path, manifest: &Path, cache: &Path, format: &str) {
    assert!(matches!(
        crate::runner::PROCESS_TEST_LOCK.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    let mut exec = Exec::new(
        "CPU checkpoint receipt fixture",
        std::env::current_exe().unwrap().to_str().unwrap(),
        vec![
            "--exact".into(),
            "launch::tests::cpu_receipt_helper".into(),
            "--nocapture".into(),
        ],
    );
    let fallback_home = cache.parent().unwrap().join("receipt-home");
    fs::create_dir_all(&fallback_home).unwrap();
    for (name, value) in [
        ("HOME", fallback_home.to_string_lossy().into_owned()),
        ("VADE_TEST_RECEIPT_MODEL", model.to_string_lossy().into_owned()),
        ("VADE_TEST_RECEIPT_FORMAT", format.into()),
        ("FLASH_NVFP4_MANIFEST", manifest.to_string_lossy().into_owned()),
        ("XDG_CACHE_HOME", cache.to_string_lossy().into_owned()),
    ] {
        exec.env.insert(name.into(), value);
    }
    exec.timeout = Duration::from_secs(30);
    exec.cancellable = false;
    let output = crate::runner::SystemRunner { dry: false }.run(&exec).unwrap();
    assert!(output.success, "{}", output.stdout);
    assert!(
        output.stdout.contains("CPU checkpoint receipt complete"),
        "{}", output.stdout
    );
    assert!(!fallback_home.join(".cache").exists());
}

#[test]
fn cpu_receipt_helper() {
    let Some(model) = std::env::var_os("VADE_TEST_RECEIPT_MODEL") else {
        return;
    };
    let plan = engine_model::flash::manifest::ReceiptPlan::open(Path::new(&model)).unwrap();
    let expected = match std::env::var("VADE_TEST_RECEIPT_FORMAT").unwrap().as_str() {
        "mixed" => engine_model::flash::WeightFormat::Nvfp4,
        "fp8" => engine_model::flash::WeightFormat::Fp8,
        other => panic!("unexpected receipt fixture format {other}"),
    };
    assert_eq!(plan.format, expected);
    let receipt = plan.finish().unwrap();
    assert!(receipt.bytes > 0);
    println!("CPU checkpoint receipt complete: {receipt:?}");
}

fn fixture_manifest(model: &Path) -> String {
    ["model.safetensors.index.json", "model.safetensors"]
        .into_iter()
        .map(|name| {
            let hash = crate::sha256::digest_reader(fs::File::open(model.join(name)).unwrap()).unwrap();
            format!("{hash}  {name}\n")
        })
        .collect()
}

#[test]
fn fp8_receipts_use_checkpoint_paths_despite_nvfp4_override() {
    let f = Fixture::new();
    fs::write(f.model.join("config.json"), model::test_config(8192)).unwrap();
    // Inspection understands FP8; the local launcher now requires FP4.
    // The serving receipt contract remains checkpoint-local for FP8.
    let discovered = model::discover(&f.home, Some(&f.model)).unwrap();
    Config::new(&f.root, &discovered).unwrap();
    let private = f.root.join("state/model-manifest.sha256");
    text::private_dir(&f.root).unwrap();
    text::private_dir(private.parent().unwrap()).unwrap();
    fs::write(&private, "NVFP4 override must remain unused\n").unwrap();
    let cache = f.root.join("state/cache");
    run_cpu_receipt(&f.model, &private, &cache, "fp8");
    let manifest = fixture_manifest(&f.model);
    assert_eq!(
        fs::read(f.model.join("manifest.sha256")).unwrap(),
        manifest.as_bytes()
    );
    assert!(f.model.join("manifest.sha256.sizes").is_file());
    assert_eq!(
        fs::read(&private).unwrap(),
        b"NVFP4 override must remain unused\n"
    );
    assert!(!f.root.join("state/model-manifest.sha256.sizes").exists());

    // FP8 replaces a stale server-written cache in that same checkpoint folder.
    fs::write(f.model.join("manifest.sha256"), "stale checkpoint cache\n").unwrap();
    fs::write(f.model.join("manifest.sha256.sizes"), "stale sizes\n").unwrap();
    run_cpu_receipt(&f.model, &private, &cache, "fp8");
    assert_eq!(
        fs::read(f.model.join("manifest.sha256")).unwrap(),
        manifest.as_bytes()
    );
    assert_ne!(
        fs::read(f.model.join("manifest.sha256.sizes")).unwrap(),
        b"stale sizes\n"
    );
    assert_eq!(
        fs::read(&private).unwrap(),
        b"NVFP4 override must remain unused\n"
    );

    // A full supported FP8 checkpoint still cannot pass one-Spark admission.
    fs::OpenOptions::new()
        .write(true)
        .open(f.model.join("model.safetensors"))
        .unwrap()
        .set_len(185_500_000_000)
        .unwrap();
    let mut launch = f.launcher();
    let error = launch.execute(Command::Install, false).unwrap_err();
    assert!(error.contains("Local launch needs"), "{error}");
    assert!(!f.root.join("config").exists());
    assert!(launch.runner.calls.is_empty());
}

#[test]
fn cold_launch_stages_only_manifest_metadata_and_keeps_receipts_private() {
    let f = Fixture::new();
    f.installed();
    let manifest = fixture_manifest(&f.model);
    let owner_sizes = b"owner's existing size cache\n";
    fs::write(f.model.join("manifest.sha256"), &manifest).unwrap();
    fs::write(f.model.join("manifest.sha256.sizes"), owner_sizes).unwrap();
    let weights = fs::read(f.model.join("model.safetensors")).unwrap();
    let mut launch = f.launcher();
    launch.runner.write_manifest_receipts = true;
    launch.execute(Command::Launch(Default::default()), false).unwrap();

    let request = launch
        .runner
        .calls
        .iter()
        .find(|e| e.label == "local start")
        .unwrap();
    let fields = text::parse(std::str::from_utf8(&request.stdin).unwrap()).unwrap();
    assert_eq!(fields["model_path"], f.model.to_string_lossy());
    assert_eq!(fields["arg.5"], f.model.to_string_lossy());
    let staged = f.root.join("state/model-manifest.sha256");
    assert_eq!(fields["env.FLASH_NVFP4_MANIFEST"], staged.to_string_lossy());
    assert_eq!(fs::read(&staged).unwrap(), manifest.as_bytes());
    assert_eq!(
        fs::read(f.model.join("manifest.sha256")).unwrap(),
        manifest.as_bytes()
    );
    assert_eq!(
        fs::read(f.model.join("manifest.sha256.sizes")).unwrap(),
        owner_sizes
    );
    assert_eq!(
        fs::read(f.model.join("model.safetensors")).unwrap(),
        weights
    );
    let sizes = fs::read_to_string(f.root.join("state/model-manifest.sha256.sizes")).unwrap();
    let manifest_hash = crate::sha256::digest_reader(manifest.as_bytes()).unwrap();
    assert_eq!(sizes.lines().next(), Some(manifest_hash.as_str()));
    assert!(sizes.lines().any(|line| line == "32  model.safetensors"));
    assert!(f.root.join("state/cache/vade-manifest-v1").is_dir());
    assert!(!f.home.join(".cache").exists());

    // No checkpoint payload or discovery metadata belongs in the install tree.
    let mut pending = vec![f.root.clone()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                assert_ne!(entry.file_name(), "config.json");
                assert_ne!(entry.file_name(), "tokenizer.json");
                assert_ne!(
                    entry.path().extension(),
                    Some(std::ffi::OsStr::new("safetensors"))
                );
                assert_ne!(fs::read(entry.path()).unwrap(), weights);
            }
        }
    }

    // A resident attach must not reread either discovery or manifest metadata.
    fs::remove_file(f.model.join("config.json")).unwrap();
    let changed_manifest = vec![b'x'; 1024 * 1024 + 1];
    fs::write(f.model.join("manifest.sha256"), &changed_manifest).unwrap();
    launch.runner.calls.clear();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    assert_eq!(launch.detector.calls, 1);
    assert_eq!((launch.runner.starts, launch.runner.chats), (1, 2));
    assert_eq!(launch.runner.worker_ops(), ["listener"]);
    assert_eq!(launch.runner.calls.len(), 3);
    assert_eq!(fs::read(staged).unwrap(), manifest.as_bytes());
    assert_eq!(
        fs::read(f.model.join("manifest.sha256")).unwrap(),
        changed_manifest
    );
    assert_eq!(
        fs::read(f.model.join("manifest.sha256.sizes")).unwrap(),
        owner_sizes
    );
    assert_eq!(
        fs::read(f.model.join("model.safetensors")).unwrap(),
        weights
    );
}

#[test]
fn generated_private_manifest_survives_restart_and_warm_attach() {
    use std::os::unix::fs::MetadataExt;

    let f = Fixture::new();
    f.installed();
    let mut launch = f.launcher();
    launch.runner.write_manifest_receipts = true;
    assert!(!f.model.join("manifest.sha256").exists());
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    let staged = f.root.join("state/model-manifest.sha256");
    let generated = fs::read(&staged).unwrap();
    let inode = fs::metadata(&staged).unwrap().ino();
    let sizes = fs::read(f.root.join("state/model-manifest.sha256.sizes")).unwrap();
    assert!(!generated.is_empty());
    assert!(!f.model.join("manifest.sha256").exists());
    assert!(!f.model.join("manifest.sha256.sizes").exists());

    launch.execute(Command::Down, false).unwrap();
    assert_eq!(fs::read(&staged).unwrap(), generated);
    launch.runner.calls.clear();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    assert_eq!(
        launch.runner.worker_ops(),
        ["probe", "start", "inspect", "listener"]
    );
    assert_eq!((launch.detector.calls, launch.runner.starts), (2, 2));
    assert_eq!(fs::read(&staged).unwrap(), generated);
    assert_eq!(fs::metadata(&staged).unwrap().ino(), inode);
    assert_eq!(
        fs::read(f.root.join("state/model-manifest.sha256.sizes")).unwrap(),
        sizes
    );

    fs::remove_file(f.model.join("config.json")).unwrap();
    launch.runner.calls.clear();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    assert_eq!(launch.runner.worker_ops(), ["listener"]);
    assert_eq!(launch.runner.calls.len(), 3);
    assert_eq!(
        (
            launch.detector.calls,
            launch.runner.starts,
            launch.runner.chats
        ),
        (2, 2, 3)
    );
    assert_eq!(fs::read(&staged).unwrap(), generated);
    assert_eq!(fs::metadata(&staged).unwrap().ino(), inode);
    assert!(!f.model.join("manifest.sha256").exists());
    assert!(!f.model.join("manifest.sha256.sizes").exists());
}

#[test]
fn oversized_manifest_refuses_before_engine_start_without_copying_it() {
    let f = Fixture::new();
    f.installed();
    let manifest = vec![b'x'; 1024 * 1024 + 1];
    fs::write(f.model.join("manifest.sha256"), &manifest).unwrap();
    let mut launch = f.launcher();
    let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
    assert!(error.contains("Model checksum manifest exceeds 1 MiB."));
    assert!(error.contains("fixture engine diagnostic\nFull log: vade logs"));
    assert_eq!(launch.runner.starts, 0);
    assert_eq!(launch.runner.chats, 0);
    assert_eq!(launch.runner.worker_ops(), ["probe", "logs", "stop", "clean"]);
    assert!(!f.root.join("state/model-manifest.sha256").exists());
    assert!(!f.root.join("state/run.conf").exists());
    assert_eq!(fs::read(f.model.join("manifest.sha256")).unwrap(), manifest);
}

#[test]
fn insufficient_memory_or_either_disk_refuses_before_install() {
    for (memory, install_disk, model_disk, reason) in [
        (8 * GIB, 80 * GIB, 60 * GIB, "Not enough memory"),
        (100 * GIB, GIB, 60 * GIB, "Not enough install disk"),
        (100 * GIB, 80 * GIB, 1, "Not enough model disk"),
    ] {
        let f = Fixture::new();
        let mut launch = f.launcher();
        launch.detector.identity.memory_available = memory;
        launch.detector.identity.install_free = install_disk;
        launch.detector.identity.models_free = model_disk;
        let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
        assert!(error.starts_with(reason), "{error}");
        assert_eq!(error.lines().count(), 1);
        assert!(!f.root.join("config").exists());
        assert!(!f.root.join("stage").exists());
        assert!(launch.runner.calls.is_empty());
        assert_eq!(launch.runner.starts, 0);
    }
}

#[test]
fn install_is_idempotent_and_does_not_start_or_chat() {
    let f = Fixture::new();
    f.installed();
    let mut launch = f.launcher();
    launch.execute(Command::Install, false).unwrap();
    let config = fs::read(f.root.join("config")).unwrap();
    let artifacts = fs::read(f.root.join("stage/install.conf")).unwrap();
    launch.execute(Command::Install, false).unwrap();
    assert_eq!(launch.detector.calls, 2);
    assert!(launch.runner.calls.is_empty());
    assert_eq!((launch.runner.starts, launch.runner.chats), (0, 0));
    assert!(!f.root.join("state/run.conf").exists());
    assert_eq!(fs::read(f.root.join("config")).unwrap(), config);
    assert_eq!(
        fs::read(f.root.join("stage/install.conf")).unwrap(),
        artifacts
    );
}

#[test]
fn missing_saved_model_refuses_without_starting_or_reselecting() {
    let f = Fixture::new();
    f.installed();
    f.configured();
    let config = fs::read(f.root.join("config")).unwrap();
    fs::remove_dir_all(&f.model).unwrap();
    let mut launch = f.launcher();
    let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
    assert!(
        error.contains("Saved model folder") && error.contains("unavailable"),
        "{error}"
    );
    assert_eq!(launch.detector.calls, 0);
    assert!(launch.runner.calls.is_empty());
    assert_eq!(fs::read(f.root.join("config")).unwrap(), config);
}

#[test]
fn unhealthy_or_unowned_resident_never_spawns_a_duplicate() {
    for listener_owned in [true, false] {
        let f = Fixture::new();
        f.installed();
        let mut launch = f.launcher();
        launch.execute(Command::Launch(Default::default()), false).unwrap();
        let record = fs::read(f.root.join("state/run.conf")).unwrap();
        launch.runner.calls.clear();
        launch.runner.health_ready = false;
        launch.runner.listener_owned = listener_owned;
        assert!(launch.execute(Command::Launch(Default::default()), false).is_err());
        assert_eq!(launch.runner.starts, 1);
        assert_eq!(launch.runner.chats, 1);
        assert_eq!(launch.detector.calls, 1);
        assert_eq!(launch.runner.worker_ops(), ["listener"]);
        assert!(launch.runner.cleanup_ops.is_empty());
        assert_eq!(fs::read(f.root.join("state/run.conf")).unwrap(), record);
    }
}

#[test]
fn owner_status_forms_verify_the_same_running_engine() {
    use crate::{evals::json, status::Snapshot};
    let f = Fixture::new();
    f.installed();
    let mut launch = f.launcher();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    let record = fs::read(f.root.join("state/run.conf")).unwrap();
    let config = Config::load(&f.root).unwrap().unwrap();
    let artifacts = install::load(&f.root).unwrap().unwrap();
    for healthy in [true, false] {
        for owned in [true, false] {
            launch.runner.health_ready = healthy;
            launch.runner.listener_owned = owned;
            launch.runner.calls.clear();
            assert_eq!(
                launch.execute(Command::Status, false).is_ok(),
                healthy && owned
            );
            let mut report = Snapshot::new("local", false);
            launch.status_snapshot(&mut report, false).unwrap();
            let (encoded, code) = report.finish(Duration::ZERO);
            let value = json::parse(&encoded).unwrap();
            assert_eq!(value["phase"].as_str(), Some("running"));
            assert_eq!(value["model"].as_str(), Some("checkpoint"));
            assert_eq!(
                value["endpoint"].as_str(),
                Some(format!("{}/v1", config.endpoint()).as_str())
            );
            assert_eq!(value["ready"], json::Value::Bool(healthy && owned));
            assert_eq!(value["generation"].as_str(), Some(artifacts["generation"].as_str()));
            assert_eq!(value["build"].as_str(), Some(artifacts["engine_sha256"].as_str()));
            assert_eq!(value["settings"]["context_limit"], json::Value::Int(config.context as i64));
            assert_eq!(value["settings"]["output_limit"], json::Value::Int(config.output as i64));
            assert_eq!(value["settings"]["fp4_local"], json::Value::Bool(true));
            assert_eq!(value["settings"]["mtp_source"].as_str(), Some("engine_default"));
            assert_eq!(value["settings"]["mtp"], json::Value::Null);
            assert_eq!(value["settings"]["requested_chats"], json::Value::Null);
            assert_eq!(value["tokens_per_second"], json::Value::Null);
            assert_eq!(code, i32::from(!(healthy && owned)));
            assert!(!encoded.contains("qwen3-8b"));
            assert!(value.get("token").is_none());
            assert!(!encoded.contains(&launch.record().unwrap().unwrap()["token"]));
            assert_eq!(fs::read(f.root.join("state/run.conf")).unwrap(), record);
        }
    }
    assert_eq!(launch.runner.starts, 1);
    assert_eq!(launch.runner.chats, 1);
}

#[test]
fn owner_json_status_does_not_call_an_incomplete_installation_stopped() {
    use crate::{evals::json, status::Snapshot};
    let f = Fixture::new();
    f.installed();
    let mut launch = f.launcher();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    fs::remove_file(f.root.join("config")).unwrap();
    assert!(launch.execute(Command::Status, false).is_err());
    let mut report = Snapshot::new("local", false);
    let error = launch.status_snapshot(&mut report, false).unwrap_err();
    report.error(error);
    let (encoded, code) = report.finish(Duration::ZERO);
    let value = json::parse(&encoded).unwrap();
    assert_eq!(code, 1);
    assert_eq!(value["phase"].as_str(), Some("running"));
    assert_eq!(value["ready"], json::Value::Null);
    assert!(!value["errors"].as_array().unwrap().is_empty());
}

#[test]
fn failed_start_cleans_only_its_ownership_and_keeps_install_for_retry() {
    let f = Fixture::new();
    f.installed();
    let mut launch = f.launcher();
    launch.runner.fail_start = true;
    let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
    assert!(error.starts_with("Fixture engine failed to start.\n"));
    assert!(error.contains("fixture engine diagnostic\nFull log: vade logs"));
    assert_eq!(
        launch.runner.worker_ops(),
        ["probe", "start", "logs", "stop", "clean"]
    );
    assert_eq!(launch.runner.cleanup_ops, ["stop", "clean"]);
    let tokens = launch
        .runner
        .calls
        .iter()
        .filter(|e| e.args == ["__worker"])
        .map(|e| text::parse(std::str::from_utf8(&e.stdin).unwrap()).unwrap()["token"].clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(tokens.len(), 1);
    assert_eq!(launch.runner.chats, 0);
    assert!(!f.root.join("state/run.conf").exists());
    assert!(f.root.join("config").is_file());
    assert!(install::load(&f.root).unwrap().is_some());
    assert!(f.model.join("model.safetensors").is_file());
}

#[path = "attach_tests.rs"]
mod attach_tests;

#[test]
fn resident_attach_with_real_subprocesses_reuses_saved_state() {
    attach_tests::check_resident_attach();
}

#[test]
fn default_context_is_32768_and_steps_down_only_on_engine_memory_refusal() {
    let f = Fixture::new();
    fs::write(f.model.join("config.json"), model::test_mixed_config(262144)).unwrap();
    f.installed();
    let mut launch = f.launcher();
    // A per-token pre-estimate would refuse 32768 here; only the engine decides.
    let model = model::discover(&f.home, Some(&f.model)).unwrap();
    launch.detector.identity.memory_available = model.bytes + 8 * GIB;
    launch.execute(Command::Install, false).unwrap();
    let saved = Config::load(&f.root).unwrap().unwrap();
    assert_eq!((saved.context, saved.output), (32768, 8192));
    // A weights refusal is not about context: one start, the engine's text, config untouched.
    let config = fs::read(f.root.join("config")).unwrap();
    launch.runner.refuse_weights = true;
    let error = launch.execute(Command::Up(Default::default()), false).unwrap_err();
    assert!(error.contains("CONTEXT and OUTPUT not supplied to allocator"), "{error}");
    assert!(!error.contains("refused --context"), "{error}");
    assert_eq!(launch.runner.starts, 1);
    assert_eq!(fs::read(f.root.join("config")).unwrap(), config);
    assert!(!f.root.join("state/run.conf").exists());
    launch.runner.refuse_weights = false;
    // Nothing smaller than 4096 fits: the engine's diagnostic, and the saved 32768 stays.
    launch.runner.refuse_above = Some(2048);
    let error = launch.execute(Command::Up(Default::default()), false).unwrap_err();
    assert!(error.contains("Requested size does not fit."), "{error}");
    assert_eq!(launch.runner.starts, 5);
    assert_eq!(fs::read(f.root.join("config")).unwrap(), config);
    assert!(!f.root.join("state/run.conf").exists());
    launch.runner.refuse_above = Some(8192);
    let capture = crate::output::testing::Capture::default();
    output::with_writer(capture.clone(), || launch.execute(Command::Up(Default::default()), false).unwrap());
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Context 8192 selected (the engine refused 32768 and 16384 for memory); saved; run vade launch --context 32768 to try the default again.\n"), "{printed}");
    // Refused 32768 and 16384; started 8192 and saved it only then.
    assert_eq!(launch.runner.starts, 8);
    let saved = Config::load(&f.root).unwrap().unwrap();
    assert_eq!((saved.context, saved.output), (8192, 8192));
    let record = read_fields(&f.root.join("state/run.conf")).unwrap().unwrap();
    assert_eq!(record["context_tokens"], "8192");
    launch.execute(Command::Down, false).unwrap();
    // An explicit --output that the smaller context lowers is named too.
    fs::write(f.root.join("config"), &config).unwrap();
    let options = crate::cli::parse(&["up", "--output", "16384"].map(str::to_owned)).unwrap().command;
    let capture = crate::output::testing::Capture::default();
    output::with_writer(capture.clone(), || launch.execute(options, false).unwrap());
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Context 8192 selected (the engine refused 32768 and 16384 for memory); output 16384 lowered to 8192; saved; run vade launch --context 32768 to try the default again.\n"), "{printed}");
    let saved = Config::load(&f.root).unwrap().unwrap();
    assert_eq!((saved.context, saved.output), (8192, 8192));
    launch.execute(Command::Down, false).unwrap();
}

#[test]
fn step_down_and_refusal_parsing() {
    let model_root = Fixture::new();
    fs::write(model_root.model.join("config.json"), model::test_mixed_config(262144)).unwrap();
    let model = model::discover(&model_root.home, Some(&model_root.model)).unwrap();
    let mut config = Config::new(&model_root.root, &model).unwrap();
    assert_eq!(config.context, 32768);
    let mut steps = Vec::new();
    while let Some(next) = config.step_down(false) { steps.push((next, config.output)); }
    assert_eq!(steps, [(16384, 8192), (8192, 8192), (4096, 4096)]);
    config.context = 65536;
    config.output = 16384;
    assert_eq!(config.step_down(true), Some(32768));
    assert_eq!(config.output, 16384);
    assert!(config::memory_refusal(REFUSAL_LOG).unwrap().starts_with("FAIL Requested size does not fit.\nAsked: weights/rank already loaded;"));
    assert_eq!(config::memory_refusal(WEIGHTS_LOG), None);
    // The weight loader's refusal (engine/crates/gpu/src/backend/weights.rs) names CONTEXT the same way.
    assert_eq!(config::memory_refusal("FAIL Requested size does not fit.\nAsked: weights/rank total 93204889600 bytes including live allocations; graphs 0 bytes; sessions, CONTEXT and OUTPUT not supplied to weight loader; total 86.80 GiB (93204889600 bytes).\nAvailable: 74.51 GiB (80000000000 bytes) from the configured allocation ceiling.\n"), None);
    assert_eq!(config::memory_refusal("fixture engine diagnostic\n"), None);
    assert_eq!(config::memory_refusal("Requested size does not fit. input=9; OUTPUT/max_tokens=4\n"), None);
}

#[test]
fn memory_floor_and_explicit_context_refusal_never_step_down() {
    let f = Fixture::new();
    f.installed();
    f.configured();
    let config = fs::read(f.root.join("config")).unwrap();
    let mut launch = f.launcher();
    // Memory cannot hold the checkpoint plus the working reserve: refuse before start.
    let model = model::discover(&f.home, Some(&f.model)).unwrap();
    launch.detector.identity.memory_available = model.bytes + 7 * GIB;
    let error = launch.execute(Command::Launch(Default::default()), false).unwrap_err();
    assert!(error.contains("Not enough memory"), "{error}");
    assert!(launch.runner.calls.is_empty());
    assert!(!f.root.join("state/run.conf").exists());
    assert_eq!(fs::read(f.root.join("config")).unwrap(), config);
    launch.detector.identity.memory_available = 100 * GIB;
    fs::write(f.model.join("config.json"), model::test_mixed_config(262144)).unwrap();
    launch.runner.refuse_above = Some(32768);
    let raised = crate::cli::parse(&["up", "--context", "65536", "--output", "16384", "--cache-bytes", "0", "--mtp", "0"].map(str::to_owned)).unwrap().command;
    let error = launch.execute(raised.clone(), false).unwrap_err();
    assert!(error.starts_with("The engine refused --context 65536 for memory:\nFAIL Requested size does not fit."), "{error}");
    assert!(error.contains("Available: 10.0 GiB"), "{error}");
    assert_eq!(launch.runner.starts, 1);
    assert!(!f.root.join("state/run.conf").exists());
    launch.runner.refuse_above = None;
    launch.execute(raised, false).unwrap();
    let saved = Config::load(&f.root).unwrap().unwrap();
    assert_eq!((saved.context, saved.output), (65536, 16384));
    let start = launch.runner.calls.iter().find(|exec| exec.label == "local start").unwrap();
    let fields = text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap();
    assert_eq!(fields["env.MTP"], "0");
    assert_eq!(fields["env.VADE_FLASHNEXT_PREFIX_CACHE"], "0");
    let mut saved_env = Fields::new();
    saved.engine_env(&mut saved_env);
    assert_eq!(saved_env["env.MTP"], "0");
    assert_eq!(saved_env["env.VADE_FLASHNEXT_PREFIX_CACHE"], "0");
    assert!(launch.runner.calls.iter().any(|exec| std::str::from_utf8(&exec.stdin).unwrap_or("").contains("--fp4-local")));
    launch.execute(Command::Doctor, false).unwrap();
    assert!(launch.execute(Command::Update, false).unwrap_err().contains("vade down"));
    launch.execute(Command::Down, false).unwrap();

}

#[path = "regression_tests.rs"]
mod regressions;

#[test]
fn unsupported_local_controls_are_refused_without_saving_or_starting() {
    for configured in [false, true] {
        for (flag, value) in [("--chats", "1"), ("--chats", "16"), ("--cache-bytes", "1")] {
            for verb in ["launch", "up"] {
                let f = Fixture::new();
                if configured { f.configured(); f.installed(); }
                let before = fs::read(f.root.join("config")).ok();
                let command = crate::cli::parse(&[verb, flag, value].map(str::to_owned)).unwrap().command;
                let mut launch = f.launcher();
                let error = launch.execute(command, false).unwrap_err();
                assert!(error.contains(flag) && !error.contains('\n'), "{error}");
                assert_eq!(fs::read(f.root.join("config")).ok(), before);
                assert!(!f.root.join("state/run.conf").exists());
                assert_eq!(launch.detector.calls, 0);
                assert!(launch.runner.calls.is_empty());
            }
        }
    }
}

#[test]
fn obsolete_saved_local_controls_allow_recovery_but_refuse_relaunch() {
    for setting in ["chats=1\n", "cache_bytes=1024\n"] {
        let f = Fixture::new();
        f.configured(); f.installed();
        let mut launch = f.launcher();
        launch.execute(Command::Up(Default::default()), false).unwrap();
        let path = f.root.join("config");
        let saved = fs::read_to_string(&path).unwrap() + setting;
        fs::write(&path, &saved).unwrap();
        assert!(Config::load(&f.root).unwrap().is_some());
        assert!(launch.execute(Command::Up(Default::default()), false).is_err());
        launch.execute(Command::Status, false).unwrap();
        launch.execute(Command::Logs, false).unwrap();
        launch.execute(Command::Down, false).unwrap();
        assert!(launch.execute(Command::Up(Default::default()), false).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), saved);
        assert_eq!(launch.runner.starts, 1);
        assert!(!f.root.join("state/run.conf").exists());
    }
}

#[test]
fn old_running_zero_cache_claim_requires_restart_before_reuse() {
    let f = Fixture::new();
    f.configured(); f.installed();
    let mut launch = f.launcher();
    let command = crate::cli::parse(&["up", "--cache-bytes", "0"].map(str::to_owned)).unwrap().command;
    launch.execute(command.clone(), false).unwrap();
    let path = f.root.join("state/run.conf");
    let mut record = read_fields(&path).unwrap().unwrap();
    assert_eq!(record.remove("prefix_cache").as_deref(), Some("0"));
    text::atomic_write(&path, &text::encode(&record)).unwrap();
    for command in [command, Command::Up(Default::default())] {
        let error = launch.execute(command, false).unwrap_err();
        assert!(error.contains("vade down") && error.contains("prefix retention"), "{error}");
    }
    let mut report = crate::status::Snapshot::new("local", false);
    launch.status_snapshot(&mut report, false).unwrap();
    // Recovery status must not turn the old unverified config into a live zero-cache claim.
    let report = crate::evals::json::parse(&report.finish(Duration::ZERO).0).unwrap();
    assert_eq!(report["settings"]["retained_cache_bytes"], crate::evals::json::Value::Null);
    launch.execute(Command::Down, false).unwrap();
    launch.execute(Command::Up(Default::default()), false).unwrap();
    assert_eq!(read_fields(&path).unwrap().unwrap()["prefix_cache"], "0");
    assert_eq!(launch.runner.starts, 2);
}

#[test]
fn busy_port_names_the_command_the_user_ran() {
    for (command, retry) in [(Command::Up(Default::default()), "vade up"), (Command::Launch(Default::default()), "vade launch")] {
        let f = Fixture::new();
        f.configured(); f.installed();
        let mut launch = f.launcher();
        launch.runner.port_busy = true;
        let error = launch.execute(command, false).unwrap_err();
        assert!(error.starts_with("Port ") && error.ends_with(&format!("is in use; close the application using it and retry {retry}.")), "{error}");
        assert_eq!(launch.runner.starts, 0);
    }
}

#[test]
fn route_a_defaults_equal_controller_defaults_and_explicit_context_wins() {
    let f = Fixture::new();
    fs::write(f.model.join("config.json"), model::test_mixed_config(262144)).unwrap();
    let discovered = model::discover(&f.home, Some(&f.model)).unwrap();
    let local = Config::new(&f.root, &discovered).unwrap();
    let Command::Up(controller) = crate::cli::parse(&["up".to_owned()]).unwrap().command else { panic!() };
    assert_eq!((local.context, local.output), (crate::cli::FP4_CONTEXT, crate::cli::fp4_output(crate::cli::FP4_CONTEXT)));
    assert_eq!((controller.context, controller.output, controller.chats), (local.context, local.output, None));
    let Command::Up(explicit) = crate::cli::parse(&["up", "--context", "262144"].map(str::to_owned)).unwrap().command else { panic!() };
    assert!(explicit.supplied_options.contains("--context"));
    assert_eq!((explicit.context, explicit.output), (262144, local.output));
}
