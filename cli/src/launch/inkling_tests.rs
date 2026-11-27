use super::*;
use crate::runner::Output;

const GIB: u64 = 1024 * 1024 * 1024;

fn spark_identity() -> detect::Identity {
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

struct InklingDetector;
impl Detector for InklingDetector {
    fn detect(&mut self, _: &Path, _: &Path) -> Result<detect::Identity, String> {
        Ok(spark_identity())
    }
}

struct InklingRunner {
    start_requests: Vec<Fields>,
    chats: usize,
    terminal: bool,
}

impl InklingRunner {
    fn new() -> Self {
        Self { start_requests: Vec::new(), chats: 0, terminal: true }
    }
    fn last_args(&self) -> Vec<String> {
        let req = self.start_requests.last().expect("start request");
        (0..).map(|i| req.get(&format!("arg.{i}")).cloned()).take_while(Option::is_some).flatten().collect()
    }
}

impl Runner for InklingRunner {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        if exec.args == ["__worker"] {
            let req = text::parse(std::str::from_utf8(&exec.stdin).unwrap()).unwrap();
            let out = match req["op"].as_str() {
                "warm-ptx" => "ptx_cache=warm\n",
                "probe" => "phase=absent\n",
                "start" => {
                    self.start_requests.push(req);
                    "alive=true\nphase=running\n"
                }
                "inspect" => "alive=true\nphase=running\n",
                "listener" => "listener=owned\n",
                "stop" => "alive=false\nphase=stopped\n",
                "clean" => "phase=absent\n",
                "logs" => "log=inkling ready\n",
                other => panic!("unexpected op {other}"),
            };
            return Ok(Output { success: true, stdout: out.into() });
        }
        if exec.label == "local health" {
            return Ok(Output { success: true, stdout: r#"{"status":"ready","model":"Inkling-2-Mini"}"#.into() });
        }
        if exec.label == "local events" {
            return Ok(Output { success: true, stdout: r#"{"counts":{},"recent":[]}"#.into() });
        }
        panic!("unexpected exec {} {:?}", exec.program, exec.args);
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> {
        assert_eq!(exec.args[0], "agent");
        self.chats += 1;
        Ok(true)
    }
    fn terminal(&self) -> bool { self.terminal }
    fn port_free(&mut self, _: u16) -> Result<(), String> { Ok(()) }
    fn pause(&mut self, _: Duration) {}
}

struct InklingFixture {
    base: PathBuf,
    home: PathBuf,
    root: PathBuf,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl InklingFixture {
    fn new() -> Self {
        let guard = crate::runner::process_test_lock();
        let base = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("vade-inkling-kiss-{}", text::random_hex(8).unwrap()));
        let home = base.join("home");
        let root = home.join(".vade");
        fs::create_dir_all(&home).unwrap();
        let bundle = base.join("bundle");
        fs::create_dir_all(&bundle).unwrap();
        let mut elf = [0u8; 64];
        elf[..6].copy_from_slice(b"\x7fELF\x02\x01");
        elf[18] = 183;
        fs::write(bundle.join("flash-serve"), elf).unwrap();
        fs::write(bundle.join("vade-harness"), elf).unwrap();
        fs::write(bundle.join("flash.ptx"), ".version 8.0\n").unwrap();
        install::publish_test_bundle(&root, &bundle).unwrap();
        Self { base, home, root, _guard: guard }
    }

    fn write_inkling(&self, path: &Path) {
        model::test_write_unserved_checkpoint(path, None);
        // Real Inkling 2 Mini uses tiktoken/tokenizer.model and has model_max_length: 1048576.
        let cfg = fs::read_to_string(path.join("config.json")).unwrap();
        let cfg = cfg.replace("\"model_max_length\": 4096", "\"model_max_length\": 1048576");
        fs::write(path.join("config.json"), cfg).unwrap();
        fs::create_dir_all(path.join("tiktoken")).unwrap();
        fs::write(path.join("tiktoken/tokenizer.model"), "aGVsbG8= 0\n").unwrap();
    }

    fn launcher(&self) -> Launcher<InklingRunner, InklingDetector> {
        Launcher {
            root: self.root.clone(),
            home: self.home.clone(),
            runner: InklingRunner::new(),
            detector: InklingDetector,
        }
    }
}

impl Drop for InklingFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

fn parse_cmd(args: &str) -> Command {
    crate::cli::parse(&args.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
        .unwrap()
        .command
}

#[test]
fn vade_up_inkling_defaults_to_512k_16k_mtp_7_and_prompt_cache_on() {
    let f = InklingFixture::new();
    let ckpt = f.home.join("models/thinkingmachines/Inkling-2-Mini");
    f.write_inkling(&ckpt);
    let mut launcher = f.launcher();
    let capture = output::testing::Capture::default();
    output::with_writer(capture.clone(), || launcher.execute(parse_cmd("up inkling"), false)).unwrap();

    let args = launcher.runner.last_args();
    let expected_cache = f.root.join("cache/prefix").to_string_lossy().into_owned();
    assert_eq!(args[6], "524288", "default Inkling context should be 512K: {args:?}");
    assert!(args.windows(2).any(|p| p == ["--mtp", "7"]), "default Inkling MTP should be 7: {args:?}");
    assert!(!args.iter().any(|a| a == "--diagnostic-unqualified-text"), "Inkling is first-class qualified: {args:?}");
    assert!(args.windows(2).any(|p| p == ["--disk-cache-dir", expected_cache.as_str()]), "default NVMe disk prefix cache should be wired: {args:?}");
    assert!(!args.iter().any(|a| a == "--no-prompt-keep"), "prompt cache should be ON by default: {args:?}");
    assert_eq!(launcher.runner.start_requests[0]["env.FLASH_SERVE_OUTPUT_TOKENS"], "16384");

    // Seed a mock .vdpc entry and verify status + doctor report entries & bytes.
    fs::write(f.root.join("cache/prefix/sample.vdpc"), vec![0u8; 4096]).unwrap();

    // Status and doctor both succeed cleanly on tiktoken Inkling.
    let status_cap = output::testing::Capture::default();
    output::with_writer(status_cap.clone(), || launcher.execute(Command::Status, false)).unwrap();
    let status_text = String::from_utf8(status_cap.bytes.borrow().clone()).unwrap();
    assert!(status_text.contains("524288 tokens, output 16384"), "{status_text}");
    assert!(status_text.contains("MTP 7 (default)"), "{status_text}");
    assert!(status_text.contains("Disk cache") && status_text.contains("1 entry, 4.0 KiB"), "{status_text}");

    let doctor_cap = output::testing::Capture::default();
    output::with_writer(doctor_cap.clone(), || launcher.execute(Command::Doctor, false)).unwrap();
    let doctor_text = String::from_utf8(doctor_cap.bytes.borrow().clone()).unwrap();
    assert!(doctor_text.contains("tiktoken/tokenizer.model is present"), "{doctor_text}");
    assert!(doctor_text.contains("default context") && doctor_text.contains("524288 tokens"), "{doctor_text}");
    assert!(doctor_text.contains("default output") && doctor_text.contains("16384 tokens"), "{doctor_text}");
    assert!(doctor_text.contains("on (MTP 7)"), "{doctor_text}");
    assert!(doctor_text.contains("disk prefix cache") && doctor_text.contains("1 entry, 4.0 KiB"), "{doctor_text}");
    assert!(doctor_text.contains("PASS"), "{doctor_text}");

    launcher.execute(Command::Down, false).unwrap();
}

#[test]
fn vade_up_inkling_supports_bind_custom_disk_cache_and_no_disk_cache() {
    let f = InklingFixture::new();
    let ckpt = f.home.join("models/thinkingmachines/Inkling-2-Mini");
    f.write_inkling(&ckpt);
    let custom_dir = f.base.join("nvme-prefix");
    let mut launcher = f.launcher();

    launcher.execute(
        parse_cmd(&format!("up inkling --bind 0.0.0.0:18095 --disk-cache-dir {} --disk-cache-bytes 2147483648", custom_dir.display())),
        false,
    ).unwrap();
    let args = launcher.runner.last_args();
    assert_eq!(args[7], "0.0.0.0:18095", "explicit --bind must pass straight to flash-serve: {args:?}");
    assert!(args.windows(2).any(|p| p == ["--disk-cache-dir", &custom_dir.to_string_lossy()]), "{args:?}");
    assert!(args.windows(2).any(|p| p == ["--disk-cache-bytes", "2147483648"]), "{args:?}");
    assert!(custom_dir.is_dir(), "disk cache directory should be created automatically");
    launcher.execute(Command::Down, false).unwrap();

    launcher.execute(parse_cmd("up inkling --no-disk-cache"), false).unwrap();
    let args = launcher.runner.last_args();
    assert!(!args.iter().any(|a| a == "--disk-cache-dir" || a == "--disk-cache-bytes"), "{args:?}");
    launcher.execute(Command::Down, false).unwrap();
}

#[test]
fn vade_up_inkling_supports_explicit_mtp_0_cache_off_and_1m_context() {
    let f = InklingFixture::new();
    let ckpt = f.home.join("models/thinkingmachines/Inkling-2-Mini");
    f.write_inkling(&ckpt);
    let mut launcher = f.launcher();

    launcher.execute(parse_cmd("up inkling --context 32768 --mtp 0 --cache-bytes 0"), false).unwrap();
    let args = launcher.runner.last_args();
    assert_eq!(args[6], "32768");
    assert!(args.windows(2).any(|p| p == ["--mtp", "0"]), "{args:?}");
    assert!(args.iter().any(|a| a == "--no-prompt-keep"), "{args:?}");
    launcher.execute(Command::Down, false).unwrap();

    launcher.execute(parse_cmd("up inkling --context 1048576 --output 32768"), false).unwrap();
    let args = launcher.runner.last_args();
    assert_eq!(args[6], "1048576");
    assert_eq!(launcher.runner.start_requests.last().unwrap()["env.FLASH_SERVE_OUTPUT_TOKENS"], "32768");
    launcher.execute(Command::Down, false).unwrap();
}

#[test]
fn vade_launch_inkling_starts_engine_and_opens_chat() {
    let f = InklingFixture::new();
    let ckpt = f.home.join("models/thinkingmachines/Inkling-2-Mini");
    f.write_inkling(&ckpt);
    let mut launcher = f.launcher();

    launcher.execute(parse_cmd("launch inkling"), false).unwrap();
    assert_eq!(launcher.runner.chats, 1);
    let args = launcher.runner.last_args();
    assert_eq!(args[6], "524288");
    assert!(args.windows(2).any(|p| p == ["--mtp", "7"]), "{args:?}");
    launcher.execute(Command::Down, false).unwrap();
}

#[test]
fn vade_up_inkling_discovers_hf_cache_readonly_without_writing_into_it() {
    let f = InklingFixture::new();
    let hf_base = f.home.join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini");
    let blobs = hf_base.join("blobs");
    let rev = crate::pull::alias::INKLING_REV;
    let snap = hf_base.join("snapshots").join(rev);
    let raw = f.base.join("raw-inkling");
    f.write_inkling(&raw);
    fs::create_dir_all(&blobs).unwrap();
    fs::create_dir_all(snap.join("tiktoken")).unwrap();
    // Simulate standard HF Hub cache layout: real files in blobs/, symlinks in snapshots/<rev>/.
    for name in [
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
        "mtp.safetensors",
        "tiktoken/tokenizer.model",
    ] {
        let blob_name = name.replace('/', "_");
        fs::rename(raw.join(name), blobs.join(&blob_name)).unwrap();
        let target = if name.contains('/') {
            PathBuf::from("../../../blobs").join(&blob_name)
        } else {
            PathBuf::from("../../blobs").join(&blob_name)
        };
        std::os::unix::fs::symlink(target, snap.join(name)).unwrap();
    }

    let snap_entries_before: Vec<_> = fs::read_dir(&snap).unwrap().map(|e| e.unwrap().file_name()).collect();
    let mut launcher = f.launcher();
    launcher.execute(parse_cmd("up inkling"), false).unwrap();

    // HF cache snapshot was not modified at all.
    let snap_entries_after: Vec<_> = fs::read_dir(&snap).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(snap_entries_before, snap_entries_after);
    // Engine receives the materialized hardlink view under VADE_HOME/hf-cache/.
    let args = launcher.runner.last_args();
    assert!(args[5].ends_with("thinkingmachines/Inkling-2-Mini") && args[5].contains("hf-cache/"), "{args:?}");
    assert_eq!(args[6], "524288");
    assert!(args.windows(2).any(|p| p == ["--mtp", "7"]), "{args:?}");
    launcher.execute(Command::Down, false).unwrap();
}
