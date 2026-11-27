//! Loop text service (Inkling 2 Mini on 1 Spark or 2-Spark RDMA `--pair`) and resident
//! text service (GLM-5.3-Flash without arbitrary Spark caps or `--diagnostic-unqualified-text`).
//! The plan in config.json is the signal. The folder name is not.
use super::*;
use std::fs;

const LOOP: &str = crate::text_service::LOOP_CONFIG;
const LOOP_DRAFT: &str = crate::text_service::LOOP_DRAFT_CONFIG;
const OTHER: &str = r#"{"model_type":"other","architectures":["Inkling2LoopForConditionalGeneration"]}"#;

fn with_config(count: usize, folder: &str, config: Option<&str>) -> Fixture {
    let mut f = Fixture::fp4(count, false);
    let root = f.dir.join(folder);
    fs::create_dir_all(&root).unwrap();
    if let Some(config) = config {
        fs::write(root.join("config.json"), config).unwrap();
    }
    let path = fs::canonicalize(&root).unwrap();
    for machine in &mut f.app.config.machines {
        machine.model_path = path.to_string_lossy().into_owned();
    }
    f
}

fn started(f: &Fixture) -> bool {
    f.app.runner.calls.iter().any(|call| call.label.ends_with(" start") && call.label != "controller tunnel start")
}

fn depth(args: &[String], value: &str) -> bool {
    args.windows(2).any(|pair| pair[0] == "--mtp" && pair[1] == value)
}

#[test]
fn a_loop_checkpoint_starts_on_one_spark() {
    let mut f = with_config(1, "checkpoint", Some(LOOP));
    let manifest = f.app.manifest().unwrap();
    // Works cleanly WITHOUT --diagnostic-unqualified-text, and also accepts it as a backward-compatible no-op.
    let direct = Session::new(&f.app.config, &fp4_options(&["--sparks", "1"]), &manifest, "a".repeat(64)).unwrap();
    assert_eq!((direct.group_len(), direct.topology(), direct.two_sparks()), (1, "path", false));
    assert_eq!((direct.mtp, direct.mtp_explicit, direct.context, direct.output), (0, true, 524_288, 16384));
    assert!(direct.loop_service);
    let args = direct.engine_args(0);
    assert_eq!(&args[..4], &[String::from("0"), String::from("1"), String::from("127.0.0.1:0"), String::from("127.0.0.1:0")]);
    assert!(!args.iter().any(|arg| arg == "--diagnostic-unqualified-text" || arg == "--pair") && depth(&args, "0"));
    assert!(args.windows(2).any(|p| p == ["--disk-cache-dir", "~/.vade/cache/prefix"]), "{args:?}");
    assert!(!args.iter().any(|arg| arg == "--fp4-local" || arg == "--decode-fuse"));
    assert!(!std::path::Path::new(&direct.ranks[0].machine.model_path).file_name().unwrap().to_string_lossy().contains("inkling"));
    f.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap();
    let saved = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    let saved_args = saved.engine_args(0);
    assert_eq!(&saved_args[..4], &[String::from("0"), String::from("1"), String::from("127.0.0.1:0"), String::from("127.0.0.1:0")]);
    assert!(!saved_args.iter().any(|arg| arg == "--diagnostic-unqualified-text") && depth(&saved_args, "0"));
    assert!(!saved_args.iter().any(|arg| arg == "--fp4-local" || arg == "--decode-fuse"));
    assert_eq!((saved.group_len(), saved.mtp, saved.topology()), (1, 0, "path"));
    assert!(!f.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
}

#[test]
fn inkling_two_spark_pair_passes_pair_mtp_7_and_omits_disk_cache() {
    let mut f = with_config(2, "checkpoint", Some(LOOP_DRAFT));
    let Command::Up(options) = crate::cli::parse(&["up", "inkling", "--sparks", "2", "--bind", "0.0.0.0:18080"].map(str::to_owned)).unwrap().command else { panic!() };
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.up(&options)).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Launch plan:") && printed.contains("MTP 7"), "{printed}");
    assert!(printed.contains("0.0.0.0:18080"), "{printed}");
    assert!(printed.contains("omitted for 2-Spark --pair; each Spark has its own disk"), "{printed}");
    let saved = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert!(saved.loop_service && !saved.resident_text && !saved.diagnostic_unqualified_text);
    assert_eq!((saved.group_len(), saved.topology(), saved.mtp, saved.mtp_explicit, saved.context, saved.output), (2, "ring", 7, true, 524_288, 16384));
    assert_eq!(saved.bind.as_deref(), Some("0.0.0.0:18080"));
    assert!(saved.disk_cache_dir.is_none() && saved.disk_cache_bytes.is_none());
    for rank in 0..2 {
        let args = saved.engine_args(rank);
        assert_eq!((args[0].as_str(), args[1].as_str(), args[7].as_str()), (rank.to_string().as_str(), "2", "0.0.0.0:18080"));
        assert!(args.iter().any(|arg| arg == "--pair"), "{args:?}");
        assert!(depth(&args, "7"), "{args:?}");
        assert!(!args.iter().any(|arg| matches!(arg.as_str(), "--diagnostic-unqualified-text" | "--fp4-local" | "--decode-fuse" | "--disk-cache-dir" | "--disk-cache-bytes")), "{args:?}");
    }
    // Status & doctor report the 2-Spark --pair settings cleanly.
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.execute(Command::Status)).unwrap();
    let status = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(status.contains("omitted for 2-Spark --pair; each Spark has its own disk"), "{status}");
    assert!(status.contains("0.0.0.0:18080"), "{status}");
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.doctor_with(Ok(true))).unwrap();
    let doctor = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(doctor.contains("on (MTP 7)"), "{doctor}");
    // Explicit --disk-cache-dir or --disk-cache-bytes is refused with a clear explanation.
    for bad in [&["up", "inkling", "--sparks", "2", "--disk-cache-dir", "/tmp/cache"][..], &["up", "inkling", "--sparks", "2", "--disk-cache-bytes", "1048576"][..]] {
        let mut f2 = with_config(2, "checkpoint", Some(LOOP_DRAFT));
        let Command::Up(opts) = crate::cli::parse(&bad.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).unwrap().command else { panic!() };
        let err = f2.app.up(&opts).unwrap_err();
        assert!(err.contains("Two-Spark Inkling (--pair) does not support --disk-cache-dir yet"), "{err}");
        assert!(!started(&f2) && !f2.recorded());
    }
    // 2-Spark Inkling requires RDMA (refuses TCP locally before any remote call).
    let f_tcp = with_config(2, "checkpoint", Some(LOOP_DRAFT));
    let Command::Up(opts) = crate::cli::parse(&["up", "inkling", "--sparks", "2"].map(str::to_owned)).unwrap().command else { panic!() };
    let err = f_tcp.app.requested_session(&opts, None, false).unwrap_err();
    assert!(err.contains("Two-Spark Inkling (--pair) requires RDMA"), "{err}");
    // Update restart hint round-trips `vade up inkling --sparks 2 ...` cleanly.
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.update().unwrap());
    let update = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    let line = update.lines().find(|l| l.trim_start().starts_with("vade up ")).unwrap_or_else(|| panic!("{update}"));
    assert!(line.contains("vade up inkling ") && line.contains("--mtp 7") && line.contains("--bind 0.0.0.0:18080"), "{line}");
    assert!(!line.contains("--diagnostic-unqualified-text") && !line.contains("--fp4-local") && !line.contains("--disk-cache-dir"), "{line}");
}

#[test]
fn a_loop_checkpoint_above_two_sparks_is_refused_while_pairs_replicas_work() {
    // 3 and 4 Sparks remain refused for Inkling (1 Spark or 2-Spark RDMA pair only).
    for count in [3usize, 4] {
        let mut f = with_config(count, "checkpoint", Some(LOOP_DRAFT));
        let count_s = count.to_string();
        let err = f.app.up(&fp4_options(&["--sparks", &count_s])).unwrap_err();
        assert!(err.contains("Inkling 2 Mini runs on 1 Spark or a 2-Spark RDMA pair"), "{err}");
        assert!(!started(&f) && !f.recorded());
    }
    // --replicas 2 --pairs starts two independent 2-Spark RDMA Inkling pairs!
    let mut pairs = with_config(4, "checkpoint", Some(LOOP_DRAFT));
    let Command::Up(options) = crate::cli::parse(&["up", "inkling", "--replicas", "2", "--pairs"].map(str::to_owned)).unwrap().command else { panic!() };
    pairs.app.up(&options).unwrap();
    let saved = Session::load(&pairs.app.config.state_dir).unwrap().unwrap();
    assert_eq!((saved.group_len(), saved.ranks.len(), saved.replica_sparks, saved.loop_service), (2, 4, 2, true));
    for idx in 0..4 {
        let args = saved.engine_args(idx);
        assert!(args.iter().any(|arg| arg == "--pair") && depth(&args, "7"), "{idx}: {args:?}");
        assert!(!args.iter().any(|arg| arg == "--disk-cache-dir"), "{idx}: {args:?}");
    }
}

#[test]
fn loop_replicas_are_one_spark_each() {
    // Works first-class with `vade up inkling --replicas 2` (no --diagnostic-unqualified-text required)
    // and respects --bind on the gateway + --stable-prefix + NVMe disk cache on each replica!
    let mut f = with_config(2, "checkpoint", Some(LOOP_DRAFT));
    let Command::Up(options) = crate::cli::parse(&["up", "inkling", "--replicas", "2", "--bind", "0.0.0.0:18092"].map(str::to_owned)).unwrap().command else { panic!() };
    f.app.up(&options).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_eq!((session.group_len(), session.ranks.len(), session.replica_sparks, session.local_port), (1, 2, 1, 18092));
    assert_eq!(session.gateway_bind(), "0.0.0.0:18092");
    for index in 0..2 {
        let args = session.engine_args(index);
        assert_eq!(&args[..2], &[String::from("0"), String::from("1")]);
        assert_eq!((&*session.ranks[index].listen, &*session.ranks[index].next), ("127.0.0.1:0", "127.0.0.1:0"));
        assert!(depth(&args, "7") && !args.iter().any(|arg| arg == "--fp4-local" || arg == "--decode-fuse" || arg == "--diagnostic-unqualified-text" || arg == "--pair"), "{args:?}");
        assert!(args.windows(2).any(|p| p == ["--disk-cache-dir", "~/.vade/cache/prefix"]), "{args:?}");
    }
}

#[test]
fn folder_name_never_overrides_local_config_plan() {
    let mut named = with_config(2, "inkling2-loop-checkpoint", Some(OTHER));
    named.app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    assert!(!named.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
    let session = Session::load(&named.app.config.state_dir).unwrap().unwrap();
    assert!(session.resident_text && !session.loop_service);
    assert_eq!(session.mtp, crate::state::TEXT_DRAFT_DEPTH);
    assert!(depth(&session.engine_args(0), "3") && depth(&session.engine_args(1), "3"));
    assert!(!session.engine_args(0).iter().any(|arg| arg == "--pair"));
}

fn up_options(extra: &[&str]) -> UpOptions {
    let args = std::iter::once("up").chain(extra.iter().copied()).map(str::to_string).collect::<Vec<_>>();
    let Command::Up(options) = crate::cli::parse(&args).unwrap().command else { panic!() };
    options
}

#[test]
fn a_named_draft_depth_is_refused_when_the_plan_has_no_draft() {
    for depth_arg in ["1", "3", "7"] {
        let mut f = with_config(1, "checkpoint", Some(LOOP));
        let err = f.app.up(&up_options(&["--sparks", "1", "--mtp", depth_arg])).unwrap_err();
        assert!(err.contains("Drafting is not built") && err.contains("--mtp 0") && err.contains("leave --mtp out"), "{err}");
        assert!(!started(&f) && !f.recorded());
    }
    let mut off = with_config(1, "checkpoint", Some(LOOP));
    off.app.up(&up_options(&["--sparks", "1", "--mtp", "0"])).unwrap();
    assert_eq!(Session::load(&off.app.config.state_dir).unwrap().unwrap().mtp, 0);
}

#[test]
fn a_loop_checkpoint_with_declared_draft_accepts_named_mtp_up_to_its_plan_max() {
    // Bare start defaults to the checkpoint's max drafting depth (`--mtp 7`) and 512K context / 16K output.
    let mut bare = with_config(1, "checkpoint", Some(LOOP_DRAFT));
    bare.app.up(&up_options(&["--sparks", "1"])).unwrap();
    let saved = Session::load(&bare.app.config.state_dir).unwrap().unwrap();
    assert_eq!((saved.group_len(), saved.mtp, saved.mtp_explicit, saved.context, saved.output), (1, 7, true, 524_288, 16384));
    assert!(depth(&saved.engine_args(0), "7"));
    // Explicit --mtp 0..=7 passes through to flash-serve on both 1 and 2 Sparks.
    for depth_arg in ["0", "1", "3", "7"] {
        let mut f = with_config(1, "checkpoint", Some(LOOP_DRAFT));
        f.app.up(&up_options(&["--sparks", "1", "--mtp", depth_arg])).unwrap();
        let saved = Session::load(&f.app.config.state_dir).unwrap().unwrap();
        assert_eq!((saved.group_len(), saved.mtp, saved.mtp_explicit), (1, depth_arg.parse::<u32>().unwrap(), true));
        assert!(depth(&saved.engine_args(0), depth_arg), "--mtp {depth_arg}");
    }
    // Above the plan's declared block_size - 1 (gamma) is refused with the plan's own max.
    let small_gamma = LOOP_DRAFT.replace("\"block_size\": 8", "\"block_size\": 4");
    let mut over = with_config(1, "checkpoint", Some(&small_gamma));
    let err = over.app.up(&up_options(&["--sparks", "1", "--mtp", "4"])).unwrap_err();
    assert_eq!(err, "Set MTP to a whole number from 0 to 3.");
    assert!(!started(&over) && !over.recorded());
    // Two Sparks work for a loop checkpoint with declared draft (`--pair --mtp 7`).
    let mut two = with_config(2, "checkpoint", Some(LOOP_DRAFT));
    two.app.up(&up_options(&["--sparks", "2", "--mtp", "7"])).unwrap();
    let saved = Session::load(&two.app.config.state_dir).unwrap().unwrap();
    assert!(saved.loop_service && saved.group_len() == 2 && depth(&saved.engine_args(0), "7") && saved.engine_args(0).iter().any(|a| a == "--pair"));
    // Remote Spark reporting text_kind=loop enables 1-Spark and 2-Spark `--pair --mtp 7` over SSH.
    let remote_draft = "text_kind=loop\ntext_sparks=1\ntext_draft_default=7\ntext_draft_max=7\n";
    let mut one_f = Fixture::fp4(1, false);
    let mut one = take(&mut one_f, remote_draft, true);
    one.up(&up_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap();
    let saved = Session::load(&one.config.state_dir).unwrap().unwrap();
    assert_eq!((saved.group_len(), saved.mtp, saved.mtp_explicit, saved.loop_service), (1, 7, true, true));
    assert!(depth(&saved.engine_args(0), "7"));
    let mut two_f = Fixture::fp4(2, false);
    let mut two_remote = take(&mut two_f, remote_draft, true);
    two_remote.up(&up_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    let saved = Session::load(&two_remote.config.state_dir).unwrap().unwrap();
    assert_eq!((saved.group_len(), saved.mtp, saved.loop_service), (2, 7, true));
    assert!(saved.engine_args(0).iter().any(|a| a == "--pair") && depth(&saved.engine_args(0), "7"));
}

#[test]
fn mismatched_local_plans_across_sparks_do_not_start() {
    let mut f = Fixture::fp4(2, false);
    for (index, (folder, config)) in [("one", LOOP), ("two", OTHER)].into_iter().enumerate() {
        let root = f.dir.join(folder);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), config).unwrap();
        f.app.config.machines[index].model_path = fs::canonicalize(&root).unwrap().to_string_lossy().into_owned();
    }
    let err = f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err();
    assert!(err.contains("do not declare the same text service") && !started(&f) && !f.recorded(), "{err}");
}

struct Reports {
    inner: FakeRunner,
    stdout: String,
    success: bool,
}
impl Runner for Reports {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let request = if matches!(exec.program.as_str(), "ssh" | "sh") && exec.args.last().is_some_and(|arg| arg.contains("__worker")) {
            text::parse(&String::from_utf8_lossy(&exec.stdin)).unwrap_or_default()
        } else {
            Fields::new()
        };
        if request.get("op").map(String::as_str) == Some("text-service") {
            self.inner.calls.push(exec.clone());
            return Ok(Output { success: self.success, stdout: self.stdout.clone() });
        }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: std::time::Duration) { self.inner.pause(duration); }
}

fn take(f: &mut Fixture, stdout: &str, success: bool) -> App<Reports> {
    App { config: f.app.config.clone(), dry: false, runner: Reports { inner: std::mem::take(&mut f.app.runner), stdout: stdout.into(), success } }
}

#[test]
fn a_spark_report_sets_the_service_kind_and_backward_compat_workers_still_work() {
    let alone = "text_sparks=1\ntext_draft_default=0\ntext_draft_max=0\n";
    let mut one_f = Fixture::fp4(1, false);
    let mut one = take(&mut one_f, alone, true);
    one.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap();
    let saved = Session::load(&one.config.state_dir).unwrap().unwrap();
    assert_eq!((saved.group_len(), saved.mtp, saved.loop_service), (1, 0, true));
    assert!(depth(&saved.engine_args(0), "0"));
    let asked = one.runner.inner.calls.iter().find(|call| call.label.ends_with(" text service")).unwrap();
    let request = text::parse(&String::from_utf8_lossy(&asked.stdin)).unwrap();
    assert_eq!(request.get("op").map(String::as_str), Some("text-service"));
    assert!(request.get("model_path").is_some_and(|path| path.starts_with("~/")) && !request.contains_key("text_sparks"));
    // Two Sparks also probe `text-service` so remote Inkling gets `--pair` automatically.
    let mut two_f = Fixture::fp4(2, false);
    let mut two = take(&mut two_f, alone, true);
    two.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    assert!(two.runner.inner.calls.iter().any(|call| call.label.ends_with(" text service")));
    let saved = Session::load(&two.config.state_dir).unwrap().unwrap();
    assert!(saved.loop_service && saved.engine_args(0).iter().any(|a| a == "--pair"));
    // Older worker without text_sparks/text_kind falls back cleanly to Resident (GLM) when --diagnostic-unqualified-text is given.
    let mut old_f = Fixture::fp4(2, false);
    let mut old = take(&mut old_f, "phase=running\ngpu=empty\n", true);
    old.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    let saved = Session::load(&old.config.state_dir).unwrap().unwrap();
    assert!(saved.resident_text && saved.mtp == crate::state::TEXT_DRAFT_DEPTH);
    // Remote resident report starts GLM cleanly without --diagnostic-unqualified-text.
    let resident = "text_kind=resident\ntext_sparks=2\ntext_draft_default=3\ntext_draft_max=7\n";
    let mut glm_f = Fixture::fp4(2, false);
    glm_f.app.config.model = "glm-5.3-flash-fp4-local".into();
    glm_f.app.config.model_repo = "nvidia/GLM-5.3-Flash-FP4-Local".into();
    glm_f.app.runner.ready_model = Some("glm-5.3-flash-fp4-local".into());
    let mut glm = take(&mut glm_f, resident, true);
    glm.up(&fp4_options(&["--sparks", "2"])).unwrap();
    let saved = Session::load(&glm.config.state_dir).unwrap().unwrap();
    assert!(saved.resident_text && !saved.diagnostic_unqualified_text && saved.mtp == 3);
    // Error from worker stops start.
    let mut broken_f = Fixture::fp4(1, false);
    let mut broken = take(&mut broken_f, "error=Cannot read the checkpoint config. missing\n", false);
    let err = broken.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("Cannot read the checkpoint config") && err.contains("Nothing started") && !broken.runner.inner.calls.iter().any(|call| call.label.ends_with(" start")), "{err}");
}

/// The helper's hash command is recorded before the text-service question.
fn verified_before_ask(calls: &[Exec]) {
    let mut asked = 0;
    for (index, call) in calls.iter().enumerate() {
        if !call.label.ends_with(" text service") {
            continue;
        }
        asked += 1;
        let name = call.label.split_whitespace().next().unwrap();
        let hash = format!("{name} artifact hashes");
        assert!(calls[..index].iter().any(|earlier| earlier.label == hash), "{hash} is missing before {}", call.label);
    }
    assert!(asked > 0, "the checkpoint was not asked");
}

fn start_args(calls: &[Exec]) -> Vec<String> {
    let call = calls.iter().find(|call| call.label.ends_with(" start") && call.label != "controller tunnel start").unwrap();
    let fields = text::parse(&String::from_utf8_lossy(&call.stdin)).unwrap();
    (0..).map(|index| fields.get(&format!("arg.{index}")).cloned()).take_while(Option::is_some).flatten().collect()
}

fn dry_reports(f: &mut Fixture, stdout: &str) -> App<Reports> {
    App { config: f.app.config.clone(), dry: true, runner: Reports { inner: std::mem::take(&mut f.app.runner), stdout: stdout.into(), success: true } }
}

#[test]
fn the_helper_is_verified_before_it_is_asked() {
    let mut broken = Fixture::fp4(1, false);
    broken.app.runner.bad_hash = true;
    let err = broken.app.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("staged hashes differ"), "{err}");
    assert!(broken.app.runner.calls.iter().any(|call| call.label.ends_with(" artifact hashes")));
    assert!(!broken.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")) && !started(&broken));
    let alone = "text_sparks=1\ntext_draft_default=0\ntext_draft_max=0\n";
    let mut one_f = Fixture::fp4(1, false);
    let mut one = take(&mut one_f, alone, true);
    one.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"])).unwrap();
    verified_before_ask(&one.runner.inner.calls);
}

/// `SystemRunner` in dry mode returns success and empty output. That reply is not a width.
#[test]
fn a_dry_run_with_an_empty_reply_does_not_guess_the_width() {
    let mut one_f = Fixture::fp4(1, true);
    let mut one = dry_reports(&mut one_f, "");
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || one.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("does not know the text-service width") && printed.contains("not refused"), "{printed}");
    assert!(printed.contains("Start planned") && printed.contains("1 Spark"), "{printed}");
    assert!(printed.contains("The drafting depth is not known in this dry run."), "{printed}");
    assert!(!printed.contains("engine default") && !printed.contains("MTP 5"), "{printed}");
    let args = start_args(&one.runner.inner.calls);
    assert_eq!(&args[..2], &[String::from("0"), String::from("1")]);
    assert!(!args.iter().any(|arg| arg == "--fp4-local"), "{args:?}");
    assert_eq!(one.runner.inner.calls.iter().filter(|call| call.label == "a artifact hashes").count(), 1);
    verified_before_ask(&one.runner.inner.calls);
    assert!(!one.config.state_dir.join("session.conf").exists());
    // When config names GLM, dry run knows the text service is Resident and plans MTP 3 cleanly.
    let mut two_f = Fixture::fp4(2, true);
    two_f.app.config.model = "glm-5.3-flash-fp4-local".into();
    two_f.app.config.model_repo = "nvidia/GLM-5.3-Flash-FP4-Local".into();
    let mut two = dry_reports(&mut two_f, "");
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || two.up(&fp4_options(&["--sparks", "2"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(!printed.contains("does not know the text-service width"), "{printed}");
    assert!(printed.contains("2 Sparks") && printed.contains("MTP 3"), "{printed}");
    assert!(start_args(&two.runner.inner.calls).windows(2).any(|pair| pair == ["--mtp", "3"]));
    for name in ["a", "b"] {
        assert_eq!(two.runner.inner.calls.iter().filter(|call| call.label == format!("{name} artifact hashes")).count(), 1, "{name}");
    }
    let mut named_f = Fixture::fp4(1, true);
    let mut named = dry_reports(&mut named_f, "");
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || named.up(&fp4_options(&["--sparks", "1", "--diagnostic-unqualified-text", "--mtp", "5"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("The drafting depth is not known in this dry run."), "{printed}");
    assert!(!printed.contains("MTP 5"), "{printed}");
}

#[test]
fn a_dry_run_of_a_local_loop_plan_still_reads_the_file() {
    let mut one = with_config(1, "checkpoint", Some(LOOP));
    one.app.dry = true;
    one.app.up(&fp4_options(&["--sparks", "1"])).unwrap();
    assert!(!one.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
    let args = start_args(&one.app.runner.calls);
    assert_eq!(&args[..2], &[String::from("0"), String::from("1")]);
    assert!(depth(&args, "0"), "{args:?}");
    let mut two = with_config(2, "checkpoint", Some(LOOP_DRAFT));
    two.app.dry = true;
    two.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
    let args2 = start_args(&two.app.runner.calls);
    assert!(args2.iter().any(|a| a == "--pair") && depth(&args2, "7"), "{args2:?}");
    let mut four = with_config(4, "checkpoint", Some(LOOP_DRAFT));
    four.app.dry = true;
    let err = four.app.up(&fp4_options(&["--sparks", "4"])).unwrap_err();
    assert!(err.contains("Inkling 2 Mini runs on 1 Spark or a 2-Spark RDMA pair"), "{err}");
}

#[test]
fn each_helper_hash_is_checked_once() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap();
    for name in ["a", "b"] {
        let label = format!("{name} artifact hashes");
        assert_eq!(f.app.runner.calls.iter().filter(|call| call.label == label).count(), 1, "{label}");
    }
    assert!(f.app.runner.calls.iter().any(|call| call.label.ends_with(" text service")));
}

struct ByMachine { inner: FakeRunner }
impl Runner for ByMachine {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        if exec.label.ends_with(" text service") {
            self.inner.calls.push(exec.clone());
            let stdout = if exec.label.starts_with("b ") {
                "text_kind=resident\ntext_sparks=2\ntext_draft_default=3\ntext_draft_max=7\n"
            } else {
                "text_kind=loop\ntext_sparks=1\ntext_draft_default=0\ntext_draft_max=0\n"
            };
            return Ok(Output { success: true, stdout: stdout.into() });
        }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: std::time::Duration) { self.inner.pause(duration); }
}

#[test]
fn each_machine_is_asked_and_a_disagreement_does_not_start() {
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: ByMachine { inner: std::mem::take(&mut f.app.runner) } };
    let err = app.up(&fp4_options(&["--replicas", "2", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("do not declare the same text service"), "{err}");
    assert!(!app.runner.inner.calls.iter().any(|call| call.label.ends_with(" start") && call.label != "controller tunnel start"), "{err}");
    verified_before_ask(&app.runner.inner.calls);
    assert!(app.runner.inner.calls.iter().any(|call| call.label == "a text service"));
    assert!(app.runner.inner.calls.iter().any(|call| call.label == "b text service"));
}

#[test]
fn a_loop_neighbor_log_and_legacy_refusal_hints_stay_helpful() {
    let neighbor = "FAIL this text service has no transport between Sparks: RANK 1 of WORLD 2 has 1 neighbor links; start it as RANK 0 of WORLD 1\n";
    let hint = health::text_refusal_hint(neighbor).unwrap();
    assert!(hint.contains("vade up inkling --sparks 1") && hint.contains("vade up inkling --sparks 2"), "{hint}");
    let absent = "text service refused: this engine has not qualified this model family; it starts only as an unqualified diagnostic service on a loopback bind: give --diagnostic-unqualified-text\n";
    let hint = health::text_refusal_hint(absent).unwrap();
    assert!(hint.contains("This older engine build requires --diagnostic-unqualified-text for GLM."), "{hint}");
    assert!(health::text_refusal_hint("flashnext ready world=2").is_none());
}

#[test]
fn admitted_chats_reads_loop_driver_and_loop_driver_pair_ready_lines() {
    let line = "text_service qualification=qualified ready bind=127.0.0.1:9 rank=0 world=1 path=loop_driver backend=device context=32768 output=8192 resident_bytes=4 session_bytes=5 session_pool_bytes=6 max_chats=1 drafting=off cache=false";
    assert_eq!(crate::state::admitted_chats(&format!("LOOP TEXT ADMISSION\nLOOP TEXT LOAD\n{line}\n")), Some(1));
    assert_eq!(crate::state::admitted_chats(&line.replace("max_chats=1", "max_chats=4")), Some(4));
    let pair_line = "text_service qualification=qualified ready bind=0.0.0.0:18080 rank=0 world=2 path=loop_driver_pair backend=device context=524288 output=16384 resident_bytes=4 session_bytes=5 session_pool_bytes=6 max_chats=32 drafting=on cache=true";
    assert_eq!(crate::state::admitted_chats(pair_line), Some(32));
    assert_eq!(crate::state::admitted_chats("LOOP TEXT ADMISSION\nLOOP TEXT LOAD\n"), None);
    assert_eq!(crate::state::admitted_chats(&line.replace(" max_chats=1", "")), None);
    assert_eq!(crate::state::admitted_chats(&line.replace("max_chats=1", "max_chats=0")), None);
    assert_eq!(crate::state::admitted_chats(&line.replace("max_chats=1", "max_chats=89")), Some(89));
    let flash = "flashnext ready world=1 model=m context=32768 concurrency=rows active=8 connections=16";
    assert_eq!(crate::state::admitted_chats(&format!("{line}\n{flash}\n")), Some(8));
    assert_eq!(crate::state::admitted_chats(&format!("{flash}\n{line}\n")), Some(1));
    assert_eq!(crate::state::admitted_chats("flash_serve ready slots=900"), None);
}

#[test]
fn admitted_chats_leaves_the_two_spark_text_ready_line_unread_as_main_does() {
    let line = "text_service qualification=qualified ready bind=127.0.0.1:9 rank=0 world=2 path=tensor_split context=131072 resident_bytes=1 session_bytes=2 shared_workspace_bytes=3 session_pool_bytes=4 max_chats=8 default_draft=alone default_depth=3 predictor_depth=3 cache=true kept_cap_bytes=1 largest_kept_prompt_tokens=2";
    assert_eq!(crate::state::admitted_chats(line), None);
    assert_eq!(crate::state::admitted_chats(&format!("load\n{line}\n")), None);
}
