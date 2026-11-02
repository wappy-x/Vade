//! Two FP4-Local Sparks run the tensor split, the only two-Spark path for FP4-Local: vade up sets
//! no level, carries no path switch and refuses TCP before any remote call; FP8 pairs keep their
//! ring and may use TCP. Status names the path and why for every group of Sparks on one model.
use super::*;

/// The worker start requests of one `vade up --sparks N`.
fn starts(f: &Fixture) -> Vec<Fields> {
    f.app.runner.calls.iter().filter(|c| c.label.ends_with(" start") && c.label != "controller tunnel start")
        .map(|start| text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap()).collect()
}

#[test]
fn up_with_two_sparks_runs_the_split_over_rdma_only() {
    const CASE: &str = "VADE_TEST_TWO_SPARK_SPLIT";
    if let Ok(case) = std::env::var(CASE) {
        let mut f = Fixture::fp4(2, false);
        if case == "tcp" {
            // The refusal is local: no SSH, fabric, artifact, path or GPU check, no ownership record.
            for options in [fp4_options(&["--sparks", "2"]), fp4_options(&["--sparks", "2", "--mtp", "0"])] {
                let error = f.app.up(&options).unwrap_err();
                assert!(error.contains("needs RDMA") && error.contains("FP4-Local"), "{error}");
                assert!(f.app.runner.calls.is_empty(), "{:?}", f.app.runner.calls.iter().map(|c| &c.label).collect::<Vec<_>>());
                assert!(f.app.runner.ports.is_empty() && f.app.runner.started.is_empty() && !f.recorded());
            }
            // FP8 keeps its ring on two Sparks, and TCP serves it.
            let mut fp8 = Fixture::new(2, false);
            fp8.app.up(&fp8_options()).unwrap();
            let requests = starts(&fp8);
            assert_eq!(requests.len(), 2);
            assert!(requests.iter().all(|request| request["env.SEQPAR_RDMA"] == "0"), "{requests:?}");
            return;
        }
        f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        let requests = starts(&f);
        assert_eq!(requests.len(), 2);
        for request in &requests {
            assert!(request.keys().all(|key| !key.contains("VADE_W2")), "{request:?}");
            assert_eq!(request["env.SEQPAR_RDMA"], "1");
        }
        // Four Sparks carry no path switch either: the engine chooses the split in parts over RDMA
        // or the chain.
        let mut chain = Fixture::fp4(4, false);
        chain.app.up(&fp4_options(&["--sparks", "4", "--mtp", "0"])).unwrap();
        assert!(starts(&chain).iter().all(|request| request.keys().all(|key| !key.contains("VADE_W2"))));
        return;
    }
    for (case, rdma) in [("rdma", None), ("rdma", Some("1")), ("tcp", Some("0"))] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "app::tests::two_spark::up_with_two_sparks_runs_the_split_over_rdma_only", "--nocapture"]).env(CASE, case);
        match rdma { Some(value) => command.env("SEQPAR_RDMA", value), None => command.env_remove("SEQPAR_RDMA") };
        let result = command.output().unwrap();
        assert!(result.status.success(), "{case}: {}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"), "the child ran the case");
    }
}

const SPLIT: &str = r#","vade":{"numerics":{"world_size":2,"topology":"pipeline","path":"tensor_split","path_reason":"FP4-Local on two Sparks","tensor_split":{"experts":"output_columns","projection_level":3,"wire":"write","head":"vocabulary_halves"}}}"#;
const CHAIN: &str = r#","vade":{"numerics":{"world_size":4,"topology":"pipeline","path":"layer_pipeline","path_reason":"FP4-Local on three or more Sparks without the tensor split (TCP, or a world the split does not take)"}}"#;
const RING: &str = r#","vade":{"numerics":{"world_size":2,"topology":"ring","path":"ring","path_reason":"FP8: the tensor split does not serve FP8"}}"#;

/// `vade status` text and `vade status --json` of a running launch.
fn status(f: &mut Fixture) -> (String, crate::evals::json::Value) {
    let capture = crate::output::testing::Capture::default();
    let printed = capture.bytes.clone();
    crate::output::with_writer(capture, || f.app.execute(Command::Status)).unwrap();
    let printed = String::from_utf8(printed.borrow().clone()).unwrap();
    let mut report = crate::status::Snapshot::new("configured", false);
    f.app.status_snapshot(&mut report).unwrap();
    (printed, crate::evals::json::parse(&report.finish(Duration::ZERO).0).unwrap())
}

#[test]
fn status_names_every_multi_spark_path_and_its_reason() {
    use crate::evals::json::{self, Value};
    let split_levels = "projection level 3, write wire, head split by vocabulary halves";
    let cases: [(&str, fn() -> Fixture, UpOptions, &str, String, &str); 4] = [
        ("split", || Fixture::fp4(2, false), fp4_options(&["--sparks", "2"]), SPLIT,
            format!("tensor split (FP4-Local on two Sparks; {split_levels})"),
            r#"{"path":"tensor_split","path_reason":"FP4-Local on two Sparks","projection_level":3.0,"wire":"write","head":"vocabulary_halves"}"#),
        ("chain", || Fixture::fp4(4, false), fp4_options(&["--sparks", "4"]), CHAIN,
            "layer chain (FP4-Local on three or more Sparks without the tensor split (TCP, or a world the split does not take))".into(),
            r#"{"path":"layer_pipeline","path_reason":"FP4-Local on three or more Sparks without the tensor split (TCP, or a world the split does not take)","projection_level":null,"wire":null,"head":null}"#),
        ("FP8 ring", || Fixture::new(2, false), fp8_options(), RING,
            "ring (FP8: the tensor split does not serve FP8)".into(),
            r#"{"path":"ring","path_reason":"FP8: the tensor split does not serve FP8","projection_level":null,"wire":null,"head":null}"#),
        ("older engine", || Fixture::fp4(2, false), fp4_options(&["--sparks", "2"]), "", "not reported by this engine".into(), "null"),
    ];
    for (name, fixture, options, extra, row, expected) in cases {
        let mut f = fixture();
        f.app.runner.health_extra = extra.into();
        f.app.up(&options).unwrap();
        let (printed, value) = status(&mut f);
        assert!(printed.contains(&format!("\n  Multi-Spark path       {row}\n")), "{name}: {printed}");
        assert_eq!(value["ready"], Value::Bool(true), "{name}");
        assert_eq!(value["multi_spark_path"], json::parse(expected).unwrap(), "{name}");
    }
    // The split without a reason (an older receipt), and the rank path, which only --tensor-map selects.
    let split = health::MultiSparkPath::from_health(&format!("{{\"status\":\"ready\"{}}}", SPLIT.replace(r#""path_reason":"FP4-Local on two Sparks","#, ""))).unwrap();
    assert_eq!(split.label(), format!("tensor split ({split_levels})"));
    // The split over more ranks: its own reason, projection level 2 over the relay, the head in parts.
    let parts = health::MultiSparkPath::from_health(r#"{"vade":{"numerics":{"path":"tensor_split","path_reason":"FP4-Local on more than two Sparks whose widths cut into whole parts","tensor_split":{"experts":"output_columns","long_prompts":"chain_tiles_then_state_gather","projection_level":2,"wire":"relay","head":"vocabulary_parts"}}}}"#).unwrap();
    assert_eq!(parts.label(), "tensor split (FP4-Local on more than two Sparks whose widths cut into whole parts; projection level 2, relay wire, head split by vocabulary parts)");
    let rank = health::MultiSparkPath::from_health(r#"{"vade":{"numerics":{"path":"rank_path","path_reason":"FP4-Local with --tensor-map"}}}"#).unwrap();
    assert_eq!(rank.label(), "rank path (FP4-Local with --tensor-map)");
    assert_eq!(health::MultiSparkPath::from_health(r#"{"status":"ready","vade":{"numerics":{"topology":"local"}}}"#), None);
    // One Spark and independent replicas serve no model together: no path row or field.
    for (count, options) in [(1, fp4_options(&[])), (2, fp4_options(&["--replicas", "2"]))] {
        let mut f = Fixture::fp4(count, false);
        f.app.runner.health_extra = SPLIT.into();
        f.app.up(&options).unwrap();
        let (printed, value) = status(&mut f);
        assert!(!printed.contains("Multi-Spark path"), "{printed}");
        assert_eq!(value.get("multi_spark_path"), None, "{value:?}");
        f.app.down().unwrap();
    }
}

/// `FakeRunner`, with one first start that the engine refuses for memory.
struct RefusesOnce { inner: FakeRunner, refused: bool }
impl Runner for RefusesOnce {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let request = text::parse(&String::from_utf8_lossy(&exec.stdin)).unwrap_or_default();
        let op = request.get("op").map(String::as_str);
        if op == Some("start") && !self.refused { self.inner.exit_on_health = Some("a".into()); }
        if op == Some("logs") && !self.refused {
            self.refused = true;
            let log = "Requested size does not fit.\n  Asked: CONTEXT=131072 OUTPUT=8192\n  Available: 1 GiB\n";
            return Ok(Output { success: true, stdout: text::encode(&Fields::from([("log".into(), log.into())])) });
        }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}

fn saved_context(f: &Fixture) -> u32 { Session::load(&f.app.config.state_dir).unwrap().unwrap().context }

#[test]
fn a_group_of_sparks_defaults_to_the_long_context_and_one_spark_does_not() {
    let context = |sparks: usize, options: &[&str]| {
        let mut f = Fixture::fp4(sparks, false);
        f.app.up(&fp4_options(options)).unwrap();
        saved_context(&f)
    };
    assert_eq!(context(1, &[]), 32768);
    assert_eq!(context(2, &["--sparks", "2"]), 131072);
    assert_eq!(context(4, &["--sparks", "4"]), 131072);
    assert_eq!(context(2, &["--hosts", "a,b"]), 131072);
    // A plain up after setup picks one Spark, and each plain replica is one Spark.
    assert_eq!(context(2, &[]), 32768);
    assert_eq!(context(2, &["--replicas", "2"]), 32768);
    // Each pair is two Sparks.
    assert_eq!(context(4, &["--replicas", "2", "--pairs"]), 131072);
    // A context the person names stays, on any count.
    assert_eq!(context(2, &["--sparks", "2", "--context", "32768"]), 32768);
    assert_eq!(context(2, &["--sparks", "2", "--context", "8192"]), 8192);
}

#[test]
fn a_running_engine_keeps_its_context_and_a_restart_keeps_it_too() {
    let mut f = Fixture::fp4(2, false);
    f.app.up(&fp4_options(&["--sparks", "2", "--context", "32768"])).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    let recorded = crate::app::resident::recorded_options(&session);
    // A bare up, and the same up with no --context, attach with the saved size.
    let Command::Up(bare) = crate::cli::parse(&["up".to_string()]).unwrap().command else { panic!() };
    f.app.up(&bare).unwrap();
    f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
    assert_eq!(saved_context(&f), 32768);
    // vade update restarts from the saved options.
    f.app.down().unwrap();
    f.app.up(&recorded).unwrap();
    assert_eq!(saved_context(&f), 32768);
}

#[test]
fn a_long_default_that_does_not_fit_falls_to_the_short_one_and_says_so() {
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesOnce { inner: std::mem::take(&mut f.app.runner), refused: false } };
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || app.up(&fp4_options(&["--sparks", "2"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Context 131072 does not fit in memory on these Sparks. Starting with context 32768.\n"), "{printed}");
    assert!(printed.contains("context 32768 tokens"), "the ready line names the chosen size: {printed}");
    assert_eq!(Session::load(&app.config.state_dir).unwrap().unwrap().context, 32768);
    // A size that the person named stays their choice: the refusal is the answer.
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesOnce { inner: std::mem::take(&mut f.app.runner), refused: false } };
    let error = app.up(&fp4_options(&["--sparks", "2", "--context", "131072"])).unwrap_err();
    assert!(error.contains("Requested size does not fit."), "{error}");
    assert!(Session::load(&app.config.state_dir).unwrap().is_none());
}

/// The manifest of an older vade: one rank kept the folder of an earlier install.
#[test]
fn a_rank_in_an_older_install_folder_makes_up_install_again_and_start() {
    let mut f = Fixture::fp4(2, false);
    let path = f.app.config.build_manifest.clone();
    let mut manifest = Manifest::load(&path).unwrap();
    manifest.artifacts.get_mut("b").unwrap().ptx = "~/stage/older/flash.ptx".into();
    text::atomic_write(&path, &manifest.encode()).unwrap();
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.up(&fp4_options(&["--sparks", "2"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("These Sparks hold Vade in different install folders, so vade up installs Vade on each Spark again."), "{printed}");
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_eq!(session.ranks[0].artifact.ptx, session.ranks[1].artifact.ptx);
    assert!(session.ranks[0].artifact.ptx.contains("/builds/"), "{}", session.ranks[0].artifact.ptx);
    assert_eq!(f.app.runner.started, ["a".to_owned(), "b".to_owned()].into());
}

/// A hand-written config can name two stage folders. No install fixes that, so the refusal stays.
#[test]
fn two_configured_stage_folders_keep_the_refusal_and_install_nothing() {
    let mut f = Fixture::fp4(2, false);
    f.app.config.machines[1].stage_dir = "~/other-stage".into();
    let path = f.app.config.build_manifest.clone();
    let mut manifest = Manifest::load(&path).unwrap();
    manifest.artifacts.get_mut("b").unwrap().ptx = "~/other-stage/flash.ptx".into();
    text::atomic_write(&path, &manifest.encode()).unwrap();
    let error = f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err();
    assert!(error.contains("expanded absolute PTX paths differ"), "{error}");
    assert!(!f.app.runner.calls.iter().any(|exec| exec.label == "harness build"), "no install");
    assert!(!f.recorded());
}

/// A start that loses SSH shows ssh's own line and its fix, not a guess.
#[test]
fn the_check_before_a_start_shows_the_real_ssh_line() {
    let mut f = Fixture::fp4(2, false);
    f.app.runner.silent_probe = true;
    f.app.runner.ssh_says = Some("you@a-ssh: Permission denied (publickey).\n");
    let error = f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err();
    assert!(error.starts_with("ssh a-ssh failed: you@a-ssh: Permission denied (publickey). Vade never edits ~/.ssh. Run ssh-copy-id a-ssh, then run vade doctor."), "{error}");
    assert!(!error.contains("did not reply"), "{error}");
    assert!(!f.recorded());
}

#[test]
fn the_ready_line_names_each_spark_and_the_context() {
    let mut f = Fixture::fp4(2, false);
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.up(&fp4_options(&["--sparks", "2"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Vade is ready on 2 Sparks (a-ssh, b-ssh) at http://localhost:18080: context 131072 tokens, drafting depth "), "{printed}");
}
