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
    #[allow(clippy::type_complexity)] // test table of two-Spark fixture cases.
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
    let parts = health::MultiSparkPath::from_health(r#"{"vade":{"numerics":{"path":"tensor_split","path_reason":"FP4-Local on more than two Sparks whose widths cut into whole parts","tensor_split":{"experts":"output_columns","long_prompts":"sequence_split_or_chain_tiles","projection_level":2,"wire":"relay","head":"vocabulary_parts"}}}}"#).unwrap();
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

/// `FakeRunner`, with an engine that refuses a text checkpoint on every normal start.
struct RefusesText { inner: FakeRunner }
const TEXT_REFUSAL_LOG: &str = "text service refused: unsupported execution: complete resident decoder is absent.; source precision and zero-sign policy require owner approval\n";
impl Runner for RefusesText {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let request = text::parse(&String::from_utf8_lossy(&exec.stdin)).unwrap_or_default();
        let op = request.get("op").map(String::as_str);
        if op == Some("start") { self.inner.exit_on_health = Some("a".into()); }
        if op == Some("logs") { return Ok(Output { success: true, stdout: text::encode(&Fields::from([("log".into(), TEXT_REFUSAL_LOG.into())])) }); }
        self.inner.run(exec)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.inner.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.inner.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.inner.pause(duration); }
}

/// A normal start of a text checkpoint stops with the engine's line and one plain hint: why, and the one way to start.
#[test]
fn a_normal_start_that_meets_the_text_refusal_says_how_to_start() {
    let mut f = Fixture::fp4(2, false);
    let mut app = App { config: f.app.config.clone(), dry: false, runner: RefusesText { inner: std::mem::take(&mut f.app.runner) } };
    let error = app.up(&fp4_options(&["--sparks", "2", "--context", "32768"])).unwrap_err();
    assert!(error.contains("Last engine log lines:\ntext service refused:"), "{error}");
    assert!(error.ends_with("\nVade has not qualified this text checkpoint, so a normal start refuses it.\nIt starts only as unqualified diagnostics on two Sparks: vade up --sparks 2 --diagnostic-unqualified-text"), "{error}");
    assert!(Session::load(&app.config.state_dir).unwrap().is_none());
    // The hint names an option that the command line takes, and the words that the engine prints.
    let Command::Up(options) = crate::cli::parse(&["up", "--sparks", "2", "--diagnostic-unqualified-text"].map(str::to_owned)).unwrap().command else { panic!() };
    assert!(options.diagnostic_unqualified_text);
    assert!(include_str!("../../../../engine/crates/tp/src/bin/flash-serve/text.rs").contains(&format!("format!(\"{} {{e}};", health::TEXT_REFUSAL)));
    // Any other stop keeps its own words and gets no hint.
    for log in ["", "Requested size does not fit.\n  Asked: CONTEXT=131072 OUTPUT=8192\n  Available: 1 GiB\n", "flashnext ready world=2"] { assert_eq!(health::text_refusal_hint(log), None, "{log}"); }
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
    assert!(error.starts_with("SSH checks failed. Vade never edits ~/.ssh."), "{error}");
    assert!(error.contains("ssh a-ssh failed: you@a-ssh: Permission denied (publickey)."), "{error}");
    assert_eq!(error.matches("Read docs/INSTALL.md for the SSH check command.").count(), 2, "{error}");
    assert!(!error.contains("did not reply"), "{error}");
    assert!(!f.recorded());
}

#[test]
fn the_ready_line_names_each_spark_and_the_context() {
    let mut f = Fixture::fp4(2, false);
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.up(&fp4_options(&["--sparks", "2"]))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Vade is ready on 2 Sparks (a, b) at http://localhost:18080: context 131072 tokens, drafting depth "), "{printed}");
}

/// Destinations stay private, including after a config edit.
#[test]
fn up_status_and_doctor_hide_address_destinations_and_old_machine_names() {
    let mut f = Fixture::fp4(2, false);
    let hosts = ["user@192.0.2.10", "user@192.0.2.11"];
    for (machine, host) in f.app.config.machines.iter_mut().zip(hosts) { machine.ssh = host.into(); }
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.execute(Command::Up(fp4_options(&["--sparks", "2"])))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("Machines") && printed.contains("Vade is ready on 2 Sparks (a, b)"), "{printed}");
    for prefix in ["192.0.2.", "192-0-2-", "user@"] { assert!(!printed.contains(prefix), "{printed}"); }
    let mut session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    for (index, rank) in session.ranks.iter_mut().enumerate() { rank.machine.name = format!("192-0-2-{}", index + 10); }
    session.save(&f.app.config.state_dir).unwrap();
    // The current config no longer has these destinations. The old record still does.
    for machine in &mut f.app.config.machines { machine.ssh = format!("{}-ssh", machine.name); }
    for command in [Command::Status, Command::Doctor] {
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || f.app.execute(command));
        let printed = format!("{}{}", String::from_utf8(capture.bytes.borrow().clone()).unwrap(), result.err().unwrap_or_default());
        assert!(printed.contains("spark-1") && printed.contains("spark-2"), "{printed}");
        for prefix in ["192.0.2.", "192-0-2-", "user@"] { assert!(!printed.contains(prefix), "{printed}"); }
    }
    let mut report = crate::status::Snapshot::new("configured", false);
    f.app.status_snapshot(&mut report).unwrap();
    let printed = report.finish(Duration::ZERO).0;
    for prefix in ["192.0.2.", "192-0-2-", "user@"] { assert!(!printed.contains(prefix), "{printed}"); }
    crate::evals::json::parse(&printed).unwrap();
}

#[test]
fn diagnostic_text_opt_in_persists_but_bare_up_does_not_reuse_it() {
    let f=Fixture::fp4(2,false);
    let options=fp4_options(&["--context","4096","--output","64","--diagnostic-unqualified-text"]);
    let diagnostic=f.app.requested_session(&options,None,true).unwrap();
    assert!(diagnostic.diagnostic_unqualified_text);
    for at in 0..diagnostic.ranks.len() {
        let args=diagnostic.engine_args(at);
        assert_eq!(args.iter().filter(|s|s.as_str()=="--diagnostic-unqualified-text").count(),1);
        assert!(!args.iter().any(|s|s=="--fp4-local" || s=="--decode-fuse"));
        let depth=args.iter().position(|s|s=="--mtp").unwrap();assert_eq!(args[depth+1],"0");
    }
    diagnostic.save(&f.app.config.state_dir).unwrap();
    let saved=Session::load(&f.app.config.state_dir).unwrap().unwrap();assert!(saved.diagnostic_unqualified_text);
    assert!(f.app.requested_session(&UpOptions::default(),Some(&saved),true).unwrap_err().contains("--diagnostic-unqualified-text"));
    let plain=f.app.requested_session(&UpOptions::default(),None,true).unwrap();
    assert!(!plain.diagnostic_unqualified_text);assert!(!resident::same_launch(&saved,&plain));
    assert!(!plain.encode().contains("diagnostic_unqualified_text"));
}

#[test]
fn second_rail_follows_configured_fabric_aux_and_tp_dual_rail_opt_out() {
    const CASE: &str = "VADE_TEST_TWO_SPARK_AUX";
    if let Ok(case) = std::env::var(CASE) {
        let lo = std::net::Ipv4Addr::LOCALHOST;
        if case == "invalid" {
            let mut f = Fixture::fp4(2, false);
            f.app.config.machines[0].fabric_aux.insert("b".into(), "aux-a-to-b".into());
            f.app.config.machines[1].fabric_aux.insert("a".into(), "aux-b-to-a".into());
            let err = f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err();
            assert!(err.contains("TP_DUAL_RAIL"), "{err}");
            assert!(f.app.runner.calls.is_empty());
            return;
        }
        if case == "optout" {
            let mut f = Fixture::fp4(2, false);
            f.app.config.machines[0].fabric_aux.insert("b".into(), "aux-a-to-b".into());
            f.app.config.machines[1].fabric_aux.insert("a".into(), "aux-b-to-a".into());
            f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
            for req in starts(&f) {
                assert_eq!(req.get("env_unset.TP_AUX_LISTEN").map(String::as_str), Some("1"));
                assert_eq!(req.get("env_unset.TP_AUX_NEXT").map(String::as_str), Some("1"));
                assert!(!req.contains_key("env.TP_AUX_LISTEN") && !req.contains_key("env.TP_AUX_NEXT"));
            }
            return;
        }
        // Unconfigured two-Spark pair leaves TP_AUX_* unset exactly as on main.
        let mut unconf = Fixture::fp4(2, false);
        unconf.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        for req in starts(&unconf) {
            assert_eq!(req.get("env_unset.TP_AUX_LISTEN").map(String::as_str), Some("1"));
            assert_eq!(req.get("env_unset.TP_AUX_NEXT").map(String::as_str), Some("1"));
            assert!(!req.contains_key("env.TP_AUX_LISTEN") && !req.contains_key("env.TP_AUX_NEXT"));
        }
        assert!(!Session::load(&unconf.app.config.state_dir).unwrap().unwrap().encode().contains("aux_listen"));

        // Configured fabric_aux at both ends sets TP_AUX_LISTEN and TP_AUX_NEXT on tp_port + 1.
        let mut conf = Fixture::fp4(2, false);
        let aux_port = conf.app.config.tp_port + 1;
        conf.app.config.machines[0].fabric_aux.insert("b".into(), "aux-a-to-b".into());
        conf.app.config.machines[1].fabric_aux.insert("a".into(), "aux-b-to-a".into());
        conf.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        let reqs = starts(&conf);
        assert_eq!(reqs.len(), 2);
        for req in &reqs {
            assert!(!req.contains_key("env_unset.TP_AUX_LISTEN") && !req.contains_key("env_unset.TP_AUX_NEXT"));
            assert_eq!(req.get("env.TP_AUX_LISTEN").map(String::as_str), Some(format!("{lo}:{aux_port}").as_str()));
            assert_eq!(req.get("env.TP_AUX_NEXT").map(String::as_str), Some(format!("{lo}:{aux_port}").as_str()));
        }
        // Changing fabric_aux refuses resident reuse without touching the running pair.
        conf.app.config.machines[0].fabric_aux.insert("b".into(), "another-aux-a".into());
        conf.app.runner.calls.clear();
        assert!(conf.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err().contains("different launch"));
        assert!(conf.app.runner.calls.is_empty());

        // One-sided fabric_aux is refused locally before any SSH call.
        let mut one_sided = Fixture::fp4(2, false);
        one_sided.app.config.machines[0].fabric_aux.insert("b".into(), "aux-only-a".into());
        assert!(one_sided.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err().contains("fabric_aux"));
        assert!(one_sided.app.runner.calls.is_empty());

        // Four-Spark FP4-Local chain keeps TP_AUX_* unset even when fabric_aux is present.
        let mut four = Fixture::fp4(4, false);
        for pair in [(0, 1, "a", "b"), (1, 2, "b", "c"), (2, 3, "c", "d")] {
            four.app.config.machines[pair.0].fabric_aux.insert(pair.3.into(), format!("aux-{}-{}", pair.2, pair.3));
            four.app.config.machines[pair.1].fabric_aux.insert(pair.2.into(), format!("aux-{}-{}", pair.3, pair.2));
        }
        four.app.up(&fp4_options(&["--sparks", "4", "--mtp", "0"])).unwrap();
        for req in starts(&four) {
            assert_eq!(req.get("env_unset.TP_AUX_LISTEN").map(String::as_str), Some("1"));
            assert!(!req.contains_key("env.TP_AUX_LISTEN"));
        }
        return;
    }
    for (case, dual) in [("default", None), ("default", Some("1")), ("optout", Some("0")), ("invalid", Some("maybe"))] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "app::tests::two_spark::second_rail_follows_configured_fabric_aux_and_tp_dual_rail_opt_out", "--nocapture"]).env(CASE, case);
        match dual { Some(v) => command.env("TP_DUAL_RAIL", v), None => command.env_remove("TP_DUAL_RAIL") };
        command.env_remove("SEQPAR_RDMA");
        let result = command.output().unwrap();
        assert!(result.status.success(), "{case}: {}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
    }
}
