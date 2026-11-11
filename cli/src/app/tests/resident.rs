//! Resident reuse rejects changed configuration and preserves owned state.
use super::*;

fn assert_refused_without_touching_resident(f: &mut Fixture, options: &UpOptions, change: &str) {
    let saved = fs::read(f.app.config.state_dir.join("session.conf")).unwrap();
    for command in [
        Command::Up(options.clone()),
        Command::Up(UpOptions::default()),
        Command::Chat(Default::default()),
    ] {
        f.app.runner.calls.clear();
        let error = f.app.execute(command).unwrap_err();
        assert!(error.contains("different launch"), "{change}: {error}");
        assert!(
            f.app.runner.calls.is_empty(),
            "{change}: mismatched launch performed remote work"
        );
        assert_eq!(
            fs::read(f.app.config.state_dir.join("session.conf")).unwrap(),
            saved, "{change}: ownership record changed"
        );
        assert_eq!(f.app.runner.started.len(), 2, "{change}: resident server count changed");
    }
}

#[test]
#[allow(clippy::type_complexity)] // test table of configuration mutators.
fn resident_reuse_compares_current_configuration_on_every_rank() {
    let changes: &[(&str, fn(&mut Config))] = &[
        ("model", |c| c.model = "another-model".into()),
        ("model repository", |c| {
            c.model_repo = "another/repository".into()
        }),
        ("model revision", |c| c.model_revision = "d".repeat(40)),
        ("first model path", |c| {
            c.machines[0].model_path = "~/models/another".into()
        }),
        ("last model path", |c| {
            c.machines[1].model_path = "~/models/another".into()
        }),
        ("first state directory", |c| {
            c.machines[0].state_dir = "~/another-state".into()
        }),
        ("last state directory", |c| {
            c.machines[1].state_dir = "~/another-state".into()
        }),
        ("machine target", |c| {
            c.machines[1].ssh = "another-machine".into()
        }),
        ("machine stage directory", |c| {
            c.machines[1].stage_dir = "~/another-stage".into()
        }),
        ("first fabric address", |c| {
            c.machines[0].fabric.insert("b".into(), "192.0.2.10".into());
        }),
        ("last fabric address", |c| {
            c.machines[1].fabric.insert("a".into(), "192.0.2.11".into());
        }),
        ("fabric port", |c| c.tp_port += 1),
        ("local port", |c| c.local_port += 1),
        ("serve port", |c| c.serve_port += 1),
    ];
    let options = fp4_options(&["--sparks", "2"]);
    for (name, change) in changes {
        let mut f = Fixture::fp4(2, false);
        f.app.up(&options).unwrap();
        change(&mut f.app.config);
        assert_refused_without_touching_resident(&mut f, &options, name);
    }
}

#[test]
#[allow(clippy::type_complexity)] // test table of manifest mutators.
fn resident_reuse_checks_the_current_build_and_all_artifact_identities() {
    let changes: &[(&str, fn(&mut Manifest))] = &[
        ("engine source", |m| m.engine_sha = "d".repeat(40)),
        ("CLI source", |m| m.cli_sha = "e".repeat(40)),
        ("engine bytes", |m| {
            for a in m.artifacts.values_mut() {
                a.binary_sha256 = "6".repeat(64);
            }
        }),
        ("PTX bytes", |m| {
            for a in m.artifacts.values_mut() {
                a.ptx_sha256 = "7".repeat(64);
            }
        }),
        ("helper bytes", |m| {
            m.artifacts.get_mut("b").unwrap().worker_sha256 = "8".repeat(64)
        }),
        ("harness bytes", |m| m.harness_sha256 = Some("9".repeat(64))),
        ("engine path", |m| {
            m.artifacts.get_mut("a").unwrap().binary = "~/new/flash-serve".into()
        }),
        ("PTX path", |m| {
            m.artifacts.get_mut("b").unwrap().ptx = "~/new/flash.ptx".into()
        }),
        ("helper path", |m| {
            m.artifacts.get_mut("b").unwrap().worker = "~/new/vade".into()
        }),
    ];
    let options = fp4_options(&["--sparks", "2"]);
    for (name, change) in changes {
        let mut f = Fixture::fp4(2, false);
        f.app.up(&options).unwrap();
        let mut manifest = f.app.manifest().unwrap();
        change(&mut manifest);
        text::atomic_write(&f.app.config.build_manifest, &manifest.encode()).unwrap();
        assert_refused_without_touching_resident(&mut f, &options, name);
    }
}

#[test]
fn activating_a_new_install_refuses_the_old_resident_even_for_the_same_source() {
    let mut f = Fixture::fp4(2, false);
    let options = fp4_options(&["--sparks", "2"]);
    f.app.execute(Command::Install).unwrap();
    f.app.execute(Command::Up(options.clone())).unwrap();
    let previous = f.app.config.build_manifest.clone();
    f.app.execute(Command::Install).unwrap();
    assert_ne!(f.app.config.build_manifest, previous);
    assert_refused_without_touching_resident(&mut f, &options, "new installation");
}

#[test]
fn copied_ownership_record_cannot_reuse_a_different_local_state_directory() {
    let mut f = Fixture::fp4(2, false);
    let options = fp4_options(&["--sparks", "2"]);
    f.app.up(&options).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    f.app.config.state_dir = f.dir.join("other-state");
    session.save(&f.app.config.state_dir).unwrap();
    assert_refused_without_touching_resident(&mut f, &options, "copied state directory");
}

#[test]
fn unchanged_resident_preserves_nondefault_options_and_expanded_paths() {
    let mut f = Fixture::fp4(2, false);
    let options = fp4_options(&[
        "--hosts",
        "b,a",
        "--context",
        "8192",
        "--output",
        "4096",
        "--mtp",
        "0",
        "--chats",
        "4",
        "--memory-bytes",
        "128000000000",
        "--cache-bytes",
        "4000000000",
    ]);
    f.app.up(&options).unwrap();
    let saved = fs::read(f.app.config.state_dir.join("session.conf")).unwrap();
    let session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert!(session.ranks[0].machine.model_path.starts_with("/srv/"));
    assert_ne!(
        session.ranks[0].machine.model_path,
        f.app.config.machines[1].model_path
    );
    assert!(
        session.ranks[0]
            .listen
            .parse::<std::net::SocketAddr>()
            .is_ok()
    );
    f.app.runner.calls.clear();
    for command in [
        Command::Up(UpOptions::default()),
        Command::Up(options),
        Command::Chat(Default::default()),
    ] {
        f.app.execute(command).unwrap();
    }
    assert_eq!(
        fs::read(f.app.config.state_dir.join("session.conf")).unwrap(),
        saved
    );
    assert!(
        !f.app
            .runner
            .calls
            .iter()
            .any(|e| e.label.ends_with(" start")
                || e.label.ends_with(" stop")
                || e.label.ends_with(" paths")
                || e.label == "fabric address resolution")
    );
}

#[test]
fn legacy_ownership_records_refuse_reuse_but_still_allow_owned_cleanup() {
    let mut f = Fixture::fp4(2, false);
    let options = fp4_options(&["--sparks", "2"]);
    f.app.up(&options).unwrap();
    let mut session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    session.launch_fingerprint = None;
    session.save(&f.app.config.state_dir).unwrap();
    assert_refused_without_touching_resident(&mut f, &options, "legacy ownership record");
    f.app.down().unwrap();
    assert!(!f.recorded() && f.app.runner.started.is_empty());
}

#[test]
fn changed_transport_refuses_reuse_after_loading_the_saved_record() {
    const CASE: &str = "VADE_TEST_RESIDENT_TRANSPORT";
    if let Ok(previous) = std::env::var(CASE) {
        // An FP8 pair: two FP4-Local Sparks refuse TCP before any launch (tests/two_spark.rs).
        let mut f = Fixture::new(2, false);
        let options = fp8_options();
        // Construct the launch under the other transport without mutating the
        // test process environment shared by parallel tests.
        let mut session = f
            .app
            .requested_session(&options, None, previous == "1")
            .unwrap();
        f.app.resolve_fabric(&mut session).unwrap();
        f.app.resolve_rank_paths(&mut session).unwrap();
        session.phase = "running".into();
        for rank in &mut session.ranks {
            rank.attempted = true;
            f.app.runner.started.insert(rank.machine.name.clone());
        }
        session.save(&f.app.config.state_dir).unwrap();
        assert_refused_without_touching_resident(&mut f, &options, "changed transport");
        return;
    }
    for (previous, current) in [("0", "1"), ("1", "0")] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "app::tests::resident_tests::changed_transport_refuses_reuse_after_loading_the_saved_record", "--nocapture"])
            .env(CASE, previous).env("SEQPAR_RDMA", current).output().unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}


fn isolated_parser_case(name: &str, mtp: Option<&str>) -> bool {
    const CASE: &str = "VADE_TEST_RESIDENT_PARSER";
    if std::env::var(CASE).as_deref() == Ok(name) {
        return true;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.args(["--exact", &format!("app::tests::resident_tests::{name}"), "--nocapture"])
        .env(CASE, name).env_remove("MTP");
    if let Some(value) = mtp { command.env("MTP", value); }
    let output = command.output().unwrap();
    assert!(output.status.success(), "{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    false
}

fn parsed_up(args: &[&str]) -> Command {
    crate::cli::parse(&["up"].into_iter().chain(args.iter().copied()).map(str::to_owned).collect::<Vec<_>>()).unwrap().command
}

#[test]
fn parsed_explicit_default_context_never_reuses_a_different_resident() {
    if !isolated_parser_case("parsed_explicit_default_context_never_reuses_a_different_resident", None) { return; }
    let mut f = Fixture::fp4(1, false);
    f.app.execute(parsed_up(&["--context", "8192"])).unwrap();
    let saved = fs::read(f.app.config.state_dir.join("session.conf")).unwrap();
    f.app.runner.calls.clear();
    let error = f.app.execute(parsed_up(&["--context", "262144"])).unwrap_err();
    assert!(error.contains("different launch"), "{error}");
    assert!(f.app.runner.calls.is_empty());
    assert_eq!(fs::read(f.app.config.state_dir.join("session.conf")).unwrap(), saved);
    f.app.execute(parsed_up(&[])).unwrap();
    assert_eq!(fs::read(f.app.config.state_dir.join("session.conf")).unwrap(), saved);
}

#[test]
fn parsed_bare_up_with_environment_mtp_reuses_recorded_fp4_launch() {
    if !isolated_parser_case("parsed_bare_up_with_environment_mtp_reuses_recorded_fp4_launch", Some("0")) { return; }
    let mut f = Fixture::fp4(1, false);
    f.app.execute(parsed_up(&["--fp4-local"])).unwrap();
    let saved = fs::read(f.app.config.state_dir.join("session.conf")).unwrap();
    f.app.runner.calls.clear();
    f.app.execute(parsed_up(&[])).unwrap();
    assert_eq!(fs::read(f.app.config.state_dir.join("session.conf")).unwrap(), saved);
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" start") || e.label.ends_with(" stop")));
    f.app.execute(Command::Down).unwrap(); f.app.execute(parsed_up(&["--mtp", "3"])).unwrap();
    f.app.runner.calls.clear();
    assert!(f.app.execute(parsed_up(&[])).unwrap_err().contains("different launch"));
    assert!(f.app.runner.calls.is_empty());
}

#[test]
fn decode_fuse_rides_the_controller_args_on_one_spark_and_stops_early_on_more() {
    // The controller path passes decode_fuse to the engine and keeps it across a reload.
    let f = Fixture::fp4(1, false);
    let fused = Session::new(&f.app.config, &fp4_options(&["--sparks", "1"]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
    assert!(fused.engine_args(0).ends_with(&["--decode-fuse".to_string(), "on".to_string()]));
    fused.save(&f.app.config.state_dir).unwrap();
    assert!(Session::load(&f.app.config.state_dir).unwrap().unwrap().decode_fuse);
    let plain = Session::new(&f.app.config, &fp4_options(&["--sparks", "1", "--decode-fuse", "off"]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
    assert!(!plain.decode_fuse);
    assert!(plain.engine_args(0).ends_with(&["--decode-fuse".to_string(), "off".to_string()]));
    plain.save(&f.app.config.state_dir).unwrap();
    assert!(!Session::load(&f.app.config.state_dir).unwrap().unwrap().decode_fuse);
    for count in [2, 4] {
        let f = Fixture::fp4(count, false);
        let plain = Session::new(&f.app.config, &fp4_options(&["--sparks", &count.to_string()]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
        assert!(!plain.decode_fuse);
        for index in 0..count {
            assert!(!plain.engine_args(index).iter().any(|arg| arg == "--decode-fuse"));
        }
    }
    // More Sparks refuse before any remote start.
    let mut f = Fixture::fp4(2, false);
    let mut options = fp4_options(&["--sparks", "2"]);
    options.decode_fuse = Some(true);
    let error = f.app.up(&options).unwrap_err();
    assert_eq!(error, "--decode-fuse on runs on one FP4-Local Spark. Leave it off for more Sparks or FP8.");
    assert!(!f.app.runner.calls.iter().any(|call| call.label.ends_with(" start")));
}
