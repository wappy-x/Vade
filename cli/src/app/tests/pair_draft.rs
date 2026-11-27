//! The GLM text service (`--diagnostic-unqualified-text`) loads predictor depth 3
//! when the user did not pass `--mtp`. A normal FP4 start, including Qwen pairs, does not.
use super::*;

const GLM_MODEL: &str = "glm-5.3-flash-fp4-local";
const GLM_REPO: &str = "nvidia/GLM-5.3-Flash-FP4-Local";

fn passes(args: &[String], depth: &str) -> bool {
    args.windows(2).any(|pair| pair[0] == "--mtp" && pair[1] == depth)
}

/// The published GLM checkpoint, not the Qwen fixture this module's normal-pair test keeps.
fn glm_pair(dry: bool) -> Fixture {
    let mut f = Fixture::fp4(2, dry);
    f.app.config.model = GLM_MODEL.into();
    f.app.config.model_repo = GLM_REPO.into();
    f.app.config.model_revision = "76348ddb2be2e35e39ecfc00e75dc07703f4f3d5".into();
    f.app.runner.ready_model = Some(GLM_MODEL.into());
    f
}

fn assert_glm(config: &crate::config::Config, session: &Session) {
    assert_eq!((session.model.as_str(), config.model_repo.as_str()), (GLM_MODEL, GLM_REPO));
    assert!(crate::card::is_glm_model(&session.model));
    assert!(!session.model.contains("qwen") && !config.model_repo.contains("Qwen"));
}

#[test]
fn glm_text_service_loads_draft_depth_three_unless_mtp_is_set() {
    let f = glm_pair(false);
    let manifest = f.app.manifest().unwrap();
    // Works first-class WITHOUT --diagnostic-unqualified-text AND accepts it as a backward-compatible no-op.
    for extra in [&["--sparks", "2"][..], &["--sparks", "2", "--diagnostic-unqualified-text"][..]] {
        let open = Session::new(&f.app.config, &fp4_options(extra), &manifest, "a".repeat(64)).unwrap();
        assert_glm(&f.app.config, &open);
        assert!(open.resident_text && !open.diagnostic_unqualified_text && open.mtp_explicit && open.mtp == crate::state::TEXT_DRAFT_DEPTH);
        assert_eq!((open.mtp_label(), open.drafting_plan()), ("3".into(), "MTP 3".into()));
        assert!(passes(&open.engine_args(0), "3") && passes(&open.engine_args(1), "3"));
        assert!(!open.engine_args(0).iter().any(|arg| arg == "--diagnostic-unqualified-text" || arg == "--fp4-local" || arg == "--decode-fuse"));
    }
    for depth in ["0", "1", "2"] {
        let set = Session::new(&f.app.config, &fp4_options(&["--sparks", "2", "--mtp", depth]), &manifest, "b".repeat(64)).unwrap();
        assert!(set.resident_text && set.mtp_explicit && set.mtp == depth.parse::<u32>().unwrap() && passes(&set.engine_args(0), depth), "{depth}");
    }
    assert_eq!(Session::new(&f.app.config, &fp4_options(&["--sparks", "2", "--mtp", "0"]), &manifest, "c".repeat(64)).unwrap().mtp_label(), "0 (off, --mtp 0)");
    // LAN --bind works cleanly for GLM without loopback refusal or --diagnostic-unqualified-text.
    let lan = Session::new(&f.app.config, &fp4_options(&["--sparks", "2", "--bind", "0.0.0.0:18080"]), &manifest, "d".repeat(64)).unwrap();
    assert_eq!(lan.bind.as_deref(), Some("0.0.0.0:18080"));
    assert_eq!(lan.engine_args(0)[7], "0.0.0.0:18080");
    assert_eq!(lan.remote_bind_host(), "127.0.0.1");
}

#[test]
fn qwen_fp4_pair_and_pairs_keep_mains_record() {
    let two = Fixture::fp4(2, false);
    let session = two.app.requested_session(&fp4_options(&["--sparks", "2"]), None, true).unwrap();
    assert_eq!(session.model, "qwen3.8-flash-next-fp4-local");
    assert!(!crate::card::is_glm_model(&session.model));
    assert_eq!((session.mtp, session.mtp_explicit), (2, false));
    assert!(!session.engine_args(0).iter().any(|arg| arg == "--mtp") && !session.engine_args(1).iter().any(|arg| arg == "--mtp"));
    assert_eq!(session.mtp_label(), "engine default");
    let again = two.app.requested_session(&crate::cli::UpOptions::default(), Some(&session), true).unwrap();
    assert!(super::super::resident::same_launch(&session, &again) && again.mtp == 2 && !again.mtp_explicit);
    assert_eq!(again.engine_args(0), session.engine_args(0));
    let pairs = Fixture::fp4(4, false);
    let first = pairs.app.requested_session(&fp4_options(&["--replicas", "2", "--pairs"]), None, true).unwrap();
    assert_eq!((first.mtp, first.mtp_explicit), (2, false));
    assert!(!first.engine_args(0).iter().any(|arg| arg == "--mtp"));
    let second = pairs.app.requested_session(&fp4_options(&["--replicas", "2", "--pairs"]), Some(&first), true).unwrap();
    assert!(super::super::resident::same_launch(&first, &second) && second.mtp == 2 && !second.mtp_explicit);
}

#[test]
fn an_old_glm_text_session_stays_off_and_does_not_attach_as_depth_three() {
    let mut f = glm_pair(false);
    let old = f.app.requested_session(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text", "--mtp", "0"]), None, true).unwrap();
    assert_glm(&f.app.config, &old);
    assert_eq!((old.mtp, old.mtp_explicit, old.mtp_label()), (0, true, "0 (off, --mtp 0)".into()));
    assert!(passes(&old.engine_args(0), "0"));
    old.save(&f.app.config.state_dir).unwrap();
    f.app.dry = true;
    let mut report = crate::status::Snapshot::new("configured", true);
    f.app.status_snapshot(&mut report).unwrap();
    let value = crate::evals::json::parse(&report.finish(std::time::Duration::ZERO).0).unwrap();
    assert_eq!(value["model"].as_str(), Some(GLM_MODEL));
    assert_eq!(value["settings"]["mtp"], crate::evals::json::Value::from(0usize));
    assert_eq!(value["settings"]["mtp_source"].as_str(), Some("explicit"));
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.doctor_with(Ok(true))).unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(printed.contains("off (MTP 0)") && !printed.contains("on (MTP 3)"), "{printed}");
    f.app.dry = false;
    let err = f.app.up(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"])).unwrap_err();
    assert!(err.contains("default drafting depth") && err.contains("vade down") && err.contains("no predictor"), "{err}");
    let fresh = f.app.requested_session(&fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"]), Some(&old), true).unwrap();
    assert_eq!(fresh.mtp, 3);
    assert!(passes(&fresh.engine_args(0), "3"));
    assert!(!super::super::resident::same_launch(&old, &fresh));
}

#[test]
fn glm_text_service_default_start_and_second_up_show_depth_three() {
    let mut f = glm_pair(false);
    let options = fp4_options(&["--sparks", "2", "--diagnostic-unqualified-text"]);
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.up(&options)).unwrap();
    let started = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(started.contains("Launch plan:") && started.contains(GLM_MODEL), "{started}");
    assert!(started.contains("Drafting request") && started.contains("MTP 3"), "{started}");
    assert!(started.contains("Vade is ready on 2 Sparks (a, b) at http://localhost:18080: context 131072 tokens, drafting depth 3.\n"), "{started}");
    let saved = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert_glm(&f.app.config, &saved);
    assert_eq!((saved.mtp, saved.mtp_explicit), (3, true));
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.execute(Command::Status)).unwrap();
    let status = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    let mtp_line = status.lines().find(|line| line.contains("MTP")).unwrap_or("");
    assert!(mtp_line.contains('3') && !mtp_line.contains("engine default") && !mtp_line.contains("off"), "{status}");
    let mut report = crate::status::Snapshot::new("configured", false);
    f.app.status_snapshot(&mut report).unwrap();
    let value = crate::evals::json::parse(&report.finish(std::time::Duration::ZERO).0).unwrap();
    assert_eq!(value["model"].as_str(), Some(GLM_MODEL));
    assert_eq!(value["settings"]["mtp"], crate::evals::json::Value::from(3usize));
    assert_eq!(value["settings"]["mtp_source"].as_str(), Some("explicit"));
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.doctor_with(Ok(true))).unwrap();
    let doctor = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(doctor.contains("on (MTP 3)") && !doctor.contains("engine default"), "{doctor}");
    let starts = f.app.runner.calls.iter().filter(|call| call.label.ends_with(" start") && call.label != "controller tunnel start").count();
    let capture = crate::output::testing::Capture::default();
    crate::output::with_writer(capture.clone(), || f.app.up(&options)).unwrap();
    let again = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(again.contains("already running"), "{again}");
    assert_eq!(f.app.runner.calls.iter().filter(|call| call.label.ends_with(" start") && call.label != "controller tunnel start").count(), starts);
    assert_eq!(Session::load(&f.app.config.state_dir).unwrap().unwrap().mtp, 3);
}

#[test]
fn update_restart_command_round_trips_the_glm_text_service() {
    for extra in [vec!["--sparks", "2"], vec!["--sparks", "2", "--mtp", "0"], vec!["--sparks", "2", "--bind", "0.0.0.0:18080"]] {
        let mut f = glm_pair(false);
        let options = fp4_options(&extra);
        f.app.up(&options).unwrap();
        let saved = Session::load(&f.app.config.state_dir).unwrap().unwrap();
        let config = f.app.config.clone();
        let manifest = f.app.manifest().unwrap();
        let capture = crate::output::testing::Capture::default();
        crate::output::with_writer(capture.clone(), || f.app.update().unwrap());
        let output = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        let command = output.lines().find_map(|line| line.trim().strip_prefix("vade up ")).unwrap_or("");
        // GLM is first-class: restart command carries the depth and --bind, with no --diagnostic-unqualified-text or --fp4-local.
        let depth = if extra.contains(&"--mtp") { 0 } else { crate::state::TEXT_DRAFT_DEPTH };
        assert_eq!(saved.mtp, depth, "{command}");
        assert!(!command.contains("--diagnostic-unqualified-text") && !command.contains("--fp4-local"), "{command}");
        assert_eq!(command.matches(&format!(" --mtp {depth}")).count(), 1, "{command}");
        if extra.contains(&"--bind") {
            assert!(command.contains(" --bind 0.0.0.0:18080"), "{command}");
        }
        let args = std::iter::once("up").chain(command.split_whitespace()).map(str::to_string).collect::<Vec<_>>();
        let Command::Up(restart) = crate::cli::parse(&args).unwrap().command else { panic!("not an up command: {command}") };
        assert!(!restart.diagnostic_unqualified_text && restart.mtp_explicit && restart.mtp == saved.mtp, "{command}");
        let again = Session::new(&config, &restart, &manifest, saved.token.clone()).unwrap();
        assert_glm(&config, &again);
        assert_eq!((again.resident_text, again.diagnostic_unqualified_text, again.mtp, again.mtp_explicit, again.context, again.output, again.bind.as_deref()),
            (true, false, saved.mtp, saved.mtp_explicit, saved.context, saved.output, saved.bind.as_deref()));
        assert!(!again.engine_args(0).iter().any(|arg| arg == "--diagnostic-unqualified-text"));
        assert!(passes(&again.engine_args(0), &saved.mtp.to_string()));
        assert_eq!(again.ranks.iter().map(|rank| rank.machine.name.as_str()).collect::<Vec<_>>(), saved.ranks.iter().map(|rank| rank.machine.name.as_str()).collect::<Vec<_>>());
    }
}

#[test]
fn glm_resident_text_has_no_arbitrary_spark_cap_and_preserves_path_endpoints() {
    // Rule 17: GLM resident text service accepts 1, 2, or 4 Sparks without an arbitrary 2-Spark cap.
    for count in [1usize, 2, 4] {
        let mut f = Fixture::fp4(count, false);
        f.app.config.model = GLM_MODEL.into();
        f.app.config.model_repo = GLM_REPO.into();
        f.app.runner.ready_model = Some(GLM_MODEL.into());
        let count_s = count.to_string();
        f.app.up(&fp4_options(&["--sparks", &count_s])).unwrap();
        let saved = Session::load(&f.app.config.state_dir).unwrap().unwrap();
        assert!(saved.resident_text && !saved.diagnostic_unqualified_text);
        assert_eq!(saved.group_len(), count);
        if count == 4 {
            assert_eq!(saved.topology(), "path");
            // Check raw Session::new fabric endpoints before FakeRunner resolves every hostname to 127.0.0.1.
            let manifest = f.app.manifest().unwrap();
            let raw = Session::new(&f.app.config, &fp4_options(&["--sparks", "4"]), &manifest, "a".repeat(64)).unwrap();
            assert!(!raw.ranks[0].listen.starts_with("127.0.0.1:0"), "{}", raw.ranks[0].listen);
            assert!(!raw.ranks[3].next.starts_with("127.0.0.1:0"), "{}", raw.ranks[3].next);
            assert_ne!(raw.ranks[3].next, raw.ranks[0].listen, "tail points back to rank 2, not rank 0");
        }
    }
}
