//! Two Sparks run the tensor split by default: vade up sets no level itself, carries this
//! shell's debug switch and temporary level names to both ranks, and status names the path.
use super::*;

/// The worker start requests of one `vade up --sparks N`.
fn starts(f: &Fixture) -> Vec<Fields> {
    f.app.runner.calls.iter().filter(|c| c.label.ends_with(" start") && c.label != "controller tunnel start")
        .map(|start| text::parse(std::str::from_utf8(&start.stdin).unwrap()).unwrap()).collect()
}

#[test]
fn switches_name_every_value_and_the_pipeline_has_one_way_in() {
    use std::env::VarError::{NotPresent, NotUnicode};
    let env = |pairs: &'static [(&'static str, &'static str)]| move |name: &str| pairs.iter().find(|(n, _)| *n == name).map(|(_, v)| v.to_string()).ok_or(NotPresent);
    let unset: Vec<(&str, Option<String>)> = plan::TWO_SPARK_SWITCHES.iter().map(|&name| (name, None)).collect();
    assert_eq!(plan::two_spark_switches(env(&[]), true), Ok(unset.clone()));
    let mut path = unset.clone(); path[0].1 = Some("pipeline".into());
    assert_eq!(plan::two_spark_switches(env(&[("VADE_W2_PATH", "pipeline")]), true), Ok(path.clone()));
    // TCP runs the pipeline only when it is named.
    assert_eq!(plan::two_spark_switches(env(&[("VADE_W2_PATH", "pipeline")]), false), Ok(path));
    assert!(plan::two_spark_switches(env(&[]), false).unwrap_err().contains("needs RDMA"));
    let mut levels = unset.clone(); levels[1].1 = Some("1".into()); levels[3].1 = Some("2".into());
    assert_eq!(plan::two_spark_switches(env(&[("VADE_W2_TP_PROJ", "2"), ("VADE_W2_TP", "1"), ("VADE_W2_PP", "1")]), true), Ok(levels));
    // The old spelling of the pipeline, and a split forced against it, stop vade up.
    assert!(plan::two_spark_switches(env(&[("VADE_W2_TP", "0")]), true).unwrap_err().contains("VADE_W2_PATH=pipeline"));
    assert!(plan::two_spark_switches(env(&[("VADE_W2_TP", "0"), ("VADE_W2_PATH", "pipeline")]), true).is_err());
    assert!(plan::two_spark_switches(env(&[("VADE_W2_TP", "1"), ("VADE_W2_PATH", "pipeline")]), true).unwrap_err().contains("disagree"));
    for path in ["", "split", "Pipeline"] {
        let read = move |name: &str| if name == "VADE_W2_PATH" { Ok(path.to_string()) } else { Err(NotPresent) };
        assert!(plan::two_spark_switches(read, true).unwrap_err().contains("VADE_W2_PATH must be unset or pipeline"), "{path:?}");
    }
    let read = |name: &str| if name == "VADE_W2_TP_HEAD" { Err(NotUnicode("x".into())) } else { Err(NotPresent) };
    assert!(plan::two_spark_switches(read, true).unwrap_err().contains("VADE_W2_TP_HEAD"));
}

#[test]
fn a_changed_two_spark_switch_is_a_different_launch() {
    const CASE: &str = "VADE_TEST_TWO_SPARK_FINGERPRINT";
    if std::env::var(CASE).is_ok() {
        // The resident started with no switch set; this shell now sets one level.
        let mut f = Fixture::fp4(2, false);
        let options = fp4_options(&["--sparks", "2"]);
        let mut session = f.app.requested_session_with(&options, None, true, |_| Err(std::env::VarError::NotPresent)).unwrap();
        assert_ne!(session.launch_fingerprint, f.app.requested_session(&options, None, true).unwrap().launch_fingerprint);
        f.app.resolve_fabric(&mut session).unwrap();
        f.app.resolve_rank_paths(&mut session).unwrap();
        session.phase = "running".into();
        for rank in &mut session.ranks {
            rank.attempted = true;
            f.app.runner.started.insert(rank.machine.name.clone());
        }
        session.save(&f.app.config.state_dir).unwrap();
        super::resident_tests::assert_refused_without_touching_resident(&mut f, &options, "changed VADE_W2_TP_PROJ");
        return;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.args(["--exact", "app::tests::two_spark::a_changed_two_spark_switch_is_a_different_launch", "--nocapture"]).env(CASE, "1");
    for name in plan::TWO_SPARK_SWITCHES { command.env_remove(name); }
    command.env_remove("SEQPAR_RDMA").env("VADE_W2_TP_PROJ", "2");
    let result = command.output().unwrap();
    assert!(result.status.success(), "{}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
    assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"), "the child ran the case");
}

#[test]
fn up_with_two_sparks_leaves_the_split_levels_to_the_engine() {
    const CASE: &str = "VADE_TEST_TWO_SPARK_SWITCHES";
    if let Ok(case) = std::env::var(CASE) {
        let mut f = Fixture::fp4(2, false);
        if case == "bad" {
            let error = f.app.up(&fp4_options(&["--sparks", "2"])).unwrap_err();
            assert!(["VADE_W2_PATH must be unset or pipeline", "VADE_W2_TP=0 no longer", "needs RDMA"].iter().any(|text| error.contains(text)), "{error}");
            assert!(starts(&f).is_empty() && f.app.runner.started.is_empty());
            return;
        }
        f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        let requests = starts(&f);
        assert_eq!(requests.len(), 2);
        for request in &requests {
            let carried: Vec<_> = request.keys().filter(|key| key.starts_with("env.VADE_W2_TP") || key.starts_with("env.VADE_W2_PATH")).cloned().collect();
            let removed: Vec<_> = request.keys().filter_map(|key| key.strip_prefix("env_unset.")).filter(|name| plan::TWO_SPARK_SWITCHES.contains(name)).collect();
            match case.as_str() {
                "empty" => {
                    assert!(carried.is_empty(), "{carried:?}");
                    assert_eq!(removed.len(), 5, "every unset switch is removed on the rank: {removed:?}");
                }
                _ => {
                    assert_eq!(carried, ["env.VADE_W2_PATH", "env.VADE_W2_TP_PROJ"]);
                    assert_eq!((request["env.VADE_W2_PATH"].as_str(), request["env.VADE_W2_TP_PROJ"].as_str()), ("pipeline", "2"));
                    assert_eq!(removed, ["VADE_W2_TP", "VADE_W2_TP_HEAD", "VADE_W2_TP_WIRE"]);
                }
            }
            assert!(!request.contains_key("env.VADE_W2_PP") && !request.contains_key("env_unset.VADE_W2_PP_SPLIT"));
        }
        // A chain of four keeps its own recipe; the 2-Spark switches stay out of it.
        let mut chain = Fixture::fp4(4, false);
        chain.app.up(&fp4_options(&["--sparks", "4", "--mtp", "0"])).unwrap();
        assert!(starts(&chain).iter().all(|request| request.keys().all(|key| !key.contains("VADE_W2_TP") && !key.contains("VADE_W2_PATH"))));
        return;
    }
    for (case, set) in [("empty", &[][..]), ("pipeline", &[("VADE_W2_PATH", "pipeline"), ("VADE_W2_TP_PROJ", "2")]), ("bad", &[("VADE_W2_PATH", "layer")]),
        ("bad", &[("VADE_W2_TP", "0")]), ("bad", &[("SEQPAR_RDMA", "0")])] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", "app::tests::two_spark::up_with_two_sparks_leaves_the_split_levels_to_the_engine", "--nocapture"]).env(CASE, case);
        for name in plan::TWO_SPARK_SWITCHES { command.env_remove(name); }
        command.env_remove("SEQPAR_RDMA");
        for (name, value) in set { command.env(name, value); }
        let result = command.output().unwrap();
        assert!(result.status.success(), "{case}: {}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
    }
}

const SPLIT: &str = r#","vade":{"numerics":{"world_size":2,"topology":"pipeline","path":"tensor_split","tensor_split":{"experts":"output_columns","projection_level":3,"wire":"write","head":"vocabulary_halves"}}}"#;

#[test]
fn status_names_the_two_spark_path_and_its_levels() {
    use crate::{evals::json::{self, Value}, status::Snapshot};
    let path = health::TwoSparkPath::from_health(&format!("{{\"status\":\"ready\"{SPLIT}}}")).unwrap();
    assert_eq!(path.label(), "tensor split (projection level 3, write wire, head split by vocabulary halves)");
    let pipeline = health::TwoSparkPath::from_health(r#"{"status":"ready","vade":{"numerics":{"path":"layer_pipeline"}}}"#).unwrap();
    assert_eq!((pipeline.label().as_str(), pipeline.projection_level), ("layer pipeline (VADE_W2_PATH=pipeline, or TCP)", None));
    assert_eq!(health::TwoSparkPath::from_health(r#"{"status":"ready","vade":{"numerics":{"topology":"local"}}}"#), None);
    for (extra, row, json_path) in [(SPLIT, "tensor split (projection level 3, write wire, head split by vocabulary halves)", Some("tensor_split")),
        ("", "not reported by this engine", None)] {
        let mut f = Fixture::fp4(2, false);
        f.app.runner.health_extra = extra.into();
        f.app.up(&fp4_options(&["--sparks", "2"])).unwrap();
        let capture = crate::output::testing::Capture::default();
        let printed = capture.bytes.clone();
        crate::output::with_writer(capture, || f.app.execute(Command::Status)).unwrap();
        let printed = String::from_utf8(printed.borrow().clone()).unwrap();
        assert!(printed.contains(&format!("\n  Two-Spark path         {row}\n")), "{printed}");
        let mut report = Snapshot::new("configured", false);
        f.app.status_snapshot(&mut report).unwrap();
        let value = json::parse(&report.finish(Duration::ZERO).0).unwrap();
        match json_path {
            Some(name) => {
                assert_eq!(value["two_spark_path"]["path"].as_str(), Some(name));
                assert_eq!((&value["two_spark_path"]["projection_level"], value["two_spark_path"]["wire"].as_str(), value["two_spark_path"]["head"].as_str()),
                    (&Value::from(3.0), Some("write"), Some("vocabulary_halves")));
            }
            None => assert_eq!(value["two_spark_path"], Value::Null),
        }
    }
    // One Spark has no 2-Spark path row.
    let mut one = Fixture::fp4(1, false);
    one.app.runner.health_extra = SPLIT.into();
    one.app.up(&fp4_options(&[])).unwrap();
    let capture = crate::output::testing::Capture::default();
    let printed = capture.bytes.clone();
    crate::output::with_writer(capture, || one.app.execute(Command::Status)).unwrap();
    assert!(!String::from_utf8(printed.borrow().clone()).unwrap().contains("Two-Spark path"));
}
