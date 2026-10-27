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
            fp8.app.up(&legacy_options()).unwrap();
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
        // A chain of four carries no path switch either: the engine chooses the chain.
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
const CHAIN: &str = r#","vade":{"numerics":{"world_size":4,"topology":"pipeline","path":"layer_pipeline","path_reason":"FP4-Local on three or more Sparks"}}"#;
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
            "layer chain (FP4-Local on three or more Sparks)".into(),
            r#"{"path":"layer_pipeline","path_reason":"FP4-Local on three or more Sparks","projection_level":null,"wire":null,"head":null}"#),
        ("FP8 ring", || Fixture::new(2, false), legacy_options(), RING,
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
