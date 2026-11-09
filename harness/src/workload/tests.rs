use super::report::percentile;
use super::*;
#[test]
fn project_memory_does_not_change_workload_requests() {
    let fixture = crate::test_workspace::Fixture::new();
    let workspace = fixture.workspace();
    let config = Config::default();
    for role in [Role::Main, Role::Subordinate] {
        let before = conversation(&config, role, 0).unwrap().request(limits().wire).to_json().unwrap();
        crate::memory::change(&workspace, "workload-sentinel", Some("Keep this out of benchmark prompts.")).unwrap();
        let after = conversation(&config, role, 0).unwrap().request(limits().wire).to_json().unwrap();
        assert_eq!(before, after);
        assert!(!after.contains("workload-sentinel"));
        assert!(!after.contains(crate::memory::LABEL));
    }
}
#[test]
fn seeds_sizes_and_closed_synthetic_history_are_repeatable() {
    let c = Config {
        model: "mock".into(),
        ..Config::default()
    };
    assert_eq!(filler(41, 100), filler(41, 100));
    assert_ne!(filler(41, 100), filler(42, 100));
    assert_eq!(filler(41, 137).len(), 137);
    let a = conversation(&c, Role::Main, 0).unwrap();
    let b = conversation(&c, Role::Main, 0).unwrap();
    assert_eq!(
        a.request(limits().wire).to_json().unwrap(),
        b.request(limits().wire).to_json().unwrap()
    );
    assert_eq!(
        a.messages().iter().map(|m| m.role()).collect::<Vec<_>>(),
        ["system", "assistant", "tool", "user"]
    );
    assert!(
        a.serialized_bytes()
            > conversation(&c, Role::Subordinate, 1)
                .unwrap()
                .serialized_bytes()
    );
    assert!(
        !a.request(limits().wire)
            .to_json()
            .unwrap()
            .contains("workspace")
    );
}
#[test]
fn bounds_and_configuration_fail_before_io() {
    let base = [
        "--endpoint",
        "http://127.0.0.1:1",
        "--model",
        "mock",
        "--json",
        "report.json",
    ];
    let parse =
        |extra: &[&str]| Config::parse(base.iter().chain(extra.iter()).map(|s| s.to_string()));
    assert!(parse(&[]).is_ok());
    for extra in [
        &["--sessions", "0"][..],
        &["--sessions", "33"],
        &["--main-sessions", "3"],
        &["--turns", "1"],
        &["--turns", "1000"],
        &["--output-tokens", "0"],
        &["--main-tool-bytes", "262145"],
        &["--seed", "-1"],
        &["--turns", "2", "--turns", "3"],
        &["--wat", "1"],
    ] {
        assert!(parse(extra).is_err(), "{extra:?}");
    }
    for endpoint in [
        "https://host",
        "http://user:secret@host",
        "http://host/?secret=x",
        "http://host/#x",
    ] {
        assert!(
            Config::parse(
                ["--endpoint", endpoint, "--model", "mock", "--json", "out"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .is_err()
        );
    }
}
#[test]
fn percentile_is_nearest_rank() {
    assert_eq!(percentile(&[], 95), None);
    assert_eq!(percentile(&[1., 2., 3., 4.], 50), Some(2.));
    assert_eq!(percentile(&[1., 2., 3., 4.], 95), Some(4.));
}
