use std::process::Command;

const WORKFLOW: &str = include_str!("release.yml");

fn guard() -> String {
    let steps = WORKFLOW.split_once("    steps:\n").expect("release steps").1;
    let first = steps.split_once("      - uses: actions/checkout@").expect("checkout after guard").0;
    let script = first.strip_prefix("      - name: Require a version tag\n        run: |\n")
        .expect("ref guard must run before checkout and toolchain setup");
    script.lines().map(|line| format!("{}\n", line.strip_prefix("          ").expect("guard indentation"))).collect()
}

fn accepts(reference: &str, kind: &str) -> bool {
    Command::new("bash").args(["--noprofile", "--norc", "-e", "-c", &guard()])
        .env_clear().env("GITHUB_REF", reference).env("GITHUB_REF_TYPE", kind)
        .output().expect("run release guard offline").status.success()
}

#[test]
fn rejects_branches_and_non_version_tags() {
    for (reference, kind) in [("refs/heads/main", "branch"), ("refs/heads/v0.2.0", "branch"),
        ("refs/tags/release-0.2.0", "tag"), ("refs/pull/1/merge", "branch"), ("", "")] {
        assert!(!accepts(reference, kind), "guard accepted {:?}", reference);
    }
}

#[test]
fn accepts_version_tags_and_checks_out_the_full_ref() {
    for reference in ["refs/tags/v0.2.0", "refs/tags/v0.2.0-rc.1"] {
        assert!(accepts(reference, "tag"), "guard rejected {:?}", reference);
    }
    let checkout = WORKFLOW.split_once("      - uses: actions/checkout@").expect("checkout").1
        .split("      - name:").next().unwrap();
    assert!(checkout.contains("          ref: ${{ github.ref }}\n"), "checkout must use the full tag ref");
}
