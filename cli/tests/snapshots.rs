//! Golden output for help pages, common errors and the color switches.
//! Each case runs the real binary with a fixed environment and compares
//! stdout, stderr and the exit code with `tests/snapshots/<name>.txt`.
//! After an intended change, rerun with VADE_UPDATE_SNAPSHOTS=1 and review the diff.
use std::{fs, path::Path, process::Command};

const CASES: &[(&str, &[&str])] = &[
    ("help", &["--help"]),
    ("help-launch", &["launch", "--help"]),
    ("help-up", &["up", "--help"]),
    ("help-down", &["down", "--help"]),
    ("help-status", &["status", "--help"]),
    ("help-logs", &["logs", "--help"]),
    ("help-chat", &["chat", "--help"]),
    ("help-doctor", &["doctor", "--help"]),
    ("help-install", &["install", "--help"]),
    ("help-setup", &["setup", "--help"]),
    ("help-update", &["update", "--help"]),
    ("help-detect", &["detect", "--help"]),
    ("help-version", &["version", "--help"]),
    ("help-run", &["run", "--help"]),
    ("help-verify", &["verify", "--help"]),
    ("help-read", &["read", "--help"]),
    ("help-config", &["config", "--help"]),
    ("help-bench", &["bench", "--help"]),
    ("help-test", &["test", "--help"]),
    ("help-pull", &["pull", "--help"]),
    ("help-models", &["models", "--help"]),
    ("error-config-argument", &["config", "--all"]),
    ("error-json-elsewhere", &["down", "--json"]),
    ("error-unknown-command", &["frobnicate"]),
    ("error-unknown-option", &["up", "--bogus"]),
    ("error-missing-value", &["launch", "--context"]),
    ("error-out-of-range", &["up", "--mtp", "9"]),
    ("error-repeated-option", &["up", "--mtp", "1", "--mtp", "2"]),
    ("error-unexpected-argument", &["down", "--force"]),
    ("error-output-over-context", &["up", "--context", "4096", "--output", "8192"]),
    ("error-status-json-options", &["status", "--json", "--watch"]),
    ("error-version-arguments", &["--version", "now"]),
    ("error-help-topic", &["help", "nonsense"]),
    ("error-missing-config", &["status"]),
    ("error-verify-option", &["verify", "--bogus"]),
    ("error-chat-prompt", &["chat", "explain main.rs"]),
    ("error-chat-json", &["chat", "--json"]),
];

fn command(args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
    command.args(args).env_clear().env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", "/nonexistent-vade-snapshot-home").env("VADE_HOME", "/nonexistent-vade-snapshot-home/.vade")
        .env("VADE_CONFIG", "/nonexistent-vade-snapshot/config").env("TERM", "xterm-256color");
    command
}

fn render(args: &[&str], command: &mut Command) -> String {
    let output = command.output().unwrap();
    format!("$ vade {}\n{}--- stderr\n{}--- exit {}\n", args.join(" "), String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr), output.status.code().map_or("signal".into(), |code| code.to_string()))
}

fn check(name: &str, actual: &str, failures: &mut Vec<String>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots").join(format!("{name}.txt"));
    if std::env::var_os("VADE_UPDATE_SNAPSHOTS").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, actual).unwrap();
        return;
    }
    let expected = fs::read_to_string(&path).unwrap_or_default();
    if expected != actual { failures.push(format!("{name}: expected\n{expected}\nactual\n{actual}")); }
}

#[test]
fn help_and_errors_match_their_snapshots() {
    let mut failures = Vec::new();
    for (name, args) in CASES { check(name, &render(args, &mut command(args)), &mut failures); }
    assert!(failures.is_empty(), "{} snapshot(s) differ; rerun with VADE_UPDATE_SNAPSHOTS=1 after an intended change:\n{}",
        failures.len(), failures.join("\n"));
}

#[test]
fn color_is_opt_in_for_pipes_and_no_color_always_wins() {
    let mut failures = Vec::new();
    for (name, args) in [("color-error", &["up", "--bogus"][..]), ("color-help", &["status", "--help"])] {
        check(name, &render(args, command(args).env("CLICOLOR_FORCE", "1")), &mut failures);
        let plain = render(args, command(args).env("CLICOLOR_FORCE", "1").env("NO_COLOR", "1"));
        assert!(!plain.contains('\x1b'), "NO_COLOR must win: {plain}");
        assert_eq!(plain, render(args, &mut command(args)), "piped output is plain by default");
    }
    for args in [&["status", "--json"][..], &["--version"]] {
        assert!(!render(args, command(args).env("CLICOLOR_FORCE", "1")).contains('\x1b'), "machine output is never colored: {args:?}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
