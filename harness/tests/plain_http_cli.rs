//! Process-level option and warning checks. No upstream connections.
#[path = "support/home.rs"]
mod home;
use std::process::Command;
const WARNING: &str = "Warning: --allow-plain-http is on. Links to other machines are not encrypted. Anyone on the network can read and change your chat, your tool results and your API key.";

#[test]
fn every_upstream_mode_shows_switch_and_warns_once() {
    for mode in ["direct", "agent", "gateway", "workload", "quality-card", "suite", "doctor", "vade-gateway"] {
        for allow in [false, true] {
            let mut cmd = home::harness();
            cmd.env_remove("VADE_API_KEY_FILE").arg(mode);
            if allow { cmd.args(["--allow-plain-http", "--allow-plain-http"]); }
            let output = cmd.arg("--help").output().unwrap();
            assert!(output.status.success(), "{mode}: {output:?}");
            let help = String::from_utf8(output.stdout).unwrap();
            let err = String::from_utf8(output.stderr).unwrap();
            assert!(help.contains("--allow-plain-http") && help.contains("same machine"), "{mode}: {help}");
            assert_eq!(err.lines().filter(|line| *line == WARNING).count(), usize::from(allow), "{mode}: {err}");
        }
    }
    for allow in [false, true] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vade-gateway"));
        cmd.env_remove("VADE_API_KEY_FILE");
        if allow { cmd.arg("--allow-plain-http"); }
        let output = cmd.arg("--help").output().unwrap();
        assert!(output.status.success());
        assert_eq!(String::from_utf8(output.stderr).unwrap().lines().filter(|line| *line == WARNING).count(), usize::from(allow));
    }
}

#[test]
fn utility_parsers_accept_flag_without_a_value() {
    for args in [
        vec!["workload", "--allow-plain-http"],
        vec!["quality-card", "--allow-plain-http"],
        vec!["suite", "--allow-plain-http"],
        vec!["doctor", "http://127.0.0.1:1", "--allow-plain-http", "--max-seconds", "0"],
        vec!["vade-gateway", "--allow-plain-http", "--bind", "invalid"],
    ] {
        let out = home::harness().env_remove("VADE_API_KEY_FILE").args(&args).output().unwrap();
        let err = String::from_utf8(out.stderr).unwrap();
        assert!(!out.status.success());
        assert_eq!(err.lines().filter(|line| *line == WARNING).count(), 1, "{args:?}: {err}");
        assert!(!err.contains("unknown") && !err.contains("option needs a value"), "{args:?}: {err}");
    }
}

#[test]
fn direct_agent_and_core_gateway_parse_switch_before_dispatch() {
    // A closed local port causes an ordinary connect error after parsing.
    for mode in ["direct", "agent", "gateway"] {
        let mut cmd = home::harness();
        cmd.env_remove("VADE_API_KEY_FILE").arg(mode).arg("--allow-plain-http");
        if mode == "gateway" {
            cmd.args(["--upstream", "http://127.0.0.1:1", "--bind", "invalid"]);
        } else {
            cmd.args(["--endpoint", "http://127.0.0.1:1", "--model", "mock", "--prompt", "hello", "--workspace"])
                .arg(home::path()).args(["--plain", "--no-instructions", "--no-web", "--no-memory"]);
        }
        let output = cmd.output().unwrap();
        let err = String::from_utf8(output.stderr).unwrap();
        assert_eq!(err.lines().filter(|line| *line == WARNING).count(), 1, "{mode}: {err}");
        assert!(!err.contains("unknown option"), "{mode}: {err}");
    }
}

#[test]
fn a_prompt_value_is_not_an_opt_in_flag() {
    let out = home::harness().env_remove("VADE_API_KEY_FILE")
        .args(["direct", "--endpoint", "http://127.0.0.1:1", "--model", "mock", "--prompt", "--allow-plain-http", "--workspace"])
        .arg(home::path()).args(["--session", "--help", "--plain", "--no-instructions", "--no-web", "--no-memory"]).output().unwrap();
    assert!(!String::from_utf8(out.stderr).unwrap().contains(WARNING));
}
