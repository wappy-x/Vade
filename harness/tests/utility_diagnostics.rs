//! Public utility diagnostics identify options and remedies without dispatch.
#[path = "support/cli.rs"]
#[allow(dead_code)]
mod cli;
use std::{fs, net::TcpListener, process::Command};

fn failure(args: &[&str], mode: &str) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
    cmd.args(args).env_remove("VADE_API_KEY_FILE");
    let output = cli::Cli::start(&mut cmd).finish();
    assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
    assert!(output.stdout.is_empty());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains(&format!("See vade-harness {mode} --help.")), "{error}");
    assert_eq!(error.lines().count(), 1, "{error}");
    assert!(!error.contains("PRIVATE") && !error.contains(['\x1b', '\r', '\u{202e}']), "{error}");
    error
}

#[test]
fn utility_options_name_the_problem_without_echoing_values_or_blame_from_other_flags() {
    let peer = TcpListener::bind("127.0.0.1:0").unwrap();
    peer.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}/v1", peer.local_addr().unwrap());
    let groups: &[(&str, &[&str], &[&str], &[&str])] = &[
        ("setup", &["setup", "generic"],
            &["--endpoint", "--model", "--gateway", "--target", "--write", "--context-window", "--max-output-tokens", "--client-bin"],
            &["--context-window", "--max-output-tokens"]),
        ("doctor", &["doctor", &endpoint], &["--model", "--max-seconds"], &["--max-seconds"]),
        ("workload", &["workload"],
            &["--endpoint", "--model", "--json", "--sessions", "--main-sessions", "--turns", "--main-system-bytes", "--sub-system-bytes", "--main-user-bytes", "--sub-user-bytes", "--main-tool-bytes", "--sub-tool-bytes", "--output-tokens", "--seed", "--max-seconds"],
            &["--sessions", "--main-sessions", "--turns", "--main-system-bytes", "--sub-system-bytes", "--main-user-bytes", "--sub-user-bytes", "--main-tool-bytes", "--sub-tool-bytes", "--output-tokens", "--seed", "--max-seconds"]),
    ];
    let mut cases = 0;
    for (mode, prefix, options, numeric) in groups {
        for option in *options {
            let mut args = prefix.to_vec(); args.push(option);
            assert!(failure(&args, mode).contains(&format!("{option}:")));
            cases += 1;
        }
        for option in *numeric {
            for invalid in ["", "PRIVATE_VALUE\x1b[2J\u{202e}"] {
                let mut args = prefix.to_vec(); args.extend([option, invalid]);
                assert!(failure(&args, mode).contains(&format!("{option}:")));
                cases += 1;
            }
        }
        for unknown in ["--unknown", "--unknown=PRIVATE_VALUE", "--PRIVATE\x1b[2J", "PRIVATE_POSITIONAL"] {
            let mut args = prefix.to_vec(); args.extend([unknown, "PRIVATE_VALUE"]);
            let error = failure(&args, mode);
            if unknown.starts_with("--unknown") { assert!(error.contains("--unknown:"), "{error}"); }
            cases += 1;
        }
        let option = if *mode == "workload" { "--seed" } else { "--model" };
        let mut args = prefix.to_vec(); args.extend([option, "1", option, "2"]);
        assert!(failure(&args, mode).contains(&format!("{option}:")));
        cases += 1;
    }
    let error = failure(&["setup", "generic", "--endpoint", &endpoint, "--model", "mock", "--write", "unused", "--target", "both"], "setup");
    assert!(!error.starts_with("vade: --target:"), "{error}");
    let error = failure(&["workload", "--endpoint", &endpoint, "--model", "mock", "--json", "unused", "--sessions", "0", "--seed", "41"], "workload");
    assert!(!error.starts_with("vade: --seed:"), "{error}");
    cases += 2;
    for (option, value) in [("--write", ""), ("--client-bin", ""), ("--client-bin", "PRIVATE\x1b[2J")] {
        let error = failure(&["setup", "generic", option, value], "setup");
        assert!(error.contains(&format!("{option}:")) && !error.contains("unknown"), "{error}");
        cases += 1;
    }
    assert_eq!(cases, 75);
    assert_eq!(peer.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
}

#[test]
fn utility_output_failures_preserve_existing_files_and_explain_a_new_destination() {
    let root = std::env::temp_dir().join(format!("vade-utility-output-errors-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let keep = root.join("existing"); fs::write(&keep, b"keep this receipt").unwrap();
    let peer = TcpListener::bind("127.0.0.1:0").unwrap();
    peer.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}/v1", peer.local_addr().unwrap());
    for (path, remedy) in [
        (keep.clone(), "Keep it and choose a new output path"),
        (root.join("missing/output"), "under an existing directory"),
        (keep.join("output"), "under an existing directory"),
    ] {
        for mode in ["setup", "workload"] {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
            cmd.arg(mode).env_remove("VADE_API_KEY_FILE");
            if mode == "setup" { cmd.args(["generic", "--target", "direct"]); }
            cmd.args(["--endpoint", &endpoint, "--model", "mock", if mode == "setup" { "--write" } else { "--json" }]).arg(&path);
            let output = cli::Cli::start(&mut cmd).finish();
            assert!(!output.status.success());
            let error = String::from_utf8(output.stderr).unwrap();
            assert!(error.contains(remedy), "{mode}: {error}");
            assert!(output.stdout.is_empty());
            assert_eq!(fs::read(&keep).unwrap(), b"keep this receipt");
            assert_eq!(peer.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
        }
    }
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    fs::remove_dir_all(root).unwrap();
}
