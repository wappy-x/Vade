//! Help must be available offline, before credential, recording or stats I/O.
#[path = "support/cli.rs"]
#[allow(dead_code)]
mod cli;
use std::{fs, process::Command};

#[test]
fn gateway_and_replay_help_do_not_read_or_create_files_in_any_option_position() {
    let root = std::env::temp_dir().join(format!("vade-gateway-help-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let new = root.join("must-not-be-created");
    let new = new.to_str().unwrap();
    let missing_key = root.join("missing-key");
    let cases: &[&[&str]] = &[
        &["--help"], &["-h"],
        &["--stats", new, "--help"],
        &["--record", new, "-h"],
        &["--upstream", "invalid", "--stats", new, "--help"],
        &["--record", new, "--help", "--stats", new],
        &["replay", "--help"], &["replay", "-h"],
        &["replay", new, "--stats", new, "--help"],
        &["replay", "--upstream", "invalid", "--help"],
        &["replay", new, "-h", "--stats", new],
        &["replay", new, "--concurrency", "invalid", "--help"],
    ];
    for args in cases {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vade-gateway"));
        cmd.args(*args).env("VADE_API_KEY_FILE", &missing_key)
            .env("VADE_GATEWAY_UPSTREAM", "invalid");
        let output = cli::Cli::start(&mut cmd).finish();
        assert!(output.status.success(), "{args:?}: {output:?}");
        assert!(output.stderr.is_empty(), "{args:?}: {output:?}");
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("--api-key-file PATH"), "{text}");
        if args.first() == Some(&"replay") {
            assert!(text.contains("Usage: vade-gateway replay DIR"), "{text}");
            assert!(text.contains("1..64") && text.contains("does not execute tools"), "{text}");
        } else { assert!(text.contains("Generation defaults:"), "{text}"); }
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0, "{args:?} created files");
    }
    fs::remove_dir_all(root).unwrap();
}
