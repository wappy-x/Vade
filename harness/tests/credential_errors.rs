//! File failures must be actionable before a listener, workspace or request exists.
#[path = "support/home.rs"]
mod home;
#[path = "support/cli.rs"]
#[allow(dead_code)]
mod cli;
use std::{fs, net::TcpListener, path::Path, process::Command};

fn command(mode: &str, endpoint: &str, workspace: &Path) -> Command {
    let mut cmd = home::command(if mode == "standalone" || mode == "replay" {
        env!("CARGO_BIN_EXE_vade-gateway")
    } else { env!("CARGO_BIN_EXE_vade-harness") });
    cmd.env_remove("VADE_API_KEY_FILE");
    match mode {
        "direct" | "agent" => {
            cmd.args([mode, "--state-in-project", "--endpoint", endpoint, "--model", "mock", "--prompt", "hello", "--workspace"])
                .arg(workspace);
        }
        "gateway" => { cmd.args(["gateway", "--upstream", endpoint, "--bind", "127.0.0.1:0"]); }
        "standalone" => { cmd.args(["--upstream", endpoint, "--bind", "127.0.0.1:0"]); }
        "replay" => { cmd.arg("replay").arg(workspace).args(["--upstream", endpoint]); }
        _ => unreachable!(),
    }
    cmd
}

#[test]
fn credential_failures_explain_cause_and_recovery_without_file_contents_or_side_effects() {
    let root = std::env::temp_dir().join(format!("vade-credential-errors-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let workspace = root.join("untouched-workspace");
    let peer = TcpListener::bind("127.0.0.1:0").unwrap();
    peer.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}/v1", peer.local_addr().unwrap());
    fs::write(root.join("binary"), b"PRIVATE_FILE_CONTENT\xff").unwrap();
    fs::write(root.join("short"), b"SECRET").unwrap();
    fs::write(root.join("multiline"), b"PRIVATE_FILE_CONTENT\nANOTHER_KEY").unwrap();
    fs::write(root.join("open"), b"PRIVATE_FILE_CONTENT_0123456789").unwrap();
    fs::write(root.join("large"), "PRIVATE_FILE_CONTENT".repeat(205)).unwrap();
    // Only the owner may read or write a key file. The file "open" keeps a group read bit.
    #[cfg(unix)]
    for (name, mode) in [("binary", 0o600), ("short", 0o600), ("multiline", 0o600), ("large", 0o600), ("open", 0o640)] {
        fs::set_permissions(root.join(name), std::os::unix::fs::PermissionsExt::from_mode(mode)).unwrap();
    }
    let cases = [
        #[cfg(unix)]
        (root.join("open"), "group or others can read or write the file. Run chmod 600"),
        (root.join("large"), "the file is larger than 4096 bytes. Replace the file contents"),
        (root.join("missing"), "file not found. Check the selected path"),
        (root.clone(), "path is a directory. Select a regular file"),
        (root.join("binary"), "file is not valid UTF-8. Save"),
        (root.join("short"), "Replace the file contents"),
        (root.join("multiline"), "Replace the file contents"),
        (root.join("short/child"), "a path component is not a directory. Correct"),
        (root.join("missing\n\t\x1b[2J\u{202e}\u{2028}"), "file not found. Check the selected path"),
    ];
    let mut count = 0;
    for mode in ["direct", "agent", "gateway", "standalone", "replay"] {
        for (path, action) in &cases {
            let mut cmd = command(mode, &endpoint, &workspace);
            if count % 2 == 0 { cmd.arg("--api-key-file").arg(path); }
            else { cmd.env("VADE_API_KEY_FILE", path); }
            let output = cli::Cli::start(&mut cmd).finish();
            assert_eq!(output.status.code(), Some(1), "{mode}: {output:?}");
            assert!(output.stdout.is_empty(), "{mode}: {output:?}");
            let error = String::from_utf8(output.stderr).unwrap();
            assert!(error.contains(action), "{mode}: {error}");
            assert!(error.contains("--api-key-file PATH") && error.contains("VADE_API_KEY_FILE"), "{error}");
            assert!(!error.contains("PRIVATE_FILE_CONTENT") && !error.contains("SECRET") && !error.contains("ANOTHER_KEY"), "{error}");
            assert!(!error.contains(['\x1b', '\r', '\t', '\u{202e}', '\u{2028}']), "{error}");
            assert_eq!(error.lines().count(), 1, "{error}");
            if path.file_name().is_some_and(|name| name.to_string_lossy().contains('\n')) {
                assert!(error.contains("\\n") && error.contains("\\t") && error.contains("\\u{202e}"), "{error}");
            }
            assert!(!workspace.exists(), "{mode} created a workspace before rejecting credentials");
            assert_eq!(peer.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "{mode} contacted upstream");
            count += 1;
        }
    }
    assert_eq!(count, 5 * cases.len());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn credential_argument_errors_recover_and_offline_help_never_reads_the_file() {
    let missing = std::env::temp_dir().join(format!("missing-help-key-{}", std::process::id()));
    for binary in [env!("CARGO_BIN_EXE_vade-harness"), env!("CARGO_BIN_EXE_vade-gateway")] {
        for args in [vec!["--api-key-file"], vec!["--api-key-file", ""], vec!["--api-key-file", "one", "--api-key-file", "two"]] {
            let mut cmd = home::command(binary);
            cmd.args(args).env_remove("VADE_API_KEY_FILE");
            let output = cli::Cli::start(&mut cmd).finish();
            assert!(!output.status.success());
            let error = String::from_utf8(output.stderr).unwrap();
            assert!(error.contains("Use --api-key-file PATH") || error.contains("Pass this option once"), "{error}");
        }
    }
    for mode in ["direct", "agent", "gateway", "standalone", "doctor", "setup", "workload"] {
        let mut cmd = home::command(if mode == "standalone" || mode == "replay" { env!("CARGO_BIN_EXE_vade-gateway") } else { env!("CARGO_BIN_EXE_vade-harness") });
        if mode != "standalone" { cmd.arg(mode); }
        cmd.arg("--help").env("VADE_API_KEY_FILE", &missing);
        let output = cli::Cli::start(&mut cmd).finish();
        assert!(output.status.success(), "{mode}: {output:?}");
        assert!(output.stderr.is_empty(), "{mode}: {output:?}");
        assert!(!output.stdout.is_empty());
    }
}
