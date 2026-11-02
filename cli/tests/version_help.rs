use std::{os::{fd::OwnedFd, unix::net::UnixStream}, process::{Command, Stdio}};
fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
    // Explicit invalid settings would fail if discovery entered lifecycle setup.
    command.env("VADE_HOME", "relative-invalid").env("VADE_CONFIG", "").env("VADE_API_KEY_FILE", "missing-credential");
    command
}
#[test]
fn version_and_help_work_without_valid_setup() {
    for argument in ["--version", "-V", "version"] {
        let output = command().arg(argument).output().unwrap();
        assert!(output.status.success());assert!(output.stderr.is_empty());
        assert_version(&String::from_utf8(output.stdout).unwrap());
    }
    for arguments in [&["--help"][..], &["help", "version"][..], &["status", "--help"][..]] {
        let output = command().args(arguments).output().unwrap();
        assert!(output.status.success());assert!(output.stderr.is_empty());
        let stdout = String::from_utf8(output.stdout).unwrap();assert!(stdout.contains(if arguments[0] == "status" { "--json" } else { "version" }));
    }
    let output = command().args(["--version", "unexpected"]).output().unwrap();
    assert_eq!(output.status.code(), Some(1));assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr).unwrap().contains("vade --version without additional arguments"));
}
#[test]
fn version_and_help_allow_consumers_to_close_stdout() {
    for argument in ["--version", "--help"] {
        let (writer, reader) = UnixStream::pair().unwrap();drop(reader);
        let fd: OwnedFd = writer.into();
        let output = command().arg(argument).stdout(Stdio::from(fd)).output().unwrap();
        assert!(output.status.success());assert!(output.stderr.is_empty());
    }
}

#[test]
fn evaluation_and_benchmark_help_is_offline_and_tolerates_closed_consumers() {
    let cases: &[&[&str]] = &[
        &["test"], &["test", "--help"], &["test", "-h"],
        &["help", "test"], &["--help", "test"], &["-h", "test"],
        &["test", "build", "--help"], &["test", "build", "-h"],
        &["test", "determinism", "--help"], &["test", "longctx", "--help"],
        &["test", "longctx", "prepare", "--help"],
        &["test", "longctx", "run", "--help"],
        &["test", "longctx", "compare", "--help"],
        &["test", "gdn-state", "--help"], &["test", "unix-flags-abi", "--help"],
        &["bench", "--help"], &["bench", "--suite", "--help"], &["help", "bench"], &["bench", "verdict", "--help"],
    ];
    for args in cases {
        // Invalid setup and tool paths prove help reaches no build, capture,
        // credential read or engine operation, including the benchmark entry.
        let mut cmd = command();
        cmd.args(*args).env("RUSTC", "missing-help-only-compiler")
            .env("PRIVATE_DENYLIST", "missing-help-only-denylist");
        let normal = cmd.output().unwrap();
        assert!(normal.status.success() && normal.stderr.is_empty(), "{args:?}: {normal:?}");
        assert!(!normal.stdout.is_empty());
        let (writer, reader) = UnixStream::pair().unwrap(); drop(reader);
        let fd: OwnedFd = writer.into();
        let closed = cmd.stdout(Stdio::from(fd)).output().unwrap();
        assert!(closed.status.success() && closed.stderr.is_empty(), "{args:?}: {closed:?}");
    }
}

#[test]
fn output_failures_other_than_broken_pipe_fail_even_for_benchmark_help() {
    use std::{fs, os::unix::process::CommandExt};
    #[repr(C)]
    struct Limit { current: u64, maximum: u64 }
    unsafe extern "C" { fn setrlimit(resource: i32, limit: *const Limit) -> i32; fn signal(number: i32, handler: usize) -> usize; }
    let root = std::env::temp_dir().join(format!("vade-help-fsize-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    for args in [&["--help"][..], &["--version"], &["test", "--help"], &["bench", "--help"], &["bench", "--suite", "--help"]] {
        let file = fs::File::create(root.join("stdout")).unwrap();
        let mut cmd = command(); cmd.args(args).stdout(file);
        // RLIMIT_FSIZE/SIGXFSZ are 1/25 on the supported macOS and Linux hosts.
        // Ignore the signal so the real file write returns EFBIG to Rust.
        unsafe { cmd.pre_exec(|| {
            signal(25, 1);
            if setrlimit(1, &Limit { current: 0, maximum: 0 }) != 0 { return Err(std::io::Error::last_os_error()); }
            Ok(())
        }); }
        let output = cmd.output().unwrap();
        assert_eq!(output.status.code(), Some(1), "{args:?}: {output:?}");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("Cannot write") && error.contains("output destination"), "{args:?}: {error}");
        assert!(!error.contains("panicked"));
        assert!(fs::read(root.join("stdout")).unwrap().is_empty());
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn errors_do_not_panic_when_stderr_is_closed() {
    for args in [&["unknown-command"][..], &["run", "qwen3-8b", "--unknown"]] {
        let (writer, reader) = UnixStream::pair().unwrap(); drop(reader);
        let fd: OwnedFd = writer.into();
        let output = command().args(args).stderr(Stdio::from(fd)).output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn failed_gate_keeps_its_status_when_stdout_is_closed() {
    let args = ["test", "determinism", "--bench", "/usr/bin/true"];
    let normal = command().args(args).output().unwrap();
    assert_eq!(normal.status.code(), Some(2), "{normal:?}");
    assert!(!normal.stdout.is_empty(), "gate must write before failing");

    let (writer, reader) = UnixStream::pair().unwrap();
    drop(reader);
    let fd: OwnedFd = writer.into();
    let closed = command().args(args).stdout(Stdio::from(fd)).output().unwrap();
    assert_eq!(closed.status.code(), normal.status.code(), "{closed:?}");
    let error = String::from_utf8(closed.stderr).unwrap();
    assert!(error.contains("FAIL inputs:") && error.contains("result=2"), "{error}");
    assert!(!error.contains("panicked"));
}

#[test]
fn failed_status_and_helper_reports_keep_their_status_when_stdout_is_closed() {
    for args in [&["status", "--json"][..], &["__worker"]] {
        let normal = command().args(args).stdin(Stdio::null()).output().unwrap();
        assert_eq!(normal.status.code(), Some(1), "{args:?}: {normal:?}");
        assert!(!normal.stdout.is_empty());
        let (writer, reader) = UnixStream::pair().unwrap();
        drop(reader);
        let fd: OwnedFd = writer.into();
        let closed = command().args(args).stdin(Stdio::null()).stdout(Stdio::from(fd)).output().unwrap();
        assert_eq!(closed.status.code(), normal.status.code(), "{args:?}: {closed:?}");
    }
}
/// Argon review of 7c9881a, item 6: `vade read` needs no key, so the key
/// settings do not apply to it and its own parser refuses the key option.
#[test]
fn read_ignores_key_settings_and_refuses_the_key_option() {
    let dir = std::env::temp_dir().join(format!("vade-read-key-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("notes.txt");
    std::fs::write(&file, "hello\n").unwrap();
    let output = command().env("VADE_API_KEY_FILE", "").arg("read").arg(&file).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8(output.stdout).unwrap().ends_with("\nhello\n"));
    let output = command().args(["read", "--api-key-file", "key.txt"]).arg(&file).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// One line: the package version, then the short commit of this build, or a
/// plain statement that the build has no commit.
fn assert_version(line: &str) {
    let rest = line.strip_prefix(concat!("vade ", env!("CARGO_PKG_VERSION"), " (")).and_then(|rest| rest.strip_suffix(")\n"));
    let rest = rest.unwrap_or_else(|| panic!("{line:?}"));
    if rest == "unknown commit" { return; }
    let short = rest.strip_suffix(", modified").unwrap_or(rest);
    assert!(short.len() == 7 && short.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)), "{line:?}");
}

#[test]
fn version_json_names_the_same_commit_as_the_version_line() {
    let line = String::from_utf8(command().arg("--version").output().unwrap().stdout).unwrap();
    assert_version(&line);
    for arguments in [&["--version", "--json"][..], &["version", "--json"], &["-V", "--json"]] {
        let output = command().args(arguments).output().unwrap();
        assert!(output.status.success() && output.stderr.is_empty(), "{output:?}");
        let json = String::from_utf8(output.stdout).unwrap();
        let start = concat!("{\"schema\":\"vade.version.v1\",\"version\":\"", env!("CARGO_PKG_VERSION"), "\",\"sha\":");
        let rest = json.strip_prefix(start).unwrap_or_else(|| panic!("{json}"));
        let (sha, modified) = rest.split_once(",\"modified\":").unwrap_or_else(|| panic!("{json}"));
        assert_eq!(modified.trim_end().trim_end_matches('}').parse::<bool>().unwrap(), line.contains(", modified"), "{json}");
        if sha == "null" {
            assert!(line.contains("(unknown commit)"), "{line}");
        } else {
            let sha = sha.trim_matches('"');
            assert!(sha.len() == 40 && line.contains(&sha[..7]), "{sha} {line}");
        }
    }
}

#[test]
fn tools_and_mcp_help_print_with_no_harness_installed() {
    let home = std::env::temp_dir().join(format!("vade-no-harness-{}", std::process::id()));
    for (args, usage) in [(&["tools", "--help"][..], "Usage: vade tools"), (&["tools", "-h"], "Usage: vade tools"), (&["mcp", "--help"], "Usage: vade mcp"),
        (&["mcp", "-h"], "Usage: vade mcp"), (&["mcp"], "Usage: vade mcp"), (&["help", "tools"], "Usage: vade tools"), (&["help", "mcp"], "Usage: vade mcp")] {
        let output = Command::new(env!("CARGO_BIN_EXE_vade")).args(args).env_remove("VADE_CONFIG").env_remove("VADE_API_KEY_FILE")
            .env("VADE_HOME", &home).env("HOME", &home).env("XDG_CONFIG_HOME", &home).output().unwrap();
        assert_eq!(output.status.code(), Some(0), "{args:?}: {output:?}");
        assert!(output.stderr.is_empty() && String::from_utf8(output.stdout).unwrap().starts_with(usage), "{args:?}");
    }
    // A real run still needs the install, and says so.
    let output = Command::new(env!("CARGO_BIN_EXE_vade")).arg("tools").env_remove("VADE_CONFIG").env_remove("VADE_API_KEY_FILE")
        .env("VADE_HOME", &home).env("HOME", &home).env("XDG_CONFIG_HOME", &home).output().unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8(output.stderr).unwrap().contains("The chat harness is not installed."));
    assert!(!home.exists(), "help and the install check write nothing");
}
