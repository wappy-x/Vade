//! Public utility output must remain usable when a pipeline consumer exits.
#[path = "support/home.rs"]
mod home;
#[cfg(unix)]
#[test]
fn utility_output_survives_a_closed_consumer_without_panicking() {
    use std::{
        os::{fd::OwnedFd, unix::net::UnixStream},
        process::Stdio,
        thread,
        time::{Duration, Instant},
    };
    let harness = env!("CARGO_BIN_EXE_vade-harness");
    let gateway = env!("CARGO_BIN_EXE_vade-gateway");
    let cases: &[(&str, &[&str])] = &[
        (harness, &["--help"]),
        (harness, &["direct", "--help"]),
        (harness, &["agent", "--help"]),
        (harness, &["gateway", "--help"]),
        (harness, &["--version"]),
        (harness, &["setup", "--help"]),
        (harness, &["doctor", "--help"]),
        (harness, &["workload", "--help"]),
        (harness, &["demo-compaction"]),
        (harness, &["setup", "generic", "--endpoint", "http://127.0.0.1:1/v1", "--model", "fixture"]),
        (gateway, &["--help"]),
        (gateway, &["replay", "--help"]),
        (gateway, &["replay", "--stats", "unused", "-h"]),
    ];
    let mut failures = Vec::new();
    for (binary, args) in cases {
        // An unnamed pair creates no filesystem sockets. Closing the peer before
        // spawning makes EPIPE deterministic, including for one-line --version.
        let (writer, reader) = UnixStream::pair().unwrap();
        drop(reader);
        let fd: OwnedFd = writer.into();
        let mut child = home::command(binary)
            .args(*args)
            .env_remove("VADE_API_KEY_FILE")
            .env_remove("VADE_HARNESS_EFFORT")
            .stdin(Stdio::null())
            .stdout(Stdio::from(fd))
            .stderr(Stdio::piped())
            .spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("utility did not exit: {args:?}");
            }
            thread::sleep(Duration::from_millis(1));
        }
        let output = child.wait_with_output().unwrap();
        if !output.status.success() || !output.stderr.is_empty() {
            failures.push(format!("{args:?}: {} {}", output.status, String::from_utf8_lossy(&output.stderr)));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(unix)]
#[test]
fn diagnostics_are_best_effort_when_stderr_and_stdout_are_closed() {
    use std::{os::{fd::OwnedFd, unix::net::UnixStream}, process::Stdio};
    fn closed() -> Stdio {
        let (writer, reader) = UnixStream::pair().unwrap(); drop(reader);
        let fd: OwnedFd = writer.into(); Stdio::from(fd)
    }
    let cases: &[(&str, &[&str], i32)] = &[
        (env!("CARGO_BIN_EXE_vade-harness"), &["unknown-command"], 2),
        (env!("CARGO_BIN_EXE_vade-harness"), &["doctor", "--unknown"], 2),
        (env!("CARGO_BIN_EXE_vade-harness"), &["direct", "--endpoint", "http://127.0.0.1:1", "--api-key-file", "/missing-synthetic-key", "--model", "mock", "--prompt", "fixture"], 1),
        (env!("CARGO_BIN_EXE_vade-harness"), &["setup", "generic", "--endpoint", "http://127.0.0.1:1", "--model", "fixture", "--target", "direct"], 0),
        (env!("CARGO_BIN_EXE_vade-gateway"), &["--unknown"], 1),
        (env!("CARGO_BIN_EXE_vade-gateway"), &["--upstream", "http://127.0.0.1:1", "--api-key-file", "/missing-synthetic-key"], 1),
    ];
    for (binary, args, code) in cases {
        let output = home::command(binary).args(*args).env_remove("VADE_API_KEY_FILE")
            .stdin(Stdio::null()).stdout(closed()).stderr(closed()).output().unwrap();
        assert_eq!(output.status.code(), Some(*code), "{args:?}: {:?}", output.status);
    }
}
