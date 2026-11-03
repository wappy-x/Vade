//! Host-only subprocess, progress and interruption tests.
use super::*;
#[test]
fn only_transient_probe_failures_suppress_the_retained_log_warning() {
    let _guard = process_test_lock();
    if std::env::var_os("VADE_TEST_TRANSIENT_PROBE").is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "runner::tests::only_transient_probe_failures_suppress_the_retained_log_warning"])
            .env("VADE_TEST_TRANSIENT_PROBE", "1").output().unwrap();
        assert!(output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(!stderr.contains("expected probe"));
        assert!(stderr.contains("ordinary failure logs retained privately at"));
        return;
    }
    for transient in [true, false] {
        let mut exec = Exec::new(if transient { "expected probe" } else { "ordinary failure" }, "sh", vec!["-c".into(), "exit 7".into()]);
        exec.transient_probe = transient;
        assert!(!SystemRunner { dry: false }.run(&exec).unwrap().success);
    }
}
#[test]
fn readiness_keeps_only_the_latest_private_diagnostic() {
    let _guard = process_test_lock();
    let root = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-probe-{}", crate::text::random_hex(8).unwrap()));
    crate::text::private_dir(&root).unwrap();
    let log = root.join("last.log");
    for reason in ["connection refused", "wrong model"] {
        let mut exec = Exec::new("readiness", "sh", vec!["-c".into(), format!("printf '%s' '{}'; printf '%s' '{}' >&2; exit 7", reason, reason)]);
        exec.transient_probe = true;
        exec.probe_log = Some(log.clone());
        assert!(!SystemRunner { dry: false }.run(&exec).unwrap().success);
        let diagnostic = std::fs::read_to_string(&log).unwrap();
        assert!(diagnostic.contains(reason));
        if reason == "wrong model" { assert!(!diagnostic.contains("connection refused")); }
    }
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(std::fs::metadata(&log).unwrap().permissions().mode() & 0o777, 0o600);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn progress_is_periodic_terminal_safe_and_does_not_touch_command_data() {
    let mut progress = Progress::new("build\n\x1b[31m");
    assert!(progress.line(Duration::from_secs(29)).is_none());
    let first = progress.line(Duration::from_secs(30)).unwrap();
    assert!(first.contains("30s elapsed") && !first.contains('\n') && !first.contains('\x1b'));
    assert!(progress.line(Duration::from_secs(59)).is_none());
    assert!(progress.line(Duration::from_secs(60)).unwrap().contains("60s elapsed"));
    assert!(progress.line(Duration::from_secs(61)).is_none());
    let label = "x".repeat(5000);
    let mut long = Progress::new(&label);
    assert!(long.line(Duration::from_secs(30)).unwrap().len() < 1100);
}
#[test]
fn real_interrupt_is_prompt_and_recorded_cleanup_can_still_run() {
    let _process_guard = process_test_lock();
    if std::env::var_os("VADE_TEST_INTERRUPT_CONTROL").is_none() {
        let result = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "runner::tests::real_interrupt_is_prompt_and_recorded_cleanup_can_still_run"])
            .env("VADE_TEST_INTERRUPT_CONTROL", "1").output().unwrap();
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stdout)); return;
    }
    let _guard = Interruptions::install();
    unsafe extern "C" { fn raise(number: i32) -> i32; }
    unsafe { raise(2); }
    let mut runner = SystemRunner { dry: false };
    let exec = Exec::new("cleanup control", "sh", vec!["-c".into(), "printf cleaned".into()]);
    assert!(runner.run(&exec).unwrap_err().contains("Interrupted"));
    let before = Instant::now(); runner.pause(Duration::from_secs(180));
    assert!(before.elapsed() < Duration::from_millis(100));
    assert_eq!(runner.cleanup(&exec).unwrap().stdout, "cleaned");
}
#[test]
fn timeout_stops_owned_descendants_and_preserves_foreign_child() {
    let _process_guard = process_test_lock();
    let root = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-job-test-{}", crate::text::random_hex(8).unwrap()));
    crate::text::private_dir(&root).unwrap();
    let mut foreign = Command::new("sleep").arg("30").spawn().unwrap();
    let marker = root.join("child");
    let mut exec = Exec::new("owned timeout control", "sh", vec!["-c".into(), "sleep 30 & printf '%s\\n' \"$!\" > \"$1\"; wait".into(), "vade-test".into(), marker.to_string_lossy().into_owned()]);
    exec.timeout = Duration::from_millis(250);
    let result = SystemRunner { dry: false }.run(&exec);
    let foreign_alive = foreign.try_wait().unwrap().is_none();
    let _ = foreign.kill(); let _ = foreign.wait();
    let pid = std::fs::read_to_string(&marker).unwrap().trim().to_string();
    let until = Instant::now() + Duration::from_secs(3); let mut dead = false;
    while Instant::now() < until {
        let status = Command::new("ps").args(["-o", "stat=", "-p", &pid]).output().unwrap();
        let state = String::from_utf8_lossy(&status.stdout);
        if state.trim().is_empty() || state.trim().starts_with('Z') { dead = true; break; }
        std::thread::sleep(Duration::from_millis(20));
    }
    std::fs::remove_dir_all(root).unwrap();
    assert!(result.unwrap_err().contains("timed out") && foreign_alive && dead);
}
#[test]
fn stream_delivers_lines_and_bounds_inherited_pipe_lifetime() {
    let _process_guard = process_test_lock();
    let mut runner = SystemRunner { dry: false }; let mut lines = Vec::new();
    let exec = Exec::new("stream control", "sh", vec!["-c".into(), "printf 'one\\ntwo\\n'".into()]);
    let output = runner.stream(&exec, &mut |line| { lines.push(line.to_string()); Ok(()) }).unwrap();
    assert!(output.success); assert_eq!(lines, ["one", "two"]);
    let mut exec = Exec::new("stream timeout control", "sh", vec!["-c".into(), "sleep 30 & exit 0".into()]); exec.timeout = Duration::from_millis(250);
    assert!(runner.stream(&exec, &mut |_| Ok(())).unwrap_err().contains("timed out"));
}
#[test]
fn previews_propagate_output_errors_before_running_commands() {
    use crate::output::{self, testing::Capture};
    for kind in [std::io::ErrorKind::BrokenPipe, std::io::ErrorKind::PermissionDenied] {
        output::with_writer(Capture { fail_on: Some(("$ ", kind)), ..Capture::default() }, || {
            let mut runner = SystemRunner { dry: true };
            let command = Exec::new("preview only", "missing-preview-program", vec![]);
            assert!(runner.run(&command).unwrap_err().contains("standard output"));
            assert_eq!(output::failure(), Some(kind));
        });
    }
}
#[test]
fn stream_consumer_error_stops_and_reaps_the_owned_process_promptly() {
    let _guard = process_test_lock();
    let root = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-consumer-{}", crate::text::random_hex(8).unwrap()));
    crate::text::private_dir(&root).unwrap();
    let fifo = root.join("release");
    assert!(Command::new("mkfifo").arg(&fifo).status().unwrap().success());
    let mut release = std::fs::OpenOptions::new().read(true).write(true).open(&fifo).unwrap();
    // The owned child and its pipe-holding descendant cannot finish naturally
    // until the test releases them.
    // Readiness excludes process startup from the bounded cancellation wait.
    let mut exec = Exec::new("consumer cancellation", "sh", vec!["-c".into(),
        "read -r release < \"$1\" & child=$!; printf '%s\\n' \"$$\"; wait \"$child\"".into(), "vade-test".into(), fifo.to_string_lossy().into_owned()]);
    exec.timeout = Duration::from_secs(120);
    let (ready_send, ready) = std::sync::mpsc::channel();
    let (done_send, done) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut delivered = 0;
        let result = SystemRunner { dry: false }.stream(&exec, &mut |line| {
            delivered += 1; let _ = ready_send.send(line.to_string());
            Err("consumer output failed".into())
        });
        let _ = done_send.send((result, delivered));
    });
    let timeout = Duration::from_secs(60);
    let pid = ready.recv_timeout(timeout);
    let completed = done.recv_timeout(timeout);
    // Inspect before releasing the gate: cleanup must not make a premature
    // stream return look like successful cancellation and reaping.
    let status = pid.as_ref().ok().filter(|_| completed.is_ok()).map(|pid|
        Command::new("ps").args(["-o", "stat=", "-p", pid]).output());
    release.write_all(b"release\n").unwrap();
    if completed.is_ok() || done.recv_timeout(timeout).is_ok() { worker.join().unwrap(); }
    drop(release);
    std::fs::remove_dir_all(root).unwrap();
    assert!(pid.expect("owned child did not become ready").parse::<u32>().unwrap() > 0);
    let (result, delivered) = completed.expect("consumer cancellation did not finish while the owned child was gated");
    let error = result.unwrap_err();
    assert!(error.contains("consumer output failed"));
    assert_eq!(delivered, 1);
    let status = status.expect("missing owned child inspection").unwrap();
    assert!(status.stderr.is_empty(), "cannot inspect owned child: {}", String::from_utf8_lossy(&status.stderr));
    assert!(String::from_utf8_lossy(&status.stdout).trim().is_empty(), "owned child was not reaped");
}
#[test]
fn process_test_lock_recovers_from_poisoning() {
    // Poison the shared lock in a disposable thread, then prove the next
    // acquisition still succeeds instead of failing with PoisonError.
    let panicking = std::thread::spawn(|| {
        let _guard = process_test_lock();
        panic!("deliberate test poisoning (expected)");
    });
    assert!(panicking.join().is_err());
    assert!(PROCESS_TEST_LOCK.is_poisoned());
    let _guard = process_test_lock();
}

#[test]
fn parallel_jobs_both_start_before_either_can_finish() {
    let _guard = process_test_lock();
    let dir = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-parallel-{}", crate::text::random_hex(8).unwrap()));
    crate::text::private_dir(&dir).unwrap();
    let job = |own: &str, peer: &str| {
        let mut exec = Exec::new("parallel test", "/bin/sh", vec!["-c".into(),
            "touch \"$1\"; n=0; while ! test -e \"$2\"; do n=$((n+1)); test $n -lt 200 || exit 7; sleep 0.01; done; printf ready".into(),
            "parallel-test".into(), dir.join(own).to_string_lossy().into_owned(), dir.join(peer).to_string_lossy().into_owned()]);
        exec.timeout = Duration::from_secs(5); exec
    };
    let (a, b) = SystemRunner { dry: false }.parallel(&job("a", "b"), &job("b", "a")).unwrap();
    assert!(a.success && b.success); assert_eq!((&*a.stdout, &*b.stdout), ("ready", "ready"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn an_ssh_failure_shows_the_last_line_and_the_fix_for_that_cause() {
    let keyscan = "Run ssh user@host-b true and accept the key, or run ssh-keyscan -H host-b >> ~/.ssh/known_hosts if host-b is the real host name, then run vade doctor.";
    let stop = SshStop::from_output("user@host-b", "No ED25519 host key is known for host-b and you have requested strict checking.\nHost key verification failed.\n").unwrap();
    assert_eq!((stop.error.as_str(), stop.fix.as_str()), ("ssh user@host-b failed: Host key verification failed. Vade never edits ~/.ssh.", keyscan));
    // A saved key that changed is not a key to add: the fix asks the person first.
    let stop = SshStop::from_output("host-b", "@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\nHost key verification failed.\n").unwrap();
    assert!(stop.fix.starts_with("Ask who runs host-b why its key changed.") && stop.fix.contains("run the ssh-keygen -R command that it prints") && !stop.fix.contains("ssh-keyscan"), "{stop:?}");
    let stop = SshStop::from_output("host-b", "user@host-b: Permission denied (publickey).\n").unwrap();
    assert_eq!(stop.error, "ssh host-b failed: user@host-b: Permission denied (publickey). Vade never edits ~/.ssh.");
    assert_eq!(stop.fix, "Run ssh-copy-id host-b, then run vade doctor.");
    let stop = SshStop::from_output("host-b", "ssh: connect to host host-b port 22: Connection refused\n\n").unwrap();
    assert_eq!(stop.error, "ssh host-b failed: ssh: connect to host host-b port 22: Connection refused.");
    assert_eq!(stop.fix, "Run ssh host-b true to see the full SSH error, then run vade doctor.");
    // The text stays one short plain line, and no text means no claim.
    assert!(SshStop::from_output("host-b", &format!("\x1b[31m{}\n", "x".repeat(900))).unwrap().error.len() < 240);
    assert_eq!(SshStop::from_output("host-b", " \n"), None);
}

#[test]
fn failed_and_unfinished_ssh_checks_point_to_the_install_guide() {
    struct Fails { calls: usize }
    impl Runner for Fails {
        fn run(&mut self, _: &Exec) -> Result<Output, String> {
            self.calls += 1;
            if self.calls == 1 { Err("probe did not finish".into()) } else { Ok(Output::default()) }
        }
        fn interactive(&mut self, _: &Exec) -> Result<bool, String> { unreachable!() }
        fn port_free(&mut self, _: u16) -> Result<(), String> { unreachable!() }
        fn pause(&mut self, _: Duration) { unreachable!() }
    }
    let _host = fake_host(Host::default());
    let mut runner = Fails { calls: 0 };
    let error = check_ssh(&mut runner, &[("first", "first"), ("second", "second")]).unwrap_err();
    assert_eq!(runner.calls, 2);
    assert!(error.contains("first: SSH did not finish.") && error.contains("second: SSH failed."), "{error}");
    assert_eq!(error.matches("Read docs/INSTALL.md for the SSH check command.").count(), 2, "{error}");
    assert!(!error.contains("Run ssh "), "{error}");
}

#[test]
fn only_the_ssh_check_shows_standard_error_and_it_runs_nothing_remote() {
    let exec = ssh_check("host-b").unwrap();
    assert_eq!((exec.program.as_str(), exec.args[1].as_str()), ("/bin/sh", "ssh \"$@\" 2>&1 >/dev/null; test $? -ne 255"));
    assert_eq!(exec.args[exec.args.len() - 2..], ["host-b", "true"]);
    assert!(exec.args.windows(2).any(|pair| pair == ["-o", "BatchMode=yes"]) && exec.transient_probe);
    assert_eq!(ssh_target(&ssh("host-b", "true".into(), "step")), Some("host-b"));
    assert_eq!(ssh_target(&copy_to("step", "host-b", "~/stage", Vec::new(), vec!["file".into()])), Some("host-b"));
    assert_eq!(ssh_target(&Exec::new("step", "git", Vec::new())), None);
}
