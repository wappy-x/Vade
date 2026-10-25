//! Host-only build ownership, cancellation and recovery tests.
use super::*;
#[test]
fn helper_process() {
    if let (Ok(mode), Some(job)) = (
        std::env::var("VADE_TEST_BUILD_MODE"),
        std::env::var_os("VADE_TEST_BUILD_JOB"),
    ) {
        if mode == "__build-driver" {
            let _ = run(Path::new(&job));
        } else if mode == "__build-recover-fifo" {
            assert!(recover(Path::new(&job)).unwrap_err().contains("regular file"));
        } else {
            helper(&mode, Path::new(&job)).unwrap();
        }
    }
}
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("vade-build-test-{}", text::random_hex(8).unwrap()));
        text::private_dir(&root).unwrap();
        Self(root)
    }
    fn job(&self, exit: bool, timeout: Duration) -> PathBuf {
        // A shell is the mock compiler, and sleep is its mock descendant.
        let script = format!(
            "sleep 30 & printf '%s\\n' \"$!\" > \"$1\"; {}",
            if exit { "exit 7" } else { "wait" }
        );
        let mut exec = Exec::new(
            "mock build",
            "sh",
            vec![
                "-c".into(),
                script,
                "control".into(),
                self.0.join("descendant").to_string_lossy().into_owned(),
            ],
        );
        exec.timeout = timeout;
        prepare(&self.0, &mut exec).unwrap();
        let job = exec.job.unwrap();
        assert!(read(&job, "intent.conf").unwrap().is_some());
        assert!(read(&job, "anchor.conf").unwrap().is_none());
        assert!(!self.0.join("descendant").exists());
        job
    }
    fn assert_stopped(&self, job: &Path) {
        assert!(read(job, "stopped.conf").unwrap().is_some());
        let anchor = read(job, "anchor.conf").unwrap().unwrap();
        assert_eq!(anchor["pid"], anchor["pgid"]);
        assert_eq!(anchor["pid"], anchor["sid"]);
        assert!(!group_active(anchor["pgid"].parse().unwrap()).unwrap());
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Foreign(Child);
impl Drop for Foreign {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
fn mock_compiler_failure_timeout_and_cli_loss_stop_whole_recorded_group() {
    let _process_guard = crate::runner::process_test_lock();
    let mut foreign = Foreign(Command::new("sleep").arg("30").spawn().unwrap());
    let failure = Fixture::new();
    let job = failure.job(true, Duration::from_secs(5));
    assert!(!run(&job).unwrap().success);
    failure.assert_stopped(&job);
    let timeout = Fixture::new();
    let job = timeout.job(false, Duration::from_millis(300));
    assert!(run(&job).unwrap_err().contains("timed out"));
    timeout.assert_stopped(&job);
    let lost = Fixture::new();
    let job = lost.job(false, Duration::from_secs(30));
    let mut cli = helper_command("__build-driver", &job)
        .unwrap()
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !lost.0.join("descendant").exists() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
    // SIGKILL the disposable CLI driver while mock compiler descendants run.
    // The supervisor must clean on EOF without a recovery/cancel request.
    cli.kill().unwrap();
    cli.wait().unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while read(&job, "stopped.conf").unwrap().is_none() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
    recover(&lost.0).unwrap();
    lost.assert_stopped(&job);
    assert!(foreign.0.try_wait().unwrap().is_none());
}
#[test]
fn recovery_cancels_active_and_unspawned_jobs_without_unrecorded_signals() {
    let _process_guard = crate::runner::process_test_lock();
    let f = Fixture::new();
    let job = f.job(false, Duration::from_secs(30));
    let mut supervisor = helper_command("__build-supervisor", &job)
        .unwrap()
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !f.0.join("descendant").exists() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
    recover(&f.0).unwrap();
    supervisor.wait().unwrap();
    f.assert_stopped(&job);
    let pending = Fixture::new();
    let job = pending.job(false, Duration::from_secs(30));
    recover(&pending.0).unwrap();
    assert!(!pending.0.join("descendant").exists());
    // A late helper respects the durable cancellation tombstone.
    assert!(
        run(&job)
            .unwrap_err()
            .contains("cancelled before execution")
    );
    assert!(read(&job, "anchor.conf").unwrap().is_none());
}
#[test]
fn recovery_clears_a_lost_supervisor_record_only_when_its_processes_are_gone() {
    let _process_guard = crate::runner::process_test_lock();
    // A supervisor that failed or was killed after registering, with no anchor.
    let f = Fixture::new();
    let job = f.job(false, Duration::from_secs(30));
    let intent = read(&job, "intent.conf").unwrap().unwrap();
    let mut gone = Command::new("sleep").arg("30").spawn().unwrap();
    let mut record = snapshot(gone.id()).unwrap();
    record.insert("token".into(), intent["token"].clone());
    save(&job, "supervisor.conf", &record).unwrap();
    gone.kill().unwrap();
    gone.wait().unwrap();
    recover(&f.0).unwrap();
    assert!(read(&job, "stopped.conf").unwrap().unwrap()["error"].contains("supervisor was lost"));
    recover(&f.0).unwrap();
    assert!(run(&job).unwrap_err().contains("cancelled before execution"));
    assert!(!f.0.join("descendant").exists());
    // A free lock with a live recorded identity is not proof; keep the record.
    let live = Fixture::new();
    let job = live.job(false, Duration::from_secs(30));
    let mut record = snapshot(std::process::id()).unwrap();
    record.insert("token".into(), intent["token"].clone());
    save(&job, "supervisor.conf", &record).unwrap();
    assert!(recover(&live.0).unwrap_err().contains("supervisor was lost"));
    assert!(read(&job, "stopped.conf").unwrap().is_none());
    // A recorded anchor from an earlier boot cannot still run.
    save(&job, "anchor.conf", &Fields::from([
        ("pid".into(), std::process::id().to_string()),
        ("boot".into(), "an-earlier-boot".into()),
        ("birth".into(), "0".into()),
    ])).unwrap();
    record.insert("boot".into(), "an-earlier-boot".into());
    save(&job, "supervisor.conf", &record).unwrap();
    recover(&live.0).unwrap();
    assert!(read(&job, "stopped.conf").unwrap().is_some());
}
#[test]
fn recovery_waits_for_a_killed_supervisors_build_then_clears_it() {
    let _process_guard = crate::runner::process_test_lock();
    let f = Fixture::new();
    // The mock compiler waits for its `sleep 30` descendant.
    let job = f.job(false, Duration::from_secs(30));
    let mut supervisor = Foreign(helper_command("__build-supervisor", &job)
        .unwrap()
        .stdin(Stdio::piped())
        .spawn()
        .unwrap());
    let deadline = Instant::now() + Duration::from_secs(5);
    while !f.0.join("descendant").exists() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
    supervisor.0.kill().unwrap();
    supervisor.0.wait().unwrap();
    // The compiler still runs under its anchor: keep the record.
    let anchor: u32 = read(&job, "anchor.conf").unwrap().unwrap()["pid"].parse().unwrap();
    assert!(recover(&f.0).unwrap_err().contains("supervisor was lost"));
    assert!(read(&job, "stopped.conf").unwrap().is_none());
    // Once the build ends, the orphaned anchor exits and recovery clears the record.
    let descendant = fs::read_to_string(f.0.join("descendant")).unwrap();
    assert!(Command::new("kill").args(["-9", descendant.trim()]).status().unwrap().success());
    let deadline = Instant::now() + Duration::from_secs(5);
    while group_active(anchor).unwrap() {
        assert!(Instant::now() < deadline, "orphaned anchor kept running");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(read(&job, "result.conf").unwrap().is_some());
    recover(&f.0).unwrap();
    assert!(read(&job, "stopped.conf").unwrap().unwrap()["error"].contains("supervisor was lost"));
}
#[test]
fn recovery_rejects_fifo_records_and_lock_within_bounded_subprocesses() {
    let _process_guard = crate::runner::process_test_lock();
    for name in ["intent.conf", "stopped.conf", "supervisor.conf", "supervisor.lock"] {
        let f = Fixture::new();
        let job = f.job(false, Duration::from_secs(30));
        let path = job.join(name);
        if path.exists() { fs::remove_file(&path).unwrap(); }
        assert!(Command::new("mkfifo").arg(&path).status().unwrap().success());
        let mut child = Foreign(helper_command("__build-recover-fifo", &f.0).unwrap().spawn().unwrap());
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                assert!(status.success(), "recovery should reject {name}");
                break;
            }
            assert!(Instant::now() < deadline, "recovery hung on {name}");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!f.0.join("descendant").exists());
    }
}

#[test]
fn ownership_record_reads_are_bounded_and_keep_invalid_evidence() {
    let f = Fixture::new();
    let path = f.0.join("intent.conf");
    fs::write(&path, vec![b'x'; 1024 * 1024 + 1]).unwrap();
    assert!(read(&f.0, "intent.conf").unwrap_err().contains("exceeds 1 MiB"));
    assert_eq!(fs::metadata(&path).unwrap().len(), 1024 * 1024 + 1);
    fs::write(&path, [0xff]).unwrap();
    assert!(read(&f.0, "intent.conf").unwrap_err().contains("Cannot read"));
    assert_eq!(fs::read(path).unwrap(), [0xff]);
}

#[test]
fn mock_anchor_identity_recheck_refuses_reuse_reboot_and_incomplete_records() {
    let intent = Fields::from([("token".into(), "a".repeat(64))]);
    let mut saved = Fields::from([
        ("pid".into(), "42".into()),
        ("pgid".into(), "42".into()),
        ("sid".into(), "42".into()),
        ("birth".into(), "100".into()),
        ("boot".into(), "boot-a".into()),
        ("token".into(), intent["token"].clone()),
    ]);
    let now = saved.clone();
    check_anchor(&intent, &saved, &now, 42).unwrap();
    for key in ["birth", "boot", "pid", "pgid", "sid"] {
        let mut changed = saved.clone();
        changed.insert(key.into(), "changed".into());
        assert!(check_anchor(&intent, &saved, &changed, 42).is_err());
    }
    let mut wrong_token = saved.clone();
    wrong_token.insert("token".into(), "wrong-token".into());
    assert!(check_anchor(&intent, &wrong_token, &now, 42).is_err());
    for key in ["birth", "boot", "pid", "pgid", "sid", "token"] {
        let value = saved.remove(key).unwrap();
        assert!(check_anchor(&intent, &saved, &now, 42).is_err());
        saved.insert(key.into(), value);
    }
}
