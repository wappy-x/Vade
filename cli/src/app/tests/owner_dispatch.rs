//! Complete controller lifecycle after either kind of local setup. A child
//! test process isolates selection environment from the other unit tests.
use super::*;

// Capture to files so a child or inherited pipe cannot prevent the parent
// enforcing its deadline. The guard also reaps/kills on assertion unwinding.
struct SupervisedChild(Option<std::process::Child>);
impl Drop for SupervisedChild {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn kill(pid: i32, signal: i32) -> i32;
        }
        if let Some(child) = &mut self.0 {
            unsafe {
                kill(-(child.id() as i32), 9);
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn bounded_child_output(
    command: &mut std::process::Command,
    timeout: std::time::Duration,
) -> std::io::Result<std::process::Output> {
    use std::{
        io::{Read, Seek},
        os::unix::process::CommandExt,
        process::Stdio,
        time::Instant,
    };
    let dir = std::env::temp_dir().join(format!(
        "vade-controller-lifecycle-{}-{}",
        std::process::id(),
        crate::text::random_hex(8).unwrap()
    ));
    fs::create_dir(&dir)?;
    let open = |name| {
        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(dir.join(name))
    };
    let mut stdout = open("stdout")?;
    let mut stderr = open("stderr")?;
    command
        .stdin(Stdio::null())
        .stdout(stdout.try_clone()?)
        .stderr(stderr.try_clone()?)
        .process_group(0);
    let mut child = SupervisedChild(Some(command.spawn()?));
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.0.as_mut().unwrap().try_wait()? {
            // Reaped: stop owning its PID so cleanup cannot signal a reused ID.
            child.0 = None;
            break Some(status);
        }
        if Instant::now() >= deadline {
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    drop(child);
    let captured = |file: &mut fs::File| {
        file.rewind()?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        std::io::Result::Ok(bytes)
    };
    let out = captured(&mut stdout)?;
    let err = captured(&mut stderr)?;
    drop(stdout);
    drop(stderr);
    fs::remove_dir_all(&dir)?;
    match status {
        Some(status) => Ok(std::process::Output {
            status,
            stdout: out,
            stderr: err,
        }),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!(
                "controller lifecycle child exceeded {timeout:?}; stdout: {}; stderr: {}",
                String::from_utf8_lossy(&out),
                String::from_utf8_lossy(&err)
            ),
        )),
    }
}

#[test]
fn lifecycle_child_deadline_terminates_a_stuck_child() {
    let started = std::time::Instant::now();
    let mut command = std::process::Command::new("/bin/sh");
    command.args(["-c", "exec sleep 60"]);
    let error = bounded_child_output(&mut command, std::time::Duration::from_millis(500))
        .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    assert!(error.to_string().contains("controller lifecycle child exceeded"), "{error}");
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

#[test]
fn alternate_start_and_every_lifecycle_command_keep_the_explicit_controller() {
    const CHILD: &str = "VADE_TEST_CONTROLLER_LIFECYCLE";
    if std::env::var_os(CHILD).is_none() {
        let _process_guard = crate::runner::process_test_lock();
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "app::tests::owner_dispatch::alternate_start_and_every_lifecycle_command_keep_the_explicit_controller", "--nocapture"])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env(CHILD, "1");
        let output = bounded_child_output(&mut command, std::time::Duration::from_secs(30)).unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    for owner in [false, true] {
        let mut f = Fixture::new(2, false);
        let local = fs::canonicalize(&f.dir).unwrap().join("separate-local");
        fs::create_dir_all(local.join("state")).unwrap();
        let untouched = if owner {
            local.join("config")
        } else {
            local.join("setup.conf")
        };
        fs::write(&untouched, "local configuration must remain untouched\n").unwrap();
        unsafe {
            std::env::set_var("HOME", &local);
            std::env::set_var("VADE_HOME", &local);
            std::env::set_var(
                "VADE_CONFIG",
                fs::canonicalize(&f.dir).unwrap().join("config"),
            );
        }
        let args = ["up", "--fp8-legacy", "--context", "8192"].map(str::to_owned);
        let start = crate::cli::parse(&args).unwrap().command;
        let mut commands = vec![
            start,
            Command::Chat,
            Command::Status,
            Command::Logs,
            Command::Down,
        ];
        for command in commands.drain(..) {
            assert!(
                !crate::launch::handles(&command),
                "owner={owner}, {command:?}"
            );
            assert!(
                !crate::run::handles(&command, &local),
                "owner={owner}, {command:?}"
            );
            f.app.execute(command.clone()).unwrap();
            if matches!(command, Command::Up(_)) {
                let running = Session::load(&f.app.config.state_dir).unwrap().unwrap();
                assert_eq!(running.context, 8192);
                assert_eq!(running.phase, "running");
                assert_eq!(f.app.runner.started.len(), 2);
            }
        }
        assert!(f.app.runner.started.is_empty());
        assert!(Session::load(&f.app.config.state_dir).unwrap().is_none());
        assert!(
            f.app
                .runner
                .calls
                .iter()
                .any(|exec| exec.label == "Vade chat")
        );
        assert_eq!(
            fs::read_to_string(&untouched).unwrap(),
            "local configuration must remain untouched\n"
        );
        assert!(!local.join("state/run.conf").exists());
        assert!(!local.join("run.conf").exists());
    }
    // A resident launched through the implicit default config keeps its
    // lifecycle until shutdown, even if a different local start is attempted.
    let mut f = Fixture::new(2, false);
    let base = fs::canonicalize(&f.dir).unwrap();
    let local = base.join("new-local");
    let xdg = base.join("xdg");
    fs::create_dir_all(xdg.join("vade")).unwrap();
    fs::write(
        xdg.join("vade/config"),
        format!(
            "[global]\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nstate_dir = {}\n[machine fixture]\nssh = fixture-no-network\n",
            fs::canonicalize(&f.app.config.state_dir).unwrap().display()
        ),
    )
    .unwrap();
    unsafe {
        std::env::remove_var("VADE_CONFIG");
        std::env::set_var("VADE_HOME", &local);
        std::env::set_var("XDG_CONFIG_HOME", &xdg);
    }
    let start = crate::cli::parse(&["up".into(), "--fp8-legacy".into(), "--context".into(), "8192".into()])
        .unwrap()
        .command;
    assert!(!crate::launch::handles(&start));
    f.app.execute(start).unwrap();
    let record = fs::read(f.app.config.state_dir.join("session.conf")).unwrap();
    assert!(
        crate::launch::run(Command::Launch(Default::default()), true)
            .unwrap_err()
            .contains("controller")
    );
    let legacy = Command::Run(crate::cli::RunOptions::default());
    assert!(
        crate::run::validate_start(&legacy)
            .unwrap_err()
            .contains("controller")
    );
    assert!(!local.exists());
    assert_eq!(
        fs::read(f.app.config.state_dir.join("session.conf")).unwrap(),
        record
    );
    assert!(!crate::launch::handles(&Command::Install));
    for command in [Command::Chat, Command::Status, Command::Logs, Command::Down] {
        assert!(!crate::launch::handles(&command));
        assert!(!crate::run::handles(&command, &local));
        f.app.execute(command).unwrap();
    }
    assert!(Session::load(&f.app.config.state_dir).unwrap().is_none());
    crate::launch::run(Command::Launch(Default::default()), true).unwrap();
    crate::run::validate_start(&legacy).unwrap();
}
