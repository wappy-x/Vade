//! SSH failure advice after public machine labels replace private destinations.
use super::*;

const TARGETS: [&str; 3] = ["user@192.0.2.10", "203.0.113.10", "fixture-ssh"];
const KEY_ERROR: &str = "Host key verification failed.\n";

fn assert_fix(target: &str, printed: &str) {
    assert!(printed.contains("Host key verification failed."), "{printed}");
    assert_advice(target, printed);
    if target == "fixture-ssh" { assert!(printed.contains("Run ssh fixture-ssh true and accept the key"), "{printed}"); }
}

fn assert_advice(target: &str, printed: &str) {
    assert!(!printed.contains("192.0.2.") && !printed.contains("203.0.113."), "{printed}");
    if target == "fixture-ssh" {
        assert!(printed.contains("ssh fixture-ssh true"), "{printed}");
        assert!(!printed.contains("Read docs/INSTALL.md"), "{printed}");
    } else {
        assert!(printed.contains("Read docs/INSTALL.md for the SSH check command."), "{printed}");
        assert!(!printed.contains(" true") && !printed.contains("ssh-keyscan") && !printed.contains(target), "{printed}");
    }
}

#[test]
fn doctor_fallback_ssh_fixes_use_the_destination_kind() {
    let _host = runner::fake_host(runner::Host::default());
    for target in TARGETS {
        for failure in ["runner", "ssh", "script"] {
            let mut f = Fixture::fp4(1, false);
            f.app.config.machines[0].ssh = target.into();
            f.app.runner.probe = Some((false, if failure == "script" { "reached=1\n" } else { "" }));
            if failure == "runner" { f.app.runner.fail = Some("a prerequisites".into()); }
            let names = crate::config::PublicNames::controller(&f.app.config);
            let capture = crate::output::testing::Capture::default();
            let result = crate::output::with_writer(capture.clone(), || names.run(|| f.app.doctor_with(Ok(false))));
            assert!(result.is_err());
            let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
            assert_advice(target, &printed);
            if failure != "runner" {
                let machine = f.app.config.machines[0].clone();
                let error = names.run(|| f.app.machine_checks_for_build(&machine, true)).unwrap_err();
                assert_advice(target, &error);
            }
        }
    }
}

#[test]
fn staged_build_ssh_fix_uses_the_destination_kind() {
    let _host = runner::fake_host(runner::Host::default());
    for target in TARGETS {
        let mut f = Fixture::fp4(1, false);
        f.app.config.machines[0].ssh = target.into();
        f.app.runner.hashes = Some((false, ""));
        let machine = f.app.config.machines[0].clone();
        let artifact = f.app.manifest().unwrap().artifacts["a"].clone();
        let names = crate::config::PublicNames::controller(&f.app.config);
        let error = names.run(|| f.app.verify_staged(&machine, &artifact)).unwrap_err();
        assert_advice(target, &error);
    }
}

#[test]
fn doctor_file_check_ssh_fix_uses_the_destination_kind() {
    let _host = runner::fake_host(runner::Host::default());
    for target in TARGETS {
        let mut f = Fixture::fp4(1, false);
        f.app.config.machines[0].ssh = target.into();
        f.app.runner.fail = Some("a doctor files".into());
        let machine = f.app.config.machines[0].clone();
        let names = crate::config::PublicNames::controller(&f.app.config);
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || names.run(|| {
            let mut report = crate::checks::Report::new("controller", false);
            let state = crate::doctor::RemoteState { owned: false, memory: None, split: false, cache_default: None };
            crate::doctor::machine(&mut report, &mut f.app.runner, &machine, &[], false, state)?;
            report.finish()
        }));
        assert!(result.is_err());
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        assert_advice(target, &printed);
    }
}

#[test]
fn doctor_ssh_stop_fix_uses_the_destination_kind() {
    let _host = runner::fake_host(runner::Host::default());
    for target in TARGETS {
        let mut f = Fixture::fp4(1, false);
        f.app.config.machines[0].ssh = target.into();
        f.app.runner.probe = Some((false, ""));
        f.app.runner.ssh_says = Some(KEY_ERROR);
        let machine = f.app.config.machines[0].clone();
        let names = crate::config::PublicNames::controller(&f.app.config);
        // The install error uses Probe.error. Doctor prints SshStop.fix in a row.
        let error = names.run(|| f.app.machine_checks_for_build(&machine, true)).unwrap_err();
        assert_fix(target, &error);
        let capture = crate::output::testing::Capture::default();
        let result = crate::output::with_writer(capture.clone(), || names.run(|| f.app.doctor_with(Ok(false))));
        assert!(result.is_err());
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        assert_fix(target, &printed);
        assert!(printed.lines().any(|line| line.starts_with("FAIL  a ssh")), "{printed}");
    }
}

#[test]
fn worker_ssh_stop_fix_uses_the_destination_kind() {
    let _host = runner::fake_host(runner::Host::default());
    for target in TARGETS {
        let mut f = Fixture::fp4(1, false);
        f.app.config.machines[0].ssh = target.into();
        f.app.runner.silent_probe = true;
        f.app.runner.ssh_says = Some(KEY_ERROR);
        let session = Session::new(&f.app.config, &fp4_options(&[]), &f.app.manifest().unwrap(), "a".repeat(64)).unwrap();
        let names = crate::config::PublicNames::controller(&f.app.config);
        let error = names.run(|| f.app.worker(&session, &session.ranks[0], "probe", Fields::new())).unwrap_err();
        assert_fix(target, &error);
    }
}

#[test]
fn checked_ssh_stop_fix_uses_the_destination_kind() {
    let _host = runner::fake_host(runner::Host::default());
    for target in TARGETS {
        let mut f = Fixture::fp4(1, false);
        f.app.config.machines[0].ssh = target.into();
        f.app.runner.hashes = Some((false, ""));
        f.app.runner.ssh_says = Some(KEY_ERROR);
        let names = crate::config::PublicNames::controller(&f.app.config);
        let exec = runner::ssh(target, "true".into(), "a artifact hashes");
        let error = names.run(|| f.app.checked(exec, "The check failed.")).unwrap_err();
        assert_fix(target, &error);
    }
}

struct CopyFailure { inner: FakeRunner }
impl Runner for CopyFailure {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        if exec.label == "a prebuilt copy" && exec.args.iter().any(|arg| arg.ends_with("/flash-serve")) {
            self.inner.calls.push(exec.clone());
            return Ok(Output { success: false, stdout: String::new() });
        }
        self.inner.run(exec)
    }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> { unreachable!() }
    fn port_free(&mut self, _: u16) -> Result<(), String> { unreachable!() }
    fn pause(&mut self, _: Duration) { unreachable!() }
}

#[test]
fn prebuilt_ssh_stop_fix_uses_the_destination_kind() {
    let _host = runner::fake_host(runner::Host::default());
    for target in TARGETS {
        let mut f = install_tests::with_bundle();
        f.app.config.machines[0].ssh = target.into();
        f.app.runner.ssh_says = Some(KEY_ERROR);
        let bundle = crate::prebuilt::Bundle::find(f.app.config.prebuilt_dir.as_deref(), &"a".repeat(40), &"b".repeat(40)).unwrap().unwrap();
        let mut app = App { config: f.app.config.clone(), runner: CopyFailure { inner: std::mem::take(&mut f.app.runner) }, dry: false };
        let machine = app.config.machines[0].clone();
        let names = crate::config::PublicNames::controller(&app.config);
        let error = names.run(|| app.install_prebuilt(&machine, &bundle, "~/stage/build")).unwrap_err();
        assert_fix(target, &error);
        assert_eq!(app.runner.inner.calls.iter().filter(|exec| exec.label == "a prebuilt copy").count(), 2);
    }
}
