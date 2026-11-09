//! A bundle-only install on two fake hosts: the real install code and real
//! processes, with the tools in tests/fixtures/fake_hosts first on PATH. It runs
//! in this process because a Mac cannot pass the host platform check for a Linux
//! bundle ([`crate::prebuilt::LINUX_HOST`]), and cli/tests/setup.rs runs the real
//! binary on the same tools up to that check.
use super::*;
use crate::runner::{Runner, SystemRunner};
use std::sync::{Arc, Condvar, Mutex};

/// Holds each rank in a step until the other rank has started the same step.
#[derive(Default)]
struct Meet { started: Mutex<BTreeMap<String, usize>>, changed: Condvar }
impl Meet {
    fn step(&self, step: &str) -> Result<(), String> {
        let mut started = self.started.lock().unwrap();
        let count = started.entry(step.into()).or_default();
        *count += 1;
        let goal = count.div_ceil(2) * 2;
        self.changed.notify_all();
        // Only a run with no overlap waits this long.
        let deadline = Instant::now() + Duration::from_secs(60);
        while started[step] < goal {
            let left = deadline.checked_duration_since(Instant::now()).ok_or_else(|| format!("{step}: the ranks ran one after the other."))?;
            started = self.changed.wait_timeout(started, left).unwrap().0;
        }
        Ok(())
    }
}

struct FakeHosts { root: PathBuf, tools: PathBuf, system: SystemRunner, deny_on_copy: Option<String>, meet: Option<Arc<Meet>>, fail: Option<&'static str>, ended: Arc<Mutex<Vec<String>>> }
impl FakeHosts {
    fn new(root: PathBuf) -> Self {
        let tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_hosts");
        for host in ["host-a", "host-b", "host-c", "host-d"] { fs::create_dir_all(root.join(host)).unwrap(); }
        Self { root, tools, system: SystemRunner { dry: false }, deny_on_copy: None, meet: None, fail: None, ended: Arc::default() }
    }
    fn log(&self) -> String { fs::read_to_string(self.root.join("calls.log")).unwrap_or_default() }
    fn know(&self, host: &str) { fs::write(self.root.join(host).join("known"), "").unwrap(); }
}
impl Runner for FakeHosts {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let mut exec = exec.clone();
        // SSH breaks between the checks and the copy of the large files.
        if let Some(host) = self.deny_on_copy.as_ref().filter(|_| exec.program == "rsync" && exec.args.iter().any(|arg| arg.ends_with("flash.ptx"))) {
            fs::write(self.root.join(host).join("denied"), "").unwrap();
        }
        // The default limit is for a real Spark. The fake storage guard is a long chain of
        // forks, which a loaded test computer can hold for more than 60 s.
        exec.timeout = exec.timeout.max(Duration::from_secs(600));
        // The stage holds Linux programs, which this computer may not start.
        if exec.label == "harness doctor availability" { return Ok(Output { success: true, stdout: String::new() }); }
        // The parts of a rank's install that must overlap with the other rank.
        if let (Some(meet), Some((_, step))) = (&self.meet, exec.label.split_once(' ').filter(|(_, step)| ["prebuilt copy", "install preparation"].contains(step))) { meet.step(step)?; }
        if self.fail == Some(exec.label.as_str()) { return Ok(Output { success: false, stdout: String::new() }); }
        exec.env.insert("PATH".into(), format!("{}:/usr/bin:/bin", self.tools.display()));
        exec.env.insert("FAKE_HOSTS".into(), self.root.to_string_lossy().into_owned());
        let output = self.system.run(&exec);
        self.ended.lock().unwrap().push(exec.label.clone());
        output
    }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> { Ok(true) }
    fn port_free(&mut self, _: u16) -> Result<(), String> { Ok(()) }
    fn pause(&mut self, _: Duration) {}
    fn worker(&self) -> Option<Box<dyn Runner + Send>> {
        Some(Box::new(FakeHosts { root: self.root.clone(), tools: self.tools.clone(), system: SystemRunner { dry: false }, deny_on_copy: self.deny_on_copy.clone(), meet: self.meet.clone(), fail: self.fail, ended: self.ended.clone() }))
    }
}

/// An `installed-bundle` config for host-a and host-b, with a floor in `MANIFEST`.
fn app() -> (Fixture, App<FakeHosts>) { sparks(2) }

/// The same config for `count` fake hosts.
fn sparks(count: usize) -> (Fixture, App<FakeHosts>) {
    let f = Fixture::fp4(count, false);
    let bundle = f.dir.join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    let mut conf = format!("format=vade-launch-bundle-v1\ntarget=aarch64-unknown-linux-gnu\nengine_sha={0}\ncli_sha={1}\nharness_sha={1}\n", "d".repeat(40), "e".repeat(40));
    for (name, key) in [("flash-serve", "engine_sha256"), ("flash.ptx", "ptx_sha256"), ("vade", "vade_sha256"), ("vade-harness", "harness_sha256")] {
        let mut bytes = vec![0; 32]; bytes[..6].copy_from_slice(b"\x7fELF\x02\x01"); bytes[18] = 183; bytes[31] = name.len() as u8;
        fs::write(bundle.join(name), &bytes).unwrap();
        conf.push_str(&format!("{key}={}\n", crate::sha256::sha(&bytes)));
    }
    fs::write(bundle.join("launcher-bundle.conf"), conf).unwrap();
    fs::write(bundle.join("MANIFEST"), "format=vade-release-v1\nmin_driver=580\nglibc=2.39\n").unwrap();
    for source in [&f.app.config.engine_source, &f.app.config.harness_source, &f.app.config.cli_source] { fs::remove_dir_all(source).unwrap(); }
    let mut config = f.app.config.clone();
    config.prebuilt_dir = Some(fs::canonicalize(&bundle).unwrap());
    config.installed_bundle = true;
    for (machine, host) in config.machines.iter_mut().zip(["host-a", "host-b", "host-c", "host-d"]) { machine.ssh = host.into(); }
    let runner = FakeHosts::new(fs::canonicalize(&f.dir).unwrap().join("hosts"));
    (f, App { config, dry: false, runner })
}

fn install(app: &mut App<FakeHosts>) -> Result<(), String> {
    crate::prebuilt::LINUX_HOST.set(true);
    let result = app.execute(Command::Install);
    crate::prebuilt::LINUX_HOST.set(false);
    result
}

#[test]
fn a_bundle_only_install_copies_to_each_host_and_shows_the_real_ssh_error() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = app();
    app.runner.know("host-a");
    // host-b has no saved host key: the error is ssh's own, with its fix.
    let error = install(&mut app).unwrap_err();
    assert!(error.starts_with("SSH checks failed. Vade never edits ~/.ssh."), "{error}");
    assert!(error.contains("ssh host-b failed: Host key verification failed.") && error.contains("Read docs/INSTALL.md for the SSH check command."), "{error}");
    assert!(!app.runner.root.join("host-b/.ssh").exists() && !app.runner.root.join("host-a/.ssh").exists());
    // The person runs that command, then the same install passes.
    app.runner.know("host-b");
    install(&mut app).unwrap();
    let log = app.runner.log();
    assert!(!log.contains("rustc was asked") && !log.contains("cargo was asked") && !log.contains("for tool in cargo"), "{log}");
    for host in ["host-a", "host-b"] {
        assert!(log.contains(&format!("rsync {host} vade\n")) && log.contains(&format!("rsync {host} flash-serve flash.ptx\n")), "{log}");
        assert!(log.contains(&format!("sha256sum {host}\n")), "the hash runs on the rank: {log}");
        let builds = app.runner.root.join(host).join("vade-stage/builds");
        let build = fs::read_dir(builds).unwrap().next().unwrap().unwrap().path();
        for name in ["vade", "flash-serve", "flash.ptx"] {
            assert_eq!(fs::read(build.join(name)).unwrap(), fs::read(app.config.prebuilt_dir.as_ref().unwrap().join(name)).unwrap(), "{name}");
        }
    }
}

#[test]
fn a_copy_that_differs_on_the_host_is_refused_there() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = app();
    app.runner.know("host-a");
    app.runner.know("host-b");
    fs::write(app.runner.root.join("host-b/corrupt"), "").unwrap();
    assert!(install(&mut app).unwrap_err().starts_with("b: Prebuilt artifact rejected:"));
    assert!(!app.runner.log().contains("for tool in cargo"), "no source route follows");
}

#[test]
fn a_host_below_the_floor_stops_a_bundle_install_before_any_copy() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = app();
    app.runner.know("host-a");
    app.runner.know("host-b");
    fs::write(app.runner.root.join("host-b/driver"), "575.51.03\n").unwrap();
    let error = install(&mut app).unwrap_err();
    assert!(error.starts_with("b: the bundle does not fit: it needs NVIDIA driver 580 or newer, and this Spark has 575.51.03. A source build needs Rust on each Spark and a checkout on this computer."), "{error}");
    assert!(error.contains("git clone") && error.contains("docs/INSTALL.md#install-rust-source-builds-only"), "{error}");
    assert!(!app.runner.log().lines().any(|row| row.starts_with("rsync ")), "{}", app.runner.log());
}

#[test]
fn a_source_build_that_lacks_rust_names_the_install_commands() {
    let mut f = Fixture::fp4(1, false);
    f.app.runner.probe = Some((false, "reached=1\nmissing=cargo\n"));
    let error = f.app.execute(Command::Install).unwrap_err();
    assert!(error.starts_with("a is missing cargo, and a source build there needs Rust. Run curl --proto '=https'"), "{error}");
    for command in ["sh /tmp/vade-rustup-init.sh -y --default-toolchain none --no-modify-path", "rustup toolchain install 1.99.0 --profile minimal --no-self-update",
        "rustup toolchain install nightly-2026-10-01 --profile minimal --no-self-update --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda"] {
        assert!(error.contains(command), "{error}");
    }
    // Every command stays the same as in docs/INSTALL.md.
    let guide = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/INSTALL.md")).unwrap();
    for command in crate::app::diagnostics::RUST_STEPS.split(" && ") {
        assert!(guide.lines().any(|line| line == command), "{command}");
    }
}

/// A second Spark set on the same controller state, as a larger setup is: host-a
/// and host-b have the bundle from the install before, and host-c has nothing.
#[test]
fn a_new_spark_moves_every_rank_to_one_folder() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = app();
    for host in ["host-a", "host-b", "host-c"] { app.runner.know(host); }
    install(&mut app).unwrap();
    let mut third = app.config.machines[1].clone();
    (third.name, third.ssh) = ("c".into(), "host-c".into());
    app.config.machines.push(third);
    app.config.machines[1].fabric.insert("c".into(), "192.0.2.3".into());
    app.config.machines[2].fabric = BTreeMap::from([("b".to_owned(), "192.0.2.4".to_owned())]);
    fs::remove_file(app.runner.root.join("calls.log")).unwrap();
    app.config.stage_dir = app.config.stage_root.clone();
    app.config.resolve_generation().unwrap();
    install(&mut app).unwrap();
    let manifest = Manifest::load(&app.config.build_manifest).unwrap();
    let folders: BTreeSet<&str> = manifest.artifacts.values().map(|artifact| artifact.ptx.as_str()).collect();
    assert_eq!((manifest.artifacts.len(), folders.len()), (3, 1), "{:?}", manifest.artifacts);
    let log = app.runner.log();
    for host in ["host-a", "host-b", "host-c"] { assert!(log.contains(&format!("rsync {host} flash-serve flash.ptx\n")), "{log}"); }
}

/// Nothing changed, so a second install copies nothing, on every rank.
#[test]
fn a_second_install_of_the_same_sparks_copies_nothing() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = app();
    for host in ["host-a", "host-b"] { app.runner.know(host); }
    install(&mut app).unwrap();
    fs::remove_file(app.runner.root.join("calls.log")).unwrap();
    app.config.stage_dir = app.config.stage_root.clone();
    app.config.resolve_generation().unwrap();
    install(&mut app).unwrap();
    assert!(!app.runner.log().lines().any(|row| row.starts_with("rsync ")), "{}", app.runner.log());
}

/// The copy of the large files fails on ssh's own error, with its fix.
#[test]
fn a_copy_that_loses_ssh_shows_the_real_line() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = app();
    for host in ["host-a", "host-b"] { app.runner.know(host); }
    app.runner.deny_on_copy = Some("host-b".into());
    let error = install(&mut app).unwrap_err();
    assert!(error.starts_with("b: ssh host-b failed: fake@host-b: Permission denied (publickey). Vade never edits ~/.ssh. Run ssh-copy-id host-b, then run vade doctor."), "{error}");
    assert!(!error.contains("Prebuilt copy failed"), "{error}");
}

/// Four Sparks from one bundle: no Rust on a rank, no checkout, and one copy route.
#[test]
fn four_sparks_take_the_bundle_with_no_rust_and_one_copy_route() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = sparks(4);
    for host in ["host-a", "host-b", "host-c", "host-d"] { app.runner.know(host); }
    install(&mut app).unwrap();
    let log = app.runner.log();
    assert!(!log.contains("rustc was asked") && !log.contains("cargo was asked") && !log.contains("for tool in cargo"), "{log}");
    for host in ["host-a", "host-b", "host-c", "host-d"] {
        assert_eq!(log.lines().filter(|row| row.starts_with(&format!("rsync {host} "))).count(), 2, "the helper, then the engine and the PTX: {log}");
        assert!(log.contains(&format!("sha256sum {host}\n")), "the hash runs on the rank: {log}");
    }
    let manifest = Manifest::load(&app.config.build_manifest).unwrap();
    assert_eq!(manifest.artifacts.values().map(|artifact| artifact.ptx.as_str()).collect::<BTreeSet<_>>().len(), 1);
}

/// Install on both hosts, with each rank held until the other one starts its copy
/// and its preparation. Returns the result and what the person sees.
fn install_together(app: &mut App<FakeHosts>) -> (Result<(), String>, String) {
    app.runner.know("host-a");
    app.runner.know("host-b");
    app.runner.meet = Some(Arc::default());
    let capture = crate::output::testing::Capture::default();
    let result = crate::output::with_writer(capture.clone(), || install(app));
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    (result, printed)
}

#[test]
fn both_hosts_copy_and_prepare_at_the_same_time_and_report_in_rank_order() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = app();
    let (result, printed) = install_together(&mut app);
    result.unwrap();
    let rows = printed.lines().filter(|row| ["Preparing ", "Copying Vade", "a: ", "b: ", "Installed build"].iter().any(|start| row.starts_with(start)) && !row.contains("driver and install disk")).collect::<Vec<_>>();
    assert_eq!(rows, ["Preparing a, b at the same time; each Spark's lines follow when it is done.",
        "Copying Vade to a.", "a: all model shards present.", "a: install caches prepared model_hashed_bytes=0 model_reused_files=132.",
        "Copying Vade to b.", "b: all model shards present.", "b: install caches prepared model_hashed_bytes=0 model_reused_files=132.",
        &*format!("Installed build {}. Models were already present.", "d".repeat(40))], "{printed}");
    assert_eq!(Manifest::load(&app.config.build_manifest).unwrap().artifacts.keys().cloned().collect::<Vec<_>>(), ["a", "b"]);
}

#[test]
fn a_failed_rank_is_named_the_other_rank_ends_and_the_activation_stays() {
    let _guard = crate::runner::process_test_lock();
    for (fail, other, error) in [("a install preparation", "b install preparation", "a: Install cache preparation failed; the previous activation was kept."),
        ("b install preparation", "a install preparation", "b: Install cache preparation failed; the previous activation was kept.")] {
        let (_f, mut app) = app();
        app.runner.fail = Some(fail);
        let before = (fs::read(&app.config.activation).unwrap(), app.config.build_manifest.clone());
        let (result, printed) = install_together(&mut app);
        assert!(result.unwrap_err().starts_with(error), "{printed}");
        assert_eq!((fs::read(&app.config.activation).unwrap(), app.config.build_manifest.clone()), before);
        assert!(app.runner.ended.lock().unwrap().iter().any(|label| label == other), "the other rank ran to its end");
        assert_eq!(printed.contains("A Spark failed; waiting for the other Sparks to end."), fail.starts_with('a'), "{printed}");
        // Output stops at the first failing rank; a rank before it is shown whole.
        assert_eq!(printed.contains("a: install caches prepared"), fail.starts_with('b'), "{printed}");
        assert!(!printed.contains("b: install caches prepared") && !printed.contains("Installed build"), "{printed}");
    }
}

#[test]
fn a_dry_run_has_no_workers_so_rank_a_ends_before_rank_b_starts() {
    let _guard = crate::runner::process_test_lock();
    let (_f, app) = app();
    let mut app = App { config: app.config, dry: true, runner: SystemRunner { dry: true } };
    let capture = crate::output::testing::Capture::default();
    crate::prebuilt::LINUX_HOST.set(true);
    let result = crate::output::with_writer(capture.clone(), || app.execute(Command::Install));
    crate::prebuilt::LINUX_HOST.set(false);
    result.unwrap();
    let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
    assert!(!printed.contains("at the same time"), "{printed}");
    let at = |needle: &str| printed.find(needle).unwrap_or_else(|| panic!("{needle} missing from {printed}"));
    assert!(at("Copying Vade to a.") < at("Copying Vade to b."));
    let last_of_a = printed.rfind("host-a").unwrap();
    assert!(last_of_a < at("Copying Vade to b."), "{printed}");
}

#[test]
fn one_up_checks_every_box_and_names_both_host_key_failures() {
    let _guard = crate::runner::process_test_lock();
    let (_f, mut app) = sparks(3);
    app.runner.know("host-b");
    let capture = crate::output::testing::Capture::default();
    let error = crate::output::with_writer(capture, || app.up(&fp4_options(&["--sparks", "3"]))).unwrap_err();
    assert!(error.starts_with("SSH checks failed."), "{error}");
    for host in ["host-a", "host-c"] { assert!(error.contains(&format!("ssh {host} failed: Host key verification failed.")), "{error}"); }
    assert_eq!(error.matches("Read docs/INSTALL.md for the SSH check command.").count(), 2, "{error}");
    assert_eq!(error.matches("Host key verification failed.").count(), 2, "{error}");
    let calls = app.runner.log();
    assert_eq!(calls.lines().collect::<Vec<_>>(), ["ssh host-a true", "ssh host-b true", "ssh host-c true"], "{calls}");
    for host in ["host-a", "host-b", "host-c"] { assert!(!app.runner.root.join(host).join(".ssh").exists()); }
    assert!(!app.config.state_dir.join("session.conf").exists());
}
