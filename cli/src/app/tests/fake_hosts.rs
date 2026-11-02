//! A bundle-only install on two fake hosts: the real install code and real
//! processes, with the tools in tests/fixtures/fake_hosts first on PATH. It runs
//! in this process because a Mac cannot pass the host platform check for a Linux
//! bundle ([`crate::prebuilt::LINUX_HOST`]), and cli/tests/setup.rs runs the real
//! binary on the same tools up to that check.
use super::*;
use crate::runner::{Runner, SystemRunner};

struct FakeHosts { root: PathBuf, tools: PathBuf, system: SystemRunner }
impl FakeHosts {
    fn new(root: PathBuf) -> Self {
        let tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_hosts");
        for host in ["host-a", "host-b"] { fs::create_dir_all(root.join(host)).unwrap(); }
        Self { root, tools, system: SystemRunner { dry: false } }
    }
    fn log(&self) -> String { fs::read_to_string(self.root.join("calls.log")).unwrap_or_default() }
    fn know(&self, host: &str) { fs::write(self.root.join(host).join("known"), "").unwrap(); }
}
impl Runner for FakeHosts {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let mut exec = exec.clone();
        // The stage holds Linux programs, which this computer may not start.
        if exec.label == "harness doctor availability" { return Ok(Output { success: true, stdout: String::new() }); }
        exec.env.insert("PATH".into(), format!("{}:/usr/bin:/bin", self.tools.display()));
        exec.env.insert("FAKE_HOSTS".into(), self.root.to_string_lossy().into_owned());
        self.system.run(&exec)
    }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> { Ok(true) }
    fn port_free(&mut self, _: u16) -> Result<(), String> { Ok(()) }
    fn pause(&mut self, _: Duration) {}
}

/// An `installed-bundle` config for host-a and host-b, with a floor in `MANIFEST`.
fn app() -> (Fixture, App<FakeHosts>) {
    let f = Fixture::fp4(2, false);
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
    for (machine, host) in config.machines.iter_mut().zip(["host-a", "host-b"]) { machine.ssh = host.into(); }
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
    assert!(error.starts_with("ssh host-b failed: Host key verification failed. Vade never edits ~/.ssh. Run ssh host-b true and accept the key, or run ssh-keyscan -H host-b >> ~/.ssh/known_hosts if host-b is the real host name, then run vade doctor."), "{error}");
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
    assert!(install(&mut app).unwrap_err().starts_with("Prebuilt artifact rejected:"));
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
