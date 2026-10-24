//! Owner/controller selection through the public executable. Every operation
//! is a dry run: no GPU, network connection or helper process is needed.
#[allow(dead_code)]
#[path = "../src/evals/json.rs"]
mod json;

use std::{
    fs,
    io::{self, Read, Seek},
    os::unix::process::CommandExt,
    path::PathBuf,
    process::{Child, Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
const MODEL: &str = "owner-checkpoint";
const PORT: u16 = 18123;

// Capture to files so a child or inherited pipe cannot prevent the parent
// enforcing its deadline. The guard also reaps/kills on assertion unwinding.
struct SupervisedChild(Option<Child>);
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

struct Capture(PathBuf);
impl Drop for Capture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

trait BoundedOutput {
    fn deadline_output(&mut self, timeout: Duration) -> io::Result<Output>;
    fn bounded_output(&mut self) -> io::Result<Output> {
        self.deadline_output(Duration::from_secs(30))
    }
}
impl BoundedOutput for Command {
    fn deadline_output(&mut self, timeout: Duration) -> io::Result<Output> {
        let capture = Capture(std::env::temp_dir().join(format!(
            "vade-owner-dispatch-output-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        fs::create_dir(&capture.0)?;
        let open = |name| {
            fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(capture.0.join(name))
        };
        let mut stdout = open("stdout")?;
        let mut stderr = open("stderr")?;
        self.stdin(Stdio::null())
            .stdout(stdout.try_clone()?)
            .stderr(stderr.try_clone()?)
            .process_group(0);
        let mut child = SupervisedChild(Some(self.spawn()?));
        let deadline = Instant::now() + timeout;
        loop {
            let status = child.0.as_mut().unwrap().try_wait()?;
            if status.is_some() || Instant::now() >= deadline {
                // On timeout kill the still-owned process group. Once reaped,
                // stop owning its PID so cleanup cannot signal a reused ID.
                if status.is_some() {
                    child.0 = None;
                }
                drop(child);
                stdout.rewind()?;
                stderr.rewind()?;
                let mut out = Vec::new();
                let mut err = Vec::new();
                stdout.read_to_end(&mut out)?;
                stderr.read_to_end(&mut err)?;
                return match status {
                    Some(status) => Ok(Output {
                        status,
                        stdout: out,
                        stderr: err,
                    }),
                    None => Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        format!(
                            "dispatch child exceeded {timeout:?}; stdout: {}; stderr: {}",
                            String::from_utf8_lossy(&out),
                            String::from_utf8_lossy(&err)
                        ),
                    )),
                };
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[test]
fn dispatch_deadline_terminates_a_stuck_child() {
    let error = Command::new("/bin/sh")
        .args(["-c", "exec sleep 60"])
        .deadline_output(Duration::from_millis(500))
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    // A busy host may not schedule the child before its deadline. Only require
    // the parent's diagnostic; successful dispatch tests verify captured output.
    assert!(
        error.to_string().contains("dispatch child exceeded"),
        "{error}"
    );
}

struct Fixture {
    base: PathBuf,
    owner: PathBuf,
    override_home: bool,
}

impl Fixture {
    fn new(override_home: bool) -> Self {
        let base = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "vade-owner-dispatch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let owner = if override_home {
            base.join("local")
        } else {
            base.join("user/.vade")
        };
        fs::create_dir_all(owner.join("state")).unwrap();
        fs::create_dir_all(base.join("user")).unwrap();
        fs::create_dir_all(base.join("xdg/vade")).unwrap();
        // A valid competing controller config catches accidental fallback even
        // when VADE_CONFIG is absent and the default config path is consulted.
        fs::write(
            base.join("xdg/vade/config"),
            "[global]\nstate_dir = controller-state\nmodel = qwen3.8-flash-next-fp4-local\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n[machine first]\nssh = fixture-no-network\nmodel_path = /controller/model\nfabric.second = 192.0.2.1\n[machine second]\nssh = fixture-second-no-network\nmodel_path = /controller/model\nfabric.first = 192.0.2.2\n",
        )
        .unwrap();
        Self {
            base,
            owner,
            override_home,
        }
    }

    fn configure_owner(&self) {
        fs::write(
            self.owner.join("config"),
            format!(
                "format=vade-owner-v1\nmodel_path=/owner/model\nmodel_bytes=1024\ncontext_tokens=8192\noutput_tokens=4096\nport={PORT}\nstage_dir={}\nstate_dir={}\n",
                self.owner.join("stage").display(),
                self.owner.join("state").display()
            ),
        )
        .unwrap();
    }

    fn record_owner(&self) -> String {
        let record = format!(
            "version=1\nphase=running\nport={PORT}\ntoken={}\nmodel_path=/owner/model\nengine={}/stage/flash\nendpoint_model={MODEL}\n",
            "a".repeat(64),
            self.owner.display()
        );
        fs::write(self.owner.join("state/run.conf"), &record).unwrap();
        record
    }

    fn command(&self, exported_config: bool) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", self.base.join("user"))
            .env("XDG_CONFIG_HOME", self.base.join("xdg"))
            .arg("--dry-run");
        if self.override_home {
            command.env("VADE_HOME", &self.owner);
        }
        if exported_config {
            command.env("VADE_CONFIG", self.base.join("xdg/vade/config"));
        }
        command
    }

    fn no_runtime_writes(&self) {
        assert!(!self.owner.join("state/worker").exists());
        assert!(!self.owner.join("stage").exists());
        assert!(!self.base.join("xdg/vade/controller-state").exists());
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.base).unwrap();
    }
}

fn stdout(output: Output) -> String {
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn launch_and_lifecycle_resolve_the_same_generated_config() {
    for override_home in [false, true] {
        let f = Fixture::new(override_home);
        f.configure_owner();
        let config = fs::read(f.owner.join("config")).unwrap();
        for exported_config in [false] {
            for command in ["launch", "down", "chat", "logs", "install", "up"] {
                let text = stdout(
                    f.command(exported_config)
                        .arg(command)
                        .bounded_output()
                        .unwrap(),
                );
                assert!(
                    text.contains(f.owner.join("config").to_str().unwrap()),
                    "{command}, VADE_CONFIG={exported_config}: {text}"
                );
                assert!(!text.contains("fixture-no-network"), "{command}: {text}");
            }
        }
        assert_eq!(fs::read(f.owner.join("config")).unwrap(), config);
        f.no_runtime_writes();
    }
}

#[test]
fn inherited_mtp_does_not_change_bare_up_selection() {
    let f = Fixture::new(true);
    for configured in [false, true] {
        if configured {
            f.configure_owner();
        }
        for exported_config in [false, true] {
            let baseline = stdout(
                f.command(exported_config)
                    .arg("up")
                    .bounded_output()
                    .unwrap(),
            );
            for mtp in ["0", "2"] {
                let text = stdout(
                    f.command(exported_config)
                        .args(["up"])
                        .env("MTP", mtp)
                        .bounded_output()
                        .unwrap(),
                );
                if configured && !exported_config {
                    assert_eq!(
                        text, baseline,
                        "owner up must use saved defaults, MTP={mtp}"
                    );
                    assert!(text.contains(f.owner.join("config").to_str().unwrap()));
                } else {
                    assert!(text.contains("Launch plan:"), "controller up: {text}");
                    assert!(text.contains("fixture-no-network"), "controller up: {text}");
                    assert!(!text.contains(f.owner.join("config").to_str().unwrap()));
                }
            }
        }
    }
    f.no_runtime_writes();
}

#[test]
fn alternate_starts_require_a_consistent_installation_selector() {
    let f = Fixture::new(true);
    f.configure_owner();
    for args in [
        vec!["up", "--fp8-legacy"],
        vec!["run", "qwen3-8b"],
    ] {
        let rejected = f.command(false).args(&args).bounded_output().unwrap();
        assert_eq!(rejected.status.code(), Some(1), "{args:?}: {rejected:?}");
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("VADE_CONFIG"));
        assert!(!String::from_utf8_lossy(&rejected.stdout).contains("Launch plan:"));

    }
    for args in [["up", "--context", "65536"], ["launch", "--output", "16384"]] {
        let text = stdout(f.command(false).args(args).bounded_output().unwrap());
        assert!(text.contains("Local defaults:"), "{text}");
    }
    let rejected = f.command(true).arg("launch").bounded_output().unwrap();
    assert_eq!(rejected.status.code(), Some(1), "{rejected:?}");
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("unset VADE_CONFIG"));
    f.no_runtime_writes();
}

#[test]
fn legacy_generated_config_rejects_controller_start_without_explicit_selection() {
    let f = Fixture::new(true);
    fs::write(f.owner.join("setup.conf"), "").unwrap();
    for args in [
        vec!["up"],
        vec!["up", "--context", "8192"],
        vec!["launch"],
        vec!["install"],
    ] {
        let rejected = f.command(false).args(&args).bounded_output().unwrap();
        assert_eq!(rejected.status.code(), Some(1), "{args:?}: {rejected:?}");
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("vade run qwen3-8b"));
    }
    let rejected = f
        .command(true)
        .args(["run", "qwen3-8b"])
        .bounded_output()
        .unwrap();
    assert_eq!(rejected.status.code(), Some(1), "{rejected:?}");
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("unset VADE_CONFIG"));
    let start = stdout(
        f.command(false)
            .args(["run", "qwen3-8b"])
            .bounded_output()
            .unwrap(),
    );
    assert!(start.contains("official Qwen/Qwen3-8B"), "{start}");
    let chat = f.command(false).arg("chat").bounded_output().unwrap();
    assert!(String::from_utf8_lossy(&chat.stderr).contains("vade run qwen3-8b"));
    let status = stdout(
        f.command(false)
            .args(["status", "--json"])
            .bounded_output()
            .unwrap(),
    );
    let value = json::parse(&status).unwrap();
    assert_eq!(value["mode"].as_str(), Some("local"));
    assert_eq!(value["model"].as_str(), Some("qwen3-8b"));
    assert!(
        stdout(f.command(false).arg("down").bounded_output().unwrap()).contains("Vade: stopped.")
    );
    f.no_runtime_writes();
}

#[test]
fn explicit_controller_selection_wins_for_every_lifecycle_command() {
    let f = Fixture::new(true);
    f.configure_owner();
    let record = f.record_owner();
    let start = stdout(
        f.command(true)
            .args(["up", "--context", "8192"])
            .bounded_output()
            .unwrap(),
    );
    assert!(start.contains("fixture-no-network"), "{start}");
    let chat = f.command(true).arg("chat").bounded_output().unwrap();
    assert_eq!(chat.status.code(), Some(1), "{chat:?}");
    assert!(String::from_utf8_lossy(&chat.stderr).contains("Vade is stopped. Run vade up"));
    for command in ["status", "logs", "down"] {
        let text = stdout(f.command(true).arg(command).bounded_output().unwrap());
        assert!(
            !text.contains(f.owner.join("config").to_str().unwrap()),
            "{command}: {text}"
        );
    }
    let status = stdout(
        f.command(true)
            .args(["status", "--json"])
            .bounded_output()
            .unwrap(),
    );
    let value = json::parse(&status).unwrap();
    assert_eq!(value["mode"].as_str(), Some("configured"));
    assert_eq!(value["phase"].as_str(), Some("stopped"));
    assert_eq!(
        fs::read_to_string(f.owner.join("state/run.conf")).unwrap(),
        record
    );
    f.no_runtime_writes();
}

#[test]
fn plain_and_json_status_use_the_selected_owner_record() {
    for override_home in [false, true] {
        let f = Fixture::new(override_home);
        f.configure_owner();
        let record = f.record_owner();
        // The legacy path has setup but no running engine. JSON must not report
        // that path's stopped/qwen3-8b state for the running owner engine.
        let legacy = if override_home {
            f.owner.clone()
        } else {
            f.base.join("user/.local/share/vade")
        };
        fs::create_dir_all(&legacy).unwrap();
        fs::write(legacy.join("setup.conf"), "").unwrap();
        for exported_config in [false] {
            let plain = stdout(
                f.command(exported_config)
                    .arg("status")
                    .bounded_output()
                    .unwrap(),
            );
            let text = stdout(
                f.command(exported_config)
                    .args(["status", "--json"])
                    .bounded_output()
                    .unwrap(),
            );
            assert_eq!(text.lines().count(), 1, "JSON must be one object: {text}");
            let value = json::parse(&text).unwrap();
            assert_eq!(value["schema"].as_str(), Some("vade.status.v1"));
            assert_eq!(value["phase"].as_str(), Some("running"));
            assert_eq!(value["mode"].as_str(), Some("local"));
            assert_eq!(value["model"].as_str(), Some(MODEL));
            assert_eq!(
                value["endpoint"].as_str(),
                Some(format!("http://127.0.0.1:{PORT}/v1").as_str())
            );
            assert_eq!(
                value["ready"],
                json::Value::Null,
                "dry status must not invent readiness"
            );
            assert!(
                value["checks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|check| check["ok"] == json::Value::Null)
            );
            assert!(value["errors"].as_array().unwrap().is_empty());
            assert!(value.get("token").is_none());
            assert!(!text.contains(&"a".repeat(64)));
            assert!(plain.contains("running"), "{plain}");
            assert!(plain.contains(&format!("127.0.0.1:{PORT}")), "{plain}");
            assert!(!plain.contains("stopped"), "{plain}");
        }
        assert_eq!(
            fs::read_to_string(f.owner.join("state/run.conf")).unwrap(),
            record
        );
        f.no_runtime_writes();
    }
}

#[test]
fn missing_owner_config_preserves_running_ownership_for_status_and_shutdown() {
    let f = Fixture::new(true);
    let record = f.record_owner();
    for exported_config in [false] {
        let output = f
            .command(exported_config)
            .args(["status", "--json"])
            .bounded_output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        assert!(output.stderr.is_empty(), "{output:?}");
        let text = String::from_utf8(output.stdout).unwrap();
        assert_eq!(text.lines().count(), 1, "{text}");
        let value = json::parse(&text).unwrap();
        assert_eq!(value["schema"].as_str(), Some("vade.status.v1"));
        assert_eq!(value["mode"].as_str(), Some("local"));
        assert_eq!(value["phase"].as_str(), Some("running"));
        assert_eq!(value["ready"], json::Value::Null);
        assert!(!value["errors"].as_array().unwrap().is_empty());
        let down = stdout(
            f.command(exported_config)
                .arg("down")
                .bounded_output()
                .unwrap(),
        );
        assert!(
            down.contains(f.owner.join("config").to_str().unwrap()),
            "{down}"
        );
        assert!(!down.contains("fixture-no-network"), "{down}");
    }
    assert_eq!(
        fs::read_to_string(f.owner.join("state/run.conf")).unwrap(),
        record
    );
    f.no_runtime_writes();
}

#[test]
fn new_local_defaults_cannot_hide_a_default_controller_record() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new(true);
    let state = f.base.join("xdg/vade/controller-state");
    fs::create_dir_all(&state).unwrap();
    let record = state.join("session.conf");
    for kind in ["regular", "directory", "dangling-link"] {
        match kind {
            "regular" => fs::write(&record, "retain even malformed ownership evidence\n").unwrap(),
            "directory" => fs::create_dir(&record).unwrap(),
            _ => symlink(state.join("missing-target"), &record).unwrap(),
        }
        for args in [vec!["launch"], vec!["run", "qwen3-8b"]] {
            let rejected = f.command(false).args(&args).bounded_output().unwrap();
            assert_eq!(
                rejected.status.code(),
                Some(1),
                "{kind}, {args:?}: {rejected:?}"
            );
            assert!(String::from_utf8_lossy(&rejected.stderr).contains("controller"));
            assert!(!f.owner.join("config").exists());
            assert!(!f.owner.join("setup.conf").exists());
            assert!(fs::symlink_metadata(&record).is_ok());
        }
        if kind == "directory" {
            fs::remove_dir(&record).unwrap();
        } else {
            fs::remove_file(&record).unwrap();
        }
    }
    assert!(
        f.command(false)
            .arg("launch")
            .bounded_output()
            .unwrap()
            .status
            .success()
    );
    assert!(
        f.command(false)
            .args(["run", "qwen3-8b"])
            .bounded_output()
            .unwrap()
            .status
            .success()
    );
    let config = f.base.join("xdg/vade/config");
    fs::remove_file(&config).unwrap();
    fs::create_dir(&config).unwrap();
    for args in [vec!["launch"], vec!["run", "qwen3-8b"]] {
        let rejected = f.command(false).args(args).bounded_output().unwrap();
        assert_eq!(rejected.status.code(), Some(1), "{rejected:?}");
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("controller config"));
    }
    fs::remove_dir(&config).unwrap();
    for command in ["launch", "install", "up"] {
        let text = stdout(f.command(false).arg(command).bounded_output().unwrap());
        assert!(text.contains(f.owner.join("config").to_str().unwrap()));
    }
}
