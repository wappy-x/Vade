//! Strict JSON framing from the public executable; no Spark or external process.
#[allow(dead_code)]
#[path = "../src/json.rs"]
mod independent_json;
#[allow(dead_code)]
#[path = "../src/evals/json.rs"]
mod json;
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "vade-json-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("home")).unwrap();
        Self(root)
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
        command
            .args(["status", "--json"])
            .env("VADE_HOME", self.0.join("home"))
            .env("VADE_CONFIG", self.0.join("missing"))
            .env_remove("VADE_API_KEY_FILE");
        command
    }
    fn config(&self) -> PathBuf {
        let path = self.0.join("config");
        fs::write(&path, "[global]\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nstate_dir = state\nmodel = qwen3.8-flash-next-fp4-local\n[machine spark]\nssh = fixture-no-network\nmodel_path = /models/fixture\n").unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn receipt(output: Output, code: i32) -> json::Value {
    assert_eq!(
        output.status.code(),
        Some(code),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        stdout.lines().count(),
        1,
        "status must emit exactly one JSON object"
    );
    assert_eq!(
        independent_json::string(&stdout, "schema").as_deref(),
        Some("vade.status.v1"),
        "independent strict parser rejected JSON"
    );
    assert!(
        output.stderr.is_empty(),
        "unexpected prose: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!stdout.chars().any(|c| c.is_control() && c != '\n'));
    json::parse(&stdout).unwrap()
}
#[test]
fn stopped_and_invalid_configuration_remain_machine_readable() {
    let f = Fixture::new();
    let missing = f.0.join("missing\x1b[2J\u{009b}\u{202e}.conf");
    for dry in [false, true] {
        let mut command = f.command();
        command.env("VADE_CONFIG", &missing);
        if dry {
            command.arg("--dry-run");
        }
        let value = receipt(command.output().unwrap(), 1);
        assert_eq!(value["phase"].as_str(), Some("unknown"));
        assert_eq!(value["ready"], json::Value::Null);
        assert!(value["errors"][0].as_str().unwrap().contains("INSTALL.md"));
        let mut command = f.command();
        command.env("VADE_CONFIG", f.config());
        if dry {
            command.arg("--dry-run");
        }
        let value = receipt(command.output().unwrap(), 0);
        assert_eq!(value["phase"].as_str(), Some("stopped"));
        assert_eq!(value["ready"], json::Value::Bool(false));
        assert_eq!(value["checks"].as_array().unwrap().len(), 0);
        assert!(!f.0.join("state").exists());
    }
    fs::create_dir(f.0.join("state")).unwrap();
    fs::write(f.0.join("state/session.conf"), "bad record\n").unwrap();
    let value = receipt(
        f.command().env("VADE_CONFIG", f.config()).output().unwrap(),
        1,
    );
    assert_eq!(value["ready"], json::Value::Null);
    assert_eq!(
        fs::read_to_string(f.0.join("state/session.conf")).unwrap(),
        "bad record\n"
    );
}
#[test]
fn offline_local_ownership_is_reported_without_probing_or_claiming_health() {
    let f = Fixture::new();
    let home = f.0.join("home");
    fs::write(home.join("setup.conf"), "").unwrap();
    let value = receipt(f.command().env_remove("VADE_CONFIG").output().unwrap(), 0);
    assert_eq!(value["mode"].as_str(), Some("local"));
    assert_eq!(value["phase"].as_str(), Some("stopped"));
    let record = format!(
        "version=1\nphase=running\nport=8080\ntoken={}\ngeneration={}\n",
        "a".repeat(64),
        "b".repeat(24)
    );
    fs::write(home.join("run.conf"), &record).unwrap();
    for _ in 0..2 {
        let value = receipt(
            f.command()
                .arg("--dry-run")
                .env_remove("VADE_CONFIG")
                .output()
                .unwrap(),
            0,
        );
        assert_eq!(value["phase"].as_str(), Some("running"));
        assert_eq!(value["ready"], json::Value::Null);
        assert_eq!(value["ranks"][0]["alive"], json::Value::Null);
        assert!(
            value["checks"]
                .as_array()
                .unwrap()
                .iter()
                .all(|check| check["ok"] == json::Value::Null)
        );
        assert!(value.get("token").is_none());
        assert_eq!(fs::read_to_string(home.join("run.conf")).unwrap(), record);
        assert!(
            !home.join("worker").exists(),
            "offline status must not invoke the worker"
        );
    }
}
#[test]
fn a_closed_status_output_pipe_does_not_panic_or_print_noise() {
    use std::{
        os::{fd::OwnedFd, unix::net::UnixStream},
        process::Stdio,
    };
    let f = Fixture::new();
    let (writer, reader) = UnixStream::pair().unwrap();
    drop(reader);
    let fd: OwnedFd = writer.into();
    let output = f
        .command()
        .env("VADE_CONFIG", f.config())
        .stdout(Stdio::from(fd))
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
}
