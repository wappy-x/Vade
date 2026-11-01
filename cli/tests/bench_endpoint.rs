//! Prove benchmark discovery reaches the selected port without loading a GPU.
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "vade-bench-endpoint-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
        command.arg("bench").current_dir(&self.0)
            .env("HOME", &self.0).env("XDG_CONFIG_HOME", self.0.join("config-home"))
            .env("XDG_DATA_HOME", self.0.join("data-home"))
            .env_remove("VADE_HOME").env_remove("VADE_CONFIG")
            .env_remove("VADE_API_URL").env_remove("VADE_API_KEY_FILE");
        command
    }
    fn owner(&self, root: &Path, port: u16) {
        fs::create_dir_all(root.join("state")).unwrap();
        fs::write(root.join("config"), format!("format=vade-owner-v1\nport={port}\n")).unwrap();
        fs::write(root.join("state/run.conf"), format!(
            "version=1\nphase=running\nport={port}\ntoken={}\n", "a".repeat(64),
        )).unwrap();
    }
    fn controller(&self, port: u16) -> PathBuf {
        let path = self.0.join("controller.conf");
        fs::write(&path, format!(
            "[global]\nlocal_port = {port}\nmodel_revision = 1111111111111111111111111111111111111111\n[machine fixture]\nssh = unused-host\nmodel_path = /unused-model\n",
        )).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); }
}

struct Server {
    port: u16,
    worker: thread::JoinHandle<bool>,
}
impl Server {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                        let mut bytes = Vec::new();
                        while !bytes.windows(4).any(|part| part == b"\r\n\r\n") {
                            let mut chunk = [0; 4096];
                            let count = stream.read(&mut chunk).unwrap();
                            assert!(count != 0 && bytes.len() < 16384);
                            bytes.extend_from_slice(&chunk[..count]);
                        }
                        assert!(bytes.starts_with(b"GET /health HTTP/1.1\r\n"));
                        // End at health: endpoint selection is independent of timing,
                        // request construction and receipt semantics tested by bench.
                        stream.write_all(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                        return true;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(2)),
                    Err(e) => panic!("mock accept: {e}"),
                }
            }
            false
        });
        Self { port, worker }
    }
    fn url(&self) -> String { format!("http://127.0.0.1:{}", self.port) }
    fn assert_reached(self, mut command: Command) {
        let output = command.output().unwrap();
        assert!(!output.status.success(), "mock health must fail: {output:?}");
        assert!(self.worker.join().unwrap(), "did not discover selected port: {output:?}");
    }
}

#[test]
fn no_flags_uses_default_owner_recorded_port() {
    let fixture = Fixture::new(); let server = Server::start();
    fixture.owner(&fixture.0.join(".vade"), server.port);
    server.assert_reached(fixture.command());
}

#[test]
fn no_flags_uses_selected_vade_home_before_legacy_and_implicit_controller() {
    let fixture = Fixture::new(); let server = Server::start();
    let owner = fixture.0.join("selected-owner"); fixture.owner(&owner, server.port);
    fs::write(owner.join("run.conf"), "port=1\n").unwrap();
    let implicit = fixture.0.join("config-home/vade"); fs::create_dir_all(&implicit).unwrap();
    fs::write(implicit.join("config"), "invalid controller config").unwrap();
    let mut command = fixture.command(); command.env("VADE_HOME", owner);
    server.assert_reached(command);
}

#[test]
fn explicit_controller_selects_its_port_even_with_owner_state() {
    let fixture = Fixture::new(); let server = Server::start();
    fixture.owner(&fixture.0.join(".vade"), 1);
    let config = fixture.controller(server.port);
    let mut command = fixture.command(); command.env("VADE_CONFIG", config);
    server.assert_reached(command);
}

#[test]
fn api_url_bypasses_invalid_installation_state() {
    let fixture = Fixture::new(); let server = Server::start();
    let mut command = fixture.command();
    command.env("VADE_CONFIG", "").env("VADE_HOME", "invalid").env("VADE_API_URL", server.url());
    server.assert_reached(command);
}

#[test]
fn explicit_url_overrides_api_url_and_invalid_installation_state() {
    let fixture = Fixture::new(); let server = Server::start();
    let mut command = fixture.command();
    command.args(["--url", &server.url()]).env("VADE_CONFIG", "")
        .env("VADE_HOME", "invalid").env("VADE_API_URL", "invalid");
    server.assert_reached(command);
}

#[test]
fn no_recorded_server_names_the_url_flag_instead_of_guessing_a_port() {
    let fixture = Fixture::new();
    let output = fixture.command().output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("No recorded local server endpoint") && error.contains("--url"), "{error}");
    assert!(!fixture.0.join("vade-bench.json").exists());
}

#[test]
fn inconsistent_owner_state_fails_before_contacting_the_server() {
    let fixture = Fixture::new();
    let root = fixture.0.join(".vade"); fixture.owner(&root, 18923);
    fs::write(root.join("config"), "format=vade-owner-v1\nport=18924\n").unwrap();
    let output = fixture.command().output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("names no running engine") && error.contains("--url"), "{error}");
}
