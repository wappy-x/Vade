//! End-to-end CLI checks with an explicitly selected first-party Rust peer.
//! No installed external agent, account files, shell, or network is used.
#![cfg(unix)]
#[path = "support/home.rs"]
mod home;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

fn compile_mock(provider: &str, source_text: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let directory = root
        .join("target")
        .join(format!("{provider}-cli-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let source = directory.join("mock.rs");
    fs::write(
        &source,
        format!(
            "#[path = {:?}] mod json;\n{source_text}",
            root.join("src/json.rs")
        ),
    )
    .unwrap();
    let binary = directory.join("mock-child");
    let compiler = std::env::var_os("VADE_TEST_RUSTC")
        .or_else(|| std::env::var_os("RUSTC"))
        .unwrap_or_else(|| "rustc".into());
    assert!(
        Command::new(compiler)
            .args([
                "--edition=2024",
                "--crate-name=vade_cli_fixture",
                "-Adead_code"
            ])
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .status()
            .unwrap()
            .success()
    );
    binary
}
fn mock() -> &'static Path {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY
        .get_or_init(|| compile_mock("codex", include_str!("fixtures/codex_child.rs")))
        .as_path()
}
fn hermes_mock() -> &'static Path {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY
        .get_or_init(|| compile_mock("hermes", include_str!("fixtures/hermes_child.rs")))
        .as_path()
}
fn hermes_command() -> Command {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let workspace = hermes_mock().parent().unwrap().join(format!(
        "workspace-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&workspace).unwrap();
    let mut command = home::harness();
    command
        .args([
            "hermes",
            "--state-in-project",
            "--permission",
            "deny-callbacks",
            "--plain",
            "--hermes-bin",
        ])
        .arg(hermes_mock())
        .arg("--workspace")
        .arg(workspace);
    command
}

fn base_command(plain: bool) -> Command {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let workspace = mock().parent().unwrap().join(format!(
        "workspace-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&workspace).unwrap();
    let mut command = home::harness();
    command
        .args(["codex", "--state-in-project", "--permission", "read-only", "--codex-bin"])
        .arg(mock())
        .arg("--workspace")
        .arg(workspace);
    if plain {
        command.arg("--plain");
    }
    command
}

fn command(prompt: &str) -> Command {
    let mut command = base_command(true);
    command.args(["--prompt", prompt]);
    command
}

fn lines(input: &str) -> Output {
    let mut command = base_command(true);
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    finish(child)
}

fn finish(mut child: Child) -> Output {
    let started = std::time::Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if started.elapsed() > std::time::Duration::from_secs(8) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("mock CLI exceeded local test deadline");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

fn stdout(output: &Output) -> &str {
    std::str::from_utf8(&output.stdout).unwrap()
}
fn stderr(output: &Output) -> &str {
    std::str::from_utf8(&output.stderr).unwrap()
}

#[test]
fn plain_cli_prints_authoritative_answer_and_exact_input() {
    let output = command("disagreement").output().unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stdout(&output), "correct final\n");
    assert!(!stdout(&output).contains("wrong draft"));
    let text = "  exact\n  Unicode 🦀\t\"quoted\"  ";
    let output = command(text).output().unwrap();
    assert!(output.status.success());
    assert_eq!(stdout(&output), format!("{text}\n"));
}

#[test]
fn incomplete_and_unknown_streams_fail_without_provisional_output() {
    for input in [
        "unfinished-item",
        "partial",
        "unknown-event",
        "unknown-item",
    ] {
        let output = command(input).output().unwrap();
        assert!(!output.status.success(), "{input}");
        assert!(stdout(&output).is_empty(), "{input}: {}", stdout(&output));
        assert!(stderr(&output).contains("outcome is unknown"));
    }
}

#[test]
fn provider_failure_details_are_suppressed_at_cli_boundary() {
    let output = command("failed").output().unwrap();
    assert!(!output.status.success());
    assert!(!stderr(&output).contains("mock provider failed"));
    assert!(stderr(&output).contains("(details hidden)"));
}

#[test]
fn line_controls_separate_queued_steering_from_followups() {
    let output = lines("steer\n/steer   exact 🦀  \n/queue follow-up\n");
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stdout(&output), "  exact 🦀  \nfollow-up\n");
    let log = stderr(&output);
    assert!(log.find("Steering #1 queued").unwrap() < log.find("Steering #1: accepted").unwrap());
    assert!(log.contains("Follow-up queued"));
}

#[test]
fn failed_turn_holds_queued_followups_without_automatic_replay() {
    let output = lines("failed\n/queue must-not-dispatch\n");
    assert!(!output.status.success());
    assert!(stdout(&output).is_empty());
    assert!(stderr(&output).contains("remain unsent at EOF"));
}

#[test]
fn line_controls_cancel_while_steering_ack_is_delayed() {
    let mut command = base_command(true);
    let args: Vec<_> = command.get_args().collect();
    let position = args.iter().position(|arg| *arg == "--workspace").unwrap();
    let marker = Path::new(args[position + 1]).join("steering-received");
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    input.write_all(b"steer-wait\n/steer adjustment\n").unwrap();
    let started = std::time::Instant::now();
    while !marker.exists() {
        if started.elapsed() > std::time::Duration::from_secs(5) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("fixture never received steering");
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let cancel_at = std::time::Instant::now();
    input.write_all(b"/cancel\n").unwrap();
    drop(input);
    let output = finish(child);
    eprintln!(
        "CLI cancel after mock received steering: {} ms to terminal result (two 120 ms fixture delays)",
        cancel_at.elapsed().as_millis()
    );
    assert!(!output.status.success());
    let log = stderr(&output);
    assert!(log.contains("acknowledged interruption"), "{log}");
    assert!(!log.contains("outcome unknown"));
    assert!(
        log.find("interruption requested").unwrap()
            < log.find("acknowledged interruption").unwrap()
    );
}

#[test]
fn hermes_cli_exact_input_completed_projection_and_denial() {
    for (input, expected) in [
        ("  exact\n  🦀  ", "  exact\n  🦀  "),
        ("interleaved", "first message 🦀\nsecond"),
        ("callbacks", "all callbacks denied"),
    ] {
        let output = hermes_command().args(["--prompt", input]).output().unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        assert_eq!(stdout(&output), format!("{expected}\n"));
        assert!(stderr(&output).contains("No external OS sandbox or read boundary is verified"));
    }
}
#[test]
fn hermes_cli_incomplete_output_and_rpc_errors_are_never_answers() {
    for input in [
        "partial",
        "unknown-event",
        "unfinished-tool",
        "rpc-error",
        "refusal",
    ] {
        let output = hermes_command().args(["--prompt", input]).output().unwrap();
        assert!(!output.status.success(), "{input}");
        assert!(stdout(&output).is_empty());
        assert!(!stderr(&output).contains("private provider diagnostic"));
    }
}
#[test]
fn hermes_cli_followups_are_exact_and_not_live_steering() {
    let mut child = hermes_command()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"first\n/steer must-not-dispatch\n/queue second\n")
        .unwrap();
    let output = finish(child);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stdout(&output), "first\nsecond\n");
    assert!(stderr(&output).contains("Live steering is available in Codex mode"));
}
#[test]
fn hermes_cli_cancel_during_delayed_permission_reports_actual_ack() {
    let mut command = hermes_command();
    let args: Vec<_> = command.get_args().collect();
    let position = args.iter().position(|arg| *arg == "--workspace").unwrap();
    let marker = Path::new(args[position + 1]).join("tool-waiting");
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    input.write_all(b"cancel-permission\n").unwrap();
    let started = std::time::Instant::now();
    while !marker.exists() {
        if started.elapsed() > std::time::Duration::from_secs(5) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("ACP mock did not start its tool");
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let cancel_at = std::time::Instant::now();
    input.write_all(b"/cancel\n").unwrap();
    drop(input);
    let output = finish(child);
    eprintln!(
        "ACP CLI cancellation acknowledgment: {} ms (120 ms mock delay)",
        cancel_at.elapsed().as_millis()
    );
    assert!(!output.status.success());
    assert!(stdout(&output).is_empty());
    let log = stderr(&output);
    assert!(log.contains("Hermes interruption requested"), "{log}");
    assert!(log.contains("acknowledged interruption"), "{log}");
    assert!(!log.contains("outcome unknown"));
    assert!(
        log.find("interruption requested").unwrap()
            < log.find("acknowledged interruption").unwrap()
    );
}

#[cfg(target_os = "macos")]
mod mac_terminal {
    use super::*;
    use std::ffi::{c_int, c_void};
    use std::fs::File;
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::sync::mpsc::{self, Receiver};
    use std::sync::{Arc, atomic::AtomicBool};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};

    #[repr(C)]
    struct Size {
        rows: u16,
        columns: u16,
        x: u16,
        y: u16,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Termios {
        input: u64,
        output: u64,
        control: u64,
        local: u64,
        chars: [u8; 20],
        input_speed: u64,
        output_speed: u64,
    }
    unsafe extern "C" {
        fn openpty(
            master: *mut c_int,
            slave: *mut c_int,
            name: *mut u8,
            term: *const c_void,
            size: *const Size,
        ) -> c_int;
        fn tcgetattr(fd: c_int, term: *mut Termios) -> c_int;
        fn ioctl(fd: c_int, request: usize, ...) -> c_int;
        fn kill(pid: c_int, signal: c_int) -> c_int;
        fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
    }
    fn attributes(file: &File) -> Termios {
        let mut value = std::mem::MaybeUninit::uninit();
        // Darwin's termios ABI, matching the supported production platform.
        assert_eq!(
            unsafe { tcgetattr(file.as_raw_fd(), value.as_mut_ptr()) },
            0
        );
        unsafe { value.assume_init() }
    }
    struct Pty {
        child: Child,
        master: File,
        slave: Option<File>,
        saved: Termios,
        rx: Receiver<Vec<u8>>,
        reader: Option<JoinHandle<()>>,
        read_stop: Arc<AtomicBool>,
        output: Vec<u8>,
        workspace: PathBuf,
    }
    impl Pty {
        fn spawn() -> Self {
            let mut command = base_command(false);
            let args: Vec<_> = command.get_args().collect();
            let position = args.iter().position(|arg| *arg == "--workspace").unwrap();
            let workspace = PathBuf::from(args[position + 1]);
            let (mut master, mut slave) = (-1, -1);
            let size = Size {
                rows: 30,
                columns: 80,
                x: 0,
                y: 0,
            };
            assert_eq!(
                unsafe {
                    openpty(
                        &mut master,
                        &mut slave,
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        &size,
                    )
                },
                0
            );
            let master = unsafe { File::from_raw_fd(master) };
            // Test-side PTY I/O also needs bounded waits. Duplicates share this
            // nonblocking file status, so both the writer and reader can stop.
            let flags = unsafe { fcntl(master.as_raw_fd(), 3) };
            assert!(flags >= 0);
            assert_eq!(unsafe { fcntl(master.as_raw_fd(), 4, flags | 4) }, 0);
            let slave = unsafe { File::from_raw_fd(slave) };
            let saved = attributes(&slave);
            command
                .env("TERM", "xterm-256color")
                .env("NO_COLOR", "1")
                .stdin(Stdio::from(slave.try_clone().unwrap()))
                .stdout(Stdio::from(slave.try_clone().unwrap()))
                .stderr(Stdio::from(slave.try_clone().unwrap()));
            let child = command.spawn().unwrap();
            drop(command);
            let mut read = master.try_clone().unwrap();
            let (tx, rx) = mpsc::sync_channel(32);
            let read_stop = Arc::new(AtomicBool::new(false));
            let stop = Arc::clone(&read_stop);
            let reader = thread::spawn(move || {
                let mut bytes = [0; 4096];
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    match read.read(&mut bytes) {
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(2));
                        }
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            if tx.send(bytes[..n].to_vec()).is_err() {
                                break;
                            }
                        }
                    }
                }
            });
            Self {
                child,
                master,
                slave: Some(slave),
                saved,
                rx,
                reader: Some(reader),
                read_stop,
                output: Vec::new(),
                workspace,
            }
        }
        fn send(&mut self, bytes: &[u8]) {
            let started = Instant::now();
            let mut remaining = bytes;
            while !remaining.is_empty() {
                assert!(
                    started.elapsed() < Duration::from_secs(5),
                    "test PTY write timed out"
                );
                match self.master.write(remaining) {
                    Ok(0) => panic!("PTY closed during write"),
                    Ok(n) => remaining = &remaining[n..],
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("PTY write: {error}"),
                }
                // A large paste and concurrent redraw can fill both PTY
                // directions. Consume bounded display data while writing input.
                while let Ok(bytes) = self.rx.try_recv() {
                    self.output.extend(bytes);
                }
                assert!(self.output.len() <= 4 * 1024 * 1024);
            }
        }
        fn contains(&self, needle: &str) -> bool {
            self.output
                .windows(needle.len())
                .any(|w| w == needle.as_bytes())
        }
        fn wait_for(&mut self, needle: &str) {
            let start = Instant::now();
            while !self.contains(needle) {
                assert!(
                    start.elapsed() < Duration::from_secs(5),
                    "PTY did not emit {needle:?}; tail: {}",
                    String::from_utf8_lossy(&self.output[self.output.len().saturating_sub(1500)..])
                );
                if let Ok(bytes) = self.rx.recv_timeout(Duration::from_millis(10)) {
                    self.output.extend(bytes);
                }
                assert!(
                    self.output.len() <= 4 * 1024 * 1024,
                    "PTY fixture output exceeded bound"
                );
            }
        }
        fn resize(&self, columns: u16) {
            let size = Size {
                rows: 30,
                columns,
                x: 0,
                y: 0,
            };
            assert_eq!(
                unsafe { ioctl(self.master.as_raw_fd(), 0x80087467, &size) },
                0
            );
        }
        fn wait_exit_and_check_restore(&mut self) {
            let start = Instant::now();
            while self.child.try_wait().unwrap().is_none() {
                assert!(
                    start.elapsed() < Duration::from_secs(5),
                    "PTY child did not exit"
                );
                if let Ok(bytes) = self.rx.recv_timeout(Duration::from_millis(10)) {
                    self.output.extend(bytes);
                }
            }
            assert_eq!(
                attributes(self.slave.as_ref().unwrap()),
                self.saved,
                "terminal attributes must be restored"
            );
        }
    }
    impl Drop for Pty {
        fn drop(&mut self) {
            if self.child.try_wait().ok().flatten().is_none() {
                unsafe {
                    kill(self.child.id() as c_int, 15);
                }
                let start = Instant::now();
                while self.child.try_wait().ok().flatten().is_none()
                    && start.elapsed() < Duration::from_secs(7)
                {
                    let _ = self.rx.recv_timeout(Duration::from_millis(10));
                }
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
            self.slave.take();
            self.read_stop.store(true, Ordering::Relaxed);
            // Release producer backpressure before joining its read loop.
            let (_, empty) = mpsc::channel();
            self.rx = empty;
            if let Some(reader) = self.reader.take() {
                let _ = reader.join();
            }
        }
    }

    #[test]
    fn raw_controls_edit_steer_resize_cancel_and_restore() {
        let mut pty = Pty::spawn();
        pty.wait_for("Codex owns delegated context");
        pty.send(b"steer-wait\r");
        pty.wait_for("Codex thread: mock-thread");
        let steering = Instant::now();
        pty.send(b"/steer axc\x1b[D\x7fb\r");
        pty.wait_for("Steering #1 queued");
        let queued_ms = steering.elapsed().as_millis();
        pty.resize(24);
        pty.wait_for("Steering #1: accepted");
        assert_eq!(
            fs::read_to_string(pty.workspace.join("steering-input")).unwrap(),
            "abc"
        );
        pty.resize(120);
        pty.send(b"/status\r");
        pty.wait_for("steering pending: 0");
        let cancel = Instant::now();
        pty.send(b"\x03");
        pty.wait_for("interruption requested");
        let requested_ms = cancel.elapsed().as_millis();
        assert!(!pty.contains("acknowledged interruption"));
        pty.wait_for("acknowledged interruption");
        eprintln!(
            "Mac PTY: steering queued in {queued_ms} ms; Ctrl-C request shown in {requested_ms} ms; provider interruption in {} ms (120 ms mock terminal delay)",
            cancel.elapsed().as_millis()
        );
        pty.send(b"/quit\r");
        pty.wait_exit_and_check_restore();
    }

    #[test]
    fn raw_oversized_bracketed_paste_never_launches_a_turn() {
        let mut pty = Pty::spawn();
        pty.wait_for("Codex owns delegated context");
        let mut paste = b"\x1b[200~/quit\n".to_vec();
        paste.extend(vec![b'x'; 65536]);
        paste.extend_from_slice(b"\x1b[201~\r");
        pty.send(&paste);
        pty.wait_for("entire input discarded without sending");
        assert!(!pty.contains("Codex launch"));
        pty.send(b"/quit\r");
        pty.wait_exit_and_check_restore();
    }
}
