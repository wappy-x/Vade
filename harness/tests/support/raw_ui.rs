//! Shared first-party macOS PTY fixture for actual native/delegated UI checks.
use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

#[repr(C)]
struct Poll(i32, i16, i16);
static PTY_OPEN: std::sync::Mutex<()> = std::sync::Mutex::new(());

unsafe extern "C" {
    fn poll(fds: *mut Poll, count: u32, timeout: i32) -> i32;
    fn fcntl(fd: i32, command: i32, ...) -> i32;
    fn ioctl(fd: i32, request: usize, ...) -> i32;
    fn openpty(
        master: *mut i32,
        slave: *mut i32,
        name: *mut u8,
        termios: *const std::ffi::c_void,
        size: *const std::ffi::c_void,
    ) -> i32;
}
pub struct Ui {
    child: Child,
    slave: Option<File>,
    input: File,
    output: Receiver<Vec<u8>>,
    pending: RefCell<Vec<u8>>,
    reader: Option<thread::JoinHandle<()>>,
}
impl Ui {
    #[allow(dead_code)]
    pub fn discard_output(&self) {
        self.pending.borrow_mut().clear();
        while self.output.try_recv().is_ok() {}
    }
    pub fn resize(&self, rows: u16, columns: u16) {
        let size = [rows, columns, 0u16, 0u16];
        assert_eq!(
            unsafe {
                ioctl(
                    std::os::fd::AsRawFd::as_raw_fd(&self.input),
                    0x80087467,
                    &size,
                )
            },
            0
        );
    }

    pub fn start(mode: &str, endpoint: &str, workspace: &str, children: bool) -> Self {
        Self::start_with_args(
            mode,
            endpoint,
            workspace,
            if children { &["--children"] } else { &[] },
        )
    }
    pub fn start_with_args(mode: &str, endpoint: &str, workspace: &str, extra: &[&str]) -> Self {
        Self::start_with_flags(mode, endpoint, workspace, extra, false)
    }
    #[allow(dead_code)] // This shared fixture is also compiled by contract tests.
    pub fn start_nonblocking(workspace: &str, extra: &[&str]) -> Self {
        Self::start_with_flags("agent", "http://localhost:9", workspace, extra, true)
    }
    fn start_with_flags(
        mode: &str,
        endpoint: &str,
        workspace: &str,
        extra: &[&str],
        nonblocking: bool,
    ) -> Self {
        let (mut master, mut slave) = (0, 0);
        let size = [24u16, 110, 0, 0];
        // Concurrent Darwin openpty calls can race during slave allocation.
        let allocation = PTY_OPEN.lock().unwrap();
        // Darwin openpty defaults, each returned descriptor gets one owner.
        assert_eq!(
            unsafe {
                openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    (&size as *const [u16; 4]).cast(),
                )
            },
            0,
            "openpty: {}",
            std::io::Error::last_os_error()
        );
        for fd in [master, slave] {
            assert_eq!(unsafe { fcntl(fd, 2, 1) }, 0); // F_SETFD, FD_CLOEXEC
        }
        drop(allocation);
        let master = unsafe { File::from_raw_fd(master) };
        if nonblocking {
            assert_eq!(unsafe { fcntl(slave, 4, fcntl(slave, 3) | 4) }, 0);
        }
        let slave = unsafe { File::from_raw_fd(slave) };
        // Model a terminal emulator keeping its slave open: Darwin can discard
        // unread output on the last slave close, even after a successful flush.
        let keep_slave = slave.try_clone().unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_vade-harness"))
            .args([
                mode,
                "--state-in-project",
                "--endpoint",
                endpoint,
                "--model",
                "mock",
                "--workspace",
                workspace,
            ])
            .args(extra)
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave))
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut output = master.try_clone().unwrap();
        let (tx, rx) = mpsc::channel();
        let reader = thread::spawn(move || {
            let mut buffer = [0; 4096];
            let mut ready = Poll(output.as_raw_fd(), 1, 0);
            loop {
                if unsafe { poll(&mut ready, 1, -1) } < 0 {
                    continue;
                }
                let n = match output.read(&mut buffer) {
                    Ok(n) => n,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                };
                // Darwin can read zero during termios restoration, before the
                // exit receipt. Only POLLHUP confirms the slave has closed.
                if n == 0 && ready.2 & 16 == 0 {
                    continue;
                }
                if n == 0 || tx.send(buffer[..n].to_vec()).is_err() {
                    break;
                }
            }
        });
        let ui = Self {
            child,
            slave: Some(keep_slave),
            input: master,
            output: rx,
            pending: RefCell::new(Vec::new()),
            reader: Some(reader),
        };
        ui.until(
            if extra.contains(&"--plain") {
                "> "
            } else {
                "│> "
            },
            Duration::from_secs(3),
        );
        ui
    }
    pub fn send(&mut self, bytes: &[u8]) {
        self.input.write_all(bytes).unwrap();
    }
    #[allow(dead_code)]
    pub fn wait_exit(&mut self) {
        // Bounded: a cleanup hang must fail the test, never hang the suite.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "UI exited with {status}");
                return;
            }
            // On timeout the assert fails and Drop kills the child.
            assert!(
                Instant::now() < deadline,
                "UI did not exit within 10s"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
    pub fn until(&self, needle: &str, budget: Duration) -> String {
        self.until_any(&[needle], budget)
    }
    pub fn until_any(&self, needles: &[&str], budget: Duration) -> String {
        // Match an observed UI event; short call-site budgets are not latency
        // contracts. Allow scheduler contention, but keep a bounded hang guard.
        let budget = budget.max(Duration::from_secs(30));
        let start = Instant::now();
        let mut output = self.pending.take();
        loop {
            if let Some((matched, complete)) = receipt(&output, needles) {
                // Return a complete frame, but retain everything after the
                // matched event: a second receipt may share this same read.
                *self.pending.borrow_mut() = output[matched..].to_vec();
                return String::from_utf8_lossy(&output[..complete]).into_owned();
            }
            output.extend(
                self.output
                    .recv_timeout(budget.saturating_sub(start.elapsed()))
                    .unwrap_or_else(|error| panic!("UI did not respond before deadline ({error:?}); waiting for {needles:?}; captured {:?}", String::from_utf8_lossy(&output))),
            );
            assert!(output.len() <= 2 * 1024 * 1024);
        }
    }
}
fn receipt(output: &[u8], needles: &[&str]) -> Option<(usize, usize)> {
    const OPEN: &[u8] = b"\x1b[?2026h";
    const CLOSE: &[u8] = b"\x1b[?2026l";
    let (start, matched) = needles
        .iter()
        .filter_map(|needle| {
            output
                .windows(needle.len())
                .position(|s| s == needle.as_bytes())
                .map(|start| (start, start + needle.len()))
        })
        .min()?;
    let prefix = &output[..start];
    let framed = prefix
        .windows(OPEN.len())
        .rposition(|s| s == OPEN)
        .is_some_and(|at| !prefix[at..].windows(CLOSE.len()).any(|s| s == CLOSE));
    // Synchronized-output end is the readiness signal for all rows in a
    // redraw. A PTY read boundary can split the footer from the draft row.
    let complete = if framed {
        matched
            + output[matched..].windows(CLOSE.len()).position(|s| s == CLOSE)?
            + CLOSE.len()
    } else {
        matched
    };
    Some((matched, complete))
}
#[test]
fn receipts_wait_for_split_frames_and_retain_coalesced_events() {
    let bytes = b"\x1b[?2026hready draft\x1b[?2026lnext";
    let (matched, complete) = receipt(bytes, &["ready"]).unwrap();
    for end in matched..complete {
        assert!(receipt(&bytes[..end], &["ready"]).is_none());
    }
    assert!(String::from_utf8_lossy(&bytes[..complete]).contains("draft"));
    assert!(receipt(&bytes[matched..], &["draft"]).is_some());
    assert!(receipt(&bytes[matched..], &["next"]).is_some());
}
impl Drop for Ui {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.slave.take();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
