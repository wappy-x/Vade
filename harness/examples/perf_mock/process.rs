//! Bounded process/PTY accounting for the macOS-only benchmark.
use std::{
    fs::File,
    io::{self, Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

#[repr(C)]
#[derive(Default)]
struct Timeval {
    seconds: i64,
    micros: i32,
}
#[repr(C)]
#[derive(Default)]
struct Usage {
    user: Timeval,
    system: Timeval,
    counters: [i64; 14],
}
#[repr(C)]
struct Poll(i32, i16, i16);
unsafe extern "C" {
    fn wait4(pid: i32, status: *mut i32, options: i32, usage: *mut Usage) -> i32;
    fn openpty(
        master: *mut i32,
        slave: *mut i32,
        name: *mut u8,
        termios: *const std::ffi::c_void,
        size: *const std::ffi::c_void,
    ) -> i32;
    fn poll(fds: *mut Poll, count: u32, timeout: i32) -> i32;
    fn fcntl(fd: i32, command: i32, ...) -> i32;
}
#[repr(C)]
#[derive(Default)]
pub struct LiveUsage {
    uuid: [u8; 16],
    pub counters: [u64; 10],
}
#[link(name = "proc")]
unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
}
#[repr(C)]
#[derive(Default)]
struct Timebase {
    numerator: u32,
    denominator: u32,
}
unsafe extern "C" {
    fn mach_timebase_info(info: *mut Timebase) -> i32;
}
pub fn tick_seconds(ticks: u64) -> io::Result<f64> {
    let mut scale = Timebase::default();
    if unsafe { mach_timebase_info(&mut scale) } != 0 || scale.denominator == 0 {
        return Err(io::Error::other("cannot read Mach clock scale"));
    }
    Ok(ticks as f64 * scale.numerator as f64 / scale.denominator as f64 / 1e9)
}
pub fn usage(pid: u32) -> io::Result<LiveUsage> {
    let mut result = LiveUsage::default();
    if unsafe { proc_pid_rusage(pid as i32, 0, (&mut result as *mut LiveUsage).cast()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(result)
}
pub fn timeout() -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, "mock benchmark deadline exceeded")
}

/// The benchmark alone reaps its child, retaining wait4 CPU/RSS accounting.
/// Every failure path stops that same child; no process-name lookup is used.
pub struct OwnedChild {
    pub child: Child,
    reaped: bool,
}
impl OwnedChild {
    pub fn spawn(command: &mut Command) -> io::Result<Self> {
        Ok(Self {
            child: command.env_remove("VADE_API_KEY_FILE").spawn()?,
            reaped: false,
        })
    }
    fn try_wait(&mut self) -> io::Result<Option<(f64, f64, i64)>> {
        let mut result = Usage::default();
        let mut status = 0;
        let got = unsafe { wait4(self.child.id() as i32, &mut status, 1, &mut result) }; // WNOHANG
        if got == 0 {
            return Ok(None);
        }
        if got < 0 {
            let error = io::Error::last_os_error();
            return if error.kind() == io::ErrorKind::Interrupted {
                Ok(None)
            } else {
                Err(error)
            };
        }
        self.reaped = true;
        if status != 0 {
            return Err(io::Error::other(format!(
                "benchmark child exited unsuccessfully: wait status {status}"
            )));
        }
        Ok(Some((
            result.user.seconds as f64 + result.user.micros as f64 / 1e6,
            result.system.seconds as f64 + result.system.micros as f64 / 1e6,
            result.counters[0],
        )))
    }
    pub fn wait(mut self, deadline: Instant) -> io::Result<(f64, f64, i64)> {
        loop {
            if let Some(result) = self.try_wait()? {
                return Ok(result);
            }
            if Instant::now() >= deadline {
                return Err(timeout());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub struct Workspace(pub PathBuf);
impl Workspace {
    pub fn new() -> io::Result<Self> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!("vade-perf-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path)?;
        Ok(Self(path.canonicalize()?))
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub struct Pty {
    pub input: File,
    _slave: File,
}
impl Pty {
    pub fn attach(command: &mut Command) -> io::Result<Self> {
        let (mut master, mut slave) = (0, 0);
        let size = [24u16, 110, 0, 0];
        if unsafe {
            openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null(),
                (&size as *const [u16; 4]).cast(),
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        let master = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        for fd in [master.as_raw_fd(), slave.as_raw_fd()] {
            if unsafe { fcntl(fd, 2, 1) } != 0 {
                return Err(io::Error::last_os_error());
            } // FD_CLOEXEC
        }
        command
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .stdin(Stdio::from(slave.try_clone()?))
            .stdout(Stdio::from(slave.try_clone()?));
        // Keep a slave open until all output is read: Darwin may otherwise
        // discard unread bytes at the last slave close.
        Ok(Self {
            input: master,
            _slave: slave,
        })
    }
    pub fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.input.write_all(bytes)
    }
}

pub struct Output {
    file: File,
    pending: Vec<u8>,
    pub all: Vec<u8>,
    raw: bool,
}
impl Output {
    pub fn new(file: File, raw: bool) -> Self {
        Self {
            file,
            pending: Vec::new(),
            all: Vec::new(),
            raw,
        }
    }
    fn read(&mut self, deadline: Instant) -> io::Result<bool> {
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(timeout());
            }
            let mut ready = Poll(self.file.as_raw_fd(), 1, 0);
            let got = unsafe { poll(&mut ready, 1, left.as_millis().clamp(1, 1000) as i32) };
            if got < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            if got == 0 {
                continue;
            }
            let mut bytes = [0; 16 * 1024];
            match self.file.read(&mut bytes) {
                Ok(0) if self.raw && ready.2 & 16 == 0 => continue,
                Ok(0) => return Ok(false),
                Ok(n) => {
                    if self.all.len() + n > 64 * 1024 * 1024 {
                        return Err(io::Error::other("benchmark output exceeds 64 MiB"));
                    }
                    self.pending.extend_from_slice(&bytes[..n]);
                    self.all.extend_from_slice(&bytes[..n]);
                    return Ok(true);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
    }
    pub fn until(&mut self, needle: &[u8], deadline: Instant) -> io::Result<Instant> {
        let mut searched = 0;
        loop {
            if let Some(at) = self.pending[searched..]
                .windows(needle.len())
                .position(|value| value == needle)
            {
                let at = searched + at;
                if frame_end(&self.pending, at, needle.len(), self.raw).is_some() {
                    let seen = Instant::now();
                    // Preserve a second receipt coalesced into this same frame.
                    self.pending.drain(..at + needle.len());
                    return Ok(seen);
                }
                searched = at;
            } else {
                searched = self
                    .pending
                    .len()
                    .saturating_sub(needle.len().saturating_sub(1));
            }
            let got = self.read(deadline).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "{error}; waiting for {:?}; output tail {:?}",
                        String::from_utf8_lossy(needle),
                        String::from_utf8_lossy(&self.all[self.all.len().saturating_sub(512)..])
                    ),
                )
            })?;
            if !got {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "missing benchmark receipt {}",
                        String::from_utf8_lossy(needle)
                    ),
                ));
            }
        }
    }
    pub fn drain(&mut self, deadline: Instant) -> io::Result<()> {
        while self.read(deadline)? {}
        Ok(())
    }
    /// Keep draining the PTY while the child exits; a final redraw can fill its
    /// kernel buffer and block the child before wait4 can observe termination.
    pub fn finish(
        &mut self,
        mut child: OwnedChild,
        deadline: Instant,
    ) -> io::Result<(f64, f64, i64)> {
        loop {
            if let Some(result) = child.try_wait()? {
                return Ok(result);
            }
            if Instant::now() >= deadline {
                return Err(timeout());
            }
            match self.read(deadline.min(Instant::now() + Duration::from_millis(10))) {
                Ok(false) => return child.wait(deadline),
                Ok(true) => {}
                Err(error) if error.kind() == io::ErrorKind::TimedOut => {}
                Err(error) => return Err(error),
            }
        }
    }
}
fn frame_end(bytes: &[u8], at: usize, size: usize, raw: bool) -> Option<usize> {
    const OPEN: &[u8] = b"\x1b[?2026h";
    const CLOSE: &[u8] = b"\x1b[?2026l";
    let end = at + size;
    let framed = raw
        && bytes[..at]
            .windows(OPEN.len())
            .rposition(|x| x == OPEN)
            .is_some_and(|start| !bytes[start..at].windows(CLOSE.len()).any(|x| x == CLOSE));
    if framed {
        bytes[end..]
            .windows(CLOSE.len())
            .position(|x| x == CLOSE)
            .map(|n| end + n + CLOSE.len())
    } else {
        Some(end)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_child_fixture() {
        match std::env::var("VADE_PERF_CHILD_FIXTURE").as_deref() {
            Ok("output") => std::io::stdout()
                .write_all(&vec![b'x'; 2 * 1024 * 1024])
                .unwrap(),
            Ok("wait") => loop {
                std::thread::sleep(Duration::from_secs(1));
            },
            _ => {}
        }
    }
    fn fixture(mode: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "process::tests::owned_child_fixture",
                "--nocapture",
            ])
            .env("VADE_PERF_CHILD_FIXTURE", mode)
            .stderr(Stdio::null());
        command
    }
    #[test]
    fn exit_drains_more_than_the_terminal_buffer() {
        let mut command = fixture("output");
        let pty = Pty::attach(&mut command).unwrap();
        let child = OwnedChild::spawn(&mut command).unwrap();
        let mut output = Output::new(pty.input.try_clone().unwrap(), true);
        output
            .finish(child, Instant::now() + Duration::from_secs(5))
            .unwrap();
        // The last PTY write can remain buffered after wait4 reports success.
        // Reading >1MiB is enough to prove that shutdown cannot wait blindly.
        assert!(output.all.len() > 1024 * 1024);
    }
    #[test]
    fn timeout_stops_and_reaps_only_the_owned_child() {
        let child = OwnedChild::spawn(&mut fixture("wait")).unwrap();
        let pid = child.child.id() as i32;
        let start = Instant::now();
        assert_eq!(
            child
                .wait(start + Duration::from_millis(50))
                .unwrap_err()
                .kind(),
            io::ErrorKind::TimedOut
        );
        assert!(start.elapsed() < Duration::from_secs(3));
        let mut status = 0;
        let mut usage = Usage::default();
        assert_eq!(unsafe { wait4(pid, &mut status, 1, &mut usage) }, -1);
        assert_eq!(io::Error::last_os_error().raw_os_error(), Some(10)); // ECHILD
    }
    #[test]
    fn marker_waits_for_the_complete_terminal_frame() {
        let bytes = b"\x1b[?2026hBEGIN payload\x1b[?2026l";
        for length in 13..bytes.len() {
            assert!(frame_end(&bytes[..length], 8, 5, true).is_none());
        }
        assert_eq!(frame_end(bytes, 8, 5, true), Some(bytes.len()));
        assert_eq!(frame_end(b"BEGIN", 0, 5, false), Some(5));
    }
}
