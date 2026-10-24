//! Exercise the real prompt and interruption handler on a controlling PTY.
use super::*;
use std::{
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::process::CommandExt,
    },
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

#[cfg_attr(target_os = "linux", link(name = "util"))]
unsafe extern "C" {
    fn fcntl(fd: i32, command: i32, ...) -> i32;
    fn openpty(
        master: *mut i32,
        slave: *mut i32,
        name: *mut u8,
        termios: *const std::ffi::c_void,
        size: *const std::ffi::c_void,
    ) -> i32;
    fn setsid() -> i32;
    fn ioctl(fd: i32, request: usize, ...) -> i32;
    fn kill(pid: i32, signal: i32) -> i32;
}

struct Prompt {
    child: Child,
    master: File,
    // Darwin can discard unread output when the final slave descriptor closes.
    _slave: File,
    transcript: String,
}
impl Prompt {
    fn start(home: &Path) -> Self {
        let (mut master, mut slave) = (-1, -1);
        assert_eq!(
            unsafe {
                openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                )
            },
            0,
            "openpty: {}",
            io::Error::last_os_error()
        );
        let master = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        for fd in [master.as_raw_fd(), slave.as_raw_fd()] {
            assert_eq!(unsafe { fcntl(fd, 2, 1) }, 0); // F_SETFD, FD_CLOEXEC
        }
        let flags = unsafe { fcntl(master.as_raw_fd(), crate::unix_flags::F_GETFL) };
        assert!(flags >= 0);
        assert_eq!(
            unsafe {
                fcntl(
                    master.as_raw_fd(),
                    crate::unix_flags::F_SETFL,
                    flags | crate::unix_flags::O_NONBLOCK,
                )
            },
            0
        );
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "launch::model::prompt_tests::signals_cancel_model_prompt_without_enter",
                "--nocapture",
            ])
            .env_clear()
            .env("HOME", home)
            .env("VADE_TEST_MODEL_PROMPT", home)
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave.try_clone().unwrap()));
        unsafe {
            command.pre_exec(|| {
                if setsid() < 0 {
                    return Err(io::Error::last_os_error());
                }
                #[cfg(target_os = "macos")]
                let tiocsctty = 0x20007461;
                #[cfg(target_os = "linux")]
                let tiocsctty = 0x540e;
                if ioctl(0, tiocsctty, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        Self {
            child,
            master,
            _slave: slave,
            transcript: String::new(),
        }
    }
    fn drain(&mut self) {
        let mut buffer = [0; 4096];
        loop {
            match self.master.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => self
                    .transcript
                    .push_str(&String::from_utf8_lossy(&buffer[..count])),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => panic!("PTY read: {error}; {}", self.transcript),
            }
            assert!(
                self.transcript.len() < 1024 * 1024,
                "unbounded prompt output"
            );
        }
    }
    fn wait_for_prompt(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            self.drain();
            if self
                .transcript
                .contains("FP4-Local model folder (existing weights): ")
            {
                return;
            }
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "exited before prompt: {}",
                self.transcript
            );
            assert!(
                Instant::now() < deadline,
                "no model prompt: {}",
                self.transcript
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    fn interrupt(&mut self, signal: i32) {
        if signal == 2 {
            // A partially typed answer must be cancelled without ever sending Enter.
            self.master
                .write_all(b"/unfinished/model/path\x03")
                .unwrap();
        } else {
            assert_eq!(unsafe { kill(self.child.id() as i32, signal) }, 0);
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            self.drain();
            if let Some(status) = self.child.try_wait().unwrap() {
                self.drain();
                assert_eq!(status.code(), Some(1), "{}", self.transcript);
                assert!(
                    self.transcript
                        .contains("Model selection interrupted; nothing was installed."),
                    "{}",
                    self.transcript
                );
                return;
            }
            assert!(
                Instant::now() < deadline,
                "signal {signal} did not cancel model prompt: {}",
                self.transcript
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
impl Drop for Prompt {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn signals_cancel_model_prompt_without_enter() {
    if let Some(home) = std::env::var_os("VADE_TEST_MODEL_PROMPT") {
        let home = PathBuf::from(home);
        let root = home.join(".vade");
        let _interruptions = crate::runner::Interruptions::install();
        let flags = unsafe { fcntl(0, crate::unix_flags::F_GETFL) };
        let mut launcher = crate::launch::Launcher {
            root: root.clone(),
            home,
            runner: crate::runner::SystemRunner { dry: false },
            detector: crate::launch::Native,
        };
        // Run the production lifecycle through the prompt, before hardware
        // detection, so the same regression executes on both Mac and Linux.
        let error = launcher
            .execute(crate::cli::Command::Launch(Default::default()), false)
            .unwrap_err();
        assert!(error.contains("Model selection interrupted"), "{error}");
        assert_eq!(
            unsafe { fcntl(0, crate::unix_flags::F_GETFL) },
            flags,
            "stdin flags were not restored"
        );
        assert!(
            !root.join("config").exists(),
            "cancelled setup saved defaults"
        );
        assert!(
            !root.join("state/run.conf").exists(),
            "cancelled setup saved an engine"
        );
        let lock = crate::state::Lock::acquire(&root)
            .expect("cancelled launch retained its operation lock");
        drop(lock);
        crate::fail(&error);
    }
    let _process_guard = crate::runner::process_test_lock();
    for (ambiguous, signal) in [(false, 2), (true, 2), (false, 15)] {
        let fixture = super::tests::Fixture::new();
        if ambiguous {
            fixture.model("models/first");
            fixture.model("models/second");
        }
        let mut prompt = Prompt::start(&fixture.0);
        prompt.wait_for_prompt();
        if ambiguous {
            assert!(
                prompt.transcript.contains("More than one model"),
                "{}",
                prompt.transcript
            );
        }
        assert!(
            crate::state::Lock::acquire(&fixture.0.join(".vade")).is_err(),
            "prompt did not hold the launch lock"
        );
        prompt.interrupt(signal);
        let _lock = crate::state::Lock::acquire(&fixture.0.join(".vade")).unwrap();
    }
}
