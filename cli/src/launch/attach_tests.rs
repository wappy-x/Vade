//! Run the real attach doubles in a disposable process, including interactive
//! chat, whose normal runner deliberately has no timeout.
use super::*;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command as Process, Stdio};

const CHILD_TEST: &str = "launch::tests::attach_tests::attach_child";

unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

struct ChildGroup(Option<Child>);
impl Drop for ChildGroup {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            // The child is still unreaped, so its private group ID cannot have
            // been reused. Chat and its descendants inherit this group.
            unsafe {
                kill(-(child.id() as i32), 9);
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn assert_process_lock_held() {
    assert!(
        matches!(
            crate::runner::PROCESS_TEST_LOCK.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        ),
        "real attach subprocesses must hold the process-test mutex"
    );
}

fn supervise(f: &Fixture, hang: bool) -> bool {
    // Fixture holds this process's mutex from setup through child cleanup. A
    // mutex in the separate child alone cannot serialize this parent's fork.
    assert_process_lock_held();
    let log = f.base.join("attach-child.log");
    let stdout = fs::File::create(&log).unwrap();
    let stderr = stdout.try_clone().unwrap();
    let child = Process::new(std::env::current_exe().unwrap())
        .args(["--exact", CHILD_TEST, "--nocapture", "--test-threads=1"])
        .env("VADE_TEST_ATTACH_BASE", &f.base)
        .env("VADE_TEST_ATTACH_HANG", if hang { "1" } else { "0" })
        .env_remove("VADE_MODELS")
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .process_group(0)
        .spawn()
        .unwrap();
    let mut child = ChildGroup(Some(child));
    let mut deadline = Instant::now() + Duration::from_secs(30);
    let mut observed_hang = false;
    let timed_out = loop {
        if let Some(status) = child.0.as_mut().unwrap().try_wait().unwrap() {
            child.0 = None;
            assert!(status.success(), "{}", fs::read_to_string(&log).unwrap());
            break false;
        }
        // Give a loaded host the full setup budget, then deliberately exercise
        // the same deadline/cleanup path once the blocked chat is running.
        if hang
            && !observed_hang
            && fs::read_to_string(f.base.join("chat-started"))
                .ok()
                .and_then(|pid| pid.trim().parse::<i32>().ok())
                .is_some_and(|pid| pid > 0)
        {
            observed_hang = true;
            deadline = Instant::now() + Duration::from_millis(200);
        }
        if Instant::now() >= deadline {
            break true;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    drop(child); // kill the child group and reap before inspecting or deleting fixtures
    print!("{}", fs::read_to_string(log).unwrap());
    if hang {
        assert!(observed_hang, "chat did not reach the deliberate hang");
    }
    timed_out
}

fn resident_fixture() -> Fixture {
    let f = Fixture::new();
    f.installed();
    f.launcher().execute(Command::Launch(Default::default()), false).unwrap();
    fs::remove_file(f.model.join("config.json")).unwrap();
    f
}

pub(super) fn check_resident_attach() {
    let f = resident_fixture();
    assert!(
        !supervise(&f, false),
        "resident attach exceeded its supervised 30-second deadline"
    );
}

#[test]
fn hung_interactive_attach_is_killed_with_its_descendant() {
    let f = resident_fixture();
    assert!(
        supervise(&f, true),
        "deliberately blocked chat escaped the deadline"
    );
    // Check both the interactive shell and its child, not merely the test
    // process. Allow the OS to reap killed orphans without a latency target.
    for name in ["chat-started", "chat-descendant"] {
        let pid = fs::read_to_string(f.base.join(name))
            .unwrap()
            .trim()
            .parse::<i32>()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while unsafe { kill(pid, 0) } == 0 {
            assert!(
                Instant::now() < deadline,
                "{name} process {pid} survived group cleanup"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    assert!(crate::state::Lock::acquire(&f.root).is_ok());
}

struct ProcessRunner {
    system: crate::runner::SystemRunner,
    calls: Vec<String>,
    base: PathBuf,
    hang: bool,
}
impl Runner for ProcessRunner {
    fn interrupted(&self) -> bool {
        false
    }
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        assert_process_lock_held();
        self.calls.push(exec.label.clone());
        let script = if exec.args == ["__worker"] {
            let fields = text::parse(std::str::from_utf8(&exec.stdin).unwrap()).unwrap();
            match fields["op"].as_str() {
                "inspect" => "while IFS= read -r line; do :; done\nprintf 'alive=true\\nphase=running\\n'",
                "listener" => "while IFS= read -r line; do :; done\nprintf 'listener=owned\\n'",
                op => panic!("resident launch tried to start or reinstall ({op})"),
            }
        } else {
            assert_eq!(exec.label, "local health");
            assert_eq!(exec.program, "curl");
            assert!(exec.args.last().unwrap().starts_with("http://127.0.0.1:"));
            "printf '%s\\n' '{\"status\":\"ready\",\"model\":\"checkpoint\"}'"
        };
        let mut fixture = exec.clone();
        fixture.program = "/bin/sh".into();
        fixture.args = vec!["-c".into(), script.into()];
        fixture.cancellable = false;
        // These two owned jobs finish or clean themselves up well inside the
        // outer deadline; interactive chat inherits the supervised group.
        fixture.timeout = Duration::from_secs(5);
        self.system.run(&fixture)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> {
        assert_process_lock_held();
        assert_eq!(exec.label, "local chat");
        self.calls.push(exec.label.clone());
        let mut fixture = exec.clone();
        fixture.program = "/bin/sh".into();
        let script = if self.hang {
            "sh -c 'printf %s \"$$\" > \"$VADE_ATTACH_DIR/chat-descendant\"; while :; do sleep 60; done' & while ! test -s \"$VADE_ATTACH_DIR/chat-descendant\"; do sleep 0.01; done; printf %s \"$$\" > \"$VADE_ATTACH_DIR/chat-started\"; wait"
        } else {
            "printf 'fixture chat attached\\n'"
        };
        fixture.args = vec!["-c".into(), script.into()];
        fixture.env.insert(
            "VADE_ATTACH_DIR".into(),
            self.base.to_string_lossy().into_owned(),
        );
        self.system.interactive(&fixture)
    }
    fn port_free(&mut self, _: u16) -> Result<(), String> {
        panic!("resident attach must not reserve another port");
    }
    fn pause(&mut self, _: Duration) {
        panic!("resident attach must not poll loading");
    }
}

#[test]
fn attach_child() {
    let Some(base) = std::env::var_os("VADE_TEST_ATTACH_BASE") else {
        return;
    };
    // Keep the mutex explicit at the real fork sites. The parent holds its own
    // copy while spawning and supervising this disposable test process.
    let _process_guard = crate::runner::process_test_lock();
    let base = PathBuf::from(base);
    let home = base.join("home");
    let root = home.join(".vade");
    let config = fs::read(root.join("config")).unwrap();
    let record = fs::read(root.join("state/run.conf")).unwrap();
    let mut launch = Launcher {
        root: root.clone(),
        home,
        runner: ProcessRunner {
            system: crate::runner::SystemRunner { dry: false },
            calls: Vec::new(),
            base,
            hang: std::env::var("VADE_TEST_ATTACH_HANG").unwrap() == "1",
        },
        detector: MockDetector {
            identity: identity(),
            calls: 0,
        },
    };
    let started = Instant::now();
    launch.execute(Command::Launch(Default::default()), false).unwrap();
    let elapsed = started.elapsed();
    assert_eq!(
        launch.runner.calls,
        ["local inspect", "local listener", "local health", "local chat"]
    );
    assert_eq!(launch.detector.calls, 0);
    assert_eq!(fs::read(root.join("config")).unwrap(), config);
    assert_eq!(fs::read(root.join("state/run.conf")).unwrap(), record);
    println!(
        "Mac resident attach with three real subprocesses: {:.3} s",
        elapsed.as_secs_f64()
    );
}
