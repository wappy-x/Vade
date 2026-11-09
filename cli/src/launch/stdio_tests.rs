//! Host checks for the Up pipe and the engine log. The Linux worker replies are
//! scripted. The engine is a real child with the production stream setup.
use super::*;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Command as Process, Stdio};
use std::sync::mpsc;

const MODE: &str = "VADE_TEST_STDIO_MODE";
const ROOT: &str = "VADE_TEST_STDIO_ROOT";
const VALUE: &str = "VADE_TEST_STDIO_VALUE";

struct EngineRunner {
    mock: MockRunner,
    root: PathBuf,
    test: &'static str,
}

impl Runner for EngineRunner {
    fn interrupted(&self) -> bool { false }
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let output = self.mock.run(exec)?;
        if exec.args == ["__worker"] {
            let request = text::parse(std::str::from_utf8(&exec.stdin).unwrap()).unwrap();
            if request["op"] == "start" {
                let log = Path::new(&request["state_dir"]).join("server.log");
                text::private_dir(log.parent().unwrap())?;
                let file = fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&log).unwrap();
                fs::write(self.root.join("log-path"), log.as_os_str().as_encoded_bytes()).unwrap();
                let mut command = Process::new(std::env::current_exe().unwrap());
                command.args(["--exact", self.test, "--nocapture", "--test-threads=1"])
                    .env(MODE, "engine").env(ROOT, &self.root).env(VALUE, "fixture value")
                    .current_dir(&self.root);
                let args = command.get_args().map(|v| v.to_owned()).collect::<Vec<_>>();
                let env = command.get_envs().map(|(k, v)| (k.to_owned(), v.map(|v| v.to_owned()))).collect::<Vec<_>>();
                crate::runner::detached_stdio(&mut command, file)?;
                assert_eq!(command.get_args().collect::<Vec<_>>(), args.iter().collect::<Vec<_>>());
                assert_eq!(command.get_envs().map(|(k, v)| (k.to_owned(), v.map(|v| v.to_owned()))).collect::<Vec<_>>(), env);
                let mut engine = command.spawn().unwrap();
                let deadline = Instant::now() + Duration::from_secs(5);
                while !self.root.join("engine-ready").exists() {
                    assert!(engine.try_wait().unwrap().is_none(), "fake engine exited before readiness");
                    assert!(Instant::now() < deadline, "fake engine did not start");
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
        Ok(output)
    }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { self.mock.interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { self.mock.port_free(port) }
    fn pause(&mut self, duration: Duration) { self.mock.pause(duration); }
}

fn child(test: &'static str, root: &Path) {
    match std::env::var(MODE).unwrap().as_str() {
        "up" => {
            // Keep the fixture for inspection. Its normal Drop deletes it.
            let fixture = std::mem::ManuallyDrop::new(Fixture::new());
            fixture.configured();
            fixture.installed();
            let mut launch = Launcher {
                root: fixture.root.clone(), home: fixture.home.clone(),
                runner: EngineRunner { mock: MockRunner::new(&fixture.root), root: root.into(), test },
                detector: MockDetector { identity: identity(), calls: 0 },
            };
            let parsed = crate::cli::parse(&["up".into()]).unwrap();
            launch.execute(parsed.command, false).unwrap();
        }
        "engine" => {
            assert_eq!(std::env::var(VALUE).unwrap(), "fixture value");
            let mut input = Vec::new();
            assert_eq!(std::io::stdin().read_to_end(&mut input).unwrap(), 0);
            let listener = UnixListener::bind("engine.sock").unwrap();
            std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream.write_all(b"running\n").unwrap();
            });
            writeln!(std::io::stdout(), "fake engine stdout before sleep").unwrap();
            writeln!(std::io::stderr(), "fake engine stderr before sleep").unwrap();
            fs::write(root.join("engine-ready"), "ready").unwrap();
            std::thread::sleep(Duration::from_secs(60));
            writeln!(std::io::stdout(), "fake engine stdout after sleep").unwrap();
            writeln!(std::io::stderr(), "fake engine stderr after sleep").unwrap();
            fs::write(root.join("engine-finished"), "finished").unwrap();
        }
        "tunnel" => {
            let mut input = Vec::new();
            assert_eq!(std::io::stdin().read_to_end(&mut input).unwrap(), 0);
            writeln!(std::io::stdout(), "fake tunnel stdout").unwrap();
            writeln!(std::io::stderr(), "fake tunnel stderr").unwrap();
        }
        _ => panic!("Unknown stdio test mode"),
    }
}

fn read_pipe(pipe: impl Read + Send + 'static, stream: usize, send: mpsc::Sender<(usize, Option<String>, Instant)>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        for line in BufReader::new(pipe).lines() {
            send.send((stream, Some(line.unwrap()), Instant::now())).unwrap();
        }
        send.send((stream, None, Instant::now())).unwrap();
    })
}

fn check(test: &'static str, logs: bool) {
    if let Some(root) = std::env::var_os(ROOT) {
        child(test, Path::new(&root));
        return;
    }
    let _guard = crate::runner::process_test_lock();
    // A short path also fits the macOS Unix socket limit.
    let root = fs::canonicalize("/tmp").unwrap().join(format!("vade-stdio-{}", text::random_hex(8).unwrap()));
    text::private_dir(&root).unwrap();
    let began = Instant::now();
    let mut up = Process::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads=1"])
        .env(MODE, "up").env(ROOT, &root).env_remove("VADE_MODELS")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    up.stdin.take().unwrap().write_all(b"input must not reach the engine\n").unwrap();
    let (send, receive) = mpsc::channel();
    let stdout = read_pipe(up.stdout.take().unwrap(), 0, send.clone());
    let stderr = read_pipe(up.stderr.take().unwrap(), 1, send);
    let mut ready = None;
    let mut eof = [None, None];
    let mut printed = String::new();
    let deadline = began + Duration::from_secs(5);
    while eof.iter().any(Option::is_none) {
        let Ok((stream, line, at)) = receive.recv_timeout(deadline.saturating_duration_since(Instant::now())) else { break };
        if let Some(line) = line {
            if line.starts_with("Up ready in ") { ready = Some(at); }
            printed.push_str(&line);
            printed.push('\n');
        } else { eof[stream] = Some(at); }
    }
    let pipe_closed = eof.iter().all(Option::is_some);
    let after_ready = ready.and_then(|ready| eof.iter().copied().collect::<Option<Vec<_>>>()
        .map(|times| times.into_iter().max().unwrap().saturating_duration_since(ready)));
    let running = if pipe_closed {
        let mut stream = UnixStream::connect(root.join("engine.sock")).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).unwrap();
        answer == "running\n" && !root.join("engine-finished").exists()
    } else { false };
    let status = loop {
        if let Some(status) = up.try_wait().unwrap() { break Some(status); }
        if Instant::now() >= deadline { break None; }
        std::thread::sleep(Duration::from_millis(10));
    };
    // The engine ends on its own. Keep all test files and send no signal.
    let end = Instant::now() + Duration::from_secs(65);
    while root.join("engine-ready").exists() && !root.join("engine-finished").exists() {
        assert!(Instant::now() < end, "fake engine did not finish its sleep");
        std::thread::sleep(Duration::from_millis(20));
    }
    if status.is_some() || pipe_closed {
        stdout.join().unwrap();
        stderr.join().unwrap();
    }
    assert!(status.is_some_and(|status| status.success()), "Up did not exit successfully within 5 s: {printed}");
    assert!(pipe_closed, "Up stdout or stderr kept the pipe open beyond 5 s: {printed}");
    assert!(after_ready.is_some_and(|elapsed| elapsed < Duration::from_secs(2)), "Up pipe did not close within 2 s of ready: {printed}");
    assert!(running, "fake engine was not alive after pipe EOF");
    assert!(!printed.contains("fake engine stdout") && !printed.contains("fake engine stderr"));
    if logs {
        let path = fs::read_to_string(root.join("log-path")).unwrap();
        assert_eq!(printed.matches(&format!("Engine log: {path}")).count(), 1);
        let log = fs::read_to_string(path).unwrap();
        for line in ["fake engine stdout before sleep", "fake engine stderr before sleep", "fake engine stdout after sleep", "fake engine stderr after sleep"] {
            assert!(log.contains(line), "missing engine log line: {line}");
        }
        let log = root.join("tunnel.log");
        let mut exec = Exec::new("test tunnel", &std::env::current_exe().unwrap().to_string_lossy(),
            ["--exact", test, "--nocapture", "--test-threads=1"].map(str::to_owned).to_vec());
        exec.env.insert(MODE.into(), "tunnel".into());
        exec.env.insert(ROOT.into(), root.to_string_lossy().into_owned());
        exec.detached_log = Some(log.clone());
        exec.timeout = Duration::from_secs(5);
        let capture = output::testing::Capture::default();
        let result = output::with_writer(capture.clone(), || crate::runner::SystemRunner { dry: false }.run(&exec)).unwrap();
        assert!(result.success && result.stdout.is_empty());
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        assert_eq!(printed.matches(&format!("test tunnel log: {}", log.display())).count(), 1);
        let log = fs::read_to_string(log).unwrap();
        assert!(log.contains("fake tunnel stdout") && log.contains("fake tunnel stderr"));
    }
    println!("stdio receipt: stdout EOF {} ms, stderr EOF {} ms, ready to EOF {} ms; engine alive after EOF; sleep 60 s",
        eof[0].unwrap().duration_since(began).as_millis(), eof[1].unwrap().duration_since(began).as_millis(), after_ready.unwrap().as_millis());
}

#[test]
fn up_pipe_closes_while_engine_runs() {
    check("launch::tests::stdio_tests::up_pipe_closes_while_engine_runs", false);
}

#[test]
fn up_engine_log_keeps_both_streams() {
    check("launch::tests::stdio_tests::up_engine_log_keeps_both_streams", true);
}
