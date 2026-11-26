//! Run the shipping binaries, with real sockets and no test-only policy override.
//! Cost: local processes and loopback sockets. Gain: mutations of the shipping
//! switch or refusal rule fail these tests, including process cleanup and output.
#[path = "support/home.rs"]
mod home;
use std::{fs, io::{Read, Write}, net::{TcpListener, TcpStream}, path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio}, sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}},
    thread, time::{Duration, Instant}};

const WARNING: &str = "Warning: --allow-plain-http is on. Links to other machines are not encrypted. Anyone on the network can read and change your chat, your tool results and your API key.";
const HELP: &str = "Allow plain HTTP to another machine. Not encrypted: others on the network can read and change the chat and the API key.";
const STRICT: &str = "VADE_HTTP_TREAT_PEERS_AS_REMOTE";
const HARNESS: &str = env!("CARGO_BIN_EXE_vade-harness");
const GATEWAY: &str = env!("CARGO_BIN_EXE_vade-gateway");
static NEXT: AtomicUsize = AtomicUsize::new(0);

fn refusal(host: &str, port: &str, _gateway: bool) -> String {
    let local = if port.parse::<u16>().is_ok_and(|number| number < 1024) { "8080" } else { port };
    format!("Vade sent nothing to {host}:{port}.\n{host} is another machine, and this link is plain HTTP. Plain HTTP is not encrypted. Anyone on the network between the two machines can read and change your chat, your tool results and your API key.\nDo one of these:\n  1. Use an SSH tunnel. It encrypts the link. `vade up` sets one up for `vade chat`. By hand: run  ssh -N -L {local}:127.0.0.1:{port} USER@{host}  in another terminal. Replace USER with your login name on that machine. Then start Vade again and use http://127.0.0.1:{local} as the address.\n  2. Accept the risk for this run: add --allow-plain-http")
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("vade-http-programs-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&path).unwrap();
        for sub in ["home", "state", "work", "tmp"] { fs::create_dir(path.join(sub)).unwrap(); }
        Self(path)
    }
    fn command(&self, binary: &str) -> Command {
        let mut cmd = home::command(binary);
        cmd.env_clear().env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .env("HOME", self.0.join("home")).env("VADE_HOME", self.0.join("state"))
            .env("TMPDIR", self.0.join("tmp")).env("NO_COLOR", "1").env("TERM", "dumb")
            .current_dir(self.0.join("work")).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd
    }
}
impl Drop for Temp { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

// A listener records every received byte, including requests with incomplete
// framing. Each accepted stream is read to EOF or a complete request.
struct Wire {
    port: String, bytes: Arc<Mutex<Vec<u8>>>, connections: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>, worker: Option<thread::JoinHandle<()>>,
}
impl Wire {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        // Preserve the user's port spelling throughout every diagnostic.
        let port = format!("0{}", listener.local_addr().unwrap().port());
        listener.set_nonblocking(true).unwrap();
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let connections = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (log, count, done) = (bytes.clone(), connections.clone(), stop.clone());
        let worker = thread::spawn(move || {
            let mut clients = Vec::new();
            loop {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        count.fetch_add(1, Ordering::Relaxed);
                        let log = log.clone();
                        clients.push(thread::spawn(move || {
                            // macOS can inherit nonblocking mode from the listener.
                            socket.set_nonblocking(false).unwrap();
                            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                            let mut request = Vec::new();
                            let mut buffer = [0; 8192];
                            loop {
                                match socket.read(&mut buffer) {
                                    Ok(0) | Err(_) => break,
                                    Ok(n) => { log.lock().unwrap().extend_from_slice(&buffer[..n]); request.extend_from_slice(&buffer[..n]); }
                                }
                                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                                    let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                                    let length = headers.lines().find_map(|line| line.strip_prefix("content-length:")).map_or(0, |s| s.trim().parse::<usize>().unwrap());
                                    if request.len() < end + 4 + length { continue; }
                                    let metadata = headers.starts_with("get ") && headers.contains("capabilities");
                                    let (status, body) = if metadata { ("404 Not Found", "{}".to_owned()) }
                                    else if headers.starts_with("post ") && request[end + 4..].windows(13).any(|w| w == b"\"stream\":true") {
                                        ("200 OK", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":100}}}\n\ndata: [DONE]\n\n".to_owned())
                                    } else { ("200 OK", "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"OK\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":1}}".to_owned()) };
                                    let kind = if body.starts_with("data:") { "text/event-stream" } else { "application/json" };
                                    let response = format!("HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                                    let _ = socket.write_all(response.as_bytes());
                                    // The reply has a length. Let the client close
                                    // after consuming it, avoiding a macOS race
                                    // between peer close and socket timeout setup.
                                    while let Ok(n @ 1..) = socket.read(&mut buffer) {
                                        log.lock().unwrap().extend_from_slice(&buffer[..n]);
                                    }
                                    break;
                                }
                            }
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if done.load(Ordering::Acquire) { break; }
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("listener: {error}"),
                }
            }
            for client in clients { client.join().unwrap(); }
        });
        Self { port, bytes, connections, stop, worker: Some(worker) }
    }
    fn url(&self) -> String { format!("http://127.0.0.1:{}", self.port) }
    fn finish(&mut self) -> Vec<u8> {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() { worker.join().unwrap(); }
        self.bytes.lock().unwrap().clone()
    }
}
impl Drop for Wire { fn drop(&mut self) { self.finish(); } }

struct Process { child: Child, out: Option<thread::JoinHandle<String>>, err: Option<thread::JoinHandle<String>> }
impl Process {
    fn start(command: &mut Command) -> Self {
        let mut child = command.spawn().unwrap();
        fn drain(mut pipe: impl Read + Send + 'static) -> thread::JoinHandle<String> {
            thread::spawn(move || { let mut bytes = Vec::new(); pipe.read_to_end(&mut bytes).unwrap(); String::from_utf8_lossy(&bytes).into_owned() })
        }
        let out = Some(drain(child.stdout.take().unwrap()));
        let err = Some(drain(child.stderr.take().unwrap()));
        Self { child, out, err }
    }
    fn finish(&mut self) -> (ExitStatus, String, String) {
        let deadline = Instant::now() + Duration::from_secs(30);
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() { break status; }
            if Instant::now() >= deadline {
                self.child.kill().unwrap(); self.child.wait().unwrap();
                panic!("program did not exit within 30 seconds");
            }
            thread::sleep(Duration::from_millis(5));
        };
        (status, self.out.take().unwrap().join().unwrap(), self.err.take().unwrap().join().unwrap())
    }
}
impl Drop for Process { fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); } }

// The normal MCP config starts this script. After stdin closes it deliberately
// stays alive; only Connection's normal destructor can stop and reap it.
#[cfg(unix)]
fn mcp(temp: &Temp) -> PathBuf {
    let script = temp.0.join("server.sh");
    let pid = temp.0.join("mcp.pid");
    fs::write(&script, r#"#!/bin/sh
printf '%s' "$$" > "$1"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"initialize"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"test","version":"1"}}}\n' "$id" ;;
    *'"tools/list"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[]}}\n' "$id" ;;
  esac
done
exec sleep 60
"#).unwrap();
    let config = format!("{{\"servers\":{{\"slow\":{{\"command\":\"/bin/sh\",\"args\":[{},{}]}}}}}}", vade_harness::json::quote(script.to_str().unwrap()), vade_harness::json::quote(pid.to_str().unwrap()));
    fs::write(temp.0.join("state/mcp.json"), config).unwrap();
    pid
}
#[cfg(unix)]
fn assert_mcp_stopped(pid_file: &Path) {
    let pid = fs::read_to_string(pid_file).expect("MCP was not started through its config");
    let running = Command::new("/bin/kill").args(["-0", &pid]).stdout(Stdio::null()).stderr(Stdio::null()).status().unwrap().success();
    if running {
        // Clean up only the stub created by this test, even when testing a mutant.
        let _ = Command::new("/bin/kill").args(["-KILL", &format!("-{pid}")]).status();
    }
    assert!(!running, "MCP stub {pid} survived the harness exit");
}

fn suite(temp: &Temp) -> PathBuf {
    let root = temp.0.join("suite");
    let task = root.join("tasks/wire");
    fs::create_dir_all(task.join("expected")).unwrap();
    fs::write(root.join("VERSION"), "1\n").unwrap();
    fs::write(task.join("task.txt"), "id: wire\ntitle: Wire test\ncategory: network\nmode: direct\nrequest: request.json\ncheck: file-equals reply.json expected/reply.txt\n").unwrap();
    fs::write(task.join("rubric.txt"), "may_change: reply.json\nmax_changed_lines: 20\nanswer_max_chars: 100\n").unwrap();
    fs::write(task.join("request.json"), "{\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}\n").unwrap();
    fs::write(task.join("expected/reply.txt"), "OK").unwrap();
    for args in [vec!["init", "-q"], vec!["add", "."]] {
        assert!(Command::new("git").current_dir(&root).args(args).status().unwrap().success());
    }
    root
}

fn harness_case(mode: &str, strict: bool, allow: bool) {
    let temp = Temp::new();
    let mut wire = Wire::new();
    let mut cmd = temp.command(HARNESS);
    let output = temp.0.join("output.json");
    let card = temp.0.join("card.txt");
    cmd.arg(if mode.starts_with("quality") { "quality-card" } else { mode });
    if strict { cmd.env(STRICT, "1"); }
    if mode == "doctor" { cmd.arg(format!("{}/v1", wire.url())).args(["--model", "test", "--max-seconds", "10"]); }
    else { cmd.args(["--endpoint", &wire.url(), "--model", "test"]); }
    match mode {
        "direct" | "agent" => { cmd.args(["--prompt", "Say OK", "--plain", "--no-instructions", "--no-memory"]).arg("--workspace").arg(temp.0.join("work")); }
        "workload" => { cmd.args(["--sessions", "32", "--main-sessions", "1", "--turns", "2", "--max-seconds", "10"]).arg("--json").arg(&output); }
        "quality-card" | "quality-exact" | "quality-engine" => {
            cmd.arg("--json").arg(&output).arg("--card").arg(&card);
            if mode == "quality-engine" { cmd.args(["--set", "engine-1"]); }
            else { cmd.args(["--only", if mode == "quality-exact" { "exactness" } else { "quality" }, "--no-code-run"]); }
        }
        "suite" => { cmd.arg("--live").arg("--out").arg(&output).arg("--suite").arg(suite(&temp)).arg("--toolchain").arg(Path::new(env!("CARGO")).parent().unwrap().parent().unwrap()); }
        "doctor" => {}
        _ => unreachable!(),
    }
    if allow { cmd.arg("--allow-plain-http"); }
    #[cfg(unix)]
    let stub = (mode == "agent").then(|| mcp(&temp));
    let (status, stdout, stderr) = Process::start(&mut cmd).finish();
    #[cfg(unix)]
    if let Some(pid) = stub { assert_mcp_stopped(&pid); }
    let repeated = if strict && !allow && matches!(mode, "workload" | "quality-card" | "quality-exact" | "quality-engine" | "suite") {
        assert!(!output.exists() && !card.exists(), "refusal left output: {stderr}");
        Some(Process::start(&mut cmd).finish())
    } else { None };
    let bytes = wire.finish();
    let detail = format!("{mode} strict={strict} allow={allow}: {status}\n{stdout}\n{stderr}\nWire: {}", String::from_utf8_lossy(&bytes));
    assert_eq!(stderr.matches(WARNING).count(), usize::from(allow), "{detail}");
    if strict && !allow {
        assert_eq!(status.code(), Some(1), "{detail}");
        assert!(wire.connections.load(Ordering::Relaxed) > 0, "no attempted connection: {detail}");
        let attempts = if mode == "workload" { 64 } else if repeated.is_some() { 2 } else { 1 };
        assert!(wire.connections.load(Ordering::Relaxed) <= attempts, "retried a refusal: {detail}");
        assert!(bytes.is_empty(), "sent {} bytes: {detail}", bytes.len());
        let expected = refusal("127.0.0.1", &wire.port, false);
        assert_eq!(stderr.matches(&expected).count(), 1, "{detail}");
        assert_eq!(stderr.matches("Vade sent nothing to ").count(), 1, "{detail}");
        assert!(!output.exists() && !card.exists(), "refusal left output: {detail}");
        assert!(!temp.0.join("tmp/vade-harness-suite-live").exists(), "suite left its work folder: {detail}");
        // Reuse the same path: it must reach the policy again, not AlreadyExists.
        if matches!(mode, "workload" | "quality-card" | "quality-exact" | "quality-engine" | "suite") {
            let (_, _, again) = repeated.unwrap();
            assert_eq!(again.matches(&expected).count(), 1, "second run: {again}");
            assert!(!output.exists() && !card.exists());
        }
    } else {
        assert!(!bytes.is_empty(), "request never arrived: {detail}");
        assert!(!stderr.contains("Vade sent nothing to "), "{detail}");
        if matches!(mode, "direct" | "agent" | "suite") { assert!(status.success(), "{detail}"); }
        if matches!(mode, "workload" | "quality-card" | "quality-exact" | "quality-engine" | "suite") { assert!(output.exists(), "missing output: {detail}"); }
        // A canned response need not pass doctor/quality/workload scoring.
    }
}

macro_rules! mode_test {
    ($name:ident, $mode:literal) => {
        #[test] fn $name() {
            harness_case($mode, true, false);
            harness_case($mode, true, true);
            harness_case($mode, false, false);
        }
    };
}
mode_test!(direct_wire_policy, "direct");
mode_test!(agent_wire_policy_and_mcp_cleanup, "agent");
mode_test!(doctor_wire_policy, "doctor");
mode_test!(workload_parallel_wire_policy_and_output, "workload");
mode_test!(quality_wire_policy_and_outputs, "quality-card");
mode_test!(quality_exact_wire_policy_and_outputs, "quality-exact");
mode_test!(quality_engine_wire_policy_and_outputs, "quality-engine");
mode_test!(suite_wire_policy_and_output, "suite");

#[test]
fn gateway_wire_policy() {
    for (strict, allow) in [(true, false), (true, true), (false, false)] {
        let temp = Temp::new();
        let mut wires = [Wire::new(), Wire::new()];
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let mut cmd = temp.command(GATEWAY);
        cmd.args(["--bind", &addr.to_string(), "--upstream", &wires[0].url(), "--upstream", &wires[1].url()]);
        if strict { cmd.env(STRICT, "1"); }
        if allow { cmd.arg("--allow-plain-http"); }
        let mut process = Process::start(&mut cmd);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if TcpStream::connect(addr).is_ok() { break; }
            assert!(Instant::now() < deadline, "gateway did not listen");
            assert!(process.child.try_wait().unwrap().is_none(), "gateway exited");
            thread::sleep(Duration::from_millis(5));
        }
        for _ in 0..6 {
            let mut stream = TcpStream::connect(addr).unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            write!(stream, "GET /v1/models HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").unwrap();
            let mut reply = String::new(); stream.read_to_string(&mut reply).unwrap();
            if strict && !allow {
                assert!(reply.starts_with("HTTP/1.1 502"), "{reply}");
                assert!(wires.iter().any(|wire| reply.contains(&format!("The gateway sent nothing to 127.0.0.1:{}. It is another machine and the link is not encrypted. Use an SSH tunnel or start the gateway with --allow-plain-http.", wire.port))), "{reply}");
            } else { assert!(reply.starts_with("HTTP/1.1 200"), "{reply}"); }
        }
        process.child.kill().unwrap();
        let (_, _, stderr) = process.finish();
        assert_eq!(stderr.matches(WARNING).count(), usize::from(allow), "{stderr}");
        for wire in &mut wires {
            let bytes = wire.finish();
            assert!(wire.connections.load(Ordering::Relaxed) > 0, "upstream not exercised");
            if strict && !allow {
                assert!(bytes.is_empty(), "gateway wrote {} bytes", bytes.len());
                assert_eq!(stderr.matches(&refusal("127.0.0.1", &wire.port, true)).count(), 1, "{stderr}");
            } else { assert!(!bytes.is_empty()); }
        }
        assert_eq!(stderr.matches("Vade sent nothing to ").count(), if strict && !allow { 2 } else { 0 }, "{stderr}");
    }
}

#[test]
fn help_words_and_warning_once() {
    let temp = Temp::new();
    for (binary, mode) in [(HARNESS, "direct"), (HARNESS, "agent"), (HARNESS, "doctor"), (HARNESS, "workload"), (HARNESS, "quality-card"), (HARNESS, "suite"), (GATEWAY, "")] {
        let mut cmd = temp.command(binary);
        if !mode.is_empty() { cmd.arg(mode); }
        cmd.args(["--allow-plain-http", "--allow-plain-http", "--help"]);
        let (status, stdout, stderr) = Process::start(&mut cmd).finish();
        assert!(status.success(), "{mode}: {stderr}");
        assert!(stdout.contains(HELP), "{mode}: {stdout}");
        assert_eq!(stderr.trim(), WARNING, "{mode}");
    }
}

#[test]
fn documentation_policy_is_consolidated() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let full = ["docs/network-setup.md", "harness/README.md", "harness/docs/invisible-gateway.md"];
    let files = Command::new("git").current_dir(root).args(["ls-files", "*.md"]).output().unwrap();
    assert!(files.status.success());
    let mut found = Vec::new();
    for name in String::from_utf8(files.stdout).unwrap().lines() {
        let text = fs::read_to_string(root.join(name)).unwrap();
        assert!(!text.contains("For current `vade-harness` and `vade-gateway` commands"), "old paragraph: {name}");
        if text.contains("HTTP links to another machine are refused by default") {
            found.push(name.to_owned());
            assert!(text.contains("your chat, your tool results and your API key"), "{name}");
            assert!(text.contains("A forwarder on this machine that is not SSH can pass the traffic on without\nencryption, and Vade cannot see that."), "{name}");
        }
        for line in text.lines().filter(|l| l.starts_with("HTTP links stay on this machine by default;")) {
            let target = line.split("[network setup](").nth(1).unwrap().split(')').next().unwrap();
            assert_eq!(fs::canonicalize(root.join(name).parent().unwrap().join(target)).unwrap(), fs::canonicalize(root.join(full[0])).unwrap());
        }
    }
    found.sort();
    assert_eq!(found, full);
}

#[test]
fn gateway_replay_wire_policy_and_output() {
    for (strict, allow) in [(true, false), (true, true), (false, false)] {
        let temp = Temp::new();
        let mut wire = Wire::new();
        let records = temp.0.join("records");
        for i in 0..16 {
            let dir = records.join(format!("request-{i:04}"));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("meta.json"), format!("{{\"version\":1,\"complete\":true,\"method\":\"POST\",\"path\":\"/v1/chat/completions\",\"session\":\"{i}\",\"headers\":[]}}")).unwrap();
            fs::write(dir.join("request.body"), "{\"messages\":[]}").unwrap();
        }
        let stats = temp.0.join("stats.jsonl");
        let mut cmd = temp.command(GATEWAY);
        cmd.arg("replay").arg(&records).args(["--upstream", &wire.url(), "--concurrency", "8"]).arg("--stats").arg(&stats);
        if strict { cmd.env(STRICT, "1"); }
        if allow { cmd.arg("--allow-plain-http"); }
        let (status, stdout, stderr) = Process::start(&mut cmd).finish();
        assert_eq!(stderr.matches(WARNING).count(), usize::from(allow), "{stderr}");
        if strict && !allow {
            assert!(!status.success());
            assert!(stdout.is_empty(), "{stdout}");
            assert!(!stats.exists());
            assert_eq!(stderr.matches(&refusal("127.0.0.1", &wire.port, true)).count(), 1, "{stderr}");
            assert!(wire.connections.load(Ordering::Relaxed) <= 8, "replay retried a refusal");
            // Refusal must never delete or append to an existing report.
            fs::write(&stats, "keep\n").unwrap();
            let (status, _, _) = Process::start(&mut cmd).finish();
            assert!(!status.success());
            assert_eq!(fs::read_to_string(&stats).unwrap(), "keep\n");
            assert!(wire.finish().is_empty());
        } else {
            assert!(status.success(), "{stderr}");
            assert_eq!(fs::read_to_string(stats).unwrap().lines().count(), 16);
            assert!(!wire.finish().is_empty());
        }
    }
}
