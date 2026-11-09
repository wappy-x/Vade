//! A saved header value must never become additional HTTP headers on replay.
#[path = "support/cli.rs"]
#[allow(dead_code)]
mod cli;
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};
use vade_harness::json;

const KEYS: &[&str] = &[
    "content-type", "content-encoding", "accept", "x-session-id",
    "session-id", "session_id", "x-codex-session-id", "x-claude-code-session-id",
];
const BODY: &[u8] = br#"{ "seed":713, "messages":[], "extension":1.20e0 }"#;

struct Root(PathBuf);
impl Root {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!("vade-replay-headers-{}-{label}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Root {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}
fn save(root: &Path, name: &str, headers: &[(&str, &str)]) -> Vec<u8> {
    let dir = root.join(name);
    fs::create_dir_all(&dir).unwrap();
    let headers: Vec<_> = headers.iter().map(|(k,v)| format!("[{},{}]", json::quote(k), json::quote(v))).collect();
    let meta = format!(r#"{{"version":1,"complete":true,"method":"POST","path":"/v1/chat/completions","session":"fixture","headers":[{}]}}"#, headers.join(",")).into_bytes();
    fs::write(dir.join("meta.json"), &meta).unwrap();
    fs::write(dir.join("request.body"), BODY).unwrap();
    meta
}
fn command(root: &Path, listener: &TcpListener) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vade-gateway"));
    cmd.env_remove("VADE_API_KEY_FILE").env_remove("VADE_GATEWAY_UPSTREAM")
        .arg("replay").arg(root)
        .args(["--upstream", &format!("http://{}", listener.local_addr().unwrap())]);
    cmd
}

#[test]
fn replay_refuses_recorded_header_controls_before_any_output_or_dispatch() {
    let root = Root::new("invalid");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    // A valid earlier request must not escape the full inventory preflight.
    save(&root.0, "request-0001", &[("accept", "application/json")]);
    let mut cases: Vec<(&str, String)> = (0u8..=31).chain([127])
        .filter(|b| *b != b'\t')
        .map(|b| ("accept", format!("PRIVATE{}fixture", char::from(b))))
        .collect();
    cases.extend(KEYS.iter().map(|key| (*key, "PRIVATE\r\nAuthorization: Bearer synthetic-only".into())));
    for (key, value) in &cases {
        let saved = save(&root.0, "request-0002", &[(key, value)]);
        let stats = root.0.join("stats.jsonl");
        let output = cli::Cli::start(command(&root.0, &listener).arg("--stats").arg(&stats)).finish();
        assert!(!output.status.success(), "{key} accepted {value:?}");
        assert!(output.stdout.is_empty());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("recorded header contains control characters"), "{error}");
        assert!(error.contains("preserve the recording") && error.contains("unmodified complete recording"), "{error}");
        assert!(!error.contains("PRIVATE") && !error.contains("synthetic-only"), "{error}");
        assert!(!stats.exists());
        assert_eq!(listener.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
        assert_eq!(fs::read(root.0.join("request-0002/meta.json")).unwrap(), saved);
    }
    assert_eq!(cases.len(), 40);
}

#[test]
fn replay_preserves_valid_headers_and_body_without_restoring_unlisted_credentials() {
    let root = Root::new("valid");
    let mut headers: Vec<_> = KEYS.iter().map(|k| (*k, "fixture\tvalue-é")).collect();
    headers.push(("authorization", "Bearer never-replay\r\nX-Injected: discarded"));
    save(&root.0, "request-0001", &headers);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut cmd = command(&root.0, &listener);
    let worker = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(4);
        let socket = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => thread::sleep(Duration::from_millis(1)),
                Err(error) => panic!("replay did not connect: {error}"),
            }
        };
        socket.set_nonblocking(false).unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        socket.set_write_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut reader = BufReader::new(socket);
        let mut head = String::new();
        loop {
            let mut line = String::new();
            assert!(reader.read_line(&mut line).unwrap() > 0);
            head.push_str(&line);
            assert!(head.len() < 8192);
            if line == "\r\n" { break; }
        }
        let mut body = vec![0; BODY.len()]; reader.read_exact(&mut body).unwrap();
        reader.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        (head, body)
    });
    let output = cli::Cli::start(&mut cmd).finish();
    let (head, body) = worker.join().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.stderr.is_empty());
    assert!(head.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
    for key in KEYS { assert!(head.contains(&format!("{key}: fixture\tvalue-é\r\n")), "{head}"); }
    assert!(!head.to_ascii_lowercase().contains("authorization") && !head.contains("X-Injected"));
    assert_eq!(body, BODY);
}

#[cfg(unix)]
#[test]
fn replay_stops_queued_sessions_without_panicking_when_stdout_closes() {
    use std::os::{fd::OwnedFd, unix::net::UnixStream};
    use std::process::Stdio;
    for stats_enabled in [false, true] {
        let root = Root::new(if stats_enabled { "closed-stats" } else { "closed" });
        for index in 0..3 {
            let name = format!("request-{index:04}");
            let meta = save(&root.0, &name, &[("content-type", "application/json")]);
            // Independent sessions must remain queued after the first output failure.
            let meta = String::from_utf8(meta).unwrap().replace("\"session\":\"fixture\"", &format!("\"session\":\"{index}\""));
            fs::write(root.0.join(name).join("meta.json"), meta).unwrap();
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let peer = listener.try_clone().unwrap();
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            let socket = loop {
                match peer.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => thread::sleep(Duration::from_millis(1)),
                    Err(error) => panic!("missing first replay: {error}"),
                }
            };
            socket.set_nonblocking(false).unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            socket.set_write_timeout(Some(Duration::from_secs(3))).unwrap();
            let mut reader = BufReader::new(socket);
            loop {
                let mut line = String::new(); assert!(reader.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" { break; }
            }
            let mut body = vec![0; BODY.len()]; reader.read_exact(&mut body).unwrap();
            assert_eq!(body, BODY);
            reader.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        });
        let (writer, reader) = UnixStream::pair().unwrap(); drop(reader);
        let output_fd: OwnedFd = writer.into();
        let mut cmd = command(&root.0, &listener);
        let stats = root.0.join("stats.jsonl");
        if stats_enabled { cmd.arg("--stats").arg(&stats); }
        let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::from(output_fd)).stderr(Stdio::piped()).spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(4);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline { let _ = child.kill(); let _ = child.wait(); panic!("replay continued after stdout closed"); }
            thread::sleep(Duration::from_millis(1));
        }
        let output = child.wait_with_output().unwrap(); server.join().unwrap();
        assert!(!output.status.success(), "incomplete replay cannot report success");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(!error.contains("panicked"), "{error}");
        assert!(error.contains("cannot write replay statistics to standard output") && error.contains("Inspect any --stats file"), "{error}");
        assert_eq!(listener.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "a queued session was dispatched");
        if stats_enabled {
            let text = fs::read_to_string(stats).unwrap(); assert_eq!(text.lines().count(), 1);
            assert_eq!(json::parse(text.trim()).unwrap().get("complete"), Some(&json::Value::Bool(true)));
        }
    }
}

#[cfg(unix)]
#[test]
fn several_workers_stop_queued_sessions_when_the_statistics_file_cannot_be_written() {
    use std::{os::unix::process::CommandExt, process::Stdio};
    #[repr(C)]
    struct Limit { current: u64, maximum: u64 }
    unsafe extern "C" { fn setrlimit(resource: i32, limit: *const Limit) -> i32; fn signal(number: i32, handler: usize) -> usize; }
    for concurrency in [2, 4] {
        let root = Root::new(&format!("failed-stats-{concurrency}"));
        for index in 0..12 {
            let name = format!("request-{index:04}");
            let meta = save(&root.0, &name, &[("content-type", "application/json")]);
            let meta = String::from_utf8(meta).unwrap().replace("\"session\":\"fixture\"", &format!("\"session\":\"{index:04}\""));
            fs::write(root.0.join(name).join("meta.json"), meta).unwrap();
        }
        let stats = root.0.join("stats.jsonl");
        fs::write(&stats, b"previous statistics\n").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let peer = listener.try_clone().unwrap();
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut readers = Vec::new();
            // Hold every initial response until all workers have an in-flight request.
            for _ in 0..concurrency {
                let socket = loop {
                    match peer.accept() {
                        Ok((socket, _)) => break socket,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => thread::sleep(Duration::from_millis(1)),
                        Err(error) => panic!("missing initial replay worker: {error}"),
                    }
                };
                socket.set_nonblocking(false).unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                socket.set_write_timeout(Some(Duration::from_secs(3))).unwrap();
                let mut reader = BufReader::new(socket);
                loop {
                    let mut line = String::new(); assert!(reader.read_line(&mut line).unwrap() > 0);
                    if line == "\r\n" { break; }
                }
                let mut body = vec![0; BODY.len()]; reader.read_exact(&mut body).unwrap();
                assert_eq!(body, BODY);
                readers.push(reader);
            }
            for reader in &mut readers {
                reader.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
            }
        });
        let mut cmd = command(&root.0, &listener);
        cmd.args(["--concurrency", &concurrency.to_string(), "--stats"]).arg(&stats);
        // Real file-write failure, independent of stdout: EFBIG on macOS/Linux.
        unsafe { cmd.pre_exec(|| {
            signal(25, 1); // Ignore SIGXFSZ so write returns the error.
            if setrlimit(1, &Limit { current: 0, maximum: 0 }) != 0 { return Err(std::io::Error::last_os_error()); }
            Ok(())
        }); }
        let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(7);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline { let _ = child.kill(); let _ = child.wait(); panic!("replay continued after the statistics file failed"); }
            thread::sleep(Duration::from_millis(1));
        }
        let output = child.wait_with_output().unwrap(); server.join().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty(), "file failure precedes the stdout row");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("cannot write replay statistics file (--stats)") && error.contains("Replay stopped"), "{error}");
        assert!(!error.contains("see request stats") && !error.contains("panicked") && !error.contains("standard output"), "{error}");
        assert_eq!(fs::read(&stats).unwrap(), b"previous statistics\n");
        assert_eq!(listener.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "a session beyond the initial workers was dispatched");
    }
}
