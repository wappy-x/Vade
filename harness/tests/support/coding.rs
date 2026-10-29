//! Temporary workspaces and a scripted chat-completions mock shared by the
//! native coding-agent CLI tests. Local loopback only.
// Each test file uses a different subset of these helpers.
#![allow(dead_code, unused_imports)]
pub use std::io::{Read, Write};
pub use std::net::{TcpListener, TcpStream};
pub use std::process::Command;
pub use std::time::{Duration, Instant};
pub use vade_harness::json::{self, Value};

use super::cli;
pub struct Fixture(pub std::path::PathBuf);
impl Fixture {
    pub fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "vade-cli-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let fixture = Self(path);
        std::fs::create_dir(fixture.home()).unwrap();
        fixture
    }
    /// The HOME of every command: a developer's own ~/.vade/instructions.md
    /// never reaches a test prompt.
    pub fn home(&self) -> std::path::PathBuf {
        let mut name = self.0.as_os_str().to_owned();
        name.push("-home");
        name.into()
    }
    pub fn command(&self, url: &str) -> Command {
        let mut c = crate::home::harness();
        c.args([
            "agent",
            "--state-in-project",
            "--verbose",
            "--endpoint",
            url,
            "--model",
            "mock",
            "--workspace",
        ])
        .arg(&self.0)
        .env("NO_COLOR", "1")
        .env("HOME", self.home());
        c
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
        let _ = std::fs::remove_dir_all(self.home());
    }
}
pub fn request(stream: &mut TcpStream) -> Value {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut byte = [0];
    while !bytes.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
        assert!(bytes.len() < 64 * 1024);
    }
    let headers = std::str::from_utf8(&bytes).unwrap();
    if headers.starts_with("GET /v1/capabilities HTTP/1.1\r\n") {
        // Optional contract discovery is unsupported by these standard-chat
        // mocks. Keep the connection framed/reusable for the generation POST.
        stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n{}").unwrap();
        return request(stream);
    }
    let length: usize = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|n| n.trim().parse().unwrap())
        })
        .unwrap();
    assert!(length <= 4 * 1024 * 1024);
    let mut body = vec![0; length];
    stream.read_exact(&mut body).unwrap();
    let request = json::parse_bytes(&body).unwrap();
    for message in request.get("messages").unwrap().as_array().unwrap() {
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let arguments = call.get("function").unwrap().get("arguments").unwrap();
                json::parse(arguments.as_str().expect("arguments must be a JSON string"))
                    .expect("mock upstream must be able to render argument JSON");
            }
        }
    }
    request
}
pub fn response(stream: &mut TcpStream, body: &str) {
    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
}
pub fn answer() -> String {
    r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"mock complete"}}]}"#.into()
}
pub fn call(id: &str, name: &str, args: Value) -> Value {
    call_with_arguments(id, name, &json::stringify(&args).unwrap())
}
pub fn call_with_arguments(id: &str, name: &str, arguments: &str) -> Value {
    Value::Object(vec![
        ("id".into(), Value::String(id.into())),
        ("type".into(), Value::String("function".into())),
        (
            "function".into(),
            Value::Object(vec![
                ("name".into(), Value::String(name.into())),
                ("arguments".into(), Value::String(arguments.into())),
            ]),
        ),
    ])
}
pub fn tools(calls: Vec<Value>) -> String {
    json::stringify(&Value::Object(vec![(
        "choices".into(),
        Value::Array(vec![Value::Object(vec![
            ("finish_reason".into(), Value::String("tool_calls".into())),
            (
                "message".into(),
                Value::Object(vec![
                    ("role".into(), Value::String("assistant".into())),
                    ("tool_calls".into(), Value::Array(calls)),
                ]),
            ),
        ])]),
    )]))
    .unwrap()
}
pub fn patch() -> Value {
    Value::Object(vec![(
        "patch".into(),
        Value::String(
            "*** Begin Patch\n*** Add File: result.txt\n+exact result\n*** End Patch".into(),
        ),
    )])
}
pub fn mock(replies: Vec<String>) -> (String, std::thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    listener.set_nonblocking(true).unwrap();
    let handle = std::thread::spawn(move || {
        let mut requests = Vec::new();
        for reply in replies {
            let start = Instant::now();
            let mut stream = loop {
                match listener.accept() {
                    Ok((s, _)) => break s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            start.elapsed() < Duration::from_secs(8),
                            "mock request deadline"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("mock accept: {e}"),
                }
            };
            requests.push(request(&mut stream));
            response(&mut stream, &reply);
        }
        requests
    });
    (url, handle)
}
pub fn run_bounded(command: &mut Command, input: &str) -> std::process::Output {
    let mut child = cli::Cli::start(command);
    for line in input.lines() {
        child.send(&format!("{line}\n"));
        // Main's control lane intentionally consumes /quit and context commands
        // during work. Fixture scripts must wait for a turn receipt before
        // sending a command that depends on that turn's completed journal.
        if !line.is_empty() && !line.starts_with('/') {
            child.until(true, "Native run:");
        } else if line.starts_with("/compact ") || line == "/compact" {
            child.until(false, "Context compacted.");
        }
    }
    child.finish()
}
