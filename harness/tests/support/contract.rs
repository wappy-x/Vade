//! Loopback fixtures shared by the contract, workload and CLI mock tests.
//! First-party only: no Engine binary, model, delegate or GPU.
// Each test file uses a different subset of these imports and helpers.
#![allow(dead_code, unused_imports)]
pub use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
pub use vade_harness::{
    agent::{
        self, Conversation, Limits, Model, StopReason, ToolError, ToolExecutor, ToolInvocation,
    },
    agent_wire::FinishReason,
    contract::{Capabilities, InputCount},
    json,
    protocol::Message,
    transport::{ChatClient, NativeModel},
};

pub const CAPS: &str = include_str!("../fixtures/engine-capabilities-v1.json");
pub const DONE: &str = "data: [DONE]\n\n";
pub const ANSWER: &str =
    "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n";

pub struct WorkspaceFixture(pub std::path::PathBuf);
impl WorkspaceFixture {
    pub fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "vade-contract-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for WorkspaceFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn fixture_read<T>(mut read: impl FnMut() -> io::Result<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match read() {
            Ok(value) => return value,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                )
                    && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => panic!("mock read failed while waiting for request: {error}"),
        }
    }
}
// Cancellation is a fresh control capability, outside the model prompt.
pub fn without_cancel(body: &str) -> String {
    let value = json::view(body).unwrap();
    if let Some(key) = value.get("cancel") {
        let key = key.as_str().unwrap();
        assert!(key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()));
    }
    format!("{{{}}}", value.fields().unwrap().filter(|(k, _)| k != "cancel")
        .map(|(k, v)| format!("{}:{}", json::quote(&k), v.raw())).collect::<Vec<_>>().join(","))
}
pub fn request(reader: &mut BufReader<TcpStream>) -> (String, String) {
    let mut first = String::new();
    // Approval/UI work may leave a keepalive socket idle. WouldBlock from
    // nonblocking I/O or a Darwin read timeout does not mean the peer closed.
    fixture_read(|| reader.read_line(&mut first));
    let mut length = None;
    loop {
        let mut line = String::new();
        assert!(fixture_read(|| reader.read_line(&mut line)) > 0);
        if line == "\r\n" {
            break;
        }
        if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = Some(n.trim().parse::<usize>().unwrap());
        }
    }
    let length = length.unwrap();
    let mut body = Vec::with_capacity(length);
    // Retain partial bytes and Take's remaining length across transient errors.
    let mut limited = reader.take(length as u64);
    fixture_read(|| limited.read_to_end(&mut body));
    assert_eq!(body.len(), length, "truncated mock request body");
    (first.trim_end().into(), String::from_utf8(body).unwrap())
}
pub fn respond(reader: &mut BufReader<TcpStream>, status: u16, kind: &str, body: &str) {
    write!(
        reader.get_mut(),
        "HTTP/1.1 {status} Mock\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
}
pub fn listen() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    (listener, endpoint)
}
pub fn accept(listener: &TcpListener) -> BufReader<TcpStream> {
    listener.set_nonblocking(true).unwrap();
    let start = Instant::now();
    loop {
        match listener.accept() {
            Ok((socket, _)) => {
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                return BufReader::new(socket);
            }
            Err(e)
                if e.kind() == io::ErrorKind::WouldBlock
                    && start.elapsed() < Duration::from_secs(3) =>
            {
                thread::sleep(Duration::from_millis(2))
            }
            other => panic!("mock accept failed: {other:?}"),
        }
    }
}
pub fn capabilities(reader: &mut BufReader<TcpStream>, caps: &str) {
    assert_eq!(
        request(reader),
        ("GET /v1/capabilities HTTP/1.1".into(), String::new())
    );
    respond(reader, 200, "application/json", caps);
}
pub fn admission_body(prompt: u64, output: u64, admitted: bool) -> String {
    format!(
        r#"{{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","prompt_tokens":{prompt},"output_tokens":{output},"context_limit":4096,"input_limit":4000,"output_limit":32,"admitted":{admitted},"reason":null,"admission_kind":"context_fit","reserves_resources":false}}"#
    )
}
pub fn admission(reader: &mut BufReader<TcpStream>, prompt: u64, admitted: bool) -> String {
    let (line, envelope) = request(reader);
    assert_eq!(line, "POST /v1/admission HTTP/1.1");
    let v = json::view(&envelope).unwrap();
    assert_eq!(
        v.get("endpoint").unwrap().as_str().unwrap(),
        "/v1/chat/completions"
    );
    assert_eq!(
        v.get("contract_version").unwrap().as_str().unwrap(),
        "vade.engine.v1"
    );
    let body = v.get("request").unwrap().raw().to_owned();
    let output: u64 = v.get("output_tokens").unwrap().raw().parse().unwrap();
    let actual = json::view(&body).unwrap();
    assert_eq!(
        actual
            .get("max_completion_tokens")
            .or_else(|| actual.get("max_tokens"))
            .unwrap()
            .raw(),
        output.to_string()
    );
    respond(
        reader,
        200,
        "application/json",
        &admission_body(prompt, output, admitted),
    );
    body
}

// Separate admission-only count of the retained completed transcript. The
// fixture verifies this request never substitutes for a generation admission.
pub fn rendered_history_admission(reader: &mut BufReader<TcpStream>, tokens: u64) -> String {
    let body = admission(reader, tokens, true);
    let v = json::view(&body).unwrap();
    let messages: Vec<_> = v.get("messages").unwrap().array().unwrap().collect();
    assert_eq!(
        messages
            .last()
            .unwrap()
            .get("role")
            .unwrap()
            .as_str()
            .as_deref(),
        Some("assistant")
    );
    assert!(v.get("vade_kv").is_none(), "counting must not retain KV");
    body
}

pub fn usage(prompt: u64, cached: u64) -> String {
    format!(
        "data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{prompt},\"completion_tokens\":3,\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{cached}}}}}}}\n\n",
        prompt + 3
    )
}
pub fn closed_without_replay(reader: &mut BufReader<TcpStream>, listener: &TcpListener) {
    // Closing without draining a rejected HTTP body can produce TCP RST on
    // macOS instead of FIN. Both prove closure; any generation byte is failure.
    match reader.read(&mut [0]) {
        Ok(0) => (),
        Err(error) if error.kind() == io::ErrorKind::ConnectionReset => (),
        other => panic!("no generation may follow refusal: {other:?}"),
    }
    listener.set_nonblocking(true).unwrap();
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
}

// Workload mode uses this same first-party contract mock: exact admissions,
// persistent sockets and real native HTTP/SSE assembly. No live server is used.
pub fn workload_config(
    endpoint: &str,
    sessions: usize,
    turns: usize,
) -> vade_harness::workload::Config {
    vade_harness::workload::Config {
        endpoint: endpoint.into(),
        model: "mock".into(),
        json_path: "unused-metrics.json".into(),
        sessions,
        main_sessions: 1,
        turns,
        main_system_bytes: 128,
        sub_system_bytes: 32,
        main_user_bytes: 64,
        sub_user_bytes: 16,
        main_tool_bytes: 64,
        sub_tool_bytes: 16,
        output_tokens: 17,
        max_seconds: 10,
        ..vade_harness::workload::Config::default()
    }
}
pub fn real(value: &json::Value) -> f64 {
    let json::Value::Number(n) = value else {
        panic!("not a number: {value:?}")
    };
    n.parse().unwrap()
}

/// A streamed native conversation with one tool and a non-ASCII prompt.
pub fn native_conversation() -> Conversation {
    Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"max_tokens":17,"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":" exact system\n"},{"role":"user","content":"hello 🦀"}]}"#).unwrap(), Limits::default()).unwrap()
}

/// Answers the `lookup` tool of [`native_conversation`] with fixed text.
pub struct MockTools;
impl ToolExecutor for MockTools {
    fn execute(&mut self, call: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        assert_eq!(call.call.name, "lookup");
        Ok(" exact result\n🦀 ".into())
    }
}
