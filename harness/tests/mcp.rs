//! MCP client tests against a tiny fake stdio server in this same binary.
//!
//! The test runs without the libtest harness (see `Cargo.toml`): libtest writes
//! to stdout, and an MCP server's stdout may hold only JSON-RPC messages. With
//! `FAKE_MCP` set, `main` is the fake server; without it, `main` runs the tests.
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use vade_harness::json::{self, Value};
use vade_harness::mcp::{Hub, config};
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[allow(dead_code)]
#[path = "support/coding.rs"]
mod coding;
#[path = "support/home.rs"]
mod home;

fn main() {
    if let Ok(mode) = std::env::var("FAKE_MCP") {
        return serve(&mode);
    }
    let tests: [(&str, fn()); 14] = [
        ("initialize_and_list", initialize_and_list),
        ("calls_progress_and_server_requests", calls_progress_and_server_requests),
        ("timeout_and_malformed_json", timeout_and_malformed_json),
        ("dead_server_restarts_once", dead_server_restarts_once),
        ("mcp_server_cwd_and_environment_stay_inside", mcp_server_cwd_and_environment_stay_inside),
        ("stalled_server_and_canceled_start", stalled_server_and_canceled_start),
        ("cancel_after_send_keeps_unknown_outcome", cancel_after_send_keeps_unknown_outcome),
        ("secrets_never_leave_quoted_stderr", secrets_never_leave_quoted_stderr),
        ("wrapper_children_stop_with_the_server", wrapper_children_stop_with_the_server),
        ("startup_failures_are_reported", startup_failures_are_reported),
        ("cli_manages_and_lists_servers", cli_manages_and_lists_servers),
        ("config_writes_keep_a_readable_file", config_writes_keep_a_readable_file),
        ("chat_calls_mcp_tools_and_resumes_web_on", || chat_calls_mcp_tools_and_resumes(true)),
        ("chat_calls_mcp_tools_and_resumes_web_off", || chat_calls_mcp_tools_and_resumes(false)),
    ];
    let filter: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with('-')).collect();
    let mut passed = 0;
    for (name, test) in tests {
        if !filter.is_empty() && !filter.iter().any(|f| name.contains(f.as_str())) {
            continue;
        }
        let started = Instant::now();
        test();
        println!("test {name} ... ok ({} ms)", started.elapsed().as_millis());
        passed += 1;
    }
    // The libtest summary line, so CI logs count these tests.
    println!("test result: ok. {passed} passed; 0 failed; 0 ignored; 0 measured; {} filtered out", tests.len() - passed);
}

// ---- The fake server ----

fn send(text: &str) {
    let mut out = std::io::stdout().lock();
    out.write_all(text.as_bytes()).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}
fn respond(id: &Value, result: &str) {
    send(&format!(r#"{{"jsonrpc":"2.0","id":{},"result":{result}}}"#, json::stringify(id).unwrap()));
}
fn text_result(id: &Value, text: &str) {
    respond(id, &format!(r#"{{"content":[{{"type":"text","text":{}}}]}}"#, json::quote(text)));
}

fn serve(mode: &str) {
    let log = std::env::var_os("FAKE_MCP_LOG").map(PathBuf::from);
    let note = |line: &str| {
        if let Some(log) = &log {
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(log).unwrap();
            // One write: a wrapper and its child append to the same log, and
            // writeln! writes the text and the newline separately.
            file.write_all(format!("{line}\n").as_bytes()).unwrap();
        }
    };
    if mode == "wrapper" || mode == "wrapper-exit" {
        // A wrapper command: the real (deaf) server is its child and holds the pipes.
        let mut child = home::command(&exe()).env("FAKE_MCP", "deaf").spawn().unwrap();
        note(&format!("child {}", child.id()));
        if mode == "wrapper" {
            let _ = child.wait();
        }
        return;
    }
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    while let Some(Ok(line)) = lines.next() {
        let message = json::parse(&line).expect("the client sends JSON");
        assert_eq!(message.get("jsonrpc").and_then(Value::as_str), Some("2.0"));
        let method = message.get("method").and_then(Value::as_str).unwrap_or("").to_owned();
        note(&method);
        let Some(id) = message.get("id").cloned() else { continue };
        let params = message.get("params");
        match method.as_str() {
            "initialize" => {
                let asked = params.and_then(|p| p.get("protocolVersion")).and_then(Value::as_str).unwrap();
                note(&format!("asked {asked}"));
                match mode {
                    "silent" => continue,
                    "garbage" => send(r#"{"jsonrpc":"2.0","id":1,"result":"#),
                    "badversion" => respond(&id, r#"{"protocolVersion":"1999-01-01","capabilities":{}}"#),
                    "exit" => {
                        eprintln!("fake: no config file");
                        std::process::exit(2);
                    }
                    _ => {
                        if mode == "banner" {
                            // Not UTF-8, and not a message: the client skips it.
                            let mut out = std::io::stdout().lock();
                            out.write_all(b"fake server \xff ready\n").unwrap();
                            out.flush().unwrap();
                        }
                        if mode == "initprogress" {
                            // Progress for a request that asked for none must not extend it.
                            for step in 1..=4 {
                                std::thread::sleep(Duration::from_millis(150));
                                send(&format!(r#"{{"jsonrpc":"2.0","method":"notifications/progress","params":{{"progressToken":"vade-1","progress":{step}}}}}"#));
                            }
                        }
                        respond(&id, r#"{"protocolVersion":"2025-06-18","capabilities":{"tools":{"listChanged":true},"resources":{},"prompts":{}},"serverInfo":{"name":"fake","version":"1.0"}}"#);
                    }
                }
            }
            "tools/list" => {
                let tools = [
                    r#"{"name":"echo","description":"Echo the arguments. Second sentence.","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}"#,
                    r#"{"name":"slow","inputSchema":{"type":"object","properties":{"ms":{"type":"integer"}}}}"#,
                    r#"{"name":"progress"}"#,
                    r#"{"name":"malformed"}"#,
                    r#"{"name":"die"}"#,
                    r#"{"name":"big"}"#,
                    r#"{"name":"fail"}"#,
                    r#"{"name":"ping"}"#,
                    r#"{"name":"dotted.name"}"#,
                ];
                // "grow": after a restart, larger schemas and no "fail" tool.
                let restarted = mode == "grow" && log.as_ref().and_then(|log| std::fs::read_to_string(log).ok()).is_some_and(|text| text.lines().filter(|line| *line == "initialize").count() > 1);
                let grown: Vec<String> = tools.iter().filter(|tool| !tool.contains(r#""fail""#)).map(|tool| {
                    let name = json::parse(tool).unwrap().get("name").and_then(Value::as_str).unwrap().to_owned();
                    format!(r#"{{"name":{},"inputSchema":{{"type":"object","description":"{}"}}}}"#, json::quote(&name), "x".repeat(16000))
                }).collect();
                let tools: Vec<&str> = if restarted { grown.iter().map(String::as_str).collect() } else { tools.to_vec() };
                // Two pages, to cover nextCursor.
                if params.and_then(|p| p.get("cursor")).is_none() {
                    respond(&id, &format!(r#"{{"tools":[{}],"nextCursor":"page-2"}}"#, tools[..4].join(",")));
                } else {
                    respond(&id, &format!(r#"{{"tools":[{}]}}"#, tools[4..].join(",")));
                }
            }
            "resources/list" => respond(&id, r#"{"resources":[{"uri":"file:///notes.txt","name":"notes"}]}"#),
            "prompts/list" => {
                respond(&id, r#"{"prompts":[{"name":"review"},{"name":"summarize"}]}"#);
                if mode == "deaf" {
                    // Started; from now on the server never reads its input.
                    loop {
                        std::thread::sleep(Duration::from_secs(60));
                    }
                }
            }
            "tools/call" => {
                let params = params.unwrap();
                let arguments = params.get("arguments").unwrap();
                let token = params.get("_meta").and_then(|m| m.get("progressToken")).cloned();
                match params.get("name").and_then(Value::as_str).unwrap() {
                    // {"env":NAME} answers whether the server can see NAME.
                    "echo" if let Some(name) = arguments.get("env").and_then(Value::as_str) => {
                        text_result(&id, &format!("{name}={}", std::env::var_os(name).is_some()))
                    }
                    "echo" => text_result(&id, &json::stringify(arguments).unwrap()),
                    "slow" => {
                        let ms = arguments.get("ms").and_then(|v| match v { Value::Number(n) => n.parse().ok(), _ => None }).unwrap_or(1000);
                        std::thread::sleep(Duration::from_millis(ms));
                        note("slow done");
                        text_result(&id, "slow done");
                    }
                    "progress" => {
                        for step in 1..=4 {
                            std::thread::sleep(Duration::from_millis(150));
                            send(&format!(r#"{{"jsonrpc":"2.0","method":"notifications/progress","params":{{"progressToken":{},"progress":{step},"total":4,"message":"step {step}"}}}}"#, json::stringify(token.as_ref().unwrap()).unwrap()));
                        }
                        // A progress note for another token does not extend the call.
                        send(r#"{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","data":"log line"}}"#);
                        text_result(&id, "progress done");
                    }
                    "malformed" => {
                        send("{this is not json");
                        text_result(&id, "late");
                    }
                    "die" => {
                        if let Some(Value::Array(parts)) = arguments.get("say") {
                            // Write in parts, so the client reads them one at a time.
                            for part in parts {
                                eprint!("{}", part.as_str().unwrap());
                                std::io::stderr().flush().unwrap();
                                std::thread::sleep(Duration::from_millis(100));
                            }
                            std::process::exit(3);
                        }
                        let secret = std::env::var("FAKE_SECRET").unwrap_or_default();
                        match arguments.get("part").and_then(|v| match v { Value::Number(n) => n.parse::<usize>().ok(), _ => None }) {
                            // Stop in the middle of the value.
                            Some(part) => eprint!("fake: dying on purpose; token {}", &secret[..part]),
                            None => eprintln!("fake: dying on purpose; token {secret}"),
                        }
                        std::process::exit(3);
                    }
                    "big" if arguments.get("over").is_none() => text_result(&id, &"x".repeat(100_000)),
                    "big" => {
                        // One stdout message over the size limit; the server stays up.
                        send(&format!(r#"{{"jsonrpc":"2.0","id":{},"result":"{}"}}"#, json::stringify(&id).unwrap(), "x".repeat(vade_harness::mcp::client::MAX_MESSAGE_BYTES)));
                    }
                    "fail" => respond(&id, r#"{"content":[{"type":"text","text":"it failed"}],"isError":true}"#),
                    "ping" => {
                        send(r#"{"jsonrpc":"2.0","id":"srv-1","method":"ping"}"#);
                        send(r#"{"jsonrpc":"2.0","id":"srv-2","method":"sampling/createMessage","params":{}}"#);
                        let mut answers = Vec::new();
                        for _ in 0..2 {
                            let reply = json::parse(&lines.next().unwrap().unwrap()).unwrap();
                            answers.push(json::stringify(&reply).unwrap());
                        }
                        let ok = answers[0] == r#"{"jsonrpc":"2.0","id":"srv-1","result":{}}"#
                            && answers[1].contains(r#""id":"srv-2","error":{"code":-32601"#);
                        text_result(&id, if ok { "pong ok" } else { "pong bad" });
                    }
                    _ => send(&format!(r#"{{"jsonrpc":"2.0","id":{},"error":{{"code":-32602,"message":"Unknown tool"}}}}"#, json::stringify(&id).unwrap())),
                }
            }
            _ => send(&format!(r#"{{"jsonrpc":"2.0","id":{},"error":{{"code":-32601,"message":"Method not found"}}}}"#, json::stringify(&id).unwrap())),
        }
    }
}

// ---- Helpers ----

struct Temp(PathBuf);
impl Temp {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("vade-mcp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path.canonicalize().unwrap())
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn exe() -> String {
    std::env::current_exe().unwrap().to_string_lossy().into_owned()
}
fn spec(mode: &str, timeout_ms: u64, log: Option<&Path>) -> config::ServerSpec {
    let mut env = vec![format!(r#""FAKE_MCP":"{mode}""#)];
    env.extend(log.map(|log| format!(r#""FAKE_MCP_LOG":{}"#, json::quote(&log.to_string_lossy()))));
    let entry = json::parse(&format!(r#"{{"command":{},"env":{{{}}},"timeout_ms":{timeout_ms}}}"#, json::quote(&exe()), env.join(","))).unwrap();
    config::parse_entry("fake", &entry, config::Source::User).unwrap()
}
fn secret_spec(log: &Path) -> config::ServerSpec {
    let mut spec = spec("normal", 2000, Some(log));
    // A shorter value inside the secret comes first; the longer one must still be hidden whole.
    spec.env.push(("FAKE_SHORT".into(), "hunt".into()));
    spec.env.push(("FAKE_SECRET".into(), "hunter2-secret".into()));
    spec
}
fn call(hub: &mut Hub, tool: &str, arguments: &str) -> (String, bool) {
    let (result, uncertain) = hub.call(tool, &json::parse(arguments).unwrap(), 64 * 1024, &AtomicBool::new(false));
    (result.unwrap(), uncertain)
}
fn log_lines(log: &Path) -> Vec<String> {
    std::fs::read_to_string(log).unwrap_or_default().lines().map(str::to_owned).collect()
}
/// Wait until `line` is in the log `count` times; the server runs in its own process.
fn wait_for_line(log: &Path, line: &str, count: usize) {
    let end = Instant::now() + Duration::from_secs(10);
    while log_lines(log).iter().filter(|l| *l == line).count() < count {
        assert!(Instant::now() < end, "{line} x{count} never reached the log: {:?}", log_lines(log));
        std::thread::sleep(Duration::from_millis(10));
    }
}
/// Wait for a whole line that starts with `prefix`, and return the rest of it.
fn wait_for_value(log: &Path, prefix: &str) -> String {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        let text = std::fs::read_to_string(log).unwrap_or_default();
        if let Some(value) = text.split_inclusive('\n').find_map(|line| line.strip_prefix(prefix)?.strip_suffix('\n')) {
            return value.to_owned();
        }
        assert!(Instant::now() < end, "no {prefix:?} line reached the log: {text:?}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

// ---- Tests ----

fn initialize_and_list() {
    let temp = Temp::new("init");
    let log = temp.0.join("log");
    let hub = Hub::start(vec![spec("normal", 5000, Some(&log))], &temp.0);
    assert!(hub.notes.is_empty(), "{:?}", hub.notes);
    let server = &hub.servers[0];
    let info = server.info().expect("connected");
    assert_eq!((info.name.as_str(), info.version.as_str(), info.protocol.as_str()), ("fake", "1.0", "2025-06-18"));
    assert!(info.tools && info.resources && info.prompts);
    let names: Vec<_> = server.tools.iter().map(|t| t.exposed.as_str()).collect();
    assert_eq!(names, ["fake__echo", "fake__slow", "fake__progress", "fake__malformed", "fake__die", "fake__big", "fake__fail", "fake__ping", "fake__dotted_name"]);
    assert_eq!(server.tools[8].name, "dotted.name");
    assert_eq!((server.resources.len(), server.prompts.len()), (1, 2));
    let definitions = hub.definitions();
    assert_eq!(definitions.len(), 9);
    assert_eq!(definitions[0].get("function").and_then(|f| f.get("parameters")).and_then(|p| p.get("properties")).and_then(|p| p.get("text")).and_then(|t| t.get("type")).and_then(Value::as_str), Some("string"));
    drop(hub);
    assert_eq!(log_lines(&log), ["initialize", "asked 2025-06-18", "notifications/initialized", "tools/list", "tools/list", "resources/list", "prompts/list"]);
}

fn calls_progress_and_server_requests() {
    let temp = Temp::new("calls");
    let mut hub = Hub::start(vec![spec("banner", 300, None)], &temp.0);
    assert!(hub.notes.is_empty(), "{:?}", hub.notes);
    assert_eq!(call(&mut hub, "fake__echo", r#"{"text":"hi \"there\""}"#), (r#"{"text":"hi \"there\""}"#.to_owned(), false));
    // Four progress notes 150 ms apart: 600 ms in total with a 300 ms timeout.
    assert_eq!(call(&mut hub, "fake__progress", "{}"), ("progress done".to_owned(), false));
    assert_eq!(call(&mut hub, "fake__ping", "{}").0, "pong ok");
    assert_eq!(call(&mut hub, "fake__dotted_name", "{}").0, call_error_text());
    let (failed, _) = call(&mut hub, "fake__fail", "{}");
    assert_eq!(json::parse(&failed).unwrap().get("error").and_then(Value::as_str), Some("mcp_tool_error"));
    let (big, _) = call(&mut hub, "fake__big", "{}");
    assert!(big.len() <= vade_harness::mcp::MAX_RESULT_BYTES && big.ends_with("of 100000 bytes]"), "{}", big.len());
    let small = hub.call("fake__big", &json::parse("{}").unwrap(), 1000, &AtomicBool::new(false)).0.unwrap();
    assert!(small.len() <= 1000 && small.starts_with("xxx"));
    let (missing, _) = call(&mut hub, "fake__nothing", "{}");
    assert!(missing.contains("does not offer fake__nothing"), "{missing}");
    let (other, _) = call(&mut hub, "other__echo", "{}");
    assert!(other.contains("not configured"), "{other}");
    // A call canceled before it is sent returns at once; its outcome is known.
    let cancel = AtomicBool::new(true);
    assert_eq!(hub.call("fake__echo", &json::parse("{}").unwrap(), 4096, &cancel), (Err(vade_harness::agent::ToolError::Cancelled), false));
}
fn call_error_text() -> &'static str {
    r#"{"ok":false,"error":"mcp_call_failed","message":"MCP tool dotted.name: the server returned error -32602: Unknown tool"}"#
}

fn timeout_and_malformed_json() {
    let temp = Temp::new("timeout");
    let log = temp.0.join("log");
    let mut hub = Hub::start(vec![spec("normal", 300, Some(&log))], &temp.0);
    let started = Instant::now();
    let (text, uncertain) = call(&mut hub, "fake__slow", r#"{"ms":1000}"#);
    let waited = started.elapsed();
    assert!(uncertain && text.contains("mcp_timeout") && text.contains("300 ms"), "{text}");
    // The call ends at its timeout, well before the tool does.
    assert!(waited >= Duration::from_millis(300) && waited < Duration::from_millis(1500), "{waited:?}");
    // Vade stops the server at the time limit, so the tool never finishes;
    // each later call starts it again, and a stop at a time limit uses no
    // restart of the one a crashed server gets.
    std::thread::sleep(Duration::from_millis(1200));
    assert!(!log_lines(&log).contains(&"slow done".to_owned()), "{:?}", log_lines(&log));
    assert_eq!(call(&mut hub, "fake__echo", r#"{"after":"slow"}"#).0, r#"{"after":"slow"}"#);
    assert!(call(&mut hub, "fake__slow", r#"{"ms":1000}"#).0.contains("mcp_timeout"));
    assert_eq!(call(&mut hub, "fake__echo", r#"{"after":"second"}"#).0, r#"{"after":"second"}"#);
    assert_eq!(log_lines(&log).iter().filter(|line| *line == "initialize").count(), 3);
    let (text, uncertain) = call(&mut hub, "fake__malformed", "{}");
    assert!(uncertain && text.contains("mcp_server_failed") && text.contains("malformed JSON-RPC"), "{text}");
    // The connection survives one bad line.
    assert_eq!(call(&mut hub, "fake__echo", r#"{"n":1}"#).0, r#"{"n":1}"#);
}

fn mcp_server_cwd_and_environment_stay_inside() {
    let temp = Temp::new("inside");
    std::fs::create_dir(temp.0.join("sub")).unwrap();
    let parse = |cwd: &str| config::parse_entry("fake", &json::parse(&format!(r#"{{"command":"x","cwd":{}}}"#, json::quote(cwd))).unwrap(), config::Source::User);
    for outside in ["../outside", "sub/../../outside", "/tmp"] {
        assert!(parse(outside).unwrap_err().contains("inside the workspace"), "{outside}");
    }
    assert_eq!(parse("./sub").unwrap().cwd.as_deref(), Some("./sub"));
    // A link inside the workspace that leads out is refused at start.
    std::os::unix::fs::symlink(std::env::temp_dir(), temp.0.join("out")).unwrap();
    let mut linked = spec("normal", 2000, None);
    linked.cwd = Some("out".into());
    let hub = Hub::start(vec![linked], &temp.0);
    assert!(hub.notes.iter().any(|note| note.contains("cwd is not a folder inside the workspace")), "{:?}", hub.notes);
    // The server sees ordinary names and its own env entries, not the rest.
    let mut inside = spec("normal", 2000, None);
    inside.cwd = Some("sub".into());
    let mut hub = Hub::start(vec![inside], &temp.0);
    assert!(hub.notes.is_empty(), "{:?}", hub.notes);
    assert_eq!(call(&mut hub, "fake__echo", r#"{"env":"PATH"}"#).0, "PATH=true");
    assert_eq!(call(&mut hub, "fake__echo", r#"{"env":"FAKE_MCP"}"#).0, "FAKE_MCP=true");
    assert!(std::env::var_os("CARGO_PKG_NAME").is_some());
    assert_eq!(call(&mut hub, "fake__echo", r#"{"env":"CARGO_PKG_NAME"}"#).0, "CARGO_PKG_NAME=false");
}

fn dead_server_restarts_once() {
    let temp = Temp::new("restart");
    // A restart keeps the definitions the chat started with, even when the
    // server now lists larger ones, and drops a tool it no longer lists.
    let log = temp.0.join("log-grow");
    let mut hub = Hub::start(vec![spec("grow", 2000, Some(&log))], &temp.0);
    let before = hub.definitions();
    assert!(call(&mut hub, "fake__die", "{}").1);
    assert_eq!(call(&mut hub, "fake__echo", r#"{"n":1}"#).0, r#"{"n":1}"#);
    assert_eq!(log_lines(&log).iter().filter(|line| *line == "initialize").count(), 2);
    let after = hub.definitions();
    let kept: Vec<Value> = before.iter().filter(|d| !json::stringify(d).unwrap().contains("fake__fail")).cloned().collect();
    assert_eq!((after.len(), after), (before.len() - 1, kept));
    assert!(call(&mut hub, "fake__fail", "{}").0.contains("does not offer fake__fail"));

    let log = temp.0.join("log");
    let mut hub = Hub::start(vec![secret_spec(&log)], &temp.0);
    let (text, uncertain) = call(&mut hub, "fake__die", "{}");
    assert!(uncertain && text.contains("mcp_server_failed") && text.contains("dying on purpose"), "{text}");
    // A message over the size limit stops the connection; ending it does not wait.
    let mut over = Hub::start(vec![spec("normal", 2000, None)], &temp.0);
    let (failed, unknown) = call(&mut over, "fake__big", r#"{"over":true}"#);
    assert!(unknown && failed.contains("larger than"), "{failed}");
    let started = Instant::now();
    drop(over);
    // The reader ended with the oversized message: nothing to drain, so no 500 ms grace.
    assert!(started.elapsed() < Duration::from_millis(300), "{:?}", started.elapsed());
    // A configured env value never reaches the model, even when the server prints it.
    assert!(text.contains("token [env value]") && !text.contains("hunter2-secret") && !text.contains("er2-secret"), "{text}");
    assert_eq!(call(&mut hub, "fake__echo", r#"{"n":2}"#).0, r#"{"n":2}"#);
    assert_eq!(log_lines(&log).iter().filter(|line| *line == "initialize").count(), 2);
    call(&mut hub, "fake__die", "{}");
    let (text, uncertain) = call(&mut hub, "fake__echo", "{}");
    assert!(!uncertain && text.contains("mcp_unavailable") && text.contains("after one restart"), "{text}");
    assert_eq!(log_lines(&log).iter().filter(|line| *line == "initialize").count(), 2);
}

fn stalled_server_and_canceled_start() {
    let temp = Temp::new("stalled");
    let log = temp.0.join("log");
    let mut hub = Hub::start(vec![spec("deaf", 1000, Some(&log))], &temp.0);
    assert!(hub.notes.is_empty(), "{:?}", hub.notes);
    // The arguments are larger than a pipe holds, and the server never reads them.
    let large = format!(r#"{{"text":"{}"}}"#, "a".repeat(256 * 1024));
    let started = Instant::now();
    let (text, uncertain) = call(&mut hub, "fake__echo", &large);
    let waited = started.elapsed();
    assert!(uncertain && text.contains("mcp_timeout"), "{text}");
    assert!(waited >= Duration::from_millis(1000) && waited < Duration::from_secs(3), "{waited:?}");
    // Ctrl-C ends a call whose request the server never reads. The server
    // has started fully, so the cancel cannot land in a restart, and the
    // cancel waits until the client queued the request for the writer.
    let mut hub = Hub::start(vec![spec("deaf", 60_000, Some(&log))], &temp.0);
    let queued = hub.servers[0].connection.as_ref().unwrap().queued();
    let before = queued.load(std::sync::atomic::Ordering::Relaxed);
    let cancel = AtomicBool::new(false);
    let (result, canceled) = std::thread::scope(|scope| {
        let canceler = scope.spawn(|| {
            let end = Instant::now() + Duration::from_secs(10);
            while queued.load(std::sync::atomic::Ordering::Relaxed) == before && Instant::now() < end {
                std::thread::sleep(Duration::from_millis(5));
            }
            cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            Instant::now()
        });
        let result = hub.call("fake__echo", &json::parse(&large).unwrap(), 4096, &cancel);
        (result, canceler.join().unwrap())
    });
    // Sent, so the outcome is unknown.
    assert_eq!(result, (Err(vade_harness::agent::ToolError::Cancelled), true));
    // A watchdog only: the 60 s timeout never ends this call.
    assert!(canceled.elapsed() < Duration::from_secs(10), "{:?}", canceled.elapsed());

    // Ctrl-C during a restart stops initialize and tells the server.
    let mut hub = Hub::start(vec![spec("normal", 2000, None)], &temp.0);
    assert!(call(&mut hub, "fake__die", "{}").1);
    let log = temp.0.join("log-silent");
    hub.servers[0].spec = spec("silent", 60_000, Some(&log));
    let cancel = AtomicBool::new(false);
    let (result, canceled) = std::thread::scope(|scope| {
        let canceler = scope.spawn(|| {
            // Cancel once the restarted server has the initialize request.
            wait_for_line(&log, "initialize", 1);
            cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            Instant::now()
        });
        let result = hub.call("fake__echo", &json::parse("{}").unwrap(), 4096, &cancel);
        (result, canceler.join().unwrap())
    });
    // The tool call was never sent, so its outcome is known.
    assert_eq!(result, (Err(vade_harness::agent::ToolError::Cancelled), false));
    assert!(canceled.elapsed() < Duration::from_secs(10), "{:?}", canceled.elapsed());
    drop(hub);
    wait_for_line(&log, "notifications/cancelled", 1);
    assert_eq!(log_lines(&log), ["initialize", "asked 2025-06-18", "notifications/cancelled"]);
}

fn startup_failures_are_reported() {
    let temp = Temp::new("startup");
    let mut missing = spec("normal", 300, None);
    missing.command = temp.0.join("no-such-server").to_string_lossy().into_owned();
    let cases = [
        (spec("silent", 300, None), "did not answer in time"),
        (spec("garbage", 300, None), "malformed JSON-RPC"),
        (spec("badversion", 300, None), "speaks MCP \"1999-01-01\""),
        (spec("exit", 300, None), "no config file"),
        (spec("initprogress", 300, None), "did not answer in time"),
        (missing, "cannot start the server"),
    ];
    for (spec, reason) in cases {
        let started = Instant::now();
        let hub = Hub::start(vec![spec], &temp.0);
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(hub.servers[0].connection.is_none() && hub.definitions().is_empty());
        assert!(hub.notes.len() == 1 && hub.notes[0].contains("did not start") && hub.notes[0].contains(reason), "{:?}", hub.notes);
    }
    // Several servers start at once; one failure does not stop the others.
    let mut second = spec("normal", 2000, None);
    second.name = "second".into();
    let hub = Hub::start(vec![spec("silent", 300, None), second], &temp.0);
    assert_eq!(hub.definitions().len(), 9);
    assert!(hub.servers[1].tools[0].exposed == "second__echo");
    // A server that failed at the start has no tool in this chat and is not
    // started again for a call.
    let mut hub = Hub::start(vec![spec("exit", 300, None)], &temp.0);
    let log = temp.0.join("log-late");
    hub.servers[0].spec = spec("normal", 2000, Some(&log));
    let (text, uncertain) = call(&mut hub, "fake__echo", "{}");
    assert!(!uncertain && text.contains("did not start when this chat began"), "{text}");
    assert!(log_lines(&log).is_empty(), "{:?}", log_lines(&log));
}

/// Quoted stderr hides a configured value that a cut or the end of the
/// output splits.
fn secrets_never_leave_quoted_stderr() {
    let temp = Temp::new("secrets");
    let with_secret = |secret: &str| {
        let mut spec = spec("normal", 2000, None);
        spec.env.push(("FAKE_SECRET".into(), secret.into()));
        Hub::start(vec![spec], &temp.0)
    };
    // A multiline value longer than the kept tail: every cut lands inside it.
    let long = (0..300).map(|i| format!("key line {i}")).collect::<Vec<_>>().join("\n");
    let (text, uncertain) = call(&mut with_secret(&long), "fake__die", "{}");
    assert!(uncertain && text.contains("[env value]") && !text.contains("key line"), "{text}");
    // A value that starts inside another match, written whole and in parts.
    for say in [r#"["token prefix-hunter2-secret end"]"#, r#"["token prefix-hunt", "er2-secret end"]"#] {
        let mut spec = spec("normal", 2000, None);
        spec.env.push(("FAKE_PREFIX".into(), "prefix-hunt".into()));
        spec.env.push(("FAKE_TOKEN".into(), "hunter2-secret".into()));
        let (text, _) = call(&mut Hub::start(vec![spec], &temp.0), "fake__die", &format!(r#"{{"say":{say}}}"#));
        assert!(text.contains("token [env value] end") && !text.contains("er2") && !text.contains("hunt"), "{text}");
    }
    // Output that ends inside a value shows none of it.
    let (text, _) = call(&mut with_secret("partial-secret-value"), "fake__die", r#"{"part":10}"#);
    assert!(text.contains("token") && !text.contains("partial"), "{text}");
}

/// A wrapper command's child stops with the server when Vade stops it, also
/// when the wrapper exited first.
fn wrapper_children_stop_with_the_server() {
    let temp = Temp::new("wrapper");
    for (mode, blocked) in [("wrapper", true), ("wrapper-exit", true), ("wrapper-exit", false)] {
        let log = temp.0.join(format!("log-{mode}-{blocked}"));
        // A start on a busy host can take seconds, so only the call below has the short limit.
        let mut hub = Hub::start(vec![spec(mode, 60_000, Some(&log))], &temp.0);
        assert!(hub.notes.is_empty(), "{mode}: {:?}", hub.notes);
        // The wrapper writes this line after the spawn, so its child can finish the whole start first.
        let child = wait_for_value(&log, "child ");
        let alive = || std::process::Command::new("/bin/kill").args(["-0", &child]).stderr(std::process::Stdio::null()).status().unwrap().success();
        assert!(alive(), "{mode}: {:?}", log_lines(&log));
        if blocked {
            // The child never reads this request; the timeout stops the server.
            hub.servers[0].spec.timeout = Duration::from_millis(500);
            let large = format!(r#"{{"text":"{}"}}"#, "a".repeat(256 * 1024));
            let (text, uncertain) = call(&mut hub, "fake__echo", &large);
            assert!(uncertain && text.contains("mcp_timeout"), "{mode}: {text}");
        }
        // Chat ends: the child still holds the pipes and ignores closed input.
        drop(hub);
        let end = Instant::now() + Duration::from_secs(10);
        while alive() {
            assert!(Instant::now() < end, "{mode}: the wrapper's child {child} still runs");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// A cancel after the request reached the server keeps the outcome unknown,
/// and the chat journal saves the unknown-outcome result.
fn cancel_after_send_keeps_unknown_outcome() {
    use vade_harness::agent::{self, Limits, Model, ModelRequest, ToolError, ToolExecutor, ToolInvocation};
    use vade_harness::agent_wire::AssistantTurn;
    use vade_harness::coding_tools::Preview;
    use vade_harness::mcp::McpTools;
    use vade_harness::shell::ApprovalPolicy;
    let temp = Temp::new("cancel-sent");
    let log = temp.0.join("log");
    let cancel_when_called = |cancel: &AtomicBool| {
        wait_for_line(&log, "tools/call", 1);
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    };

    let mut hub = Hub::start(vec![spec("normal", 60_000, Some(&log))], &temp.0);
    let cancel = AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        scope.spawn(|| cancel_when_called(&cancel));
        hub.call("fake__slow", &json::parse(r#"{"ms":1500}"#).unwrap(), 4096, &cancel)
    });
    assert_eq!(result, (Err(ToolError::Cancelled), true));
    // The cancel notice does not stop the tool; its work still ends.
    wait_for_line(&log, "slow done", 1);

    struct CallSlow;
    impl Model for CallSlow {
        fn complete(&mut self, request: ModelRequest<'_>, _: &AtomicBool, _: &mut dyn FnMut(&str) -> std::io::Result<()>) -> std::io::Result<AssistantTurn> {
            let reply = r#"{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"fake__slow","arguments":"{\"ms\":1500}"}}]}}]}"#;
            AssistantTurn::from_response(&json::parse(reply).unwrap(), request.wire)
        }
    }
    struct NoTools;
    impl ToolExecutor for NoTools {
        fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
            Err(ToolError::UnknownTool)
        }
    }
    let project = temp.0.join("project");
    std::fs::create_dir(&project).unwrap();
    let workspace = vade_harness::tools::Workspace::new(&project).unwrap();
    let request = json::parse(r#"{"tools":[{"type":"function","function":{"name":"fake__slow"}}],"messages":[{"role":"user","content":"wait"}]}"#).unwrap();
    let mut conversation = agent::Conversation::from_request(request, Limits::default()).unwrap();
    let mut journal = vade_harness::journal::Journal::create(&workspace, "mcp-cancel", &conversation).unwrap();
    let hub = std::sync::Mutex::new(Hub::start(vec![spec("normal", 60_000, Some(&log))], &temp.0));
    let mut tools = McpTools::new(NoTools, Some(&hub), ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
    let cancel = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            wait_for_line(&log, "tools/call", 2);
            cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        agent::run_observed(&mut conversation, &mut CallSlow, &mut tools, Limits::default(), &cancel, &mut |_| {}, &mut journal);
    });
    assert!(tools.last_side_effect_uncertain());
    drop(journal);
    let (_, resumed, _) = vade_harness::journal::Journal::resume(&workspace, "mcp-cancel", Limits::default()).unwrap();
    let result = resumed.messages().last().unwrap();
    assert_eq!(result.tool_call_id(), Some("call-1"));
    let content = result.value().get("content").and_then(Value::as_str).unwrap();
    assert!(content.contains("unknown_outcome"), "{content}");
    wait_for_line(&log, "slow done", 2);
}

fn harness(home: &Path, project: &Path, args: &[&str], mcp: Option<&str>) -> (bool, String, String) {
    // The fixture HOME and VADE_MCP=off come first; each case sets its own.
    let mut command = home::harness();
    command.args(args).current_dir(project).env("VADE_HOME", home).env_remove("FAKE_MCP").env_remove("VADE_MEMORY");
    match mcp {
        Some(value) => command.env("VADE_MCP", value),
        None => command.env_remove("VADE_MCP"),
    };
    let output = command.output().unwrap();
    (output.status.success(), String::from_utf8(output.stdout).unwrap(), String::from_utf8(output.stderr).unwrap())
}

fn cli_manages_and_lists_servers() {
    let home = Temp::new("home");
    let project = Temp::new("project");
    let (h, p) = (&home.0, &project.0);
    let run = |args: &[&str]| harness(h, p, args, None);
    let (ok, out, _) = run(&["mcp", "list"]);
    assert!(ok && out.contains("No MCP servers"), "{out}");
    let (ok, out, err) = run(&["mcp", "add", "fake", "--env", "FAKE_MCP=normal", "--env", "SECRET_TOKEN=hidden-value", "--timeout-ms", "5000", "--", &exe(), "--flag"]);
    assert!(ok && out.contains("Added MCP server fake"), "{out}{err}");
    let user = std::fs::read_to_string(h.join("mcp.json")).unwrap();
    assert!(user.starts_with("{\n  \"servers\": {\n    \"fake\": {\n"), "{user}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(h.join("mcp.json")).unwrap().permissions().mode() & 0o777, 0o600);
    }
    let (ok, out, err) = run(&["mcp", "add", "fake", "--", "x"]);
    assert!(!ok && err.contains("already in"), "{out}{err}");
    let (ok, _, err) = run(&["mcp", "add", "bad_name", "--", "x"]);
    assert!(!ok && err.contains("letters, digits or hyphens"), "{err}");
    for (args, reason) in [
        (&["mcp", "add", "x", "-e", "A=B", "--", "y"][..], "unknown option"),
        (&["mcp", "add", "x", "extra", "--", "y"], "put the server command after --"),
        (&["mcp", "add", "x", "--timeout-ms", "+100", "--", "y"], "--timeout-ms needs"),
        (&["mcp", "add", "x", "--env", "T=a", "--env", "T=b", "--", "y"], "env sets T more than once"),
    ] {
        let (ok, _, err) = run(args);
        assert!(!ok && err.contains(reason), "{args:?}: {err}");
    }
    let (ok, _, err) = run(&["mcp", "add", "plain", "--timeout-ms", "0100", "--", "y"]);
    assert!(ok, "{err}");
    let user = std::fs::read_to_string(h.join("mcp.json")).unwrap();
    assert!(user.contains("\"timeout_ms\": 100,") && !user.contains("\"x\""), "{user}");
    assert!(run(&["mcp", "remove", "plain"]).0);
    let (ok, out, _) = run(&["mcp", "list"]);
    assert!(ok && out.contains("fake  user  enabled") && out.contains("--flag") && out.contains("env: FAKE_MCP, SECRET_TOKEN"), "{out}");
    assert!(!out.contains("hidden-value"), "list must not print env values: {out}");
    let (ok, out, err) = run(&["mcp", "test", "fake"]);
    assert!(ok && out.contains("Server: fake 1.0, protocol 2025-06-18") && out.contains("Tools (9):") && out.contains("fake__dotted_name"), "{out}{err}");
    assert!(out.contains("Resources (1, listed only): file:///notes.txt") && out.contains("Prompts (2, listed only): review, summarize"), "{out}");
    let (ok, out, _) = run(&["tools"]);
    assert!(ok && out.contains("Native tools (") && out.contains("  workspace_search  ") && out.contains("MCP tools (9):") && out.contains("  fake__echo  Echo the arguments\n"), "{out}");
    let (ok, out, _) = harness(h, p, &["tools"], Some("off"));
    assert!(ok && out.contains("MCP tools (0):"), "{out}");
    // The list matches agent chat: the memory tools unless VADE_MEMORY=0.
    assert!(out.contains("  memory_save  ") && out.contains("  memory_forget  "), "{out}");
    let output = home::harness().args(["tools"]).current_dir(p).env("VADE_HOME", h).env("VADE_MEMORY", "0").output().unwrap();
    let out = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success() && out.contains("Native tools (") && !out.contains("memory_"), "{out}");

    // A project entry starts only after a yes; a changed entry needs a new yes.
    let entry = format!(r#"{{"command":{},"env":{{"FAKE_MCP":"normal"}}}}"#, json::quote(&exe()));
    std::fs::create_dir_all(p.join(".vade")).unwrap();
    std::fs::write(p.join(".vade/mcp.json"), format!(r#"{{"mcpServers":{{"shared":{entry}}}}}"#)).unwrap();
    let (ok, out, err) = run(&["tools"]);
    assert!(ok && out.contains("MCP tools (9):") && err.contains("project server shared is not trusted"), "{out}{err}");
    let (_, out, _) = run(&["mcp", "list"]);
    assert!(out.contains("shared  project  not trusted"), "{out}");
    let (ok, out, err) = run(&["mcp", "trust", "shared"]);
    assert!(ok && out.contains("Trusted project MCP server shared"), "{out}{err}");
    let (ok, out, _) = run(&["tools"]);
    assert!(ok && out.contains("MCP tools (18):") && out.contains("shared__echo"), "{out}");
    std::fs::write(p.join(".vade/mcp.json"), format!(r#"{{"mcpServers":{{"shared":{}}}}}"#, entry.replace("normal", "banner"))).unwrap();
    let (_, out, _) = run(&["mcp", "list"]);
    assert!(out.contains("shared  project  not trusted"), "{out}");
    // Adding to the project list is your own yes.
    let (ok, out, err) = run(&["mcp", "add", "--project", "mine", "--", &exe()]);
    assert!(ok, "{out}{err}");
    let (_, out, _) = run(&["mcp", "list"]);
    assert!(out.contains("mine  project  enabled"), "{out}");
    let project_file = std::fs::read_to_string(p.join(".vade/mcp.json")).unwrap();
    assert!(project_file.contains("\"mcpServers\"") && !project_file.contains("\"servers\""), "{project_file}");
    let (ok, _, err) = run(&["mcp", "remove", "mine"]);
    assert!(!ok && err.contains("add --project"), "{err}");
    let (ok, _, _) = run(&["mcp", "remove", "mine", "--project"]);
    assert!(ok);
    let user = std::fs::read_to_string(h.join("mcp.json")).unwrap();
    assert!(!user.contains("\"mine\""), "{user}");
    let (ok, _, _) = run(&["mcp", "remove", "fake"]);
    assert!(ok);
    let (_, out, _) = run(&["mcp", "list"]);
    assert!(!out.contains("fake  user"), "{out}");
    let (ok, _, err) = harness(Path::new("/tmp/a/../b"), p, &["mcp", "list"], None);
    assert!(ok && err.contains("without '..'"), "{err}");
    // A broken file is reported and never overwritten.
    std::fs::write(h.join("mcp.json"), "{broken").unwrap();
    let (ok, _, err) = run(&["mcp", "add", "x", "--", "y"]);
    assert!(!ok && err.contains("not valid JSON"), "{err}");
    assert_eq!(std::fs::read_to_string(h.join("mcp.json")).unwrap(), "{broken");
    let (ok, out, _) = run(&["mcp", "--help"]);
    assert!(ok && out.starts_with("Usage: vade mcp"));
    let (ok, _, err) = run(&["mcp", "launch"]);
    assert!(!ok && err.contains("unknown mcp command"), "{err}");
    let workspace = p.to_string_lossy();
    let (ok, _, err) = run(&["mcp", "--workspace", &workspace]);
    assert!(!ok && err.contains("name an mcp command") && !err.contains("panicked"), "{err}");
    let (ok, _, err) = run(&["tools", "extra"]);
    assert!(!ok && err.contains("Run vade tools --help"), "{err}");
}

/// A change the next read would refuse is never saved; the old file stays.
fn config_writes_keep_a_readable_file() {
    let home = Temp::new("limits-home");
    let project = Temp::new("limits-project");
    let (h, p) = (&home.0, &project.0);
    let run = |args: &[&str]| harness(h, p, args, Some("off"));
    // Trust copies the project entry one level deeper into the user file.
    let deep = format!("{}0{}", "[".repeat(61), "]".repeat(61));
    std::fs::create_dir_all(p.join(".vade")).unwrap();
    std::fs::write(p.join(".vade/mcp.json"), format!(r#"{{"servers":{{"deep":{{"command":"unused","metadata":{deep}}}}}}}"#)).unwrap();
    let user = r#"{"servers":{"existing":{"command":"unused"}}}"#;
    std::fs::write(h.join("mcp.json"), user).unwrap();
    let (ok, out, err) = run(&["mcp", "trust", "deep"]);
    assert!(!ok && err.contains("nesting or value limits"), "{out}{err}");
    assert_eq!(std::fs::read_to_string(h.join("mcp.json")).unwrap(), user);
    let (ok, out, err) = run(&["mcp", "list"]);
    assert!(ok && out.contains("existing  user  enabled") && out.contains("deep  project  not trusted"), "{out}{err}");
    // One more server pushes a file near the value limit over it.
    let user = format!(r#"{{"note":[{}],"servers":{{}}}}"#, vec!["0"; 99_994].join(","));
    std::fs::write(h.join("mcp.json"), &user).unwrap();
    let (ok, out, err) = run(&["mcp", "add", "added", "--", "unused"]);
    assert!(!ok && err.contains("nesting or value limits"), "{out}{err}");
    assert_eq!(std::fs::read_to_string(h.join("mcp.json")).unwrap(), user);
    let (ok, out, err) = run(&["mcp", "list"]);
    assert!(ok && !err.contains("skipped"), "{out}{err}");
}

fn chat_calls_mcp_tools_and_resumes(web: bool) {
    use coding::{Fixture, answer, call, mock, run_bounded, tools};
    let home = Temp::new("chat-home");
    std::fs::write(home.0.join("mcp.json"), format!(r#"{{"servers":{{"fake":{{"command":{},"env":{{"FAKE_MCP":"normal"}}}}}}}}"#, json::quote(&exe()))).unwrap();
    let f = Fixture::new();
    let web_setting = if web { "1" } else { "0" };
    let (url, server) = mock(vec![tools(vec![call("m1", "fake__echo", json::parse(r#"{"text":"from model"}"#).unwrap())]), answer()]);
    let output = run_bounded(
        f.command(&url).args(["--session", "mcp-chat", "--approval", "auto", "--prompt", "echo it"]).env("VADE_HOME", &home.0).env_remove("VADE_MCP").env("VADE_WEB", web_setting),
        "",
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    let requests = server.join().unwrap();
    let names = |request: &Value| -> Vec<String> {
        request.get("tools").and_then(Value::as_array).unwrap().iter()
            .map(|tool| tool.get("function").and_then(|f| f.get("name")).and_then(Value::as_str).unwrap().to_owned())
            .collect()
    };
    let offered = names(&requests[0]);
    // The native list comes first and is unchanged; MCP tools follow it.
    let native: Vec<String> = vade_harness::coding_tools::definitions_with_web(
        vade_harness::contract::guaranteed_native_reply_tokens(), vade_harness::coding_tools::shell_offered(), web,
    ).as_array().unwrap().iter()
        .map(|tool| tool.get("function").and_then(|f| f.get("name")).and_then(Value::as_str).unwrap().to_owned())
        .collect();
    assert_eq!(offered[..native.len()], native[..]);
    for name in ["web_fetch", "web_search"] {
        assert_eq!(offered.iter().any(|tool| tool == name), web);
    }
    assert_eq!(offered[native.len()..].len(), 9);
    assert_eq!(offered[native.len()], "fake__echo");
    let result = requests[1].get("messages").and_then(Value::as_array).unwrap().iter()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("tool")).unwrap()
        .get("content").and_then(Value::as_str).unwrap().to_owned();
    assert_eq!(result, r#"{"text":"from model"}"#);
    // Resume keeps the frozen tool list, MCP tools included.
    let (url, server) = mock(vec![answer()]);
    let output = run_bounded(
        f.command(&url).args(["--resume", "mcp-chat", "--approval", "auto", "--prompt", "again"]).env("VADE_HOME", &home.0).env_remove("VADE_MCP").env("VADE_WEB", web_setting),
        "",
    );
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(names(&server.join().unwrap()[0]), offered);
    // With MCP off, a fresh chat offers the native list only.
    let (url, server) = mock(vec![answer()]);
    let output = run_bounded(f.command(&url).args(["--prompt", "plain"]).env("VADE_HOME", &home.0).env("VADE_MCP", "off").env("VADE_WEB", web_setting), "");
    assert!(output.status.success());
    assert_eq!(names(&server.join().unwrap()[0]), native);
    // The launcher passes --vade-home and drops VADE_HOME; the list comes from that home.
    let (url, server) = mock(vec![answer()]);
    let mut command = home::harness();
    command.args(["agent", "--vade-home"]).arg(&home.0).args(["--endpoint", &url, "--model", "mock", "--prompt", "home", "--workspace"]).arg(&f.0)
        .env("NO_COLOR", "1").env("HOME", f.home()).env_remove("VADE_HOME").env_remove("VADE_MCP").env("VADE_WEB", web_setting);
    let output = run_bounded(&mut command, "");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(names(&server.join().unwrap()[0]), offered);
}
