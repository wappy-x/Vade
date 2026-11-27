//! The agent never sends the same external tool call again on its own: not
//! after a tool error, not after the vendor's server dies during the call, and
//! not on a resume. The external vendor is a fake MCP server in this binary
//! that logs every tools/call it receives, so the count comes from the vendor.
//!
//! The test runs without the libtest harness (see `Cargo.toml`): an MCP
//! server's stdout may hold only JSON-RPC messages. With `FAKE_VENDOR_LOG` set,
//! `main` is the fake vendor; without it, `main` runs the tests.
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use vade_harness::json::{self, Value};
#[path = "support/home.rs"]
mod home;
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/coding.rs"]
mod coding;
use coding::{Fixture, answer, call, mock, run_bounded, tools};

fn main() {
    if let Some(log) = std::env::var_os("FAKE_VENDOR_LOG") {
        return serve(Path::new(&log));
    }
    let tests: [(&str, fn()); 1] = [("tool_calls_reach_the_vendor_once", tool_calls_reach_the_vendor_once)];
    for (name, test) in tests {
        test();
        println!("test {name} ... ok");
    }
    println!("test result: ok. {} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out", tests.len());
}

// ---- The fake vendor: "charge" refuses, "ship" dies during the call ----

fn serve(log: &Path) {
    let note = |line: &str| {
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(log).unwrap();
        writeln!(file, "{line}").unwrap();
    };
    let send = |text: String| {
        let mut out = std::io::stdout().lock();
        out.write_all(text.as_bytes()).unwrap();
        out.write_all(b"\n").unwrap();
        out.flush().unwrap();
    };
    for line in std::io::stdin().lock().lines() {
        let message = json::parse(&line.unwrap()).unwrap();
        let method = message.get("method").and_then(Value::as_str).unwrap_or("").to_owned();
        let Some(id) = message.get("id").map(|id| json::stringify(id).unwrap()) else { continue };
        let result = match method.as_str() {
            "initialize" => {
                note("initialize");
                r#"{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"vendor","version":"1.0"}}"#.to_owned()
            }
            "tools/list" => r#"{"tools":[{"name":"charge","inputSchema":{"type":"object","properties":{"amount":{"type":"integer"}}}},{"name":"ship","inputSchema":{"type":"object"}}]}"#.to_owned(),
            "tools/call" => {
                let params = message.get("params").unwrap();
                let name = params.get("name").and_then(Value::as_str).unwrap();
                note(&format!("call {name} {}", json::stringify(params.get("arguments").unwrap()).unwrap()));
                if name == "ship" {
                    // The vendor goes away after it took the call: the outcome is unknown.
                    std::process::exit(3);
                }
                r#"{"content":[{"type":"text","text":"card declined"}],"isError":true}"#.to_owned()
            }
            _ => "{}".to_owned(),
        };
        send(format!(r#"{{"jsonrpc":"2.0","id":{id},"result":{result}}}"#));
    }
}

fn calls(log: &Path) -> Vec<String> {
    std::fs::read_to_string(log).unwrap_or_default().lines().filter(|l| l.starts_with("call ")).map(str::to_owned).collect()
}
fn tool_results(request: &Value) -> Vec<String> {
    request.get("messages").and_then(Value::as_array).unwrap().iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("tool"))
        .map(|m| m.get("content").and_then(Value::as_str).unwrap().to_owned())
        .collect()
}

fn tool_calls_reach_the_vendor_once() {
    let home = Fixture::new();
    let log: PathBuf = home.0.join("vendor.log");
    let exe = std::env::current_exe().unwrap();
    std::fs::write(home.0.join("mcp.json"), format!(r#"{{"servers":{{"vendor":{{"command":{},"env":{{"FAKE_VENDOR_LOG":{}}}}}}}}}"#,
        json::quote(&exe.to_string_lossy()), json::quote(&log.to_string_lossy()))).unwrap();
    let f = Fixture::new();
    let agent = |url: &str, args: &[&str], ok: bool| {
        let mut command = f.command(url);
        command.args(["--approval", "auto"]).args(args).env("VADE_HOME", &home.0).env_remove("VADE_MCP").env("VADE_WEB", "0");
        let output = run_bounded(&mut command, "");
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert_eq!(output.status.success(), ok, "{stderr}");
        stderr
    };

    // 1. A tool error goes back to the model; the agent does not call again.
    let (url, server) = mock(vec![tools(vec![call("c1", "vendor__charge", json::parse(r#"{"amount":5}"#).unwrap())]), answer()]);
    agent(&url, &["--session", "vendor", "--prompt", "pay"], true);
    let requests = server.join().unwrap();
    assert_eq!(calls(&log), [r#"call charge {"amount":5}"#]);
    let results = tool_results(&requests[1]);
    assert_eq!(results.len(), 1);
    assert!(results[0].contains("card declined"), "{results:?}");

    // 2. The vendor dies after it took the call. The run stops for the user:
    // no further model turn, no server restart and no second call.
    let (url, server) = mock(vec![tools(vec![call("c2", "vendor__ship", json::parse("{}").unwrap())])]);
    let stderr = agent(&url, &["--resume", "vendor", "--prompt", "ship it"], false);
    assert_eq!(server.join().unwrap().len(), 1);
    assert!(stderr.contains("Stopped: a tool failed"), "{stderr}");
    assert_eq!(calls(&log), [r#"call charge {"amount":5}"#, "call ship {}"]);
    let initializes = || std::fs::read_to_string(&log).unwrap().lines().filter(|l| *l == "initialize").count();
    assert_eq!(initializes(), 2, "one start per chat; no restart without a new call");

    // 3. A resume replays both results to the model and sends no call.
    let (url, server) = mock(vec![answer()]);
    agent(&url, &["--resume", "vendor", "--prompt", "what happened?"], true);
    let requests = server.join().unwrap();
    assert_eq!(calls(&log).len(), 2, "a resume sent a tool call again");
    let results = tool_results(&requests[0]);
    assert!(results.len() == 2 && results[0].contains("card declined") && results[1].contains("unknown_outcome"), "{results:?}");
}
