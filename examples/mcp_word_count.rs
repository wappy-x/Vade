//! A small MCP tool server for a first test of `vade mcp`: one tool,
//! `word_count`, over stdio. It needs no token, no network and no runtime
//! besides the Rust that `scripts/install.sh` installs. Build and add it:
//!   cargo build --release --offline --locked --manifest-path tools/Cargo.toml --example mcp-word-count
//!   vade mcp add words -- "$PWD/tools/target/release/examples/mcp-word-count"
//! See docs/mcp.md.
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../tools/src/json.rs"] mod json;
use json::{Value, parse, quote};
use std::io::{BufRead, Write};

const TOOLS: &str = r#"{"tools":[{"name":"word_count","description":"Count the words, lines and characters of a text.","inputSchema":{"type":"object","properties":{"text":{"type":"string","description":"The text to count"}},"required":["text"]}}]}"#;

/// The reply to one JSON-RPC line, or `None` for a notification.
fn reply(line: &str) -> Option<String> {
    let message = match parse(line) {
        Ok(message) => message,
        Err(_) => return Some(error(&Value::Null, -32700, "parse error")),
    };
    let id = message.get("id");
    if *id == Value::Null { return None; }
    let result = match message.get("method").text().unwrap_or("") {
        "initialize" => {
            let asked = message.get("params").get("protocolVersion").text().unwrap_or("2025-06-18");
            format!(r#"{{"protocolVersion":{},"capabilities":{{"tools":{{}}}},"serverInfo":{{"name":"word-count","version":"0.1.0"}}}}"#, quote(asked))
        }
        "ping" => "{}".into(),
        "tools/list" => TOOLS.into(),
        "tools/call" => {
            let params = message.get("params");
            if params.get("name").text().ok() != Some("word_count") { return Some(error(id, -32602, "unknown tool")); }
            let (text, failed) = match params.get("arguments").get("text") {
                Value::String(text) => (count(text), false),
                _ => ("give text as a string".to_owned(), true),
            };
            format!(r#"{{"content":[{{"type":"text","text":{}}}],"isError":{failed}}}"#, quote(&text))
        }
        _ => return Some(error(id, -32601, "method not found")),
    };
    Some(format!(r#"{{"jsonrpc":"2.0","id":{},"result":{result}}}"#, id.encode()))
}

fn error(id: &Value, code: i32, message: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","id":{},"error":{{"code":{code},"message":{}}}}}"#, id.encode(), quote(message))
}

fn count(text: &str) -> String {
    format!("words: {}, lines: {}, characters: {}", text.split_whitespace().count(), text.lines().count(), text.chars().count())
}

fn main() {
    let mut out = std::io::stdout().lock();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() { continue; }
        if let Some(reply) = reply(&line)
            && writeln!(out, "{reply}").and_then(|()| out.flush()).is_err() { break; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn answers_the_handshake_list_and_call() {
        let init = reply(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}"#).unwrap();
        assert_eq!(parse(&init).unwrap().get("result").get("protocolVersion").text().unwrap(), "2025-06-18");
        assert_eq!(reply(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#), None);
        let list = parse(&reply(r#"{"jsonrpc":"2.0","id":"a","method":"tools/list"}"#).unwrap()).unwrap();
        assert_eq!(list.get("id").text().unwrap(), "a");
        assert_eq!(list.get("result").get("tools").get("0").get("name").text().unwrap(), "word_count");
        let call = parse(&reply(r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"word_count","arguments":{"text":"one two\nthree \"four\" five"}}}"#).unwrap()).unwrap();
        assert_eq!(call.get("result").get("content").get("0").get("text").text().unwrap(), "words: 5, lines: 2, characters: 25");
        assert_eq!(call.get("result").get("isError"), &Value::Bool(false));
    }
    #[test]
    fn bad_input_gets_an_error_not_a_crash() {
        assert!(reply("not json").unwrap().contains("-32700"));
        assert!(reply(r#"{"jsonrpc":"2.0","id":4,"method":"resources/read"}"#).unwrap().contains("-32601"));
        assert!(reply(r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"other"}}"#).unwrap().contains("-32602"));
        let call = parse(&reply(r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"word_count","arguments":{}}}"#).unwrap()).unwrap();
        assert_eq!(call.get("result").get("isError"), &Value::Bool(true));
    }
}
