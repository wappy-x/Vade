// First-party ACP v1 mock. It never opens a network socket or runs a tool.
use json::Value;
use std::io::{self, BufRead, Write};
use std::time::Duration;
fn obj(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn s(text: &str) -> Value {
    Value::String(text.into())
}
fn get<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap()
}
fn read(reader: &mut impl BufRead) -> Value {
    let mut line = String::new();
    assert!(reader.read_line(&mut line).unwrap() > 0);
    let value = json::parse(&line).unwrap();
    assert_eq!(get(&value, "jsonrpc"), "2.0");
    value
}
fn send(mut fields: Vec<(&str, Value)>) {
    fields.push(("jsonrpc", s("2.0")));
    let bytes = format!("{}\n", json::stringify(&obj(fields)).unwrap());
    let mut out = io::stdout().lock();
    // Deliberately fragment UTF-8 and JSON framing across writes.
    for chunk in bytes.as_bytes().chunks(3) {
        out.write_all(chunk).unwrap();
    }
    out.flush().unwrap();
}
fn reply(id: &Value, result: Value) {
    send(vec![("id", id.clone()), ("result", result)]);
}
fn update(session: &str, update: Value) {
    send(vec![
        ("method", s("session/update")),
        (
            "params",
            obj(vec![("sessionId", s(session)), ("update", update)]),
        ),
    ]);
}
fn text(session: &str, id: &str, text: &str) {
    update(
        session,
        obj(vec![
            ("sessionUpdate", s("agent_message_chunk")),
            ("messageId", s(id)),
            ("content", obj(vec![("type", s("text")), ("text", s(text))])),
        ]),
    );
}
fn tool(session: &str, kind: &str, id: &str, status: &str) {
    update(
        session,
        obj(vec![
            ("sessionUpdate", s(kind)),
            ("toolCallId", s(id)),
            ("title", s("mock tool (never executed)")),
            ("status", s(status)),
        ]),
    );
}
fn stop(id: &Value, reason: &str) {
    reply(
        id,
        obj(vec![
            ("stopReason", s(reason)),
            ("_meta", obj(vec![("opaque", s("retained externally"))])),
        ]),
    );
}
fn callback(
    reader: &mut impl BufRead,
    session: &str,
    method: &str,
    reject: bool,
    cancelling: bool,
) {
    let options = if reject {
        vec![
            obj(vec![("optionId", s("yes")), ("kind", s("allow_once"))]),
            obj(vec![("optionId", s("no")), ("kind", s("reject_once"))]),
        ]
    } else {
        vec![obj(vec![
            ("optionId", s("yes")),
            ("kind", s("allow_always")),
        ])]
    };
    send(vec![
        ("id", s("mock-callback")),
        ("method", s(method)),
        (
            "params",
            obj(vec![
                ("sessionId", s(session)),
                ("options", Value::Array(options)),
            ]),
        ),
    ]);
    loop {
        let response = read(reader);
        if response.get("method").and_then(Value::as_str) == Some("session/cancel") {
            assert!(cancelling);
            continue;
        }
        assert_eq!(response.get("id"), Some(&s("mock-callback")));
        if method == "session/request_permission" {
            let outcome = response.get("result").unwrap().get("outcome").unwrap();
            if reject && !cancelling {
                assert_eq!(get(outcome, "outcome"), "selected");
                assert_eq!(get(outcome, "optionId"), "no");
            } else {
                assert_eq!(get(outcome, "outcome"), "cancelled");
            }
        } else {
            assert_eq!(
                response.get("error").unwrap().get("code"),
                Some(&Value::Number("-32601".into()))
            );
        }
        break;
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(&args[1..], ["acp"]);
    let mode = std::fs::read_to_string("fixture-mode").unwrap_or_default();
    if mode == "init-hang" {
        std::thread::sleep(Duration::from_secs(30));
    }
    let mut reader = io::stdin().lock();
    let init = read(&mut reader);
    assert_eq!(get(&init, "method"), "initialize");
    let params = init.get("params").unwrap();
    assert_eq!(
        params.get("protocolVersion"),
        Some(&Value::Number("1".into()))
    );
    let caps = params.get("clientCapabilities").unwrap();
    assert_eq!(caps.get("terminal"), Some(&Value::Bool(false)));
    assert_eq!(
        caps.get("fs").unwrap().get("readTextFile"),
        Some(&Value::Bool(false))
    );
    assert_eq!(
        caps.get("fs").unwrap().get("writeTextFile"),
        Some(&Value::Bool(false))
    );
    assert_eq!(
        caps.get("auth").unwrap().get("terminal"),
        Some(&Value::Bool(false))
    );
    if mode == "init-error" {
        send(vec![
            ("id", init.get("id").unwrap().clone()),
            (
                "error",
                obj(vec![
                    ("code", Value::Number("-32000".into())),
                    ("message", s("private provider diagnostic")),
                ]),
            ),
        ]);
        return;
    }
    let mut session_caps = vec![
        ("close", obj(vec![])),
        ("fork", obj(vec![])),
        ("list", obj(vec![])),
    ];
    if mode == "resume" {
        session_caps.push(("resume", obj(vec![])));
    }
    reply(
        init.get("id").unwrap(),
        obj(vec![
            (
                "protocolVersion",
                Value::Number(if mode == "v2" { "2" } else { "1" }.into()),
            ),
            (
                "agentCapabilities",
                obj(vec![
                    ("loadSession", Value::Bool(mode != "no-load")),
                    ("sessionCapabilities", obj(session_caps)),
                ]),
            ),
            ("_meta", obj(vec![("providerExtension", s("opaque setup"))])),
        ]),
    );
    if matches!(mode.as_str(), "v2" | "no-load") {
        std::thread::sleep(Duration::from_secs(30));
        return;
    }
    let setup = read(&mut reader);
    let method = get(&setup, "method");
    assert!(matches!(
        method,
        "session/new" | "session/load" | "session/resume"
    ));
    let params = setup.get("params").unwrap();
    assert_eq!(
        std::path::Path::new(get(params, "cwd")),
        std::env::current_dir().unwrap()
    );
    assert_eq!(params.get("mcpServers"), Some(&Value::Array(vec![])));
    let session = params
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or("mock-session")
        .to_owned();
    if mode == "resume" {
        assert_eq!(method, "session/resume");
    }
    if method == "session/load" {
        text(&session, "history", "old external history");
    }
    if mode == "wrong-setup-session" {
        reply(
            setup.get("id").unwrap(),
            obj(vec![("sessionId", s("different"))]),
        );
    } else {
        reply(
            setup.get("id").unwrap(),
            obj(vec![
                ("sessionId", s(&session)),
                ("_meta", obj(vec![("setupMethod", s(method))])),
            ]),
        );
    }
    loop {
        let prompt = read(&mut reader);
        assert_eq!(get(&prompt, "method"), "session/prompt");
        let params = prompt.get("params").unwrap();
        assert_eq!(get(params, "sessionId"), session);
        let blocks = params.get("prompt").unwrap().as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(get(&blocks[0], "type"), "text");
        let input = get(&blocks[0], "text");
        let id = prompt.get("id").unwrap();
        match input {
            "crash" => std::process::exit(17),
            "partial" => {
                print!("{{\"jsonrpc\":\"2.0\"");
                io::stdout().flush().unwrap();
                return;
            }
            "hang" | "ignore-cancel" => std::thread::sleep(Duration::from_secs(30)),
            "unknown-event" => update(&session, obj(vec![("sessionUpdate", s("future_update"))])),
            "wrong-session" => text("different", "m", "untrusted"),
            "unknown-response" => stop(&s("unknown-request"), "end_turn"),
            "unknown-notification" => send(vec![
                ("method", s("future/notice")),
                ("params", obj(vec![])),
            ]),
            "unknown-stop" => stop(id, "future_reason"),
            "rpc-error" => send(vec![
                ("id", id.clone()),
                (
                    "error",
                    obj(vec![
                        ("code", Value::Number("-32000".into())),
                        ("message", s("private provider diagnostic")),
                    ]),
                ),
            ]),
            "unknown-tool" => tool(&session, "tool_call_update", "unknown", "completed"),
            "unfinished-tool" => {
                tool(&session, "tool_call", "t", "in_progress");
                stop(id, "end_turn");
            }
            "reopened-tool" => {
                tool(&session, "tool_call", "t", "completed");
                tool(&session, "tool_call_update", "t", "in_progress");
                stop(id, "end_turn");
            }
            "unknown-tool-status" => tool(&session, "tool_call", "t", "future_status"),
            "oversized" => {
                let mut out = io::stdout().lock();
                out.write_all(&vec![b'x'; 1024 * 1024 + 1]).unwrap();
                out.flush().unwrap();
            }
            "callbacks" => {
                callback(
                    &mut reader,
                    &session,
                    "session/request_permission",
                    true,
                    false,
                );
                callback(
                    &mut reader,
                    &session,
                    "session/request_permission",
                    false,
                    false,
                );
                for method in [
                    "fs/read_text_file",
                    "fs/write_text_file",
                    "terminal/create",
                    "terminal/output",
                    "terminal/release",
                    "_future/run",
                    "authenticate",
                ] {
                    callback(&mut reader, &session, method, false, false);
                }
                text(&session, "m", "all callbacks denied");
                stop(id, "end_turn");
            }
            "cancel-tool" | "cancel-permission" | "complete-race" => {
                tool(&session, "tool_call", "t", "in_progress");
                std::fs::write("tool-waiting", "ready").unwrap();
                let cancel = read(&mut reader);
                assert_eq!(get(&cancel, "method"), "session/cancel");
                assert!(cancel.get("id").is_none());
                assert_eq!(get(cancel.get("params").unwrap(), "sessionId"), session);
                if input == "cancel-permission" {
                    callback(
                        &mut reader,
                        &session,
                        "session/request_permission",
                        true,
                        true,
                    );
                }
                std::thread::sleep(Duration::from_millis(120));
                if input == "complete-race" {
                    tool(&session, "tool_call_update", "t", "completed");
                    text(&session, "m", "completed before cancellation");
                    stop(id, "end_turn");
                } else {
                    stop(id, "cancelled");
                }
            }
            "interleaved" => {
                text(&session, "one", "first ");
                tool(&session, "tool_call", "t", "in_progress");
                text(&session, "two", "second");
                text(&session, "one", "message 🦀");
                update(
                    &session,
                    obj(vec![
                        ("sessionUpdate", s("plan")),
                        ("entries", Value::Array(vec![])),
                        ("_meta", obj(vec![("opaque", s("provider field"))])),
                    ]),
                );
                tool(&session, "tool_call_update", "t", "completed");
                stop(id, "end_turn");
            }
            "max_tokens" | "max_turn_requests" | "refusal" => {
                text(&session, "m", "provisional");
                stop(id, input);
            }
            _ => {
                text(&session, "m", input);
                stop(id, "end_turn");
            }
        }
    }
}
