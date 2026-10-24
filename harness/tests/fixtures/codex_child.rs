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
    json::parse(&line).unwrap()
}
fn send(value: Value) {
    let mut out = io::stdout().lock();
    out.write_all(json::stringify(&value).unwrap().as_bytes())
        .unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}
fn reply(id: &Value, result: Value) {
    send(obj(vec![("id", id.clone()), ("result", result)]));
}
fn turn(id: &str, status: &str) -> Value {
    obj(vec![
        ("id", s(id)),
        ("status", s(status)),
        ("items", Value::Array(vec![])),
        ("error", Value::Null),
    ])
}
fn delta(thread: &str, turn: &str, text: &str) {
    let value = obj(vec![
        ("method", s("item/agentMessage/delta")),
        (
            "params",
            obj(vec![
                ("threadId", s(thread)),
                ("turnId", s(turn)),
                ("itemId", s("mock-item")),
                ("delta", s(text)),
            ]),
        ),
    ]);
    let bytes = format!("{}\n", json::stringify(&value).unwrap());
    let mut out = io::stdout().lock();
    for chunk in bytes.as_bytes().chunks(3) {
        out.write_all(chunk).unwrap();
    }
    out.flush().unwrap();
}
fn completed(thread: &str, id: &str, status: &str) {
    send(obj(vec![
        ("method", s("turn/completed")),
        (
            "params",
            obj(vec![("threadId", s(thread)), ("turn", turn(id, status))]),
        ),
    ]));
}
fn item(thread: &str, turn: &str, method: &str, kind: &str, text: &str) {
    send(obj(vec![
        ("method", s(method)),
        (
            "params",
            obj(vec![
                ("threadId", s(thread)),
                ("turnId", s(turn)),
                (
                    "item",
                    obj(vec![
                        ("id", s("mock-item")),
                        ("type", s(kind)),
                        ("text", s(text)),
                    ]),
                ),
            ]),
        ),
    ]));
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args[1], "app-server");
    assert!(args.iter().any(|a| a == "approval_policy=\"never\""));
    assert!(args.iter().any(|a| a == "mcp_servers={}"));
    let fixture_mode = std::fs::read_to_string("fixture-mode").unwrap_or_default();
    if fixture_mode == "init-hang" {
        std::thread::sleep(Duration::from_secs(30));
    }
    let mut reader = io::stdin().lock();
    let init = read(&mut reader);
    assert_eq!(get(&init, "method"), "initialize");
    assert_eq!(
        get(
            init.get("params").unwrap().get("clientInfo").unwrap(),
            "name"
        ),
        "vade_harness"
    );
    assert_eq!(
        init.get("params")
            .unwrap()
            .get("capabilities")
            .unwrap()
            .get("experimentalApi"),
        Some(&Value::Bool(false))
    );
    if fixture_mode == "bad-init" {
        reply(
            init.get("id").unwrap(),
            obj(vec![("protocolVersion", s("unknown"))]),
        );
        return;
    }
    reply(
        init.get("id").unwrap(),
        obj(vec![(
            "userAgent",
            s(if fixture_mode == "v2" {
                "mock/2"
            } else {
                "mock/1"
            }),
        )]),
    );
    assert_eq!(get(&read(&mut reader), "method"), "initialized");
    let start = read(&mut reader);
    assert!(matches!(
        get(&start, "method"),
        "thread/start" | "thread/resume"
    ));
    let params = start.get("params").unwrap();
    assert_eq!(get(params, "approvalPolicy"), "never");
    let cwd = get(params, "cwd");
    assert_eq!(std::env::current_dir().unwrap().to_str().unwrap(), cwd);
    let mode = get(params, "sandbox");
    assert!(matches!(mode, "read-only" | "workspace-write"));
    let thread = params
        .get("threadId")
        .and_then(Value::as_str)
        .unwrap_or("mock-thread");
    let model = params
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("mock");
    let mut sandbox = if model == "escalate" {
        obj(vec![("type", s("dangerFullAccess"))])
    } else if mode == "read-only" {
        obj(vec![
            ("type", s("readOnly")),
            ("networkAccess", Value::Bool(false)),
        ])
    } else {
        obj(vec![
            ("type", s("workspaceWrite")),
            ("writableRoots", Value::Array(vec![s(cwd)])),
            ("networkAccess", Value::Bool(false)),
            ("excludeSlashTmp", Value::Bool(true)),
            ("excludeTmpdirEnvVar", Value::Bool(true)),
        ])
    };
    if let Value::Object(fields) = &mut sandbox {
        if model == "missing-network" {
            fields.retain(|(k, _)| k != "networkAccess");
        }
        if model == "network-enabled" {
            fields
                .iter_mut()
                .find(|(k, _)| k == "networkAccess")
                .unwrap()
                .1 = Value::Bool(true);
        }
        if model == "outside-root" {
            fields
                .iter_mut()
                .find(|(k, _)| k == "writableRoots")
                .unwrap()
                .1 = Value::Array(vec![s("/")]);
        }
        if model == "tmp-write" {
            fields
                .iter_mut()
                .find(|(k, _)| k == "excludeSlashTmp")
                .unwrap()
                .1 = Value::Bool(false);
        }
    }
    let returned_thread = if model == "wrong-resume" {
        "wrong-thread"
    } else {
        thread
    };
    reply(
        start.get("id").unwrap(),
        obj(vec![
            ("thread", obj(vec![("id", s(returned_thread))])),
            ("cwd", s(if model == "wrong-cwd" { "/" } else { cwd })),
            (
                "approvalPolicy",
                s(if model == "wrong-approval" {
                    "on-request"
                } else {
                    "never"
                }),
            ),
            ("sandbox", sandbox),
        ]),
    );
    if model == "block-input" {
        std::thread::sleep(Duration::from_secs(30));
    }
    let mut counter = 0;
    let mut prefix = String::from("mock-owned system and tools\n");
    loop {
        let request = read(&mut reader);
        assert_eq!(get(&request, "method"), "turn/start");
        counter += 1;
        let params = request.get("params").unwrap();
        assert_eq!(get(params, "threadId"), thread);
        assert_eq!(get(params, "cwd"), cwd);
        assert_eq!(get(params, "approvalPolicy"), "never");
        let policy = params.get("sandboxPolicy").unwrap();
        assert_eq!(
            get(policy, "type"),
            if mode == "read-only" {
                "readOnly"
            } else {
                "workspaceWrite"
            }
        );
        assert_eq!(policy.get("networkAccess"), Some(&Value::Bool(false)));
        if mode == "workspace-write" {
            assert_eq!(
                policy.get("writableRoots").unwrap().as_array().unwrap()[0].as_str(),
                Some(cwd)
            );
            assert_eq!(policy.get("excludeSlashTmp"), Some(&Value::Bool(true)));
        }
        let text = get(&params.get("input").unwrap().as_array().unwrap()[0], "text");
        if text == "rpc-error" {
            send(obj(vec![
                ("id", request.get("id").unwrap().clone()),
                (
                    "error",
                    obj(vec![
                        ("code", Value::Number("-32000".into())),
                        ("message", s("mock start rejected")),
                    ]),
                ),
            ]));
            continue;
        }
        let id = format!("mock-turn-{counter}");
        if text == "wrong-response" {
            reply(&Value::Number("999999".into()), obj(vec![]));
        }
        if text == "pre-ack" {
            delta(thread, "stale-turn", "not displayed");
            delta("other-thread", &id, "not displayed");
            delta(thread, &id, "before 🦀");
        }
        if text == "initial-terminal" {
            delta(thread, "stale-turn", "not displayed");
        }
        reply(
            request.get("id").unwrap(),
            obj(vec![(
                "turn",
                turn(
                    &id,
                    if text == "initial-terminal" {
                        "completed"
                    } else {
                        "inProgress"
                    },
                ),
            )]),
        );
        if model == "prefix-probe" {
            // The mock owns delegated history. Only the exact new user input
            // crosses the bridge, even after mock-owned tool activity.
            let inputs = params.get("input").unwrap().as_array().unwrap();
            assert_eq!(inputs.len(), 1);
            assert_eq!(get(&inputs[0], "type"), "text");
            assert!(params.get("instructions").is_none() && params.get("history").is_none());
            let expected = [
                " exact first\n🦀 ",
                "quotes \" and slash \\",
                "third\n\n  whitespace",
            ];
            assert_eq!(text, expected[counter - 1]);
            let old = prefix.clone();
            prefix.push_str(&json::stringify(&inputs[0]).unwrap());
            prefix.push_str("\nmock lookup {}\nmock result 🦀\n");
            send(obj(vec![
                ("method", s("item/completed")),
                (
                    "params",
                    obj(vec![
                        ("threadId", s(thread)),
                        ("turnId", s(&id)),
                        (
                            "item",
                            obj(vec![
                                ("type", s("dynamicToolCall")),
                                ("id", s("mock-tool")),
                                ("status", s("completed")),
                            ]),
                        ),
                    ]),
                ),
            ]));
            assert!(prefix.starts_with(&old));
            delta(thread, &id, text);
            item(thread, &id, "item/completed", "agentMessage", text);
            completed(thread, &id, "completed");
            continue;
        }
        match text {
            "initial-terminal" => {}
            "steer" | "steer-reject" | "steer-wrong" | "steer-timeout" | "steer-race"
            | "steer-wait" => {
                delta(thread, &id, "provisional work");
                let control = read(&mut reader);
                assert_eq!(get(&control, "method"), "turn/steer");
                let p = control.get("params").unwrap();
                assert_eq!(get(p, "threadId"), thread);
                assert_eq!(get(p, "expectedTurnId"), id);
                assert_eq!(p.get("input").unwrap().as_array().unwrap().len(), 1);
                let steer_text = get(&p.get("input").unwrap().as_array().unwrap()[0], "text");
                assert!(p.get("cwd").is_none() && p.get("sandboxPolicy").is_none());
                std::fs::write("steering-input", steer_text).unwrap();
                std::fs::write("steering-received", b"yes").unwrap();
                if text == "steer-timeout" {
                    std::thread::sleep(Duration::from_secs(30));
                    continue;
                }
                if text == "steer-race" {
                    item(
                        thread,
                        &id,
                        "item/completed",
                        "agentMessage",
                        "completed before steering acknowledgment",
                    );
                    completed(thread, &id, "completed");
                    continue;
                }
                if text == "steer-reject" {
                    send(obj(vec![
                        ("id", control.get("id").unwrap().clone()),
                        (
                            "error",
                            obj(vec![
                                ("code", Value::Number("-32000".into())),
                                ("message", s("fixture steering rejected")),
                            ]),
                        ),
                    ]));
                } else {
                    if text == "steer-wait" {
                        std::thread::sleep(Duration::from_millis(120));
                    }
                    reply(
                        control.get("id").unwrap(),
                        obj(vec![(
                            "turnId",
                            s(if text == "steer-wrong" {
                                "wrong-turn"
                            } else {
                                &id
                            }),
                        )]),
                    );
                }
                if text == "steer-wait" {
                    let interrupt = read(&mut reader);
                    assert_eq!(get(&interrupt, "method"), "turn/interrupt");
                    reply(interrupt.get("id").unwrap(), obj(vec![]));
                    std::thread::sleep(Duration::from_millis(120));
                    completed(thread, &id, "interrupted");
                } else {
                    item(thread, &id, "item/completed", "agentMessage", steer_text);
                    completed(thread, &id, "completed");
                }
            }
            "burst-control" => {
                for _ in 0..2000 {
                    delta(thread, &id, &"b".repeat(128));
                }
                let interrupt = read(&mut reader);
                assert_eq!(get(&interrupt, "method"), "turn/interrupt");
                reply(interrupt.get("id").unwrap(), obj(vec![]));
                std::thread::sleep(Duration::from_millis(120));
                completed(thread, &id, "interrupted");
            }

            "pre-ack" => {
                delta(thread, &id, "after é");
                item(
                    thread,
                    &id,
                    "item/completed",
                    "agentMessage",
                    "before 🦀after é",
                );
                completed(thread, &id, "completed");
            }
            "failed" | "contradictory" | "retry" => {
                send(obj(vec![
                    ("method", s("error")),
                    (
                        "params",
                        obj(vec![
                            ("threadId", s(thread)),
                            ("turnId", s(&id)),
                            ("willRetry", Value::Bool(text == "retry")),
                            ("error", obj(vec![("message", s("mock provider failed"))])),
                        ]),
                    ),
                ]));
                completed(
                    thread,
                    &id,
                    if text == "failed" {
                        "failed"
                    } else {
                        "completed"
                    },
                );
            }
            "interrupt" | "interrupt-timeout" | "cancel-tool" | "cancel-race"
            | "interrupt-rejected" => {
                if text == "cancel-tool" {
                    item(thread, &id, "item/started", "commandExecution", "");
                }
                let interrupt = read(&mut reader);
                assert_eq!(get(&interrupt, "method"), "turn/interrupt");
                let p = interrupt.get("params").unwrap();
                assert_eq!(get(p, "threadId"), thread);
                assert_eq!(get(p, "turnId"), id);
                if text == "interrupt-rejected" {
                    send(obj(vec![
                        ("id", interrupt.get("id").unwrap().clone()),
                        (
                            "error",
                            obj(vec![
                                ("code", Value::Number("-32000".into())),
                                ("message", s("fixture rejected interruption")),
                            ]),
                        ),
                    ]));
                    continue;
                }
                reply(interrupt.get("id").unwrap(), obj(vec![]));
                if text == "cancel-race" {
                    completed(thread, &id, "completed");
                } else if text != "interrupt-timeout" {
                    completed(thread, &id, "interrupted");
                } else {
                    std::thread::sleep(Duration::from_secs(30));
                }
            }
            "approval" => {
                for (method, request_id) in [
                    ("item/commandExecution/requestApproval", 101),
                    ("item/permissions/requestApproval", 102),
                    ("item/tool/call", 103),
                    ("unknown/futureCallback", 104),
                    ("thread/shellCommand", 105),
                    ("item/fileChange/requestApproval", 106),
                    ("mcpServer/elicitation/request", 107),
                ] {
                    let request_id = Value::Number(request_id.to_string());
                    send(obj(vec![
                        ("method", s(method)),
                        ("id", request_id.clone()),
                        (
                            "params",
                            obj(vec![("threadId", s(thread)), ("turnId", s(&id))]),
                        ),
                    ]));
                    let decision = read(&mut reader);
                    assert_eq!(decision.get("id"), Some(&request_id));
                    match method {
                        "item/commandExecution/requestApproval"
                        | "item/fileChange/requestApproval" => {
                            assert_eq!(get(decision.get("result").unwrap(), "decision"), "decline")
                        }
                        "item/permissions/requestApproval" => assert_eq!(
                            decision.get("result").unwrap().get("permissions"),
                            Some(&obj(vec![]))
                        ),
                        "mcpServer/elicitation/request" => {
                            assert_eq!(get(decision.get("result").unwrap(), "action"), "decline")
                        }
                        _ => assert!(decision.get("error").is_some()),
                    }
                }
                completed(thread, &id, "completed");
            }
            "oversize" => {
                let mut out = io::stdout().lock();
                out.write_all(&vec![b'x'; 1024 * 1024 + 1]).unwrap();
                out.write_all(b"\n").unwrap();
                out.flush().unwrap();
            }
            "partial" => {
                let mut out = io::stdout().lock();
                out.write_all(b"{\"method\":").unwrap();
                out.flush().unwrap();
                return;
            }
            "malformed" => {
                println!("{{invalid}}");
                return;
            }
            "unknown-event" => {
                send(obj(vec![
                    ("method", s("turn/futureEvent")),
                    (
                        "params",
                        obj(vec![("threadId", s(thread)), ("turnId", s(&id))]),
                    ),
                ]));
            }
            "unknown-item" => {
                item(thread, &id, "item/completed", "futureItem", "");
                completed(thread, &id, "completed");
            }
            "unfinished-item" => {
                delta(thread, &id, "draft");
                completed(thread, &id, "completed");
            }
            "disagreement" => {
                delta(thread, &id, "wrong draft");
                item(
                    thread,
                    &id,
                    "item/completed",
                    "agentMessage",
                    "correct final",
                );
                completed(thread, &id, "completed");
            }
            "conflicting-item" => {
                item(thread, &id, "item/completed", "agentMessage", "first");
                item(thread, &id, "item/completed", "agentMessage", "second");
                completed(thread, &id, "completed");
            }
            "hang" => std::thread::sleep(Duration::from_secs(30)),
            "exit" => std::process::exit(7),
            "flood" => {
                for _ in 0..100 {
                    delta(thread, &id, "x");
                }
                item(
                    thread,
                    &id,
                    "item/completed",
                    "agentMessage",
                    &"x".repeat(100),
                );
                completed(thread, &id, "completed");
            }
            _ => {
                delta(thread, &id, text);
                item(thread, &id, "item/completed", "agentMessage", text);
                completed(thread, &id, "completed");
            }
        }
    }
}
