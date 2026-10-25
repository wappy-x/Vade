//! The workload driver against first-party loopback fixtures: concurrent
//! sessions, per-role reports, telemetry gaps, refusals and the JSON/table CLI.
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/contract.rs"]
mod contract;
use contract::*;

fn member<'a>(value: &'a json::Value, path: &[&str]) -> &'a json::Value {
    path.iter().fold(value, |v, key| v.get(key).unwrap())
}
fn integer(value: &json::Value) -> u64 {
    let json::Value::Number(n) = value else {
        panic!("not a number: {value:?}")
    };
    n.parse().unwrap()
}
#[test]
fn workload_concurrent_native_sessions_grow_and_report_per_role() {
    use std::sync::{Arc, Mutex, atomic::AtomicUsize};
    let (listener, endpoint) = listen();
    let active = Arc::new(AtomicUsize::new(0));
    let keys = Arc::new(Mutex::new(std::collections::HashSet::new()));
    let server_active = Arc::clone(&active);
    let server_keys = Arc::clone(&keys);
    let server = thread::spawn(move || {
        let mut workers = Vec::new();
        for _ in 0..6 {
            let mut reader = accept(&listener);
            let active = Arc::clone(&server_active);
            let keys = Arc::clone(&server_keys);
            workers.push(thread::spawn(move || {
                let mut prefix = String::new();
                let mut affinity = String::new();
                let mut main = false;
                for turn in 1..=3 {
                    if turn > 1 {
                        rendered_history_admission(
                            &mut reader,
                            (if main { 1000 } else { 100 }) + (turn as u64 - 2) * 100 + 3,
                        );
                    }
                    capabilities(&mut reader, CAPS);
                    let (line, envelope) = request(&mut reader);
                    assert_eq!(line, "POST /v1/admission HTTP/1.1");
                    let envelope = json::view(&envelope).unwrap();
                    let body = envelope.get("request").unwrap().raw().to_owned();
                    let v = json::view(&body).unwrap();
                    let messages: Vec<_> = v.get("messages").unwrap().array().unwrap().collect();
                    assert_eq!(messages.len(), 4 * turn);
                    assert_eq!(
                        messages[messages.len() - 2]
                            .get("role")
                            .unwrap()
                            .as_str()
                            .unwrap(),
                        "tool"
                    );
                    if turn > 1 {
                        assert_eq!(
                            messages[messages.len() - 4]
                                .get("content")
                                .unwrap()
                                .as_str()
                                .unwrap(),
                            "answer"
                        );
                    }
                    assert!(body.starts_with(&prefix), "real request prefix changed");
                    prefix = body[..body.len() - 2].into();
                    let key = v
                        .get("prompt_cache_key")
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .into_owned();
                    if turn == 1 {
                        assert!(keys.lock().unwrap().insert(key.clone()));
                        affinity = key.clone();
                        main = messages[0]
                            .get("content")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .contains("role main.");
                    }
                    assert_eq!(key, affinity);
                    let prompt = (if main { 1000 } else { 100 }) + (turn as u64 - 1) * 100;
                    respond(
                        &mut reader,
                        200,
                        "application/json",
                        &admission_body(prompt, 17, true),
                    );
                    assert_eq!(
                        request(&mut reader),
                        ("POST /v1/chat/completions HTTP/1.1".into(), body)
                    );
                    if turn == 1 {
                        active.fetch_add(1, Ordering::SeqCst);
                        let start = Instant::now();
                        while active.load(Ordering::SeqCst) < 6 {
                            assert!(
                                start.elapsed() < Duration::from_secs(3),
                                "sessions were serialized"
                            );
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                    // Delay all turns so wall-window aggregate can be compared
                    // with per-session output rates without sub-ms noise.
                    thread::sleep(Duration::from_millis(10));
                    let u = prompt - 100 + 3;
                    let cached = if turn == 1 {
                        0
                    } else if main {
                        u + 2
                    } else {
                        u / 2
                    };
                    respond(
                        &mut reader,
                        200,
                        "text/event-stream",
                        &format!("{ANSWER}{}{DONE}", usage(prompt, cached)),
                    );
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let report = vade_harness::workload::run(workload_config(&endpoint, 6, 3)).unwrap();
    server.join().unwrap();
    assert!(report.successful());
    assert_eq!(keys.lock().unwrap().len(), 6);
    let body = report.to_json().unwrap();
    assert!(!body.contains(&endpoint));
    assert!(!body.contains("synthetic-0-1"));
    assert!(!body.contains("vade-"));
    let doc = json::parse(&body).unwrap();
    let main = member(&doc, &["roles", "main"]);
    let sub = member(&doc, &["roles", "subordinate"]);
    assert_eq!(integer(member(main, &["sum_u"])), 2106);
    assert_eq!(integer(member(main, &["sum_min_c_u"])), 2106); // min(C,U), not C
    assert_eq!(real(member(main, &["cache_ratio"])), 1.0);
    assert_eq!(integer(member(sub, &["sum_u"])), 1530);
    assert_eq!(integer(member(sub, &["sum_min_c_u"])), 760);
    assert!((real(member(sub, &["cache_ratio"])) - 760.0 / 1530.0).abs() < 1e-10);
    assert_eq!(member(main, &["u_estimated"]), &json::Value::Bool(true));
    assert_eq!(
        member(main, &["exact_target_verified"]),
        &json::Value::Bool(false)
    );
    assert_eq!(integer(member(main, &["admission_based_samples"])), 2);
    assert_eq!(integer(member(sub, &["ttft_samples"])), 10);
    assert_eq!(
        integer(member(&doc, &["all_turn_requests", "generations"])),
        18
    );
    assert_eq!(
        integer(member(&doc, &["aggregate", "requests", "generations"])),
        12
    );
    let aggregate = real(member(&doc, &["aggregate", "output_tok_s"]));
    let sessions = member(&doc, &["sessions"]).as_array().unwrap();
    for session in sessions {
        let rate = real(member(session, &["summary", "output_tok_s"]));
        assert!(
            aggregate > rate * 2.0,
            "aggregate must use overlapping wall time"
        );
        let turns = member(session, &["turns"]).as_array().unwrap();
        assert_eq!(member(&turns[0], &["u"]), &json::Value::Null);
        assert_eq!(member(&turns[0], &["measured"]), &json::Value::Bool(false));
        assert!(real(member(&turns[1], &["ttft_ms"])) >= 9.0);
    }
    println!("{}", report.table());
}

#[test]
fn workload_missing_telemetry_and_json_fallback_cannot_claim_a_pass_or_ttft() {
    for json_response in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            assert_eq!(request(&mut reader).0, "GET /v1/capabilities HTTP/1.1");
            respond(&mut reader, 404, "application/json", "{}");
            let mut prefix = String::new();
            for turn in 1..=3 {
                let (line, body) = request(&mut reader);
                assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
                assert!(body.starts_with(&prefix));
                prefix = body[..body.len() - 2].into();
                let v = json::view(&body).unwrap();
                assert!(v.get("prompt_cache_key").is_none());
                assert_eq!(
                    v.get("stream_options")
                        .unwrap()
                        .get("include_usage")
                        .unwrap()
                        .raw(),
                    "true"
                );
                let telemetry = if turn == 2 {
                    "".into()
                } else {
                    usage(1000, 1000)
                };
                if json_response {
                    respond(
                        &mut reader,
                        200,
                        "application/json",
                        r#"{"choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}"#,
                    );
                } else {
                    respond(
                        &mut reader,
                        200,
                        "text/event-stream",
                        &format!("{ANSWER}{telemetry}{DONE}"),
                    );
                }
            }
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 3)).unwrap();
        server.join().unwrap();
        assert!(report.successful());
        let doc = json::parse(&report.to_json().unwrap()).unwrap();
        let main = member(&doc, &["roles", "main"]);
        assert_eq!(member(main, &["cache_ratio"]), &json::Value::Null);
        assert_eq!(member(main, &["output_tok_s"]), &json::Value::Null);
        assert_eq!(
            member(main, &["target_status"]).as_str(),
            Some("insufficient_data")
        );
        assert_eq!(
            integer(member(&doc, &["all_turn_requests", "admissions"])),
            0
        );
        if json_response {
            assert_eq!(member(main, &["ttft_p50_ms"]), &json::Value::Null);
        }
        assert!(report.sessions[0].samples[1].u_estimated);
        assert_eq!(report.sessions[0].samples[1].u_basis, "wire_bytes_estimate");
    }
}

#[test]
fn workload_admission_refusal_and_http_overload_are_counted_without_replay() {
    // Status failures at both admission and generation; context-fit refusal is
    // separate from an HTTP 429/503, and each occurs after one successful turn.
    for (stage, status) in [
        ("admission", 200),
        ("admission", 429),
        ("admission", 503),
        ("generation", 429),
        ("generation", 503),
    ] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(100, 0)),
            );
            rendered_history_admission(&mut reader, 103);
            capabilities(&mut reader, CAPS);
            if stage == "admission" && status == 200 {
                admission(&mut reader, 4000, false);
            } else if stage == "admission" {
                assert_eq!(request(&mut reader).0, "POST /v1/admission HTTP/1.1");
                respond(
                    &mut reader,
                    status,
                    "application/json",
                    "{\"private\":\"must not log body\"}",
                );
            } else {
                let body = admission(&mut reader, 200, true);
                assert_eq!(request(&mut reader).1, body);
                respond(
                    &mut reader,
                    status,
                    "application/json",
                    "{\"private\":\"must not log body\"}",
                );
            }
            closed_without_replay(&mut reader, &listener);
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 3)).unwrap();
        server.join().unwrap();
        assert!(!report.successful());
        assert_eq!(report.sessions[0].samples.len(), 2);
        let raw = report.to_json().unwrap();
        assert!(!raw.contains("must not log body"));
        let doc = json::parse(&raw).unwrap();
        let main = member(&doc, &["roles", "main"]);
        assert_eq!(integer(member(main, &["errors"])), 1);
        assert_eq!(integer(member(main, &["unattempted_samples"])), 1);
        assert_eq!(integer(member(main, &["requests", "admissions"])), 1);
        assert_eq!(
            integer(member(main, &["requests", "refused"])),
            u64::from(status == 200)
        );
        assert_eq!(
            integer(member(main, &["requests", "http_429"])),
            u64::from(status == 429)
        );
        assert_eq!(
            integer(member(main, &["requests", "http_503"])),
            u64::from(status == 503)
        );
        assert_eq!(member(main, &["cache_ratio"]), &json::Value::Null);
        let sample = &report.sessions[0].samples[1];
        assert_eq!(sample.last_http_stage, stage);
        assert_eq!(sample.last_http_status, Some(status));
    }
}

#[test]
fn workload_ttft_includes_reasoning_but_not_role_event_or_final_usage() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        for turn in 1..=2 {
            if turn > 1 {
                rendered_history_admission(&mut reader, (turn - 1) * 100 + 3);
            }
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, turn * 100, true);
            assert_eq!(request(&mut reader).1, body);
            let role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
            let reasoning =
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"considering\"}}]}\n\n";
            let tail = format!("{ANSWER}{}{DONE}", usage(turn * 100, 100));
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{role}", role.len() + reasoning.len() + tail.len()).unwrap();
            thread::sleep(Duration::from_millis(40));
            reader.get_mut().write_all(reasoning.as_bytes()).unwrap();
            thread::sleep(Duration::from_millis(100));
            reader.get_mut().write_all(tail.as_bytes()).unwrap();
        }
    });
    let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
    server.join().unwrap();
    assert!(report.successful());
    let sample = &report.sessions[0].samples[1];
    let ttft = sample.ttft_ms.unwrap();
    assert!(ttft >= 30.0, "role event must not count as a token: {ttft}");
    assert!(
        (sample.end_s - sample.start_s) * 1000.0 - ttft >= 70.0,
        "must measure reasoning before answer text"
    );
}

#[test]
fn workload_reasoning_only_replies_round_trip_eight_turns_with_exact_prefixes() {
    // One streaming session and one JSON fallback session, both length-limited
    // like the reported native failure. Validate the history at admission and
    // require those exact same bytes at generation, on the original socket.
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut workers = Vec::new();
        for streaming in [true, false] {
            let mut reader = accept(&listener);
            workers.push(thread::spawn(move || {
                let mut prefix = String::new();
                for turn in 1..=8 {
                    if turn > 1 { rendered_history_admission(&mut reader, (turn - 1) * 100 + 3); }
                    capabilities(&mut reader, CAPS);
                    let body = admission(&mut reader, turn * 100, true);
                    assert!(body.starts_with(&prefix), "old request bytes changed on turn {turn}");
                    prefix = body[..body.len() - 2].into();
                    let request_value = json::parse(&body).unwrap();
                    let messages = request_value.get("messages").unwrap().as_array().unwrap();
                    let mut reasoning_turns = 0;
                    for message in messages {
                        if message.get("role").and_then(json::Value::as_str) != Some("assistant") {
                            continue;
                        }
                        let has_calls = message.get("tool_calls").and_then(json::Value::as_array).is_some_and(|calls| !calls.is_empty());
                        // Mirrors the documented Engine shape constraint only;
                        // no engine implementation, tokenizer, or code is used.
                        assert!(has_calls || matches!(message.get("content"), Some(json::Value::String(_) | json::Value::Array(_))), "assistant lacks legal content on turn {turn}");
                        if let Some(reasoning) = message.get("reasoning_content").and_then(json::Value::as_str) {
                            reasoning_turns += 1;
                            assert_eq!(message.get("content").and_then(json::Value::as_str), Some(""));
                            assert_eq!(reasoning, format!("thinking {reasoning_turns}\n 🦀"));
                            assert_eq!(message.get("provider").unwrap().get("opaque"), Some(&json::Value::Bool(true)));
                        }
                    }
                    assert_eq!(reasoning_turns, turn - 1);
                    assert_eq!(request(&mut reader), ("POST /v1/chat/completions HTTP/1.1".into(), body));
                    let content = if turn % 2 == 0 { ",\"content\":null" } else { "" };
                    if streaming {
                        let role = format!("data: {{\"choices\":[{{\"delta\":{{\"role\":\"assistant\"{content}}}}}]}}\n\n");
                        let first = format!("data: {{\"choices\":[{{\"delta\":{{\"reasoning_content\":\"thinking {turn}\"}}}}]}}\n\n");
                        let last = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"\\n 🦀\",\"provider\":{\"opaque\":true}},\"finish_reason\":\"length\"}]}\n\n";
                        respond(&mut reader, 200, "text/event-stream", &format!("{role}{first}{last}{}{DONE}", usage(turn * 100, 0)));
                    } else {
                        respond(&mut reader, 200, "application/json", &format!(r#"{{"choices":[{{"message":{{"role":"assistant"{content},"reasoning_content":"thinking {turn}\n 🦀","provider":{{"opaque":true}}}},"finish_reason":"length"}}],"usage":{{"prompt_tokens":{},"completion_tokens":3}}}}"#, turn * 100));
                    }
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    let report = vade_harness::workload::run(workload_config(&endpoint, 2, 8)).unwrap();
    server.join().unwrap();
    assert!(report.successful());
    for session in &report.sessions {
        assert_eq!(session.samples.len(), 8);
        assert!(session.stopped.is_none());
        for sample in &session.samples {
            assert_eq!(sample.outcome, "length");
            assert!(sample.error.is_none());
            assert!(sample.prefix_preserved);
            assert_eq!(sample.requests.admissions, 1);
            assert_eq!(sample.requests.generations, 1);
        }
    }
}

#[test]
fn workload_cli_writes_private_json_and_one_table_without_clobbering() {
    use std::process::Command;
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        for turn in 1..=2 {
            if turn > 1 {
                rendered_history_admission(&mut reader, (turn - 1) * 100 + 3);
            }
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, turn * 100, true);
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!(
                    "{ANSWER}{}{DONE}",
                    usage(turn * 100, if turn == 1 { 0 } else { 103 })
                ),
            );
        }
    });
    let file = std::env::temp_dir().join(format!(
        "vade-workload-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let args = [
        "workload",
        "--endpoint",
        &endpoint,
        "--model",
        "mock",
        "--sessions",
        "1",
        "--turns",
        "2",
        "--output-tokens",
        "17",
        "--json",
        file.to_str().unwrap(),
    ];
    let output = Command::new(env!("CARGO_BIN_EXE_vade-harness"))
        .args(args)
        .output()
        .unwrap();
    server.join().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.matches("Scope             Done/plan").count(), 1);
    assert!(stdout.contains("~1.0000"));
    assert!(!stdout.contains("Synthetic observation"));
    let original = std::fs::read_to_string(&file).unwrap();
    json::parse(&original).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    // No server remains. Existing-file rejection must occur before any network.
    let rejected = Command::new(env!("CARGO_BIN_EXE_vade-harness"))
        .args(args)
        .output()
        .unwrap();
    assert_eq!(rejected.status.code(), Some(2));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
    std::fs::remove_file(file).unwrap();
}

#[test]
fn workload_deadline_cancels_and_joins_a_pending_native_read() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        let body = admission(&mut reader, 100, true);
        assert_eq!(request(&mut reader).1, body);
        // Leave generation pending. The workload owns and must close this socket.
        assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    });
    let mut config = workload_config(&endpoint, 1, 3);
    config.max_seconds = 1;
    let start = Instant::now();
    let report = vade_harness::workload::run(config).unwrap();
    server.join().unwrap();
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(report.timed_out);
    assert!(!report.successful());
    assert_eq!(report.sessions[0].samples.len(), 1);
    assert_eq!(report.sessions[0].samples[0].error, Some("cancelled"));
    assert_eq!(report.sessions[0].samples[0].requests.generations, 1);
    assert_eq!(report.sessions[0].samples[0].usage.cached_tokens, None);
}

#[test]
fn workload_truncation_and_unrequested_tools_never_become_success_or_real_execution() {
    for tool in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, body);
            if tool {
                respond(
                    &mut reader,
                    200,
                    "application/json",
                    r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"unexpected","type":"function","function":{"name":"workspace_read","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
                );
                closed_without_replay(&mut reader, &listener);
            } else {
                let body = format!("{ANSWER}{}{DONE}", usage(100, 100));
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len() + 5).unwrap();
            }
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 3)).unwrap();
        server.join().unwrap();
        assert!(!report.successful());
        assert_eq!(report.sessions[0].samples.len(), 1);
        let sample = &report.sessions[0].samples[0];
        assert_eq!(sample.usage.cached_tokens, None);
        assert!(sample.ttft_ms.is_none());
        assert_eq!(
            sample.error,
            Some(if tool {
                "native_runtime_limit"
            } else {
                "incomplete_response"
            })
        );
    }
}

#[test]
fn workload_length_finish_is_measured_and_zero_cache_is_distinct_from_zero_denominator() {
    for zero_prompt in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            for turn in 1..=2 {
                if turn > 1 {
                    rendered_history_admission(&mut reader, if zero_prompt { 0 } else { 100 });
                }
                capabilities(&mut reader, CAPS);
                let prompt = if zero_prompt { 0 } else { 100 };
                let body = admission(&mut reader, prompt, true);
                assert_eq!(request(&mut reader).1, body);
                let answer = ANSWER.replace("stop", "length");
                let telemetry =
                    usage(prompt, 0).replace("\"completion_tokens\":3", "\"completion_tokens\":0");
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{answer}{telemetry}{DONE}"),
                );
            }
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
        server.join().unwrap();
        assert!(report.successful());
        assert_eq!(report.sessions[0].samples[1].outcome, "length");
        assert_eq!(report.sessions[0].samples[1].usage.cached_tokens, Some(0));
        let doc = json::parse(&report.to_json().unwrap()).unwrap();
        let main = member(&doc, &["roles", "main"]);
        if zero_prompt {
            assert_eq!(member(main, &["cache_ratio"]), &json::Value::Null);
            assert_eq!(
                member(main, &["target_status"]).as_str(),
                Some("insufficient_data")
            );
        } else {
            assert_eq!(real(member(main, &["cache_ratio"])), 0.0);
            assert_eq!(
                member(main, &["target_status"]).as_str(),
                Some("estimate_below_target")
            );
        }
    }
}

#[test]
fn workload_contract_identity_change_and_inconsistent_counts_are_visible() {
    for mismatch in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            for turn in 1..=2 {
                let caps = if turn == 2 && !mismatch {
                    CAPS.replace("mock-template", "mock-template-v2").replacen(
                        "\"object\"",
                        "\"build_hash\":\"build-v2\",\"object\"",
                        1,
                    )
                } else {
                    CAPS.into()
                };
                if turn > 1 {
                    rendered_history_admission(&mut reader, (turn - 1) * 100 + 3);
                }
                capabilities(&mut reader, &caps);
                let body = admission(&mut reader, 100 * turn, true);
                assert_eq!(request(&mut reader).1, body);
                let prompt = if turn == 2 && mismatch {
                    201
                } else {
                    100 * turn
                };
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{ANSWER}{}{DONE}", usage(prompt, 100)),
                );
            }
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
        server.join().unwrap();
        let sample = &report.sessions[0].samples[1];
        if mismatch {
            assert_eq!(sample.error, Some("admission_usage_mismatch"));
            assert!(!report.successful());
        } else {
            assert_eq!(sample.cache_domain_changes, 1);
            assert_eq!(sample.u, None);
            assert_eq!(sample.u_basis, "unavailable_cache_identity_change");
        }
        let doc = json::parse(&report.to_json().unwrap()).unwrap();
        assert_eq!(
            member(&doc, &["roles", "main", "cache_ratio"]),
            &json::Value::Null
        );
    }
}

#[test]
fn workload_counts_full_rendered_history_instead_of_generated_output_and_prints_each_turn() {
    for rendered in [173, 400] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let first = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, first);
            let assistant = r#"{"role":"assistant","content":"answer","reasoning_content":"retained reasoning","provider":{"opaque":true}}"#;
            // Output count absent: exact rendering must not use a byte fallback.
            respond(
                &mut reader,
                200,
                "application/json",
                &format!(
                    r#"{{"choices":[{{"message":{assistant},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":100,"prompt_tokens_details":{{"cached_tokens":0}}}}}}"#
                ),
            );
            let closed = rendered_history_admission(&mut reader, rendered);
            let v = json::view(&closed).unwrap();
            let messages: Vec<_> = v.get("messages").unwrap().array().unwrap().collect();
            assert_eq!(messages.last().unwrap().raw(), assistant);
            assert_eq!(messages.len(), 5);
            capabilities(&mut reader, CAPS);
            let next = admission(&mut reader, 300, true);
            assert!(next.starts_with(first.strip_suffix("]}").unwrap()));
            assert_eq!(request(&mut reader).1, next);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(300, 160)),
            );
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
        server.join().unwrap();
        assert!(report.successful());
        let sample = &report.sessions[0].samples[1];
        assert_eq!(sample.u, Some(rendered.min(300)));
        assert_eq!(sample.rendered_previous_tokens, Some(rendered));
        assert_eq!(sample.u_basis, "exact_rendered_history_counts");
        assert!(sample.u_estimated);
        assert_eq!(sample.requests.admissions, 1);
        assert_eq!(sample.requests.generations, 1);
        assert_eq!(sample.prefix_count_requests.admissions, 1);
        assert_eq!(sample.prefix_count_requests.generations, 0);
        assert!(matches!(
            sample.input,
            InputCount::Exact {
                prompt_tokens: 300,
                ..
            }
        ));
        let table = report.table();
        assert!(table.contains("Per-turn cache"));
        assert!(
            table.contains("first_turn / stop")
                && table.contains("exact_rendered_history_counts / stop")
        );
        assert!(table.contains(if rendered == 173 { "92.49" } else { "53.33" }));
        assert!(!table.contains("retained reasoning"));
        let doc = json::parse(&report.to_json().unwrap()).unwrap();
        assert_eq!(
            integer(member(&doc, &["prefix_count_requests", "admissions"])),
            1
        );
        assert_eq!(
            integer(member(&doc, &["all_turn_requests", "generations"])),
            2
        );
    }
}

#[test]
fn workload_failed_render_count_leaves_u_unknown_and_counts_the_probe_without_replay() {
    for mode in 0..3 {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let first = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, first);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(100, 0)),
            );
            let (line, envelope) = request(&mut reader);
            assert_eq!(line, "POST /v1/admission HTTP/1.1");
            assert_eq!(
                json::view(&envelope)
                    .unwrap()
                    .get("request")
                    .unwrap()
                    .get("messages")
                    .unwrap()
                    .array()
                    .unwrap()
                    .last()
                    .unwrap()
                    .get("role")
                    .unwrap()
                    .as_str()
                    .as_deref(),
                Some("assistant")
            );
            match mode {
                0 => respond(
                    &mut reader,
                    429,
                    "application/json",
                    "{\"private\":\"count failure sentinel\"}",
                ),
                1 => respond(
                    &mut reader,
                    200,
                    "application/json",
                    &admission_body(4000, 17, false),
                ),
                _ => respond(
                    &mut reader,
                    200,
                    "application/json",
                    &admission_body(173, 17, true)
                        .replace("\"output_tokens\":17", "\"output_tokens\":16"),
                ),
            }
            // HTTP/invalid metadata drops that observation's socket; the next
            // generation opens another, without replaying the count request.
            if mode != 1 {
                reader = accept(&listener);
            }
            capabilities(&mut reader, CAPS);
            let next = admission(&mut reader, 300, true);
            assert_eq!(request(&mut reader).1, next);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(300, 160)),
            );
        });
        let report = vade_harness::workload::run(workload_config(&endpoint, 1, 2)).unwrap();
        server.join().unwrap();
        assert!(
            report.successful(),
            "generations complete even if measurement fails"
        );
        let sample = &report.sessions[0].samples[1];
        assert_eq!(sample.u, None);
        assert_eq!(sample.u_basis, "unavailable_rendered_history_count_failed");
        assert_eq!(sample.prefix_count_requests.admissions, 1);
        assert_eq!(sample.prefix_count_requests.http_429, u64::from(mode == 0));
        assert_eq!(sample.prefix_count_requests.refused, u64::from(mode == 1));
        let raw = report.to_json().unwrap();
        assert!(!raw.contains("count failure sentinel"));
        let doc = json::parse(&raw).unwrap();
        assert_eq!(
            member(&doc, &["roles", "main", "cache_ratio"]),
            &json::Value::Null
        );
    }
}
