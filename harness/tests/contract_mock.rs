//! First-party loopback fixtures only. No Engine binary, model, delegate or GPU.
//! Contract negotiation, compaction, native prompt resets and response timing.
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/contract.rs"]
mod contract;
use contract::*;
#[cfg(target_os = "macos")]
#[path = "support/children_contract.rs"]
mod children_contract;

#[test]
fn metadata_transport_break_retries_once_without_replaying_generation() {
    for admission_break in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            if admission_break { capabilities(&mut reader, CAPS); }
            let lost = request(&mut reader);
            if admission_break {
                assert_eq!(lost.0, "POST /v1/admission HTTP/1.1");
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{{").unwrap();
            } else { assert_eq!(lost.0, "GET /v1/capabilities HTTP/1.1"); }
            drop(reader);
            let mut reader = accept(&listener);
            let retried = request(&mut reader);
            assert_eq!(retried, lost, "retry must retain exact metadata request bytes");
            if admission_break {
                let v = json::view(&retried.1).unwrap();
                let output = v.get("output_tokens").unwrap().raw().parse().unwrap();
                respond(&mut reader, 200, "application/json", &admission_body(100, output, true));
            } else {
                respond(&mut reader, 200, "application/json", CAPS);
                admission(&mut reader, 100, true);
            }
            assert_eq!(request(&mut reader).0, "POST /v1/chat/completions HTTP/1.1");
            // A generation failure must propagate, without a second generation.
            drop(reader);
        });
        let mut model = NativeModel::new(&endpoint).unwrap();
        let c = native_conversation();
        let cancel = AtomicBool::new(false);
        let count = model.count_input(c.request(Limits::default().wire), &cancel);
        assert!(count.is_ok(), "read-only metadata recovery failed: {count:?}");
        assert!(model.complete(c.request(Limits::default().wire), &cancel, &mut |_| Ok(())).is_err());
        assert_eq!(model.context().requests.generations, 1);
        assert_eq!(model.context().requests.capabilities, if admission_break { 1 } else { 2 });
        assert_eq!(model.context().requests.admissions, if admission_break { 2 } else { 1 });
        server.join().unwrap();
    }
}

#[test]
fn cutoff_only_recovery_requires_a_tool_and_releases_policy_after_progress() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut generations = 0;
        while generations < 4 {
            let (line, body) = request(&mut reader);
            match line.as_str() {
                "GET /v1/capabilities HTTP/1.1" => respond(&mut reader, 200, "application/json", CAPS),
                "POST /v1/admission HTTP/1.1" => {
                    let v = json::view(&body).unwrap();
                    let output = v.get("output_tokens").unwrap().raw().parse().unwrap();
                    respond(&mut reader, 200, "application/json", &admission_body(100, output, true));
                }
                "POST /v1/chat/completions HTTP/1.1" => {
                    let v = json::view(&body).unwrap();
                    assert_eq!(v.get("tool_choice").and_then(|v| v.as_str()).as_deref(), (generations == 2).then_some("required"));
                    assert_eq!(v.get("max_tokens").unwrap().raw(), "17");
                    let reply = match generations {
                        0 | 1 => r#"{"choices":[{"message":{"role":"assistant","reasoning_content":"still planning","content":""},"finish_reason":"length"}]}"#,
                        2 => r#"{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"recovered","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
                        _ => r#"{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}"#,
                    };
                    respond(&mut reader, 200, "application/json", reply);
                    generations += 1;
                }
                _ => panic!("unexpected request {line}"),
            }
        }
    });
    let mut c = native_conversation();
    let mut model = NativeModel::new(&endpoint).unwrap();
    let limits = Limits { max_model_turns: 4, ..Limits::default() };
    let report = agent::run(&mut c, &mut model, &mut MockTools, limits, &AtomicBool::new(false), &mut |_| {});
    assert_eq!((report.reason, report.model_turns, report.tool_calls), (StopReason::Finished(FinishReason::Stop), 4, 1));
    assert_eq!(c.messages().iter().filter(|m| m.role() == "tool").count(), 1);
    server.join().unwrap();
}

#[test]
fn oversized_prompt_is_counted_before_save_and_resume_accepts_next_prompt() {
    use std::process::Command;
    let workspace = WorkspaceFixture::new();
    let (listener, endpoint) = listen();
    let oversized = "oversize-marker-".to_owned() + &"x".repeat(32_000 - 16);
    assert_eq!(oversized.len(), 32_000);
    let server = thread::spawn(move || {
        let caps = CAPS.replace("4096", "8192").replace("4000", "8160");
        let mut generations = 0;
        for _ in 0..2 {
            let mut reader = accept(&listener);
            while !reader.fill_buf().unwrap().is_empty() {
                let (line, body) = request(&mut reader);
                match line.as_str() {
                    "GET /v1/capabilities HTTP/1.1" => {
                        respond(&mut reader, 200, "application/json", &caps);
                    }
                    "POST /v1/admission HTTP/1.1" => {
                        let large = body.contains("oversize-marker-");
                        let reply = admission_body(if large { 10_000 } else { 100 }, 32, !large)
                            .replace("4096", "8192")
                            .replace("4000", "8160");
                        respond(&mut reader, 200, "application/json", &reply);
                    }
                    "POST /v1/chat/completions HTTP/1.1" => {
                        assert!(!body.contains("oversize-marker-"));
                        generations += 1;
                        respond(
                            &mut reader,
                            200,
                            "text/event-stream",
                            &format!("{ANSWER}{DONE}"),
                        );
                    }
                    _ => panic!("unexpected mock request"),
                }
            }
        }
        generations
    });
    let launch = |resume: bool, prompt: &str| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        command
            .args([
                "agent",
                "--state-in-project",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--workspace",
            ])
            .arg(&workspace.0)
            .args([
                "--plain",
                "--verbose",
                "--approval",
                "never",
                "--context-tokens",
                "8192",
                if resume { "--resume" } else { "--session" },
                "oversize",
                "--prompt",
                prompt,
            ]);
        cli::Cli::start(&mut command).finish()
    };
    let rejected = launch(false, &oversized);
    let raw = std::fs::read(workspace.0.join(".vade/sessions/oversize.journal")).unwrap();
    let clean = !raw.windows(16).any(|w| w == b"oversize-marker-");
    let next = launch(true, "tiny follow-up");
    let generations = server.join().unwrap();
    eprintln!(
        "oversize reproduction: journal_clean={clean}; next_success={}; generations={generations}",
        next.status.success()
    );
    assert!(clean, "oversized prompt was persisted before admission");
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("it was not sent"));
    assert!(
        next.status.success(),
        "{}",
        String::from_utf8_lossy(&next.stderr)
    );
    assert_eq!(generations, 1);
}

#[test]
fn presave_exact_count_accepts_a_prompt_that_the_local_estimate_would_veto() {
    use std::process::Command;
    let workspace = WorkspaceFixture::new();
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        let body = admission(&mut reader, 100, true);
        assert!(body.contains(&"x".repeat(32_000)));
        assert_eq!(
            request(&mut reader),
            ("POST /v1/chat/completions HTTP/1.1".into(), body)
        );
        respond(
            &mut reader,
            200,
            "text/event-stream",
            &format!("{ANSWER}{DONE}"),
        );
    });
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
    command
        .args([
            "agent",
            "--state-in-project",
            "--endpoint",
            &endpoint,
            "--model",
            "mock",
            "--workspace",
        ])
        .arg(&workspace.0)
        .args([
            "--plain",
            "--approval",
            "never",
            "--context-tokens",
            "8192",
            "--prompt",
            &"x".repeat(32_000),
        ]);
    let output = cli::Cli::start(&mut command).finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    server.join().unwrap();
}

#[test]
fn cached_native_refusal_cannot_be_used_to_send_generation() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        admission(&mut reader, 4090, false);
        closed_without_replay(&mut reader, &listener);
    });
    let mut model = NativeModel::new(&endpoint).unwrap();
    let c = native_conversation();
    let cancel = AtomicBool::new(false);
    for _ in 0..2 {
        assert!(matches!(
            model
                .count_input(c.request(Limits::default().wire), &cancel)
                .unwrap(),
            InputCount::Exact {
                admitted: false,
                ..
            }
        ));
    }
    assert!(
        model
            .complete(c.request(Limits::default().wire), &cancel, &mut |_| Ok(()))
            .is_err()
    );
    assert_eq!(model.context().requests.generations, 0);
    drop(model);
    server.join().unwrap();
}

// The returned slice comes from the HTTP body read off the socket, including
// every byte of options/tools and the complete leading developer record.
fn socket_head(body: &str) -> &str {
    let v = json::view(body).unwrap();
    let developer = v.get("messages").unwrap().array().unwrap().nth(1).unwrap();
    assert_eq!(
        developer.get("role").unwrap().as_str().as_deref(),
        Some("developer")
    );
    let raw = developer.raw();
    let end = raw.as_ptr() as usize - body.as_ptr() as usize + raw.len();
    &body[..end]
}

#[test]
fn socket_captured_generation_head_is_identical_after_compaction() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut bodies = Vec::new();
        for _ in 0..2 {
            capabilities(&mut reader, CAPS);
            let admitted = admission(&mut reader, 100, true);
            let (line, captured) = request(&mut reader);
            assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
            assert_eq!(captured, admitted);
            bodies.push(captured);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{DONE}"),
            );
        }
        assert_eq!(
            socket_head(&bodies[0]).as_bytes(),
            socket_head(&bodies[1]).as_bytes()
        );
        let messages: Vec<_> = json::view(&bodies[1])
            .unwrap()
            .get("messages")
            .unwrap()
            .array()
            .unwrap()
            .collect();
        assert_eq!(
            messages[2].get("name").unwrap().as_str().as_deref(),
            Some("vade_context_summary")
        );
        assert!(bodies[1].len() < bodies[0].len());
    });
    let request = format!(
        r#"{{"model":"mock","tools":[{{"type":"function","function":{{"name":"read","parameters":{{"type":"object"}}}}}}],"opaque":{{"order":[3,1,2]}},"messages":[{{"role":"system","content":"stable 🦀\\n rules"}},{{"role":"developer","content":" scoped rules  \\n keep spacing","opaque":[2,1]}},{{"role":"user","content":"old constraint"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"recent constraint"}},{{"role":"assistant","content":"recent answer"}},{{"role":"user","content":"next task"}}]}}"#,
        "discardable old results ".repeat(100)
    );
    let original =
        Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
    let mut model = NativeModel::new(&endpoint)
        .unwrap()
        .with_compaction_prewarm(true);
    let cancel = AtomicBool::new(false);
    model
        .count_input(original.request(Limits::default().wire), &cancel)
        .unwrap();
    model
        .complete(
            original.request(Limits::default().wire),
            &cancel,
            &mut |_| Ok(()),
        )
        .unwrap();
    let compacted =
        vade_harness::context::compact(&original, "Resolved older work.", 1, Limits::default())
            .unwrap();
    model.prompt_rewritten(); // automatic path: no admission/decode for a warm
    model
        .count_input(compacted.request(Limits::default().wire), &cancel)
        .unwrap();
    model
        .warm_compacted(compacted.request(Limits::default().wire), &cancel)
        .unwrap();
    model
        .complete(
            compacted.request(Limits::default().wire),
            &cancel,
            &mut |_| Ok(()),
        )
        .unwrap();
    assert!(model.context().compaction_warm.is_none());
    assert_eq!(model.context().requests.generations, 2);
    server.join().unwrap();
}

#[test]
fn native_capability_snapshot_survives_compaction_and_reconciles_one_new_build() {
    let initial = CAPS.replacen("\"object\"", "\"build_hash\":\"build-a\",\"object\"", 1);
    assert_native_capability_reconcile(initial, "build-a", "build-b");
}

#[test]
fn native_capability_snapshot_reconciles_cache_domain_once_without_build_hash() {
    assert_native_capability_reconcile(CAPS.into(), "mock-load-epoch", "mock-load-epoch-2");
}

#[test]
fn native_capability_snapshot_reconciles_each_fingerprint_once_without_build_hash() {
    for fingerprint in ["mock-weights", "mock-tokenizer", "mock-template"] {
        assert_native_capability_reconcile(CAPS.into(), fingerprint, &format!("{fingerprint}-2"));
    }
}

fn assert_native_capability_reconcile(initial: String, old_epoch: &str, new_epoch: &str) {
    let (listener, endpoint) = listen();
    let drift = initial
        .replace("\"default_thinking\": false", "\"default_thinking\": true")
        .replace(
            "\"prompt_cache_key\": \"scheduling_hint\"",
            "\"prompt_cache_key\": null",
        );
    let next = drift.replace(old_epoch, new_epoch);
    let later_drift = initial.replace(old_epoch, new_epoch);
    let expected_initial = Capabilities::parse(&initial).unwrap().unwrap();
    let expected_next = Capabilities::parse(&next).unwrap().unwrap();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut bodies = Vec::new();
        for caps in [initial, drift, next, later_drift] {
            capabilities(&mut reader, &caps);
            admission(&mut reader, 100, true);
            let (line, body) = request(&mut reader);
            assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
            bodies.push(body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{DONE}"),
            );
        }
        assert_eq!(socket_head(&bodies[0]), socket_head(&bodies[1]));
        assert_ne!(socket_head(&bodies[1]), socket_head(&bodies[2]));
        assert_eq!(socket_head(&bodies[2]), socket_head(&bodies[3]));
    });
    let mut c = Conversation::from_request(json::parse(r#"{"model":"mock","messages":[{"role":"system","content":"stable"},{"role":"developer","content":"scoped rules"},{"role":"user","content":"first"}]}"#).unwrap(), Limits::default()).unwrap();
    let mut model = NativeModel::new(&endpoint).unwrap();
    let cancel = AtomicBool::new(false);
    for turn in 0..4 {
        if turn == 1 {
            model.reset_prompt();
        }
        if turn > 0 {
            c.push_user(
                json::parse(&format!(r#"{{"role":"user","content":"turn {turn}"}}"#)).unwrap(),
                Limits::default(),
            )
            .unwrap();
        }
        model
            .count_input(c.request(Limits::default().wire), &cancel)
            .unwrap();
        model
            .complete(c.request(Limits::default().wire), &cancel, &mut |_| Ok(()))
            .unwrap();
        assert_eq!(
            model.context().capability_reconciliations,
            u64::from(turn >= 2)
        );
        assert_eq!(model.context().cache_domain_changes, u64::from(turn >= 2));
        assert_eq!(
            model.context().capabilities.as_ref(),
            Some(if turn < 2 {
                &expected_initial
            } else {
                &expected_next
            })
        );
    }
    assert!(
        model
            .context()
            .describe()
            .contains("Engine build changed; capabilities reconciled")
    );
    server.join().unwrap();
}

#[test]
fn manual_compaction_counts_before_save_and_warm_failure_allows_real_followup() {
    use std::{process::Command, sync::Arc};
    // Oversized candidate, non-shrinking exact count, and successful compaction
    // followed by an HTTP-failed opt-in warm request.
    for scenario in 0..3 {
        let workspace = WorkspaceFixture::new();
        let (listener, endpoint) = listen();
        let done = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&done);
        let server = thread::spawn(move || {
            listener.set_nonblocking(true).unwrap();
            let mut generations = 0;
            let mut failed_warms = 0;
            while !stopped.load(Ordering::Relaxed) {
                let socket = match listener.accept() {
                    Ok((socket, _)) => socket,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(socket);
                while !reader.fill_buf().unwrap().is_empty() {
                    let (line, body) = request(&mut reader);
                    match line.as_str() {
                        "GET /v1/capabilities HTTP/1.1" => {
                            respond(&mut reader, 200, "application/json", CAPS)
                        }
                        "POST /v1/admission HTTP/1.1" => {
                            let envelope = json::view(&body).unwrap();
                            let summary = envelope
                                .get("request")
                                .unwrap()
                                .get("messages")
                                .unwrap()
                                .raw()
                                .contains("vade_context_summary");
                            let output = envelope
                                .get("output_tokens")
                                .unwrap()
                                .raw()
                                .parse()
                                .unwrap();
                            let tokens = if summary {
                                [4090, 512, 100][scenario]
                            } else {
                                512
                            };
                            respond(
                                &mut reader,
                                200,
                                "application/json",
                                &admission_body(tokens, output, !(summary && scenario == 0)),
                            );
                        }
                        "POST /v1/chat/completions HTTP/1.1" => {
                            let v = json::view(&body).unwrap();
                            if v.get("max_tokens").unwrap().raw() == "1" {
                                failed_warms += 1;
                                respond(
                                    &mut reader,
                                    503,
                                    "application/json",
                                    r#"{"error":"mock warm failure"}"#,
                                );
                            } else {
                                generations += 1;
                                let text = if generations <= 2 {
                                    "discardable old prose ".repeat(150)
                                } else {
                                    "latest answer".into()
                                };
                                let event = format!(
                                    "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\"}}]}}\n\n{DONE}",
                                    json::quote(&text)
                                );
                                respond(&mut reader, 200, "text/event-stream", &event);
                            }
                        }
                        _ => panic!("unexpected mock route"),
                    }
                }
            }
            (generations, failed_warms)
        });
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
        command
            .args([
                "agent",
                "--state-in-project",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--workspace",
            ])
            .arg(&workspace.0)
            .args([
                "--plain",
                "--verbose",
                "--approval",
                "never",
                "--session",
                "manual",
                "--prewarm-compaction",
            ]);
        let mut child = cli::Cli::start(&mut command);
        for prompt in ["first", "second", "latest"] {
            child.send(&format!("{prompt}\n"));
            child.until(true, "Native run:");
        }
        let path = workspace.0.join(".vade/sessions/manual.journal");
        let before = std::fs::read(&path).unwrap();
        child.send("/compact 1 Resolved old work.\n");
        if scenario < 2 {
            child.until(false, "Compaction refused:");
            assert_eq!(
                std::fs::read(&path).unwrap(),
                before,
                "refused plan must not write a journal record"
            );
        } else {
            child.until(true, "optional pre-warm failed or cancelled");
            assert_ne!(std::fs::read(&path).unwrap(), before);
        }
        child.send("tiny follow-up\n");
        child.until(true, "Native run:");
        child.send("/quit\n");
        let output = child.finish();
        done.store(true, Ordering::Relaxed);
        assert_eq!(server.join().unwrap(), (4, usize::from(scenario == 2)));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn explicit_native_prompt_reset_preserves_session_affinity_and_usage_counters() {
    native_prompt_reset(false);
    native_prompt_reset(true);
}

fn native_prompt_reset(prewarm: bool) {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut first_key = String::new();
        for turn in 0..2 {
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 117, true);
            let v = json::view(&body).unwrap();
            let key = v
                .get("prompt_cache_key")
                .unwrap()
                .as_str()
                .unwrap()
                .into_owned();
            if turn == 0 {
                first_key = key;
            } else {
                assert_eq!(first_key, key);
            }
            if turn == 1 && prewarm {
                let warm = admission(&mut reader, 117, true);
                let w = json::view(&warm).unwrap();
                assert_eq!(w.get("max_tokens").unwrap().raw(), "1");
                for field in ["tools", "messages", "prompt_cache_key"] {
                    assert_eq!(w.get(field).unwrap().raw(), v.get(field).unwrap().raw());
                }
                assert_eq!(request(&mut reader).1, warm);
                respond(
                    &mut reader,
                    200,
                    "application/json",
                    r#"{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"discarded warm"}}],"usage":{"prompt_tokens":117,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":100}}}"#,
                );
            }
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(117, 0)),
            );
        }
    });
    let mut model = NativeModel::new(&endpoint)
        .unwrap()
        .with_compaction_prewarm(prewarm);
    let first = native_conversation();
    assert!(matches!(
        model
            .count_input(
                first.request(Limits::default().wire),
                &AtomicBool::new(false)
            )
            .unwrap(),
        InputCount::Exact {
            prompt_tokens: 117,
            ..
        }
    ));
    model
        .complete(
            first.request(Limits::default().wire),
            &AtomicBool::new(false),
            &mut |_| Ok(()),
        )
        .unwrap();
    let requests = model.context().requests;
    assert!(model.context().response_streamed);
    assert!(model.context().first_token_elapsed.is_some());
    assert_eq!(model.context().last_http_status, Some(200));
    assert_eq!(model.context().last_http_stage, "generation");
    model.reset_prompt(); // owner-declared compaction; no change to session identity
    assert_eq!(model.context().requests, requests);
    assert_eq!(model.context().completed_responses, 1);
    assert!(!model.context().response_streamed);
    assert!(model.context().first_token_elapsed.is_none());
    assert_eq!(model.context().last_http_status, None);
    assert_eq!(model.context().last_http_stage, "none");
    assert!(model.context().turns.is_empty());
    assert!(model.context().prefix_pending);
    let changed = Conversation::from_request(
        json::parse(
            &first
                .request(Limits::default().wire)
                .to_json()
                .unwrap()
                .replace("hello 🦀", "explicitly rebuilt context"),
        )
        .unwrap(),
        Limits::default(),
    )
    .unwrap();
    model
        .count_input(
            changed.request(Limits::default().wire),
            &AtomicBool::new(false),
        )
        .unwrap();
    model
        .warm_compacted(
            changed.request(Limits::default().wire),
            &AtomicBool::new(false),
        )
        .unwrap();
    model
        .complete(
            changed.request(Limits::default().wire),
            &AtomicBool::new(false),
            &mut |_| Ok(()),
        )
        .unwrap();
    assert_eq!(model.context().completed_responses, 2);
    assert_eq!(model.context().compaction_warm.is_some(), prewarm);
    if prewarm {
        model.reset_prompt();
        let bad = Conversation::from_request(
            json::parse(
                &first
                    .request(Limits::default().wire)
                    .to_json()
                    .unwrap()
                    .replace(r#""role":"system""#, r#""role":"user""#),
            )
            .unwrap(),
            Limits::default(),
        )
        .unwrap();
        assert!(
            model
                .warm_compacted(bad.request(Limits::default().wire), &AtomicBool::new(false))
                .is_err()
        );
        assert_eq!(model.context().completed_responses, 2); // rejected before warm I/O
    }
    server.join().unwrap();
}

#[test]
fn foreign_html_discovery_and_custom_routes_keep_standard_chat_behavior() {
    for custom in [false, true] {
        let (listener, mut endpoint) = listen();
        if custom {
            endpoint.push_str("/custom/rpc");
        }
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            if !custom {
                assert_eq!(request(&mut reader).0, "GET /v1/capabilities HTTP/1.1");
                respond(
                    &mut reader,
                    200,
                    "text/html",
                    "<html>foreign service</html>",
                );
            }
            let (line, body) = request(&mut reader);
            assert_eq!(
                line,
                if custom {
                    "POST /custom/rpc HTTP/1.1"
                } else {
                    "POST /v1/chat/completions HTTP/1.1"
                }
            );
            assert!(json::view(&body).unwrap().get("prompt_cache_key").is_none());
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(100, 40)),
            );
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        client.stream("mock", &[], &mut |_| {}, None).unwrap();
        assert!(matches!(client.context().input, InputCount::Estimate(_)));
        assert_eq!(
            client.context().turns.back().unwrap().usage.cached_tokens,
            Some(40)
        );
        server.join().unwrap();
    }
}

#[test]
fn response_timing_in_native_and_direct_excludes_role_and_usage_trailer_waits() {
    for native in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            admission(&mut reader, 100, true);
            request(&mut reader);
            let role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
            let first = "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n";
            let tail = format!("{}{DONE}", usage(100, 0));
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{role}", role.len() + first.len() + ANSWER.len() + tail.len()).unwrap();
            thread::sleep(Duration::from_millis(40));
            reader.get_mut().write_all(first.as_bytes()).unwrap();
            thread::sleep(Duration::from_millis(80));
            reader.get_mut().write_all(ANSWER.as_bytes()).unwrap();
            thread::sleep(Duration::from_millis(160));
            reader.get_mut().write_all(tail.as_bytes()).unwrap();
        });
        let context = if native {
            let mut model = NativeModel::new(&endpoint).unwrap();
            let conversation = Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"messages":[{"role":"user","content":"timing"}]}"#).unwrap(), Limits::default()).unwrap();
            model
                .complete(
                    conversation.request(Limits::default().wire),
                    &AtomicBool::new(false),
                    &mut |_| Ok(()),
                )
                .unwrap();
            model.context().clone()
        } else {
            let mut client = ChatClient::new(&endpoint).unwrap();
            client
                .stream(
                    "mock",
                    &[Message {
                        role: "user".into(),
                        content: "timing".into(),
                    }],
                    &mut |_| {},
                    None,
                )
                .unwrap();
            client.context().clone()
        };
        server.join().unwrap();
        let turn = context.turns.back().unwrap();
        assert!(turn.first_token_elapsed.unwrap() >= Duration::from_millis(30));
        let decode = turn.decode_elapsed.unwrap();
        assert!(
            decode >= Duration::from_millis(60) && decode < Duration::from_millis(180),
            "{decode:?}"
        );
        assert!(turn.speed().contains("prompt 100 tokens · cached 0 tokens"));
    }
}

#[test]
fn native_queue_waits_beyond_thirty_seconds_before_headers_or_output() {
    // Exercise real socket polls and the production timeout policy. These three
    // fixtures run concurrently so the regression adds about 31s, not 93s.
    thread::scope(|scope| {
        for kind in ["headers", "role", "json"] {
            scope.spawn(move || {
                let (listener, endpoint) = listen();
                let server = thread::spawn(move || {
                    let mut reader = accept(&listener);
                    for turn in 1..=2 {
                        if turn > 1 { rendered_history_admission(&mut reader, (turn - 1) * 100 + 3); }
                    capabilities(&mut reader, CAPS);
                        let body = admission(&mut reader, turn * 100, true);
                        assert_eq!(request(&mut reader).1, body);
                        let role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
                        let answer = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"still thinking\"},\"finish_reason\":\"length\"}]}\n\n";
                        let tail = format!("{answer}{}{DONE}", usage(turn * 100, 0));
                        let json_body = format!(r#"{{"choices":[{{"message":{{"role":"assistant","reasoning_content":"still thinking"}},"finish_reason":"length"}}],"usage":{{"prompt_tokens":{},"completion_tokens":3}}}}"#, turn * 100);
                        if kind == "role" {
                            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{role}", role.len() + tail.len()).unwrap();
                        } else if kind == "json" {
                            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", json_body.len()).unwrap();
                        }
                        if turn == 1 { thread::sleep(Duration::from_secs(31)); }
                        match kind {
                            "role" => reader.get_mut().write_all(tail.as_bytes()).unwrap(),
                            "json" => reader.get_mut().write_all(json_body.as_bytes()).unwrap(),
                            _ => respond(&mut reader, 200, "text/event-stream", &tail),
                        }
                    }
                    // Two requests on this socket; no timeout reconnect/replay.
                });
                let mut config = workload_config(&endpoint, 1, 2);
                config.max_seconds = 60;
                let report = vade_harness::workload::run(config).unwrap();
                server.join().unwrap();
                assert!(report.successful(), "{kind}: {report:?}");
                assert_eq!(report.sessions[0].samples.len(), 2);
                let first = &report.sessions[0].samples[0];
                assert!(first.end_s - first.start_s >= 30.0);
                if kind != "json" { assert!(first.ttft_ms.unwrap() >= 30_000.0); }
            });
        }
    });
}

#[test]
fn model_progress_waits_for_generated_reasoning_and_clears_on_cancel_or_completion() {
    use std::sync::Arc;
    use vade_harness::transport::NativePhase;
    for (native, cancel_after_reasoning) in
        [(true, false), (true, true), (false, false), (false, true)]
    {
        let (listener, endpoint) = listen();
        let (role_tx, role_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 100, true);
            assert_eq!(request(&mut reader).1, body);
            let role = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
            let reasoning = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\",\"content\":\"\\n\\n\"}}]}\n\n";
            let done = format!(
                "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"length\"}}]}}\n\n{DONE}"
            );
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{role}", role.len() + reasoning.len() + done.len()).unwrap();
            role_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
            reader.get_mut().write_all(reasoning.as_bytes()).unwrap();
            if !cancel_after_reasoning {
                reader.get_mut().write_all(done.as_bytes()).unwrap();
            }
            closed_without_replay(&mut reader, &listener);
        });
        let (phase_tx, phase_rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let worker = thread::spawn(move || {
            let observer = Arc::new(move |phase| phase_tx.send(phase).unwrap());
            if native {
                let mut model = NativeModel::new(&endpoint).unwrap();
                model.set_progress_observer(Some(observer));
                model
                    .complete(
                        native_conversation().request(Limits::default().wire),
                        &worker_cancel,
                        &mut |_| panic!("reasoning must not be shown as answer text"),
                    )
                    .map(|turn| turn.finish_reason() == &FinishReason::Length)
            } else {
                let mut client = ChatClient::new(&endpoint).unwrap();
                client.set_progress_observer(Some(observer));
                client
                    .stream(
                        "mock",
                        &[],
                        &mut |_| panic!("reasoning must not be shown as answer text"),
                        Some(&worker_cancel),
                    )
                    .map(|turn| turn.finish_reason.as_deref() == Some("length"))
            }
        });
        let phase = || phase_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        assert_eq!(phase(), NativePhase::Preparing);
        assert_eq!(phase(), NativePhase::Queued);
        role_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        // A role event alone must not move the UI to generating.
        assert!(phase_rx.recv_timeout(Duration::from_millis(50)).is_err());
        release_tx.send(()).unwrap();
        assert_eq!(phase(), NativePhase::Thinking);
        if cancel_after_reasoning {
            cancel.store(true, Ordering::Relaxed);
        }
        assert_eq!(phase(), NativePhase::Idle);
        let result = worker.join().unwrap();
        if cancel_after_reasoning {
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
        } else {
            assert!(result.unwrap());
        }
        server.join().unwrap();
    }
}
