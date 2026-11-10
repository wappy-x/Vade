//! Exact admission against first-party loopback fixtures: affinity, cache counts,
//! denied or malformed admission, contract changes and transport failures.
//! No Engine binary, model, delegate or GPU.
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/contract.rs"]
mod contract;
use contract::*;

#[test]
fn direct_exact_request_admission_stable_affinity_and_actual_cache_counts() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut prefix = String::new();
        let mut key = String::new();
        for (prompt, cached) in [(117, 0), (146, 109)] {
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, prompt, true);
            assert!(body.starts_with(&prefix));
            prefix = body[..body.len() - 2].to_owned();
            let v = json::view(&body).unwrap();
            let next = v
                .get("prompt_cache_key")
                .unwrap()
                .as_str()
                .unwrap()
                .into_owned();
            if key.is_empty() {
                key = next;
            } else {
                assert_eq!(key, next);
            }
            assert_eq!(
                request(&mut reader),
                ("POST /v1/chat/completions HTTP/1.1".into(), body)
            );
            // Duplicated final snapshots must count only once.
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{ANSWER}{}{DONE}", usage(prompt, cached).repeat(2)),
            );
        }
    });
    let mut client = ChatClient::new(&endpoint).unwrap();
    let mut messages = vec![Message {
        role: "user".into(),
        content: " exact\n🦀".into(),
    }];
    for (prompt, cached) in [(117, 0), (146, 109)] {
        let result = client.stream("mock", &messages, &mut |_| {}, None).unwrap();
        let ctx = client.context();
        assert_eq!(
            ctx.input,
            InputCount::Exact {
                prompt_tokens: prompt,
                output_tokens: 32,
                context_limit: 4096,
                admitted: true
            }
        );
        assert_eq!(ctx.turns.back().unwrap().usage.cached_tokens, Some(cached));
        assert!(ctx.describe().contains(&format!("cached tokens {cached}")));
        messages.push(Message {
            role: "assistant".into(),
            content: result.text,
        });
        messages.push(Message {
            role: "user".into(),
            content: "next".into(),
        });
    }
    assert_eq!(client.context().reported_cached_tokens, 109);
    assert_eq!(client.context().responses_with_cached_usage, 2);
    server.join().unwrap();
}

#[test]
fn native_tool_loop_uses_complete_admission_and_keeps_decorated_prefix() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut prefix = String::new();
        for turn in 0..2 {
            if turn == 1 {
                capabilities(&mut reader, CAPS);
                let staged = admission(&mut reader, 130, true);
                assert!(staged.contains("not_executed"));
            }
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 120 + turn * 20, true);
            assert!(body.starts_with(&prefix));
            prefix = body[..body.len() - 2].into();
            assert_eq!(
                request(&mut reader),
                ("POST /v1/chat/completions HTTP/1.1".into(), body.clone())
            );
            let text = if turn == 0 {
                r#"{"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":"opaque reasoning","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":120,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":0}}}"#.to_owned()
            } else {
                let parsed = json::parse(&body).unwrap();
                let last = parsed
                    .get("messages")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap();
                assert_eq!(last.get("tool_call_id").unwrap().as_str(), Some("call-1"));
                assert_eq!(
                    last.get("content").unwrap().as_str(),
                    Some(" exact result\n🦀 ")
                );
                r#"{"choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":140,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":112}}}"#.to_owned()
            };
            respond(&mut reader, 200, "application/json", &text);
        }
    });
    let mut model = NativeModel::new(&endpoint).unwrap();
    let report = agent::run(
        &mut native_conversation(),
        &mut model,
        &mut MockTools,
        Limits::default(),
        &AtomicBool::new(false),
        &mut |_| {},
    );
    assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
    assert_eq!(report.tool_calls, 1);
    assert_eq!(model.context().turns.len(), 2);
    assert_eq!(model.context().turns[1].usage.cached_tokens, Some(112));
    server.join().unwrap();
}

#[test]
fn absent_and_unknown_contracts_degrade_without_custom_fields_or_fake_hits() {
    for status in [404, 405, 501, 200] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            assert_eq!(request(&mut reader).0, "GET /v1/capabilities HTTP/1.1");
            respond(
                &mut reader,
                status,
                "application/json",
                &CAPS.replace("vade.engine.v1", "vade.engine.future"),
            );
            for _ in 0..2 {
                let (line, body) = request(&mut reader);
                assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
                let value = json::view(&body).unwrap();
                assert!(value.get("prompt_cache_key").is_none());
                assert_eq!(value.get("max_tokens").unwrap().raw(), "8192");
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{ANSWER}{DONE}"),
                );
            }
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        for _ in 0..2 {
            client.stream("mock", &[], &mut |_| {}, None).unwrap();
            assert!(matches!(client.context().input, InputCount::Estimate(_)));
            assert_eq!(
                client.context().turns.back().unwrap().usage.cached_tokens,
                None
            );
            assert!(
                client
                    .context()
                    .describe()
                    .contains("cached tokens unknown")
            );
        }
        server.join().unwrap();
    }
}

#[test]
fn denied_exact_admission_never_sends_generation_or_replays() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        admission(&mut reader, 4090, false);
        closed_without_replay(&mut reader, &listener);
    });
    let mut client = ChatClient::new(&endpoint).unwrap();
    assert!(client.stream("mock", &[], &mut |_| {}, None).is_err());
    assert_eq!(
        client.context().input,
        InputCount::Exact {
            prompt_tokens: 4090,
            output_tokens: 32,
            context_limit: 4096,
            admitted: false
        }
    );
    assert!(client.context().describe().contains("nothing was sent"));
    drop(client);
    server.join().unwrap();
}

#[test]
fn malformed_admission_is_not_bypassed_with_estimates() {
    for response in [
        admission_body(4090, 32, true), // claims fit despite prompt + output overflow
        admission_body(100, 31, true),  // counted a different output reservation
        admission_body(100, 32, true).replace("vade.engine.v1", "future"),
        admission_body(100, 32, true).replace("\"context_limit\":4096", "\"context_limit\":8192"),
        admission_body(100, 32, true).replace("\"model\":\"mock\"", "\"model\":\"other\""),
        admission_body(100, 32, true).replace(
            "\"reserves_resources\":false",
            "\"reserves_resources\":true",
        ),
        admission_body(100, 32, true).replace("\"prompt_tokens\":100", "\"prompt_tokens\":1e2"),
    ] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            assert_eq!(request(&mut reader).0, "POST /v1/admission HTTP/1.1");
            respond(&mut reader, 200, "application/json", &response);
            closed_without_replay(&mut reader, &listener);
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        assert!(client.stream("mock", &[], &mut |_| {}, None).is_err());
        assert!(!matches!(client.context().input, InputCount::Estimate(_)));
        assert_eq!(client.context().completed_responses, 0);
        drop(client);
        server.join().unwrap();
    }
}

#[test]
fn unadvertised_tools_and_wrong_model_or_dialect_fail_before_dispatch() {
    for caps in [
        CAPS.replace("\"tools\": true", "\"tools\": null"),
        CAPS.replace("\"model\": \"mock\"", "\"model\": \"other\""),
        CAPS.replace("\"/v1/chat/completions\", ", ""),
    ] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, &caps);
            closed_without_replay(&mut reader, &listener);
        });
        let mut model = NativeModel::new(&endpoint).unwrap();
        assert!(
            model
                .complete(
                    native_conversation().request(Limits::default().wire),
                    &AtomicBool::new(false),
                    &mut |_| Ok(())
                )
                .is_err()
        );
        drop(model);
        server.join().unwrap();
    }
}

#[test]
fn restart_and_fingerprint_change_are_observable_and_affinity_is_not_a_hit() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut key = String::new();
        for turn in 0..3 {
            let caps = if turn == 0 {
                CAPS.into()
            } else {
                CAPS.replace("mock-load-epoch", "mock-restart")
                    .replace("mock-template", "mock-template-2")
            };
            capabilities(&mut reader, &caps);
            let body = admission(&mut reader, 130, true);
            let next = json::view(&body)
                .unwrap()
                .get("prompt_cache_key")
                .unwrap()
                .as_str()
                .unwrap()
                .into_owned();
            if turn == 0 {
                key = next;
            } else if turn == 1 {
                assert_eq!(
                    key, next,
                    "session affinity survives restart but says nothing about reuse"
                );
            } else {
                assert_ne!(key, next, "explicit new context gets a fresh hint");
            }
            assert_eq!(request(&mut reader).1, body);
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!(
                    "{ANSWER}{}{DONE}",
                    usage(130, if turn == 0 { 100 } else { 0 })
                ),
            );
        }
    });
    let mut client = ChatClient::new(&endpoint).unwrap();
    client.stream("mock", &[], &mut |_| {}, None).unwrap();
    client.stream("mock", &[], &mut |_| {}, None).unwrap();
    assert_eq!(client.context().cache_domain_changes, 1);
    assert_eq!(
        client.context().turns.back().unwrap().usage.cached_tokens,
        Some(0)
    );
    client.reset_context();
    client.stream("mock", &[], &mut |_| {}, None).unwrap();
    assert_eq!(client.context().completed_responses, 1);
    server.join().unwrap();
}

#[test]
fn established_contract_cannot_disappear_or_downgrade_silently() {
    for next in [
        CAPS.replace("vade.engine.v1", "future"),
        CAPS.replace("\"admission\": \"/v1/admission\"", "\"admission\": null"),
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
                &format!("{ANSWER}{DONE}"),
            );
            capabilities(&mut reader, &next);
            closed_without_replay(&mut reader, &listener);
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        client.stream("mock", &[], &mut |_| {}, None).unwrap();
        assert!(client.stream("mock", &[], &mut |_| {}, None).is_err());
        assert!(client.context().describe().contains("/new"));
        drop(client);
        server.join().unwrap();
    }
}

#[test]
fn cancellation_during_admission_drops_socket_without_generation() {
    let (listener, endpoint) = listen();
    let cancel = AtomicBool::new(false);
    let (ready, waiting) = mpsc::channel();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, CAPS);
        assert_eq!(request(&mut reader).0, "POST /v1/admission HTTP/1.1");
        ready.send(()).unwrap();
        closed_without_replay(&mut reader, &listener);
    });
    thread::scope(|scope| {
        let flag = &cancel;
        scope.spawn(move || {
            waiting.recv_timeout(Duration::from_secs(3)).unwrap();
            flag.store(true, Ordering::Relaxed);
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        let error = client
            .stream("mock", &[], &mut |_| {}, Some(&cancel))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
        assert_eq!(client.context().completed_responses, 0);
    });
    server.join().unwrap();
}

#[test]
fn advertised_admission_is_optional_and_unknown_features_do_not_enable_it() {
    let caps = CAPS.replace(
        "\"admission\": \"/v1/admission\"",
        "\"admission\": \"https://unsupported.invalid/count\"",
    );
    assert!(!Capabilities::parse(&caps).unwrap().unwrap().exact_admission);
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        capabilities(&mut reader, &caps);
        let (line, body) = request(&mut reader);
        assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
        assert_eq!(json::view(&body).unwrap().get("max_tokens").unwrap().raw(), "32");
        respond(
            &mut reader,
            200,
            "text/event-stream",
            &format!("{ANSWER}{DONE}"),
        );
    });
    let mut client = ChatClient::new(&endpoint).unwrap();
    client.stream("mock", &[], &mut |_| {}, None).unwrap();
    assert!(matches!(client.context().input, InputCount::Estimate(_)));
    server.join().unwrap();
}

#[test]
fn truncated_usage_stream_does_not_publish_cache_hits() {
    for native in [false, true] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 117, true);
            assert_eq!(request(&mut reader).1, body);
            let data = format!("{ANSWER}{}{DONE}", usage(117, 87));
            // Even a valid usage event and [DONE] cannot hide lost HTTP framing.
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{data}", data.len() + 1).unwrap();
        });
        if native {
            let mut model = NativeModel::new(&endpoint).unwrap();
            assert!(
                model
                    .complete(
                        native_conversation().request(Limits::default().wire),
                        &AtomicBool::new(false),
                        &mut |_| Ok(())
                    )
                    .is_err()
            );
            assert_eq!(model.context().reported_cached_tokens, 0);
            assert!(model.context().turns.is_empty());
        } else {
            let mut client = ChatClient::new(&endpoint).unwrap();
            assert!(client.stream("mock", &[], &mut |_| {}, None).is_err());
            assert_eq!(client.context().reported_cached_tokens, 0);
            assert!(client.context().turns.is_empty());
        }
        server.join().unwrap();
    }
}

#[test]
fn transport_failures_and_http_errors_during_admission_never_dispatch_generation() {
    for status in [0, 400, 404, 503] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            assert_eq!(request(&mut reader).0, "POST /v1/admission HTTP/1.1");
            if status != 0 {
                respond(
                    &mut reader,
                    status,
                    "application/json",
                    r#"{"error":{"message":"mock sensitive upstream detail"}}"#,
                );
                closed_without_replay(&mut reader, &listener);
            } else {
                drop(reader);
                let mut retry = accept(&listener);
                assert_eq!(request(&mut retry).0, "POST /v1/admission HTTP/1.1");
                // The second broken transport is final; generation stays unsent.
            }
            listener
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        let error = client.stream("mock", &[], &mut |_| {}, None).unwrap_err();
        assert!(!error.to_string().contains("sensitive"));
        assert!(client.context().describe().contains("nothing was sent"));
        drop(client);
        let listener = server.join().unwrap();
        listener.set_nonblocking(true).unwrap();
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
}
