    use super::*;
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn endpoint_rejects_tls_secrets_and_injection() {
        assert_eq!(
            Endpoint::parse("https://127.0.0.1:1").unwrap_err().kind(),
            io::ErrorKind::Unsupported
        );
        for endpoint in [
            "http://@localhost",
            "http://localhost/\r\nX: y",
            "http://localhost:0",
            "http://localhost#x",
            "http://::1",
        ] {
            assert!(Endpoint::parse(endpoint).is_err());
        }
        assert_eq!(
            Endpoint::parse("http://[::1]:8080/v1?x=1").unwrap().path,
            "/v1/chat/completions?x=1"
        );
        let response_endpoint =
            Endpoint::parse("http://127.0.0.1:8080/proxy/v1/responses?x=1").unwrap();
        assert_eq!(
            response_endpoint
                .gateway_route("/chat/completions", Some("y=2"))
                .unwrap()
                .path,
            "/proxy/v1/chat/completions?x=1&y=2"
        );
        assert_eq!(
            response_endpoint
                .gateway_route("/responses", None)
                .unwrap()
                .path,
            "/proxy/v1/responses?x=1"
        );
    }

    #[test]
    fn completions_routes_preserve_prefixes_queries_and_sibling_api_names() {
        for source in ["/v1", "/v1/chat/completions", "/v1/responses", "/v1/messages"] {
            let endpoint = Endpoint::parse(&format!("http://localhost:8080/proxy{source}?configured=1")).unwrap();
            for suffix in ["/completions", "/chat/completions", "/responses", "/messages", "/capabilities", "/models", "/admission"] {
                assert_eq!(endpoint.gateway_route(suffix, Some("client=2")).unwrap().path,
                    format!("/proxy/v1{suffix}?configured=1&client=2"), "{source} -> {suffix}");
            }
        }
        for source in ["/completions", "/v1/completions"] {
            let endpoint = Endpoint::parse(&format!("http://localhost:8080{source}?configured=1")).unwrap();
            let base = source.strip_suffix("/completions").unwrap();
            for suffix in ["/completions", "/chat/completions", "/responses"] {
                assert_eq!(endpoint.gateway_route(suffix, Some("client=2")).unwrap().path,
                    format!("{base}{suffix}?configured=1&client=2"));
            }
        }
        for exact in ["/custom", "/custom/completions", "/proxy/v1/completions"] {
            let endpoint = Endpoint::parse(&format!("http://localhost:8080{exact}?configured=1")).unwrap();
            for suffix in ["/completions", "/chat/completions", "/responses"] {
                assert_eq!(endpoint.gateway_route(suffix, Some("client=2")).unwrap().path,
                    format!("{exact}?configured=1&client=2"));
            }
        }
    }

    #[test]
    fn sse_every_byte_utf8_crlf_and_multiline() {
        let input = "\u{feff}:comment\r\ndata: {\r\ndata: \"choices\": [{\"delta\": {\"content\": \"é🦀\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n";
        let mut decoder = SseDecoder::default();
        let mut completion = Completion::default();
        let mut chunks = Vec::new();
        for byte in input.as_bytes() {
            decoder
                .feed(&[*byte], &mut |data| {
                    if data == "[DONE]" {
                        return Ok(true);
                    }
                    protocol::apply_event(&json::parse(data)?, &mut completion, &mut |text| {
                        chunks.push(text.to_owned())
                    })?;
                    Ok(false)
                })
                .unwrap();
        }
        assert!(decoder.done);
        assert_eq!(completion.text, "é🦀");
        assert_eq!(chunks, ["é🦀"]);
    }

    #[test]
    fn framing_rejects_smuggling_truncation_and_limits() {
        for bytes in [
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n".as_slice(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n",
            b"HTTP/1.1 200 OK\n\n",
        ] {
            assert!(read_head(&mut Cursor::new(bytes)).is_err());
        }
        let head = Head {
            first: String::new(),
            fields: vec![("transfer-encoding".into(), "chunked".into())],
        };
        let mut body = BodyReader::new(
            Cursor::new(b"3;x=y\r\nabc\r\n0\r\nX-Trailer: ignored\r\n\r\n"),
            &head,
            false,
            3,
        )
        .unwrap();
        let mut out = Vec::new();
        body.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"abc");
        let mut truncated = BodyReader::new(Cursor::new(b"3\r\na"), &head, false, 3).unwrap();
        assert!(truncated.read_to_end(&mut Vec::new()).is_err());
        let mut large =
            BodyReader::new(Cursor::new(b"4\r\nabcd\r\n0\r\n\r\n"), &head, false, 3).unwrap();
        assert!(large.read_to_end(&mut Vec::new()).is_err());
    }

    #[test]
    fn native_reply_budget_keeps_token_floor_near_task_deadline() {
        let mut contract = crate::contract::Contract::default();
        assert_eq!(contract.native_reply_tokens(Some(std::time::Duration::ZERO)), 512);
        assert_eq!(contract.native_reply_time(Some(std::time::Duration::ZERO)), Some(std::time::Duration::ZERO));
        assert_eq!(contract.native_reply_tokens(Some(std::time::Duration::from_secs(60))), 512);
        // 30 s floor: a late reply can still finish a small patch, without
        // eating the 10 s reserve kept after it (35 s left -> 25 s reply).
        assert_eq!(contract.native_reply_time(Some(std::time::Duration::from_secs(60))), Some(std::time::Duration::from_secs(30)));
        assert_eq!(contract.native_reply_time(Some(std::time::Duration::from_secs(35))), Some(std::time::Duration::from_secs(25)));
        assert_eq!(contract.native_reply_time(Some(std::time::Duration::from_secs(2400))), Some(std::time::Duration::from_secs(120)));
        assert_eq!(crate::contract::guaranteed_native_reply_tokens(), 2048);
        let mut usage = crate::contract::UsageCollector::default();
        usage.observe_data(r#"{"usage":{"completion_tokens":201,"vade":{"decode_ms":10000}}}"#).unwrap();
        contract.completed(usage);
        // Measured 20 tok/s over the floored 30 s reply window.
        assert_eq!(contract.native_reply_tokens(Some(std::time::Duration::from_secs(60))), 600);
        contract.options.reply_tokens = Some(37);
        assert_eq!(contract.native_reply_tokens(Some(std::time::Duration::ZERO)), 37);
    }

    #[test]
    fn advertised_guidance_and_reply_floor_match_the_native_cap() {
        use crate::agent::{Conversation, Limits};
        let cap = crate::contract::guaranteed_native_reply_tokens();
        assert_eq!(cap, crate::contract::Contract::default().native_reply_tokens(None));
        assert!((crate::contract::NATIVE_REPLY_MIN_TOKENS..=crate::contract::NATIVE_REPLY_MAX_TOKENS).contains(&cap));
        assert!(crate::coding_tools::write_guidance(cap).contains("~40 lines"));
        let tools = json::stringify(&crate::coding_tools::definitions()).unwrap();
        assert!(tools.contains("~40 lines") && !tools.contains("~200 lines"));
        let c = Conversation::from_request(json::parse(r#"{"messages":[{"role":"user","content":"x"}]}"#).unwrap(), Limits::default()).unwrap();
        let exact = crate::context::Usage::for_conversation(&c).with_count(&crate::contract::InputCount::Exact { prompt_tokens: 100, output_tokens: 4096, context_limit: 32768, admitted: true }).unwrap();
        assert!(!exact.fits(32768, 8192));
        assert!(exact.fits_with_reply_cap(32768, 8192, crate::contract::NATIVE_REPLY_MAX_TOKENS));
    }

    #[test]
    fn reply_cap_excludes_reasoning_and_rolls_back_entire_tool_groups() {
        use crate::agent::{Conversation, Limits, Model};
        let mut budget = crate::contract::Contract::default();
        assert_eq!(budget.native_reply_tokens(None), 2048);
        let mut usage = crate::contract::UsageCollector::default();
        usage.observe(json::view(r#"{"usage":{"completion_tokens":101,"vade":{"decode_ms":10000}}}"#).unwrap());
        budget.completed(usage);
        assert_eq!(budget.native_reply_tokens(None), 1200);
        budget.options.reply_tokens = Some(37);
        assert_eq!(budget.native_reply_tokens(None), 37);
        for (tools, disabled, prose, size) in [(false, false, false, 15000), (true, false, false, 15000), (true, true, false, 15000), (false, false, true, 15000), (true, false, false, 9000)] {
            let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let server = thread::spawn(move || {
                let (socket, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
                {
                    let head = legacy_head(&mut reader);
                    let mut body = vec![0; head.get("content-length").unwrap_or("0").parse::<usize>().unwrap()];
                    reader.read_exact(&mut body).unwrap();
                    let first = if tools { format!(r#"{{"tool_calls":[{{"index":0,"id":"whole","type":"function","function":{{"name":"read","arguments":{}}}}}]}}"#, json::quote(&format!("{{\"path\":\"{}", "x".repeat(size)))) } else if prose { format!(r#"{{"content":{}}}"#, json::quote(&"x".repeat(size))) } else { format!(r#"{{"reasoning_content":{}}}"#, json::quote(&"plan ".repeat(4000))) };
                    let last = if tools { r#"{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}"# } else { r#"{"content":"done"}"# };
                    let sse = format!("data: {{\"choices\":[{{\"delta\":{first},\"finish_reason\":null}}]}}\n\ndata: {{\"choices\":[{{\"delta\":{last},\"finish_reason\":\"{}\"}}]}}\n\ndata: [DONE]\n\n", if tools { "tool_calls" } else { "stop" });
                    write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}", sse.len()).unwrap();
                }
            });
            let c = Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"messages":[{"role":"user","content":"build"}]}"#).unwrap(), Limits::default()).unwrap();
            let mut model = NativeModel::new(&endpoint).unwrap().with_generation_options(crate::contract::GenerationOptions { reply_tokens: disabled.then_some(0), ..Default::default() });
            let turn = model.complete(c.request(Limits::default().wire), &AtomicBool::new(false), &mut |_| Ok(())).unwrap();
            let exhausted = (tools && size >= 12288 || prose) && !disabled;
            assert_eq!(turn.reply_budget_exhausted(), exhausted);
            assert_eq!(turn.finish_reason(), &if exhausted { crate::agent_wire::FinishReason::Length } else if tools { crate::agent_wire::FinishReason::ToolCalls } else { crate::agent_wire::FinishReason::Stop });
            if tools && exhausted { assert!(turn.truncated_tool_call()); assert!(turn.message().tool_calls().is_empty()); assert!(turn.discarded_bytes() > 15000); }
            if tools && !exhausted { assert_eq!(turn.message().tool_calls()[0].arguments, format!("{{\"path\":\"{}\"}}", "x".repeat(size))); }
            if prose { assert_eq!(turn.message().value().get("content").and_then(json::Value::as_str), Some("x".repeat(size).as_str())); }
            server.join().unwrap();
        }
    }

    #[test]
    fn reply_deadline_keeps_prose_and_reasoning_but_rolls_back_partial_tool() {
        use crate::agent::{Conversation, Limits, Model};
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (release, hold) = std::sync::mpsc::channel::<()>();
        let server = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
            let head = legacy_head(&mut reader);
            let mut body = vec![0; head.get("content-length").unwrap_or("0").parse::<usize>().unwrap()];
            reader.read_exact(&mut body).unwrap();
            let event = r#"{"choices":[{"delta":{"content":"Working on it.","reasoning_content":"Check the file first.","tool_calls":[{"index":0,"id":"partial","type":"function","function":{"name":"read","arguments":"{\"path\":\""}}]},"finish_reason":null}]}"#;
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {event}\n\ndata: {{").unwrap();
            // Keep the stream open until the client's absolute deadline expires.
            let _ = hold.recv_timeout(Duration::from_secs(10));
        });
        let conversation = Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"messages":[{"role":"user","content":"build"}]}"#).unwrap(), Limits::default()).unwrap();
        let mut model = NativeModel::new(&endpoint).unwrap();
        model.set_work_remaining(Some(Duration::from_secs(8)));
        let mut emitted = String::new();
        let result = model.complete(conversation.request(Limits::default().wire), &AtomicBool::new(false), &mut |text| { emitted.push_str(text); Ok(()) });
        release.send(()).unwrap();
        server.join().unwrap();
        let turn = result.unwrap();
        assert!(turn.reply_budget_exhausted());
        assert!(turn.truncated_tool_call());
        assert_eq!(turn.finish_reason(), &crate::agent_wire::FinishReason::Length);
        assert!(turn.message().tool_calls().is_empty());
        assert_eq!(emitted, "Working on it.");
        assert_eq!(turn.message().value().get("content").and_then(json::Value::as_str), Some(emitted.as_str()));
        assert_eq!(turn.message().value().get("reasoning_content").and_then(json::Value::as_str), Some("Check the file first."));
    }

    #[test]
    fn reply_deadline_before_first_token_returns_timeout() {
        use crate::agent::{Conversation, Limits, Model};
        for prelude in ["", "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n", "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"] {
            let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let (release, hold) = std::sync::mpsc::channel::<()>();
            let server = thread::spawn(move || {
                let (socket, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
                let head = legacy_head(&mut reader);
                let mut body = vec![0; head.get("content-length").unwrap_or("0").parse::<usize>().unwrap()];
                reader.read_exact(&mut body).unwrap();
                reader.get_mut().write_all(prelude.as_bytes()).unwrap();
                let _ = hold.recv_timeout(Duration::from_secs(10));
            });
            let conversation = Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"messages":[{"role":"user","content":"build"}]}"#).unwrap(), Limits::default()).unwrap();
            let mut model = NativeModel::new(&endpoint).unwrap();
            model.set_work_remaining(Some(Duration::from_secs(8)));
            let error = model.complete(conversation.request(Limits::default().wire), &AtomicBool::new(false), &mut |_| panic!("no tokens")).unwrap_err();
            release.send(()).unwrap(); server.join().unwrap();
            assert_eq!(error.kind(), io::ErrorKind::TimedOut);
            assert!(error.to_string().contains("before the first generated token"));
        }
    }

    #[test]
    fn mock_http_fragmented_chunked_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
            let head = legacy_head(&mut reader);
            assert_eq!(head.first, "POST /v1/chat/completions HTTP/1.1");
            let mut body = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
            let mut bytes = Vec::new();
            body.read_to_end(&mut bytes).unwrap();
            let value = json::parse_bytes(&bytes).unwrap();
            assert_eq!(value.get("model").unwrap().as_str(), Some("mock"));
            assert_eq!(
                value.get("messages").unwrap().as_array().unwrap()[0]
                    .get("content")
                    .unwrap()
                    .as_str(),
                Some("prefix\n")
            );
            let socket = body.reader.get_mut();
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
            let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hé🦀\"},\"finish_reason\":null}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            for (i, byte) in sse.as_bytes().iter().enumerate() {
                socket.write_all(b"1\r\n").unwrap();
                socket.write_all(&[*byte]).unwrap();
                if i + 1 == sse.len() {
                    let _ = socket.write_all(b"\r\n");
                } else {
                    socket.write_all(b"\r\n").unwrap();
                }
            }
            // The client may close immediately after [DONE].
            let _ = socket.write_all(b"0\r\n\r\n");
        });
        let messages = [Message {
            role: "system".into(),
            content: "prefix\n".into(),
        }];
        let mut deltas = String::new();
        let result = stream_chat(&format!("http://{address}"), "mock", &messages, &mut |s| {
            deltas.push_str(s)
        })
        .unwrap();
        assert_eq!(result.text, "hé🦀");
        assert_eq!(deltas, result.text);
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
        server.join().unwrap();
    }

    #[test]
    fn cancelled_pending_read_returns_promptly() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (_server, _) = listener.accept().unwrap();
        let cancel = AtomicBool::new(false);
        thread::scope(|scope| {
            let flag = &cancel;
            scope.spawn(move || {
                thread::sleep(Duration::from_millis(40));
                flag.store(true, Ordering::Relaxed);
            });
            let mut socket = DeadlineStream::new(client, Some(&cancel)).unwrap();
            let started = Instant::now();
            assert_eq!(
                socket.read(&mut [0; 1]).unwrap_err().kind(),
                io::ErrorKind::ConnectionAborted
            );
            assert!(started.elapsed() < Duration::from_secs(2));
        });
    }

    fn mock_response(bytes: Vec<u8>, on_delta: &mut dyn FnMut(&str)) -> io::Result<Completion> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(DeadlineStream::new(socket, None).unwrap());
            let head = legacy_head(&mut reader);
            let mut body = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
            body.read_to_end(&mut Vec::new()).unwrap();
            body.reader.get_mut().write_all(&bytes).unwrap();
        });
        let result = stream_chat(&format!("http://{address}"), "mock", &[], on_delta);
        worker.join().unwrap();
        result
    }

    #[test]
    fn nonstream_json_fallback_and_status_error_privacy() {
        let body = br#"{"choices":[{"message":{"role":"assistant","content":"hello\n"},"finish_reason":"stop"}]}"#;
        let mut reply = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
        reply.extend_from_slice(body);
        let mut deltas = Vec::new();
        let completion = mock_response(reply, &mut |text| deltas.push(text.to_owned())).unwrap();
        assert_eq!(deltas, ["hello\n"]);
        assert_eq!(completion.text, "hello\n");
        assert_eq!(completion.finish_reason.as_deref(), Some("stop"));
        let error = mock_response(b"HTTP/1.1 429 Limited\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"sensitive upstream details\"}".to_vec(), &mut |_| {}).unwrap_err();
        assert_eq!(error.to_string(), "upstream HTTP status 429");
        assert_eq!(http_status(&error), Some(429));
        assert_eq!(http_status(&io::Error::other("upstream HTTP status 429")), None);
    }

    #[test]
    fn truncated_sse_never_becomes_a_successful_completion() {
        for (code, kind) in [("client_cancelled", io::ErrorKind::Interrupted), ("generation_deadline", io::ErrorKind::TimedOut)] {
            let mut partial = String::new();
            let data = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {{\"choices\":[{{\"delta\":{{\"content\":\"partial\"}}}}]}}\n\ndata: {{\"error\":{{\"type\":\"{code}\",\"message\":\"private\"}}}}\n\ndata: [DONE]\n\n");
            let error = mock_response(data.into_bytes(), &mut |s| partial.push_str(s)).unwrap_err();
            assert_eq!((partial.as_str(), error.kind()), ("partial", kind));
            assert!(!error.to_string().contains("private"));
        }
        let mut emitted = String::new();
        let error = mock_response(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: {\"choices\":[{".to_vec(), &mut |text| emitted.push_str(text)).unwrap_err();
        assert_eq!(emitted, "partial");
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        let error = mock_response(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\n\r\ndata: ".to_vec(), &mut |_| {}).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn deadline_checks_cover_idle_and_absolute_lifetime_without_sleeping() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (_peer, _) = listener.accept().unwrap();
        let mut stream = DeadlineStream::new(client, None).unwrap();
        stream.progress = Instant::now() - IDLE_TIMEOUT;
        assert_eq!(
            stream.read(&mut [0; 1]).unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
        stream.progress = Instant::now();
        stream.start = Instant::now() - LIFETIME;
        assert_eq!(
            stream.write(b"x").unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
        stream.start = Instant::now();
        stream.deadline = Some(Instant::now());
        assert_eq!(stream.flush().unwrap_err().kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn native_response_idle_budget_keeps_absolute_deadline_and_cancellation() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (_peer, _) = listener.accept().unwrap();
        let cancel = AtomicBool::new(false);
        let mut stream = DeadlineStream::new(client, Some(&cancel)).unwrap();
        stream.idle_timeout = LIFETIME;
        stream.progress = Instant::now() - IDLE_TIMEOUT - Duration::from_secs(1);
        assert!(stream.check().is_ok());
        stream.start = Instant::now() - LIFETIME;
        assert_eq!(stream.check().unwrap_err().kind(), io::ErrorKind::TimedOut);
        stream.start = Instant::now();
        stream.deadline = Some(Instant::now());
        assert_eq!(stream.check().unwrap_err().kind(), io::ErrorKind::TimedOut);
        stream.deadline = None;
        cancel.store(true, Ordering::Relaxed);
        assert_eq!(
            stream.check().unwrap_err().kind(),
            io::ErrorKind::ConnectionAborted
        );
    }

    #[test]
    fn cumulative_chunk_body_and_sse_event_limits_are_enforced() {
        let head = Head {
            first: String::new(),
            fields: vec![("transfer-encoding".into(), "chunked".into())],
        };
        let mut body = BodyReader::new(
            Cursor::new(b"2\r\nab\r\n2\r\ncd\r\n0\r\n\r\n"),
            &head,
            false,
            3,
        )
        .unwrap();
        assert!(body.read_to_end(&mut Vec::new()).is_err());
        let head = Head {
            first: String::new(),
            fields: Vec::new(),
        };
        let mut close_body = BodyReader::new(Cursor::new(b"abcd"), &head, false, 3).unwrap();
        assert!(close_body.read_to_end(&mut Vec::new()).is_err());
        let mut decoder = SseDecoder::default();
        assert!(
            decoder
                .feed(&vec![b'x'; MAX_EVENT + 1], &mut |_| Ok(false))
                .is_err()
        );
        let mut decoder = SseDecoder::default();
        assert!(decoder.feed(b"data: \xff\n\n", &mut |_| Ok(false)).is_err());
    }

    fn legacy_head<S: Read + Write>(reader: &mut BufReader<S>) -> Head {
        let mut head = read_head(reader).unwrap();
        if head.first == "GET /v1/capabilities HTTP/1.1" {
            assert_eq!(head.get("content-length"), Some("0"));
            reader
                .get_mut()
                .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            head = read_head(reader).unwrap();
        }
        head
    }

    fn receive_request(reader: &mut BufReader<TcpStream>) -> Vec<u8> {
        let head = legacy_head(reader);
        assert_eq!(head.get("connection"), Some("keep-alive"));
        let mut body = BodyReader::new(reader, &head, true, json::MAX_INPUT_BYTES).unwrap();
        let mut bytes = Vec::new();
        body.read_to_end(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn native_tool_session_wire_prefix_and_keepalive() {
        use crate::agent::{
            self, Conversation, Limits, StopReason, ToolError, ToolExecutor, ToolInvocation,
        };
        use crate::agent_wire::FinishReason;
        struct MockTools;
        impl ToolExecutor for MockTools {
            fn execute(
                &mut self,
                call: ToolInvocation<'_>,
                _: &AtomicBool,
            ) -> Result<String, ToolError> {
                assert_eq!(call.call.name, "lookup");
                Ok(" exact tool result\n🦀 ".into())
            }
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let upstream = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut reader = BufReader::new(socket);
            let mut prior = Vec::new();
            let mut tools = None;
            for turn in 0..6 {
                let body = receive_request(&mut reader);
                assert!(
                    body.starts_with(&prior),
                    "wire prefix changed on model turn {turn}"
                );
                prior = body[..body.len() - 2].to_vec(); // JSON envelope delimiters only
                let request = json::parse_bytes(&body).unwrap();
                let definitions = json::stringify(request.get("tools").unwrap()).unwrap();
                if let Some(tools) = &tools {
                    assert_eq!(tools, &definitions);
                }
                tools = Some(definitions);
                if turn % 2 == 1 {
                    let messages = request.get("messages").unwrap().as_array().unwrap();
                    let result = messages.last().unwrap();
                    assert_eq!(
                        result.get("tool_call_id").unwrap().as_str(),
                        Some(format!("call-{}", turn / 2).as_str())
                    );
                    assert_eq!(
                        result.get("content").unwrap().as_str(),
                        Some(" exact tool result\n🦀 ")
                    );
                    assert!(
                        messages[messages.len() - 2]
                            .get("reasoning_content")
                            .is_some()
                    );
                }
                let event = if turn % 2 == 0 {
                    format!(
                        r#"{{"choices":[{{"delta":{{"content":null,"reasoning_content":"retain reasoning","tool_calls":[{{"index":0,"id":"call-{}","type":"function","function":{{"name":"lookup","arguments":"{{}}"}},"vendor":true}}]}},"finish_reason":"tool_calls"}}]}}"#,
                        turn / 2
                    )
                } else {
                    r#"{"choices":[{"delta":{"content":"exact answer 🦀\n"},"finish_reason":"stop"}]}"#.into()
                };
                let data = format!("data: {event}\n\ndata: [DONE]\n\n");
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}", data.len(), data).unwrap();
            }
            // All six requests must have used this one socket, including the
            // three explicit user prompts and three ID-linked tool continuations.
        });
        let mut conversation = Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"seed":42,"prompt_cache_key":"mock-stable","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"messages":[{"role":"system","content":"Exact system\n  🦀"}]}"#).unwrap(), Limits::default()).unwrap();
        let mut model = NativeModel::new(&endpoint).unwrap();
        for turn in 0..3 {
            conversation
                .push_user(
                    json::parse(&format!(
                        r#"{{"role":"user","content":" prompt {turn} 🦀\n"}}"#
                    ))
                    .unwrap(),
                    Limits::default(),
                )
                .unwrap();
            let report = agent::run(
                &mut conversation,
                &mut model,
                &mut MockTools,
                Limits::default(),
                &AtomicBool::new(false),
                &mut |_| {},
            );
            assert_eq!(report.reason, StopReason::Finished(FinishReason::Stop));
            assert_eq!((report.model_turns, report.tool_calls), (2, 1));
        }
        upstream.join().unwrap();
    }

    #[test]
    fn direct_keepalive_consumes_chunk_trailers_before_reusing() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut reader = BufReader::new(socket);
            for _ in 0..3 {
                receive_request(&mut reader);
                let data = b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n", data.len()).unwrap();
                reader.get_mut().write_all(data).unwrap();
                reader
                    .get_mut()
                    .write_all(b"\r\n0\r\nX-Trailer: done\r\n\r\n")
                    .unwrap();
            }
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        for _ in 0..3 {
            assert_eq!(
                client.stream("mock", &[], &mut |_| {}, None).unwrap().text,
                "x"
            );
        }
        server.join().unwrap();
    }

    #[test]
    fn done_cannot_hide_truncated_framing_or_post_done_data() {
        let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        for response in [
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{payload}",
                payload.len() + 1
            ),
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{payload}\r\n",
                payload.len()
            ),
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{payload}data: {{}}\n\n"
            ),
        ] {
            assert!(mock_response(response.into_bytes(), &mut |_| {}).is_err());
        }
    }

    #[test]
    fn cancellation_cannot_promote_truncated_framing_after_done() {
        // Stop + [DONE] arrive, then the socket is held open with one
        // promised body byte missing. Cancelling while the client waits
        // for that byte is an error, never a validated completion.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(socket);
            receive_request(&mut reader);
            let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{payload}", payload.len() + 1).unwrap();
            // Hold the socket open; the client must fail and close it (a
            // reset close after cancellation is equally a close).
            match reader.read(&mut [0]) {
                Ok(0) | Err(_) => (),
                other => panic!("client kept a truncated stream open: {other:?}"),
            }
        });
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let client_cancel = std::sync::Arc::clone(&cancel);
        let (waiting_tx, waiting) = std::sync::mpsc::channel::<()>();
        let (cancelled_tx, cancelled) = std::sync::mpsc::channel::<()>();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let client = thread::spawn(move || {
            let mut client = ChatClient::new(&endpoint).unwrap();
            // Armed only after [DONE] is decoded, and invoked only when the
            // real socket read times out waiting for the missing body byte.
            // Hold that read boundary until cancellation has been set.
            client.after_done_read_wait = Some(Box::new(move || {
                waiting_tx.send(()).unwrap();
                cancelled.recv_timeout(Duration::from_secs(5)).unwrap();
            }));
            let result = client.stream("mock", &[], &mut |_| {}, Some(&client_cancel));
            result_tx.send(result.map(|done| done.text)).unwrap();
        });
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        cancel.store(true, Ordering::Relaxed);
        cancelled_tx.send(()).unwrap();
        let result = result_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        client.join().unwrap();
        server.join().unwrap();
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
    }

    #[test]
    fn sse_borrowed_fast_path_and_all_fragment_boundaries_agree() {
        let input = "\u{feff}:first\r\ndata: one 🦀\n\ndata: two\r\n\r\nevent: ignored\ndata: a\ndata: b\n\ndata: end\r\r";
        for split in 0..=input.len() {
            let mut decoder = SseDecoder::default();
            let mut actual = Vec::new();
            let mut emit = |s: &str| {
                actual.push(s.to_owned());
                Ok(false)
            };
            decoder.feed(&input.as_bytes()[..split], &mut emit).unwrap();
            decoder.feed(&input.as_bytes()[split..], &mut emit).unwrap();
            assert_eq!(actual, ["one 🦀", "two", "a\nb", "end"], "split {split}");
        }
        let mut decoder = SseDecoder::default();
        decoder.feed(b":init\n", &mut |_| Ok(false)).unwrap();
        let input = b"data: borrowed\n\n";
        decoder
            .feed(input, &mut |s| {
                assert_eq!(
                    s.as_ptr(),
                    input[6..].as_ptr(),
                    "common SSE data must borrow input"
                );
                Ok(false)
            })
            .unwrap();
        assert!(decoder.line.capacity() < input.len());
    }

    #[test]
    fn close_errors_cancellation_and_expired_idle_sockets_are_not_reused() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            for phase in 0..4 {
                let (socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut reader = BufReader::new(socket);
                receive_request(&mut reader);
                let data = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                let close = if phase == 0 {
                    "Connection: close\r\n"
                } else {
                    ""
                };
                let length = data.len() + usize::from(phase == 1);
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\n{close}Content-Type: text/event-stream\r\nContent-Length: {length}\r\n\r\n{data}").unwrap();
                if phase == 2 {
                    // Cancellation during callback must close even when the
                    // entire body was already received into BufReader.
                    assert_eq!(reader.read(&mut [0]).unwrap(), 0);
                }
            }
        });
        let mut client = ChatClient::new(&endpoint).unwrap();
        client.stream("mock", &[], &mut |_| {}, None).unwrap();
        assert!(client.pool.0.lock().unwrap().is_empty());
        assert_eq!(
            client
                .stream("mock", &[], &mut |_| {}, None)
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert!(client.pool.0.lock().unwrap().is_empty());
        let cancel = AtomicBool::new(false);
        assert_eq!(
            client
                .stream(
                    "mock",
                    &[],
                    &mut |_| {
                        cancel.store(true, Ordering::Relaxed);
                    },
                    Some(&cancel)
                )
                .unwrap_err()
                .kind(),
            io::ErrorKind::ConnectionAborted
        );
        assert!(client.pool.0.lock().unwrap().is_empty());
        client.stream("mock", &[], &mut |_| {}, None).unwrap();
        let mut pool = client.pool.0.lock().unwrap();
        assert_eq!(pool.len(), 1);
        pool[0].1 = Instant::now() - Duration::from_secs(16);
        drop(pool);
        assert!(client.pool.take(&client.endpoint).is_none());
        server.join().unwrap();
    }

    #[test]
    fn post_connection_failure_is_never_automatically_replayed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            receive_request(&mut BufReader::new(socket));
            listener
        });
        assert!(
            ChatClient::new(&endpoint)
                .unwrap()
                .stream("mock", &[], &mut |_| {}, None)
                .is_err()
        );
        let listener = server.join().unwrap();
        listener.set_nonblocking(true).unwrap();
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
