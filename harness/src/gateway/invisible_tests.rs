use super::*; use std::sync::mpsc; fn relay(streaming: bool) { let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let origin = Origin::parse(&format!("http://{}/v1", upstream.local_addr().unwrap())).unwrap();
let listener = TcpListener::bind("127.0.0.1:0").unwrap(); let address = listener.local_addr().unwrap(); let (release, wait) = mpsc::channel(); let payload: &[u8] = if streaming { b": comment\n\ndata: {\"delta\":\"hi\"}\n\n" } else {
b"\x00\xffopaque error\n" }; let server = thread::spawn(move || { let mut socket = BufReader::new(upstream.accept().unwrap().0); let request = WireHead::read(&mut socket).unwrap();
assert_eq!(request.parsed.first, "POST /v1/custom?x=1 HTTP/1.1"); assert!(request.lines.iter().any(|v| v == b"x-Opaque:\t keep  ")); let mut request = BodyReader::new(socket, &request.body_head(), true, 1024).unwrap();
let mut bytes = Vec::new(); request.read_to_end(&mut bytes).unwrap(); assert_eq!(bytes, b" not json \xff"); let writer = request.reader.get_mut(); let status = if streaming { "200 Exact Reason" } else { "429 Exact Refusal" };
if streaming {write!(writer, "HTTP/1.1 {status}\r\nx-Custom:\t exact  \r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n", payload.len()).unwrap();} else {write!(writer,"HTTP/1.0 {status}\r\nx-Custom:\t exact  \r\nContent-Length: {}\r\n\r\n",payload.len()).unwrap();} writer.write_all(payload).unwrap(); if streaming {writer.write_all(b"\r\n").unwrap();} writer.flush().unwrap();
wait.recv_timeout(Duration::from_secs(3)).unwrap(); if streaming {writer.write_all(b"0\r\n\r\n").unwrap();} }); let gateway = thread::spawn(move || { handle( listener.accept().unwrap().0, &Runtime::single(origin), None, None, ) .unwrap() });
let mut client = DeadlineStream::new(TcpStream::connect(address).unwrap(), None).unwrap(); client.write_all(b"POST /v1/custom?x=1 HTTP/1.1\r\nHost: localhost\r\nx-Opaque:\t keep  \r\nContent-Length: 11\r\n\r\n not json \xff").unwrap();
let mut client = BufReader::new(client); let head = WireHead::read(&mut client).unwrap(); assert_eq!( head.parsed.first, if streaming { "HTTP/1.1 200 Exact Reason" } else { "HTTP/1.0 429 Exact Refusal" } );
assert!(head.lines.iter().any(|v| v == b"x-Custom:\t exact  ")); assert!(head.parsed.get("x-vade-gateway").is_none()); let mut client = BodyReader::new(client, &head.body_head(), false, 1024).unwrap();
let mut bytes = vec![0; payload.len()]; client.read_exact(&mut bytes).unwrap();
    assert_eq!(bytes, payload); // Received before upstream is allowed to finish.
release.send(()).unwrap(); assert_eq!(client.read(&mut [0]).unwrap(), 0); server.join().unwrap(); gateway.join().unwrap(); }
#[test]
fn streamed_reply_is_incremental_and_byte_identical() { relay(true); }
#[test]
fn nonstream_error_status_headers_and_binary_body_are_identical() { relay(false); }
#[test]
fn busy_refusal_has_one_stats_row_without_upstream_dispatch() {
    let path=std::env::temp_dir().join(format!("vade-gateway-busy-{}",std::process::id()));let log=StatsLog::open(path.to_str().unwrap()).unwrap();let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();let worker=thread::spawn(move || busy(listener.accept().unwrap().0,Some(&log)).unwrap());
    let mut client=TcpStream::connect(address).unwrap();client.write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();let mut bytes=Vec::new();client.read_to_end(&mut bytes).unwrap();worker.join().unwrap();assert!(bytes.starts_with(b"HTTP/1.1 503"));let lines=std::fs::read_to_string(&path).unwrap();assert_eq!(lines.lines().count(),1);let row=crate::json::parse(lines.trim()).unwrap();assert_eq!(row.get("path").unwrap().as_str(),Some("/v1/models"));assert_eq!(row.get("status"),Some(&crate::json::Value::Number("503".into())));std::fs::remove_file(path).unwrap();
}
fn final_event_close(truncated: bool) {
    let path=std::env::temp_dir().join(format!("vade-gateway-final-{}-{truncated}",std::process::id()));let log=StatsLog::open(path.to_str().unwrap()).unwrap();let upstream=TcpListener::bind("127.0.0.1:0").unwrap();let origin=Origin::parse(&format!("http://{}",upstream.local_addr().unwrap())).unwrap();let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();let (release,wait)=mpsc::channel();
    let event=b"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"input_tokens_details\":{\"cached_tokens\":9}}},\"vade\":{\"prefill_cache_restored_tokens\":9}}\n\n";
    let server=thread::spawn(move || {let socket=upstream.accept().unwrap().0;let mut r=BufReader::new(socket);let h=transport::read_head(&mut r).unwrap();let mut b=BodyReader::new(r,&h,true,1024).unwrap();let mut input=Vec::new();b.read_to_end(&mut input).unwrap();let w=b.reader.get_mut();write!(w,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n",event.len()).unwrap();w.write_all(event).unwrap();w.write_all(b"\r\n").unwrap();w.flush().unwrap();wait.recv_timeout(Duration::from_secs(3)).unwrap();thread::sleep(Duration::from_millis(20));if !truncated {w.write_all(b"0\r\n\r\n").unwrap();}});
    let worker=thread::spawn(move || {let _=handle(listener.accept().unwrap().0,&Runtime::single(origin),Some(&log),None);});let mut c=TcpStream::connect(address).unwrap();c.write_all(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}").unwrap();let mut c=BufReader::new(c);let h=WireHead::read(&mut c).unwrap();let mut b=BodyReader::new(c,&h.body_head(),false,1024).unwrap();let mut bytes=vec![0;event.len()];b.read_exact(&mut bytes).unwrap();assert_eq!(bytes,event);drop(b);release.send(()).unwrap();server.join().unwrap();worker.join().unwrap();let line=std::fs::read_to_string(&path).unwrap();let row=crate::json::parse(line.trim()).unwrap();assert_eq!(row.get("complete"),Some(&crate::json::Value::Bool(!truncated)));assert_eq!(row.get("cached_tokens"),Some(&if truncated {crate::json::Value::Null} else {crate::json::Value::Number("9".into())}));assert_eq!(row.get("downstream_complete"),Some(&crate::json::Value::Bool(false)));std::fs::remove_file(path).unwrap();
}

#[test]
fn client_close_after_final_event_keeps_validated_upstream_usage() { final_event_close(false); }
#[test]
fn final_event_cannot_hide_truncated_upstream_framing() { final_event_close(true); }

#[test]
fn api_base_suffix_preserves_arbitrary_origin_paths_and_queries() { for url in ["http://127.0.0.1:1","http://127.0.0.1:1/v1/"] { let origin=Origin::parse(url).unwrap();for path in ["/v1/models","/v1/capabilities","/custom?x=1","/v1/kv/release"] {assert_eq!(origin.target(path),path);} } }

// Exercise the standalone relay's socket handler, including HTTP framing.
fn mock_relay(path: &str, body: &[u8], response: Vec<u8>, flags: Flags) -> (Vec<u8>, Vec<u8>, io::Result<()>) {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let runtime = Runtime::new(&[format!("http://{}", upstream.local_addr().unwrap())], flags).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let route = format!("POST {path} HTTP/1.1");
    let server = thread::spawn(move || {
        let socket = upstream.accept().unwrap().0;
        socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut reader = BufReader::new(socket);
        let head = WireHead::read(&mut reader).unwrap();
        assert_eq!(head.parsed.first, route);
        let mut input = BodyReader::new(reader, &head.body_head(), true, 8192).unwrap();
        let mut received = Vec::new();
        input.read_to_end(&mut received).unwrap();
        input.reader.get_mut().write_all(&response).unwrap();
        received
    });
    let gateway = thread::spawn(move || handle(listener.accept().unwrap().0, &runtime, None, None));
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    write!(client, "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len()).unwrap();
    client.write_all(body).unwrap();
    let mut reply = Vec::new();
    client.read_to_end(&mut reply).unwrap();
    // EOF is stable: no second response or synthetic terminal after closure.
    assert_eq!(client.read(&mut [0]).unwrap(), 0);
    (server.join().unwrap(), reply, gateway.join().unwrap())
}

#[test]
fn n_turn_prefix_is_byte_identical_through_invisible_gateway() {
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages?beta=true"] {
        for stable in [false, true] {
            let field = if path == "/v1/responses" { "input" } else { "messages" };
            let mut history = String::new();
            let mut previous = Vec::<String>::new();
            let mut key = None;
            for turn in 0..4 {
                if turn > 0 { history.push_str(",\n "); }
                history.push_str(&format!(r#"{{ "role":"user", "content":"turn {turn}: \u0061\n雪" }}, {{"role":"assistant","content":"ok"}}"#));
                let body = format!(r#"{{ "model":"mock", "{field}":[{history}], "temperature":0, "seed":42, "top_p":1, "extension":1.20e0 }}"#);
                let payload = b"data: {\"delta\":\"greedy text\"}\n\ndata: [DONE]\n\n";
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n", payload.len()).into_bytes();
                let (received, reply, result) = mock_relay(path, body.as_bytes(), [response, payload.to_vec()].concat(), Flags { stable, ..Flags::default() });
                result.unwrap();
                let original = crate::json::view(&body).unwrap();
                let forwarded = crate::json::view(std::str::from_utf8(&received).unwrap()).unwrap();
                assert_eq!(forwarded.get(field).unwrap().raw(), original.get(field).unwrap().raw());
                assert_eq!(forwarded.get("extension").unwrap().raw(), "1.20e0");
                let items: Vec<_> = forwarded.get(field).unwrap().array().unwrap().map(|v| v.raw().to_owned()).collect();
                assert_eq!(items[..previous.len()], previous);
                previous = items;
                if !stable { assert_eq!(received, body.as_bytes()); }
                else {
                    // Every generation API, Messages too, gets the cache opt-in the engine needs to reuse a prefix.
                    assert_eq!(forwarded.get("cache").unwrap().raw(), "true", "{path}");
                    let current = forwarded.get("prompt_cache_key").unwrap().raw().to_owned();
                    if let Some(previous) = &key { assert_eq!(&current, previous); }
                    key = Some(current);
                }
                let mut reader = BufReader::new(io::Cursor::new(reply));
                let head = WireHead::read(&mut reader).unwrap();
                let mut output = Vec::new();
                BodyReader::new(reader, &head.body_head(), false, 8192).unwrap().read_to_end(&mut output).unwrap();
                assert_eq!(output, payload);
            }
        }
    }
}

#[test]
fn relayed_stream_terminates_once_even_when_upstream_cuts() {
    for (path, terminal) in [
        ("/v1/chat/completions", "data: [DONE]\n\n"),
        ("/v1/responses", "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n"),
        ("/v1/messages?beta=true", "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
    ] {
        // Complete; cut after terminal SSE; cut inside terminal SSE/HTTP chunk.
        for cut in 0..3 {
            let terminal_bytes = if cut == 2 { &terminal.as_bytes()[..terminal.len()/2] } else { terminal.as_bytes() };
            let payload = [b": heartbeat\n\n".as_slice(), terminal_bytes].concat();
            let mut response = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
            write!(response, "{:x}\r\n", payload.len() + usize::from(cut == 2)).unwrap();
            response.extend_from_slice(&payload);
            if cut != 2 { response.extend_from_slice(b"\r\n"); }
            if cut == 0 { response.extend_from_slice(b"0\r\n\r\n"); }
            let (_, reply, result) = mock_relay(path, b"{}", response, Flags::default());
            assert_eq!(result.is_ok(), cut == 0);
            assert_eq!(reply.windows(b"0\r\n\r\n".len()).filter(|w| *w == b"0\r\n\r\n").count(), usize::from(cut == 0));
            let mut reader = BufReader::new(io::Cursor::new(reply));
            let head = WireHead::read(&mut reader).unwrap();
            assert_eq!(head.parsed.first, "HTTP/1.1 200 OK");
            let mut output = Vec::new();
            let decoded = BodyReader::new(reader, &head.body_head(), false, 8192).unwrap().read_to_end(&mut output);
            assert_eq!(decoded.is_ok(), cut == 0);
            assert_eq!(output, payload);
            assert_eq!(std::str::from_utf8(&output).unwrap().matches(terminal).count(), usize::from(cut != 2));
        }
    }
}

#[test]
fn completions_alias_preserves_queries_prefixes_and_custom_routes() {
    for (base, expected) in [
        ("http://localhost:8080", "/v1/completions?x=1"),
        ("http://localhost:8080/v1", "/v1/completions?x=1"),
        ("http://localhost:8080/proxy/v1", "/proxy/v1/completions?x=1"),
        ("http://localhost:8080/proxy", "/proxy/v1/completions?x=1"),
    ] {
        assert_eq!(Origin::parse(base).unwrap().target("/completions?x=1"), expected);
    }
    let origin = Origin::parse("http://localhost:8080/v1").unwrap();
    for path in ["/custom/completions?x=1", "/completions/extra", "/v1/completions?x=1", "/unknown"] {
        assert_eq!(origin.target(path), path);
    }
}

fn guarded(runtime: &Runtime, access: &Access, request: &str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::scope(|scope| {
        let gateway = scope.spawn(|| handle_guarded(listener.accept().unwrap().0, runtime, None, None, access));
        let mut client = TcpStream::connect(address).unwrap();
        client.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        client.write_all(request.replace("PORT", &address.port().to_string()).as_bytes()).unwrap();
        let mut reply = Vec::new();
        client.read_to_end(&mut reply).unwrap();
        let _ = gateway.join().unwrap();
        String::from_utf8(reply).unwrap()
    })
}

#[test]
fn browser_guards_answer_before_any_upstream_contact() {
    // Port 1 is closed: a request that passes every check fails only upstream (502).
    let runtime = Runtime::single(Origin::parse("http://127.0.0.1:1").unwrap());
    let open = Access::default();
    let mut locked = Access::default();
    locked.allow_origin("https://app.example").unwrap();
    locked.allow_host("vade.lan").unwrap();
    locked.require_token("t0k").unwrap();
    let post = |host: &str, extra: &str| format!("POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n{extra}Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}");
    for (access, request, status) in [
        (&open, post("127.0.0.1:PORT", ""), "502"),
        (&open, post("localhost:PORT", ""), "502"),
        (&open, post("evil.example", ""), "421"),
        (&open, post("evil.example:PORT", "Origin: http://evil.example:PORT\r\n"), "421"),
        (&open, post("localhost:1", ""), "421"),
        (&open, post("127.0.0.1:PORT", "Origin: https://evil.example\r\n"), "403"),
        (&open, post("127.0.0.1:PORT", "Origin: null\r\n"), "403"),
        (&open, "GET /v1/models HTTP/1.1\r\n\r\n".into(), "400"),
        (&locked, post("vade.lan:PORT", "Authorization: Bearer t0k\r\n"), "502"),
        (&locked, post("127.0.0.1:PORT", "Origin: https://app.example\r\nAuthorization: Bearer t0k\r\n"), "502"),
        (&locked, post("127.0.0.1:PORT", "Origin: https://evil.example\r\nAuthorization: Bearer t0k\r\n"), "403"),
        (&locked, post("127.0.0.1:PORT", ""), "401"),
        (&locked, post("127.0.0.1:PORT", "Authorization: Bearer t0kk\r\n"), "401"),
    ] {
        let reply = guarded(&runtime, access, &request);
        assert!(reply.starts_with(&format!("HTTP/1.1 {status} ")), "{request:?} -> {reply}");
    }
    let preflight = guarded(&runtime, &locked, "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost:PORT\r\nOrigin: https://app.example\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: authorization, content-type\r\n\r\n");
    assert!(preflight.starts_with("HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: https://app.example\r\nVary: Origin\r\nAccess-Control-Allow-Methods: POST\r\n"), "{preflight}");
}

#[test]
fn local_errors_speak_the_dialect_of_the_route() {
    // Claude Code reads Anthropic error bodies; OpenAI SDKs read error.type.
    let runtime = Runtime::single(Origin::parse("http://127.0.0.1:1").unwrap());
    let mut locked = Access::default();
    locked.require_token("t0k").unwrap();
    let post = |path: &str| format!("POST {path} HTTP/1.1\r\nHost: 127.0.0.1:PORT\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}");
    for (access, path, body) in [
        (&locked, "/v1/messages?beta=true", r#"{"type":"error","error":{"type":"authentication_error","message":"#),
        (&locked, "/v1/chat/completions", r#"{"error":{"type":"authentication_error","message":"#),
        (&Access::default(), "/v1/messages", r#"{"type":"error","error":{"type":"api_error","message":"upstream connection failed; request not replayed"}}"#),
        (&Access::default(), "/v1/responses", r#"{"error":{"type":"server_error","message":"upstream connection failed; request not replayed","param":null,"code":null}}"#),
    ] {
        let reply = guarded(&runtime, access, &post(path));
        let (_, actual) = reply.split_once("\r\n\r\n").unwrap();
        assert!(actual.starts_with(body), "{path} -> {actual}");
    }
}

#[test]
fn allowed_browser_gets_one_cors_answer_and_the_token_stays_local() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let runtime = Runtime::single(Origin::parse(&format!("http://{}/v1", upstream.local_addr().unwrap())).unwrap());
    let mut access = Access::default();
    access.allow_origin("https://app.example").unwrap();
    access.require_token("t0k").unwrap();
    let server = thread::spawn(move || {
        let mut reader = BufReader::new(upstream.accept().unwrap().0);
        let head = WireHead::read(&mut reader).unwrap();
        assert!(head.parsed.get("authorization").is_none());
        assert!(!head.lines.iter().any(|line| line.to_ascii_lowercase().starts_with(b"authorization")));
        assert_eq!(head.parsed.get("x-api-key"), Some("upstream-key"));
        let mut input = BodyReader::new(reader, &head.body_head(), true, 1024).unwrap();
        input.read_to_end(&mut Vec::new()).unwrap();
        input.reader.get_mut().write_all(b"HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\naccess-control-allow-credentials: true\r\nContent-Length: 2\r\n\r\n{}").unwrap();
    });
    let reply = guarded(&runtime, &access, "POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1:PORT\r\nOrigin: https://app.example\r\nAuthorization: Bearer t0k\r\nX-Api-Key: upstream-key\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}");
    server.join().unwrap();
    assert!(reply.starts_with("HTTP/1.1 200 OK\r\n"), "{reply}");
    let lower = reply.to_ascii_lowercase();
    assert_eq!(lower.matches("access-control-").count(), 1, "{reply}");
    assert!(reply.contains("Access-Control-Allow-Origin: https://app.example\r\nVary: Origin\r\n"));
}

#[test]
fn replicas_route_to_fewest_in_flight_rotate_ties_and_keep_conversations_sticky() {
    let r = Router::new(3, false); let mut held: Vec<_> = (0..3).map(|_| r.acquire(None, None, &[]).unwrap()).collect(); assert_eq!(held.iter().map(|l| l.index).collect::<Vec<_>>(), [0, 1, 2]);
    held.remove(1); assert_eq!(r.acquire(None, None, &[]).unwrap().index, 1); assert_eq!(r.acquire(None, None, &[1]).unwrap().index, 2); assert!(r.acquire(None, None, &[0, 1, 2]).is_none());
    let chat = |turns: &str| format!(r#"{{"model":"m","messages":[{{"role":"system","content":"{}"}},{{"role":"user","content":"q"}}{turns}]}}"#, "s".repeat(2100));
    let (one, two) = (route::conversation_key(chat("").as_bytes()), route::conversation_key(chat(r#",{"role":"assistant","content":"a"},{"role":"user","content":"q2"}"#).as_bytes()));
    assert!(one.is_some() && one == two && one != route::conversation_key(br#"{"messages":[{"role":"user","content":"other"}]}"#) && route::conversation_key(br#"{"messages":[]}"#).is_none());
    for sticky in [true, false] { let r = Router::new(2, sticky); let first = r.acquire(None, one, &[]).unwrap().index; assert_eq!(r.acquire(None, one, &[]).unwrap().index == first, sticky);
        let busy = r.acquire(None, None, &[1 - first]).unwrap(); assert_eq!(r.acquire(None, one, &[]).unwrap().index, 1 - first, "a busier remembered upstream yields"); drop(busy); }
}

fn upstream(replies: Vec<&'static str>) -> (String, thread::JoinHandle<usize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap(); let url = format!("http://{}", listener.local_addr().unwrap());
    (url, thread::spawn(move || { listener.set_nonblocking(true).unwrap(); let (mut served, idle) = (0, Instant::now() + Duration::from_millis(1500)); for reply in replies { let socket = loop { match listener.accept() { Ok((s, _)) => break s, Err(_) if Instant::now() < idle => thread::sleep(Duration::from_millis(5)), Err(_) => return served } };
        socket.set_nonblocking(false).unwrap(); let mut r = BufReader::new(socket); let h = transport::read_head(&mut r).unwrap(); let mut b = BodyReader::new(r, &h, true, 1 << 20).unwrap(); b.read_to_end(&mut Vec::new()).unwrap(); b.reader.get_mut().write_all(reply.as_bytes()).unwrap(); served += 1; } served }))
}
fn ask(urls: &[String], request: &str) -> String {
    let runtime = Runtime::new(urls, Flags::default()).unwrap(); let listener = TcpListener::bind("127.0.0.1:0").unwrap(); let mut c = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let gateway = thread::spawn(move || { let _ = handle(listener.accept().unwrap().0, &runtime, None, None); }); c.write_all(request.as_bytes()).unwrap(); let mut reply = String::new(); let _ = c.read_to_string(&mut reply); gateway.join().unwrap(); reply
}
const HEALTH: &str = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
fn health(model: &str) -> &'static str { let body = format!(r#"{{"status":"ready","model":"{model}","active":1,"queued":0,"vade":{{"ptx_sha256":"p"}}}}"#); Box::leak(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}", body.len()).into_boxed_str()) }

#[test]
fn replica_health_aggregates_upstreams_and_refuses_mixed_models() {
    let (a, b) = (upstream(vec![health("m"), health("m")]), upstream(vec![health("m"), health("x")])); let urls = [a.0, b.0];
    let ready = ask(&urls, HEALTH); assert!(ready.starts_with("HTTP/1.1 200") && ready.contains(r#"{"status":"ready","model":"m","active":2,"queued":0,"vade":{"ptx_sha256":"p"}"#) && ready.contains(r#""upstreams":[{"upstream":0,"status":"ready","http":200,"model":"m","active":1,"queued":0},{"upstream":1,"#), "{ready}");
    let mixed = ask(&urls, HEALTH); assert!(mixed.starts_with("HTTP/1.1 502") && mixed.contains(r#""error":"upstream model ids differ""#), "{mixed}"); assert_eq!((a.1.join().unwrap(), b.1.join().unwrap()), (2, 2));
}

#[test]
fn replica_retries_only_before_the_first_reply_byte() {
    let post = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
    let (busy, fine) = (upstream(vec!["HTTP/1.1 503 Loading\r\nContent-Length: 0\r\n\r\n"]), upstream(vec!["HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"]));
    let reply = ask(&[busy.0, fine.0], post); assert!(reply.starts_with("HTTP/1.1 200") && reply.ends_with("\r\n2\r\nok\r\n0\r\n\r\n"), "{reply}"); assert_eq!((busy.1.join().unwrap(), fine.1.join().unwrap()), (1, 1));
    let (cut, spare) = (upstream(vec!["HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\npart"]), upstream(vec!["HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"]));
    let reply = ask(&[cut.0, spare.0], post); assert!(reply.ends_with("\r\n4\r\npart\r\n"), "streamed bytes are never replayed: {reply}"); assert_eq!((cut.1.join().unwrap(), spare.1.join().unwrap()), (1, 0));
}

#[test]
fn replica_cancel_and_release_reach_every_upstream_and_prefer_a_success() {
    for (path, field) in [("/v1/cancel", "cancelled"), ("/v1/kv/release", "released")] {
        let reply = |ok: bool| -> &'static str {
            let body = format!(r#"{{"{field}":{ok}}}"#);
            Box::leak(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).into_boxed_str())
        };
        let key = r#"{"key":"k1"}"#;
        let post = format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{key}", key.len());
        // The idle replica answers first and does not know the key; the owner does.
        for (first, second, expected) in [(false, true, true), (false, false, false), (true, false, true)] {
            let (a, b) = (upstream(vec![reply(first)]), upstream(vec![reply(second)]));
            let answer = ask(&[a.0, b.0], &post);
            assert!(answer.starts_with("HTTP/1.1 200 ") && answer.contains(&format!(r#"{{"{field}":{expected}}}"#)), "{path}: {answer}");
            assert_eq!((a.1.join().unwrap(), b.1.join().unwrap()), (1, 1), "{path}: every replica gets the call");
        }
    }
    // A down replica does not block the answer from the live one.
    let live = upstream(vec!["HTTP/1.1 200 OK\r\nContent-Length: 18\r\n\r\n{\"cancelled\":true}"]);
    let answer = ask(&["http://127.0.0.1:1".into(), live.0], "POST /v1/cancel HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}");
    assert!(answer.starts_with("HTTP/1.1 200 ") && answer.contains(r#"{"cancelled":true}"#), "{answer}");
    assert_eq!(live.1.join().unwrap(), 1);
}

#[test]
fn accept_errors_are_logged_and_the_relay_keeps_serving() {
    // Before this, one failed accept (or a peer gone before peer_addr) ended the relay.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Length: 2\r\n\r\n{{}}", address.port()).as_bytes()).unwrap();
    let incoming = vec![
        Err(io::Error::from(io::ErrorKind::ConnectionAborted)),
        Err(io::Error::from(io::ErrorKind::Interrupted)),
        Err(io::Error::other("too many open files")),
        Ok(listener.accept().unwrap().0),
    ];
    // An owned upstream with a fixed reply: the request served after the
    // failed accepts arrives there and its answer comes back.
    let fixture = upstream(vec!["HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"ok\":true}"]);
    let runtime = Arc::new(Runtime::single(Origin::parse(&fixture.0).unwrap()));
    serve(incoming.into_iter(), runtime, None, None, Arc::new(Access::default())).unwrap();
    let mut reply = String::new();
    client.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.1 200 ") && reply.contains(r#"{"ok":true}"#), "{reply}");
    assert_eq!(fixture.1.join().unwrap(), 1);
}
