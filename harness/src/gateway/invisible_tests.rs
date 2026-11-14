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
    let mut client=TcpStream::connect(address).unwrap();client.write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();let mut bytes=Vec::new();client.read_to_end(&mut bytes).unwrap();worker.join().unwrap();assert!(bytes.starts_with(b"HTTP/1.1 503"));let lines=std::fs::read_to_string(&path).unwrap();assert_eq!(lines.lines().count(),1);let row=crate::json::parse(lines.trim()).unwrap();assert_eq!(row.get("path").unwrap().as_str(),Some("/v1/models"));assert_eq!(row.get("status"),Some(&crate::json::Value::Number("503".into())));assert_eq!(row.get("complete"),Some(&crate::json::Value::Bool(false)));std::fs::remove_file(path).unwrap();
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
    guarded_recorded(runtime, access, None, request)
}

fn guarded_recorded(runtime: &Runtime, access: &Access, recorder: Option<&Recorder>, request: &str) -> String {
    guarded_logged(runtime, access, None, recorder, request)
}

fn guarded_logged(runtime: &Runtime, access: &Access, log: Option<&StatsLog>, recorder: Option<&Recorder>, request: &str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::scope(|scope| {
        let gateway = scope.spawn(|| handle_guarded(listener.accept().unwrap().0, runtime, log, recorder, access));
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
    locked.require_token("t0k-0123456789abcdef").unwrap();
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
        (&locked, post("vade.lan:PORT", "Authorization: Bearer t0k-0123456789abcdef\r\n"), "502"),
        (&locked, post("127.0.0.1:PORT", "Origin: https://app.example\r\nAuthorization: Bearer t0k-0123456789abcdef\r\n"), "502"),
        (&locked, post("127.0.0.1:PORT", "Origin: https://evil.example\r\nAuthorization: Bearer t0k-0123456789abcdef\r\n"), "403"),
        (&locked, post("127.0.0.1:PORT", ""), "401"),
        (&locked, post("127.0.0.1:PORT", "Authorization: Bearer t0k-0123456789abcdefk\r\n"), "401"),
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
    locked.require_token("t0k-0123456789abcdef").unwrap();
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
    access.require_token("t0k-0123456789abcdef").unwrap();
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
    let reply = guarded(&runtime, &access, "POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1:PORT\r\nOrigin: https://app.example\r\nAuthorization: Bearer t0k-0123456789abcdef\r\nX-Api-Key: upstream-key\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}");
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
    let (one, two) = (conversation(&chat(""), None), conversation(&chat(r#",{"role":"assistant","content":"a"},{"role":"user","content":"q2"}"#), None));
    assert!(one.is_some_and(|c| !c.continues) && two.is_some_and(|c| c.continues) && one.map(|c| c.key) == two.map(|c| c.key));
    assert!(one != conversation(r#"{"messages":[{"role":"user","content":"other"}]}"#, None) && conversation(r#"{"messages":[]}"#, None).is_none());
    for sticky in [true, false] { let r = Router::new(2, sticky); let first = r.acquire(None, one, &[]).unwrap().index; assert_eq!(r.acquire(None, one, &[]).unwrap().index == first, sticky);
        let busy = r.acquire(None, None, &[1 - first]).unwrap(); assert_eq!(r.acquire(None, one, &[]).unwrap().index, 1 - first, "a busier remembered upstream yields a first turn"); drop(busy); }
}

fn conversation(body: &str, session: Option<&str>) -> Option<route::Conversation> {
    let fields = session.map(|id| vec![("X-Claude-Code-Session-Id".to_string(), id.to_string())]).unwrap_or_default();
    route::conversation_key(body.as_bytes(), &Head { first: "POST /v1/chat/completions HTTP/1.1".into(), fields })
}
const SHORT_CHAT: [&str; 3] = [r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}]}"#,
    r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"more"}]}"#,
    r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"more"},{"role":"assistant","content":"ok"},{"role":"user","content":"end"}]}"#];

#[test]
fn conversation_key_is_the_same_on_every_turn_of_a_short_chat() {
    // Before, the key hashed through the last user message: a chat under 2048 bytes got a new key each turn.
    let turns: Vec<_> = SHORT_CHAT.iter().map(|body| conversation(body, None).unwrap()).collect();
    assert!(turns.iter().all(|c| c.key == turns[0].key), "{turns:?}");
    assert_eq!(turns.iter().map(|c| c.continues).collect::<Vec<_>>(), [false, true, true]);
    // A different system prompt, tools or first user message is another chat.
    for other in [SHORT_CHAT[0].replace(r#""content":"s""#, r#""content":"t""#), SHORT_CHAT[0].replace(r#""content":"hi""#, r#""content":"ho""#),
        SHORT_CHAT[0].replace(r#"{"model":"m","#, r#"{"model":"m","tools":[{"type":"function"}],"#)] {
        assert_ne!(conversation(&other, None).unwrap().key, turns[0].key, "{other}");
    }
    // An explicit cache key or a session header names the chat, whatever the messages say.
    let keyed = |body: &str| body.replacen('{', r#"{"prompt_cache_key":"chat-7","#, 1);
    assert_eq!(conversation(&keyed(SHORT_CHAT[0]), None).unwrap().key, conversation(&keyed(&SHORT_CHAT[1].replace("hi", "x")), None).unwrap().key);
    assert_eq!(conversation(SHORT_CHAT[0], Some("s1")).unwrap().key, conversation(&SHORT_CHAT[2].replace("hi", "x"), Some("s1")).unwrap().key);
    assert_ne!(conversation(SHORT_CHAT[0], Some("s1")).unwrap().key, conversation(SHORT_CHAT[0], Some("s2")).unwrap().key);
    // Responses input arrays and Messages tool results count user turns the same way.
    let input = conversation(r#"{"model":"m","input":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}]}"#, None).unwrap();
    assert!(input.continues && input.key == conversation(r#"{"model":"m","input":[{"role":"user","content":"a"}]}"#, None).unwrap().key);
    let tool = conversation(r#"{"model":"m","system":"s","messages":[{"role":"user","content":"a"},{"role":"assistant","content":[{"type":"tool_use"}]},{"role":"user","content":[{"type":"tool_result"}]}]}"#, None).unwrap();
    assert!(tool.continues);
    // Tool loops add no user message: a Chat tool reply or a Responses function output still continues the chat.
    let chat_tool = conversation(r#"{"model":"m","messages":[{"role":"user","content":"a"},{"role":"assistant","tool_calls":[{"id":"c1"}]},{"role":"tool","tool_call_id":"c1","content":"r"}]}"#, None).unwrap();
    let responses_tool = conversation(r#"{"model":"m","input":[{"role":"user","content":"a"},{"type":"function_call","call_id":"c1"},{"type":"function_call_output","call_id":"c1","output":"r"}]}"#, None).unwrap();
    let first = conversation(r#"{"model":"m","messages":[{"role":"user","content":"a"}]}"#, None).unwrap();
    assert!(chat_tool.continues && responses_tool.continues && chat_tool.key == first.key, "{chat_tool:?} {responses_tool:?}");
    assert!(conversation(r#"{"model":"m","input":"text"}"#, None).is_none() && conversation(r#"{"model":"m","prompt":"p"}"#, None).is_none());
}

#[test]
fn a_continuation_stays_on_its_upstream_until_the_turn_limit() {
    let (first, next) = (conversation(SHORT_CHAT[0], None), conversation(SHORT_CHAT[1], None));
    let r = Router::new(2, true); let home = r.acquire(None, first, &[]).unwrap().index; let other = 1 - home;
    // The home replica is busier than the idle one, yet the next turn stays: its cache is there.
    let mut held: Vec<_> = (1..route::STICKY_TURN_LIMIT).map(|_| r.acquire(Some(home), None, &[]).unwrap()).collect();
    let stay = r.acquire(None, next, &[]).unwrap(); assert_eq!(stay.index, home, "{} in flight on the home replica", held.len());
    // A first turn with the same key is new work: it goes to the least busy replica.
    assert_eq!(r.acquire(None, first, &[]).unwrap().index, other);
    // At the limit the engine would queue the turn, so load wins.
    held.push(stay); assert_eq!(r.acquire(None, next, &[]).unwrap().index, other);
    held.pop(); assert_eq!(r.acquire(None, next, &[]).unwrap().index, other, "the last turn moved the chat");
    let r = Router::new(2, false); let _busy = r.acquire(Some(0), None, &[]).unwrap(); assert_eq!(r.acquire(None, next, &[]).unwrap().index, 1, "--no-sticky routes by load");
    // Past the limit on every replica (C16 over two pairs), a home that is among the least busy keeps the turn.
    let r = Router::new(2, true); assert_eq!(r.acquire(None, first, &[]).unwrap().index, 0);
    let _full: Vec<_> = (0..2 * route::STICKY_TURN_LIMIT).map(|i| r.acquire(Some(i % 2), None, &[]).unwrap()).collect();
    assert_eq!(r.acquire(None, next, &[]).unwrap().index, 0, "a tie at the limit keeps the cache");
}

#[test]
fn identical_first_turns_spread_evenly_over_the_replicas() {
    // C4 and C8 cards send the same prompt at once: equal halves, never a pile on one pair.
    let first = conversation(SHORT_CHAT[0], None);
    for chats in [4, 8] {
        let r = Router::new(2, true); let held: Vec<_> = (0..chats).map(|_| r.acquire(None, first, &[]).unwrap()).collect();
        assert_eq!(held.iter().filter(|l| l.index == 0).count(), chats / 2, "{chats} chats");
    }
    // Different chats rotate on ties too.
    let r = Router::new(2, true); let a = r.acquire(None, conversation(SHORT_CHAT[0], None), &[]).unwrap();
    let b = r.acquire(None, conversation(&SHORT_CHAT[0].replace("hi", "yo"), None), &[]).unwrap(); assert_ne!(a.index, b.index);
}

#[test]
fn a_dropped_replica_cuts_its_stream_and_the_next_turn_lands_on_a_live_one() {
    let post = |body: &str| format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    let ok = |text: &'static str| -> &'static str { Box::leak(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{text}", text.len()).into_boxed_str()) };
    // Replica 0 answers turn 1, then dies in the middle of turn 2 and stops listening.
    let (dying, live) = (upstream(vec![ok("a1"), "HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\npart"]), upstream(vec![ok("b3"), ok("b4")]));
    let runtime = Runtime::new(&[dying.0, live.0], Flags::default()).unwrap();
    let send = |body: &str| thread::scope(|scope| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap(); let mut c = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let runtime = &runtime; let gateway = scope.spawn(move || { let _ = handle(listener.accept().unwrap().0, runtime, None, None); });
        c.write_all(post(body).as_bytes()).unwrap(); let mut reply = String::new(); let _ = c.read_to_string(&mut reply); gateway.join().unwrap(); reply
    });
    let one = send(SHORT_CHAT[0]); assert!(one.starts_with("HTTP/1.1 200") && one.ends_with("\r\n2\r\na1\r\n0\r\n\r\n"), "{one}");
    // Bytes already reached the client: the gateway ends the reply there and never replays it.
    let two = send(SHORT_CHAT[1]); assert!(two.starts_with("HTTP/1.1 200") && two.ends_with("\r\n4\r\npart\r\n"), "the client sees a cut stream: {two}");
    assert_eq!(dying.1.join().unwrap(), 2);
    // The chat's replica is gone: the next turn fails over before any byte, and later turns follow it.
    let three = send(SHORT_CHAT[2]); assert!(three.starts_with("HTTP/1.1 200") && three.contains("\r\nb3\r\n"), "{three}");
    let four = send(&SHORT_CHAT[2].replace("end", "again")); assert!(four.contains("\r\nb4\r\n"), "{four}");
    assert_eq!(live.1.join().unwrap(), 2);
}

fn upstream(replies: Vec<&'static str>) -> (String, thread::JoinHandle<usize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap(); let url = format!("http://{}", listener.local_addr().unwrap());
    (url, thread::spawn(move || { listener.set_nonblocking(true).unwrap(); let mut served = 0; for reply in replies { let idle = Instant::now() + Duration::from_millis(1500); let socket = loop { match listener.accept() { Ok((s, _)) => break s, Err(_) if Instant::now() < idle => thread::sleep(Duration::from_millis(5)), Err(_) => return served } };
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

#[test]
fn refused_requests_leave_no_recording_and_a_shared_record_folder_becomes_private() {
    let dir = std::env::temp_dir().join(format!("vade-record-guard-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let recorder = Recorder::open(dir.to_str().unwrap()).unwrap();
    let runtime = Runtime::single(Origin::parse("http://127.0.0.1:1").unwrap());
    let mut locked = Access::default();
    locked.require_token("t0k-0123456789abcdef").unwrap();
    let post = |extra: &str| format!("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:PORT\r\n{extra}Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}");
    let folders = || fs_entries(&dir);
    // Before the fix each of these left a request-* folder: no token, a wrong
    // token, no Host, a duplicate Host and a malformed head.
    for (request, status) in [
        (post(""), "401"),
        (post("Authorization: Bearer wrong\r\n"), "401"),
        ("GET /v1/models HTTP/1.1\r\nAuthorization: Bearer t0k-0123456789abcdef\r\n\r\n".to_owned(), "400"),
        (post("Authorization: Bearer t0k-0123456789abcdef\r\nHost: evil.example\r\n"), "400"),
        ("GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:PORT\r\nAuthorization: Bearer t0k-0123456789abcdef\r\nContent-Length: x\r\n\r\n".to_owned(), "400"),
    ] {
        let reply = guarded_recorded(&runtime, &locked, Some(&recorder), &request);
        assert!(reply.starts_with(&format!("HTTP/1.1 {status} ")), "{request:?} -> {reply}");
        assert_eq!(folders(), 0, "{request:?} left a recording");
    }
    // A request that passes the checks is still recorded, with its metadata.
    let reply = guarded_recorded(&runtime, &locked, Some(&recorder), &post("Authorization: Bearer t0k-0123456789abcdef\r\n"));
    assert!(reply.starts_with("HTTP/1.1 502 "), "{reply}");
    assert_eq!(folders(), 1);
    let saved = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
    assert!(saved.join("meta.json").exists() && saved.join("request.body").exists());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        Recorder::open(dir.to_str().unwrap()).unwrap();
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700, "a shared folder becomes private");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

fn fs_entries(dir: &std::path::Path) -> usize { std::fs::read_dir(dir).unwrap().count() }

#[test]
fn a_stalled_replica_holds_neither_the_cancel_answer_nor_another_replica() {
    // Replica 0 takes the connection and never answers; before, the serial
    // fan-out waited 30 s for it before it asked replica 1.
    let stalled = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", stalled.local_addr().unwrap());
    let (release, hold) = mpsc::channel::<()>();
    let staller = thread::spawn(move || { let socket = stalled.accept().unwrap().0; let _ = hold.recv_timeout(Duration::from_secs(20)); drop(socket); });
    let live = upstream(vec!["HTTP/1.1 200 OK\r\nContent-Length: 18\r\n\r\n{\"cancelled\":true}"]);
    let started = Instant::now();
    let answer = ask(&[url, live.0], "POST /v1/cancel HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}");
    let took = started.elapsed();
    let _ = release.send(());
    staller.join().unwrap();
    assert!(answer.starts_with("HTTP/1.1 200 ") && answer.contains(r#"{"cancelled":true}"#), "{answer}");
    assert!(took < Duration::from_secs(4), "{took:?}");
    assert_eq!(live.1.join().unwrap(), 1);
    // An oversized control reply is skipped; the bounded one answers.
    let body = "x".repeat(CONTROL_REPLY_BYTES);
    let big: &'static str = Box::leak(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}", body.len()).into_boxed_str());
    let (a, b) = (upstream(vec![big]), upstream(vec!["HTTP/1.1 200 OK\r\nContent-Length: 19\r\n\r\n{\"cancelled\":false}"]));
    let answer = ask(&[a.0, b.0], "POST /v1/cancel HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}");
    assert!(answer.contains(r#"{"cancelled":false}"#), "{}", &answer[..answer.len().min(200)]);
    let _ = (a.1.join(), b.1.join());
}

#[test]
fn a_head_the_gateway_cannot_forward_gets_a_400_not_a_close_or_a_502() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let runtime = Runtime::single(Origin::parse(&format!("http://{}", upstream.local_addr().unwrap())).unwrap());
    // 101 header lines: before, the relay closed the socket with no reply.
    let many: String = (0..101).map(|i| format!("X-{i}: 1\r\n")).collect();
    let reply = guarded(&runtime, &Access::default(), &format!("GET /v1/models HTTP/1.1\r\nHost: localhost:PORT\r\n{many}\r\n"));
    assert!(reply.starts_with("HTTP/1.1 400 "), "{reply}");
    // Claude Code reads the Anthropic error shape on Messages, even for a bad head.
    let reply = guarded(&runtime, &Access::default(), &format!("POST /v1/messages HTTP/1.1\r\nHost: localhost:PORT\r\n{many}\r\n"));
    assert!(reply.starts_with("HTTP/1.1 400 ") && reply.contains(r#"{"type":"error","error":{"type":"invalid_request_error""#), "{reply}");
    // A head just under the limit grows past it with the gateway's own lines:
    // before, the upstream refused it and the client saw a 502.
    let mut request = "GET /v1/models HTTP/1.1\r\nHost: localhost:PORT\r\n".to_owned();
    while request.len() < transport::MAX_HEADERS - 8000 {
        request.push_str(&format!("X-Pad-{}: {}\r\n", request.len(), "y".repeat(7000)));
    }
    request.push_str(&format!("X-Last: {}\r\n\r\n", "z".repeat(transport::MAX_HEADERS - 20 - request.len())));
    let dir = std::env::temp_dir().join(format!("vade-record-head-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let recorder = Recorder::open(dir.to_str().unwrap()).unwrap();
    let stats = dir.with_extension("jsonl");
    let log = StatsLog::open(stats.to_str().unwrap()).unwrap();
    let reply = guarded_logged(&runtime, &Access::default(), Some(&log), Some(&recorder), &request);
    assert!(reply.starts_with("HTTP/1.1 400 ") && reply.contains("oversized HTTP headers"), "{}", &reply[..reply.len().min(200)]);
    // Refused before any recording folder or upstream connection.
    assert_eq!(fs_entries(&dir), 0);
    upstream.set_nonblocking(true).unwrap();
    assert_eq!(upstream.accept().unwrap_err().kind(), io::ErrorKind::WouldBlock, "no upstream connection");
    // Nothing reached an upstream, so the stats row is not a completion.
    let row = crate::json::parse(std::fs::read_to_string(&stats).unwrap().trim()).unwrap();
    assert_eq!(row.get("complete"), Some(&crate::json::Value::Bool(false)));
    assert_eq!(row.get("downstream_complete"), Some(&crate::json::Value::Bool(false)));
    upstream.set_nonblocking(false).unwrap();
    // A forwarded head of exactly the limit, with its real length line, still goes upstream.
    let pad = |n: usize| format!("GET /v1/models HTTP/1.1\r\nHost: localhost:PORT\r\nX-A: {}\r\nX-B: {}\r\nX-C: {}\r\nX-D: {}\r\nX-E: {}\r\n\r\n",
        "a".repeat(7000), "b".repeat(7000), "c".repeat(7000), "d".repeat(7000), "e".repeat(n));
    let forwarded = |n: usize| upstream_head(&WireHead::read(&mut io::Cursor::new(pad(n).into_bytes())).unwrap(), runtime.origin(0), 0).unwrap().len();
    let fits = 4000 + transport::MAX_HEADERS - forwarded(4000);
    assert_eq!(forwarded(fits), transport::MAX_HEADERS);
    let server = thread::spawn(move || {
        let mut socket = BufReader::new(upstream.accept().unwrap().0);
        let head = WireHead::read(&mut socket).unwrap();
        socket.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}").unwrap();
        head.lines.iter().map(|line| line.len() + 2).sum::<usize>() + 2
    });
    let reply = guarded_recorded(&runtime, &Access::default(), Some(&recorder), &pad(fits));
    assert!(reply.starts_with("HTTP/1.1 200 "), "{}", &reply[..reply.len().min(200)]);
    assert_eq!(server.join().unwrap(), transport::MAX_HEADERS);
    assert_eq!(fs_entries(&dir), 1);
    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_file(stats).unwrap();
}

#[test]
fn a_head_that_fits_only_before_the_policy_lengthens_the_body_gets_a_400_with_its_policy() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let runtime = Runtime::new(&[format!("http://{}", upstream.local_addr().unwrap())], Flags { greedy: true, ..Flags::default() }).unwrap();
    let body = r#"{"model":"m","messages":[]}"#;
    let pad = |n: usize| format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:PORT\r\nContent-Type: application/json\r\nContent-Length: {}\r\nX-A: {}\r\nX-B: {}\r\nX-C: {}\r\nX-D: {}\r\nX-E: {}\r\n\r\n{body}",
        body.len(), "a".repeat(7000), "b".repeat(7000), "c".repeat(7000), "d".repeat(7000), "e".repeat(n));
    let head = |n: usize| WireHead::read(&mut io::Cursor::new(pad(n).into_bytes())).unwrap();
    let forwarded = |n: usize, length: usize| upstream_head(&head(n), runtime.origin(0), length).unwrap().len();
    // --greedy-when-unset adds sampling fields, so the length line gets more digits.
    let longer = runtime.prepare("POST", "/v1/chat/completions", &head(0).parsed, body.as_bytes()).body.len();
    assert!(longer.to_string().len() > body.len().to_string().len(), "{longer}");
    let fits = 4000 + transport::MAX_HEADERS - forwarded(4000, body.len());
    assert_eq!(forwarded(fits, body.len()), transport::MAX_HEADERS);
    assert!(forwarded(fits, longer) > transport::MAX_HEADERS);
    let dir = std::env::temp_dir().join(format!("vade-record-policy-head-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let recorder = Recorder::open(dir.to_str().unwrap()).unwrap();
    let stats = dir.with_extension("jsonl");
    let log = StatsLog::open(stats.to_str().unwrap()).unwrap();
    let reply = guarded_logged(&runtime, &Access::default(), Some(&log), Some(&recorder), &pad(fits));
    assert!(reply.starts_with("HTTP/1.1 400 ") && reply.contains("oversized HTTP headers"), "{}", &reply[..reply.len().min(200)]);
    assert_eq!(fs_entries(&dir), 0);
    upstream.set_nonblocking(true).unwrap();
    assert_eq!(upstream.accept().unwrap_err().kind(), io::ErrorKind::WouldBlock, "no upstream connection");
    // The row keeps what the policy did, and it is not a completion.
    let row = crate::json::parse(std::fs::read_to_string(&stats).unwrap().trim()).unwrap();
    assert_eq!(row.get("complete"), Some(&crate::json::Value::Bool(false)));
    assert_eq!(row.get("policy").and_then(|policy| policy.get("greedy_added")), Some(&crate::json::Value::Bool(true)), "{row:?}");
    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_file(stats).unwrap();
}

#[test]
fn with_clamp_limits_an_unforwardable_head_gets_its_400_before_the_admission_call() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = upstream.local_addr().unwrap();
    let url = format!("http://{address}");
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let server = thread::spawn({
        let stop = stop.clone();
        move || {
            let mut reader = BufReader::new(upstream.accept().unwrap().0);
            transport::read_head(&mut reader).unwrap();
            let caps = r#"{"object":"engine.capabilities","contract_version":"vade.engine.v1","model":"mock","context_limit":100,"input_limit":99,"output_limit":16,"admission":"/v1/admission","admission_kind":"context_fit"}"#;
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{caps}", caps.len()).unwrap();
            // Count every later connection; the gateway must open none.
            let mut later = 0;
            for socket in upstream.incoming() {
                if stop.load(std::sync::atomic::Ordering::SeqCst) { break; }
                drop(socket);
                later += 1;
            }
            later
        }
    });
    let runtime = Runtime::new(&[url], Flags { clamp: true, ..Flags::default() }).unwrap();
    let body = r#"{"model":"mock","messages":[]}"#;
    let mut request = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:PORT\r\nContent-Length: {}\r\n", body.len());
    while request.len() < transport::MAX_HEADERS - 8000 {
        request.push_str(&format!("X-Pad-{}: {}\r\n", request.len(), "y".repeat(7000)));
    }
    request.push_str(&format!("X-Last: {}\r\n\r\n{body}", "z".repeat(transport::MAX_HEADERS - 20 - request.len())));
    let reply = guarded(&runtime, &Access::default(), &request);
    assert!(reply.starts_with("HTTP/1.1 400 ") && reply.contains("oversized HTTP headers"), "{}", &reply[..reply.len().min(200)]);
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    TcpStream::connect(address).unwrap();
    assert_eq!(server.join().unwrap(), 0, "the client's header lines reached the upstream in an admission request");
}

#[test]
fn a_local_refusal_or_502_is_not_a_completion_in_stats_or_recordings() {
    let dir = std::env::temp_dir().join(format!("vade-record-local-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let recorder = Recorder::open(dir.to_str().unwrap()).unwrap();
    let stats = dir.with_extension("jsonl");
    let log = StatsLog::open(stats.to_str().unwrap()).unwrap();
    // Ports that were just released are closed, so a request that passes every check gets a 502.
    let closed: Vec<String> = [TcpListener::bind("127.0.0.1:0").unwrap(), TcpListener::bind("127.0.0.1:0").unwrap()].iter().map(|l| format!("http://{}", l.local_addr().unwrap())).collect();
    let one = Runtime::single(Origin::parse(&closed[0]).unwrap());
    let two = Runtime::new(&closed, Flags::default()).unwrap();
    // An upstream that answers 101 gets the gateway's own 502.
    let upgrade = upstream(vec!["HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: x\r\n\r\n"]);
    let switching = Runtime::single(Origin::parse(&upgrade.0).unwrap());
    // Replicas answer /health as one engine, and the gateway writes that answer itself.
    let (a, b) = (upstream(vec![health("m")]), upstream(vec![health("m")]));
    let replicas = Runtime::new(&[a.0.clone(), b.0.clone()], Flags::default()).unwrap();
    let (open, mut browser) = (Access::default(), Access::default());
    browser.allow_origin("https://app.example").unwrap();
    let many: String = (0..101).map(|i| format!("X-{i}: 1\r\n")).collect();
    let cases: [(&Runtime, &Access, String, &str); 11] = [
        (&one, &open, format!("GET /v1/models HTTP/1.1\r\nHost: localhost:PORT\r\n{many}\r\n"), "400"),
        (&one, &open, "GET v1/models HTTP/1.1\r\nHost: localhost:PORT\r\n\r\n".into(), "400"),
        (&one, &open, "GET /v1/models HTTP/1.1\r\nHost: evil.example\r\n\r\n".into(), "421"),
        (&one, &open, "GET /v1/models HTTP/1.1\r\nHost: localhost:PORT\r\nUpgrade: websocket\r\n\r\n".into(), "400"),
        (&one, &open, "POST /v1/responses HTTP/1.1\r\nHost: localhost:PORT\r\nContent-Length: 999999999\r\n\r\n".into(), "413"),
        (&one, &open, "POST /v1/responses HTTP/1.1\r\nHost: localhost:PORT\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n".into(), "400"),
        (&one, &open, "GET /v1/models HTTP/1.1\r\nHost: localhost:PORT\r\n\r\n".into(), "502"),
        (&two, &open, "POST /v1/cancel HTTP/1.1\r\nHost: localhost:PORT\r\nContent-Length: 2\r\n\r\n{}".into(), "502"),
        (&switching, &open, "GET /v1/models HTTP/1.1\r\nHost: localhost:PORT\r\n\r\n".into(), "502"),
        (&one, &browser, "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost:PORT\r\nOrigin: https://app.example\r\nAccess-Control-Request-Method: POST\r\n\r\n".into(), "204"),
        (&replicas, &open, "GET /health HTTP/1.1\r\nHost: localhost:PORT\r\n\r\n".into(), "200"),
    ];
    for (runtime, access, request, status) in &cases {
        let reply = guarded_logged(runtime, access, Some(&log), Some(&recorder), request);
        assert!(reply.starts_with(&format!("HTTP/1.1 {status} ")), "{request:?} -> {reply}");
    }
    assert_eq!((upgrade.1.join().unwrap(), a.1.join().unwrap(), b.1.join().unwrap()), (1, 1, 1));
    let rows = std::fs::read_to_string(&stats).unwrap();
    assert_eq!(rows.lines().count(), cases.len());
    for (row, (_, _, request, _)) in rows.lines().zip(&cases) {
        assert_eq!(crate::json::parse(row).unwrap().get("complete"), Some(&crate::json::Value::Bool(false)), "{request:?} -> {row}");
    }
    // The three 502s passed every check, so they have folders, but replay must skip them.
    assert_eq!(fs_entries(&dir), 3);
    for folder in std::fs::read_dir(&dir).unwrap() {
        let meta = std::fs::read_to_string(folder.unwrap().path().join("meta.json")).unwrap();
        assert!(meta.ends_with(",\"complete\":false}"), "{meta}");
    }
    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_file(stats).unwrap();
}
