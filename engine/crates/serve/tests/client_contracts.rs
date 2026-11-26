//! Client contracts for cancellation, accounting, isolation, and stream endings. The fixture drives passes, not time.
#[path = "support/client_contract_engine.rs"]
mod fixture;
use fixture::*;
use engine_model::json::{self, Value};
use spark_serve::{cache::SessionKey, Limits};
use std::{collections::BTreeMap, sync::atomic::Ordering, time::Duration};

fn object(value: &Value) -> &BTreeMap<String, Value> { value.object().unwrap() }
fn parse(text: &str) -> Value { json::parse(text.as_bytes(), Default::default()).unwrap() }
fn number(n: usize) -> Value { Value::Number(n.to_string()) }
fn body(wire: &str) -> &str { wire.split_once("\r\n\r\n").unwrap().1 }
fn header<'a>(wire: &'a str, name: &str) -> &'a str {
    wire.split("\r\n\r\n").next().unwrap().lines().find_map(|s| s.strip_prefix(&format!("{name}: "))).unwrap_or("")
}
fn complete_http(wire: &str, status: usize) -> Value {
    assert!(wire.starts_with(&format!("HTTP/1.1 {status} ")), "wrong HTTP status");
    assert_eq!(header(wire, "Content-Length").parse::<usize>().unwrap(), body(wire).len(), "truncated HTTP body");
    parse(body(wire))
}
fn frames(wire: &str) -> Vec<Value> {
    let Some((_, body)) = wire.split_once("\r\n\r\n") else { return Vec::new(); };
    body.split_inclusive("\n\n").filter(|frame| frame.ends_with("\n\n")).filter_map(|frame| {
        let data = frame.lines().find_map(|s| s.strip_prefix("data: "))?;
        if data == "[DONE]" { None } else { Some(parse(data)) }
    }).collect()
}
fn request(mode: u32, max: usize, extra: &str) -> String {
    format!(r#"{{"prompt_token_ids":[{mode},1],"max_tokens":{max},"temperature":0,"seed":7,"stream":true{extra}}}"#)
}
fn chat(mode: u32, extra: &str) -> String {
    format!(r#"{{"messages":[{{"role":"user","content":"{mode}"}}],"max_tokens":100,"temperature":0,"seed":7,"stream":true{extra}}}"#)
}
fn usage(value: &Value, prompt: usize, completion: usize, cached: usize, responses: bool) {
    let value = object(value);
    let (input, output) = if responses { ("input_tokens", "output_tokens") } else { ("prompt_tokens", "completion_tokens") };
    assert_eq!(value[input], number(prompt), "prompt ledger");
    assert_eq!(value[output], number(completion), "completion ledger");
    assert_eq!(value["total_tokens"], number(prompt + completion), "total ledger");
    assert_eq!(object(&value[&format!("{input}_details")])["cached_tokens"], number(cached), "restore ledger");
}
fn ending(wire: &str, reason: &str, prompt: usize, completion: usize, cached: usize) {
    ending_with_usage(wire, reason, prompt, completion, cached, None);
}
fn ending_with_usage(wire: &str, reason: &str, prompt: usize, completion: usize, cached: usize, include_usage: Option<bool>) {
    assert_eq!(body(wire).matches("data: [DONE]\n\n").count(), 1, "one DONE marker");
    assert!(wire.ends_with("data: [DONE]\n\n"), "DONE must be last");
    let data = frames(wire);
    let finals: Vec<_> = data.iter().filter(|v| object(v).get("choices").is_some_and(|c|
        c.array().unwrap().iter().any(|c| object(c).get("finish_reason").is_some_and(|r| *r != Value::Null)))).collect();
    assert_eq!(finals.len(), 1, "one finish chunk");
    let last = data.last().unwrap();
    let finish_at = data.len() - if include_usage == Some(true) { 2 } else { 1 };
    assert_eq!(finals[0], &data[finish_at], "finish must immediately precede usage or DONE");
    assert_eq!(object(&object(finals[0])["choices"].array().unwrap()[0])["finish_reason"], Value::String(reason.into()));
    let usage_at: Vec<_> = data.iter().enumerate().filter(|(_, v)| object(v).contains_key("usage")).map(|(i, _)| i).collect();
    if include_usage == Some(false) {
        assert!(usage_at.is_empty(), "explicit false must omit usage");
    } else {
        assert_eq!(usage_at, vec![data.len() - 1], "usage appears only on the final JSON frame");
        usage(&object(last)["usage"], prompt, completion, cached, false);
    }
    if include_usage == Some(true) {
        assert_eq!(object(last)["choices"], Value::Array(vec![]), "requested usage has no choices");
        assert_eq!(object(last).keys().map(String::as_str).collect::<Vec<_>>(),
            ["choices", "created", "id", "model", "object", "usage", "vade"], "exact usage frame fields");
        for key in ["id", "object", "created", "model"] { assert_eq!(object(last)[key], object(&data[0])[key], "usage frame identity"); }
    }
    assert!(!data.iter().any(|v| object(v).contains_key("error")), "successful stream has an error");
}
fn content(wire: &str, chat: bool) -> String {
    frames(wire).iter().filter_map(|v| object(v).get("choices"))
        .flat_map(|v| v.array().unwrap()).filter_map(|v| {
            if chat { object(v).get("delta").and_then(|d| object(d).get("content")) } else { object(v).get("text") }
        }).filter_map(|v| v.string().ok()).collect()
}
fn failure(wire: &str, code: &str, committed: usize) {
    failure_cached(wire, code, committed, 0);
}
fn failure_cached(wire: &str, code: &str, committed: usize, cached: usize) {
    assert_eq!(body(wire).matches("data: [DONE]\n\n").count(), 1, "one failure terminator");
    assert!(wire.ends_with("data: [DONE]\n\n"), "failure DONE must be last");
    for frame in body(wire).split_inclusive("\n\n") {
        if frame.starts_with(':') { assert!(frame.ends_with("\n\n"), "complete comment frame required"); continue; }
        let data = frame.strip_prefix("data: ").and_then(|s| s.strip_suffix("\n\n")).expect("complete data frame required");
        if data != "[DONE]" { parse(data); }
    }
    let data = frames(wire);
    let errors: Vec<_> = data.iter().enumerate().filter(|(_, v)| object(v).contains_key("error")).collect();
    assert_eq!(errors.len(), 1, "one error frame");
    let (at, error) = errors[0];
    assert_eq!(object(&object(error)["error"])["type"], Value::String(code.into()));
    assert_eq!(at + 2, data.len(), "error then usage then DONE");
    let tail = object(data.last().unwrap());
    assert_eq!(tail.len(), 2, "usage-only terminal frame");
    assert_eq!(tail["choices"], Value::Array(vec![]));
    usage(&tail["usage"], 2, committed, cached, false);
    for frame in &data {
        if let Some(choices) = object(frame).get("choices") { for choice in choices.array().unwrap() {
            assert!(object(choice).get("finish_reason").is_none_or(|r| *r == Value::Null), "failure has a successful finish");
        } }
    }
}
/// Remove only per-run receipt data. Keep the complete ordered client payload bytes.
fn payload(wire: &str) -> String {
    body(wire).split("\n\n").filter_map(|frame| {
        let data = frame.lines().find_map(|s| s.strip_prefix("data: "))?;
        if data == "[DONE]" || object(&parse(data)).contains_key("usage") { return None; }
        let start = data.find("\"choices\":[")? + "\"choices\":".len();
        let (mut depth, mut quoted, mut escaped) = (0, false, false);
        for (i, b) in data.as_bytes()[start..].iter().enumerate() {
            if escaped { escaped = false; continue; }
            if quoted && *b == b'\\' { escaped = true; continue; }
            if *b == b'"' { quoted = !quoted; continue; }
            if quoted { continue; }
            if *b == b'[' { depth += 1; }
            if *b == b']' { depth -= 1; if depth == 0 { return Some(data[start..=start + i].to_owned()); } }
        }
        panic!("incomplete choices array");
    }).collect::<Vec<_>>().join("\n")
}
fn health(server: &Server) -> Value { parse(body(&server.finish(&mut server.get("/health", "")))) }
fn count(value: &Value, key: &str) -> usize { object(value)[key].u64().unwrap() as usize }

#[test]
fn cache_capabilities_cannot_cross_scopes() {
    let (server, bob) = Server::keyed();
    let alice = "obviously-fake-alice-scope";
    let bob = bob.as_str();
    let cold = server.run("/v1/completions", &request(10, 4, r#","cache":false"#), bob);
    let other = server.run("/v1/completions", &request(17, 4, r#","cache":true,"kv_checkpoint":true"#), bob);
    let warm = server.run("/v1/completions", &request(10, 4, r#","cache":true,"kv_checkpoint":true,"prompt_cache_key":"shared""#), alice);
    assert_ne!(content(&other, false), content(&warm, false), "fixture must distinguish private prompts");
    let data = frames(&warm);
    let receipt = object(&object(&object(data.last().unwrap())["usage"])["vade"]);
    let kv = object(&receipt["kv"]);
    let handle = object(&kv["checkpoint"])["handle"].string().unwrap();
    let session = header(&warm, "X-Session-Capability");
    assert!(!session.is_empty(), "retained session capability missing");
    for extra in [format!(r#","cache":true,"cache_salt":"attempt-a","session":"{session}","fork":true"#),
        format!(r#","cache":true,"cache_salt":"attempt-b","kv_handle":"{handle}","prompt_cache_key":"shared""#)] {
        let wire = server.run("/v1/completions", &request(10, 4, &extra), bob);
        ending(&wire, "length", 2, 4, 0);
        assert_eq!(payload(&wire), payload(&cold), "scope changed client payload");
        assert_eq!(header(&wire, "X-Prefix-Reused-Tokens"), "0");
    }
    complete_http(&server.run("/v1/kv/release", &format!(r#"{{"kv_handle":"{handle}"}}"#), bob), 200);
    let owner = server.run("/v1/completions", &request(10, 4, &format!(r#","cache":true,"kv_handle":"{handle}""#)), alice);
    ending(&owner, "length", 2, 4, 2);
    let owner_frames = frames(&owner);
    let owner_kv = object(&object(&object(&object(owner_frames.last().unwrap())["usage"])["vade"])["kv"]);
    assert_eq!(owner_kv["reuse"], Value::String("hit".into()), "foreign release deleted the owner's checkpoint");
    assert_eq!(payload(&owner), payload(&cold));
    for path in ["/v1/models", "/v1/events", "/v1/capabilities"] {
        let refused = server.finish(&mut server.get(path, ""));
        assert_eq!(complete_http(&refused, 401), parse(spark_serve::auth::UNAUTHORIZED_BODY));
    }
    for path in ["/tokenize", "/v1/completions", "/v1/chat/completions", "/v1/responses", "/v1/messages",
        "/v1/messages/count_tokens", "/v1/cancel", "/v1/kv/release", "/v1/admission"] {
        let refused = server.run(path, "{}", "");
        assert_eq!(complete_http(&refused, 401), parse(spark_serve::auth::UNAUTHORIZED_BODY));
    }
    let open = server.finish(&mut server.get("/health", ""));
    assert_eq!(complete_http(&open, 200), parse(r#"{"status":"ready"}"#));
    server.stop();
}

#[test]
fn nonfinite_batch_row_fails_only_its_request() {
    for mode in 18..=20 {
        let server = Server::new();
        let cold = server.run("/v1/completions", &request(10, 12, r#","cache":false"#), "");
        server.fault.store(true, Ordering::Relaxed);
        server.hold.store(true, Ordering::Relaxed);
        let mut bad = server.send("/v1/completions", &request(mode, 12, r#","cache":false"#), "");
        let mut peer = server.send("/v1/completions", &request(10, 12, r#","cache":false"#), "");
        server.until(|| { let ledger = server.ledger.lock().unwrap(); ledger.attempts.get(&mode) == Some(&1)
            && ledger.attempts.get(&10) == Some(&2) });
        server.hold.store(false, Ordering::Relaxed);
        let bad = server.finish(&mut bad);
        failure(&bad, "generation_error", 2);
        let peer = server.finish(&mut peer);
        ending(&peer, "length", 2, 12, 0);
        assert_eq!(payload(&peer), payload(&cold), "bad row changed sibling payload");
        assert!(server.ledger.lock().unwrap().batches.iter().any(|&n| n > 1), "no batch exercised");
        assert_eq!(object(&health(&server))["status"], Value::String("ready".into()));
        server.stop();
    }
}

#[test]
// Finding S1 ruled: the client gets a generic message; the server keeps the cause.
fn malformed_batch_result_retires_all_affected_rows() {
    // The event ring is process-global and bounded. Give this scenario its own process.
    // The child is this binary run for this one test alone. It is marked by its argument list being EXACTLY these
    // four (no environment switch). Any other list, even one that contains them beside another filter, spawns the child.
    const ALONE: [&str; 4] = ["--exact", "malformed_batch_result_retires_all_affected_rows", "--nocapture", "--test-threads=1"];
    if !std::env::args().skip(1).eq(ALONE) {
        let output = std::process::Command::new(std::env::current_exe().unwrap()).args(ALONE).output().unwrap();
        assert!(output.status.success(), "isolated retirement scenario failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        return;
    }
    let server = Server::new();
    let cold = server.run("/v1/completions", &request(10, 6, r#","cache":false"#), "");
    server.hold.store(true, Ordering::Relaxed);
    let mut a = server.send("/v1/completions", &request(10, 6, r#","cache":false"#), "");
    let mut b = server.send("/v1/completions", &request(17, 6, r#","cache":false"#), "");
    server.until(|| server.ledger.lock().unwrap().attempts.get(&17) == Some(&1));
    server.hold.store(false, Ordering::Relaxed);
    // Prefill is serial; wait until each request has emitted a sampled token.
    server.until(|| { a.poll(); b.poll(); !content(&a.wire, false).is_empty() && !content(&b.wire, false).is_empty() });
    let before = server.ledger.lock().unwrap().steps.values().sum::<usize>();
    let affected: Vec<_> = { let ledger = server.ledger.lock().unwrap(); ledger.modes.keys()
        .filter(|id| !ledger.dropped_ids.contains(id)).copied().collect() };
    assert_eq!(affected.len(), 2, "both affected sessions must still be live");
    let stopped_before = engine_model::events::count("request_stopped");
    server.malformed.store(true, Ordering::Relaxed);
    let a = server.finish(&mut a); let b = server.finish(&mut b);
    let events = parse(&engine_model::events::json());
    assert_eq!(engine_model::events::count("request_stopped"), stopped_before + 2, "each failed row must log its cause");
    assert_eq!(object(&events)["recent"].array().unwrap().iter().filter(|event| {
        let event = object(event);
        event["name"] == Value::String("request_stopped".into())
            && event["line"] == Value::String("request stopped: incomplete batch result".into())
    }).count(), 2, "server event record lost the batch failure detail");
    for wire in [&a, &b] {
        failure(wire, "generation_error", content(wire, false).len());
        let data = frames(wire);
        let error = data.iter().find_map(|value| object(value).get("error")).unwrap();
        assert_eq!(*error, parse(r#"{"type":"generation_error","message":"generation stopped before completion","param":null,"code":null}"#), "client error must keep internal details private");
        assert!(!body(wire).contains("incomplete batch result"), "internal batch cause leaked to the client");
    }
    assert!(server.ledger.lock().unwrap().steps.values().sum::<usize>() > before, "fault must follow state mutation");
    {
        let ledger = server.ledger.lock().unwrap().clone();
        for id in affected {
            assert_eq!(ledger.retired.get(&id), Some(&1), "affected session retirement missing or repeated");
            assert!(ledger.dropped_ids.contains(&id), "affected session ownership survives before follower admission");
        }
    }
    let fresh = server.run("/v1/completions", &request(10, 6, ""), "");
    ending(&fresh, "length", 2, 6, 0); assert_eq!(payload(&fresh), payload(&cold));
    server.stop();
}

#[test]
fn recoverable_admission_error_releases_budget() {
    // There are two slots, but the byte budget funds exactly one session.
    let server = Server::budgeted();
    let cold = server.run("/v1/completions", &request(10, 4, r#","cache":false"#), "");
    server.hold.store(true, Ordering::Relaxed);
    let mut healthy = server.send("/v1/completions", &request(17, 4, r#","cache":false"#), "");
    server.until(|| server.ledger.lock().unwrap().attempts.get(&17) == Some(&1));
    let mut refused = server.send("/v1/completions", &request(22, 4, r#","cache":false"#), "");
    let mut follower = server.send("/v1/completions", &request(10, 4, r#","cache":false"#), "");
    server.until(|| { let ledger = server.ledger.lock().unwrap(); ledger.validated.get(&22) == Some(&1) && ledger.validated.get(&10) == Some(&2) });
    assert_eq!(count(&health(&server), "queued"), 2);
    server.hold.store(false, Ordering::Relaxed); server.finish(&mut healthy);
    let refused = complete_http(&server.finish(&mut refused), 503);
    assert_eq!(object(&refused)["error"], parse(r#"{"type":"engine_error","message":"session unavailable","param":null,"code":null}"#));
    {
        let ledger = server.ledger.lock().unwrap().clone();
        let refused: Vec<_> = ledger.modes.iter().filter(|(_, mode)| **mode == 22).map(|(id, _)| *id).collect();
        assert_eq!(refused.len(), 1, "refusal must exercise an allocated session");
        assert!(ledger.dropped_ids.contains(&refused[0]), "refused admission kept its session budget");
    }
    let follower = server.finish(&mut follower);
    ending(&follower, "length", 2, 4, 0); assert_eq!(payload(&follower), payload(&cold));
    assert_eq!(count(&health(&server), "active"), 0);
    server.stop();
}

#[test]
fn queued_cancel_never_allocates_the_target() {
    let cold_server = Server::new();
    let cold = cold_server.run("/v1/completions", &request(17, 3, r#","cache":false"#), "");
    cold_server.stop();
    let server = Server::start(1, 8, Limits { intake_burst: Duration::ZERO, ..Default::default() });
    server.hold.store(true, Ordering::Relaxed);
    let mut active = server.send("/v1/completions", &request(10, 3, r#","cache":false"#), "");
    server.until(|| server.ledger.lock().unwrap().attempts.get(&10) == Some(&1));
    let cancel = SessionKey::generate().unwrap().expose();
    let target_body = request(17, 3, &format!(r#","cache":true,"cancel":"{cancel}""#)).replace("\"stream\":true", "\"stream\":false");
    let mut target = server.send("/v1/completions", &target_body, "");
    server.until(|| server.ledger.lock().unwrap().validated.get(&17) == Some(&1));
    assert_eq!(count(&health(&server), "queued"), 1, "target must be queued");
    let ack = server.run("/v1/cancel", &format!(r#"{{"cancel":"{cancel}"}}"#), "");
    assert_eq!(complete_http(&ack, 200), parse(r#"{"cancelled":true}"#));
    let target = server.finish(&mut target);
    assert_eq!(header(&target, "x-should-retry"), "false");
    let error = complete_http(&target, 409);
    assert_eq!(object(&object(&error)["error"])["type"], Value::String("client_cancelled".into()));
    assert!(!object(&error).contains_key("choices"));
    let target_allocated = server.ledger.lock().unwrap().attempts.contains_key(&17);
    assert!(!target_allocated, "queued cancel allocated target");
    server.hold.store(false, Ordering::Relaxed); server.finish(&mut active);
    let follower = server.run("/v1/completions", &request(17, 3, ""), "");
    ending(&follower, "length", 2, 3, 0);
    assert_eq!(payload(&follower), payload(&cold), "queued cancellation changed the cold follower");
    server.stop();
}

#[test]
fn cancel_capability_targets_only_its_request() {
    let server = Server::new();
    let cold = server.run("/v1/completions", &request(10, 100, r#","cache":false"#), "");
    let cancel = SessionKey::generate().unwrap().expose();
    server.hold.store(true, Ordering::Relaxed);
    let mut peer = server.send("/v1/completions", &request(10, 100, r#","cache":false"#), "");
    let mut target = server.send("/v1/completions", &request(17, 100, &format!(r#","cache":false,"cancel":"{cancel}""#)), "");
    server.until(|| server.ledger.lock().unwrap().attempts.get(&17) == Some(&1));
    let unknown = SessionKey::generate().unwrap().expose();
    let ack = server.run("/v1/cancel", &format!(r#"{{"cancel":"{unknown}"}}"#), "");
    assert_eq!(complete_http(&ack, 200), parse(r#"{"cancelled":true}"#));
    assert_eq!(count(&health(&server), "active"), 2);
    server.hold.store(false, Ordering::Relaxed);
    server.until(|| { target.poll(); !content(&target.wire, false).is_empty() });
    let mut first = server.send("/v1/cancel", &format!(r#"{{"cancel":"{cancel}"}}"#), "");
    let mut repeat = server.send("/v1/cancel", &format!(r#"{{"cancel":"{cancel}"}}"#), "");
    for ack in [&mut first, &mut repeat] { assert_eq!(complete_http(&server.finish(ack), 200), parse(r#"{"cancelled":true}"#)); }
    let target = server.finish(&mut target);
    let used = content(&target, false).len();
    assert!((1..100).contains(&used)); failure(&target, "client_cancelled", used);
    let peer = server.finish(&mut peer); ending(&peer, "length", 2, 100, 0);
    assert_eq!(payload(&peer), payload(&cold), "cancel changed sibling payload");
    server.stop();
}

#[test]
fn midstream_error_has_exact_terminal_shape_and_usage() {
    let server = Server::new();
    for (path, input) in [("/v1/completions", request(16, 20, r#","cache":false"#)), ("/v1/chat/completions", chat(16, r#","cache":false"#)),
        ("/v1/responses", r#"{"input":"16","max_output_tokens":20,"stream":true,"temperature":0,"cache":false}"#.into()),
        ("/v1/messages", r#"{"messages":[{"role":"user","content":"16"}],"max_tokens":20,"stream":true,"temperature":0,"cache":false}"#.into())] {
        let wire = server.run(path, &input, "");
        match path {
            "/v1/completions" | "/v1/chat/completions" => { failure(&wire, "generation_error", 2); assert_eq!(content(&wire, path.contains("chat")), "NN"); }
            "/v1/responses" => {
                assert_eq!(body(&wire).matches("event: response.failed\n").count(), 1);
                assert!(!wire.contains("event: response.completed"));
                let data = frames(&wire); let response = object(&object(data.last().unwrap())["response"]);
                assert_eq!(response["status"], Value::String("failed".into()));
                assert_eq!(response["error"], parse(r#"{"code":"generation_error","message":"generation stopped before completion"}"#), "Responses terminal error object");
                usage(&response["usage"], 2, 2, 0, true);
                let item = object(&response["output"].array().unwrap()[0]);
                assert_eq!(object(&item["content"].array().unwrap()[0])["text"], Value::String("NN".into()), "partial output lost");
            }
            _ => {
                assert_eq!(body(&wire).matches("event: error\n").count(), 1); assert!(!wire.contains("event: message_stop"));
                let data = frames(&wire);
                let text: String = data.iter().filter_map(|v| object(v).get("delta"))
                    .filter_map(|v| object(v).get("text")).map(|v| v.string().unwrap()).collect();
                assert_eq!(text, "NN", "Messages partial text lost");
                let error = object(data.last().unwrap());
                assert_eq!(error["type"], Value::String("error".into()));
                assert_eq!(error["error"], parse(r#"{"type":"api_error","message":"generation stopped before completion"}"#));
                assert!(wire.ends_with("\n\n") && body(&wire).rsplit("\n\n").nth(1).unwrap().starts_with("event: error\n"), "Messages error must be last");
            }
        }
    }
    let forced = r#","cache":false,"tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}],"tool_choice":"required""#;
    let wire = server.run("/v1/chat/completions", &chat(11, forced), "");
    failure(&wire, "invalid_model_output", 3);
    server.stop();
}

#[test]
fn successful_stream_schema_and_unique_ending() {
    let server = Server::new();
    let tool = r#","tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}]"#;
    for (include_usage, options) in [(None, ""), (Some(false), r#","stream_options":{"include_usage":false}"#),
        (Some(true), r#","stream_options":{"include_usage":true}"#)] {
        // Plain completions retain final-choice usage for every option; only chat has a role delta.
        let plain = server.run("/v1/completions", &request(11, 9, &format!(r#"{options},"cache":false"#)), "");
        ending(&plain, "stop", 2, 3, 0); assert_eq!(content(&plain, false), "ok");
        for value in frames(&plain) {
            assert_eq!(object(&value)["object"], Value::String("text_completion".into()));
            let choices = object(&value)["choices"].array().unwrap(); assert_eq!(choices.len(), 1);
            assert!(!object(&choices[0]).contains_key("delta")); assert!(!object(&choices[0]).contains_key("role"));
        }
        for (mode, expected, reason, committed) in [(11, "ok", "stop", 3), (13, "€!", "stop", 5),
            (14, "", "tool_calls", 66), (15, "ok", "stop", 21)] {
            let wire = server.run("/v1/chat/completions", &chat(mode, &format!(r#"{options}{tool},"cache":false"#)), "");
            // The script bytes are sampled once each; EOS is also billed.
            let committed = if mode == 14 { b"<tool_call>{\"name\":\"search\",\"arguments\":{\"x\":1}}</tool_call>".len() + 1 } else { committed };
            ending_with_usage(&wire, reason, 2, committed, 0, include_usage); assert_eq!(content(&wire, true), expected, "text or UTF-8 lost");
            let data = frames(&wire); let id = object(&data[0])["id"].clone(); let mut call_id = None; let mut arguments = String::new();
            for (at, value) in data.iter().enumerate() {
                let value = object(value);
                assert_eq!(value["object"], Value::String("chat.completion.chunk".into()));
                assert_eq!(value["id"], id); assert_eq!(value["model"], Value::String(spark_serve::MODEL_ID.into()));
                assert_eq!(value["created"], object(&data[0])["created"]);
                let choices = value["choices"].array().unwrap();
                if include_usage == Some(true) && at + 1 == data.len() { assert!(choices.is_empty()); continue; }
                assert_eq!(choices.len(), 1);
                let choice = object(&choices[0]); assert_eq!(choice["index"], number(0));
                let delta = object(&choice["delta"]);
                if at == 0 { assert_eq!(delta.get("role"), Some(&Value::String("assistant".into())), "first delta role"); }
                else { assert!(!delta.contains_key("role"), "role repeated after first delta"); }
                if let Some(calls) = delta.get("tool_calls") { for call in calls.array().unwrap() {
                    let call = object(call); assert_eq!(call["index"], number(0)); let function = object(&call["function"]);
                    if let Some(id) = call.get("id") {
                        assert!(call_id.is_none(), "duplicate tool start"); call_id = Some(id.clone());
                        assert_eq!(call["type"], Value::String("function".into()));
                        assert_eq!(function["name"], Value::String("search".into()));
                    }
                    if let Some(fragment) = function.get("arguments") { assert!(call_id.is_some(), "arguments before tool start"); arguments.push_str(fragment.string().unwrap()); }
                } }
            }
            if mode == 14 { assert!(call_id.is_some()); assert_eq!(arguments, r#"{"x":1}"#); }
            if mode == 15 {
                let reasoning: String = data.iter().flat_map(|v| object(v)["choices"].array().unwrap())
                    .filter_map(|v| object(&object(v)["delta"]).get("reasoning_content"))
                    .map(|v| v.string().unwrap()).collect();
                assert_eq!(reasoning, "why", "reasoning text lost");
            }
        }
    }
    server.stop();
}

#[test]
fn usage_matches_committed_tokens_on_all_endings() {
    let server = Server::new();
    for draft in ["plain", "strict"] {
    for (mode, max, extra, reason, completion, text) in [(11, 9, "", "stop", 3, "ok"),
        (12, 20, r#","stop":"STOP""#, "stop", 6, "ok"), (10, 3, "", "length", 3, "") ] {
        let cold = server.run("/v1/completions", &request(mode, max, &format!(r#"{extra},"cache":true,"kv_checkpoint":true,"cache_salt":"{draft}","draft":"{draft}""#)), "");
        ending(&cold, reason, 2, completion, 0);
        if !text.is_empty() { assert_eq!(content(&cold, false), text); }
        let cached = server.run("/v1/completions", &request(mode, max, &format!(r#"{extra},"cache":true,"cache_salt":"{draft}","draft":"{draft}""#)), "");
        ending(&cached, reason, 2, completion, 2);
        assert_eq!(payload(&cached), payload(&cold));
    }
    }
    let failed = server.run("/v1/completions", &request(16, 20, ""), ""); failure(&failed, "generation_error", 2);
    let plain = server.run("/v1/completions", &request(23, 16, r#","cache":false,"draft":"plain""#), "");
    let strict = server.run("/v1/completions", &request(23, 16, r#","cache":false,"draft":"strict""#), "");
    ending(&strict, "length", 2, 16, 0);
    assert_eq!(payload(&strict), payload(&plain), "rejected drafts changed output");
    assert!(server.ledger.lock().unwrap().rejected_proposals > 0, "no rejected draft exercised");
    for cached in [0, 2] {
        let cancel = SessionKey::generate().unwrap().expose();
        let mut client = server.send("/v1/completions", &request(24, 100, &format!(r#","cache":true,"cancel":"{cancel}""#)), "");
        server.until(|| { client.poll(); content(&client.wire, false).len() >= 2 });
        complete_http(&server.run("/v1/cancel", &format!(r#"{{"cancel":"{cancel}"}}"#), ""), 200);
        let wire = server.finish(&mut client);
        failure_cached(&wire, "client_cancelled", content(&wire, false).len(), cached);
        let wire = server.run("/v1/completions", &request(16, 20, r#","cache":true"#), "");
        // An earlier failure already kept this prompt. Both restores bill only accepted tokens.
        failure_cached(&wire, "generation_error", 2, 2);
    }
    assert!(server.ledger.lock().unwrap().restores.contains(&2), "no restored prompt work");
    server.stop();
}

#[test]
fn http_framing_refusals_never_allocate_or_split_requests() {
    let server = Server::new();
    let valid = request(10, 2, "");
    let heads = [format!("Content-Type: application/json\r\nContent-Length: {}\r\nContent-Length: {}\r\n", valid.len(), valid.len()),
        format!("Content-Type: application/json\r\nContent-Length: {}\r\nTransfer-Encoding: chunked\r\n", valid.len()),
        format!("Content-Type: application/json\r\nContent-Length: {}\r\nContent-Encoding: gzip\r\n", valid.len()),
        format!("Content-Length: {}\r\n", valid.len()),
        "Content-Type: application/json\r\nContent-Length: 4194305\r\n".into()];
    for (headers, message) in heads.iter().zip(["duplicate HTTP header", "Transfer-Encoding is unsupported", "Content-Encoding is unsupported", "Content-Type must be application/json", "body"]) {
        let before = server.ledger.lock().unwrap().opened;
        let mut client = server.fragmented("/v1/completions", &valid, headers);
        let wire = server.finish(&mut client); let value = complete_http(&wire, 400);
        assert!(object(&object(&value)["error"])["message"].string().unwrap().contains(message), "wrong framing refusal");
        assert_eq!(wire.matches("HTTP/1.1 ").count(), 1, "body became a second request");
        assert_eq!(server.ledger.lock().unwrap().opened, before, "invalid head allocated");
    }
    let before = server.ledger.lock().unwrap().opened;
    let headers = format!("Content-Type: application/json\r\nContent-Length: {}\r\n", valid.len());
    for trailing in ["extra", "POST /v1/completions HTTP/1.1\r\nContent-Length: 0\r\n\r\n"] {
        let mut client = server.raw("/v1/completions", &format!("{valid}{trailing}"), &headers);
        let wire = server.finish(&mut client);
        let error = complete_http(&wire, 400);
        assert_eq!(object(&object(&error)["error"])["message"], Value::String("unexpected trailing request bytes".into()));
        assert_eq!(wire.matches("HTTP/1.1 ").count(), 1, "trailing bytes made another reply");
        assert_eq!(server.ledger.lock().unwrap().opened, before, "trailing bytes allocated a session");
    }
    let too_large = server.run("/v1/completions", &request(10, 2, r#","cache":true,"cache_salt":"cannot-fit""#), "");
    let value = complete_http(&too_large, 503);
    assert_eq!(object(&object(&value)["error"])["type"], Value::String("capacity_exhausted".into()));
    assert_eq!(server.ledger.lock().unwrap().opened, 0, "cannot-fit request allocated");
    ending(&server.run("/v1/completions", &valid, ""), "length", 2, 2, 0);
    server.stop();
    let full = Server::start(8, 8, Limits { intake_burst: Duration::ZERO, ..Default::default() });
    full.hold.store(true, Ordering::Relaxed);
    let mut active: Vec<_> = (30..38).map(|mode| full.send("/v1/completions", &request(mode, 3, r#","cache":false"#), "")).collect();
    full.until(|| full.ledger.lock().unwrap().attempts.values().sum::<usize>() == 8);
    assert_eq!(count(&health(&full), "active"), 8);
    let ninth = full.run("/v1/completions", &valid, "");
    assert_eq!(complete_http(&ninth, 503), parse(r#"{"error":{"type":"overloaded","message":"connection capacity reached","param":null,"code":null}}"#));
    // Fill the remaining intake slot with a complete normal head. Its body waits
    // under the long configured bound; subsequent complete heads use control intake.
    let mut pending = full.raw("/v1/completions", "", &headers);
    full.pass();
    // Buffer all heads before any pass. Later readers use the control reserve.
    let mut refusals: Vec<_> = heads.iter().map(|headers| full.raw("/v1/completions", &valid, headers)).collect();
    for refused in &mut refusals {
        complete_http(&full.finish(refused), 400);
        assert_eq!(full.ledger.lock().unwrap().opened, 8, "full-table refusal allocated");
    }
    pending.append(valid.as_bytes());
    complete_http(&full.finish(&mut pending), 503);
    assert_eq!(count(&health(&full), "active"), 8, "control route lost admission");
    full.hold.store(false, Ordering::Relaxed);
    for client in &mut active { ending(&full.finish(client), "length", 2, 3, 0); }
    full.stop();
}
