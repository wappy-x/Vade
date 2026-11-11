use super::*;

#[test]
fn receipt_identity_is_optional_bounded_and_excludes_paths() {
    let metadata = json::parse(r#"{"build":{"git_sha":"abc","dirty":false,"source":"private/source"},"ptx_sha256":"def","layer_ptx_sha256":null,"checkpoint":{"sha256":"ghi","hash_state":"ready","directory":"private/model"},"server_start_unix_ms":123}"#).unwrap();
    let selected = identity(&metadata);
    assert_eq!(
        selected.get("build").unwrap().get("git_sha"),
        Some(&Value::Str("abc".into()))
    );
    assert_eq!(
        selected.get("checkpoint").unwrap().get("hash_state"),
        Some(&Value::Str("ready".into()))
    );
    assert_eq!(selected.get("server_start_unix_ms"), Some(&Value::Int(123)));
    assert!(!json::packed(&selected).contains("private"));
    let invalid = json::object(vec![
        ("ptx_sha256", "x".repeat(129).into()),
        ("server_start_unix_ms", (-1i64).into()),
        ("build", json::object(vec![("dirty", "false".into())])),
    ]);
    let selected = identity(&invalid);
    assert_eq!(selected.get("ptx_sha256"), Some(&Value::Null));
    assert_eq!(selected.get("server_start_unix_ms"), Some(&Value::Null));
    assert_eq!(
        selected.get("build").unwrap().get("dirty"),
        Some(&Value::Null)
    );
    assert!(json::packed(&identity(&Value::Null)).len() < 256);
    assert!(
        parse_stream(&transcript(false), false, 2)
            .unwrap()
            .identity
            .get("build")
            .is_some()
    );
}

fn transcript(on: bool) -> String {
    let draft = if on { "strict" } else { "plain" };
    format!(
        concat!(
            "data: {{\"model\":\"mock\",\"choices\":[{{\"index\":0,\"text\":\"café\",\"token_id\":4294967295,\"finish_reason\":null}}]}}\n\n",
            "data: {{\"model\":\"mock\",\"choices\":[{{\"index\":0,\"text\":\"\",\"token_id\":2,\"finish_reason\":null}}]}}\n\n",
            "data: {{\"model\":\"mock\",\"choices\":[{{\"index\":0,\"text\":\"\",\"finish_reason\":\"length\"}}],",
            "\"usage\":{{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}},",
            "\"vade\":{{\"draft\":\"{}\",\"mtp_depth\":2,\"thinking\":false,\"draft_used\":{},\"model_draft_used\":false,",
            "\"sampling\":{{\"temperature\":0,\"top_p\":1,\"top_k\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,\"seed\":0}}}}}}\n\n",
            "data: [DONE]\n\n"
        ),
        draft, on
    )
}

#[test]
fn every_id_and_effective_setting_is_required() {
    let source = transcript(true);
    let reply = parse_stream(&source, true, 2).unwrap();
    assert_eq!(reply.token_ids, [u32::MAX, 2]);
    assert_eq!(reply.token_texts, ["café", ""]);
    assert!(reply.draft_used);
    for broken in [
        source.replace(",\"token_id\":2", ""),
        source.replace("4294967295", "4294967296"),
        source.replace("4294967295", "1.5"),
        source.replace("\"temperature\":0", "\"temperature\":0.1"),
        source.replace("\"seed\":0", "\"seed\":1"),
        source.replace("\"total_tokens\":7", "\"total_tokens\":6"),
        source.replace("\"draft\":\"strict\"", "\"draft\":\"auto\""),
        source.replace("\"thinking\":false", "\"thinking\":true"),
        source.replace("\"draft_used\":true", "\"draft_used\":\"true\""),
        source.replace("\"finish_reason\":\"length\"", "\"finish_reason\":\"stop\""),
        source.replace("data: [DONE]\n\n", ""),
        source.trim_end_matches('\n').to_owned(),
    ] {
        assert!(
            parse_stream(&broken, true, 2).is_err(),
            "accepted poisoned transcript: {broken}"
        );
    }
    assert!(parse_stream(&transcript(false), false, 2).is_ok());
    assert!(
        parse_stream(
            &transcript(false).replace("\"draft_used\":false", "\"draft_used\":true"),
            false,
            2
        )
        .is_err()
    );
}

#[test]
fn strict_validation_precedes_permissive_receipt_json_parser() {
    let source = transcript(true);
    for broken in [
        source.replacen("\"index\":0", "\"index\":1,\"index\":0", 1),
        source.replacen("\"index\":0", "\"index\":00", 1),
        source.replacen("\"index\":0", "\"index\":0,\"ignored\":NaN", 1),
        source.replacen("\"index\":0", "\"index\":0,\"ignored\":1e999", 1),
        source.replacen("\"model\":\"mock\"", "\"model\":\"other\"", 1),
        format!("{source}data: {{}}\n\n"),
    ] {
        assert!(parse_stream(&broken, true, 2).is_err());
    }
}

#[test]
fn decoder_tail_keeps_last_token_fragment() {
    let source = transcript(false);
    let tail = "data: {\"model\":\"mock\",\"choices\":[{\"index\":0,\"text\":\"�\",\"finish_reason\":null}]}\n\n";
    let end = source.find("data: {\"model\":\"mock\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"length\"").unwrap();
    let source = format!("{}{tail}{}", &source[..end], &source[end..]);
    assert_eq!(parse_stream(&source, false, 2).unwrap().token_texts[1], "�");
    assert!(parse_stream(&format!("{tail}{source}"), false, 2).is_err());
}

#[test]
fn chunked_http_decodes_utf8_only_after_reassembly() {
    let body = transcript(false);
    let mut wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
    for byte in body.as_bytes() {
        wire.extend_from_slice(b"1;extension=ok\r\n");
        wire.push(*byte);
        wire.extend_from_slice(b"\r\n");
    }
    wire.extend_from_slice(b"0\r\nX-Check: complete\r\n\r\n");
    assert_eq!(receive(&mut io::Cursor::new(&wire)).unwrap(), body);
    wire.truncate(wire.len() - 2);
    assert!(receive(&mut io::Cursor::new(&wire)).is_err());
}

#[test]
fn rejects_truncated_and_ambiguous_http_bodies() {
    for wire in [
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 20\r\n\r\ndata: [DONE]\n\n",
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 0\r\nTransfer-Encoding: chunked\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\ny0\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 0\r\n\r\n",
        "HTTP/1.1 500 Oops\r\n\r\n",
    ] {
        assert!(
            receive(&mut io::Cursor::new(wire)).is_err(),
            "accepted {wire}"
        );
    }
}

#[test]
fn endpoint_validation_and_explicit_request_are_offline() {
    for url in [
        "http://localhost",
        "http://localhost:18080/v1/",
        "http://[::1]:8080/v1/completions",
    ] {
        assert!(Endpoint::parse(url).is_ok());
    }
    for url in [
        "https://localhost",
        "http://user:pass@localhost",
        "http://localhost/?secret=1",
        "http://localhost:0",
        "http://localhost:65536",
        "http://localhost:80\r\nInjected:1",
        "http://[::1]bad",
        "http://:80",
        "http://localhost/other",
    ] {
        assert!(Endpoint::parse(url).is_err(), "accepted {url}");
    }
    let request = request_body("quote: \"\n雪", true, 2);
    assert_eq!(
        crate::json::string(&request, "prompt").unwrap(),
        "quote: \"\n雪"
    );
    assert_eq!(crate::json::string(&request, "draft").unwrap(), "strict");
    assert_eq!(
        json::parse(&request).unwrap().get("ignore_eos"),
        Some(&Value::Bool(true))
    );
}

#[test]
fn endpoint_identity_normalizes_authority_and_api_suffixes() {
    for (a, b) in [
        ("http://LOCALHOST", "http://localhost:00080/v1/completions"),
        ("http://[0:0:0:0:0:0:0:1]/v1", "http://[::1]:80"),
        (
            "http://127.0.0.1:8080/",
            "http://127.0.0.1:8080/v1/completions/",
        ),
    ] {
        assert!(
            Endpoint::parse(a)
                .unwrap()
                .same_address(&Endpoint::parse(b).unwrap())
        );
    }
    assert!(
        !Endpoint::parse("http://localhost:8080")
            .unwrap()
            .same_address(&Endpoint::parse("http://localhost:8081").unwrap())
    );
}

#[test]
fn stalled_dns_cannot_exceed_the_request_deadline() {
    let (release, wait) = std::sync::mpsc::channel();
    let (done, completed) = std::sync::mpsc::channel();
    let result = resolve_before(Instant::now() + Duration::from_millis(20), move || {
        wait.recv().unwrap();
        done.send(()).unwrap();
        Ok(vec!["127.0.0.1:80".parse().unwrap()])
    });
    assert!(result.unwrap_err().contains("deadline exceeded"));
    release.send(()).unwrap();
    completed.recv_timeout(Duration::from_secs(2)).unwrap();
    let address = "127.0.0.1:80".parse().unwrap();
    assert_eq!(
        resolve_before(Instant::now() + Duration::from_secs(2), move || Ok(vec![
            address
        ]))
        .unwrap(),
        [address]
    );
}
