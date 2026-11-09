use super::*;
use std::net::TcpListener;
use std::thread;
use std::time::Instant;

fn server(replies: Vec<(&'static str, String)>) -> (String, thread::JoinHandle<Vec<(String, Option<Value>)>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let worker = thread::spawn(move || {
        let mut requests = Vec::new();
        for (content_type, reply) in replies {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
                    Err(error) => panic!("accept: {error}"),
                }
            };
            // macOS can inherit the listener's nonblocking mode on accept.
            socket.set_nonblocking(false).unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            socket.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut first = String::new();
            reader.read_line(&mut first).unwrap();
            let mut length = 0;
            loop {
                let mut line = String::new();
                assert!(reader.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" { break; }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut bytes = vec![0; length];
            reader.read_exact(&mut bytes).unwrap();
            requests.push((first, if bytes.is_empty() { None } else { Some(json::parse(std::str::from_utf8(&bytes).unwrap()).unwrap()) }));
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).unwrap();
        }
        requests
    });
    (url, worker)
}

fn args(url: &str, root: &Path) -> LongArgs {
    LongArgs::parse(&[
        "run".into(), "--manifest".into(), root.join("manifest.json").to_str().unwrap().into(),
        "--a-url".into(), url.into(), "--b-url".into(), url.into(),
        "--output".into(), root.join("capture").to_str().unwrap().into(), "--top-k".into(), "1".into(),
    ]).unwrap()
}

#[test]
fn legacy_frozen_bodies_dispatch_seedless_for_retrieval_and_scoring() {
    let root = crate::test_gate::TempDir::new("vade-legacy-greedy-capture").unwrap();
    let mut body = json::object(sampling());
    body.insert("model", "qwen3.8-flash-next-fp8".into());
    body.insert("seed", 42i64.into());
    body.insert("prompt", "frozen prompt".into());
    body.insert("draft", "plain".into());
    body.insert("cache", Value::Bool(false));
    let frozen = json::packed(&body);
    fs::write(root.path.join("legacy.body.json"), &frozen).unwrap();
    let cases = ["teacher", "retrieval"].into_iter().map(|kind| json::object(vec![
        ("id", kind.into()), ("kind", kind.into()), ("body_file", "legacy.body.json".into()),
        ("length", 1i64.into()), ("prompt_sha256", sha(b"frozen prompt").into()),
        ("input_ids_sha256", sha(&7u32.to_le_bytes()).into()),
    ])).collect();
    let manifest = json::object(vec![
        ("cases", Value::Array(cases)), ("max_tokens", 1i64.into()), ("checkpoint_sha256", CHECKPOINT.into()),
    ]);
    // Pin the mock independently: changing production MODEL must break this legacy capture.
    let capabilities = r#"{"model":"qwen3.8-flash-next-fp8","context_limit":131073,"identity":{"weights":"frozen","tokenizer_fingerprint":"frozen"}}"#.to_owned();
    let scoring = concat!(
        "{\"object\":\"vade.score.v1\",\"path\":\"prefill_all\",\"prompt_tokens\":1,\"top_k\":1}\n",
        "{\"position\":0,\"input_id\":7,\"top1_id\":7,\"topk\":[[7,1]],\"scores\":[[7,1]]}\n",
        "{\"object\":\"vade.score.end\",\"rows\":1}\n",
    ).to_owned();
    let retrieval = format!(r#"{{"choices":[{{"text":"answer"}}],"usage":{{"prompt_tokens":1}},"vade":{{"sampling":{},"draft":"plain","mtp_depth":0,"thinking":false}}}}"#, json::packed(&json::object(sampling())));
    let replies = vec![("application/json", capabilities), ("application/x-ndjson", scoring), ("application/json", retrieval)];
    let (url, worker) = server([replies.clone(), replies].concat());
    let result = run(&mut args(&url, &root.path), &manifest);
    let requests = worker.join().unwrap();
    result.unwrap();
    assert_eq!(requests.len(), 6);
    for (index, (line, sent)) in requests.iter().enumerate() {
        if index % 3 == 0 { assert!(sent.is_none()); continue; }
        assert!(line.starts_with(if index % 3 == 1 { "POST /v1/score " } else { "POST /v1/completions " }));
        let sent = sent.as_ref().unwrap();
        assert_eq!(sent["model"].as_str(), Some("qwen3.8-flash-next-fp8"));
        assert!(sent.get("seed").is_none(), "{sent:?}");
        assert!(greedy(sent));
        assert_eq!(sent["prompt"], body["prompt"]);
        if index == 4 { assert_eq!(sent["score_ids"].as_str(), Some("7")); }
    }
    assert_eq!(fs::read_to_string(root.path.join("legacy.body.json")).unwrap(), frozen);
}

#[test]
fn dispatch_preserves_explicit_sampled_seeds() {
    let root = crate::test_gate::TempDir::new("vade-sampled-capture").unwrap();
    let (url, worker) = server(vec![("application/json", "{}".into())]);
    let body = json::parse(r#"{"temperature":0.7,"seed":713,"prompt":"sampled"}"#).unwrap();
    http_json(&url, "/v1/completions", Some(&body), &args(&url, &root.path)).unwrap();
    assert_eq!(worker.join().unwrap()[0].1.as_ref(), Some(&body));
}
