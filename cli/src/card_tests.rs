use super::*;
use std::{
    net::TcpListener,
    sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc, Mutex},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn strings(arguments: &[&str]) -> Vec<String> { arguments.iter().map(|arg| (*arg).to_owned()).collect() }

/// A card file with these 66 token ID arrays in receipt order, laid out the
/// way the RigMark cards and this client save them (decode in send order).
fn card_from(ids: &[Vec<i64>]) -> Value {
    let row = |ids: &Vec<i64>| object(vec![("token_ids", Value::Array(ids.iter().map(|id| Value::Int(*id)).collect()))]);
    let runs = |rows: &[Vec<i64>]| object(vec![("runs", Value::Array(rows.iter().map(row).collect()))]);
    let mut concurrency = object(vec![]);
    let mut next = 21;
    for level in LEVELS {
        let rounds = (0..ROUNDS).map(|_| { let streams = &ids[next..next + level]; next += level; object(vec![("streams", Value::Array(streams.iter().map(row).collect()))]) }).collect();
        concurrency.insert(level.to_string(), object(vec![("rounds", Value::Array(rounds))]));
    }
    object(vec![
        ("card_draft", "served".into()),
        ("decode", object(vec![("code", runs(&ids[5..10])), ("prose", runs(&ids[0..5])), ("structured", runs(&ids[10..15]))])),
        ("prefill", object(vec![("8192", object(vec![("cold", runs(&ids[15..18])), ("warm_replay", runs(&ids[18..21]))]))])),
        ("concurrency", concurrency),
    ])
}

fn known_rows() -> Vec<Vec<i64>> { (0..66).map(|i| vec![i, 1000 + i, 151643]).collect() }

#[test]
fn the_digest_is_sha256_of_compact_json_in_receipt_order() {
    // Expected values from Python: sha256(json.dumps(rows, separators=(",", ":"))).
    let mut rows = known_rows();
    assert_eq!(digest(&card_from(&rows)).unwrap(), "c570bc144c4bffd2529ad123dffad71c01c658a58a68f445b1aa66b6f424de42");
    rows[40] = vec![40, 1040, 7];
    assert_eq!(digest(&card_from(&rows)).unwrap(), "4823b5d616265b08e0acee63c272bba49516c968567423ebcd5896fd4b81bf9f");
    let card = card_from(&rows);
    let labels: Vec<String> = token_rows(&card).unwrap().into_iter().map(|(label, _)| label).collect();
    assert_eq!(RESPONSES, 66);
    assert_eq!((labels[0].as_str(), labels[5].as_str(), labels[10].as_str()), ("prose run 1", "code run 1", "structured run 1"));
    assert_eq!((labels[15].as_str(), labels[18].as_str(), labels[21].as_str()), ("cold 8K run 1", "warm 8K run 1", "C1 round 1 stream 1"));
    assert_eq!((labels[24].as_str(), labels[65].as_str()), ("C2 round 1 stream 1", "C8 round 3 stream 8"));
}

#[test]
fn an_incomplete_card_has_no_digest() {
    let mut card = card_from(&known_rows());
    card.insert("concurrency", object(vec![]));
    assert_eq!(digest(&card).unwrap_err(), "the card has no C1 rounds; it is incomplete");
    let mut card = card_from(&known_rows());
    card.insert("prefill", object(vec![]));
    assert_eq!(digest(&card).unwrap_err(), "the card has no cold 8K runs; it is incomplete");
}

#[test]
fn compare_counts_identical_arrays_or_matches_a_receipt_digest() {
    let rows = known_rows();
    let card = card_from(&rows);
    let verdict = compare(&card, &card).unwrap();
    assert!(verdict.identical);
    assert!(verdict.lines.contains(&"ids 66/66 identical".to_owned()), "{:?}", verdict.lines);
    let mut other = rows.clone();
    other[40] = vec![40, 1040, 7, 8];
    let mut other = card_from(&other);
    other.insert("card_draft", "plain".into());
    let verdict = compare(&card, &other).unwrap();
    assert!(!verdict.identical);
    assert!(verdict.lines.contains(&"ids 65/66 identical".to_owned()), "{:?}", verdict.lines);
    assert!(verdict.lines.contains(&"First difference: C4 round 3 stream 3, token 2 (3 and 4 tokens).".to_owned()), "{:?}", verdict.lines);
    assert!(verdict.lines.iter().any(|line| line == "Note: this card's chats were sent as served (no draft field); the other card's were sent with \"draft\":\"plain\". Drafting ON and OFF must give the same token IDs."));
    let receipt = |sha: &str| object(vec![("card_draft", "served".into()), ("responses", 66usize.into()), ("token_ids_sha256", sha.into())]);
    let verdict = compare(&card, &receipt("c570bc144c4bffd2529ad123dffad71c01c658a58a68f445b1aa66b6f424de42")).unwrap();
    assert!(verdict.identical && verdict.lines.contains(&"ids 66/66 identical".to_owned()), "{:?}", verdict.lines);
    let verdict = compare(&card, &receipt("805f06d0a7e80c33d3c311b65178aadc29de800a5750602e3dbdc83c60fd006a")).unwrap();
    assert!(!verdict.identical && verdict.lines.iter().any(|line| line.starts_with("ids differ from the receipt.")), "{:?}", verdict.lines);
    assert!(compare(&card, &object(vec![])).is_err());
}

#[test]
fn nonces_match_the_recorded_card_requests() {
    // The user messages of the 2026-10-31 one-Spark card begin with these nonces.
    assert_eq!(nonce(&["decode", "code", "1"]), "0ec96af2ba9fe6a86208aaebe21a7796");
    assert_eq!(nonce(&["concurrency", "code", "4", "2", "3"]), "07cb84b8e0dc63e19a51d00bca7d95e4");
}

#[test]
fn glm_bodies_ask_low_effort_and_never_thinking_off() {
    let glm = json::packed(&chat_payload("glm-5.3-flash-fp4-local", "S", "U", 8192, SEED, true));
    assert_eq!(glm, r#"{"model":"glm-5.3-flash-fp4-local","messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}],"temperature":0.0,"top_p":1.0,"seed":20260905,"max_tokens":8192,"stream":true,"stream_options":{"include_usage":true},"reasoning_effort":"low","top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"cache":false,"mtp_depth":0,"draft":"plain"}"#);
    assert!(!glm.contains("enable_thinking"));
    let qwen = json::packed(&chat_payload("qwen3.8-flash-next-fp8", "S", "U", 8192, SEED, true));
    assert!(qwen.contains(r#""chat_template_kwargs":{"enable_thinking":false}"#));
    assert!(!qwen.contains("reasoning_effort") && !qwen.contains("mtp_depth"));
    assert!(is_glm_model("GLM-5.3-Flash-FP4-Local") && !is_glm_model("qwen3.8-flash-next-fp8"));
}

#[test]
fn request_bodies_carry_the_card_settings() {
    let served = json::packed(&chat_payload("m", "S", "U", 8192, SEED, false));
    assert_eq!(served, r#"{"model":"m","messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}],"temperature":0.0,"top_p":1.0,"seed":20260905,"max_tokens":8192,"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":false},"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"cache":false}"#);
    assert!(json::packed(&chat_payload("m", "S", "U", 256, SEED + 1, true)).ends_with(r#""cache":false,"draft":"plain"}"#));
    assert_eq!(json::packed(&prefill_payload("m", &[1, 2], "salt")), r#"{"model":"m","prompt":[1,2],"add_special_tokens":false,"max_tokens":8,"ignore_eos":true,"temperature":0.0,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0,"draft":"plain","stream":true,"stream_options":{"include_usage":true},"cache_salt":"salt"}"#);
    assert_eq!(prefill_tokens(&[9, 8], &[1, 2, 3], 7).unwrap(), [9, 8, 1, 2, 3, 1, 2]);
    assert!(prefill_tokens(&[1; 9], &[1], 8).is_err());
    assert!(prefill_tokens(&[], &[1], 8).is_err());
}

#[test]
fn the_stream_parser_reads_token_ids_usage_and_the_echo() {
    // Lines in the server's own format (engine/crates/serve/src/http.rs, sse_chunk).
    let payload = chat_payload("m", "S", "U", 8, SEED, false);
    let echo = r#"{"sampling":{"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":20260905},"thinking":false,"build":{"git_sha":"abc"}}"#;
    let mut events = Events::default();
    let lines = [
        r#"data: {"id":"completion-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null,"token_id":9707}]}"#.to_owned(),
        String::new(),
        r#"data: {"id":"completion-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null,"token_id":385}]}"#.to_owned(),
        format!(r#"data: {{"id":"completion-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":20,"completion_tokens":3,"total_tokens":23}},"vade":{echo}}}"#),
    ];
    for (at, line) in lines.iter().enumerate() { assert!(!events.line(&format!("{line}\n"), 0.5 + at as f64).unwrap()); }
    assert!(events.line("data: [DONE]\n", 4.0).unwrap());
    let row = events.row(&payload, 4.5).unwrap();
    assert_eq!(json::packed(&row["token_ids"]), "[9707,385]");
    assert_eq!((row["output"].as_str(), row["finish_reason"].as_str(), row["completion_tokens"].as_i64()), (Some("Hello"), Some("stop"), Some(3)));
    assert_eq!((row["ttft_seconds"].as_f64(), row["decode_seconds"].as_f64(), row["decode_tokens_per_second"].as_f64()), (Some(0.5), Some(2.0), Some(1.0)));
    assert_eq!(row["stream_done_marker"], Value::Bool(true));
    let mut bad = Events::default();
    bad.line(&lines[0], 0.1).unwrap();
    assert_eq!(bad.row(&payload, 1.0).unwrap_err(), "the server sent no final token usage");
    assert!(Events::default().line(r#"data: {"choices":[{"index":0,"token_id":"x"}]}"#, 0.0).is_err());
    assert!(Events::default().line(r#"data: {"choices":[{"index":1}]}"#, 0.0).is_err());
}

#[test]
fn the_echo_must_match_the_request() {
    let payload = chat_payload("m", "S", "U", 8, SEED, false);
    let echo = |temperature: &str, thinking: bool| json::parse(&format!(r#"{{"sampling":{{"temperature":{temperature},"top_k":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":20260905}},"thinking":{thinking}}}"#)).unwrap();
    assert!(check_echo(&payload, &echo("0", false)).is_ok());
    assert_eq!(check_echo(&payload, &echo("0.7", false)).unwrap_err(), "the server used temperature 0.7 for a request that sent 0.0");
    assert_eq!(check_echo(&payload, &echo("0", true)).unwrap_err(), "the server did not use the requested thinking mode");
    // The 8K prompt sends no seed and no thinking field; the server picks its own seed.
    let prefill = prefill_payload("m", &[1], "s");
    assert!(check_echo(&prefill, &json::parse(r#"{"sampling":{"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":17951147121147274963}}"#).unwrap()).is_ok());
}

#[test]
fn the_structured_gate_needs_fifty_squares() {
    let output = |count: i64, bad: i64| format!("[{}]", (1..=count).map(|i| format!(r#"{{"index":{i},"square":{}}}"#, if i == bad { 0 } else { i * i })).collect::<Vec<_>>().join(","));
    let row = |output: String| object(vec![("finish_reason", "stop".into()), ("stream_done_marker", true.into()), ("output", output.into())]);
    assert!(structured_gate(&row(output(50, 0))).is_ok());
    assert_eq!(structured_gate(&row(output(49, 0))).unwrap_err(), "expected an array of 50 objects");
    assert_eq!(structured_gate(&row(output(50, 7))).unwrap_err(), "incorrect object at index 7");
    assert!(structured_gate(&row("not json".into())).unwrap_err().starts_with("invalid JSON"));
}

#[test]
fn options_and_prompts_are_checked() {
    assert_eq!(parse(&strings(&["--card", "--prompts", "p.json", "--draft", "plain", "--url", "http://127.0.0.1:1"])).unwrap(),
        Options { url: Some("http://127.0.0.1:1".into()), prompts: Some("p.json".into()), draft_plain: true, ..Options::default() });
    assert!(parse(&strings(&["--card"])).unwrap_err().starts_with("Give --prompts the RigMark prompts.json at revision 52c29c3."));
    assert_eq!(parse(&strings(&["--card", "--prompts", "p", "--draft", "strict"])).unwrap_err(), "With --card, --draft must be plain or served. Run vade bench --help.");
    assert_eq!(parse(&strings(&["--card", "--bogus"])).unwrap_err(), "Unknown option \"--bogus\" for vade bench --card. Run vade bench --help.");
    assert_eq!(parse(&strings(&["--card", "--compare", "a"])).unwrap_err(), "Give --compare two card or receipt files. Run vade bench --help.");
    assert!(parse(&strings(&["--card", "--compare", "a", "b", "--prompts", "p"])).is_err());
    assert_eq!(parse(&strings(&["--card", "--compare", "a", "b"])).unwrap().compare, Some(("a".into(), "b".into())));
    assert!(load_prompts(br#"{"version":"1.0.0"}"#).unwrap_err().contains(PROMPTS_SHA256));
    assert!(Server::parse("https://example.test").is_err());
    assert!(Server::parse("http://user@127.0.0.1:8").is_err());
    assert_eq!(Server::parse("http://127.0.0.1:18080/v1").unwrap().authority, "127.0.0.1:18080");
    assert_eq!(Server::parse("http://localhost").unwrap().authority, "localhost:80");
}

/// A fake server: answers the card's endpoints, streams token IDs derived from
/// each request body, and records every request in arrival order.
struct Fake { url: String, seen: Arc<Mutex<Vec<(String, Value)>>>, stop: Arc<AtomicBool>, task: Option<std::thread::JoinHandle<()>> }

impl Fake {
    fn start() -> Self { Self::refusing_after("/v1/chat/completions", usize::MAX) }

    /// A fake that answers HTTP 500 to every request for `path` after the first `count`.
    fn refusing_after(path: &'static str, count: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (seen, stop) = (Arc::new(Mutex::new(Vec::new())), Arc::new(AtomicBool::new(false)));
        let task = { let (seen, stop) = (seen.clone(), stop.clone()); std::thread::spawn(move || while !stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => { let seen = seen.clone(); std::thread::spawn(move || serve(stream, &seen, path, count)); }
                Err(_) => std::thread::sleep(Duration::from_millis(2)),
            }
        }) };
        Self { url, seen, stop, task: Some(task) }
    }
}

impl Drop for Fake {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); let _ = self.task.take().unwrap().join(); }
}

/// The IDs the fake streams for a body: two tokens from its SHA-256.
fn fake_ids(body: &str) -> [i64; 2] {
    let sha = crate::sha256::sha(body.as_bytes());
    [i64::from_str_radix(&sha[..6], 16).unwrap(), i64::from_str_radix(&sha[6..12], 16).unwrap()]
}

fn serve(stream: TcpStream, seen: &Mutex<Vec<(String, Value)>>, refuse: &str, count: usize) {
    stream.set_nonblocking(false).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let path = line.split_whitespace().nth(1).unwrap().to_owned();
    let mut length = 0;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        if header.trim().is_empty() { break; }
        if let Some(n) = header.to_ascii_lowercase().strip_prefix("content-length:") { length = n.trim().parse().unwrap(); }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let text = String::from_utf8(body).unwrap();
    let request = if text.is_empty() { Value::Null } else { json::parse(&text).unwrap() };
    let refused = {
        let mut seen = seen.lock().unwrap();
        seen.push((path.clone(), request.clone()));
        path == refuse && seen.iter().filter(|(p, _)| p == refuse).count() > count
    };
    let mut stream = stream;
    if refused {
        let _ = stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
        return;
    }
    let json_reply = |body: &str| format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    let reply = match path.as_str() {
        "/v1/models" => json_reply(r#"{"object":"list","data":[{"id":"fake-model","object":"model"}]}"#),
        "/health" => json_reply(r#"{"status":"ok"}"#),
        "/tokenize" => json_reply(&format!(r#"{{"tokens":[{}]}}"#, request["prompt"].as_str().unwrap().bytes().map(|b| b.to_string()).collect::<Vec<_>>().join(","))),
        _ => {
            let chat = path == "/v1/chat/completions";
            let structured = chat && request["messages"][1]["content"].as_str().unwrap().ends_with("Return JSON.");
            let ids = fake_ids(&text);
            let parts = if structured { ["[{\"index\":1,\"square\":1}".to_owned(), "]".to_owned()] } else { ["Hel".to_owned(), "lo".to_owned()] };
            let chunk = |text: &str, id: i64| if chat { format!(r#"{{"choices":[{{"index":0,"delta":{{"content":{}}},"finish_reason":null,"token_id":{id}}}]}}"#, json::packed(&text.into())) }
                else { format!(r#"{{"choices":[{{"index":0,"text":{},"finish_reason":null,"token_id":{id}}}]}}"#, json::packed(&text.into())) };
            let sampling = ["temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty", "frequency_penalty", "seed"].iter()
                .map(|key| format!("\"{key}\":{}", request.get(key).map_or("7".into(), json::packed))).collect::<Vec<_>>().join(",");
            let prompt_tokens = request["prompt"].as_array().map_or(20, Vec::len);
            let last = format!(r#"{{"choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":3,"prompt_tokens_details":{{"cached_tokens":0}}}},"vade":{{"sampling":{{{sampling}}},"thinking":false,"build":{{"git_sha":"fakebuild"}}}}}}"#);
            let first = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {}\n\n", chunk(&parts[0], ids[0]));
            stream.write_all(first.as_bytes()).unwrap();
            // Two events must arrive at different times for a decode rate.
            std::thread::sleep(Duration::from_millis(2));
            format!("data: {}\n\ndata: {last}\n\ndata: [DONE]\n\n", chunk(&parts[1], ids[1]))
        }
    };
    let _ = stream.write_all(reply.as_bytes());
}

#[test]
fn a_fake_server_gets_the_66_requests_in_card_order() {
    let fake = Fake::start();
    let server = Server::parse(&fake.url).unwrap();
    let prompts = Prompts { version: "1.0.0".into(), system: "S".into(), prefill_unit: "unit text ".into(),
        workloads: vec![("code".into(), "Write code.".into()), ("prose".into(), "Write prose.".into()), ("structured".into(), "Return JSON.".into())] };
    let mut card = object(vec![]);
    run_card(&server, &prompts, false, &mut card, || false).unwrap();
    let seen = fake.seen.lock().unwrap().clone();
    let streams: Vec<&(String, Value)> = seen.iter().filter(|(path, _)| path.starts_with("/v1/") && path != "/v1/models").collect();
    assert_eq!(streams.len(), RESPONSES);
    assert_eq!(seen.iter().filter(|(path, _)| path == "/tokenize").count(), 1 + PREFILL_RUNS);
    let user = |row: &Value| row["messages"][1]["content"].as_str().unwrap().to_owned();
    for (index, (name, run)) in ["code", "prose", "structured"].iter().flat_map(|name| (1..=5).map(move |run| (*name, run))).enumerate() {
        assert_eq!(streams[index].0, "/v1/chat/completions");
        assert!(user(&streams[index].1).starts_with(&format!("Request nonce: {}\n\n", nonce(&["decode", name, &run.to_string()]))), "{index}");
        assert_eq!(streams[index].1.get("draft"), None);
    }
    for run in 0..PREFILL_RUNS {
        let (cold, warm) = (&streams[15 + 2 * run].1, &streams[16 + 2 * run].1);
        assert_eq!(cold, warm, "the warm request repeats the cold one");
        assert_eq!(cold["prompt"].as_array().unwrap().len(), 8192);
    }
    assert_ne!(streams[15].1["cache_salt"], streams[17].1["cache_salt"]);
    let mut next = 21;
    for level in LEVELS {
        for round in 1..=ROUNDS {
            let mut seeds: Vec<i64> = streams[next..next + level].iter().map(|(_, row)| { assert_eq!(row["max_tokens"].as_i64(), Some(256)); row["seed"].as_i64().unwrap() }).collect();
            seeds.sort_unstable();
            assert_eq!(seeds, (0..level as i64).map(|s| SEED + s).collect::<Vec<_>>(), "C{level} round {round}");
            next += level;
        }
    }
    // Every saved array is what the server streamed for that request, and the
    // card digest is taken over them in receipt order.
    let rows = token_rows(&card).unwrap();
    let expected: Vec<Value> = rows.iter().map(|(_, ids)| (*ids).clone()).collect();
    for (label, ids) in &rows { assert_eq!(ids.as_array().unwrap().len(), 2, "{label}"); }
    let decode_code_1 = &card["decode"]["code"]["runs"][0];
    assert_eq!(json::packed(&decode_code_1["token_ids"]), format!("{:?}", fake_ids(&json::packed(&decode_code_1["request_body"]))).replace(' ', ""));
    assert_eq!(digest(&card).unwrap(), crate::sha256::sha(json::packed(&Value::Array(expected)).as_bytes()));
    assert_eq!(card["decode"]["code"]["runs"][0]["completion_validation"]["valid"], Value::Bool(true));
    assert_eq!(card["decode"]["structured"]["runs"][0]["completion_validation"]["valid"], Value::Bool(false));
    assert_eq!(card["model"].as_str(), Some("fake-model"));
    assert!(summary(&card).ends_with("Output checks 10/15."), "{}", summary(&card));
}

#[test]
fn a_plain_card_sends_draft_plain_in_every_chat() {
    let fake = Fake::start();
    let prompts = Prompts { version: "1.0.0".into(), system: "S".into(), prefill_unit: "u".into(),
        workloads: vec![("code".into(), "c".into()), ("prose".into(), "p".into()), ("structured".into(), "s".into())] };
    run_card(&Server::parse(&fake.url).unwrap(), &prompts, true, &mut object(vec![]), || false).unwrap();
    let seen = fake.seen.lock().unwrap().clone();
    let chats: Vec<&Value> = seen.iter().filter(|(path, _)| path == "/v1/chat/completions").map(|(_, row)| row).collect();
    assert_eq!(chats.len(), 60);
    assert!(chats.iter().all(|row| row["draft"].as_str() == Some("plain")));
}

#[test]
fn an_interrupt_stops_before_the_next_request() {
    let fake = Fake::start();
    let prompts = Prompts { version: "1.0.0".into(), system: "S".into(), prefill_unit: "u".into(),
        workloads: vec![("code".into(), "c".into()), ("prose".into(), "p".into()), ("structured".into(), "s".into())] };
    assert_eq!(run_card(&Server::parse(&fake.url).unwrap(), &prompts, false, &mut object(vec![]), || true).unwrap_err(), "Interrupted.");
    assert!(fake.seen.lock().unwrap().iter().all(|(path, _)| !path.contains("completions")));
}

#[test]
fn the_card_file_holds_the_digest_and_the_verdict_matches_a_receipt() {
    let fake = Fake::start();
    let directory = std::env::temp_dir().join(format!("vade-card-test-{}-{}-{}", std::process::id(), now(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&directory).unwrap();
    let prompts = Prompts { version: "1.0.0".into(), system: "S".into(), prefill_unit: "u".into(),
        workloads: vec![("code".into(), "c".into()), ("prose".into(), "p".into()), ("structured".into(), "s".into())] };
    let first = directory.join("first.json");
    let options = |output: &Path, receipt: Option<PathBuf>| Options { url: Some(fake.url.clone()), output: Some(output.into()), receipt, ..Options::default() };
    assert_eq!(card_file(&options(&first, None), &prompts, || false).unwrap(), 0);
    let card = json::read(&first).unwrap();
    assert_eq!((card["complete"].clone(), card["responses"].as_i64()), (Value::Bool(true), Some(66)));
    assert_eq!(card["token_ids_sha256"].as_str().unwrap(), digest(&card).unwrap());
    let (before, bytes) = (fake.seen.lock().unwrap().len(), std::fs::read(&first).unwrap());
    assert!(card_file(&options(&first, None), &prompts, || false).unwrap_err().contains("exists"), "never overwrites");
    assert_eq!((fake.seen.lock().unwrap().len(), std::fs::read(&first).unwrap()), (before, bytes), "no request, file unchanged");
    // The fake streams the same IDs for the same body, but the 8K requests carry a fresh salt.
    let receipt = directory.join("receipt.json");
    std::fs::write(&receipt, format!(r#"{{"responses":66,"token_ids_sha256":"{}"}}"#, card["token_ids_sha256"].as_str().unwrap())).unwrap();
    assert_eq!(card_file(&options(&directory.join("second.json"), Some(receipt)), &prompts, || false).unwrap(), 1);
    assert_eq!(card_file(&options(&directory.join("third.json"), Some(first)), &prompts, || false).unwrap(), 1);
    let third = json::read(&directory.join("third.json")).unwrap();
    let verdict = compare(&third, &card).unwrap();
    assert!(verdict.lines.contains(&"ids 60/66 identical".to_owned()), "{:?}", verdict.lines);
    assert!(verdict.lines.contains(&"First difference: cold 8K run 1, token 0 (2 and 2 tokens).".to_owned()), "{:?}", verdict.lines);
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn a_reply_must_carry_one_token_id_per_token() {
    // On the recorded 2026-10-31 cards, a reply that stops has one ID fewer than
    // completion_tokens (the stop token has none), and a capped reply has one per token.
    let payload = chat_payload("m", "S", "U", 8, SEED, false);
    let echo = r#""vade":{"sampling":{"temperature":0,"top_k":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"seed":20260905},"thinking":false}"#;
    let reply = |ids: &[i64], completion: usize, finish: &str| {
        let mut events = Events::default();
        for (at, id) in ids.iter().enumerate() {
            events.line(&format!(r#"data: {{"choices":[{{"index":0,"delta":{{"content":"t"}},"finish_reason":null,"token_id":{id}}}]}}"#), at as f64).unwrap();
        }
        // A reply with no IDs still needs measurable text to reach the count check.
        if ids.is_empty() { for at in 0..2 { events.line(r#"data: {"choices":[{"index":0,"delta":{"content":"t"},"finish_reason":null}]}"#, at as f64).unwrap(); } }
        events.line(&format!(r#"data: {{"choices":[{{"index":0,"delta":{{}},"finish_reason":"{finish}"}}],"usage":{{"prompt_tokens":5,"completion_tokens":{completion}}},{echo}}}"#), 9.0).unwrap();
        events.row(&payload, 10.0)
    };
    assert!(reply(&[1, 2, 3], 3, "length").is_ok());
    assert!(reply(&[1, 2], 3, "stop").is_ok());
    assert_eq!(reply(&[1, 2], 3, "length").unwrap_err(), "the server sent 2 token IDs for 3 completion tokens (finish reason length)");
    assert_eq!(reply(&[], 2, "stop").unwrap_err(), "the server sent 0 token IDs for 2 completion tokens (finish reason stop)");
    assert!(reply(&[1, 2, 3, 4], 3, "stop").is_err());
}

#[test]
fn a_card_with_empty_arrays_or_wrong_sections_does_not_compare() {
    // Two tokenless cards must not read as 66/66 identical.
    let empty = card_from(&vec![Vec::new(); 66]);
    assert_eq!(compare(&empty, &empty).unwrap_err(), "the card has no token IDs for prose run 1");
    let mut rows = known_rows();
    rows[30] = Vec::new();
    assert_eq!(digest(&card_from(&rows)).unwrap_err(), "the card has no token IDs for C4 round 1 stream 1");
    // 66 arrays in total, but six prose runs and four code runs.
    let mut card = card_from(&known_rows());
    let mut prose = card["decode"]["prose"]["runs"].as_array().unwrap().clone();
    prose.push(card["decode"]["code"]["runs"][0].clone());
    let code = card["decode"]["code"]["runs"].as_array().unwrap()[1..].to_vec();
    let mut decode = card["decode"].clone();
    decode.insert("prose", object(vec![("runs", Value::Array(prose))]));
    decode.insert("code", object(vec![("runs", Value::Array(code))]));
    card.insert("decode", decode);
    assert_eq!(digest(&card).unwrap_err(), "the card has 6 prose runs, not 5");
}

/// One stream to one client: the headers, then a chunk every 20 ms until the
/// client hangs up or 10 s pass. `SENT` counts the chunks written.
fn streaming_server(sent: &'static std::sync::atomic::AtomicUsize) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 65536];
        let _ = stream.read(&mut request);
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n").unwrap();
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(10) {
            let chunk = r#"data: {"choices":[{"index":0,"delta":{"content":"t"},"finish_reason":null,"token_id":1}]}"#;
            if stream.write_all(format!("{chunk}\n\n").as_bytes()).is_err() { break; }
            sent.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(20));
        }
    });
    (url, task)
}

#[test]
fn ctrl_c_stops_a_reply_that_is_still_streaming() {
    // The reply never pauses for a second, so only a check before each read can stop it.
    static SENT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static AT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let (url, task) = streaming_server(&SENT);
    // Ctrl-C arrives once the client has seen five chunks; record the count at that moment.
    let interrupted = || { let sent = SENT.load(Ordering::SeqCst); sent >= 5 && { let _ = AT.compare_exchange(0, sent, Ordering::SeqCst, Ordering::SeqCst); true } };
    let server = Server { interrupted, ..Server::parse(&url).unwrap() };
    let error = server.stream("/v1/chat/completions", &chat_payload("m", "S", "U", 8, SEED, false)).unwrap_err();
    assert!(error.contains("Interrupted"), "{error}");
    drop(server);
    task.join().unwrap();
    // The server stops when the client hangs up: a few chunks after the interrupt, not 10 s worth (500).
    let (at, sent) = (AT.load(Ordering::SeqCst), SENT.load(Ordering::SeqCst));
    assert!(at >= 5 && sent < at + 20, "interrupt at chunk {at}, server sent {sent}");
}

#[test]
fn ctrl_c_stops_a_reply_that_has_stalled() {
    // Headers, then nothing. Ctrl-C comes 1.5 s after the headers: by then the client
    // can only be in the one-second wait step, so this test reaches that path.
    static HEADERS: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 65536];
        let _ = stream.read(&mut request);
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n").unwrap();
        let _ = HEADERS.set(Instant::now());
        // Hold the connection until the client closes it.
        let _ = stream.read(&mut request);
    });
    let server = Server { interrupted: || HEADERS.get().is_some_and(|at| at.elapsed() >= Duration::from_millis(1500)), ..Server::parse(&url).unwrap() };
    let error = server.stream("/v1/chat/completions", &chat_payload("m", "S", "U", 8, SEED, false)).unwrap_err();
    assert!(error.contains("Interrupted"), "{error}");
    // The read waited past the flag and returned within one more wait step.
    let waited = HEADERS.get().unwrap().elapsed();
    assert!(waited >= Duration::from_millis(1500) && waited < Duration::from_millis(3500), "{waited:?}");
    drop(server);
    task.join().unwrap();
}

#[test]
fn the_bearer_key_goes_with_every_request() {
    // A server that answers only requests with the key, the way an engine with --api-key-file does.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = std::thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = vec![0; 65536];
            let n = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..n]).into_owned();
            let reply = if request.contains("\r\nAuthorization: Bearer card-test-key\r\n") {
                let body = r#"{"data":[{"id":"m"}]}"#;
                format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len())
            } else { "HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\n\r\n{}".to_owned() };
            stream.write_all(reply.as_bytes()).unwrap();
        }
    });
    let keyed = Server { key: Some("card-test-key".into()), ..Server::parse(&url).unwrap() };
    assert_eq!(keyed.json("GET", "/v1/models", None).unwrap()["data"][0]["id"].as_str(), Some("m"));
    let error = Server::parse(&url).unwrap().json("GET", "/v1/models", None).unwrap_err();
    assert!(error.starts_with("/v1/models answered HTTP 401"), "{error}");
    task.join().unwrap();
}

#[test]
fn a_run_with_no_reply_saves_nothing_and_a_stopped_run_keeps_its_replies() {
    let directory = std::env::temp_dir().join(format!("vade-card-partial-{}-{}-{}", std::process::id(), now(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&directory).unwrap();
    let prompts = Prompts { version: "1.0.0".into(), system: "S".into(), prefill_unit: "u".into(),
        workloads: vec![("code".into(), "c".into()), ("prose".into(), "p".into()), ("structured".into(), "s".into())] };
    // No server on this port: the first request fails, and the file name stays free.
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", closed.local_addr().unwrap());
    drop(closed);
    let output = directory.join("none.json");
    let options = Options { url: Some(url), output: Some(output.clone()), ..Options::default() };
    let error = card_file(&options, &prompts, || false).unwrap_err();
    assert!(error.ends_with("No reply arrived, so nothing was saved."), "{error}");
    assert!(!output.exists());
    // The server refuses the third chat: the card keeps code runs 1 and 2 and says why it stopped.
    let fake = Fake::refusing_after("/v1/chat/completions", 2);
    let output = directory.join("partial.json");
    let options = Options { url: Some(fake.url.clone()), output: Some(output.clone()), ..Options::default() };
    let error = card_file(&options, &prompts, || false).unwrap_err();
    assert!(error.contains("answered HTTP 500") && error.contains("The partial card is in"), "{error}");
    let card = json::read(&output).unwrap();
    assert_eq!((card["complete"].clone(), card["decode"]["code"]["runs"].as_array().map(Vec::len)), (Value::Bool(false), Some(2)));
    assert!(card["error"].as_str().unwrap().contains("HTTP 500"));
    assert!(card.get("token_ids_sha256").is_none() && card.get("prefill").is_none());
    assert_eq!(fake.seen.lock().unwrap().iter().filter(|(p, _)| p == "/v1/chat/completions").count(), 3, "no chat after the refused one");
    // A stop inside an 8K pair: pair 1 is kept; the cold reply of pair 2, whose warm replay failed, is not.
    let fake = Fake::refusing_after("/v1/completions", 3);
    let output = directory.join("pair.json");
    let options = Options { url: Some(fake.url.clone()), output: Some(output.clone()), ..Options::default() };
    assert!(card_file(&options, &prompts, || false).unwrap_err().contains("answered HTTP 500"));
    let card = json::read(&output).unwrap();
    let pairs = |part: &str| card["prefill"]["8192"][part]["runs"].as_array().map(Vec::len);
    assert_eq!((card["decode"]["prose"]["runs"].as_array().map(Vec::len), pairs("cold"), pairs("warm_replay")), (Some(5), Some(1), Some(1)));
    assert!(card.get("concurrency").is_none());
    // A stop inside a concurrency round: the three C1 rounds are kept; the C2 round with a refused stream is not.
    let fake = Fake::refusing_after("/v1/chat/completions", 15 + 3 + 1);
    let output = directory.join("round.json");
    let options = Options { url: Some(fake.url.clone()), output: Some(output.clone()), ..Options::default() };
    assert!(card_file(&options, &prompts, || false).unwrap_err().contains("answered HTTP 500"));
    let card = json::read(&output).unwrap();
    assert_eq!(card["concurrency"]["1"]["rounds"].as_array().map(Vec::len), Some(3));
    assert!(card["concurrency"].get("2").is_none() && card["complete"] == Value::Bool(false));
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn ctrl_c_returns_130_with_no_file_before_a_reply_and_a_partial_card_after() {
    #[allow(clippy::type_complexity)] // test capture log shared with the mock server.
    static SEEN: std::sync::OnceLock<Arc<Mutex<Vec<(String, Value)>>>> = std::sync::OnceLock::new();
    let directory = std::env::temp_dir().join(format!("vade-card-stop-{}-{}-{}", std::process::id(), now(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&directory).unwrap();
    let prompts = Prompts { version: "1.0.0".into(), system: "S".into(), prefill_unit: "u".into(),
        workloads: vec![("code".into(), "c".into()), ("prose".into(), "p".into()), ("structured".into(), "s".into())] };
    let fake = Fake::start();
    // Ctrl-C before the first reply: exit status 130 and no file.
    let none = directory.join("none.json");
    let options = Options { url: Some(fake.url.clone()), output: Some(none.clone()), ..Options::default() };
    assert_eq!(card_file(&options, &prompts, || true).unwrap(), 130);
    assert!(!none.exists());
    // Ctrl-C once the server has the second chat: the first reply is saved, the second is not.
    let _ = SEEN.set(fake.seen.clone());
    let partial = directory.join("partial.json");
    let options = Options { url: Some(fake.url.clone()), output: Some(partial.clone()), ..Options::default() };
    let interrupted = || SEEN.get().is_some_and(|seen| seen.lock().unwrap().iter().filter(|(p, _)| p == "/v1/chat/completions").count() >= 2);
    assert_eq!(card_file(&options, &prompts, interrupted).unwrap(), 130);
    let card = json::read(&partial).unwrap();
    assert_eq!((card["complete"].clone(), card["decode"]["code"]["runs"].as_array().map(Vec::len)), (Value::Bool(false), Some(1)));
    let _ = std::fs::remove_dir_all(&directory);
}
