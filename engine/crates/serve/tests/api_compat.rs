//! Requests that real OpenAI clients send, and the replies the OpenAI API
//! reference promises them, against a mock engine on a real socket.
use spark_serve::{Decoder, Engine, Limits, Session, Tokenizer, cache::CacheLimits, protocol, scheduler};
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    time::Duration,
};

struct Model;
struct Kv(Vec<u32>);
impl Session for Kv {
    fn advance(&mut self, token: u32) -> Result<(), String> {
        std::thread::sleep(Duration::from_millis(2));
        self.0.push(token);
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.advance(token)?;
        // Never EOS (7): every reply runs to its max_tokens.
        let mut logits = vec![0.; 8];
        logits[self.0.len() % 7] = 1.;
        Ok(logits)
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> { self.0.truncate(position); Ok(()) }
    fn fork(&self) -> Result<Self, String> { Ok(Self(self.0.clone())) }
}
impl Engine for Model {
    type Session<'a> = Kv;
    fn session(&self) -> Result<Kv, String> { Ok(Kv(Vec::new())) }
    fn vocab_size(&self) -> usize { 8 }
    fn context_limit(&self) -> usize { 1024 }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { Some(16384) }
}
struct Codec;
impl Decoder for Codec {
    fn push(&mut self, _: u32) -> Result<String, String> { Ok("x".into()) }
    fn finish(&mut self) -> Result<String, String> { Ok(String::new()) }
}
impl Tokenizer for Codec {
    type Decoder<'a> = Codec;
    fn encode(&self, _: &str) -> Result<Vec<u32>, String> { Ok(vec![1]) }
    fn decoder(&self) -> Codec { Codec }
    fn has_token(&self, id: u32) -> bool { id < 8 }
    fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { Ok(vec![1]) }
}

struct Server { address: SocketAddr, stop: Arc<AtomicBool>, worker: Option<std::thread::JoinHandle<()>> }
impl Server {
    fn start() -> Self { Self::with_limits(Limits::default()) }
    fn with_limits(limits: Limits) -> Self { Self::with_sharing(limits, scheduler::Sharing::default()) }
    fn with_sharing(limits: Limits, sharing: scheduler::Sharing) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let shutdown = stop.clone();
        let worker = std::thread::spawn(move || {
            scheduler::serve(listener, &Model, Some(&Codec), &limits,
                &CacheLimits { entries: 8, retained_bytes: 1 << 20, tokens_per_entry: 1024,
                    logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
                &scheduler::Config { active: 3, connections: 16, prefill_chunk: 1, foreground_rows: 8,
                    background_rows: 16, active_bytes: 32768, session_budget_bytes: 2 << 20,
                    draft: protocol::DraftMode::Strict, mtp_depth: 0, draft_global_bank: false,
                    prefix_cache: false, sharing },
                &shutdown).unwrap();
        });
        Self { address, stop, worker: Some(worker) }
    }
    /// One raw HTTP/1.1 exchange; `head` adds header lines.
    fn raw(&self, method: &str, path: &str, head: &str, body: &str) -> String {
        let mut socket = TcpStream::connect(self.address).unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let length = if method == "POST" { format!("Content-Type: application/json\r\nContent-Length: {}\r\n", body.len()) } else { String::new() };
        write!(socket, "{method} {path} HTTP/1.1\r\nHost: {}\r\n{head}{length}\r\n{body}", self.address).unwrap();
        let mut reply = String::new();
        socket.read_to_string(&mut reply).unwrap();
        reply
    }
    fn post(&self, path: &str, body: &str) -> String { self.raw("POST", path, "", body) }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
    }
}
fn status(reply: &str) -> u16 { reply[9..12].parse().unwrap() }
fn body(reply: &str) -> &str { reply.split_once("\r\n\r\n").unwrap().1 }
fn events(reply: &str) -> Vec<&str> {
    body(reply).split("\n\n").filter_map(|e| e.strip_prefix("data: ")).filter(|d| *d != "[DONE]").collect()
}
fn json(text: &str) -> engine_model::json::Value {
    engine_model::json::parse(text.as_bytes(), Default::default()).unwrap_or_else(|e| panic!("{e}: {text}"))
}
fn field<'a>(value: &'a engine_model::json::Value, key: &str) -> Option<&'a engine_model::json::Value> {
    value.object().unwrap().get(key)
}

const CHAT: &str = r#""model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}],"max_tokens":3"#;

// Finding 3: OpenAI's first chunk carries delta.role "assistant"; SDK stream
// accumulators take the message role from it.
#[test]
fn first_stream_chunk_carries_the_assistant_role() {
    let server = Server::start();
    let reply = server.post("/v1/chat/completions", &format!("{{{CHAT},\"stream\":true}}"));
    assert_eq!(status(&reply), 200, "{reply}");
    let chunks = events(&reply);
    let first = json(chunks[0]);
    let delta = field(&field(&first, "choices").unwrap().array().unwrap()[0], "delta").unwrap();
    assert_eq!(field(delta, "role").and_then(|v| v.string().ok()), Some("assistant"), "{}", chunks[0]);
    assert_eq!(chunks.iter().filter(|c| c.contains("\"role\"")).count(), 1, "{reply}");
    // Content is unchanged: three tokens, three x.
    let text: String = chunks.iter().map(|c| json(c)).filter_map(|v| {
        let choice = field(&v, "choices")?.array().ok()?.first()?.clone();
        field(&choice, "delta")?.object().ok()?.get("content")?.string().ok().map(str::to_owned)
    }).collect();
    assert_eq!(text, "xxx");
}

// Finding 3, second half: the message openai-python rebuilds from a stream and
// sends back on the next turn (model_dump of ChatCompletionMessage).
#[test]
fn a_streamed_message_sent_back_is_accepted() {
    let server = Server::start();
    let body = r#"{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"xxx","refusal":null,"tool_calls":null,"function_call":null,"audio":null,"annotations":[]},{"role":"user","content":"Again"}],"max_tokens":1}"#;
    let reply = server.post("/v1/chat/completions", body);
    assert_eq!(status(&reply), 200, "{reply}");
}

// Finding 21: every OpenAI error body has message, type, param and code (null
// when unset). Keep the existing error type for local clients.
#[test]
fn error_bodies_carry_param_and_code() {
    let server = Server::start();
    for request in [
        "{not json",
        r#"{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}],"n":2}"#,
        r#"{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}],"stop":""}"#,
        r#"{"model":"qwen3-8b","messages":[]}"#,
    ] {
        let reply = server.post("/v1/chat/completions", request);
        assert_eq!(status(&reply), 400, "{reply}");
        let value = json(body(&reply));
        let error = field(&value, "error").unwrap();
        for key in ["message", "type", "param", "code"] { assert!(field(error, key).is_some(), "{key}: {reply}"); }
        assert_eq!(field(error, "param"), Some(&engine_model::json::Value::Null), "{reply}");
    }
}

#[test]
fn completion_accepts_one_string_prompt() {
    let server = Server::start();
    let reply = server.post("/v1/completions", r#"{"model":"qwen3-8b","prompt":["Say hello"],"max_tokens":1}"#);
    assert_eq!(status(&reply), 200, "{reply}");
    let parsed = protocol::parse(br#"{"prompt":["Say hello"]}"#, false, &Limits::default()).unwrap();
    assert!(matches!(parsed.prompt, protocol::Prompt::Text(s) if s == "Say hello"));
    let reply = server.post("/v1/completions", r#"{"model":"qwen3-8b","prompt":["one","two"]}"#);
    assert_eq!(status(&reply), 400, "{reply}");
    assert!(body(&reply).contains("only one string prompt is supported"), "{reply}");
    // Existing token arrays retain their meaning and order.
    let parsed = protocol::parse(br#"{"prompt":[1,2]}"#, false, &Limits::default()).unwrap();
    assert!(matches!(parsed.prompt, protocol::Prompt::Tokens(ids) if ids == [1, 2]));
}

#[test]
fn tokenizer_limits_have_context_error_codes() {
    for message in ["chat template byte limit exceeded", "input byte limit exceeded", "token count limit exceeded"] {
        let value = json(&protocol::error_json("invalid_request_error", message));
        assert_eq!(field(field(&value, "error").unwrap(), "code").and_then(|v| v.string().ok()),
            Some("context_length_exceeded"), "{message}");
    }
    for message in ["body byte limit exceeded", "output byte limit exceeded"] {
        assert!(!protocol::error_json("invalid_request_error", message).contains("context_length_exceeded"));
    }
}

#[test]
fn requested_stream_usage_has_an_empty_choices_chunk() {
    let server = Server::start();
    for option in ["true", "false"] {
        let reply = server.post("/v1/chat/completions", &format!("{{{CHAT},\"stream\":true,\"stream_options\":{{\"include_usage\":{option}}}}}"));
        let chunks: Vec<_> = events(&reply).into_iter().map(json).collect();
        let usage: Vec<_> = chunks.iter().filter(|v| field(v, "usage").is_some_and(|v| *v != engine_model::json::Value::Null)).collect();
        if option == "false" { assert!(usage.is_empty(), "{reply}"); continue; }
        assert_eq!(usage.len(), 1, "{reply}");
        let last = chunks.last().unwrap();
        assert!(field(last, "choices").unwrap().array().unwrap().is_empty(), "{reply}");
        for key in ["id", "object", "created", "model"] { assert_eq!(field(last, key), field(&chunks[0], key), "{key}: {reply}"); }
        let counts = field(last, "usage").unwrap();
        assert_eq!(field(counts, "completion_tokens").unwrap().u64().unwrap(), 3);
        assert!(chunks[..chunks.len()-1].iter().all(|v| field(v, "usage").is_none_or(|v| *v == engine_model::json::Value::Null)));
    }
}

#[test]
fn sdk_requests_preserve_auth_health_and_private_usage_contracts() {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::temp_dir().join(format!("vade-api-compat-key-{}", std::process::id()));
    std::fs::write(&path, b"0123456789abcdef\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let key = spark_serve::auth::ApiKey::load(Some(&path), false).unwrap();
    std::fs::remove_file(&path).unwrap();
    let server = Server::with_sharing(Limits { api_key: key, ..Limits::default() },
        scheduler::Sharing { private_counts: true, ..Default::default() });
    const AUTH: &str = "Authorization: Bearer 0123456789abcdef\r\n";
    for route in ["/v1/models", "/v1/chat/completions"] {
        let reply = server.raw("GET", route, "", "");
        assert_eq!(status(&reply), 401);
        assert_eq!(body(&reply), spark_serve::auth::UNAUTHORIZED_BODY);
    }
    let reply = server.raw("GET", "/health", "", "");
    assert_eq!(status(&reply), 200);
    assert_eq!(body(&reply), r#"{"status":"ready"}"#);
    engine_model::events::note("prefix_cache_api_compat_hidden", format_args!("test"));
    for route in ["/health", "/v1/events"] {
        let reply = server.raw("GET", route, AUTH, "");
        assert_eq!(status(&reply), 200);
        assert!(!reply.contains("prefix_cache_api_compat_hidden"));
        if route == "/health" {
            assert_eq!(field(&json(body(&reply)), "device_bytes"), Some(&engine_model::json::Value::Null));
        }
    }
    let reply = server.raw("GET", "/v1/models", AUTH, "");
    assert_eq!(status(&reply), 200);
    let models = json(body(&reply));
    let model = &field(&models, "data").unwrap().array().unwrap()[0];
    assert!(field(model, "created").unwrap().u64().is_ok());
    let reply = server.raw("POST", "/v1/chat/completions", AUTH,
        &format!("{{{CHAT},\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"n\":1}}"));
    assert_eq!(status(&reply), 200);
    let chunks: Vec<_> = events(&reply).into_iter().map(json).collect();
    let last = chunks.last().unwrap();
    assert!(field(last, "choices").unwrap().array().unwrap().is_empty());
    assert_eq!(field(field(last, "usage").unwrap(), "completion_tokens").unwrap().u64().unwrap(), 3);
    assert!(!reply.contains("prefix_cache_api_compat_hidden"));
    assert!(reply.ends_with("data: [DONE]\n\n"));
}
