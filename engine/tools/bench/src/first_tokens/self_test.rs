//! Small local socket proof. Synthetic replies never stand for model evidence.
use super::*;
use std::{io::{BufRead, BufReader}, net::{Ipv4Addr, TcpListener}, thread, time::{SystemTime, UNIX_EPOCH}};

const SHA: &str = "1111111111111111111111111111111111111111";
const CHECKPOINT: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const KERNEL: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const ECHO: &str = r#"{"sampling":{"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0},"draft":"plain","mtp_depth":0,"thinking":false,"draft_used":false,"model_draft_used":false}"#;
static FIXTURE_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Fixture(std::path::PathBuf);
impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
impl Fixture {
    fn new() -> Result<Self> {
        let serial = FIXTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("vade-first-tokens-test-{}-{}-{serial}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos()));
        fs::create_dir(&root).map_err(|e| e.to_string())?;
        let fixture = Self(root);
        write(&fixture.0.join("config.json"), b"{}")?;
        write(&fixture.0.join("tokenizer.json"), include_bytes!("../../../../crates/tokenizer/tests/fixtures/host_edges/tokenizer.json"))?;
        let config = o([
            ("chat_template", s(include_str!("../../../../crates/tokenizer/templates/qwen3.jinja"))),
            ("eos_token", s("<|im_end|>")), ("pad_token", s("<|endoftext|>")),
            ("add_bos_token", b(false)), ("add_prefix_space", b(false)),
            ("clean_up_tokenization_spaces", b(false)), ("split_special_tokens", b(false)),
        ]);
        write(&fixture.0.join("tokenizer_config.json"), encode(&config).as_bytes())?;
        Ok(fixture)
    }
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(format!("self-test: {message}")) }
}
fn health(start: u64) -> Value {
    o([("status", s("ready")), ("model", s("fixture")), ("vade", o([
        ("build", o([("git_sha", s(SHA)), ("dirty", b(false))])),
        ("checkpoint", o([("sha256", s(CHECKPOINT)), ("hash_state", s("ready"))])),
        ("ptx_sha256", s(KERNEL)), ("layer_ptx_sha256", s(KERNEL)),
        ("numerics", o([("world_size", n(2)), ("topology", s("tensor_split")), ("mtp_depth", n(0))])),
        ("server_start_unix_ms", n(start)),
    ]))])
}
fn serve(listener: TcpListener, start: u64, prepared: Value) -> Result<()> {
    // Capture reads health before and after its one chat request.
    for step in 0..3 {
        let (mut socket, _) = listener.accept().map_err(|e| e.to_string())?;
        socket.set_read_timeout(Some(Duration::from_secs(10))).map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(socket.try_clone().map_err(|e| e.to_string())?);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        require(line.starts_with(if step == 1 { "POST /v1/chat/completions " } else { "GET /health " }), "wrong request order")?;
        let mut length = 0;
        loop {
            line.clear(); reader.read_line(&mut line).map_err(|e| e.to_string())?;
            if line == "\r\n" { break; }
            if let Some(value) = line.strip_prefix("Content-Length: ") { length = value.trim().parse::<usize>().map_err(|e| e.to_string())?; }
            require(!line.is_empty(), "request headers ended early")?;
        }
        if step != 1 {
            let body = encode(&health(start));
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).map_err(|e| e.to_string())?;
            continue;
        }
        require(length < 2 * 1024 * 1024, "mock request too large")?;
        let mut bytes = vec![0; length]; reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        let mut expected = prepared.get("request").clone(); expected.put("model", s("fixture"));
        require(encode(&parse(&bytes)?) == encode(&expected), "request settings or chat changed")?;
        let mut body = String::new();
        for _ in 0..TOKENS {
            body.push_str("data: {\"model\":\"fixture\",\"choices\":[{\"index\":0,\"token_id\":256,\"delta\":{\"content\":\"ab\"}}]}\n\n");
        }
        let prompt = ids(prepared.get("prompt_ids"))?.len();
        body.push_str(&format!("data: {{\"model\":\"fixture\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"length\"}}],\"usage\":{{\"prompt_tokens\":{prompt},\"completion_tokens\":{TOKENS},\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":0}}}},\"vade\":{ECHO}}}\n\ndata: [DONE]\n\n", prompt + TOKENS));
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}").map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn captures(fixture: &Fixture, prepared: &Value) -> Result<Vec<Value>> {
    let mut records = Vec::new();
    for start in 1..=REPEATS {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(|e| e.to_string())?;
        let suffix = ["", "/v1/", "/v1/chat/completions"][start - 1];
        let url = format!("http://{}{suffix}", listener.local_addr().map_err(|e| e.to_string())?);
        let input = prepared.clone();
        let worker = thread::spawn(move || serve(listener, start as u64, input));
        let result = capture(&url, &fixture.0, prepared, SHA, CHECKPOINT, 2, || false);
        worker.join().map_err(|_| "mock server panicked")??;
        records.push(result?);
    }
    Ok(records)
}
fn refusals(records: &[Value]) -> Result<()> {
    require(compare(records, Some(257)).is_err(), "accepted the wrong host first token")?;
    for field in ["token_ids", "decoded_text", "reply_text", "prepared_sha256", "request_sha256", "completion_tokens", "sampling_echo"] {
        let mut bad = records.to_vec(); bad[1].put(field, Value::Null);
        require(compare(&bad, None).is_err(), &format!("accepted missing {field}"))?;
    }
    let mut bad = records.to_vec(); bad[1] = bad[0].clone();
    require(compare(&bad, None).is_err(), "accepted a reused server process")?;
    let mut bad = records.to_vec();
    let mut echo = bad[1].get("sampling_echo").clone(); echo.put("draft_used", b(true)); bad[1].put("sampling_echo", echo);
    require(compare(&bad, None).is_err(), "accepted a draft response")?;
    let mut bad = records.to_vec(); bad[1].put("completion_tokens", n(TOKENS - 1));
    require(compare(&bad, None).is_err(), "accepted early completion")?;
    require(compare(&records[..2], None).is_err(), "accepted a partial repeat set")
}

pub(super) fn run() -> Result<()> {
    require(token_bytes_sha256(&[1,256])==digest(&[1,0,0,0,0,1,0,0]), "token hash does not use little-endian u32 bytes")?;
    let fixture = Fixture::new()?;
    let (tokenizer, prepared) = prepare(&fixture.0, "ab <|im_end|> café\n")?;
    let prompt = ids(prepared.get("prompt_ids"))?;
    require(prompt.iter().filter(|&&id| id == 151645).count() == 1, "user text became a template control")?;
    require(tokenizer.decode(&[256; TOKENS], false).map_err(|e| e.to_string())? == "ab".repeat(TOKENS), "fixture decode changed")?;
    let records = captures(&fixture, &prepared)?;
    require(compare(&records, Some(256))?.get("status").str() == Some("repeat-pass"), "repeat proof failed")?;
    compare_expected(&records,&[256;TOKENS])?;
    let mut expected=[256;TOKENS];expected[TOKENS-1]=257;
    require(compare_expected(&records,&expected).is_err(),"accepted the wrong final host token")?;
    require(compare_expected(&records,&expected[..TOKENS-1]).is_err(),"accepted a partial host token sequence")?;
    refusals(&records)?;
    let before = health(1);
    require(identity(&before, CHECKPOINT, CHECKPOINT, 2).is_err(), "accepted the wrong build")?;
    require(identity(&before, SHA, KERNEL, 2).is_err(), "accepted the wrong checkpoint")?;
    require(identity(&before, SHA, CHECKPOINT, 4).is_err(), "accepted the wrong Spark count")?;
    let mut changed = prepared.clone(); changed.put("prompt_ids", a(vec![n(0)]));
    require(capture("invalid", &fixture.0, &changed, SHA, CHECKPOINT, 2, || false).is_err(), "accepted changed prompt IDs")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chat_loop_and_repeat_failures() { run().unwrap(); }
    #[test]
    fn prepare_refuses_unknown_template_and_metadata_changes() {
        let fixture = Fixture::new().unwrap();
        let (_, prepared) = prepare(&fixture.0, "ab").unwrap();
        fs::write(fixture.0.join("config.json"), b"{\"changed\":true}").unwrap();
        assert!(capture("invalid", &fixture.0, &prepared, SHA, CHECKPOINT, 2, || false).unwrap_err().contains("metadata"));
        fs::write(fixture.0.join("tokenizer_config.json"), b"{\"chat_template\":\"unknown\"}").unwrap();
        assert!(prepare(&fixture.0, "ab").err().unwrap().contains("unsupported chat template"));
    }
    #[test]
    fn output_files_are_never_replaced() {
        let fixture = Fixture::new().unwrap();
        assert!(write(&fixture.0.join("config.json"), b"changed").is_err());
        assert_eq!(fs::read(fixture.0.join("config.json")).unwrap(), b"{}");
    }
    #[test]
    fn prepared_plain_turn_pins_glm_ids_and_eight_greedy_tokens() {
        let fixture = Fixture::new().unwrap();
        fs::write(fixture.0.join("tokenizer.json"), include_bytes!("../../../../crates/tokenizer/tests/glm-tokenizer-slice.json")).unwrap();
        fs::write(fixture.0.join("tokenizer_config.json"), b"{}").unwrap();
        fs::write(fixture.0.join("chat_template.jinja"), include_bytes!("../../../../crates/tokenizer/templates/glm.jinja")).unwrap();
        let (_, prepared) = prepare(&fixture.0, "Say hello.").unwrap();
        let expected = [154822,154824,154826,25062,287,29905,371,25,7487,154827,45494,23745,13,154828,154841];
        assert_eq!(ids(prepared.get("prompt_ids")).unwrap(), expected);
        let request = prepared.get("request");
        assert_eq!(request.get("max_tokens").uint(), Some(8));
        assert_eq!(encode(request.get("temperature")), "0");
        assert_eq!(encode(request.get("draft")), "\"plain\"");
        assert_eq!(request.get("mtp_depth").uint(), Some(0));
        assert_eq!(encode(request.get("cache")), "false");
        let mut changed = prepared.clone();
        changed.put("prompt_ids", encoded_ids(&expected[..expected.len()-1]));
        assert!(capture("invalid", &fixture.0, &changed, SHA, CHECKPOINT, 2, || false).unwrap_err().contains("prepared chat"));
        changed = prepared.clone();
        let mut request = changed.get("request").clone(); request.put("max_tokens", n(32)); changed.put("request", request);
        assert!(capture("invalid", &fixture.0, &changed, SHA, CHECKPOINT, 2, || false).unwrap_err().contains("prepared chat"));
    }
}
