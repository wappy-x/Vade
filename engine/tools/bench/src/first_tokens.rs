//! Chat preparation and greedy repeat receipts through the shared HTTP client.
//! This client does not load weights or select an engine execution path.
use crate::{http::{self, Endpoint, Reply}, timed::{stream, value::*}};
use engine_tokenizer::{Tokenizer, chat::{self, Message, Role}};
use std::{fs, io::{Read, Write}, path::Path, sync::{Arc, atomic::AtomicBool}, time::{Duration, Instant}};

const TOKENS: usize = 8;
const REPEATS: usize = 3;
const HELP: &str = "first-tokens prepare CHECKPOINT USER_TEXT_FILE NEW_DIRECTORY
first-tokens capture URL CHECKPOINT PREPARED_JSON NEW_RECEIPT SHA40 CHECKPOINT_SHA256 SPARK_COUNT
first-tokens run URL CHECKPOINT PREPARED_JSON NEW_RECEIPT SHA40 CHECKPOINT_SHA256 SPARK_COUNT
first-tokens compare RECEIPT RECEIPT RECEIPT [--first-id HOST_ID_FILE]
first-tokens compare RECEIPT RECEIPT RECEIPT --expected-ids HOST_IDS_FILE
first-tokens --self-test

Also available as: vade bench --first-tokens ...
Prepare uses the checkpoint tokenizer and chat template. Capture sends one chat.
Each capture requires a fresh server process. Restart every rank between captures.
Capture requests 8 greedy tokens, thinking off, cache off, and drafting off.
Run is an alias for capture. The checkpoint template controls the prompt.
Compare checks IDs, decoded text, reply text, settings, and server start times.
The host ID file contains one unsigned token ID from an independent reference.
The expected IDs file contains all 8 independent reference IDs in order.
Without that file, first-step parity stays open. Peer logs must prove peer restarts.
The self-test uses local mock replies. It is not a device check.
";

type Result<T> = std::result::Result<T, String>;
fn read(path: &Path, cap: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path).and_then(|file| file.take(cap as u64 + 1).read_to_end(&mut bytes)).map_err(|e| e.to_string())?;
    if bytes.len() > cap { return Err("input file exceeds its byte limit".into()); }
    Ok(bytes)
}
fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::OpenOptions::new().write(true).create_new(true).open(path)
        .and_then(|mut file| file.write_all(bytes)).map_err(|e| e.to_string())
}
fn load(path: &str) -> Result<Value> { parse(&read(Path::new(path), 8 * 1024 * 1024)?) }
fn encoded_ids(ids: &[u32]) -> Value { a(ids.iter().map(n).collect()) }
fn token_bytes_sha256(tokens: &[u32]) -> String {
    digest(&tokens.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>())
}
fn ids(value: &Value) -> Result<Vec<u32>> {
    let Value::Array(items) = value else { return Err("missing token ID array".into()) };
    items.iter().map(|v| v.uint().and_then(|v| u32::try_from(v).ok()).ok_or("invalid token ID".into())).collect()
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value.get(key).str().ok_or_else(|| format!("missing string field {key}"))
}
fn hash(value: &str, len: usize) -> Result<()> {
    if value.len() != len || !value.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return Err("expected a full lowercase hexadecimal hash".into());
    }
    Ok(())
}
fn from_json(value: &engine_model::json::Value) -> Value {
    use engine_model::json::Value as J;
    match value {
        J::Null => Value::Null, J::Bool(v) => b(*v), J::Number(v) => Value::Number(v.clone()), J::String(v) => s(v),
        J::Array(v) => a(v.iter().map(from_json).collect()),
        J::Object(v) => Value::Object(v.iter().map(|(k, v)| (k.clone(), from_json(v))).collect()),
    }
}

fn prepare(root: &Path, user: &str) -> Result<(Tokenizer, Value)> {
    if user.is_empty() || user.len() > 1024 * 1024 { return Err("user text must be nonempty and fit the byte limit".into()); }
    let files = [("config.json", 1024 * 1024), ("tokenizer.json", 32 * 1024 * 1024), ("tokenizer_config.json", 2 * 1024 * 1024)];
    let metadata = Value::Object(files.iter().map(|(name, cap)|
        read(&root.join(name), *cap).map(|bytes| ((*name).into(), s(digest(&bytes))))).collect::<Result<_>>()?);
    let tokenizer = Tokenizer::from_directory(root).map_err(|e| format!("checkpoint tokenizer: {e}"))?;
    let messages = [Message::new(Role::User, user)];
    let options = chat::Options { enable_thinking: false, ..tokenizer.chat_options() };
    let prompt = chat::encode(&tokenizer, &messages, &options).map_err(|e| e.to_string())?;
    if prompt.is_empty() { return Err("chat template produced no prompt IDs".into()); }
    let rendered = tokenizer.format_chat(&messages, &options).map_err(|e| e.to_string())?;
    let mut request = stream::sampling();
    for (key, value) in [
        ("messages", a(vec![o([("role", s("user")), ("content", s(user))])])),
        ("stream", b(true)), ("stream_options", o([("include_usage", b(true))])),
        ("max_tokens", n(TOKENS)), ("cache", b(false)), ("ignore_eos", b(true)),
        ("enable_thinking", b(false)), ("chat_template_kwargs", o([("enable_thinking", b(false))])),
        ("draft", s("plain")), ("mtp_depth", n(0)),
    ] { request.put(key, value); }
    Ok((tokenizer, o([
        ("schema", s("vade.first-tokens.prepared.v1")), ("metadata_sha256", metadata),
        ("user_text", s(user)), ("rendered_prompt", s(rendered)), ("prompt_ids", encoded_ids(&prompt)),
        ("request", request),
    ])))
}

// Keep only public identity fields. Endpoints and local paths do not enter receipts.
fn identity(health: &Value, sha: &str, checkpoint: &str, sparks: usize) -> Result<Value> {
    // Validate the identity as data. No rank count selects a client execution path.
    for (field, expected, error) in [
        ("vade.build.git_sha", s(sha), "server build differs from the requested commit"),
        ("vade.build.dirty", b(false), "server must report a clean build"),
        ("vade.checkpoint.sha256", s(checkpoint), "server checkpoint hash differs from the requested hash"),
        ("vade.numerics.world_size", n(sparks), "server Spark count differs from the requested count"),
    ] {
        if encode(health.path(field)) != encode(&expected) { return Err(error.into()); }
    }
    let start = health.path("vade.server_start_unix_ms");
    if start.uint().is_none_or(|n| n == 0) { return Err("server has no positive start time".into()); }
    let model = health.get("model").str().filter(|v| !v.is_empty()).ok_or("server has no model ID")?;
    for field in ["vade.ptx_sha256", "vade.layer_ptx_sha256"] {
        hash(health.path(field).str().ok_or("server has no kernel hash")?, 64)?;
    }
    Ok(o([
        ("model", s(model)), ("build_sha", s(sha)), ("checkpoint_sha256", s(checkpoint)),
        ("numerics", health.path("vade.numerics").clone()),
        ("ptx_sha256", health.path("vade.ptx_sha256").clone()),
        ("layer_ptx_sha256", health.path("vade.layer_ptx_sha256").clone()),
        ("server_start_unix_ms", start.clone()),
    ]))
}

fn check_reply(reply: &Reply, prompt_len: usize, model: &str) -> Result<()> {
    if reply.status != "ok" { return Err(reply.error.clone().unwrap_or_else(|| reply.status.clone())); }
    reply.verify_greedy()?;
    let echo = reply.vade.as_ref().map(from_json).unwrap_or(Value::Null);
    for (key, value) in [("draft", s("plain")), ("mtp_depth", n(0)), ("thinking", b(false)),
        ("draft_used", b(false)), ("model_draft_used", b(false))] {
        // Value equality permits Python-style bool/number equivalence. Use JSON bytes here.
        if encode(echo.get(key)) != encode(&value) { return Err(format!("unexpected request echo: {key}")); }
    }
    if reply.token_ids.len() != TOKENS || reply.completion_tokens != Some(TOKENS as u64) {
        return Err("completion IDs and usage must contain the requested token count".into());
    }
    if reply.prompt_tokens != Some(prompt_len as u64) || reply.total_tokens != Some((prompt_len + TOKENS) as u64) {
        return Err("prompt or total usage differs from local prompt IDs".into());
    }
    if reply.cached_tokens != Some(0) || reply.reused_tokens.is_some_and(|n| n != 0) {
        return Err("capture requires a cold request with zero cached tokens".into());
    }
    if reply.response_model.as_deref() != Some(model) { return Err("reply model differs from server identity".into()); }
    if reply.finish_reason.as_deref() != Some("length") { return Err("capture requires a length finish".into()); }
    Ok(())
}

fn capture(url: &str, root: &Path, prepared: &Value, sha: &str, checkpoint: &str, sparks: usize,
    interrupted: fn() -> bool) -> Result<Value> {
    hash(sha, 40)?; hash(checkpoint, 64)?;
    std::num::NonZeroUsize::new(sparks).ok_or("Spark count must be positive")?;
    let (tokenizer, current) = prepare(root, text(prepared, "user_text")?)?;
    if encode(&current) != encode(prepared) { return Err("prepared chat differs from checkpoint metadata or template output".into()); }
    // The health reader accepts a base URL, while the chat client also accepts its full route.
    let base = url.trim_end_matches('/');
    let health_url = base.strip_suffix("/chat/completions").unwrap_or(base);
    let before = identity(&stream::get_health_with_cancel(health_url, 30.0, interrupted)?, sha, checkpoint, sparks)?;
    let mut request = prepared.get("request").clone();
    request.put("model", before.get("model").clone());
    let body = encode(&request);
    let endpoint = Endpoint::parse(url)?;
    if interrupted() { return Err("chat proof interrupted".into()); }
    let reply = http::request(&endpoint, &body, Instant::now() + Duration::from_secs(1800),
        Arc::new(AtomicBool::new(false)), None);
    let after = identity(&stream::get_health_with_cancel(health_url, 30.0, interrupted)?, sha, checkpoint, sparks)?;
    let mut errors = Vec::new();
    if encode(&before) != encode(&after) { errors.push(s("server identity changed during capture")); }
    if let Err(error) = check_reply(&reply, ids(prepared.get("prompt_ids"))?.len(), text(&before, "model")?) { errors.push(s(error)); }
    let decoded = match tokenizer.decode(&reply.token_ids, false) {
        Ok(text) => s(text), Err(error) => { errors.push(s(error.to_string())); Value::Null }
    };
    Ok(o([
        ("schema", s("vade.first-tokens.capture.v1")), ("status", s(if errors.is_empty() { "pass" } else { "fail" })),
        ("errors", a(errors)), ("identity", before), ("identity_after", after),
        ("prepared_sha256", s(digest(encode(prepared).as_bytes()))), ("prepared", prepared.clone()),
        ("request_sha256", s(digest(body.as_bytes()))), ("request", request),
        ("token_ids", encoded_ids(&reply.token_ids)), ("decoded_text", decoded), ("reply_text", s(reply.text)),
        ("sampling_echo", reply.vade.as_ref().map(from_json).unwrap_or(Value::Null)),
        ("prompt_tokens", reply.prompt_tokens.map(n).unwrap_or(Value::Null)),
        ("completion_tokens", reply.completion_tokens.map(n).unwrap_or(Value::Null)),
        ("total_tokens", reply.total_tokens.map(n).unwrap_or(Value::Null)),
        ("cached_tokens", reply.cached_tokens.map(n).unwrap_or(Value::Null)),
        ("reused_tokens", reply.reused_tokens.map(n).unwrap_or(Value::Null)),
        ("response_model", reply.response_model.map(s).unwrap_or(Value::Null)),
        ("finish_reason", reply.finish_reason.map(s).unwrap_or(Value::Null)),
        ("elapsed_ms", n(reply.elapsed_ms)), ("decode_tokens_per_second", reply.decode_tps.map(n).unwrap_or(Value::Null)),
    ]))
}

fn compare(records: &[Value], reference: Option<u32>) -> Result<Value> {
    if records.len() != REPEATS { return Err("compare requires the full repeat set".into()); }
    let first = &records[0];
    let mut starts = std::collections::BTreeSet::new();
    for record in records {
        if record.get("schema").str() != Some("vade.first-tokens.capture.v1") || record.get("status").str() != Some("pass") {
            return Err("repeat set contains a failed or invalid capture".into());
        }
        let tokens = ids(record.get("token_ids"))?;
        if tokens.len() != TOKENS { return Err("repeat has the wrong token count".into()); }
        let prepared = record.get("prepared");
        if record.get("prepared_sha256").str() != Some(digest(encode(prepared).as_bytes()).as_str())
            || record.get("request_sha256").str() != Some(digest(encode(record.get("request")).as_bytes()).as_str()) {
            return Err("capture input hash differs from its saved bytes".into());
        }
        let reply = Reply {
            status: "ok".into(), token_ids: tokens.clone(),
            prompt_tokens: record.get("prompt_tokens").uint(), completion_tokens: record.get("completion_tokens").uint(),
            total_tokens: record.get("total_tokens").uint(), cached_tokens: record.get("cached_tokens").uint(),
            reused_tokens: record.get("reused_tokens").uint(), response_model: record.get("response_model").str().map(str::to_owned),
            finish_reason: record.get("finish_reason").str().map(str::to_owned),
            vade: Some(engine_model::json::parse(encode(record.get("sampling_echo")).as_bytes(), Default::default()).map_err(|e| e.to_string())?),
            ..Default::default()
        };
        check_reply(&reply, ids(prepared.get("prompt_ids"))?.len(), text(record.get("identity"), "model")?)?;
        for field in ["decoded_text", "reply_text"] { text(record, field)?; }
        if reference.is_some_and(|id| tokens.first() != Some(&id)) { return Err("first token differs from the host reference".into()); }
        for field in ["prepared_sha256", "prepared", "request_sha256", "request", "token_ids", "decoded_text", "reply_text"] {
            if record.get(field) == &Value::Null || encode(record.get(field)) != encode(first.get(field)) {
                return Err(format!("repeat differs at {field}"));
            }
        }
        let mut stable = record.get("identity").clone();
        let mut base = first.get("identity").clone();
        let start = stable.get("server_start_unix_ms").uint().filter(|n| *n > 0).ok_or("capture has no server start time")?;
        if !starts.insert(start) { return Err("repeat set reuses a server start time; restart every rank".into()); }
        stable.del("server_start_unix_ms"); base.del("server_start_unix_ms");
        if encode(&stable) != encode(&base) { return Err("repeat server identity differs".into()); }
        if encode(record.get("identity")) != encode(record.get("identity_after")) { return Err("capture server identity changed".into()); }
    }
    Ok(o([
        ("status", s("repeat-pass")), ("runs", n(records.len())), ("token_ids", first.get("token_ids").clone()),
        ("token_ids_u32le_sha256", s(token_bytes_sha256(&ids(first.get("token_ids"))?))),
        ("decoded_text", first.get("decoded_text").clone()), ("reply_text", first.get("reply_text").clone()),
        ("first_id_reference", reference.map(n).unwrap_or(Value::Null)),
        ("peer_process_freshness", s("requires station rank logs")),
        ("decode_tokens_per_second", a(records.iter().map(|v| v.get("decode_tokens_per_second").clone()).collect())),
    ]))
}

fn compare_expected(records:&[Value], expected:&[u32])->Result<Value> {
    if expected.len()!=TOKENS {return Err("host reference must contain the complete output ID sequence".into());}
    let mut result=compare(records,expected.first().copied())?;
    if ids(result.get("token_ids"))?!=expected {return Err("output token IDs differ from the host reference".into());}
    result.put("expected_token_ids",encoded_ids(expected));Ok(result)
}

/// Run the generic proof client. A capture never starts or stops a server.
pub fn run(arguments: &[String], interrupted: fn() -> bool) -> Result<i32> {
    let arg: Vec<_> = arguments.iter().map(String::as_str).collect();
    match arg.as_slice() {
        [] | ["--help" | "-h"] => { crate::write_output(format_args!("{HELP}"), false)?; }
        ["prepare", root, user, output] => {
            let user = String::from_utf8(read(Path::new(user), 1024 * 1024)?).map_err(|e| e.to_string())?;
            let (_, prepared) = prepare(Path::new(root), &user)?;
            fs::create_dir(output).map_err(|e| e.to_string())?;
            let dir = Path::new(output);
            write(&dir.join("prepared.json"), encode(&prepared).as_bytes())?;
            write(&dir.join("request.json"), encode(prepared.get("request")).as_bytes())?;
            write(&dir.join("prompt.txt"), text(&prepared, "rendered_prompt")?.as_bytes())?;
            let tokens = ids(prepared.get("prompt_ids"))?;
            write(&dir.join("prompt.ids"), tokens.iter().map(|id| format!("{id}\n")).collect::<String>().as_bytes())?;
            crate::write_output(format_args!("PREPARED prompt_tokens={} request_sha256={}", tokens.len(), digest(encode(prepared.get("request")).as_bytes())), true)?;
        }
        ["capture" | "run", url, root, input, output, sha, checkpoint, sparks] => {
            let record = capture(url, Path::new(root), &load(input)?, sha, checkpoint,
                sparks.parse().map_err(|_| "invalid Spark count")?, interrupted)?;
            write(Path::new(output), encode(&record).as_bytes())?;
            crate::write_output(format_args!("CAPTURE {} tokens={} decode_tokens_per_second={}", text(&record, "status")?,
                record.get("token_ids").arr().len(), encode(record.get("decode_tokens_per_second"))), true)?;
            crate::write_output(format_args!("TOKEN_IDS {} token_ids_u32le_sha256={}", encode(record.get("token_ids")),
                token_bytes_sha256(&ids(record.get("token_ids"))?)), true)?;
            if record.get("status").str() != Some("pass") { return Err(format!("capture failed: {}", encode(record.get("errors")))); }
        }
        ["compare", one, two, three] | ["compare", one, two, three, "--first-id", _] => {
            let reference = if arg.len() == 6 {
                Some(std::str::from_utf8(&read(Path::new(arg[5]), 128)?).map_err(|e| e.to_string())?.trim().parse::<u32>()
                    .map_err(|_| "host reference must contain one unsigned token ID")?)
            } else { None };
            let result = compare(&[load(one)?, load(two)?, load(three)?], reference)?;
            crate::write_output(format_args!("{}", encode(&result)), true)?;
        }
        ["compare", one, two, three, "--expected-ids", reference] => {
            let bytes=read(Path::new(reference),4096)?;
            let expected=std::str::from_utf8(&bytes).map_err(|e|e.to_string())?.split_whitespace()
                .map(|id|id.parse::<u32>().map_err(|_|"host reference contains an invalid token ID".to_owned())).collect::<Result<Vec<_>>>()?;
            let result=compare_expected(&[load(one)?,load(two)?,load(three)?],&expected)?;
            crate::write_output(format_args!("{}", encode(&result)), true)?;
        }
        ["--self-test"] => { self_test::run()?; crate::write_output(format_args!("SELFTEST PASS mock chat preparation, captures, repeats, and refusal checks"), true)?; }
        _ => return Err(HELP.into()),
    }
    Ok(0)
}

mod self_test;
