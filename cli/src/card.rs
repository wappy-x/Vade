//! `vade bench --card`: send the 66 requests of a published card to a running
//! server, save every token ID in card order, and compare the IDs with a
//! receipt or another card.
//!
//! The card is the RigMark protocol 1.3.0 card at RigMark revision
//! `52c29c3`, thinking off, with the request bodies the published cards
//! sent (`bench/receipts/2026-10-31/README.md`). The prompts are RigMark's
//! `prompts.json` at that revision. They are not copied into this repository:
//! the user passes the file, and the client refuses any other SHA-256.
//!
//! Requests go out in the card's order: 15 single-stream chats (5 each of the
//! workloads, in the order of `prompts.json`), three cold and warm 8K prompt
//! pairs, then 3 rounds each of 1, 2, 4 and 8 concurrent code chats. The
//! token IDs are the `token_id` of each streamed chunk. `token_ids_sha256` is
//! the SHA-256 of the compact JSON array of the 66 ID arrays in receipt order:
//! prose, code and structured runs; the cold, then the warm 8K runs; the C1,
//! C2, C4 and C8 rounds and their streams.
use crate::evals::json::{self, object, Value};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const RIGMARK_REVISION: &str = "52c29c3fc1d741b9aef8895b23372aafa5dc6161";
pub const PROMPTS_SHA256: &str = "0c3ac4015813af6a1b967659bd2289990fde895a5e43ae065b894d1e1f909ecb";
const PROTOCOL_VERSION: &str = "1.3.0";
const NONCE_VERSION: &str = "1.1.0";
const COMPARISON_ID: &str = "2026-10-11-vade-rigmark-compat-thinking-off";
const SEED: i64 = 20260905;
const DECODE_RUNS: usize = 5;
const DECODE_MAX_TOKENS: i64 = 8192;
const PREFILL_DEPTH: usize = 8192;
const PREFILL_RUNS: usize = 3;
const PREFILL_MAX_TOKENS: i64 = 8;
const LEVELS: [usize; 4] = [1, 2, 4, 8];
const ROUNDS: usize = 3;
const CONCURRENCY_MAX_TOKENS: i64 = 256;
const CONCURRENCY_WORKLOAD: &str = "code";
/// The workloads in receipt order. `prompts.json` must hold exactly these.
pub const WORKLOADS: [&str; 3] = ["prose", "code", "structured"];
/// 15 single-stream chats, 3 cold and 3 warm 8K prompts, 3 rounds of 1+2+4+8 streams.
pub const RESPONSES: usize = WORKLOADS.len() * DECODE_RUNS + 2 * PREFILL_RUNS + ROUNDS * (1 + 2 + 4 + 8);
const TIMEOUT: Duration = Duration::from_secs(7200);
const MAX_BODY: u64 = 64 * 1024 * 1024;
const MAX_LINE: u64 = 4 * 1024 * 1024;

#[derive(Debug, Default, PartialEq)]
struct Options {
    url: Option<String>,
    prompts: Option<PathBuf>,
    draft_plain: bool,
    output: Option<PathBuf>,
    receipt: Option<PathBuf>,
    compare: Option<(PathBuf, PathBuf)>,
}

fn parse(arguments: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut index = 0;
    let value = |index: usize, flag: &str| arguments.get(index + 1).filter(|v| !v.is_empty())
        .cloned().ok_or_else(|| format!("Give {flag} a value. Run vade bench --help."));
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        match flag {
            "--card" => { index += 1; continue; }
            "--url" => options.url = Some(value(index, flag)?),
            "--prompts" => options.prompts = Some(value(index, flag)?.into()),
            "--output" | "--out" => options.output = Some(value(index, flag)?.into()),
            "--receipt" => options.receipt = Some(value(index, flag)?.into()),
            "--draft" => match value(index, flag)?.as_str() {
                "plain" => options.draft_plain = true,
                "served" => options.draft_plain = false,
                _ => return Err("With --card, --draft must be plain or served. Run vade bench --help.".into()),
            },
            "--compare" => {
                let second = arguments.get(index + 2).filter(|v| !v.is_empty())
                    .ok_or("Give --compare two card or receipt files. Run vade bench --help.")?;
                options.compare = Some((value(index, flag)?.into(), second.into()));
                index += 3;
                continue;
            }
            _ => return Err(format!("Unknown option {} for vade bench --card. Run vade bench --help.", crate::text::argument(flag))),
        }
        index += 2;
    }
    if options.compare.is_some() && (options.url.is_some() || options.prompts.is_some() || options.output.is_some() || options.receipt.is_some()) {
        return Err("--compare reads two files and sends no requests; give it no other option. Run vade bench --help.".into());
    }
    if options.compare.is_none() && options.prompts.is_none() {
        return Err(format!("Give --prompts the RigMark prompts.json at revision {}. Run vade bench --help.", &RIGMARK_REVISION[..7]));
    }
    Ok(options)
}

/// `vade bench --card ...`. Exit status 0 when the card ran (and matched the
/// receipt, if one was given), 1 when the token IDs differ, 130 after Ctrl-C.
/// An error is also exit status 1, through main's `fail`.
pub fn run(arguments: &[String], interrupted: fn() -> bool) -> Result<i32, String> {
    let options = parse(arguments)?;
    if let Some((mine, other)) = &options.compare {
        let verdict = compare(&json::read(mine).map_err(|e| file_error(mine, &e))?, &json::read(other).map_err(|e| file_error(other, &e))?)?;
        for line in &verdict.lines { say(line); }
        return Ok(if verdict.identical { 0 } else { 1 });
    }
    let path = options.prompts.as_deref().unwrap_or(Path::new(""));
    let prompts = load_prompts(&std::fs::read(path).map_err(|e| crate::text::file_error("Cannot read --prompts", path, &e))?)?;
    card_file(&options, &prompts, interrupted)
}

/// Run the card, save it (complete, or partial once a reply has arrived), print the cells and the
/// verdict against `--receipt`.
fn card_file(options: &Options, prompts: &Prompts, interrupted: fn() -> bool) -> Result<i32, String> {
    let output = options.output.clone().unwrap_or_else(|| "vade-card.json".into());
    if output.exists() {
        return Err(format!("{} exists. Give --output a new file name; the card never overwrites a file.", output.display()));
    }
    let receipt = match &options.receipt {
        Some(path) => Some(json::read(path).map_err(|e| file_error(path, &e))?),
        None => None,
    };
    if let Some(receipt) = &receipt && !card_like(receipt) && receipt["token_ids_sha256"].as_str().is_none() {
        return Err("--receipt is neither a card with token IDs nor a receipt with token_ids_sha256.".into());
    }
    let url = match options.url.clone() { Some(url) => url, None => crate::endpoint::resolve(None, false)?.0 };
    // --api-key-file or VADE_API_KEY_FILE, taken out of the arguments by main.
    let server = Server { key: crate::auth::bearer_key()?.map(str::to_owned), interrupted, ..Server::parse(&url)? };
    let mut card = object(vec![
        ("format", "vade-card-1".into()),
        ("card_draft", if options.draft_plain { "plain" } else { "served" }.into()),
        ("protocol", object(vec![
            ("rigmark_revision", RIGMARK_REVISION.into()), ("protocol_version", PROTOCOL_VERSION.into()),
            ("nonce_version", NONCE_VERSION.into()), ("comparison_id", COMPARISON_ID.into()),
            ("prompts_version", prompts.version.clone().into()), ("prompts_sha256", PROMPTS_SHA256.into()),
        ])),
        ("start_unix", now().into()),
    ]);
    let result = run_card(&server, prompts, options.draft_plain, &mut card, interrupted);
    card.insert("end_unix", now().into());
    if !interrupted() && let Ok(health) = server.json("GET", "/health", None) { card.insert("health_at_end", health); }
    let result = result.and_then(|()| digest(&card));
    let digest = match &result {
        Ok(digest) => Some(digest.clone()),
        Err(error) => { card.insert("complete", false.into()); card.insert("error", error.clone().into()); None }
    };
    if let Some(digest) = &digest {
        card.insert("complete", true.into());
        card.insert("responses", RESPONSES.into());
        card.insert("token_ids_sha256", digest.clone().into());
    }
    if let Err(error) = &result && card.get("decode").is_none() {
        // No reply arrived (no server, a wrong port, a refused key): keep the file name free for the next try.
        if interrupted() { say("Stopped before any reply; nothing was saved."); return Ok(130); }
        return Err(format!("{}. No reply arrived, so nothing was saved.", error.trim_end_matches('.')));
    }
    json::write_json(&output, &card).map_err(|e| format!("Cannot write {}: {e}", output.display()))?;
    say(&format!("Saved {}", output.display()));
    if let Err(error) = result {
        if interrupted() { say(&format!("Stopped. The partial card is in {}.", output.display())); return Ok(130); }
        return Err(format!("{}. The partial card is in {}.", error.trim_end_matches('.'), output.display()));
    }
    say(&summary(&card));
    let Some(receipt) = receipt else { return Ok(0) };
    let verdict = compare(&card, &receipt)?;
    for line in &verdict.lines { say(line); }
    Ok(if verdict.identical { 0 } else { 1 })
}

fn say(line: &str) { crate::output::report(format_args!("{line}"), true); }
fn file_error(path: &Path, error: &str) -> String { format!("Cannot read {}: {error}", path.display()) }
fn now() -> f64 { SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64()) }
fn round6(value: f64) -> f64 { (value * 1e6).round() / 1e6 }

/// The RigMark prompt set: the system message, the three workloads in file
/// order and the 8K prompt unit.
#[derive(Debug)]
pub struct Prompts {
    pub version: String,
    pub system: String,
    pub workloads: Vec<(String, String)>,
    pub prefill_unit: String,
}

pub fn load_prompts(bytes: &[u8]) -> Result<Prompts, String> {
    let sha = crate::sha256::sha(bytes);
    if sha != PROMPTS_SHA256 {
        return Err(format!("--prompts has SHA-256 {sha}. The card needs RigMark prompts.json at revision {RIGMARK_REVISION}, SHA-256 {PROMPTS_SHA256}."));
    }
    let value = json::parse(std::str::from_utf8(bytes).map_err(|_| "--prompts is not UTF-8")?)?;
    let text = |v: &Value, name: &str| v.as_str().map(str::to_owned).ok_or(format!("--prompts has no {name} text"));
    let workloads = match value.get("workloads") {
        Some(Value::Object(fields)) => fields.iter().map(|(k, v)| Ok((k.clone(), text(v, k)?))).collect::<Result<Vec<_>, String>>()?,
        _ => return Err("--prompts has no workloads".into()),
    };
    let mut names: Vec<&str> = workloads.iter().map(|(k, _)| k.as_str()).collect();
    names.sort_unstable();
    if names != ["code", "prose", "structured"] { return Err("--prompts must hold the prose, code and structured workloads".into()); }
    Ok(Prompts { version: text(&value["version"], "version")?, system: text(&value["system"], "system")?, workloads, prefill_unit: text(&value["prefill_unit"], "prefill_unit")? })
}

/// RigMark's request nonce: the first 32 hex digits of SHA-256 over the nonce
/// version, the comparison ID and the parts, joined with colons.
pub fn nonce(parts: &[&str]) -> String {
    let source = [NONCE_VERSION, COMPARISON_ID].iter().chain(parts).copied().collect::<Vec<_>>().join(":");
    crate::sha256::sha(source.as_bytes())[..32].to_owned()
}

/// True when the served model id names the GLM family. The GLM template has
/// no thinking-off form and its server refuses that ask, so GLM requests
/// carry the maker's low-effort field instead. Every other family keeps the
/// thinking-off ask its saved answers were recorded with.
pub fn is_glm_model(model: &str) -> bool { model.to_ascii_lowercase().contains("glm") }

/// One chat request body, in RigMark's field order with the card's extra fields.
pub fn chat_payload(model: &str, system: &str, prompt: &str, max_tokens: i64, seed: i64, draft_plain: bool) -> Value {
    let message = |role: &str, content: &str| object(vec![("role", role.into()), ("content", content.into())]);
    let glm = is_glm_model(model);
    let thinking: (&str, Value) = if glm { ("reasoning_effort", "low".into()) } else { ("chat_template_kwargs", object(vec![("enable_thinking", false.into())])) };
    let mut payload = object(vec![
        ("model", model.into()),
        ("messages", Value::Array(vec![message("system", system), message("user", prompt)])),
        ("temperature", 0.0.into()), ("top_p", 1.0.into()), ("seed", seed.into()), ("max_tokens", max_tokens.into()),
        ("stream", true.into()), ("stream_options", object(vec![("include_usage", true.into())])),
        thinking,
        ("top_k", 0i64.into()), ("repetition_penalty", 1i64.into()), ("presence_penalty", 0i64.into()),
        ("frequency_penalty", 0i64.into()), ("cache", false.into()),
    ]);
    if glm { payload.insert("mtp_depth", 0i64.into()); }
    if draft_plain { payload.insert("draft", "plain".into()); }
    payload
}

/// One cold or warm 8K prompt request body: the prompt is token IDs.
pub fn prefill_payload(model: &str, tokens: &[i64], cache_salt: &str) -> Value {
    object(vec![
        ("model", model.into()), ("prompt", Value::Array(tokens.iter().map(|t| Value::Int(*t)).collect())),
        ("add_special_tokens", false.into()), ("max_tokens", PREFILL_MAX_TOKENS.into()), ("ignore_eos", true.into()),
        ("temperature", 0.0.into()), ("top_p", 1.0.into()), ("top_k", 0i64.into()), ("repetition_penalty", 1.0.into()),
        ("presence_penalty", 0.0.into()), ("frequency_penalty", 0.0.into()), ("draft", "plain".into()),
        ("stream", true.into()), ("stream_options", object(vec![("include_usage", true.into())])),
        ("cache_salt", cache_salt.into()),
    ])
}

/// The 8K prompt: the tokenized nonce line, then the prompt unit repeated, cut to `depth`.
pub fn prefill_tokens(prefix: &[i64], unit: &[i64], depth: usize) -> Result<Vec<i64>, String> {
    if prefix.is_empty() || unit.is_empty() { return Err("the server tokenized the 8K prompt prefix or unit to nothing".into()); }
    if prefix.len() > depth { return Err(format!("the 8K prompt prefix is longer than {depth} tokens")); }
    let mut tokens = prefix.to_vec();
    while tokens.len() < depth { tokens.extend(unit.iter().take(depth - tokens.len())); }
    Ok(tokens)
}

fn run_card(server: &Server, prompts: &Prompts, draft_plain: bool, card: &mut Value, interrupted: fn() -> bool) -> Result<(), String> {
    let stop = || if interrupted() { Err("Interrupted.".to_owned()) } else { Ok(()) };
    let models = server.json("GET", "/v1/models", None)?;
    let model = match models["data"].as_array().map(Vec::as_slice) {
        Some([only]) => only["id"].as_str().ok_or("/v1/models has no model id")?.to_owned(),
        _ => return Err("The server must list exactly one model at /v1/models.".into()),
    };
    card.insert("model", model.clone().into());
    card.insert("health_at_start", server.json("GET", "/health", None)?);
    let mut decode = object(vec![]);
    for (name, prompt) in &prompts.workloads {
        say(&format!("decode/{name}: {DECODE_RUNS} runs"));
        let mut runs = Vec::new();
        for run in 1..=DECODE_RUNS {
            stop()?;
            let text = format!("Request nonce: {}\n\n{prompt}", nonce(&["decode", name, &run.to_string()]));
            let mut row = server.stream("/v1/chat/completions", &chat_payload(&model, &prompts.system, &text, DECODE_MAX_TOKENS, SEED, draft_plain))?;
            let gate = if name == "structured" { structured_gate(&row) } else { visible_gate(&row) };
            say(&format!("  {run}: {} tok/s, TTFT {:.3}s, {} tokens", row["decode_tokens_per_second"].as_f64().unwrap_or(0.0), row["ttft_seconds"].as_f64().unwrap_or(0.0), row["completion_tokens"].as_i64().unwrap_or(0)));
            row.insert("run", run.into());
            row.insert("completion_validation", match gate { Ok(()) => object(vec![("valid", true.into())]), Err(e) => object(vec![("valid", false.into()), ("error", e.into())]) });
            runs.push(row);
            // Save each reply as it arrives, so a stopped card keeps the finished ones.
            decode.insert(name.clone(), object(vec![("runs", Value::Array(runs.clone()))]));
            card.insert("decode", decode.clone());
        }
    }
    say(&format!("prefill/{PREFILL_DEPTH}: {PREFILL_RUNS} cold/warm pairs"));
    let tokenize = |text: &str| -> Result<Vec<i64>, String> {
        let body = object(vec![("model", model.clone().into()), ("prompt", text.into()), ("add_special_tokens", false.into())]);
        server.json("POST", "/tokenize", Some(&body))?["tokens"].as_array()
            .and_then(|ids| ids.iter().map(Value::as_i64).collect::<Option<Vec<_>>>()).ok_or("/tokenize returned no integer tokens".into())
    };
    let unit = tokenize(&prompts.prefill_unit)?;
    let (mut cold_runs, mut warm_runs) = (Vec::new(), Vec::new());
    for run in 1..=PREFILL_RUNS {
        stop()?;
        let prefix = tokenize(&format!("Unique cold-prefill nonce {}.\n", nonce(&["prefill", &PREFILL_DEPTH.to_string(), &run.to_string()])))?;
        let tokens = prefill_tokens(&prefix, &unit, PREFILL_DEPTH)?;
        // A fresh namespace per pair keeps the cold request cold; it does not change the prompt.
        let salt = crate::text::random_hex(32)?;
        let payload = prefill_payload(&model, &tokens, &salt);
        let mut cold = server.stream("/v1/completions", &payload)?;
        if cold["cached_prompt_tokens"].as_i64().is_some_and(|n| n > 0) {
            return Err("The cold 8K prompt reported cached tokens; the server did not keep cache_salt namespaces apart.".into());
        }
        let mut warm = server.stream("/v1/completions", &payload)?;
        for row in [&mut cold, &mut warm] {
            if row["prompt_tokens"].as_i64() != Some(PREFILL_DEPTH as i64) {
                return Err(format!("The server counted {} prompt tokens for an {PREFILL_DEPTH}-token prompt.", row["prompt_tokens"].as_i64().unwrap_or(-1)));
            }
            row.insert("run", run.into());
        }
        say(&format!("  {run}: cold TTFT {:.3}s, warm TTFT {:.3}s", cold["ttft_seconds"].as_f64().unwrap_or(0.0), warm["ttft_seconds"].as_f64().unwrap_or(0.0)));
        cold_runs.push(cold);
        warm_runs.push(warm);
        let depth = PREFILL_DEPTH.to_string();
        card.insert("prefill", object(vec![(depth.as_str(), object(vec![
            ("cold", object(vec![("runs", Value::Array(cold_runs.clone()))])), ("warm_replay", object(vec![("runs", Value::Array(warm_runs.clone()))]))]))]));
    }
    let prompt = &prompts.workloads.iter().find(|(name, _)| name == CONCURRENCY_WORKLOAD).ok_or("--prompts has no code workload")?.1;
    let mut concurrency = object(vec![]);
    for level in LEVELS {
        say(&format!("concurrency/{level}: {ROUNDS} rounds of {CONCURRENCY_WORKLOAD}"));
        let mut rounds = Vec::new();
        for round in 1..=ROUNDS {
            stop()?;
            let payloads: Vec<Value> = (0..level).map(|stream| {
                let text = format!("Request nonce: {}\n\n{prompt}", nonce(&["concurrency", CONCURRENCY_WORKLOAD, &level.to_string(), &round.to_string(), &(stream + 1).to_string()]));
                chat_payload(&model, &prompts.system, &text, CONCURRENCY_MAX_TOKENS, SEED + stream as i64, draft_plain)
            }).collect();
            let barrier = std::sync::Barrier::new(level);
            let started = Instant::now();
            let streams = std::thread::scope(|scope| {
                let tasks: Vec<_> = payloads.iter().map(|payload| { let barrier = &barrier; scope.spawn(move || { barrier.wait(); server.stream("/v1/chat/completions", payload) }) }).collect();
                tasks.into_iter().map(|task| task.join().unwrap_or_else(|_| Err("a stream thread failed".into()))).collect::<Result<Vec<_>, String>>()
            })?;
            let wall = started.elapsed().as_secs_f64();
            let completion: i64 = streams.iter().map(|row| row["completion_tokens"].as_i64().unwrap_or(0)).sum();
            let aggregate = completion as f64 / wall.max(1e-9);
            say(&format!("  {round}: aggregate {aggregate:.1} tok/s"));
            rounds.push(object(vec![("round", round.into()), ("wall_seconds", round6(wall).into()), ("completion_tokens", completion.into()),
                ("aggregate_end_to_end_tokens_per_second", ((aggregate * 1e3).round() / 1e3).into()), ("streams", Value::Array(streams))]));
            concurrency.insert(level.to_string(), object(vec![("rounds", Value::Array(rounds.clone()))]));
            card.insert("concurrency", concurrency.clone());
        }
    }
    Ok(())
}

/// Code and prose must end normally with visible text.
fn visible_gate(row: &Value) -> Result<(), String> {
    if row["output"].as_str().unwrap_or("").trim().is_empty() { return Err("no visible answer".into()); }
    normal_end(row)
}

fn normal_end(row: &Value) -> Result<(), String> {
    let reason = row["finish_reason"].as_str().unwrap_or("none");
    if reason != "stop" { return Err(format!("non-normal finish reason: {reason}")); }
    if row["stream_done_marker"] != Value::Bool(true) { return Err("stream ended without a [DONE] marker".into()); }
    Ok(())
}

/// Structured output must be a JSON array of 50 objects {index: i, square: i*i}.
pub fn structured_gate(row: &Value) -> Result<(), String> {
    normal_end(row)?;
    let value = json::parse(row["output"].as_str().unwrap_or("")).map_err(|e| format!("invalid JSON: {e}"))?;
    let items = value.as_array().filter(|items| items.len() == 50).ok_or("expected an array of 50 objects")?;
    for (index, item) in (1i64..).zip(items) {
        let fields = match item { Value::Object(fields) => fields, _ => return Err(format!("incorrect object at index {index}")) };
        let mut keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
        keys.sort_unstable();
        if keys != ["index", "square"] || item["index"].as_i64() != Some(index) || item["square"].as_i64() != Some(index * index) {
            return Err(format!("incorrect object at index {index}"));
        }
    }
    Ok(())
}

/// The 66 token ID arrays of a card file (this client's or a RigMark card
/// with token IDs), in receipt order, each with a label.
pub fn token_rows(card: &Value) -> Result<Vec<(String, &Value)>, String> {
    let mut rows = Vec::new();
    let count = |items: &Vec<Value>, want: usize, what: &str| if items.len() == want { Ok(()) } else { Err(format!("the card has {} {what}, not {want}", items.len())) };
    for name in WORKLOADS {
        let runs = card["decode"][name]["runs"].as_array().ok_or(format!("the card has no {name} runs; it is incomplete"))?;
        count(runs, DECODE_RUNS, &format!("{name} runs"))?;
        for (index, row) in runs.iter().enumerate() {
            rows.push((format!("{name} run {}", index + 1), &row["token_ids"]));
        }
    }
    for (part, label) in [("cold", "cold 8K"), ("warm_replay", "warm 8K")] {
        let runs = card["prefill"][PREFILL_DEPTH.to_string().as_str()][part]["runs"].as_array().ok_or(format!("the card has no {label} runs; it is incomplete"))?;
        count(runs, PREFILL_RUNS, &format!("{label} runs"))?;
        for (index, row) in runs.iter().enumerate() { rows.push((format!("{label} run {}", index + 1), &row["token_ids"])); }
    }
    for level in LEVELS {
        let rounds = card["concurrency"][level.to_string().as_str()]["rounds"].as_array().ok_or(format!("the card has no C{level} rounds; it is incomplete"))?;
        count(rounds, ROUNDS, &format!("C{level} rounds"))?;
        for (round, value) in rounds.iter().enumerate() {
            let streams = value["streams"].as_array().ok_or(format!("the card has no C{level} streams"))?;
            count(streams, level, &format!("C{level} streams in round {}", round + 1))?;
            for (stream, row) in streams.iter().enumerate() {
                rows.push((format!("C{level} round {} stream {}", round + 1, stream + 1), &row["token_ids"]));
            }
        }
    }
    if rows.len() != RESPONSES { return Err(format!("the card has {} responses, not {RESPONSES}", rows.len())); }
    if let Some((label, _)) = rows.iter().find(|(_, ids)| !ids.as_array().is_some_and(|ids| !ids.is_empty() && ids.iter().all(|id| matches!(id, Value::Int(n) if *n >= 0)))) {
        return Err(format!("the card has no token IDs for {label}"));
    }
    Ok(rows)
}

/// SHA-256 of the compact JSON array of the 66 token ID arrays in receipt order.
pub fn digest(card: &Value) -> Result<String, String> {
    let rows = token_rows(card)?;
    Ok(crate::sha256::sha(json::packed(&Value::Array(rows.into_iter().map(|(_, ids)| ids.clone()).collect())).as_bytes()))
}

#[derive(Debug, PartialEq)]
pub struct Verdict { pub identical: bool, pub lines: Vec<String> }

/// Compare a card with a receipt (digest only) or with another card (every array).
pub fn compare(card: &Value, other: &Value) -> Result<Verdict, String> {
    let mine = token_rows(card)?;
    let digest = digest(card)?;
    let mut lines = vec![format!("This card: token_ids_sha256 {digest}")];
    let identical = if card_like(other) {
        let theirs = token_rows(other)?;
        let same = mine.iter().zip(&theirs).filter(|((_, a), (_, b))| a == b).count();
        lines.push(format!("Other card: token_ids_sha256 {}", self::digest(other)?));
        lines.push(format!("ids {same}/{RESPONSES} identical"));
        if let Some(((label, a), (_, b))) = mine.iter().zip(&theirs).find(|((_, a), (_, b))| a != b) {
            let (a, b) = (a.as_array().map_or(&[][..], Vec::as_slice), b.as_array().map_or(&[][..], Vec::as_slice));
            let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
            lines.push(format!("First difference: {label}, token {at} ({} and {} tokens).", a.len(), b.len()));
        }
        same == RESPONSES
    } else {
        let theirs = other["token_ids_sha256"].as_str().ok_or("The other file is neither a card with token IDs nor a receipt with token_ids_sha256.")?;
        if other["responses"].as_i64().is_some_and(|n| n != RESPONSES as i64) {
            return Err(format!("The receipt covers {} responses, not {RESPONSES}.", other["responses"].as_i64().unwrap_or(0)));
        }
        lines.push(format!("Receipt: token_ids_sha256 {theirs}"));
        if theirs == digest {
            lines.push(format!("ids {RESPONSES}/{RESPONSES} identical"));
        } else {
            lines.push("ids differ from the receipt. The receipt holds only the digest, so compare with a full card to count the identical responses.".into());
        }
        theirs == digest
    };
    let draft = |v: &Value| v["card_draft"].as_str().map(str::to_owned);
    if let (Some(a), Some(b)) = (draft(card), draft(other)) && a != b {
        lines.push(format!("Note: this card's chats were sent {}; the other card's were sent {}. Drafting ON and OFF must give the same token IDs.", label(&a), label(&b)));
    }
    let prompts = |v: &Value| [&v["protocol"]["prompts_sha256"], &v["client"]["prompts_sha256"]].into_iter().find_map(|p| p.as_str().map(str::to_owned));
    if let (Some(a), Some(b)) = (prompts(card), prompts(other)) && a != b {
        lines.push(format!("Note: the prompt files differ ({} and {}).", &a[..a.len().min(8)], &b[..b.len().min(8)]));
    }
    let build = |v: &Value| [&v["server"]["git_sha"], &v["sha"], &first_echo(v)["build"]["git_sha"]].into_iter().find_map(|p| p.as_str().map(str::to_owned));
    if let (Some(a), Some(b)) = (build(card), build(other)) {
        lines.push(format!("Server build: this card {}, the other {}.", &a[..a.len().min(12)], &b[..b.len().min(12)]));
    }
    Ok(Verdict { identical, lines })
}

/// How a card's chats asked for drafting. "served" sends no draft field, so the server's own setting decides.
fn label(draft: &str) -> &str { if draft == "plain" { "with \"draft\":\"plain\"" } else { "as served (no draft field)" } }
fn card_like(value: &Value) -> bool { value.get("decode").is_some() }
fn first_echo(card: &Value) -> &Value { &card["decode"][WORKLOADS[0]]["runs"][0]["vade"] }

/// The card's cells with the receipts' names: median decode rate per
/// workload, median aggregate rate per concurrency, median cold 8K TTFT.
fn summary(card: &Value) -> String {
    let median = |mut values: Vec<f64>| -> f64 {
        values.sort_by(f64::total_cmp);
        let n = values.len();
        if n == 0 { 0.0 } else if n % 2 == 1 { values[n / 2] } else { (values[n / 2 - 1] + values[n / 2]) / 2.0 }
    };
    let rows = |v: &Value, key: &str| v.as_array().map(|rows| rows.iter().filter_map(|row| row[key].as_f64()).collect()).unwrap_or_default();
    let mut cells: Vec<String> = WORKLOADS.iter().map(|w| format!("{w} {:.1}", median(rows(&card["decode"][*w]["runs"], "decode_tokens_per_second")))).collect();
    cells.extend(LEVELS.iter().map(|l| format!("c{l} {:.1}", median(rows(&card["concurrency"][l.to_string().as_str()]["rounds"], "aggregate_end_to_end_tokens_per_second")))));
    let gates: usize = WORKLOADS.iter().map(|w| card["decode"][*w]["runs"].as_array().map_or(0, |runs| runs.iter().filter(|r| r["completion_validation"]["valid"] == Value::Bool(true)).count())).sum();
    format!("Cells (tok/s, median): {}. Cold 8K TTFT median {:.3}s. Output checks {gates}/{}.",
        cells.join(", "), median(rows(&card["prefill"][PREFILL_DEPTH.to_string().as_str()]["cold"]["runs"], "ttft_seconds")), WORKLOADS.len() * DECODE_RUNS)
}

/// A plain-HTTP server root such as `http://127.0.0.1:18080`, the bearer key
/// to send, and the interrupt check that stops a read in progress.
pub struct Server { authority: String, key: Option<String>, interrupted: fn() -> bool }

fn never() -> bool { false }

/// A socket that checks for Ctrl-C before every read and waits for data in
/// one-second steps, up to `TIMEOUT`. The signal handler only records SIGINT,
/// and std's buffered reads retry an interrupted read, so this check is what
/// stops a reply that is still streaming or has stalled.
struct Patient { socket: TcpStream, interrupted: fn() -> bool }

impl Read for Patient {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let started = Instant::now();
        loop {
            if (self.interrupted)() { return Err(std::io::Error::other("Interrupted")); }
            match self.socket.read(bytes) {
                Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {
                    if started.elapsed() >= TIMEOUT { return Err(e); }
                }
                result => return result,
            }
        }
    }
}

impl Server {
    pub fn parse(url: &str) -> Result<Self, String> {
        let rest = url.strip_prefix("http://").ok_or("The card client needs an http:// server URL, such as http://127.0.0.1:18080.")?;
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        if authority.is_empty() || authority.contains(['@', '?', '#', ' ']) || !matches!(path.trim_end_matches('/'), "" | "v1") {
            return Err("Give --url a server root such as http://127.0.0.1:18080, without credentials or a path.".into());
        }
        let authority = if authority.contains(':') && !authority.ends_with(']') { authority.to_owned() } else { format!("{authority}:80") };
        Ok(Self { authority, key: None, interrupted: never })
    }

    fn connect(&self, method: &str, path: &str, body: Option<&Value>) -> Result<BufReader<Patient>, String> {
        // Try every address of the name: localhost can resolve to ::1 before 127.0.0.1.
        let mut failure = "Cannot resolve the server.".to_owned();
        let mut socket = None;
        for address in self.authority.to_socket_addrs().map_err(|e| format!("Cannot resolve the server: {e}"))? {
            match TcpStream::connect_timeout(&address, Duration::from_secs(30)) {
                Ok(connected) => { socket = Some(connected); break; }
                Err(e) => failure = format!("Cannot connect to the server: {e}"),
            }
        }
        let mut socket = socket.ok_or(failure)?;
        socket.set_read_timeout(Some(Duration::from_secs(1))).and_then(|()| socket.set_write_timeout(Some(TIMEOUT))).and_then(|()| socket.set_nodelay(true)).map_err(|e| e.to_string())?;
        let body = body.map(json::packed).unwrap_or_default();
        let authorization = self.key.as_ref().map(|key| format!("Authorization: Bearer {key}\r\n")).unwrap_or_default();
        let content = if method == "POST" { format!("Content-Type: application/json\r\nContent-Length: {}\r\n", body.len()) } else { String::new() };
        let request = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\n{authorization}{content}Connection: close\r\n\r\n{body}", self.authority);
        socket.write_all(request.as_bytes()).map_err(|e| format!("Cannot send {path}: {e}"))?;
        Ok(BufReader::new(Patient { socket, interrupted: self.interrupted }))
    }

    /// One JSON request; the reply must be HTTP 200 with a JSON body.
    pub fn json(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
        let mut reader = self.connect(method, path, body)?;
        let headers = head(&mut reader, path)?;
        let mut bytes = Vec::new();
        match headers.iter().find(|(k, _)| k == "content-length") {
            Some((_, n)) => { let n: u64 = n.parse().map_err(|_| format!("{path}: invalid Content-Length"))?; if n > MAX_BODY { return Err(format!("{path}: reply too large")); } reader.take(n).read_to_end(&mut bytes) }
            None => reader.take(MAX_BODY).read_to_end(&mut bytes),
        }.map_err(|e| format!("{path}: {e}"))?;
        json::parse(std::str::from_utf8(&bytes).map_err(|_| format!("{path}: reply is not UTF-8"))?).map_err(|e| format!("{path}: {e}"))
    }

    /// One streamed completion: token IDs from each chunk's `token_id`, the
    /// server's `vade` echo, usage, and client-side timing.
    pub fn stream(&self, path: &str, payload: &Value) -> Result<Value, String> {
        let started = Instant::now();
        let mut reader = self.connect("POST", path, Some(payload))?;
        let headers = head(&mut reader, path)?;
        if !headers.iter().any(|(k, v)| k == "content-type" && v.to_ascii_lowercase().starts_with("text/event-stream")) {
            return Err(format!("{path} did not answer with an event stream"));
        }
        if headers.iter().any(|(k, _)| k == "transfer-encoding") { return Err(format!("{path}: chunked replies are not supported")); }
        let mut events = Events::default();
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = (&mut reader).take(MAX_LINE).read_until(b'\n', &mut line).map_err(|e| format!("{path}: {e}"))?;
            if read == 0 { break; }
            if events.line(std::str::from_utf8(&line).map_err(|_| format!("{path}: stream is not UTF-8"))?, started.elapsed().as_secs_f64())? { break; }
        }
        events.row(payload, started.elapsed().as_secs_f64())
    }
}

fn head(reader: &mut impl BufRead, path: &str) -> Result<Vec<(String, String)>, String> {
    let mut line = String::new();
    reader.by_ref().take(8192).read_line(&mut line).map_err(|e| format!("{path}: {e}"))?;
    let code = line.split_whitespace().nth(1).unwrap_or("").to_owned();
    let mut headers = Vec::new();
    loop {
        let mut header = String::new();
        if reader.by_ref().take(8192).read_line(&mut header).map_err(|e| format!("{path}: {e}"))? == 0 || header.trim().is_empty() { break; }
        if let Some((k, v)) = header.split_once(':') { headers.push((k.trim().to_ascii_lowercase(), v.trim().to_owned())); }
        if headers.len() > 200 { return Err(format!("{path}: too many headers")); }
    }
    if code != "200" {
        let mut body = Vec::new();
        let _ = reader.by_ref().take(2048).read_to_end(&mut body);
        let body: String = String::from_utf8_lossy(&body).chars().filter(|c| !c.is_control()).take(300).collect();
        return Err(format!("{path} answered HTTP {}: {body}", if code.is_empty() { "?" } else { &code }));
    }
    Ok(headers)
}

/// The state of one event stream, read one line at a time.
#[derive(Default)]
pub struct Events {
    token_ids: Vec<Value>,
    vade: Option<Value>,
    usage: Option<Value>,
    finish_reason: Option<String>,
    output: String,
    reasoning: usize,
    first: Option<f64>,
    last: f64,
    measured: usize,
    done: bool,
}

impl Events {
    /// Read one SSE line seen `at` seconds after the request; true at `[DONE]`.
    pub fn line(&mut self, line: &str, at: f64) -> Result<bool, String> {
        let Some(data) = line.trim().strip_prefix("data:") else { return Ok(false) };
        let data = data.trim();
        if data == "[DONE]" { self.done = true; return Ok(true); }
        let event = json::parse(data).map_err(|e| format!("invalid stream event: {e}"))?;
        if let Some(vade) = event.get("vade").filter(|v| **v != Value::Null) { self.vade = Some(vade.clone()); }
        if let Some(usage @ Value::Object(_)) = event.get("usage") { self.usage = Some(usage.clone()); }
        let Some(choices) = event["choices"].as_array().filter(|c| !c.is_empty()) else { return Ok(false) };
        let choice = &choices[0];
        if choices.len() != 1 || !matches!(choice, Value::Object(_)) || choice["index"].as_i64().unwrap_or(0) != 0 {
            return Err("the stream returned several choices or a nonzero index".into());
        }
        match &choice["token_id"] { Value::Null => {}, id @ Value::Int(_) => self.token_ids.push(id.clone()), _ => return Err("the stream sent a token_id that is not an integer".into()) }
        if let Some(reason) = choice["finish_reason"].as_str() { self.finish_reason = Some(reason.into()); }
        let delta = &choice["delta"];
        let reasoning: String = ["reasoning", "reasoning_content"].iter().filter_map(|k| delta[*k].as_str()).collect();
        let content = delta["content"].as_str().filter(|c| !c.is_empty()).or_else(|| choice["text"].as_str()).unwrap_or("");
        if !reasoning.is_empty() || !content.is_empty() {
            self.first.get_or_insert(at);
            self.last = at;
            self.measured += 1;
            self.reasoning += reasoning.chars().count();
            self.output.push_str(content);
        }
        Ok(false)
    }

    /// The saved row for this response.
    pub fn row(self, payload: &Value, wall: f64) -> Result<Value, String> {
        let first = self.first.ok_or("the stream sent no output")?;
        let usage = self.usage.ok_or("the server sent no final token usage")?;
        let count = |key: &str| usage[key].as_i64().filter(|n| *n >= 0).ok_or(format!("the server sent no integer {key}"));
        let (prompt, completion) = (count("prompt_tokens")?, count("completion_tokens")?);
        // Every generated token comes with its token_id, except a stop token, which ends the reply
        // without one. Any other count means the IDs cannot be compared.
        let ids = self.token_ids.len() as i64;
        if ids == 0 || !(ids == completion || (ids + 1 == completion && self.finish_reason.as_deref() == Some("stop"))) {
            return Err(format!("the server sent {ids} token IDs for {completion} completion tokens (finish reason {})", self.finish_reason.as_deref().unwrap_or("none")));
        }
        let vade = self.vade.ok_or("the server sent no vade echo; the card needs a Vade server")?;
        check_echo(payload, &vade)?;
        let window = self.last - first;
        if completion > 1 && (self.measured < 2 || window <= 0.0) { return Err("the server sent the reply in one event; the decode rate cannot be measured".into()); }
        let rate = if completion <= 1 { 0.0 } else { (completion - 1) as f64 / window };
        Ok(object(vec![
            ("request_body", payload.clone()), ("vade", vade), ("token_ids", Value::Array(self.token_ids)),
            ("prompt_tokens", prompt.into()), ("completion_tokens", completion.into()),
            ("cached_prompt_tokens", usage["prompt_tokens_details"]["cached_tokens"].as_i64().map_or(Value::Null, Value::Int)),
            ("ttft_seconds", round6(first).into()), ("time_to_last_output_seconds", round6(self.last).into()),
            ("decode_seconds", round6(window.max(0.0)).into()), ("decode_tokens_per_second", ((rate * 1e3).round() / 1e3).into()),
            ("measured_sse_events", self.measured.into()), ("wall_seconds", round6(wall).into()),
            ("stream_done_marker", self.done.into()), ("finish_reason", self.finish_reason.map_or(Value::Null, Value::Str)),
            ("output", self.output.into()), ("reasoning_characters", self.reasoning.into()),
        ]))
    }
}

/// The server must echo the sampling and thinking settings it was sent, or the
/// IDs would not belong to this card's requests.
pub fn check_echo(payload: &Value, vade: &Value) -> Result<(), String> {
    for field in ["temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty", "frequency_penalty", "seed"] {
        let Some(sent) = payload.get(field) else { continue };
        let echoed = &vade["sampling"][field];
        let same = match (sent, echoed) { (Value::Int(a), Value::Int(b)) => a == b, _ => sent.as_f64().is_some() && sent.as_f64() == echoed.as_f64() };
        if !same { return Err(format!("the server used {field} {} for a request that sent {}", json::packed(echoed), json::packed(sent))); }
    }
    if let Value::Bool(thinking) = &payload["chat_template_kwargs"]["enable_thinking"] && vade["thinking"] != Value::Bool(*thinking) {
        return Err("the server did not use the requested thinking mode".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "card_tests.rs"]
mod tests;
