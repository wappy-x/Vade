//! Token and accounting determinism over the existing streaming HTTP protocol.
use super::{Fixture, Result};
#[allow(dead_code)]
#[path = "../../../../../tools/bench/src/http.rs"] mod http;
use engine_model::json::{self, Value};
use engine_tokenizer::chat::quote;
use std::{collections::BTreeMap, fs, sync::{Arc, Barrier, atomic::AtomicBool}, time::{Duration, Instant}};

fn encode(value: &Value) -> String {
    match value {
        Value::Null => "null".into(), Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.clone(), Value::String(v) => quote(v),
        Value::Array(v) => format!("[{}]", v.iter().map(encode).collect::<Vec<_>>().join(",")),
        Value::Object(v) => format!("{{{}}}", v.iter().map(|(k, v)|
            format!("{}:{}", quote(k), encode(v))).collect::<Vec<_>>().join(",")),
    }
}
fn response(reply: &http::Reply, prompt: usize) -> Result<Vec<u8>> {
    if reply.status != "ok" { return Err(format!("server {}: {}", reply.status, reply.error.as_deref().unwrap_or("missing reply"))); }
    reply.verify_greedy()?;
    eprintln!("server_sampling prompt={prompt} vade={}", encode(reply.vade.as_ref().unwrap()));
    let (input, output, total, cached) = (reply.prompt_tokens.ok_or("missing prompt usage")?,
        reply.completion_tokens.ok_or("missing completion usage")?, reply.total_tokens.ok_or("missing total usage")?,
        reply.cached_tokens.ok_or("missing cache usage")?);
    if input != prompt as u64 || output == 0 || output > 64 || total != input + output || cached > input {
        return Err(format!("invalid server usage: prompt={input} output={output} total={total} cached={cached}, expected prompt={prompt}"));
    }
    let finish = reply.finish_reason.as_deref().ok_or("missing finish reason")?;
    if !matches!(finish, "stop" | "length") { return Err(format!("unexpected finish reason {finish}")); }
    // Generation charges terminal EOS to usage and returns Finished instead of
    // Token, so its ID is not streamed. Never invent that unobserved ID.
    let ids = reply.token_ids.len() as u64;
    if ids != output && !(finish == "stop" && ids + 1 == output) {
        return Err(format!("stream IDs={ids}, completion usage={output}, finish={finish}"));
    }
    let mut bytes = reply.token_ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    bytes.extend_from_slice(&(finish.len() as u64).to_le_bytes());
    bytes.extend_from_slice(finish.as_bytes());
    for n in [input, output, total] { bytes.extend_from_slice(&n.to_le_bytes()); }
    Ok(bytes)
}

fn cohort(endpoint: &http::Endpoint, bodies: &[&str], deadline: Instant,
    cancel: &Arc<AtomicBool>) -> Result<Vec<http::Reply>> {
    let barrier = Barrier::new(bodies.len());
    std::thread::scope(|scope| {
        let jobs = bodies.iter().map(|&body| {
            let (barrier, cancel) = (&barrier, cancel.clone());
            scope.spawn(move || {
                barrier.wait();
                http::request(endpoint, body, deadline, cancel, None)
            })
        }).collect::<Vec<_>>();
        jobs.into_iter().map(|job| job.join().map_err(|_| "server gate client panicked".to_owned()))
            .collect::<Result<Vec<_>>>()
    })
}

fn compare(expected: &[u8], actual: &[u8], prompt: usize, case: &str, row: usize) -> (usize, usize) {
    let different = expected.iter().zip(actual).filter(|(a, b)| a != b).count()
        + expected.len().abs_diff(actual.len());
    if different != 0 {
        let offset = expected.iter().zip(actual).position(|(a, b)| a != b).unwrap_or(expected.len().min(actual.len()));
        eprintln!("first_difference server prompt={prompt} case={case} row={row} byte={offset} token_position={}", offset/4);
    }
    (expected.len().max(actual.len()), different)
}

// Cold must bypass cache; the sequential replay must exercise a real hit.
// Concurrent admission can reclaim idle prefixes from the elastic pool.
// A cohort miss still compares its complete output with the cold reference.
fn cache_valid(prompt: usize, case: &str, row: usize, cached: u64) -> bool {
    let valid = match case {
        "cold" => cached == 0,
        "cache-hit" => cached > 0,
        _ => true,
    };
    if !valid {
        eprintln!("cache_reuse_failed prompt={prompt} case={case} row={row} cached_tokens={cached}");
    } else if cached == 0 && case != "cold" {
        eprintln!("cache_miss prompt={prompt} case={case} row={row} retention=elastic");
    }
    valid
}

#[cfg(test)] #[test] fn cache_checks_respect_elastic_retention() {
    for (case, cached, valid) in [("cold", 0, true), ("cold", 1, false), ("cache-hit", 0, false), ("cache-hit", 8192, true), ("cohort-2", 0, true), ("cohort-8", 0, true), ("cohort-16", 0, true), ("cohort-mixed", 0, true)] { assert_eq!(cache_valid(8192, case, 14, cached), valid, "{case}"); }
}

pub fn run() -> Result<()> {
    let args = std::env::args().skip(2).collect::<Vec<_>>();
    if args.len() != 2 && !(args.len() == 4 && args[2] == "--bodies") {
        return Err("usage: flash-bench --gate-server URL FIXTURE.json[,FIXTURE.json...] [--bodies REQUEST.json[,REQUEST.json...]]; URL is an http:// root; bodies use /v1/chat/completions, otherwise exact prompt IDs use /v1/completions".into());
    }
    let fixtures = args[1].split(',').map(Fixture::read).collect::<Result<Vec<_>>>()?;
    let mut lengths = fixtures.iter().map(|f| f.prompt.len()).collect::<Vec<_>>();
    lengths.sort_unstable();
    if lengths != [285, 2048, 8192] { return Err("server gate needs the three 285/2048/8192-token fixtures".into()); }
    let bodies = if args.len() == 4 { Some(args[3].split(',').map(|p| {
        json::parse(&fs::read(p).map_err(|e| e.to_string())?, json::Limits::default())
            .map_err(|e| e.to_string())?.object().map(|v| v.clone()).map_err(|e| e.to_string())
    }).collect::<Result<Vec<_>>>()?) } else { None };
    if bodies.as_ref().is_some_and(|b| b.len() != fixtures.len()) { return Err("server bodies must correspond to the three fixtures in order".into()); }
    let url = format!("{}/v1/{}", args[0].trim_end_matches('/'),
        if bodies.is_some() { "chat/completions" } else { "completions" });
    let endpoint = http::Endpoint::parse(&url)?;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(900);
    let cancel = Arc::new(AtomicBool::new(false));
    let mut failures = false;
    let mut mixed_bodies = Vec::new();
    let mut references = Vec::new();
    println!("case rows bytes_compared bytes_different result");
    for (index, fixture) in fixtures.iter().enumerate() {
        let mut body = bodies.as_ref().map_or_else(BTreeMap::new, |b| b[index].clone());
        if bodies.is_none() {
            body.insert("prompt_token_ids".into(), Value::Array(fixture.prompt.iter().map(|t| Value::Number(t.to_string())).collect()));
        }
        // Every cohort sends the same greedy request; only cold's cache setting differs.
        for (key, value) in [("stream", Value::Bool(true)), ("max_tokens", Value::Number("64".into())),
            ("temperature", Value::Number("0".into())), ("top_p", Value::Number("1".into())),
            ("top_k", Value::Number("0".into())), ("seed", Value::Number("0".into())),
            ("repetition_penalty", Value::Number("1".into())),
            ("presence_penalty", Value::Number("0".into())), ("frequency_penalty", Value::Number("0".into())),
            ("draft", Value::String("plain".into())), ("include_usage_timing", Value::Bool(false))] {
            body.insert(key.into(), value);
        }
        let mut reference = None;
        for (case, count, caching) in [("cold", 1, false), ("prime", 1, true),
            ("cache-hit", 1, true), ("cohort-2", 2, true), ("cohort-8", 8, true),
            ("cohort-16", 16, true)] {
            body.insert("cache".into(), Value::Bool(caching));
            let body = encode(&Value::Object(body.clone()));
            let replies = cohort(&endpoint, &vec![body.as_str(); count], deadline, &cancel)?;
            let (mut compared, mut different, mut valid) = (0, 0, true);
            for (row, reply) in replies.iter().enumerate() {
                let actual = match response(reply, fixture.prompt.len()) {
                    Ok(bytes) => bytes,
                    Err(error) => { eprintln!("server_error prompt={} case={case} row={row}: {error}", fixture.prompt.len()); valid = false; continue; }
                };
                valid &= cache_valid(fixture.prompt.len(), case, row, reply.cached_tokens.unwrap());
                if case == "cold" { reference = Some(actual); continue; }
                let expected = reference.as_ref().ok_or("server cold reference failed")?;
                let (bytes, changed) = compare(expected, &actual, fixture.prompt.len(), case, row);
                compared += bytes;
                different += changed;
            }
            let pass = valid && different == 0;
            // Cache coverage is diagnostic separately from byte equality.
            let hits = replies.iter().filter(|reply| reply.cached_tokens.is_some_and(|n| n > 0)).count();
            eprintln!("cache_coverage prompt={} case={case} hits={hits} rows={count}", fixture.prompt.len());
            println!("server-{}-{case} {count} {compared} {different} {}", fixture.prompt.len(),
                if !pass { "FAIL" } else if case == "cold" { "REF" } else { "PASS" });
            failures |= !pass;
        }
        references.push(reference.ok_or("server cold reference failed")?);
        mixed_bodies.push(encode(&Value::Object(body)));
    }
    let replies = cohort(&endpoint, &mixed_bodies.iter().map(String::as_str).collect::<Vec<_>>(), deadline, &cancel)?;
    let (mut compared, mut different, mut valid) = (0, 0, true);
    for (row, (fixture, reply)) in fixtures.iter().zip(&replies).enumerate() {
        match response(reply, fixture.prompt.len()) {
            Ok(actual) => {
                let (bytes, changed) = compare(&references[row], &actual, fixture.prompt.len(), "cohort-mixed", row);
                compared += bytes; different += changed;
                valid &= cache_valid(fixture.prompt.len(), "cohort-mixed", row, reply.cached_tokens.unwrap());
            }
            Err(error) => { eprintln!("server_error prompt={} case=cohort-mixed row={row}: {error}", fixture.prompt.len()); valid = false; }
        }
    }
    println!("server-cohort-mixed {} {compared} {different} {}", replies.len(), if valid && different == 0 { "PASS" } else { "FAIL" });
    failures |= !valid || different != 0;
    eprintln!("server_gate complete elapsed_ms={:.3}; verified echoed greedy sampling; compared token IDs, finish reason, prompt/completion/total usage; cached-token accounting checked separately", started.elapsed().as_secs_f64()*1000.0);
    if failures { Err("server cache determinism differs".into()) } else { Ok(()) }
}
