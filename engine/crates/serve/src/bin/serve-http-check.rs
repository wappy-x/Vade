//! Focused actual-server check: branch equality, foreground latency, cancellation.
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    time::{Duration, Instant},
};

fn connect(
    address: SocketAddr,
    prompt: &[u32],
    output: usize,
    extra: &str,
    draft: Option<&str>,
) -> Result<TcpStream, String> {
    let mut socket = TcpStream::connect(address).map_err(|e| e.to_string())?;
    socket
        .set_read_timeout(Some(Duration::from_secs(180)))
        .map_err(|e| e.to_string())?;
    let draft = draft.map(str::to_owned).unwrap_or_else(|| std::env::var("VADE_CHECK_DRAFT").unwrap_or_else(|_| "plain".into()));
    let body = format!(
        "{{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":{prompt:?},\"max_tokens\":{output},\"draft\":{}{extra}}}",
        spark_serve::protocol::quote(&draft)
    );
    println!("REQUEST prompt_token_ids={prompt:?} max_tokens={output} draft={draft} cache_disabled={} session_supplied={} fork={}",
        extra.contains("\"cache\":false"), extra.contains("\"session\":"), extra.contains("\"fork\":true"));
    write!(
        socket,
        "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .map_err(|e| e.to_string())?;
    Ok(socket)
}
struct Reply {
    tokens: Vec<u32>,
    key: Option<String>,
    reused: usize,
    ttft_ms: f64,
    total_ms: f64,
}
fn run(address: SocketAddr, prompt: &[u32], output: usize, extra: &str, draft: Option<&str>) -> Result<Reply, String> {
    let start = Instant::now();
    let mut socket = connect(address, prompt, output, extra, draft)?;
    let mut wire = Vec::new();
    let mut ttft_ms = None;
    let mut buffer = [0; 8192];
    loop {
        let n = socket.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        wire.extend_from_slice(&buffer[..n]);
        if ttft_ms.is_none() && wire.windows(10).any(|w| w == b"\"token_id\"") {
            ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.);
        }
        if wire.len() > 4 * 1024 * 1024 {
            return Err("check response too large".into());
        }
    }
    let text = String::from_utf8(wire).map_err(|e| e.to_string())?;
    let reply = parse_reply(&text, ttft_ms.unwrap_or_default(), start.elapsed().as_secs_f64() * 1000.)?;
    println!("REPLY token_ids={:?} reused={} session_returned={}", reply.tokens, reply.reused, reply.key.is_some());
    Ok(reply)
}
fn parse_reply(text: &str, ttft_ms: f64, total_ms: f64) -> Result<Reply, String> {
    if !text.starts_with("HTTP/1.1 200")
        || text.contains("generation_error")
        || !text.ends_with("data: [DONE]\n\n")
    {
        return Err("server response was unsuccessful".into());
    }
    let header = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(name))
            .map(str::trim)
    };
    let key = header("X-Session-Capability:").map(str::to_owned);
    let mut tokens = Vec::new();
    let mut reused = None;
    for data in text.lines().filter_map(|line| line.strip_prefix("data: ")).filter(|data| *data != "[DONE]") {
        let value = engine_model::json::parse(data.as_bytes(), engine_model::json::Limits::default()).map_err(|e| e.to_string())?;
        let object = value.object().map_err(|e| e.to_string())?;
        if let Some(choices) = object.get("choices") {
            for choice in choices.array().map_err(|e| e.to_string())? {
                if let Some(token) = choice.object().map_err(|e| e.to_string())?.get("token_id") {
                    tokens.push(u32::try_from(token.u64().map_err(|e| e.to_string())?).map_err(|_| "invalid token ID")?);
                }
            }
        }
        // The reuse header is opt-in and is sent before recurrent replay. Final usage
        // always reports the actual restored rows, including a cache miss.
        if let Some(usage) = object.get("usage").filter(|v| **v != engine_model::json::Value::Null) {
            let usage = usage.object().map_err(|e| e.to_string())?;
            let details = engine_model::json::field(usage, "prompt_tokens_details").map_err(|e| e.to_string())?
                .object().map_err(|e| e.to_string())?;
            reused = Some(usize::try_from(engine_model::json::field(details, "cached_tokens").map_err(|e| e.to_string())?
                .u64().map_err(|e| e.to_string())?).map_err(|_| "invalid reused count")?);
        }
    }
    if tokens.is_empty() { return Err("no generated token".into()); }
    Ok(Reply {
        tokens,
        key,
        reused: reused.ok_or("missing cached token usage")?,
        ttft_ms,
        total_ms,
    })
}
fn compare_cached(label: &str, reference: &Reply, repeated: &Reply, prompt_tokens: usize) -> Result<&'static str, String> {
    if reference.tokens != repeated.tokens {
        let at = reference.tokens.iter().zip(&repeated.tokens).position(|(a, b)| a != b)
            .unwrap_or(reference.tokens.len().min(repeated.tokens.len()));
        return Err(format!("{label} token-ID mismatch: index={at} reference={:?} repeated={:?} reference_tokens={} repeated_tokens={} reused={}",
            reference.tokens.get(at), repeated.tokens.get(at), reference.tokens.len(), repeated.tokens.len(), repeated.reused));
    }
    // A session capability does not restrict automatic prefix selection. An
    // earlier request can supply more rows than the named parent's prefix.
    // Without the selected snapshot's identity, only the request is a bound.
    if repeated.reused > prompt_tokens { return Err(format!("{label} invalid reuse count: reused={} prompt_tokens={prompt_tokens}", repeated.reused)); }
    // Recurrent backends can restore a checkpoint at or below the prompt end, or none.
    // A miss still checks token equality, but does not establish cached exactness.
    Ok(if repeated.reused == 0 { "SKIP" } else { "PASS" })
}
fn read_health(address: SocketAddr) -> Result<String, String> {
    let mut health = TcpStream::connect(address).map_err(|e| e.to_string())?;
    health.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    health.write_all(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n").as_bytes()).map_err(|e| e.to_string())?;
    let mut response = String::new();
    health.read_to_string(&mut response).map_err(|e| e.to_string())?;
    if !response.starts_with("HTTP/1.1 200") { return Err("health response was unsuccessful".into()); }
    let body = response.split_once("\r\n\r\n").ok_or("missing health body")?.1;
    engine_model::json::parse(body.as_bytes(), Default::default()).map_err(|e| e.to_string())?;
    Ok(body.to_owned())
}
fn check() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--health") {
        let address = args.get(1).ok_or("usage: serve-http-check --health ADDRESS")?.parse().map_err(|_| "invalid address")?;
        println!("{}", read_health(address)?);
        return Ok(());
    }
    let address: SocketAddr = args
        .first()
        .ok_or("usage: serve-http-check ADDRESS [OUTPUT_TOKENS] [FIXTURE.json]")?
        .parse()
        .map_err(|_| "invalid address")?;
    let long_tokens = args
        .get(1)
        .map(|s| s.parse::<usize>().map_err(|_| "invalid output token count"))
        .transpose()?
        .unwrap_or(64);
    let prompt: Vec<u32> = if let Some(path) = args.get(2) {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let value = engine_model::json::parse(&bytes, engine_model::json::Limits::default()).map_err(|e| e.to_string())?;
        engine_model::json::field(value.object().map_err(|e| e.to_string())?, "prompt_ids").map_err(|e| e.to_string())?
            .array().map_err(|e| e.to_string())?.iter().map(|v| v.u64().map(|n| n as u32).map_err(|e| e.to_string())).collect::<Result<_, _>>()?
    } else { (0..16).map(|i| [1, 42, 1024, 17][i % 4]).collect() };
    check_requests(address, long_tokens, &prompt, None, std::env::var("VADE_CHECK_AUTOMATIC").as_deref() == Ok("1"))
}
fn check_requests(address: SocketAddr, long_tokens: usize, prompt: &[u32], draft: Option<&str>, automatic: bool) -> Result<(), String> {
    if prompt.len() <= 8 { return Err("fixture needs more than 8 prompt tokens".into()); }
    let run = |address, prompt: &[u32], output, extra: &str| run(address, prompt, output, extra, draft);
    let (mut hits, mut misses) = (0, 0);
    let mut compare = |label: &str, reference: &Reply, repeated: &Reply, prompt_tokens| {
        let status = compare_cached(label, reference, repeated, prompt_tokens)?;
        if status == "PASS" { hits += 1; } else { misses += 1; }
        Ok::<_, String>(status)
    };
    if automatic {
        let parent = run(address, &prompt, long_tokens, "")?;
        let mut suffix = prompt.to_vec(); suffix.extend_from_slice(&parent.tokens); suffix.extend_from_slice(&[220, 17, 42]);
        for shared in [prompt.len() + parent.tokens.len().saturating_sub(1), prompt.len() - 1] {
            if shared < prompt.len() { suffix[shared] = (suffix[shared] + 1) % 1000; }
            let cold = run(address, &suffix, long_tokens, ",\"cache\":false")?;
            let warm = run(address, &suffix, long_tokens, "")?;
            let status = compare("automatic suffix", &cold, &warm, suffix.len())?;
            if cold.reused != 0 { return Err("cache-disabled request restored prefix tokens".into()); }
            println!("{status} automatic_prefix token_ids_equal=true shared={shared} reused={} output={} cold_ttft_ms={:.3} warm_ttft_ms={:.3}", warm.reused, warm.tokens.len(), cold.ttft_ms, warm.ttft_ms);
        }
        println!("CACHE_COVERAGE hits={hits} misses={misses} token_ids_equal=true named_parent_source=UNPROVEN");
        return Ok(());
    }
    let replay = run(address, &prompt, 1, ",\"cache\":true,\"priority\":1")?;
    let replay_key = replay.key.as_ref().ok_or("missing replay capability")?;
    let repeated = run(address, &prompt, 1, &format!(",\"cache\":true,\"priority\":1,\"session\":\"{replay_key}\",\"fork\":true"))?;
    let status = compare("exact replay", &replay, &repeated, prompt.len())?;
    println!("{status} exact_replay token_ids_equal=true prompt_tokens={} cold_ttft_ms={:.3} cached_ttft_ms={:.3} reused={}",
        prompt.len(), replay.ttft_ms, repeated.ttft_ms, repeated.reused);
    let cold = run(address, &prompt, 4, ",\"cache\":true,\"priority\":1")?;
    let key = cold.key.as_ref().ok_or("missing session capability")?;
    let cached = format!(",\"cache\":true,\"priority\":1,\"session\":\"{key}\",\"fork\":true");
    let warm = run(address, &prompt, 4, &cached)?;
    let status = compare("prompt fork", &cold, &warm, prompt.len())?;
    if warm.key == cold.key {
        return Err("prompt fork reused the parent capability".into());
    }
    println!(
        "{status} prompt_fork token_ids_equal=true cold_prompt_tokens={} cold_ttft_ms={:.3} warm_ttft_ms={:.3} reused={} output={:?}",
        prompt.len(), cold.ttft_ms, warm.ttft_ms, warm.reused, warm.tokens
    );
    let mut divergent = prompt.to_vec();
    divergent[8] = 220;
    let shared = prompt.iter().zip(&divergent).take_while(|(a, b)| a == b).count();
    let branch = run(address, &divergent, 4, &cached)?;
    let fresh = run(address, &divergent, 4, ",\"priority\":1,\"cache\":false")?;
    let branch_status = compare("divergent prefix fork", &fresh, &branch, divergent.len())?;
    if fresh.reused != 0 { return Err("cache-disabled request restored prefix tokens".into()); }
    let parent = run(address, &prompt, 4, &cached)?;
    let parent_status = compare("parent replay", &cold, &parent, prompt.len())?;
    println!("{parent_status} parent_replay token_ids_equal=true reused={}", parent.reused);
    println!(
        "{branch_status} divergent_prefix token_ids_equal=true shared_prefix={shared} reused={} reuse_bound={} named_parent_source=UNPROVEN exceeds_named_parent={} branch_ttft_ms={:.3} fresh_ttft_ms={:.3} parent_reused={} parent_cache={parent_status} output={:?}",
        branch.reused, divergent.len(), branch.reused > shared, branch.ttft_ms, fresh.ttft_ms, parent.reused, branch.tokens
    );
    let isolated = run(address, &prompt[..4], 4, ",\"priority\":255")?;
    let background_prompt: Vec<_> = (0..512).map(|i| [1, 42, 1024, 17][i % 4]).collect();
    let mut background = connect(address, &background_prompt, 128, "", draft)?;
    let mut prefill = Vec::new();
    while !prefill.windows(9).any(|w| w == b": prefill") {
        let mut bytes = [0; 1024];
        let n = background.read(&mut bytes).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("background closed before prefill".into());
        }
        prefill.extend_from_slice(&bytes[..n]);
    }
    let foreground = run(address, &prompt[..4], 4, ",\"priority\":255")?;
    if foreground.tokens != isolated.tokens {
        return Err("interleaving changed foreground output".into());
    }
    drop(background);
    println!(
        "PASS main_prompt_tokens=4 isolated_ttft_ms={:.3} during_background_512_ttft_ms={:.3} foreground_total_ms={:.3}",
        isolated.ttft_ms, foreground.ttft_ms, foreground.total_ms
    );
    let long = run(address, &prompt[..4], long_tokens, ",\"priority\":1")?;
    if long.tokens.len() != long_tokens {
        return Err("long generation stopped before requested length".into());
    }
    println!(
        "PASS long_output_tokens={} ttft_ms={:.3} total_ms={:.3} decode_tokens_per_sec={:.3}",
        long.tokens.len(),
        long.ttft_ms,
        long.total_ms,
        long.tokens.len().saturating_sub(1) as f64 * 1000. / (long.total_ms - long.ttft_ms)
    );
    let response = read_health(address)?;
    if !response.contains("\"active\":0") || !response.contains("\"queued\":0") {
        return Err("disconnect left active or queued work".into());
    }
    println!("PASS disconnect_cleanup active=0 queued=0");
    println!("CACHE_COVERAGE hits={hits} misses={misses} token_ids_equal=true named_parent_source=UNPROVEN");
    Ok(())
}
fn main() {
    if let Err(error) = check() {
        eprintln!("FAIL {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn wire(tokens: &[u32], reused: usize, header: &str) -> String {
        let mut wire = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Session-Capability: test-key\r\n{header}\r\n: prefill\n\n");
        for token in tokens { wire.push_str(&format!("data: {{\"choices\":[{{\"text\":\"x\",\"token_id\":{token}}}]}}\n\n")); }
        wire.push_str(&format!("data: {{\"choices\":[{{\"finish_reason\":\"length\"}}],\"usage\":{{\"prompt_tokens_details\":{{\"cached_tokens\":{reused}}}}}}}\n\ndata: [DONE]\n\n"));
        wire
    }
    pub(super) fn reply(tokens: &[u32], reused: usize) -> Reply { parse_reply(&wire(tokens, reused, ""), 1., 2.).unwrap() }

    #[test]
    fn final_usage_works_without_the_opt_in_reuse_header() {
        let r = reply(&[42], 16);
        assert_eq!((r.tokens, r.reused, r.key.as_deref()), (vec![42], 16, Some("test-key")));
        let r = parse_reply(&wire(&[42], 8, "X-Prefix-Reused-Tokens: 16\r\n"), 1., 2.).unwrap();
        assert_eq!(r.reused, 8);
    }

    #[test]
    fn a_cache_miss_skips_cache_coverage_but_checks_every_token() {
        let reference = reply(&[42], 0);
        let repeated = reply(&[42], 0);
        // A short-prompt control restored no prefix. A reuse requirement
        // must not serve as evidence of unequal emitted tokens.
        assert_eq!(compare_cached("exact replay", &reference, &repeated, 16).unwrap(), "SKIP");
        let error = compare_cached("exact replay", &reference, &reply(&[43], 0), 16).unwrap_err();
        assert!(error.contains("index=0 reference=Some(42) repeated=Some(43)"), "{error}");
    }

    #[test]
    fn recurrent_checkpoint_reuse_need_not_reach_the_prompt_end() {
        let reference = reply(&[42, 17], 0);
        for restored in [8, 16] {
            assert_eq!(compare_cached("prompt fork", &reference, &reply(&[42, 17], restored), 16).unwrap(), "PASS");
        }
        assert!(compare_cached("branch", &reference, &reply(&[42, 17], 9), 8).unwrap_err().contains("invalid reuse count"));
        for (tokens, at) in [(&[42, 18][..], 1), (&[42][..], 1), (&[42, 17, 18][..], 2)] {
            let error = compare_cached("prompt fork", &reference, &reply(tokens, 8), 16).unwrap_err();
            assert!(error.contains(&format!("index={at}")), "{error}");
        }
    }

    #[test]
    fn missing_usage_is_not_a_cache_miss() {
        let wire = "HTTP/1.1 200 OK\r\n\r\ndata: {\"choices\":[{\"token_id\":42}]}\n\ndata: [DONE]\n\n";
        assert_eq!(parse_reply(wire, 1., 2.).err().unwrap(), "missing cached token usage");
    }

    #[test]
    fn only_choice_token_ids_enter_the_comparison() {
        let wire = wire(&[42], 0, "").replace("\"text\":\"x\"", "\"text\":\"quoted \\\"token_id\\\":17\",\"logprobs\":{\"token_id\":18}");
        assert_eq!(parse_reply(&wire, 99., 101.).unwrap().tokens, vec![42]);
    }

    #[test]
    fn replay_over_http_distinguishes_a_miss_from_a_partial_hit() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for reused in [0, 8] {
                let (mut socket, _) = listener.accept().unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut request = Vec::new();
                let mut buffer = [0; 4096];
                loop {
                    let n = socket.read(&mut buffer).unwrap(); assert_ne!(n, 0); request.extend_from_slice(&buffer[..n]);
                    if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let header = std::str::from_utf8(&request[..end]).unwrap();
                        let length: usize = header.lines().find_map(|line| line.strip_prefix("Content-Length: ")).unwrap().parse().unwrap();
                        if request.len() == end + 4 + length { break; }
                    }
                }
                let request = String::from_utf8(request).unwrap();
                assert!(request.contains("\"temperature\":0"));
                assert!(request.contains("\"max_tokens\":1"));
                if reused > 0 { assert!(request.contains("\"session\":\"test-key\",\"fork\":true")); }
                socket.write_all(wire(&[42], reused, "").as_bytes()).unwrap();
            }
        });
        let prompt = [1, 42, 1024, 17].repeat(4);
        let reference = run(address, &prompt, 1, ",\"cache\":true", Some("plain")).unwrap();
        let repeated = run(address, &prompt, 1, ",\"cache\":true,\"session\":\"test-key\",\"fork\":true", Some("plain")).unwrap();
        assert_eq!(compare_cached("exact replay", &reference, &reference, prompt.len()).unwrap(), "SKIP");
        assert_eq!(compare_cached("exact replay", &reference, &repeated, prompt.len()).unwrap(), "PASS");
        server.join().unwrap();
    }
}

#[cfg(test)]
#[path = "serve-http-check/history_tests.rs"]
mod history_tests;
