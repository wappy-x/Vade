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
) -> Result<TcpStream, String> {
    let mut socket = TcpStream::connect(address).map_err(|e| e.to_string())?;
    socket
        .set_read_timeout(Some(Duration::from_secs(180)))
        .map_err(|e| e.to_string())?;
    let body = format!(
        "{{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":{prompt:?},\"max_tokens\":{output},\"draft\":{}{extra}}}",
        spark_serve::protocol::quote(&std::env::var("VADE_CHECK_DRAFT").unwrap_or_else(|_| "plain".into()))
    );
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
fn run(address: SocketAddr, prompt: &[u32], output: usize, extra: &str) -> Result<Reply, String> {
    let start = Instant::now();
    let mut socket = connect(address, prompt, output, extra)?;
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
    let reused = header("X-Prefix-Reused-Tokens:")
        .unwrap_or("0")
        .parse()
        .map_err(|_| "invalid reused count")?;
    let tokens = text
        .split("\"token_id\":")
        .skip(1)
        .map(|part| {
            part.split(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap_or("")
                .parse::<u32>()
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Reply {
        tokens,
        key,
        reused,
        ttft_ms: ttft_ms.ok_or("no generated token")?,
        total_ms: start.elapsed().as_secs_f64() * 1000.,
    })
}
fn check() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
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
    if std::env::var("VADE_CHECK_AUTOMATIC").as_deref() == Ok("1") {
        let parent = run(address, &prompt, long_tokens, "")?;
        let mut suffix = prompt.clone(); suffix.extend_from_slice(&parent.tokens); suffix.extend_from_slice(&[220, 17, 42]);
        for shared in [prompt.len() + parent.tokens.len().saturating_sub(1), prompt.len() - 1] {
            if shared < prompt.len() { suffix[shared] = (suffix[shared] + 1) % 1000; }
            let cold = run(address, &suffix, long_tokens, ",\"cache\":false")?;
            let warm = run(address, &suffix, long_tokens, "")?;
            if cold.tokens != warm.tokens || warm.reused < shared.min(prompt.len())
                || warm.reused > shared.min(prompt.len()) + usize::from(shared >= prompt.len()) || cold.reused != 0 {
                return Err(format!("automatic suffix mismatch: shared={shared} reused={} cold={:?} warm={:?}", warm.reused, cold.tokens, warm.tokens));
            }
            println!("PASS automatic_prefix shared={shared} reused={} output={} cold_ttft_ms={:.3} warm_ttft_ms={:.3}", warm.reused, warm.tokens.len(), cold.ttft_ms, warm.ttft_ms);
        }
        return Ok(());
    }
    let replay = run(address, &prompt, 1, ",\"cache\":true,\"priority\":1")?;
    let replay_key = replay.key.as_ref().ok_or("missing replay capability")?;
    let repeated = run(address, &prompt, 1, &format!(",\"cache\":true,\"priority\":1,\"session\":\"{replay_key}\",\"fork\":true"))?;
    let replay_reused = prompt.len();
    if replay.tokens != repeated.tokens || repeated.reused != replay_reused { return Err("exact cached replay mismatch".into()); }
    println!("PASS exact_replay prompt_tokens={} cold_ttft_ms={:.3} cached_ttft_ms={:.3} reused={}",
        prompt.len(), replay.ttft_ms, repeated.ttft_ms, repeated.reused);
    let cold = run(address, &prompt, 4, ",\"cache\":true,\"priority\":1")?;
    let key = cold.key.as_ref().ok_or("missing session capability")?;
    let cached = format!(",\"cache\":true,\"priority\":1,\"session\":\"{key}\",\"fork\":true");
    let warm = run(address, &prompt, 4, &cached)?;
    // Prompt-end logits survive the completion snapshot.
    if cold.tokens != warm.tokens || warm.reused != prompt.len() || warm.key == cold.key {
        return Err(format!(
            "exact prompt fork mismatch: cold={:?} warm={:?} reused={} independent_key={}",
            cold.tokens,
            warm.tokens,
            warm.reused,
            warm.key != cold.key
        ));
    }
    println!(
        "PASS cold_prompt_tokens={} cold_ttft_ms={:.3} warm_ttft_ms={:.3} reused={} output={:?}",
        prompt.len(), cold.ttft_ms, warm.ttft_ms, warm.reused, warm.tokens
    );
    let mut divergent = prompt.clone();
    divergent[8] = 220;
    let branch = run(address, &divergent, 4, &cached)?;
    let fresh = run(address, &divergent, 4, ",\"priority\":1,\"cache\":false")?;
    if branch.tokens != fresh.tokens || branch.reused != 8 {
        return Err("divergent prefix fork mismatch".into());
    }
    let parent = run(address, &prompt, 4, &cached)?;
    if parent.tokens != cold.tokens || parent.reused != prompt.len() {
        return Err("parent snapshot mutated or unexpectedly evicted".into());
    }
    println!(
        "PASS shared_prefix=8 branch_ttft_ms={:.3} fresh_ttft_ms={:.3} parent_reused={} output={:?}",
        branch.ttft_ms, fresh.ttft_ms, parent.reused, branch.tokens
    );
    let isolated = run(address, &prompt[..4], 4, ",\"priority\":255")?;
    let background_prompt: Vec<_> = (0..512).map(|i| [1, 42, 1024, 17][i % 4]).collect();
    let mut background = connect(address, &background_prompt, 128, "")?;
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
    let mut health = TcpStream::connect(address).map_err(|e| e.to_string())?;
    health
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    health
        .write_all(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n").as_bytes())
        .map_err(|e| e.to_string())?;
    let mut response = String::new();
    health
        .read_to_string(&mut response)
        .map_err(|e| e.to_string())?;
    if !response.contains("\"active\":0") || !response.contains("\"queued\":0") {
        return Err("disconnect left active or queued work".into());
    }
    println!("PASS disconnect_cleanup active=0 queued=0");
    Ok(())
}
fn main() {
    if let Err(error) = check() {
        eprintln!("FAIL {error}");
        std::process::exit(1);
    }
}
