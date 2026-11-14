//! Compare streamed IDs and text over HTTP, with cold prefill for every mode.
use std::{io::{BufRead, BufReader, Write}, net::{SocketAddr, TcpStream}, path::Path, time::{Duration, Instant}};
use engine_model::json::{self, Value};

struct Reply { ids: Vec<u32>, text: String, count: usize, ttft: f64, prefill: f64, total: f64 }
fn request(address: SocketAddr, prompt: &[u32], mode: &str, sampled: bool, max_new: usize) -> Result<Reply, String> {
    let start = Instant::now();
    let mut socket = TcpStream::connect(address).map_err(|e| e.to_string())?;
    socket.set_read_timeout(Some(Duration::from_secs(300))).map_err(|e| e.to_string())?;
    let sampling = if sampled { "\"temperature\":0.7,\"top_k\":20,\"top_p\":0.8" } else { "\"temperature\":0" };
    let body = format!("{{\"stream\":true,\"cache\":false,\"seed\":42,{sampling},\"prompt_token_ids\":{prompt:?},\"max_tokens\":{max_new},\"draft\":\"{mode}\"}}");
    write!(socket, "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).map_err(|e| e.to_string())?;
    let mut reply = Reply { ids: Vec::new(), text: String::new(), count: 0, ttft: 0.0, prefill: 0.0, total: 0.0 };
    let mut done = false;
    for line in BufReader::new(socket).lines() {
        let line = line.map_err(|e| e.to_string())?;
        let elapsed = start.elapsed().as_secs_f64()*1000.0;
        if line.starts_with("HTTP/") && !line.starts_with("HTTP/1.1 200") { return Err(line); }
        if line == ": prefill" { reply.prefill = elapsed; }
        if line == "data: [DONE]" { done = true; break; }
        let Some(data) = line.strip_prefix("data: ") else { continue; };
        let value = json::parse(data.as_bytes(), json::Limits::default()).map_err(|e| e.to_string())?;
        let object = value.object().map_err(|e| e.to_string())?;
        if object.contains_key("error") { return Err(data.into()); }
        let choices = json::field(object, "choices").and_then(Value::array).map_err(|e| e.to_string())?;
        let choice = choices.first().ok_or("missing choice")?.object().map_err(|e| e.to_string())?;
        if let Some(id) = choice.get("token_id") {
            if reply.ids.is_empty() { reply.ttft = elapsed; }
            reply.ids.push(id.u64().map_err(|e| e.to_string())? as u32);
        }
        if let Some(text) = choice.get("text") { reply.text.push_str(text.string().map_err(|e| e.to_string())?); }
        if let Some(usage) = object.get("usage") {
            reply.count = json::field(usage.object().map_err(|e| e.to_string())?, "completion_tokens")
                .and_then(Value::u64).map_err(|e| e.to_string())? as usize;
        }
    }
    reply.total = start.elapsed().as_secs_f64()*1000.0;
    if !done || reply.ids.is_empty() || reply.prefill == 0.0 { return Err("incomplete stream or missing prefill boundary".into()); }
    Ok(reply)
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) { return Err("usage: serve-draft-check ADDRESS PROMPT_DIR [MAX_NEW=128]".into()); }
    let address = args[0].parse().map_err(|_| "invalid address")?;
    let max_new = args.get(2).map_or(Ok(128), |s| s.parse::<usize>()).map_err(|e| e.to_string())?;
    let cases = std::env::var("VADE_DRAFT_CASES").unwrap_or_else(|_| "code,json,quotes,prose".into());
    let modes = std::env::var("VADE_DRAFT_MODES").unwrap_or_else(|_| "plain,strict,fast".into());
    for name in cases.split(',') {
        let prompt = std::fs::read_to_string(Path::new(&args[1]).join(format!("{name}.ids"))).map_err(|e| e.to_string())?
            .split_whitespace().map(|s| s.parse::<u32>().map_err(|e| e.to_string())).collect::<Result<Vec<_>, _>>()?;
        for sampled in [false, true] {
            if sampled && std::env::var("VADE_DRAFT_SAMPLING").as_deref() == Ok("greedy") { continue; }
            let mut plain: Option<Reply> = None;
            for mode in modes.split(',') {
                let r = request(address, &prompt, mode, sampled, max_new)?;
                let same = plain.as_ref().is_none_or(|p| p.ids == r.ids && p.text == r.text && p.count == r.count);
                println!("HTTP case={name} sampled={sampled} mode={mode} prompt={} generated={} tokens_text_equal={same} tok_s={:.3} ttft_ms={:.3} prefill_ms={:.3} total_ms={:.3}",
                    prompt.len(), r.count, r.count as f64*1000.0/(r.total-r.prefill), r.ttft, r.prefill, r.total);
                if !same { return Err(format!("stream differs from plain: {name}, sampled={sampled}, mode={mode}")); }
                if mode == "plain" { plain = Some(r); }
            }
        }
    }
    Ok(())
}
fn main() { if let Err(e) = run() { eprintln!("FAIL {e}"); std::process::exit(1); } }
