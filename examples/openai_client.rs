//! Run with cargo run --manifest-path tools/Cargo.toml --example openai-client -- [stream|tools|cache ...].
//! Plain HTTP only. Use the shell example for HTTPS. The cache demo needs the engine prefix cache.
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../tools/src/json.rs"] mod json;
use std::{io::{self,BufRead,BufReader,Read,Write},net::TcpStream,time::Duration};
use json::{Value,parse,quote};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
struct Body { reader: BufReader<TcpStream>, chunked: bool, left: usize, ended: bool }
impl Read for Body {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.ended { return Ok(0); }
        if self.chunked && self.left == 0 {
            let mut line = String::new(); self.reader.read_line(&mut line)?;
            self.left = usize::from_str_radix(line.trim().split(';').next().unwrap(),16).map_err(io::Error::other)?;
            if self.left == 0 { self.ended = true; return Ok(0); }
        }
        let size = buf.len().min(self.left); let n = self.reader.read(&mut buf[..size])?;
        if n == 0 && self.left != usize::MAX && self.left != 0 { return Err(io::Error::other("short HTTP body")); }
        if self.left != usize::MAX { self.left -= n; }
        if self.chunked && self.left == 0 {
            let mut crlf = [0;2]; self.reader.read_exact(&mut crlf)?;
            if crlf != *b"\r\n" { return Err(io::Error::other("invalid HTTP chunk")); }
        }
        Ok(n)
    }
}
fn call(base: &str, path: &str, body: Option<&str>) -> Result<Body> {
    let host = base.strip_prefix("http://").ok_or("use an http:// endpoint or the shell example for HTTPS")?;
    let (authority,prefix) = host.split_once('/').unwrap_or((host,""));
    if authority.contains(['\r','\n']) || prefix.contains(['\r','\n']) { return Err("invalid HTTP endpoint".into()); }
    let address = if authority.contains(':') { authority.to_owned() } else { format!("{authority}:80") };
    let mut tcp = TcpStream::connect(address)?; tcp.set_read_timeout(Some(Duration::from_secs(600)))?; tcp.set_write_timeout(Some(Duration::from_secs(600)))?;
    let body_text = body.unwrap_or(""); let path = format!("/{}{path}",prefix.trim_end_matches('/')).replacen("//","/",1);
    let mut auth = String::new();
    if let Some(file) = std::env::var_os("VADE_API_KEY_FILE") {
        let key = std::fs::read_to_string(file)?; let key = key.trim();
        if key.contains(['\r','\n']) { return Err("invalid bearer key".into()); } auth = format!("Authorization: Bearer {key}\r\n");
    }
    write!(tcp,"{} {path} HTTP/1.1\r\nHost: {authority}\r\nContent-Type: application/json\r\n{auth}Connection: close\r\nContent-Length: {}\r\n\r\n{body_text}",if body.is_some() {"POST"} else {"GET"},body_text.len())?;
    let mut reader = BufReader::new(tcp); let mut status = String::new(); reader.read_line(&mut status)?;
    let code: u16 = status.split_whitespace().nth(1).ok_or("missing HTTP status")?.parse()?;
    if !(200..300).contains(&code) { return Err(format!("HTTP request failed: {}",status.trim()).into()); }
    let mut left = usize::MAX; let mut chunked = false;
    loop {
        let mut line = String::new(); if reader.read_line(&mut line)? == 0 { return Err("short HTTP headers".into()); } if line == "\r\n" { break; }
        if let Some((key,value)) = line.split_once(':') {
            if key.eq_ignore_ascii_case("content-length") { left = value.trim().parse()?; }
            if key.eq_ignore_ascii_case("transfer-encoding") { chunked = value.trim().eq_ignore_ascii_case("chunked"); }
        }
    }
    Ok(Body {reader,chunked,left:if chunked {0} else {left},ended:false})
}
fn chat(base: &str, body: &str) -> Result<Value> {
    let mut text = String::new(); call(base,"/v1/chat/completions",Some(body))?.read_to_string(&mut text)?; Ok(parse(&text)?)
}
fn stream(base: &str, model: &str) -> Result<()> {
    let body = format!("{{\"model\":{},\"stream\":true,\"max_tokens\":64,\"enable_thinking\":false,\"messages\":[{{\"role\":\"user\",\"content\":\"Write a haiku about a small computer that runs a large model.\"}}]}}",quote(model));
    let mut usage = Value::Null;
    for line in BufReader::new(call(base,"/v1/chat/completions",Some(&body))?).lines() {
        let line = line?; let Some(data) = line.strip_prefix("data: ") else { continue; }; if data == "[DONE]" { continue; }
        let chunk = parse(data)?;
        for choice in chunk.get("choices").array()? {
            if let Ok(content) = choice.get("delta").get("content").text() { print!("{content}"); io::stdout().flush()?; }
        }
        if chunk.get("usage") != &Value::Null { usage = chunk.get("usage").clone(); }
    }
    println!("\n[usage] prompt {}, completion {}",usage.get("prompt_tokens").text()?,usage.get("completion_tokens").text()?); Ok(())
}
fn tools(base: &str, model: &str) -> Result<()> {
    let schema = r#"{"type":"function","function":{"name":"get_weather","description":"Current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"#;
    let body = format!("\"model\":{},\"tools\":[{schema}],\"max_tokens\":256,\"enable_thinking\":false",quote(model));
    let messages = r#"{"role":"user","content":"What is the weather in Paris right now?"}"#;
    let reply = chat(base,&format!("{{{body},\"messages\":[{messages}]}}"))?; let choice = reply.get("choices").get("0"); let message = choice.get("message");
    let call = message.get("tool_calls").get("0"); let args = parse(call.get("function").get("arguments").text()?)?;
    println!("[finish_reason] {}",choice.get("finish_reason").text()?);
    println!("[tool call] {}({})",call.get("function").get("name").text()?,args.encode());
    let result = format!("{{\"city\":{},\"temperature_c\":18,\"sky\":\"clear\"}}",args.get("city").encode());
    let reply = chat(base,&format!("{{{body},\"messages\":[{messages},{},{{\"role\":\"tool\",\"tool_call_id\":{},\"content\":{}}}]}}",message.encode(),call.get("id").encode(),quote(&result)))?;
    println!("[answer] {}",reply.get("choices").get("0").get("message").get("content").text()?.trim()); Ok(())
}
fn cache(base: &str, model: &str) -> Result<()> {
    let notes = (1..=120).map(|i|format!("Rule {i}: keep function {i} under {} lines.",10+i)).collect::<Vec<_>>().join(" ");
    let mut messages = format!("{{\"role\":\"system\",\"content\":{}}},{{\"role\":\"user\",\"content\":\"How many lines may function 7 have? Answer with the number.\"}}",quote(&format!("You review code against these house rules. {notes}")));
    for turn in 1..=2 {
        let reply = chat(base,&format!("{{\"model\":{},\"max_tokens\":16,\"enable_thinking\":false,\"prompt_cache_key\":\"examples-review\",\"messages\":[{messages}]}}",quote(model)))?;
        let usage = reply.get("usage"); let answer = reply.get("choices").get("0").get("message").get("content").text()?.trim();
        println!("[turn {turn}] prompt {} tokens, cached {}: {answer}",usage.get("prompt_tokens").text()?,usage.get("prompt_tokens_details").get("cached_tokens").text()?);
        messages.push_str(&format!(",{{\"role\":\"assistant\",\"content\":{}}},{{\"role\":\"user\",\"content\":\"And function 9? Answer with the number.\"}}",quote(answer)));
    } Ok(())
}
fn run() -> Result<()> {
    let base = std::env::var("VADE_API_URL").unwrap_or("http://127.0.0.1:18080".into()); let base = base.trim_end_matches('/'); let base = base.strip_suffix("/v1").unwrap_or(base);
    let mut text = String::new(); call(base,"/v1/models",None)?.read_to_string(&mut text)?; let value = parse(&text)?; let model = value.get("data").get("0").get("id").text()?;
    println!("[model] {model} at {base}"); let mut demos: Vec<_> = std::env::args().skip(1).collect(); if demos.is_empty() { demos = vec!["stream".into(),"tools".into(),"cache".into()]; }
    for demo in demos { println!("== {demo}"); match demo.as_str() { "stream" => stream(base,model)?, "tools" => tools(base,model)?, "cache" => cache(base,model)?, _ => return Err("use stream, tools, or cache".into()) } } Ok(())
}
fn main() { if let Err(e) = run() { eprintln!("API example: {e}"); std::process::exit(1); } }
#[cfg(test)]
#[path = "../tools/tests/common/api_checks.rs"] mod tests;
