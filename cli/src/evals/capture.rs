//! Capture the same frozen requests and validate a complete stream before reuse.
use super::{
    CHECKPOINT, LongArgs, MODEL,
    json::{self, Value},
    longctx, sampling, sha,
};
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

struct Endpoint {
    secure: bool,
    host: String,
    port: u16,
    authority: String,
    target: String,
}
fn endpoint(url: &str, path: &str) -> Result<Endpoint, String> {
    let fail = "server URL must be an HTTP(S) base without credentials/query/fragment";
    // urlsplit removes leading C0/space and embedded tab/newline characters.
    let normalized = url
        .trim_start_matches(|c: char| c <= ' ')
        .replace(['\r', '\n', '\t'], "");
    let (scheme, rest) = normalized.split_once("://").ok_or(fail)?;
    let scheme = scheme.to_ascii_lowercase();
    if !matches!(scheme.as_str(), "http" | "https") {
        return Err(fail.into());
    }
    let rest = match rest.split_once('#') {
        Some((url, "")) => url,
        Some(_) => return Err(fail.into()),
        None => rest,
    };
    let rest = match rest.split_once('?') {
        Some((url, "")) => url,
        Some(_) => return Err(fail.into()),
        None => rest,
    };
    let (authority, base) = rest.split_once('/').map_or((rest, ""), |(a, b)| (a, b));
    if authority.is_empty() || authority.contains('@') {
        return Err(fail.into());
    }
    let (host, port) = if authority.starts_with('[') {
        let end = authority.find(']').ok_or(fail)?;
        let host = &authority[1..end];
        let suffix = &authority[end + 1..];
        (
            host,
            if suffix.is_empty() {
                None
            } else {
                Some(suffix.strip_prefix(':').ok_or(fail)?)
            },
        )
    } else {
        authority
            .rsplit_once(':')
            .map_or((authority, None), |(h, p)| (h, Some(p)))
    };
    if host.is_empty() {
        return Err(fail.into());
    }
    let port = port
        .map(str::parse::<u16>)
        .transpose()
        .map_err(|_| fail)?
        .unwrap_or(if scheme == "https" { 443 } else { 80 });
    let mut base = format!("/{}", base.trim_end_matches('/'));
    if base == "/" {
        base.clear();
    }
    if base.ends_with("/v1") {
        base.truncate(base.len() - 3);
    }
    let host = host.to_ascii_lowercase();
    let authority = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.clone()
    };
    let authority = if port == if scheme == "https" { 443 } else { 80 } {
        authority
    } else {
        format!("{authority}:{port}")
    };
    let target = format!("{base}{path}");
    if authority
        .chars()
        .chain(target.chars())
        .any(|c| c <= ' ' || c == '\u{7f}')
    {
        return Err("InvalidURL".into());
    }
    Ok(Endpoint {
        secure: scheme == "https",
        host,
        port,
        authority,
        target,
    })
}
struct Chunked {
    stream: BufReader<Box<dyn Read>>,
    remaining: usize,
    end: bool,
}
impl Read for Chunked {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.end {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut line = String::new();
            self.stream.read_line(&mut line)?;
            let size = line.trim().split(';').next().unwrap_or("");
            self.remaining = usize::from_str_radix(size, 16)
                .map_err(|_| io::Error::new(io::ErrorKind::UnexpectedEof, "invalid chunk"))?;
            if self.remaining == 0 {
                self.end = true;
                return Ok(0);
            }
        }
        let n = buf.len().min(self.remaining);
        let got = self.stream.read(&mut buf[..n])?;
        if got == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete chunk",
            ));
        }
        self.remaining -= got;
        if self.remaining == 0 {
            let mut end = [0; 2];
            self.stream.read_exact(&mut end)?;
            if end != *b"\r\n" {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "invalid chunk",
                ));
            }
        }
        Ok(got)
    }
}
struct ChildGuard(Option<Child>);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}
struct Response {
    status: u16,
    content_type: String,
    body: BufReader<Box<dyn Read>>,
    child: ChildGuard,
    expected_length: Option<u64>,
}
impl Response {
    fn finish(&mut self) -> Result<(), String> {
        if let Some(child) = self.child.0.as_mut() {
            if !child.wait().map_err(json::io_error)?.success() {
                return Err("OSError".into());
            }
        }
        Ok(())
    }
}
fn curl_quote(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}
fn request(
    url: &str,
    path: &str,
    body: Option<&Value>,
    args: &LongArgs,
) -> Result<Response, String> {
    let endpoint = endpoint(url, path)?;
    if args.timeout.is_nan() {
        return Err("Invalid value NaN (not a number)".into());
    }
    if !args.timeout.is_finite() || args.timeout.abs() >= 9223372036.854776 {
        return Err("OverflowError".into());
    }
    if args.timeout < 0.0 {
        return Err("Timeout value out of range".into());
    }
    if args.timeout == 0.0 {
        return Err("BlockingIOError".into());
    }
    if let Some(body) = body {
        json::validate(body)?;
    }
    // Legacy frozen bodies can contain a sampling seed. Strip it only for
    // explicit greedy requests, at the common HTTP/HTTPS dispatch boundary
    // used by both retrieval and teacher scoring; keep the frozen file intact.
    let payload = body.map(|body| {
        let mut body = body.clone();
        if json::numeric(&body["temperature"]) == Some(0.) {
            if let Value::Object(fields) = &mut body {
                fields.retain(|(key, _)| key != "seed");
            }
        }
        json::packed(&body)
    });
    let authorization = if let Some(name) = &args.api_key_env {
        let key = std::env::var(name).map_err(|_| "KeyError")?;
        if key.contains(['\r', '\n']) {
            return Err("Invalid header value".into());
        }
        Some(format!("Bearer {key}"))
    } else {
        None
    };
    let mut child = ChildGuard(None);
    let stream: Box<dyn Read> = if !endpoint.secure {
        let addresses = (endpoint.host.as_str(), endpoint.port)
            .to_socket_addrs()
            .map_err(json::io_error)?;
        let timeout = Duration::try_from_secs_f64(args.timeout).map_err(|_| "OverflowError")?;
        let mut connected = None;
        let mut connection_error = "OSError".to_string();
        for addr in addresses {
            match TcpStream::connect_timeout(&addr, timeout) {
                Ok(stream) => {
                    connected = Some(stream);
                    break;
                }
                Err(error) => connection_error = json::io_error(error),
            }
        }
        let mut stream = connected.ok_or(connection_error)?;
        stream
            .set_read_timeout(Some(timeout))
            .map_err(json::io_error)?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(json::io_error)?;
        let method = if payload.is_some() { "POST" } else { "GET" };
        let mut headers = format!(
            "{method} {} HTTP/1.1\r\nHost: {}\r\nAccept-Encoding: identity\r\nContent-Type: application/json\r\nConnection: close\r\n",
            endpoint.target, endpoint.authority
        );
        if let Some(key) = authorization {
            headers.push_str(&format!("Authorization: {key}\r\n"));
        }
        if let Some(payload) = &payload {
            headers.push_str(&format!("Content-Length: {}\r\n", payload.len()));
        }
        headers.push_str("\r\n");
        stream
            .write_all(headers.as_bytes())
            .map_err(json::io_error)?;
        if let Some(payload) = payload {
            stream
                .write_all(payload.as_bytes())
                .map_err(json::io_error)?;
        }
        Box::new(stream)
    } else {
        // The standard library has no TLS. Use the CLI's existing curl runtime
        // for HTTPS, sending credentials only through its private stdin config.
        child.0 = Some(
            Command::new("curl")
                .args([
                    "--disable",
                    "--silent",
                    "--show-error",
                    "--include",
                    "--no-buffer",
                    "--http1.1",
                    "--noproxy",
                    "*",
                    "--config",
                    "-",
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(json::io_error)?,
        );
        let proc = child.0.as_mut().ok_or("OSError")?;
        let mut config = format!(
            "url = \"{}\"\nrequest = \"{}\"\nheader = \"Content-Type: application/json\"\nheader = \"Connection: close\"\nconnect-timeout = {}\nspeed-limit = 1\nspeed-time = {}\n",
            curl_quote(&format!(
                "https://{}{}",
                endpoint.authority, endpoint.target
            )),
            if payload.is_some() { "POST" } else { "GET" },
            args.timeout,
            args.timeout.ceil()
        );
        if let Some(key) = authorization {
            config.push_str(&format!(
                "header = \"Authorization: {}\"\n",
                curl_quote(&key)
            ));
        }
        if let Some(payload) = payload {
            config.push_str(&format!("data-binary = \"{}\"\n", curl_quote(&payload)));
        }
        proc.stdin
            .take()
            .ok_or("OSError")?
            .write_all(config.as_bytes())
            .map_err(json::io_error)?;
        let stdout = proc.stdout.take().ok_or("OSError")?;
        Box::new(stdout)
    };
    let mut stream = BufReader::new(stream);
    let (status, content_type, length, chunked) = loop {
        let mut line = String::new();
        stream.read_line(&mut line).map_err(json::io_error)?;
        let status = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or("HTTPException")?;
        let mut content_type = String::new();
        let mut length = None;
        let mut chunked = false;
        loop {
            line.clear();
            if stream.read_line(&mut line).map_err(json::io_error)? == 0 {
                return Err("HTTPException".into());
            }
            if line == "\r\n" || line == "\n" {
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                match key.to_ascii_lowercase().as_str() {
                    "content-type" => content_type = value.trim().into(),
                    "content-length" => length = value.trim().parse::<u64>().ok(),
                    "transfer-encoding" => chunked = value.trim().eq_ignore_ascii_case("chunked"),
                    _ => {}
                }
            }
        }
        if status != 100 {
            break (status, content_type, length, chunked);
        }
    };
    // curl removes transfer framing itself; raw TCP keeps the HTTP decoder.
    let body: Box<dyn Read> = if chunked && child.0.is_none() {
        Box::new(Chunked {
            stream,
            remaining: 0,
            end: false,
        })
    } else if let Some(length) = length {
        Box::new(stream.take(length))
    } else {
        Box::new(stream)
    };
    Ok(Response {
        status,
        content_type,
        body: BufReader::new(body),
        child,
        expected_length: if chunked { None } else { length },
    })
}
fn http_json(
    url: &str,
    path: &str,
    body: Option<&Value>,
    args: &LongArgs,
) -> Result<Value, String> {
    let mut response = request(url, path, body, args)?;
    if response.status != 200 {
        return Err(format!("server request failed (HTTP {})", response.status));
    }
    let mut text = String::new();
    response
        .body
        .read_to_string(&mut text)
        .map_err(json::io_error)?;
    if response
        .expected_length
        .is_some_and(|length| length != text.len() as u64)
    {
        return Err("IncompleteRead".into());
    }
    response.finish()?;
    json::parse(&text)
}
fn capture_score(
    url: &str,
    destination: &Path,
    case: &Value,
    request_body: &Value,
    reference: Option<&Path>,
    args: &LongArgs,
) -> Result<(), String> {
    let mut body = request_body.clone();
    body.insert("score_top_k", args.top_k.into());
    if let Some(reference) = reference {
        let mut rows = Vec::new();
        for row in longctx::score_rows(reference, case, args.top_k)? {
            let row = row?;
            rows.push(
                row.topk
                    .iter()
                    .map(|(id, _)| id.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        body.insert("score_ids", rows.join(";").into());
    }
    json::validate(&body)?;
    if json::packed(&body).len() > 16 * 1024 * 1024 {
        return Err("scoring body exceeds 16 MiB; reduce --top-k".into());
    }
    let id = case["id"].as_str().ok_or("KeyError")?;
    let path = destination.join(format!("{id}.score.jsonl"));
    let temporary = path.with_extension("partial");
    let captured = (|| {
        let mut response = request(url, "/v1/score", Some(&body), args)?;
        if response.status != 200 {
            return Err(format!("scoring request failed (HTTP {})", response.status));
        }
        if response.content_type.split(';').next() != Some("application/x-ndjson") {
            return Err("server lacks teacher-forced NDJSON scoring".into());
        }
        let mut out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(json::io_error)?;
        let mut line = String::new();
        response.body.read_line(&mut line).map_err(json::io_error)?;
        let mut meta = json::parse(&line)?;
        meta.insert("prompt_sha256", case["prompt_sha256"].clone());
        meta.insert("checkpoint_sha256", CHECKPOINT.into());
        json::validate(&meta)?;
        writeln!(out, "{}", json::packed(&meta)).map_err(json::io_error)?;
        // Text lines remain byte-identical to the server's UTF-8 NDJSON.
        io::copy(&mut response.body, &mut out).map_err(json::io_error)?;
        response.finish()?;
        drop(out);
        fs::rename(&temporary, &path).map_err(json::io_error)?;
        for row in longctx::score_rows(destination, case, args.top_k)? {
            row?;
        }
        Ok(())
    })();
    if captured.is_err() && path.exists() {
        let _ = fs::rename(&path, &temporary);
    }
    captured
}
fn numeric_eq(a: &Value, b: &Value) -> bool {
    json::equal(a, b)
}
fn greedy(value: &Value) -> bool {
    sampling()
        .iter()
        .all(|(k, v)| value.get(k).is_some_and(|found| numeric_eq(found, v)))
}
/// `vade test longctx run`: send every frozen request to each server and save
/// the validated captures.
pub fn run(args: &mut LongArgs, manifest: &Value) -> Result<(), String> {
    let mut urls = vec![
        ("A", args.a_url.clone(), args.a_build.clone()),
        ("B", args.b_url.clone(), args.b_build.clone()),
    ];
    if let Some(url) = &args.c_url {
        urls.push(("C", url.clone(), args.c_build.clone()));
    }
    let mut identity: Option<Value> = None;
    for (arm, url, build) in urls {
        let capabilities = http_json(&url, "/v1/capabilities", None, args)?;
        if capabilities["model"].as_str() != Some(MODEL)
            || json::numeric(&capabilities["context_limit"]).unwrap_or(0.)
                < 131072. + json::numeric(&manifest["max_tokens"]).ok_or("KeyError")?
        {
            return Err("server model/context cannot run the frozen suite".into());
        }
        let current = &capabilities["identity"];
        if !matches!(current, Value::Object(_))
            || !json::truthy(&current["weights"])
            || !json::truthy(&current["tokenizer_fingerprint"])
        {
            return Err("server checkpoint/tokenizer identity is unavailable".into());
        }
        let stable = json::object(vec![
            ("weights", current["weights"].clone()),
            (
                "tokenizer_fingerprint",
                current["tokenizer_fingerprint"].clone(),
            ),
        ]);
        if identity
            .as_ref()
            .is_some_and(|old| !json::equal(old, &stable))
        {
            return Err("server checkpoint/tokenizer identities differ".into());
        }
        identity = Some(stable);
        let destination = args.output.join(arm);
        if destination.exists() {
            return Err("FileExistsError".into());
        }
        fs::create_dir_all(&destination).map_err(json::io_error)?;
        json::write_json(
            &destination.join("provenance.json"),
            &json::object(vec![
                ("checkpoint_sha256", manifest["checkpoint_sha256"].clone()),
                ("identity", current.clone()),
                ("build", build.into()),
                (
                    "verification",
                    "operator build/checkpoint pin; capabilities identity compared across arms"
                        .into(),
                ),
            ]),
        )?;
        for case in manifest["cases"].as_array().ok_or("KeyError")? {
            let body_path = args
                .manifest
                .parent()
                .unwrap_or(Path::new("."))
                .join(case["body_file"].as_str().ok_or("KeyError")?);
            let body = json::read(&body_path)?;
            if sha(body["prompt"].as_str().ok_or("KeyError")?.as_bytes())
                != case["prompt_sha256"].as_str().ok_or("KeyError")?
            {
                return Err("frozen prompt body changed".into());
            }
            if !greedy(&body)
                || body["draft"].as_str() != Some("plain")
                || !matches!(body["cache"], Value::Bool(false))
            {
                return Err(
                    "frozen request no longer has explicit greedy uncached settings".into(),
                );
            }
            if arm == "C" && case["kind"].as_str() != Some("teacher") {
                continue;
            }
            if case["kind"].as_str() == Some("teacher") {
                let reference = args.output.join("A");
                capture_score(
                    &url,
                    &destination,
                    case,
                    &body,
                    if arm == "A" { None } else { Some(&reference) },
                    args,
                )?;
            } else {
                let response = http_json(&url, "/v1/completions", Some(&body), args)?;
                let vade = &response["vade"];
                if !greedy(&vade["sampling"]) {
                    return Err("server did not echo the requested greedy sampling".into());
                }
                if vade["draft"].as_str() != Some("plain")
                    || json::numeric(&vade["mtp_depth"]) != Some(0.)
                    || !matches!(vade["thinking"], Value::Bool(false))
                {
                    return Err("server did not honor plain MTP-0 thinking-off generation".into());
                }
                let text = response["choices"][0]["text"].as_str().ok_or("KeyError")?;
                if !json::equal(&response["usage"]["prompt_tokens"], &case["length"]) {
                    return Err("retrieval tokenization differs from manifest".into());
                }
                let saved = json::object(vec![
                    ("case_id", case["id"].clone()),
                    ("prompt_sha256", case["prompt_sha256"].clone()),
                    ("text", text.into()),
                    ("usage", response["usage"].clone()),
                    (
                        "vade",
                        json::object(vec![
                            ("sampling", vade["sampling"].clone()),
                            ("draft", vade["draft"].clone()),
                            ("mtp_depth", vade["mtp_depth"].clone()),
                            ("thinking", vade["thinking"].clone()),
                        ]),
                    ),
                ]);
                json::write_json(
                    &destination.join(format!(
                        "{}.reply.json",
                        case["id"].as_str().ok_or("KeyError")?
                    )),
                    &saved,
                )?;
            }
            crate::output::stderr_line(format_args!("captured {arm} {}", case["id"].as_str().ok_or("KeyError")?));
        }
    }
    args.a_dumps = args.output.join("A");
    args.b_dumps = args.output.join("B");
    args.c_dumps = args.c_url.as_ref().map(|_| args.output.join("C"));
    Ok(())
}

#[cfg(test)]
#[path = "capture_tests.rs"]
mod tests;
