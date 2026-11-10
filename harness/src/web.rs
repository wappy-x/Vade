//! Bounded public web reads. No workspace files, credentials or curl config.
use crate::{agent::ToolError, json::Value};
use std::io::{self, Read};
use std::net::{IpAddr, Ipv4Addr};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const DEFAULT_BYTES: usize = 262144;
const MAX_BYTES: usize = 2097152;
const MAX_URL: usize = 8192;
const HEADER_BYTES: usize = 65536;
const REDIRECTS: usize = 5;

pub fn offered() -> bool {
    enabled(std::env::var_os("VADE_WEB").as_deref())
}
fn enabled(value: Option<&std::ffi::OsStr>) -> bool { value.is_none_or(|v| v != "0") }

#[derive(Debug)]
struct Reply { body: Vec<u8>, headers: String, code: Option<i32>, truncated: bool }
struct Page { url: String, kind: String, body: Vec<u8>, truncated: bool }
#[derive(Debug, PartialEq, Eq)]
enum Failure { Cancelled, Timeout, Refused(&'static str), Fetch(&'static str) }
impl Failure {
    fn message(&self) -> &'static str {
        match self {
            Self::Cancelled => "request cancelled",
            Self::Timeout => "request timed out",
            Self::Refused(message) | Self::Fetch(message) => message,
        }
    }
}

fn private_v4(ip: Ipv4Addr) -> bool {
    ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified()
        || ip.is_broadcast() || ip.is_multicast() || ip.octets()[0] == 0
}
fn validate_url(url: &str) -> Result<(), Failure> {
    let refuse = |message| Err(Failure::Refused(message));
    if url.len() > MAX_URL || !url.is_ascii() || url.bytes().any(|b| b <= b' ' || b == 127 || b == b'\\') {
        return refuse("URL must be bounded ASCII without spaces or control characters");
    }
    let Some((scheme, rest)) = url.split_once("://") else { return refuse("URL requires http or https"); };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return refuse("URL requires http or https");
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains(['@', '%']) { return refuse("URL credentials and encoded hosts are refused"); }
    let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
        let Some((host, tail)) = bracketed.split_once(']') else { return refuse("invalid URL host"); };
        if !tail.is_empty() && !tail.starts_with(':') { return refuse("invalid URL port"); }
        (host, tail.strip_prefix(':'))
    } else {
        authority.split_once(':').map_or((authority, None), |(host, port)| (host, Some(port)))
    };
    if port.is_some_and(|p| p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) || p.parse::<u16>().ok().is_none_or(|p| p == 0)) {
        return refuse("invalid URL port");
    }
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") { return refuse("local hosts are refused"); }
    if let Ok(ip) = host.parse::<IpAddr>() {
        let private = match ip {
            IpAddr::V4(ip) => private_v4(ip),
            IpAddr::V6(ip) => ip.is_loopback() || ip.is_unspecified() || ip.is_unique_local()
                || ip.is_unicast_link_local() || ip.is_multicast() || ip.to_ipv4().is_some_and(private_v4),
        };
        if private { return refuse("local and private address ranges are refused"); }
    } else {
        // Curl accepts short, integer, octal and hexadecimal address spellings.
        // Require a DNS name with a nonnumeric final label instead.
        let labels: Vec<_> = host.split('.').collect();
        if labels.len() < 2 || labels.iter().any(|label| label.is_empty() || label.len() > 63
            || label.starts_with('-') || label.ends_with('-')
            || !label.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'))
            || labels.last().is_none_or(|label| label.bytes().all(|b| b.is_ascii_digit()) || label.starts_with("0x")) {
            return refuse("invalid or ambiguous URL host");
        }
    }
    Ok(())
}

fn curl_command(url: &str, cap: usize, remaining: Duration) -> Command {
    let mut command = Command::new("/usr/bin/curl");
    // Drive redirects one hop at a time so each destination is checked first.
    // Ignore Content-Length to retain a prefix of large responses. The pipe
    // cap also covers curl versions that only check the announced size.
    // HTTP/1.1 with Connection: close gives the ignored-length body an end.
    command.env_clear().args(["-q", "-sSL", "--globoff", "--proto", "=http,https", "--proto-redir", "=http,https",
        "--max-redirs", "0", "--user-agent", "vade-harness", "--proxy", "", "--dump-header", "/dev/stderr",
        "--write-out", "%{stderr}\nVADE_REDIRECT:%{redirect_url}\n", "--ignore-content-length", "--http1.1", "--header", "Connection: close"])
        .args(["--max-time", &format!("{:.3}", remaining.as_secs_f64()), "--max-filesize", &cap.to_string(), "--url", url])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    command
}

fn capture(mut stream: impl Read, cap: usize, overflow: &AtomicBool) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buf = [0; 8192];
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 { break; }
        let take = n.min(cap.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&buf[..take]);
        if take < n { overflow.store(true, Ordering::Relaxed); break; }
    }
    Ok(bytes)
}
fn run_command(mut command: Command, cap: usize, deadline: Instant, cancel: &AtomicBool) -> Result<Reply, Failure> {
    let mut child = command.spawn().map_err(|_| Failure::Fetch("could not start system curl"))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let body_full = AtomicBool::new(false);
    let headers_full = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let body = scope.spawn(|| capture(stdout, cap, &body_full));
        let headers = scope.spawn(|| capture(stderr, HEADER_BYTES, &headers_full));
        let status = loop {
            let failure = if cancel.load(Ordering::Relaxed) { Some(Failure::Cancelled) }
                else if Instant::now() >= deadline { Some(Failure::Timeout) }
                else if headers_full.load(Ordering::Relaxed) { Some(Failure::Fetch("response headers exceed the limit")) }
                else { None };
            if let Some(failure) = failure { let _ = child.kill(); let _ = child.wait(); break Err(failure); }
            if body_full.load(Ordering::Relaxed) { let _ = child.kill(); break child.wait().map_err(|_| Failure::Fetch("could not wait for curl")); }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(_) => { let _ = child.kill(); let _ = child.wait(); break Err(Failure::Fetch("could not wait for curl")); }
            }
        };
        let body = body.join().map_err(|_| Failure::Fetch("body reader stopped"))?.map_err(|_| Failure::Fetch("could not read body"))?;
        let headers = headers.join().map_err(|_| Failure::Fetch("header reader stopped"))?.map_err(|_| Failure::Fetch("could not read headers"))?;
        let status = status?;
        if headers_full.load(Ordering::Relaxed) { return Err(Failure::Fetch("response headers exceed the limit")); }
        Ok(Reply { body, headers: String::from_utf8_lossy(&headers).into_owned(), code: status.code(), truncated: body_full.load(Ordering::Relaxed) })
    })
}
fn run(url: &str, cap: usize, deadline: Instant, cancel: &AtomicBool) -> Result<Reply, Failure> {
    // Tests inject a runner into fetch_with. An accidental real fetch in a unit
    // test must fail before a process starts.
    if cfg!(test) { return Err(Failure::Fetch("network is disabled in tests")); }
    run_command(curl_command(url, cap, deadline.saturating_duration_since(Instant::now())), cap, deadline, cancel)
}
fn fetch_with(url: &str, cap: usize, timeout: usize, cancel: &AtomicBool,
    mut runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>) -> Result<Page, Failure> {
    let deadline = Instant::now() + Duration::from_secs(timeout as u64);
    let mut url = url.to_owned();
    for hop in 0..=REDIRECTS {
        validate_url(&url)?;
        if cancel.load(Ordering::Relaxed) { return Err(Failure::Cancelled); }
        if Instant::now() >= deadline { return Err(Failure::Timeout); }
        let reply = runner(&url, cap, deadline, cancel)?;
        if reply.code == Some(28) { return Err(Failure::Timeout); }
        let mut status = None;
        let mut kind = "application/octet-stream";
        let mut redirect = None;
        let mut in_headers = false;
        for line in reply.headers.lines() {
            if line.starts_with("HTTP/") {
                in_headers = true;
                status = line.split_whitespace().nth(1).and_then(|s| s.parse::<u16>().ok());
                kind = "application/octet-stream";
            } else if line.is_empty() {
                in_headers = false;
            } else if let Some((name, value)) = line.split_once(':') {
                if in_headers && name.eq_ignore_ascii_case("content-type") { kind = value.trim(); }
                if !in_headers && name == "VADE_REDIRECT" { redirect = (!value.trim().is_empty()).then_some(value.trim()); }
            }
        }
        if matches!(status, Some(301 | 302 | 303 | 307 | 308)) {
            if !matches!(reply.code, Some(0 | 47)) { return Err(Failure::Fetch("curl redirect failed")); }
            if hop == REDIRECTS { return Err(Failure::Fetch("redirect limit reached")); }
            url = redirect.ok_or(Failure::Fetch("redirect has no usable destination"))?.to_owned();
            continue;
        }
        let truncated = reply.truncated || reply.code == Some(63) || reply.body.len() > cap;
        if reply.code != Some(0) && !truncated { return Err(Failure::Fetch("curl request failed")); }
        if !status.is_some_and(|s| (200..300).contains(&s)) { return Err(Failure::Fetch("server returned an unsuccessful HTTP status")); }
        let mut body = reply.body;
        body.truncate(cap);
        return Ok(Page { url, kind: kind.into(), body, truncated });
    }
    unreachable!()
}

fn integer(args: &Value, key: &str, default: usize, max: usize) -> Result<usize, ToolError> {
    let n = match args.get(key) {
        None => default,
        Some(Value::Number(n)) => n.parse().map_err(|_| ToolError::InvalidArguments("web limits must be positive integers within the schema bounds"))?,
        _ => return Err(ToolError::InvalidArguments("web limits must be positive integers within the schema bounds")),
    };
    if n == 0 || n > max { return Err(ToolError::InvalidArguments("web limits must be positive integers within the schema bounds")); }
    Ok(n)
}

fn cut(mut text: String, cap: usize, already_cut: bool) -> String {
    if text.len() > cap || already_cut {
        let marker = format!("\n[truncated at {cap} bytes]");
        let mut end = cap.min(text.len());
        while !text.is_char_boundary(end) { end -= 1; }
        text.truncate(end);
        text.push_str(&marker);
    }
    text
}

fn encode(text: &str) -> String {
    let mut encoded = String::new();
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { encoded.push(b as char); }
        else { encoded.push_str(&format!("%{b:02X}")); }
    }
    encoded
}
fn decode(text: &str) -> Option<String> {
    let mut bytes = Vec::new();
    let mut input = text.bytes();
    while let Some(b) = input.next() {
        bytes.push(match b {
            b'+' => b' ',
            b'%' => (input.next()? as char).to_digit(16)? as u8 * 16 + (input.next()? as char).to_digit(16)? as u8,
            _ => b,
        });
    }
    String::from_utf8(bytes).ok()
}
fn entities(text: &str) -> String {
    entities_with_scan(text, || {})
}
fn entities_with_scan(text: &str, mut scan: impl FnMut()) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some((head, tail)) = rest.split_once('&') {
        out.push_str(head);
        // Inspect only the supported name length and its terminator. A long
        // attribute with many ampersands must not rescan the whole suffix.
        let end = tail.bytes().take(17).position(|b| { scan(); b == b';' });
        if let Some(end) = end {
            let (name, after) = (&tail[..end], &tail[end + 1..]);
            let ch = match name {
                "amp" => Some('&'), "quot" => Some('"'), "apos" | "#39" => Some('\''),
                "lt" => Some('<'), "gt" => Some('>'), "nbsp" => Some(' '),
                _ => name.strip_prefix("#x").or_else(|| name.strip_prefix("#X"))
                    .and_then(|n| u32::from_str_radix(n, 16).ok()).or_else(|| name.strip_prefix('#').and_then(|n| n.parse().ok()))
                    .and_then(char::from_u32),
            };
            if let Some(ch) = ch { out.push(ch); rest = after; continue; }
        }
        out.push('&'); rest = tail;
    }
    out.push_str(rest);
    out
}
// Only the result anchors and snippets are needed. Attributes may use either
// quote, or no quote. A quote keeps a '>' inside an attribute out of the lexer.
struct Tag<'a> { name: String, attrs: &'a str, end: usize, closing: bool }
fn tag(html: &str, start: usize) -> Option<Tag<'_>> {
    let mut quote = None;
    let mut end = None;
    for (i, c) in html[start + 1..].char_indices() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (None, '\'' | '"') => quote = Some(c),
            (None, '>') => { end = Some(start + 1 + i); break; }
            _ => (),
        }
    }
    let end = end?;
    let raw = html[start + 1..end].trim();
    let closing = raw.starts_with('/');
    let raw = raw.trim_start_matches('/');
    let name_end = raw.find(|c: char| c.is_ascii_whitespace() || c == '/').unwrap_or(raw.len());
    Some(Tag { name: raw[..name_end].to_ascii_lowercase(), attrs: &raw[name_end..], end: end + 1, closing })
}
fn attr(mut attrs: &str, key: &str) -> Option<String> {
    loop {
        attrs = attrs.trim_start();
        if attrs.is_empty() || attrs == "/" { return None; }
        let end = attrs.find(|c: char| c.is_ascii_whitespace() || c == '=').unwrap_or(attrs.len());
        let name = &attrs[..end];
        attrs = attrs[end..].trim_start();
        if let Some(value) = attrs.strip_prefix('=') {
            attrs = value.trim_start();
            let value;
            if attrs.starts_with(['\'', '"']) {
                let quote = attrs.as_bytes()[0] as char;
                let tail = &attrs[1..];
                let end = tail.find(quote)?;
                value = &tail[..end]; attrs = &tail[end + 1..];
            } else {
                let end = attrs.find(char::is_whitespace).unwrap_or(attrs.len());
                value = &attrs[..end]; attrs = &attrs[end..];
            }
            if name.eq_ignore_ascii_case(key) { return Some(entities(value)); }
        } else if end == 0 { return None; }
    }
}

// A known download cut can leave a partial UTF-8 character. Keep its valid
// prefix before the reader selects an encoding. A complete response with a
// damaged UTF-8 BOM tail is an error, not a cut sample. UTF-16 has its own
// byte boundaries and must reach the reader unchanged.
fn reader_source(bytes: &[u8], truncated: bool) -> Result<&[u8], Failure> {
    if !crate::readers::is_utf16(bytes) && let Err(error) = std::str::from_utf8(bytes) {
        if error.error_len().is_none() {
            if truncated { return Ok(&bytes[..error.valid_up_to()]); }
            if bytes.starts_with(b"\xef\xbb\xbf") {
                return Err(Failure::Fetch("complete response has an incomplete UTF-8 character"));
            }
        }
    }
    Ok(bytes)
}

fn render(page: Page, cap: usize, output_cap: usize, cancel: &AtomicBool) -> Result<String, Failure> {
    render_with(page, cap, output_cap, cancel, crate::readers::read)
}
fn render_with(page: Page, cap: usize, output_cap: usize, cancel: &AtomicBool,
    mut reader: impl FnMut(&[u8], &str, &crate::readers::Caps) -> Result<(crate::readers::Format, crate::readers::Doc), crate::readers::ReadError>) -> Result<String, Failure> {
    let mime = page.kind.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    let header = format!("url: {}\ncontent type: {}\nbytes: {}\n\n", page.url, page.kind, page.body.len());
    // Set the conversion allowance before calling the reader. The download
    // limit does not enlarge the retained result limit.
    // Reserve the header and both possible truncation markers first.
    let conversion_cap = cap.min(output_cap.saturating_sub(header.len()).saturating_sub(128));
    if conversion_cap == 0 { return Err(Failure::Fetch("response metadata exceeds the result limit")); }
    let mut truncated = page.truncated;
    let mut read = |name: &str| {
        let caps = crate::readers::Caps { max_bytes: cap, max_chars: conversion_cap, ..Default::default() };
        let format = crate::readers::sniff(&page.body, name).map(|(format, _)| format);
        let source = if format.is_some_and(|format| format.is_plain_text()) {
            reader_source(&page.body, page.truncated)?
        } else { &page.body };
        let (_, doc) = reader(source, name, &caps)
            .map_err(|_| Failure::Fetch("document reader could not read the response"))?;
        truncated |= doc.warnings.iter().any(|w| w.contains("cap"));
        let mut body = doc.markdown;
        if !doc.warnings.is_empty() { body.push_str(&format!("\nReader notes: {}\n", doc.warnings.join("; "))); }
        Ok::<_, Failure>(body)
    };
    let body = if matches!(mime.as_str(), "text/plain" | "text/markdown" | "application/json" | "text/json") || mime.ends_with("+json") {
        String::from_utf8_lossy(&page.body).into_owned()
    } else if mime == "text/html" || mime == "application/xhtml+xml" {
        read("web.html")?
    } else if mime == "application/pdf" || crate::pdf::is_pdf(&page.body) {
        crate::pdf::text(&page.body, cancel).map_err(|_| Failure::Fetch("document reader could not read the response"))?
    } else {
        let name = match mime.as_str() {
            "text/rtf" | "application/rtf" => "web.rtf",
            "text/csv" => "web.csv",
            "text/tab-separated-values" => "web.tsv",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => "web.docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "web.xlsx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation" => "web.pptx",
            "application/epub+zip" => "web.epub",
            _ => page.url.split(['?', '#']).next().unwrap_or(""),
        };
        match crate::readers::sniff(&page.body, name) {
            Some((format, Some(_))) if !format.is_plain_text() || matches!(format, crate::readers::Format::Html | crate::readers::Format::Csv | crate::readers::Format::Tsv) => read(name)?,
            _ => format!("binary {mime}, {} bytes", page.body.len()),
        }
    };
    if cancel.load(Ordering::Relaxed) { return Err(Failure::Cancelled); }
    Ok(format!("{header}{}", cut(body, conversion_cap, truncated)))
}

pub(crate) fn execute(name: &str, args: &Value, output_cap: usize, cancel: &AtomicBool, log: &mut dyn FnMut(&str)) -> Result<String, ToolError> {
    execute_logged(name, args, output_cap, cancel, run, log)
}
#[cfg(test)]
fn execute_with(name: &str, args: &Value, output_cap: usize, cancel: &AtomicBool,
    runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>) -> Result<String, ToolError> {
    execute_logged(name, args, output_cap, cancel, runner, &mut |_| ())
}
fn execute_logged(name: &str, args: &Value, output_cap: usize, cancel: &AtomicBool,
    runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>, log: &mut dyn FnMut(&str)) -> Result<String, ToolError> {
    let search = name == "web_search";
    let allowed: &[&str] = if search { &["query", "count"] } else { &["url", "max_bytes", "timeout_s"] };
    let Value::Object(fields) = args else { return Err(ToolError::InvalidArguments("web arguments must be a JSON object")); };
    for (i, (key, _)) in fields.iter().enumerate() {
        if !allowed.contains(&key.as_str()) || fields[..i].iter().any(|(old, _)| old == key) {
            return Err(ToolError::InvalidArguments("web arguments have unknown or repeated fields"));
        }
    }
    let key = if search { "query" } else { "url" };
    let input = args.get(key).and_then(Value::as_str).ok_or(ToolError::InvalidArguments("web query or url must be a string"))?;
    if input.trim().is_empty() || input.len() > if search { 2048 } else { MAX_URL } {
        return Err(ToolError::InvalidArguments("web query or url is empty or too long"));
    }
    let max_bytes = if search { DEFAULT_BYTES } else { integer(args, "max_bytes", DEFAULT_BYTES, MAX_BYTES)? };
    let timeout = if search { 20 } else { integer(args, "timeout_s", 20, 60)? };
    let count = if search { integer(args, "count", 5, 10)? } else { 0 };
    // Leave room for the explicit truncation marker inside the executor cap.
    if output_cap < 128 { return Err(ToolError::OutputLimit); }
    if search { return search::run(input, count, max_bytes, timeout, output_cap, cancel, runner, log); }
    let result = fetch_with(input, max_bytes, timeout, cancel, runner).and_then(|page| render(page, max_bytes, output_cap, cancel));
    let text = match result {
        Ok(text) => text,
        Err(Failure::Cancelled) => return Err(ToolError::Cancelled),
        Err(failure) => format!("Web fetch failed: {}.", failure.message()),
    };
    Ok(cut(text, output_cap.saturating_sub(64), false))
}

mod search;
pub(crate) use search::append_log;

#[cfg(test)]
mod tests;
