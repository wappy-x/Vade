//! Bounded public web reads. No workspace files, credentials or curl config.
use crate::{agent::ToolError, json::Value};
use std::io::{self, Read};
use std::net::{IpAddr, Ipv4Addr};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
enum Failure { Cancelled, Timeout, Denied, Refused(&'static str), Fetch(&'static str) }
impl Failure {
    fn message(&self) -> &'static str {
        match self {
            Self::Cancelled => "request cancelled",
            Self::Timeout => "request timed out",
            Self::Denied => "the redirect to another site was not approved",
            Self::Refused(message) | Self::Fetch(message) => message,
        }
    }
}

fn private_v4(ip: Ipv4Addr) -> bool {
    let [first, second, third, _] = ip.octets();
    // The shared address space of RFC 6598 (100.64/10) is for carrier NAT and
    // private overlay networks. 198.18.0.0/15 is for tests between networks, and local
    // container and VM bridges use it. 240.0.0.0/4 is reserved.
    // The documentation ranges, 192.0.0.0/24 (protocol assignments) and
    // 192.88.99.0/24 (the old 6to4 relay range) are not global either.
    ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified()
        || ip.is_broadcast() || ip.is_multicast() || ip.is_documentation() || first == 0 || first >= 240
        || (first == 100 && second & 0xc0 == 64) || (first == 198 && second & 0xfe == 18)
        || (first == 192 && second == 0 && third == 0) || (first == 192 && second == 88 && third == 99)
}
/// An address on this machine or a private network, as a literal in a URL or
/// as the answer to a name lookup. An IPv6 address that carries an IPv4
/// address is as private as the IPv4 address that it carries: a translator
/// on the path (NAT64, 6to4) sends the packet to that IPv4 address.
fn private_ip(ip: IpAddr) -> bool {
    let ip = match ip {
        IpAddr::V4(ip) => return private_v4(ip),
        IpAddr::V6(ip) => ip,
    };
    // Keep the existing IPv4-mapped and IPv4-compatible forms.
    if let Some(ip) = ip.to_ipv4() { return private_v4(ip); }
    let carried = |high: u16, low: u16| private_v4(Ipv4Addr::from((u32::from(high) << 16) | u32::from(low)));
    // ISATAP (RFC 5214) carries IPv4 in the interface ID under any prefix.
    // Check it separately so neither encoding can hide a refused address.
    let isatap_private = matches!(ip.segments(), [.., 0 | 0x0200, 0x5efe, high, low] if carried(high, low));
    let carries_private = match ip.segments() {
        // Keep the existing NAT64 well-known and IPv4-translated forms.
        [0x64, 0xff9b, 0, 0, 0, 0, high, low] | [0, 0, 0, 0, 0xffff, 0, high, low] => carried(high, low),
        // All other IPv6 destinations must be in global unicast (2000::/3).
        // This also refuses local NAT64, discard, dummy and local SRv6 space.
        [first, ..] if first & 0xe000 != 0x2000 => true,
        // Refuse the whole IETF assignment block (2001::/23), including
        // Teredo and identifier space. No public service exceptions apply.
        [0x2001, 0..=0x01ff, ..] | [0x2001, 0x0db8, ..] => true,
        // 6to4 still checks its IPv4 destination. Documentation stays refused.
        [0x2002, high, low, ..] => carried(high, low),
        [0x3fff, second, ..] => second & 0xf000 == 0,
        _ => false,
    };
    isatap_private || carries_private
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
    let dotted = host.ends_with('.');
    let host = host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase();
    // Refuse local namespaces before DNS, even if they return global addresses.
    // Include common local aliases as well as reserved local suffixes.
    if ["localhost", "local", "home.arpa", "internal", "lan", "home", "corp", "localdomain", "test", "invalid", "example", "onion"].iter()
        .any(|suffix| host == *suffix || host.strip_suffix(suffix).is_some_and(|prefix| prefix.ends_with('.'))) {
        return refuse("local hosts are refused");
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        // Curl reads an address with a final dot as a name and looks it up.
        if dotted { return refuse("invalid or ambiguous URL host"); }
        if private_ip(ip) { return refuse("local and private address ranges are refused"); }
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

/// The site of a URL that `validate_url` accepted: its scheme and its host
/// with the port, in lower case. The port that the scheme implies is left
/// out, so `https://a.example:443` and `https://a.example` are one site. An
/// approval names a site.
pub(crate) fn site(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    let authority = rest.split(['/', '?', '#']).next()?.to_ascii_lowercase();
    let implied = if scheme == "https" { ":443" } else { ":80" };
    Some(format!("{scheme}://{}", authority.strip_suffix(implied).unwrap_or(&authority)))
}
/// The key that a saved approval holds for a site: one spelling for one
/// place that curl connects to. On top of `site`, a final dot on a name,
/// leading zeros in a port and the spelling of an IPv6 address are made
/// plain, since they do not change the connection. `None` when `key` is not
/// the site of a URL that `validate_url` accepts.
pub(crate) fn saved_site(key: &str) -> Option<String> {
    validate_url(key).ok()?;
    let key = site(key)?;
    let (scheme, authority) = key.split_once("://")?;
    let (host, port) = match authority.strip_prefix('[') {
        Some(bracketed) => {
            let (host, tail) = bracketed.split_once(']')?;
            (format!("[{}]", host.parse::<std::net::Ipv6Addr>().ok()?), tail.strip_prefix(':'))
        }
        None => {
            let (host, port) = authority.split_once(':').map_or((authority, None), |(host, port)| (host, Some(port)));
            (host.strip_suffix('.').unwrap_or(host).to_owned(), port)
        }
    };
    let port = match port {
        Some(port) => Some(port.parse::<u16>().ok()?),
        None => None,
    };
    let implied = if scheme == "https" { 443 } else { 80 };
    Some(match port.filter(|port| *port != implied) {
        Some(port) => format!("{scheme}://{host}:{port}"),
        None => format!("{scheme}://{host}"),
    })
}
/// Whether `text` is a saved site key exactly as `saved_site` writes it. A
/// saved approval must be one: no pattern, path, user info, upper-case
/// letter, default port or second spelling of one site.
pub(crate) fn site_key(text: &str) -> bool {
    saved_site(text).as_deref() == Some(text)
}
/// Whether a redirect leaves the site that got the request: another host or
/// port, or a step from https to http. Only a step from http to https on the
/// same host stays, with the same port after each scheme's default port is
/// dropped: 80 to 443, or one explicit port on both sides. A name and
/// its `www.` name are two sites here, as for a direct fetch and for a saved
/// approval, so a redirect between them asks.
fn leaves(from: &str, to: &str) -> bool {
    let (Some(from), Some(to)) = (site(from), site(to)) else { return true };
    let host = |site: &str| site.split_once("://").map_or(String::new(), |(_, host)| host.to_owned());
    host(&from) != host(&to) || (from.starts_with("https://") && !to.starts_with("https://"))
}
/// The host and port of a URL that `validate_url` accepted: the name or the
/// address literal as curl reads it, and the port that the scheme implies.
fn endpoint(url: &str) -> Option<(&str, u16)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let (host, port) = match authority.strip_prefix('[') {
        Some(bracketed) => { let (host, tail) = bracketed.split_once(']')?; (host, tail.strip_prefix(':')) }
        None => authority.split_once(':').map_or((authority, None), |(host, port)| (host, Some(port))),
    };
    let default = if scheme.eq_ignore_ascii_case("https") { 443 } else { 80 };
    Some((host, port.map_or(Some(default), |port| port.parse().ok())?))
}
/// The addresses that curl may connect to, from the answer to the name lookup.
/// One local or private address refuses the whole answer: a public name can
/// point at this machine. An address literal was checked with the URL.
fn public_addresses(host: &str, lookup: impl FnOnce(&str) -> Result<Vec<IpAddr>, Failure>) -> Result<Vec<IpAddr>, Failure> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        if private_ip(ip) { return Err(Failure::Refused("local and private address ranges are refused")); }
        return Ok(vec![ip]);
    }
    let name = host.trim_end_matches('.').to_ascii_lowercase();
    let addresses = lookup(&name)?;
    if addresses.is_empty() { return Err(Failure::Fetch("could not find the address of the host")); }
    if addresses.iter().any(|ip| private_ip(*ip)) { return Err(Failure::Refused("the host name resolves to a local or private address")); }
    let mut unique = Vec::new();
    for ip in addresses { if !unique.contains(&ip) { unique.push(ip); } }
    Ok(unique)
}
/// The `--resolve` entries that make curl connect to `addresses` and to no
/// other address, so the checked address is the one that gets the request.
/// An address literal needs none.
fn pin(host: &str, port: u16, addresses: &[IpAddr]) -> Vec<String> {
    if host.parse::<IpAddr>().is_ok() { return Vec::new(); }
    let list = addresses.iter().map(|ip| match ip { IpAddr::V4(ip) => ip.to_string(), IpAddr::V6(ip) => format!("[{ip}]") }).collect::<Vec<_>>().join(",");
    // Curl reads a host name in lower case. It can keep or drop a final dot
    // of the name; pin both spellings. The last entry covers each other
    // spelling: `*` sends every name that curl looks up for this port to the
    // checked addresses (curl 7.64 and later; an older curl ignores it).
    let host = host.to_ascii_lowercase();
    let name = host.trim_end_matches('.');
    let mut names = vec![name];
    if name.len() != host.len() { names.push(&host); }
    names.push("*");
    names.into_iter().map(|name| format!("{name}:{port}:{list}")).collect()
}
/// Whether curl reports a connection to one of the checked addresses. A reply
/// that holds an HTTP response must also hold the report, unless curl was
/// stopped at the size limit before it could write the report.
fn connected_to(headers: &str, addresses: &[IpAddr], stopped: bool) -> bool {
    let mut reports = headers.lines().filter_map(|line| line.strip_prefix("VADE_REMOTE:")).filter(|ip| !ip.is_empty()).peekable();
    if reports.peek().is_none() { return stopped || !headers.lines().any(|line| line.starts_with("HTTP/")); }
    reports.all(|ip| ip.parse::<IpAddr>().is_ok_and(|ip| addresses.contains(&ip)))
}
/// The system name lookup, off this thread so the deadline and a cancel apply.
fn lookup(host: &str, port: u16, deadline: Instant, cancel: &AtomicBool) -> Result<Vec<IpAddr>, Failure> {
    use std::net::ToSocketAddrs;
    if cfg!(test) { return Err(Failure::Fetch("network is disabled in tests")); }
    let name = host.to_owned();
    lookup_with(move || (name.as_str(), port).to_socket_addrs().map(|list| list.map(|address| address.ip()).collect()), deadline, cancel)
}
/// Lookup threads still running, including those past their deadline: such
/// a thread lives until the system resolver gives up.
static LOOKUPS: AtomicUsize = AtomicUsize::new(0);
const MAX_LOOKUPS: usize = 8;
// Release on unwind or failed thread start, but not while a canceled lookup runs.
struct LookupSlot(&'static AtomicUsize);
impl Drop for LookupSlot {
    fn drop(&mut self) { self.0.fetch_sub(1, Ordering::AcqRel); }
}
fn lookup_with(query: impl FnOnce() -> io::Result<Vec<IpAddr>> + Send + 'static, deadline: Instant, cancel: &AtomicBool) -> Result<Vec<IpAddr>, Failure> {
    lookup_capped(query, deadline, cancel, &LOOKUPS)
}
fn lookup_capped(query: impl FnOnce() -> io::Result<Vec<IpAddr>> + Send + 'static, deadline: Instant, cancel: &AtomicBool,
    running: &'static AtomicUsize) -> Result<Vec<IpAddr>, Failure> {
    if running.fetch_add(1, Ordering::AcqRel) >= MAX_LOOKUPS {
        running.fetch_sub(1, Ordering::AcqRel);
        return Err(Failure::Fetch("too many address lookups are still running; try again later"));
    }
    let slot = LookupSlot(running);
    let (found, answer) = std::sync::mpsc::channel();
    // Bind the slot inside the thread so unwind drops it before the sender.
    let thread = std::thread::Builder::new().name("vade-web-resolve".into())
        .spawn(move || { let slot = slot; let addresses = query(); drop(slot); let _ = found.send(addresses); });
    if thread.is_err() {
        return Err(Failure::Fetch("could not start the address lookup"));
    }
    loop {
        if cancel.load(Ordering::Relaxed) { return Err(Failure::Cancelled); }
        if Instant::now() >= deadline { return Err(Failure::Timeout); }
        match answer.recv_timeout(Duration::from_millis(10)) {
            Ok(Ok(addresses)) => return Ok(addresses),
            Ok(Err(_)) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Err(Failure::Fetch("could not find the address of the host")),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}

fn curl_command(url: &str, pins: &[String], cap: usize, remaining: Duration) -> Command {
    let mut command = Command::new("/usr/bin/curl");
    // Drive redirects one hop at a time so each destination is checked first.
    // Ignore Content-Length to retain a prefix of large responses. The pipe
    // cap also covers curl versions that only check the announced size.
    // HTTP/1.1 with Connection: close gives the ignored-length body an end.
    command.env_clear().args(["-q", "-sSL", "--globoff", "--proto", "=http,https", "--proto-redir", "=http,https",
        "--max-redirs", "0", "--user-agent", "vade-harness", "--proxy", "", "--dump-header", "/dev/stderr",
        "--write-out", "%{stderr}\nVADE_REDIRECT:%{redirect_url}\nVADE_REMOTE:%{remote_ip}\n", "--ignore-content-length", "--http1.1", "--header", "Connection: close"])
        .args(pins.iter().flat_map(|pin| ["--resolve", pin.as_str()]))
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
    run_with(url, cap, deadline, cancel, lookup, |command| run_command(command, cap, deadline, cancel))
}
/// Resolve the name, check each address, then give curl only those addresses
/// and check the peer address curl reports.
fn run_with(url: &str, cap: usize, deadline: Instant, cancel: &AtomicBool,
    resolve: impl FnOnce(&str, u16, Instant, &AtomicBool) -> Result<Vec<IpAddr>, Failure>,
    start: impl FnOnce(Command) -> Result<Reply, Failure>) -> Result<Reply, Failure> {
    validate_url(url)?;
    let (host, port) = endpoint(url).ok_or(Failure::Refused("invalid URL host"))?;
    let addresses = public_addresses(host, |name| resolve(name, port, deadline, cancel))?;
    if cancel.load(Ordering::Relaxed) { return Err(Failure::Cancelled); }
    if Instant::now() >= deadline { return Err(Failure::Timeout); }
    let pins = pin(host, port, &addresses);
    let reply = start(curl_command(url, &pins, cap, deadline.saturating_duration_since(Instant::now())))?;
    if !connected_to(&reply.headers, &addresses, reply.truncated) {
        return Err(Failure::Refused("the connection went to an address that was not checked"));
    }
    Ok(reply)
}
/// A fetch that follows each redirect, for tests of the fetch itself.
#[cfg(test)]
fn fetch_with(url: &str, cap: usize, timeout: usize, cancel: &AtomicBool,
    runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>) -> Result<Page, Failure> {
    fetch_hops(url, cap, timeout, cancel, runner, &mut |_, _| Ok(()))
}
/// A redirect can send the URL that the model wrote, or a part of it, to a
/// site that no one approved: many sites redirect to any address in the
/// query. So `hop` is asked, with the old and the new URL, before a redirect
/// leaves the site that got the request. An error stops the fetch.
fn fetch_hops(url: &str, cap: usize, timeout: usize, cancel: &AtomicBool,
    mut runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>,
    hop: &mut dyn FnMut(&str, &str) -> Result<(), ToolError>) -> Result<Page, Failure> {
    let mut deadline = Instant::now() + Duration::from_secs(timeout as u64);
    let mut url = url.to_owned();
    for step in 0..=REDIRECTS {
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
            if step == REDIRECTS { return Err(Failure::Fetch("redirect limit reached")); }
            let next = redirect.ok_or(Failure::Fetch("redirect has no usable destination"))?.to_owned();
            validate_url(&next)?;
            if leaves(&url, &next) {
                // A person can read a prompt here. That wait is not fetch time.
                let asked = Instant::now();
                hop(&url, &next).map_err(|error| if error == ToolError::Cancelled { Failure::Cancelled } else { Failure::Denied })?;
                deadline += asked.elapsed();
            }
            url = next;
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

/// Where a call sends data, for the approval prompt: the approval key and the
/// text to show. A fetch gives its site and its URL; a search gives one fixed
/// key and its query. `None` when the arguments name no usable target;
/// `execute` then refuses the call before it sends a request.
pub(crate) fn target(name: &str, args: &Value) -> Option<(String, String)> {
    if name == "web_search" {
        return Some((SEARCH_TARGET.into(), args.get("query")?.as_str()?.to_owned()));
    }
    let url = args.get("url")?.as_str()?;
    validate_url(url).ok()?;
    Some((site(url)?, url.to_owned()))
}
/// The approval key of every `web_search` call. It is not a site.
pub(crate) const SEARCH_TARGET: &str = "web search";

/// `hop` decides each redirect of a fetch that leaves the site of the
/// request (see `fetch_hops`). A search keeps its redirects with its provider.
pub(crate) fn execute(name: &str, args: &Value, output_cap: usize, cancel: &AtomicBool, log: &mut dyn FnMut(&str),
    hop: &mut dyn FnMut(&str, &str) -> Result<(), ToolError>) -> Result<String, ToolError> {
    execute_logged(name, args, output_cap, cancel, run, log, hop)
}
#[cfg(test)]
fn execute_with(name: &str, args: &Value, output_cap: usize, cancel: &AtomicBool,
    runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>) -> Result<String, ToolError> {
    execute_logged(name, args, output_cap, cancel, runner, &mut |_| (), &mut |_, _| Ok(()))
}
fn execute_logged(name: &str, args: &Value, output_cap: usize, cancel: &AtomicBool,
    runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>, log: &mut dyn FnMut(&str),
    hop: &mut dyn FnMut(&str, &str) -> Result<(), ToolError>) -> Result<String, ToolError> {
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
    let result = fetch_hops(input, max_bytes, timeout, cancel, runner, hop).and_then(|page| render(page, max_bytes, output_cap, cancel));
    let text = match result {
        Ok(text) => text,
        Err(Failure::Cancelled) => return Err(ToolError::Cancelled),
        Err(Failure::Denied) => return Err(ToolError::Denied),
        Err(failure) => format!("Web fetch failed: {}.", failure.message()),
    };
    Ok(cut(text, output_cap.saturating_sub(64), false))
}

mod search;
pub(crate) use search::append_log;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod guard_tests;
