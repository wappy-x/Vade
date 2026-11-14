//! One `web_search` call: a fixed list of tries, one class for each reply, and
//! text that never lets the model read a failed search as "does not exist".
use super::{Failure, Reply, Tag, attr, cut, decode, encode, fetch_hops, tag, validate_url};
use crate::agent::ToolError;
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A silent provider must leave time for the next one in the shared limit.
const TRY_SECONDS: usize = 6;
const LOG_FILE: &str = "web-search.log";
const LOG_BYTES: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Provider { DuckDuckGo, Bing }
/// The tries of one call, each with the pause before it. DuckDuckGo answers
/// many curl clients with a bot challenge; Bing is the keyless second source.
const TRIES: [(Provider, Duration); 4] = [
    (Provider::DuckDuckGo, Duration::ZERO), (Provider::DuckDuckGo, Duration::from_secs(1)),
    (Provider::Bing, Duration::ZERO), (Provider::Bing, Duration::from_secs(3)),
];
impl Provider {
    fn name(self) -> &'static str {
        match self { Self::DuckDuckGo => "DuckDuckGo", Self::Bing => "Bing" }
    }
    fn label(self) -> &'static str {
        match self { Self::DuckDuckGo => "duckduckgo", Self::Bing => "bing" }
    }
    fn url(self, query: &str) -> String {
        match self {
            Self::DuckDuckGo => format!("https://html.duckduckgo.com/html/?q={}", encode(query)),
            Self::Bing => format!("https://www.bing.com/search?q={}", encode(query)),
        }
    }
    /// Whether a redirect stays with this provider. The query must not go to
    /// another site: a provider can redirect a query to a site that it names.
    fn owns(self, url: &str) -> bool {
        let domain = match self { Self::DuckDuckGo => "duckduckgo.com", Self::Bing => "bing.com" };
        url.strip_prefix("https://").and_then(|rest| rest.split(['/', '?', '#', ':']).next()).map(str::to_ascii_lowercase)
            .is_some_and(|host| host == domain || host.ends_with(&format!(".{domain}")))
    }
    fn scan(self, html: &str, count: usize) -> Scan {
        match self {
            Self::DuckDuckGo => scan(html, count, duck_mark, duck_url),
            Self::Bing => scan(html, count, bing_marks(), bing_url),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class { Results, NoResults, Blocked, BadPage, Failed }
impl Class {
    fn label(self) -> &'static str {
        match self {
            Self::Results => "results", Self::NoResults => "no-results", Self::Blocked => "blocked",
            Self::BadPage => "bad-page", Self::Failed => "failed",
        }
    }
}

type Hit = (String, String, String);
#[derive(Default)]
struct Scan { hits: Vec<Hit>, none: bool, blocked: bool }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark { Skip, Item, Title, Snippet, NoResults, Blocked }

// A provider can decode to a raw UTF-8 path, and validate_url takes ASCII only.
// A non-ASCII host stays refused: its escapes fail the check on the host.
fn ascii_url(url: &str) -> String {
    url.bytes().map(|b| if b.is_ascii() { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}
fn duck_url(href: &str) -> Option<String> {
    let href = if href.starts_with("//") { format!("https:{href}") }
        else if href.starts_with('/') { format!("https://duckduckgo.com{href}") }
        else { href.into() };
    let host = href.split_once("://")?.1.split('/').next()?.to_ascii_lowercase();
    let url = if matches!(host.as_str(), "duckduckgo.com" | "html.duckduckgo.com") {
        let query = href.split_once('?')?.1.split('#').next()?;
        let encoded = query.split('&').find_map(|p| p.strip_prefix("uddg="))?;
        ascii_url(&decode(encoded)?)
    } else { href };
    validate_url(&url).ok()?;
    Some(url)
}
fn base64url(text: &str) -> Option<String> {
    let (mut bits, mut filled, mut bytes) = (0u32, 0u32, Vec::with_capacity(text.len() / 4 * 3 + 3));
    for b in text.bytes().take_while(|b| *b != b'=') {
        let value = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'-' | b'+' => 62,
            b'_' | b'/' => 63,
            _ => return None,
        };
        bits = bits << 6 | u32::from(value);
        filled += 6;
        if filled >= 8 {
            filled -= 8;
            bytes.push((bits >> filled) as u8);
            bits &= (1 << filled) - 1;
        }
    }
    String::from_utf8(bytes).ok()
}
// Bing wraps each result in bing.com/ck/a with the target as `u=a1<base64url>`.
// A wrapped link that does not decode is dropped, not passed on.
fn bing_url(href: &str) -> Option<String> {
    let authority = href.split_once("://")?.1.split(['/', '?', '#']).next()?.to_ascii_lowercase();
    let host = authority.split(':').next()?;
    let url = if host == "bing.com" || host.ends_with(".bing.com") {
        let query = href.split_once('?')?.1.split('#').next()?;
        ascii_url(&base64url(query.split('&').find_map(|p| p.strip_prefix("u=a1"))?)?)
    } else { href.into() };
    validate_url(&url).ok()?;
    Some(url)
}
fn has(class: &str, name: &str) -> bool { class.split_whitespace().any(|s| s == name) }
fn duck_mark(t: &Tag<'_>, class: &str) -> Mark {
    if t.name == "a" && has(class, "result__a") { Mark::Title }
    else if has(class, "result__snippet") { Mark::Snippet }
    else if has(class, "no-results") || has(class, "result--no-result") { Mark::NoResults }
    else if class.split_whitespace().any(|s| s.starts_with("anomaly-modal"))
        || attr(t.attrs, "id").as_deref() == Some("challenge-form") { Mark::Blocked }
    else { Mark::Skip }
}
// A Bing result is `li.b_algo > h2 > a`, then a `p.b_lineclamp<n>` snippet.
fn bing_marks() -> impl FnMut(&Tag<'_>, &str) -> Mark {
    let (mut inside, mut heading) = (false, false);
    move |t, class| {
        // Only the tag directly after the heading can be its title link.
        let after_heading = std::mem::take(&mut heading);
        if t.closing {
            if matches!(t.name.as_str(), "li" | "ol") { inside = false; }
            return Mark::Skip;
        }
        match t.name.as_str() {
            "li" => {
                inside = has(class, "b_algo");
                if has(class, "b_no") { Mark::NoResults } else { Mark::Item }
            }
            "h2" if inside => { heading = true; Mark::Skip }
            "a" if after_heading => Mark::Title,
            "p" if inside && class.split_whitespace().any(|s| s.starts_with("b_lineclamp")) => Mark::Snippet,
            _ => Mark::Skip,
        }
    }
}

fn plain_html(html: &str, cap: usize) -> String {
    let caps = crate::readers::Caps { max_chars: cap, ..Default::default() };
    crate::readers::html_to_markdown_in(html, &mut crate::readers::Budget::new(&caps))
        .split_whitespace().collect::<Vec<_>>().join(" ")
}
fn scan(html: &str, count: usize, mut mark: impl FnMut(&Tag<'_>, &str) -> Mark, url: fn(&str) -> Option<String>) -> Scan {
    let lower = html.to_ascii_lowercase();
    let mut found = Scan::default();
    let mut pos = 0;
    let mut active = None;
    while let Some(start) = html[pos..].find('<').map(|n| pos + n) {
        if html[start..].starts_with("<!--") {
            if let Some(end) = html[start + 4..].find("-->") { pos = start + 4 + end + 3; continue; }
            break;
        }
        let Some(t) = tag(html, start) else { break; };
        pos = t.end;
        // A closing tag has no class. A rule sees it only to end its own state.
        if t.closing { mark(&t, ""); continue; }
        if matches!(t.name.as_str(), "script" | "style") {
            let closing = format!("</{}", t.name);
            if let Some(end) = lower[pos..].find(&closing) { pos += end; }
            else { break; }
            continue;
        }
        let class = attr(t.attrs, "class").unwrap_or_default();
        let title = match mark(&t, &class) {
            Mark::Skip => continue,
            Mark::Item => { active = None; continue; }
            Mark::NoResults => { found.none = true; continue; }
            Mark::Blocked => { found.blocked = true; continue; }
            Mark::Title => true,
            Mark::Snippet => false,
        };
        let close = format!("</{}", t.name);
        let Some(end) = lower[pos..].find(&close) else { break; };
        // Bing ends a snippet with a "Read more" link. An anchor cannot nest in
        // a title, so this cut changes only such a snippet.
        let fragment = html[pos..pos + end].split("<a ").next().unwrap_or("");
        let text = plain_html(fragment, if title { 512 } else { 2048 });
        if title {
            if found.hits.len() == count { break; }
            active = None;
            if let Some(url) = attr(t.attrs, "href").and_then(|href| url(&href)).filter(|_| !text.is_empty()) {
                if !found.hits.iter().any(|(_, old, _)| old == &url) {
                    active = Some(found.hits.len());
                    found.hits.push((text, url, String::new()));
                }
            }
        } else if let Some(i) = active.take() { found.hits[i].2 = text; }
        pos += end;
    }
    found
}
fn numbered(hits: &[Hit]) -> String {
    hits.iter().enumerate().map(|(i, (title, url, snippet))| format!("{}. {title}\n{url}\n{snippet}", i + 1)).collect::<Vec<_>>().join("\n\n")
}

struct Attempt { class: Class, hits: Vec<Hit>, truncated: bool, reason: String }
fn http_status(headers: &str) -> Option<u16> {
    headers.lines().rev().find(|line| line.starts_with("HTTP/"))
        .and_then(|line| line.split_whitespace().nth(1)).and_then(|code| code.parse().ok())
}
fn classify(provider: Provider, status: Option<u16>, fetched: Result<super::Page, Failure>, count: usize) -> Result<Attempt, Failure> {
    let name = provider.name();
    let http = status.map_or_else(|| "no HTTP status".to_owned(), |code| format!("HTTP {code}"));
    let blocked = |truncated| Attempt { class: Class::Blocked, hits: Vec::new(), truncated, reason: format!("{name} blocked the request ({http})") };
    let page = match fetched {
        Err(Failure::Cancelled) => return Err(Failure::Cancelled),
        Err(_) if matches!(status, Some(403 | 429)) => return Ok(blocked(false)),
        Err(failure) => return Ok(Attempt { class: Class::Failed, hits: Vec::new(), truncated: false, reason: format!("{name} failed ({})", failure.message()) }),
        Ok(page) => page,
    };
    let truncated = page.truncated;
    let found = provider.scan(&String::from_utf8_lossy(&page.body), count);
    if !found.hits.is_empty() { return Ok(Attempt { class: Class::Results, hits: found.hits, truncated, reason: String::new() }); }
    if found.blocked || status == Some(202) { return Ok(blocked(truncated)); }
    if found.none { return Ok(Attempt { class: Class::NoResults, hits: Vec::new(), truncated, reason: String::new() }); }
    Ok(Attempt { class: Class::BadPage, hits: Vec::new(), truncated, reason: format!("{name} returned a page without known markers ({http})") })
}

fn log_line(time: u64, provider: Provider, attempt: usize, status: Option<u16>, class: Class, results: usize, ms: u128) -> String {
    let status = status.map_or_else(|| "none".to_owned(), |code| code.to_string());
    format!("{time} web_search provider={} try={attempt} status={status} class={} results={results} ms={ms}", provider.label(), class.label())
}
/// A diagnostic only: a failed write never fails the search.
pub(crate) fn append_log(workspace: &crate::tools::Workspace, line: &str) {
    // Child agents search at the same time. Seek and write must stay one step.
    static WRITER: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _writer = WRITER.lock();
    let _ = (|| -> std::io::Result<()> {
        let mut file = workspace.state_dir()?.journal_file(LOG_FILE)?;
        if file.metadata()?.len() >= LOG_BYTES { file.set_len(0)?; }
        file.seek(SeekFrom::End(0))?;
        file.write_all(format!("{line}\n").as_bytes())
    })();
}

fn pause(wait: Duration, cancel: &AtomicBool) {
    let end = Instant::now() + wait;
    while Instant::now() < end && !cancel.load(Ordering::Relaxed) { std::thread::sleep(Duration::from_millis(10)); }
}
fn search_with(query: &str, count: usize, cap: usize, total: Duration, cancel: &AtomicBool,
    mut runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>,
    mut wait: impl FnMut(Duration, &AtomicBool), log: &mut dyn FnMut(&str)) -> Result<String, Failure> {
    let deadline = Instant::now() + total;
    let mut tries = 0;
    let mut reasons: Vec<(Provider, String)> = Vec::new();
    for (provider, delay) in TRIES {
        if Instant::now() + delay >= deadline { break; }
        if !delay.is_zero() { wait(delay, cancel); }
        if cancel.load(Ordering::Relaxed) { return Err(Failure::Cancelled); }
        tries += 1;
        let started = Instant::now();
        let mut status = None;
        let fetched = fetch_hops(&provider.url(query), cap, TRY_SECONDS, cancel, |url, cap, limit, cancel| {
            status = None;
            let reply = runner(url, cap, limit.min(deadline), cancel)?;
            status = http_status(&reply.headers);
            Ok(reply)
        }, &mut |_, to| if provider.owns(to) { Ok(()) } else { Err(ToolError::Denied) });
        let attempt = classify(provider, status, fetched, count)?;
        let time = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs());
        log(&log_line(time, provider, tries, status, attempt.class, attempt.hits.len(), started.elapsed().as_millis()));
        match attempt.class {
            Class::Results if provider == TRIES[0].0 => return Ok(cut(numbered(&attempt.hits), cap, attempt.truncated)),
            Class::Results => return Ok(cut(format!("Results from {}, because {} did not answer. {} can return pages that do not match the query, so check each title.\n\n{}",
                provider.name(), TRIES[0].0.name(), provider.name(), numbered(&attempt.hits)), cap, attempt.truncated)),
            // A provider's own "no results" is an answer. Bing returned unrelated
            // pages for gibberish, so a fallback here would invent hits.
            Class::NoResults if provider == TRIES[0].0 => return Ok(format!("No results for {query} (provider said none).")),
            Class::NoResults => return Ok(format!("No results for {query} (provider said none). Only {} answered; {} did not. Do not conclude that the thing does not exist.", provider.name(), TRIES[0].0.name())),
            _ => {
                reasons.retain(|(old, _)| *old != provider);
                reasons.push((provider, attempt.reason));
            }
        }
    }
    let reason = if reasons.is_empty() { "no try could start in the time limit".to_owned() }
        else { reasons.into_iter().map(|(_, reason)| reason).collect::<Vec<_>>().join(", ") };
    let tries = if tries == 1 { "1 try".to_owned() } else { format!("{tries} tries") };
    Ok(format!("Web search failed: {reason} after {tries}; the web may be unreachable from this machine. Do not conclude that the thing does not exist."))
}

pub(super) fn run(query: &str, count: usize, cap: usize, seconds: usize, output_cap: usize, cancel: &AtomicBool,
    runner: impl FnMut(&str, usize, Instant, &AtomicBool) -> Result<Reply, Failure>, log: &mut dyn FnMut(&str)) -> Result<String, ToolError> {
    match search_with(query, count, cap, Duration::from_secs(seconds as u64), cancel, runner, pause, log) {
        Ok(text) => Ok(cut(text, output_cap.saturating_sub(64), false)),
        Err(_) => Err(ToolError::Cancelled),
    }
}

#[cfg(test)]
mod tests;
