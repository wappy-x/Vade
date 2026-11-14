use super::super::{DEFAULT_BYTES, decode, entities, entities_with_scan};
use super::*;
use crate::json;

// Recorded 2026-11-01 from the live pages, trimmed, tokens replaced.
const DUCK_RESULTS: &str = include_str!("../../../tests/fixtures/web-search/ddg-results.html");
const DUCK_NONE: &str = include_str!("../../../tests/fixtures/web-search/ddg-no-results.html");
const DUCK_CHALLENGE: &str = include_str!("../../../tests/fixtures/web-search/ddg-challenge.html");
const BING_RESULTS: &str = include_str!("../../../tests/fixtures/web-search/bing-results.html");
const FAILED_TAIL: &str = "the web may be unreachable from this machine. Do not conclude that the thing does not exist.";

fn reply(status: u16, body: &str) -> Reply {
    Reply { body: body.as_bytes().to_vec(), headers: format!("HTTP/1.1 {status} Status\r\nContent-Type: text/html\r\n\r\n\nVADE_REDIRECT:\n"), code: Some(0), truncated: false }
}
fn timed_out() -> Result<Reply, Failure> { Ok(Reply { body: Vec::new(), headers: String::new(), code: Some(28), truncated: false }) }

struct Run { text: String, urls: Vec<String>, waits: Vec<Duration>, logs: Vec<String> }
/// One scripted call. A runner call past the end of the script panics, so the
/// script also proves the number of tries.
fn drive(script: Vec<Result<Reply, Failure>>) -> Run {
    let mut script = script.into_iter();
    let (mut urls, mut waits, mut logs) = (Vec::new(), Vec::new(), Vec::new());
    let text = search_with("flight school", 2, DEFAULT_BYTES, Duration::from_secs(20), &AtomicBool::new(false), |url, _, _, _| {
        urls.push(url.to_owned());
        script.next().expect("no more tries are expected")
    }, |wait, _| waits.push(wait), &mut |line| logs.push(line.to_owned())).unwrap();
    assert!(script.next().is_none(), "the call stopped before the script ended");
    Run { text, urls, waits, logs }
}
fn class_of(provider: Provider, status: Option<u16>, fetched: Result<super::super::Page, Failure>) -> Class {
    classify(provider, status, fetched, 5).unwrap().class
}
fn page(body: &str) -> Result<super::super::Page, Failure> {
    Ok(super::super::Page { url: "https://example.org".into(), kind: "text/html".into(), body: body.as_bytes().to_vec(), truncated: false })
}

#[test]
fn fixtures_and_failures_each_get_their_class() {
    use Provider::{Bing, DuckDuckGo};
    assert_eq!(class_of(DuckDuckGo, Some(200), page(DUCK_RESULTS)), Class::Results);
    assert_eq!(class_of(Bing, Some(200), page(BING_RESULTS)), Class::Results);
    assert_eq!(class_of(DuckDuckGo, Some(200), page(DUCK_NONE)), Class::NoResults);
    assert_eq!(class_of(DuckDuckGo, Some(202), page(DUCK_CHALLENGE)), Class::Blocked);
    // The challenge marker alone is enough, and so is the status alone.
    assert_eq!(class_of(DuckDuckGo, Some(200), page(DUCK_CHALLENGE)), Class::Blocked);
    assert_eq!(class_of(Bing, Some(202), page("<html><p>Please try again.</p></html>")), Class::Blocked);
    for code in [403, 429] {
        assert_eq!(class_of(DuckDuckGo, Some(code), Err(Failure::Fetch("server returned an unsuccessful HTTP status"))), Class::Blocked);
    }
    assert_eq!(class_of(DuckDuckGo, Some(503), Err(Failure::Fetch("server returned an unsuccessful HTTP status"))), Class::Failed);
    assert_eq!(class_of(DuckDuckGo, None, Err(Failure::Timeout)), Class::Failed);
    assert_eq!(class_of(Bing, None, Err(Failure::Fetch("curl request failed"))), Class::Failed);
    // Any changed markup is a bad page, never "no results".
    assert_eq!(class_of(DuckDuckGo, Some(200), page("<html><p>No results.</p></html>")), Class::BadPage);
    assert_eq!(class_of(Bing, Some(200), page(DUCK_RESULTS)), Class::BadPage);
    assert_eq!(class_of(DuckDuckGo, Some(200), page("")), Class::BadPage);
    // No real Bing no-results page was seen. This marker is hand-written.
    assert_eq!(class_of(Bing, Some(200), page(r#"<ol id="b_results"><li class="b_no"><h1>There are no results</h1></li></ol>"#)), Class::NoResults);
    assert!(matches!(classify(DuckDuckGo, None, Err(Failure::Cancelled), 5), Err(Failure::Cancelled)));
}

#[test]
fn results_win_and_a_query_echo_cannot_fake_a_marker() {
    let echo = r#"<input class="search__input" value="no-results anomaly-modal challenge-form"><p>no-results anomaly-modal</p>"#;
    assert_eq!(class_of(Provider::DuckDuckGo, Some(200), page(echo)), Class::BadPage);
    let both = format!("{DUCK_CHALLENGE}{DUCK_NONE}{DUCK_RESULTS}");
    assert_eq!(class_of(Provider::DuckDuckGo, Some(202), page(&both)), Class::Results);
}

#[test]
fn first_provider_results_keep_the_plain_list() {
    let run = drive(vec![Ok(reply(200, DUCK_RESULTS))]);
    assert_eq!(run.urls, ["https://html.duckduckgo.com/html/?q=flight%20school"]);
    assert!(run.waits.is_empty());
    assert!(run.text.starts_with("1. "), "{}", run.text);
    assert!(run.text.contains("\nhttps://doc.rust-lang.org/std/process/struct.Command.html\n"), "{}", run.text);
    assert!(run.text.contains("\n\n2. "), "{}", run.text);
    assert!(!run.text.contains("\n\n3. "), "{}", run.text);
    assert!(!run.text.contains("duckduckgo.com"), "{}", run.text);
    assert!(!run.text.contains("Results from"), "{}", run.text);
    assert_eq!(run.logs.len(), 1);
}

#[test]
fn a_retry_follows_a_block_a_bad_page_or_a_failure() {
    for first in [Ok(reply(202, DUCK_CHALLENGE)), Ok(reply(200, "<html></html>")), Ok(reply(429, "")), timed_out()] {
        let run = drive(vec![first, Ok(reply(200, DUCK_RESULTS))]);
        assert_eq!(run.waits, [Duration::from_secs(1)]);
        assert_eq!(run.urls.len(), 2);
        assert_eq!(run.urls[0], run.urls[1]);
        assert!(run.text.starts_with("1. "), "{}", run.text);
    }
}

#[test]
fn no_results_stops_at_once_with_the_exact_text() {
    let run = drive(vec![Ok(reply(200, DUCK_NONE))]);
    assert_eq!(run.text, "No results for flight school (provider said none).");
    assert_eq!(run.urls.len(), 1);
    assert!(run.waits.is_empty());
    // A provider that answers after the first one was blocked ends the call
    // too, and the text says that only the fallback answered.
    let none = r#"<li class="b_no">There are no results</li>"#;
    let run = drive(vec![Ok(reply(202, DUCK_CHALLENGE)), Ok(reply(202, DUCK_CHALLENGE)), Ok(reply(200, none))]);
    assert_eq!(run.text, "No results for flight school (provider said none). Only Bing answered; DuckDuckGo did not. Do not conclude that the thing does not exist.");
}

#[test]
fn two_blocks_move_to_the_second_provider_and_name_it() {
    let run = drive(vec![Ok(reply(202, DUCK_CHALLENGE)), Ok(reply(202, DUCK_CHALLENGE)), Ok(reply(200, BING_RESULTS))]);
    assert_eq!(run.urls[2], "https://www.bing.com/search?q=flight%20school");
    assert_eq!(run.waits, [Duration::from_secs(1)]);
    assert!(run.text.starts_with("Results from Bing, because DuckDuckGo did not answer. Bing can return pages that do not match the query, so check each title.\n\n1. **Rust** Programming Language\nhttps://rust-lang.org/\nRust is blazingly fast"), "{}", run.text);
    assert!(run.text.contains("\n\n2. **Rust** - Explore, Build and Survive\nhttps://rust.facepunch.com/\n"), "{}", run.text);
    assert!(!run.text.contains("bing.com") && !run.text.contains("Read more"), "{}", run.text);
    let run = drive(vec![timed_out(), timed_out(), timed_out(), Ok(reply(200, BING_RESULTS))]);
    assert_eq!(run.waits, [Duration::from_secs(1), Duration::from_secs(3)]);
    assert!(run.text.starts_with("Results from Bing"), "{}", run.text);
}

#[test]
fn every_try_failing_gives_the_exact_text() {
    let run = drive(vec![Ok(reply(202, DUCK_CHALLENGE)), Ok(reply(202, DUCK_CHALLENGE)), timed_out(), timed_out()]);
    assert_eq!(run.text, format!("Web search failed: DuckDuckGo blocked the request (HTTP 202), Bing failed (request timed out) after 4 tries; {FAILED_TAIL}"));
    assert_eq!(run.logs.len(), 4);
    let run = drive(vec![Ok(reply(200, "<p>changed</p>")), Err(Failure::Fetch("could not start system curl")),
        Ok(reply(403, "")), Ok(reply(200, "<p>changed</p>"))]);
    assert_eq!(run.text, format!("Web search failed: DuckDuckGo failed (could not start system curl), Bing returned a page without known markers (HTTP 200) after 4 tries; {FAILED_TAIL}"));
    assert!(!run.text.contains("No results"));
}

#[test]
fn the_call_stays_inside_the_time_limit_and_each_try_inside_both() {
    let mut deadlines = Vec::new();
    let start = Instant::now();
    let text = search_with("q", 5, DEFAULT_BYTES, Duration::from_millis(1500), &AtomicBool::new(false), |_, _, deadline, _| {
        deadlines.push(deadline);
        Ok(reply(202, DUCK_CHALLENGE))
    }, |wait, _| assert_eq!(wait, Duration::from_secs(1)), &mut |_| ()).unwrap();
    // The 3 s pause does not fit, so the fourth try never starts.
    assert_eq!(text, format!("Web search failed: DuckDuckGo blocked the request (HTTP 202), Bing blocked the request (HTTP 202) after 3 tries; {FAILED_TAIL}"));
    assert_eq!(deadlines.len(), 3);
    assert!(deadlines.iter().all(|deadline| *deadline <= start + Duration::from_millis(1600)));
    // A long limit keeps the cap for one try.
    let mut limit = Duration::MAX;
    search_with("q", 5, DEFAULT_BYTES, Duration::from_secs(20), &AtomicBool::new(false), |_, _, deadline, _| {
        limit = deadline.saturating_duration_since(Instant::now());
        Ok(reply(200, DUCK_RESULTS))
    }, |_, _| (), &mut |_| ()).unwrap();
    assert!(limit <= Duration::from_secs(TRY_SECONDS as u64));
    let text = search_with("q", 5, DEFAULT_BYTES, Duration::ZERO, &AtomicBool::new(false), |_, _, _, _| panic!("no try fits"), |_, _| panic!("no pause fits"), &mut |_| ()).unwrap();
    assert_eq!(text, format!("Web search failed: no try could start in the time limit after 0 tries; {FAILED_TAIL}"));
}

#[test]
fn cancel_stops_before_a_try_and_during_a_pause() {
    let cancel = AtomicBool::new(true);
    let result = search_with("q", 5, DEFAULT_BYTES, Duration::from_secs(20), &cancel, |_, _, _, _| panic!("cancelled call reached the runner"), |_, _| (), &mut |_| ());
    assert_eq!(result, Err(Failure::Cancelled));
    let cancel = AtomicBool::new(false);
    let mut calls = 0;
    let result = search_with("q", 5, DEFAULT_BYTES, Duration::from_secs(20), &cancel, |_, _, _, _| { calls += 1; Ok(reply(202, DUCK_CHALLENGE)) },
        |_, cancel| cancel.store(true, Ordering::Relaxed), &mut |_| ());
    assert_eq!(result, Err(Failure::Cancelled));
    assert_eq!(calls, 1);
    let started = Instant::now();
    pause(Duration::from_secs(30), &AtomicBool::new(true));
    assert!(started.elapsed() < Duration::from_secs(5));
    let started = Instant::now();
    pause(Duration::from_millis(20), &AtomicBool::new(false));
    assert!(started.elapsed() >= Duration::from_millis(20));
    let args = json::parse(r#"{"query":"q"}"#).unwrap();
    assert_eq!(super::super::execute_with("web_search", &args, 4096, &AtomicBool::new(true), |_, _, _, _| panic!()), Err(ToolError::Cancelled));
}

#[test]
fn redirects_inside_a_try_still_pass_the_url_check() {
    let run = drive(vec![Ok(Reply { body: Vec::new(), headers: "HTTP/1.1 302 Found\r\n\r\nVADE_REDIRECT:http://169.254.169.254/\n".into(), code: Some(47), truncated: false }),
        Ok(reply(200, DUCK_RESULTS))]);
    assert_eq!(run.urls.len(), 2);
    assert!(run.urls.iter().all(|url| url.starts_with("https://html.duckduckgo.com/")));
    assert!(run.logs[0].contains(" status=302 class=failed results=0 "), "{}", run.logs[0]);
}

#[test]
fn a_search_redirect_stays_with_its_provider() {
    // A provider can redirect a query to a site that the query names. Such a try fails; the query goes to no other site.
    let away = |target: &str| Ok(Reply { body: Vec::new(), headers: format!("HTTP/1.1 302 Found\r\n\r\nVADE_REDIRECT:{target}\n"), code: Some(47), truncated: false });
    let run = drive(vec![away("https://other.example/search?q=flight+school"), away("http://html.duckduckgo.com/html/?q=x"), Ok(reply(200, BING_RESULTS))]);
    assert_eq!(run.urls.len(), 3);
    assert!(run.urls[..2].iter().all(|url| url.starts_with("https://html.duckduckgo.com/")) && run.urls[2].starts_with("https://www.bing.com/"), "{:?}", run.urls);
    assert!(run.logs[..2].iter().all(|line| line.contains(" status=302 class=failed ")), "{:?}", run.logs);
    // A redirect inside the provider's own domain is followed.
    let run = drive(vec![away("https://duckduckgo.com/html/?q=flight+school"), Ok(reply(200, DUCK_RESULTS))]);
    assert_eq!(run.urls, ["https://html.duckduckgo.com/html/?q=flight%20school", "https://duckduckgo.com/html/?q=flight+school"]);
    for (provider, url, owns) in [(Provider::DuckDuckGo, "https://html.duckduckgo.com/html/", true), (Provider::DuckDuckGo, "https://DuckDuckGo.com:443/", true),
        (Provider::DuckDuckGo, "https://notduckduckgo.com/", false), (Provider::DuckDuckGo, "https://duckduckgo.com.other.example/", false),
        (Provider::Bing, "https://cn.bing.com/search?q=x", true), (Provider::Bing, "http://www.bing.com/search?q=x", false), (Provider::Bing, "https://duckduckgo.com/", false)] {
        assert_eq!(provider.owns(url), owns, "{url}");
    }
}

#[test]
fn the_log_has_one_line_per_try_and_no_query() {
    assert_eq!(log_line(1793584594, Provider::DuckDuckGo, 2, Some(202), Class::Blocked, 0, 207),
        "1793584594 web_search provider=duckduckgo try=2 status=202 class=blocked results=0 ms=207");
    assert_eq!(log_line(7, Provider::Bing, 4, None, Class::Failed, 0, 6000),
        "7 web_search provider=bing try=4 status=none class=failed results=0 ms=6000");
    let run = drive(vec![Ok(reply(202, DUCK_CHALLENGE)), Ok(reply(202, DUCK_CHALLENGE)), timed_out(), Ok(reply(200, BING_RESULTS))]);
    let lines: Vec<_> = run.logs.iter().map(|line| line.split_once(" web_search ").unwrap().1.rsplit_once(" ms=").unwrap().0).collect();
    assert_eq!(lines, ["provider=duckduckgo try=1 status=202 class=blocked results=0", "provider=duckduckgo try=2 status=202 class=blocked results=0",
        "provider=bing try=3 status=none class=failed results=0", "provider=bing try=4 status=200 class=results results=2"]);
    assert!(run.logs.iter().all(|line| !line.contains("flight") && !line.contains("school") && !line.contains("http")));
}

#[test]
fn the_log_file_is_private_and_bounded() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("vade-web-log-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let workspace = crate::tools::Workspace::open(&root, crate::tools::StateLocation::Project).unwrap();
    append_log(&workspace, "first");
    append_log(&workspace, "second");
    let path = root.canonicalize().unwrap().join(".vade").join(LOG_FILE);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "first\nsecond\n");
    assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(LOG_BYTES).unwrap();
    append_log(&workspace, "after");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "after\n");
    let lines: Vec<_> = (0..8).map(|n| format!("line {n}")).collect();
    std::thread::scope(|scope| { for line in &lines { scope.spawn(|| append_log(&workspace, line)); } });
    let mut written: Vec<_> = std::fs::read_to_string(&path).unwrap().lines().map(str::to_owned).collect();
    written.sort();
    assert_eq!(written, ["after", "line 0", "line 1", "line 2", "line 3", "line 4", "line 5", "line 6", "line 7"]);
    // A symlink is refused, and the search does not see the failure.
    std::fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink(root.join("target"), &path).unwrap();
    append_log(&workspace, "never");
    assert!(!root.join("target").exists());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_bing_heading_without_a_direct_link_gives_no_hit() {
    let html = r#"<li class="b_algo"><h2>Title</h2><div><a href="https://example.org/other">Other</a></div></li>
<li class="b_algo"><h2><a href="https://example.net/real">Real</a></h2><p class="b_lineclamp3">Text</p></li>
<li class="b_ad"><h2><a href="https://example.com/ad">Ad</a></h2></li>
<li class="b_algo"><h2>No link</h2><a href="https://example.com/after">After</a></li>
<li class="b_algo"><h2><a href="https://example.net/last">Last</a></h2></li></ol>
<aside><h2><a href="https://example.com/side">Side</a></h2><p class="b_lineclamp2">Side text</p></aside>"#;
    assert_eq!(numbered(&Provider::Bing.scan(html, 5).hits), "1. Real\nhttps://example.net/real\nText\n\n2. Last\nhttps://example.net/last\n");
}

#[test]
fn non_ascii_targets_keep_their_hit_and_a_non_ascii_host_stays_refused() {
    assert_eq!(duck_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fzh.example.org%2Fwiki%2F%E4%B8%AD%E6%96%87"), Some("https://zh.example.org/wiki/%E4%B8%AD%E6%96%87".into()));
    assert_eq!(bing_url("https://www.bing.com/ck/a?u=a1aHR0cHM6Ly96aC5leGFtcGxlLm9yZy93aWtpL-S4reaWhw"), Some("https://zh.example.org/wiki/%E4%B8%AD%E6%96%87".into()));
    assert_eq!(duck_url("//duckduckgo.com/l/?uddg=https%3A%2F%2F%E4%B8%AD.example.org%2F"), None);
}

#[test]
fn bing_links_decode_or_drop() {
    assert_eq!(base64url("aHR0cHM6Ly9ydXN0LWxhbmcub3JnLw"), Some("https://rust-lang.org/".into()));
    assert_eq!(base64url("Pz8_"), Some("???".into()));
    assert_eq!(base64url("Pj4-"), Some(">>>".into()));
    assert_eq!(base64url("YQ=="), Some("a".into()));
    // Decoded bytes that are not UTF-8 give no URL.
    assert_eq!(base64url("_w"), None);
    assert_eq!(base64url("!!!!"), None);
    let wrap = |target: &str| format!("https://www.bing.com/ck/a?!&&p=0&u=a1{target}&ntb=1");
    assert_eq!(bing_url(&wrap("aHR0cHM6Ly9ydXN0LWxhbmcub3JnLw")), Some("https://rust-lang.org/".into()));
    assert_eq!(bing_url("https://example.org/direct"), Some("https://example.org/direct".into()));
    // aHR0cDovLzEyNy4wLjAuMS8 is http://127.0.0.1/.
    assert_eq!(bing_url(&wrap("aHR0cDovLzEyNy4wLjAuMS8")), None);
    assert_eq!(bing_url(&wrap("%%%")), None);
    assert_eq!(bing_url("https://www.bing.com/search?q=other"), None);
    for host in ["cn.bing.com", "www.bing.com:443", "bing.com"] {
        assert_eq!(bing_url(&format!("https://{host}/ck/a?u=a1aHR0cHM6Ly9ydXN0LWxhbmcub3JnLw")), Some("https://rust-lang.org/".into()));
    }
    // A host that only ends in the same letters is not a wrapper.
    assert_eq!(bing_url("https://notbing.com/ck/a?u=a1aHR0cHM6Ly9ydXN0LWxhbmcub3JnLw"), Some("https://notbing.com/ck/a?u=a1aHR0cHM6Ly9ydXN0LWxhbmcub3JnLw".into()));
    assert_eq!(bing_url("javascript:alert(1)"), None);
    assert_eq!(bing_url("data:text/plain,x"), None);
}

const SEARCH: &str = r#"<html><body>
<a href="/settings">Settings</a>
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fguide%3Fa%3D1%26b%3D2&amp;rut=x">Current <b>guide</b></a>
<a class='result__snippet' href='#'>A small &amp; useful <b>snippet</b>.</a>
<a href='https://example.net/news' class='other result__a'>News</a>
<div class=result__snippet>Latest release.</div>
<a class='result__a' href='data:text/plain,wrong'>Wrong</a>
<a class='result__snippet'>Must not attach to News.</a>
</body></html>"#;
fn duck(html: &str, count: usize) -> String { numbered(&Provider::DuckDuckGo.scan(html, count).hits) }

#[test]
fn search_entity_lookahead_is_bounded_with_many_ampersands() {
    for tail in ["", ";", "é&amp;&#233;&#xE9;"] {
        let input = format!("{}{tail}", "&".repeat(250_000));
        let mut scans = 0;
        let decoded = entities_with_scan(&input, || scans += 1);
        assert!((250_000..=input.len() * 17).contains(&scans));
        assert_eq!(decoded, format!("{}{}", "&".repeat(250_000), if tail.starts_with('é') { "é&éé" } else { tail }));
        let html = format!(r#"<a class="{input} result__a" href="https://example.org">Guide</a>"#);
        assert!(html.len() < DEFAULT_BYTES);
        assert_eq!(duck(&html, 5), "1. Guide\nhttps://example.org\n");
    }
    assert_eq!(entities("&ééééééééé;&amp;"), "&ééééééééé;&");
}

#[test]
fn search_parses_only_real_results_and_decodes_redirects() {
    let results = duck(SEARCH, 5);
    assert!(results.starts_with("1. Current **guide**\nhttps://example.org/guide?a=1&b=2\nA small & useful **snippet**."));
    assert!(results.contains("2. News\nhttps://example.net/news\nLatest release."));
    assert!(!results.contains("Settings") && !results.contains("Wrong") && !results.contains("Must not attach"));
    assert!(!duck(SEARCH, 1).contains("News"));
    assert_eq!(duck("<p>No results.</p>", 5), "");
    assert_eq!(decode("a+b%20c%2F%C3%A9"), Some("a b c/é".into()));
    assert_eq!(decode("%xx"), None);
    assert_eq!(duck(r#"<!-- <a class="result__a" href="https://example.org">Fake</a> -->"#, 5), "");
    let args = json::parse(r#"{"query":"Rust & café","count":1}"#).unwrap();
    let result = super::super::execute_with("web_search", &args, 4096, &AtomicBool::new(false), |url, cap, _, _| {
        assert_eq!(url, "https://html.duckduckgo.com/html/?q=Rust%20%26%20caf%C3%A9"); assert_eq!(cap, DEFAULT_BYTES);
        Ok(reply(200, SEARCH))
    }).unwrap();
    assert!(result.contains("Current")); assert!(!result.contains("News"));
}

#[test]
fn a_cut_page_says_so_and_the_output_stays_inside_the_cap() {
    let text = search_with("q", 5, DEFAULT_BYTES, Duration::from_secs(20), &AtomicBool::new(false), |_, _, _, _| {
        let mut page = reply(200, DUCK_RESULTS);
        page.truncated = true;
        Ok(page)
    }, |_, _| (), &mut |_| ()).unwrap();
    assert!(text.ends_with(&format!("[truncated at {DEFAULT_BYTES} bytes]")), "{text}");
    let long = "q".repeat(2048);
    let text = run(&long, 5, DEFAULT_BYTES, 20, 256, &AtomicBool::new(false), |_, _, _, _| Ok(reply(200, DUCK_NONE)), &mut |_| ()).unwrap();
    assert!(text.len() <= 256);
    assert!(text.starts_with("No results for qqq"));
}
