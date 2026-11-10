use super::*;
use crate::json;
use std::io::Write;

fn response(body: &[u8], kind: &str) -> Reply {
    Reply { body: body.to_vec(), headers: format!("HTTP/2 200\r\nContent-Type: {kind}\r\n\r\n"), code: Some(0), truncated: false }
}
fn redirect(url: &str) -> Reply {
    Reply { body: Vec::new(), headers: format!("HTTP/2 302\r\n\r\nVADE_REDIRECT:{url}\n"), code: Some(47), truncated: false }
}
fn call(name: &str, args: &str, reply: Reply) -> String {
    let mut reply = Some(reply);
    execute_with(name, &json::parse(args).unwrap(), 65536, &AtomicBool::new(false), |_, _, _, _| Ok(reply.take().unwrap())).unwrap()
}

#[test]
fn url_checks_run_before_the_runner() {
    let mut refused: Vec<String> = ["file:///private", "ftp://example.org/a", "data:text/plain,x", "http://localhost/a",
        "https://sub.localhost./a", "https://%6cocalhost/a", "https://example.org\\x",
        "https://example.org:bad/a", "https://example.org:0", "https://example.org/a\nheader", "https://example.org bad",
        "https://example.org:65536", "https://[bad]/", "https://example.org:/"]
        .into_iter().map(str::to_owned).collect();
    let numeric = u32::from(Ipv4Addr::from([127, 0, 0, 1]));
    refused.push(format!("https://name:secret{}example.org", '@'));
    refused.extend([format!("https://{numeric}"), format!("https://0x{numeric:x}")]);
    for octets in [[127, 0, 0, 1], [10, 20, 30, 40], [172, 16, 0, 1], [172, 31, 255, 255], [192, 168, 4, 5], [169, 254, 2, 3]] {
        let ip = Ipv4Addr::from(octets);
        refused.push(format!("http://{ip}/"));
        refused.push(format!("http://[{}]/", ip.to_ipv6_mapped()));
        refused.push(format!("http://[{}]/", ip.to_ipv6_compatible()));
    }
    for segments in [[0, 0, 0, 0, 0, 0, 0, 1], [0xfc00, 0, 0, 0, 0, 0, 0, 1], [0xfdff, 0, 0, 0, 0, 0, 0, 1], [0xfe80, 0, 0, 0, 0, 0, 0, 1]] {
        refused.push(format!("https://[{}]/", std::net::Ipv6Addr::from(segments)));
    }
    for url in refused {
        assert!(matches!(fetch_with(&url, 64, 1, &AtomicBool::new(false), |_, _, _, _| panic!("refused URL reached runner")), Err(Failure::Refused(_))));
    }
    for url in ["https://example.org/a?q=x#part", "http://example.org:8080/", "HTTPS://EXAMPLE.ORG/"] { assert!(validate_url(url).is_ok()); }
}

#[test]
fn redirects_validate_each_destination_and_share_a_deadline() {
    let cancel = AtomicBool::new(false);
    let mut calls = 0;
    let mut first_deadline = None;
    let page = fetch_with("https://example.org/start", 64, 20, &cancel, |url, cap, deadline, _| {
        calls += 1; assert_eq!(cap, 64);
        if let Some(first) = first_deadline { assert_eq!(deadline, first); } else { first_deadline = Some(deadline); }
        Ok(if calls == 1 { redirect("https://example.org/end") } else { assert!(url.ends_with("/end")); response(b"done", "text/plain") })
    }).unwrap();
    assert_eq!(page.url, "https://example.org/end"); assert_eq!(page.body, b"done"); assert_eq!(calls, 2);
    for target in ["file:///private".into(), format!("http://{}/", Ipv4Addr::from([10, 1, 2, 3]))] {
        calls = 0;
        let result = fetch_with("https://example.org", 64, 20, &cancel, |_, _, _, _| { calls += 1; Ok(redirect(&target)) });
        assert!(matches!(result, Err(Failure::Refused(_)))); assert_eq!(calls, 1);
    }
    calls = 0;
    let result = fetch_with("https://example.org", 64, 20, &cancel, |_, _, _, _| { calls += 1; Ok(redirect("https://example.org/again")) });
    assert!(matches!(result, Err(Failure::Fetch("redirect limit reached")))); assert_eq!(calls, REDIRECTS + 1);
}

#[test]
fn fetch_truncates_text_and_context_on_utf8_boundaries() {
    let text = call("web_fetch", r#"{"url":"https://example.org","max_bytes":7}"#, response("abcdéé!".as_bytes(), "text/plain"));
    assert!(text.contains("bytes: 7\n")); assert!(text.contains("[truncated at 7 bytes]"));
    let args = json::parse(r#"{"url":"https://example.org"}"#).unwrap();
    let text = execute_with("web_fetch", &args, 200, &AtomicBool::new(false), |_, _, _, _| Ok(response("é".repeat(1000).as_bytes(), "text/plain"))).unwrap();
    assert!(text.len() <= 200); assert!(text.contains("[truncated at"));
    assert_eq!(cut("aéz".into(), 2, false), "a\n[truncated at 2 bytes]");
    let mut reply = response(b"part", "text/plain"); reply.code = Some(63);
    assert!(call("web_fetch", r#"{"url":"https://example.org","max_bytes":4}"#, reply).contains("[truncated at 4 bytes]"));
}

#[test]
fn html_uses_the_document_reader_and_other_types_stay_clear() {
    let html = b"<html><head><script>secret()</script></head><body><h1>Title</h1><p>Hello <strong>reader</strong>.</p></body></html>";
    let expected = crate::readers::read(html, "web.html", &crate::readers::Caps::default()).unwrap().1.markdown;
    let text = call("web_fetch", r#"{"url":"https://example.org"}"#, response(html, "text/html; charset=utf-8"));
    assert!(text.contains(&expected)); assert!(text.contains("# Title")); assert!(text.contains("**reader**")); assert!(!text.contains("secret"));
    assert!(call("web_fetch", r#"{"url":"https://example.org"}"#, response(br"{\rtf1 hello}", "application/rtf")).contains("hello"));
    for kind in ["text/plain", "text/markdown", "application/json", "application/problem+json"] {
        assert!(call("web_fetch", r#"{"url":"https://example.org"}"#, response(b"unchanged", kind)).ends_with("unchanged"));
    }
    assert!(call("web_fetch", r#"{"url":"https://example.org"}"#, response(&[0, 255, 1], "image/png")).ends_with("binary image/png, 3 bytes"));
    assert!(call("web_fetch", r#"{"url":"https://example.org"}"#, response(b"%PDF-invalid", "application/pdf")).contains("document reader"));
}

#[test]
fn html_download_cuts_preserve_complete_utf8_characters() {
    for (path, kind, start) in [
        ("", "text/html", "<p>"),
        ("/page.htm", "application/octet-stream", "<p>"),
        ("/page.xhtml", "application/octet-stream", "<p>"),
        ("/page.xht", "application/octet-stream", "<p>"),
        ("/page.shtml", "application/octet-stream", "<p>"),
        ("", "application/octet-stream", "<html><p>"),
    ] {
        for bom in ["", "\u{feff}"] {
            let html = format!("{bom}{start}ééé</p>");
            let limit = bom.len() + start.len() + 5;
            let args = format!(r#"{{"url":"https://example.org{path}","max_bytes":{limit}}}"#);
            // Exercise the pipe cap, curl's size-limit exit, and a runner that
            // returns more than requested. Each reaches the same reader path.
            for mode in 0..3 {
                let mut reply = response(html.as_bytes(), kind);
                if mode != 2 { reply.body.truncate(limit); }
                reply.truncated = mode == 0;
                if mode == 1 { reply.code = Some(63); }
                let text = call("web_fetch", &args, reply);
                assert_eq!(text.split_once("\n\n").unwrap().1, format!("éé\n\n[truncated at {limit} bytes]"));
                assert!(text.contains(&format!("bytes: {limit}\n")));
            }
            let args = format!(r#"{{"url":"https://example.org{path}"}}"#);
            let complete = call("web_fetch", &args, response(html.as_bytes(), kind));
            assert_eq!(complete.split_once("\n\n").unwrap().1, "ééé\n");
        }
        let mut damaged = format!("\u{feff}{start}caf").into_bytes();
        damaged.push(0xc3);
        let args = format!(r#"{{"url":"https://example.org{path}"}}"#);
        assert_eq!(call("web_fetch", &args, response(&damaged, kind)),
            "Web fetch failed: complete response has an incomplete UTF-8 character.");
    }
    // A legacy encoding error inside the source is not a cut UTF-8 tail.
    let legacy = b"<p>caf\xe9</p>";
    assert_eq!(reader_source(legacy, true).unwrap(), legacy);
    let utf16 = b"\xff\xfe<\0p\0>\0a";
    assert_eq!(reader_source(utf16, true).unwrap(), utf16);
}

#[test]
fn reader_allowance_is_set_before_conversion_storage_is_built() {
    let cancel = AtomicBool::new(false);
    let page = fetch_with("https://example.org", MAX_BYTES, 20, &cancel,
        |_, _, _, _| Ok(response(b"<p>data</p>", "text/html"))).unwrap();
    let mut called = false;
    let text = render_with(page, MAX_BYTES, 4096, &cancel, |bytes, name, caps| {
        called = true;
        assert_eq!(bytes, b"<p>data</p>");
        assert_eq!(name, "web.html");
        assert_eq!(caps.max_bytes, MAX_BYTES);
        let header = "url: https://example.org\ncontent type: text/html\nbytes: 11\n\n";
        assert_eq!(caps.max_chars, 4096 - header.len() - 128);
        assert_eq!(crate::readers::Budget::new(caps).room(), crate::readers::MIN_WORK_BYTES);
        crate::readers::read(bytes, name, caps)
    }).unwrap();
    assert!(called);
    assert!(text.ends_with("data\n"));
    let page = Page { url: "https://example.org".into(), kind: "text/html".into(), body: Vec::new(), truncated: false };
    assert_eq!(render_with(page, MAX_BYTES, 128, &cancel, |_, _, _| panic!("reader ran without result room")),
        Err(Failure::Fetch("response metadata exceeds the result limit")));
}

#[test]
fn failures_and_bad_arguments_are_explicit() {
    let args = json::parse(r#"{"url":"https://example.org"}"#).unwrap();
    let mut reply = response(b"", "text/plain"); reply.code = Some(28);
    assert_eq!(call("web_fetch", r#"{"url":"https://example.org"}"#, reply), "Web fetch failed: request timed out.");
    let mut reply = response(b"", "text/plain"); reply.code = Some(6); reply.headers.clear();
    assert_eq!(call("web_fetch", r#"{"url":"https://example.org"}"#, reply), "Web fetch failed: curl request failed.");
    let mut reply = response(b"", "text/plain"); reply.headers = "HTTP/2 404\r\n\r\n".into();
    assert_eq!(call("web_fetch", r#"{"url":"https://example.org"}"#, reply), "Web fetch failed: server returned an unsuccessful HTTP status.");
    let cancel = AtomicBool::new(true);
    assert_eq!(execute_with("web_fetch", &args, 4096, &cancel, |_, _, _, _| panic!()), Err(ToolError::Cancelled));
    for (name, args) in [("web_fetch", r#"{"url":"https://example.org","max_bytes":0}"#),
        ("web_fetch", r#"{"url":"https://example.org","max_bytes":2097153}"#),
        ("web_fetch", r#"{"url":"https://example.org","timeout_s":61}"#),
        ("web_fetch", r#"{"url":"https://example.org","timeout_s":"20"}"#),
        ("web_fetch", r#"{"url":"https://example.org","cookie":"secret"}"#),
        ("web_search", r#"{"query":"q","count":11}"#), ("web_search", r#"{"query":""}"#), ("web_fetch", "[]")] {
        assert!(matches!(execute_with(name, &json::parse(args).unwrap(), 4096, &AtomicBool::new(false), |_, _, _, _| panic!()), Err(ToolError::InvalidArguments(_))));
    }
    assert!(enabled(None)); assert!(!enabled(Some(std::ffi::OsStr::new("0")))); assert!(enabled(Some(std::ffi::OsStr::new("1"))));
    assert!(matches!(run("https://example.org", 64, Instant::now(), &AtomicBool::new(false)), Err(Failure::Fetch("network is disabled in tests"))));
}

#[test]
fn curl_command_has_fixed_policy() {
    let command = curl_command("https://example.org", 123, Duration::from_secs(7));
    assert_eq!(command.get_program(), "/usr/bin/curl");
    let args: Vec<_> = command.get_args().map(|s| s.to_str().unwrap()).collect();
    assert_eq!(args[0], "-q");
    for pair in [["--proto", "=http,https"], ["--proto-redir", "=http,https"], ["--max-redirs", "0"],
        ["--user-agent", "vade-harness"], ["--max-filesize", "123"], ["--max-time", "7.000"], ["--proxy", ""]] {
        assert!(args.windows(2).any(|a| a == pair));
    }
    assert_eq!(command.get_envs().count(), 0);
}

// A Rust child fixture exercises pipes and process cleanup without any socket.
#[test]
fn process_fixture() {
    let Ok(mode) = std::env::var("VADE_WEB_TEST_FIXTURE") else { return; };
    if mode == "wait" { std::thread::sleep(Duration::from_secs(2)); return; }
    let mut stderr = io::stderr().lock();
    let mut stdout = io::stdout().lock();
    if mode == "headers" { let _ = stderr.write_all(&vec![b'h'; HEADER_BYTES * 2]); return; }
    stderr.write_all(b"HTTP/2 200\r\nContent-Type: text/plain\r\n\r\n").unwrap();
    stderr.flush().unwrap();
    let _ = stdout.write_all(&vec![b'x'; 65536]);
}
fn fixture(mode: &str) -> Command {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--exact", "web::tests::process_fixture", "--nocapture"]).env("VADE_WEB_TEST_FIXTURE", mode)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()); c
}
#[test]
fn process_runner_bounds_pipes_and_reaps_on_timeout_or_cancel() {
    let result = run_command(fixture("body"), 1024, Instant::now() + Duration::from_secs(5), &AtomicBool::new(false)).unwrap();
    assert!(result.truncated); assert_eq!(result.body.len(), 1024);
    assert_eq!(run_command(fixture("wait"), 1024, Instant::now() + Duration::from_millis(50), &AtomicBool::new(false)).unwrap_err(), Failure::Timeout);
    assert_eq!(run_command(fixture("wait"), 1024, Instant::now() + Duration::from_secs(5), &AtomicBool::new(true)).unwrap_err(), Failure::Cancelled);
    assert_eq!(run_command(fixture("headers"), 1024, Instant::now() + Duration::from_secs(5), &AtomicBool::new(false)).unwrap_err(), Failure::Fetch("response headers exceed the limit"));
}
