//! The quality card against a first-party loopback SSE fixture: request
//! shapes, token-id comparison, draft receipt matching and card output.
//! The CLI test runs the real binary so the shared credential path in
//! `main` (`auth::configure` before dispatch) is covered, not only `run()`.
#[path = "support/home.rs"]
mod home;
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use vade_harness::quality::{Config, parse_endpoint, run, send};

const ORDER_ANSWER: &str =
    "{\\\"order_id\\\":\\\"A-1042\\\",\\\"items\\\":3,\\\"total_cents\\\":4599,\\\"priority\\\":\\\"high\\\",\\\"paid\\\":false}";

#[derive(Clone, Copy, Default)]
struct Mock {
    /// Plain arms emit a different second token id.
    differ: bool,
    /// Chunks carry no `token_id` field at all.
    omit_ids: bool,
    /// Chunks carry fractional token ids (11.1 then 12.9).
    fractional_ids: bool,
    /// The stream never sends the terminal `[DONE]` event.
    omit_done: bool,
    /// The strict arm reports plain drafting back (engine downgrade).
    downgrade_strict: bool,
}

/// Serve `requests` streamed replies. The reported draft mode follows the
/// request draft field (omitted means auto), like the engine echo.
fn serve(
    listener: TcpListener,
    requests: usize,
    mock: Mock,
    seen_auth: Option<Arc<Mutex<Vec<Option<String>>>>>,
) {
    listener.set_nonblocking(true).expect("nonblocking listener");
    for _ in 0..requests {
        // Bounded accept: an early client exit must not park this thread.
        let accept_deadline = Instant::now() + Duration::from_secs(60);
        let (stream, _) = loop {
            match listener.accept() {
                Ok(pair) => break pair,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < accept_deadline, "mock accept timed out");
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => panic!("accept: {error}"),
            }
        };
        stream.set_nonblocking(false).expect("blocking stream");
        let mut reader = BufReader::new(stream);
        let mut length = 0usize;
        let mut auth = None;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("header line");
            if line == "\r\n" || line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    length = value.trim().parse().expect("length");
                }
                if name.eq_ignore_ascii_case("authorization") {
                    auth = Some(value.trim().to_owned());
                }
            }
        }
        if let Some(seen) = &seen_auth {
            seen.lock().expect("auth log").push(auth);
        }
        let mut body = vec![0u8; length];
        reader.read_exact(&mut body).expect("body");
        let body = String::from_utf8(body).expect("utf8 body");
        assert!(body.contains("\"temperature\":0"), "greedy pinned: {body}");
        assert!(body.contains("\"cache\":false"), "cache off: {body}");
        let requested = if body.contains("\"draft\":\"plain\"") {
            "plain"
        } else if body.contains("\"draft\":\"strict\"") {
            "strict"
        } else {
            "auto"
        };
        let reported = if mock.downgrade_strict && requested == "strict" {
            "plain"
        } else {
            requested
        };
        let draft_used = reported != "plain";
        let second = if mock.differ && requested == "plain" { 99 } else { 12 };
        let (tok1, tok2) = if mock.omit_ids {
            (String::new(), String::new())
        } else if mock.fractional_ids {
            (",\"token_id\":11.1".to_owned(), ",\"token_id\":12.9".to_owned())
        } else {
            (",\"token_id\":11".to_owned(), format!(",\"token_id\":{second}"))
        };
        let done = if mock.omit_done { "" } else { "data: [DONE]\r\n\r\n" };
        let content = if body.contains("order note") { ORDER_ANSWER } else { "ok" };
        let first_half = &content[..content.len() / 2];
        let second_half = &content[content.len() / 2..];
        let events = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n\
data: {{\"id\":\"completion-7\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{first_half}\"}},\"finish_reason\":null{tok1}}}]}}\r\n\r\n\
data: {{\"id\":\"completion-7\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{second_half}\"}},\"finish_reason\":null{tok2}}}]}}\r\n\r\n\
data: {{\"id\":\"completion-7\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":20,\"completion_tokens\":2,\"total_tokens\":22,\"prompt_tokens_details\":{{\"cached_tokens\":0}}}},\"vade\":{{\"draft\":\"{reported}\",\"mtp_depth\":3,\"thinking\":false,\"draft_used\":{draft_used},\"model_draft_used\":{draft_used},\"prefill_ms\":12.0}}}}\r\n\r\n\
{done}"
        );
        reader.get_mut().write_all(events.as_bytes()).expect("write events");
        reader.get_mut().flush().expect("flush");
    }
}

/// The mock CLI key, built from short parts at run time so this file
/// carries no literal in the bearer-token shape the release scan flags.
fn mock_api_key() -> String {
    format!("{}-{}-{}-{}-{}", "quality", "card", "test", "key", "0123456789")
}

fn mock_config(endpoint: String, server_log: Option<std::path::PathBuf>) -> Config {
    Config {
        endpoint,
        model: "mock".into(),
        json_path: "unused.json".into(),
        server_log,
        code_run: false,
        request_seconds: 30,
        ..Config::default()
    }
}

fn run_against(mock: Mock) -> vade_harness::quality::Report {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    let server = thread::spawn(move || serve(listener, 36, mock, None));
    let report = run(mock_config(endpoint, None)).expect("run");
    server.join().expect("server thread");
    report
}

#[test]
fn quality_card_runs_against_loopback_mock() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    let server = thread::spawn(move || serve(listener, 36, Mock::default(), None));
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let log_path = std::env::temp_dir().join(format!("vade-quality-mock-{stamp}.log"));
    std::fs::write(
        &log_path,
        "draft id=7 cycles=10 rows=40 accepted=25 proposed=30 max_rows=4 target_ms=120.500 draft_ms=20.250 guards=0/0 host_bytes=1 bank_host_bytes=0\n",
    )
    .expect("write log");
    let report = run(mock_config(endpoint, Some(log_path.clone()))).expect("run");
    let _ = std::fs::remove_file(&log_path);
    server.join().expect("server thread");

    assert_eq!(report.quality.len(), 12);
    assert!(report.quality.iter().all(|item| item.error.is_none()));
    let order = report
        .quality
        .iter()
        .find(|item| item.case.id == "json-order")
        .expect("order case");
    assert!(order.full_pass(), "mock golden JSON passes: {:?}", order.checks);
    assert_eq!(report.exact_identical(), (12, 12));
    assert_eq!(report.verify.cycles, 10);
    assert_eq!(report.verify.verify_ms_per_step(), Some(12.05));
    assert!(report.successful());
    let cards = report.cards();
    assert!(cards.contains("IDS IDENTICAL 12/12"));
    assert!(cards.contains("Verify ms per step 12.050"));
    let receipt = report.to_json().expect("receipt");
    assert!(vade_harness::json::parse(&receipt).is_ok());
}

#[test]
fn quality_card_reports_differing_ids() {
    let report = run_against(Mock { differ: true, ..Mock::default() });

    assert_eq!(report.exact_identical(), (0, 12));
    assert!(!report.successful());
    assert!(
        report
            .exactness
            .iter()
            .all(|item| item.first_diff() == Some(1)),
        "second token differs"
    );
    assert_eq!(report.verify_basis, "no server log supplied");
    let cards = report.cards();
    assert!(cards.contains("Not measured: no server log supplied"), "{cards}");
    // A scored run still finishes inside its per-request deadlines.
    assert!(report.elapsed_s < Duration::from_secs(120).as_secs_f64());
}

#[test]
fn quality_card_fails_without_token_ids() {
    let report = run_against(Mock { omit_ids: true, ..Mock::default() });

    assert_eq!(report.exact_identical(), (0, 12));
    assert!(!report.successful());
    assert!(
        report
            .exactness
            .iter()
            .all(|item| item.failure_reason().as_deref() == Some("no token ids")),
        "every exactness case fails for missing ids"
    );
    let cards = report.cards();
    assert!(cards.contains("no token ids"), "{cards}");
    let receipt = report.to_json().expect("receipt");
    assert!(receipt.contains("no token ids"), "receipt names the failure");
}

#[test]
fn quality_card_fails_when_strict_arm_reports_plain() {
    let report = run_against(Mock { downgrade_strict: true, ..Mock::default() });

    assert_eq!(report.exact_identical(), (0, 12));
    assert!(!report.successful());
    assert!(
        report.exactness.iter().all(|item| item.ids_identical() == Some(true)),
        "ids match; only the draft modes are wrong"
    );
    assert!(
        report.exactness.iter().all(|item| {
            item.failure_reason().is_some_and(|reason| {
                reason.contains("strict arm reported plain (want strict)")
                    && reason.contains("plain arm reported plain (want plain)")
            })
        }),
        "every case prints both reported modes"
    );
    let cards = report.cards();
    assert!(cards.contains("draft mode mismatch"), "{cards}");
    let receipt = report.to_json().expect("receipt");
    assert!(receipt.contains("draft mode mismatch"), "receipt names the failure");
}

#[test]
fn quality_card_fails_on_fractional_token_ids() {
    let report = run_against(Mock { fractional_ids: true, ..Mock::default() });

    assert_eq!(report.exact_identical(), (0, 12));
    assert!(!report.successful());
    assert!(
        report
            .exactness
            .iter()
            .all(|item| item.failure_reason().as_deref() == Some("invalid token id")),
        "fractional ids are invalid evidence on every case"
    );
}

#[test]
fn quality_card_fails_without_the_terminal_event() {
    let report = run_against(Mock { omit_done: true, ..Mock::default() });

    assert_eq!(report.exact_identical(), (0, 12));
    assert!(!report.successful());
    assert!(
        report.exactness.iter().all(|item| item.failure_reason().is_some()),
        "no exactness case passes without the terminal event"
    );
}

#[test]
fn quality_card_cli_sends_the_configured_api_key() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = thread::spawn({
        let seen = seen.clone();
        move || serve(listener, 36, Mock::default(), Some(seen))
    });
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let root = std::env::temp_dir().join(format!("vade-quality-cli-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&root).expect("temp dir");
    let key_path = root.join("key.txt");
    let json_path = root.join("receipt.json");
    let key = mock_api_key();
    std::fs::write(&key_path, format!("{key}\n")).expect("write key");
    let mut child = home::harness()
        .args([
            "quality-card",
            "--endpoint",
            &endpoint,
            "--model",
            "mock",
            "--json",
            json_path.to_string_lossy().as_ref(),
            "--api-key-file",
            key_path.to_string_lossy().as_ref(),
            "--no-code-run",
            "--request-seconds",
            "30",
        ])
        .env_remove("VADE_API_KEY_FILE")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("run binary");
    let child_deadline = Instant::now() + Duration::from_secs(180);
    while child.try_wait().expect("poll binary").is_none() {
        assert!(Instant::now() < child_deadline, "quality-card CLI timed out");
        thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output().expect("collect binary output");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    server.join().expect("server thread");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let seen = seen.lock().expect("auth log");
    assert_eq!(seen.len(), 36);
    let expected_auth = format!("Bearer {key}");
    assert!(
        seen.iter().all(|auth| auth.as_deref() == Some(expected_auth.as_str())),
        "every request carries the configured key: {seen:?}"
    );
    let receipt = std::fs::read_to_string(&json_path).expect("receipt file");
    assert!(!receipt.contains(&key), "key never reaches the receipt");
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert!(stdout.contains("IDS IDENTICAL 12/12"), "{stdout}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn trickling_stream_hits_the_request_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = parse_endpoint(&format!("http://{}", listener.local_addr().expect("addr")))
        .expect("endpoint");
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let mut reader = BufReader::new(stream);
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; length];
        let _ = reader.read_exact(&mut body);
        let stream = reader.get_mut();
        if stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"partial\"")
            .is_err()
        {
            return;
        }
        // One byte at a time, never a newline: only a total deadline ends this.
        loop {
            thread::sleep(Duration::from_millis(50));
            if stream.write_all(b"x").and_then(|()| stream.flush()).is_err() {
                return;
            }
        }
    });
    let started = Instant::now();
    let result = send(&endpoint, "{}", None, Duration::from_millis(400));
    assert!(result.is_err(), "trickling stream must fail");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "deadline bounds the whole request"
    );
    server.join().expect("server thread");
}

#[test]
fn quality_card_keeps_the_run_when_the_server_log_is_missing() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    let server = thread::spawn(move || serve(listener, 36, Mock::default(), None));
    let missing = std::env::temp_dir().join("vade-quality-log-that-does-not-exist.log");
    let report = run(mock_config(endpoint, Some(missing))).expect("run keeps its answers");
    server.join().expect("server thread");
    assert_eq!(report.exact_identical(), (12, 12));
    assert!(report.successful());
    assert_eq!(report.verify.receipts, 0);
    assert!(
        report.verify_basis.contains("server log unreadable"),
        "{}",
        report.verify_basis
    );
}

#[test]
fn quality_card_cli_retry_works_after_a_blocked_card_file() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let root = std::env::temp_dir().join(format!("vade-quality-retry-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&root).expect("temp dir");
    let json_path = root.join("receipt.json");
    let card_path = root.join("card.txt");
    std::fs::write(&card_path, "already here\n").expect("seed card file");
    let blocked = home::harness()
        .args([
            "quality-card",
            "--endpoint",
            "http://127.0.0.1:9",
            "--model",
            "mock",
            "--json",
            json_path.to_string_lossy().as_ref(),
            "--card",
            card_path.to_string_lossy().as_ref(),
        ])
        .env_remove("VADE_API_KEY_FILE")
        .output()
        .expect("run binary");
    assert_eq!(blocked.status.code(), Some(2), "setup refusal exits 2");
    assert!(
        !json_path.exists(),
        "the empty JSON file from the refused setup must not block a retry"
    );
    // The same JSON path now runs to completion against the mock.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    let server = thread::spawn(move || serve(listener, 36, Mock::default(), None));
    let retry_card = root.join("retry-card.txt");
    let output = home::harness()
        .args([
            "quality-card",
            "--endpoint",
            &endpoint,
            "--model",
            "mock",
            "--json",
            json_path.to_string_lossy().as_ref(),
            "--card",
            retry_card.to_string_lossy().as_ref(),
            "--no-code-run",
            "--request-seconds",
            "30",
        ])
        .env_remove("VADE_API_KEY_FILE")
        .output()
        .expect("run binary");
    server.join().expect("server thread");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(json_path.exists(), "retry writes the receipt");
    let _ = std::fs::remove_dir_all(&root);
}
