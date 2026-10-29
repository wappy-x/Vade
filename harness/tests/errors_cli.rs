//! Public process-level diagnostics against bounded, first-party loopback peers.
#[path = "support/home.rs"]
mod home;
#[path = "support/cli.rs"]
mod cli;
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    process::Command,
    sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}},
    thread,
    time::Duration,
};

struct Mock {
    endpoint: String,
    stop: Arc<AtomicBool>,
    posts: Arc<AtomicUsize>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn status(status: u16) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let running = stop.clone();
        let posts = Arc::new(AtomicUsize::new(0));
        let count = posts.clone();
        let worker = thread::spawn(move || {
            while !running.load(Ordering::Relaxed) {
                let (socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("mock accept: {e}"),
                };
                // Poll accepts, then read each complete request with bounded timeouts.
                socket.set_nonblocking(false).unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                socket.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
                let mut reader = BufReader::new(socket);
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut size = 0;
                loop {
                    let mut line = String::new();
                    assert!(reader.read_line(&mut line).unwrap() > 0);
                    if line == "\r\n" { break; }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            size = value.trim().parse::<usize>().unwrap();
                        }
                    }
                }
                assert!(size < 64 * 1024);
                reader.read_exact(&mut vec![0; size]).unwrap();
                let reply_status = if first.starts_with("GET /v1/capabilities ") {
                    404 // Exercise the supported compatibility path.
                } else {
                    assert!(first.starts_with("POST /v1/chat/completions "), "{first}");
                    count.fetch_add(1, Ordering::Relaxed);
                    status
                };
                let body = "{\"error\":\"PRIVATE_TOKEN_123 https://private.invalid/?key=secret \\u001b[2J\"}";
                let reply = format!("HTTP/1.1 {reply_status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = reader.get_mut().write_all(reply.as_bytes());
            }
        });
        Self { endpoint, stop, posts, worker: Some(worker) }
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let joined = self.worker.take().unwrap().join();
        if !thread::panicking() { joined.unwrap(); }
    }
}

#[test]
fn direct_http_failures_offer_specific_recovery_without_body_disclosure_or_retry() {
    let root = std::env::temp_dir().join(format!("vade-error-cli-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (status, category, next) in [
        (401, "HTTP 401", "API_KEY_FILE"),
        (403, "HTTP 403", "API key"),
        (404, "HTTP 404", "--endpoint"),
        (413, "HTTP 413", "Shorten the prompt"),
        (422, "HTTP 422", "output limits"),
        (429, "HTTP 429", "wait for capacity"),
        (503, "HTTP 503", "wait for capacity"),
        (504, "HTTP 504", "timed out"),
    ] {
        let mock = Mock::status(status);
        let mut command = home::harness();
        command.args(["direct", "--state-in-project", "--endpoint", &mock.endpoint, "--model", "mock", "--plain", "--prompt", "hello", "--max-output-tokens", "32", "--workspace"])
            .arg(&root).env_remove("VADE_API_KEY_FILE");
        let mut process = cli::Cli::start(&mut command);
        process.send("");
        process.until(true, next);
        let output = process.finish();
        assert!(!output.status.success());
        let text = String::from_utf8_lossy(&output.stderr);
        assert!(text.contains(category), "status {status}: {text}");
        assert!(text.contains("will not resend"), "{text}");
        for secret in ["PRIVATE_TOKEN_123", "private.invalid", "key=secret", "\x1b"] {
            assert!(!text.contains(secret), "untrusted error leaked: {text}");
            assert!(!String::from_utf8_lossy(&output.stdout).contains(secret));
        }
        assert_eq!(mock.posts.load(Ordering::Relaxed), 1, "failed generation replayed");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn both_gateway_entry_points_explain_an_occupied_port() {
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    for standalone in [false, true] {
        let mut command = home::command(if standalone {
            env!("CARGO_BIN_EXE_vade-gateway")
        } else {
            env!("CARGO_BIN_EXE_vade-harness")
        });
        if !standalone { command.arg("gateway"); }
        command.args(["--bind", &reservation.local_addr().unwrap().to_string(), "--upstream", "http://127.0.0.1:1/v1"])
            .env_remove("VADE_API_KEY_FILE");
        let output = cli::Cli::start(&mut command).finish();
        assert!(!output.status.success());
        let text = String::from_utf8_lossy(&output.stderr);
        assert!(text.contains("--bind 127.0.0.1:PORT"), "{text}");
        assert!(text.contains("gateway you started"), "{text}");
    }
}

#[test]
fn argument_errors_name_options_and_mode_help_without_echoing_values() {
    fn failure(args: &[&str]) -> String {
        let mut command = home::harness();
        command.args(args).env_remove("VADE_API_KEY_FILE")
            .env_remove("VADE_HARNESS_EFFORT");
        let output = cli::Cli::start(&mut command).finish();
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?}");
        let text = String::from_utf8(output.stderr).unwrap();
        assert_eq!(text.lines().count(), 1, "{args:?}: {text}");
        assert!(!text.contains("PRIVATE") && !text.contains('\x1b') && !text.contains('\r')
            && !text.contains('\u{202e}'), "{args:?}: {text}");
        text
    }
    let options = [
        "--mode", "--endpoint", "--model", "--prompt", "--workspace", "--session",
        "--bind", "--upstream", "--hermes-bin", "--external-session", "--codex-bin",
        "--thread", "--permission", "--approval", "--resume", "--context-tokens",
        "--agent-time-limit-seconds", "--config", "--rules-scope", "--shell-exclude",
        "--shell-toolchain", "--child-edit", "--child-shell", "--reply-tokens",
        "--max-output-tokens", "--thinking", "--effort", "--check", "--temperature", "--seed",
    ];
    for option in options {
        for args in [vec!["agent", option], vec!["agent", option, ""]] {
            let text = failure(&args);
            assert!(text.starts_with(&format!("vade: {option}: ")), "{text}");
            assert!(text.contains("See vade-harness agent --help."), "{text}");
        }
    }
    for mode in ["direct", "agent", "gateway"] {
        let text = failure(&[mode, "--unknown=PRIVATE_VALUE"]);
        assert!(text.contains("--unknown: unknown argument"), "{text}");
        assert!(text.contains(&format!("See vade-harness {mode} --help.")), "{text}");
    }
    for option in ["--seed", "--temperature", "--thinking", "--permission",
        "--approval", "--max-output-tokens", "--reply-tokens", "--context-tokens",
        "--agent-time-limit-seconds", "--effort"] {
        let text = failure(&["agent", option, "PRIVATE_VALUE\x1b[2J\r\n\u{202e}"]);
        assert!(text.starts_with(&format!("vade: {option}: ")), "{text}");
    }
    for value in ["--bad\x1b[2J", "--bad\r\nPRIVATE", "--bad\u{202e}PRIVATE",
        "PRIVATE_POSITIONAL", "--aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"] {
        let text = failure(&["direct", value]);
        assert!(text.contains("unknown argument"), "{text}");
        assert!(text.len() < 200, "{text}");
    }
    let text = failure(&["direct", "--seed", "713", "--seed", "PRIVATE_VALUE"]);
    assert!(text.contains("--seed: an option was specified more than once"), "{text}");
    // The older flag name is named too, in both errors.
    let text = failure(&["agent", "--no-compact", "--no-auto-compact"]);
    assert!(text.contains("--no-compact or --no-auto-compact was specified more than once"), "{text}");
    let text = failure(&["direct", "--no-auto-compact"]);
    assert!(text.contains("--no-compact and --no-auto-compact require agent mode"), "{text}");
    let text = failure(&["direct", "--greedy", "--temperature", "0.5", "--plain"]);
    assert!(text.starts_with("vade: choose --greedy or --temperature"), "{text}");
    assert!(!text.contains("--plain:"), "cross-option error blamed final flag: {text}");
    let text = failure(&["demo-compaction", "PRIVATE_VALUE"]);
    assert_eq!(text.trim_end(), "vade: demo-compaction takes no arguments; run vade-harness demo-compaction on its own.");
}

#[test]
fn child_shell_is_refused_where_the_shell_is_not_offered() {
    // VADE_HARNESS_SHELL=off gives macOS the Linux shape; on Linux it changes nothing.
    let mut command = home::harness();
    command
        .args(["agent", "--children", "--child-shell", "checks", "--endpoint",
            "http://127.0.0.1:9/v1/chat/completions", "--model", "mock", "--prompt", "x"])
        .env("VADE_HARNESS_SHELL", "off");
    let output = cli::Cli::start(&mut command).finish();
    assert_eq!(output.status.code(), Some(2));
    let text = String::from_utf8(output.stderr).unwrap();
    let expected = if vade_harness::coding_tools::SHELL_AVAILABLE {
        "vade: --child-shell: VADE_HARNESS_SHELL=off hides workspace_shell; unset it or drop --child-shell."
    } else {
        "vade: --child-shell needs the macOS sandbox; workspace_shell is not offered on this platform."
    };
    assert!(text.starts_with(expected), "{text}");
}

#[test]
fn standalone_gateway_argument_errors_keep_option_context_and_hide_values() {
    let mut cases = Vec::new();
    for option in ["--upstream", "--bind", "--record", "--stats"] {
        cases.push((vec![option], option));
        cases.push((vec![option, ""], option));
    }
    cases.push((vec!["replay"], "replay DIR"));
    for option in ["--upstream", "--stats", "--concurrency"] {
        cases.push((vec!["replay", "unused", option], option));
        cases.push((vec!["replay", "unused", option, ""], option));
    }
    cases.extend([
        (vec!["--unknown=PRIVATE_VALUE"], "--unknown"),
        (vec!["--unknown\x1b[2J\nPRIVATE"], "unknown gateway argument"),
        (vec!["--unknown\u{202e}PRIVATE"], "unknown gateway argument"),
        (vec!["replay", "unused", "--unknown=PRIVATE_VALUE"], "--unknown"),
    ]);
    for (args, context) in cases {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade-gateway"));
        command.args(&args).env_remove("VADE_GATEWAY_UPSTREAM")
            .env_remove("VADE_API_KEY_FILE");
        let output = cli::Cli::start(&mut command).finish();
        assert_eq!(output.status.code(), Some(1), "{args:?}");
        assert!(output.stdout.is_empty());
        let text = String::from_utf8(output.stderr).unwrap();
        assert!(text.contains(context) && text.contains("vade-gateway --help"), "{text}");
        assert_eq!(text.lines().count(), 1, "{text}");
        assert!(!text.contains("PRIVATE") && !text.contains('\x1b') && !text.contains('\u{202e}'), "{text}");
    }
}
