//! CLI fixtures: Rust-only loopback mocks, no installed clients or Engine.
#![cfg(target_os = "macos")]
#[path = "support/cli.rs"]
#[allow(dead_code)]
mod cli;
use std::{
    fs,
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    os::unix::fs::{PermissionsExt, symlink},
    path::PathBuf,
    process::{Command, Output},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use vade_harness::{
    gateway::{self, Cluster, RoutingConfig},
    json,
};

const CAPS: &str = include_str!("fixtures/engine-capabilities-v1.json");
const ASSISTANT: &str = r#"{"role":"assistant","content":"OK","reasoning_content":"synthetic reasoning","opaque":{"retain":"unchanged"}}"#;
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "vade-setup-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_vade-harness"));
    command.env_clear();
    command
}
fn invoke(args: &[&str]) -> Output {
    cli::Cli::start(command().args(args)).finish()
}
fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).unwrap()
}
fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).unwrap()
}

#[test]
fn all_five_setups_print_both_exact_endpoints_offline_without_writes() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!(
        "http://{}/proxy/v1/responses",
        listener.local_addr().unwrap()
    );
    let expected = endpoint.strip_suffix("/responses").unwrap();
    let temp = Temp::new();
    for client in ["codex", "hermes", "opencode", "pi", "generic"] {
        let output = cli::Cli::start(command().current_dir(&temp.0).args([
            "setup",
            client,
            "--endpoint",
            &endpoint,
            "--model",
            "org/mock:Q4_K_M",
            "--gateway",
            "http://127.0.0.1:43210/v1/chat/completions",
        ]))
        .finish();
        assert!(output.status.success(), "{}", stderr(&output));
        let text = stdout(&output);
        assert!(text.contains("(a) Client -> Vade Engine directly"));
        assert!(text.contains("(b) Client -> Vade gateway -> Engine"));
        assert!(text.contains(expected));
        assert!(text.contains("http://127.0.0.1:43210/v1"));
        assert!(text.contains("org/mock:Q4_K_M"));
        assert!(text.contains("Source: https://"));
        assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    }
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
}

#[test]
fn selected_templates_have_documented_provider_settings_and_clean_stdout() {
    for target in ["direct", "gateway"] {
        let base = if target == "direct" {
            "http://127.0.0.1:18080/v1"
        } else {
            "http://127.0.0.1:8787/v1"
        };
        for client in ["codex", "hermes", "opencode", "pi", "generic"] {
            let output = invoke(&[
                "setup",
                client,
                "--endpoint",
                "http://127.0.0.1:18080",
                "--model",
                "org/mock",
                "--target",
                target,
            ]);
            assert!(output.status.success());
            let text = stdout(&output);
            assert!(stderr(&output).contains("Source: https://"));
            assert!(!text.contains("Source:") && !text.contains("```"));
            assert!(text.contains(base));
            match client {
                "codex" => {
                    assert!(text.starts_with("model = \"org/mock\"\nmodel_provider = \"vade\"\n"));
                    for setting in [
                        "[model_providers.vade]",
                        "wire_api = \"responses\"",
                        "requires_openai_auth = false",
                        "supports_websockets = false",
                    ] {
                        assert!(text.contains(setting));
                    }
                    assert!(!text.contains("env_key") && !text.contains("api_key"));
                }
                "hermes" => {
                    assert!(text.starts_with("model:\n  default: \"org/mock\"\n"));
                    assert!(text.contains("provider: custom"));
                    assert!(!text.contains("OPENAI_BASE_URL") && !text.contains("LLM_MODEL"));
                }
                "opencode" => {
                    let v = json::parse(&text).unwrap();
                    assert_eq!(v.get("model").unwrap().as_str(), Some("vade/org/mock"));
                    let provider = v.get("provider").unwrap().get("vade").unwrap();
                    assert_eq!(
                        provider.get("npm").unwrap().as_str(),
                        Some("@ai-sdk/openai-compatible")
                    );
                    assert_eq!(
                        provider
                            .get("options")
                            .unwrap()
                            .get("baseURL")
                            .unwrap()
                            .as_str(),
                        Some(base)
                    );
                    assert!(provider.get("models").unwrap().get("org/mock").is_some());
                }
                "pi" => {
                    let v = json::parse(&text).unwrap();
                    let provider = v.get("providers").unwrap().get("vade").unwrap();
                    assert_eq!(
                        provider.get("api").unwrap().as_str(),
                        Some("openai-completions")
                    );
                    assert_eq!(provider.get("baseUrl").unwrap().as_str(), Some(base));
                    let models = provider.get("models").unwrap().as_array().unwrap();
                    assert_eq!(models[0].get("id").unwrap().as_str(), Some("org/mock"));
                    assert!(stderr(&output).contains("pi --provider vade --model 'org/mock'"));
                }
                _ => {
                    assert!(text.starts_with("curl -q --fail-with-body --max-time 60 "));
                    assert!(text.contains(&format!("'{base}/chat/completions'")));
                    let payload = text
                        .split("--data-binary '")
                        .nth(1)
                        .unwrap()
                        .trim_end()
                        .strip_suffix('\'')
                        .unwrap();
                    let value = json::parse(payload).unwrap();
                    assert_eq!(value.get("model").unwrap().as_str(), Some("org/mock"));
                }
            }
        }
    }
}

#[test]
fn writes_only_explicit_new_private_file_and_refuses_overwrite_or_symlink() {
    let temp = Temp::new();
    for client in ["codex", "hermes", "opencode", "pi", "generic"] {
        let path = temp.0.join(client);
        let args = [
            "setup",
            client,
            "--endpoint",
            "http://example.invalid",
            "--model",
            "mock",
            "--target",
            "gateway",
        ];
        let expected = invoke(&args);
        let mut command = command();
        command.args(args).args(["--write", path.to_str().unwrap()]);
        let output = cli::Cli::start(&mut command).finish();
        assert!(output.status.success(), "{}", stderr(&output));
        assert!(output.stdout.is_empty());
        assert_eq!(fs::read(&path).unwrap(), expected.stdout);
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let before = fs::read(&path).unwrap();
        let refused = cli::Cli::start(&mut command).finish();
        assert!(!refused.status.success());
        assert_eq!(fs::read(&path).unwrap(), before);
    }
    let original = temp.0.join("original");
    fs::write(&original, "sentinel").unwrap();
    let link = temp.0.join("link");
    symlink(&original, &link).unwrap();
    let output = invoke(&[
        "setup",
        "pi",
        "--endpoint",
        "http://example.invalid",
        "--model",
        "mock",
        "--target",
        "direct",
        "--write",
        link.to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    assert_eq!(fs::read_to_string(original).unwrap(), "sentinel");
    assert!(!stderr(&output).contains(temp.0.to_str().unwrap()));
    let missing_parent = temp.0.join("missing/config");
    let output = invoke(&[
        "setup",
        "pi",
        "--endpoint",
        "http://example.invalid",
        "--model",
        "mock",
        "--target",
        "direct",
        "--write",
        missing_parent.to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    assert!(!temp.0.join("missing").exists());
}

#[test]
fn dedicated_homes_create_client_files_and_refuse_existing_state() {
    let temp = Temp::new();
    let parent = fs::canonicalize(&temp.0).unwrap();
    const OPTIONS: &str = "--endpoint http://example.invalid --model mock --target gateway --dedicated-home --context-window 32768 --max-output-tokens 1024 --write";
    for client in ["codex", "hermes", "opencode", "pi"] {
        let home = parent.join(client);
        let mut cmd = command();
        cmd.args(["setup", client])
            .args(OPTIONS.split_whitespace())
            .arg(&home);
        let output = cli::Cli::start(&mut cmd).finish();
        assert!(output.status.success(), "{}", stderr(&output));
        let instructions = if client == "hermes" {
            "SOUL.md"
        } else {
            "AGENTS.md"
        };
        assert!(
            fs::read_to_string(home.join(instructions))
                .unwrap()
                .contains("call the tool immediately")
        );
        let launcher = fs::read_to_string(home.join("launch.sh")).unwrap();
        assert!(launcher.contains("env -i") && launcher.contains("HOME="));
        assert!(!cli::Cli::start(&mut cmd).finish().status.success());
    }
}

#[test]
fn invalid_setup_and_doctor_inputs_fail_without_echoing_secrets_or_connecting() {
    let temp = Temp::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    for extra in [
        vec!["--model", "${PRIVATE_VALUE}"],
        vec!["--model", "!private-command"],
        vec!["--model", "{file:private}"],
        vec!["--model", "mock", "--write", "config"],
        vec!["--model", "mock", "--unknown", "private"],
        vec!["--model", "mock", "--model", "mock"],
    ] {
        let mut command = command();
        command
            .current_dir(&temp.0)
            .args(["setup", "pi", "--endpoint", &base])
            .args(extra);
        let output = cli::Cli::start(&mut command).finish();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(!stderr(&output).contains("PRIVATE_VALUE"));
        assert!(!stderr(&output).contains("private-command"));
    }
    for url in [
        "https://example.invalid",
        "http://user:private@example.invalid",
        "http://example.invalid/v1?api_key=private",
        "http://example.invalid/a#private",
        "http://example.invalid/${PRIVATE_VALUE}/v1",
    ] {
        for mode in ["setup", "doctor"] {
            let mut args = vec![mode];
            if mode == "setup" {
                args.extend(["codex", "--endpoint"]);
            }
            args.push(url);
            args.extend(["--model", "mock"]);
            let output = invoke(&args);
            assert_eq!(output.status.code(), Some(2));
            assert!(!stderr(&output).contains("private"));
        }
    }
    for seconds in ["0", "301", "no"] {
        let output = invoke(&["doctor", &base, "--max-seconds", seconds]);
        assert_eq!(output.status.code(), Some(2));
    }
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    let oversized = format!("http://{}", "a".repeat(8180));
    let output = invoke(&["doctor", &oversized]);
    assert_eq!(output.status.code(), Some(2));
}

#[derive(Clone)]
struct Request {
    path: String,
    body: String,
}
struct Reply {
    gateway: bool,
    status: u16,
    content_type: &'static str,
    body: String,
    truncate: bool,
    delay: Duration,
}
impl Reply {
    fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            gateway: false,
            status,
            content_type: "application/json",
            body: body.into(),
            truncate: false,
            delay: Duration::ZERO,
        }
    }
    fn sse(body: impl Into<String>) -> Self {
        Self {
            content_type: "text/event-stream",
            ..Self::json(200, body)
        }
    }
}
struct Mock {
    url: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn start(respond: impl Fn(&Request) -> Reply + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let running = stop.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let worker = thread::spawn(move || {
            while !running.load(Ordering::Relaxed) {
                let (socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("fixture accept: {e}"),
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut reader = BufReader::new(socket);
                let request = match read_request(&mut reader) {
                    Ok(Some(r)) => r,
                    Ok(None) => continue,
                    Err(e) => panic!("fixture request: {e}"),
                };
                seen.lock().unwrap().push(request.clone());
                let reply = respond(&request);
                thread::sleep(reply.delay);
                let length = reply.body.len() + if reply.truncate { 50 } else { 0 };
                let _ = write!(
                    reader.get_mut(),
                    "HTTP/1.1 {} Mock\r\n{}Content-Type: {}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n{}",
                    reply.status,
                    if reply.gateway {
                        "X-Vade-Gateway: v1\r\n"
                    } else {
                        ""
                    },
                    reply.content_type,
                    reply.body
                );
            }
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }
    fn seen(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let result = self.worker.take().unwrap().join();
        if !thread::panicking() {
            result.unwrap();
        }
    }
}
fn read_request(reader: &mut BufReader<TcpStream>) -> io::Result<Option<Request>> {
    let mut first = String::new();
    if reader.read_line(&mut first)? == 0 {
        return Ok(None);
    }
    let mut length = 0;
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line)? > 0);
        if line == "\r\n" {
            break;
        }
        let (key, value) = line.split_once(':').unwrap();
        assert!(
            !key.eq_ignore_ascii_case("authorization"),
            "doctor never scavenges credentials"
        );
        if key.eq_ignore_ascii_case("content-length") {
            length = value.trim().parse::<usize>().unwrap();
        }
    }
    assert!(length < 256 * 1024);
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request {
        path: first.split_whitespace().nth(1).unwrap().into(),
        body: String::from_utf8(body).unwrap(),
    }))
}
fn second(request: &Request) -> bool {
    json::view(&request.body)
        .ok()
        .and_then(|v| {
            v.get("messages")
                .or_else(|| v.get("request").and_then(|r| r.get("messages")))
        })
        .and_then(|v| v.array())
        .is_some_and(|a| a.count() == 4)
}
fn usage(request: &Request, cached: Option<u64>) -> String {
    let prompt = if second(request) { 120 } else { 100 };
    let details = cached.map_or(String::new(), |n| {
        format!(",\"prompt_tokens_details\":{{\"cached_tokens\":{n}}}")
    });
    format!("{{\"prompt_tokens\":{prompt},\"completion_tokens\":2{details}}}")
}
fn answer(request: &Request, cached: Option<u64>) -> Reply {
    Reply::json(
        200,
        format!(
            "{{\"model\":\"mock\",\"choices\":[{{\"index\":0,\"message\":{ASSISTANT},\"finish_reason\":\"stop\"}}],\"usage\":{}}}",
            usage(request, cached)
        ),
    )
}
fn normal(request: &Request) -> Reply {
    match request.path.as_str() {
        "/_vade/gateway/stats" => Reply::json(400, "private-direct-route-error"),
        "/v1/capabilities" => Reply::json(
            200,
            CAPS.replace(
                "\"prefix_cache\": true,",
                "\"prefix_cache\": true, \"cache_granularity_tokens\": 32,",
            ),
        ),
        "/v1/admission" => {
            let v = json::view(&request.body).unwrap();
            assert_eq!(
                v.get("contract_version").unwrap().as_str().as_deref(),
                Some("vade.engine.v1")
            );
            let output = v.get("output_tokens").unwrap().raw();
            let prompt = if second(request) { 120 } else { 100 };
            Reply::json(
                200,
                format!(
                    "{{\"object\":\"engine.admission\",\"contract_version\":\"vade.engine.v1\",\"model\":\"mock\",\"prompt_tokens\":{prompt},\"output_tokens\":{output},\"context_limit\":4096,\"input_limit\":4000,\"output_limit\":32,\"admitted\":true,\"admission_kind\":\"context_fit\",\"reserves_resources\":false}}"
                ),
            )
        }
        "/v1/chat/completions" => answer(request, Some(if second(request) { 80 } else { 0 })),
        _ => panic!("unexpected mock path {}", request.path),
    }
}
fn doctor(mock: &Mock, extra: &[&str]) -> Output {
    let mut args = vec!["doctor", &mock.url, "--max-seconds", "5"];
    args.extend_from_slice(extra);
    invoke(&args)
}

#[test]
fn doctor_reports_contract_identity_admission_and_actual_reuse_with_full_prefix() {
    let mock = Mock::start(normal);
    let output = doctor(&mock, &[]);
    assert!(
        output.status.success(),
        "{}\n{}",
        stdout(&output),
        stderr(&output)
    );
    let text = stdout(&output);
    for line in [
        "Contract: vade.engine.v1 (recognized)",
        "Model: \"mock\" (advertised)",
        "Context limit: 4096",
        "weights=\"sha256:mock-weights\"",
        "Tools: yes (advertised); thinking: yes (advertised)",
        "Admission: PASS",
        "cached_tokens=80",
        "reuse=66.67%",
        "Gateway cluster: not probed",
        "Two-turn cache check: PASS",
    ] {
        assert!(text.contains(line), "{text}");
    }
    assert!(!text.contains("synthetic reasoning") && !text.contains("amber birch"));
    let seen = mock.seen();
    assert!(seen.iter().all(|r| r.path != "/_vade/gateway/stats"));
    let posts: Vec<_> = seen
        .iter()
        .filter(|r| r.path.ends_with("/chat/completions"))
        .collect();
    assert_eq!(posts.len(), 2);
    let admissions: Vec<_> = seen
        .iter()
        .filter(|r| r.path.ends_with("/admission"))
        .collect();
    assert_eq!(admissions.len(), 3);
    assert_eq!(
        json::view(&admissions[0].body)
            .unwrap()
            .get("request")
            .unwrap()
            .get("messages")
            .unwrap()
            .raw(),
        json::view(&posts[0].body)
            .unwrap()
            .get("messages")
            .unwrap()
            .raw()
    );
    for (post, admission) in posts.iter().zip(admissions.into_iter().skip(1)) {
        let v = json::view(&admission.body).unwrap();
        assert_eq!(v.get("request").unwrap().raw(), post.body);
    }
    assert!(
        posts[1]
            .body
            .starts_with(posts[0].body.strip_suffix("]}").unwrap())
    );
    let first = json::view(&posts[0].body).unwrap();
    assert_eq!(first.get("cache").unwrap().raw(), "true");
    let next = json::view(&posts[1].body).unwrap();
    assert_eq!(
        first.get("prompt_cache_key").unwrap().raw(),
        next.get("prompt_cache_key").unwrap().raw()
    );
    assert_eq!(
        next.get("messages")
            .unwrap()
            .array()
            .unwrap()
            .nth(2)
            .unwrap()
            .raw(),
        ASSISTANT
    );
    assert!(first.get("tools").is_none());
    assert_eq!(
        seen.iter().filter(|r| r.path == "/v1/capabilities").count(),
        3
    );
}

#[test]
fn doctor_distinguishes_zero_missing_and_impossible_cache_counts() {
    for (cached, expected) in [
        (Some(0), "NO HIT"),
        (None, "UNKNOWN"),
        (Some(999), "UNKNOWN"),
    ] {
        let mock = Mock::start(move |r| {
            if r.path == "/v1/chat/completions" && second(r) {
                answer(r, cached)
            } else {
                normal(r)
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(
            stdout(&output).contains(&format!("Two-turn cache check: {expected}")),
            "{}",
            stdout(&output)
        );
        assert!(!stdout(&output).contains("Two-turn cache check: PASS"));
    }
}

#[test]
fn doctor_never_reports_cache_hit_from_truncated_or_http_error_response() {
    for http_error in [false, true] {
        let mock = Mock::start(move |r| {
            if r.path == "/v1/chat/completions" && second(r) {
                if http_error {
                    Reply::json(503, "private-error-body")
                } else {
                    Reply {
                        truncate: true,
                        ..normal(r)
                    }
                }
            } else {
                normal(r)
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(stdout(&output).contains("Two-turn cache check: INCOMPLETE"));
        assert!(
            !stdout(&output).contains("cached_tokens=80")
                && !stdout(&output).contains("private-error-body")
        );
        assert_eq!(
            mock.seen()
                .iter()
                .filter(|r| r.path == "/v1/chat/completions")
                .count(),
            2,
            "no retry"
        );
    }
}

#[test]
fn admission_refusal_and_invalid_admission_stop_generation() {
    for invalid in [false, true] {
        let mock = Mock::start(move |r| {
            if r.path == "/v1/admission" {
                let mut response = normal(r);
                response.body = if invalid {
                    response
                        .body
                        .replace("\"output_tokens\":32", "\"output_tokens\":31")
                } else {
                    response
                        .body
                        .replace("\"admitted\":true", "\"admitted\":false")
                };
                response
            } else {
                normal(r)
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(stdout(&output).contains(if invalid {
            "sizing admission: FAILED or REFUSED"
        } else {
            "sizing admission: FAILED or REFUSED"
        }));
        assert!(mock.seen().iter().all(|r| r.path != "/v1/chat/completions"));
    }
}

#[test]
fn discovery_expectation_survives_first_native_refresh_without_silent_downgrade() {
    for change in 0..3 {
        let probes = AtomicUsize::new(0);
        let mock = Mock::start(move |r| {
            if r.path == "/v1/capabilities" && probes.fetch_add(1, Ordering::Relaxed) > 0 {
                match change {
                    0 => Reply::json(404, "{}"),
                    1 => Reply::json(200, CAPS.replace("vade.engine.v1", "foreign.v2")),
                    _ => Reply::json(
                        200,
                        CAPS.replace(
                            "\"admission_kind\": \"context_fit\"",
                            "\"admission_kind\": null",
                        ),
                    ),
                }
            } else {
                normal(r)
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1), "{}", stdout(&output));
        assert!(stdout(&output).contains("Two-turn cache check: INCOMPLETE"));
        assert!(
            mock.seen().iter().all(|r| r.path != "/v1/chat/completions"),
            "no generation after discovery downgrade (one initial sizing admission is allowed)"
        );
    }
}

#[test]
fn usage_must_match_admission_and_output_reservation() {
    for replacement in [
        "\"prompt_tokens\":119,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":80}",
        "\"prompt_tokens\":120,\"completion_tokens\":33,\"prompt_tokens_details\":{\"cached_tokens\":80}",
        "\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":121}",
    ] {
        let mock = Mock::start(move |r| {
            let mut reply = normal(r);
            if r.path == "/v1/chat/completions" && second(r) {
                reply.body = reply
                    .body
                    .replace(&usage(r, Some(80)), &format!("{{{replacement}}}"));
            }
            reply
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(stdout(&output).contains("Two-turn cache check: UNKNOWN"));
    }
}

#[test]
fn malformed_oversized_or_truncated_metadata_never_authorizes_generation() {
    for mode in 0..3 {
        let mock = Mock::start(move |r| {
            if r.path == "/v1/capabilities" {
                match mode {
                    0 => Reply::json(200, "{not-json-private-sentinel"),
                    1 => Reply::json(200, format!("{{\"padding\":\"{}\"}}", "x".repeat(65536))),
                    _ => Reply {
                        truncate: true,
                        ..normal(r)
                    },
                }
            } else {
                normal(r)
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(stdout(&output).contains("Contract: FAILED"));
        assert!(!stdout(&output).contains("private-sentinel"));
        assert!(mock.seen().iter().all(|r| r.body.is_empty()));
    }
}

#[test]
fn unknown_contract_requires_explicit_model_and_never_sends_admission_or_hint() {
    for unsupported in [false, true] {
        let mock = Mock::start(move |r| {
            if r.path == "/v1/capabilities" {
                if unsupported {
                    Reply::json(404, "{}")
                } else {
                    Reply::json(200, CAPS.replace("vade.engine.v1", "foreign.v2"))
                }
            } else {
                normal(r)
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(stdout(&output).contains("supply --model NAME"));
        assert!(mock.seen().iter().all(|r| r.body.is_empty()));
        let output = doctor(&mock, &["--model", "mock"]);
        assert!(output.status.success(), "{}", stdout(&output));
        assert!(stdout(&output).contains("context limit: unknown"));
        for r in mock.seen().iter().filter(|r| !r.body.is_empty()) {
            assert_eq!(r.path, "/v1/chat/completions");
            assert!(!r.body.contains("prompt_cache_key"));
        }
    }
}

#[test]
fn malformed_metadata_model_mismatch_and_unsupported_dialect_fail_closed() {
    for mode in 0..4 {
        let mock = Mock::start(move |r| {
            if r.path == "/v1/capabilities" {
                match mode {
                    0 => Reply::json(
                        200,
                        CAPS.replace("\"context_limit\": 4096", "\"context_limit\": 0"),
                    ),
                    1 => normal(r),
                    2 => Reply::json(200, CAPS.replace("\"/v1/chat/completions\", ", "")),
                    _ => Reply::json(409, "private-cluster-ambiguity"),
                }
            } else {
                normal(r)
            }
        });
        let extra = if mode == 1 {
            vec!["--model", "different-model"]
        } else {
            vec![]
        };
        let output = doctor(&mock, &extra);
        assert_eq!(output.status.code(), Some(1));
        assert!(stdout(&output).contains("Two-turn cache check: INCOMPLETE"));
        assert!(!stdout(&output).contains("private-cluster-ambiguity"));
        assert!(mock.seen().iter().all(|r| r.body.is_empty()));
    }
}

#[test]
fn null_capability_flags_stay_unknown_and_length_finish_is_labeled() {
    let mock = Mock::start(|r| {
        let mut response = normal(r);
        if r.path == "/v1/capabilities" {
            response.body = response
                .body
                .replace("\"tools\": true", "\"tools\": null")
                .replace("\"thinking\": true", "\"thinking\": null");
        }
        if r.path == "/v1/chat/completions" {
            response.body = response
                .body
                .replace("\"finish_reason\":\"stop\"", "\"finish_reason\":\"length\"");
        }
        response
    });
    let output = doctor(&mock, &[]);
    assert!(output.status.success(), "{}", stdout(&output));
    assert!(stdout(&output).contains("Tools: unknown; thinking: unknown"));
    assert!(stdout(&output).contains("complete (output token limit)"));
}

#[test]
fn sse_usage_is_validated_and_conflicts_cannot_become_a_hit() {
    for conflict in [false, true] {
        let mock = Mock::start(move |r| {
            if r.path == "/v1/chat/completions" {
                let telemetry = usage(r, Some(if second(r) { 80 } else { 0 }));
                let bad = if conflict && second(r) {
                    format!(
                        "data: {{\"choices\":[],\"usage\":{}}}\n\n",
                        usage(r, Some(79))
                    )
                } else {
                    String::new()
                };
                Reply::sse(format!(
                    "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"OK\"}},\"finish_reason\":\"stop\"}}]}}\n\ndata: {{\"choices\":[],\"usage\":{telemetry}}}\n\n{bad}data: [DONE]\n\n"
                ))
            } else {
                normal(r)
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.success(), !conflict, "{}", stdout(&output));
        assert!(stdout(&output).contains(if conflict {
            "Two-turn cache check: UNKNOWN"
        } else {
            "Two-turn cache check: PASS"
        }));
    }
}

#[test]
fn unexpected_tools_are_rejected_without_execution_or_second_turn() {
    let mock = Mock::start(|r| {
        if r.path == "/v1/chat/completions" {
            Reply::json(
                200,
                r#"{"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_mock","type":"function","function":{"name":"shell","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":80}}}"#,
            )
        } else {
            normal(r)
        }
    });
    let output = doctor(&mock, &[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stdout(&output).contains("Two-turn cache check: INCOMPLETE"));
    assert_eq!(
        mock.seen()
            .iter()
            .filter(|r| r.path == "/v1/chat/completions")
            .count(),
        1
    );
}

#[test]
fn deadline_stops_slow_probe_without_retries() {
    let mock = Mock::start(|r| {
        if r.path == "/v1/chat/completions" {
            Reply {
                delay: Duration::from_millis(1500),
                ..normal(r)
            }
        } else {
            normal(r)
        }
    });
    let start = Instant::now();
    let output = invoke(&["doctor", &mock.url, "--max-seconds", "1"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(stdout(&output).contains("deadline reached"));
    assert_eq!(
        mock.seen()
            .iter()
            .filter(|r| r.path == "/v1/chat/completions")
            .count(),
        1
    );
}

fn stats(healthy: bool) -> String {
    format!(
        "{{\"object\":\"vade.gateway.stats\",\"endpoints\":[{{\"endpoint\":0,\"group\":0,\"healthy\":{healthy},\"cooling_down\":false,\"in_flight\":0,\"requests\":99,\"failures\":0,\"prompt_tokens\":10000,\"cached_tokens\":9900}}]}}"
    )
}
#[test]
fn cluster_health_and_aggregate_cache_do_not_substitute_for_two_turn_check() {
    for healthy in [false, true] {
        let mock = Mock::start(move |r| {
            if r.path == "/_vade/gateway/stats" {
                Reply::json(200, stats(healthy))
            } else if healthy && r.path == "/v1/chat/completions" {
                answer(r, None)
            } else {
                let mut reply = normal(r);
                reply.gateway = r.path == "/v1/capabilities";
                reply
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(stdout(&output).contains(if healthy {
            "Two-turn cache check: UNKNOWN"
        } else {
            "Gateway cluster: 1 endpoints; DEGRADED"
        }));
        assert!(stdout(&output).contains("cumulative, not the two-turn check"));
    }
}

struct RunningGateway {
    url: String,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<io::Result<()>>>,
}
impl RunningGateway {
    fn start(mocks: &[Mock]) -> Self {
        let cluster = Cluster::new(
            &mocks.iter().map(|m| m.url.clone()).collect::<Vec<_>>(),
            RoutingConfig::default(),
        )
        .unwrap();
        // The public server API accepts an address, not a pre-bound listener.
        // Other parallel fixtures can take the released ephemeral reservation.
        // Retry only a confirmed bind collision, before any server dispatch.
        for _ in 0..8 {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            drop(listener);
            let stop = Arc::new(AtomicBool::new(false));
            let running = stop.clone();
            let owner = cluster.clone();
            let worker = thread::spawn(move || {
                gateway::run_cluster_until(&address.to_string(), owner, &running)
            });
            let mut result = Self {
                url: format!("http://{address}"),
                stop,
                worker: Some(worker),
            };
            let deadline = Instant::now() + Duration::from_secs(3);
            while !cluster.stats().endpoints.iter().all(|e| e.healthy) {
                if result.worker.as_ref().unwrap().is_finished() {
                    let error = result.worker.take().unwrap().join().unwrap().unwrap_err();
                    assert_eq!(
                        error.kind(),
                        io::ErrorKind::AddrInUse,
                        "gateway fixture startup"
                    );
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "mock cluster discovery deadline: {:?}",
                    cluster.stats()
                );
                thread::sleep(Duration::from_millis(5));
            }
            if result.worker.is_some() {
                return result;
            }
        }
        panic!("unable to reserve a loopback fixture port");
    }
}
impl Drop for RunningGateway {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                result.unwrap().unwrap();
            }
        }
    }
}
#[test]
fn doctor_runs_through_real_gateway_with_mock_upstreams_and_reports_ambiguity() {
    for ambiguous in [false, true] {
        let mocks = [
            Mock::start(normal),
            Mock::start(move |r| {
                let mut response = normal(r);
                if ambiguous && r.path == "/v1/capabilities" {
                    response.body = response.body.replace("mock-weights", "different-weights");
                }
                response
            }),
        ];
        let gateway = RunningGateway::start(&mocks);
        let output = invoke(&["doctor", &gateway.url, "--max-seconds", "5"]);
        assert_eq!(
            output.status.success(),
            !ambiguous,
            "{}\n{}",
            stdout(&output),
            stderr(&output)
        );
        assert!(stdout(&output).contains("Gateway cluster: 2 endpoints; healthy"));
        if ambiguous {
            assert!(stdout(&output).contains("Gateway discovery is ambiguous"));
            assert!(
                mocks
                    .iter()
                    .all(|m| m.seen().iter().all(|r| r.body.is_empty()))
            );
        } else {
            assert!(stdout(&output).contains("Two-turn cache check: PASS"));
            assert_eq!(
                mocks
                    .iter()
                    .map(|m| m
                        .seen()
                        .iter()
                        .filter(|r| r.path == "/v1/chat/completions")
                        .count())
                    .sum::<usize>(),
                2
            );
        }
    }
}

fn large_probe_count(request: &Request) -> u64 {
    let v = json::view(&request.body).unwrap();
    let v = v.get("request").unwrap_or(v);
    // Deliberately fewer tokens than words to force exact-count sizing growth.
    let words = v
        .get("messages")
        .unwrap()
        .array()
        .unwrap()
        .nth(1)
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap()
        .split_whitespace()
        .count();
    words as u64 / 2 + if second(request) { 100 } else { 80 }
}

#[test]
fn doctor_sizes_above_hybrid_cache_tiles_and_fallback_with_exact_admission() {
    for advertised in [false, true] {
        let caps = CAPS
            .replace("\"context_limit\": 4096", "\"context_limit\": 65536")
            .replace("\"input_limit\": 4000", "\"input_limit\": 65504");
        let caps = if advertised {
            caps.replacen('{', "{\"cache_granularity_tokens\":4096,", 1)
        } else {
            caps
        };
        let mock = Mock::start(move |r| match r.path.as_str() {
            "/v1/capabilities" => Reply::json(200, &caps),
            "/v1/admission" => {
                let count = large_probe_count(r);
                Reply::json(
                    200,
                    format!(
                        r#"{{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","prompt_tokens":{count},"output_tokens":32,"context_limit":65536,"input_limit":65504,"output_limit":32,"admitted":true,"admission_kind":"context_fit","reserves_resources":false}}"#
                    ),
                )
            }
            "/v1/chat/completions" => {
                let count = large_probe_count(r);
                assert!(
                    count > 8192,
                    "undersized generation must never be dispatched"
                );
                let cached = if second(r) { 8192 } else { 0 };
                Reply::json(
                    200,
                    format!(
                        r#"{{"choices":[{{"message":{{"role":"assistant","content":"OK"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{count},"completion_tokens":2,"prompt_tokens_details":{{"cached_tokens":{cached}}}}}}}"#
                    ),
                )
            }
            _ => panic!("direct doctor attempted a gateway route"),
        });
        let output = doctor(&mock, &[]);
        assert!(output.status.success(), "{}", stdout(&output));
        let text = stdout(&output);
        assert!(text.contains("target at least 8193 rendered prompt tokens"));
        assert!(text.contains("3 sizing check(s), no inference"));
        assert!(text.contains(if advertised {
            "above two advertised cache blocks"
        } else {
            "fallback above 8192"
        }));
        assert!(text.contains("reuse=") && text.contains('%'));
        let seen = mock.seen();
        assert_eq!(
            seen.iter()
                .filter(|r| r.path.ends_with("/admission"))
                .count(),
            5
        );
        let posts: Vec<_> = seen
            .iter()
            .filter(|r| r.path.ends_with("/chat/completions"))
            .collect();
        assert_eq!(posts.len(), 2);
        assert!(
            posts[1]
                .body
                .starts_with(posts[0].body.strip_suffix("]}").unwrap())
        );
        assert!(!text.contains("amber") && !text.contains(&mock.url));
    }
}

#[test]
fn doctor_cannot_claim_cache_failure_from_an_unfit_or_unverified_small_probe() {
    for granularity in [None, Some(4096), Some(u64::MAX)] {
        let caps = granularity.map_or_else(
            || CAPS.to_owned(),
            |g| CAPS.replacen('{', &format!("{{\"cache_granularity_tokens\":{g},"), 1),
        );
        let mock = Mock::start(move |r| {
            if r.path == "/v1/capabilities" {
                Reply::json(200, &caps)
            } else {
                panic!("unfit probe must not dispatch admission or generation")
            }
        });
        let output = doctor(&mock, &[]);
        assert_eq!(output.status.code(), Some(1));
        assert!(stdout(&output).contains("diagnostic bounds"));
        assert!(stdout(&output).contains("Two-turn cache check: INCOMPLETE"));
    }
    let mock = Mock::start(|r| {
        if r.path == "/v1/capabilities" {
            Reply::json(404, "{}")
        } else if r.path == "/v1/chat/completions" {
            answer(r, Some(0))
        } else {
            panic!("unknown direct endpoint must receive standard chat only")
        }
    });
    let output = doctor(&mock, &["--model", "mock"]);
    assert!(stdout(&output).contains("INCONCLUSIVE (reported probe size below target)"));
    assert!(!stdout(&output).contains("NO HIT"));
}
