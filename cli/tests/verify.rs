//! Exercise the public exactness command against local HTTP mocks, without a GPU.
#[allow(dead_code)]
#[path = "../src/evals/json.rs"]
mod json;
#[allow(dead_code)]
#[path = "../src/json.rs"]
mod strict_json;

use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Command, Output},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "vade-verify-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn base_command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vade"));
        command
            .current_dir(&self.0)
            .args(["verify", "--max-tokens", "3"])
            .env("VADE_CONFIG", self.0.join("missing-config"))
            .env("VADE_HOME", self.0.join("unused-home"))
            .env_remove("VADE_API_KEY_FILE")
            .env_remove("VADE_API_URL");
        command
    }

    fn command(&self, url: &str) -> Command {
        let mut command = self.base_command();
        command.args(["--url", url, "--timeout", "3"]);
        command
    }

    fn config(&self, second_url: &str) -> PathBuf {
        let path = self.0.join("config");
        fs::write(
            &path,
            format!(
                "[global]\nstate_dir = state\nmodel = mock-model\nmodel_revision = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\
                 [machine second]\nssh = unused-fixture-host\n\
                 model_path = /unused-fixture-model\nverify_url = {second_url}\n"
            ),
        )
        .unwrap();
        path
    }

    fn owner(&self, root: &std::path::Path, url: &str) {
        fs::create_dir_all(root.join("state")).unwrap();
        let port = url.rsplit(':').next().unwrap();
        fs::write(root.join("config"), format!(
            "format=vade-owner-v1\nmodel_path=/owner/model\nmodel_bytes=1024\ncontext_tokens=8192\noutput_tokens=4096\nport={port}\nstage_dir={}\nstate_dir={}\n",
            root.join("stage").display(), root.join("state").display(),
        )).unwrap();
        fs::write(root.join("state/run.conf"), format!(
            "version=1\nphase=running\nport={port}\ntoken={}\nmodel_path=/owner/model\nengine=/owner/engine\nendpoint_model=mock-model\n",
            "a".repeat(64),
        )).unwrap();
    }

    fn receipt(&self, name: &str) -> json::Value {
        let text = fs::read_to_string(self.0.join(name)).unwrap();
        assert!(text.len() < 16_384, "receipt should remain small");
        assert_eq!(
            strict_json::string(&text, "schema").as_deref(),
            Some("vade.verify.v1"),
            "receipt must be strict JSON: {text}"
        );
        json::parse(&text).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

struct Server {
    url: String,
    stopped: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<json::Value>>>,
    worker: Option<JoinHandle<()>>,
}
impl Server {
    fn start(reply: impl Fn(&json::Value, usize) -> String + Send + 'static) -> Self {
        Self::streaming(move |request, index, stream| {
            let response = reply(request, index);
            // Rejections may close early; that is part of the failure cases.
            let _ = stream.write_all(response.as_bytes());
        })
    }

    fn streaming(reply: impl Fn(&json::Value, usize, &mut TcpStream) + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let stopped = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = stopped.clone();
        let received = requests.clone();
        let worker = thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // macOS inherits O_NONBLOCK from the listening socket.
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(3)))
                            .unwrap();
                        stream
                            .set_write_timeout(Some(Duration::from_secs(3)))
                            .unwrap();
                        let request = read_request(&mut stream);
                        let mut requests = received.lock().unwrap();
                        let index = requests.len();
                        requests.push(request.clone());
                        drop(requests);
                        reply(&request, index, &mut stream);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("mock accept: {error}"),
                }
            }
        });
        Self {
            url,
            stopped,
            requests,
            worker: Some(worker),
        }
    }

    fn normal() -> Self {
        Self::start(|request, _| response(&events(request, None)))
    }

    fn requests(&self) -> Vec<json::Value> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take()
            && let Err(error) = worker.join()
            && !thread::panicking() {
            std::panic::resume_unwind(error);
        }
    }
}

fn read_request(stream: &mut TcpStream) -> json::Value {
    let mut bytes = Vec::new();
    let head_end = loop {
        let mut buf = [0; 4096];
        let count = stream.read(&mut buf).unwrap();
        assert!(count > 0, "request ended inside headers");
        bytes.extend_from_slice(&buf[..count]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
        assert!(bytes.len() < 16_384, "mock header limit");
    };
    let headers = String::from_utf8(bytes[..head_end].to_vec()).unwrap();
    assert!(
        headers.starts_with("POST /v1/completions HTTP/1.1\r\n"),
        "{headers}"
    );
    let length: usize = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("Content-Length")
                .then(|| value.trim().parse().unwrap())
        })
        .expect("Content-Length request header");
    assert!(length < 262_144, "mock body limit");
    while bytes.len() - head_end < length {
        let mut buf = [0; 4096];
        let count = stream.read(&mut buf).unwrap();
        assert!(count > 0, "request ended inside body");
        bytes.extend_from_slice(&buf[..count]);
    }
    json::parse(std::str::from_utf8(&bytes[head_end..head_end + length]).unwrap()).unwrap()
}

fn events(request: &json::Value, different: Option<(u32, &str)>) -> String {
    let draft = request["draft"].as_str().unwrap();
    assert!(
        matches!(draft, "plain" | "strict"),
        "unexpected draft {draft}"
    );
    let on = draft == "strict";
    let mut body = String::new();
    let (second_id, second_text) = different.unwrap_or((102, " world"));
    for (id, text) in [(101, "Hello"), (second_id, second_text), (103, "!")] {
        body.push_str(&format!(
            "data: {{\"model\":\"mock-model\",\"choices\":[{{\"index\":0,\"text\":{},\"token_id\":{id},\"finish_reason\":null}}]}}\n\n",
            json::packed(&json::Value::Str(text.into()))
        ));
    }
    body.push_str(&format!(
        "data: {{\"model\":\"mock-model\",\"choices\":[{{\"index\":0,\"text\":\"\",\"finish_reason\":\"length\"}}],\
         \"usage\":{{\"prompt_tokens\":12,\"completion_tokens\":3,\"total_tokens\":15}},\
         \"vade\":{{\"sampling\":{{\"temperature\":0,\"top_p\":1,\"top_k\":0,\"repetition_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,\"seed\":0}},\
         \"draft\":\"{draft}\",\"thinking\":false,\"mtp_depth\":2,\"draft_used\":{on},\"model_draft_used\":{on}}}}}\n\ndata: [DONE]\n\n"
    ));
    body
}

fn response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn line(output: Output, code: i32) -> String {
    assert_eq!(output.status.code(), Some(code), "{output:?}");
    assert!(
        output.stderr.is_empty(),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = String::from_utf8(output.stdout).unwrap();
    assert_eq!(line.lines().count(), 1, "one plain line: {line:?}");
    assert!(
        !line.chars().any(|c| c.is_control() && c != '\n'),
        "terminal controls: {line:?}"
    );
    line
}

fn assert_requests(requests: &[json::Value], count: usize) {
    assert_eq!(requests.len(), count * 2);
    for (index, request) in requests.iter().enumerate() {
        assert_eq!(
            request["draft"].as_str(),
            Some(if index % 2 == 0 { "plain" } else { "strict" })
        );
        assert_eq!(
            request["prompt"], requests[0]["prompt"],
            "the same frozen prompt must be sent every time"
        );
        assert!(!request["prompt"].as_str().unwrap().is_empty());
        assert_eq!(request["temperature"].as_f64(), Some(0.0));
        assert_eq!(request["top_p"].as_f64(), Some(1.0));
        assert_eq!(request["top_k"].as_i64(), Some(0));
        assert_eq!(request["repetition_penalty"].as_f64(), Some(1.0));
        assert_eq!(request["presence_penalty"].as_f64(), Some(0.0));
        assert_eq!(request["frequency_penalty"].as_f64(), Some(0.0));
        assert_eq!(request["seed"].as_i64(), Some(0));
        assert_eq!(request["cache"], json::Value::Bool(false));
        assert_eq!(request["ignore_eos"], json::Value::Bool(true));
        assert_eq!(request["enable_thinking"], json::Value::Bool(false));
        assert_eq!(request["stream"], json::Value::Bool(true));
        assert_eq!(
            request["stream_options"]["include_usage"],
            json::Value::Bool(true)
        );
        assert_eq!(request["max_tokens"].as_i64(), Some(3));
    }
}

#[test]
fn default_twenty_runs_compare_on_and_off_and_write_receipt() {
    let fixture = Fixture::new();
    let server = Server::normal();
    let text = line(fixture.command(&server.url).output().unwrap(), 0);
    assert_eq!(text, "IDENTICAL: 20/20 runs, 1 machine, 3 tokens\n");
    assert_requests(&server.requests(), 20);
    let receipt = fixture.receipt("vade-verify.json");
    assert_eq!(receipt["result"].as_str(), Some("identical"));
    assert_eq!(receipt["runs_requested"].as_i64(), Some(20));
    assert_eq!(receipt["runs_completed"].as_i64(), Some(20));
    assert_eq!(receipt["requests_completed"].as_i64(), Some(40));
    assert_eq!(receipt["tokens_per_run"].as_i64(), Some(3));
    assert_eq!(receipt["reference_ids_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(receipt["prompt_sha256"].as_str().unwrap().len(), 64);
    assert!(receipt["elapsed_ms"].as_f64().unwrap() >= 0.0);
}

#[test]
fn two_configured_machines_use_both_modes_and_compare_to_one_baseline() {
    let fixture = Fixture::new();
    let primary = Server::normal();
    let second = Server::normal();
    let text = line(
        fixture
            .command(&primary.url)
            .args([
                "-n",
                "3",
                "--machine",
                "second",
                "--receipt",
                "two-machines.json",
            ])
            .env("VADE_CONFIG", fixture.config(&second.url))
            .output()
            .unwrap(),
        0,
    );
    assert_eq!(text, "IDENTICAL: 3/3 runs, 2 machines, 3 tokens\n");
    assert_requests(&primary.requests(), 3);
    assert_requests(&second.requests(), 3);
    assert_eq!(
        primary.requests()[0]["prompt"],
        second.requests()[0]["prompt"]
    );
    let receipt = fixture.receipt("two-machines.json");
    assert_eq!(receipt["result"].as_str(), Some("identical"));
    assert_eq!(receipt["requests_completed"].as_i64(), Some(12));
    assert_eq!(receipt["runs_completed"].as_i64(), Some(3));
}

#[test]
fn first_difference_reports_one_based_run_token_and_both_ids_and_texts() {
    let fixture = Fixture::new();
    let server = Server::start(|request, index| {
        response(&events(
            request,
            (index == 13).then_some((999, " changed\n\u{1b}[2J")),
        ))
    });
    let text = line(
        fixture
            .command(&server.url)
            .args(["-n", "9"])
            .output()
            .unwrap(),
        1,
    );
    assert!(text.starts_with("DIFFERENT at run 7 token 2:"), "{text}");
    for expected in ["102", "999", "world", "changed"] {
        assert!(text.contains(expected), "{text}");
    }
    assert_eq!(server.requests().len(), 14, "stop at first divergence");
    let receipt = fixture.receipt("vade-verify.json");
    assert_eq!(receipt["result"].as_str(), Some("different"));
    let difference = &receipt["first_difference"];
    assert_eq!(difference["run"].as_i64(), Some(7));
    assert_eq!(difference["token"].as_i64(), Some(2));
    assert_eq!(difference["expected_id"].as_i64(), Some(102));
    assert_eq!(difference["actual_id"].as_i64(), Some(999));
    assert_eq!(difference["expected_text"].as_str(), Some(" world"));
    assert_eq!(
        difference["actual_text"].as_str(),
        Some(" changed\n\u{1b}[2J")
    );
}

#[test]
fn a_second_machine_difference_cannot_pass_as_locally_repeatable() {
    let fixture = Fixture::new();
    let primary = Server::normal();
    let second = Server::start(|request, _| response(&events(request, Some((999, " elsewhere")))));
    let text = line(
        fixture
            .command(&primary.url)
            .args(["-n", "2", "--machine", "second"])
            .env("VADE_CONFIG", fixture.config(&second.url))
            .output()
            .unwrap(),
        1,
    );
    assert!(text.starts_with("DIFFERENT at run 1 token 2:"), "{text}");
    assert!(text.contains("102") && text.contains("999"), "{text}");
    assert_eq!(
        fixture.receipt("vade-verify.json")["result"].as_str(),
        Some("different")
    );
}

#[test]
fn truncated_malformed_or_unverifiable_streams_never_report_identical() {
    let mutations: &[fn(String) -> String] = &[
        |body| body.replace("data: [DONE]\n\n", ""),
        |body| body.replace("\"token_id\":102", "\"token_id\":102.5"),
        |body| body.replace("\"token_id\":102,", ""),
        |body| body.replace("\"completion_tokens\":3", "\"completion_tokens\":4"),
        |body| body.replace("\"temperature\":0", "\"temperature\":1"),
        |body| body.replace("\"thinking\":false", "\"thinking\":true"),
        |body| body.replace("\"finish_reason\":\"length\"", "\"finish_reason\":\"stop\""),
        |body| body.replacen("data: {", "data: {BROKEN", 1),
    ];
    for (index, mutation) in mutations.iter().copied().enumerate() {
        let fixture = Fixture::new();
        let server = Server::start(move |request, _| response(&mutation(events(request, None))));
        let text = line(
            fixture
                .command(&server.url)
                .args(["-n", "1"])
                .output()
                .unwrap(),
            1,
        );
        assert!(text.starts_with("ERROR:"), "mutation {index}: {text}");
        assert_eq!(
            fixture.receipt("vade-verify.json")["result"].as_str(),
            Some("error")
        );
    }
}

#[test]
fn requested_drafting_without_any_actual_draft_is_not_a_proof() {
    let fixture = Fixture::new();
    let server = Server::start(|request, _| {
        response(
            &events(request, None)
                .replace("\"draft_used\":true", "\"draft_used\":false")
                .replace("\"model_draft_used\":true", "\"model_draft_used\":false"),
        )
    });
    let text = line(
        fixture
            .command(&server.url)
            .args(["-n", "2"])
            .output()
            .unwrap(),
        1,
    );
    assert!(
        text.starts_with("ERROR:") && text.to_lowercase().contains("draft"),
        "{text}"
    );
    assert_eq!(
        fixture.receipt("vade-verify.json")["result"].as_str(),
        Some("error")
    );
}

#[test]
fn one_successful_draft_cannot_hide_a_later_on_request_without_drafting() {
    let fixture = Fixture::new();
    let server = Server::start(|request, index| {
        let body = events(request, None);
        response(&if index == 3 {
            body.replace("\"draft_used\":true", "\"draft_used\":false")
                .replace("\"model_draft_used\":true", "\"model_draft_used\":false")
        } else {
            body
        })
    });
    let text = line(
        fixture
            .command(&server.url)
            .args(["-n", "3"])
            .output()
            .unwrap(),
        1,
    );
    assert!(
        text.starts_with("ERROR:")
            && text.contains("run 2")
            && text.to_lowercase().contains("draft"),
        "{text}"
    );
    assert_eq!(
        server.requests().len(),
        4,
        "stop at first unproven ON request"
    );
    let receipt = fixture.receipt("vade-verify.json");
    assert_eq!(receipt["result"].as_str(), Some("error"));
    assert_eq!(receipt["runs_completed"].as_i64(), Some(1));
}

#[test]
fn default_endpoint_discovery_uses_config_port_and_environment_url() {
    for environment_url in [false, true] {
        let fixture = Fixture::new();
        let server = Server::normal();
        let mut command = fixture.base_command();
        command.args(["-n", "1", "--timeout", "3"]);
        if environment_url {
            // An explicit endpoint works independently of lifecycle config.
            command.env("VADE_API_URL", &server.url);
        } else {
            let path = fixture.config(&server.url);
            let port = server.url.rsplit(':').next().unwrap();
            let config = fs::read_to_string(&path).unwrap().replacen(
                "[global]\n",
                &format!("[global]\nlocal_port = {port}\n"),
                1,
            );
            fs::write(&path, config).unwrap();
            command.env("VADE_CONFIG", path);
        }
        assert_eq!(
            line(command.output().unwrap(), 0),
            "IDENTICAL: 1/1 runs, 1 machine, 3 tokens\n"
        );
        assert_requests(&server.requests(), 1);
    }
}

#[test]
fn the_same_endpoint_under_a_completion_path_cannot_count_as_two_machines() {
    let fixture = Fixture::new();
    let server = Server::normal();
    let text = line(fixture.command(&server.url)
        .args(["-n", "1", "--machine", "second"])
        .env("VADE_CONFIG", fixture.config(&format!("{}/v1/completions", server.url)))
        .output().unwrap(), 1);
    assert!(text.starts_with("ERROR:") && text.contains("distinct"), "{text}");
    assert!(server.requests().is_empty(), "reject duplicate endpoints before requesting inference");
    let receipt = fixture.receipt("vade-verify.json");
    assert_eq!(receipt["result"].as_str(), Some("error"));
    assert_eq!(receipt["requests_completed"].as_i64(), Some(0));
}

#[test]
fn legacy_local_state_takes_precedence_over_an_implicit_default_config() {
    let fixture = Fixture::new();
    let local = Server::normal();
    let configured = Server::normal();
    let owned_home = fixture.0.join("owned-home");
    fs::create_dir(&owned_home).unwrap();
    fs::write(owned_home.join("setup.conf"), "").unwrap();
    fs::write(owned_home.join("run.conf"), format!(
        "version=1\nphase=running\nport={}\n", local.url.rsplit(':').next().unwrap(),
    )).unwrap();
    let xdg = fixture.0.join("xdg");
    fs::create_dir_all(xdg.join("vade")).unwrap();
    let explicit_config = fixture.config(&configured.url);
    let config = fs::read_to_string(explicit_config).unwrap().replacen(
        "[global]\n",
        &format!("[global]\nlocal_port = {}\n", configured.url.rsplit(':').next().unwrap()),
        1,
    );
    fs::write(xdg.join("vade/config"), config).unwrap();
    let text = line(fixture.base_command().args(["-n", "1", "--timeout", "3"])
        .env_remove("VADE_CONFIG").env("VADE_HOME", owned_home)
        .env("XDG_CONFIG_HOME", xdg).output().unwrap(), 0);
    assert_eq!(text, "IDENTICAL: 1/1 runs, 1 machine, 3 tokens\n");
    assert_requests(&local.requests(), 1);
    assert!(configured.requests().is_empty(), "default config must not override owned local engine state");
    let fixture = Fixture::new();
    let text = line(fixture.base_command().env_remove("VADE_CONFIG")
        .env("HOME", &fixture.0).env("XDG_CONFIG_HOME", fixture.0.join("config-home"))
        .output().unwrap(), 1);
    assert!(text.contains("vade up") && !text.contains("8080"), "{text}");
}

#[test]
fn owner_endpoint_takes_precedence_over_legacy_state_and_implicit_controller() {
    for override_home in [false, true] {
        let fixture = Fixture::new();
        let owner = Server::normal();
        let other = Server::normal();
        let home = fixture.0.join("home");
        let root = if override_home { fixture.0.join("owner") } else { home.join(".vade") };
        fixture.owner(&root, &owner.url);
        let record = fs::read(root.join("state/run.conf")).unwrap();
        let legacy = if override_home { root.clone() } else { home.join(".local/share/vade") };
        fs::create_dir_all(&legacy).unwrap();
        fs::write(legacy.join("setup.conf"), "").unwrap();
        fs::write(legacy.join("run.conf"), format!("port={}\n", other.url.rsplit(':').next().unwrap())).unwrap();
        let xdg = fixture.0.join("xdg");
        fs::create_dir_all(xdg.join("vade")).unwrap();
        let config = fs::read_to_string(fixture.config(&other.url)).unwrap().replacen(
            "[global]\n", &format!("[global]\nlocal_port = {}\n", other.url.rsplit(':').next().unwrap()), 1,
        );
        fs::write(xdg.join("vade/config"), config).unwrap();
        let mut command = fixture.base_command();
        command.args(["-n", "1", "--timeout", "3"])
            .env_remove("VADE_CONFIG").env_remove("VADE_HOME").env_remove("XDG_DATA_HOME")
            .env("HOME", home).env("XDG_CONFIG_HOME", xdg);
        if override_home { command.env("VADE_HOME", &root); }
        assert_eq!(line(command.output().unwrap(), 0), "IDENTICAL: 1/1 runs, 1 machine, 3 tokens\n");
        assert_requests(&owner.requests(), 1);
        assert!(other.requests().is_empty(), "owner selection must match lifecycle selection");
        assert_eq!(fs::read(root.join("state/run.conf")).unwrap(), record);
    }
}

#[test]
fn explicit_endpoint_and_controller_selectors_override_owner_state_in_order() {
    for selector in ["controller", "environment", "url"] {
        let fixture = Fixture::new();
        let selected = Server::normal();
        let other = Server::normal();
        let root = fixture.0.join("owner");
        fixture.owner(&root, &other.url);
        let controller = if selector == "controller" { &selected.url } else { &other.url };
        let path = fixture.config(controller);
        let config = fs::read_to_string(&path).unwrap().replacen(
            "[global]\n", &format!("[global]\nlocal_port = {}\n", controller.rsplit(':').next().unwrap()), 1,
        );
        fs::write(&path, config).unwrap();
        let mut command = fixture.base_command();
        command.args(["-n", "1", "--timeout", "3"])
            .env("VADE_HOME", root).env("VADE_CONFIG", path);
        if selector == "environment" { command.env("VADE_API_URL", &selected.url); }
        if selector == "url" {
            command.env("VADE_API_URL", &other.url).args(["--url", &selected.url]);
        }
        assert_eq!(line(command.output().unwrap(), 0), "IDENTICAL: 1/1 runs, 1 machine, 3 tokens\n");
        assert_requests(&selected.requests(), 1);
        assert!(other.requests().is_empty(), "explicit selector precedence changed: {selector}");
    }
}

#[test]
fn invalid_owner_endpoint_state_fails_without_falling_back_or_changing_records() {
    for fault in ["missing-record", "missing-config", "starting", "port", "mismatch", "format", "token", "malformed", "large", "directory", "symlink"] {
        let fixture = Fixture::new();
        let other = Server::normal();
        let root = fixture.0.join("owner");
        fixture.owner(&root, &other.url);
        let record = root.join("state/run.conf");
        let config = root.join("config");
        // A legacy endpoint must never rescue invalid owner state.
        fs::write(root.join("run.conf"), format!("port={}\n", other.url.rsplit(':').next().unwrap())).unwrap();
        fs::write(root.join("setup.conf"), "").unwrap();
        match fault {
            "missing-record" => fs::remove_file(&record).unwrap(),
            "missing-config" => fs::remove_file(&config).unwrap(),
            "starting" => fs::write(&record, fs::read_to_string(&record).unwrap().replace("phase=running", "phase=starting")).unwrap(),
            "port" => fs::write(&record, "version=1\nphase=running\nport=0\n").unwrap(),
            "mismatch" => fs::write(&config, "format=vade-owner-v1\nport=0\n").unwrap(),
            "format" => fs::write(&config, "format=unknown\n").unwrap(),
            "token" => fs::write(&record, fs::read_to_string(&record).unwrap().replace(&"a".repeat(64), "invalid")).unwrap(),
            "malformed" => fs::write(&record, "broken record").unwrap(),
            "large" => fs::write(&record, "x".repeat(65537)).unwrap(),
            "directory" => { fs::remove_file(&record).unwrap(); fs::create_dir(&record).unwrap(); }
            "symlink" => { fs::remove_file(&record).unwrap(); std::os::unix::fs::symlink(root.join("run.conf"), &record).unwrap(); }
            _ => unreachable!(),
        }
        let before = fs::read(&record).ok();
        let text = line(fixture.base_command().args(["-n", "1", "--timeout", "3"])
            .env_remove("VADE_CONFIG").env("VADE_HOME", &root)
            .env("XDG_CONFIG_HOME", fixture.0.join("unused-config"))
            .output().unwrap(), 1);
        assert!(text.starts_with("ERROR:"), "{fault}: {text}");
        assert!(!text.contains("owner state"), "{fault}: {text}");
        if matches!(fault, "missing-record" | "missing-config") {
            assert_eq!(text, "ERROR: The engine is not running on this Spark. Start it with vade up, or give the address of a running engine with --url.\n", "{fault}");
        }
        assert!(other.requests().is_empty(), "invalid owner state fell back: {fault}");
        assert_eq!(fs::read(&record).ok(), before, "verify changed owner evidence: {fault}");
    }
}

#[test]
fn an_unwritable_receipt_destination_fails_before_inference() {
    let fixture = Fixture::new();
    let server = Server::normal();
    let text = line(fixture.command(&server.url)
        .args(["-n", "1", "--receipt", "missing-parent/receipt.json"])
        .output().unwrap(), 1);
    assert!(text.starts_with("ERROR:") && text.to_lowercase().contains("receipt"), "{text}");
    assert!(server.requests().is_empty(), "receipt failures must be detected before inference");
    assert!(!fixture.0.join("missing-parent/receipt.json").exists());
    assert!(!fixture.0.join("vade-verify.json").exists());
}

#[test]
fn model_or_prompt_token_count_drift_invalidates_the_comparison() {
    for model_drift in [false, true] {
        let fixture = Fixture::new();
        let server = Server::start(move |request, index| {
            let body = events(request, None);
            response(&if index != 0 {
                if model_drift {
                    body.replace("mock-model", "changed-model")
                } else {
                    body.replace("\"prompt_tokens\":12", "\"prompt_tokens\":13")
                        .replace("\"total_tokens\":15", "\"total_tokens\":16")
                }
            } else {
                body
            })
        });
        let text = line(
            fixture
                .command(&server.url)
                .args(["-n", "2"])
                .output()
                .unwrap(),
            1,
        );
        assert!(
            text.starts_with("ERROR:") && text.contains("model or prompt"),
            "{text}"
        );
        assert_eq!(server.requests().len(), 2);
        assert_eq!(
            fixture.receipt("vade-verify.json")["result"].as_str(),
            Some("error")
        );
    }
}

#[test]
fn timeout_is_a_request_deadline_even_when_the_peer_keeps_sending_bytes() {
    let fixture = Fixture::new();
    let server = Server::streaming(|_, _, stream| {
        // Each gap is below --timeout. An idle-only timeout would last the
        // full 2.4 seconds, whereas the command must stop after one second.
        for byte in b"HTTP/1.1 200" {
            if stream.write_all(&[*byte]).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(220));
        }
    });
    let started = Instant::now();
    let text = line(
        fixture
            .base_command()
            .args(["--url", &server.url, "--timeout", "1", "-n", "1"])
            .output()
            .unwrap(),
        1,
    );
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "request deadline was reset by peer progress"
    );
    assert!(text.starts_with("ERROR:"), "{text}");
    assert_eq!(
        fixture.receipt("vade-verify.json")["result"].as_str(),
        Some("error")
    );
}

#[test]
fn chunked_http_with_utf8_and_split_sse_boundaries_is_accepted() {
    let fixture = Fixture::new();
    let server = Server::start(|request, _| {
        let body = events(request, Some((102, " café")));
        let mut wire = String::from(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        );
        // ASCII protocol cuts exercise the decoder independently of SSE events.
        // Keep chunks UTF-8 here so the mock itself remains a String transport.
        let mut start = 0;
        while start < body.len() {
            let mut end = (start + 7).min(body.len());
            while !body.is_char_boundary(end) {
                end += 1;
            }
            let part = &body[start..end];
            wire.push_str(&format!("{:x}\r\n{part}\r\n", part.len()));
            start = end;
        }
        wire.push_str("0\r\n\r\n");
        wire
    });
    let text = line(
        fixture
            .command(&server.url)
            .args(["-n", "1"])
            .output()
            .unwrap(),
        0,
    );
    assert_eq!(text, "IDENTICAL: 1/1 runs, 1 machine, 3 tokens\n");
}

#[test]
fn bad_http_status_and_incomplete_content_length_are_errors() {
    for status_error in [false, true] {
        let fixture = Fixture::new();
        let server = Server::start(move |request, _| {
            if status_error {
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbusy".into()
            } else {
                let body = events(request, None);
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len() + 9,
                )
            }
        });
        let text = line(
            fixture
                .command(&server.url)
                .args(["-n", "1"])
                .output()
                .unwrap(),
            1,
        );
        assert!(text.starts_with("ERROR:"), "{text}");
        assert_eq!(
            fixture.receipt("vade-verify.json")["result"].as_str(),
            Some("error")
        );
    }
}

#[test]
fn help_requires_no_valid_configuration_or_credentials() {
    let fixture = Fixture::new();
    for arguments in [["verify", "--help"], ["help", "verify"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_vade"))
            .current_dir(&fixture.0)
            .args(arguments)
            .env("VADE_CONFIG", "")
            .env("VADE_HOME", "relative-invalid")
            .env("VADE_API_KEY_FILE", "missing-credential")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(output.stderr.is_empty(), "{output:?}");
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(
            text.contains("-n") && text.contains("--machine") && text.contains("--receipt"),
            "{text}"
        );
    }
    assert!(!fixture.0.join("vade-verify.json").exists());
}

#[test]
fn all_bundled_prompt_sizes_are_stable_across_runs_without_external_files() {
    let mut lengths = Vec::new();
    for prompt in ["short", "2k", "8k"] {
        let fixture = Fixture::new();
        let server = Server::normal();
        let text = line(
            fixture
                .command(&server.url)
                .args(["-n", "2", "--prompt", prompt])
                .output()
                .unwrap(),
            0,
        );
        assert!(text.starts_with("IDENTICAL: 2/2"), "{text}");
        let requests = server.requests();
        assert_requests(&requests, 2);
        lengths.push(requests[0]["prompt"].as_str().unwrap().len());
    }
    assert!(
        lengths.windows(2).all(|pair| pair[0] < pair[1]),
        "prompt sizes: {lengths:?}"
    );
}
