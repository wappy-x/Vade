//! Independent HTTP reference clients, real gateway processes, synthetic peers.
//! These fixtures prove wire compatibility; they are not installed-vendor proofs.
use std::{
    collections::BTreeMap,
    io::{self, BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use vade_harness::json;

#[derive(Clone)]
struct Case {
    api: &'static str,
    name: &'static str,
    path: &'static str,
    request: Vec<u8>,
    response: Vec<u8>,
    status: u16,
    stream: bool,
    truncated: bool,
    defaults: bool,
}
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for api in ["chat", "responses", "messages"] {
        let (path, input, answer, tool, followup, stream) = match api {
            "chat" => (
                "/v1/chat/completions?fixture=1",
                r#""messages":[{"role":"user","content":"Read marker.txt"}]"#,
                r#"{"choices":[{"message":{"role":"assistant","content":"TOKEN_FIXTURE_314"},"finish_reason":"stop"}],"usage":{"prompt_tokens":101,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":97}}}"#,
                r#"{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"fixture-call","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"marker.txt\"}"}}]},"finish_reason":"tool_calls"}]}"#,
                r#""messages":[{"role":"user","content":"Read marker.txt"},{"role":"assistant","tool_calls":[{"id":"fixture-call","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"marker.txt\"}"}}]},{"role":"tool","tool_call_id":"fixture-call","content":"TOKEN_FIXTURE_314"}]"#,
                "data: {\"choices\":[{\"delta\":{\"content\":\"雪 café 👋\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
            ),
            "responses" => (
                "/v1/responses?fixture=1",
                r#""input":[{"role":"user","content":"Read marker.txt"}]"#,
                r#"{"id":"fixture-response","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"TOKEN_FIXTURE_314","annotations":[]}]}],"usage":{"input_tokens":101,"output_tokens":2,"input_tokens_details":{"cached_tokens":97}}}"#,
                r#"{"id":"fixture-response","object":"response","status":"completed","output":[{"type":"function_call","id":"item-1","call_id":"fixture-call","name":"read_file","arguments":"{\"path\":\"marker.txt\"}"}]}"#,
                r#""input":[{"role":"user","content":"Read marker.txt"},{"type":"function_call","id":"item-1","call_id":"fixture-call","name":"read_file","arguments":"{\"path\":\"marker.txt\"}"},{"type":"function_call_output","call_id":"fixture-call","output":"TOKEN_FIXTURE_314"}]"#,
                "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"雪 café 👋\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[]}}\n\n",
            ),
            _ => (
                "/v1/messages?beta=true",
                r#""max_tokens":32,"messages":[{"role":"user","content":"Read marker.txt"}]"#,
                r#"{"id":"fixture-message","type":"message","role":"assistant","content":[{"type":"text","text":"TOKEN_FIXTURE_314"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":2,"cache_read_input_tokens":97}}"#,
                r#"{"id":"fixture-message","type":"message","role":"assistant","content":[{"type":"tool_use","id":"fixture-call","name":"read_file","input":{"path":"marker.txt"}}],"stop_reason":"tool_use"}"#,
                r#""max_tokens":32,"messages":[{"role":"user","content":"Read marker.txt"},{"role":"assistant","content":[{"type":"tool_use","id":"fixture-call","name":"read_file","input":{"path":"marker.txt"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"fixture-call","content":"TOKEN_FIXTURE_314"}]}]"#,
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"雪 café 👋\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            ),
        };
        for (index, name) in [
            "text",
            "fragmented_sse",
            "tool_call",
            "tool_result",
            "usage",
            "upstream_error",
            "explicit_null",
            "greedy_default",
            "truncated_sse",
        ]
        .into_iter()
        .enumerate()
        {
            let defaults = index == 7;
            let fields = if index == 3 { followup } else { input };
            let sampling = if defaults {
                ""
            } else if index == 6 {
                r#", "seed":null, "generation_config":{"future":true}"#
            } else {
                r#", "temperature":0, "seed":20260905"#
            };
            let tool_schema = if matches!(index, 2 | 3) {
                match api {
                    "chat" => {
                        r#", "tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]"#
                    }
                    "responses" => {
                        r#", "tools":[{"type":"function","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]"#
                    }
                    _ => {
                        r#", "tools":[{"name":"read_file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]"#
                    }
                }
            } else {
                ""
            };
            let streaming = matches!(index, 1 | 8);
            let request = format!(r#"{{ "model":"fixture", {fields}{sampling}{tool_schema}, "stream":{streaming}, "extension":1.20e0, "escaped":"\u0061\n雪" }}"#).into_bytes();
            let response = match index {
                1 => stream.as_bytes().to_vec(),
                2 => tool.as_bytes().to_vec(),
                5 => b"{\"error\":{\"message\":\"opaque provider failure\",\"code\":\"fixture_busy\"},\"raw\":\"\\u001b[2J\"}".to_vec(),
                8 => stream.as_bytes()[..stream.len() / 2].to_vec(),
                _ => answer.as_bytes().to_vec(),
            };
            cases.push(Case {
                api,
                name,
                path,
                request,
                response,
                status: if index == 5 { 429 } else { 200 },
                stream: matches!(index, 1 | 8),
                truncated: index == 8,
                defaults,
            });
        }
    }
    cases
}

// This decoder intentionally shares no transport code with either gateway.
struct Reply {
    status: u16,
    fields: BTreeMap<String, String>,
    body: Vec<u8>,
    complete: bool,
}
fn head(reader: &mut impl BufRead) -> io::Result<(String, BTreeMap<String, String>)> {
    let mut first = String::new();
    if reader.read_line(&mut first)? == 0 {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    let mut fields = BTreeMap::new();
    for _ in 0..128 {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        if line == "\r\n" {
            return Ok((first, fields));
        }
        let (key, value) = line.split_once(':').ok_or(io::ErrorKind::InvalidData)?;
        fields.insert(key.to_ascii_lowercase(), value.trim().to_owned());
    }
    Err(io::ErrorKind::InvalidData.into())
}
fn read_reply(reader: &mut impl BufRead, mut progress: impl FnMut()) -> io::Result<Reply> {
    let (first, fields) = head(reader)?;
    let status = first.split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut body = Vec::new();
    let mut complete = true;
    if fields
        .get("transfer-encoding")
        .is_some_and(|value| value == "chunked")
    {
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                complete = false;
                break;
            }
            let size =
                usize::from_str_radix(line.trim(), 16).map_err(|_| io::ErrorKind::InvalidData)?;
            assert!(size <= 64 * 1024);
            if size == 0 {
                let mut end = [0; 2];
                complete = reader.read_exact(&mut end).is_ok() && end == *b"\r\n";
                break;
            }
            let start = body.len();
            reader.take(size as u64).read_to_end(&mut body)?;
            if body.len() > start {
                progress();
            }
            let mut end = [0; 2];
            if body.len() - start != size || reader.read_exact(&mut end).is_err() || end != *b"\r\n"
            {
                complete = false;
                break;
            }
        }
    } else {
        let size: usize = fields["content-length"].parse().unwrap();
        assert!(size < 64 * 1024);
        while body.len() < size {
            let mut chunk = [0; 512];
            let amount = (size - body.len()).min(chunk.len());
            let read = reader.read(&mut chunk[..amount])?;
            if read == 0 {
                complete = false;
                break;
            }
            body.extend_from_slice(&chunk[..read]);
            progress();
        }
    }
    Ok(Reply {
        status,
        fields,
        body,
        complete,
    })
}

struct Mock {
    address: SocketAddr,
    release: mpsc::Sender<()>,
    seen: Arc<Mutex<Vec<(usize, String)>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn start(cases: Arc<Vec<Case>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let running = stop.clone();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let captured = seen.clone();
        let (release, wait) = mpsc::channel();
        let worker = thread::spawn(move || {
            while !running.load(Ordering::Relaxed) {
                let (socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(e) => panic!("fixture accept failed: {e}"),
                };
                // Accepted sockets may inherit O_NONBLOCK on macOS.
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut reader = BufReader::new(socket);
                let (first, fields) = head(&mut reader).unwrap();
                let index: usize = fields["x-fixture-case"].parse().unwrap();
                let arm = &fields["x-fixture-arm"];
                let case = &cases[index];
                assert_eq!(first.trim(), format!("POST {} HTTP/1.1", case.path));
                assert_eq!(fields["authorization"], "Bearer fixture-key-0000");
                assert_eq!(fields["anthropic-version"], "2023-06-01");
                let size: usize = fields["content-length"].parse().unwrap();
                assert!(size < 64 * 1024);
                let mut body = vec![0; size];
                reader.read_exact(&mut body).unwrap();
                let expected = if arm != "raw" && case.defaults {
                    let input = std::str::from_utf8(&case.request).unwrap();
                    format!(
                        "{},\"temperature\":0,\"seed\":20260905,\"top_p\":1.0,\"top_k\":0,\"repetition_penalty\":1.0,\"presence_penalty\":0.0,\"frequency_penalty\":0.0}}",
                        input.strip_suffix('}').unwrap()
                    )
                    .into_bytes()
                } else {
                    case.request.clone()
                };
                assert_eq!(
                    body, expected,
                    "{} {} {arm}: request changed",
                    case.api, case.name
                );
                captured.lock().unwrap().push((index, arm.clone()));
                let writer = reader.get_mut();
                write!(writer, "HTTP/1.1 {} Fixture\r\nContent-Type: {}\r\nContent-Length: {}\r\nX-Fixture-Exact: keep\r\nRetry-After: 7\r\nConnection: close\r\n\r\n", case.status, if case.stream { "text/event-stream" } else { "application/json" }, case.response.len() + usize::from(case.truncated)).unwrap();
                if case.name == "fragmented_sse" {
                    // The upstream cannot finish until the client has observed
                    // body bytes. Split inside a multibyte Unicode character.
                    let split = case.response.iter().position(|byte| *byte == 0xe9).unwrap() + 1;
                    writer.write_all(&case.response[..split]).unwrap();
                    writer.flush().unwrap();
                    wait.recv_timeout(Duration::from_secs(3))
                        .expect("gateway buffered the entire stream");
                    for fragment in case.response[split..].chunks(3) {
                        writer.write_all(fragment).unwrap();
                    }
                } else {
                    writer.write_all(&case.response).unwrap();
                }
            }
        });
        Self {
            address,
            release,
            seen,
            stop,
            worker: Some(worker),
        }
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
struct Gateway(Child, SocketAddr);
impl Gateway {
    fn start(upstream: SocketAddr, standalone: bool) -> Self {
        Self::with_stats(upstream, standalone, None)
    }
    fn with_stats(upstream: SocketAddr, standalone: bool, stats: Option<&std::path::Path>) -> Self {
        let reserve = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reserve.local_addr().unwrap();
        drop(reserve);
        let mut command = Command::new(if standalone {
            env!("CARGO_BIN_EXE_vade-gateway")
        } else {
            env!("CARGO_BIN_EXE_vade-harness")
        });
        if !standalone {
            command.arg("gateway");
        }
        // An inherited client token would answer these unauthenticated calls with 401.
        command.env_remove("VADE_GATEWAY_TOKEN");
        if let Some(path) = stats { command.arg("--stats").arg(path); }
        let child = command
            .args([
                "--upstream",
                &format!("http://{upstream}/v1"),
                "--bind",
                &address.to_string(),
            ])
            .env_remove("VADE_API_KEY_FILE")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut gateway = Self(child, address);
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            assert!(
                gateway.0.try_wait().unwrap().is_none(),
                "gateway failed at startup"
            );
            if TcpStream::connect_timeout(&address, Duration::from_millis(20)).is_ok() {
                return gateway;
            }
            assert!(
                Instant::now() < deadline,
                "gateway startup exceeded deadline"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }
}
impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn read_tool(api: &str, reply: &[u8], workspace: &std::path::Path) -> (String, String) {
    let value = json::parse_bytes(reply).unwrap();
    let tool = match api {
        "chat" => &value.get("choices").unwrap().as_array().unwrap()[0]
            .get("message")
            .unwrap()
            .get("tool_calls")
            .unwrap()
            .as_array()
            .unwrap()[0],
        "responses" => &value.get("output").unwrap().as_array().unwrap()[0],
        _ => &value.get("content").unwrap().as_array().unwrap()[0],
    };
    let id = tool
        .get(if api == "responses" { "call_id" } else { "id" })
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned();
    let function = if api == "chat" {
        tool.get("function").unwrap()
    } else {
        tool
    };
    assert_eq!(function.get("name").unwrap().as_str(), Some("read_file"));
    let input = if api == "messages" {
        function.get("input").unwrap().clone()
    } else {
        json::parse(function.get("arguments").unwrap().as_str().unwrap()).unwrap()
    };
    let path = input.get("path").unwrap().as_str().unwrap();
    assert_eq!(
        path, "marker.txt",
        "fixture tools cannot read arbitrary files"
    );
    (id, std::fs::read_to_string(workspace.join(path)).unwrap())
}

#[test]
fn three_api_reference_clients_keep_27_transcripts_exact_through_both_gateways() {
    let cases = Arc::new(cases());
    assert_eq!(cases.len(), 27);
    let mock = Mock::start(cases.clone());
    let invisible = Gateway::start(mock.address, true);
    let core = Gateway::start(mock.address, false);
    let workspace =
        std::env::temp_dir().join(format!("vade-reference-clients-{}", std::process::id()));
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(workspace.join("marker.txt"), "TOKEN_FIXTURE_314").unwrap();
    let mut tool_results = BTreeMap::new();
    for (index, case) in cases.iter().enumerate() {
        for (arm, address) in [
            ("raw", mock.address),
            ("invisible", invisible.1),
            ("core", core.1),
        ] {
            let mut body = case.request.clone();
            if case.name == "tool_result" {
                let (id, result): &(String, String) = &tool_results[&(case.api, arm)];
                body = String::from_utf8(body)
                    .unwrap()
                    .replace("fixture-call", id)
                    .replace("TOKEN_FIXTURE_314", result)
                    .into_bytes();
            }
            let mut socket = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(4)))
                .unwrap();
            socket
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            write!(socket, "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAuthorization: Bearer fixture-key-0000\r\nAnthropic-Version: 2023-06-01\r\nX-Fixture-Case: {index}\r\nX-Fixture-Arm: {arm}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", case.path, body.len()).unwrap();
            socket.write_all(&body).unwrap();
            let mut released = false;
            let reply = read_reply(&mut BufReader::new(socket), || {
                if case.name == "fragmented_sse" && !released {
                    mock.release.send(()).unwrap();
                    released = true;
                }
            })
            .unwrap();
            assert_eq!(
                reply.status, case.status,
                "{} {} {arm}",
                case.api, case.name
            );
            assert_eq!(
                reply.body, case.response,
                "{} {} {arm}: response changed",
                case.api, case.name
            );
            assert_eq!(
                reply.complete, !case.truncated,
                "truncation cannot become success"
            );
            assert_eq!(reply.fields["x-fixture-exact"], "keep");
            assert_eq!(reply.fields["retry-after"], "7");
            if case.name == "fragmented_sse" {
                assert!(released);
            }
            if case.name == "tool_call" {
                tool_results.insert(
                    (case.api, arm),
                    read_tool(case.api, &reply.body, &workspace),
                );
            }
            if case.name == "tool_result" {
                assert!(String::from_utf8_lossy(&reply.body).contains("TOKEN_FIXTURE_314"));
            }
            eprintln!(
                "client_conformance api={} case={} arm={arm} status={} body_bytes={} complete={}",
                case.api,
                case.name,
                reply.status,
                reply.body.len(),
                reply.complete
            );
        }
    }
    assert_eq!(
        tool_results.len(),
        9,
        "one real reference read-tool round trip per API/arm"
    );
    let seen = mock.seen.lock().unwrap();
    assert_eq!(
        seen.len(),
        81,
        "hidden probes or retries changed dispatch count"
    );
    for index in 0..27 {
        for arm in ["raw", "invisible", "core"] {
            assert_eq!(
                seen.iter().filter(|(i, a)| *i == index && a == arm).count(),
                1
            );
        }
    }
    drop(seen);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[test]
fn completions_stats_observe_streamed_text_without_changing_reply_bytes() {
    let stream = "data: {\"choices\":[{\"text\":\"雪 café\",\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"text\":\" done\",\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n";
    let response = br#"{"choices":[{"text":"complete answer"}],"usage":{"prompt_tokens":11,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":0}}}"#;
    let cases = Arc::new(vec![
        Case { api: "completions", name: "fragmented_sse", path: "/v1/completions?case=fragmented_sse", request: br#"{"prompt":"fixture","seed":713,"stream":true}"#.to_vec(), response: stream.as_bytes().to_vec(), status: 200, stream: true, truncated: false, defaults: false },
        Case { api: "completions", name: "json", path: "/v1/completions?case=json", request: br#"{"prompt":"fixture","seed":713}"#.to_vec(), response: response.to_vec(), status: 200, stream: false, truncated: false, defaults: false },
        Case { api: "completions", name: "error", path: "/v1/completions?case=error", request: br#"{"prompt":"fixture","seed":713}"#.to_vec(), response: br#"{"error":{"message":"opaque"}}"#.to_vec(), status: 429, stream: false, truncated: false, defaults: false },
        Case { api: "completions", name: "truncated", path: "/v1/completions?case=truncated", request: br#"{"prompt":"fixture","seed":713,"stream":true}"#.to_vec(), response: stream.as_bytes().to_vec(), status: 200, stream: true, truncated: true, defaults: false },
    ]);
    let root = std::env::temp_dir().join(format!("vade-completions-stats-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("stats.jsonl");
    let mock = Mock::start(cases.clone());
    let gateway = Gateway::with_stats(mock.address, true, Some(&path));
    for (index, case) in cases.iter().enumerate() {
        for (arm, address) in [("raw", mock.address), ("invisible", gateway.1)] {
            let mut socket = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(4))).unwrap();
            socket.set_write_timeout(Some(Duration::from_secs(3))).unwrap();
            write!(socket, "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAuthorization: Bearer fixture-key-0000\r\nAnthropic-Version: 2023-06-01\r\nX-Fixture-Case: {index}\r\nX-Fixture-Arm: {arm}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", case.path, case.request.len()).unwrap();
            socket.write_all(&case.request).unwrap();
            let mut released = false;
            let reply = read_reply(&mut BufReader::new(socket), || {
                if case.name == "fragmented_sse" && !released { mock.release.send(()).unwrap(); released = true; }
            }).unwrap();
            assert_eq!(reply.body, case.response);
            assert_eq!(reply.status, case.status);
            assert_eq!(reply.complete, !case.truncated);
            assert_eq!(reply.fields["x-fixture-exact"], "keep");
        }
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    let rows = loop {
        let text = std::fs::read_to_string(&path).unwrap();
        let observed: Vec<_> = text.lines().filter_map(|line| json::parse(line).ok()).collect();
        let ordered: Option<Vec<_>> = cases.iter().map(|case| observed.iter().find(|row|
            row.get("path").and_then(json::Value::as_str) == Some(case.path)).cloned()).collect();
        if let Some(rows) = ordered { break rows; }
        assert!(Instant::now() < deadline, "missing stats rows: {text}");
        thread::sleep(Duration::from_millis(1));
    };
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.get("telemetry_invalid"), Some(&json::Value::Bool(index >= 2)));
        if index < 2 {
            assert_eq!(row.get("completion_tokens"), Some(&json::Value::Number("2".into())));
            assert_eq!(row.get("cached_tokens"), Some(&json::Value::Number("0".into())));
        } else { assert_eq!(row.get("completion_tokens"), Some(&json::Value::Null)); }
    }
    assert_ne!(rows[0].get("first_token_ms"), Some(&json::Value::Null));
    assert_eq!(rows[1].get("first_token_ms"), Some(&json::Value::Null));
    assert_eq!(mock.seen.lock().unwrap().len(), 8, "no hidden probes or retries");
    drop(gateway);
    std::fs::remove_dir_all(root).unwrap();
}
