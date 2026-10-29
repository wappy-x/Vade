//! Interactive steering uses the real CLI and cancellation HTTP route.
#[path = "support/home.rs"]
mod home;
#[path = "support/cli.rs"]
mod cli;
use std::{io::{BufRead, BufReader, Read, Write}, net::{TcpListener, TcpStream},
    sync::mpsc::{self, Receiver}, thread, time::Duration};
use vade_harness::json;

struct Mock {
    endpoint: String,
    requests: Receiver<(TcpStream, json::Value)>,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    admissions: Receiver<(TcpStream, json::Value)>,
    cancels: Receiver<String>,
    done: mpsc::Sender<()>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn new() -> Self { Self::with_ack(true) }
    fn with_ack(ack: bool) -> Self { Self::build(ack, false) }
    #[cfg(target_os = "macos")]
    fn with_admission() -> Self { Self::build(true, true) }
    fn build(ack: bool, admission: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (requests, incoming) = mpsc::channel();
        let (admission_tx, admission_rx) = mpsc::channel();
        let (cancels, cancelled) = mpsc::channel();
        let (done, stop) = mpsc::channel();
        let worker = thread::spawn(move || loop {
            if stop.try_recv().is_ok() { break; }
            let (socket, _) = match listener.accept() {
                Ok(socket) => socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(1)); continue;
                }
                Err(error) => panic!("{error}"),
            };
            socket.set_nonblocking(false).unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut reader = BufReader::new(socket);
            let mut first = String::new(); reader.read_line(&mut first).unwrap();
            let mut size = 0;
            loop {
                let mut line = String::new(); assert!(reader.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" { break; }
                if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    size = n.trim().parse().unwrap();
                }
            }
            let mut bytes = vec![0; size]; reader.read_exact(&mut bytes).unwrap();
            let mut socket = reader.into_inner();
            if first.starts_with("GET /v1/capabilities ") {
                let caps = include_str!("fixtures/engine-capabilities-v1.json");
                let caps = if admission { caps.replace("\"output_limit\": 32", "\"output_limit\": 65536") }
                    else { caps.replace("\"/v1/admission\"", "\"unavailable\"") };
                let caps = caps.replace("4096", "262144").replace("4000", "262000");
                reply(&mut socket, &caps);
            } else if first.starts_with("POST /v1/admission ") {
                admission_tx.send((socket, json::parse_bytes(&bytes).unwrap())).unwrap();
            } else if first.starts_with("POST /v1/cancel ") {
                let body = json::parse_bytes(&bytes).unwrap();
                cancels.send(body.get("cancel").unwrap().as_str().unwrap().to_owned()).unwrap();
                reply(&mut socket, if ack { r#"{"cancelled":true}"# } else { r#"{"cancelled":false}"# });
            } else {
                assert!(first.starts_with("POST /v1/chat/completions "), "{first}");
                requests.send((socket, json::parse_bytes(&bytes).unwrap())).unwrap();
            }
        });
        Self { endpoint, requests: incoming, admissions: admission_rx, cancels: cancelled, done, worker: Some(worker) }
    }
    fn request(&self) -> (TcpStream, json::Value) {
        let request = self.requests.recv_timeout(Duration::from_secs(5)).unwrap();
        let system = request.1.get("messages").unwrap().as_array().unwrap()[0].get("content").unwrap().as_str().unwrap();
        assert!(system.contains("Served model id (configured): \"mock\"; family: not specified."));
        request
    }
    fn cancelled(&self, body: &json::Value) {
        assert_eq!(self.cancels.recv_timeout(Duration::from_secs(5)).unwrap(),
            body.get("cancel").unwrap().as_str().unwrap());
    }
    #[cfg(target_os = "macos")]
    fn admission(&self) -> (TcpStream, json::Value) {
        self.admissions.recv_timeout(Duration::from_secs(5)).unwrap()
    }
    /// Answer any further admission counts until the generation request lands.
    #[cfg(target_os = "macos")]
    fn generation(&self) -> (TcpStream, json::Value) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok((mut socket, body)) = self.admissions.try_recv() { admit(&mut socket, &body); }
            match self.requests.recv_timeout(Duration::from_millis(50)) {
                Ok(request) => {
                    let system = request.1.get("messages").unwrap().as_array().unwrap()[0].get("content").unwrap().as_str().unwrap();
                    assert!(system.contains("Served model id (configured): \"mock\"; family: not specified."));
                    return request;
                }
                Err(mpsc::RecvTimeoutError::Timeout) =>
                    assert!(std::time::Instant::now() < deadline, "generation request never arrived"),
                Err(error) => panic!("{error}"),
            }
        }
    }
    fn cli(&self) -> cli::Cli {
        cli::Cli::start(home::harness()
            .args(["direct", "--state-in-project", "--endpoint", &self.endpoint, "--model", "mock", "--plain"]))
    }
    /// Stop accepting new connections, as if the server died; generation
    /// sockets already handed to the test stay as the test left them.
    fn kill_server(&mut self) {
        let _ = self.done.send(());
        self.worker.take().unwrap().join().unwrap();
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        let _ = self.done.send(());
        if let Some(worker) = self.worker.take() { worker.join().unwrap(); }
    }
}
fn reply(socket: &mut TcpStream, body: &str) {
    write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
}
/// Admit a counted prompt, echoing the envelope's output reservation. A
/// cancelled count may already have closed its socket, so a failed write
/// here is not a test failure.
#[cfg(target_os = "macos")]
fn admit(socket: &mut TcpStream, requested: &json::Value) {
    let output = match requested.get("output_tokens") {
        Some(json::Value::Number(number)) => number.clone(),
        _ => "32".to_owned(),
    };
    let body = format!(r#"{{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","admission_kind":"context_fit","reserves_resources":false,"prompt_tokens":100,"output_tokens":{output},"context_limit":262144,"input_limit":262000,"output_limit":65536,"admitted":true}}"#);
    let _ = write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len());
}
#[cfg(target_os = "macos")]
fn users(body: &json::Value) -> Vec<&str> {
    // Admission bodies wrap the generation request in an envelope.
    let inner = body.get("request").unwrap_or(body);
    history(inner).into_iter().filter(|(role, _)| *role == "user").map(|(_, text)| text).collect()
}
fn stream(socket: &mut TcpStream, text: &str) {
    socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
    write!(socket, "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n", json::quote(text)).unwrap();
}
fn finish(socket: &mut TcpStream) {
    reply(socket, r#"{"choices":[{"message":{"role":"assistant","content":"FINAL"},"finish_reason":"stop"}]}"#);
}
fn history(body: &json::Value) -> Vec<(&str, &str)> {
    body.get("messages").unwrap().as_array().unwrap().iter().map(|m|
        (m.get("role").unwrap().as_str().unwrap(), m.get("content").unwrap().as_str().unwrap())).collect()
}
#[test]
fn interrupt_mid_stream_and_twice_preserves_partial_context() {
    let mock = Mock::new(); let mut cli = mock.cli(); cli.send("original\n");
    let (mut first, body) = mock.request(); stream(&mut first, "PART_ONE");
    cli.until(false, "PART_ONE"); cli.send("steer one\n"); mock.cancelled(&body);
    let (mut second, body) = mock.request();
    assert_eq!(&history(&body)[1..], &[("user", "original"), ("assistant", "PART_ONE\n[interrupted]"), ("user", "steer one")]);
    assert_eq!(first.read(&mut [0]).unwrap(), 0);
    stream(&mut second, "PART_TWO"); cli.until(false, "PART_TWO"); cli.send("steer two\n"); mock.cancelled(&body);
    let (mut third, body) = mock.request();
    assert_eq!(&history(&body)[3..], &[("user", "steer one"), ("assistant", "PART_TWO\n[interrupted]"), ("user", "steer two")]);
    finish(&mut third); drop(third); cli.until(false, "FINAL");
    let output = cli.finish(); assert!(output.status.success(), "{:?}", output);
}
#[test]
fn interrupt_before_headers_cancels_before_first_token() {
    let mock = Mock::new(); let mut cli = mock.cli(); cli.send("original\n");
    let (mut first, body) = mock.request(); cli.send("replacement\n"); mock.cancelled(&body);
    let (mut next, body) = mock.request();
    assert_eq!(&history(&body)[1..], &[("user", "original"), ("assistant", "\n[interrupted]"), ("user", "replacement")]);
    assert_eq!(first.read(&mut [0]).unwrap(), 0);
    finish(&mut next); drop(next); cli.until(false, "FINAL"); assert!(cli.finish().status.success());
}
#[test]
fn exit_during_generation_waits_for_server_cancel() {
    let mock = Mock::new(); let mut cli = mock.cli(); cli.send("original\n");
    let (mut first, body) = mock.request(); stream(&mut first, "PART_EXIT");
    cli.until(false, "PART_EXIT"); cli.send("/exit\n"); mock.cancelled(&body);
    let output = cli.finish(); assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("[interrupted]"));
    assert_eq!(first.read(&mut [0]).unwrap(), 0);
    assert!(mock.requests.try_recv().is_err());
}
#[test]
fn queue_toggle_keeps_the_current_answer_running() {
    let mock = Mock::new(); let mut cli = mock.cli(); cli.send("/queue on\noriginal\n");
    let (mut first, _) = mock.request(); stream(&mut first, "PART_QUEUE");
    cli.until(false, "PART_QUEUE"); cli.send("follow up\n"); cli.until(true, "Follow-up queued");
    assert!(mock.cancels.try_recv().is_err());
    first.write_all(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n").unwrap(); drop(first);
    let (mut next, body) = mock.request();
    assert_eq!(&history(&body)[1..], &[("user", "original"), ("assistant", "PART_QUEUE"), ("user", "follow up")]);
    finish(&mut next); drop(next); cli.until(false, "FINAL"); assert!(cli.finish().status.success());
}

#[cfg(target_os = "macos")]
#[path = "support/raw_ui.rs"]
mod raw_ui;
#[cfg(target_os = "macos")]
fn native_case(before_token: bool, twice: bool, exit: bool) {
    let mock = Mock::new();
    let root = std::env::temp_dir().join(format!("vade-native-steering-{}-{before_token}-{twice}-{exit}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let mut ui = raw_ui::Ui::start("agent", &mock.endpoint, root.to_str().unwrap(), false);
    ui.resize(24, 110);
    ui.send(b"original\r");
    let (mut first, body) = mock.request();
    if !before_token { stream(&mut first, "NATIVE_PART"); ui.until("NATIVE_PART", Duration::from_secs(5)); }
    ui.send(if exit { b"/exit\r" } else { b"replacement\r" });
    mock.cancelled(&body);
    if !exit {
        let (mut next, body) = mock.request();
        let messages = history(&body);
        assert!(messages.windows(3).any(|m| m == [("user", "original"),
            ("assistant", if before_token { "\n[interrupted]" } else { "NATIVE_PART\n[interrupted]" }), ("user", "replacement")]));
        if twice {
            stream(&mut next, "NATIVE_TWO"); ui.until("NATIVE_TWO", Duration::from_secs(5));
            ui.send(b"third\r"); mock.cancelled(&body);
            let (socket, body) = mock.request(); next = socket;
            assert!(history(&body).windows(3).any(|m| m == [("user", "replacement"), ("assistant", "NATIVE_TWO\n[interrupted]"), ("user", "third")]));
        }
        finish(&mut next); drop(next); ui.until("FINAL", Duration::from_secs(5)); ui.send(b"/exit\r");
    }
    ui.wait_exit(); drop(ui);
    assert_eq!(first.read(&mut [0]).unwrap(), 0);
    let workspace = vade_harness::tools::Workspace::new(&root).unwrap();
    let sessions = vade_harness::journal::list(&workspace).unwrap();
    assert_eq!(sessions.len(), 1);
    // Reopening the journal must retain the interrupted partial answer.
    let (_store, conversation, _torn) = vade_harness::journal::Journal::resume(
        &workspace, &sessions[0], vade_harness::agent::Limits::default()).unwrap();
    let contents: Vec<String> = conversation.messages().iter()
        .filter(|message| message.role() == "assistant")
        .filter_map(|message| message.value().get("content").and_then(|value| value.as_str()).map(str::to_owned))
        .collect();
    assert!(contents.iter().any(|text| text.ends_with("\n[interrupted]")),
        "reopened journal assistant messages: {contents:?}");
    drop(conversation); drop(_store);
    drop(workspace);
    std::fs::remove_dir_all(root).unwrap();
}
#[cfg(target_os = "macos")]
#[test]
fn native_interrupt_mid_stream() { native_case(false, false, false); }
#[cfg(target_os = "macos")]
#[test]
fn native_interrupt_before_first_token() { native_case(true, false, false); }
#[cfg(target_os = "macos")]
#[test]
fn native_two_interrupts() { native_case(false, true, false); }
#[cfg(target_os = "macos")]
#[test]
fn native_exit_cancels_generation() { native_case(false, false, true); }

#[test]
fn failed_cancel_ack_holds_the_steering_message() {
    let mock = Mock::with_ack(false); let mut cli = mock.cli(); cli.send("original\n");
    let (mut first, body) = mock.request(); stream(&mut first, "PART_FAIL");
    cli.until(false, "PART_FAIL"); cli.send("replacement\n"); mock.cancelled(&body);
    cli.until(true, "not acknowledged");
    assert!(mock.requests.try_recv().is_err());
    cli.send("/exit\n"); assert!(!cli.finish().status.success());
}

#[test]
fn overflow_keeps_the_steering_correction_held() {
    // A partial answer past direct-history capacity must not delete the
    // correction that interrupted it; it stays held for /new then /next.
    let mock = Mock::new(); let mut cli = mock.cli(); cli.send("original\n");
    let (mut first, body) = mock.request();
    for chunk in 0..41 {
        let receipt = format!("OVERFLOW_CHUNK_{chunk:02}");
        let text = format!("{}{receipt}", "x".repeat(8192));
        if chunk == 0 { stream(&mut first, &text); }
        else {
            write!(first, "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n", json::quote(&text)).unwrap();
        }
        // Each receipt proves the UI consumed this chunk before another is
        // sent. The 64 KiB render lane cannot drop the final marker even if
        // the producer would otherwise outrun the UI.
        cli.until(false, &receipt);
    }
    cli.send("correction after overflow\n"); mock.cancelled(&body);
    cli.until(true, "exceeds history capacity");
    assert!(mock.requests.try_recv().is_err(), "held correction must not auto-send");
    cli.send("/new\n/next\n");
    let (mut next, body) = mock.request();
    assert_eq!(&history(&body)[1..], &[("user", "correction after overflow")]);
    finish(&mut next); drop(next); cli.until(false, "FINAL");
    drop(cli);
}

#[test]
fn pruning_keeps_the_steering_correction_held() {
    // Same preservation when the interrupted exchange fits but recording it
    // prunes the oldest of the 32 retained direct turns.
    let mock = Mock::new(); let mut cli = mock.cli();
    for turn in 0..32 {
        cli.send(&format!("turn {turn}\n"));
        let (mut socket, _) = mock.request();
        finish(&mut socket); drop(socket);
        // The status line is printed inside completion handling, after the
        // pending turn is taken; FINAL alone is only a streamed delta and
        // the next line could still race the completion event.
        cli.until(true, "╰");
    }
    cli.send("original\n");
    let (mut first, body) = mock.request(); stream(&mut first, "PART_PRUNE");
    cli.until(false, "PART_PRUNE");
    cli.send("correction after pruning\n"); mock.cancelled(&body);
    cli.until(true, "exceeds history capacity");
    assert!(mock.requests.try_recv().is_err(), "held correction must not auto-send");
    cli.send("/new\n/next\n");
    let (mut next, body) = mock.request();
    assert_eq!(&history(&body)[1..], &[("user", "correction after pruning")]);
    finish(&mut next); drop(next); cli.until(false, "FINAL");
    drop(cli);
}

#[cfg(target_os = "macos")]
#[test]
fn native_steering_during_admission_preserves_both_prompts() {
    // The second prompt arrives while the first is still being counted for
    // admission; both must reach generation, in the order typed.
    let mock = Mock::with_admission();
    let root = std::env::temp_dir().join(format!("vade-native-admission-steer-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let mut ui = raw_ui::Ui::start("agent", &mock.endpoint, root.to_str().unwrap(), false);
    ui.resize(24, 110);
    ui.send(b"first task\r");
    let (first_count, body) = mock.admission();
    assert_eq!(users(&body), ["first task"], "first count holds only the first prompt");
    ui.send(b"second task\r");
    // Keep the first count unanswered: only local cancellation of that count
    // can produce what comes next, so this cannot race into generation.
    let (mut second_count, body) = mock.admission();
    drop(first_count);
    assert_eq!(users(&body), ["first task", "second task"],
        "admission must preserve both prompts in order");
    admit(&mut second_count, &body); drop(second_count);
    let (mut generation, body) = mock.generation();
    assert_eq!(users(&body), ["first task", "second task"],
        "generation must preserve both prompts in order");
    assert!(mock.cancels.try_recv().is_err(), "counting has no server generation to cancel");
    finish(&mut generation); drop(generation);
    ui.until("FINAL", Duration::from_secs(5));
    ui.send(b"/exit\r");
    ui.wait_exit(); drop(ui);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn native_ctrl_d_cancels_generation_before_exit() {
    // Ctrl+D on an empty editor must cancel like /exit, not wait for a
    // generation that may never finish.
    let mock = Mock::new();
    let root = std::env::temp_dir().join(format!("vade-native-ctrl-d-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let mut ui = raw_ui::Ui::start("agent", &mock.endpoint, root.to_str().unwrap(), false);
    ui.resize(24, 110);
    ui.send(b"original\r");
    let (mut first, body) = mock.request();
    stream(&mut first, "PART_CTRL_D");
    ui.until("PART_CTRL_D", Duration::from_secs(5));
    ui.send(b"\x04");
    mock.cancelled(&body);
    ui.wait_exit(); drop(ui);
    assert_eq!(first.read(&mut [0]).unwrap(), 0);
    assert!(mock.requests.try_recv().is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn cancel_lands_within_one_token_interval_and_nothing_is_duplicated() {
    // The mock holds its next token back on a two-second cadence; steering
    // must cancel long before another token could land, keep exactly the
    // tokens already shown once, and mark them interrupted once.
    let mock = Mock::new(); let mut cli = mock.cli(); cli.send("original\n");
    let (mut first, body) = mock.request(); stream(&mut first, "TOK_ONE");
    cli.until(false, "TOK_ONE");
    let steered = std::time::Instant::now();
    cli.send("steer now\n");
    assert_eq!(mock.cancels.recv_timeout(Duration::from_secs(1)).unwrap(),
        body.get("cancel").unwrap().as_str().unwrap(),
        "cancel must arrive within one token interval");
    assert!(steered.elapsed() < Duration::from_secs(1), "cancel took {:?}", steered.elapsed());
    let (mut second, body) = mock.request();
    assert_eq!(&history(&body)[1..], &[("user", "original"), ("assistant", "TOK_ONE\n[interrupted]"), ("user", "steer now")]);
    assert_eq!(first.read(&mut [0]).unwrap(), 0, "generation socket must be closed");
    finish(&mut second); drop(second); cli.until(false, "FINAL");
    let output = cli.finish(); assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.matches("TOK_ONE").count(), 1, "partial printed once: {stdout}");
    assert_eq!(stdout.matches("[interrupted]").count(), 1, "marked once: {stdout}");
}

#[test]
fn two_quick_messages_become_one_steering_turn_in_order() {
    let mock = Mock::new(); let mut cli = mock.cli(); cli.send("original\n");
    let (mut first, body) = mock.request(); stream(&mut first, "PART_QUICK");
    cli.until(false, "PART_QUICK");
    cli.send("steer one\nsteer two\n");
    mock.cancelled(&body);
    let (mut second, body) = mock.request();
    assert_eq!(&history(&body)[1..], &[
        ("user", "original"),
        ("assistant", "PART_QUICK\n[interrupted]"),
        ("user", "steer one\n\nsteer two"),
    ]);
    assert!(mock.cancels.try_recv().is_err(), "exactly one server cancellation");
    assert_eq!(first.read(&mut [0]).unwrap(), 0);
    finish(&mut second); drop(second); cli.until(false, "FINAL");
    let output = cli.finish(); assert!(output.status.success(), "{output:?}");
    assert!(mock.requests.try_recv().is_err(), "no third generation");
}

#[test]
fn steering_to_a_dead_server_fails_fast_with_a_clear_line() {
    let mut mock = Mock::new(); let mut cli = mock.cli(); cli.send("original\n");
    let (mut first, _body) = mock.request(); stream(&mut first, "PART_DEAD");
    cli.until(false, "PART_DEAD");
    mock.kill_server();
    let steered = std::time::Instant::now();
    cli.send("steer after death\n");
    // Connect-refused cancellation is quick; the bounded Cli::finish below
    // also proves /exit cannot hang on the dead endpoint.
    cli.until(true, "not acknowledged");
    assert!(steered.elapsed() < Duration::from_secs(5), "dead-server steering took {:?}", steered.elapsed());
    cli.send("/exit\n");
    let output = cli.finish();
    assert!(!output.status.success(), "held steering to a dead server is a failure: {output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PART_DEAD") && stdout.contains("[interrupted]"),
        "partial answer kept and marked: {stdout}");
}
