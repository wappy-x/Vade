//! First-party loopback mocks only. Never discover or connect an engine.
use super::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

const CAPS: &str = include_str!("../../tests/fixtures/engine-capabilities-v1.json");

fn listen() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    (listener, endpoint)
}
fn accept(listener: &TcpListener) -> BufReader<TcpStream> {
    let end = Instant::now() + Duration::from_secs(5);
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                return BufReader::new(stream);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < end => {
                thread::park_timeout(Duration::from_millis(1))
            }
            other => panic!("loopback accept failed: {other:?}"),
        }
    }
}
fn request(reader: &mut BufReader<TcpStream>) -> (String, String) {
    let mut start = String::new();
    reader.read_line(&mut start).unwrap();
    assert!(start.len() < 1024);
    let mut length = 0;
    let mut total = start.len();
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        total += line.len();
        assert!(total < 16 * 1024);
        if line == "\r\n" {
            break;
        }
        if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = n.trim().parse().unwrap();
        }
    }
    assert!(length < 2 * 1024 * 1024);
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    (
        start.trim_end().to_owned(),
        String::from_utf8(body).unwrap(),
    )
}
fn respond(reader: &mut BufReader<TcpStream>, status: u16, body: &str) {
    write!(reader.get_mut(), "HTTP/1.1 {status} Mock\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
}
fn prepare(listener: &TcpListener, caps: &str) -> (BufReader<TcpStream>, String) {
    {
        let mut reader = accept(listener);
        assert_eq!(
            request(&mut reader),
            ("GET /v1/capabilities HTTP/1.1".into(), String::new())
        );
        respond(&mut reader, 200, caps);
    }
    let mut reader = accept(listener);
    let (line, envelope) = request(&mut reader);
    assert_eq!(line, "POST /v1/admission HTTP/1.1");
    let envelope = json::view(&envelope).unwrap();
    let output = envelope.get("output_tokens").unwrap().raw();
    let admitted = envelope.get("request").unwrap().raw().to_owned();
    respond(
        &mut reader,
        200,
        &format!(
            r#"{{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","prompt_tokens":50,"output_tokens":{output},"context_limit":4096,"input_limit":4000,"output_limit":32,"admitted":true,"admission_kind":"context_fit","reserves_resources":false}}"#
        ),
    );
    (reader, admitted)
}
fn response(message: Value, completion_tokens: u64) -> String {
    let finish = if message.get("tool_calls").is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    format!(
        r#"{{"choices":[{{"message":{},"finish_reason":"{finish}"}}],"usage":{{"prompt_tokens":50,"completion_tokens":{completion_tokens},"total_tokens":{},"prompt_tokens_details":{{"cached_tokens":20}}}}}}"#,
        json::stringify(&message).unwrap(),
        50 + completion_tokens
    )
}
fn serve(listener: &TcpListener, caps: &str, message: Value, output: u64) -> String {
    let (reader, admitted) = prepare(listener, caps);
    drop(reader);
    let mut reader = accept(listener);
    let (line, body) = request(&mut reader);
    assert_eq!(line, "POST /v1/chat/completions HTTP/1.1");
    assert_eq!(
        body, admitted,
        "generation must use exactly the admitted bytes"
    );
    respond(&mut reader, 200, &response(message, output));
    body
}
fn http_factory(endpoint: String) -> Arc<Factory> {
    Arc::new(move |_, _| {
        Ok(ChildWorker {
            observer: None,
            journal_name: None,
            model: Box::new(HttpChildModel::new(&endpoint)?),
            tools: Box::new(ReadTools),
        })
    })
}
fn http_task() -> Task {
    let mut task = Task::read_only("mock HTTP task");
    task.quota.output_tokens_per_turn = 16;
    task
}

#[test]
fn exact_http_child_prefix_and_affinity_are_independent_from_parent_and_sibling() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        vec![
            serve(&listener, CAPS, answer("parent"), 3),
            serve(
                &listener,
                CAPS,
                proposal("r1", "workspace_read", val(r#"{"path":"sample"}"#)),
                3,
            ),
            serve(&listener, CAPS, answer("first child"), 3),
            serve(&listener, CAPS, answer("second child"), 3),
        ]
    });
    let parent = parent();
    let mut native = crate::transport::NativeModel::new(&endpoint).unwrap();
    native
        .complete(
            parent.request(WireLimits::default()),
            &AtomicBool::new(false),
            &mut |_| Ok(()),
        )
        .unwrap();
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        http_factory(endpoint),
    );
    let a = scheduler.spawn(http_task()).unwrap();
    let snapshot = finished(&scheduler, a);
    assert_eq!(snapshot.usage.completed_responses, 2);
    assert_eq!(snapshot.usage.responses_with_cached_usage, 2);
    assert_eq!(snapshot.usage.reported_cached_tokens, 40);
    assert_eq!(snapshot.usage.last.unwrap().cached_tokens, Some(20));
    let first = scheduler.take_result(a).unwrap();
    assert_eq!(first.outcome, Outcome::Completed);
    assert_eq!(first.tokens_reserved, 132);
    assert_eq!(first.last_usage.unwrap().cached_tokens, Some(20));
    let b = scheduler.spawn(http_task()).unwrap();
    finished(&scheduler, b);
    assert_eq!(
        scheduler.take_result(b).unwrap().outcome,
        Outcome::Completed
    );
    let bodies = server.join().unwrap();
    let parsed: Vec<Value> = bodies
        .iter()
        .map(|body| json::parse(body).unwrap())
        .collect();
    let key = |i: usize| parsed[i].get("prompt_cache_key").unwrap().as_str().unwrap();
    assert_ne!(key(0), key(1));
    assert_eq!(key(1), key(2));
    assert_ne!(key(1), key(3));
    for child in &parsed[1..] {
        assert_eq!(child.get("cache"), Some(&Value::Bool(true)));
        assert_eq!(child.get("tools"), parsed[0].get("tools"));
        let Value::Array(messages) = child.get("messages").unwrap() else {
            unreachable!()
        };
        let Value::Array(original) = parsed[0].get("messages").unwrap() else {
            unreachable!()
        };
        assert_eq!(&messages[..2], &original[..2]);
        assert!(
            !json::stringify(child)
                .unwrap()
                .contains("private parent history")
        );
    }
    // Exact serialized child history is append-only over its tool round.
    assert!(bodies[2].starts_with(&bodies[1][..bodies[1].len() - 2]));
}

#[test]
fn affinity_is_omitted_unless_contract_advertises_it() {
    let (listener, endpoint) = listen();
    let caps = CAPS.replace("\"prompt_cache_key\": \"scheduling_hint\",", "");
    let server = thread::spawn(move || serve(&listener, &caps, answer("done"), 1));
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        http_factory(endpoint),
    );
    let id = scheduler.spawn(http_task()).unwrap();
    finished(&scheduler, id);
    assert_eq!(
        scheduler.take_result(id).unwrap().outcome,
        Outcome::Completed
    );
    assert!(
        json::parse(&server.join().unwrap())
            .unwrap()
            .get("prompt_cache_key")
            .is_none()
    );
}

#[test]
fn token_quota_denies_generation_after_exact_admission() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let _ = prepare(&listener, CAPS);
        listener
    });
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        http_factory(endpoint),
    );
    let mut task = http_task();
    task.quota.tokens = 60;
    let id = scheduler.spawn(task).unwrap();
    finished(&scheduler, id);
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.outcome, Outcome::QuotaTokens);
    assert_eq!(result.tokens_reserved, 0);
    assert_eq!(
        server.join().unwrap().accept().unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
}

#[test]
fn unavailable_exact_count_stops_child_without_generation() {
    for advertised in [true, false] {
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            assert_eq!(request(&mut reader).0, "GET /v1/capabilities HTTP/1.1");
            if advertised {
                respond(
                    &mut reader,
                    200,
                    &CAPS.replace("\"admission\": \"/v1/admission\",", ""),
                );
            } else {
                respond(&mut reader, 404, "{}");
            }
            listener
        });
        let mut scheduler = scheduler(
            Config::default(),
            Arc::new(AtomicBool::new(false)),
            http_factory(endpoint),
        );
        let id = scheduler.spawn(http_task()).unwrap();
        finished(&scheduler, id);
        let result = scheduler.take_result(id).unwrap();
        assert_eq!(
            result.outcome,
            Outcome::Runtime(StopReason::ModelError(io::ErrorKind::Unsupported))
        );
        assert_eq!(result.tokens_reserved, 0);
        assert_eq!(
            server.join().unwrap().accept().unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
}

#[test]
fn contradictory_usage_refuses_model_tools_instead_of_claiming_quota_success() {
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        serve(
            &listener,
            CAPS,
            proposal("over-budget", "workspace_read", val(r#"{"path":"x"}"#)),
            17,
        )
    });
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        http_factory(endpoint),
    );
    let id = scheduler.spawn(http_task()).unwrap();
    let snapshot = finished(&scheduler, id);
    assert_eq!(snapshot.usage, ChildUsage::default());
    assert!(ui::context(&[snapshot]).contains("reported cached tokens: unknown across 0/0"));
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(
        result.outcome,
        Outcome::Runtime(StopReason::ModelError(io::ErrorKind::InvalidData))
    );
    assert!(result.tools.is_empty());
    assert_eq!(result.tokens_reserved, 66);
    server.join().unwrap();
}

#[test]
fn parent_cancel_interrupts_pending_child_http_without_inventing_remote_ack() {
    let (listener, endpoint) = listen();
    let (tx, rx) = mpsc::sync_channel(1);
    let server = thread::spawn(move || {
        let (reader, admitted) = prepare(&listener, CAPS);
        drop(reader);
        let mut reader = accept(&listener);
        assert_eq!(request(&mut reader).1, admitted);
        tx.send(()).unwrap();
        assert_eq!(
            reader.read(&mut [0]).unwrap(),
            0,
            "cancelled child must close its socket"
        );
    });
    let cancel = Arc::new(AtomicBool::new(false));
    let mut scheduler = scheduler(
        Config::default(),
        Arc::clone(&cancel),
        http_factory(endpoint),
    );
    let id = scheduler.spawn(http_task()).unwrap();
    rx.recv_timeout(Duration::from_secs(3)).unwrap();
    cancel.store(true, Ordering::Release);
    finished(&scheduler, id);
    let result = scheduler.take_result(id).unwrap();
    assert_eq!(result.outcome, Outcome::Cancelled);
    assert_eq!(result.stop_requested, Some(StopRequest::ParentCancelled));
    assert_eq!(result.tokens_reserved, 66);
    server.join().unwrap();
}
