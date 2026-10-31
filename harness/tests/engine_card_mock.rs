//! The engine card (`quality-card --set engine-1`) against a first-party
//! loopback SSE server: every check passes on a clean server, and each
//! failure mode the card exists to catch fails its own check and grade.
//! The CLI test runs the real binary for the files and the exit code.
#[path = "support/home.rs"]
mod home;
use std::{
    collections::{HashMap, HashSet},
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use vade_harness::json::{self, Value};
use vade_harness::quality::engine_cases::{
    CACHE_PROMPTS, CONCURRENCY_PROMPTS, CONCURRENT_COPIES, JSON_CASES, NEEDLES, PROBE_SENTENCES, TOOL_CASES,
};
use vade_harness::quality::{CardSet, Config, EngineReport, Grade, run_engine};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
/// After the last copy arrives, how long the copies keep their tail so each
/// client thread can read its head first (see `handle`).
const CLIENT_READ_MARGIN: Duration = Duration::from_millis(250);

/// Which defects the mock server shows.
#[derive(Clone, Copy, Default)]
struct Mock {
    /// JSON items answered inside a code fence (count from the first).
    fenced_json: usize,
    /// Tool items answered with a wrong argument (count from the first).
    wrong_tools: usize,
    /// The 32K needle answer misses the code.
    miss_long_needle: bool,
    /// Warm runs of cache prompt 5 change one token id.
    warm_differs: bool,
    /// `"cache": false` is ignored and a previous run left every cache
    /// prompt stored, so even the cold runs report cached tokens.
    cache_flag_ignored: bool,
    /// A request that starts while 4 others are in flight changes one id.
    batch_differs: bool,
    /// Requests run one at a time (no overlap).
    serial: bool,
    /// The server context in tokens, as Vade checks it: a chat prompt of
    /// this many tokens or more gets HTTP 400, and `/v1/admission` reports
    /// it. None admits every prompt.
    context: Option<u64>,
    /// The server has no `/v1/admission` route (HTTP 404).
    no_admission: bool,
}

struct State {
    mock: Mock,
    seen: Mutex<HashSet<String>>,
    in_flight: AtomicUsize,
    one_at_a_time: Mutex<()>,
    json_count: AtomicUsize,
    tool_count: AtomicUsize,
    /// Arrivals per concurrency prompt: the first is the run alone, the
    /// next `CONCURRENT_COPIES` are the copies. Never decremented.
    arrivals: Mutex<HashMap<String, usize>>,
    arrived: Condvar,
    /// Chat requests refused at the context limit with HTTP 400.
    refused: Arc<AtomicUsize>,
}

/// One value the schema accepts (first enum choice, fixed value per type).
fn instance(schema: &Value) -> Value {
    if let Some(first) = schema.get("enum").and_then(Value::as_array).and_then(|c| c.first()) {
        return first.clone();
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("object") => Value::Object(match schema.get("properties") {
            Some(Value::Object(properties)) => {
                properties.iter().map(|(key, sub)| (key.clone(), instance(sub))).collect()
            }
            _ => Vec::new(),
        }),
        Some("array") => Value::Array(vec![Value::String("item".into())]),
        Some("integer") => Value::Number("7".into()),
        Some("number") => Value::Number("2.5".into()),
        Some("boolean") => Value::Bool(true),
        _ => Value::String("text".into()),
    }
}

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

fn chunk(delta: &str, token_id: u32) -> String {
    format!(
        "data: {{\"id\":\"completion-9\",\"object\":\"chat.completion.chunk\",\"model\":\"mock\",\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":null,\"token_id\":{token_id}}}]}}\r\n\r\n"
    )
}

/// The mock's prompt count: 4 bytes of the last user message per token.
fn user_tokens(body: &Value) -> u64 {
    let messages = body.get("messages").and_then(Value::as_array).expect("messages");
    let user = messages.last().and_then(|m| m.get("content")).and_then(Value::as_str).expect("user");
    (user.len() as u64 / 4).max(1)
}

fn reply_json(stream: &mut TcpStream, status: &str, body: &str) {
    let head = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    stream.write_all(head.as_bytes()).expect("write head");
    stream.write_all(body.as_bytes()).expect("write body");
    stream.flush().expect("flush");
}

/// `/v1/admission` as Vade answers it: the exact chat body is counted and
/// checked against the context with `output_tokens` of answer room.
fn admission(stream: &mut TcpStream, state: &State, raw: &[u8]) {
    if state.mock.no_admission {
        return reply_json(stream, "404 Not Found", "{\"error\":{\"message\":\"not found\"}}");
    }
    let body = json::parse(std::str::from_utf8(raw).expect("utf8")).expect("admission JSON");
    assert_eq!(body.get("endpoint").and_then(Value::as_str), Some("/v1/chat/completions"));
    let output = body.get("output_tokens").and_then(|v| match v {
        Value::Number(text) => text.parse::<u64>().ok(),
        _ => None,
    });
    let request = body.get("request").expect("admission request");
    assert_eq!(request.get("temperature"), Some(&Value::Number("0".into())), "the exact chat body");
    let (tokens, context) = (user_tokens(request), state.mock.context.unwrap_or(1 << 30));
    let admitted = tokens < context && tokens + output.expect("output_tokens") <= context;
    reply_json(
        stream,
        "200 OK",
        &format!(
            "{{\"object\":\"engine.admission\",\"prompt_tokens\":{tokens},\"output_tokens\":{},\"context_limit\":{context},\"input_limit\":{},\"output_limit\":8192,\"admitted\":{admitted},\"reason\":{}}}",
            output.unwrap(),
            context - 1,
            if admitted { "null" } else { "\"Requested size does not fit.\"" }
        ),
    );
}

fn handle(stream: TcpStream, state: &State) {
    let mut reader = BufReader::new(stream);
    let mut length = 0usize;
    let mut request_line = String::new();
    reader.read_line(&mut request_line).expect("request line");
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
        }
    }
    let mut raw = vec![0u8; length];
    reader.read_exact(&mut raw).expect("body");
    if request_line.starts_with("POST /v1/admission ") {
        return admission(reader.get_mut(), state, &raw);
    }
    assert!(request_line.starts_with("POST /v1/chat/completions "), "{request_line}");
    let body = json::parse(std::str::from_utf8(&raw).expect("utf8")).expect("request JSON");
    assert_eq!(body.get("temperature"), Some(&Value::Number("0".into())), "greedy pinned");
    assert!(body.get("draft").is_none(), "drafting as served");
    let messages = body.get("messages").and_then(Value::as_array).expect("messages");
    let user = messages.last().and_then(|m| m.get("content")).and_then(Value::as_str).expect("user").to_owned();
    if !user.contains("<notes>") {
        // Only the prime and warm runs of the cache check may use the cache.
        assert_eq!(body.get("cache"), Some(&Value::Bool(false)), "every other request runs fresh");
    }
    // The served prefix cache is opt-in: only "cache": true reads or stores.
    let cache_on = body.get("cache") == Some(&Value::Bool(true));
    let mock = state.mock;
    if mock.context.is_some_and(|context| user_tokens(&body) >= context) {
        // Vade refuses a prompt that leaves no room for one output token.
        state.refused.fetch_add(1, Ordering::SeqCst);
        return reply_json(reader.get_mut(), "400 Bad Request", "{\"error\":{\"message\":\"Requested size does not fit.\",\"code\":\"context_length_exceeded\"}}");
    }
    let _serial = mock.serial.then(|| state.one_at_a_time.lock().expect("serial"));
    let running = state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;

    let mut prompt_tokens = user.len() as u64 / 4;
    let mut cached = 0u64;
    let mut parts: Vec<String> = Vec::new();
    let mut finish = "stop";
    let base = (fnv(&user) % 50_000) as u32;
    let mut ids: Vec<u32> = (0..4).map(|i| base + i).collect();
    let mut hold = false;
    if body.get("tools").is_some() {
        let index = state.tool_count.fetch_add(1, Ordering::SeqCst);
        let case = TOOL_CASES.iter().find(|case| case.prompt == user).expect("known tool prompt");
        let arguments = if index < mock.wrong_tools { case.arguments.replacen('"', "\"x", 3) } else { case.arguments.to_owned() };
        let (head, tail) = arguments.split_at(arguments.len() / 2);
        parts.push(format!(
            "{{\"tool_calls\":[{{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{{\"name\":\"{}\",\"arguments\":\"\"}}}}]}}",
            case.name
        ));
        for piece in [head, tail] {
            parts.push(format!("{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":{}}}}}]}}", json::quote(piece)));
        }
        finish = "tool_calls";
    } else if let Some(format) = body.get("response_format") {
        let index = state.json_count.fetch_add(1, Ordering::SeqCst);
        let schema = format.get("json_schema").and_then(|f| f.get("schema")).expect("json_schema schema");
        let mut text = json::stringify(&instance(schema)).expect("instance");
        if index < mock.fenced_json {
            text = format!("```json\n{text}\n```");
        }
        parts.push(format!("{{\"content\":{}}}", json::quote(&text)));
    } else if user.contains("<document>") {
        let mut answer = NEEDLES.iter().find(|needle| user.contains(needle.answer)).map_or("none", |needle| needle.answer);
        if mock.miss_long_needle && answer == NEEDLES[1].answer {
            answer = "COBALT-2846";
        }
        parts.push(format!("{{\"content\":{}}}", json::quote(answer)));
    } else if user.contains("<notes>") {
        let repeat = !state.seen.lock().expect("seen").insert(user.clone());
        if repeat && (cache_on || mock.cache_flag_ignored) {
            cached = prompt_tokens / 64 * 64;
            if mock.warm_differs && user == vade_harness::quality::engine_cases::cache_prompt(4) {
                ids[2] += 1;
            }
        }
        parts.push("{\"content\":\"The notes name rivers.\"}".to_owned());
    } else {
        // The copies (not the run alone) hold their tail until all of them
        // have arrived and sent their head. The card times the overlap at the
        // client (first token read before the first copy ends), which no server
        // condition can order, so the tails then wait a margin for the client
        // threads to read the heads. A serial server cannot overlap, so it does
        // not wait.
        let mut arrivals = state.arrivals.lock().expect("arrivals");
        let seen = arrivals.entry(user.clone()).or_insert(0);
        *seen += 1;
        hold = *seen > 1 && !mock.serial;
        state.arrived.notify_all();
        drop(arrivals);
        if mock.batch_differs && running >= 5 {
            ids[3] += 1;
        }
        parts.push("{\"content\":\"A steady answer.\"}".to_owned());
    }
    if prompt_tokens == 0 {
        prompt_tokens = 1;
    }
    let mut events = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
    );
    for (at, id) in ids.iter().enumerate() {
        let delta = parts.get(at).cloned().unwrap_or_else(|| "{}".to_owned());
        events.push_str(&chunk(&delta, *id));
    }
    let stream = reader.get_mut();
    stream.write_all(events.as_bytes()).expect("write head");
    stream.flush().expect("flush");
    if hold {
        let arrivals = state.arrivals.lock().expect("arrivals");
        let (arrivals, waited) = state
            .arrived
            .wait_timeout_while(arrivals, Duration::from_secs(20), |a| a.get(&user).copied().unwrap_or(0) < 1 + CONCURRENT_COPIES)
            .expect("arrivals wait");
        assert!(!waited.timed_out(), "all {CONCURRENT_COPIES} copies arrived: {:?}", arrivals.get(&user));
        drop(arrivals);
        thread::sleep(CLIENT_READ_MARGIN);
    }
    state.in_flight.fetch_sub(1, Ordering::SeqCst);
    let tail = format!(
        "data: {{\"id\":\"completion-9\",\"object\":\"chat.completion.chunk\",\"model\":\"mock-served\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish}\"}}],\"usage\":{{\"prompt_tokens\":{prompt_tokens},\"completion_tokens\":{},\"total_tokens\":0,\"prompt_tokens_details\":{{\"cached_tokens\":{cached}}}}},\"vade\":{{\"draft\":\"auto\",\"mtp_depth\":3,\"thinking\":false,\"draft_used\":true,\"model_draft_used\":true,\"build\":{{\"git_sha\":\"{SHA}\",\"dirty\":false,\"source\":\"git\"}},\"numerics\":{{\"nvfp4\":true,\"mtp_depth\":3,\"world_size\":1,\"topology\":\"local\"}}}}}}\r\n\r\ndata: [DONE]\r\n\r\n",
        ids.len()
    );
    stream.write_all(tail.as_bytes()).expect("write tail");
    stream.flush().expect("flush");
}

/// Serve until `stop`, one thread per connection so copies run together.
fn serve(listener: TcpListener, mock: Mock, stop: Arc<AtomicBool>, accepts: Arc<AtomicUsize>, refused: Arc<AtomicUsize>) {
    listener.set_nonblocking(true).expect("nonblocking listener");
    let state = Arc::new(State {
        mock,
        seen: Mutex::new(if mock.cache_flag_ignored {
            (0..CACHE_PROMPTS).map(vade_harness::quality::engine_cases::cache_prompt).collect()
        } else {
            HashSet::new()
        }),
        in_flight: AtomicUsize::new(0),
        one_at_a_time: Mutex::new(()),
        json_count: AtomicUsize::new(0),
        tool_count: AtomicUsize::new(0),
        arrivals: Mutex::new(HashMap::new()),
        arrived: Condvar::new(),
        refused,
    });
    let deadline = Instant::now() + Duration::from_secs(240);
    let mut workers = Vec::new();
    while !stop.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "mock server ran too long");
        match listener.accept() {
            Ok((stream, _)) => {
                accepts.fetch_add(1, Ordering::SeqCst);
                stream.set_nonblocking(false).expect("blocking stream");
                let state = state.clone();
                workers.push(thread::spawn(move || handle(stream, &state)));
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(2)),
            Err(error) => panic!("accept: {error}"),
        }
    }
    for worker in workers {
        worker.join().expect("mock worker");
    }
}

/// Start the mock; the counters hold the connections it accepted and the
/// chat requests it refused at the context limit.
fn start(mock: Mock) -> (String, Arc<AtomicBool>, thread::JoinHandle<()>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    let stop = Arc::new(AtomicBool::new(false));
    let accepts = Arc::new(AtomicUsize::new(0));
    let refused = Arc::new(AtomicUsize::new(0));
    let server = thread::spawn({
        let (stop, accepts, refused) = (stop.clone(), accepts.clone(), refused.clone());
        move || serve(listener, mock, stop, accepts, refused)
    });
    (endpoint, stop, server, accepts, refused)
}

fn run_against(mock: Mock) -> EngineReport {
    run_counted(mock).0
}

/// One run and the chat requests the mock refused at its context limit.
fn run_counted(mock: Mock) -> (EngineReport, usize) {
    let (endpoint, stop, server, _, refused) = start(mock);
    let config = Config {
        endpoint,
        model: "mock".into(),
        json_path: "unused.json".into(),
        request_seconds: 30,
        set: CardSet::Engine,
        ..Config::default()
    };
    let report = run_engine(config).expect("run");
    stop.store(true, Ordering::SeqCst);
    server.join().expect("server thread");
    let refused = refused.load(Ordering::SeqCst);
    (report, refused)
}

#[test]
fn engine_card_grades_a_clean_server_a_plus() {
    let report = run_against(Mock::default());
    assert_eq!(report.needle_score(), (2, 2), "{:?}", report.needles.iter().map(|n| n.failure_reason()).collect::<Vec<_>>());
    assert!(report.needles.iter().all(|item| item.reply.as_ref().and_then(|r| r.prompt_tokens) >= Some(item.needle.target_tokens)));
    assert_eq!(report.json_score(), (50, 50));
    assert_eq!(report.tool_score(), (20, 20));
    assert_eq!(report.cache_score(), (CACHE_PROMPTS, CACHE_PROMPTS));
    assert_eq!(report.concurrency_score(), (24, 24));
    assert!(report.concurrency.iter().all(|item| item.overlapped()));
    assert_eq!(report.grade(), Grade::APlus);
    assert_eq!(report.server.git_sha.as_deref(), Some(SHA));
    assert_eq!(report.server.world_size, Some(1));
    let card = report.card();
    assert!(card.contains("GRADE A+  ·  5/5 CHECKS PASS"), "{card}");
    assert!(card.contains(SHA), "{card}");
    assert!(card.contains("MTP on, depth 3 (draft auto)"), "{card}");
    let receipt = json::parse(&report.to_json().expect("receipt")).expect("receipt parses");
    assert_eq!(receipt.get("grade").and_then(Value::as_str), Some("A+"));
}

#[test]
fn engine_card_soft_misses_grade_a_then_b() {
    let one = run_against(Mock { fenced_json: 1, ..Mock::default() });
    assert_eq!(one.json_score(), (49, 50));
    assert_eq!(one.grade(), Grade::A);
    assert!(one.card().contains("✗ json person-1: not one JSON value"));
    let two = run_against(Mock { fenced_json: 1, wrong_tools: 1, ..Mock::default() });
    assert_eq!((two.json_score(), two.tool_score()), ((49, 50), (19, 20)));
    assert_eq!(two.grade(), Grade::B);
    assert!(two.card().contains("✗ tool weather-lisbon: arguments"), "{}", two.card());
}

#[test]
fn engine_card_two_tool_misses_grade_c() {
    let report = run_against(Mock { wrong_tools: 2, ..Mock::default() });
    assert_eq!(report.tool_score(), (18, 20));
    assert_eq!(report.grade(), Grade::C);
}

#[test]
fn engine_card_missed_needle_is_hard() {
    let report = run_against(Mock { miss_long_needle: true, ..Mock::default() });
    assert_eq!(report.needle_score(), (1, 2));
    assert_eq!(report.grade(), Grade::C);
    assert!(report.needles[1].failure_reason().is_some_and(|r| r.contains("does not hold COBALT-2846-WILLOW")));
}

#[test]
fn engine_card_catches_a_warm_cache_id_change() {
    let report = run_against(Mock { warm_differs: true, ..Mock::default() });
    assert_eq!(report.cache_score(), (CACHE_PROMPTS - 1, CACHE_PROMPTS));
    assert!(report.cache[4].failure_reason().is_some_and(|r| r.contains("ids differ at token 2")));
    assert_eq!(report.grade(), Grade::C);
    let receipt = report.to_json().expect("receipt");
    assert!(receipt.contains("ids differ at token 2"));
}

#[test]
fn engine_card_refuses_a_cold_run_that_read_the_cache() {
    let report = run_against(Mock { cache_flag_ignored: true, ..Mock::default() });
    assert_eq!(report.cache_score(), (0, CACHE_PROMPTS));
    assert!(report.cache.iter().all(|item| item.failure_reason().is_some_and(|r| r.starts_with("cold run read"))));
    assert_eq!(report.grade(), Grade::C);
}

#[test]
fn engine_card_catches_a_batch_dependent_id() {
    let report = run_against(Mock { batch_differs: true, ..Mock::default() });
    // The copies arrive one by one and hold until all 8 are in, so in each
    // group exactly the 5th to the 8th start with 4 or more others in flight.
    assert!(report.concurrency.iter().all(|item| item.overlapped()));
    assert_eq!(report.concurrency_score(), (12, 24), "4 of 8 copies differ per prompt");
    assert_eq!(report.grade(), Grade::C);
    assert!(report.card().contains("ids differ at token 3"), "{}", report.card());
}

#[test]
fn engine_card_needs_copies_in_flight_together() {
    let report = run_against(Mock { serial: true, ..Mock::default() });
    assert!(report.concurrency.iter().all(|item| !item.overlapped()));
    assert_eq!(report.concurrency_score(), (0, 24));
    assert_eq!(report.grade(), Grade::C);
    assert!(report.card().contains("copies did not overlap"));
}

#[test]
fn engine_card_cli_writes_new_files_and_exits_on_grade() {
    let (endpoint, stop, server, accepts, _) = start(Mock::default());
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("engine-card-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("temp dir");
    let json_path = root.join("receipt.json");
    let card_path = root.join("card.txt");
    let run = |extra: &[&str]| {
        let mut args = vec!["quality-card", "--set", "engine-1", "--endpoint", &endpoint, "--model", "mock", "--request-seconds", "30"];
        args.extend_from_slice(extra);
        home::harness().args(&args).env_remove("VADE_API_KEY_FILE").output().expect("run binary")
    };
    let refused = run(&["--json", json_path.to_str().unwrap(), "--only", "quality"]);
    assert_eq!(refused.status.code(), Some(2));
    assert!(!json_path.exists(), "a refused run writes nothing");
    let existing = root.join("existing.json");
    std::fs::write(&existing, b"keep\n").expect("existing receipt");
    let kept = run(&["--json", existing.to_str().unwrap()]);
    assert_eq!(kept.status.code(), Some(2), "an existing receipt is never overwritten");
    assert_eq!(std::fs::read(&existing).expect("existing"), b"keep\n", "existing receipt bytes unchanged");
    let output = run(&["--json", json_path.to_str().unwrap(), "--card", card_path.to_str().unwrap()]);
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GRADE A+"), "{stdout}");
    assert_eq!(std::fs::read_to_string(&card_path).expect("card file"), stdout);
    // The mock accepted exactly one run of requests: neither refused run
    // above opened a connection (a queued one would be counted first).
    // Each needle is checked once at /v1/admission, then sent.
    let one_run = PROBE_SENTENCES.len()
        + NEEDLES.len() * 2
        + JSON_CASES.len() * 5
        + TOOL_CASES.len()
        + CACHE_PROMPTS * 3
        + CONCURRENCY_PROMPTS.len() * (1 + CONCURRENT_COPIES);
    assert_eq!(accepts.load(Ordering::SeqCst), one_run, "connections accepted");
    let receipt = json::parse(&std::fs::read_to_string(&json_path).expect("receipt")).expect("receipt parses");
    assert_eq!(receipt.get("schema").and_then(Value::as_str), Some("vade.engine-card.v1"));
    let (receipt_bytes, card_bytes) = (std::fs::read(&json_path).expect("receipt"), std::fs::read(&card_path).expect("card"));
    let accepted = accepts.load(Ordering::SeqCst);
    let again = run(&["--json", json_path.to_str().unwrap(), "--card", card_path.to_str().unwrap()]);
    assert_eq!(again.status.code(), Some(2), "an existing receipt is never overwritten");
    assert_eq!(accepts.load(Ordering::SeqCst), accepted, "a refused rerun opens no connection");
    assert_eq!(std::fs::read(&json_path).expect("receipt"), receipt_bytes, "receipt bytes unchanged");
    assert_eq!(std::fs::read(&card_path).expect("card"), card_bytes, "card bytes unchanged");
    stop.store(true, Ordering::SeqCst);
    server.join().expect("server thread");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn engine_card_does_not_send_a_needle_the_server_context_cannot_hold() {
    // vade up's default context: 32,768 tokens in all, prompt and answer.
    let (report, refused) = run_counted(Mock { context: Some(32_768), ..Mock::default() });
    assert_eq!(refused, 0, "no chat request reached the context limit");
    assert_eq!(report.needle_score(), (1, 2));
    assert!(report.needles[0].passed(), "{:?}", report.needles[0].failure_reason());
    assert!(report.needles[1].reply.is_none(), "the 32K needle was not sent");
    assert_eq!(report.grade(), Grade::C, "a 32K needle stays required for A+");
    let card = report.card();
    for text in ["needle 32K  NOT RUN (refused before sending)", "NOT RUN: the server admits at most", "vade up --context 40960"] {
        assert!(card.contains(text), "{text}: {card}");
    }
    // The other four checks still run in full.
    assert_eq!((report.json_score(), report.tool_score()), ((50, 50), (20, 20)));
    assert_eq!(report.concurrency_score(), (24, 24));
}

#[test]
fn engine_card_sizes_the_needle_to_the_server_context() {
    // The calibrated 32K prompt is over this context; the card steps the
    // filler down with the server's count and the needle runs and passes.
    let (report, refused) = run_counted(Mock { context: Some(33_400), ..Mock::default() });
    assert_eq!(refused, 0, "no chat request reached the context limit");
    assert_eq!(report.needle_score(), (2, 2), "{:?}", report.needles.iter().map(|n| n.failure_reason()).collect::<Vec<_>>());
    let tokens = report.needles[1].reply.as_ref().and_then(|r| r.prompt_tokens).expect("32K prompt tokens");
    assert!((32_768..33_400 - 32).contains(&tokens), "{tokens}");
    assert_eq!(report.grade(), Grade::APlus);
    // A server with a large context gets the calibrated size at once.
    let (wide, refused) = run_counted(Mock { context: Some(40_960), ..Mock::default() });
    assert_eq!((refused, wide.needle_score(), wide.grade()), (0, (2, 2), Grade::APlus));
}

#[test]
fn engine_card_without_an_admission_route_sends_as_calibrated() {
    let report = run_against(Mock { no_admission: true, ..Mock::default() });
    assert_eq!(report.needle_score(), (2, 2));
    assert_eq!(report.grade(), Grade::APlus);
    assert!(report.card().contains("sent unchecked"), "{}", report.card());
}
