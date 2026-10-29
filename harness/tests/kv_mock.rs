//! First-party stateful loopback model of the checkpoint wire contract. No
//! Engine code, model, tokenizer, GPU, or remote endpoint is used by these tests.
#[path = "support/home.rs"]
mod home;
use std::{
    collections::VecDeque,
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use vade_harness::{
    agent::{self, Conversation, Limits, ToolError, ToolExecutor, ToolInvocation},
    agent_wire::FinishReason,
    contract::PromptBoundary,
    journal::Journal,
    json::{self, Value},
    kv::Reuse,
    tools::Workspace,
    transport::NativeModel,
};

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
fn n(v: &Value, key: &str) -> u64 {
    match v.get(key).unwrap() {
        Value::Number(n) => n.parse().unwrap(),
        _ => panic!("number required"),
    }
}
struct Saved {
    handle: String,
    prompt: String,
    expiry: u64,
    bytes: u64,
    foreign: bool,
}
struct State {
    enabled: bool,
    domain: String,
    count: usize,
    bytes: u64,
    ttl: u64,
    tick: u64,
    serial: u64,
    entries: VecDeque<Saved>,
    evicted: usize,
    requests: Vec<Value>,
    released: Vec<String>,
    admission: Vec<String>,
    malformed: bool,
    mismatch: bool,
    truncated: bool,
    retention_failure: bool,
}
impl Default for State {
    fn default() -> Self {
        Self {
            enabled: true,
            domain: "mock-kv-epoch-1".into(),
            count: 8,
            bytes: 1_000_000,
            ttl: 300_000,
            tick: 0,
            serial: 0,
            entries: VecDeque::new(),
            evicted: 0,
            requests: Vec::new(),
            released: Vec::new(),
            admission: Vec::new(),
            malformed: false,
            mismatch: false,
            truncated: false,
            retention_failure: false,
        }
    }
}
// Deliberately simple authored framing. The generation suffix is significant:
// a system-only canonical input does NOT prefix a following user message.
fn render(v: &Value) -> String {
    let mut out = json::stringify(v.get("tools").unwrap_or(&Value::Null)).unwrap();
    for message in v.get("messages").unwrap().as_array().unwrap() {
        let role = message.get("role").unwrap().as_str().unwrap();
        out.push_str(&format!(
            "<{role}>{}</{role}>",
            message.get("content").and_then(Value::as_str).unwrap_or("")
        ));
    }
    out.push_str("<assistant>");
    out
}
impl State {
    fn reply(&mut self, method: &str, path: &str, body: &str) -> (u16, &'static str, String, bool) {
        if method == "GET" && path == "/v1/capabilities" {
            let base = include_str!("fixtures/engine-capabilities-v1.json").replace(
                "4096,\n  \"input_limit\": 4000",
                "65536,\n  \"input_limit\": 65000",
            );
            let caps = if self.enabled {
                base.replace("\"kv_branch_handles\": false,", &format!(r#""kv_branch_handles":true,"kv_checkpoints":{{"scope":"authorization","cache_domain":{},"max_handles_per_scope":{},"max_bytes_per_scope":{},"ttl_ms":{},"release":"/v1/kv/release"}},"#, json::quote(&self.domain), self.count, self.bytes, self.ttl))
            } else {
                base
            };
            return (200, "application/json", caps, false);
        }
        let v = json::parse(body).unwrap();
        if path == "/v1/admission" {
            let request = v.get("request").unwrap();
            self.admission.push(
                json::view(body)
                    .unwrap()
                    .get("request")
                    .unwrap()
                    .raw()
                    .into(),
            );
            let p = render(request).len();
            return (
                200,
                "application/json",
                format!(
                    r#"{{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","prompt_tokens":{p},"output_tokens":{},"context_limit":65536,"input_limit":65000,"output_limit":32,"admitted":true,"reason":null,"admission_kind":"context_fit","reserves_resources":false}}"#,
                    n(&v, "output_tokens")
                ),
                false,
            );
        }
        if path == "/v1/kv/release" {
            let handle = v.get("kv_handle").unwrap().as_str().unwrap();
            self.released.push(handle.into());
            self.entries.retain(|e| e.handle != handle || e.foreign);
            return (
                200,
                "application/json",
                r#"{"released":true}"#.into(),
                false,
            );
        }
        assert_eq!(path, "/v1/chat/completions");
        assert!(
            self.admission.iter().any(|b| b == body),
            "admission must see exact final bytes"
        );
        assert!(
            n(&v, "max_tokens") > 0,
            "contract has no prefill-only request"
        );
        let prompt = render(&v);
        let at = now() + self.tick;
        self.entries.retain(|e| e.expiry > at);
        let mut reuse = "not_requested";
        let mut reused = 0;
        if let Some(handle) = v.get("kv_handle").and_then(Value::as_str) {
            reuse = "unavailable";
            if let Some(index) = self
                .entries
                .iter()
                .position(|e| e.handle == handle && !e.foreign)
            {
                let saved = self.entries.remove(index).unwrap();
                reuse = if prompt.starts_with(&saved.prompt) && !self.mismatch {
                    reused = saved.prompt.len();
                    "hit"
                } else {
                    "prompt_mismatch"
                };
                self.entries.push_back(saved); // fixed expiry, read refreshes LRU only
            }
        }
        let mut retention = "not_requested";
        let mut checkpoint = "null".to_owned();
        if v.get("kv_checkpoint") == Some(&Value::Bool(true)) {
            let bytes = prompt.len() as u64 * 4 + 16;
            retention = "capacity";
            if bytes <= self.bytes && !self.retention_failure {
                while self.entries.len() >= self.count
                    || self.entries.iter().map(|e| e.bytes).sum::<u64>() + bytes > self.bytes
                {
                    self.entries.pop_front();
                    self.evicted += 1;
                }
                self.serial += 1;
                let handle = format!("{:064x}", self.serial);
                let expiry = at + self.ttl;
                let scope = self.entries.iter().map(|e| e.bytes).sum::<u64>() + bytes;
                checkpoint = format!(
                    r#"{{"handle":"{handle}","prompt_tokens":{},"reusable_prefix_tokens":{},"expires_at_ms":{expiry},"retained_bytes":{bytes},"scope_retained_bytes":{scope}}}"#,
                    prompt.len(),
                    prompt.len()
                );
                self.entries.push_back(Saved {
                    handle,
                    prompt: prompt.clone(),
                    expiry,
                    bytes,
                    foreign: false,
                });
                retention = "retained";
            }
        }
        let kv = if self.enabled {
            format!(
                r#", "vade":{{"kv":{{"cache_domain":{},"reuse":"{reuse}","reused_tokens":{},"retention":"{retention}","checkpoint":{checkpoint}}}}}"#,
                json::quote(&self.domain),
                if self.malformed {
                    prompt.len() + 1
                } else {
                    reused
                }
            )
        } else {
            String::new()
        };
        let usage = format!(
            r#"{{"prompt_tokens":{},"completion_tokens":1,"prompt_tokens_details":{{"cached_tokens":{reused}}}{kv}}}"#,
            prompt.len()
        );
        let streaming = v.get("stream") == Some(&Value::Bool(true));
        self.requests.push(v);
        if streaming {
            (
                200,
                "text/event-stream",
                format!(
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":\"answer\"}},\"finish_reason\":\"stop\"}}]}}\n\ndata: {{\"choices\":[],\"usage\":{usage}}}\n\ndata: [DONE]\n\n"
                ),
                self.truncated,
            )
        } else {
            (
                200,
                "application/json",
                format!(
                    r#"{{"choices":[{{"message":{{"role":"assistant","content":"answer"}},"finish_reason":"stop"}}],"usage":{usage}}}"#
                ),
                self.truncated,
            )
        }
    }
}
struct Mock {
    endpoint: String,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (data, done) = (Arc::clone(&state), Arc::clone(&stop));
        let worker = thread::spawn(move || {
            let mut workers = Vec::new();
            while !done.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let (state, stop) = (Arc::clone(&data), Arc::clone(&done));
                        workers.push(thread::spawn(move || serve(stream, state, stop)));
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1))
                    }
                    Err(e) => panic!("mock accept: {e}"),
                }
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            endpoint,
            state,
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
fn serve(stream: TcpStream, state: Arc<Mutex<State>>, stop: Arc<AtomicBool>) {
    stream.set_nonblocking(false).unwrap();
    stream.set_nodelay(true).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    while !stop.load(Ordering::Relaxed) {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => (),
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(_) => return,
        }
        let first = line.clone();
        let mut length = 0;
        loop {
            line.clear();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = value.trim().parse().unwrap();
            }
        }
        let mut bytes = vec![0; length];
        reader.read_exact(&mut bytes).unwrap();
        let mut parts = first.split_whitespace();
        let (status, kind, body, truncated) = state.lock().unwrap().reply(
            parts.next().unwrap(),
            parts.next().unwrap(),
            std::str::from_utf8(&bytes).unwrap(),
        );
        if write!(
            reader.get_mut(),
            "HTTP/1.1 {status} Mock\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len() + usize::from(truncated)
        )
        .is_err()
        {
            return;
        }
        if truncated {
            return;
        }
    }
}
struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(vade_harness::journal::fresh_name());
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn workspace(&self) -> Workspace {
        Workspace::new(&self.0).unwrap()
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
struct NoTools;
impl ToolExecutor for NoTools {
    fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
        panic!("no tools in fixture")
    }
}
fn conversation(stream: bool) -> Conversation {
    Conversation::from_request(json::parse(&format!(r#"{{"model":"mock","stream":{stream},"tools":[],"messages":[{{"role":"system","content":"exact rules and project prefix"}}]}}"#)).unwrap(), Limits::default()).unwrap()
}
fn user(c: &mut Conversation, text: &str) {
    c.push_user(
        Value::Object(vec![
            ("role".into(), Value::String("user".into())),
            ("content".into(), Value::String(text.into())),
        ]),
        Limits::default(),
    )
    .unwrap();
}
fn run(c: &mut Conversation, m: &mut NativeModel, journal: &mut Journal, text: &str) {
    m.bind_branch(journal.name()).unwrap();
    user(c, text);
    journal.save(c).unwrap();
    let report = agent::run_observed(
        c,
        m,
        &mut NoTools,
        Limits::default(),
        &AtomicBool::new(false),
        &mut |_| {},
        journal,
    );
    assert_eq!(
        report.reason,
        agent::StopReason::Finished(FinishReason::Stop)
    );
}
fn reuse(m: &NativeModel) -> Reuse {
    m.context().turns.back().unwrap().kv.as_ref().unwrap().reuse
}

#[test]
fn fork_switch_resume_and_children_reuse_full_prompts_and_release_shared_ownership() {
    for stream in [false, true] {
        let mock = Mock::new();
        let temp = Temp::new();
        let w = temp.workspace();
        let mut parent = conversation(stream);
        let mut journal = Journal::create(&w, "parent", &parent).unwrap();
        let mut model = NativeModel::new(&mock.endpoint).unwrap();
        let early_child_model = model.branch_client(); // share ownership before discovery too
        run(&mut parent, &mut model, &mut journal, "first task");
        assert_eq!(reuse(&model), Reuse::NotRequested);
        assert_eq!(
            parent.checkpoint_hints()[0].messages,
            2,
            "answer is not checkpointed by the Engine contract"
        );
        let (mut child_journal, mut child) = journal
            .fork(&w, "child", "alternative", Limits::default())
            .unwrap();
        let mut child_model = model.branch_client();
        run(
            &mut child,
            &mut child_model,
            &mut child_journal,
            "child task",
        );
        assert_eq!(reuse(&child_model), Reuse::Hit);
        assert!(child_model.context().describe().contains("KV handle: hit"));
        assert!(mock.state.lock().unwrap().released.is_empty());
        let mut prefix_only = conversation(stream);
        prefix_only.inherit_checkpoint_hints(&parent);
        assert!(
            prefix_only.checkpoint_hints().is_empty(),
            "a system-only child cannot reuse a user-containing checkpoint"
        );
        let mut inherited = parent.clone();
        inherited.inherit_checkpoint_hints(&parent);
        assert_eq!(inherited.checkpoint_hints().len(), 1);
        let mut child_two = early_child_model;
        let mut j2 = Journal::create(&w, "child-two", &inherited).unwrap();
        run(&mut inherited, &mut child_two, &mut j2, "another child");
        assert_eq!(reuse(&child_two), Reuse::Hit);
        assert_eq!(
            child_model
                .delete_branch(child_journal, &w)
                .unwrap()
                .released,
            1
        );
        assert_eq!(
            mock.state.lock().unwrap().released.len(),
            1,
            "parent reference survives child deletion"
        );
        drop(child_model);
        drop(child_two);
        drop(j2);
        let before = mock.state.lock().unwrap().released.len();
        let mut switched = model.branch_client();
        drop(model);
        drop(journal);
        let (mut j, mut restored, torn) = Journal::resume(&w, "parent", Limits::default()).unwrap();
        assert!(!torn);
        assert_eq!(restored.checkpoint_hints(), parent.checkpoint_hints());
        switched.reestablish_prompt(PromptBoundary::Resume);
        run(&mut restored, &mut switched, &mut j, "parent again");
        assert_eq!(reuse(&switched), Reuse::Hit);
        assert_eq!(mock.state.lock().unwrap().released.len(), before);
        for record in 0..j.record_count() {
            assert!(
                !j.inspect_record(record, 0, 4096)
                    .unwrap()
                    .contains(&format!("{:064x}", 1))
            );
        }
        assert!(!format!("{:?}", restored.checkpoint_hints()).contains(&format!("{:064x}", 1)));
        drop(switched);
        assert!(
            mock.state.lock().unwrap().entries.is_empty(),
            "last client drop releases the entire owned session"
        );
    }
}

#[test]
fn expiry_foreign_quota_eviction_restart_and_capacity_use_normal_full_prompt_fallback() {
    let mock = Mock::new();
    let temp = Temp::new();
    let w = temp.workspace();
    let mut c = conversation(true);
    let mut j = Journal::create(&w, "main", &c).unwrap();
    let mut m = NativeModel::new(&mock.endpoint).unwrap();
    run(&mut c, &mut m, &mut j, "first");
    mock.state.lock().unwrap().tick = 300_001;
    run(&mut c, &mut m, &mut j, "expired remotely");
    assert_eq!(reuse(&m), Reuse::Unavailable);
    for entry in &mut mock.state.lock().unwrap().entries {
        entry.foreign = true;
    }
    run(&mut c, &mut m, &mut j, "foreign scope");
    assert_eq!(reuse(&m), Reuse::Unavailable);
    {
        let mut s = mock.state.lock().unwrap();
        s.count = 1;
        s.entries.clear();
    }
    let mut other = conversation(true);
    let mut oj = Journal::create(&w, "other", &other).unwrap();
    let mut om = NativeModel::new(&mock.endpoint).unwrap();
    run(&mut other, &mut om, &mut oj, "other scope consumer");
    run(&mut other, &mut om, &mut oj, "quota pressure");
    assert!(mock.state.lock().unwrap().evicted > 0);
    run(&mut c, &mut m, &mut j, "evicted by quota");
    assert_eq!(reuse(&m), Reuse::Unavailable);
    {
        let mut s = mock.state.lock().unwrap();
        s.domain = "mock-kv-epoch-2".into();
        s.entries.clear();
    }
    run(&mut c, &mut m, &mut j, "new engine epoch");
    assert_eq!(reuse(&m), Reuse::NotRequested);
    assert!(
        mock.state
            .lock()
            .unwrap()
            .requests
            .last()
            .unwrap()
            .get("kv_handle")
            .is_none()
    );
    mock.state.lock().unwrap().retention_failure = true;
    run(&mut c, &mut m, &mut j, "capacity refusal still answers");
    assert_eq!(
        m.context()
            .turns
            .back()
            .unwrap()
            .kv
            .as_ref()
            .unwrap()
            .retention,
        vade_harness::kv::Retention::Capacity
    );
    for request in &mock.state.lock().unwrap().requests {
        assert!(request.get("messages").unwrap().as_array().unwrap().len() >= 2);
    }
}

#[test]
fn malformed_telemetry_and_truncated_stream_never_publish_checkpoint_hints() {
    for truncated in [false, true] {
        let mock = Mock::new();
        let temp = Temp::new();
        let w = temp.workspace();
        {
            let mut s = mock.state.lock().unwrap();
            s.malformed = !truncated;
            s.truncated = truncated;
        }
        let mut c = conversation(true);
        let mut j = Journal::create(&w, "main", &c).unwrap();
        let mut m = NativeModel::new(&mock.endpoint).unwrap();
        user(&mut c, "task");
        j.save(&c).unwrap();
        let result = agent::run_observed(
            &mut c,
            &mut m,
            &mut NoTools,
            Limits::default(),
            &AtomicBool::new(false),
            &mut |_| {},
            &mut j,
        );
        assert!(c.checkpoint_hints().is_empty());
        if truncated {
            assert!(matches!(result.reason, agent::StopReason::ModelError(_)));
            assert!(m.context().turns.is_empty());
        } else {
            assert_eq!(
                result.reason,
                agent::StopReason::Finished(FinishReason::Stop)
            );
            assert!(m.context().turns.back().unwrap().kv_invalid);
        }
        assert!(mock.state.lock().unwrap().released.is_empty());
    }
}

#[test]
fn no_advertisement_and_cache_false_preserve_ordinary_request_behavior() {
    for disabled in [false, true] {
        let mock = Mock::new();
        let temp = Temp::new();
        let w = temp.workspace();
        mock.state.lock().unwrap().enabled = disabled;
        let mut c = if disabled {
            Conversation::from_request(json::parse(r#"{"model":"mock","stream":true,"cache":false,"messages":[{"role":"system","content":"rules"}]}"#).unwrap(),Limits::default()).unwrap()
        } else {
            conversation(true)
        };
        let mut j = Journal::create(&w, "main", &c).unwrap();
        let mut m = NativeModel::new(&mock.endpoint).unwrap();
        run(&mut c, &mut m, &mut j, "first");
        run(&mut c, &mut m, &mut j, "next");
        drop(m);
        let s = mock.state.lock().unwrap();
        assert_eq!(s.requests.len(), 2);
        assert!(s.released.is_empty());
        for r in &s.requests {
            assert!(r.get("kv_handle").is_none());
            assert!(r.get("kv_checkpoint").is_none());
        }
    }
}

#[path = "support/cli.rs"]
mod cli;

#[test]
fn cli_fork_switch_context_and_exit_use_checkpoint_lifecycle() {
    let mock = Mock::new();
    let temp = Temp::new();
    let mut command = home::harness();
    command
        .args([
            "agent",
            "--state-in-project",
            "--endpoint",
            &mock.endpoint,
            "--model",
            "mock",
            "--workspace",
        ])
        .arg(&temp.0)
        .args([
            "--verbose",
            "--session",
            "parent",
            "--approval",
            "never",
            "--plain",
        ]);
    let mut child = cli::Cli::start(&mut command);
    child.send("parent task\n");
    child.until(true, "Native run:");
    child.send("/fork child\nchild task\n");
    child.until(true, "Native run:");
    child.send("/context\n/switch parent\nparent followup\n");
    child.until(true, "Native run:");
    child.send("/context\n/quit\n");
    let output = child.finish();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("KV handle: hit"), "{stdout}");
    assert!(!stdout.contains(&format!("{:064x}", 1)));
    assert!(!String::from_utf8_lossy(&output.stderr).contains(&format!("{:064x}", 1)));
    let s = mock.state.lock().unwrap();
    assert_eq!(
        s.requests.len(),
        3,
        "fork/switch must not generate hidden warm-ups"
    );
    assert!(s.requests[0].get("kv_handle").is_none());
    assert!(s.requests[1].get("kv_handle").is_some());
    assert!(s.requests[2].get("kv_handle").is_some());
    assert!(
        s.entries.is_empty(),
        "normal process exit releases known handles"
    );
}

#[test]
fn prompt_mismatch_and_byte_pressure_do_not_retry_or_change_transcript() {
    let mock = Mock::new();
    let temp = Temp::new();
    let w = temp.workspace();
    let mut c = conversation(false);
    let mut j = Journal::create(&w, "main", &c).unwrap();
    let mut m = NativeModel::new(&mock.endpoint).unwrap();
    run(&mut c, &mut m, &mut j, "first");
    let first = c.checkpoint_hints()[0].checkpoint.retained_bytes;
    {
        let mut s = mock.state.lock().unwrap();
        s.mismatch = true;
        s.bytes = first * 2;
    }
    run(&mut c, &mut m, &mut j, "a different framing was reported");
    assert_eq!(reuse(&m), Reuse::PromptMismatch);
    assert_eq!(mock.state.lock().unwrap().requests.len(), 2);
    assert_eq!(c.messages().len(), 5);
    assert_eq!(
        c.checkpoint_hints().len(),
        1,
        "rejected input handle is retired"
    );
    assert!(
        mock.state.lock().unwrap().evicted > 0,
        "the byte quota evicts despite count headroom"
    );
    let cancelled = AtomicBool::new(true);
    let mut child_model = m.branch_client();
    let report = agent::run(
        &mut c,
        &mut child_model,
        &mut NoTools,
        Limits::default(),
        &cancelled,
        &mut |_| {},
    );
    assert_eq!(report.reason, agent::StopReason::Cancelled);
    drop(child_model);
    assert_eq!(mock.state.lock().unwrap().requests.len(), 2);
    assert!(
        !mock.state.lock().unwrap().entries.is_empty(),
        "cancelled child must not release its parent"
    );
}

#[test]
fn simultaneous_child_clients_share_one_parent_reference_and_keep_independent_tails() {
    let mock = Mock::new();
    let temp = Temp::new();
    let w = temp.workspace();
    let mut c = conversation(true);
    let mut j = Journal::create(&w, "parent", &c).unwrap();
    let mut m = NativeModel::new(&mock.endpoint).unwrap();
    run(&mut c, &mut m, &mut j, "parent task");
    thread::scope(|scope| {
        for id in 0..4 {
            let mut child = c.clone();
            child.inherit_checkpoint_hints(&c);
            let mut model = m.branch_client();
            let w = &w;
            scope.spawn(move || {
                let mut journal = Journal::create(w, &format!("child-{id}"), &child).unwrap();
                run(
                    &mut child,
                    &mut model,
                    &mut journal,
                    &format!("independent task {id}"),
                );
                assert_eq!(reuse(&model), Reuse::Hit);
            });
        }
    });
    assert_eq!(mock.state.lock().unwrap().requests.len(), 5);
    assert!(mock.state.lock().unwrap().released.is_empty());
    assert_eq!(
        m.release_checkpoints().released,
        5,
        "shared handle released once after all children join"
    );
    assert!(mock.state.lock().unwrap().entries.is_empty());
}
