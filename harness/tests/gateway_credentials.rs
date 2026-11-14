//! File credential selection at the two actual gateway process boundaries.
#[path = "support/home.rs"]
mod home;
#[path = "support/cli.rs"]
#[allow(dead_code)] // This process matrix needs only the bounded start/finish driver.
mod cli;
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, atomic::{AtomicBool, Ordering}, mpsc},
    thread,
    time::{Duration, Instant},
};

const CLIENT: &str = "fixture-client-key-0123456789";
const ENV_KEY: &str = "fixture-environment-key-0123456789";
const FLAG_KEY: &str = "fixture-option-key-0123456789";
const BODY: &str = r#"{ "model":"mock", "messages":[], "seed":713, "max_tokens":32 }"#;

struct Root(PathBuf);
impl Root {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!("vade-gateway-auth-{}-{label}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn key(&self, name: &str, value: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, format!("{value}\n")).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
        path
    }
}
impl Drop for Root { fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); } }

struct Request { first: String, headers: Vec<String>, body: Vec<u8> }
impl Request {
    fn credentials(&self) -> Vec<(String, &str)> {
        self.headers.iter().filter_map(|line| {
            let (name, value) = line.split_once(':').unwrap();
            (name.eq_ignore_ascii_case("authorization") || name.eq_ignore_ascii_case("x-api-key"))
                .then(|| (name.to_ascii_lowercase(), value.trim()))
        }).collect()
    }
}
struct Mock {
    address: SocketAddr,
    requests: mpsc::Receiver<Request>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let running = stop.clone();
        let (tx, requests) = mpsc::channel();
        let worker = thread::spawn(move || {
            while !running.load(Ordering::Relaxed) {
                let (socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1)); continue;
                    }
                    Err(error) => panic!("{error}"),
                };
                socket.set_nonblocking(false).unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                socket.set_write_timeout(Some(Duration::from_secs(3))).unwrap();
                let mut reader = BufReader::new(socket);
                let mut first = String::new(); reader.read_line(&mut first).unwrap();
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    assert!(reader.read_line(&mut line).unwrap() > 0);
                    if line == "\r\n" { break; }
                    headers.push(line.trim_end_matches("\r\n").to_owned());
                }
                let size = headers.iter().find_map(|line| {
                    let (name, value) = line.split_once(':').unwrap();
                    name.eq_ignore_ascii_case("content-length").then(|| value.trim().parse::<usize>().unwrap())
                }).unwrap_or(0);
                assert!(size < 16384);
                let mut body = vec![0; size]; reader.read_exact(&mut body).unwrap();
                let response = if first.starts_with("GET /v1/capabilities ") {
                    include_str!("fixtures/engine-capabilities-v1.json")
                } else if first.starts_with("POST /v1/admission ") {
                    r#"{"object":"engine.admission","contract_version":"vade.engine.v1","model":"mock","admission_kind":"context_fit","reserves_resources":false,"prompt_tokens":16,"output_tokens":32,"context_limit":4096,"input_limit":3072,"output_limit":1024,"admitted":true}"#
                } else { r#"{"fixture":"complete"}"# };
                tx.send(Request { first, headers, body }).unwrap();
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAuthorization: Bearer response-opaque\r\nX-Fixture: preserved\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).unwrap();
            }
        });
        Self { address, requests, stop, worker: Some(worker) }
    }
    fn receive(&self) -> Request { self.requests.recv_timeout(Duration::from_secs(4)).unwrap() }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let result = self.worker.take().unwrap().join();
        if !thread::panicking() { result.unwrap(); }
    }
}

struct Gateway(Child);
impl Drop for Gateway {
    fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); }
}
fn free_address() -> SocketAddr { TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap() }
fn command(standalone: bool) -> Command {
    let mut command = home::command(if standalone { env!("CARGO_BIN_EXE_vade-gateway") } else { env!("CARGO_BIN_EXE_vade-harness") });
    if !standalone { command.arg("gateway"); }
    // An inherited client token would answer these unauthenticated calls with 401.
    command.env_remove("VADE_API_KEY_FILE").env_remove("VADE_GATEWAY_UPSTREAM").env_remove("VADE_GATEWAY_TOKEN");
    command
}
fn call(process: &mut Gateway, address: SocketAddr) -> Vec<u8> {
    let deadline = Instant::now() + Duration::from_secs(4);
    let mut socket = loop {
        match TcpStream::connect_timeout(&address, Duration::from_millis(100)) {
            Ok(socket) => break socket,
            Err(_) => {
                assert!(process.0.try_wait().unwrap().is_none(), "gateway exited before serving");
                assert!(Instant::now() < deadline, "gateway startup timeout");
                thread::sleep(Duration::from_millis(1));
            }
        }
    };
    socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    socket.set_write_timeout(Some(Duration::from_secs(3))).unwrap();
    write!(socket, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:{}\r\nContent-Type: application/json\r\nAuthorization: Bearer {CLIENT}\r\nX-Api-Key: {CLIENT}\r\nX-Fixture: untouched\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{BODY}", address.port(), BODY.len()).unwrap();
    let mut response = Vec::new(); socket.read_to_end(&mut response).unwrap(); response
}
fn recorded_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if root.exists() {
        for entry in fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() { files.extend(recorded_files(&path)); } else { files.push(path); }
        }
    }
    files
}
fn response_body(response: &[u8]) -> Vec<u8> {
    let split = response.windows(4).position(|bytes| bytes == b"\r\n\r\n").unwrap();
    let headers = String::from_utf8_lossy(&response[..split]).to_ascii_lowercase();
    let mut body = &response[split + 4..];
    if !headers.contains("transfer-encoding: chunked") { return body.to_vec(); }
    let mut decoded = Vec::new();
    loop {
        let end = body.windows(2).position(|bytes| bytes == b"\r\n").unwrap();
        let size = usize::from_str_radix(std::str::from_utf8(&body[..end]).unwrap(), 16).unwrap();
        body = &body[end + 2..];
        if size == 0 { assert_eq!(body, b"\r\n"); return decoded; }
        decoded.extend_from_slice(&body[..size]);
        assert_eq!(&body[size..size + 2], b"\r\n"); body = &body[size + 2..];
    }
}

#[test]
fn configured_credentials_cover_both_gateways_admission_and_recorded_replay() {
    let root = Root::new("matrix");
    let environment = root.key("environment.key", ENV_KEY);
    let explicit = root.key("explicit.key", FLAG_KEY);
    for (index, (standalone, selected, clamp, record)) in [
        (false, 0, false, false), (true, 0, false, false),
        (false, 1, false, false), (true, 1, false, false),
        (false, 2, false, false), (true, 2, false, false),
        (true, 2, true, true),
    ].into_iter().enumerate() {
        let mock = Mock::new(); let address = free_address();
        let mut cmd = command(standalone);
        cmd.args(["--upstream", &format!("http://{}/v1", mock.address), "--bind", &address.to_string()]);
        if selected > 0 { cmd.env("VADE_API_KEY_FILE", &environment); }
        if selected == 2 { cmd.arg("--api-key-file").arg(&explicit); }
        if clamp { cmd.arg("--clamp-limits"); }
        let recordings = root.0.join(format!("record-{index}"));
        let stats = root.0.join(format!("stats-{index}.jsonl"));
        if record { cmd.arg("--record").arg(&recordings).arg("--stats").arg(&stats); }
        let mut gateway = Gateway(cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap());
        let response = call(&mut gateway, address);
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert_eq!(response_body(&response), br#"{"fixture":"complete"}"#);
        assert!(String::from_utf8_lossy(&response).to_ascii_lowercase().contains("authorization: bearer response-opaque"));
        for number in 0..if clamp { 3 } else { 1 } {
            let request = mock.receive();
            if selected == 0 {
                assert_eq!(request.credentials(), [("authorization".into(), format!("Bearer {CLIENT}").as_str()), ("x-api-key".into(), CLIENT)]);
            } else {
                let key = if selected == 1 { ENV_KEY } else { FLAG_KEY };
                assert_eq!(request.credentials(), [("authorization".into(), format!("Bearer {key}").as_str())]);
            }
            if !clamp || number == 2 {
                assert!(request.first.starts_with("POST /v1/chat/completions "));
                assert_eq!(request.body, BODY.as_bytes());
                assert!(request.headers.iter().any(|line| line.eq_ignore_ascii_case("X-Fixture: untouched")));
            } else {
                assert!(request.first.starts_with(if number == 0 { "GET /v1/capabilities " } else { "POST /v1/admission " }));
            }
        }
        if record {
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                if recorded_files(&recordings).iter().any(|path| path.file_name().unwrap() == "meta.json")
                    && fs::read_to_string(&stats).is_ok_and(|text| text.ends_with('\n')) { break; }
                assert!(Instant::now() < deadline); thread::sleep(Duration::from_millis(1));
            }
            drop(gateway);
            for path in recorded_files(&recordings) {
                let bytes = fs::read(path).unwrap(); let text = String::from_utf8_lossy(&bytes);
                for secret in [CLIENT, ENV_KEY, FLAG_KEY] { assert!(!text.contains(secret)); }
            }
            let mut replay = command(true);
            replay.arg("replay").arg(&recordings).args(["--upstream", &format!("http://{}/v1", mock.address)])
                .arg("--api-key-file").arg(&explicit).arg("--stats").arg(&stats);
            let output = cli::Cli::start(&mut replay).finish();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            let request = mock.receive();
            assert_eq!(request.credentials(), [("authorization".into(), format!("Bearer {FLAG_KEY}").as_str())]);
            assert_eq!(request.body, BODY.as_bytes());
            let saved = fs::read_to_string(&stats).unwrap();
            assert_eq!(saved.lines().count(), 2, "gateway and replay must each append statistics");
            for line in saved.lines() { vade_harness::json::parse(line).unwrap(); }
            for secret in [CLIENT, ENV_KEY, FLAG_KEY] { assert!(!saved.contains(secret)); }
        }
    }
}

#[test]
fn missing_credentials_fail_before_listening_but_help_stays_offline() {
    let root = Root::new("missing"); let missing = root.0.join("missing.key");
    for standalone in [false, true] {
        let address = free_address(); let mut cmd = command(standalone);
        cmd.args(["--upstream", "http://127.0.0.1:1/v1", "--bind", &address.to_string()])
            .arg("--api-key-file").arg(&missing);
        let output = cli::Cli::start(&mut cmd).finish();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("cannot read --api-key-file"));
        assert!(output.stdout.is_empty());
        let _available = TcpListener::bind(address).unwrap();
        let mut help = command(standalone);
        help.arg("--help").env("VADE_API_KEY_FILE", &missing);
        let output = cli::Cli::start(&mut help).finish();
        assert!(output.status.success()); assert!(output.stderr.is_empty());
    }
}
