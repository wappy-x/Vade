//! Private, streaming recordings and ordered session replay. No header credentials.
use super::*;
use crate::json::{self, Value};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Mutex, atomic::AtomicBool},
    time::{SystemTime, UNIX_EPOCH},
};

/// The most `request-*` folders one recording folder holds; replay refuses more.
const MAX_RECORDINGS: usize = 100_000;
pub(super) struct Recorder {
    root: PathBuf,
    sequence: AtomicUsize,
    /// New recordings this run may add before the folder reaches [`MAX_RECORDINGS`].
    room: usize,
}
impl Recorder {
    pub(super) fn open(path: &str) -> io::Result<Self> {
        let root = PathBuf::from(path);
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&root)?;
        if fs::symlink_metadata(&root)?.file_type().is_symlink() {
            return Err(transport::invalid("record root must be a real directory"));
        }
        // Recordings hold prompts and replies; an existing folder becomes private too.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&root)?.permissions().mode();
            if mode & 0o077 != 0 {
                fs::set_permissions(&root, fs::Permissions::from_mode(mode & 0o7700))?;
                crate::diagnostics::stderr_line(format_args!("vade-gateway: the --record folder was open to other users; it now has owner-only permissions."));
            }
        }
        let mut existing = 0;
        for entry in fs::read_dir(&root)? {
            existing += usize::from(entry?.file_name().to_string_lossy().starts_with("request-"));
        }
        Ok(Self {
            root,
            sequence: AtomicUsize::new(0),
            room: MAX_RECORDINGS.saturating_sub(existing),
        })
    }
    /// A new recording, or `None` after a report when the folder is full or unwritable.
    pub(super) fn begin(&self) -> Option<Recording> {
        match self.start() {
            Ok(recording) => Some(recording),
            Err(error) if error.kind() == io::ErrorKind::StorageFull => None,
            Err(_) => {
                crate::diagnostics::stderr_line(format_args!("vade-gateway: recorder start failed; this request is forwarded but not recorded. Check the --record folder's permissions and free space."));
                None
            }
        }
    }
    pub(super) fn start(&self) -> io::Result<Recording> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        if sequence >= self.room {
            if sequence == self.room {
                crate::diagnostics::stderr_line(format_args!("vade-gateway: the --record folder holds {MAX_RECORDINGS} recordings, the replay limit; later requests are forwarded but not recorded. Move old recordings out, then restart the gateway."));
            }
            return Err(io::Error::new(io::ErrorKind::StorageFull, "recording limit reached"));
        }
        let dir = self.root.join(format!("request-{stamp:020}-{sequence:08}"));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&dir)?;
        let response = private_file(&dir.join("response.body"))?;
        Ok(Recording {
            dir,
            response,
            metadata: None,
        })
    }
}
fn private_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}
pub(super) struct Recording {
    dir: PathBuf,
    response: File,
    metadata: Option<String>,
}
impl Recording {
    pub(super) fn request(
        &mut self,
        method: &str,
        path: &str,
        head: &Head,
        body: &[u8],
    ) -> io::Result<()> {
        private_file(&self.dir.join("request.body"))?.write_all(body)?;
        // Deliberate allowlist: auth, cookies and arbitrary client headers are not persisted.
        let headers: Vec<_> = head
            .fields
            .iter()
            .filter(|(k, _)| {
                matches!(
                    k.as_str(),
                    "content-type"
                        | "content-encoding"
                        | "accept"
                        | "x-session-id"
                        | "session-id"
                        | "session_id"
                        | "x-codex-session-id"
                        | "x-claude-code-session-id"
                )
            })
            .map(|(k, v)| format!("[{},{}]", json::quote(k), json::quote(v)))
            .collect();
        let session = std::str::from_utf8(body)
            .ok()
            .and_then(|s| json::view(s).ok())
            .and_then(|v| v.get("prompt_cache_key"))
            .and_then(json::View::as_str)
            .map(|v| v.into_owned())
            .or_else(|| {
                head.get("x-session-id")
                    .or_else(|| head.get("session_id"))
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "unkeyed".into());
        self.metadata = Some(format!(
            "\"version\":1,\"method\":{},\"path\":{},\"headers\":[{}],\"session\":{}",
            json::quote(method),
            json::quote(path),
            headers.join(","),
            json::quote(&session)
        ));
        Ok(())
    }
    pub(super) fn effective(&mut self, body: &[u8]) -> io::Result<()> {
        private_file(&self.dir.join("upstream-request.body"))?.write_all(body)
    }
    pub(super) fn payload(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.response.write_all(bytes)
    }
    pub(super) fn finish(mut self, timings: &str) -> io::Result<()> {
        self.response.flush()?;
        private_file(&self.dir.join("timings.json"))?.write_all(timings.as_bytes())?;
        if let Some(metadata) = self.metadata {
            let complete = json::view(timings)?
                .get("complete")
                .is_some_and(|v| v.raw() == "true");
            let temporary = self.dir.join("meta.new");
            private_file(&temporary)?
                .write_all(format!("{{{metadata},\"complete\":{complete}}}").as_bytes())?;
            fs::rename(temporary, self.dir.join("meta.json"))?;
        }
        Ok(())
    }
}
pub(super) fn observe(
    recording: &mut Option<Recording>,
    action: impl FnOnce(&mut Recording) -> io::Result<()>,
) {
    if recording.as_mut().is_some_and(|r| action(r).is_err()) {
        crate::diagnostics::stderr_line(format_args!("vade-gateway: recorder write failed; reply forwarding continues"));
        *recording = None;
    }
}

struct Saved {
    dir: PathBuf,
    method: String,
    path: String,
    headers: Vec<(String, String)>,
}
fn read_bounded(path: &Path, bound: u64) -> io::Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.len() > bound {
        return Err(transport::invalid("invalid or oversized recording file"));
    }
    let mut bytes = Vec::new();
    File::open(path)?.take(bound + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > bound {
        return Err(transport::invalid("recording exceeds size limit"));
    }
    Ok(bytes)
}
fn string(v: &Value, key: &str) -> io::Result<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| transport::invalid("missing recording metadata"))
}
fn inventory(root: &Path) -> io::Result<Vec<Vec<Saved>>> {
    let mut dirs = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        // Incomplete folders have no meta.json; they never count against the limit.
        if entry.file_type()?.is_dir()
            && entry.file_name().to_string_lossy().starts_with("request-")
            && entry.path().join("meta.json").exists()
        {
            dirs.push(entry.path());
        }
        if dirs.len() > MAX_RECORDINGS {
            return Err(transport::invalid("recording inventory exceeds limit"));
        }
    }
    dirs.sort();
    let mut sessions: BTreeMap<String, Vec<Saved>> = BTreeMap::new();
    for dir in dirs {
        let meta = json::parse_bytes(&read_bounded(&dir.join("meta.json"), 64 * 1024)?)?;
        if meta.get("complete") != Some(&Value::Bool(true)) {
            continue;
        }
        let method = string(&meta, "method")?;
        let path = string(&meta, "path")?;
        let session = string(&meta, "session")?;
        if method.is_empty()
            || !method
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || !path.starts_with('/')
            || path.bytes().any(|b| b <= 0x20 || b == 0x7f)
            || path.contains(['#', '\\'])
        {
            return Err(transport::invalid("invalid recorded request target"));
        }
        let mut headers = Vec::new();
        for pair in meta
            .get("headers")
            .and_then(Value::as_array)
            .ok_or_else(|| transport::invalid("invalid recorded headers"))?
        {
            let pair = pair
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(|| transport::invalid("invalid recorded header"))?;
            let key = pair[0]
                .as_str()
                .ok_or_else(|| transport::invalid("invalid recorded header name"))?;
            let value = pair[1]
                .as_str()
                .ok_or_else(|| transport::invalid("invalid recorded header value"))?;
            // Revalidate the persisted allowlist even if a recording was edited.
            if matches!(
                key,
                "content-type"
                    | "content-encoding"
                    | "accept"
                    | "x-session-id"
                    | "session-id"
                    | "session_id"
                    | "x-codex-session-id"
                    | "x-claude-code-session-id"
            ) {
                // Rebuilding HTTP from saved JSON must not turn a newline in
                // a value into another header. Match the live HTTP validator.
                if value.bytes().any(|byte| (byte < 0x20 && byte != b'\t') || byte == 0x7f) {
                    return Err(transport::invalid(
                        "recorded header contains control characters; preserve the recording and choose an unmodified complete recording",
                    ));
                }
                headers.push((key.into(), value.into()));
            }
        }
        sessions.entry(session.clone()).or_default().push(Saved {
            dir,
            method,
            path,
            headers,
        });
    }
    Ok(sessions.into_values().collect())
}
fn replay_body(saved: &Saved) -> io::Result<Vec<u8>> {
    let effective = saved.dir.join("upstream-request.body");
    let request = if effective.exists() {
        effective
    } else {
        saved.dir.join("request.body")
    };
    read_bounded(&request, transport::MAX_BODY as u64)
}
#[derive(Clone, Copy, Debug)]
enum ReplayOutputFailure { Stdout = 1, StatisticsFile = 2 }
impl std::fmt::Display for ReplayOutputFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stdout => "cannot write replay statistics to standard output",
            Self::StatisticsFile => "cannot write replay statistics file (--stats)",
        })
    }
}
impl std::error::Error for ReplayOutputFailure {}

fn replay_one(origin: &Origin, saved: &Saved, log: Option<&StatsLog>) -> io::Result<()> {
    let body = replay_body(saved)?;
    let mut wire = format!("{} {} HTTP/1.1\r\n", saved.method, saved.path);
    for (key, value) in &saved.headers {
        wire.push_str(&format!("{key}: {value}\r\n"));
    }
    wire.push_str("\r\n");
    let request = WireHead::read(&mut io::Cursor::new(wire))?;
    let mut metrics = Metrics::new(true);
    metrics.path = saved.path.clone();
    metrics.request(&body);
    let result = origin.connect().and_then(|upstream| {
        exchange(
            upstream,
            Dispatch {
                origin,
                request: &request,
                body: &body,
                cors: "",
                retry: false,
            },
            &mut io::sink(),
            &mut metrics,
            &mut None,
        ).map(drop)
    });
    metrics.finish(result.is_ok());
    let line = metrics.json();
    if let Some(log) = log {
        log.write(&line).map_err(|error| io::Error::new(error.kind(), ReplayOutputFailure::StatisticsFile))?;
    }
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{line}")
        .and_then(|()| stdout.flush())
        .map_err(|error| io::Error::new(error.kind(), ReplayOutputFailure::Stdout))?;
    result
}
pub(super) fn replay(mut args: impl Iterator<Item = String>) -> io::Result<()> {
    let dir = argument(&mut args, "replay DIR")?;
    let mut url = std::env::var("VADE_GATEWAY_UPSTREAM").ok();
    let mut concurrency = 1;
    let mut stats_path = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--upstream" => url = Some(argument(&mut args, "--upstream")?),
            "--stats" => stats_path = Some(argument(&mut args, "--stats")?),
            "--concurrency" => {
                concurrency = argument(&mut args, "--concurrency")?
                    .parse::<usize>()
                    .map_err(|_| transport::invalid("invalid replay concurrency"))?
            }
            _ => return Err(io::Error::new(io::ErrorKind::InvalidInput,
                crate::diagnostics::option_name(&arg).map_or_else(
                    || "unknown replay argument".to_owned(),
                    |name| format!("unknown replay option {name}")))),
        }
    }
    if !(1..=super::super::MAX_CONNECTIONS).contains(&concurrency) {
        return Err(transport::invalid("replay concurrency must be 1..64"));
    }
    crate::auth::preload()?;
    let origin =
        Origin::parse(&url.ok_or_else(|| transport::invalid("replay --upstream is required"))?)?;
    let sessions = inventory(Path::new(&dir))?;
    if sessions.is_empty() {
        return Err(transport::invalid("no complete recorded sessions"));
    }
    let stats = stats_path.as_deref().map(StatsLog::open).transpose()
        .map_err(|error| io::Error::new(error.kind(), ReplayOutputFailure::StatisticsFile))?
        .map(Arc::new);
    let queue = Arc::new(Mutex::new(VecDeque::from(sessions)));
    let failed = Arc::new(AtomicBool::new(false));
    let output_failed = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::new();
    for _ in 0..concurrency {
        let (queue, failed, output_failed, origin, stats) =
            (queue.clone(), failed.clone(), output_failed.clone(), origin.clone(), stats.clone());
        workers.push(
            thread::Builder::new()
                .name("gateway-replay".into())
                .spawn(move || {
                    loop {
                        if output_failed.load(Ordering::Acquire) != 0 {
                            break;
                        }
                        let session = queue.lock().unwrap_or_else(|p| p.into_inner()).pop_front();
                        let Some(session) = session else {
                            break;
                        };
                        for saved in session {
                            if output_failed.load(Ordering::Acquire) != 0 {
                                break;
                            }
                            if let Err(error) = replay_one(&origin, &saved, stats.as_deref()) {
                                if let Some(failure) = error.get_ref().and_then(|inner| inner.downcast_ref::<ReplayOutputFailure>()) {
                                    let _ = output_failed.compare_exchange(0, *failure as usize, Ordering::AcqRel, Ordering::Acquire);
                                }
                                failed.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                })?,
        );
    }
    for worker in workers {
        if worker.join().is_err() {
            failed.store(true, Ordering::Relaxed);
        }
    }
    if output_failed.load(Ordering::Acquire) != 0 {
        Err(io::Error::other(if output_failed.load(Ordering::Acquire) == 1 {
            ReplayOutputFailure::Stdout
        } else {
            ReplayOutputFailure::StatisticsFile
        }))
    } else if failed.load(Ordering::Relaxed) {
        Err(io::Error::other(
            "one or more replay sessions failed; see request stats",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)] mod tests { use super::*;
    #[test]
fn recording_is_exact_private_and_replay_keeps_session_order_at_concurrency_two() { let dir = std::env::temp_dir().join(format!( "vade-gateway-record-{}-{}", std::process::id(), SystemTime::now() .duration_since(UNIX_EPOCH) .unwrap()
.as_nanos() )); let recorder = Recorder::open(dir.to_str().unwrap()).unwrap(); let head = Head { first: "POST /v1/responses HTTP/1.1".into(), fields: vec![ ("content-type".into(), "application/json".into()), ( "authorization".into(),
"Bearer synthetic-never-persist".into(), ), ], }; for session in ["a", "b"] { for turn in 1..=2 { let body = format!( "{{ \"model\":\"mock\", \"prompt_cache_key\":\"{session}\", \"turn\":{turn} }}" );
let mut record = recorder.start().unwrap(); record .request("POST", "/v1/responses", &head, body.as_bytes()) .unwrap(); record.payload(b"opaque\x00\xff response").unwrap(); let folder = record.dir.clone();
record.finish("{\"complete\":true}").unwrap(); assert_eq!( fs::read(folder.join("request.body")).unwrap(), body.as_bytes() ); assert_eq!( fs::read(folder.join("response.body")).unwrap(), b"opaque\x00\xff response" ); assert!(
!fs::read_to_string(folder.join("meta.json")) .unwrap() .contains("synthetic-never-persist") ); #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; assert_eq!( fs::metadata(folder.join("request.body")) .unwrap() .permissions() .mode()
& 0o777, 0o600 ); } } } let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
let firsts = Arc::new(AtomicUsize::new(0)); let seen = Arc::new(Mutex::new(Vec::new())); let observed = seen.clone(); let server = thread::spawn(move || { let mut workers = Vec::new(); for _ in 0..4 {
let socket = listener.accept().unwrap().0; let firsts = firsts.clone(); let seen = seen.clone(); workers.push(thread::spawn(move || {
socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();let mut reader=BufReader::new(socket);let head=transport::read_head(&mut reader).unwrap();assert!(head.get("authorization").is_none());
let mut body=BodyReader::new(reader,&head,true,1024).unwrap();let mut bytes=Vec::new();body.read_to_end(&mut bytes).unwrap();let v=json::parse_bytes(&bytes).unwrap();let session=string(&v,"prompt_cache_key").unwrap();let turn=v.get("turn").cloned().unwrap();seen.lock().unwrap().push((session,turn.clone()));
if turn==Value::Number("1".into()) { firsts.fetch_add(1,Ordering::Relaxed);let until=Instant::now()+Duration::from_secs(2);while firsts.load(Ordering::Relaxed)<2 && Instant::now()<until { thread::sleep(Duration::from_millis(1)); }assert_eq!(firsts.load(Ordering::Relaxed),2); }
let reply=r#"{"usage":{"input_tokens":10,"output_tokens":2,"input_tokens_details":{"cached_tokens":0}}}"#;
write!(body.reader.get_mut(),"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",reply.len()).unwrap(); })); } for worker in workers { worker.join().unwrap(); } });
let stats = dir.join("replay.jsonl"); replay( vec![ dir.to_str().unwrap().into(), "--upstream".into(), url, "--concurrency".into(), "2".into(), "--stats".into(), stats.to_str().unwrap().into(), ] .into_iter(), ) .unwrap();
server.join().unwrap(); let seen = observed.lock().unwrap(); for session in ["a", "b"] { assert_eq!( seen.iter() .filter(|(s, _)| s == session) .map(|(_, t)| t.clone()) .collect::<Vec<_>>(),
vec![Value::Number("1".into()), Value::Number("2".into())] ); } assert_eq!(fs::read_to_string(stats).unwrap().lines().count(), 4); fs::remove_dir_all(dir).unwrap(); } }

#[cfg(test)]
mod limit_tests {
    use super::*;

    #[test]
    fn a_full_folder_stops_recording_and_replay_ignores_incomplete_folders() {
        let dir = std::env::temp_dir().join(format!("vade-record-limit-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let mut recorder = Recorder::open(dir.to_str().unwrap()).unwrap();
        assert_eq!(recorder.room, MAX_RECORDINGS);
        // An interrupted request leaves a folder without meta.json.
        drop(recorder.start().unwrap());
        assert_eq!(Recorder::open(dir.to_str().unwrap()).unwrap().room, MAX_RECORDINGS - 1);
        recorder.room = 2;
        let head = Head { first: "POST /v1/responses HTTP/1.1".into(), fields: Vec::new() };
        let mut complete = recorder.start().unwrap();
        complete.request("POST", "/v1/responses", &head, b"{}").unwrap();
        complete.finish("{\"complete\":true}").unwrap();
        assert_eq!(recorder.start().err().map(|e| e.kind()), Some(io::ErrorKind::StorageFull));
        assert!(recorder.begin().is_none());
        let sessions = inventory(&dir).unwrap();
        assert_eq!(sessions.iter().map(Vec::len).sum::<usize>(), 1);
        fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod replay_body_tests {
    use super::*;

    #[test]
    fn recorded_replay_bodies_are_exact_for_defaults_explicit_sampling_and_boundary() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/replay-fixtures")
            .join(format!(
                "replay-body-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        let recorder = Recorder::open(dir.to_str().unwrap()).unwrap();
        let head = Head {
            first: String::new(),
            fields: Vec::new(),
        };
        let runtime = Runtime::single(Origin::parse("http://127.0.0.1:1").unwrap());
        let mut originals = vec![br#"{ "input":"exact", "extension":1.20e0 }"#.to_vec()];
        for &(field, explicit) in super::super::super::sampling::EXPLICIT_SAMPLING_EXAMPLES {
            originals
                .push(format!(r#" {{ "{field}":{explicit}, "extension":1.20e0 }} "#).into_bytes());
        }
        let mut boundary = br#"{"input":""#.to_vec();
        boundary.resize(json::MAX_INPUT_BYTES - 2, b'x');
        boundary.extend_from_slice(b"\"}");
        originals.push(boundary);
        for original in originals {
            let prepared = runtime.prepare("POST", "/v1/responses", &head, &original);
            let mut recording = recorder.start().unwrap();
            recording
                .request("POST", "/v1/responses", &head, &original)
                .unwrap();
            if prepared.body.as_ref() != original {
                recording.effective(&prepared.body).unwrap();
            }
            let saved = Saved {
                dir: recording.dir.clone(),
                method: "POST".into(),
                path: "/v1/responses".into(),
                headers: Vec::new(),
            };
            recording.finish("{\"complete\":true}").unwrap();
            assert_eq!(fs::read(saved.dir.join("request.body")).unwrap(), original);
            assert_eq!(replay_body(&saved).unwrap(), prepared.body.as_ref());
            if original.len() == json::MAX_INPUT_BYTES {
                assert!(replay_body(&saved).unwrap().len() > json::MAX_INPUT_BYTES);
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
