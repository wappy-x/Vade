//! `vade pull` against a fake Hub: a std-only HTTP server on the loopback
//! interface that serves the revision API, a paged tree listing and file
//! downloads with byte ranges, and can refuse, cut, corrupt or fail requests.
use super::*;
use std::{collections::BTreeSet, io::{BufRead, BufReader, Write as _}, net::{TcpListener, TcpStream},
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}};

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const REPO: &str = "org/model";
const README: &str = "---\nlicense: apache-2.0\ntags:\n- text\n---\n# Model\n";

/// What the fake Hub does differently from a healthy one.
#[derive(Clone, Default)]
struct Behavior {
    /// Refuse every request without this bearer token.
    token: Option<String>,
    /// Send only this many bytes of a file on its next request, then close.
    cut: BTreeMap<String, usize>,
    /// Serve these files with one changed byte.
    corrupt: BTreeSet<String>,
    /// Change a byte only on the next request for each named file.
    corrupt_once: BTreeSet<String>,
    /// Ignore Range headers and always send the whole file.
    no_range: bool,
    /// Files per tree page; 0 means one page.
    page: usize,
    /// Answer this many revision requests with HTTP 503 first.
    unavailable: usize,
    /// Serve these files with this many bytes added, and a matching length.
    extra: BTreeMap<String, usize>,
    /// Serve these files with no length and bytes that never end.
    endless: BTreeSet<String>,
    /// Send the next tree page to another address.
    foreign_next: bool,
    /// Name the next tree page with an unquoted rel=next, after another link.
    bare_rel: bool,
    /// Answer a file request with a 307 and a body of this many bytes, as the
    /// Hub does; the address it names serves the file. 0 serves it at once.
    redirect: usize,
    /// Serve these files with their last byte missing, and a matching length.
    short: BTreeSet<String>,
    /// End the next response at this byte, with a matching length.
    short_once: BTreeMap<String, usize>,
}

/// One request the fake Hub saw.
#[derive(Clone, Debug)]
struct Seen { path: String, range: Option<String>, auth: Option<String> }

struct State { behavior: Behavior, seen: Vec<Seen> }

struct Hub { url: String, files: Vec<(String, Vec<u8>, bool)>, state: Arc<Mutex<State>>, stop: Arc<AtomicBool>, task: Option<std::thread::JoinHandle<()>> }

impl Hub {
    fn start(files: Vec<(String, Vec<u8>, bool)>, behavior: Behavior) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State { behavior, seen: Vec::new() }));
        let stop = Arc::new(AtomicBool::new(false));
        let task = {
            let (files, state, stop, url) = (files.clone(), state.clone(), stop.clone(), url.clone());
            std::thread::spawn(move || while !stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => { let _ = serve(stream, &files, &state, &url); }
                    Err(_) => std::thread::sleep(Duration::from_millis(2)),
                }
            })
        };
        Self { url, files, state, stop, task: Some(task) }
    }
    fn seen(&self) -> Vec<Seen> { self.state.lock().unwrap().seen.clone() }
    fn set(&self, change: impl FnOnce(&mut Behavior)) { change(&mut self.state.lock().unwrap().behavior); }
    fn downloads(&self) -> Vec<Seen> { self.seen().into_iter().filter(|seen| seen.path.contains("/resolve/")).collect() }
    fn content(&self, path: &str) -> &[u8] { &self.files.iter().find(|(name, ..)| name == path).unwrap().1 }
}

impl Drop for Hub {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); let _ = self.task.take().unwrap().join(); }
}

fn reply(stream: &mut TcpStream, status: &str, headers: &str, body: &[u8]) -> std::io::Result<()> {
    write!(stream, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n", body.len())?;
    stream.write_all(body)
}

fn serve(mut stream: TcpStream, files: &[(String, Vec<u8>, bool)], state: &Mutex<State>, url: &str) -> std::io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let target = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
    let (mut range, mut auth) = (None, None);
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header == "\r\n" { break; }
        let (name, value) = header.split_once(':').unwrap_or((&header, ""));
        let value = Some(value.trim().to_owned());
        if name.eq_ignore_ascii_case("range") { range = value; } else if name.eq_ignore_ascii_case("authorization") { auth = value; }
    }
    let mut state = state.lock().unwrap();
    state.seen.push(Seen { path: target.clone(), range: range.clone(), auth: auth.clone() });
    if state.behavior.token.as_ref().is_some_and(|token| auth.as_deref() != Some(&format!("Bearer {token}"))) {
        return reply(&mut stream, "401 Unauthorized", "", b"{}");
    }
    let api = format!("/api/models/{REPO}/");
    if let Some(revision) = target.strip_prefix(&format!("{api}revision/")) {
        if state.behavior.unavailable > 0 { state.behavior.unavailable -= 1; return reply(&mut stream, "503 Service Unavailable", "", b"{}"); }
        if revision != "main" && revision != COMMIT { return reply(&mut stream, "404 Not Found", "", b"{}"); }
        return reply(&mut stream, "200 OK", "", format!("{{\"id\":\"{REPO}\",\"sha\":\"{COMMIT}\",\"private\":false}}").as_bytes());
    }
    if let Some(query) = target.strip_prefix(&format!("{api}tree/{COMMIT}?recursive=true")) {
        let cursor: usize = query.strip_prefix("&cursor=").map_or(0, |cursor| cursor.parse().unwrap());
        let size = if state.behavior.page == 0 { files.len() } else { state.behavior.page };
        let mut entries = vec!["{\"type\":\"directory\",\"oid\":\"0000000000000000000000000000000000000000\",\"size\":0,\"path\":\"tensors\"}".to_owned()];
        entries.extend(files.iter().skip(cursor).take(size).map(|(path, bytes, lfs)| {
            let lfs = if *lfs { format!(",\"lfs\":{{\"oid\":\"{}\",\"size\":{},\"pointerSize\":133}}", crate::sha256::sha(bytes), bytes.len()) } else { String::new() };
            let mut blob = format!("blob {}\0", bytes.len()).into_bytes();
            blob.extend_from_slice(bytes);
            format!("{{\"type\":\"file\",\"oid\":\"{}\",\"size\":{},\"path\":\"{path}\"{lfs}}}", fetch::sha1_hex(&blob), bytes.len())
        }));
        let url = if state.behavior.foreign_next { "https://elsewhere.example" } else { url };
        let rel = if state.behavior.bare_rel { format!("<{url}{api}tree/{COMMIT}?recursive=true>; rel=first, ") } else { String::new() };
        let next = if state.behavior.bare_rel { "rel=next" } else { "rel=\"next\"" };
        let link = if cursor + size < files.len() { format!("Link: {rel}<{url}{api}tree/{COMMIT}?recursive=true&cursor={}>; {next}\r\n", cursor + size) } else { String::new() };
        return reply(&mut stream, "200 OK", &link, format!("[{}]", entries.join(",")).as_bytes());
    }
    let (resolve, cache) = (format!("/{REPO}/resolve/{COMMIT}/"), format!("/api/resolve-cache/models/{REPO}/{COMMIT}/"));
    if let Some(path) = target.strip_prefix(&resolve).filter(|_| state.behavior.redirect > 0) {
        return reply(&mut stream, "307 Temporary Redirect", &format!("Location: {cache}{path}\r\n"), &vec![b'r'; state.behavior.redirect]);
    }
    let Some(path) = target.strip_prefix(&resolve).or_else(|| target.strip_prefix(&cache)) else { return reply(&mut stream, "404 Not Found", "", b"") };
    let Some((_, bytes, _)) = files.iter().find(|(name, ..)| name == path) else { return reply(&mut stream, "404 Not Found", "", b"") };
    let mut bytes = bytes.clone();
    if state.behavior.corrupt.contains(path) || state.behavior.corrupt_once.remove(path) { bytes[0] ^= 1; }
    if let Some(extra) = state.behavior.extra.get(path) { bytes.extend(std::iter::repeat_n(b'x', *extra)); }
    if state.behavior.short.contains(path) { bytes.pop(); }
    if let Some(size) = state.behavior.short_once.remove(path) { bytes.truncate(size); }
    let from: usize = range.as_deref().filter(|_| !state.behavior.no_range).and_then(|range| range.strip_prefix("bytes=")?.strip_suffix('-')?.parse().ok()).unwrap_or(0);
    if from >= bytes.len() && from > 0 { return reply(&mut stream, "416 Range Not Satisfiable", "", b""); }
    let (status, headers) = if from > 0 { ("206 Partial Content", format!("Content-Range: bytes {from}-{}/{}\r\n", bytes.len() - 1, bytes.len())) } else { ("200 OK", String::new()) };
    let body = &bytes[from..];
    if state.behavior.endless.contains(path) {
        // No length: the reply ends only when the client closes it (or at 256 MiB).
        write!(stream, "HTTP/1.1 {status}\r\nConnection: close\r\n{headers}\r\n")?;
        stream.write_all(body)?;
        for _ in 0..4096 { stream.write_all(&[b'x'; 65_536])?; }
        return stream.flush();
    }
    if let Some(keep) = state.behavior.cut.remove(path) {
        write!(stream, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n", body.len())?;
        stream.write_all(&body[..keep.min(body.len())])?;
        return stream.flush();
    }
    reply(&mut stream, status, &headers, body)
}

/// The fake repository: a small FP8 checkpoint Vade's launcher accepts,
/// a model card, and files the default selection leaves out.
fn repository() -> Vec<(String, Vec<u8>, bool)> {
    let mut files: Vec<(String, Vec<u8>, bool)> = crate::launch::test_checkpoint().into_iter()
        .map(|(name, bytes)| (name.to_owned(), if name == "model.safetensors" { (0..200_000u32).map(|i| (i % 251) as u8).collect() } else { bytes }, name.ends_with(".safetensors") || name == "tokenizer.json"))
        .collect();
    files.push(("README.md".into(), README.into(), false));
    files.push((".gitattributes".into(), b"*.safetensors filter=lfs\n".to_vec(), false));
    files.push(("pytorch_model.bin".into(), vec![7; 64], true));
    files.push(("images/card.png".into(), vec![1; 8], false));
    // Paths a hostile listing could use to reach the pull state.
    files.push((".vade-pull/lock".into(), b"x".to_vec(), false));
    files.push((".vade-model.json".into(), b"{}".to_vec(), false));
    files
}

struct Fixture { root: PathBuf, hub: Hub, _guard: std::sync::MutexGuard<'static, ()> }

impl Fixture {
    fn new(behavior: Behavior) -> Self { Self::with(repository(), behavior) }
    fn with(files: Vec<(String, Vec<u8>, bool)>, behavior: Behavior) -> Self {
        let guard = crate::runner::process_test_lock();
        let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-pull-test-{}", crate::text::random_hex(6).unwrap()));
        fs::create_dir_all(&root).unwrap();
        Self { root, hub: Hub::start(files, behavior), _guard: guard }
    }
    fn setup(&self, token: Option<&str>) -> Setup {
        Setup { endpoint: self.hub.url.clone(), token: token.map(str::to_owned), home: self.root.clone(), root: self.root.join(".vade"), program: "curl".into(),
            backoff: Duration::from_millis(10), live: false }
    }
    fn folder(&self) -> PathBuf { self.root.join("models").join(REPO) }
    fn pull(&self, args: &str, token: Option<&str>, dry: bool) -> (Result<(), String>, Vec<String>) {
        let request = parse(&args.split_whitespace().map(str::to_owned).collect::<Vec<_>>()).unwrap();
        let mut lines = Vec::new();
        let result = pull(&request, &self.setup(token), dry, &mut |line| { lines.push(line.to_owned()); Ok(()) });
        (result, lines)
    }
    fn part(&self, path: &str) -> PathBuf { self.folder().join(STATE).join(format!("{}.part", &crate::sha256::sha(path.as_bytes())[..24])) }
    fn bad(&self, path: &str) -> Vec<PathBuf> {
        let base = self.folder().join(STATE).join("bad").join(path);
        let prefix = format!("{}.bad-", base.file_name().unwrap().to_string_lossy());
        fs::read_dir(base.parent().unwrap()).into_iter().flatten().flatten().filter(|entry| entry.path() == base || entry.file_name().to_str()
            .and_then(|name| name.strip_prefix(&prefix)).is_some_and(|time| !time.is_empty() && time.bytes().all(|byte| byte.is_ascii_digit())))
            .map(|entry| entry.path()).collect()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.root); }
}

#[test]
fn names_revisions_and_endpoints_are_checked() {
    for good in ["nvidia/Qwen3.8-Flash-Next-FP4-Local", "Qwen/Qwen3-8B", "a/b", "org_1/name.v2"] { assert!(hub::repo(good).is_ok(), "{good}"); }
    for bad in ["model", "a/b/c", "../x", "a/..", "a/.hidden", "-a/b", "a/b c", "a/", "/b", "a/b;rm"] { assert!(hub::repo(bad).is_err(), "{bad}"); }
    for good in ["main", "v1.0", "refs/pr/1", COMMIT] { assert!(hub::revision(good).is_ok(), "{good}"); }
    for bad in ["", "../main", "a//b", "-x", "a b", "a/"] { assert!(hub::revision(bad).is_err(), "{bad}"); }
    assert_eq!(hub::endpoint("https://hub.example/").unwrap(), "https://hub.example");
    assert!(hub::endpoint("http://127.0.0.1:8080").is_ok() && hub::endpoint("http://localhost").is_ok());
    for bad in ["http://hub.example", "ftp://x", "https://", "https://a/b", "http://127.0.0.1.example"] { assert!(hub::endpoint(bad).is_err(), "{bad}"); }
    assert_eq!(hub::encode("a b/c%", true), "a%20b/c%25");
    assert_eq!(hub::revision_url("https://h", "o/n", "refs/pr/1"), "https://h/api/models/o/n/revision/refs%2Fpr%2F1");
    assert_eq!(hub::file_url("https://h", "o/n", COMMIT, "tensors/a b.json"), format!("https://h/o/n/resolve/{COMMIT}/tensors/a%20b.json"));
}

#[test]
fn arguments_parse_into_one_request() {
    let args = |text: &str| parse(&text.split_whitespace().map(str::to_owned).collect::<Vec<_>>());
    assert_eq!(args("org/name").unwrap(), Request { repo: "org/name".into(), revision: "main".into(), include: vec![], to: None });
    assert_eq!(args("org/name@v2 --include *.json --include tensors/* --to /data/m").unwrap(),
        Request { repo: "org/name".into(), revision: "v2".into(), include: vec!["*.json".into(), "tensors/*".into()], to: Some("/data/m".into()) });
    for (text, error) in [("", "Name the model"), ("a/b c/d", "Give one model"), ("a/b --include", "Give --include a value"),
        ("a/b --to x --to y", "--to only once"), ("a/b --bogus", "Unknown option"), ("a/b@", "revision"), ("a", "org/name")] {
        assert!(args(text).unwrap_err().contains(error), "{text}: {:?}", args(text));
    }
}

#[test]
fn selection_keeps_what_vade_serves_and_patterns_replace_it() {
    let remote = |path: &str| Remote { path: path.into(), size: 1, git: "0".repeat(40), sha256: None };
    let files: Vec<Remote> = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "README.md", "LICENSE", "model.safetensors.index.json", "tensors/model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors", ".gitattributes", "pytorch_model.bin", "original/consolidated.pth", "images/a.png", ".cache/x.json"]
        .iter().map(|path| remote(path)).collect();
    let names = |chosen: Vec<Remote>| chosen.into_iter().map(|file| file.path).collect::<Vec<_>>();
    assert_eq!(names(hub::select(&files, &[])), ["LICENSE", "README.md", "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
        "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "model-00002-of-00002.safetensors",
        "tensors/model-00001-of-00002.safetensors"]);
    assert_eq!(names(hub::select(&files, &["*.json".into()])), ["config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json"]);
    // No pattern reaches a path with a dot part, such as the pull state.
    assert!(names(hub::select(&files, &["*".into()])).iter().all(|path| !path.split('/').any(|part| part.starts_with('.'))));
    assert_eq!(names(hub::select(&files, &["tensors/*".into(), "*.bin".into()])), ["pytorch_model.bin", "tensors/model-00001-of-00002.safetensors"]);
    for (pattern, text, matched) in [("*", "a/b", true), ("a?c", "abc", true), ("a?c", "ac", false), ("*.json", "x/y.json", true),
        ("model-*-of-*.safetensors", "model-1-of-2.safetensors", true), ("*a*b", "xaxb", true), ("*a*b", "xbxa", false), ("", "", true), ("", "a", false)] {
        assert_eq!(hub::glob(pattern, text), matched, "{pattern} {text}");
    }
}

#[test]
fn listing_headers_card_and_hashes_parse() {
    let page = format!(r#"[{{"type":"directory","oid":"{z}","size":0,"path":"d"}},{{"type":"file","oid":"{a}","size":3,"path":"d/x.json"}},
        {{"type":"file","oid":"{a}","size":5,"path":"w.safetensors","lfs":{{"oid":"{s}","size":5,"pointerSize":1}}}}]"#, z = "0".repeat(40), a = "a".repeat(40), s = "b".repeat(64));
    assert_eq!(hub::tree_page(&page).unwrap(), vec![
        Remote { path: "d/x.json".into(), size: 3, git: "a".repeat(40), sha256: None },
        Remote { path: "w.safetensors".into(), size: 5, git: "a".repeat(40), sha256: Some("b".repeat(64)) }]);
    for bad in [page.replace("d/x.json", "../x"), page.replace("\"size\":5,\"pointerSize\"", "\"size\":6,\"pointerSize\""), page.replace(&"b".repeat(64), "short"), "{}".into(), "[".into()] {
        assert!(hub::tree_page(&bad).is_err(), "{bad}");
    }
    let next = "https://h/api/models/o/n/tree/c?recursive=true&cursor=x";
    assert_eq!(hub::next_page(&format!("HTTP/1.1 200 OK\r\nlink: <{next}>; rel=\"next\"\r\n"), "https://h").unwrap().as_deref(), Some(next));
    assert_eq!(hub::next_page("Link: <https://h/api/models/x>; rel=\"prev\"\r\n", "https://h"), Ok(None));
    assert_eq!(hub::next_page("Content-Type: application/json\r\n", "https://h"), Ok(None));
    // A next page on another address, or a broken link, is an error: the list would be short.
    // RFC 8288 forms: an unquoted rel, several relation types, parameters in
    // any order, a quoted comma, and several links in one header.
    for good in [format!("Link: <{next}>; rel=next\r\n"), format!("Link: <{next}>;rel=\"prev next\"\r\n"), format!("Link: <{next}>; title=\"a, b\"; REL=Next\r\n"),
        format!("Link: <https://h/api/models/o/n/tree/c?recursive=true>; rel=first, <{next}>; rel=next\r\n")] {
        assert_eq!(hub::next_page(&good, "https://h").unwrap().as_deref(), Some(next), "{good}");
    }
    assert_eq!(hub::next_page("Link: <https://h/api/models/x>; rel=nextpage\r\n", "https://h"), Ok(None));
    for bad in ["Link: <https://elsewhere/api/models/x>; rel=\"next\"\r\n", "Link: https://h/api/models/x; rel=\"next\"\r\n",
        "Link: <https://elsewhere/api/models/x>; rel=next\r\n", "Link: <https://h/api/models/x>; rel=\"next\r\n"] {
        assert!(hub::next_page(bad, "https://h").unwrap_err().contains("another address or a malformed link"), "{bad}");
    }
    assert_eq!(hub::license(README).as_deref(), Some("apache-2.0"));
    assert_eq!(hub::license("---\nlicense: \"other\"\n---\n").as_deref(), Some("other"));
    assert_eq!(hub::license("---\nlicense: other\nlicense_name: nvidia-open-model-agreement\n---\n").as_deref(), Some("nvidia-open-model-agreement"));
    assert_eq!(hub::license("# no front matter\nlicense: mit\n"), None);
    assert_eq!(fetch::sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    assert_eq!(fetch::sha1_hex(&[b'a'; 1000]), "291e9a6c66994949b57ba5e650361e98fc36b1ba");
    assert_eq!(utc(std::time::UNIX_EPOCH + Duration::from_secs(1_793_246_400)), "2026-10-29T04:00:00Z");
    assert_eq!(utc(std::time::UNIX_EPOCH + Duration::from_secs(951_782_400)), "2000-02-29T00:00:00Z");
}

#[test]
fn pull_downloads_checks_and_records_a_servable_model() {
    let f = Fixture::new(Behavior { page: 3, ..Behavior::default() });
    let (result, lines) = f.pull(REPO, None, false);
    result.unwrap();
    let folder = fs::canonicalize(f.folder()).unwrap();
    let chosen = ["README.md", "config.json", "generation_config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"];
    for name in chosen { assert_eq!(fs::read(folder.join(name)).unwrap(), f.hub.content(name), "{name}"); }
    for name in [".gitattributes", "pytorch_model.bin", "images/card.png"] { assert!(!folder.join(name).exists(), "{name}"); }
    assert!(lines[0].starts_with(&format!("Pulling {REPO} at {} (main): 7 files", &COMMIT[..12])), "{lines:?}");
    assert!(lines.iter().any(|line| line.starts_with("Supported: Qwen3.8 Flash-Next FP8")), "{lines:?}");
    assert!(lines.iter().any(|line| line.contains("vade up --fp8")), "{lines:?}");
    let manifest = json::parse(&fs::read_to_string(folder.join(MANIFEST)).unwrap()).unwrap();
    assert_eq!((manifest["schema"].as_str(), manifest["repo"].as_str(), manifest["revision"].as_str(), manifest["commit"].as_str(), manifest["license"].as_str()),
        (Some("vade.model.v1"), Some(REPO), Some("main"), Some(COMMIT), Some("apache-2.0")));
    assert_eq!(manifest["support"]["state"].as_str(), Some("supported"));
    assert_eq!(manifest["support"]["route"].as_str(), Some("fp8"));
    assert_eq!(manifest["files"].as_array().unwrap().len(), 7);
    assert!(manifest["pulled_at"].as_str().unwrap().ends_with('Z'));
    let rows = fs::read_to_string(folder.parent().unwrap().join("model.sha256")).unwrap();
    let expected: String = { let mut names = chosen.to_vec(); names.sort(); names.iter().map(|name| format!("{}  ./{name}\n", crate::sha256::sha(f.hub.content(name)))).collect() };
    assert_eq!(rows, expected);
    // The launcher's own manifest reader accepts the list.
    assert!(crate::launch::supplied_manifest(&folder).unwrap().unwrap().contains(&format!("{}  model.safetensors\n", crate::sha256::sha(f.hub.content("model.safetensors")))));
    assert!(!fs::read_dir(folder.join(STATE)).unwrap().flatten().any(|entry| entry.file_name().to_string_lossy().ends_with(".part")));
    // The tree came in four pages of three files, and no request carried a token.
    assert_eq!(f.hub.seen().iter().filter(|seen| seen.path.contains("/tree/")).count(), 4);
    assert!(f.hub.seen().iter().all(|seen| seen.auth.is_none()));
    // A second pull finds every file checked and unchanged: no downloads.
    let before = f.hub.downloads().len();
    let (again, lines) = f.pull(REPO, None, false);
    again.unwrap();
    assert_eq!(f.hub.downloads().len(), before, "{lines:?}");
    // A changed file is fetched again.
    fs::write(folder.join("config.json"), b"{}").unwrap();
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(f.hub.downloads().len(), before + 1);
    assert_eq!(fs::read(folder.join("config.json")).unwrap(), f.hub.content("config.json"));
    // vade models lists it with its size, source and support state.
    let (rows, text) = models::tests_list(&f.root.join("models"));
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, REPO);
    assert_eq!(rows[0].1, chosen.iter().map(|name| f.hub.content(name).len() as u64).sum::<u64>() + fs::metadata(folder.join(MANIFEST)).unwrap().len());
    assert_eq!((rows[0].2.as_deref(), rows[0].3.as_deref()), (Some(REPO), Some(COMMIT)));
    assert_eq!(rows[0].4.route, models::Route::Fp8);
    assert!(text.contains(&format!("{REPO}@{}", &COMMIT[..12])) && text.contains("Supported"), "{text}");
}

#[test]
fn interrupted_downloads_resume_from_the_partial_file() {
    let f = Fixture::new(Behavior { cut: [("model.safetensors".into(), 70_000)].into(), ..Behavior::default() });
    f.pull(REPO, None, false).0.unwrap();
    let ranges: Vec<_> = f.hub.downloads().into_iter().filter(|seen| seen.path.ends_with("/model.safetensors")).map(|seen| seen.range).collect();
    assert_eq!(ranges, [None, Some("bytes=70000-".into())]);
    assert_eq!(fs::read(f.folder().join("model.safetensors")).unwrap(), f.hub.content("model.safetensors"));
    // A partial file left by an earlier, stopped pull is continued too.
    drop(f);
    let f = Fixture::new(Behavior::default());
    crate::text::private_dir(f.part("x").parent().unwrap()).unwrap();
    fs::write(f.part("model.safetensors"), &f.hub.content("model.safetensors")[..12_345]).unwrap();
    f.pull(REPO, None, false).0.unwrap();
    let ranges: Vec<_> = f.hub.downloads().into_iter().filter(|seen| seen.path.ends_with("/model.safetensors")).map(|seen| seen.range).collect();
    assert_eq!(ranges, [Some("bytes=12345-".into())]);
    assert_eq!(fs::read(f.folder().join("model.safetensors")).unwrap(), f.hub.content("model.safetensors"));
    // A server that cannot continue a file sends it again from the start.
    drop(f);
    let f = Fixture::new(Behavior { no_range: true, ..Behavior::default() });
    crate::text::private_dir(f.part("x").parent().unwrap()).unwrap();
    fs::write(f.part("model.safetensors"), &f.hub.content("model.safetensors")[..999]).unwrap();
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(fs::read(f.folder().join("model.safetensors")).unwrap(), f.hub.content("model.safetensors"));
}

#[test]
fn a_file_that_fails_its_hash_is_kept_and_retried_only_once() {
    for (name, kind) in [("model.safetensors", "SHA-256"), ("config.json", "Git ID")] {
        let f = Fixture::new(Behavior { corrupt: [name.to_owned()].into(), ..Behavior::default() });
        let error = f.pull(REPO, None, false).0.unwrap_err();
        assert!(error.starts_with(&format!("{name} has {kind} ")) && error.contains("Pull kept the latest bad copy"), "{error}");
        assert!(!f.part(name).exists() && !f.folder().join(name).exists());
        assert_eq!(f.bad(name).len(), 1);
        assert_eq!(f.hub.downloads().iter().filter(|seen| seen.path.ends_with(&format!("/{name}"))).count(), 2);
        let mut bad = f.hub.content(name).to_vec();
        bad[0] ^= 1;
        for path in f.bad(name) { assert_eq!(fs::read(path).unwrap(), bad); }
        assert!(!f.folder().join(MANIFEST).exists());
        // vade models shows the unfinished pull.
        assert_eq!(models::tests_list(&f.root.join("models")).0[0].4.route, models::Route::Incomplete);
    }
}

/// A fixture large enough to test a transfer that stops between hash blocks.
fn resume_repository() -> Vec<(String, Vec<u8>, bool)> {
    vec![("model.safetensors".into(), (0..3_000_000).map(|i| (i % 251) as u8).collect(), true)]
}

#[test]
fn a_cut_at_forty_percent_resumes_with_the_clean_sha() {
    let f = Fixture::with(resume_repository(), Behavior { cut: [("model.safetensors".into(), 1_200_000)].into(), ..Behavior::default() });
    let (result, lines) = f.pull(REPO, None, false);
    result.unwrap();
    let ranges: Vec<_> = f.hub.downloads().into_iter().map(|seen| seen.range).collect();
    assert_eq!(ranges, [None, Some("bytes=1200000-".into())]);
    let sha = crate::sha256::digest_reader(fs::File::open(f.folder().join("model.safetensors")).unwrap()).unwrap();
    assert_eq!(sha, crate::sha256::sha(f.hub.content("model.safetensors")));
    assert!(lines.last().unwrap().contains("Verified 1 files, 0.003 GiB") && lines.last().unwrap().contains("1 resumed"), "{lines:?}");
    // A clean pull to another folder gives the same SHA-256.
    let clean = f.root.join("clean");
    f.pull(&format!("{REPO} --to {}", clean.display()), None, false).0.unwrap();
    assert_eq!(crate::sha256::digest_reader(fs::File::open(clean.join("model.safetensors")).unwrap()).unwrap(), sha);
    // The next pull starts with the bytes saved by a prior process.
    let saved = f.part("model.safetensors");
    fs::rename(f.folder().join("model.safetensors"), &saved).unwrap();
    fs::OpenOptions::new().write(true).open(&saved).unwrap().set_len(1_200_000).unwrap();
    let before = f.hub.downloads().len();
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(f.hub.downloads()[before].range.as_deref(), Some("bytes=1200000-"));
    assert_eq!(crate::sha256::digest_reader(fs::File::open(f.folder().join("model.safetensors")).unwrap()).unwrap(), sha);
}

#[test]
fn a_server_that_ignores_range_restarts_with_the_clean_sha() {
    let f = Fixture::with(resume_repository(), Behavior { no_range: true, ..Behavior::default() });
    crate::text::private_dir(f.part("model.safetensors").parent().unwrap()).unwrap();
    fs::write(f.part("model.safetensors"), &f.hub.content("model.safetensors")[..1_200_000]).unwrap();
    let (result, lines) = f.pull(REPO, None, false);
    result.unwrap();
    let ranges: Vec<_> = f.hub.downloads().into_iter().map(|seen| seen.range).collect();
    assert_eq!(ranges, [Some("bytes=1200000-".into()), None]);
    assert_eq!(crate::sha256::digest_reader(fs::File::open(f.folder().join("model.safetensors")).unwrap()).unwrap(), crate::sha256::sha(f.hub.content("model.safetensors")));
    assert!(lines.last().unwrap().contains("0 resumed"), "{lines:?}");
}

#[test]
fn a_range_restart_prints_one_notice() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "pull::tests::a_server_that_ignores_range_restarts_with_the_clean_sha", "--nocapture"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(stderr.lines().filter(|line| line.contains("Pull starts the file again.")).count(), 1, "{stderr}");
}

#[test]
fn bad_bytes_are_cleared_after_one_fresh_pull_succeeds() {
    let f = Fixture::with(resume_repository(), Behavior { corrupt_once: ["model.safetensors".into()].into(), ..Behavior::default() });
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(f.hub.downloads().len(), 2);
    assert!(f.hub.downloads().iter().all(|seen| seen.range.is_none()));
    assert!(f.bad("model.safetensors").is_empty());
    assert_eq!(crate::sha256::digest_reader(fs::File::open(f.folder().join("model.safetensors")).unwrap()).unwrap(), crate::sha256::sha(f.hub.content("model.safetensors")));
    // Another pull makes no request.
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(f.hub.downloads().len(), 2);
    assert!(f.bad("model.safetensors").is_empty());
}

#[test]
fn bad_saved_files_are_cleared_after_a_fresh_pull() {
    for size in [1_200_000, 3_000_000, 3_000_001] {
        let f = Fixture::with(resume_repository(), Behavior::default());
        crate::text::private_dir(f.part("model.safetensors").parent().unwrap()).unwrap();
        let mut bad = f.hub.content("model.safetensors").to_vec();
        bad[0] ^= 1;
        bad.resize(size, 0);
        fs::write(f.part("model.safetensors"), &bad).unwrap();
        let (result, lines) = f.pull(REPO, None, false);
        result.unwrap();
        if size < 3_000_000 {
            assert_eq!(f.hub.downloads().len(), 2);
            assert_eq!(f.hub.downloads()[0].range.as_deref(), Some("bytes=1200000-"));
        } else { assert_eq!(f.hub.downloads().len(), 1); }
        assert!(f.hub.downloads().last().unwrap().range.is_none());
        assert!(lines.last().unwrap().contains("0 resumed"), "{lines:?}");
        assert!(f.bad("model.safetensors").is_empty());
        assert_eq!(fs::read(f.folder().join("model.safetensors")).unwrap(), f.hub.content("model.safetensors"));
    }
}

#[test]
fn a_bad_model_file_is_removed_when_its_replacement_also_fails() {
    let f = Fixture::with(resume_repository(), Behavior { corrupt: ["model.safetensors".into()].into(), ..Behavior::default() });
    fs::create_dir_all(f.folder()).unwrap();
    fs::write(f.folder().join("model.safetensors"), b"bad saved file").unwrap();
    assert!(f.pull(REPO, None, false).0.unwrap_err().contains("Pull kept the latest bad copy"));
    assert!(!f.folder().join("model.safetensors").exists() && !f.folder().join(MANIFEST).exists());
    assert_eq!(f.bad("model.safetensors").len(), 1);
    assert_ne!(fs::read(&f.bad("model.safetensors")[0]).unwrap(), b"bad saved file");
    assert_eq!(f.hub.downloads().len(), 2);
}

#[test]
fn file_progress_uses_gib_and_waits_five_seconds() {
    let mut progress = Progress::new(false, 4 << 30, 1 << 30);
    let now = Instant::now();
    let line = |progress: &mut Progress, seconds, finish| progress.line("model.safetensors", 1 << 30, 2 << 30, 10_000_000,
        Duration::from_secs(seconds), finish, now + Duration::from_secs(seconds));
    assert!(line(&mut progress, 4, None).is_none());
    assert_eq!(line(&mut progress, 5, None).unwrap(), "model.safetensors 1.000 / 2.000 GiB 2.0 MB/s total 2.000 / 4.000 GiB");
    assert!(line(&mut progress, 6, None).is_none());
    assert!(line(&mut progress, 9, None).is_none());
    assert!(line(&mut progress, 10, None).is_some());
    assert!(line(&mut progress, 10, None).is_none());
}

#[test]
fn repeated_failures_keep_only_the_latest_bad_copy() {
    let name = "config.json";
    let f = Fixture::with(vec![(name.into(), b"{}".to_vec(), true)], Behavior { corrupt: [name.into()].into(), ..Behavior::default() });
    assert!(f.pull(REPO, None, false).0.is_err());
    assert_eq!(f.bad(name).len(), 1);
    assert_eq!(fs::read(&f.bad(name)[0]).unwrap(), b"z}");
    f.hub.set(|behavior| { behavior.corrupt.clear(); behavior.extra.insert(name.into(), 1); });
    assert!(f.pull(REPO, None, false).0.is_err());
    assert_eq!(f.bad(name).len(), 1);
    assert_eq!(fs::read(&f.bad(name)[0]).unwrap(), b"{}x");
    assert_eq!(f.hub.downloads().len(), 4);
    assert!(!f.folder().join(name).exists());
    f.hub.set(|behavior| behavior.extra.clear());
    f.pull(REPO, None, false).0.unwrap();
    assert!(f.bad(name).is_empty());
    assert_eq!(fs::read(f.folder().join(name)).unwrap(), b"{}");
}

#[test]
fn a_good_file_clears_its_bad_copy_before_a_later_file_fails() {
    let f = Fixture::with(vec![("a.json".into(), b"{}".to_vec(), true), ("b.json".into(), b"[]".to_vec(), true)],
        Behavior { corrupt_once: ["a.json".into()].into(), corrupt: ["b.json".into()].into(), ..Behavior::default() });
    assert!(f.pull(REPO, None, false).0.unwrap_err().starts_with("b.json has SHA-256"));
    assert_eq!(fs::read(f.folder().join("a.json")).unwrap(), b"{}");
    assert!(f.bad("a.json").is_empty());
    assert_eq!(f.bad("b.json").len(), 1);
    assert!(!f.folder().join("b.json").exists());
}

#[test]
fn good_files_clear_old_bad_names_without_clearing_other_files() {
    let name = "tensors/a.json";
    let f = Fixture::with(vec![(name.into(), b"{}".to_vec(), true)], Behavior::default());
    let bad = f.folder().join(STATE).join("bad/tensors");
    crate::text::private_dir(&bad).unwrap();
    for copy in ["a.json", "a.json.bad-1", "a.json.bad-2", "a.json.bad-note", "b.json", "b.json.bad-1"] { fs::write(bad.join(copy), b"saved").unwrap(); }
    f.pull(REPO, None, false).0.unwrap();
    assert!(f.bad(name).is_empty());
    for copy in ["a.json.bad-note", "b.json", "b.json.bad-1"] { assert_eq!(fs::read(bad.join(copy)).unwrap(), b"saved"); }
    // A present file also clears its saved bad copies without a download.
    for copy in ["a.json", "a.json.bad-3"] { fs::write(bad.join(copy), b"saved").unwrap(); }
    f.pull(REPO, None, false).0.unwrap();
    assert!(f.bad(name).is_empty());
    assert_eq!(f.hub.downloads().len(), 1);
    assert_eq!(fs::read(bad.join("b.json")).unwrap(), b"saved");
}

#[test]
fn a_new_failure_replaces_bad_copies_from_older_pulls() {
    let name = "tensors/a.json";
    let f = Fixture::with(vec![(name.into(), b"{}".to_vec(), true)], Behavior { corrupt: [name.into()].into(), ..Behavior::default() });
    let bad = f.folder().join(STATE).join("bad/tensors");
    crate::text::private_dir(&bad).unwrap();
    for copy in ["a.json", "a.json.bad-1", "a.json.bad-2"] { fs::write(bad.join(copy), b"old").unwrap(); }
    assert!(f.pull(REPO, None, false).0.is_err());
    assert_eq!(f.bad(name), [bad.join("a.json")]);
    assert_eq!(fs::read(bad.join("a.json")).unwrap(), b"z}");
}

#[test]
fn an_old_revision_is_replaced_without_a_saved_bad_copy() {
    let name = "config.json";
    let f = Fixture::with(vec![(name.into(), b"{}".to_vec(), true)], Behavior::default());
    fs::create_dir_all(f.folder()).unwrap();
    fs::write(f.folder().join(name), b"{\"old\":true}").unwrap();
    fs::write(f.folder().join(MANIFEST), format!("{{\"schema\":\"vade.model.v1\",\"repo\":\"{REPO}\",\"commit\":\"{}\",\"files\":[]}}", "f".repeat(40))).unwrap();
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(fs::read(f.folder().join(name)).unwrap(), b"{}");
    assert!(!f.folder().join(STATE).join("bad").exists());
    assert_eq!(models::manifest(&f.folder()).unwrap()["commit"].as_str(), Some(COMMIT));
}

#[test]
fn a_short_successful_response_resumes_with_range() {
    let name = "model.safetensors";
    let f = Fixture::with(resume_repository(), Behavior { short_once: [(name.into(), 1_200_000)].into(), ..Behavior::default() });
    let (result, lines) = f.pull(REPO, None, false);
    result.unwrap();
    let ranges: Vec<_> = f.hub.downloads().into_iter().map(|seen| seen.range).collect();
    assert_eq!(ranges, [None, Some("bytes=1200000-".into())]);
    assert!(lines.last().unwrap().contains("1 resumed"), "{lines:?}");
    assert_eq!(crate::sha256::digest_reader(fs::File::open(f.folder().join(name)).unwrap()).unwrap(), crate::sha256::sha(f.hub.content(name)));
    assert!(!f.folder().join(STATE).join("bad").exists());
}

#[test]
fn file_completion_bypasses_the_progress_timer() {
    for live in [false, true] {
        let mut progress = Progress::new(live, 3 << 30, 1 << 30);
        let now = Instant::now();
        assert!(progress.line("a", 1 << 29, 1 << 30, 1 << 29, Duration::from_secs(5), None, now).is_some());
        let line = progress.line("a", 1 << 30, 1 << 30, 1 << 30, Duration::from_secs(5), Some("done"), now).unwrap();
        assert!(line.contains("MB/s done total 2.000 / 3.000 GiB"), "{line}");
        assert!(progress.render(&line, true).ends_with('\n'));
        let line = progress.line("b", 1 << 30, 1 << 30, 0, Duration::ZERO, Some("present"), now).unwrap();
        assert!(line.contains("present total") && !line.contains("MB/s"), "{line}");
    }
}

#[test]
fn terminal_progress_updates_in_place_and_keeps_the_total() {
    let mut progress = Progress::new(true, 4 << 30, 2 << 30);
    let now = Instant::now();
    let line = progress.line("a", 1 << 30, 2 << 30, 1_000_000, Duration::from_secs(1), None, now).unwrap();
    assert!(line.contains("total 3.000 / 4.000 GiB"), "{line}");
    let rendered = progress.render(&line, false);
    assert!(rendered.starts_with("\r\x1b[2K") && !rendered.ends_with('\n'), "{rendered:?}");
    assert!(progress.line("a", 1 << 30, 2 << 30, 1_000_000, Duration::from_secs(1), None, now + Duration::from_millis(249)).is_none());
    assert!(progress.line("a", 1 << 30, 2 << 30, 1_000_000, Duration::from_secs(1), None, now + Duration::from_millis(250)).is_some());
    assert!(progress.render(&line, true).ends_with('\n'));
}

#[test]
fn pull_progress_prints_done_and_present_for_every_file() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "pull::tests::pull_downloads_checks_and_records_a_servable_model", "--nocapture"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);
    let lines: Vec<_> = stderr.lines().filter(|line| line.starts_with('[')).collect();
    assert_eq!(lines.len(), 21, "{stderr}");
    assert!(lines[..7].iter().all(|line| line.contains(" done total ")), "{stderr}");
    assert!(lines[7..14].iter().all(|line| line.contains(" present total ") && !line.contains("MB/s")), "{stderr}");
    assert!(lines[6].starts_with("[7/7] ") && lines[13].starts_with("[7/7] "), "{stderr}");
}

#[test]
fn bad_copy_cleanup_refuses_links() {
    let name = "config.json";
    let f = Fixture::with(vec![(name.into(), b"{}".to_vec(), true)], Behavior { corrupt: [name.into()].into(), ..Behavior::default() });
    let state = f.folder().join(STATE);
    crate::text::private_dir(&state).unwrap();
    let outside = f.root.join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join(name), b"keep").unwrap();
    std::os::unix::fs::symlink(&outside, state.join("bad")).unwrap();
    assert!(f.pull(REPO, None, false).0.unwrap_err().contains("is a symlink"));
    assert_eq!(fs::read(outside.join(name)).unwrap(), b"keep");
    assert!(!f.folder().join(name).exists());
}

#[test]
fn gated_models_need_a_token_that_is_never_shown() {
    let token = "hf_testTOKEN123";
    let f = Fixture::new(Behavior { token: Some(token.into()), ..Behavior::default() });
    let error = f.pull(REPO, None, false).0.unwrap_err();
    assert!(error.starts_with("The Hub refused access to org/model. Check the model ID: the Hub also refuses an ID that does not exist.") && error.contains("HF_TOKEN"), "{error}");
    let error = f.pull(REPO, Some("hf_wrong"), false).0.unwrap_err();
    assert!(error.contains("The token was sent") && !error.contains("hf_wrong"), "{error}");
    let (result, lines) = f.pull(REPO, Some(token), false);
    result.unwrap();
    assert!(lines.iter().all(|line| !line.contains(token)));
    assert!(f.hub.downloads().iter().all(|seen| seen.auth.as_deref() == Some(&format!("Bearer {token}") as &str)));
    assert!(!fs::read_to_string(f.folder().join(MANIFEST)).unwrap().contains(token));
}

#[test]
fn busy_servers_are_retried_and_missing_names_are_final() {
    let f = Fixture::new(Behavior { unavailable: 2, ..Behavior::default() });
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(f.hub.seen().iter().filter(|seen| seen.path.contains("/revision/")).count(), 3);
    let error = f.pull(&format!("{REPO}@v9"), None, false).0.unwrap_err();
    assert!(error.starts_with("The Hub has no model org/model at revision v9."), "{error}");
    let error = f.pull("org/missing", None, false).0.unwrap_err();
    assert!(error.contains("org/missing"), "{error}");
    let error = f.pull(&format!("{REPO}@{}", "f".repeat(40)), None, false).0.unwrap_err();
    assert!(error.contains("revision"), "{error}");
}

#[test]
fn dry_runs_patterns_targets_and_locks() {
    let f = Fixture::new(Behavior::default());
    let (result, lines) = f.pull(REPO, None, true);
    result.unwrap();
    assert!(!f.folder().exists() && f.hub.downloads().is_empty());
    assert_eq!(lines.len(), 9, "{lines:?}");
    assert_eq!(lines.last().unwrap(), "Dry run: nothing was downloaded.");
    let error = f.pull(&format!("{REPO} --include *.gguf"), None, true).0.unwrap_err();
    assert!(error.contains("No file in org/model matches"), "{error}");
    let to = f.root.join("elsewhere/metadata");
    f.pull(&format!("{REPO} --include *.json --to {}", to.display()), None, false).0.unwrap();
    let mut names: Vec<_> = fs::read_dir(&to).unwrap().flatten().map(|entry| entry.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    assert_eq!(names, [".vade-model.json", ".vade-pull", ".vade-verified", "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]);
    // Without its shard the folder is not servable, and the record says why.
    let manifest = json::parse(&fs::read_to_string(to.join(MANIFEST)).unwrap()).unwrap();
    assert_eq!(manifest["support"]["state"].as_str(), Some("not supported yet"));
    assert!(manifest["support"]["reason"].as_str().unwrap().contains("model.safetensors"), "{manifest:?}");
    // One pull at a time per folder.
    let held = lock(&to.join(STATE)).unwrap();
    let error = f.pull(&format!("{REPO} --to {}", to.display()), None, false).0.unwrap_err();
    assert!(error.starts_with("Another vade pull is writing this model folder."), "{error}");
    drop(held);
    // A symlink inside the folder cannot redirect a download.
    let outside = f.root.join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::remove_file(to.join("config.json")).unwrap();
    std::os::unix::fs::symlink(outside.join("config.json"), to.join("config.json")).unwrap();
    let error = f.pull(&format!("{REPO} --include *.json --to {}", to.display()), None, false).0.unwrap_err();
    assert!(error.contains("is a symlink"), "{error}");
    assert!(!outside.join("config.json").exists());
}

#[test]
fn support_names_the_route_or_the_reason() {
    let f = Fixture::new(Behavior::default());
    let folder = f.root.join("plain");
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("config.json"), r#"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#).unwrap();
    let support = models::support(&folder, None);
    assert_eq!(support.route, models::Route::Unsupported);
    assert!(support.state() == "Not supported yet" && support.text().starts_with("This engine cannot load this model"), "{support:?}");
    let catalog: BTreeMap<String, String> = crate::catalog::MODEL_FILES.iter().map(|(name, sha)| (name.to_string(), sha.to_string())).collect();
    // Recorded hashes count only for files that are in the folder.
    assert_eq!(models::support(&folder, Some(&catalog)).route, models::Route::Unsupported);
    for (name, _) in crate::catalog::MODEL_FILES { if *name != "config.json" { fs::write(folder.join(name), b"").unwrap(); } }
    assert_eq!(models::support(&folder, Some(&catalog)).route, models::Route::Qwen3Bf16);
    // Names alone, with no record of their hashes, are not enough.
    assert_eq!(models::support(&folder, None).route, models::Route::Unsupported);
    let mut wrong = catalog.clone();
    wrong.insert("config.json".into(), "0".repeat(64));
    assert_eq!(models::support(&folder, Some(&wrong)).route, models::Route::Unsupported);
    assert!(models::tests_list(&f.root.join("nothing")).0.is_empty());
}

#[test]
fn hostile_listings_and_local_files_cannot_escape_block_or_panic() {
    let f = Fixture::new(Behavior::default());
    // --include '*' takes the extra files, but never a path with a dot part.
    f.pull(&format!("{REPO} --include *"), None, false).0.unwrap();
    assert!(f.hub.downloads().iter().all(|seen| !seen.path.contains("/.")), "{:?}", f.hub.downloads());
    let folder = f.folder();
    assert_eq!(fs::read(folder.join("pytorch_model.bin")).unwrap(), f.hub.content("pytorch_model.bin"));
    assert_eq!(json::parse(&fs::read_to_string(folder.join(MANIFEST)).unwrap()).unwrap()["schema"].as_str(), Some("vade.model.v1"));
    assert_eq!(fs::read(folder.join(STATE).join("lock")).unwrap(), b"");
    // Sizes past 16 TiB, and so any byte total that could overflow, are refused.
    let page = format!(r#"[{{"type":"file","oid":"{}","size":{},"path":"a.json"}}]"#, "a".repeat(40), i64::MAX);
    assert!(hub::tree_page(&page).unwrap_err().contains("16 TiB"));
    // A malformed record never panics vade models; its source is left out.
    let other = f.root.join("models/other/model");
    fs::create_dir_all(&other).unwrap();
    fs::write(other.join(MANIFEST), r#"{"repo":"org/name","commit":"0123456789aé0123456789"}"#).unwrap();
    fs::write(other.join("config.json"), "{}").unwrap();
    let (rows, text) = models::tests_list(&f.root.join("models"));
    let row = rows.iter().find(|row| row.0 == "other/model").unwrap();
    assert_eq!((row.2.as_deref(), row.3.as_deref(), row.4.route.clone()), (Some("org/name"), None, models::Route::Unsupported));
    // The table, where the old panic was, shows the source with no commit.
    assert!(text.lines().any(|line| line.starts_with("  other/model ") && line.contains(" org/name@? ")), "{text}");
    // A FIFO where a record should be does not block the list.
    fs::remove_file(other.join(MANIFEST)).unwrap();
    assert!(std::process::Command::new("mkfifo").arg(other.join(MANIFEST)).status().unwrap().success());
    assert_eq!(models::tests_list(&f.root.join("models")).0.len(), 2);
}

#[test]
fn the_token_file_is_bounded_and_must_be_a_regular_file() {
    let f = Fixture::new(Behavior::default());
    let path = f.root.join("hf-token");
    assert_eq!(token_file(&path).unwrap(), None);
    fs::write(&path, "  hf_abc123\n").unwrap();
    assert_eq!(token_file(&path).unwrap().as_deref(), Some("hf_abc123"));
    // A link to the real file is followed.
    let link = f.root.join("hf-token-link");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    assert_eq!(token_file(&link).unwrap().as_deref(), Some("hf_abc123"));
    fs::write(&path, "hf_a\"b").unwrap();
    assert!(token_file(&path).unwrap_err().contains("characters a token cannot have"));
    fs::write(&path, vec![b'a'; 4097]).unwrap();
    assert!(token_file(&path).unwrap_err().contains("too large"));
    fs::remove_file(&path).unwrap();
    assert!(std::process::Command::new("mkfifo").arg(&path).status().unwrap().success());
    assert!(token_file(&path).unwrap_err().contains("not a regular file"));
    let error = token_file(&f.root).unwrap_err();
    assert!(error.contains("not a regular file") || error.contains("Cannot read"), "{error}");
}

/// Run one test of this module again in a child process with `env` set, and
/// fail with its output when it fails.
fn child(test: &str, env: &[(&str, &str)], unset: &[&str]) {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.args(["--exact", &format!("pull::tests::{test}"), "--nocapture", "--test-threads", "1"]).envs(env.iter().copied());
    for name in unset { command.env_remove(name); }
    let output = command.output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success() && text.contains("1 passed"), "{text}");
}

/// When set, the hash hook sends SIGINT to this process once a hash has read
/// a whole block, and waits until the handler records it. `LATE` records a
/// hash that kept reading after that.
static ARMED: AtomicBool = AtomicBool::new(false);
static LATE: AtomicBool = AtomicBool::new(false);
fn interrupt_once(read: u64) {
    if crate::runner::interrupted() { LATE.store(true, Ordering::Relaxed); return; }
    if read < 1 << 20 || !ARMED.swap(false, Ordering::Relaxed) { return; }
    let pid = std::process::id().to_string();
    assert!(std::process::Command::new("sh").args(["-c", "kill -INT \"$1\"", "sh", &pid]).status().unwrap().success());
    let until = Instant::now() + Duration::from_secs(10);
    while !crate::runner::interrupted() && Instant::now() < until { std::thread::sleep(Duration::from_millis(5)); }
    assert!(crate::runner::interrupted());
}

#[test]
fn ctrl_c_during_a_hash_stops_the_pull_and_publishes_nothing() {
    if std::env::var_os("VADE_TEST_PULL_CANCEL").is_none() {
        let _guard = crate::runner::process_test_lock();
        return child("ctrl_c_during_a_hash_stops_the_pull_and_publishes_nothing", &[("VADE_TEST_PULL_CANCEL", "1")], &[]);
    }
    // In the child: a real SIGINT arrives inside the hash loop of a 3 MiB file.
    fetch::HASH_HOOK.set(interrupt_once).unwrap();
    let files = repository().into_iter().map(|(name, bytes, lfs)| if name == "model.safetensors" {
        (name, (0..3 << 20 | 5).map(|i: u32| (i % 251) as u8).collect(), lfs) } else { (name, bytes, lfs) }).collect();
    let f = Fixture::with(files, Behavior::default());
    let folder = f.folder();
    let list = f.root.join("models/org/model.sha256");
    // The check of the last download is cut short: the partial file stays,
    // and neither record is written.
    let interruptions = crate::runner::Interruptions::install();
    ARMED.store(true, Ordering::Relaxed);
    assert_eq!(f.pull(REPO, None, false).0.unwrap_err(), INTERRUPTED);
    assert_eq!(fs::read(f.part("model.safetensors")).unwrap(), f.hub.content("model.safetensors"));
    assert!(!folder.join("model.safetensors").exists() && !folder.join(MANIFEST).exists() && !list.exists());
    assert_eq!(f.hub.downloads().len(), 7);
    // The next run uses that file and finishes.
    drop(interruptions);
    let interruptions = crate::runner::Interruptions::install();
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(f.hub.downloads().len(), 7);
    let (manifest, rows) = (fs::read(folder.join(MANIFEST)).unwrap(), fs::read(&list).unwrap());
    // Ctrl-C while pull checks a file already in place: no download, no
    // change to the records, and the model stays complete.
    fs::remove_file(folder.join(STATE).join("checked")).unwrap();
    let before = f.hub.downloads().len();
    drop(interruptions);
    let _interruptions = crate::runner::Interruptions::install();
    ARMED.store(true, Ordering::Relaxed);
    assert_eq!(f.pull(REPO, None, false).0.unwrap_err(), INTERRUPTED);
    assert_eq!(f.hub.downloads().len(), before);
    assert_eq!((fs::read(folder.join(MANIFEST)).unwrap(), fs::read(&list).unwrap()), (manifest, rows));
    assert_eq!(fs::read(folder.join("model.safetensors")).unwrap(), f.hub.content("model.safetensors"));
    assert_eq!(models::tests_list(&f.root.join("models")).0[0].4.route, models::Route::Fp8);
    // Both hashes stopped in the block where Ctrl-C arrived.
    assert!(!ARMED.load(Ordering::Relaxed) && !LATE.load(Ordering::Relaxed));
}

#[test]
fn links_and_fifos_in_the_pull_state_are_refused() {
    let f = Fixture::new(Behavior::default());
    let state = f.folder().join(STATE);
    crate::text::private_dir(&state).unwrap();
    let outside = f.root.join("outside");
    fs::write(&outside, b"short").unwrap();
    // The partial file, and the two API reply files curl writes.
    for name in [f.part("README.md"), state.join("reply.json"), state.join("reply.headers")] {
        std::os::unix::fs::symlink(&outside, &name).unwrap();
        let error = f.pull(REPO, None, false).0.unwrap_err();
        assert!(error.ends_with("in the pull state is not a regular file. Remove it, then run vade pull again."), "{error}");
        assert_eq!(fs::read(&outside).unwrap(), b"short");
        fs::remove_file(&name).unwrap();
        // A FIFO fails at once instead of blocking curl.
        assert!(std::process::Command::new("mkfifo").arg(&name).status().unwrap().success());
        let error = f.pull(REPO, None, false).0.unwrap_err();
        assert!(error.contains("is not a regular file"), "{error}");
        fs::remove_file(&name).unwrap();
    }
    f.pull(REPO, None, false).0.unwrap();
}

#[test]
fn a_reply_longer_than_the_listed_size_is_stopped_kept_and_retried() {
    // A reply that announces too many bytes, and one with no length that never ends.
    let f = Fixture::new(Behavior { extra: [("config.json".into(), 1000)].into(), endless: ["model.safetensors".into()].into(), ..Behavior::default() });
    let error = f.pull(REPO, None, false).0.unwrap_err();
    assert!(error.starts_with("config.json has ") && error.contains("Pull kept the latest bad copy"), "{error}");
    assert!(!f.part("config.json").exists() && !f.folder().join("config.json").exists());
    assert_eq!(f.bad("config.json").len(), 1);
    f.hub.set(|behavior| behavior.extra.clear());
    // The endless reply, also when it continues a partial file.
    for partial in [0, 12_345] {
        if partial > 0 { fs::write(f.part("model.safetensors"), &f.hub.content("model.safetensors")[..partial]).unwrap(); }
        let began = Instant::now();
        let error = f.pull(REPO, None, false).0.unwrap_err();
        assert!(error.starts_with("model.safetensors ") && error.contains("Pull kept the latest bad copy"), "{error}");
        assert!(!f.part("model.safetensors").exists() && !f.folder().join("model.safetensors").exists());
        assert_eq!(f.bad("model.safetensors").len(), 1);
        assert!(began.elapsed() < Duration::from_secs(60));
    }
    f.hub.set(|behavior| behavior.endless.clear());
    f.pull(REPO, None, false).0.unwrap();
}

#[test]
fn a_redirect_longer_than_a_small_file_still_gets_the_file() {
    // The Hub answers resolve/ with a 307 whose body is longer than the small
    // files but under the floor; curl checks its size limit against that body too.
    let body = 1000;
    let f = Fixture::new(Behavior { redirect: body, ..Behavior::default() });
    assert!(f.hub.content("generation_config.json").len() < body && (body as u64) < fetch::REDIRECT_BODY);
    // Behind the redirect, a long reply is stopped and kept.
    f.hub.set(|behavior| behavior.extra = [("config.json".into(), 1000)].into());
    let error = f.pull(REPO, None, false).0.unwrap_err();
    assert!(error.starts_with("config.json has ") && error.contains("Pull kept the latest bad copy"), "{error}");
    assert!(!f.part("config.json").exists() && !f.folder().join("config.json").exists());
    assert_eq!(f.bad("config.json").len(), 1);
    // A reply one byte short is never taken as the whole file.
    f.hub.set(|behavior| { behavior.extra.clear(); behavior.short = ["config.json".into()].into(); });
    let error = f.pull(REPO, None, false).0.unwrap_err();
    assert!(error.contains("config.json") && !error.starts_with("The Hub sent more bytes"), "{error}");
    assert!(!f.folder().join("config.json").exists());
    assert_eq!(f.bad("config.json").len(), 1);
    // The healthy redirect gives every file, and each download went through it.
    f.hub.set(|behavior| behavior.short.clear());
    f.pull(REPO, None, false).0.unwrap();
    for name in ["README.md", "config.json", "generation_config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"] {
        assert_eq!(fs::read(f.folder().join(name)).unwrap(), f.hub.content(name), "{name}");
    }
    let seen = f.hub.seen();
    let (asked, cached) = (seen.iter().filter(|seen| seen.path.contains("/resolve/")).count(), seen.iter().filter(|seen| seen.path.contains("/resolve-cache/")).count());
    assert!(asked > 0 && asked == cached, "{asked} {cached}");
}

#[test]
fn a_failed_update_shows_as_incomplete_until_a_pull_finishes() {
    let f = Fixture::new(Behavior::default());
    f.pull(REPO, None, false).0.unwrap();
    let folder = f.folder();
    // The update replaces config.json first, then a later file fails its hash.
    fs::write(folder.join("config.json"), b"{}").unwrap();
    fs::remove_file(folder.join("model.safetensors")).unwrap();
    f.hub.set(|behavior| { behavior.corrupt.insert("model.safetensors".into()); });
    assert!(f.pull(REPO, None, false).0.unwrap_err().starts_with("model.safetensors has SHA-256 "));
    assert_eq!(fs::read(folder.join("config.json")).unwrap(), f.hub.content("config.json"));
    assert!(folder.join(MANIFEST).exists() && folder.join(STATE).join(UPDATING).exists());
    assert_eq!(models::tests_list(&f.root.join("models")).0[0].4.route, models::Route::Incomplete);
    // A selection cannot finish an unfinished update: its records would keep the old rows.
    let error = f.pull(&format!("{REPO} --include README.md"), None, false).0.unwrap_err();
    assert!(error.starts_with("An earlier pull of this folder stopped while it replaced files."), "{error}");
    assert!(folder.join(STATE).join(UPDATING).exists());
    assert_eq!(models::tests_list(&f.root.join("models")).0[0].4.route, models::Route::Incomplete);
    f.hub.set(|behavior| behavior.corrupt.clear());
    f.pull(REPO, None, false).0.unwrap();
    assert!(!folder.join(STATE).join(UPDATING).exists());
    assert_eq!(models::tests_list(&f.root.join("models")).0[0].4.route, models::Route::Fp8);
}

#[test]
fn a_next_page_on_another_address_fails_the_listing() {
    let f = Fixture::new(Behavior { page: 3, foreign_next: true, ..Behavior::default() });
    let error = f.pull(REPO, None, false).0.unwrap_err();
    assert!(error.starts_with("The Hub file listing has a next page that pull cannot follow"), "{error}");
    assert!(f.hub.downloads().is_empty() && !f.folder().join(MANIFEST).exists());
    // An unquoted rel=next after another link still pages through the whole list.
    drop(f);
    let f = Fixture::new(Behavior { page: 3, bare_rel: true, ..Behavior::default() });
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(f.hub.seen().iter().filter(|seen| seen.path.contains("/tree/")).count(), 4);
    assert_eq!(f.hub.downloads().len(), 7);
}

#[test]
fn the_next_step_names_the_checkpoint_launch_serves() {
    let f = Fixture::new(Behavior::default());
    f.pull(REPO, None, false).0.unwrap();
    let (saved, other) = (fs::canonicalize(f.folder()).unwrap(), f.root.join("models/nvidia/new"));
    fs::create_dir_all(&other).unwrap();
    let root = f.root.join(".vade");
    assert_eq!(crate::launch::saved_model(&root), None);
    crate::text::private_dir(&root).unwrap();
    fs::write(root.join("config"), format!("format=vade-owner-v1\nmodel_path={}\nmodel_bytes=1\ncontext_tokens=32768\noutput_tokens=8192\nport=18080\nstage_dir={}\nstate_dir={}\n",
        saved.display(), root.join("stage").display(), root.join("state").display())).unwrap();
    assert_eq!(crate::launch::saved_model(&root).as_deref(), Some(saved.as_path()));
    let fp4 = models::Support { route: models::Route::Fp4Local, reason: String::new() };
    let home = &f.root;
    assert_eq!(fp4.next(&other, None, home).unwrap(), (format!("VADE_MODELS={} vade up", other.display()), None));
    assert_eq!(fp4.next(&saved, Some(&saved), home).unwrap(), ("vade up".into(), None));
    // A valid saved checkpoint comes before VADE_MODELS, so the step uses a new home.
    // The suggested home names the org and skips a home that is already there.
    fs::create_dir_all(home.join(".vade-nvidia-new")).unwrap();
    let (next, note) = fp4.next(&other, Some(&saved), home).unwrap();
    assert_eq!(next, format!("VADE_HOME={} VADE_MODELS={} vade up", home.join(".vade-nvidia-new-2").display(), other.display()));
    assert_eq!(note.unwrap(), format!("vade up serves its saved checkpoint {} first, so this step uses a new, unused Vade home.", saved.display()));
    // Each command is one valid shell line, also with a path that needs quotes.
    let spaced = f.root.join("models/nvidia/new model (2)");
    fs::create_dir_all(&spaced).unwrap();
    for folder in [&other, &spaced] {
        for saved in [None, Some(saved.as_path())] {
            let (command, _) = fp4.next(folder, saved, home).unwrap();
            assert!(std::process::Command::new("sh").args(["-n", "-c", &command]).status().unwrap().success(), "{command}");
        }
    }
}

#[test]
fn a_partial_pull_keeps_the_records_of_a_complete_folder() {
    let f = Fixture::new(Behavior::default());
    f.pull(REPO, None, false).0.unwrap();
    let folder = fs::canonicalize(f.folder()).unwrap();
    let list = folder.parent().unwrap().join("model.sha256");
    let full = fs::read_to_string(&list).unwrap();
    // The same commit: the files left out keep their rows.
    f.pull(&format!("{REPO} --include *.json"), None, false).0.unwrap();
    assert_eq!(fs::read_to_string(&list).unwrap(), full);
    assert!(crate::launch::supplied_manifest(&folder).unwrap().unwrap().contains("  model.safetensors\n"));
    let manifest = json::parse(&fs::read_to_string(folder.join(MANIFEST)).unwrap()).unwrap();
    assert_eq!(manifest["files"].as_array().unwrap().len(), 7);
    // Another commit: the earlier rows cannot be kept, so the pull stops before any change.
    let text = fs::read_to_string(folder.join(MANIFEST)).unwrap().replace(COMMIT, &"f".repeat(40));
    fs::write(folder.join(MANIFEST), &text).unwrap();
    let before = f.hub.downloads().len();
    let error = f.pull(&format!("{REPO} --include *.json"), None, false).0.unwrap_err();
    assert!(error.starts_with("This folder holds README.md, which this pull would leave out of its records"), "{error}");
    assert_eq!((fs::read_to_string(&list).unwrap(), fs::read_to_string(folder.join(MANIFEST)).unwrap(), f.hub.downloads().len()), (full.clone(), text, before));
    // A pull of every file may replace them.
    f.pull(REPO, None, false).0.unwrap();
    assert_eq!(fs::read_to_string(&list).unwrap(), full);
    // Even a full pull stops when the list names a file the commit no longer has.
    fs::write(folder.join("extra.json"), b"{}").unwrap();
    fs::write(&list, format!("{full}{}  ./extra.json\n", crate::sha256::sha(b"{}"))).unwrap();
    let error = f.pull(REPO, None, false).0.unwrap_err();
    assert!(error.starts_with("The checksum list beside this folder names extra.json, which this pull leaves out"), "{error}");
}

#[test]
fn test_requests_skip_the_proxy_settings() {
    let local = Transport { endpoint: "http://127.0.0.1:1".into(), token: None, scratch: PathBuf::new(), program: "curl".into(), backoff: Duration::ZERO };
    assert!(local.arguments().windows(2).any(|pair| pair == ["--noproxy", "127.0.0.1,localhost,::1"]));
    let hub = Transport { endpoint: hub::DEFAULT_ENDPOINT.into(), ..local };
    assert!(!hub.arguments().iter().any(|argument| argument == "--noproxy"));
    // A full pull with every proxy variable set to a closed port still reaches the fake Hub.
    let _guard = crate::runner::process_test_lock();
    let proxy = "http://127.0.0.1:9";
    child("pull_downloads_checks_and_records_a_servable_model",
        &[("http_proxy", proxy), ("HTTP_PROXY", proxy), ("https_proxy", proxy), ("HTTPS_PROXY", proxy), ("all_proxy", proxy), ("ALL_PROXY", proxy)],
        &["no_proxy", "NO_PROXY"]);
}

#[test]
fn a_selection_never_breaks_a_complete_folder_with_no_records() {
    // A complete checkpoint from another download method: no manifest, no
    // checksum list beside it and no manifest.sha256.
    let f = Fixture::new(Behavior::default());
    let folder = f.folder();
    fs::create_dir_all(&folder).unwrap();
    for (name, bytes, _) in &f.hub.files { if hub::wanted(name) && !hub::hidden(name) { fs::write(folder.join(name), bytes).unwrap(); } }
    assert_eq!(models::support(&folder, None).route, models::Route::Fp8);
    let error = f.pull(&format!("{REPO} --include *.json"), None, false).0.unwrap_err();
    assert!(error.starts_with("This folder holds README.md, which this pull would leave out of its records"), "{error}");
    assert!(f.hub.downloads().is_empty() && !folder.join(MANIFEST).exists() && !f.root.join("models/org/model.sha256").exists());
    // A full pull checks what is there and records every file.
    f.pull(REPO, None, false).0.unwrap();
    assert!(f.hub.downloads().is_empty());
    assert_eq!(fs::read_to_string(f.root.join("models/org/model.sha256")).unwrap().lines().count(), 7);
}

#[test]
fn an_api_reply_over_the_cap_is_stopped_by_pull_itself() {
    // A stand-in for curl that writes 65 MiB without a length and keeps the
    // request open, as curl before 8.4.0 would with --max-filesize.
    let _guard = crate::runner::process_test_lock();
    let root = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-pull-cap-{}", crate::text::random_hex(6).unwrap()));
    crate::text::private_dir(&root).unwrap();
    let program = root.join("fake-curl");
    fs::write(&program, "#!/bin/sh\nwhile [ $# -gt 0 ]; do [ \"$1\" = --output ] && out=$2; shift; done\n\
        dd if=/dev/zero of=\"$out\" bs=1048576 count=65 2>/dev/null\nexec sleep 30\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&program, fs::Permissions::from_mode(0o700)).unwrap();
    let transport = Transport { endpoint: "http://127.0.0.1:1".into(), token: None, scratch: root.clone(), program: program.to_string_lossy().into_owned(), backoff: Duration::ZERO };
    let began = Instant::now();
    let result = transport.get("http://127.0.0.1:1/api/models/org/model/revision/main", &root.join("reply.json"), &root.join("reply.headers"));
    let elapsed = began.elapsed();
    let _ = fs::remove_dir_all(&root);
    assert_eq!(result, Err(Failure::TooLarge));
    assert!(elapsed < Duration::from_secs(20), "{elapsed:?}");
}

#[test]
fn a_child_with_a_copy_of_the_lock_file_does_not_keep_the_pull_lock() {
    let _guard = crate::runner::process_test_lock();
    let scratch = fs::canonicalize(std::env::temp_dir()).unwrap()
        .join(format!("vade-pull-lock-{}", crate::text::random_hex(12).unwrap()));
    crate::text::private_dir(&scratch).unwrap();
    let held = lock(&scratch).unwrap();
    assert!(lock(&scratch).is_err());
    let mut child = crate::unix_file::ChildCopy::of(&held);
    drop(held);
    let again = lock(&scratch);
    let running = child.running();
    drop(child);
    fs::remove_dir_all(&scratch).unwrap();
    assert!(running, "the child must still hold its copy");
    assert!(again.is_ok(), "the copy in a child kept the lock");
}
