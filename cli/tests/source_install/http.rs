//! The bootstrap uses real curl with a local file server. Host probes are stubs.
use super::*;
use std::{io::{BufRead, BufReader, Write}, net::{Ipv4Addr, TcpListener}, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}, thread, time::Duration};

#[derive(Clone, Copy, Debug)]
enum Reply { Full, CutOnce, CutThenUnavailable, IgnoreRange, RangeUnavailable }

struct Server {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    task: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn new(f: &Fixture, archive: Vec<u8>, reply: Reply) -> Self {
        let latest = fs::read(f.root.join("dl/main/latest.json")).unwrap();
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (log, done) = (requests.clone(), stop.clone());
        let task = thread::spawn(move || {
            let mut archive_calls = 0;
            while !done.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5)); continue;
                    }
                    Err(error) => panic!("accept: {error}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                stream.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 { break; }
                    request.push_str(&line);
                    if line == "\r\n" { break; }
                    assert!(request.len() < 16384);
                }
                log.lock().unwrap().push(request.clone());
                let path = request.lines().next().unwrap().split_whitespace().nth(1).unwrap();
                let range = request.lines().find_map(|line| line.to_ascii_lowercase().strip_prefix("range: bytes=")
                    .and_then(|range| range.strip_suffix('-')).and_then(|offset| offset.parse::<usize>().ok()));
                if path == "/main/latest.json" {
                    write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", latest.len()).unwrap();
                    stream.write_all(&latest).unwrap();
                    continue;
                }
                assert_eq!(path, format!("/main/{}.tar.gz", stem()));
                archive_calls += 1;
                if archive_calls == 1 && !matches!(reply, Reply::Full) {
                    assert_eq!(range, None);
                    write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", archive.len()).unwrap();
                    stream.write_all(&archive[..archive.len() / 2]).unwrap();
                    continue;
                }
                if archive_calls <= 3 && matches!(reply, Reply::CutThenUnavailable) {
                    write!(stream, "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                    continue;
                }
                if range.is_some() && matches!(reply, Reply::RangeUnavailable) {
                    write!(stream, "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", archive.len()).unwrap();
                    continue;
                }
                let start = if matches!(reply, Reply::IgnoreRange) { 0 } else { range.unwrap_or(0) };
                assert!(start < archive.len());
                if range.is_some() && !matches!(reply, Reply::IgnoreRange) {
                    write!(stream, "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{}\r\n", archive.len() - 1, archive.len()).unwrap();
                } else {
                    write!(stream, "HTTP/1.1 200 OK\r\n").unwrap();
                }
                write!(stream, "Content-Length: {}\r\nConnection: close\r\n\r\n", archive.len() - start).unwrap();
                // curl can close an ignored Range reply before it reads the body.
                let _ = stream.write_all(&archive[start..]);
            }
        });
        Self { url, requests, stop, task: Some(task) }
    }
    fn archive_requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().iter().filter(|request| request.lines().next().unwrap().contains(".tar.gz ")).cloned().collect()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(task) = self.task.take() {
            let result = task.join();
            if !thread::panicking() { result.unwrap(); }
        }
    }
}

fn served(reply: Reply, bad: Bad) -> (Fixture, Vec<(&'static str, String)>, Server, Vec<u8>) {
    let f = Fixture::new();
    let archive = bundle(Bad::None);
    let mut env = f.prebuilt(&archive, bad);
    let server = Server::new(&f, archive.clone(), reply);
    // The production bootstrap accepts TLS or a file URL. This test adapter
    // changes only the protocol and URL, so std can serve HTTP without TLS.
    // Every request still uses the system curl and the bootstrap's arguments.
    fs::remove_file(f.root.join("bin/curl")).unwrap();
    let curl = system_tool("curl");
    f.tool("curl", &format!(r#"
count=$#
while [ "$count" -gt 0 ]; do
    value=$1
    shift
    count=$((count - 1))
    case $value in
        =file) value='=http' ;;
        file://*/main/*) value="$INSTALL_HTTP/main/${{value##*/}}" ;;
        https://*) exit 92 ;;
    esac
    set -- "$@" "$value"
done
exec '{}' "$@"
"#, curl.display()));
    // Save the checksum that the installer reads from the downloaded bytes.
    // Manifest checks still call the real tool without this capture.
    fs::remove_file(f.root.join("bin/sha256sum")).unwrap();
    let sum = system_tool("sha256sum");
    f.tool("sha256sum", &format!(r#"
case "${{1-}}" in
    */bundle.part)
        digest=$('{}' "$@")
        printf '%s\n' "${{digest%% *}}" > "$INSTALL_ROOT/download.sha256"
        printf '%s\n' "$digest"
        ;;
    *) exec '{}' "$@" ;;
esac
"#, sum.display(), sum.display()));
    env.push(("INSTALL_HTTP", server.url.clone()));
    (f, env, server, archive)
}

fn assert_installed(f: &Fixture, output: &Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(fs::read(f.bundles().join(stem()).join("vade")).unwrap(), BUNDLE_VADE.as_bytes());
    assert_eq!(fs::read(f.bundles().join(stem()).join("flash-serve")).unwrap(), b"fixture engine\n");
    assert_eq!(f.calls(), "bundle-vade install\n");
    assert!(f.left_downloads().is_empty());
    assert_eq!(fs::read_dir(f.bundles()).unwrap().count(), 1);
    f.assert_private("HTTP bundle");
}

#[test]
fn good_http_bundle_installs_without_changing_payload_bytes() {
    let (f, env, server, _) = served(Reply::Full, Bad::None);
    assert_installed(&f, &f.run_prebuilt(&env, &[]));
    let requests = server.archive_requests();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].to_ascii_lowercase().contains("range:"));
}

#[test]
fn bad_http_bundle_hash_stops_with_one_line_and_no_install_files() {
    let (f, env, server, _) = served(Reply::Full, Bad::Sha256);
    let output = f.run_prebuilt(&env, &[]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(String::from_utf8(output.stderr).unwrap(),
        format!("install: {}.tar.gz does not match the sha256 in latest.json; nothing installed.\n", stem()));
    assert_eq!(f.calls(), "");
    assert!(!f.bundles().join(stem()).exists());
    assert!(!f.root.join("home/.local/bin/vade").exists());
    assert_eq!(fs::read_dir(f.bundles()).unwrap().count(), 0);
    assert_eq!(server.archive_requests().len(), 1);
}

#[test]
fn cut_http_bundle_resumes_with_range_and_matches_the_published_hash() {
    let (f, env, server, archive) = served(Reply::CutOnce, Bad::None);
    assert_installed(&f, &f.run_prebuilt(&env, &[]));
    let requests = server.archive_requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].to_ascii_lowercase().contains(&format!("range: bytes={}-\r\n", archive.len() / 2)));
    let downloaded = fs::read_to_string(f.root.join("download.sha256")).unwrap();
    assert_eq!(downloaded.trim(), sha256(&archive));
}

#[test]
fn servers_without_range_restart_the_bundle_without_appending_a_full_reply() {
    for reply in [Reply::IgnoreRange, Reply::RangeUnavailable] {
        let (f, env, server, archive) = served(reply, Bad::None);
        let output = f.run_prebuilt(&env, &[]);
        assert!(output.status.success(), "{reply:?}: {}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert_installed(&f, &output);
        let requests = server.archive_requests();
        assert_eq!(requests.len(), 3);
        assert!(requests[1].to_ascii_lowercase().contains(&format!("range: bytes={}-\r\n", archive.len() / 2)));
        assert!(!requests[2].to_ascii_lowercase().contains("range:"));
    }
}

#[test]
fn a_later_install_resumes_the_retained_partial_bundle() {
    let (f, env, server, archive) = served(Reply::CutThenUnavailable, Bad::None);
    let first = f.run_prebuilt(&env, &[]);
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    assert!(String::from_utf8_lossy(&first.stdout).contains("partial bytes were kept for retry"));
    assert!(!f.bundles().join(stem()).exists());
    let cache = f.bundles().join(format!(".partial-{}", sha256(&archive)));
    assert_eq!(fs::read(cache.join("bundle.part")).unwrap(), archive[..archive.len() / 2]);
    assert!(!cache.join("lock").exists());
    assert!(f.left_downloads().is_empty());
    assert_eq!(server.archive_requests().len(), 3);
    fs::write(f.root.join("calls"), "").unwrap();
    assert_installed(&f, &f.run_prebuilt(&env, &[]));
    let requests = server.archive_requests();
    assert_eq!(requests.len(), 4);
    assert!(requests[3].to_ascii_lowercase().contains(&format!("range: bytes={}-\r\n", archive.len() / 2)));
    assert_eq!(fs::read_to_string(f.root.join("download.sha256")).unwrap().trim(), sha256(&archive));
    assert!(!cache.exists());
}
