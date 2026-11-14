//! The transport for `vade pull`: the system curl from PATH, started with one
//! explicit argument list per request and no shell. The token goes to curl on
//! standard input as a config line, so it never shows in a process list.
use std::{fs, io::{Read, Write}, path::{Path, PathBuf}, process::Stdio, time::{Duration, Instant}};

/// Requests a download may make before it gives up.
const ATTEMPTS: u32 = 6;
/// The largest API reply body and header block a pull reads.
pub(super) const API_BODY: u64 = 64 << 20;
pub(super) const API_HEADERS: u64 = 1 << 20;
/// The smallest size limit a download gives curl. curl checks the limit against
/// the length of every reply it reads, a redirect too, and the Hub answers a
/// file request with a redirect whose body can be longer than a small file.
pub(super) const REDIRECT_BODY: u64 = 64 << 10;

/// Why one curl request did not finish.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Failure {
    /// The server answered with this HTTP status.
    Http(u16),
    /// curl stopped with this exit code and its last error line.
    Curl(i32, String),
    /// curl is not on PATH.
    Missing,
    /// Ctrl-C.
    Interrupted,
    /// The response grew past the size the Hub lists.
    TooLarge,
    /// An output path in the pull state is a link, FIFO or other non-regular file.
    NotRegular(PathBuf),
}

impl Failure {
    /// Network trouble worth another attempt; refusals are final.
    fn transient(&self) -> bool {
        match self {
            Self::Http(code) => matches!(code, 408 | 429 | 500..=599),
            Self::Curl(code, _) => matches!(code, 5 | 6 | 7 | 16 | 18 | 28 | 35 | 52 | 55 | 56 | 92),
            Self::Missing | Self::Interrupted | Self::TooLarge | Self::NotRegular(_) => false,
        }
    }
}

/// How to reach the Hub.
pub(super) struct Transport {
    pub endpoint: String,
    pub token: Option<String>,
    /// Private folder for curl's status and error output.
    pub scratch: PathBuf,
    pub program: String,
    /// The first wait between attempts; it doubles each time.
    pub backoff: Duration,
}

pub(super) enum Event {
    /// Current file size and bytes received in this call.
    Progress { have: u64, received: u64 },
    /// The server refused a byte range. The next request starts at zero.
    Restart,
}

impl Transport {
    pub(super) fn arguments(&self) -> Vec<String> {
        // Plain HTTP is accepted only for a loopback HF_ENDPOINT (see hub::endpoint).
        let local = !self.endpoint.starts_with("https://");
        let proto = if local { "=http,https" } else { "=https" };
        // --disable must come first: it keeps a user .curlrc out of the request.
        let mut arguments = ["--disable", "--silent", "--show-error", "--fail", "--location", "--max-redirs", "10", "--proto", proto,
            "--proto-redir", proto, "--connect-timeout", "30", "--user-agent", concat!("vade/", env!("CARGO_PKG_VERSION")),
            "--config", "-", "--write-out", "%{http_code}"].map(String::from).to_vec();
        // A proxy cannot reach this machine's loopback address. A Hub address
        // keeps the proxy settings.
        if local { arguments.extend(["--noproxy".into(), "127.0.0.1,localhost,::1".into()]); }
        arguments
    }

    /// The config text curl reads on standard input. curl sends a custom
    /// Authorization header to the first host only, never across a redirect to
    /// another host.
    fn config(&self) -> String {
        self.token.as_ref().map(|token| format!("header = \"Authorization: Bearer {token}\"\n")).unwrap_or_default()
    }

    /// Run curl once with `extra` arguments. `watch` runs while it works; when
    /// it returns true, the transfer stops as [`Failure::TooLarge`].
    fn once(&self, extra: &[String], limit: Option<Duration>, watch: &mut dyn FnMut() -> bool) -> Result<(), Failure> {
        let id = crate::text::random_hex(6).map_err(|error| Failure::Curl(-1, error))?;
        let (out, err) = (self.scratch.join(format!("curl-{id}.code")), self.scratch.join(format!("curl-{id}.error")));
        // New files only: a link or FIFO already at the name is refused.
        let create = |path: &Path| fs::File::options().write(true).create_new(true).open(path)
            .map_err(|_| Failure::Curl(-1, "Cannot write the private pull state.".into()));
        let mut arguments = self.arguments();
        arguments.extend_from_slice(extra);
        let exec = crate::runner::Exec::new("curl", &self.program, arguments);
        let mut command = crate::runner::owned_command(&exec);
        command.stdin(Stdio::piped()).stdout(create(&out)?).stderr(create(&err)?);
        let result = (|| {
            let mut child = command.spawn().map_err(|error| if error.kind() == std::io::ErrorKind::NotFound { Failure::Missing }
                else { Failure::Curl(-1, "Cannot start curl.".into()) })?;
            let sent = child.stdin.take().map(|mut input| input.write_all(self.config().as_bytes()));
            let began = Instant::now();
            let mut large = false;
            let status = loop {
                let stop = large || crate::runner::interrupted() || sent.as_ref().is_some_and(Result::is_err)
                    || limit.is_some_and(|limit| began.elapsed() >= limit);
                match child.try_wait() {
                    Ok(Some(status)) => break status,
                    Ok(None) if !stop => { large = watch(); if !large { std::thread::sleep(Duration::from_millis(100)); } }
                    _ => {
                        crate::runner::stop_job(&mut child);
                        let _ = child.wait();
                        return Err(if crate::runner::interrupted() { Failure::Interrupted } else if large { Failure::TooLarge }
                            else { Failure::Curl(28, "The request took too long.".into()) });
                    }
                }
            };
            let code = fs::read_to_string(&out).ok().and_then(|code| code.trim().parse::<u16>().ok()).unwrap_or(0);
            if status.success() && (200..300).contains(&code) { return Ok(()); }
            if code >= 400 { return Err(Failure::Http(code)); }
            let mut message = String::new();
            let _ = fs::File::open(&err).map(|file| file.take(4096).read_to_string(&mut message));
            let message = message.lines().rev().find(|line| !line.trim().is_empty()).unwrap_or("").chars().take(300).collect::<String>();
            Err(Failure::Curl(status.code().unwrap_or(-1), crate::text::terminal_text(&message).into_owned()))
        })();
        let _ = (fs::remove_file(&out), fs::remove_file(&err));
        result
    }

    /// Wait before the next attempt; Ctrl-C ends the wait.
    fn pause(&self, attempt: u32) -> Result<(), Failure> {
        let until = Instant::now() + self.backoff * 2u32.pow(attempt.min(5));
        while Instant::now() < until {
            if crate::runner::interrupted() { return Err(Failure::Interrupted); }
            std::thread::sleep(Duration::from_millis(50));
        }
        Ok(())
    }

    /// Fetch a small API reply into `body`, with the headers in `headers`.
    /// A body over [`API_BODY`] bytes or headers over [`API_HEADERS`] stop it
    /// with [`Failure::TooLarge`]: curl before 8.4.0 does not apply
    /// --max-filesize to a reply of unknown length, so the watch checks too.
    pub(super) fn get(&self, url: &str, body: &Path, headers: &Path) -> Result<(), Failure> {
        let extra = ["--max-time".into(), "120".into(), "--max-filesize".into(), API_BODY.to_string(), "--output".into(), path(body), "--dump-header".into(), path(headers), url.into()];
        let length = |path: &Path| fs::symlink_metadata(path).map_or(0, |meta| meta.len());
        let mut attempt = 0;
        loop {
            regular(body)?;
            regular(headers)?;
            let result = self.once(&extra, Some(Duration::from_secs(150)), &mut || length(body) > API_BODY || length(headers) > API_HEADERS);
            if length(body) > API_BODY || length(headers) > API_HEADERS || matches!(result, Err(Failure::Curl(63, _))) { return Err(Failure::TooLarge); }
            match result {
                Err(failure) if failure.transient() && attempt + 1 < ATTEMPTS => { self.pause(attempt)?; attempt += 1; }
                result => return result,
            }
        }
    }

    /// Download `url` into `part`, continuing from the bytes already there,
    /// until it holds `size` bytes. Return whether the last request used saved
    /// bytes. The caller checks the hash. Keep failed copies for it.
    pub(super) fn download(&self, url: &str, part: &Path, size: u64, watch: &mut dyn FnMut(Event)) -> Result<bool, Failure> {
        let length = || fs::symlink_metadata(part).map(|meta| meta.len()).unwrap_or(0);
        let mut attempt = 0;
        let mut received = 0;
        loop {
            regular(part)?;
            let have = length();
            if have == size && (size > 0 || part.exists()) { return Ok(false); }
            if have > size { return Err(Failure::TooLarge); }
            // A transfer fails after 120 s below 1 byte per second; a healthy
            // one may run for hours. curl refuses a reply that announces more
            // than `size` bytes, or [`REDIRECT_BODY`] for a smaller file; the
            // watch and the length check after curl stop any reply that writes more.
            let extra = ["--speed-limit".into(), "1".into(), "--speed-time".into(), "120".into(), "--max-filesize".into(), size.max(REDIRECT_BODY).to_string(),
                "--continue-at".into(), have.to_string(), "--output".into(), path(part), url.into()];
            let result = self.once(&extra, None, &mut || {
                let now = length();
                watch(Event::Progress { have: now.min(size), received: received + now.saturating_sub(have) });
                now > size
            });
            received += length().saturating_sub(have);
            watch(Event::Progress { have: length().min(size), received });
            if length() > size || matches!(result, Err(Failure::TooLarge | Failure::Curl(63, _))) {
                return Err(Failure::TooLarge);
            }
            let failure = match result {
                Ok(()) if length() == size => {
                    // curl writes no file for an empty body; never cut one it wrote.
                    if size == 0 { fs::File::options().write(true).create(true).truncate(false).open(part).map_err(|_| Failure::Curl(-1, "Cannot write an empty file.".into()))?; }
                    return Ok(have > 0);
                }
                Ok(()) => Failure::Curl(18, "The download ended early.".into()),
                // The server cannot continue this file: start it again.
                Err(Failure::Curl(33, _)) | Err(Failure::Http(416)) if have > 0 => {
                    regular(part)?;
                    fs::File::options().write(true).truncate(true).open(part)
                        .map_err(|_| Failure::Curl(-1, "Cannot start the partial file again.".into()))?;
                    watch(Event::Restart);
                    if attempt + 1 >= ATTEMPTS { return Err(Failure::Curl(18, "The server could not continue the file.".into())); }
                    attempt += 1;
                    continue;
                }
                Err(failure) => failure,
            };
            if !failure.transient() || attempt + 1 >= ATTEMPTS { return Err(failure); }
            self.pause(attempt)?;
            attempt += 1;
        }
    }
}

fn path(path: &Path) -> String { path.to_string_lossy().into_owned() }

/// Allow curl to write `path` only when nothing is there yet or it is a
/// regular file: never through a link, and never into a FIFO that would block it.
fn regular(path: &Path) -> Result<(), Failure> {
    match fs::symlink_metadata(path) {
        Ok(meta) if !meta.is_file() => Err(Failure::NotRegular(path.to_path_buf())),
        _ => Ok(()),
    }
}

/// Run by [`digests`] after each block with the bytes read so far, in the
/// cancellation test only.
#[cfg(test)]
pub(super) static HASH_HOOK: std::sync::OnceLock<fn(u64)> = std::sync::OnceLock::new();

/// The SHA-256 of the regular file at `path`, and its Git blob ID when `git` is
/// set, in one read. The file must hold `size` bytes; the open neither follows
/// a symlink nor blocks on a FIFO, and the read stops at `size` bytes. Ctrl-C
/// stops the read with [`std::io::ErrorKind::TimedOut`]; the caller checks
/// [`crate::runner::interrupted`].
pub(super) fn digests(path: &Path, size: u64, git: bool) -> std::io::Result<(String, Option<String>)> {
    use std::os::unix::fs::OpenOptionsExt;
    use crate::unix_flags::{O_CLOEXEC, O_NOFOLLOW, O_NONBLOCK};
    let file = fs::OpenOptions::new().read(true).custom_flags(O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW).open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.len() != size { return Err(std::io::ErrorKind::InvalidData.into()); }
    let mut file = file.take(size);
    let mut sha256 = crate::sha256::Sha256::new();
    let mut sha1 = git.then(|| { let mut hash = Sha1::new(); hash.update(format!("blob {size}\0").as_bytes()); hash });
    let mut buffer = vec![0u8; 1 << 20];
    #[cfg(test)]
    let mut read = 0u64;
    loop {
        let n = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        sha256.update(&buffer[..n]);
        if let Some(hash) = sha1.as_mut() { hash.update(&buffer[..n]); }
        #[cfg(test)]
        if let Some(hook) = HASH_HOOK.get() { read += n as u64; hook(read); }
        if crate::runner::interrupted() { return Err(std::io::ErrorKind::TimedOut.into()); }
    }
    Ok((sha256.finalize(), sha1.map(Sha1::finalize)))
}

/// SHA-1, only to check the Git blob IDs the Hub lists for small files.
struct Sha1 { state: [u32; 5], pending: Vec<u8>, total: u64 }
impl Sha1 {
    fn new() -> Self { Self { state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0], pending: Vec::with_capacity(128), total: 0 } }
    fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if !self.pending.is_empty() {
            let take = (64 - self.pending.len()).min(data.len());
            self.pending.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.pending.len() == 64 { let block = std::mem::take(&mut self.pending); self.block(&block); }
        }
        #[allow(clippy::chunks_exact_to_as_chunks)] // remainder() is used on the iterator below.
        let mut blocks = data.chunks_exact(64);
        for block in &mut blocks { self.block(block); }
        self.pending.extend_from_slice(blocks.remainder());
    }
    fn block(&mut self, block: &[u8]) {
        let mut w = [0u32; 80];
        for (word, &bytes) in w.iter_mut().zip(block.as_chunks::<4>().0) { *word = u32::from_be_bytes(bytes); }
        for i in 16..80 { w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1); }
        let [mut a, mut b, mut c, mut d, mut e] = self.state;
        for (i, &word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5a827999u32),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
                _ => (b ^ c ^ d, 0xca62c1d6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(word);
            [e, d, c, b, a] = [d, c, b.rotate_left(30), a, t];
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e]) { *s = s.wrapping_add(v); }
    }
    fn finalize(mut self) -> String {
        let bits = self.total.wrapping_mul(8);
        let mut tail = std::mem::take(&mut self.pending);
        tail.push(0x80);
        while tail.len() % 64 != 56 { tail.push(0); }
        tail.extend_from_slice(&bits.to_be_bytes());
        for block in tail.as_chunks::<64>().0 { self.block(block); }
        self.state.iter().map(|n| format!("{n:08x}")).collect()
    }
}

#[cfg(test)]
pub(super) fn sha1_hex(data: &[u8]) -> String { let mut hash = Sha1::new(); hash.update(data); hash.finalize() }
