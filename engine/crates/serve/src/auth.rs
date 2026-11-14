//! Startup credentials and fixed-work request authentication, using std only.
use std::{fmt, path::Path, sync::Arc};

// Owner policy: set true to require a key on loopback too.
const REQUIRE_LOOPBACK_KEY: bool = false;
const MAX_KEY_FILE_BYTES: u64 = 4096;
// Audited O_NONBLOCK values (as in cli/src/unix_flags.rs): opening a pipe must
// not wait for a writer. Regular files are unaffected.
#[cfg(target_os = "macos")]
const NONBLOCK: i32 = 0x4;
#[cfg(target_os = "linux")]
const NONBLOCK: i32 = 0x800;
pub const PERMISSION_REFUSAL: &str = "--api-key-file must not be readable or writable by group or others; run chmod 600 on it";
pub const REMOTE_REFUSAL: &str = "--allow-remote requires --api-key-file PATH";
pub(crate) const UNAUTHORIZED: &str = "Unauthorized";
pub const UNAUTHORIZED_BODY: &str = "{\"error\":{\"message\":\"Unauthorized\",\"type\":\"authentication_error\",\"param\":null,\"code\":\"invalid_api_key\"}}";

#[derive(Clone)]
pub struct ApiKey(Arc<[u8]>, Option<Arc<[u8]>>);
impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("ApiKey([redacted])") }
}
impl ApiKey {
    /// Read once before binding or loading weights; never include key/path in errors.
    pub fn load(path: Option<&Path>, allow_remote: bool) -> Result<Option<Self>, String> {
        if path.is_none() && (allow_remote || REQUIRE_LOOPBACK_KEY) {
            return Err(if allow_remote { REMOTE_REFUSAL } else { "loopback serving requires --api-key-file PATH" }.into());
        }
        let Some(path) = path else { return Ok(None); };
        // Prove the path is a regular file before the open: a pipe can block the open or feed a key.
        // A link is followed, as before this check; its target must be a regular file.
        let named = std::fs::symlink_metadata(path).map_err(|_| "cannot open --api-key-file")?;
        let named = if named.file_type().is_symlink() { std::fs::metadata(path).map_err(|_| "cannot open --api-key-file")? } else { named };
        if !named.is_file() {
            return Err("--api-key-file must be a regular file".into());
        }
        let mut open = std::fs::OpenOptions::new();
        open.read(true);
        // A path swapped to a pipe after the check must not wait for a writer.
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        std::os::unix::fs::OpenOptionsExt::custom_flags(&mut open, NONBLOCK);
        let mut file = open.open(path).map_err(|_| "cannot open --api-key-file")?;
        let metadata = file.metadata().map_err(|_| "cannot inspect --api-key-file permissions")?;
        if !metadata.is_file() { return Err("--api-key-file must be a regular file".into()); }
        // Another account must not read the key, or put its own key in the file.
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            if open_to_others(metadata.permissions().mode()) { return Err(PERMISSION_REFUSAL.into()); }
        }
        let mut text = String::new();
        use std::io::Read;
        file.by_ref().take(MAX_KEY_FILE_BYTES + 1).read_to_string(&mut text)
            .map_err(|_| "cannot read --api-key-file as UTF-8")?;
        if text.len() as u64 > MAX_KEY_FILE_BYTES {
            return Err("--api-key-file must be at most 4096 bytes".into());
        }
        let key = validate_key(&text)?;
        Ok(Some(Self(Arc::from(key.as_bytes()), None)))
    }
    /// Keep the owner's existing Bearer credential available in every server state.
    pub fn with_owner(mut self, owner: Option<crate::cache::SessionKey>) -> Self {
        self.1 = owner.map(|key| Arc::from(key.expose().as_bytes()));
        self
    }
    pub(crate) fn authorize(&self, authorization: Option<&str>, x_api_key: Option<&str>) -> Option<bool> {
        let api_ok = self.accepts(authorization, x_api_key);
        // Check the owner separately after API auth, with fixed work over its full key.
        let owner_ok = self.1.as_ref().map(|owner| constant_time_eq(owner,
            authorization.and_then(bearer_key).unwrap_or("").as_bytes())).unwrap_or(false)
            & (x_api_key.is_none() | constant_time_eq(&self.0, x_api_key.unwrap_or("").as_bytes()));
        (api_ok | owner_ok).then_some(owner_ok)
    }
    pub(crate) fn accepts(&self, authorization: Option<&str>, x_api_key: Option<&str>) -> bool {
        let bearer = authorization.and_then(bearer_key).unwrap_or("");
        // Always compare both slots. If both headers are supplied, both must match.
        let bearer_ok = constant_time_eq(&self.0, bearer.as_bytes());
        let x_ok = constant_time_eq(&self.0, x_api_key.unwrap_or("").as_bytes());
        (authorization.is_some() | x_api_key.is_some())
            & (authorization.is_none() | bearer_ok) & (x_api_key.is_none() | x_ok)
    }
}

#[cfg(unix)]
fn open_to_others(mode: u32) -> bool { mode & 0o066 != 0 }

fn validate_key(text: &str) -> Result<&str, &'static str> {
    let key = text.trim();
    if key.len() < 16 { return Err("--api-key-file must contain one non-empty key of at least 16 bytes"); }
    if !key.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err("--api-key-file must contain one key using only visible ASCII (0x21-0x7E)");
    }
    Ok(key)
}

fn bearer_key(value: &str) -> Option<&str> {
    let (scheme, key) = value.split_once(' ')?;
    (scheme.eq_ignore_ascii_case("Bearer") && !key.is_empty()
        && key.bytes().all(|b| (0x21..=0x7e).contains(&b))).then_some(key)
}

/// Work is fixed by the configured key length, including on length mismatch.
/// black_box preserves the XOR fold against replacement by an early-exit compare.
fn constant_time_eq(expected: &[u8], supplied: &[u8]) -> bool {
    let mut difference = expected.len() ^ supplied.len();
    for (index, byte) in expected.iter().enumerate() {
        difference |= std::hint::black_box(usize::from(byte ^ supplied.get(index).copied().unwrap_or(0)));
    }
    std::hint::black_box(difference) == 0
}

pub(crate) fn request_error(error: &str) -> (u16, String) {
    if error == UNAUTHORIZED { (401, UNAUTHORIZED_BODY.into()) }
    else { (400, crate::protocol::error_json("invalid_request_error", error)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn key_file_validation() {
        let directory = std::env::temp_dir().join(format!("vade-auth-{}-{}", std::process::id(), crate::cache::SessionKey::generate().unwrap().expose())); std::fs::create_dir(&directory).unwrap(); let path = directory.join("key");
        #[cfg(unix)] let chmod = |mode| { use std::os::unix::fs::PermissionsExt; std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap() };
        std::fs::write(&path, b"0123456789abcdef\n").unwrap();
        // A key that group or others can read or write is refused before it is read; the owner's own bits do not matter.
        #[cfg(unix)] { for mode in [0o640, 0o604, 0o644, 0o620, 0o602, 0o666] { chmod(mode); assert_eq!(ApiKey::load(Some(&path), false).unwrap_err(), PERMISSION_REFUSAL, "{mode:o}"); }
            for mode in [0o400, 0o700, 0o600] { chmod(mode); assert!(ApiKey::load(Some(&path), false).unwrap().is_some(), "{mode:o}"); } }
        for bytes in [b"".as_slice(), b"short", b"01234567\n89abcdef", b"01234567\x0189abcdef", b"0123456789abcdef\xff"] { std::fs::write(&path, bytes).unwrap(); assert!(ApiKey::load(Some(&path), false).is_err()); }
        std::fs::write(&path, "a".repeat(4097)).unwrap(); assert_eq!(ApiKey::load(Some(&path), false).unwrap_err(), "--api-key-file must be at most 4096 bytes");
        std::fs::write(&path, b"0123456789abcdef\n").unwrap(); let key = ApiKey::load(Some(&path), false).unwrap().unwrap(); assert!(key.accepts(None, Some("0123456789abcdef"))); assert_eq!(format!("{key:?}"), "ApiKey([redacted])");
        assert_eq!(ApiKey::load(Some(&directory), false).unwrap_err(), "--api-key-file must be a regular file"); std::fs::remove_dir_all(directory).unwrap();
    }
    #[test] fn visible_ascii() {
        for key in ["0123456789abcdefé", "0123456789abcdef\u{7f}", "01234567 9abcdef", "01234567\t9abcdef", "01234567\n9abcdef"] { assert!(validate_key(key).is_err()); }
        assert_eq!(validate_key("  !0123456789abcde~\n").unwrap(), "!0123456789abcde~");
        let access = crate::http::AccessPolicy::new(std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 8080)), &crate::Limits::default()).unwrap();
        for head in ["GET /héalth HTTP/1.1\r\nHost: localhost:8080", "GET /health HTTP/1.1\r\nHost: localhost:8080\r\nX-Name: é"] { assert_eq!(crate::http::parse_head(head.as_bytes(), &access, false).err().unwrap(), "invalid HTTP header bytes"); }
    }
    #[test] fn compare() {
        assert!(constant_time_eq(b"abcd", b"abcd")); assert!(!constant_time_eq(b"abcd", b"abce"));
        assert!(!constant_time_eq(b"abcd", b"abc")); assert!(!constant_time_eq(b"abcd", b"abcde"));
        assert!(constant_time_eq(b"", b"")); assert!(!constant_time_eq(b"abcd", b"")); assert!(!constant_time_eq(b"", b"a"));
    }
    #[test] fn headers() {
        let key = ApiKey(Arc::from(&b"0123456789abcdef"[..]), None);
        for (a, x, ok) in [(Some("Bearer 0123456789abcdef"), None, true), (Some("bearer 0123456789abcdef"), Some("0123456789abcdef"), true), (None, Some("0123456789abcdef"), true), (None, None, false), (Some("Basic 0123456789abcdef"), None, false), (Some("Bearer 0123456789abcdef"), Some("wrong"), false), (Some("Bearer wrong"), Some("0123456789abcdef"), false), (Some("Bearer  0123456789abcdef"), None, false), (Some("Bearer "), Some("0123456789abcdef"), false)] { assert_eq!(key.accepts(a, x), ok); }
    }
    #[test] fn http_headers() {
        let limits = crate::Limits { api_key: Some(ApiKey(Arc::from(&b"0123456789abcdef"[..]), None)), ..Default::default() };
        let access = crate::http::AccessPolicy::new(std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 8080)), &limits).unwrap();
        for (host, ok) in [("localhost:18085", true), ("127.0.0.1:18085", true), ("[::1]:18085", true), ("LOCALHOST:65535", true), ("localhost:0", true), ("localhost:65536", false), ("localhost:", false), ("localhost:bad", false), ("localhost.example:18085", false), ("127.0.0.1.example:18085", false), ("[::ffff:127.0.0.1]:18085", false)] { assert_eq!(crate::http::parse_head(format!("GET /health HTTP/1.1\r\nHost: {host}").as_bytes(), &access, false).is_ok(), ok, "{host}"); }
        for (path, header, ok) in [("/v1/models", "aUtHoRiZaTiOn: Bearer 0123456789abcdef", true), ("/v1/models", "X-Api-Key: 0123456789abcdef", true), ("/v1/models", "X-Api-Key: wrong", false), ("/v1/models", "Accept: application/json", false), ("/health", "Accept: application/json", true)] {
            let head = format!("GET {path} HTTP/1.1\r\nHost: localhost:8080\r\n{header}"); let result = crate::http::parse_head(head.as_bytes(), &access, false);
            assert_eq!(result.is_ok(), ok); if !ok { assert_eq!(result.err().unwrap(), UNAUTHORIZED); }
        }
    }
    #[test] fn health_shows_its_facts_only_to_a_request_with_the_key() {
        let owner = crate::cache::SessionKey::generate().unwrap();
        let limits = crate::Limits { api_key: Some(ApiKey(Arc::from(&b"0123456789abcdef"[..]), None).with_owner(Some(owner))), ..Default::default() };
        let address = std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 8080));
        let route = |limits: &crate::Limits, target: &str, headers: &str| {
            let access = crate::http::AccessPolicy::new(address, limits).unwrap();
            crate::http::parse_head(format!("GET {target} HTTP/1.1\r\nHost: localhost:8080{headers}").as_bytes(), &access, false).map(|head| (head.path, head.control_target))
        };
        // With a key configured: no key or a wrong key gets the liveness route. It never gets a 401, so a probe needs no secret.
        for headers in ["", "\r\nAuthorization: Bearer wrong-key-0123456", "\r\nX-Api-Key: wrong-key-0123456", "\r\nAuthorization: Basic 0123456789abcdef", "\r\nAuthorization: authenticated"] {
            assert_eq!(route(&limits, "/health", headers).unwrap(), (crate::http::HEALTH_LIVE.to_owned(), true), "{headers}");
        }
        // The API key or the owner key gets the full route.
        for headers in ["\r\nAuthorization: Bearer 0123456789abcdef".to_owned(), "\r\nX-Api-Key: 0123456789abcdef".to_owned(), format!("\r\nAuthorization: Bearer {}", owner.expose())] {
            assert_eq!(route(&limits, "/health", &headers).unwrap(), ("/health".to_owned(), true), "{headers}");
        }
        // No request can name the liveness route: a query is not part of the route.
        assert_eq!(route(&limits, "/health?live", "\r\nX-Api-Key: 0123456789abcdef").unwrap().0, "/health");
        assert_eq!(route(&limits, "/health?live", "").unwrap().0, crate::http::HEALTH_LIVE);
        // With no key configured, as on the default loopback bind, health is unchanged.
        assert_eq!(route(&crate::Limits::default(), "/health", "").unwrap(), ("/health".to_owned(), true));
        // Every other route still needs the key.
        assert_eq!(route(&limits, "/v1/models", "").unwrap_err(), UNAUTHORIZED);
    }
    #[test] fn protected_methods_and_anthropic_routes() {
        let limits = crate::Limits { api_key: Some(ApiKey(Arc::from(&b"0123456789abcdef"[..]), None)), ..Default::default() };
        let access = crate::http::AccessPolicy::new(std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 8080)), &limits).unwrap();
        for (method, path) in [("OPTIONS", "/v1/chat/completions"), ("POST", "/health"), ("POST", "/v1/messages"), ("POST", "/v1/messages/count_tokens")] {
            let error = crate::http::parse_head(format!("{method} {path} HTTP/1.1\r\nHost: localhost:8080").as_bytes(), &access, false).err().unwrap(); assert_eq!(request_error(&error), (401, UNAUTHORIZED_BODY.into()));
        }
        let head = crate::http::parse_head(b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost:8080\r\nX-Api-Key: 0123456789abcdef", &access, false).unwrap(); assert_eq!(head.path, "/cors");
    }
    #[test] fn owner_priority_with_api_auth() {
        let owner = crate::cache::SessionKey::generate().unwrap(); let bearer = format!("Bearer {}", owner.expose());
        let limits = crate::Limits { api_key: Some(ApiKey(Arc::from(&b"0123456789abcdef"[..]), None).with_owner(Some(owner))), ..Default::default() };
        let access = crate::http::AccessPolicy::new(std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 8080)), &limits).unwrap();
        let sharing = crate::scheduler::Sharing { owner_key: Some(owner), ..Default::default() };
        for (headers, priority) in [(format!("Authorization: {bearer}"), Some(255)), (format!("Authorization: {bearer}\r\nX-Api-Key: 0123456789abcdef"), Some(255)), ("Authorization: Bearer 0123456789abcdef".into(), Some(0)), (format!("Authorization: {bearer}\r\nX-Api-Key: wrong"), None), ("Authorization: authenticated-owner".into(), None)] {
            let result = crate::http::parse_head(format!("GET /v1/models HTTP/1.1\r\nHost: localhost:8080\r\n{headers}").as_bytes(), &access, false);
            assert_eq!(result.ok().map(|head| sharing.priority(head.authorization.as_deref(), 0, true)), priority);
        }
        assert_eq!(sharing.priority(Some("authenticated-owner"), 0, false), 0);
    }
    #[test] fn refusal_and_body() {
        assert_eq!(ApiKey::load(None, true).unwrap_err(), "--allow-remote requires --api-key-file PATH"); assert!(ApiKey::load(None, false).unwrap().is_none());
        assert_eq!(request_error(UNAUTHORIZED), (401, "{\"error\":{\"message\":\"Unauthorized\",\"type\":\"authentication_error\",\"param\":null,\"code\":\"invalid_api_key\"}}".into()));
    }
    #[cfg(unix)]
    #[test]
    fn fifo_without_a_writer_has_a_hard_timeout() {
        for mode in ["0666", "0600"] {
            let path = std::env::temp_dir().join(format!("vade-key-no-writer-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            assert!(std::process::Command::new("mkfifo").args(["-m", mode]).arg(&path).status().unwrap().success());
            let (send, receive) = std::sync::mpsc::channel();
            let worker = { let path = path.clone(); std::thread::spawn(move || { let _ = send.send(ApiKey::load(Some(&path), false).map(|_| ()).map_err(|e| e.to_string())); }) };
            let result = receive.recv_timeout(std::time::Duration::from_secs(2));
            std::fs::remove_file(&path).unwrap();
            // Never join a blocked loader on failure: the test must have a hard deadline.
            let error = result.expect("key reader blocked on a FIFO without a writer").unwrap_err();
            worker.join().unwrap();
            assert!(error.contains("regular file"), "{mode}: {error}");
        }
    }
    // A link to a regular file is read, as before the pre-open check. A link to a pipe is refused at once.
    #[cfg(unix)]
    #[test]
    fn linked_key_path_is_followed() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let load = |path: &std::path::Path| -> Result<String, String> { ApiKey::load(Some(path), false).map(|key| String::from_utf8(key.unwrap().0.to_vec()).unwrap()) };
        let path = std::env::temp_dir().join(format!("vade-key-link-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let (target, pipe) = (path.with_extension("file"), path.with_extension("fifo"));
        std::fs::write(&target, "0123456789abcdef").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();
        symlink(&target, &path).unwrap();
        let linked = load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(std::process::Command::new("mkfifo").args(["-m", "0600"]).arg(&pipe).status().unwrap().success());
        symlink(&pipe, &path).unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        let worker = { let path = path.clone(); std::thread::spawn(move || { let _ = send.send(load(&path)); }) };
        let result = receive.recv_timeout(std::time::Duration::from_secs(2));
        for file in [&path, &target, &pipe] { std::fs::remove_file(file).unwrap(); }
        assert_eq!(linked.unwrap(), "0123456789abcdef");
        // Never join a blocked loader on failure: the test must have a hard deadline.
        let error = result.expect("key reader blocked on a linked FIFO without a writer").unwrap_err();
        worker.join().unwrap();
        assert!(error.contains("regular file"), "{error}");
    }
    // A pipe with a writer is refused, not read, whatever its mode: another account could feed it a key.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn fifo_with_a_writer_is_refused_not_read() {
        use std::os::unix::fs::OpenOptionsExt;
        let load = |path: &std::path::Path| -> Result<String, String> { ApiKey::load(Some(path), false).map(|key| String::from_utf8(key.unwrap().0.to_vec()).unwrap()) };
        for mode in ["0666", "0600"] {
            let path = std::env::temp_dir().join(format!("vade-key-fifo-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            assert!(std::process::Command::new("mkfifo").args(["-m", mode]).arg(&path).status().unwrap().success());
            let writer = { let path = path.clone(); std::thread::spawn(move || {
                use std::io::Write;
                if let Ok(mut pipe) = std::fs::OpenOptions::new().write(true).open(&path) { let _ = pipe.write_all(b"0123456789abcdef\n"); }
            }) };
            let (send, receive) = std::sync::mpsc::channel();
            let worker = { let path = path.clone(); std::thread::spawn(move || { let _ = send.send(load(&path)); }) };
            let result = receive.recv_timeout(std::time::Duration::from_secs(2));
            // Let a writer still waiting for a reader finish.
            let reader = std::fs::OpenOptions::new().read(true).custom_flags(NONBLOCK).open(&path).unwrap();
            writer.join().unwrap(); drop(reader); std::fs::remove_file(&path).unwrap();
            let error = result.expect("key reader blocked on a FIFO").unwrap_err();
            worker.join().unwrap();
            assert!(error.contains("regular file") && !error.contains("0123456789"), "{mode}: {error}");
        }
    }

}
