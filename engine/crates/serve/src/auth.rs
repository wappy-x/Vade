//! Startup credentials and fixed-work request authentication, using std only.
use std::{fmt, path::Path, sync::Arc};

// Owner policy: set true to require a key on loopback too.
const REQUIRE_LOOPBACK_KEY: bool = false;
const MAX_KEY_FILE_BYTES: u64 = 4096;
const PERMISSION_WARNING: &str = "WARNING: --api-key-file is group- or world-readable";
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
        let mut file = std::fs::File::open(path).map_err(|_| "cannot open --api-key-file")?;
        let metadata = file.metadata().map_err(|_| "cannot inspect --api-key-file permissions")?;
        if !metadata.is_file() { return Err("--api-key-file must be a regular file".into()); }
        let mut text = String::new();
        use std::io::Read;
        file.by_ref().take(MAX_KEY_FILE_BYTES + 1).read_to_string(&mut text)
            .map_err(|_| "cannot read --api-key-file as UTF-8")?;
        if text.len() as u64 > MAX_KEY_FILE_BYTES {
            return Err("--api-key-file must be at most 4096 bytes".into());
        }
        let key = validate_key(&text)?;
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            if permission_warning(metadata.permissions().mode()) {
                eprintln!("{PERMISSION_WARNING}");
            }
        }
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
fn permission_warning(mode: u32) -> bool { mode & 0o044 != 0 }

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
        for bytes in [b"".as_slice(), b"short", b"01234567\n89abcdef", b"01234567\x0189abcdef", b"0123456789abcdef\xff"] { std::fs::write(&path, bytes).unwrap(); assert!(ApiKey::load(Some(&path), false).is_err()); }
        std::fs::write(&path, "a".repeat(4097)).unwrap(); assert_eq!(ApiKey::load(Some(&path), false).unwrap_err(), "--api-key-file must be at most 4096 bytes");
        std::fs::write(&path, b"0123456789abcdef\n").unwrap(); let key = ApiKey::load(Some(&path), false).unwrap().unwrap(); assert!(key.accepts(None, Some("0123456789abcdef"))); assert_eq!(format!("{key:?}"), "ApiKey([redacted])");
        assert_eq!(ApiKey::load(Some(&directory), false).unwrap_err(), "--api-key-file must be a regular file"); std::fs::remove_dir_all(directory).unwrap();
        #[cfg(unix)] { assert!(!permission_warning(0o600)); for mode in [0o640, 0o604, 0o644] { assert!(permission_warning(mode)); } }
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
}
