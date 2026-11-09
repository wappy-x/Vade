//! File credentials for every upstream route; never formatted or persisted.
use std::{
    fmt, io,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

#[derive(Clone)]
pub(crate) struct ApiKey(Arc<str>);
impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey([redacted])")
    }
}
struct KeyFile {
    path: Option<PathBuf>,
    key: OnceLock<Result<Option<ApiKey>, String>>,
}
static FILE: OnceLock<KeyFile> = OnceLock::new();

#[derive(Debug)]
struct ReportedFailure(String);
impl fmt::Display for ReportedFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for ReportedFailure {}

/// The credential loader already printed this local failure with its remedy.
pub fn already_reported(error: &io::Error) -> bool {
    error.get_ref().is_some_and(|cause| cause.is::<ReportedFailure>())
}

impl KeyFile {
    fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            key: OnceLock::new(),
        }
    }
    fn load(&self) -> io::Result<Option<ApiKey>> {
        self.key
            .get_or_init(|| {
                self.path
                    .as_deref()
                    .map(ApiKey::read)
                    .transpose()
                    .map_err(|error| {
                        // Agent reports retain only the error kind and gateway/doctor
                        // diagnostics hide upstream text. Emit this local-file failure
                        // once so every HTTP consumer still shows the selected path.
                        let message = error.to_string();
                        crate::diagnostics::stderr_line(format_args!("vade: {}", crate::terminal::sanitize(&message)));
                        message
                    })
            })
            .as_ref()
            .cloned()
            .map_err(|message| io::Error::other(ReportedFailure(message.clone())))
    }
}
impl ApiKey {
    fn read(path: &Path) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| read_error(path, error.kind()))?;
        let key = validate_key(&text).map_err(|message| {
            io::Error::other(format!(
                "{message}: {}. Replace the file contents with the engine's API key, or select its file with --api-key-file PATH or VADE_API_KEY_FILE.",
                quoted_path(path)
            ))
        })?;
        Ok(Self(Arc::from(key)))
    }
    pub(crate) fn configured() -> io::Result<Option<Self>> {
        FILE.get_or_init(|| KeyFile::new(std::env::var_os("VADE_API_KEY_FILE").map(PathBuf::from)))
            .load()
    }
    pub(crate) fn bearer_header(&self) -> String {
        format!("Authorization: Bearer {}\r\n", self.0)
    }
}

// Quote paths on one line, preserving printable Unicode but making terminal
// controls and directional characters visible. Never include file contents or
// an arbitrary I/O error string in a credential diagnostic.
fn quoted_path(path: &Path) -> String {
    let mut shown = String::new();
    for c in format!("{path:?}").chars() {
        if c.is_control() || matches!(c, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{2028}'..='\u{202e}' | '\u{2066}'..='\u{2069}') {
            shown.extend(c.escape_default());
        } else { shown.push(c); }
    }
    shown
}

fn read_error(path: &Path, kind: io::ErrorKind) -> io::Error {
    let detail = match kind {
        io::ErrorKind::NotFound => "file not found. Check the selected path",
        io::ErrorKind::PermissionDenied => "permission denied. Check read permission on the file and access to its parent directories",
        io::ErrorKind::IsADirectory => "path is a directory. Select a regular file containing the key",
        io::ErrorKind::NotADirectory => "a path component is not a directory. Correct the selected path",
        io::ErrorKind::InvalidData => "file is not valid UTF-8. Save the engine's API key as plain text",
        _ => "file read failed. Check that the selected file is readable and the filesystem is available",
    };
    io::Error::other(format!(
        "cannot read --api-key-file: {}: {detail}. Use --api-key-file PATH or VADE_API_KEY_FILE to select the key file.",
        quoted_path(path)
    ))
}

/// Common option for native, direct, gateway, delegated and utility commands.
/// CLI overrides the environment. Only select the path here; upstream modes
/// preload it before opening the UI or listener, other commands load on dispatch.
pub fn configure(args: &mut Vec<String>) -> io::Result<()> {
    let path = select_file(
        args,
        std::env::var_os("VADE_API_KEY_FILE").map(PathBuf::from),
    )?;
    FILE.set(KeyFile::new(path))
        .map_err(|_| io::Error::other("API credential already configured"))
}

/// Load/cache credentials before an upstream mode creates any UI or listener.
pub fn preload() -> io::Result<()> {
    ApiKey::configured().map(|_| ())
}

fn select_file(args: &mut Vec<String>, mut path: Option<PathBuf>) -> io::Result<Option<PathBuf>> {
    let mut seen = false;
    let mut index = 0;
    while index < args.len() {
        if args[index] != "--api-key-file" {
            index += 1;
            continue;
        }
        if seen {
            return Err(io::Error::other("duplicate --api-key-file. Pass this option once; it overrides VADE_API_KEY_FILE"));
        }
        seen = true;
        let value = args
            .get(index + 1)
            .filter(|v| !v.is_empty() && !v.starts_with("--"))
            .ok_or_else(|| io::Error::other("--api-key-file needs a path. Use --api-key-file PATH with a file containing the engine's API key"))?;
        path = Some(value.into());
        args.drain(index..index + 2);
    }
    Ok(path)
}

fn validate_key(text: &str) -> Result<&str, &'static str> {
    let key = text.trim();
    if key.len() < 16 || !key.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err(
            "--api-key-file must contain one key of at least 16 bytes using only visible ASCII (0x21-0x7E)",
        );
    }
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn permission_and_unusual_read_failures_have_local_remedies() {
        for (kind, action) in [
            (io::ErrorKind::PermissionDenied, "Check read permission"),
            (io::ErrorKind::Other, "Check that the selected file is readable"),
        ] {
            let error = read_error(Path::new("key\n\u{202e}"), kind).to_string();
            assert!(error.contains(action), "{error}");
            assert!(error.contains("\\n") && error.contains("\\u{202e}"), "{error}");
            assert_eq!(error.lines().count(), 1);
        }
    }
    #[test]
    fn missing_file_is_deferred_until_request() {
        let path = std::env::temp_dir().join(format!(
            "vade-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut args = vec![
            "setup".into(),
            "--api-key-file".into(),
            path.to_string_lossy().into_owned(),
        ];
        assert_eq!(
            select_file(&mut args, Some("unused-env-file".into())).unwrap(),
            Some(path.clone())
        );
        assert_eq!(args, ["setup"]);
        let file = KeyFile::new(Some(path.clone()));
        assert!(file.key.get().is_none());
        let error = file.load().unwrap_err().to_string();
        assert!(error.contains(&quoted_path(&path)), "{error}");
        assert!(error.contains("file not found. Check the selected path"), "{error}");
        assert_eq!(
            select_file(&mut vec!["setup".into()], Some(path.clone())).unwrap(),
            Some(path)
        );
        let directory = KeyFile::new(Some(std::env::temp_dir()));
        assert!(directory.key.get().is_none());
        assert!(
            directory
                .load()
                .unwrap_err()
                .to_string()
                .starts_with("cannot read --api-key-file:")
        );
    }
    #[test]
    fn file_header_and_redaction() {
        let path = std::env::temp_dir().join(format!(
            "vade-auth-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, "  !0123456789abcde~\n").unwrap();
        let key = ApiKey::read(&path).unwrap();
        assert_eq!(
            key.bearer_header(),
            "Authorization: Bearer !0123456789abcde~\r\n"
        );
        assert_eq!(format!("{key:?}"), "ApiKey([redacted])");
        std::fs::write(&path, "short").unwrap();
        assert!(ApiKey::read(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn visible_ascii() {
        for key in [
            "short",
            "0123456789abcdefé",
            "0123456789abcdef\u{7f}",
            "01234567 9abcdef",
            "01234567\t9abcdef",
            "01234567\n9abcdef",
        ] {
            assert!(validate_key(key).is_err());
        }
        assert_eq!(
            validate_key("  !0123456789abcde~\n").unwrap(),
            "!0123456789abcde~"
        );
    }
}
