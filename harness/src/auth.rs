//! File credentials for every upstream route; never formatted or persisted.
use std::{
    fmt, io,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

/// The engine's own limit for a key file.
const MAX_KEY_FILE_BYTES: u64 = 4096;
// Audited O_NONBLOCK values (as in cli/args.rs): opening a pipe must not wait
// for a writer. Regular files are unaffected.
#[cfg(target_os = "macos")]
const NONBLOCK: i32 = 0x4;
#[cfg(target_os = "linux")]
const NONBLOCK: i32 = 0x800;

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
        use io::Read;
        let check_kind = |metadata: &std::fs::Metadata| {
            if metadata.is_dir() { return Err(read_error(path, io::ErrorKind::IsADirectory)); }
            if !metadata.is_file() {
                return Err(io::Error::other(format!(
                    "cannot use --api-key-file: {}: not a regular file. Save the engine's API key in a plain file and run chmod 600 on it, or use --api-key-file PATH or VADE_API_KEY_FILE to select another key file.", quoted_path(path)
                )));
            }
            Ok(())
        };
        // Prove the path is a regular file before the open: a pipe can block
        // the open or feed a key. A link is followed, as before this check;
        // its target must be a regular file.
        let named = std::fs::symlink_metadata(path).map_err(|error| read_error(path, error.kind()))?;
        let named = if named.file_type().is_symlink() {
            std::fs::metadata(path).map_err(|error| read_error(path, error.kind()))?
        } else {
            named
        };
        check_kind(&named)?;
        let mut open = std::fs::OpenOptions::new();
        open.read(true);
        // A path swapped to a pipe after the check must not wait for a writer.
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        std::os::unix::fs::OpenOptionsExt::custom_flags(&mut open, NONBLOCK);
        let mut file = open.open(path).map_err(|error| read_error(path, error.kind()))?;
        // Another account must not read the key, or put its own key in the
        // file. The checks and the read use the same open file.
        let metadata = file.metadata().map_err(|error| read_error(path, error.kind()))?;
        check_kind(&metadata)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o066 != 0 {
                return Err(io::Error::other(format!(
                    "cannot use --api-key-file: {}: group or others can read or write the file. Run chmod 600 on it, or use --api-key-file PATH or VADE_API_KEY_FILE to select another key file.",
                    quoted_path(path)
                )));
            }
        }
        // The engine reads at most 4096 bytes of a key file. A larger file is not a key.
        let mut text = String::new();
        file.by_ref().take(MAX_KEY_FILE_BYTES + 1).read_to_string(&mut text).map_err(|error| read_error(path, error.kind()))?;
        if text.len() as u64 > MAX_KEY_FILE_BYTES {
            return Err(io::Error::other(format!(
                "cannot use --api-key-file: {}: the file is larger than 4096 bytes. Replace the file contents with the engine's API key, or select its file with --api-key-file PATH or VADE_API_KEY_FILE.",
                quoted_path(path)
            )));
        }
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
        // A key that group or others can read or write is refused, with the remedy and no key text.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for mode in [0o640, 0o604, 0o644, 0o620, 0o602] {
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
                let error = ApiKey::read(&path).unwrap_err().to_string();
                assert!(error.contains("chmod 600") && error.contains(&quoted_path(&path)) && !error.contains("0123456789"), "{mode:o}: {error}");
            }
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        // The read stops at the engine's limit: a large file is refused, not loaded.
        std::fs::write(&path, "k".repeat(4097)).unwrap();
        assert!(ApiKey::read(&path).unwrap_err().to_string().contains("larger than 4096 bytes"));
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
    // A pipe is not a key file, whatever its mode: another account could feed it a key.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn pipe_is_refused_at_any_mode() {
        use std::os::unix::fs::OpenOptionsExt;
        for mode in ["0666", "0600"] {
            let path = std::env::temp_dir().join(format!(
                "vade-key-fifo-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            assert!(std::process::Command::new("mkfifo")
                .args(["-m", mode])
                .arg(&path)
                .status()
                .unwrap()
                .success());
            // A writer, so that a loader that waits for one cannot hang the test.
            let writer = {
                let path = path.clone();
                std::thread::spawn(move || {
                    use std::io::Write;
                    if let Ok(mut pipe) = std::fs::OpenOptions::new().write(true).open(&path) {
                        let _ = pipe.write_all(b"0123456789abcdef\n");
                    }
                })
            };
            let result = ApiKey::read(&path);
            // Let a writer still waiting for a reader finish.
            let reader = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(NONBLOCK)
                .open(&path)
                .unwrap();
            writer.join().unwrap();
            drop(reader);
            std::fs::remove_file(&path).unwrap();
            let error = result.unwrap_err().to_string();
            assert!(
                error.starts_with("cannot use --api-key-file:")
                    && error.contains("not a regular file")
                    && error.contains(&quoted_path(&path))
                    && !error.contains("0123456789"),
                "{mode}: {error}"
            );
        }
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
    #[cfg(unix)]
    #[test]
    fn fifo_without_a_writer_has_a_hard_timeout() {
        for mode in ["0666", "0600"] {
            let path = std::env::temp_dir().join(format!("vade-key-no-writer-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            assert!(std::process::Command::new("mkfifo").args(["-m", mode]).arg(&path).status().unwrap().success());
            let (send, receive) = std::sync::mpsc::channel();
            let worker = { let path = path.clone(); std::thread::spawn(move || { let _ = send.send(ApiKey::read(&path).map(|_| ()).map_err(|e| e.to_string())); }) };
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
        let load = |path: &std::path::Path| -> Result<String, String> { ApiKey::read(path).map(|key| key.0.to_string()).map_err(|e| e.to_string()) };
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

}
