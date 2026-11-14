//! Invocation credentials. File paths are forwarded; passwords never enter argv.
use std::{path::{Path, PathBuf}, sync::OnceLock};
use crate::runner::Exec;

struct Credential { path: PathBuf, key: OnceLock<Result<String, String>> }
static CREDENTIAL: OnceLock<Option<Credential>> = OnceLock::new();

/// Take `--api-key-file PATH` out of `args` (or use `VADE_API_KEY_FILE`) and
/// remember the path for this process. The file is not opened yet.
pub fn configure(args: &mut Vec<String>) -> Result<(), String> {
    let path = select_file(args, std::env::var_os("VADE_API_KEY_FILE").map(PathBuf::from))?;
    CREDENTIAL.set(path.map(Credential::new)).map_err(|_| "API credential is already configured. Start a new vade command.".into())
}

// Selecting an option never opens or resolves its file, even for utility commands.
fn select_file(args: &mut Vec<String>, mut path: Option<PathBuf>) -> Result<Option<PathBuf>, String> {
    let mut seen = false;
    let mut index = 0;
    while index < args.len() {
        if args[index] != "--api-key-file" { index += 1; continue; }
        if seen { return Err("Use --api-key-file only once; choose one readable key file.".into()); }
        seen = true;
        let value = args.get(index + 1).filter(|v| !v.is_empty() && !v.starts_with("--"))
            .ok_or("Give --api-key-file a readable key file path. See vade --help.")?;
        path = Some(value.into()); args.drain(index..index + 2);
    }
    // Preserve the invocation directory when passing the path to a child.
    // absolute() does not access the credential file or resolve symlinks.
    path.map(|path| std::path::absolute(path).map_err(|_| "Cannot resolve the API key path. Change to an existing folder and retry.".into())).transpose()
}

impl Credential {
    fn new(path: PathBuf) -> Self { Self { path, key: OnceLock::new() } }
    fn key(&self) -> Result<&str, String> {
        self.key.get_or_init(|| {
            use std::io::Read;
            let failed = |error: std::io::Error| {
                if error.kind() == std::io::ErrorKind::InvalidData {
                    format!("Cannot read --api-key-file {}: invalid UTF-8. Replace its contents with the engine's visible-ASCII key.", crate::text::quoted(&self.path.to_string_lossy()))
                } else { crate::text::file_error("Cannot read --api-key-file", &self.path, &error) }
            };
            let check_kind = |metadata: &std::fs::Metadata| {
                if metadata.is_dir() { return Err(failed(std::io::ErrorKind::IsADirectory.into())); }
                if !metadata.is_file() {
                    return Err(format!("Cannot use --api-key-file {}: the path is not a regular file. Save the key in a plain file and run chmod 600 on it.", crate::text::quoted(&self.path.to_string_lossy())));
                }
                Ok(())
            };
            // Prove the path is a regular file before the open: a pipe can block the open or feed a key.
            // A link is followed, as before this check; its target must be a regular file.
            let named = std::fs::symlink_metadata(&self.path).map_err(failed)?;
            check_kind(&if named.file_type().is_symlink() { std::fs::metadata(&self.path).map_err(failed)? } else { named })?;
            // A path swapped to a pipe after the check must not wait for a writer.
            let mut file = {
                use std::os::unix::fs::OpenOptionsExt;
                std::fs::OpenOptions::new().read(true).custom_flags(crate::unix_flags::O_NONBLOCK).open(&self.path).map_err(failed)?
            };
            // Another account must not read the key, or put its own key in the file.
            // The checks and the read use the same open file.
            let metadata = file.metadata().map_err(failed)?;
            check_kind(&metadata)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if metadata.permissions().mode() & 0o066 != 0 {
                    return Err(format!("Cannot use --api-key-file {}: group or others can read or write the file. Run chmod 600 on it.", crate::text::quoted(&self.path.to_string_lossy())));
                }
            }
            // The engine reads at most 4096 bytes of a key file. A larger file is not a key.
            let mut text = String::new();
            file.by_ref().take(4097).read_to_string(&mut text).map_err(failed)?;
            if text.len() > 4096 {
                return Err(format!("Cannot use --api-key-file {}: the file is larger than 4096 bytes. Replace its contents with the engine's configured key.", crate::text::quoted(&self.path.to_string_lossy())));
            }
            validate_key(&text).map(str::to_owned)
                .map_err(|message| format!("{message}: {}. Replace its contents with the engine's configured key; do not put the key in command arguments.", crate::text::quoted(&self.path.to_string_lossy())))
        }).as_deref().map_err(Clone::clone)
    }
}

/// The configured key file, if any.
pub fn file() -> Option<&'static Path> {
    CREDENTIAL.get().and_then(Option::as_ref).map(|c| c.path.as_path())
}
/// Pass the key file's path, never the key, to a child command.
pub fn forward(exec: &mut Exec) {
    if let Some(path) = file() {
        exec.env.insert("VADE_API_KEY_FILE".into(), path.to_string_lossy().into_owned());
    }
}

fn credential(dry: bool) -> Result<Option<&'static Credential>, String> {
    if dry { return Ok(None); }
    let credential = CREDENTIAL.get().and_then(Option::as_ref);
    if let Some(credential) = credential { credential.key()?; }
    Ok(credential)
}

/// Commands that send or forward the key validate at entry, before side effects.
/// The health request to an engine that vade run owns sends the key (an engine with a key
/// shows its model and load only with it). Other health requests never load or attach it.
pub fn validate(dry: bool) -> Result<(), String> { credential(dry).map(|_| ()) }

/// The validated key for an `Authorization: Bearer` header, if configured.
pub(crate) fn bearer_key() -> Result<Option<&'static str>, String> {
    credential(false)?.map(Credential::key).transpose()
}

/// Add the key to a curl command through a config on its standard input, as
/// the evals client does. No argv, preview or file holds the key, so an
/// interrupt leaves nothing behind. A body that curl read from standard input
/// (`--data-binary @-`) moves into the same config.
pub fn curl_headers(exec: &mut Exec, dry: bool) -> Result<(), String> {
    let Some(credential) = credential(dry)? else { return Ok(()); };
    exec.stdin = curl_config(exec, credential.key()?)?;
    exec.args.extend(["--config".into(), "-".into()]);
    Ok(())
}
fn curl_config(exec: &mut Exec, key: &str) -> Result<Vec<u8>, String> {
    let mut config = format!("header = \"Authorization: Bearer {}\"\n", curl_quote(key));
    if let Some(at) = exec.args.windows(2).position(|pair| pair[0] == "--data-binary" && pair[1] == "@-") {
        exec.args.drain(at..at + 2);
        let body = std::str::from_utf8(&exec.stdin).map_err(|_| "The curl request body is not UTF-8 text.")?;
        config.push_str(&format!("data-binary = \"{}\"\n", curl_quote(body)));
    } else if !exec.stdin.is_empty() {
        return Err("The curl command already uses standard input; the API key cannot be added.".into());
    }
    Ok(config.into_bytes())
}
/// A curl config string value: backslash, quote and control escapes.
pub(crate) fn curl_quote(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn validate_key(text: &str) -> Result<&str, &'static str> {
    let key = text.trim();
    if key.len() < 16 || !key.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err("--api-key-file must contain one key of at least 16 bytes using only visible ASCII (0x21-0x7E)");
    }
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn key_and_body_reach_curl_on_stdin_without_a_file() {
        // Before, the key went to a /tmp header file that an interrupt left behind.
        let mut exec = Exec::new("answer", "curl", vec!["--data-binary".into(), "@-".into(), "http://127.0.0.1:1/v1/chat/completions".into()]);
        exec.stdin = br#"{"messages":[{"content":"a \"q\"\n"}]}"#.to_vec();
        let config = String::from_utf8(curl_config(&mut exec, "!0123456789ab\"c\\~").unwrap()).unwrap();
        assert_eq!(config, "header = \"Authorization: Bearer !0123456789ab\\\"c\\\\~\"\ndata-binary = \"{\\\"messages\\\":[{\\\"content\\\":\\\"a \\\\\\\"q\\\\\\\"\\\\n\\\"}]}\"\n");
        assert_eq!(exec.args, ["http://127.0.0.1:1/v1/chat/completions"]);
        // Other standard input is never replaced.
        let mut busy = Exec::new("other", "curl", Vec::new());
        busy.stdin = b"x".to_vec();
        assert!(curl_config(&mut busy, "0123456789abcdef").is_err());
        let path = std::env::temp_dir().join(format!("vade-auth-test-{}", crate::text::random_hex(12).unwrap())); std::fs::write(&path, "short").unwrap();
        assert!(Credential::new(path.clone()).key().is_err());
        // A key file that group or others can read or write is refused, with the remedy and no key text.
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            std::fs::write(&path, "0123456789abcdef\n").unwrap();
            for mode in [0o640, 0o604, 0o644, 0o620, 0o602] {
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
                let error = Credential::new(path.clone()).key().unwrap_err();
                assert!(error.starts_with("Cannot use --api-key-file") && error.contains("chmod 600") && !error.contains("0123456789"), "{mode:o}: {error}");
            }
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            assert_eq!(Credential::new(path.clone()).key().unwrap(), "0123456789abcdef");
        }
        // The read stops at the engine's limit: a large file is refused, not loaded.
        std::fs::write(&path, "k".repeat(4097)).unwrap();
        assert!(Credential::new(path.clone()).key().unwrap_err().contains("larger than 4096 bytes"));
        std::fs::remove_file(path).unwrap();
    }
    #[test] fn missing_file_is_deferred_until_request() {
        let path = std::env::temp_dir().join(format!("vade-missing-{}", crate::text::random_hex(12).unwrap()));
        let mut args = vec!["logs".into(), "--api-key-file".into(), path.to_string_lossy().into_owned()];
        assert_eq!(select_file(&mut args, Some("unused-env-file".into())).unwrap(), Some(path.clone())); assert_eq!(args, ["logs"]);
        let credential = Credential::new(path.clone()); assert!(credential.key.get().is_none());
        let error = credential.key().unwrap_err(); assert!(error.contains("Cannot read --api-key-file") && error.contains("path not found") && error.contains("Check the path")); assert!(error.contains(&path.to_string_lossy().to_string()));
        assert_eq!(select_file(&mut vec!["logs".into()], Some(path.clone())).unwrap(), Some(path));
        let directory = Credential::new(std::env::temp_dir()); assert!(directory.key.get().is_none()); assert!(directory.key().unwrap_err().starts_with("Cannot read --api-key-file"));
    }
    // A pipe is not a key file, whatever its mode: another account could feed it a key.
    #[cfg(unix)]
    #[test] fn pipe_is_refused_at_any_mode() {
        for mode in ["0666", "0600"] {
            let path = std::env::temp_dir().join(format!("vade-key-fifo-{}", crate::text::random_hex(12).unwrap()));
            assert!(std::process::Command::new("mkfifo").args(["-m", mode]).arg(&path).status().unwrap().success());
            // A writer, so that a loader that waits for one cannot hang the test.
            let writer = { let path = path.clone(); std::thread::spawn(move || {
                use std::io::Write;
                if let Ok(mut pipe) = std::fs::OpenOptions::new().write(true).open(&path) { let _ = pipe.write_all(b"0123456789abcdef\n"); }
            }) };
            let result = Credential::new(path.clone()).key().map(str::to_owned);
            // Let a writer still waiting for a reader finish.
            let reader = { use std::os::unix::fs::OpenOptionsExt; std::fs::OpenOptions::new().read(true).custom_flags(crate::unix_flags::O_NONBLOCK).open(&path).unwrap() };
            writer.join().unwrap(); drop(reader); std::fs::remove_file(&path).unwrap();
            let error = result.unwrap_err();
            assert!(error.starts_with("Cannot use --api-key-file") && error.contains("not a regular file") && !error.contains("0123456789"), "{mode}: {error}");
        }
    }
    #[test] fn visible_ascii() {
        for key in ["short", "0123456789abcdefé", "0123456789abcdef\u{7f}", "01234567 9abcdef", "01234567\t9abcdef", "01234567\n9abcdef"] { assert!(validate_key(key).is_err()); }
        assert_eq!(validate_key("  !0123456789abcde~\n").unwrap(), "!0123456789abcde~");
    }
    #[cfg(unix)]
    #[test]
    fn fifo_without_a_writer_has_a_hard_timeout() {
        for mode in ["0666", "0600"] {
            let path = std::env::temp_dir().join(format!("vade-key-no-writer-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            assert!(std::process::Command::new("mkfifo").args(["-m", mode]).arg(&path).status().unwrap().success());
            let (send, receive) = std::sync::mpsc::channel();
            let worker = { let path = path.clone(); std::thread::spawn(move || { let _ = send.send(Credential::new(path.clone()).key().map(|_| ()).map_err(|e| e.to_string())); }) };
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
        let load = |path: &std::path::Path| -> Result<String, String> { Credential::new(path.to_path_buf()).key().map(str::to_owned) };
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
