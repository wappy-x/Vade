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
            let text = std::fs::read_to_string(&self.path)
                .map_err(|error| {
                    if error.kind() == std::io::ErrorKind::InvalidData {
                        format!("Cannot read --api-key-file {}: invalid UTF-8. Replace its contents with the engine's visible-ASCII key.", crate::text::quoted(&self.path.to_string_lossy()))
                    } else { crate::text::file_error("Cannot read --api-key-file", &self.path, &error) }
                })?;
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
/// Open health requests never load or attach credentials.
pub fn validate(dry: bool) -> Result<(), String> { credential(dry).map(|_| ()) }

/// The validated key for an `Authorization: Bearer` header, if configured.
pub(crate) fn bearer_key() -> Result<Option<&'static str>, String> {
    credential(false)?.map(Credential::key).transpose()
}

/// curl reads a private header file, so the key never appears in previews/argv.
/// The guard removes only the temporary file we created, on success or error.
pub struct HeaderFile(PathBuf);
impl Drop for HeaderFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}
/// Add the key to a curl command through a private header file; keep the
/// returned guard alive until curl exits.
pub fn curl_headers(exec: &mut Exec, dry: bool) -> Result<Option<HeaderFile>, String> {
    let Some(credential) = credential(dry)? else { return Ok(None); };
    let guard = HeaderFile::write(credential)?;
    exec.args.extend(["--header".into(), format!("@{}", guard.0.display())]);
    Ok(Some(guard))
}
impl HeaderFile {
    fn write(credential: &Credential) -> Result<Self, String> {
        use std::{io::Write, os::unix::fs::OpenOptionsExt};
        let path = std::env::temp_dir().join(format!("vade-api-header-{}-{}", std::process::id(), crate::text::random_hex(12)?));
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600)
            .open(&path).map_err(|error| crate::text::file_error("Cannot create private auth header", &path, &error))?;
        let guard = HeaderFile(path);
        writeln!(file, "Authorization: Bearer {}", credential.key()?).map_err(|error| crate::text::file_error("Cannot write private auth header", &guard.0, &error))?;
        Ok(guard)
    }
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
    #[test] fn private_header_is_removed() {
        use std::os::unix::fs::PermissionsExt;
        let path = std::env::temp_dir().join(format!("vade-auth-test-{}", crate::text::random_hex(12).unwrap())); std::fs::write(&path, "  !0123456789abcde~\n").unwrap();
        let credential = Credential::new(path.clone()); let header = HeaderFile::write(&credential).unwrap(); let header_path = header.0.clone();
        assert_eq!(std::fs::metadata(&header_path).unwrap().permissions().mode() & 0o777, 0o600); assert_eq!(std::fs::read_to_string(&header_path).unwrap(), "Authorization: Bearer !0123456789abcde~\n"); drop(header); assert!(!header_path.exists());
        std::fs::write(&path, "short").unwrap(); assert!(Credential::new(path.clone()).key().is_err()); std::fs::remove_file(path).unwrap();
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
    #[test] fn visible_ascii() {
        for key in ["short", "0123456789abcdefé", "0123456789abcdef\u{7f}", "01234567 9abcdef", "01234567\t9abcdef", "01234567\n9abcdef"] { assert!(validate_key(key).is_err()); }
        assert_eq!(validate_key("  !0123456789abcde~\n").unwrap(), "!0123456789abcde~");
    }
}
