use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

pub type Fields = BTreeMap<String, String>;

/// Display untrusted text without allowing it to control the terminal. Keep
/// ordinary Unicode, line breaks and tabs readable; never change stored bytes.
pub fn terminal_text(value: &str) -> std::borrow::Cow<'_, str> {
    fn unsafe_char(ch: char) -> bool {
        (ch.is_control() && ch != '\n' && ch != '\t')
            || matches!(ch, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' | '\u{2028}' | '\u{2029}')
    }
    if !value.chars().any(unsafe_char) {
        return std::borrow::Cow::Borrowed(value);
    }
    let mut result = String::with_capacity(value.len());
    for ch in value.chars() {
        if unsafe_char(ch) { result.extend(ch.escape_default()); }
        else { result.push(ch); }
    }
    std::borrow::Cow::Owned(result)
}

/// Quote a bounded argument on one diagnostic line, including invisible text.
pub fn argument(value: &str) -> String {
    let short: String = value.chars().take(128).collect();
    format!("{}{}", quoted(&short), if short.len() < value.len() { "…" } else { "" })
}

pub fn quoted(value: &str) -> String {
    terminal_text(&format!("{value:?}")).into_owned()
}

/// Keep filesystem failures specific without echoing file contents or arbitrary
/// subprocess stderr. The caller supplies the operation and affected path.
pub fn file_error(operation: &str, path: &Path, error: &std::io::Error) -> String {
    use std::io::ErrorKind;
    let detail = match error.kind() {
        ErrorKind::NotFound => "path not found. Check the path and restore the required file",
        ErrorKind::PermissionDenied => "permission denied. Check the file and parent directory permissions",
        ErrorKind::AlreadyExists => "destination exists. Keep the existing file and choose another destination",
        ErrorKind::StorageFull => "storage is full. Free space on this volume and retry",
        ErrorKind::ReadOnlyFilesystem => "filesystem is read-only. Choose a writable destination",
        ErrorKind::IsADirectory | ErrorKind::NotADirectory => "wrong path type. Select the expected file or directory",
        ErrorKind::InvalidData => "invalid file data. Keep the file for inspection and run vade doctor",
        _ => "I/O failed. Check the path and available space; keep saved ownership records before retrying",
    };
    format!("{operation} {}: {detail}.", quoted(&path.to_string_lossy()))
}

pub fn parse(input: &str) -> Result<Fields, String> {
    let mut result = Fields::new();
    for line in input.lines() {
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or("Invalid record. Keep the record and run vade doctor.")?;
        if key.is_empty()
            || !key
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        {
            return Err("Invalid record key. Keep the record and run vade doctor.".into());
        }
        let mut decoded = String::new();
        let mut chars = value.chars();
        while let Some(ch) = chars.next() {
            if ch != '\\' {
                decoded.push(ch);
                continue;
            }
            decoded.push(match chars.next() {
                Some('n') => '\n',
                Some('r') => '\r',
                Some('\\') => '\\',
                _ => {
                    return Err(
                        "Invalid record escape. Keep the record and run vade doctor.".into(),
                    );
                }
            });
        }
        if decoded.contains('\0') || result.insert(key.into(), decoded).is_some() {
            return Err(
                "Invalid or duplicate record value. Keep the record and run vade doctor.".into(),
            );
        }
    }
    Ok(result)
}

pub fn encode(fields: &Fields) -> String {
    let mut result = String::new();
    for (key, value) in fields {
        result.push_str(key);
        result.push('=');
        for ch in value.chars() {
            match ch {
                '\\' => result.push_str("\\\\"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                _ => result.push(ch),
            }
        }
        result.push('\n');
    }
    result
}

pub fn private_dir(path: &Path) -> Result<(), String> {
    reject_symlinks(path)?;
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(0o700);
    builder
        .create(path)
        .map_err(|_| "Cannot create private directory. Check its permissions.")?;
    reject_symlinks(path)?;
    let metadata = fs::metadata(path)
        .map_err(|_| "Cannot inspect private directory. Check its permissions.")?;
    if !metadata.is_dir() {
        return Err(
            "A private path is not a directory. Choose a dedicated directory in config.".into(),
        );
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(format!(
            "Directory {} is shared. Run chmod 700 {}, then run vade install again.",
            path.display(), crate::runner::quote(&path.to_string_lossy())
        ));
    }
    Ok(())
}

pub fn reject_symlinks(path: &Path) -> Result<(), String> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        if fs::symlink_metadata(&current).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(
                "A state path is a symlink. Choose a private directory without symlinks.".into(),
            );
        }
    }
    Ok(())
}

pub fn atomic_write(path: &Path, value: &str) -> Result<(), String> {
    reject_symlinks(path)?;
    let parent = path
        .parent()
        .ok_or("State file needs a parent directory. Choose a private state directory in config.")?;
    private_dir(parent)?;
    let tmp = parent.join(format!(".write-{}-{}", std::process::id(), random_hex(8)?));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let outcome = (|| {
        let mut file = options
            .open(&tmp)
            .map_err(|_| "Cannot write private state. Check disk and permissions.")?;
        file.write_all(value.as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(|_| "Cannot save state. Check free disk space.")?;
        fs::rename(&tmp, path).map_err(|_| "Cannot publish state. Check directory permissions.")?;
        let directory = fs::File::open(parent).map_err(|_| "Cannot open state directory. Check its path and permissions; keep existing records.")?;
        directory
            .sync_all()
            .map_err(|_| "Cannot sync state directory. Check free space and filesystem health; keep existing records.")?;
        Ok(())
    })();
    if outcome.is_err() {
        let _ = fs::remove_file(tmp);
    }
    outcome
}

pub fn random_hex(bytes: usize) -> Result<String, String> {
    let mut value = vec![0; bytes];
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut value))
        .map_err(
            |_| "Cannot create a safe ownership token. Check the operating system random source.",
        )?;
    Ok(value.iter().map(|b| format!("{b:02x}")).collect())
}

pub fn required<'a>(fields: &'a Fields, key: &str) -> Result<&'a str, String> {
    fields
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("Record is missing {key}. Keep it and run vade doctor."))
}

pub fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protocol_round_trip() {
        let fields = Fields::from([("log".into(), "a=b\nline\\end\r".into())]);
        assert_eq!(parse(&encode(&fields)).unwrap(), fields);
        assert!(parse("a=1\na=2\n").is_err());
        assert!(parse("a=bad\\x\n").is_err());
    }
    #[test]
    fn newly_created_directories_are_private_and_existing_ancestors_keep_their_mode() {
        let root = fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("vade-dirs-{}", random_hex(12).unwrap()));
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
        private_dir(&root.join("new/nested")).unwrap();
        for relative in ["new", "new/nested"] {
            assert_eq!(
                fs::metadata(root.join(relative))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
        assert!(private_dir(&root).unwrap_err().contains("shared"));
        assert_eq!(
            fs::metadata(&root).unwrap().permissions().mode() & 0o777,
            0o755
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod diagnostic_branch_tests {
    use super::*;
    #[test]
    fn filesystem_failures_name_the_branch_and_recovery_without_raw_error_text() {
        for (kind, detail, action) in [
            (std::io::ErrorKind::PermissionDenied, "permission denied", "permissions"),
            (std::io::ErrorKind::AlreadyExists, "destination exists", "Keep the existing file"),
            (std::io::ErrorKind::StorageFull, "storage is full", "Free space"),
            (std::io::ErrorKind::ReadOnlyFilesystem, "filesystem is read-only", "writable destination"),
        ] {
            let error = file_error("Cannot save", Path::new("fixture\x1b[2J"), &std::io::Error::new(kind, "PRIVATE error payload"));
            assert!(error.contains(detail) && error.contains(action), "{error}");
            assert!(error.contains("Cannot save") && error.contains("fixture"));
            assert!(!error.contains("PRIVATE") && !error.contains('\x1b'));
        }
    }
}
