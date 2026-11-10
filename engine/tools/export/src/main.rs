//! Exact-file sanitized snapshots; never copy implementation Git history.
mod scan;
mod sha256;

use engine_model::json::{self, Value};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

const MANIFEST: &str = "EXPORT-MANIFEST.json";
const ALLOWLIST: &str = "scripts/export-allowlist.txt";
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn invalid(message: impl Into<String>) -> Box<dyn std::error::Error> {
    io::Error::new(io::ErrorKind::InvalidData, message.into()).into()
}

fn safe_name(name: &str) -> Result<()> {
    if name.contains('\\')
        || name.chars().any(char::is_control)
        || name
            .split('/')
            .any(|c| matches!(c, "" | "." | ".." | ".git") || c.starts_with("._"))
    {
        return Err(invalid("unsafe manifest path"));
    }
    Ok(())
}

fn checked_file(root: &Path, name: &str) -> Result<PathBuf> {
    safe_name(name)?;
    let mut path = root.to_owned();
    for component in name.split('/') {
        path.push(component);
        if fs::symlink_metadata(&path)?.file_type().is_symlink() {
            return Err(invalid(format!("symlink rejected: {name}")));
        }
    }
    if !fs::metadata(&path)?.is_file() {
        return Err(invalid(format!("missing regular file: {name}")));
    }
    Ok(path)
}

// Resolve existing parents without following destination symlinks, including
// symlinked directories above a destination that does not exist yet.
fn destination_path(path: &Path) -> Result<PathBuf> {
    let mut absolute = std::env::current_dir()?;
    for component in path.components() {
        match component {
            Component::CurDir => continue,
            Component::ParentDir => return Err(invalid("destination parent traversal rejected")),
            _ => absolute.push(component.as_os_str()),
        }
    }
    let mut parent = PathBuf::new();
    for component in absolute.components() {
        parent.push(component.as_os_str());
        match fs::symlink_metadata(&parent) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(invalid("destination symlink rejected"));
            }
            Ok(meta) if !meta.is_dir() => {
                return Err(invalid("destination parent is not a directory"));
            }
            Ok(_) => (),
            Err(e) if e.kind() == io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(absolute)
}

fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(invalid(format!("git {} failed", args[0])));
    }
    Ok(output.stdout)
}

// Match the original sorted, ASCII-escaped JSON so existing exports retain
// identical manifests. Input filenames are not restricted to ASCII.
fn quote(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                for word in ch.encode_utf16(&mut [0; 2]) {
                    write!(out, "\\u{word:04x}").unwrap();
                }
            }
        }
    }
    out.push('"');
    out
}

fn manifest(contents: &BTreeMap<String, Vec<u8>>, revision: Option<&str>) -> Vec<u8> {
    let mut out = String::from("{\n  \"files\": {");
    for (i, (name, data)) in contents.iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        write!(out, "\n    {}: \"{}\"", quote(name), sha256::digest(data)).unwrap();
    }
    if !contents.is_empty() {
        out.push_str("\n  ");
    }
    out.push_str("},\n");
    if revision.is_none() {
        out.push_str("  \"preview\": true,\n");
    }
    out.push_str("  \"schema\": 1,\n  \"source_revision\": ");
    out.push_str(&revision.map_or_else(|| "null".to_owned(), quote));
    out.push_str("\n}\n");
    out.into_bytes()
}

fn previous(destination: &Path, preview: bool) -> Result<BTreeMap<String, String>> {
    let mut old = BTreeMap::new();
    match fs::symlink_metadata(destination.join(MANIFEST)) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(old),
        Err(e) => return Err(e.into()),
        Ok(_) => (),
    }
    let data = fs::read(checked_file(destination, MANIFEST)?)?;
    let record = json::parse(&data, json::Limits::default())?;
    let record = record.object()?;
    if json::field(record, "schema")?.u64()? != 1 {
        return Err(invalid("invalid previous manifest"));
    }
    if preview && record.get("preview") != Some(&Value::Bool(true)) {
        return Err(invalid("preview cannot overwrite a release export"));
    }
    for (name, hash) in json::field(record, "files")?.object()? {
        safe_name(name)?;
        if name == MANIFEST {
            return Err(invalid("reserved previous manifest entry"));
        }
        let hash = hash.string()?;
        if sha256::digest(&fs::read(checked_file(destination, name)?)?) != hash {
            return Err(invalid(format!(
                "destination changed independently: {name}"
            )));
        }
        old.insert(name.clone(), hash.to_owned());
    }
    Ok(old)
}

fn inspect_destination(root: &Path, dir: &Path, old: &BTreeMap<String, String>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err(invalid("destination symlink rejected"));
        }
        let rel = path
            .strip_prefix(root)?
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 destination path"))?;
        // An independent checkout may already exist. Never read or write its history.
        if rel == ".git" {
            continue;
        }
        safe_name(rel)?;
        if kind.is_dir() {
            inspect_destination(root, &path, old)?;
        } else if !kind.is_file() || (rel != MANIFEST && !old.contains_key(rel)) {
            return Err(invalid(format!("unrelated destination file: {rel}")));
        }
    }
    Ok(())
}

fn export(source: &Path, destination: &Path, denyfile: &Path, preview: bool) -> Result<()> {
    let source = source.canonicalize()?;
    let destination = destination_path(destination)?;
    if source.starts_with(&destination) || (!preview && destination.starts_with(&source)) {
        return Err(invalid("source and destination must be separate trees"));
    }
    if let Ok(relative) = destination.strip_prefix(&source) {
        safe_name(
            relative
                .to_str()
                .ok_or_else(|| invalid("non-UTF8 destination path"))?,
        )?;
    }
    let denyfile = denyfile.canonicalize()?;
    if denyfile.starts_with(&source) || denyfile.starts_with(&destination) {
        return Err(invalid(
            "private denylist must be outside source and destination",
        ));
    }
    let denied: Vec<String> = fs::read_to_string(denyfile)?
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(str::to_owned)
        .collect();
    if denied.is_empty() {
        return Err(invalid("private denylist must be nonempty"));
    }
    let (revision, source_prefix) = if preview {
        (None, String::new())
    } else {
        let revision = String::from_utf8(git(&source, &["rev-parse", "HEAD"])?)?
            .trim()
            .to_owned();
        if !git(&source, &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
            return Err(invalid(
                "canonical tracked source must be committed before export",
            ));
        }
        let prefix = String::from_utf8(git(&source, &["rev-parse", "--show-prefix"])?)?;
        let prefix = prefix.trim_end_matches('\n');
        if !prefix.is_empty() {
            safe_name(
                prefix
                    .strip_suffix('/')
                    .ok_or_else(|| invalid("invalid Git source prefix"))?,
            )?;
        }
        (Some(revision), prefix.to_owned())
    };
    let allowlist = fs::read_to_string(checked_file(&source, ALLOWLIST)?)?;
    let mut contents = BTreeMap::new();
    for name in allowlist
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
    {
        safe_name(name)?;
        if name == MANIFEST || contents.contains_key(name) {
            return Err(invalid("duplicate/reserved allowlist entry"));
        }
        let mut file_root = source.clone();
        let mut git_name = format!("{source_prefix}{name}");
        let mut snapshot_revision = revision.clone();
        // The monorepo keeps one compiler pin at its root. Every other input
        // stays component-relative, and an existing local pin takes precedence.
        if name == "rust-toolchain.toml"
            && matches!(fs::symlink_metadata(source.join(name)), Err(error) if error.kind() == io::ErrorKind::NotFound)
        {
            let root = String::from_utf8(git(&source, &["rev-parse", "--show-toplevel"])?)?;
            file_root = PathBuf::from(root.trim_end_matches('\n')).canonicalize()?;
            if !source.starts_with(&file_root) {
                return Err(invalid("shared compiler pin must belong to the source repository"));
            }
            git_name = name.to_owned();
            if !git(
                &file_root,
                &["status", "--porcelain", "--untracked-files=all", "--", name],
            )?
            .is_empty()
            {
                return Err(invalid("shared compiler pin must be tracked and committed"));
            }
            if snapshot_revision.is_none() {
                snapshot_revision = Some(
                    String::from_utf8(git(&file_root, &["rev-parse", "HEAD"])?)?
                        .trim()
                        .to_owned(),
                );
            }
        }
        let path = checked_file(&file_root, name)?;
        if path.starts_with(&destination) {
            return Err(invalid(format!(
                "destination cannot be an export input: {name}"
            )));
        }
        let data = fs::read(path)?;
        scan::scan(name, &data, &denied).map_err(invalid)?;
        if let Some(revision) = &snapshot_revision
            && git(&source, &["show", &format!("{revision}:{git_name}")])? != data
        {
            return Err(invalid(format!("source changed during snapshot: {name}")));
        }
        contents.insert(name.to_owned(), data);
    }
    let old = previous(&destination, preview)?;
    if destination.exists() {
        inspect_destination(&destination, &destination, &old)?;
    }
    let record = manifest(&contents, revision.as_deref());
    let count = contents.len();
    let bytes: usize = contents.values().map(Vec::len).sum();
    contents.insert(MANIFEST.to_owned(), record);
    // Detect directory/file conflicts before removing or writing anything.
    for name in contents.keys() {
        let mut path = destination.clone();
        let mut parts = name.split('/').peekable();
        while let Some(part) = parts.next() {
            path.push(part);
            if let Ok(meta) = fs::symlink_metadata(&path)
                && (meta.file_type().is_symlink() || (parts.peek().is_some() != meta.is_dir()))
            {
                return Err(invalid(format!("destination path conflict: {name}")));
            }
        }
    }
    // All validation precedes mutation. The manifest is written last, so an
    // interrupted write cannot masquerade as a complete new snapshot.
    fs::create_dir_all(&destination)?;
    for name in old.keys().filter(|name| !contents.contains_key(*name)) {
        fs::remove_file(destination.join(name))?;
    }
    for (name, data) in contents
        .iter()
        .filter(|(name, _)| name.as_str() != MANIFEST)
    {
        let path = destination.join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(&path, data)?;
        #[cfg(unix)]
        if name.ends_with(".sh") || name.ends_with(".py") {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
        }
    }
    fs::write(destination.join(MANIFEST), &contents[MANIFEST])?;
    println!(
        "{} {count} files ({bytes} bytes) from {}",
        if preview { "Previewed" } else { "Exported" },
        revision.as_deref().unwrap_or("uncommitted preview")
    );
    Ok(())
}

fn run() -> Result<()> {
    let mut source: Option<OsString> = None;
    let mut destination = None;
    let mut denylist = None;
    let mut preview = false;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let slot = match arg.to_str() {
            Some("--source") => &mut source,
            Some("--destination") => &mut destination,
            Some("--denylist") => &mut denylist,
            Some("--preview") if !preview => {
                preview = true;
                continue;
            }
            Some("--help" | "-h") => {
                println!("vade-export --source DIR --destination DIR --denylist FILE [--preview]");
                return Ok(());
            }
            _ => return Err(invalid("unknown or duplicate argument; see --help")),
        };
        if slot.is_some() {
            return Err(invalid("duplicate argument"));
        }
        *slot = Some(
            args.next()
                .ok_or_else(|| invalid("missing argument value"))?,
        );
    }
    let required = |value: Option<OsString>| {
        value
            .map(PathBuf::from)
            .ok_or_else(|| invalid("--source, --destination and --denylist are required"))
    };
    export(
        &required(source)?,
        &required(destination)?,
        &required(denylist)?,
        preview,
    )
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_boundaries() {
        for name in [
            "../escape",
            "/absolute",
            "a/../b",
            ".git/config",
            "a\\b",
            "a//b",
            "a/./b",
        ] {
            assert!(safe_name(name).is_err());
        }
        let denied = vec!["fixture-host".to_owned()];
        for value in [
            format!("/{}/person/data", "home"),
            ["192", "168", "2", "3"].join("."),
            ["aa", "bb", "cc", "dd", "ee", "ff"].join(":"),
            format!("gh{}{}", "p_", "x".repeat(30)),
            ["-----BEGIN ", "PRIVATE KEY-----"].concat(),
            ["pass", "word=secret-value"].concat(),
            ["OPENAI_API_", "KEY=example-sensitive-value"].concat(),
            ["sk-", "proj-", &"x".repeat(30)].concat(),
            "FIXTURE-HOST".to_owned(),
            "a\0b".to_owned(),
        ] {
            assert!(scan::scan("text.txt", value.as_bytes(), &denied).is_err());
        }
        assert!(scan::scan("text.txt", b"127.0.0.1:0 and 0.0.0.0:8000", &denied).is_ok());
        assert!(scan::scan("text.txt", b"prefix_fixture-host", &denied).is_ok());
        assert!(scan::scan("text.txt", "jörg".as_bytes(), &["JÖRG".to_owned()]).is_err());
        assert_eq!(
            sha256::digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
