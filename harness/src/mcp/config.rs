//! MCP server lists: `<project>/.vade/mcp.json` and the user's `mcp.json` in the Vade home.
use crate::json::{self, Value};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Config files above this size are refused, not truncated.
pub const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
pub const MAX_TIMEOUT_MS: u64 = 600_000;
pub const MAX_SERVER_NAME: usize = 32;

/// Which file a server entry came from. A project entry overrides a user
/// entry with the same name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    User,
    Project,
}
impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
        }
    }
}

/// One stdio server entry, validated.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: Option<String>,
    pub enabled: bool,
    pub timeout: Duration,
    pub source: Source,
    /// User entries are always trusted; project entries only after `vade mcp trust`.
    pub trusted: bool,
    /// The exact entry from the file; trust records compare against it.
    pub entry: Value,
}
impl ServerSpec {
    /// The folder the server runs in: `cwd` resolved against the workspace root.
    pub fn directory(&self, root: &Path) -> PathBuf {
        self.cwd.as_deref().map_or_else(|| root.to_path_buf(), |cwd| root.join(cwd))
    }
}

/// Every configured server for one workspace, plus one line per skipped entry.
#[derive(Debug, Default)]
pub struct Loaded {
    pub servers: Vec<ServerSpec>,
    pub notes: Vec<String>,
}

pub fn valid_name(name: &str) -> bool {
    (1..=MAX_SERVER_NAME).contains(&name.len())
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        && name.as_bytes()[0].is_ascii_alphanumeric()
}

/// `$VADE_HOME/mcp.json`, else `$HOME/.vade/mcp.json`.
pub fn user_path() -> io::Result<PathBuf> {
    user_path_in(None)
}

/// `mcp.json` in `home` when chat got a Vade home (`--vade-home`), else the
/// [`user_path`] rule. The launcher passes the home and drops `VADE_HOME`.
pub fn user_path_in(home: Option<&Path>) -> io::Result<PathBuf> {
    let home = home.map(|home| home.as_os_str().to_owned()).or_else(|| std::env::var_os("VADE_HOME"));
    let home = match (home.filter(|v| !v.is_empty()), std::env::var_os("HOME").filter(|v| !v.is_empty())) {
        (Some(home), _) => PathBuf::from(home),
        (None, Some(home)) => PathBuf::from(home).join(".vade"),
        (None, None) => return Err(io::Error::other("cannot find the Vade home for the MCP list; set VADE_HOME or HOME")),
    };
    if !home.is_absolute() || home.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err(io::Error::other("the Vade home for the MCP list must be an absolute path without '..'"));
    }
    Ok(home.join("mcp.json"))
}
pub fn project_path(root: &Path) -> PathBuf {
    root.join(".vade").join("mcp.json")
}

/// Refuse a config file or its folder that is a symlink, so a project cannot
/// point Vade at a file or folder outside it.
fn refuse_links(path: &Path) -> io::Result<()> {
    for part in [path.parent(), Some(path)].into_iter().flatten() {
        if std::fs::symlink_metadata(part).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err(io::Error::other(format!("{} is a symlink; use a regular file and folder", part.display())));
        }
    }
    Ok(())
}

/// Read one config file. A missing file is an empty object.
pub fn read_file(path: &Path) -> io::Result<Value> {
    refuse_links(path)?;
    let mut open = std::fs::OpenOptions::new();
    open.read(true);
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        #[cfg(target_os = "macos")]
        const NONBLOCK: i32 = 0x4;
        #[cfg(target_os = "linux")]
        const NONBLOCK: i32 = 0x800;
        // Never block on a FIFO put in place of the file.
        open.custom_flags(NONBLOCK);
    }
    let file = match open.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Value::Object(Vec::new())),
        Err(error) => return Err(io::Error::new(error.kind(), format!("cannot read {}", path.display()))),
    };
    if !file.metadata().is_ok_and(|meta| meta.is_file()) {
        return Err(io::Error::other(format!("{} must be a regular file", path.display())));
    }
    let mut text = String::new();
    file.take(MAX_CONFIG_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|_| io::Error::other(format!("{} is not UTF-8 text", path.display())))?;
    if text.len() as u64 > MAX_CONFIG_BYTES {
        return Err(io::Error::other(format!("{} is larger than 1 MiB", path.display())));
    }
    let value = json::parse(&text).map_err(|e| io::Error::other(format!("{} is not valid JSON ({e})", path.display())))?;
    if !matches!(value, Value::Object(_)) {
        return Err(io::Error::other(format!("{} must hold a JSON object", path.display())));
    }
    Ok(value)
}

/// The server map of one file. `mcpServers` is accepted so a list copied from
/// another MCP client works unchanged; a file may not hold both keys.
pub fn servers(file: &Value) -> Result<&[(String, Value)], String> {
    match (file.get("servers"), file.get("mcpServers")) {
        (Some(_), Some(_)) => Err("use servers or mcpServers, not both".into()),
        (Some(Value::Object(map)), None) | (None, Some(Value::Object(map))) => Ok(map),
        (None, None) => Ok(&[]),
        _ => Err("servers must be a JSON object keyed by server name".into()),
    }
}

/// Validate one entry. `Err` is a one-line reason the entry is skipped.
pub fn parse_entry(name: &str, entry: &Value, source: Source) -> Result<ServerSpec, String> {
    if !valid_name(name) {
        return Err(format!("server name {name:?} must be 1 to {MAX_SERVER_NAME} ASCII letters, digits or hyphens, starting with a letter or digit"));
    }
    let Value::Object(fields) = entry else { return Err(format!("server {name} must be a JSON object")) };
    let mut spec = ServerSpec {
        name: name.into(),
        command: String::new(),
        args: Vec::new(),
        env: Vec::new(),
        cwd: None,
        enabled: true,
        timeout: DEFAULT_TIMEOUT,
        source,
        trusted: source == Source::User,
        entry: entry.clone(),
    };
    for (key, value) in fields {
        let wrong = |what: &str| format!("server {name}: {key} must be {what}");
        match (key.as_str(), value) {
            ("command", Value::String(command)) if !command.is_empty() => spec.command = command.clone(),
            ("command", _) => return Err(wrong("a nonempty string")),
            ("args", Value::Array(args)) => {
                spec.args = args.iter().map(|arg| arg.as_str().map(str::to_owned)).collect::<Option<_>>().ok_or_else(|| wrong("an array of strings"))?
            }
            ("env", Value::Object(env)) => {
                for (key, value) in env {
                    let value = value.as_str().ok_or_else(|| wrong("an object of strings"))?;
                    if key.is_empty() || key.contains('=') || key.contains('\0') || value.contains('\0') {
                        return Err(format!("server {name}: env has an invalid variable name or value"));
                    }
                    if spec.env.iter().any(|(old, _)| old == key) {
                        return Err(format!("server {name}: env sets {key} more than once"));
                    }
                    spec.env.push((key.clone(), value.into()));
                }
            }
            ("cwd", Value::String(cwd)) if !cwd.is_empty() => spec.cwd = Some(cwd.clone()),
            ("enabled", Value::Bool(enabled)) => spec.enabled = *enabled,
            ("timeout_ms", Value::Number(n)) => match n.parse::<u64>() {
                Ok(ms @ 1..=MAX_TIMEOUT_MS) => spec.timeout = Duration::from_millis(ms),
                _ => return Err(wrong("an integer from 1 to 600000")),
            },
            ("type", Value::String(kind)) if kind == "stdio" => {}
            ("type", _) | ("url", _) | ("headers", _) => {
                return Err(format!("server {name}: only stdio servers (command and args) are supported"));
            }
            ("args" | "env" | "cwd" | "enabled" | "timeout_ms", _) => return Err(wrong(match key.as_str() {
                "args" => "an array of strings",
                "env" => "an object of strings",
                "cwd" => "a nonempty string",
                "enabled" => "true or false",
                _ => "an integer",
            })),
            // Other clients add their own keys (for example "description");
            // they do not change how the server runs.
            _ => {}
        }
    }
    if spec.command.is_empty() {
        return Err(format!("server {name}: command is required"));
    }
    Ok(spec)
}

/// Whether the user file records a yes for this exact project entry.
pub fn is_trusted(user: &Value, root: &Path, name: &str, entry: &Value) -> bool {
    let root = root.to_string_lossy();
    let Ok(entry) = json::stringify(entry) else { return false };
    user.get("trusted").and_then(Value::as_array).is_some_and(|records| {
        records.iter().any(|record| {
            record.get("project").and_then(Value::as_str) == Some(&root)
                && record.get("server").and_then(Value::as_str) == Some(name)
                && record.get("entry").and_then(|e| json::stringify(e).ok()).as_deref() == Some(entry.as_str())
        })
    })
}

/// Merge the user and project files for `root`; `home` is the chat's Vade
/// home, if any. Bad files and entries become notes; they never stop a chat
/// from starting.
pub fn load(root: &Path, home: Option<&Path>) -> Loaded {
    let mut loaded = Loaded::default();
    let user = match user_path_in(home).and_then(|path| read_file(&path)) {
        Ok(user) => user,
        Err(error) => {
            loaded.notes.push(format!("MCP: user list skipped: {error}"));
            Value::Object(Vec::new())
        }
    };
    let project = read_file(&project_path(root)).unwrap_or_else(|error| {
        loaded.notes.push(format!("MCP: project list skipped: {error}"));
        Value::Object(Vec::new())
    });
    for (file, source) in [(&user, Source::User), (&project, Source::Project)] {
        let map = match servers(file) {
            Ok(map) => map,
            Err(error) => {
                loaded.notes.push(format!("MCP: {} list skipped: {error}", source.label()));
                continue;
            }
        };
        for (name, entry) in map {
            match parse_entry(name, entry, source) {
                Ok(mut spec) => {
                    if source == Source::Project {
                        spec.trusted = is_trusted(&user, root, name, entry);
                    }
                    loaded.servers.retain(|old| old.name != spec.name);
                    loaded.servers.push(spec);
                }
                Err(error) => loaded.notes.push(format!("MCP: skipped: {error}")),
            }
        }
    }
    loaded
}

/// Replace one top-level member, keeping the other members and their order.
pub fn set_member(file: &mut Value, key: &str, value: Value) {
    if let Value::Object(fields) = file {
        match fields.iter_mut().find(|(k, _)| k == key) {
            Some((_, old)) => *old = value,
            None => fields.push((key.into(), value)),
        }
    }
}

/// Write a config file atomically, readable only by its owner (it may hold
/// tokens in `env`).
pub fn write_file(path: &Path, file: &Value) -> io::Result<()> {
    use std::io::Write;
    refuse_links(path)?;
    let dir = path.parent().ok_or_else(|| io::Error::other("config path has no folder"))?;
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)?;
    let mut text = String::new();
    pretty(file, 0, &mut text)?;
    text.push('\n');
    // A file this command cannot read back is never written.
    if text.len() as u64 > MAX_CONFIG_BYTES {
        return Err(io::Error::other("the file would be larger than 1 MiB; remove servers or trust records first"));
    }
    // The parts were written one at a time; the whole file must still pass
    // the nesting and value limits of the next read.
    if json::parse(&text).is_err() {
        return Err(io::Error::other("the file would exceed the JSON nesting or value limits; simplify the file first"));
    }
    let temp = dir.join(format!(".mcp.json.{}.tmp", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = options.open(&temp).and_then(|mut out| {
        out.write_all(text.as_bytes())?;
        out.sync_all()
    });
    match result.and_then(|()| std::fs::rename(&temp, path)) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&temp);
            Err(error)
        }
    }
}

/// Two-space indented JSON, so people can edit the file by hand.
pub fn pretty(value: &Value, depth: usize, out: &mut String) -> io::Result<()> {
    let indent = |out: &mut String, depth: usize| out.extend(std::iter::repeat_n("  ", depth));
    match value {
        Value::Array(items) if !items.is_empty() && items.iter().any(|v| matches!(v, Value::Array(_) | Value::Object(_))) => {
            out.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                indent(out, depth + 1);
                pretty(item, depth + 1, out)?;
                out.push_str(if index + 1 < items.len() { ",\n" } else { "\n" });
            }
            indent(out, depth);
            out.push(']');
        }
        Value::Object(fields) if !fields.is_empty() => {
            out.push_str("{\n");
            for (index, (key, item)) in fields.iter().enumerate() {
                indent(out, depth + 1);
                out.push_str(&json::quote(key));
                out.push_str(": ");
                pretty(item, depth + 1, out)?;
                out.push_str(if index + 1 < fields.len() { ",\n" } else { "\n" });
            }
            indent(out, depth);
            out.push('}');
        }
        _ => out.push_str(&json::stringify(value)?),
    }
    Ok(())
}
