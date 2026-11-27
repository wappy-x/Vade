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
    /// The exact executable path and bytes checked for a trusted project server.
    pub program: Option<ProgramIdentity>,
    /// Why a project server needs a new trust record.
    pub trust_reason: Option<String>,
    /// The exact entry from the file. This includes the complete argument list.
    pub entry: Value,
}
impl ServerSpec {
    /// The folder the server runs in: `cwd` resolved against the workspace root.
    pub fn directory(&self, root: &Path) -> PathBuf {
        self.cwd.as_deref().map_or_else(|| root.to_path_buf(), |cwd| root.join(cwd))
    }
}

/// What a yes for a project server binds besides the exact config entry: the
/// program (path and bytes) and each script argument (path and bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramIdentity {
    pub path: PathBuf,
    pub target: PathBuf,
    pub sha256: String,
    /// The program file's state when `sha256` was taken; `None` where the
    /// system gives no such values.
    pub stamp: Option<FileStamp>,
    pub scripts: Vec<ScriptIdentity>,
    /// The command loads its code when it starts, and no script is bound.
    pub code_unchecked: bool,
}
impl ProgramIdentity {
    /// Whether every bound value is the same. The stamp is not a bound value.
    pub fn binds_same(&self, other: &Self) -> bool {
        self.path == other.path && self.target == other.target && self.sha256 == other.sha256 && self.scripts == other.scripts
    }
    /// The program file as it was read, for a later check that may skip the read.
    pub fn known(&self) -> Option<KnownProgram<'_>> {
        self.stamp.as_ref().map(|stamp| KnownProgram { target: &self.target, stamp, sha256: &self.sha256 })
    }
    /// The lines of `vade mcp trust` and `vade mcp list`: what is checked,
    /// then what is not.
    pub fn checked_lines(&self) -> Vec<String> {
        let mut lines = vec![format!("program: {}", self.path.display())];
        lines.extend(self.scripts.iter().map(|script| format!("script: {}", script.path.display())));
        if self.code_unchecked {
            lines.push(CODE_NOT_CHECKED.into());
        }
        lines.push(NOT_CHECKED.into());
        lines
    }
}

/// One script argument: its place in `args`, its resolved path and its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptIdentity {
    pub index: usize,
    pub path: PathBuf,
    pub sha256: String,
}

/// What tells an unchanged program file apart without reading it: device,
/// inode, size, and the modification and change times in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileStamp {
    pub device: u64,
    pub inode: u64,
    pub size: u64,
    pub modified_ns: i128,
    pub changed_ns: i128,
}
impl FileStamp {
    #[cfg(unix)]
    fn of(metadata: &std::fs::Metadata) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;
        let ns = |seconds: i64, nanos: i64| i128::from(seconds) * 1_000_000_000 + i128::from(nanos);
        Some(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.size(),
            modified_ns: ns(metadata.mtime(), metadata.mtime_nsec()),
            changed_ns: ns(metadata.ctime(), metadata.ctime_nsec()),
        })
    }
    #[cfg(not(unix))]
    fn of(_: &std::fs::Metadata) -> Option<Self> {
        None
    }
    /// The `stamp` member of a trust record.
    pub fn to_json(&self) -> Value {
        let number = |text: String| Value::Number(text);
        Value::Object(vec![
            ("device".into(), number(self.device.to_string())),
            ("inode".into(), number(self.inode.to_string())),
            ("size".into(), number(self.size.to_string())),
            ("modified_ns".into(), number(self.modified_ns.to_string())),
            ("changed_ns".into(), number(self.changed_ns.to_string())),
        ])
    }
    pub fn from_json(value: &Value) -> Option<Self> {
        fn number<T: std::str::FromStr>(value: &Value, key: &str) -> Option<T> {
            match value.get(key)? { Value::Number(text) => text.parse().ok(), _ => None }
        }
        Some(Self {
            device: number(value, "device")?,
            inode: number(value, "inode")?,
            size: number(value, "size")?,
            modified_ns: number(value, "modified_ns")?,
            changed_ns: number(value, "changed_ns")?,
        })
    }
}

/// A program file that was read before: where it is, its stamp then and its hash.
#[derive(Clone, Copy, Debug)]
pub struct KnownProgram<'a> {
    pub target: &'a Path,
    pub stamp: &'a FileStamp,
    pub sha256: &'a str,
}

/// Why a project entry cannot be trusted or started.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The command is `env`.
    Env,
    /// The command loads its code when it starts, and [`unbound_code`] says refuse.
    UnboundCode,
    Other(String),
}
impl Refusal {
    fn reason(&self) -> &str {
        match self {
            Self::Env => "its command is env, which starts another program that Vade cannot check",
            Self::UnboundCode => "its command loads its code when it starts, so Vade cannot check that code",
            Self::Other(reason) => reason,
        }
    }
    fn advice(&self) -> &'static str {
        match self {
            Self::Env => "Name that program as the command",
            Self::UnboundCode => "Put the server code in a script file and name that file as an argument",
            Self::Other(_) => "Fix the command",
        }
    }
    /// The words after `MCP: project server NAME ` for a server that did not start.
    pub fn at_start(&self, name: &str) -> String {
        let access = if *self == Self::Env { "" } else { " Project MCP programs run with your file and network access." };
        format!("did not start because {}.{access} {}, then run vade mcp trust {name}", self.reason(), self.advice())
    }
    /// The words of `vade mcp trust` and `vade mcp add --project`, which
    /// start nothing. `again` is the command to run after the fix.
    pub fn at_trust(&self, name: &str, again: &str) -> String {
        format!("cannot trust project MCP server {name} because {}. {}, then run {again}", self.reason(), self.advice())
    }
}
fn other(reason: impl Into<String>) -> Refusal {
    Refusal::Other(reason.into())
}

/// The line in place of script lines for a command whose code is not bound.
pub const CODE_NOT_CHECKED: &str = "code: not checked. This command loads its code when it starts. Your yes covers this exact command line and the program file only.";
/// The last line of every list of checked files.
pub const NOT_CHECKED: &str = "Not checked: files and folders that other arguments name, and any file the program loads after it starts.";

/// What Vade does with a project command whose code it cannot bind, such as
/// `npx -y pkg`, `uvx tool`, `python3 -m pkg`, `sh -c "..."` or `node .`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnboundCode {
    /// Trust covers the exact command line and the program file only, and
    /// the trust output says so ([`CODE_NOT_CHECKED`]).
    Allow,
    /// The entry is refused at trust time and at start.
    Refuse,
}
/// The one switch for such commands. To refuse them, return `Refuse` here.
pub fn unbound_code() -> UnboundCode {
    #[cfg(test)]
    if REFUSE_UNBOUND_CODE.get() {
        return UnboundCode::Refuse;
    }
    UnboundCode::Allow
}
#[cfg(test)]
thread_local! {
    /// Lets one test take the other side of [`unbound_code`].
    pub(crate) static REFUSE_UNBOUND_CODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// How many launch files this thread read to the end for a hash.
    pub(crate) static LAUNCH_FILE_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Project program and script files above this size are refused. Hashing
/// uses a fixed buffer, but the cap also bounds trust time for sparse files.
const MAX_TRUSTED_LAUNCH_FILE_BYTES: u64 = 1024 * 1024 * 1024;

/// An argument file with one of these extensions, in any letter case, is a script.
const SCRIPT_EXTENSIONS: [&str; 12] = ["js", "mjs", "cjs", "ts", "py", "rb", "sh", "bash", "zsh", "pl", "php", "lua"];
/// An interpreter's file argument after its first one is data when its name
/// ends in one of these extensions, in any letter case (`.env` counts).
/// Every other one is a script.
const DATA_EXTENSIONS: [&str; 21] = [
    "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf", "csv", "tsv", "txt",
    "log", "db", "sqlite", "sqlite3", "lock", "pid", "sock", "pem", "crt", "env",
];
/// Programs that run their file arguments as code.
const INTERPRETERS: [&str; 14] = ["sh", "bash", "dash", "zsh", "python", "node", "deno", "bun", "ruby", "perl", "php", "lua", "tsx", "ts-node"];
/// Programs that find or fetch the code they run by its name.
const LAUNCHERS: [&str; 8] = ["npx", "npm", "pnpm", "yarn", "uv", "uvx", "pipx", "cargo"];

fn script_extension(path: &Path) -> bool {
    path.extension().and_then(|extension| extension.to_str()).is_some_and(|extension| SCRIPT_EXTENSIONS.iter().any(|known| extension.eq_ignore_ascii_case(known)))
}
fn data_extension(path: &Path) -> bool {
    file_name(path).rsplit_once('.').is_some_and(|(_, extension)| DATA_EXTENSIONS.contains(&extension))
}
/// The file name in lower case. The macOS file system ignores letter case,
/// so `ENV` and `Python3` start `env` and `python3` there.
fn file_name(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}
/// Whether the file name, without a trailing version such as `3`, `3.12` or
/// `22`, is in `names`.
fn named(path: &Path, names: &[&str]) -> bool {
    let name = file_name(path);
    names.contains(&name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.'))
}

/// The SHA-256 of a regular launch file and its stamp. The read is skipped
/// when the file still has the stamp of `known`. The stamp is taken before
/// the read, so a write during the read leaves a stamp that no later check
/// accepts.
fn file_identity(path: &Path, known: Option<KnownProgram<'_>>) -> Result<(Option<FileStamp>, String), String> {
    let file = std::fs::File::open(path).map_err(|_| "a launch file cannot be read".to_owned())?;
    let metadata = file.metadata().map_err(|_| "a launch file is unavailable".to_owned())?;
    if !metadata.is_file() {
        return Err("a launch path does not name a regular file".into());
    }
    if metadata.len() > MAX_TRUSTED_LAUNCH_FILE_BYTES {
        return Err("a launch file exceeds the 1 GiB trust limit".into());
    }
    let stamp = FileStamp::of(&metadata);
    if let (Some(known), Some(stamp)) = (known, &stamp)
        && known.target == path
        && known.stamp == stamp
    {
        return Ok((Some(*stamp), known.sha256.to_owned()));
    }
    #[cfg(test)]
    LAUNCH_FILE_READS.set(LAUNCH_FILE_READS.get() + 1);
    let sha256 = crate::trust::digest_reader(file).map_err(|_| "a launch file cannot be read".to_owned())?;
    Ok((stamp, sha256))
}

/// The working folder, the program path that `Command` would find for this
/// server, and the file that path leads to. A command named `env` is
/// refused: it starts another program, and trust would cover only `env`.
fn resolve_program(spec: &ServerSpec, root: &Path) -> Result<(PathBuf, PathBuf, PathBuf), Refusal> {
    fn executable(path: &Path) -> bool {
        let Ok(metadata) = std::fs::metadata(path) else { return false };
        if !metadata.is_file() { return false; }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            metadata.permissions().mode() & 0o111 != 0
        }
        #[cfg(not(unix))]
        true
    }
    let command = Path::new(&spec.command);
    if file_name(command) == "env" {
        return Err(Refusal::Env);
    }
    let directory = spec.directory(root).canonicalize().map_err(|_| other("its working folder is unavailable"))?;
    let candidate = if command.is_absolute() || command.components().count() > 1 {
        Some(if command.is_absolute() { command.to_path_buf() } else { directory.join(command) })
    } else {
        let path = spec.env.iter().find(|(name, _)| name == "PATH").map(|(_, value)| std::ffi::OsString::from(value)).or_else(|| std::env::var_os("PATH"))
            .ok_or_else(|| other("its command is a bare name, but PATH is not set"))?;
        std::env::split_paths(&path).map(|folder| {
            let folder = if folder.as_os_str().is_empty() { directory.clone() } else if folder.is_absolute() { folder } else { directory.join(folder) };
            folder.join(command)
        }).find(|path| executable(path))
    }.ok_or_else(|| other("its command does not resolve to a program"))?;
    // Keep the selected path: resolving a virtual environment's interpreter
    // symlink before launch would discard that environment.
    let path = candidate;
    let metadata = std::fs::metadata(&path).map_err(|_| other("its program is unavailable"))?;
    if !metadata.is_file() {
        return Err(other("its command does not resolve to a regular program file"));
    }
    if !executable(&path) {
        return Err(other("its program file is not executable"));
    }
    let target = path.canonicalize().map_err(|_| other("its program a launch file is unavailable"))?;
    // A link with another name that leads to a file named `env` starts it too.
    if file_name(&target) == "env" {
        return Err(Refusal::Env);
    }
    Ok((directory, path, target))
}

/// Whether an interpreter invocation runs inline, stdin, module, or preloaded
/// code that cannot be bound by hashing a trailing decoy script file
/// (e.g. `sh -c "evil" decoy.sh`, `python3 -c "..." decoy.py`, `node -e "..." decoy.js`).
fn runs_unbound_code(path: &Path, target: &Path, args: &[String]) -> bool {
    let is = |names: &[&str]| named(path, names) || named(target, names);
    let shell = is(&["sh", "bash", "dash", "zsh"]);
    let python = is(&["python"]);
    let node = is(&["node", "tsx", "ts-node"]);
    let deno_or_bun = is(&["deno", "bun"]);
    let ruby = is(&["ruby"]);
    let perl = is(&["perl"]);
    let lua = is(&["lua"]);
    let php = is(&["php"]);
    let mut i = 0;
    while let Some(arg) = args.get(i) {
        if arg == "--" {
            break;
        }
        if !arg.starts_with('-') || arg == "-" {
            if deno_or_bun && matches!(arg.as_str(), "eval" | "repl") {
                return true;
            }
            break;
        }
        if shell {
            if matches!(arg.as_str(), "--command" | "--rcfile" | "--init-file")
                || arg.starts_with("--command=")
                || arg.starts_with("--rcfile=")
                || arg.starts_with("--init-file=")
            {
                return true;
            }
            if !arg.starts_with("--") {
                let letters = &arg[1..];
                if letters.contains('c') || letters.contains('s') {
                    return true;
                }
                if letters.ends_with('o') || letters.ends_with('O') {
                    i += 2;
                    continue;
                }
            }
        } else if python {
            if !arg.starts_with("--") {
                let letters = &arg[1..];
                if letters.contains('c') || letters.contains('m') {
                    return true;
                }
                if matches!(arg.as_str(), "-W" | "-X" | "-Q") {
                    i += 2;
                    continue;
                }
            }
        } else if node || deno_or_bun {
            if matches!(
                arg.as_str(),
                "-e" | "-p" | "-r" | "--eval" | "--print" | "--require" | "--import" | "--loader"
            ) || arg.starts_with("--eval=")
                || arg.starts_with("--print=")
                || arg.starts_with("--require=")
                || arg.starts_with("--import=")
                || arg.starts_with("--loader=")
            {
                return true;
            }
            if node && !arg.starts_with("--") && arg[1..].chars().any(|c| matches!(c, 'e' | 'p' | 'r')) {
                return true;
            }
        } else if ruby || perl || lua {
            if arg == "--eval" || arg.starts_with("--eval=") {
                return true;
            }
            if !arg.starts_with("--") {
                let letters = &arg[1..];
                if letters.contains('e')
                    || (perl && (letters.contains('E') || letters.contains('M') || letters.contains('m')))
                    || (ruby && letters.contains('r'))
                {
                    return true;
                }
            }
        } else if php {
            if matches!(arg.as_str(), "--run" | "--process-code")
                || arg.starts_with("--run=")
                || arg.starts_with("--process-code=")
                || (!arg.starts_with("--") && arg[1..].chars().any(|c| matches!(c, 'r' | 'B' | 'R' | 'F' | 'E')))
            {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Resolve the same program that `Command` would find for this server, and
/// read what a yes binds besides the exact config entry: the program path and
/// bytes, and each script argument's path and bytes. Other arguments are
/// passed as written; the files and folders they name are not checked, and
/// neither is a file that the program loads after it starts.
///
/// An argument that names a regular file is a script when it starts with
/// `#!`, when its name or the file it resolves to has a script extension, or
/// when the program is an interpreter and this is its first file argument or
/// a later one without a data extension. A decoy file before the real code
/// (`python3 -W ignore server`) so cannot leave the code unbound.
/// A script is read at every call. The program is read again unless it
/// is outside the project and still has the stamp of `known`.
pub fn program_identity(spec: &ServerSpec, root: &Path, known: Option<KnownProgram<'_>>) -> Result<ProgramIdentity, Refusal> {
    let (directory, path, target) = resolve_program(spec, root)?;
    // The stamp shortcut only for a program outside the project. Some volumes
    // keep no reliable change time (a FAT image, for one), so there a write
    // that restores the size and modification time leaves all five values
    // equal. The project, and the model writing inside it, can write only
    // inside the project, so a program there is read at every check, like a
    // script. A system interpreter such as node keeps the fast start.
    let inside = root.canonicalize().map_or(true, |root| target.starts_with(root));
    let known = known.filter(|_| !inside);
    let (stamp, sha256) = file_identity(&target, known).map_err(|reason| other(format!("its program {reason}")))?;
    let interpreter = named(&path, &INTERPRETERS) || named(&target, &INTERPRETERS);
    let mut scripts = Vec::new();
    let mut first_file = true;
    for (index, argument) in spec.args.iter().enumerate() {
        // An absolute argument replaces the folder.
        let candidate = directory.join(argument);
        if !std::fs::metadata(&candidate).is_ok_and(|metadata| metadata.is_file()) {
            continue;
        }
        let first = std::mem::replace(&mut first_file, false);
        let resolved = candidate.canonicalize().unwrap_or_else(|_| candidate.clone());
        let shebang = std::fs::File::open(&resolved).is_ok_and(|mut file| {
            let mut prefix = [0; 2];
            file.read_exact(&mut prefix).is_ok() && prefix == *b"#!"
        });
        // An interpreter's later file argument is data only when both the
        // name given and the file it leads to have a data extension.
        let data = data_extension(&candidate) && data_extension(&resolved);
        if shebang || script_extension(&candidate) || script_extension(&resolved) || (interpreter && (first || !data)) {
            let resolved = resolved.canonicalize().map_err(|_| other(format!("argument {} a launch file is unavailable", index + 1)))?;
            let (_, sha256) = file_identity(&resolved, None).map_err(|reason| other(format!("argument {} {reason}", index + 1)))?;
            scripts.push(ScriptIdentity { index, path: resolved, sha256 });
        }
    }
    let launcher = named(&path, &LAUNCHERS) || named(&target, &LAUNCHERS);
    let code_unchecked = launcher
        || (interpreter && (scripts.is_empty() || runs_unbound_code(&path, &target, &spec.args)));
    if code_unchecked && unbound_code() == UnboundCode::Refuse {
        return Err(Refusal::UnboundCode);
    }
    Ok(ProgramIdentity { path, target, sha256, stamp, scripts, code_unchecked })
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
        program: None,
        trust_reason: None,
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
            ("cwd", Value::String(cwd)) if !cwd.is_empty() => {
                use std::path::Component;
                if !Path::new(cwd).components().all(|part| matches!(part, Component::Normal(_) | Component::CurDir)) {
                    return Err(wrong("a relative folder inside the workspace, without .."));
                }
                spec.cwd = Some(cwd.clone())
            }
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

fn trust_record<'a>(user: &'a Value, root: &Path, name: &str, entry: &Value) -> Option<&'a Value> {
    let root = root.to_string_lossy();
    let entry = json::stringify(entry).ok()?;
    user.get("trusted").and_then(Value::as_array)?.iter().find(|record| {
        record.get("project").and_then(Value::as_str) == Some(&root)
            && record.get("server").and_then(Value::as_str) == Some(name)
            && record.get("entry").and_then(|e| json::stringify(e).ok()).as_deref() == Some(entry.as_str())
    })
}

/// The record of one yes: the project, the server, the exact entry, the
/// program (path, target, bytes and stamp) and each script (path and bytes).
/// Other arguments are not recorded: they are not checked.
pub fn trust_record_value(root: &Path, name: &str, entry: &Value, program: &ProgramIdentity) -> Value {
    let text = |path: &Path| Value::String(path.to_string_lossy().into_owned());
    let scripts = program.scripts.iter().map(|script| Value::Object(vec![
        ("index".into(), Value::Number(script.index.to_string())),
        ("path".into(), text(&script.path)),
        ("kind".into(), Value::String("script".into())),
        ("sha256".into(), Value::String(script.sha256.clone())),
    ])).collect();
    Value::Object(vec![
        ("project".into(), text(root)),
        ("server".into(), Value::String(name.into())),
        ("entry".into(), entry.clone()),
        ("program".into(), text(&program.path)),
        ("target".into(), text(&program.target)),
        ("sha256".into(), Value::String(program.sha256.clone())),
        // While the program file keeps this stamp, a check accepts the hash without reading the file.
        ("stamp".into(), program.stamp.as_ref().map_or(Value::Null, FileStamp::to_json)),
        ("arguments".into(), Value::Array(scripts)),
    ])
}

fn check_project_trust(user: &Value, root: &Path, spec: &mut ServerSpec) {
    let shown = |path: &Path| path.strip_prefix(root).map_or_else(|_| path.display().to_string(), |relative| format!("./{}", relative.display()));
    let Some(record) = trust_record(user, root, &spec.name, &spec.entry) else {
        // A yes for `env` would cover no server program, so say that first.
        if let Err(refusal @ Refusal::Env) = resolve_program(spec, root) {
            spec.trust_reason = Some(refusal.at_start(&spec.name));
            return;
        }
        let changed = user.get("trusted").and_then(Value::as_array).is_some_and(|records| records.iter().any(|record| {
            record.get("project").and_then(Value::as_str) == Some(&root.to_string_lossy())
                && record.get("server").and_then(Value::as_str) == Some(&spec.name)
        }));
        spec.trust_reason = Some(format!("is not trusted because its {}. Project MCP programs run with your file and network access. Check the command and program, then run vade mcp trust {}",
            if changed { "configuration changed" } else { "project configuration has no trust record" }, spec.name));
        return;
    };
    let old_path = record.get("program").and_then(Value::as_str);
    let old_target = record.get("target").and_then(Value::as_str);
    let old_hash = record.get("sha256").and_then(Value::as_str);
    if old_path.is_none() && old_target.is_none() && old_hash.is_none() {
        // A released Vade recorded only the entry. Nothing changed; the
        // record names no program to compare, so no file is read here.
        spec.trust_reason = Some(match resolve_program(spec, root) {
            Err(refusal) => refusal.at_start(&spec.name),
            Ok(_) => format!("did not start because its trust record is from an older Vade and names no program. Check the command, then run vade mcp trust {}", spec.name),
        });
        return;
    }
    let old_stamp = record.get("stamp").and_then(FileStamp::from_json);
    // The recorded hash stands for a program file outside the project while
    // the file keeps the recorded stamp (see `program_identity`). This check
    // never writes the record.
    let known = match (old_target, &old_stamp, old_hash) {
        (Some(target), Some(stamp), Some(sha256)) => Some(KnownProgram { target: Path::new(target), stamp, sha256 }),
        _ => None,
    };
    let current = match program_identity(spec, root, known) {
        Ok(identity) => identity,
        Err(refusal) => {
            spec.trust_reason = Some(refusal.at_start(&spec.name));
            return;
        }
    };
    // Only script entries are compared. An earlier build of this check also
    // recorded the paths of data arguments; those entries are ignored.
    let old_scripts = record.get("arguments").and_then(Value::as_array).and_then(|arguments| arguments.iter()
        .filter(|argument| argument.get("kind").and_then(Value::as_str) != Some("data path"))
        .map(|argument| Some(ScriptIdentity {
            index: match argument.get("index")? { Value::Number(index) => index.parse().ok()?, _ => return None },
            path: PathBuf::from(argument.get("path")?.as_str()?),
            sha256: match argument.get("kind")?.as_str()? {
                "script" => argument.get("sha256")?.as_str()?.to_owned(),
                _ => return None,
            },
        })).collect::<Option<Vec<_>>>());
    let current_path = current.path.to_string_lossy();
    spec.trust_reason = if old_path != Some(&current_path) || old_target != Some(current.target.to_string_lossy().as_ref()) {
        let old = old_target.or(old_path).map(Path::new).map_or_else(|| "an unrecorded path".into(), &shown);
        Some(format!("did not start because its program path changed from {:?} to {:?}. Project MCP programs run with your file and network access. Check the new program, then run vade mcp trust {}",
            old, shown(&current.target), spec.name))
    } else if old_hash != Some(current.sha256.as_str()) {
        Some(format!("did not start because its program contents changed at {:?}. Project MCP programs run with your file and network access. Check the changed program, then run vade mcp trust {}",
            shown(&current.path), spec.name))
    } else if old_scripts.as_ref() != Some(&current.scripts) {
        Some(format!("did not start because a file selected by its command arguments changed. Project MCP programs run with your file and network access. Check the argument files, then run vade mcp trust {}",
            spec.name))
    } else {
        spec.trusted = true;
        None
    };
    spec.program = Some(current);
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
                    if source == Source::Project && spec.enabled {
                        check_project_trust(&user, root, &mut spec);
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
