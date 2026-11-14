//! Trust records for the project files that change what runs with no prompt:
//! the allow list (`.vade/permissions.json`) and the check (`.vade/check`).
//! A project that you clone can ship both. Vade uses such a file only when
//! `trust.json` in the Vade home holds the SHA-256 of its exact bytes for
//! this project folder, as `vade mcp trust` does for a project MCP server.
use crate::diagnostics::{print_stdout, stderr_line};
use crate::json::{self, Value};
use crate::terminal::sanitize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The record file in the Vade home.
pub const STORE: &str = "trust.json";
/// The project files that need a record: the name in `vade trust NAME`, the
/// file in `.vade` and its size limit in bytes.
pub const FILES: [(&str, &str, u64); 3] = [("permissions", "permissions.json", 64 * 1024), ("check", "check", 4096), ("hooks", "hooks.conf", 64 * 1024)];
const MAX_STORE_BYTES: u64 = 1024 * 1024;

pub const HELP: &str = "\
Usage: vade trust [permissions|check|hooks] [--workspace DIR] [--vade-home DIR]

A project can hold three files that run commands with no prompt:
  .vade/permissions.json   Commands that run without approval in vade chat.
  .vade/check              A command that runs after the agent edits files.
  .vade/hooks.conf         Hook commands, which run outside the sandbox. Chat
                           reads this file only with --state-in-project.
Vade uses each file only after you trust its exact contents. A project that
you clone cannot trust its own files. If a file changes, trust it again.

Commands:
  vade trust               Show the files and their state. Change nothing.
  vade trust permissions   Trust .vade/permissions.json as it is now.
  vade trust check         Trust .vade/check as it is now.
  vade trust hooks         Trust .vade/hooks.conf as it is now.

Options:
  --workspace DIR          The project folder (default: the current folder).
  --vade-home DIR          The Vade home that holds the records. Give it when
                           chat runs with another home than the default one.

Records: trust.json in the Vade home ($VADE_HOME, else ~/.vade). An allow-list
rule that you save in chat with p keeps the record current. A Vade home in a
workspace, at any depth and under any name, holds no record: Vade does not
use or write records there. Only the default ~/.vade is exempt when the
workspace is the user's HOME folder.
";

/// One trusted file: the project folder, the file name in `.vade` and the
/// SHA-256 of the trusted bytes.
type Record = (String, String, String);

/// The SHA-256 of `bytes` as lowercase hex (FIPS 180-4).
pub fn digest(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut state: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut message = bytes.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&(bytes.len() as u64).wrapping_mul(8).to_be_bytes());
    for block in message.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (word, quad) in w.iter_mut().zip(block.as_chunks::<4>().0) {
            *word = u32::from_be_bytes(*quad);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for (constant, word) in K.iter().zip(w) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let t1 = h.wrapping_add(s1).wrapping_add((e & f) ^ (!e & g)).wrapping_add(*constant).wrapping_add(word);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let t2 = s0.wrapping_add((a & b) ^ (a & c) ^ (b & c));
            (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (sum, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *sum = sum.wrapping_add(value);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

fn parse(text: &str) -> Result<Vec<Record>, String> {
    let value = json::parse(text).map_err(|_| "is not valid JSON".to_owned())?;
    if !matches!(value.get("version"), Some(Value::Number(n)) if n == "1") {
        return Err("needs \"version\": 1".into());
    }
    let Some(Value::Array(records)) = value.get("trusted") else { return Err("needs a \"trusted\" array".into()) };
    records
        .iter()
        .map(|record| {
            let field = |key| record.get(key).and_then(Value::as_str).map(str::to_owned);
            match (field("project"), field("file"), field("sha256")) {
                (Some(project), Some(file), Some(sha256)) => Ok((project, file, sha256)),
                _ => Err("has a record without project, file and sha256 strings".to_owned()),
            }
        })
        .collect()
}

fn render(records: &[Record]) -> Result<String, String> {
    let text = |text: &str| Value::String(text.into());
    let records = records
        .iter()
        .map(|(project, file, sha256)| Value::Object(vec![("project".into(), text(project)), ("file".into(), text(file)), ("sha256".into(), text(sha256))]))
        .collect();
    let store = Value::Object(vec![("version".into(), Value::Number("1".into())), ("trusted".into(), Value::Array(records))]);
    let mut out = String::new();
    crate::mcp::config::pretty(&store, 0, &mut out).map_err(|error| error.to_string())?;
    out.push('\n');
    if out.len() as u64 > MAX_STORE_BYTES {
        return Err("would be larger than 1 MiB; remove records of projects that you no longer use".into());
    }
    Ok(out)
}

/// The records in an opened Vade home folder. A missing file holds none. The
/// file must be a private regular file that you own.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn read_in(folder: &crate::tools::anchored::Dir) -> Result<Vec<Record>, String> {
    use std::io::Read;
    let Some(file) = folder.optional_file(STORE, true).map_err(|_| "must be a regular file that only you can read and write (chmod 600)".to_owned())? else {
        return Ok(Vec::new());
    };
    let mut text = String::new();
    file.take(MAX_STORE_BYTES + 1).read_to_string(&mut text).map_err(|_| "is not UTF-8 text".to_owned())?;
    if text.len() as u64 > MAX_STORE_BYTES {
        return Err("is larger than 1 MiB".into());
    }
    parse(&text)
}

/// Whether the records in `home` hold these exact bytes for `file` of the
/// project at `root`. No home means no record. An unreadable record file is
/// an error: its reason goes into the note for the person. So is a home that
/// a model tool can write (`home_guard`): a record there could be planted.
pub fn holds(home: Option<&Path>, root: &Path, file: &str, bytes: &[u8]) -> Result<bool, String> {
    let Some(home) = home else { return Ok(false) };
    holds_at(home, &crate::home_guard::Root::open(root)?, file, bytes)
}

/// `holds` for a workspace root held open (by the session). The home is
/// opened once and checked (`home_guard::Home`); the records are read
/// through that handle.
pub fn holds_at(home: &Path, root: &crate::home_guard::Root, file: &str, bytes: &[u8]) -> Result<bool, String> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let Some(opened) = crate::home_guard::Home::open(home, root, false)? else { return Ok(false) };
        let (project, sha256) = (root.path().to_string_lossy(), digest(bytes));
        let records = read_in(opened.dir()).map_err(|error| format!("{}: {error}", home.join(STORE).display()))?;
        Ok(records.iter().any(|(old, name, hash)| *old == project && name == file && *hash == sha256))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    { let _ = (home, root, file, bytes); Err("trust records need macOS or Linux".into()) }
}

/// The record file in `home`, held under its lock for one read and write, so
/// two sessions that save at once keep both records.
pub struct Store {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    folder: crate::tools::anchored::Dir,
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    _lock: crate::tools::anchored::Lock,
    records: Vec<Record>,
}
impl Store {
    /// Lock and read the records in `home` to change those of the project at
    /// `root`. A home that a model tool can write is refused (`home_guard`).
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub fn open(home: &Path, root: &Path) -> Result<Self, String> {
        Self::open_at(home, &crate::home_guard::Root::open(root)?)
    }
    /// `open` for a workspace root held open. The home is opened (made when
    /// missing) and checked once; the lock, the read and the write go
    /// through that handle.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub fn open_at(home: &Path, root: &crate::home_guard::Root) -> Result<Self, String> {
        let named = |error: String| format!("{}: {error}", home.join(STORE).display());
        let opened = match crate::home_guard::Home::open(home, root, true) {
            Ok(Some(opened)) => opened,
            Ok(None) => return Err(named("the Vade home cannot be made".into())),
            Err(reason) => return Err(named(format!("nothing is written: {reason}"))),
        };
        let folder = opened.dir().duplicate().map_err(|error| named(error.to_string()))?;
        let lock = crate::shell::permissions::lock_in(&folder, STORE).map_err(named)?;
        Ok(Self { records: read_in(&folder).map_err(named)?, folder, _lock: lock })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    pub fn open(_: &Path, _: &Path) -> Result<Self, String> {
        Err("trust records need macOS or Linux".into())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    pub fn open_at(_: &Path, _: &crate::home_guard::Root) -> Result<Self, String> {
        Err("trust records need macOS or Linux".into())
    }
    /// The recorded hash for `file` of the project at `root`, if any.
    pub fn recorded(&self, root: &Path, file: &str) -> Option<String> {
        let project = root.to_string_lossy();
        self.records.iter().find(|(old, name, _)| *old == project && name == file).map(|(_, _, hash)| hash.clone())
    }
    /// Replace the record for `file` of the project at `root` with `sha256`,
    /// or drop it, and write the file.
    pub fn set(&mut self, root: &Path, file: &str, sha256: Option<&str>) -> Result<(), String> {
        let project = root.to_string_lossy().into_owned();
        let mut records = self.records.clone();
        records.retain(|(old, name, _)| *old != project || name != file);
        records.extend(sha256.map(|hash| (project, file.to_owned(), hash.to_owned())));
        let text = render(&records).map_err(|error| format!("{STORE}: {error}"))?;
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        self.folder.write_atomic(STORE, text.as_bytes()).map_err(|_| format!("cannot write {STORE} in the Vade home"))?;
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let _ = text;
        self.records = records;
        Ok(())
    }
}

/// The text of one project file in `<root>/.vade`, or `None` when the file or
/// the folder is absent. The folder and the file open without following a
/// link. A file past `cap` bytes or one that is not UTF-8 is an error.
pub fn project_file(root: &Path, name: &str, cap: u64) -> Result<Option<String>, String> {
    use std::io::Read;
    let folder = root.join(".vade");
    if std::fs::symlink_metadata(&folder).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) {
        return Ok(None);
    }
    let Some(file) = crate::shell::permissions::open_file(&folder, name)? else { return Ok(None) };
    let mut text = String::new();
    file.take(cap + 1).read_to_string(&mut text).map_err(|_| "is not UTF-8 text".to_owned())?;
    if text.len() as u64 > cap {
        return Err(format!("is larger than {cap} bytes"));
    }
    Ok(Some(text))
}

/// The startup line for a project file that Vade does not use. `effect` says
/// what does not happen; `reason` is why a record could not be checked. `home`
/// is the Vade home of this chat: the command in the line names it when
/// `vade trust` would write its record in another home.
pub fn note(name: &str, root: &Path, effect: &str, reason: Option<&str>, home: Option<&Path>) -> String {
    let file = FILES.iter().find(|(known, ..)| *known == name).map_or(name, |(_, file, _)| file);
    let path = root.join(".vade").join(file);
    match reason {
        Some(reason) => format!("{} not used ({reason}), so {effect}.", path.display()),
        None => format!("{} is not trusted, so {effect}. Read the file, then run: vade trust {name}{}", path.display(), home_words(home)),
    }
}

/// The words that make `vade trust` write to `home`: none when it is the
/// default home. A path with other characters than letters, digits and
/// `/._-` goes in single quotes for the shell.
fn home_words(chat: Option<&Path>) -> String {
    let Some(chat) = chat.filter(|chat| default_home().ok().as_deref() != Some(chat)) else { return String::new() };
    let text = chat.to_string_lossy();
    if text.bytes().all(|b| b.is_ascii_alphanumeric() || b"/._-".contains(&b)) {
        format!(" --vade-home {text}")
    } else {
        format!(" --vade-home '{}'", text.replace('\'', "'\\''"))
    }
}

/// The default Vade home of the records: `VADE_HOME`, else `~/.vade`, as for the MCP list.
fn default_home() -> Result<PathBuf, String> {
    let list = crate::mcp::config::user_path().map_err(|_| "cannot find the Vade home; set VADE_HOME or HOME".to_owned())?;
    list.parent().map(Path::to_path_buf).ok_or_else(|| "cannot find the Vade home; set VADE_HOME or HOME".to_owned())
}

/// Split off `--vade-home DIR`: the home of a chat that does not use the default one.
fn home_argument(args: &mut Vec<String>) -> Result<Option<PathBuf>, String> {
    let mut home = None;
    while let Some(at) = args.iter().position(|arg| arg == "--vade-home") {
        if home.is_some() || at + 1 >= args.len() {
            return Err("give --vade-home once, with a folder".into());
        }
        let folder = PathBuf::from(args.remove(at + 1));
        args.remove(at);
        if !folder.is_absolute() || folder.components().any(|part| matches!(part, std::path::Component::ParentDir)) {
            return Err("--vade-home must be an absolute path without '..'".into());
        }
        home = Some(folder);
    }
    Ok(home)
}

/// What one project file holds, for the person who decides to trust it.
fn describe(name: &str, text: &str) -> Result<String, String> {
    if name == "check" {
        let command = text.lines().map(str::trim).find(|line| !line.is_empty() && !line.starts_with('#'));
        return match command {
            // Chat refuses such a line at its start, so it is not trusted here.
            Some(command) if command.len() > 4096 || command.chars().any(char::is_control) => {
                Err("its command must be one line of at most 4096 bytes with no control character".into())
            }
            Some(command) => Ok(format!("  runs after edits: /bin/sh -c {}\n", sanitize(command))),
            None => Ok("  holds no command\n".into()),
        };
    }
    if name == "hooks" {
        let hooks = crate::hooks::Hooks::summary(text).map_err(|error| error.to_string())?;
        let mut out = format!("  {} hook(s) that run outside the sandbox, with --state-in-project:\n", hooks.len());
        for hook in &hooks {
            out.push_str(&format!("    {}\n", sanitize(hook)));
        }
        return Ok(out);
    }
    let rules = crate::shell::permissions::parse(text)?;
    let mut out = format!("  {} rule(s) that run without approval:\n", rules.len());
    for rule in &rules {
        out.push_str(&format!("    {}\n", sanitize(&crate::shell::permissions::rule_text(rule))));
    }
    Ok(out)
}

fn fail(message: impl std::fmt::Display) -> ExitCode {
    stderr_line(format_args!("vade: {message}"));
    ExitCode::FAILURE
}

/// `vade trust [permissions|check|hooks] [--workspace DIR] [--vade-home DIR]`.
pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let mut args: Vec<String> = args.into_iter().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return print_stdout(HELP);
    }
    let usage = |message: String| {
        stderr_line(format_args!("vade: {message}. Run vade trust --help."));
        ExitCode::from(2)
    };
    let root = match crate::mcp::cli::workspace(&mut args) {
        Ok(root) => root,
        Err(error) => return usage(error),
    };
    // Chat with `--vade-home DIR` reads the records of DIR, so the person can name it here.
    let home = match home_argument(&mut args) {
        Ok(Some(home)) => home,
        Ok(None) => match default_home() {
            Ok(home) => home,
            Err(error) => return fail(error),
        },
        Err(error) => return usage(error),
    };
    let wanted = match args.as_slice() {
        [] => None,
        [name] if FILES.iter().any(|(known, ..)| known == name) => Some(name.as_str()),
        _ => return usage("name one file to trust: permissions, check or hooks".into()),
    };
    let mut out = format!("Project: {}\nRecords: {}\n", sanitize(&root.to_string_lossy()), sanitize(&home.join(STORE).to_string_lossy()));
    for (name, file, cap) in FILES {
        if wanted.is_some_and(|wanted| wanted != name) {
            continue;
        }
        let text = match project_file(&root, file, cap) {
            Ok(Some(text)) => text,
            Ok(None) if wanted.is_some() => return fail(format!("this project has no .vade/{file}")),
            Ok(None) => {
                out.push_str(&format!(".vade/{file}: absent\n"));
                continue;
            }
            Err(error) => return fail(format!(".vade/{file} {error}")),
        };
        let described = match describe(name, &text) {
            Ok(described) => described,
            Err(error) => return fail(format!(".vade/{file} does not load ({error}); fix or remove it first")),
        };
        if wanted.is_some() {
            let saved = Store::open(&home, &root).and_then(|mut store| store.set(&root, file, Some(&digest(text.as_bytes()))));
            if let Err(error) = saved {
                return fail(error);
            }
            out.push_str(&format!("Trusted .vade/{file}:\n{described}If the file changes, run vade trust {name} again.\n"));
            continue;
        }
        let state = match holds(Some(&home), &root, file, text.as_bytes()) {
            Ok(true) => "trusted".to_owned(),
            Ok(false) => format!("not trusted; after you read it, run vade trust {name}"),
            Err(error) => return fail(error),
        };
        out.push_str(&format!(".vade/{file}: {state}\n{described}"));
    }
    print_stdout(&out)
}

#[cfg(test)]
mod tests;
