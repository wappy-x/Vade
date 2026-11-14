//! Approvals that the person saved with `a` (always) at a web or memory
//! prompt: `allowed.json` in the Vade home. Each web entry is one exact
//! site key (`web::site`: scheme, lower-case host and port) or the web search
//! key; each memory entry is one project folder. A project folder can come
//! from a hostile repository, so the file never lives there. Vade reads it
//! only when it is a private regular file that you own (chmod 600) and the
//! Vade home is outside the workspace, where no model tool can write.
//! `vade allowed` lists the entries and removes one.
use crate::diagnostics::{print_stdout, stderr_line};
use crate::json::{self, Value};
use crate::terminal::sanitize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The file in the Vade home.
pub const FILE_NAME: &str = "allowed.json";
const MAX_FILE_BYTES: u64 = 64 * 1024;
const MAX_ENTRIES: usize = 512;
/// The word before a project folder in `vade allowed` lines.
pub(crate) const MEMORY_WORD: &str = "memory";

pub const HELP: &str = "\
Usage: vade allowed [list | remove ENTRY] [--workspace DIR] [--vade-home DIR]

In vade chat, a web request, a web search and a memory note change ask first.
Answer y for once, s for the rest of the session, or a for always. An a answer
is saved here and applies in every later session until you remove it.

Commands:
  vade allowed list            Show the saved entries, one per line.
  vade allowed remove ENTRY    Remove one entry, written as the list shows it:
                                 a site such as https://docs.example.org, the words
                                 web search, or memory followed by a project
                                 folder. memory alone means the current project.

Options:
  --workspace DIR      The project folder for memory (default: the current folder).
  --vade-home DIR      The Vade home that holds the file. Give it when chat runs
                       with another home than the default one.

File: allowed.json in the Vade home ($VADE_HOME, else ~/.vade), readable and
writable only by you. Chat does not use it when the Vade home is a folder in
the workspace, at any depth and under any name. Only the default ~/.vade is
exempt when the workspace is the user's HOME folder. To never be asked at
all, start chat with --yes. See
harness/docs/agent-tools.md.
";

/// The saved entries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Saved {
    /// Site keys and the web search key.
    pub web: Vec<String>,
    /// Canonical project folders whose memory notes change with no prompt.
    pub memory: Vec<String>,
}

/// Whether `entry` is a key that a web prompt can save: the exact site key
/// that `web::site` gives for an accepted URL, or the web search key. A
/// pattern, a path, an upper-case letter or a default port is not one.
pub fn web_key(entry: &str) -> bool {
    entry == crate::web::SEARCH_TARGET || crate::web::site_key(entry)
}

/// Parse the file: `{"version": 1, "web": [...], "memory": [...]}`.
pub fn parse(text: &str) -> Result<Saved, String> {
    let value = json::parse(text).map_err(|_| "is not valid JSON".to_owned())?;
    let Value::Object(fields) = &value else { return Err("must be a JSON object".into()) };
    for (at, (key, _)) in fields.iter().enumerate() {
        if !matches!(key.as_str(), "version" | "web" | "memory") {
            return Err(format!("has the unknown key {}; only \"version\", \"web\" and \"memory\" are read", json::quote(key)));
        }
        if fields[..at].iter().any(|(old, _)| old == key) {
            return Err(format!("has the key {} twice", json::quote(key)));
        }
    }
    if !matches!(value.get("version"), Some(Value::Number(n)) if n == "1") {
        return Err("needs \"version\": 1".into());
    }
    let list = |key: &str, valid: &dyn Fn(&str) -> bool, what: &str| -> Result<Vec<String>, String> {
        let entries = match value.get(key) {
            None => return Ok(Vec::new()),
            Some(Value::Array(entries)) => entries,
            Some(_) => return Err(format!("\"{key}\" must be an array of strings")),
        };
        if entries.len() > MAX_ENTRIES {
            return Err(format!("\"{key}\" has more than {MAX_ENTRIES} entries"));
        }
        let mut out: Vec<String> = Vec::new();
        for entry in entries {
            let Some(entry) = entry.as_str().filter(|entry| valid(entry)) else {
                return Err(format!("\"{key}\" holds {}, which is not {what}", sanitize(&json::stringify(entry).unwrap_or_default())));
            };
            if !out.iter().any(|old| old == entry) {
                out.push(entry.to_owned());
            }
        }
        Ok(out)
    };
    Ok(Saved {
        web: list("web", &web_key, "an exact site such as https://docs.example.org or \"web search\"")?,
        memory: list("memory", &project_key, "an absolute project folder")?,
    })
}

fn project_key(entry: &str) -> bool {
    let path = Path::new(entry);
    path.is_absolute() && !entry.contains('\0') && !path.components().any(|part| matches!(part, std::path::Component::ParentDir))
}

/// The file text for `saved`.
pub fn render(saved: &Saved) -> String {
    let list = |entries: &[String]| -> String {
        if entries.is_empty() {
            return "[]".into();
        }
        let lines: Vec<String> = entries.iter().map(|entry| format!("    {}", json::quote(entry))).collect();
        format!("[\n{}\n  ]", lines.join(",\n"))
    };
    format!("{{\n  \"version\": 1,\n  \"web\": {},\n  \"memory\": {}\n}}\n", list(&saved.web), list(&saved.memory))
}

/// The entries saved in `home`. A missing home or file holds none.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub fn read(home: &Path) -> Result<Saved, String> {
    if std::fs::symlink_metadata(home).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) {
        return Ok(Saved::default());
    }
    let folder = crate::tools::anchored::Dir::open_root(home).map_err(|_| "the Vade home must be a real folder, not a link".to_owned())?;
    read_in(&folder)
}

/// The entries in the file of an opened Vade home folder.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn read_in(folder: &crate::tools::anchored::Dir) -> Result<Saved, String> {
    use std::io::Read;
    let Some(file) = folder.optional_file(FILE_NAME, true).map_err(|_| "must be a regular file that only you can read and write (chmod 600)".to_owned())? else {
        return Ok(Saved::default());
    };
    let mut text = String::new();
    file.take(MAX_FILE_BYTES + 1).read_to_string(&mut text).map_err(|_| "is not UTF-8 text".to_owned())?;
    if text.len() as u64 > MAX_FILE_BYTES {
        return Err("is larger than 64 KiB".into());
    }
    parse(&text)
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn read(_: &Path) -> Result<Saved, String> {
    Err("saved approvals need macOS or Linux".into())
}

/// The file in `home`, held under its lock for one read and write, so two
/// sessions that save at once keep both entries.
pub struct Store {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    folder: crate::tools::anchored::Dir,
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    _lock: crate::tools::anchored::Lock,
    pub saved: Saved,
}
impl Store {
    /// Lock and read the file. The home is made (mode 0700) when missing.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub fn open(home: &Path) -> Result<Self, String> {
        let path = home.join(FILE_NAME);
        let named = |error: String| format!("{}: {error}", path.display());
        let lock = crate::shell::permissions::lock_file(&path).map_err(named)?;
        let folder = crate::tools::anchored::Dir::open_root(home).map_err(|_| named("the Vade home must be a real folder, not a link".into()))?;
        Ok(Self { saved: read_in(&folder).map_err(named)?, folder, _lock: lock })
    }
    /// Lock and read the file through a checked, opened home: no second
    /// lookup of the home by its path.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub(crate) fn open_in(home: &crate::home_guard::Home) -> Result<Self, String> {
        let named = |error: String| format!("{}: {error}", home.path().join(FILE_NAME).display());
        let folder = home.dir().duplicate().map_err(|error| named(error.to_string()))?;
        let lock = crate::shell::permissions::lock_in(&folder, FILE_NAME).map_err(named)?;
        Ok(Self { saved: read_in(&folder).map_err(named)?, folder, _lock: lock })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    pub fn open(_: &Path) -> Result<Self, String> {
        Err("saved approvals need macOS or Linux".into())
    }
    /// Write `self.saved` (mode 0600, written beside the old file and renamed over it).
    pub fn write(&self) -> Result<(), String> {
        let text = render(&self.saved);
        if text.len() as u64 > MAX_FILE_BYTES || self.saved.web.len() > MAX_ENTRIES || self.saved.memory.len() > MAX_ENTRIES {
            return Err(format!("{FILE_NAME} would hold too much; remove entries with vade allowed remove"));
        }
        parse(&text).map_err(|error| format!("{FILE_NAME} would not load again ({error}); it is unchanged"))?;
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        return self.folder.write_atomic(FILE_NAME, text.as_bytes()).map_err(|_| format!("cannot write {FILE_NAME} in the Vade home"));
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        Err("saved approvals need macOS or Linux".into())
    }
}

/// Add one entry to the file in `home`: a web key, or (with `memory`) a
/// project folder. Returns the path of the file.
pub fn add(home: &Path, entry: &str, memory: bool) -> Result<String, String> {
    let valid = if memory { project_key(entry) } else { web_key(entry) };
    if !valid {
        return Err(format!("{} cannot be saved", sanitize(entry)));
    }
    let mut store = Store::open(home)?;
    store.add(entry, memory)?;
    Ok(home.join(FILE_NAME).display().to_string())
}

/// `add` through a checked, opened home.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn add_in(home: &crate::home_guard::Home, entry: &str, memory: bool) -> Result<(), String> {
    let valid = if memory { project_key(entry) } else { web_key(entry) };
    if !valid {
        return Err(format!("{} cannot be saved", sanitize(entry)));
    }
    Store::open_in(home)?.add(entry, memory)
}

impl Store {
    fn add(&mut self, entry: &str, memory: bool) -> Result<(), String> {
        let list = if memory { &mut self.saved.memory } else { &mut self.saved.web };
        if !list.iter().any(|old| old == entry) {
            list.push(entry.to_owned());
            self.write()?;
        }
        Ok(())
    }
}

/// The words of `vade allowed remove` for an entry.
pub fn remove_words(entry: &str, memory: bool) -> String {
    if memory { format!("vade allowed remove {MEMORY_WORD} {}", shell_word(entry)) } else { format!("vade allowed remove {}", shell_word(entry)) }
}

/// One literal argument for a POSIX shell. Terminal escaping is not quoting.
pub(crate) fn shell_word(word: &str) -> String {
    format!("'{}'", word.replace('\'', "'\\''"))
}

/// The default Vade home: `VADE_HOME`, else `~/.vade`, as for the MCP list.
fn default_home() -> Result<PathBuf, String> {
    let list = crate::mcp::config::user_path().map_err(|_| "cannot find the Vade home; set VADE_HOME or HOME".to_owned())?;
    list.parent().map(Path::to_path_buf).ok_or_else(|| "cannot find the Vade home; set VADE_HOME or HOME".to_owned())
}

/// Split off `--vade-home DIR`.
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

fn fail(message: impl std::fmt::Display) -> ExitCode {
    stderr_line(format_args!("vade: {message}"));
    ExitCode::FAILURE
}

/// `vade allowed [list | remove ENTRY] [--workspace DIR] [--vade-home DIR]`.
pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let mut args: Vec<String> = args.into_iter().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return print_stdout(HELP);
    }
    let usage = |message: String| {
        stderr_line(format_args!("vade: {message}. Run vade allowed --help."));
        ExitCode::from(2)
    };
    let home = match home_argument(&mut args) {
        Ok(Some(home)) => home,
        Ok(None) => match default_home() {
            Ok(home) => home,
            Err(error) => return fail(error),
        },
        Err(error) => return usage(error),
    };
    let root = match crate::mcp::cli::workspace(&mut args) {
        Ok(root) => root,
        Err(error) => return usage(error),
    };
    let path = home.join(FILE_NAME);
    match args.split_first() {
        None => list(&path, &home),
        Some((command, [])) if command == "list" => list(&path, &home),
        Some((command, words)) if command == "remove" && !words.is_empty() => {
            let (entry, memory) = match words {
                [word] if word == MEMORY_WORD => (root.to_string_lossy().into_owned(), true),
                [word, rest @ ..] if word == MEMORY_WORD => (rest.join(" "), true),
                _ => (words.join(" "), false),
            };
            remove(&path, &home, &entry, memory)
        }
        Some((command, _)) if command == "list" || command == "remove" => usage(format!("wrong arguments for allowed {}", sanitize(command))),
        Some((command, _)) => usage(format!("unknown allowed command {}", json::quote(&sanitize(command)))),
    }
}

fn list(path: &Path, home: &Path) -> ExitCode {
    let saved = match read(home) {
        Ok(saved) => saved,
        Err(error) => return fail(format!("{}: {error}", path.display())),
    };
    let mut out = format!("Saved approvals: {}\n", sanitize(&path.to_string_lossy()));
    if saved.web.is_empty() && saved.memory.is_empty() {
        out.push_str("None. Answer a at a web or memory prompt in vade chat to save one.\n");
    }
    for entry in &saved.web {
        out.push_str(&format!("{}\n", sanitize(entry)));
    }
    for entry in &saved.memory {
        out.push_str(&format!("{MEMORY_WORD} {}\n", sanitize(entry)));
    }
    if !saved.web.is_empty() || !saved.memory.is_empty() {
        out.push_str("Remove one with: vade allowed remove ENTRY (as shown above).\n");
    }
    print_stdout(&out)
}

fn remove(path: &Path, home: &Path, entry: &str, memory: bool) -> ExitCode {
    let mut store = match Store::open(home) {
        Ok(store) => store,
        Err(error) => return fail(error),
    };
    let list = if memory { &mut store.saved.memory } else { &mut store.saved.web };
    let before = list.len();
    list.retain(|old| old != entry);
    if list.len() == before {
        let shown = if memory { format!("{MEMORY_WORD} {}", sanitize(entry)) } else { sanitize(entry) };
        return fail(format!("{shown} is not saved in {}; run vade allowed list", path.display()));
    }
    if let Err(error) = store.write() {
        return fail(error);
    }
    let shown = if memory { format!("memory notes of {}", sanitize(entry)) } else { sanitize(entry) };
    print_stdout(&format!("Removed {shown}. Chat asks again from its next session.\n"))
}

#[cfg(test)]
mod tests;
