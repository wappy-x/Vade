//! Allow lists for commands that need approval: this session (memory only),
//! the project (`.vade/permissions.json`) and the user
//! (`<Vade home>/permissions.json`). A rule is one exact argv of plain
//! literal words (see `policy::literal`); it matches only that same argv.
//! A rule written in a file can also hold `<path>` slots after the program
//! word; each one matches one path inside the workspace
//! (`policy::workspace_path`). Blocked commands, findings that cannot go
//! on a list and words that name private files never match a rule.
use super::policy;
use super::safety::Verdict;
use crate::json::{self, Value};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::ThreadId;

/// One allow-list rule: the exact argv it approves, where a `<path>` word
/// stands for one workspace path.
pub type Rule = Vec<String>;

pub const FILE_NAME: &str = "permissions.json";
const MAX_FILE_BYTES: u64 = 64 * 1024;
const MAX_RULES: usize = 512;
const MAX_RULE_WORDS: usize = 128;

/// How far an approval reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Once,
    Session,
    Project,
    User,
    /// A web target or this project's memory notes, saved in the Vade home
    /// (`crate::saved_approvals`).
    Always,
}
impl Scope {
    pub fn label(self) -> &'static str {
        match self {
            Self::Once => "once",
            Self::Session => "session",
            Self::Project => "project",
            Self::User => "user",
            Self::Always => "always",
        }
    }
}

/// A rule as people write it: words joined by spaces.
pub fn rule_text(rule: &Rule) -> String {
    rule.join(" ")
}

fn parse_rule(value: &Value) -> Result<Rule, String> {
    let rule: Rule = match value {
        Value::String(text) => text.split_ascii_whitespace().map(str::to_owned).collect(),
        Value::Array(words) => words
            .iter()
            .map(|word| word.as_str().map(str::to_owned).ok_or("a rule array must hold only strings"))
            .collect::<Result<_, _>>()?,
        _ => return Err("each rule must be a string or an array of strings".into()),
    };
    if rule.is_empty() || rule.len() > MAX_RULE_WORDS || rule.iter().any(|w| w.is_empty() || w.len() > 4096 || w.contains('\0')) {
        return Err("a rule needs 1 to 128 nonempty words".into());
    }
    if rule[0] == policy::PATH_SLOT {
        return Err(format!("a rule starts with the program name; {} can only stand for an argument", policy::PATH_SLOT));
    }
    if let Some(word) = rule.iter().find(|word| *word != policy::PATH_SLOT && !policy::literal(word)) {
        return Err(format!("a rule is one exact command of plain words (letters, digits and - _ . , / : = + @) and {} slots; {} is not one", policy::PATH_SLOT, json::quote(word)));
    }
    if let Some(word) = rule.iter().find(|word| policy::private_word(word)) {
        return Err(format!("{} names a hidden, credential or permissions file; such a command always asks", json::quote(word)));
    }
    Ok(rule)
}

/// Parse a permissions file: `{"version": 1, "allow": [rule, ...]}`.
pub fn parse(text: &str) -> Result<Vec<Rule>, String> {
    let value = json::parse(text).map_err(|_| "not valid JSON".to_owned())?;
    let Value::Object(fields) = &value else {
        return Err("expected a JSON object with \"version\" and \"allow\"".into());
    };
    if let Some((key, _)) = fields.iter().find(|(key, _)| !matches!(key.as_str(), "version" | "allow")) {
        return Err(format!("unknown key \"{key}\"; only \"version\" and \"allow\" are read"));
    }
    if !matches!(value.get("version"), Some(Value::Number(n)) if n == "1") {
        return Err("\"version\" must be 1".into());
    }
    let rules = match value.get("allow") {
        Some(Value::Array(rules)) => rules.iter().map(parse_rule).collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
        _ => return Err("\"allow\" must be an array".into()),
    };
    if rules.len() > MAX_RULES {
        return Err(format!("at most {MAX_RULES} rules"));
    }
    Ok(rules)
}

/// The file text for `rules`. Words with spaces use the array form.
pub fn render(rules: &[Rule]) -> String {
    let mut text = String::from("{\n  \"version\": 1,\n  \"allow\": [");
    for (index, rule) in rules.iter().enumerate() {
        text.push_str(if index == 0 { "\n    " } else { ",\n    " });
        if rule.iter().any(|w| w.is_empty() || w.chars().any(|c| c.is_whitespace())) {
            let words: Vec<String> = rule.iter().map(|w| json::quote(w)).collect();
            text.push_str(&format!("[{}]", words.join(", ")));
        } else {
            text.push_str(&json::quote(&rule_text(rule)));
        }
    }
    text.push_str(if rules.is_empty() { "]\n}\n" } else { "\n  ]\n}\n" });
    text
}

/// Read a permissions file. A missing file or folder is an empty list. The
/// folder and the file open without following links, and the file without
/// blocking on a FIFO, so a link cannot point the list somewhere else.
pub fn load(path: &Path) -> Result<Vec<Rule>, String> {
    read_text(path)?.map_or(Ok(Vec::new()), |text| parse(&text))
}

/// The text of a permissions file, or `None` when the file or its folder is
/// absent. A trust record names the hash of exactly these bytes.
fn read_text(path: &Path) -> Result<Option<String>, String> {
    let (folder, name) = split(path)?;
    if std::fs::symlink_metadata(folder).is_err_and(|error| error.kind() == io::ErrorKind::NotFound) {
        return Ok(None);
    }
    let Some(file) = open_file(folder, name)? else { return Ok(None) };
    let mut text = String::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|_| "cannot read as UTF-8".to_owned())?;
    if text.len() as u64 > MAX_FILE_BYTES {
        return Err("is larger than 64 KiB".into());
    }
    Ok(Some(text))
}

/// The project list of a workspace. A cloned project can ship this file, so
/// its rules apply only when a trust record in the Vade home names its exact
/// bytes (`crate::trust`). The second value is the line for the person when
/// the file is there and its rules do not apply.
fn load_project(path: &Path, trust: Option<&Trust>) -> Result<(Vec<Rule>, Option<String>), String> {
    let Some(text) = read_text(path)? else { return Ok((Vec::new(), None)) };
    let Some(trust) = trust else { return Ok((parse(&text)?, None)) };
    let effect = "its rules do not apply and each command asks";
    let held = match (trust.home.as_deref(), &trust.dir) {
        (None, _) => Ok(false),
        (Some(home), Ok(root)) => crate::trust::holds_at(home, root, FILE_NAME, text.as_bytes()),
        (Some(_), Err(error)) => Err(error.clone()),
    };
    match held {
        Ok(true) => Ok((parse(&text)?, None)),
        Ok(false) => Ok((Vec::new(), Some(crate::trust::note("permissions", &trust.root, effect, None, trust.home.as_deref())))),
        Err(reason) => Ok((Vec::new(), Some(crate::trust::note("permissions", &trust.root, effect, Some(&reason), trust.home.as_deref())))),
    }
}

/// Replace a permissions file with `rules`. The new text must load again
/// within the size limit; it is written beside the old file and renamed
/// over it, inside a folder opened without following links.
pub fn save(path: &Path, rules: &[Rule]) -> Result<(), String> {
    let text = render(rules);
    if text.len() as u64 > MAX_FILE_BYTES {
        return Err("would be larger than 64 KiB; the file is unchanged".into());
    }
    parse(&text).map_err(|error| format!("would not load again ({error}); the file is unchanged"))?;
    let (folder, name) = split(path)?;
    write_file(folder, name, text.as_bytes())
}

fn split(path: &Path) -> Result<(&Path, &str), String> {
    let folder = path.parent().filter(|p| !p.as_os_str().is_empty()).ok_or("has no folder")?;
    let name = path.file_name().and_then(|n| n.to_str()).ok_or("has no file name")?;
    Ok((folder, name))
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn open_file(folder: &Path, name: &str) -> Result<Option<std::fs::File>, String> {
    let folder = crate::tools::anchored::Dir::open_root(folder).map_err(|_| "its folder must be a real folder, not a link".to_owned())?;
    folder.regular_file(name).map_err(|error| error.to_string())
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn write_file(folder: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    use std::os::unix::fs::DirBuilderExt;
    // Private like the project state that shares `.vade`.
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(folder).map_err(|error| format!("cannot create its folder: {error}"))?;
    let folder = crate::tools::anchored::Dir::open_root(folder).map_err(|_| "its folder must be a real folder, not a link".to_owned())?;
    if let Some(file) = folder.regular_file(name).map_err(|error| error.to_string())? {
        drop(file);
    }
    folder.write_atomic(name, bytes).map_err(|_| "cannot write the file".to_owned())
}

/// Hold the lock of a permissions file (`.<name>.lock` beside it) for one
/// read and save, so two sessions that save at once keep both rules.
/// Another session holds it for one save, so a busy lock is tried again
/// for up to two seconds; then nothing is saved.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn lock_file(path: &Path) -> Result<crate::tools::anchored::Lock, String> {
    use std::os::unix::fs::DirBuilderExt;
    let (folder, name) = split(path)?;
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(folder).map_err(|error| format!("cannot create its folder: {error}"))?;
    let folder = crate::tools::anchored::Dir::open_root(folder).map_err(|_| "its folder must be a real folder, not a link".to_owned())?;
    lock_in(&folder, name)
}

/// `lock_file` for the file `name` in a folder that is already open.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn lock_in(folder: &crate::tools::anchored::Dir, name: &str) -> Result<crate::tools::anchored::Lock, String> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        match folder.lock(&format!(".{name}.lock")) {
            Ok(lock) => return Ok(lock),
            Err(_) if std::time::Instant::now() < deadline => std::thread::sleep(std::time::Duration::from_millis(5)),
            Err(_) => return Err("another session is saving it, or its lock does not open; the file is unchanged".into()),
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) fn lock_file(path: &Path) -> Result<std::fs::File, String> {
    let (folder, name) = split(path)?;
    std::fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let file = std::fs::OpenOptions::new().write(true).create(true).open(folder.join(format!(".{name}.lock"))).map_err(|error| error.to_string())?;
    file.lock().map_err(|error| error.to_string())?;
    Ok(file)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) fn open_file(folder: &Path, name: &str) -> Result<Option<std::fs::File>, String> {
    let path = folder.join(name);
    match std::fs::symlink_metadata(&path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Ok(metadata) if metadata.is_file() => std::fs::File::open(&path).map(Some).map_err(|error| error.to_string()),
        _ => Err("must be a regular file, not a link or folder".into()),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn write_file(folder: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    std::fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let temporary = folder.join(format!(".{name}.{}.new", std::process::id()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temporary, folder.join(name))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map_err(|error| error.to_string())
}

struct List {
    path: Option<PathBuf>,
    rules: Vec<Rule>,
}

/// Where the trust record of the project list is: the workspace root that
/// the record names, and the Vade home that holds the records, if any.
struct Trust {
    root: PathBuf,
    home: Option<PathBuf>,
    /// The workspace root, held open for the session (`home_guard::Root`),
    /// or why it could not be opened.
    dir: Result<crate::home_guard::Root, String>,
}
impl Trust {
    /// Open the Vade home `home` once and check it against the session's
    /// root handle. Every read and write of a home file goes through it.
    fn open_home(&self, home: &Path, create: bool) -> Result<Option<crate::home_guard::Home>, String> {
        crate::home_guard::Home::open(home, self.dir.as_ref().map_err(Clone::clone)?, create)
    }
    /// The user list at `path` through a checked home, or why it is not used.
    fn user_home(&self, path: &Path, create: bool) -> Result<Option<crate::home_guard::Home>, String> {
        let home = path.parent().ok_or("has no folder")?;
        self.open_home(home, create).map_err(|reason| format!("{} is not used: {reason}. The commands it lists ask each time.", path.display()))
    }
}

/// The start line when the user list at `path` is not used: its home fails
/// the guard, or the file is not a regular file of yours with one link.
fn user_note(path: &Path, trust: &Trust) -> Option<String> {
    let home = match trust.user_home(path, false) {
        Ok(Some(home)) => home,
        Ok(None) => return None,
        Err(note) => return Some(note),
    };
    let name = path.file_name()?.to_str()?;
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    if home.dir().optional_file(name, false).is_err() {
        return Some(format!("{} is not used: {USER_FILE_RULE}. The commands it lists ask each time.", path.display()));
    }
    let _ = (home, name);
    None
}

/// The rules of the user list at `path`. With a workspace the Vade home is
/// opened once, checked, and the file is read through that handle.
fn load_user(path: &Path, trust: Option<&Trust>) -> Result<Vec<Rule>, String> {
    let Some(trust) = trust else { return load(path) };
    let Some(home) = trust.user_home(path, false)? else { return Ok(Vec::new()) };
    let named = |error: String| format!("{}: {error}", path.display());
    let (_, name) = split(path)?;
    read_text_in(&home, name).map_err(named)?.map_or(Ok(Vec::new()), |text| parse(&text).map_err(named))
}

/// Why the user list must be a file of yours with one link: a second link
/// from a project folder would let project code rewrite it. The mode rule
/// stays as before (a hand-made list may be readable by others).
const USER_FILE_RULE: &str = "it must be a regular file that you own, with exactly one link";

/// The text of the file `name` in a checked, opened Vade home: a regular
/// file of yours with one link.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn read_text_in(home: &crate::home_guard::Home, name: &str) -> Result<Option<String>, String> {
    let Some(file) = home.dir().optional_file(name, false).map_err(|_| USER_FILE_RULE.to_owned())? else { return Ok(None) };
    let mut text = String::new();
    file.take(MAX_FILE_BYTES + 1).read_to_string(&mut text).map_err(|_| "cannot read as UTF-8".to_owned())?;
    if text.len() as u64 > MAX_FILE_BYTES {
        return Err("is larger than 64 KiB".into());
    }
    Ok(Some(text))
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn read_text_in(_: &crate::home_guard::Home, _: &str) -> Result<Option<String>, String> {
    Err("a Vade home needs macOS or Linux".into())
}

/// Add `rules` to the user list `name` in a checked, opened Vade home:
/// lock, read, merge and write, all through that handle.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn remember_in_home(home: &crate::home_guard::Home, name: &str, list: &mut List, rules: &[Rule]) -> Result<(), String> {
    let _lock = lock_in(home.dir(), name)?;
    let mut current = match read_text_in(home, name)? {
        Some(text) => parse(&text)?,
        None => Vec::new(),
    };
    list.rules = current.clone();
    for rule in rules {
        if !current.contains(rule) {
            current.push(rule.clone());
        }
    }
    let text = render(&current);
    if current.len() > MAX_RULES || text.len() as u64 > MAX_FILE_BYTES {
        return Err(format!("at most {MAX_RULES} rules and 64 KiB"));
    }
    parse(&text).map_err(|error| format!("would not load again ({error}); the file is unchanged"))?;
    home.dir().write_atomic(name, text.as_bytes()).map_err(|_| "cannot write the file".to_owned())?;
    list.rules = current;
    Ok(())
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn remember_in_home(_: &crate::home_guard::Home, _: &str, _: &mut List, _: &[Rule]) -> Result<(), String> {
    Err("a Vade home needs macOS or Linux".into())
}

/// The allow lists and prompt state shared by the terminal and every tool
/// executor of one agent session, including child agents.
pub struct Approvals {
    interactive: AtomicBool,
    /// The experimental checker's safe class counts (`VADE_SHELL_CHECKER`).
    checker: AtomicBool,
    /// Set for a workspace: its project list needs a trust record.
    trust: Option<Trust>,
    /// The line for the person when the user list is not read: a model tool
    /// can write in the Vade home (`home_guard`).
    user_note: Option<String>,
    state: Mutex<State>,
}
struct State {
    session: Vec<Rule>,
    project: List,
    /// The line for the person when the project list is there but not used.
    project_note: Option<String>,
    user: List,
    /// The reach of each pending yes, by the tool thread that asked.
    choices: Vec<(ThreadId, Scope)>,
    /// Tool threads whose approval request the terminal could not show.
    unshown: Vec<ThreadId>,
    /// Web targets (a site: scheme, host and port; or the search key) that
    /// the person approved for this session.
    web: Vec<String>,
    /// Memory note changes approved for this session (`s` at a memory prompt).
    memory: bool,
    /// What the person saved with `a`, in the Vade home.
    always: Always,
}

/// The saved approvals of one workspace (`crate::saved_approvals`).
#[derive(Default)]
struct Always {
    /// The workspace and the Vade home, to read the file again in a new session.
    source: Option<(PathBuf, PathBuf)>,
    /// The Vade home that an `a` answer saves to; `None` when it cannot be used.
    home: Option<PathBuf>,
    /// The canonical workspace folder: the key of its memory entry.
    project: Option<String>,
    saved: crate::saved_approvals::Saved,
    /// The line for the person when the file is not used.
    note: Option<String>,
}
impl Always {
    fn open(root: &Path, home: &Path, trust: Option<&Trust>) -> Self {
        let source = Some((root.to_path_buf(), home.to_path_buf()));
        let project = root.canonicalize().ok().and_then(|root| root.to_str().map(str::to_owned));
        let memory_note = project.is_none().then(|| "Saved memory approval is off: the canonical workspace path must be UTF-8. Memory changes need approval in each session, and a cannot save them.".to_owned());
        let file = home.join(crate::saved_approvals::FILE_NAME);
        // A model tool can write in the workspace, so a file there could be
        // planted. The home is opened once and checked; the file is read
        // through that handle.
        let opened = match trust.ok_or_else(|| "the workspace folder cannot be checked".to_owned()).and_then(|trust| trust.open_home(home, false)) {
            Ok(opened) => opened,
            Err(reason) => {
                let note = format!("{} is not used and a cannot save: {reason}. Web requests and memory notes ask each time.", file.display());
                return Self { source, note: Some(note), ..Self::default() };
            }
        };
        let read = match &opened {
            None => Ok(crate::saved_approvals::Saved::default()),
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            Some(opened) => crate::saved_approvals::read_in(opened.dir()),
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            Some(_) => Err("saved approvals need macOS or Linux".to_owned()),
        };
        match read {
            Ok(saved) => Self { source, home: Some(home.to_path_buf()), project, saved, note: memory_note },
            Err(error) => Self { source, home: Some(home.to_path_buf()), project, note: Some(format!("{} is not used ({error}), so web requests and memory notes ask each time.", file.display())), ..Self::default() },
        }
    }
}

impl Approvals {
    /// Load the project and user lists. A broken file stops startup: an
    /// unread allow list would silently ask, or silently allow, the wrong things.
    pub fn new(project: Option<PathBuf>, user: Option<PathBuf>, interactive: bool) -> Result<Self, String> {
        Self::with_trust(project, user, interactive, None)
    }
    fn with_trust(project: Option<PathBuf>, user: Option<PathBuf>, interactive: bool, trust: Option<Trust>) -> Result<Self, String> {
        let named = |path: &Path, error: String| format!("{}: {error}", path.display());
        let (rules, project_note) = match &project {
            Some(path) => load_project(path, trust.as_ref()).map_err(|error| named(path, error))?,
            None => (Vec::new(), None),
        };
        let user_rules = match &user {
            Some(path) => load_user(path, trust.as_ref()).map_err(|error| if trust.is_some() { error } else { named(path, error) })?,
            None => Vec::new(),
        };
        Ok(Self {
            interactive: AtomicBool::new(interactive),
            checker: AtomicBool::new(false),
            trust,
            user_note: None,
            state: Mutex::new(State {
                session: Vec::new(),
                project: List { rules, path: project },
                project_note,
                user: List { rules: user_rules, path: user },
                choices: Vec::new(),
                unshown: Vec::new(),
                web: Vec::new(),
                memory: false,
                always: Always::default(),
            }),
        })
    }
    /// The standard places: `<workspace>/.vade/permissions.json` and
    /// `<Vade home>/permissions.json`. The project list applies only with a
    /// trust record in the Vade home (`vade trust permissions`). It also
    /// reads the approvals saved with `a` (`crate::saved_approvals`).
    pub fn for_workspace(root: &Path, vade_home: Option<&Path>, interactive: bool) -> Result<Self, String> {
        // `vade trust` records the canonical folder, so look the record up by that name.
        let trust = Trust {
            root: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()),
            home: vade_home.map(Path::to_path_buf),
            dir: crate::home_guard::Root::open(root),
        };
        // A model tool could plant command rules in a Vade home that it can write.
        let user_path = vade_home.map(|home| home.join(FILE_NAME));
        let user_note = user_path.as_deref().and_then(|path| user_note(path, &trust));
        let user = user_path.filter(|_| user_note.is_none());
        let mut approvals = Self::with_trust(Some(root.join(".vade").join(FILE_NAME)), user, interactive, Some(trust))?;
        approvals.user_note = user_note;
        if let Some(home) = vade_home {
            let always = Always::open(root, home, approvals.trust.as_ref());
            approvals.state().always = always;
        }
        Ok(approvals)
    }
    /// The line for the person when the saved approvals are not used.
    pub fn always_note(&self) -> Option<String> {
        self.state().always.note.clone()
    }
    /// The line for the person when the user list in the Vade home is not read.
    pub fn user_note(&self) -> Option<String> {
        self.user_note.clone()
    }
    /// The line for the person when the project list is there and its rules
    /// do not apply: it has no trust record, or the records do not read.
    pub fn project_note(&self) -> Option<String> {
        self.state().project_note.clone()
    }
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    /// Whether a person can answer a prompt now.
    pub fn interactive(&self) -> bool {
        self.interactive.load(Ordering::Acquire)
    }
    pub fn set_interactive(&self, interactive: bool) {
        self.interactive.store(interactive, Ordering::Release);
    }
    /// Whether commands that the experimental checker reads as safe run
    /// without approval. Off by default: every command asks.
    pub fn checker(&self) -> bool {
        self.checker.load(Ordering::Acquire)
    }
    pub fn set_checker(&self, on: bool) {
        self.checker.store(on, Ordering::Release);
    }
    /// The rule that approves `argv` with this verdict, and the list it came
    /// from. Only an Ask verdict whose argv is all plain literal words that
    /// name no private file and whose findings may go on a list, and only a
    /// rule equal to the argv, with `<path>` slots filled by paths inside
    /// `root` (`policy::rule_matches`).
    pub fn allowed(&self, argv: &[String], verdict: &Verdict, root: &Path) -> Option<(Scope, Rule)> {
        if !policy::listable(argv, verdict) {
            return None;
        }
        let state = self.state();
        let lists = [(Scope::Session, &state.session), (Scope::Project, &state.project.rules), (Scope::User, &state.user.rules)];
        lists.into_iter().find_map(|(scope, rules)| rules.iter().find(|rule| policy::rule_matches(rule, argv, root)).map(|rule| (scope, rule.clone())))
    }
    /// Whether the person approved this web target for the session, or saved it.
    pub fn web_allowed(&self, target: &str) -> bool {
        let state = self.state();
        // A saved site is held in one spelling (`web::saved_site`).
        let saved = crate::web::saved_site(target).unwrap_or_else(|| target.to_owned());
        state.web.iter().any(|old| old == target) || state.always.saved.web.contains(&saved)
    }
    /// Whether memory note changes of this project need no prompt: approved
    /// for the session, or saved for this project.
    pub fn memory_allowed(&self) -> bool {
        let state = self.state();
        state.memory || state.always.project.as_ref().is_some_and(|project| state.always.saved.memory.contains(project))
    }
    /// Approve memory note changes for the rest of the session.
    pub fn allow_memory(&self) {
        self.state().memory = true;
    }
    /// Whether an `a` answer can save: there is a Vade home outside the
    /// workspace (and, for `memory`, a project folder).
    pub fn can_save(&self, memory: bool) -> bool {
        let state = self.state();
        state.always.home.is_some() && (!memory || state.always.project.is_some())
    }
    /// Save a web target, or with `memory` this project's notes, for every
    /// later session. Returns the sentence that says how to take it back.
    pub fn save_always(&self, target: Option<&str>, memory: bool) -> Result<String, String> {
        let mut state = self.state();
        let always = &mut state.always;
        let Some(home) = always.home.clone() else {
            return Err(always.note.clone().unwrap_or_else(|| "there is no Vade home to save it in".into()));
        };
        let entry = match (memory, target) {
            (true, _) => always.project.clone().ok_or("this chat has no project folder")?,
            (false, Some(crate::web::SEARCH_TARGET)) => crate::web::SEARCH_TARGET.to_owned(),
            (false, Some(target)) => crate::web::saved_site(target).ok_or("this site has no saved form")?,
            (false, None) => return Err("no web target".into()),
        };
        // The save opens and checks the home again, and writes through
        // that handle.
        let trust = self.trust.as_ref().ok_or("this chat has no workspace to check the Vade home against")?;
        let opened = trust.open_home(&home, true)?.ok_or("the Vade home cannot be made")?;
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        crate::saved_approvals::add_in(&opened, &entry, memory)?;
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        { let _ = opened; return Err("saved approvals need macOS or Linux".into()); }
        let list = if memory { &mut always.saved.memory } else { &mut always.saved.web };
        if !list.contains(&entry) {
            list.push(entry.clone());
        }
        // A command is shown only when the terminal shows it as it is: it
        // drops control and direction characters, and a non-UTF-8 home
        // has no exact text. A changed command would remove nothing.
        // `vade allowed remove memory` with no path takes the project
        // folder it runs in, which gives the exact entry.
        let exact = |text: &str| !text.chars().any(char::is_control) && crate::terminal::sanitize(text) == text;
        let shown_entry = exact(&entry) || !memory;
        let words = if shown_entry { crate::saved_approvals::remove_words(&entry, memory) } else { format!("vade allowed remove {}", crate::saved_approvals::MEMORY_WORD) };
        let place = if shown_entry { "" } else { " in the project folder" };
        Ok(match home.to_str().filter(|home| exact(home)) {
            Some(shown) => format!("To take this back, run{place}: {words} --vade-home {}", crate::saved_approvals::shell_word(shown)),
            None => format!("To take this back, set VADE_HOME to the Vade home of this chat (its path has characters that cannot be shown here), then run{place}: {words}"),
        })
    }
    /// Keep a web target for the session. A full list keeps what it has, so
    /// the next request to a new target asks again.
    pub fn allow_web(&self, target: &str) {
        let mut state = self.state();
        if state.web.len() < MAX_RULES && !state.web.iter().any(|old| old == target) {
            state.web.push(target.to_owned());
        }
    }
    /// Start a new, resumed or forked session: forget session rules and any
    /// pending answer, and read the project and user lists again, so a rule
    /// taken out of a file since startup no longer allows anything. A file
    /// that no longer reads leaves its list empty, so its commands ask, and
    /// gives an error.
    pub fn new_session(&self) -> Result<(), String> {
        let mut state = self.state();
        state.session.clear();
        state.web.clear();
        state.memory = false;
        if let Some((root, home)) = state.always.source.clone() {
            state.always = Always::open(&root, &home, self.trust.as_ref());
        }
        state.choices.clear();
        state.unshown.clear();
        let State { project, project_note, user, .. } = &mut *state;
        let mut errors = Vec::new();
        if let Some(path) = &project.path {
            (project.rules, *project_note) = load_project(path, self.trust.as_ref()).unwrap_or_else(|error| {
                errors.push(format!("{}: {error}", path.display()));
                (Vec::new(), None)
            });
        }
        if let Some(path) = &user.path {
            // The home is opened and checked again at each read.
            user.rules = load_user(path, self.trust.as_ref()).unwrap_or_else(|error| {
                errors.push(if self.trust.is_some() { error } else { format!("{}: {error}", path.display()) });
                Vec::new()
            });
        }
        if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
    }
    /// The terminal records how far a yes reaches, for the tool thread that
    /// asked. Parent and child tools ask at the same time, so one shared
    /// slot could give one tool the reach chosen for another.
    pub fn choose_for(&self, asker: ThreadId, scope: Scope) {
        let mut state = self.state();
        state.choices.retain(|(thread, _)| *thread != asker);
        state.choices.push((asker, scope));
    }
    /// `choose_for` the calling thread.
    pub fn choose(&self, scope: Scope) {
        self.choose_for(std::thread::current().id(), scope);
    }
    /// The terminal takes back a yes that did not reach its tool.
    pub fn withdraw_choice(&self, asker: ThreadId) {
        self.state().choices.retain(|(thread, _)| *thread != asker);
    }
    /// The terminal could not show this tool's approval request, so its no
    /// is not the user's answer.
    pub fn not_shown_for(&self, asker: ThreadId) {
        let mut state = self.state();
        if !state.unshown.contains(&asker) {
            state.unshown.push(asker);
        }
    }
    /// The executor reads, and clears, whether its request was not shown.
    pub fn take_not_shown(&self) -> bool {
        let me = std::thread::current().id();
        let mut state = self.state();
        let before = state.unshown.len();
        state.unshown.retain(|thread| *thread != me);
        state.unshown.len() != before
    }
    /// The executor reads, and clears, the reach of the answer it received.
    pub fn take_choice(&self) -> Scope {
        let me = std::thread::current().id();
        let mut state = self.state();
        let at = state.choices.iter().position(|(thread, _)| *thread == me);
        at.map_or(Scope::Once, |at| state.choices.remove(at).1)
    }
    /// Add rules to a list. The project and user lists are reread and saved
    /// at once, under the file's lock. Returns where the rules went.
    pub fn remember(&self, scope: Scope, rules: &[Rule]) -> Result<String, String> {
        let mut state = self.state();
        let trust = self.trust.as_ref().filter(|_| scope == Scope::Project);
        let list = match scope {
            Scope::Once => return Ok("this command only".into()),
            Scope::Session => {
                let added = rules.iter().filter(|rule| !state.session.contains(rule)).count();
                if state.session.len() + added > MAX_RULES {
                    return Err(format!("this session already has {MAX_RULES} rules"));
                }
                for rule in rules {
                    if !state.session.contains(rule) {
                        state.session.push(rule.clone());
                    }
                }
                return Ok("this session".into());
            }
            Scope::Project => &mut state.project,
            Scope::User => &mut state.user,
            Scope::Always => return Err("a command cannot be saved with a; use s or p".into()),
        };
        let path = list.path.clone().ok_or("no file for this list")?;
        // The user list of a workspace: the Vade home is opened and checked
        // again, and the lock, read and write go through that handle.
        if scope == Scope::User && let Some(trust) = &self.trust {
            let home = match trust.user_home(&path, true) {
                Ok(Some(home)) => home,
                Ok(None) => return Err("the Vade home cannot be made".into()),
                Err(error) => {
                    list.rules.clear();
                    return Err(format!("the rule is not saved. {error}"));
                }
            };
            let (_, name) = split(&path)?;
            remember_in_home(&home, name, list, rules).map_err(|error| format!("{}: {error}", path.display()))?;
            return Ok(path.display().to_string());
        }
        // Another session can save the same file at the same time; the lock
        // keeps its read and write apart from ours (review of aacf3c9).
        let _lock = lock_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        // The file as it is now replaces the list, also when the rule is not
        // saved: a rule removed by hand no longer allows anything, and a
        // file that does not load, or that has no trust record, allows nothing.
        let named = |error: String| format!("{}: {error}", path.display());
        let mut current = match load_project(&path, trust) {
            Ok((rules, None)) => rules,
            Ok((_, Some(note))) => {
                list.rules.clear();
                return Err(format!("the project list is not used, so the rule is not saved. {note}"));
            }
            Err(error) => {
                list.rules.clear();
                return Err(named(error));
            }
        };
        list.rules = current.clone();
        for rule in rules {
            if !current.contains(rule) {
                current.push(rule.clone());
            }
        }
        if current.len() > MAX_RULES {
            return Err(format!("{}: at most {MAX_RULES} rules", path.display()));
        }
        let Some(trust) = trust else {
            save(&path, &current).map_err(named)?;
            list.rules = current;
            return Ok(path.display().to_string());
        };
        // The person approved this rule, so the new file is trusted as
        // written. The record goes first: a file that Vade wrote but could
        // not record would stop every rule in the next session.
        let home = trust.home.as_deref().ok_or_else(|| named("no Vade home for its trust record; the file is unchanged".into()))?;
        let mut records = crate::trust::Store::open_at(home, trust.dir.as_ref().map_err(Clone::clone)?)?;
        let before = records.recorded(&trust.root, FILE_NAME);
        records.set(&trust.root, FILE_NAME, Some(&crate::trust::digest(render(&current).as_bytes())))?;
        if let Err(error) = save(&path, &current) {
            let _ = records.set(&trust.root, FILE_NAME, before.as_deref());
            return Err(named(error));
        }
        list.rules = current;
        Ok(path.display().to_string())
    }
    /// Rule counts per list, for status lines.
    pub fn summary(&self) -> String {
        let state = self.state();
        format!(
            "allow lists: {} session, {} project, {} user rule(s)",
            state.session.len(),
            state.project.rules.len(),
            state.user.rules.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Rule {
        text.split(' ').map(str::to_owned).collect()
    }
    /// The rule that approves `argv` under the default policy.
    fn allowed(approvals: &Approvals, argv: &Rule) -> Option<(Scope, Rule)> {
        approvals.allowed(argv, &crate::shell::policy::decide(argv, Path::new("/w"), false), Path::new("/w"))
    }
    struct Temp(PathBuf);
    impl Temp {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!("vade-permissions-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(self.home());
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    impl Temp {
        /// A Vade home for the trust records of the project in this folder,
        /// beside it: a home inside the workspace holds no trusted record.
        fn home(&self) -> PathBuf {
            let mut home = self.0.clone().into_os_string();
            home.push("-home");
            PathBuf::from(home)
        }
        /// Write the project list by hand, then trust it as it is, as
        /// `vade trust permissions` does.
        fn write_trusted(&self, text: &str) {
            std::fs::write(self.0.join(".vade").join(FILE_NAME), text).unwrap();
            // The record names the canonical folder, as `vade trust` writes it.
            let root = self.0.canonicalize().unwrap();
            crate::trust::Store::open(&self.home(), &root).unwrap().set(&root, FILE_NAME, Some(&crate::trust::digest(text.as_bytes()))).unwrap();
        }
    }

    #[test]
    fn rules_are_exact_commands_of_plain_words() {
        let file = |rule: &str| format!("{{\"version\":1,\"allow\":[{}]}}", json::quote(rule));
        for good in ["cargo fmt", "rm -rf build", "cargo +1.99.0 test --offline", "npm install left-pad@1.3.0", "git commit -m wip"] {
            assert_eq!(parse(&file(good)).unwrap(), vec![words(good)], "{good}");
        }
        // `*`, patterns, braces, variables, quotes, `~` and spaces are not
        // plain words: a rule with one does not load (review of 583ad99).
        for bad in ["git push *", "touch item{1,2}", "touch item[12]", "cat $HOME/x", "cat ~/x", "echo 'a'", "rm -rf build?"] {
            assert!(parse(&file(bad)).unwrap_err().contains("plain words"), "{bad}");
        }
        assert!(parse(r#"{"version":1,"allow":[["rm","-rf","my dir"]]}"#).is_err());
    }

    #[test]
    fn file_format_round_trips_and_rejects_mistakes() {
        let rules = vec![words("git push origin main"), words("cargo fmt")];
        let text = render(&rules);
        assert_eq!(parse(&text).unwrap(), rules);
        assert!(text.contains("\"git push origin main\""));
        assert_eq!(parse(&render(&[])).unwrap(), Vec::<Rule>::new());
        assert_eq!(parse(r#"{"version":1}"#).unwrap(), Vec::<Rule>::new());
        for bad in [
            "",
            "[]",
            r#"{"allow":[]}"#,
            r#"{"version":2,"allow":[]}"#,
            r#"{"version":1,"allow":"git push"}"#,
            r#"{"version":1,"allow":[7]}"#,
            r#"{"version":1,"allow":[""]}"#,
            r#"{"version":1,"allow":["*"]}"#,
            r#"{"version":1,"allow":[["git", 7]]}"#,
            r#"{"version":1,"allow":[],"deny":[]}"#,
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn allow_lists_approve_only_the_exact_plain_argv() {
        let temp = Temp::new("lists");
        let project = temp.0.join("project.json");
        let user = temp.0.join("user.json");
        std::fs::write(&project, render(&[words("git push origin main")])).unwrap();
        std::fs::write(&user, render(&[words("rm -rf build")])).unwrap();
        let approvals = Approvals::new(Some(project.clone()), Some(user), true).unwrap();
        assert_eq!(allowed(&approvals, &words("git push origin main")), Some((Scope::Project, words("git push origin main"))));
        assert_eq!(allowed(&approvals, &words("rm -rf build")), Some((Scope::User, words("rm -rf build"))));
        // Other words, more words, and the same words as a script do not match.
        assert_eq!(allowed(&approvals, &words("git push origin dev")), None);
        assert_eq!(allowed(&approvals, &words("rm -rf build src")), None);
        assert_eq!(allowed(&approvals, &vec!["/bin/sh".into(), "-c".into(), "rm -rf build".into()]), None);
        // Blocked commands and root never use a rule. A rule that names a
        // private file does not load.
        std::fs::write(&project, render(&[words("sudo ls"), words("rm -rf /"), words("/usr/bin//sudo ls"), words("/bin//rm -rf /")])).unwrap();
        let approvals = Approvals::new(Some(project.clone()), None, true).unwrap();
        for text in ["sudo ls", "rm -rf /", "/usr/bin//sudo ls", "/bin//rm -rf /"] {
            assert_eq!(allowed(&approvals, &words(text)), None, "{text}");
        }
        std::fs::write(&project, render(&[words("cat .ssh/id_rsa")])).unwrap();
        assert!(Approvals::new(Some(project), None, true).err().unwrap().contains("credential"));
    }

    #[test]
    fn path_slots_match_one_workspace_path() {
        let temp = Temp::new("slots");
        let root = temp.0.canonicalize().unwrap();
        std::fs::create_dir_all(root.join("src/deep")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("src/a.rs"), "").unwrap();
        let project = root.join("rules.json");
        let text = "{\"version\":1,\"allow\":[\"ls -la <path>\",\"cargo test\",\"git status\",\"wc -l <path> <path>\"]}";
        std::fs::write(&project, text).unwrap();
        let rules = load(&project).unwrap();
        assert_eq!(parse(&render(&rules)).unwrap(), rules);
        let approvals = Approvals::new(Some(project), None, true).unwrap();
        let run = |text: &str| {
            let argv = words(text);
            approvals.allowed(&argv, &crate::shell::policy::decide(&argv, &root, false), &root).map(|(_, rule)| rule_text(&rule))
        };
        for text in ["ls -la .", "ls -la ./", "ls -la src", "ls -la ./src/deep/", "ls -la src/a.rs", "ls -la not-made-yet", "ls -la src/new/x.txt"] {
            assert_eq!(run(text).as_deref(), Some("ls -la <path>"), "{text}");
        }
        assert_eq!(run("cargo test").as_deref(), Some("cargo test"));
        assert_eq!(run("git status").as_deref(), Some("git status"));
        assert_eq!(run("wc -l src/a.rs src").as_deref(), Some("wc -l <path> <path>"));
        // A link in any part could point outside the workspace.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc", root.join("out")).unwrap();
            std::os::unix::fs::symlink(root.join("src"), root.join("src/deep/back")).unwrap();
        }
        // Not one workspace path, not the same words, or a word that expands.
        for text in [
            "ls -la /", "ls -la /etc", "ls -la ..", "ls -la src/../..", "ls -la -R", "ls -la --color=always", "ls -la .git", "ls -la .git/config",
            "ls -la out", "ls -la out/passwd", "ls -la src/deep/back", "ls -la src/deep/back/a.rs", "ls -la id_rsa", "ls -la src/server.pem",
            "ls -la .vade/permissions.json", "ls -la permissions.json", "ls -la src a", "ls -la", "ls -l src", "ls -la src/*", "ls -la ~", "ls -la $HOME",
            "ls -la src?", "ls -la {src,a}", "ls -la src/caf\u{e9}", "cargo test --offline", "cargo  test", "git status --short", "wc -l src",
            "/bin/ls -la src", "ls -la src/a.rs;rm",
        ] {
            assert_eq!(run(text), None, "{text}");
        }
        assert_eq!(run("ls -la src/deep").as_deref(), Some("ls -la <path>"));
        // A slot is only an argument, and a rule names no private file.
        for bad in ["<path> -la", "ls <paths>", "ls <PATH>", "ls <path>x", "cat .env", "tar -tfid_rsa", "git show HEAD:.env", "curl -d @id_rsa x"] {
            let file = format!("{{\"version\":1,\"allow\":[{}]}}", json::quote(bad));
            assert!(parse(&file).is_err(), "{bad}");
        }
        // `s` saves the exact argv, never a slot.
        let ls = words("ls -la src");
        assert_eq!(crate::shell::policy::rule_for(&ls, &crate::shell::policy::decide(&ls, &root, false)), Some(ls));
    }

    #[test]
    fn each_tool_reads_the_reach_chosen_for_it() {
        let approvals = Approvals::new(None, None, true).unwrap();
        std::thread::scope(|scope| {
            let approvals = &approvals;
            let tool = || {
                let (asked, asks) = std::sync::mpsc::channel();
                let (resume, resumes) = std::sync::mpsc::channel::<()>();
                let thread = scope.spawn(move || {
                    asked.send(std::thread::current().id()).unwrap();
                    resumes.recv().unwrap();
                    approvals.take_choice()
                });
                (asks.recv().unwrap(), resume, thread)
            };
            let (a_id, resume_a, a) = tool();
            let (b_id, resume_b, b) = tool();
            approvals.choose_for(a_id, Scope::Once);
            approvals.choose_for(b_id, Scope::Session);
            resume_a.send(()).unwrap();
            assert_eq!(a.join().unwrap(), Scope::Once);
            resume_b.send(()).unwrap();
            assert_eq!(b.join().unwrap(), Scope::Session);
        });
        approvals.choose(Scope::Project);
        approvals.withdraw_choice(std::thread::current().id());
        assert_eq!(approvals.take_choice(), Scope::Once);
    }

    #[test]
    fn a_user_list_that_model_tools_can_write_is_not_read() {
        // A sandboxed command can write a plain folder of the workspace, also
        // one with a non-ASCII name. Rules planted in a Vade home there are not read.
        let temp = Temp::new("planted-user");
        let root = temp.0.canonicalize().unwrap();
        let rm = words("rm -rf build");
        for home in [root.join(".config/vade"), root.join(".vade"), root.join("secrets/vade"), root.join("vade-home"), root.join("caf\u{e9}").join("vade")] {
            std::fs::create_dir_all(&home).unwrap();
            std::fs::write(home.join(FILE_NAME), render(std::slice::from_ref(&rm))).unwrap();
            #[cfg(unix)]
            { use std::os::unix::fs::PermissionsExt; std::fs::set_permissions(home.join(FILE_NAME), std::fs::Permissions::from_mode(0o600)).unwrap(); }
            let approvals = Approvals::for_workspace(&root, Some(&home), false).unwrap();
            assert_eq!(allowed(&approvals, &rm), None, "{}", home.display());
            let note = approvals.user_note().unwrap();
            assert!(note.contains("permissions.json is not used: the Vade home") && note.contains("inside this workspace"), "{note}");
            let before = std::fs::read(home.join(FILE_NAME)).unwrap();
            assert!(approvals.remember(Scope::User, std::slice::from_ref(&rm)).is_err());
            assert_eq!(std::fs::read(home.join(FILE_NAME)).unwrap(), before);
            approvals.new_session().unwrap();
            assert_eq!(allowed(&approvals, &rm), None);
        }
        // The same list in a home outside the workspace is read.
        for home in [temp.home()] {
            std::fs::create_dir_all(&home).unwrap();
            std::fs::write(home.join(FILE_NAME), render(std::slice::from_ref(&rm))).unwrap();
            let approvals = Approvals::for_workspace(&root, Some(&home), true).unwrap();
            assert_eq!(allowed(&approvals, &rm), Some((Scope::User, rm.clone())), "{}", home.display());
            assert_eq!(approvals.user_note(), None);
        }
    }

    /// The user list is checked at each read and each write, not only at
    /// start: here the home's link moves into the workspace after start.
    #[cfg(unix)]
    #[test]
    fn a_user_list_whose_home_moved_into_the_workspace_is_not_read_or_written() {
        use std::os::unix::fs::PermissionsExt;
        let temp = Temp::new("moved-user");
        for dir in ["work/sub/vade", "real/vade"] { std::fs::create_dir_all(temp.0.join(dir)).unwrap(); }
        let root = temp.0.join("work").canonicalize().unwrap();
        let link = temp.0.join("link");
        std::os::unix::fs::symlink(temp.0.join("real"), &link).unwrap();
        let home = link.join("vade");
        let approvals = Approvals::for_workspace(&root, Some(&home), true).unwrap();
        assert_eq!(approvals.user_note(), None);
        // The link now leads into the workspace, where a list was planted.
        let (rm, push) = (words("rm -rf build"), words("git push origin main"));
        let planted = root.join("sub/vade").join(FILE_NAME);
        std::fs::write(&planted, render(std::slice::from_ref(&rm))).unwrap();
        std::fs::set_permissions(&planted, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(root.join("sub"), &link).unwrap();
        let error = approvals.new_session().unwrap_err();
        assert!(error.contains("inside this workspace"), "{error}");
        assert_eq!(allowed(&approvals, &rm), None);
        let error = approvals.remember(Scope::User, std::slice::from_ref(&push)).unwrap_err();
        assert!(error.contains("not saved") && error.contains("inside this workspace"), "{error}");
        assert_eq!(allowed(&approvals, &push), None);
        assert_eq!(std::fs::read_to_string(&planted).unwrap(), render(std::slice::from_ref(&rm)));
        assert!(!temp.0.join("real/vade").join(FILE_NAME).exists());
    }

    /// Review 121e: a user list with a second (hard) link, for example from a
    /// project folder, could be rewritten by project code: it is not read
    /// and not written. A list readable by others but with one link is still
    /// read, as before.
    #[cfg(unix)]
    #[test]
    fn a_user_list_with_a_second_link_is_not_read_or_written() {
        use std::os::unix::fs::PermissionsExt;
        let temp = Temp::new("linked-user");
        std::fs::create_dir_all(temp.0.join("work")).unwrap();
        let root = temp.0.join("work").canonicalize().unwrap();
        let home = temp.home();
        std::fs::create_dir_all(&home).unwrap();
        let (rm, push) = (words("rm -rf build"), words("git push origin main"));
        let list = home.join(FILE_NAME);
        std::fs::write(&list, render(std::slice::from_ref(&rm))).unwrap();
        std::fs::set_permissions(&list, std::fs::Permissions::from_mode(0o644)).unwrap();
        let approvals = Approvals::for_workspace(&root, Some(&home), true).unwrap();
        assert_eq!(approvals.user_note(), None);
        assert_eq!(allowed(&approvals, &rm), Some((Scope::User, rm.clone())));
        // A second link from the project folder.
        std::fs::hard_link(&list, root.join("linked.json")).unwrap();
        let approvals = Approvals::for_workspace(&root, Some(&home), true).unwrap();
        let note = approvals.user_note().unwrap();
        assert!(note.contains("exactly one link"), "{note}");
        assert_eq!(allowed(&approvals, &rm), None);
        // A session that started before the link reads nothing at /new and saves nothing.
        std::fs::remove_file(root.join("linked.json")).unwrap();
        let started = Approvals::for_workspace(&root, Some(&home), true).unwrap();
        std::fs::hard_link(&list, root.join("linked.json")).unwrap();
        assert!(started.new_session().unwrap_err().contains("exactly one link"));
        assert_eq!(allowed(&started, &rm), None);
        assert!(started.remember(Scope::User, std::slice::from_ref(&push)).unwrap_err().contains("exactly one link"));
        assert_eq!(std::fs::read_to_string(&list).unwrap(), render(std::slice::from_ref(&rm)));
    }

    #[test]
    fn remembered_rules_reach_the_chosen_list() {
        let temp = Temp::new("remember");
        let project = temp.0.join(".vade").join(FILE_NAME);
        let approvals = Approvals::for_workspace(&temp.0, Some(&temp.home()), true).unwrap();
        let push = words("git push origin main");
        let rules = vec![push.clone()];
        assert_eq!(approvals.take_choice(), Scope::Once);
        approvals.choose(Scope::Session);
        assert_eq!(approvals.take_choice(), Scope::Session);
        assert_eq!(approvals.take_choice(), Scope::Once);
        assert_eq!(approvals.remember(Scope::Session, &rules).unwrap(), "this session");
        assert_eq!(allowed(&approvals, &push), Some((Scope::Session, push.clone())));
        assert!(!project.exists());
        let clean = words("cargo clean");
        approvals.remember(Scope::Project, &[clean.clone()]).unwrap();
        assert_eq!(load(&project).unwrap(), vec![clean.clone()]);
        // A second session in the same project reads the saved rule: a `p`
        // answer keeps the trust record of the file current.
        let next = Approvals::for_workspace(&temp.0, Some(&temp.home()), false).unwrap();
        assert!(allowed(&next, &clean).is_some() && allowed(&next, &push).is_none());
        assert!(!next.interactive() && next.project_note().is_none());
        assert!(Approvals::new(None, None, true).unwrap().remember(Scope::User, &rules).is_err());
        assert!(next.summary().contains("1 project"));
        // A new, resumed or forked session drops session rules; project rules stay.
        approvals.new_session().unwrap();
        assert_eq!(allowed(&approvals, &push), None);
        assert!(allowed(&approvals, &clean).is_some());
        // It reads the files again: a rule taken out of the file since
        // startup no longer allows, and a broken file allows nothing.
        temp.write_trusted("{\"version\":1,\"allow\":[]}");
        assert!(allowed(&next, &clean).is_some());
        next.new_session().unwrap();
        assert!(allowed(&next, &clean).is_none());
        temp.write_trusted("{\"version\":1,\"allow\":[\"cargo clean\"]}");
        next.new_session().unwrap();
        assert!(allowed(&next, &clean).is_some());
        temp.write_trusted("{\"version\":1,\"allow\":[7]}");
        assert!(next.new_session().unwrap_err().contains(&project.display().to_string()));
        assert!(allowed(&next, &clean).is_none());
        // A `p` answer that cannot save still takes the file as it is now:
        // a rule removed by hand stops at once (review of 583ad99, item 9).
        temp.write_trusted(&render(&[clean.clone()]));
        next.new_session().unwrap();
        assert!(allowed(&next, &clean).is_some());
        let full: Vec<Rule> = (0..MAX_RULES).map(|n| words(&format!("touch f{n}"))).collect();
        temp.write_trusted(&render(&full));
        assert!(next.remember(Scope::Project, &[words("touch extra")]).unwrap_err().contains("at most"));
        assert!(allowed(&next, &clean).is_none());
        assert!(allowed(&next, &words("touch f7")).is_some());
        temp.write_trusted("{\"version\":1,\"allow\":[7]}");
        assert!(next.remember(Scope::Project, &[words("touch extra")]).is_err());
        assert!(allowed(&next, &words("touch f7")).is_none());
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_project_list_applies_only_with_a_trust_record_of_its_exact_bytes() {
        // A project that someone else wrote ships its own allow list.
        let temp = Temp::new("trust");
        let project = temp.0.join(".vade").join(FILE_NAME);
        std::fs::create_dir(temp.0.join(".vade")).unwrap();
        let (make, clean) = (words("make"), words("cargo clean"));
        let shipped = render(std::slice::from_ref(&make));
        std::fs::write(&project, &shipped).unwrap();
        // No record: no rule applies, and the person learns what to do.
        let approvals = Approvals::for_workspace(&temp.0, Some(&temp.home()), true).unwrap();
        assert_eq!(allowed(&approvals, &make), None);
        let note = approvals.project_note().unwrap();
        assert!(note.contains(&project.display().to_string()) && note.contains("run: vade trust permissions --vade-home "), "{note}");
        assert!(approvals.summary().contains("0 project"));
        // A `p` answer adds no rule to that file and gives it no record.
        let error = approvals.remember(Scope::Project, std::slice::from_ref(&clean)).unwrap_err();
        assert!(error.contains("is not trusted") && error.contains("the rule is not saved"), "{error}");
        assert_eq!(std::fs::read_to_string(&project).unwrap(), shipped);
        approvals.new_session().unwrap();
        assert!(allowed(&approvals, &make).is_none() && approvals.project_note().is_some());
        // An untrusted file is not parsed, so a broken one cannot stop a chat.
        std::fs::write(&project, "{\"version\":1,\"allow\":[7]}").unwrap();
        assert!(Approvals::for_workspace(&temp.0, Some(&temp.home()), true).unwrap().project_note().is_some());
        // The person trusts the file as it is: its rules apply.
        temp.write_trusted(&shipped);
        approvals.new_session().unwrap();
        assert_eq!(allowed(&approvals, &make), Some((Scope::Project, make.clone())));
        assert_eq!(approvals.project_note(), None);
        // A changed file (for example after a pull) needs a new record.
        std::fs::write(&project, render(&[make.clone(), words("sh build.sh")])).unwrap();
        approvals.new_session().unwrap();
        assert!(allowed(&approvals, &make).is_none() && allowed(&approvals, &words("sh build.sh")).is_none());
        assert!(approvals.project_note().is_some());
        // The record names this project folder: the same file elsewhere has none.
        let other = Temp::new("trust-other");
        std::fs::create_dir(other.0.join(".vade")).unwrap();
        std::fs::write(other.0.join(".vade").join(FILE_NAME), &shipped).unwrap();
        temp.write_trusted(&shipped);
        assert!(Approvals::for_workspace(&other.0, Some(&temp.home()), true).unwrap().project_note().is_some());
        assert!(Approvals::for_workspace(&temp.0, Some(&temp.home()), true).unwrap().project_note().is_none());
        // With no Vade home there is no record: the list never applies and `p` saves nothing.
        let homeless = Approvals::for_workspace(&temp.0, None, true).unwrap();
        assert!(allowed(&homeless, &make).is_none() && homeless.project_note().is_some());
        std::fs::remove_file(&project).unwrap();
        assert!(homeless.remember(Scope::Project, std::slice::from_ref(&clean)).unwrap_err().contains("no Vade home"));
        assert!(!project.exists());
        // A record file that does not read trusts nothing and names its reason.
        temp.write_trusted(&shipped);
        std::fs::write(temp.home().join(crate::trust::STORE), "not json").unwrap();
        let unread = Approvals::for_workspace(&temp.0, Some(&temp.home()), true).unwrap();
        assert!(allowed(&unread, &make).is_none());
        assert!(unread.project_note().unwrap().contains("is not valid JSON"));
    }

    #[test]
    fn sessions_that_save_at_once_keep_every_rule() {
        // Review of aacf3c9, item 3: each store reads the file, adds its rule
        // and renames a new file over it; without a lock, the last rename
        // drops the rules of the others.
        let temp = Temp::new("race");
        let project = temp.0.join(".vade").join(FILE_NAME);
        let mut expected = Vec::new();
        for round in 0..4 {
            let rules: Vec<Rule> = (0..16).map(|n| words(&format!("touch r{round}-{n}"))).collect();
            let stores: Vec<Approvals> = rules.iter().map(|_| Approvals::for_workspace(&temp.0, Some(&temp.home()), true).unwrap()).collect();
            let start = std::sync::Barrier::new(stores.len());
            let results = std::thread::scope(|scope| {
                let writers: Vec<_> = stores.iter().zip(&rules).map(|(store, rule)| {
                    let start = &start;
                    scope.spawn(move || {
                        start.wait();
                        store.remember(Scope::Project, std::slice::from_ref(rule))
                    })
                }).collect();
                writers.into_iter().map(|writer| writer.join().unwrap()).collect::<Vec<_>>()
            });
            let saved = load(&project).unwrap();
            for (rule, result) in rules.iter().zip(&results) {
                match result {
                    Ok(_) => assert!(saved.contains(rule), "round {round}: {} lost", rule_text(rule)),
                    Err(error) => assert_eq!(error, &format!("{}: another session is saving it, or its lock does not open; the file is unchanged", project.display())),
                }
            }
            // A busy save can reach its deadline on a loaded host. Check all
            // successful writes first, then retry only busy saves in input order.
            for ((store, rule), result) in stores.iter().zip(&rules).zip(&results) {
                if result.is_err() {
                    store.remember(Scope::Project, std::slice::from_ref(rule)).unwrap();
                }
            }
            expected.extend(rules);
            let saved = load(&project).unwrap();
            assert_eq!(saved.len(), expected.len(), "round {round}");
            for rule in &expected {
                assert!(saved.contains(rule), "round {round}: {} lost", rule_text(rule));
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_busy_rules_file_stays_unchanged_until_a_retry() {
        let temp = Temp::new("busy");
        let project = temp.0.join(".vade").join(FILE_NAME);
        let store = Approvals::for_workspace(&temp.0, Some(&temp.home()), true).unwrap();
        let first = words("touch first");
        let next = words("touch next");
        store.remember(Scope::Project, std::slice::from_ref(&first)).unwrap();
        let before = std::fs::read(&project).unwrap();
        let held = lock_file(&project).unwrap();
        let error = store.remember(Scope::Project, std::slice::from_ref(&next)).unwrap_err();
        assert_eq!(error, format!("{}: another session is saving it, or its lock does not open; the file is unchanged", project.display()));
        assert_eq!(std::fs::read(&project).unwrap(), before);
        drop(held);
        store.remember(Scope::Project, std::slice::from_ref(&next)).unwrap();
        assert_eq!(load(&project).unwrap(), vec![first, next]);
    }

    #[test]
    fn a_broken_file_stops_startup_with_its_path() {
        let temp = Temp::new("broken");
        let path = temp.0.join(FILE_NAME);
        std::fs::write(&path, "{\"version\":1,\"allow\":[7]}").unwrap();
        let error = Approvals::new(Some(path.clone()), None, true).err().unwrap();
        assert!(error.contains(&path.display().to_string()) && error.contains("string"));
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(Approvals::new(Some(path), None, true).is_err());
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn linked_folders_fifos_and_oversized_rules_are_refused() {
        let temp = Temp::new("links");
        let outside = temp.0.join("outside");
        let workspace = temp.0.join("workspace");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::create_dir_all(&workspace).unwrap();
        let planted = render(&[words("rm -rf build")]);
        std::fs::write(outside.join(FILE_NAME), &planted).unwrap();
        std::os::unix::fs::symlink(&outside, workspace.join(".vade")).unwrap();
        let path = workspace.join(".vade").join(FILE_NAME);
        assert!(load(&path).unwrap_err().contains("link"));
        assert!(Approvals::for_workspace(&workspace, None, true).is_err());
        assert!(save(&path, &[words("git push")]).is_err());
        assert_eq!(std::fs::read_to_string(outside.join(FILE_NAME)).unwrap(), planted);
        // A linked file and a FIFO are refused at once, without blocking.
        std::fs::remove_file(workspace.join(".vade")).unwrap();
        std::fs::create_dir(workspace.join(".vade")).unwrap();
        std::os::unix::fs::symlink(outside.join(FILE_NAME), &path).unwrap();
        assert!(load(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        if std::process::Command::new("mkfifo").arg(&path).status().is_ok_and(|s| s.success()) {
            assert!(load(&path).is_err());
            std::fs::remove_file(&path).unwrap();
        }
        // A rule that would not load again keeps the old file.
        let approvals = Approvals::for_workspace(&workspace, Some(&temp.home()), true).unwrap();
        approvals.remember(Scope::Project, &[words("git push")]).unwrap();
        let long: Rule = (0..129).map(|i| format!("w{i}")).collect();
        assert!(approvals.remember(Scope::Project, &[long]).unwrap_err().contains("unchanged"));
        assert_eq!(load(&path).unwrap(), vec![words("git push")]);
    }
}
