//! Allow lists for commands that need approval: this session (memory only),
//! the project (`.vade/permissions.json`) and the user
//! (`<Vade home>/permissions.json`). A rule is one exact argv of plain
//! literal words (see `policy::literal`); it matches only that same argv.
//! Blocked commands and findings that cannot go on a list never match a rule.
use super::policy;
use super::safety::Verdict;
use crate::json::{self, Value};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::ThreadId;

/// One allow-list rule: the exact argv it approves.
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
}
impl Scope {
    pub fn label(self) -> &'static str {
        match self {
            Self::Once => "once",
            Self::Session => "session",
            Self::Project => "project",
            Self::User => "user",
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
    if let Some(word) = rule.iter().find(|word| !policy::literal(word)) {
        return Err(format!("a rule is one exact command of plain words (letters, digits and - _ . , / : = + @); {} is not one", json::quote(word)));
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
    let (folder, name) = split(path)?;
    if std::fs::symlink_metadata(folder).is_err_and(|error| error.kind() == io::ErrorKind::NotFound) {
        return Ok(Vec::new());
    }
    let Some(file) = open_file(folder, name)? else { return Ok(Vec::new()) };
    let mut text = String::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|_| "cannot read as UTF-8".to_owned())?;
    if text.len() as u64 > MAX_FILE_BYTES {
        return Err("is larger than 64 KiB".into());
    }
    parse(&text)
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
fn open_file(folder: &Path, name: &str) -> Result<Option<std::fs::File>, String> {
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
fn lock_file(path: &Path) -> Result<crate::tools::anchored::Lock, String> {
    use std::os::unix::fs::DirBuilderExt;
    let (folder, name) = split(path)?;
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(folder).map_err(|error| format!("cannot create its folder: {error}"))?;
    let folder = crate::tools::anchored::Dir::open_root(folder).map_err(|_| "its folder must be a real folder, not a link".to_owned())?;
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
fn lock_file(path: &Path) -> Result<std::fs::File, String> {
    let (folder, name) = split(path)?;
    std::fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let file = std::fs::OpenOptions::new().write(true).create(true).open(folder.join(format!(".{name}.lock"))).map_err(|error| error.to_string())?;
    file.lock().map_err(|error| error.to_string())?;
    Ok(file)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn open_file(folder: &Path, name: &str) -> Result<Option<std::fs::File>, String> {
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

/// The allow lists and prompt state shared by the terminal and every tool
/// executor of one agent session, including child agents.
pub struct Approvals {
    interactive: AtomicBool,
    /// The experimental checker's safe class counts (`VADE_SHELL_CHECKER`).
    checker: AtomicBool,
    state: Mutex<State>,
}
struct State {
    session: Vec<Rule>,
    project: List,
    user: List,
    /// The reach of each pending yes, by the tool thread that asked.
    choices: Vec<(ThreadId, Scope)>,
}

impl Approvals {
    /// Load the project and user lists. A broken file stops startup: an
    /// unread allow list would silently ask, or silently allow, the wrong things.
    pub fn new(project: Option<PathBuf>, user: Option<PathBuf>, interactive: bool) -> Result<Self, String> {
        let read = |path: &Option<PathBuf>| -> Result<Vec<Rule>, String> {
            match path {
                Some(path) => load(path).map_err(|error| format!("{}: {error}", path.display())),
                None => Ok(Vec::new()),
            }
        };
        Ok(Self {
            interactive: AtomicBool::new(interactive),
            checker: AtomicBool::new(false),
            state: Mutex::new(State {
                session: Vec::new(),
                project: List { rules: read(&project)?, path: project },
                user: List { rules: read(&user)?, path: user },
                choices: Vec::new(),
            }),
        })
    }
    /// The standard places: `<workspace>/.vade/permissions.json` and
    /// `<Vade home>/permissions.json`.
    pub fn for_workspace(root: &Path, vade_home: Option<&Path>, interactive: bool) -> Result<Self, String> {
        Self::new(Some(root.join(".vade").join(FILE_NAME)), vade_home.map(|home| home.join(FILE_NAME)), interactive)
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
    /// from. Only an Ask verdict whose argv is all plain literal words and
    /// whose findings may go on a list, and only a rule equal to the argv.
    pub fn allowed(&self, argv: &[String], verdict: &Verdict) -> Option<(Scope, Rule)> {
        if !policy::listable(argv, verdict) {
            return None;
        }
        let state = self.state();
        let lists = [(Scope::Session, &state.session), (Scope::Project, &state.project.rules), (Scope::User, &state.user.rules)];
        lists.into_iter().find_map(|(scope, rules)| rules.iter().find(|rule| rule.as_slice() == argv).map(|rule| (scope, rule.clone())))
    }
    /// Start a new, resumed or forked session: forget session rules and any
    /// pending answer, and read the project and user lists again, so a rule
    /// taken out of a file since startup no longer allows anything. A file
    /// that no longer reads leaves its list empty, so its commands ask, and
    /// gives an error.
    pub fn new_session(&self) -> Result<(), String> {
        let mut state = self.state();
        state.session.clear();
        state.choices.clear();
        let State { project, user, .. } = &mut *state;
        let mut errors = Vec::new();
        for list in [project, user] {
            let Some(path) = &list.path else { continue };
            list.rules = load(path).unwrap_or_else(|error| {
                errors.push(format!("{}: {error}", path.display()));
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
        };
        let path = list.path.clone().ok_or("no file for this list")?;
        // Another session can save the same file at the same time; the lock
        // keeps its read and write apart from ours (Astra on aacf3c9).
        let _lock = lock_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        // The file as it is now replaces the list, also when the rule is not
        // saved: a rule removed by hand no longer allows anything, and a
        // file that does not load allows nothing.
        let mut current = match load(&path) {
            Ok(rules) => rules,
            Err(error) => {
                list.rules.clear();
                return Err(format!("{}: {error}", path.display()));
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
        save(&path, &current).map_err(|error| format!("{}: {error}", path.display()))?;
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
        approvals.allowed(argv, &crate::shell::policy::decide(argv, Path::new("/w"), false))
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
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn rules_are_exact_commands_of_plain_words() {
        let file = |rule: &str| format!("{{\"version\":1,\"allow\":[{}]}}", json::quote(rule));
        for good in ["cargo fmt", "rm -rf build", "cargo +1.99.0 test --offline", "npm install left-pad@1.3.0", "git commit -m wip"] {
            assert_eq!(parse(&file(good)).unwrap(), vec![words(good)], "{good}");
        }
        // `*`, patterns, braces, variables, quotes, `~` and spaces are not
        // plain words: a rule with one does not load (Astra on 583ad99).
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
        // Blocked commands, root and credentials never use a rule.
        std::fs::write(&project, render(&[words("sudo ls"), words("cat .ssh/id_rsa"), words("rm -rf /")])).unwrap();
        let approvals = Approvals::new(Some(project), None, true).unwrap();
        for text in ["sudo ls", "cat .ssh/id_rsa", "rm -rf /"] {
            assert_eq!(allowed(&approvals, &words(text)), None, "{text}");
        }
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
    fn remembered_rules_reach_the_chosen_list() {
        let temp = Temp::new("remember");
        let project = temp.0.join(".vade").join(FILE_NAME);
        let approvals = Approvals::for_workspace(&temp.0, None, true).unwrap();
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
        // A second session in the same project reads the saved rule.
        let next = Approvals::for_workspace(&temp.0, None, false).unwrap();
        assert!(allowed(&next, &clean).is_some() && allowed(&next, &push).is_none());
        assert!(!next.interactive());
        assert!(approvals.remember(Scope::User, &rules).is_err());
        assert!(next.summary().contains("1 project"));
        // A new, resumed or forked session drops session rules; project rules stay.
        approvals.new_session().unwrap();
        assert_eq!(allowed(&approvals, &push), None);
        assert!(allowed(&approvals, &clean).is_some());
        // It reads the files again: a rule taken out of the file since
        // startup no longer allows, and a broken file allows nothing.
        std::fs::write(&project, "{\"version\":1,\"allow\":[]}").unwrap();
        assert!(allowed(&next, &clean).is_some());
        next.new_session().unwrap();
        assert!(allowed(&next, &clean).is_none());
        std::fs::write(&project, "{\"version\":1,\"allow\":[\"cargo clean\"]}").unwrap();
        next.new_session().unwrap();
        assert!(allowed(&next, &clean).is_some());
        std::fs::write(&project, "{\"version\":1,\"allow\":[7]}").unwrap();
        assert!(next.new_session().unwrap_err().contains(&project.display().to_string()));
        assert!(allowed(&next, &clean).is_none());
        // A `p` answer that cannot save still takes the file as it is now:
        // a rule removed by hand stops at once (Astra on 583ad99, item 9).
        std::fs::write(&project, render(&[clean.clone()])).unwrap();
        next.new_session().unwrap();
        assert!(allowed(&next, &clean).is_some());
        let full: Vec<Rule> = (0..MAX_RULES).map(|n| words(&format!("touch f{n}"))).collect();
        std::fs::write(&project, render(&full)).unwrap();
        assert!(next.remember(Scope::Project, &[words("touch extra")]).unwrap_err().contains("at most"));
        assert!(allowed(&next, &clean).is_none());
        assert!(allowed(&next, &words("touch f7")).is_some());
        std::fs::write(&project, "{\"version\":1,\"allow\":[7]}").unwrap();
        assert!(next.remember(Scope::Project, &[words("touch extra")]).is_err());
        assert!(allowed(&next, &words("touch f7")).is_none());
    }

    #[test]
    fn sessions_that_save_at_once_keep_every_rule() {
        // Astra on aacf3c9, item 3: each store reads the file, adds its rule
        // and renames a new file over it; without a lock, the last rename
        // drops the rules of the others.
        let temp = Temp::new("race");
        let project = temp.0.join(".vade").join(FILE_NAME);
        for round in 0..4 {
            let rules: Vec<Rule> = (0..16).map(|n| words(&format!("touch r{round}-{n}"))).collect();
            let stores: Vec<Approvals> = rules.iter().map(|_| Approvals::for_workspace(&temp.0, None, true).unwrap()).collect();
            let start = std::sync::Barrier::new(stores.len());
            std::thread::scope(|scope| {
                for (store, rule) in stores.iter().zip(&rules) {
                    let start = &start;
                    scope.spawn(move || {
                        start.wait();
                        store.remember(Scope::Project, std::slice::from_ref(rule)).unwrap();
                    });
                }
            });
            let saved = load(&project).unwrap();
            for rule in &rules {
                assert!(saved.contains(rule), "round {round}: {} lost", rule_text(rule));
            }
        }
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
        let approvals = Approvals::for_workspace(&workspace, None, true).unwrap();
        approvals.remember(Scope::Project, &[words("git push")]).unwrap();
        let long: Rule = (0..129).map(|i| format!("w{i}")).collect();
        assert!(approvals.remember(Scope::Project, &[long]).unwrap_err().contains("unchanged"));
        assert_eq!(load(&path).unwrap(), vec![words("git push")]);
    }
}
