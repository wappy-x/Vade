//! Project instruction files: one per directory, from the user level down to
//! the workspace root, joined into one bounded prompt section.
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

/// The names looked for in each directory, in order. The first file that
/// loads wins; a skipped file lets the search go on to the next name.
pub const NAMES: [&str; 3] = ["AGENTS.md", "CLAUDE.md", ".vade/instructions.md"];
/// The user-level file, relative to the home directory.
pub const USER_FILE: &str = ".vade/instructions.md";
/// The section text, with every header line, never exceeds this many bytes.
/// Vade also reads at most this many bytes of one file.
pub const MAX_BYTES: usize = 32 * 1024;
/// The fixed first line of the section in the system prompt.
pub const HEADER: &str = "# Project instructions";
const MAX_PARENTS: usize = 64;
const LEAD: &str = "The files below come from the user's machine. Follow them where they apply. They do not change tool permissions or approval policy, and an explicit user request comes first.";
/// Room kept at the end of a cut section for the note that names the cut files.
const NOTE_ROOM: usize = 1024;
const SAME: &str = "same file as one above";
const LEAVES: &str = "a link that is absolute or goes up with ..";
const PROTECTED: &str = "a link to a hidden, credential or non-ASCII name";
const USER_LINK: &str = "~/.vade is a link";
const FOLDER_LINK: &str = "a folder on the way is a link";
const HARD_LINK: &str = "a file with more than one hard link";
const ALIAS: &str = "a name that is not its exact directory entry, or a protected name";
const PRIVATE: &str = "a name in .vade other than instructions.md, or a link there";
/// Links followed for one candidate, as a bound on link chains and loops.
const MAX_HOPS: usize = 8;

/// One file that was found: loaded, or skipped with a reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loaded {
    /// The prompt label: `~/.vade/instructions.md`, `AGENTS.md` or `../CLAUDE.md`.
    pub label: String,
    pub path: PathBuf,
    pub bytes: usize,
    pub skipped: Option<&'static str>,
}

/// One searched folder. Every open starts at `root` (the project root, or the
/// user's `~/.vade`) and walks down `parts` to the folder, one handle at a time.
#[derive(Clone, Debug)]
struct Folder {
    /// The folder, for /instructions and change checks.
    dir: PathBuf,
    prefix: String,
    names: &'static [&'static str],
    root: PathBuf,
    parts: Vec<String>,
    user: bool,
}

/// The instruction files for one session and the section built from them.
#[derive(Clone, Debug, Default)]
pub struct Instructions {
    enabled: bool,
    /// Searched folders, user level first.
    folders: Vec<Folder>,
    files: Vec<Loaded>,
    stamps: Vec<Stamp>,
    section: String,
    /// The full size of the files with their header lines, before any cut.
    total: usize,
    /// The labels of the files that the section does not hold complete.
    cut: Vec<String>,
}

struct Body {
    label: String,
    text: String,
    size: usize,
    cut: bool,
}

/// The directories from the git root down to `root`, or only `root` when no
/// parent holds `.git`.
fn project_dirs(root: &Path) -> Vec<PathBuf> {
    let mut chain = vec![root.to_path_buf()];
    let mut dir = root;
    while !dir.join(".git").exists() {
        match dir.parent() {
            Some(parent) if chain.len() <= MAX_PARENTS => {
                chain.push(parent.to_path_buf());
                dir = parent;
            }
            _ => return vec![root.to_path_buf()],
        }
    }
    chain.reverse();
    chain
}

impl Instructions {
    /// No files and no section; `--no-instructions` uses this.
    pub fn disabled() -> Self {
        Self::default()
    }
    /// Reads the files for a workspace root. `home` is the user's home
    /// directory, if known.
    pub fn load(root: &Path, home: Option<&Path>) -> Self {
        let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        let mut folders = Vec::new();
        if let Some(home) = home {
            let dir = home.join(".vade");
            folders.push(Folder { dir: dir.clone(), prefix: "~/.vade/".into(), names: &["instructions.md"], root: dir, parts: Vec::new(), user: true });
        }
        let chain = project_dirs(&root);
        let (top, depth) = (chain[0].clone(), chain.len() - 1);
        for (index, dir) in chain.into_iter().enumerate() {
            // A folder name that is not UTF-8 cannot be walked by name; skip it.
            let Some(parts) = dir.strip_prefix(&top).ok().and_then(|rel| rel.iter().map(|p| p.to_str().map(str::to_owned)).collect()) else { continue };
            folders.push(Folder { dir, prefix: "../".repeat(depth - index), names: &NAMES, root: top.clone(), parts, user: false });
        }
        let mut result = Self {
            enabled: true,
            folders,
            ..Self::default()
        };
        result.read();
        result
    }
    /// A fresh stamp for every name, in the order `read` records them.
    fn probes(&self) -> impl Iterator<Item = Stamp> + '_ {
        self.folders.iter().flat_map(|f| f.names.iter().map(move |name| probe(f, name).0))
    }
    fn read(&mut self) {
        self.files.clear();
        self.stamps.clear();
        // Files already loaded, by device and inode.
        let mut seen: Vec<(u64, u64)> = Vec::new();
        let mut bodies = Vec::new();
        for folder in &self.folders {
            let mut done = false;
            for name in folder.names {
                // Every name gets a stamp, also after the folder's file loaded,
                // so a new file above it counts as a change.
                let (stamp, result) = probe(folder, name);
                self.stamps.push(stamp);
                if done {
                    continue;
                }
                let label = format!("{}{name}", folder.prefix);
                let mut file = Loaded { label, path: folder.dir.join(name), bytes: 0, skipped: None };
                match result {
                    Ok(None) => continue,
                    Ok(Some(opened)) if seen.contains(&opened.id) => file.skipped = Some(SAME),
                    Ok(Some(opened)) => match read_text(&opened.file) {
                        Ok((text, cut)) => {
                            seen.push(opened.id);
                            file.bytes = opened.size.max(text.len());
                            bodies.push(Body { label: file.label.clone(), text, size: file.bytes, cut });
                        }
                        Err(reason) => file.skipped = Some(reason),
                    },
                    Err(reason) => file.skipped = Some(reason),
                }
                done = matches!(file.skipped, None | Some(SAME));
                self.files.push(file);
            }
        }
        self.build(&bodies);
    }
    fn build(&mut self, bodies: &[Body]) {
        self.section.clear();
        self.cut.clear();
        self.total = 0;
        if bodies.is_empty() {
            return;
        }
        let mut text = format!("{HEADER}\n{LEAD}\n");
        self.total = text.len();
        // The byte range of each file's block, header line and marker included.
        let mut blocks = Vec::new();
        for body in bodies {
            let start = text.len();
            let head = format!("\n## {}\n", body.label);
            self.total += head.len() + body.size + 1;
            text.push_str(&head);
            text.push_str(body.text.trim_end());
            text.push('\n');
            if body.cut {
                // The marker stays even when trimming makes the kept text short.
                text.push_str(&format!("[{} is cut: Vade reads at most {MAX_BYTES} bytes of one file, and this file has {} bytes. The rest of the file is not sent.]\n", body.label, body.size));
                self.cut.push(body.label.clone());
            }
            blocks.push((start, text.len(), body.label.as_str()));
        }
        if text.len() > MAX_BYTES {
            let mut end = MAX_BYTES - NOTE_ROOM;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            let partial: Vec<_> = blocks.iter().filter(|b| b.0 < end && b.1 > end).map(|b| b.2).collect();
            let omitted: Vec<_> = blocks.iter().filter(|b| b.0 >= end).map(|b| b.2).collect();
            let mut note = format!("\n[Project instructions truncated at the {} KiB limit: the first {end} of {} bytes are shown.", MAX_BYTES / 1024, text.len());
            for label in &partial {
                note.push_str(&format!(" The end of {label} is cut."));
            }
            if !omitted.is_empty() {
                note.push_str(&format!(" Not sent: {}.", omitted.join(", ")));
            }
            if note.len() > NOTE_ROOM - 200 {
                note.truncate(note.find(" The end of").or_else(|| note.find(" Not sent:")).unwrap_or(note.len()));
                note.push_str(&format!(" {} files are cut or not sent, from {} on.", partial.len() + omitted.len(), partial.first().or(omitted.first()).unwrap_or(&"")));
            }
            note.push_str(" Shorten the files; /instructions lists them.]\n");
            for label in partial.into_iter().chain(omitted) {
                if !self.cut.iter().any(|c| c == label) {
                    self.cut.push(label.to_owned());
                }
            }
            text.truncate(end);
            text.push_str(&note);
        }
        self.section = text;
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    /// The complete section for the system prompt, or "" when no file loaded.
    pub fn section(&self) -> &str {
        &self.section
    }
    pub fn files(&self) -> &[Loaded] {
        &self.files
    }
    /// True when the section does not hold every loaded file complete.
    pub fn truncated(&self) -> bool {
        !self.cut.is_empty()
    }
    /// True when a candidate file was added, removed or changed its modified
    /// time or size, or a link on the way changed or was repaired, since the
    /// last read.
    pub fn changed(&self) -> bool {
        self.enabled && self.probes().ne(self.stamps.iter().cloned())
    }
    /// Reads again if a file changed. Returns one status line when the section
    /// text is different.
    pub fn refresh(&mut self) -> Option<String> {
        if !self.changed() {
            return None;
        }
        let before = std::mem::take(&mut self.section);
        self.read();
        (self.section != before).then(|| self.status())
    }
    /// The one status line for a changed section.
    pub fn status(&self) -> String {
        let names: Vec<_> = self.files.iter().filter(|f| f.skipped.is_none()).map(|f| f.label.as_str()).collect();
        if !self.enabled {
            "Project instructions are off (--no-instructions); the saved instructions no longer apply.".to_owned()
        } else if names.is_empty() {
            "Project instructions removed; no instruction file is found now.".to_owned()
        } else if self.truncated() {
            format!("Project instructions changed; reloaded {}. Not complete at the {} KiB limit: {}.", names.join(", "), MAX_BYTES / 1024, self.cut.join(", "))
        } else {
            format!("Project instructions changed; reloaded {}.", names.join(", "))
        }
    }
    /// The /instructions text.
    pub fn inspect(&self) -> String {
        if !self.enabled {
            return "Project instructions are off (--no-instructions).".into();
        }
        let mut text = String::from("Project instructions, in prompt order:\n");
        for f in &self.files {
            match f.skipped {
                None => {
                    let cut = if self.cut.contains(&f.label) { " · cut" } else { "" };
                    text.push_str(&format!("loaded {} · {} bytes{cut} · {}\n", f.label, f.bytes, f.path.display()));
                }
                Some(reason) => text.push_str(&format!("skipped {} · {reason} · {}\n", f.label, f.path.display())),
            }
        }
        if self.section.is_empty() {
            text.push_str(&format!("No instruction file found. Vade looks for {} in each folder from the git root down to the workspace, and for ~/{USER_FILE}.\n", NAMES.join(", ")));
        } else {
            text.push_str(&format!("Section: {} bytes", self.section.len()));
            if self.truncated() {
                text.push_str(&format!(", truncated from {} bytes at the {} KiB limit; not complete: {}", self.total, MAX_BYTES / 1024, self.cut.join(", ")));
            }
            text.push_str(". Vade reads the files again before a turn when they change.\n");
        }
        text
    }
}

/// The `name` of the user message that carries a changed section in an agent
/// session. The system message stays frozen, so the request stays append-only.
pub const UPDATE_NAME: &str = "vade_project_instructions";
const UPDATE_LEAD: &str = "Project instructions changed between turns. The text below replaces the Project instructions section of the system prompt and any root AGENTS.md rule in the frozen project rules.\n";
const UPDATE_EMPTY: &str = "Project instructions changed between turns. No instruction file is found now. Ignore the Project instructions section of the system prompt and any root AGENTS.md rule in the frozen project rules.";
const UPDATE_OFF: &str = "Project instructions are off for this session (--no-instructions). Ignore the Project instructions section of the system prompt and any root AGENTS.md rule in the frozen project rules.";

/// The update message for a new section. `enabled` is false for
/// `--no-instructions`.
pub fn update_message(section: &str, enabled: bool) -> crate::json::Value {
    use crate::json::Value;
    let content = match (section.is_empty(), enabled) {
        (false, _) => format!("{UPDATE_LEAD}{section}"),
        (true, true) => UPDATE_EMPTY.to_owned(),
        (true, false) => UPDATE_OFF.to_owned(),
    };
    Value::Object(vec![
        ("role".into(), Value::String("user".into())),
        ("name".into(), Value::String(UPDATE_NAME.into())),
        ("content".into(), Value::String(content)),
    ])
}

/// The last update message in a conversation. A child agent copies it, so it
/// follows the rules its parent follows now.
pub fn latest_update(c: &crate::agent::Conversation) -> Option<&crate::json::Value> {
    use crate::json::Value;
    c.messages().iter().rev().map(|m| m.value()).find(|v| {
        v.get("role").and_then(Value::as_str) == Some("user") && v.get("name").and_then(Value::as_str) == Some(UPDATE_NAME)
    })
}

/// True when the conversation gives the model other instructions than
/// `section`: the last update message, else the system prompt section. A
/// session made before the section existed froze the root AGENTS.md in its
/// project rules; it stays stale until one update message replaces that rule.
pub fn stale(c: &crate::agent::Conversation, section: &str) -> bool {
    use crate::json::Value;
    if let Some(update) = latest_update(c) {
        let text = update.get("content").and_then(Value::as_str).unwrap_or("");
        return text.strip_prefix(UPDATE_LEAD).unwrap_or("") != section;
    }
    let system = c.messages().first().and_then(|m| m.value().get("content")).and_then(Value::as_str).unwrap_or("");
    let legacy = crate::project::Snapshot::from_conversation(c).is_ok_and(|s| s.has_root_rule());
    legacy || crate::prompt::split_instructions(system).1 != section
}

/// One opened instruction file: its device and inode, size and modified time.
struct Opened {
    id: (u64, u64),
    file: std::fs::File,
    size: usize,
    modified: Option<SystemTime>,
}

/// What a candidate resolved to, for change checks: the text of each link on
/// the way, then the final file's identity, size and modified time, or the
/// reason it was skipped. Repairing or breaking a link changes the stamp.
#[derive(Clone, Debug, PartialEq)]
struct Stamp {
    links: Vec<String>,
    found: Result<Option<((u64, u64), usize, Option<SystemTime>)>, &'static str>,
}

fn probe(folder: &Folder, name: &str) -> (Stamp, Result<Option<Opened>, &'static str>) {
    let mut links = Vec::new();
    let result = open_file(folder, name, &mut links);
    let found = match &result {
        Ok(Some(o)) => Ok(Some((o.id, o.size, o.modified))),
        Ok(None) => Ok(None),
        Err(reason) => Err(*reason),
    };
    (Stamp { links, found }, result)
}

/// A folder name between the project root and the workspace: any name, but
/// it must be the exact directory entry.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn any_name(_: &str) -> bool {
    true
}
/// An instruction name or a name from a link: the workspace reader's rule,
/// plus `.vade`, whose contents `open_file` limits to the documented file.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn instruction_name(name: &str) -> bool {
    name == ".vade" || crate::tools::allowed_component(name)
}

/// The names in a link's text. No absolute target and no `..` step. Each name
/// must pass the workspace reader's rule; `.vade` is allowed only as the
/// documented path `.vade/instructions.md` at the end of the text.
fn link_parts(target: &Path) -> Result<Vec<String>, &'static str> {
    let mut parts = Vec::new();
    for component in target.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(text) => parts.push(text.to_str().ok_or(PROTECTED)?.to_owned()),
            _ => return Err(LEAVES),
        }
    }
    for (index, part) in parts.iter().enumerate() {
        let documented = part == ".vade" && index + 2 == parts.len() && parts[index + 1] == "instructions.md";
        if !documented && !crate::tools::allowed_component(part) {
            return Err(PROTECTED);
        }
    }
    Ok(parts)
}

#[cfg(test)]
thread_local! {
    /// Called with each folder name right after the walk opens it; a test
    /// swaps folders here to replay a race at an exact point.
    static AFTER_OPEN: std::cell::RefCell<Option<Box<dyn FnMut(&str)>>> = const { std::cell::RefCell::new(None) };
}

/// Opens one candidate, or `None` when nothing is there. The walk starts at
/// the folder's root handle and opens one name at a time without following a
/// link, so the file it reads is the file it checked: a folder or file that
/// is replaced by a link after a check is refused at the open. Each opened
/// name must be its exact directory entry. A link is resolved by its text
/// (see `link_parts`), and every link on the way is checked and recorded in
/// `links`. Inside a `.vade` folder only `instructions.md` opens, and it may
/// not be a link: that folder also holds private chat state. The opened
/// descriptor must be a regular file with one hard link, and the open never
/// waits on a FIFO.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn open_file(folder: &Folder, name: &str, links: &mut Vec<String>) -> Result<Option<Opened>, &'static str> {
    use crate::tools::anchored::{Dir, Entry};
    use std::os::unix::fs::MetadataExt;
    match std::fs::symlink_metadata(&folder.root) {
        Ok(meta) if meta.file_type().is_symlink() => return Err(if folder.user { USER_LINK } else { FOLDER_LINK }),
        Ok(_) => {}
        Err(_) => return Ok(None),
    }
    let mut dir = Dir::open_root(&folder.root).map_err(|_| "cannot read")?;
    // The path of `dir`, so a link's text can be read by path. The text only
    // names what to open next; every open uses the handles.
    let mut at = folder.root.clone();
    for part in &folder.parts {
        match dir.entry(part, any_name).map_err(|_| "cannot read")? {
            Entry::Dir(child) => dir = child,
            Entry::Link => return Err(FOLDER_LINK),
            Entry::Alias => return Err(ALIAS),
            Entry::Missing | Entry::Other(_) => return Ok(None),
        }
        at.push(part);
    }
    let mut in_vade = folder.user;
    let mut pending: std::collections::VecDeque<String> = name.split('/').map(str::to_owned).collect();
    let mut hops = 0;
    let missing = |hops| if hops == 0 { Ok(None) } else { Err("link target is missing") };
    while let Some(part) = pending.pop_front() {
        if in_vade && part != "instructions.md" {
            return Err(PRIVATE);
        }
        match dir.entry(&part, instruction_name).map_err(|_| "cannot read")? {
            Entry::Missing => return missing(hops),
            Entry::Alias => return Err(ALIAS),
            Entry::Link if in_vade => return Err(PRIVATE),
            Entry::Link => {
                hops += 1;
                if hops > MAX_HOPS {
                    return Err("too many links");
                }
                let target = std::fs::read_link(at.join(&part)).map_err(|_| "cannot read")?;
                links.push(target.to_string_lossy().into_owned());
                for text in link_parts(&target)?.into_iter().rev() {
                    pending.push_front(text);
                }
                if pending.is_empty() {
                    return Err("not a regular file");
                }
            }
            Entry::Dir(child) if !pending.is_empty() => {
                #[cfg(test)]
                AFTER_OPEN.with(|hook| hook.borrow_mut().as_mut().map(|hook| hook(&part)));
                in_vade = part == ".vade";
                dir = child;
                at.push(part);
            }
            Entry::Dir(_) => return Err("not a regular file"),
            Entry::Other(_) if !pending.is_empty() => return missing(hops),
            Entry::Other(file) => {
                let meta = file.metadata().map_err(|_| "cannot read")?;
                if !meta.is_file() {
                    return Err("not a regular file");
                }
                if meta.nlink() != 1 {
                    return Err(HARD_LINK);
                }
                return Ok(Some(Opened { id: (meta.dev(), meta.ino()), size: meta.len() as usize, modified: meta.modified().ok(), file }));
            }
        }
    }
    Ok(None)
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn open_file(_: &Folder, _: &str, _: &mut Vec<String>) -> Result<Option<Opened>, &'static str> {
    Err("instruction files require macOS or Linux")
}

/// At most `MAX_BYTES` of UTF-8 text, and true when the file was longer.
fn read_text(file: &std::fs::File) -> Result<(String, bool), &'static str> {
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64 + 1).read_to_end(&mut bytes).map_err(|_| "cannot read")?;
    let cut = bytes.len() > MAX_BYTES;
    if cut {
        // Keep a valid UTF-8 prefix; the section states the cut.
        bytes.truncate(MAX_BYTES);
        let valid = match std::str::from_utf8(&bytes) {
            Ok(_) => bytes.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => return Err("not UTF-8 text"),
        };
        bytes.truncate(valid);
    }
    String::from_utf8(bytes).map(|text| (text, cut)).map_err(|_| "not UTF-8 text")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_workspace::Fixture;
    use std::os::unix::fs::symlink;
    /// repo/.git, repo/AGENTS.md, repo/app/CLAUDE.md, repo/app/web (workspace)
    /// with .vade/instructions.md, and home/.vade/instructions.md.
    fn tree() -> Fixture {
        let f = Fixture::new();
        for dir in ["home/.vade", "repo/.git", "repo/app/web/.vade"] {
            std::fs::create_dir_all(f.0.join(dir)).unwrap();
        }
        f.write("home/.vade/instructions.md", "user rule\n");
        f.write("repo/AGENTS.md", "root rule\n");
        f.write("repo/app/CLAUDE.md", "app rule\n");
        f.write("repo/app/web/.vade/instructions.md", "web rule\n");
        f
    }
    fn load(f: &Fixture) -> Instructions {
        Instructions::load(&f.0.join("repo/app/web"), Some(&f.0.join("home")))
    }
    fn labels(i: &Instructions) -> Vec<&str> {
        i.files().iter().filter(|f| f.skipped.is_none()).map(|f| f.label.as_str()).collect()
    }
    fn skipped<'a>(i: &'a Instructions, label: &str) -> Option<&'a str> {
        i.files().iter().find(|f| f.label == label).and_then(|f| f.skipped)
    }
    #[test]
    fn order_is_user_then_root_then_nearer_with_one_header_per_file() {
        let f = tree();
        let i = load(&f);
        assert_eq!(labels(&i), ["~/.vade/instructions.md", "../../AGENTS.md", "../CLAUDE.md", ".vade/instructions.md"]);
        let s = i.section();
        assert!(s.starts_with("# Project instructions\n"));
        let at = |t| s.find(t).unwrap();
        assert!(at("## ~/.vade/instructions.md\nuser rule") < at("## ../../AGENTS.md\nroot rule"));
        assert!(at("root rule") < at("## ../CLAUDE.md\napp rule"));
        assert!(at("app rule") < at("## .vade/instructions.md\nweb rule"));
        assert_eq!(s.matches("\n## ").count(), 4);
        assert!(!s.contains(f.0.to_str().unwrap()), "no absolute paths in the prompt");
        assert!(i.inspect().contains("loaded ../CLAUDE.md · 9 bytes"));
    }
    #[test]
    fn first_name_wins_in_a_folder_and_search_stops_at_the_git_root() {
        let f = tree();
        f.write("AGENTS.md", "above the git root\n");
        f.write("repo/CLAUDE.md", "second name\n");
        f.write("repo/app/web/AGENTS.md", "web agents\n");
        let i = load(&f);
        let s = i.section();
        assert!(s.contains("root rule") && s.contains("web agents"));
        assert!(!s.contains("second name") && !s.contains("web rule") && !s.contains("above the git root"));
        // A skipped file does not stop the search: the next name loads.
        std::fs::write(f.0.join("repo/AGENTS.md"), [0xff, 0xfe]).unwrap();
        let i = load(&f);
        assert_eq!(skipped(&i, "../../AGENTS.md"), Some("not UTF-8 text"));
        assert!(i.section().contains("## ../../CLAUDE.md\nsecond name"));
        // No git root: only the workspace folder and the user file.
        std::fs::remove_dir(f.0.join("repo/.git")).unwrap();
        assert_eq!(labels(&load(&f)), ["~/.vade/instructions.md", "AGENTS.md"]);
        let none = Instructions::load(&f.0.join("repo/app/web"), None);
        assert_eq!(labels(&none), ["AGENTS.md"]);
    }
    #[test]
    fn a_link_to_a_loaded_file_is_not_repeated_and_bad_files_are_skipped() {
        let f = tree();
        // A link down into a subfolder is allowed; the subfolder then finds
        // the same file again and does not repeat it.
        std::fs::remove_file(f.0.join("repo/app/CLAUDE.md")).unwrap();
        symlink("web/.vade/instructions.md", f.0.join("repo/app/CLAUDE.md")).unwrap();
        std::fs::write(f.0.join("repo/app/web/AGENTS.md"), [0xff, 0xfe]).unwrap();
        std::fs::create_dir(f.0.join("repo/app/web/CLAUDE.md")).unwrap();
        let i = load(&f);
        assert_eq!(labels(&i), ["~/.vade/instructions.md", "../../AGENTS.md", "../CLAUDE.md"]);
        assert_eq!(skipped(&i, "AGENTS.md"), Some("not UTF-8 text"));
        assert_eq!(skipped(&i, "CLAUDE.md"), Some("not a regular file"));
        assert_eq!(skipped(&i, ".vade/instructions.md"), Some(SAME));
        assert_eq!(i.section().matches("web rule").count(), 1);
        assert!(i.inspect().contains("skipped AGENTS.md · not UTF-8 text"));
    }
    /// Review finding 1: a link inside an instruction location must not pull
    /// a file from outside the project (or outside ~/.vade) into the prompt.
    #[test]
    fn a_link_cannot_pull_a_file_from_outside_the_project() {
        let f = tree();
        const MARK: &str = "outside-secret-marker";
        std::fs::create_dir_all(f.0.join("outside/.vade")).unwrap();
        f.write("private.txt", MARK);
        f.write("outside/rules.md", MARK);
        f.write("outside/.vade/instructions.md", MARK);
        let secret = f.0.join("private.txt");
        let replace = |path: &str| {
            let path = f.0.join(path);
            let _ = std::fs::remove_file(&path);
            path
        };
        // Root folder: an absolute link, then a hard link, then a plain file.
        symlink(&secret, replace("repo/AGENTS.md")).unwrap();
        std::fs::hard_link(&secret, f.0.join("repo/CLAUDE.md")).unwrap();
        std::fs::create_dir(f.0.join("repo/.vade")).unwrap();
        f.write("repo/.vade/instructions.md", "root rule kept");
        // app: a link that goes up, a link through a folder link that leaves
        // the project, and a .vade folder that is an absolute link.
        symlink("../../private.txt", f.0.join("repo/app/AGENTS.md")).unwrap();
        symlink(f.0.join("outside"), f.0.join("repo/app/out")).unwrap();
        symlink("out/rules.md", replace("repo/app/CLAUDE.md")).unwrap();
        symlink(f.0.join("outside/.vade"), f.0.join("repo/app/.vade")).unwrap();
        // web: a link down inside the workspace still loads.
        std::fs::create_dir(f.0.join("repo/app/web/notes")).unwrap();
        f.write("repo/app/web/notes/rules.md", "linked down rule");
        symlink("notes/rules.md", f.0.join("repo/app/web/AGENTS.md")).unwrap();
        // User level: ~/.vade/instructions.md may not be a link at all.
        symlink(&secret, replace("home/.vade/instructions.md")).unwrap();
        let i = load(&f);
        assert!(!i.section().contains(MARK), "{}", i.section());
        assert_eq!(skipped(&i, "~/.vade/instructions.md"), Some(PRIVATE));
        assert_eq!(skipped(&i, "../../AGENTS.md"), Some(LEAVES));
        assert_eq!(skipped(&i, "../../CLAUDE.md"), Some(HARD_LINK));
        assert_eq!(skipped(&i, "../AGENTS.md"), Some(LEAVES));
        // The second link on the way, out -> an absolute folder, is checked too.
        assert_eq!(skipped(&i, "../CLAUDE.md"), Some(LEAVES));
        assert_eq!(skipped(&i, "../.vade/instructions.md"), Some(LEAVES));
        assert_eq!(labels(&i), ["../../.vade/instructions.md", "AGENTS.md"]);
        assert!(i.section().contains("root rule kept") && i.section().contains("linked down rule"));
        // A relative link inside ~/.vade is refused too.
        symlink("../../private.txt", replace("home/.vade/instructions.md")).unwrap();
        assert_eq!(skipped(&load(&f), "~/.vade/instructions.md"), Some(PRIVATE));
    }
    /// Review finding 2: the open itself must not wait on a FIFO, so a file
    /// replaced by a FIFO after any check cannot hang startup or a refresh.
    #[test]
    fn a_fifo_in_place_of_a_file_never_blocks() {
        let f = tree();
        let fifo = f.0.join("repo/AGENTS.md");
        std::fs::remove_file(&fifo).unwrap();
        assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        f.write("repo/CLAUDE.md", "next name");
        let root = std::fs::canonicalize(f.0.join("repo")).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let web = f.0.join("repo/app/web");
        std::thread::spawn(move || {
            // The open that follows the checks, as the race would meet it.
            let folder = Folder { dir: root.clone(), prefix: String::new(), names: &NAMES, root, parts: Vec::new(), user: false };
            let opened = open_file(&folder, "AGENTS.md", &mut Vec::new()).map(|found| found.is_some());
            let section = Instructions::load(&web, None).section().to_owned();
            let _ = tx.send((opened, section));
        });
        let (opened, section) = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a FIFO blocked the read");
        assert_eq!(opened, Err("not a regular file"));
        assert!(section.contains("## ../../CLAUDE.md\nnext name"), "{section}");
    }
    /// Review 2 finding 3: every link on the way is checked, not only the
    /// first, and a linked ~/.vade is not a trusted start folder.
    #[test]
    fn every_link_in_a_chain_and_a_linked_user_folder_are_checked() {
        let f = tree();
        const MARK: &str = "chain-private-marker";
        f.write("repo/private.txt", MARK);
        // repo/app/AGENTS.md -> rules.md -> ../private.txt: the final file is
        // inside the repo, but the second link goes up.
        symlink("rules.md", f.0.join("repo/app/AGENTS.md")).unwrap();
        symlink("../private.txt", f.0.join("repo/app/rules.md")).unwrap();
        // A chain that stays down still loads.
        std::fs::create_dir_all(f.0.join("repo/app/web/a/b")).unwrap();
        f.write("repo/app/web/a/b/end.md", "two hops down");
        symlink("a/step.md", f.0.join("repo/app/web/AGENTS.md")).unwrap();
        symlink("b/end.md", f.0.join("repo/app/web/a/step.md")).unwrap();
        // A link loop stops at the hop limit.
        symlink("CLAUDE.md", f.0.join("repo/CLAUDE.md")).unwrap();
        std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
        // ~/.vade itself is an absolute link to a folder outside home.
        std::fs::create_dir(f.0.join("elsewhere")).unwrap();
        f.write("elsewhere/instructions.md", MARK);
        std::fs::remove_dir_all(f.0.join("home/.vade")).unwrap();
        symlink(f.0.join("elsewhere"), f.0.join("home/.vade")).unwrap();
        let i = load(&f);
        assert!(!i.section().contains(MARK), "{}", i.section());
        assert_eq!(skipped(&i, "../AGENTS.md"), Some(LEAVES));
        assert_eq!(skipped(&i, "../../CLAUDE.md"), Some("too many links"));
        assert_eq!(skipped(&i, "~/.vade/instructions.md"), Some(USER_LINK));
        assert!(i.section().contains("## AGENTS.md\ntwo hops down"), "{}", i.section());
        assert!(i.section().contains("## ../CLAUDE.md\napp rule"), "the next name still loads");
    }
    /// Review 2 finding 2: a link may not name a file that a workspace read
    /// refuses (credential names, hidden names), at any hop.
    #[test]
    fn a_link_cannot_name_a_protected_file() {
        let f = tree();
        const MARK: &str = "protected-secret-marker";
        f.write("repo/.env", MARK);
        f.write("repo/credentials.json", MARK);
        f.write("repo/app/server.pem", MARK);
        f.write("repo/app/web/plain.md", "plain link target");
        symlink(".env", f.0.join("repo/CLAUDE.md")).unwrap();
        std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
        symlink("credentials.json", f.0.join("repo/AGENTS.md")).unwrap();
        // A chain that reaches a protected name at its second hop.
        symlink("step.md", f.0.join("repo/app/AGENTS.md")).unwrap();
        symlink("server.pem", f.0.join("repo/app/step.md")).unwrap();
        symlink("plain.md", f.0.join("repo/app/web/AGENTS.md")).unwrap();
        let i = load(&f);
        assert!(!i.section().contains(MARK), "{}", i.section());
        assert_eq!(skipped(&i, "../../AGENTS.md"), Some(PROTECTED));
        assert_eq!(skipped(&i, "../../CLAUDE.md"), Some(PROTECTED));
        assert_eq!(skipped(&i, "../AGENTS.md"), Some(PROTECTED));
        assert!(i.section().contains("## AGENTS.md\nplain link target"));
        // The protected-name rule is the workspace reader's own rule.
        for name in [".env", "credentials.json", "server.pem"] {
            assert!(!crate::tools::allowed_component(name), "{name}");
        }
    }
    /// Review 3 finding 1: a link may reach `.vade` only as the documented
    /// `.vade/instructions.md`; with --state-in-project that folder also holds
    /// private input recall, which must never reach the prompt.
    #[test]
    fn a_link_reaches_vade_only_at_the_documented_file() {
        let f = tree();
        const MARK: &str = "private-input-marker";
        f.write("repo/app/web/.vade/history", MARK);
        f.write("home/.vade/history", MARK);
        // Direct, through a folder link, and from the documented file itself.
        symlink(".vade/history", f.0.join("repo/app/web/AGENTS.md")).unwrap();
        symlink(".vade", f.0.join("repo/app/web/state")).unwrap();
        symlink("state/history", f.0.join("repo/app/web/CLAUDE.md")).unwrap();
        std::fs::remove_file(f.0.join("repo/app/web/.vade/instructions.md")).unwrap();
        symlink("history", f.0.join("repo/app/web/.vade/instructions.md")).unwrap();
        std::fs::remove_file(f.0.join("home/.vade/instructions.md")).unwrap();
        symlink("history", f.0.join("home/.vade/instructions.md")).unwrap();
        let i = load(&f);
        assert!(!i.section().contains(MARK), "{}", i.section());
        assert_eq!(skipped(&i, "AGENTS.md"), Some(PROTECTED));
        assert_eq!(skipped(&i, "CLAUDE.md"), Some(PROTECTED));
        assert_eq!(skipped(&i, ".vade/instructions.md"), Some(PRIVATE));
        assert_eq!(skipped(&i, "~/.vade/instructions.md"), Some(PRIVATE));
        // The documented file is still a valid link target.
        std::fs::remove_file(f.0.join("repo/app/CLAUDE.md")).unwrap();
        std::fs::create_dir_all(f.0.join("repo/app/docs/.vade")).unwrap();
        f.write("repo/app/docs/.vade/instructions.md", "documented target");
        symlink("docs/.vade/instructions.md", f.0.join("repo/app/CLAUDE.md")).unwrap();
        assert!(load(&f).section().contains("## ../CLAUDE.md\ndocumented target"));
        assert_eq!(link_parts(Path::new("docs/.vade/history")), Err(PROTECTED));
        assert_eq!(link_parts(Path::new(".vade/instructions.md/x")), Err(PROTECTED));
    }
    /// Review 3 finding 2: the opened name must be its exact directory entry,
    /// so a folding filesystem cannot open `picK.txt` for `pick.txt`.
    #[test]
    fn a_link_cannot_open_a_folded_alias() {
        let f = tree();
        const MARK: &str = "folded-alias-marker";
        f.write("repo/picK.txt", MARK);
        std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
        symlink("pick.txt", f.0.join("repo/AGENTS.md")).unwrap();
        let i = load(&f);
        assert!(!i.section().contains(MARK), "{}", i.section());
        // Only a folding filesystem makes pick.txt an alias; elsewhere the
        // link target is missing.
        let folds = f.0.join("repo/pick.txt").exists();
        assert_eq!(skipped(&i, "../../AGENTS.md"), Some(if folds { ALIAS } else { "link target is missing" }));
    }
    /// Review 3 finding 3: repairing a skipped link, with no change to the
    /// target file, reloads; breaking it again removes the rules.
    #[test]
    fn repairing_or_breaking_a_link_reloads() {
        let f = tree();
        f.write("repo/rules.md", "repaired rule");
        std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
        symlink(f.0.join("repo/rules.md"), f.0.join("repo/AGENTS.md")).unwrap();
        let mut i = load(&f);
        assert_eq!(skipped(&i, "../../AGENTS.md"), Some(LEAVES));
        assert!(!i.changed());
        let relink = |target: &Path| {
            std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
            symlink(target, f.0.join("repo/AGENTS.md")).unwrap();
        };
        relink(Path::new("rules.md"));
        assert!(i.changed());
        assert!(i.refresh().unwrap().contains("../../AGENTS.md"));
        assert!(i.section().contains("## ../../AGENTS.md\nrepaired rule"));
        assert!(!i.changed());
        relink(&f.0.join("repo/rules.md"));
        assert!(i.refresh().is_some());
        assert!(!i.section().contains("repaired rule"));
        assert_eq!(skipped(&i, "../../AGENTS.md"), Some(LEAVES));
    }
    /// Review 3 finding 4: the swap happens at an exact point of the full
    /// loader (right after it opened `rules`), with no timing. The loader reads
    /// through the handle it holds, so the outside file never loads; the next
    /// load meets the swapped name as a link and refuses it.
    #[test]
    fn a_folder_swapped_inside_the_loader_never_leads_outside() {
        let f = tree();
        const MARK: &str = "swapped-outside-marker";
        std::fs::create_dir_all(f.0.join("outside")).unwrap();
        f.write("outside/file.md", MARK);
        std::fs::create_dir(f.0.join("repo/rules")).unwrap();
        f.write("repo/rules/file.md", "inside rule");
        std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
        symlink("rules/file.md", f.0.join("repo/AGENTS.md")).unwrap();
        let (repo, outside) = (f.0.join("repo"), f.0.join("outside"));
        let swaps = std::rc::Rc::new(std::cell::Cell::new(0));
        let counter = swaps.clone();
        AFTER_OPEN.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |part: &str| {
                if part == "rules" && counter.get() == 0 {
                    std::fs::rename(repo.join("rules"), repo.join("rules.real")).unwrap();
                    symlink(&outside, repo.join("rules")).unwrap();
                    counter.set(1);
                }
            }));
        });
        let first = Instructions::load(&f.0.join("repo/app/web"), None);
        let second = Instructions::load(&f.0.join("repo/app/web"), None);
        AFTER_OPEN.with(|hook| *hook.borrow_mut() = None);
        assert_eq!(swaps.get(), 1, "the swap ran inside the first load");
        assert!(first.section().contains("## ../../AGENTS.md\ninside rule"), "{}", first.section());
        assert!(!first.section().contains(MARK) && !second.section().contains(MARK));
        assert_eq!(skipped(&second, "../../AGENTS.md"), Some(LEAVES));
    }
    /// Review 2 finding 1, made exact: the walk holds a handle to each folder
    /// it checked. A swap after that point cannot redirect the open, while a
    /// second open by path (the f0cc43a code) reads the outside file.
    #[test]
    fn a_held_folder_handle_is_not_redirected_by_a_swap() {
        use crate::tools::anchored::{Dir, Entry};
        let f = tree();
        const MARK: &str = "swapped-outside-marker";
        std::fs::create_dir_all(f.0.join("outside")).unwrap();
        f.write("outside/file.md", MARK);
        std::fs::create_dir(f.0.join("repo/rules")).unwrap();
        f.write("repo/rules/file.md", "inside rule");
        let repo = Dir::open_root(&f.0.join("repo")).unwrap();
        let Entry::Dir(rules) = repo.entry("rules", instruction_name).unwrap() else { panic!("rules is a folder") };
        // Another process swaps the checked folder for a link to an outside folder.
        std::fs::rename(f.0.join("repo/rules"), f.0.join("repo/rules.real")).unwrap();
        symlink(f.0.join("outside"), f.0.join("repo/rules")).unwrap();
        let Entry::Other(file) = rules.entry("file.md", instruction_name).unwrap() else { panic!("file.md is a file") };
        assert_eq!(read_text(&file).unwrap().0, "inside rule");
        assert_eq!(std::fs::read_to_string(f.0.join("repo/rules/file.md")).unwrap(), MARK, "an open by path follows the swap");
        // The loader meets the swapped state as a link and refuses it.
        assert!(matches!(repo.entry("rules", instruction_name).unwrap(), Entry::Link));
        std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
        symlink("rules/file.md", f.0.join("repo/AGENTS.md")).unwrap();
        let i = load(&f);
        assert_eq!(skipped(&i, "../../AGENTS.md"), Some(LEAVES));
        assert!(!i.section().contains(MARK));
    }
    /// Review 2 finding 1: a folder on the way that another process swaps for
    /// a link to an outside folder, again and again while the files load, must
    /// never put the outside file in the prompt. The open walks handles and
    /// never follows a link, so no check-then-open gap exists to hit. This is a
    /// timing stress run, kept out of CI; the synchronized test above is the
    /// regression test. Run it with `cargo test -- --ignored`.
    #[test]
    #[ignore = "timing stress; run with --ignored"]
    fn a_folder_swapped_for_a_link_during_reads_never_leads_outside() {
        let f = tree();
        const MARK: &str = "swapped-outside-marker";
        std::fs::create_dir_all(f.0.join("outside")).unwrap();
        f.write("outside/file.md", MARK);
        std::fs::create_dir(f.0.join("repo/rules")).unwrap();
        f.write("repo/rules/file.md", "inside rule");
        std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
        symlink("rules/file.md", f.0.join("repo/AGENTS.md")).unwrap();
        symlink(f.0.join("outside"), f.0.join("repo/rules.link")).unwrap();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let repo = f.0.join("repo");
        let flipper = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut swaps = 0u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    std::fs::rename(repo.join("rules"), repo.join("rules.real")).unwrap();
                    std::fs::rename(repo.join("rules.link"), repo.join("rules")).unwrap();
                    std::fs::rename(repo.join("rules"), repo.join("rules.link")).unwrap();
                    std::fs::rename(repo.join("rules.real"), repo.join("rules")).unwrap();
                    swaps += 1;
                }
                swaps
            })
        };
        let (mut inside, mut refused) = (0, 0);
        let start = std::time::Instant::now();
        while start.elapsed() < std::time::Duration::from_millis(1500) {
            let i = Instructions::load(&f.0.join("repo/app/web"), None);
            assert!(!i.section().contains(MARK), "an outside file reached the prompt");
            if i.section().contains("inside rule") {
                inside += 1;
            } else {
                refused += 1;
            }
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let swaps = flipper.join().unwrap();
        eprintln!("swaps {swaps}, loads with the inside rule {inside}, without {refused}");
    }
    #[test]
    fn the_section_stops_at_32_kib_and_names_what_is_cut() {
        let f = tree();
        f.write("repo/AGENTS.md", &"é".repeat(MAX_BYTES));
        let i = load(&f);
        assert!(i.truncated());
        let s = i.section();
        assert!(s.len() <= MAX_BYTES);
        assert!(s.contains("[Project instructions truncated at the 32 KiB limit: the first "));
        assert!(s.contains("The end of ../../AGENTS.md is cut. Not sent: ../CLAUDE.md, .vade/instructions.md."), "{}", &s[s.len() - 400..]);
        assert!(!s.contains("web rule"));
        let shown = i.inspect();
        assert!(shown.contains("truncated from") && shown.contains("not complete: ../../AGENTS.md, ../CLAUDE.md, .vade/instructions.md"), "{shown}");
        assert!(shown.contains(&format!("loaded ../../AGENTS.md · {} bytes · cut", 2 * MAX_BYTES)), "{shown}");
    }
    /// Review finding 6: whitespace before the cut must not hide a dropped rule.
    #[test]
    fn a_cut_file_keeps_a_marker_even_when_the_kept_text_is_short() {
        let f = tree();
        let body = format!("start\n{}mandatory final rule\n", " ".repeat(MAX_BYTES));
        f.write("repo/AGENTS.md", &body);
        let i = load(&f);
        let s = i.section();
        assert!(s.len() < 4096, "trimming leaves a short section");
        assert!(!s.contains("mandatory final rule"));
        assert!(s.contains(&format!("## ../../AGENTS.md\nstart\n[../../AGENTS.md is cut: Vade reads at most {MAX_BYTES} bytes of one file, and this file has {} bytes. The rest of the file is not sent.]\n", body.len())), "{s}");
        assert!(s.contains("web rule"), "the other files stay");
        assert!(i.truncated());
        let shown = i.inspect();
        assert!(shown.contains("not complete: ../../AGENTS.md") && shown.contains("· cut ·"), "{shown}");
        // The total counts the full file, not the trimmed text.
        assert!(shown.contains(&format!("truncated from {} bytes", i.total)) && i.total > body.len(), "{shown}");
        // A reload states the cut in its status line too.
        let mut i = Instructions::load(&f.0.join("repo/app/web"), None);
        f.write("repo/app/CLAUDE.md", "app rule, longer now\n");
        assert!(i.refresh().unwrap().ends_with("Not complete at the 32 KiB limit: ../../AGENTS.md."));
    }
    #[test]
    fn a_change_between_turns_reloads_once_with_one_status_line() {
        let f = tree();
        let mut i = load(&f);
        assert!(!i.changed());
        assert_eq!(i.refresh(), None);
        f.write("repo/app/CLAUDE.md", "app rule, longer now\n");
        assert!(i.changed());
        let line = i.refresh().unwrap();
        assert_eq!(line, "Project instructions changed; reloaded ~/.vade/instructions.md, ../../AGENTS.md, ../CLAUDE.md, .vade/instructions.md.");
        assert!(i.section().contains("app rule, longer now"));
        assert_eq!(i.refresh(), None);
        // A new file in a nearer folder is a change too.
        f.write("repo/app/AGENTS.md", "new app agents\n");
        assert!(i.refresh().is_some());
        assert!(i.section().contains("new app agents") && !i.section().contains("app rule"));
        for path in ["home/.vade/instructions.md", "repo/AGENTS.md", "repo/app/AGENTS.md", "repo/app/CLAUDE.md", "repo/app/web/.vade/instructions.md"] {
            std::fs::remove_file(f.0.join(path)).unwrap();
        }
        assert_eq!(i.refresh().unwrap(), "Project instructions removed; no instruction file is found now.");
        assert_eq!(i.section(), "");
        let mut off = Instructions::disabled();
        assert!(!off.changed() && off.refresh().is_none() && off.section().is_empty());
        assert!(off.inspect().contains("--no-instructions"));
        assert!(off.status().contains("off (--no-instructions)"));
    }
    /// A conversation as native_conversation makes it: the system prompt with
    /// `section`, then the frozen project rules. `root_rules` makes the
    /// snapshot that sessions saved before the section existed carry.
    fn conversation(f: &Fixture, section: &str, root_rules: bool) -> crate::agent::Conversation {
        use crate::json::Value;
        let workspace = crate::tools::Workspace::new(&f.0.join("repo")).unwrap();
        let project = crate::project::Snapshot::discover_with(&workspace, &[], root_rules).unwrap();
        let message = |role: &str, content: String| Value::Object(vec![("role".into(), Value::String(role.into())), ("content".into(), Value::String(content))]);
        let request = Value::Object(vec![
            ("model".into(), Value::String("mock".into())),
            ("messages".into(), Value::Array(vec![message("system", crate::prompt::with_instructions("Built in.", section)), project.message().unwrap()])),
        ]);
        crate::agent::Conversation::from_request(request, crate::agent::Limits::default()).unwrap()
    }
    /// Review findings 3 and 4: resume with --no-instructions withdraws saved
    /// instructions, and an older session's frozen root rules are reconciled.
    #[test]
    fn resume_reconciles_saved_and_older_root_rules_with_the_files() {
        let f = tree();
        let limits = crate::agent::Limits::default();
        let current = Instructions::load(&f.0.join("repo"), None);
        let section = current.section();
        // A current session: unchanged files send nothing.
        let mut saved = conversation(&f, section, false);
        assert!(!stale(&saved, section));
        // --no-instructions on resume: the section is empty, so an update
        // withdraws the saved one, and after it nothing more is sent.
        assert!(stale(&saved, ""));
        saved.push_user(update_message("", false), limits).unwrap();
        assert!(!stale(&saved, ""));
        assert!(latest_update(&saved).unwrap().get("content").unwrap().as_str().unwrap().contains("off for this session"));
        // An older session froze the root AGENTS.md in its project rules and
        // has no section. Removing the file must still withdraw that rule.
        let legacy = conversation(&f, "", true);
        assert!(crate::project::Snapshot::from_conversation(&legacy).unwrap().has_root_rule());
        std::fs::remove_file(f.0.join("repo/AGENTS.md")).unwrap();
        assert!(stale(&legacy, ""), "the removed root rule stays active without an update");
        let mut legacy = legacy;
        legacy.push_user(update_message("", true), limits).unwrap();
        assert!(!stale(&legacy, ""));
        let text = latest_update(&legacy).unwrap().get("content").unwrap().as_str().unwrap().to_owned();
        assert!(text.contains("any root AGENTS.md rule in the frozen project rules"));
        // A changed root file also replaces the frozen copy once.
        let older = conversation(&f, "", true);
        f.write("repo/AGENTS.md", "root rule changed\n");
        let changed = Instructions::load(&f.0.join("repo"), None);
        assert!(stale(&older, changed.section()));
    }
}
