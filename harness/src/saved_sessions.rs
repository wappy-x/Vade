//! Saved native sessions: list, resolve, delete and export the private journals.
//! The journal is the only store. A `session` record adds a display name and a
//! start time; readers that do not know it ignore it. Listing and export read a
//! journal without its lock and stop at an incomplete final record.
use crate::context::RESUME_NOTE;
use crate::json::{self, Value};
use crate::tools::{StateLocation, Workspace, anchored::Dir};
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::UNIX_EPOCH;

/// Resume and export keyword for the session with the latest write.
pub const LAST: &str = "last";
const MAX_ENTRIES: usize = 4096;
const MAX_PROJECTS: usize = 1024;

/// True for a usable display name: 1 to 64 characters, no control characters,
/// no path separators, no outer spaces, no leading hyphen (it reads as an
/// option on a command line), and not the `last` keyword.
pub fn valid_display_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('-')
        && name.chars().count() <= 64
        && name.trim() == name
        && name != LAST
        && !name.chars().any(|c| c.is_control() || c == '/' || c == '\\')
        && crate::terminal::sanitize(name) == name
}

/// The `session` record value: a display name and, at creation, the start time.
pub fn record(name: Option<&str>, started_ms: Option<u64>) -> Value {
    let mut fields = vec![("name".into(), name.map_or(Value::Null, |n| Value::String(n.into())))];
    if let Some(ms) = started_ms {
        fields.push(("started_ms".into(), Value::Number(ms.to_string())));
    }
    Value::Object(fields)
}

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// One saved session as `vade sessions` shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub id: String,
    pub name: Option<String>,
    pub started_ms: Option<u64>,
    /// The journal's last write: the end of the last turn or step.
    pub last_ms: Option<u64>,
    pub turns: u64,
    pub model: Option<String>,
    pub project: String,
    /// Set when a resume would refuse the journal, with the reason; the other
    /// fields may be empty.
    pub problem: Option<String>,
}

/// The sessions a scan found. A folder with more entries than the scan reads
/// adds a line to `partial`; the list may then miss sessions, so a name or
/// `last` does not select from it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Listing {
    pub sessions: Vec<Summary>,
    /// One warning per folder whose scan stopped at its entry limit.
    pub partial: Vec<String>,
}

/// One item of a readable transcript.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Message(Value),
    /// The model context was rewritten (compacted); the summary when recorded.
    Compacted(Option<String>),
    Resumed,
    /// A user-role message that Vade adds for the model, with its `name` and
    /// text (for example a project instructions update). It is not a user turn.
    Note(String, String),
}

/// A journal read back into transcript order.
#[derive(Debug, Default)]
pub struct Transcript {
    pub name: Option<String>,
    pub started_ms: Option<u64>,
    pub model: Option<String>,
    pub forked_from: Option<String>,
    pub entries: Vec<Entry>,
}

impl Transcript {
    /// User prompts, without resume notes.
    pub fn turns(&self) -> u64 {
        self.entries.iter().filter(|e| matches!(e, Entry::Message(m) if m.get("role").and_then(Value::as_str) == Some("user"))).count() as u64
    }
}

fn error(s: &str) -> io::Error {
    io::Error::other(s)
}

/// Split raw journal bytes into records. A torn final record ends the list, as
/// on resume; a damaged complete record is an error.
pub fn records(raw: &[u8]) -> io::Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while raw.len() - at >= 8 {
        let len = u64::from_le_bytes(raw[at..at + 8].try_into().unwrap());
        if len > json::MAX_INPUT_BYTES as u64 {
            return Err(error("invalid journal record length"));
        }
        let len = len as usize;
        if raw.len() - at - 8 < len + 8 {
            break;
        }
        if raw[at + 8 + len..at + 16 + len] != raw[at..at + 8] {
            return Err(error("journal record trailer corrupt"));
        }
        out.push(json::parse_bytes(&raw[at + 8..at + 8 + len]).map_err(|_| error("journal record corrupt"))?);
        at += len + 16;
    }
    Ok(out)
}

fn number(value: Option<&Value>) -> Option<u64> {
    match value {
        Some(Value::Number(n)) => n.parse().ok(),
        _ => None,
    }
}

/// Rebuild the transcript from journal records. Each record keeps `keep`
/// messages of the current view and appends its own, so a staged step that a
/// later record replaces is never shown twice. A record that rewrites the
/// context (keep 0 after the first record, or any `compaction` record) first
/// emits the old view in full, so the export keeps the turns the model no
/// longer sees. Each record passes the envelope and compaction checks that
/// replay uses; `check` runs the whole resume check. Unknown record kinds and
/// keys are ignored.
pub fn transcript(records: &[Value]) -> io::Result<Transcript> {
    let mut t = Transcript::default();
    let mut view: Vec<Value> = Vec::new();
    let mut shown = 0usize;
    let mut generation = 0;
    let emit = |t: &mut Transcript, messages: &[Value]| {
        for m in messages {
            let user = m.get("role").and_then(Value::as_str) == Some("user");
            t.entries.push(match m.get("name").and_then(Value::as_str) {
                _ if user && m.get("content").and_then(Value::as_str) == Some(RESUME_NOTE) => Entry::Resumed,
                Some(name) if user => Entry::Note(name.into(), text(m.get("content"))),
                _ => Entry::Message(m.clone()),
            });
        }
    };
    for (seq, r) in records.iter().enumerate() {
        let (keep, append) = crate::journal::envelope(r, seq as u64, view.len())?;
        if let Some(metadata) = r.get("compaction") {
            crate::journal::compaction_source(metadata, seq as u64, keep, generation, view.len())?;
            generation += 1;
        }
        if seq == 0 {
            t.model = r.get("options").and_then(|o| o.get("model")).and_then(Value::as_str).map(str::to_owned);
        }
        if let Some(session) = r.get("session") {
            if let Some(name) = session.get("name") {
                t.name = name.as_str().map(str::to_owned);
            }
            t.started_ms = t.started_ms.or(number(session.get("started_ms")));
        }
        if let Some(branch) = r.get("branch") {
            // A fork copies its parent's records; the parent's name and start are not its own.
            t.name = None;
            t.started_ms = None;
            t.forked_from = branch.get("parent_session").and_then(Value::as_str).map(str::to_owned);
        }
        if seq > 0 && (keep == 0 || r.get("compaction").is_some()) {
            emit(&mut t, &view[shown..]);
            let summary = r.get("compaction").and_then(|c| c.get("summary")).and_then(Value::as_str).map(str::to_owned);
            t.entries.push(Entry::Compacted(summary));
            view.truncate(keep);
            view.extend(append.iter().cloned());
            shown = view.len();
            continue;
        }
        view.truncate(keep);
        shown = shown.min(keep);
        view.extend(append.iter().cloned());
    }
    emit(&mut t, &view[shown..]);
    Ok(t)
}

fn read_journal(dir: &Dir, id: &str) -> io::Result<(Vec<u8>, Option<u64>, Option<u64>)> {
    let file = dir.optional_file(&format!("{id}.journal"))?.ok_or_else(|| error("session not found"))?;
    let meta = file.metadata()?;
    let ms = |t: io::Result<std::time::SystemTime>| t.ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_millis() as u64);
    let (modified, created) = (ms(meta.modified()), ms(meta.created()));
    let mut raw = Vec::new();
    file.take(crate::journal::MAX_JOURNAL + 1).read_to_end(&mut raw)?;
    if raw.len() as u64 > crate::journal::MAX_JOURNAL {
        return Err(error("session journal exceeds 64 MiB"));
    }
    Ok((raw, modified, created))
}

/// Run the resume checks (`Journal::replay`) on one journal, so the list and
/// `last` accept exactly what a resume accepts. `identity` is the folder
/// identity a resume would compare; None takes the identity that the journal
/// records, so only the journal itself is checked.
fn check(dir: &Dir, id: &str, records: &[Value], identity: Option<&str>) -> io::Result<()> {
    let file = dir.optional_file(&format!("{id}.journal"))?.ok_or_else(|| error("session not found"))?;
    let own = records.first().and_then(|r| r.get("workspace")).and_then(Value::as_str).unwrap_or("");
    crate::journal::Journal::check_saved(id, file, identity.unwrap_or(own).to_owned())
}

fn summarize(dir: &Dir, id: &str, project: &str, identity: Option<&str>) -> Summary {
    let mut summary = Summary { id: id.into(), name: None, started_ms: None, last_ms: None, turns: 0, model: None, project: project.into(), problem: None };
    let result = read_journal(dir, id).and_then(|(raw, modified, created)| {
        summary.last_ms = modified;
        let records = records(&raw)?;
        let checked = check(dir, id, &records, identity);
        let t = transcript(&records)?;
        // Sessions from older builds have no session record: use the time in
        // a generated ID, then the file's creation time where the system keeps it.
        summary.started_ms = t.started_ms.or_else(|| id_time(id)).or(created);
        summary.turns = t.turns();
        summary.name = t.name;
        summary.model = t.model;
        checked
    });
    if let Err(e) = result {
        summary.problem = Some(e.to_string());
    }
    summary
}

/// The start time inside a generated `session-<ms>-<pid>-<n>` ID.
fn id_time(id: &str) -> Option<u64> {
    id.strip_prefix("session-")?.split('-').next()?.parse().ok()
}

/// The marker `Dir::names` adds when it stops at its limit.
const LIMIT_MARKER: &str = "[listing limit reached]";

/// The session IDs in one folder, and false when the scan stopped at the
/// entry limit. Lock files and other entries count toward the limit.
fn ids(dir: &Dir) -> io::Result<(Vec<String>, bool)> {
    // One entry past the limit, so a full folder of exactly MAX_ENTRIES is complete.
    let names = dir.names(MAX_ENTRIES + 1)?;
    let complete = !names.iter().any(|s| s == LIMIT_MARKER);
    let ids = names.into_iter().filter_map(|s| s.strip_suffix(".journal").filter(|n| crate::journal::valid_name(n)).map(str::to_owned)).collect();
    Ok((ids, complete))
}

fn partial_folder(project: &str) -> String {
    format!("The session folder of {project} has more than {MAX_ENTRIES} entries, so this list may miss sessions. Use a session ID to resume or export one.")
}

fn newest_first(list: &mut [Summary]) {
    list.sort_by(|a, b| b.last_ms.cmp(&a.last_ms).then_with(|| b.id.cmp(&a.id)));
}

/// The saved sessions of this workspace, newest first. Listing never creates
/// the state folder.
pub fn list(workspace: &Workspace) -> io::Result<Listing> {
    if !workspace.state_path().join("sessions").is_dir() {
        return Ok(Listing::default());
    }
    let dir = workspace.session_dir()?;
    let project = workspace.root().to_string_lossy().into_owned();
    let (ids, complete) = ids(&dir)?;
    let identity = crate::branch::workspace_id(workspace).ok();
    let mut sessions: Vec<_> = ids.iter().map(|id| summarize(&dir, id, &project, identity.as_deref())).collect();
    newest_first(&mut sessions);
    Ok(Listing { sessions, partial: if complete { Vec::new() } else { vec![partial_folder(&project)] } })
}

/// The saved sessions of every project under a Vade home, newest first.
pub fn list_home(home: &std::path::Path) -> io::Result<Listing> {
    let projects = home.join("projects");
    if !projects.is_dir() {
        return Ok(Listing::default());
    }
    let root = Dir::open_root(&projects)?;
    let mut listing = Listing::default();
    let keys = root.names(MAX_PROJECTS + 1)?;
    if keys.iter().any(|k| k == LIMIT_MARKER) {
        listing.partial.push(format!("The Vade home has more than {MAX_PROJECTS} projects, so this list may miss sessions. Run vade sessions in a project folder to list its sessions."));
    }
    for key in keys {
        let Ok(project) = root.child(&key) else { continue };
        let mut path = Vec::new();
        if let Ok(Some(file)) = project.optional_file("workspace") {
            let _ = file.take(64 * 1024).read_to_end(&mut path);
        }
        let path = String::from_utf8_lossy(path.strip_suffix(b"\n").unwrap_or(&path)).into_owned();
        let path = if path.is_empty() { "(unknown project)".to_owned() } else { path };
        // A project with no sessions folder has no sessions; any other failure
        // is a warning, and the other projects still list.
        if std::fs::symlink_metadata(projects.join(&key).join("sessions")).is_err_and(|e| e.kind() == io::ErrorKind::NotFound) {
            continue;
        }
        let scan = project.child("sessions").and_then(|sessions| Ok((ids(&sessions)?, sessions)));
        let ((ids, complete), sessions) = match scan {
            Ok(found) => found,
            Err(e) => {
                listing.partial.push(format!("Cannot read the session folder of {path}: {e}."));
                continue;
            }
        };
        // Without the project folder, the journal's own folder identity is used.
        listing.sessions.extend(ids.iter().map(|id| summarize(&sessions, id, &path, None)));
        if !complete {
            listing.partial.push(partial_folder(&path));
        }
    }
    newest_first(&mut listing.sessions);
    Ok(listing)
}

/// Map `ID`, `last` or a display name to a session ID. An exact ID wins; a
/// name shared by several sessions selects the newest one. `last` is the newest
/// session with at least one turn; for a resume it also skips sessions that
/// another chat (or this one) has open. A name or `last` needs a complete
/// scan: a partial one may miss the newest session.
pub fn resolve(workspace: &Workspace, query: &str, resume: bool) -> io::Result<String> {
    if crate::journal::valid_name(query) && workspace.state_path().join("sessions").join(format!("{query}.journal")).is_file() {
        return Ok(query.into());
    }
    let listing = list(workspace)?;
    if !listing.partial.is_empty() {
        return Err(error(&format!("the session folder has more than {MAX_ENTRIES} entries, so a name or last may select the wrong session; select the session by ID")));
    }
    let list = listing.sessions;
    let free = |s: &&Summary| !resume || workspace.session_dir().and_then(|d| d.lock(&format!("{}.journal-lock", s.id))).is_ok();
    let found = if query == LAST { list.iter().filter(|s| s.turns > 0 && s.problem.is_none()).find(free) } else { list.iter().find(|s| s.name.as_deref() == Some(query)) };
    found.map(|s| s.id.clone()).ok_or_else(|| {
        error(match query {
            LAST if list.is_empty() => "this project has no saved sessions",
            LAST => "no saved session in this project can be resumed with last: each one has no turns, is open in a chat, or cannot be resumed; run vade sessions",
            _ => "no saved session in this project has that ID or name",
        })
    })
}

/// Delete one session's journal. A running session holds the lock, so it is
/// refused. The empty lock file stays, as with an in-app delete.
pub fn delete(workspace: &Workspace, id: &str) -> io::Result<()> {
    if !crate::journal::valid_name(id) || !workspace.state_path().join("sessions").join(format!("{id}.journal")).is_file() {
        return Err(error("no saved session in this project has that ID; run vade sessions"));
    }
    let dir = workspace.session_dir()?;
    let _lock = dir.lock(&format!("{id}.journal-lock")).map_err(|_| error("the session is open in another chat; close it first"))?;
    dir.optional_file(&format!("{id}.journal"))?.ok_or_else(|| error("session not found"))?;
    workspace.verify_root()?;
    dir.remove(&format!("{id}.journal"))
}

/// A readable Markdown transcript of one session.
pub fn export_markdown(workspace: &Workspace, id: &str) -> io::Result<String> {
    let dir = workspace.session_dir()?;
    let (raw, modified, created) = read_journal(&dir, id)?;
    let records = records(&raw)?;
    // A journal from another folder (an old clone) still exports.
    check(&dir, id, &records, None)?;
    let t = transcript(&records)?;
    let started = t.started_ms.or_else(|| id_time(id)).or(created);
    Ok(markdown(id, &workspace.root().to_string_lossy(), &t, started, modified))
}

fn time(ms: Option<u64>) -> String {
    let Some(ms) = ms else { return "-".into() };
    let secs = ms / 1000;
    let (days, rest) = ((secs / 86400) as i64, secs % 86400);
    // Civil date from days since 1970-01-01 (proleptic Gregorian calendar).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02} {:02}:{:02} UTC", rest / 3600, rest % 3600 / 60)
}

/// The text of a `content` or `refusal` field: a string or text parts.
fn text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
        None | Some(Value::Null) => String::new(),
        Some(other) => json::stringify(other).unwrap_or_default(),
    }
}

/// A code fence one backtick longer than any backtick run in the text.
fn fenced(text: &str, info: &str) -> String {
    let longest = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    let fence = "`".repeat(longest.max(2) + 1);
    format!("{fence}{info}\n{}\n{fence}\n\n", text.trim_end_matches('\n'))
}

fn markdown(id: &str, project: &str, t: &Transcript, started: Option<u64>, last: Option<u64>) -> String {
    let clean = crate::terminal::sanitize;
    let mut out = format!("# Session {}\n\n", clean(t.name.as_deref().unwrap_or(id)));
    out += &format!("- ID: `{id}`\n");
    if let Some(name) = &t.name {
        out += &format!("- Name: {}\n", clean(name));
    }
    if let Some(parent) = &t.forked_from {
        out += &format!("- Forked from: `{}`\n", clean(parent));
    }
    out += &format!("- Model: {}\n- Project: {}\n- Started: {}\n- Last turn: {}\n- Turns: {}\n\n", clean(t.model.as_deref().unwrap_or("-")), clean(project), time(started), time(last), t.turns());
    for entry in &t.entries {
        match entry {
            Entry::Resumed => out += "*The session was resumed here. No tools were run again.*\n\n",
            Entry::Note(name, text) => {
                out += &format!("*Vade added a note for the model here ({}):*\n\n", clean(name));
                out += &fenced(&clean(text), "text");
            }
            Entry::Compacted(summary) => {
                out += "---\n\n*The context was compacted here. The turns above stay in the journal; the model continued from this summary:*\n\n";
                out += &fenced(&clean(summary.as_deref().unwrap_or("(no summary recorded)")), "text");
                out += "---\n\n";
            }
            Entry::Message(m) => {
                let role = m.get("role").and_then(Value::as_str).unwrap_or("unknown");
                let body = clean(&text(m.get("content")));
                match role {
                    "system" | "developer" => {}
                    "user" => out += &format!("## User\n\n{}\n\n", body.trim_end()),
                    "assistant" => {
                        out += "## Assistant\n\n";
                        // A refusal is the visible answer when the content is empty.
                        for answer in [body, clean(&text(m.get("refusal")))] {
                            if !answer.trim().is_empty() {
                                out += &format!("{}\n\n", answer.trim_end());
                            }
                        }
                        for call in m.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                            let function = call.get("function");
                            let name = function.and_then(|f| f.get("name")).and_then(Value::as_str).unwrap_or("unknown");
                            let args = function.and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or("");
                            out += &format!("Tool call `{}`:\n\n", clean(name));
                            out += &fenced(&clean(args), "json");
                        }
                    }
                    "tool" => {
                        out += "Tool result:\n\n";
                        out += &fenced(&body, "text");
                    }
                    other => out += &format!("## {}\n\n{}\n\n", clean(other), body.trim_end()),
                }
            }
        }
    }
    out
}

/// The `vade sessions` table: one row per session, newest first.
/// A partial scan adds one warning line per folder.
pub fn table(listing: &Listing) -> String {
    let warnings: String = listing.partial.iter().map(|w| format!("Warning: {}\n", crate::terminal::sanitize(w))).collect();
    let list = &listing.sessions;
    if list.is_empty() {
        return "No saved sessions. Each agent chat is a saved session; resume it with vade chat --resume last.\n".to_owned() + &warnings;
    }
    let clean = |s: &str| crate::terminal::sanitize(s).replace(['\n', '\t'], " ");
    let rows: Vec<[String; 7]> = list
        .iter()
        .map(|s| {
            [
                s.id.clone(),
                clean(s.name.as_deref().unwrap_or("-")),
                time(s.started_ms),
                time(s.last_ms),
                if s.problem.is_some() { "-".into() } else { s.turns.to_string() },
                s.problem.as_ref().map_or_else(|| clean(s.model.as_deref().unwrap_or("-")), |p| format!("cannot resume: {}", clean(p))),
                clean(&s.project),
            ]
        })
        .collect();
    let head = ["ID", "NAME", "STARTED", "LAST TURN", "TURNS", "MODEL", "PROJECT"];
    let mut widths = head.map(|h| h.chars().count());
    for row in &rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let line = |cells: &[String]| {
        let mut text = String::new();
        for (i, cell) in cells.iter().enumerate() {
            if i + 1 == cells.len() {
                text += cell;
            } else {
                text += &format!("{cell}{}  ", " ".repeat(widths[i] - cell.chars().count()));
            }
        }
        text + "\n"
    };
    let mut out = line(&head.map(String::from));
    for row in &rows {
        out += &line(row);
    }
    out + &warnings
}

const HELP: &str = "Usage: vade-harness sessions [list] [--all] [STATE]
       vade-harness sessions export ID|NAME|last [--markdown] [STATE]
       vade-harness sessions delete ID [STATE]

Lists, exports or deletes the saved native agent sessions of a project.
list is the default. Sessions are listed newest first.

  --all                 List every project under the Vade home
  --markdown            Export a readable Markdown transcript (the only format)
  --workspace PATH      The project folder (default: the current folder)
  --vade-home DIR       State root (default VADE_HOME, else ~/.vade)
  --state-in-project    Read WORKSPACE/.vade instead of the Vade home

STATE is --workspace, --vade-home and --state-in-project. Resume a session
with vade-harness agent --resume ID|NAME|last.
";

/// `vade-harness sessions ...`: exit 0, 1 on failure, 2 on a usage error.
pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    match run(args.into_iter().collect()) {
        Ok(text) => crate::diagnostics::print_stdout(&text),
        Err((code, message)) => {
            crate::diagnostics::stderr_line(format_args!("vade: {}", crate::terminal::sanitize(&message)));
            ExitCode::from(code)
        }
    }
}

fn run(args: Vec<String>) -> Result<String, (u8, String)> {
    let usage = |m: &str| (2, format!("{m}. Run vade-harness sessions --help."));
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return Ok(HELP.into());
    }
    let (mut action, mut target, mut all, mut workspace, mut home, mut in_project) = (None, None, false, None, None, false);
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--all" if !all => all = true,
            "--markdown" => {}
            "--state-in-project" if !in_project => in_project = true,
            "--workspace" | "--vade-home" => {
                let value = args.next().filter(|v| !v.starts_with("--")).ok_or_else(|| usage(&format!("{arg} needs a value")))?;
                let slot = if arg == "--workspace" { &mut workspace } else { &mut home };
                if slot.replace(PathBuf::from(value)).is_some() {
                    return Err(usage(&format!("use {arg} only once")));
                }
            }
            "list" | "export" | "delete" if action.is_none() => action = Some(arg),
            _ if action.as_deref().is_some_and(|a| a != "list") && target.is_none() && !arg.starts_with("--") => target = Some(arg),
            _ => return Err(usage(&format!("unexpected argument {:?}", crate::terminal::sanitize(&arg)))),
        }
    }
    let action = action.unwrap_or_else(|| "list".into());
    if action != "list" && target.is_none() {
        return Err(usage(&format!("{action} needs a session ID")));
    }
    if all && action != "list" {
        return Err(usage("--all works only with list"));
    }
    if home.is_some() && in_project {
        return Err(usage("choose --vade-home DIR or --state-in-project, not both"));
    }
    let failed = |e: io::Error| (1, e.to_string());
    let state = match home {
        _ if in_project => StateLocation::Project,
        Some(path) => StateLocation::home(path).map_err(|_| usage("--vade-home must be an absolute path without '..'"))?,
        None => StateLocation::from_env().map_err(failed)?,
    };
    if all {
        let StateLocation::Home(home) = &state else {
            return Err(usage("--all reads the Vade home; it does not work with --state-in-project"));
        };
        return list_home(home).map(|l| table(&l)).map_err(failed);
    }
    let root = match workspace {
        Some(path) => path,
        None => std::env::current_dir().map_err(|_| (1, "the current folder is unavailable".to_string()))?,
    };
    let workspace = Workspace::open(&root, state).map_err(failed)?;
    match action.as_str() {
        "list" => list(&workspace).map(|l| table(&l)),
        "export" => resolve(&workspace, target.as_deref().unwrap(), false).and_then(|id| export_markdown(&workspace, &id)),
        _ => {
            let id = target.unwrap();
            delete(&workspace, &id).map(|()| format!("Deleted session {id}.\n"))
        }
    }
    .map_err(failed)
}

#[cfg(test)]
#[path = "saved_sessions/tests.rs"]
mod tests;
