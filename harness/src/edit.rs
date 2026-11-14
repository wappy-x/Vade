//! Exact, bounded text patches. The only fuzzy step is a unique match that
//! ignores spaces and tabs, reported to the caller. No external patch utility.
//! Preflight the whole batch; commit by atomic file replacement, not in-place writes.
use crate::tools::{MAX_TEXT_BYTES, Workspace, anchored::Dir};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "line_edit.rs"]
pub(crate) mod line_edit;

const MAX_PATCH: usize = 64 * 1024;
const MAX_FILES: usize = 32;
const MAX_PREVIEW: usize = 192 * 1024;
const MAX_CONTEXT_LINE: usize = 512;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub text: String,
    pub identity: (u64, u64, u64, i64, i64, u32),
}
impl Snapshot {
    pub(crate) fn read(dir: &Dir, name: &str) -> io::Result<Option<Self>> {
        use std::io::Read;
        let Some(file) = dir.workspace_file(name)? else {
            return Ok(None);
        };
        let m = file.metadata()?;
        if !m.is_file() || m.nlink() != 1 {
            return Err(err("patch requires a regular file with one hard link"));
        }
        if m.len() > MAX_TEXT_BYTES as u64 {
            return Err(err(&format!(
                "target file is {} bytes; edit target limit is {MAX_TEXT_BYTES} bytes (4 MiB)",
                m.len()
            )));
        }
        let stamp = |m: &std::fs::Metadata| {
            ((m.dev(), m.ino(), m.len(), m.mtime(), m.mtime_nsec(), m.mode()), (m.ctime(), m.ctime_nsec()))
        };
        let (identity, changed) = stamp(&m);
        let mut bytes = Vec::new();
        (&file)
            .take((MAX_TEXT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_TEXT_BYTES {
            return Err(err(&format!(
                "target read is {} bytes; edit target limit is {MAX_TEXT_BYTES} bytes (4 MiB)",
                bytes.len()
            )));
        }
        // A same-length write during the read still moves mtime or ctime.
        if stamp(&file.metadata()?) != (identity, changed) || bytes.len() as u64 != m.len() {
            return Err(err("target file changed during patch read"));
        }
        Ok(Some(Self {
            text: String::from_utf8(bytes).map_err(|_| err("patch requires UTF-8"))?,
            identity,
        }))
    }
}
struct Change {
    dir: Dir,
    /// Directories below `dir` that are absent until an added file is written.
    missing: Vec<String>,
    name: String,
    path: String,
    before: Option<Snapshot>,
    after: Option<String>,
    target: Option<(Dir, String, String)>,
}
/// A parsed patch checked against the current files: the planned changes,
/// their diff and the original patch text, ready to preview and apply.
pub struct PreparedPatch {
    changes: Vec<Change>,
    diff: String,
    patch: String,
    /// Paths whose old text was found only by ignoring spaces and tabs.
    loose: Vec<String>,
}
/// An immutable copy of the bytes offered for approval and their prepared plan.
pub struct PatchPreview {
    patch: String,
    diff: String,
    patch_hash: u64,
    diff_hash: u64,
    plan_hash: u64,
}
/// A failed patch application, and whether any write may already have
/// happened.
#[derive(Debug)]
pub struct ApplyError {
    error: io::Error,
    /// False only when apply stopped before attempting any write. An attempted
    /// namespace operation may have committed even without a success receipt.
    pub uncertain_writes: bool,
}
impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for ApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
impl ApplyError {
    fn new(error: io::Error, uncertain_writes: bool) -> Self {
        Self {
            error,
            uncertain_writes,
        }
    }
}
impl PatchPreview {
    pub fn diff(&self) -> &str {
        &self.diff
    }
}
fn byte_hash(bytes: &[u8]) -> u64 {
    // A process-local std integrity check, not a cryptographic identity. Exact
    // byte comparison below also rejects mismatches even if hashes collide.
    let mut hash = DefaultHasher::new();
    bytes.hash(&mut hash);
    hash.finish()
}
impl PreparedPatch {
    pub fn diff(&self) -> &str {
        &self.diff
    }
    pub fn preview(&self) -> PatchPreview {
        PatchPreview {
            patch: self.patch.clone(),
            diff: self.diff.clone(),
            patch_hash: byte_hash(self.patch.as_bytes()),
            diff_hash: byte_hash(self.diff.as_bytes()),
            plan_hash: self.plan_hash(),
        }
    }
    fn plan_hash(&self) -> u64 {
        let mut hash = DefaultHasher::new();
        for change in &self.changes {
            change.path.hash(&mut hash);
            change.after.hash(&mut hash);
            change
                .target
                .as_ref()
                .map(|(_, _, path)| path)
                .hash(&mut hash);
            change.mode().hash(&mut hash);
        }
        hash.finish()
    }
    /// Every path that may change, including both ends of a move.
    pub fn affected_paths(&self) -> Vec<String> {
        self.changes
            .iter()
            .flat_map(|change| {
                std::iter::once(change.path.clone())
                    .chain(change.target.as_ref().map(|(_, _, path)| path.clone()))
            })
            .collect()
    }
    pub fn apply(self, cancel: &AtomicBool) -> io::Result<String> {
        let preview = self.preview();
        self.apply_with_receipts(&preview, cancel, |_, _| {})
            .map_err(io::Error::other)
    }
    /// Report each committed namespace operation, including a move destination
    /// installed before a later source-removal failure. Paths stay relative.
    pub fn apply_with_receipts(
        self,
        preview: &PatchPreview,
        cancel: &AtomicBool,
        mut receipt: impl FnMut(&'static str, &str),
    ) -> Result<String, ApplyError> {
        // Check before preflight or any namespace operation. The same frozen
        // preview is used by the approval callback and by this apply boundary.
        if byte_hash(self.patch.as_bytes()) != preview.patch_hash
            || byte_hash(self.diff.as_bytes()) != preview.diff_hash
            || self.patch.as_bytes() != preview.patch.as_bytes()
            || self.diff.as_bytes() != preview.diff.as_bytes()
            || self.plan_hash() != preview.plan_hash
        {
            return Err(ApplyError::new(
                err(
                    "patch integrity mismatch: applied patch differs from the preview; request a new preview and approval",
                ),
                false,
            ));
        }
        for c in &self.changes {
            c.check().map_err(|e| ApplyError::new(e, false))?;
        }
        let mut applied = 0;
        for c in &self.changes {
            // Cancel stops only before the first write. A started batch runs to
            // its end so a checked multi-file patch is not left half applied.
            if applied == 0 && cancel.load(Ordering::Relaxed) {
                return Err(ApplyError::new(err("patch canceled; 0 file operations committed"), false));
            }
            if let Err(e) = c.check() {
                if applied == 0 {
                    return Err(ApplyError::new(e, false));
                }
                // Another writer changed a later file.
                let undone = undo_committed(&self.changes[..applied], &mut receipt);
                return Err(ApplyError::new(err(&format!("{e}; {undone}")), true));
            }
            let result = (|| {
                match (&c.after, &c.target) {
                    (Some(text), Some((dir, name, target))) => {
                        // A pure move is a single exclusive rename. A move+edit is
                        // two atomic namespace operations; a crash may leave both.
                        if c.before.as_ref().is_some_and(|b| b.text == *text) {
                            c.dir.move_exclusive(&c.name, dir, name)?;
                            receipt("File written", target);
                            receipt("File removed", &c.path);
                        } else {
                            dir.install(name, text.as_bytes(), c.mode(), true, || c.check())?;
                            receipt("File written", target);
                            if Snapshot::read(&c.dir, &c.name)? != c.before {
                                return Err(err(
                                    "move source changed after destination commit; both files retained",
                                ));
                            }
                            c.dir.remove(&c.name).map_err(|_| err("move destination installed; source removal failed, both files may remain"))?;
                            receipt("File removed", &c.path);
                        }
                    }
                    (Some(text), None) => {
                        let dir = c
                            .parent(true)?
                            .ok_or_else(|| err("patch parent is missing"))?;
                        dir.install(
                            &c.name,
                            text.as_bytes(),
                            c.mode(),
                            c.before.is_none(),
                            || {
                                if applied == 0 && cancel.load(Ordering::Relaxed) {
                                    return Err(err("patch canceled during staging"));
                                }
                                c.check()
                            },
                        )?;
                        receipt("File written", &c.path);
                    }
                    (None, _) => {
                        c.dir.remove(&c.name)?;
                        receipt("File removed", &c.path);
                    }
                }
                Ok(())
            })();
            if let Err(e) = result {
                // The staging check or a write failed. The earlier files are
                // undone the same way; this one may have partial changes.
                let undone = match applied {
                    0 => String::new(),
                    _ => format!("; {}", undo_committed(&self.changes[..applied], &mut receipt)),
                };
                return Err(ApplyError::new(
                    err(&format!(
                        "patch stopped after {applied} completed file operations; the failing operation may have partial changes: {e}{undone}"
                    )),
                    true,
                ));
            }
            applied += 1;
        }
        let mut summary = format!("Applied {applied} file operations.");
        if !self.loose.is_empty() {
            let paths: Vec<String> = self.loose.iter().map(|p| crate::json::quote(p)).collect();
            summary.push_str(&format!(
                "\nnote: whitespace-tolerant match used in {}; old lines differed from the file only in spaces or tabs. Unchanged lines kept the file's spacing; added lines use the patch's spacing.",
                paths.join(", ")
            ));
        }
        Ok(summary)
    }
}
/// Undo committed changes, last first, while each file still holds exactly
/// the bytes this patch wrote. Every undo is tried, even after one fails.
fn undo_committed(done: &[Change], receipt: &mut dyn FnMut(&'static str, &str)) -> String {
    let undone = done.iter().rev().fold(true, |all, change| change.undo(receipt).is_ok() && all);
    let how = if undone { "rolled back" } else { "could not roll back all of" };
    format!("{how} the {} earlier file operations of this patch", done.len())
}
impl Change {
    fn mode(&self) -> u32 {
        self.before.as_ref().map_or(0o644, |b| b.identity.5 & 0o777)
    }
    /// The file's parent directory. Missing directories are created only when
    /// `create` is set; otherwise a still-missing parent is `None`.
    fn parent(&self, create: bool) -> io::Result<Option<Dir>> {
        let mut dir = self.dir.duplicate()?;
        for name in &self.missing {
            dir = if create {
                dir.create_workspace_child(name)?
            } else {
                match dir.optional_workspace_child(name)? {
                    Some(child) => child,
                    None => return Ok(None),
                }
            };
        }
        Ok(Some(dir))
    }
    /// Undo a committed add, update or delete while the file still holds the
    /// bytes this patch wrote. A move is not undone; created folders remain.
    fn undo(&self, receipt: &mut dyn FnMut(&'static str, &str)) -> io::Result<()> {
        if self.target.is_some() {
            return Err(err("a move is not rolled back"));
        }
        let dir = self.parent(false)?.ok_or_else(|| err("patch parent is missing"))?;
        let now = Snapshot::read(&dir, &self.name)?;
        if now.as_ref().map(|s| &s.text) != self.after.as_ref() {
            return Err(err("file changed after this patch wrote it"));
        }
        let unchanged = || match Snapshot::read(&dir, &self.name)? == now {
            true => Ok(()),
            false => Err(err("file changed after this patch wrote it")),
        };
        match &self.before {
            Some(before) => {
                dir.install(&self.name, before.text.as_bytes(), self.mode(), self.after.is_none(), unchanged)?;
                receipt("File restored", &self.path);
            }
            None => {
                unchanged()?;
                dir.remove(&self.name)?;
                receipt("File removed", &self.path);
            }
        }
        Ok(())
    }
    fn check(&self) -> io::Result<()> {
        let current = match self.parent(false)? {
            Some(dir) => Snapshot::read(&dir, &self.name)?,
            None => None,
        };
        if current != self.before {
            return Err(err("patch conflict: source changed since preview"));
        }
        if let Some((dir, name, _)) = &self.target {
            if Snapshot::read(dir, name)?.is_some() {
                return Err(err("patch conflict: move destination exists"));
            }
        }
        Ok(())
    }
}
fn err(text: &str) -> io::Error {
    io::Error::other(text)
}

struct Edit {
    old_start: usize,
    old_end: usize,
    new_start: usize,
    new_end: usize,
}

/// Unchanged context lines come from the target file (up to 4 MiB), so each is
/// capped individually. Headers and -/+ lines come from bounded patch text and
/// are always shown in full; if the preview would still exceed the cap the
/// patch is refused rather than approved with changed lines hidden.
#[derive(Default)]
struct Preview {
    text: String,
    overflow: bool,
}
impl Preview {
    fn push(&mut self, text: &str) {
        if self.overflow {
            return;
        }
        if self.text.len() + text.len() > MAX_PREVIEW {
            self.overflow = true;
            return;
        }
        self.text.push_str(text);
    }
    fn escaped(&mut self, text: &str) {
        if self.overflow {
            return;
        }
        match crate::terminal::escape_preview(text, MAX_PREVIEW - self.text.len()) {
            Ok(shown) => self.push(&shown),
            Err(_) => self.overflow = true,
        }
    }
    fn line(&mut self, prefix: char, line: &str) {
        if self.overflow {
            return;
        }
        let body = line.strip_suffix('\n').unwrap_or(line);
        let mut shown = line;
        if prefix == ' ' && body.len() > MAX_CONTEXT_LINE {
            let mut end = MAX_CONTEXT_LINE;
            while !line.is_char_boundary(end) {
                end -= 1;
            }
            shown = &line[..end];
        }
        self.push(&prefix.to_string());
        self.escaped(shown);
        if shown.len() < line.len() {
            self.push(&format!(
                "\n[context line truncated: {} bytes]\n",
                body.len()
            ));
        } else if !line.ends_with('\n') {
            self.push("\n\\ No newline at end of file\n");
        }
    }
    fn finish(self) -> io::Result<String> {
        if self.overflow {
            return Err(err(&format!(
                "diff exceeds {MAX_PREVIEW}-byte (192 KiB) preview limit; every changed line must be shown for approval"
            )));
        }
        Ok(self.text)
    }
}

/// Changed ranges come from exact patch application, not a quadratic whole-file
/// comparison. Trim unchanged hunk edges and merge overlapping context windows.
fn unified_preview(preview: &mut Preview, old: &str, new: &str, edits: &[Edit]) {
    let old: Vec<&str> = old.split_inclusive('\n').collect();
    let new: Vec<&str> = new.split_inclusive('\n').collect();
    let mut index = 0;
    while index < edits.len() && !preview.overflow {
        let first = &edits[index];
        let start = first.old_start.saturating_sub(3);
        let new_start = first.new_start - (first.old_start - start);
        let mut last = index;
        let mut end = (first.old_end + 3).min(old.len());
        while last + 1 < edits.len() && edits[last + 1].old_start.saturating_sub(3) <= end {
            last += 1;
            end = (edits[last].old_end + 3).min(old.len());
        }
        let new_end = edits[last].new_end + end - edits[last].old_end;
        let old_count = end - start;
        let new_count = new_end - new_start;
        preview.push(&format!(
            "@@ -{},{} +{},{} @@\n",
            start + usize::from(old_count > 0),
            old_count,
            new_start + usize::from(new_count > 0),
            new_count
        ));
        let mut cursor = start;
        for edit in &edits[index..=last] {
            for line in &old[cursor..edit.old_start] {
                preview.line(' ', line);
            }
            for line in &old[edit.old_start..edit.old_end] {
                preview.line('-', line);
            }
            for line in &new[edit.new_start..edit.new_end] {
                preview.line('+', line);
            }
            cursor = edit.old_end;
        }
        for line in &old[cursor..end] {
            preview.line(' ', line);
        }
        index = last + 1;
    }
}

pub(crate) const BEGIN: &str = "*** Begin Patch";
pub(crate) const END: &str = "*** End Patch";
pub(crate) const FILE_HEADERS: [&str; 3] = ["*** Add File: ", "*** Update File: ", "*** Delete File: "];

/// Patch lines framed by the Begin/End markers. Either marker may be omitted
/// when the text starts with a file header; any other text before or after the
/// markers is refused, except one enclosing Markdown code fence.
fn framed_lines(patch: &str) -> io::Result<Vec<&str>> {
    let patch = if let Some((opening, body)) = patch.split_once('\n')
        && matches!(opening, "```" | "```diff" | "```patch")
        && let Some(body) = body.strip_suffix("\n```").or_else(|| body.strip_suffix("\n```\n"))
    { body } else { patch };
    let mut lines: Vec<&str> = patch
        .strip_suffix('\n')
        .unwrap_or(patch)
        .split('\n')
        .collect();
    let starts_with_header = FILE_HEADERS.iter().any(|h| lines[0].starts_with(h));
    if lines[0] != BEGIN && starts_with_header && !lines.contains(&BEGIN) {
        lines.insert(0, BEGIN);
    }
    if !lines.contains(&END) {
        lines.push(END);
    }
    if lines[0] != BEGIN || lines.last() != Some(&END) {
        return Err(err(
            "patch must start with '*** Begin Patch' (or a '*** Add/Update/Delete File: ' header) and end with '*** End Patch'; remove Markdown fences and any text outside the markers",
        ));
    }
    Ok(lines)
}

/// Conservative effect extraction for the scheduler's ownership guard. The
/// executor still validates the complete patch and workspace boundary. These
/// are the only unprefixed path headers accepted by `prepare`; content lines
/// have a space/+/- prefix. Reject malformed/oversized envelopes before dispatch.
pub(crate) fn touched_paths(patch: &str) -> io::Result<Vec<&str>> {
    if patch.len() > MAX_PATCH {
        return Err(err(&format!(
            "patch text is {} bytes; patch text limit is {MAX_PATCH} bytes (64 KiB)",
            patch.len()
        )));
    }
    let lines = framed_lines(patch)?;
    let mut paths = Vec::new();
    for &line in &lines[1..lines.len() - 1] {
        if let Some(path) = [
            "*** Add File: ",
            "*** Update File: ",
            "*** Delete File: ",
            "*** Move to: ",
        ]
        .iter()
        .find_map(|prefix| line.strip_prefix(prefix))
        {
            paths.push(path);
            if paths.len() > MAX_FILES * 2 {
                return Err(err("too many patch paths"));
            }
        } else if line.starts_with("*** ") {
            return Err(err("unknown patch header"));
        }
    }
    if paths.is_empty() {
        return Err(err("empty patch"));
    }
    Ok(paths)
}

/// Retain a bounded correction only for a single rejected Add. Existing content
/// is captured now so a later explicit retry cannot overwrite intervening edits.
pub(crate) fn prepare_add_retry(workspace: &Workspace, patch: &str) -> io::Result<(String, PreparedPatch)> {
    if patch.len() > MAX_PATCH {
        return Err(err("retry patch too large"));
    }
    let lines = framed_lines(patch)?;
    let path = lines.get(1).and_then(|line| line.strip_prefix("*** Add File: "))
        .ok_or_else(|| err("retry requires a single Add"))?;
    let body = &lines[2..lines.len() - 1];
    if body.iter().any(|line| line.starts_with("*** ")) {
        return Err(err("retry requires a single Add"));
    }
    if body.iter().any(|line| line.starts_with("@@") || line.starts_with("--- ") || line.starts_with("+++")) {
        return Err(err("retained retry cannot interpret mixed unified-diff metadata as file content; resend a canonical patch"));
    }
    let (dir, missing, name, _) = workspace.edit_parent(path)?;
    let before = if missing.is_empty() {
        Snapshot::read(&dir, &name)?
    } else {
        None
    };
    let header = format!("*** {} File: {path}", if before.is_some() { "Update" } else { "Add" });
    let mut corrected = format!("{BEGIN}\n{header}\n");
    if let Some(before) = before {
        corrected.push_str("@@\n");
        for line in before.text.split_inclusive('\n') {
            corrected.push('-');
            corrected.push_str(line);
            if !line.ends_with('\n') {
                corrected.push_str("\n\\ No newline at end of file\n");
            }
            if corrected.len() > MAX_PATCH {
                return Err(err("retry patch too large"));
            }
        }
    }
    // An entirely bare Add keeps leading '+' content. A mixed Add is refused:
    // a bare line that starts with '+' cannot be told apart from a prefix.
    let prefixed = body.iter().any(|line| line.starts_with('+'));
    if prefixed && body.iter().any(|line| !line.is_empty() && !line.starts_with('+') && *line != "\\ No newline at end of file") {
        return Err(err("retry cannot tell a '+' prefix from '+' content in a mixed Add body"));
    }
    for line in body {
        if *line == "\\ No newline at end of file" {
            corrected.push_str(line);
        } else {
            corrected.push('+');
            corrected.push_str(if prefixed { line.strip_prefix('+').unwrap_or(line) } else { line });
        }
        corrected.push('\n');
        if corrected.len() > MAX_PATCH {
            return Err(err("retry patch too large"));
        }
    }
    corrected.push_str(END);
    Ok((header, prepare(workspace, &corrected)?))
}

/// Grammar follows the familiar Begin/End Patch markers. Update hunks use bare
/// @@ separators and unique exact old text, or else a unique match ignoring
/// spaces and tabs. Add bodies may omit every '+'. A delete body is optional;
/// if given it must be the entire old text as '-' lines. Lines include LF
/// unless followed by the usual backslash no-newline marker. Missing parents
/// of an added file are created inside the workspace when it is applied.
pub fn prepare(workspace: &Workspace, patch: &str) -> io::Result<PreparedPatch> {
    if patch.len() > MAX_PATCH {
        return Err(err(&format!(
            "patch text is {} bytes; patch text limit is {MAX_PATCH} bytes (64 KiB)",
            patch.len()
        )));
    }
    let lines = framed_lines(patch)?;
    let mut i = 1;
    let mut changes = Vec::new();
    let mut used = Vec::new();
    let mut loose_paths = Vec::new();
    let mut diff = Preview::default();
    diff.push("Preview escapes: backslash is \\\\, tab is \\t, carriage return is \\r; nonprinting Unicode is \\u{hex}.\n");
    while i + 1 < lines.len() {
        if changes.len() == MAX_FILES {
            return Err(err("patch exceeds 32 files"));
        }
        let (kind, path) = if let Some(p) = lines[i].strip_prefix("*** Add File: ") {
            ("add", p)
        } else if let Some(p) = lines[i].strip_prefix("*** Update File: ") {
            ("update", p)
        } else if let Some(p) = lines[i].strip_prefix("*** Delete File: ") {
            ("delete", p)
        } else {
            let found: String = lines[i].chars().take(80).collect();
            return Err(err(&format!(
                "patch line {} must be a '*** Add File: ', '*** Update File: ' or '*** Delete File: ' header, found {}",
                i + 1,
                crate::json::quote(&found)
            )));
        };
        let (dir, missing, name, normalized) = workspace.edit_parent(path)?;
        reserve_path(workspace, &mut used, &normalized)?;
        let before = if missing.is_empty() {
            Snapshot::read(&dir, &name)?
        } else {
            None
        };
        if (kind == "add") != before.is_none() {
            let path = crate::json::quote(&normalized);
            return Err(err(&match kind {
                "add" => format!(
                    "patch conflict: {path} already exists; use '*** Update File: ' to change it"
                ),
                _ => format!(
                    "patch conflict: {path} does not exist; use '*** Add File: ' to create it"
                ),
            }));
        }
        i += 1;
        let target =
            if kind == "update" && lines.get(i).is_some_and(|l| l.starts_with("*** Move to: ")) {
                let p = lines[i].strip_prefix("*** Move to: ").unwrap();
                let (d, missing, n, p) = workspace.edit_parent(p)?;
                if !missing.is_empty() {
                    let p = crate::json::quote(&p);
                    return Err(err(&format!(
                        "move destination {p} is in a missing folder; use Add File + Delete File"
                    )));
                }
                reserve_path(workspace, &mut used, &p)?;
                if Snapshot::read(&d, &n)?.is_some() {
                    return Err(err("move destination already exists or overlaps patch"));
                }
                i += 1;
                Some((d, n, p))
            } else {
                None
            };
        let mut text = before.as_ref().map_or(String::new(), |b| b.text.clone());
        let mut old = String::new();
        let mut new = String::new();
        let mut last_prefix = None;
        let mut hunks = 0;
        let mut cursor = 0;
        let mut edits = Vec::new();
        let mut line_delta = 0isize;
        let mut operations = Vec::new();
        let mut loose = false;
        // An Add File body may omit every '+'. Mixing prefixed and unprefixed
        // lines leaves the intended content unclear, so that is refused.
        let (mut plus, mut bare) = (0, 0);
        for line in lines[i..].iter().take_while(|l| !l.starts_with("*** ")) {
            if !line.is_empty() && *line != "\\ No newline at end of file" {
                if line.starts_with('+') {
                    plus += 1;
                } else {
                    bare += 1;
                }
            }
        }
        if kind == "add" && plus > 0 && bare > 0 {
            let wrong = lines[i..].iter().position(|line| !line.is_empty() && !line.starts_with('+') && !line.starts_with("*** ") && *line != "\\ No newline at end of file").unwrap_or(0) + i + 1;
            return Err(err(&format!(
                "Add File body for {} mixes '+' lines and lines without '+' at patch line {wrong}; start every content line with '+', or none of them",
                crate::json::quote(&normalized)
            )));
        }
        let raw_add = kind == "add" && bare > 0;
        let flush = |old: &mut String,
                     new: &mut String,
                     text: &mut String,
                     cursor: &mut usize,
                     edits: &mut Vec<Edit>,
                     line_delta: &mut isize,
                     operations: &mut Vec<char>|
         -> io::Result<bool> {
            let empty_insert = old.is_empty() && text.is_empty() && *cursor == 0 && !new.is_empty();
            if old.is_empty() && !empty_insert {
                return Err(err(
                    "update hunk requires nonempty exact old text (except an empty file)",
                ));
            }
            let mut matches: Vec<usize> = if empty_insert {
                vec![0]
            } else {
                // Test every line start, including starts inside an earlier
                // match. Non-overlapping substring search hides ambiguity.
                std::iter::once(0).chain(text.match_indices('\n').map(|(i, _)| i + 1))
                    .filter(|&i| {
                        text[i..].starts_with(old.as_str())
                            && (old.ends_with('\n') || i + old.len() == text.len())
                    })
                    .take(2)
                    .collect()
            };
            // Only when exact text is missing: accept a unique block that differs
            // in spaces and tabs alone, keeping the file's own unchanged lines.
            let mut loose = None;
            if matches.is_empty() && !empty_insert {
                let found = loose_matches(text, old);
                loose = Some(found.len());
                if let [(start, end)] = found[..] {
                    if let Some(rebuilt) = keep_context(&text[start..end], new, operations) {
                        *old = text[start..end].to_owned();
                        *new = rebuilt;
                        matches.push(start);
                    }
                }
            }
            if matches.len() != 1 || matches[0] < *cursor {
                let (cause, repair) = if loose.is_some_and(|n| n > 1) {
                    (
                        "old text missing, and ignoring spaces and tabs it matches more than once",
                        "Copy the exact spaces and tabs from workspace_read_lines, or include more unchanged context",
                    )
                } else if matches.is_empty() {
                    ("old text missing", "Re-read the current lines; copy their exact spaces and text into the '-' and space-prefixed lines")
                } else if matches.len() > 1 {
                    ("old text ambiguous", "Include more unchanged space-prefixed context so the old block matches exactly once")
                } else {
                    ("old text out of order", "Resend the @@ hunks in file order, from top to bottom")
                };
                let anchor: String = old.lines().next().unwrap_or("").chars().take(120).collect();
                return Err(err(&format!(
                    "patch conflict in {}: {cause}; expected block begins {}. \
                     Use workspace_search and workspace_read_lines for this path. \
                     {repair}. Resend only the corrected small hunk; no files were written",
                    crate::json::quote(&normalized), crate::json::quote(&anchor)
                )));
            }
            let at = matches[0];
            if !new.is_empty() && !new.ends_with('\n') && at + old.len() < text.len() {
                return Err(err(
                    "no-newline marker requires a replacement at the end of the file",
                ));
            }
            let current_line = text[..at].split_inclusive('\n').count();
            let old_line = (current_line as isize - *line_delta) as usize;
            let old_lines: Vec<&str> = old.split_inclusive('\n').collect();
            let new_lines: Vec<&str> = new.split_inclusive('\n').collect();
            // A no-newline marker can join payload lines. Their prefix counts
            // then differ from the actual applied text: preview that exact text
            // as one replacement rather than trusting the marker line counts.
            if operations.iter().filter(|&&p| p != '+').count() != old_lines.len()
                || operations.iter().filter(|&&p| p != '-').count() != new_lines.len()
            {
                operations.clear();
                operations.resize(old_lines.len(), '-');
                operations.resize(old_lines.len() + new_lines.len(), '+');
            }
            let (mut old_at, mut new_at, mut op) = (0, 0, 0);
            while op < operations.len() {
                if operations[op] == ' ' {
                    old_at += 1;
                    new_at += 1;
                    op += 1;
                    continue;
                }
                let (old_start, new_start) = (old_at, new_at);
                while op < operations.len() && operations[op] != ' ' {
                    old_at += usize::from(operations[op] == '-');
                    new_at += usize::from(operations[op] == '+');
                    op += 1;
                }
                let (mut os, mut ns) = (
                    old_start.min(old_lines.len()),
                    new_start.min(new_lines.len()),
                );
                let (mut oe, mut ne) = (old_at.min(old_lines.len()), new_at.min(new_lines.len()));
                while os < oe && ns < ne && old_lines[os] == new_lines[ns] {
                    os += 1;
                    ns += 1;
                }
                while os < oe && ns < ne && old_lines[oe - 1] == new_lines[ne - 1] {
                    oe -= 1;
                    ne -= 1;
                }
                if os < oe || ns < ne {
                    edits.push(Edit {
                        old_start: old_line + os,
                        old_end: old_line + oe,
                        new_start: current_line + ns,
                        new_end: current_line + ne,
                    });
                }
            }
            operations.clear();
            *line_delta += new_lines.len() as isize - old_lines.len() as isize;
            text.replace_range(at..at + old.len(), new);
            *cursor = at + new.len();
            old.clear();
            new.clear();
            Ok(loose.is_some())
        };
        while i + 1 < lines.len() && !lines[i].starts_with("*** ") {
            let line = lines[i];
            i += 1;
            if line == "@@" || line.starts_with("@@ ") {
                if kind != "update" {
                    return Err(err(
                        "@@ lines belong only in Update File sections; remove them",
                    ));
                }
                if !old.is_empty() || !new.is_empty() {
                    loose |= flush(
                        &mut old,
                        &mut new,
                        &mut text,
                        &mut cursor,
                        &mut edits,
                        &mut line_delta,
                        &mut operations,
                    )?;
                    hunks += 1;
                }
                last_prefix = None;
                continue;
            }
            if line == "\\ No newline at end of file" {
                match last_prefix.take() {
                    Some('-') => {
                        old.pop();
                    }
                    Some('+') => {
                        new.pop();
                    }
                    Some(' ') => {
                        old.pop();
                        new.pop();
                    }
                    _ => {
                        return Err(err(
                            "misplaced no-newline marker; put it directly after a content line",
                        ));
                    }
                }
                continue;
            }
            let (prefix, value) = if raw_add || kind == "add" && line.is_empty() {
                ('+', line)
            } else if kind == "update" && line.is_empty() {
                (' ', "")
            } else {
                (
                    line.chars().next().unwrap_or('\0'),
                    line.get(1..).unwrap_or(""),
                )
            };
            if !matches!(prefix, '+' | '-' | ' ')
                || kind == "add" && prefix != '+'
                || kind == "delete" && prefix != '-'
            {
                let grammar = match kind {
                    "add" => "every Add File content line must start with '+', including empty lines",
                    "delete" => "Delete File needs no body; if one is sent, every line must start with '-' and cover the entire current file",
                    _ => "Update File content lines must start with ' ' (unchanged, so an empty unchanged line is one space), '-' (removed), or '+' (added); use bare @@ between hunks",
                };
                return Err(err(&format!(
                    "invalid patch line prefix in {} at patch line {i}: {grammar}. \
                     Resend plain patch text with these prefixes; no files were written",
                    crate::json::quote(&normalized)
                )));
            }
            operations.push(prefix);
            if prefix != '+' {
                old.push_str(value);
                old.push('\n');
            }
            if prefix != '-' {
                new.push_str(value);
                new.push('\n');
            }
            last_prefix = Some(prefix);
        }
        let after = match kind {
            "add" => Some(new),
            "delete" => {
                if !operations.is_empty() && old != text {
                    return Err(err(&format!(
                        "delete conflict in {0}: the '-' lines are not the entire current file; \
                         resend '*** Delete File: {1}' with no body lines. No files were written",
                        crate::json::quote(&normalized),
                        normalized
                    )));
                }
                None
            }
            _ => {
                if !old.is_empty() || !new.is_empty() {
                    loose |= flush(
                        &mut old,
                        &mut new,
                        &mut text,
                        &mut cursor,
                        &mut edits,
                        &mut line_delta,
                        &mut operations,
                    )?;
                    hunks += 1;
                }
                if hunks == 0 && target.is_none() {
                    return Err(err(
                        "empty update; add an @@ hunk with -/+ lines or a Move to line",
                    ));
                }
                Some(text)
            }
        };
        if let Some(result) = after.as_ref().filter(|s| s.len() > MAX_TEXT_BYTES) {
            return Err(err(&format!(
                "result file is {} bytes; edit result limit is {MAX_TEXT_BYTES} bytes (4 MiB)",
                result.len()
            )));
        }
        diff.push("--- ");
        diff.escaped(if kind == "add" {
            "/dev/null"
        } else {
            &normalized
        });
        diff.push("\n+++ ");
        diff.escaped(target.as_ref().map_or(
            if kind == "delete" {
                "/dev/null"
            } else {
                &normalized
            },
            |t| &t.2,
        ));
        diff.push("\n");
        let old_text = before.as_ref().map_or("", |b| b.text.as_str());
        let new_text = after.as_deref().unwrap_or("");
        if kind != "update" {
            edits.push(Edit {
                old_start: 0,
                old_end: old_text.split_inclusive('\n').count(),
                new_start: 0,
                new_end: new_text.split_inclusive('\n').count(),
            });
        }
        unified_preview(&mut diff, old_text, new_text, &edits);
        if loose {
            loose_paths.push(normalized.clone());
        }
        changes.push(Change {
            dir,
            missing,
            name,
            path: normalized,
            before,
            after,
            target,
        });
    }
    if changes.is_empty() {
        return Err(err(
            "empty patch; add an Add, Update or Delete File section",
        ));
    }
    Ok(PreparedPatch {
        changes,
        diff: diff.finish()?,
        patch: patch.to_owned(),
        loose: loose_paths,
    })
}

/// Reserve files and move destinations. Two spellings that differ only by
/// case are one file or folder exactly when the disk says so (see
/// `Workspace::same_entry`): then the batch is refused before any write. On a
/// case-sensitive disk they are two entries and the batch is valid.
fn reserve_path(workspace: &Workspace, used: &mut Vec<String>, path: &str) -> io::Result<()> {
    for previous in used.iter() {
        let mut a = previous.split('/');
        let mut b = path.split('/');
        let mut parent = Vec::new();
        loop {
            match (a.next(), b.next()) {
                (Some(a), Some(b)) if a == b => parent.push(a),
                (Some(a), Some(b)) => {
                    if a.eq_ignore_ascii_case(b) {
                        let same = workspace.same_entry(&parent, a, b)?;
                        let (a, b) = (crate::json::quote(a), crate::json::quote(b));
                        match same {
                            Some(false) => (),
                            Some(true) => return Err(err(&format!(
                                "patch paths spell one file or folder two ways ({a} and {b}); this disk ignores case, so use one exact spelling"
                            ))),
                            None => return Err(err(&format!(
                                "cannot tell whether this disk treats {a} and {b} as one name; send them in separate patches"
                            ))),
                        }
                    }
                    break;
                }
                (None, None) => return Err(err(
                    "patch touches a path more than once; put all changes to one file in one section",
                )),
                _ => return Err(err(&format!(
                    "patch uses {} or its folder as both a file and a folder; send them in separate patches",
                    crate::json::quote(path)
                ))),
            }
        }
    }
    used.push(path.to_owned());
    Ok(())
}

/// Line blocks equal to `old` when runs of spaces and tabs are ignored. Words
/// and line endings (including a carriage return) must still agree, and so
/// must the indent width (a tab counts 4 or 8 columns), so a block at another
/// nesting depth never matches. At most two starts and ends are returned.
fn loose_matches(text: &str, old: &str) -> Vec<(usize, usize)> {
    fn words(line: &str) -> impl Iterator<Item = &str> {
        let line = line.strip_suffix('\n').unwrap_or(line);
        line.split([' ', '\t']).filter(|w| !w.is_empty())
    }
    fn indent(line: &str, tab: usize) -> Option<usize> {
        let mut width = 0;
        for c in line.chars() {
            match c {
                ' ' => width += 1,
                '\t' => width += tab,
                '\n' | '\r' => return None,
                _ => return Some(width),
            }
        }
        None
    }
    let same_indent = |a: &str, b: &str| {
        indent(a, 4).is_none() || indent(a, 4) == indent(b, 4) || indent(a, 8) == indent(b, 8)
    };
    let old: Vec<&str> = old.split_inclusive('\n').collect();
    let mut at = 0;
    let lines: Vec<(usize, &str)> = text
        .split_inclusive('\n')
        .map(|line| {
            at += line.len();
            (at - line.len(), line)
        })
        .collect();
    lines
        .windows(old.len())
        .filter(|block| {
            block
                .iter()
                .zip(&old)
                .all(|((_, a), b)| {
                    a.ends_with('\n') == b.ends_with('\n') && words(a).eq(words(b)) && same_indent(a, b)
                })
        })
        .take(2)
        .map(|block| {
            (
                block[0].0,
                block[old.len() - 1].0 + block[old.len() - 1].1.len(),
            )
        })
        .collect()
}

/// New text for a loose match: unchanged lines keep the file's bytes and added
/// lines keep the patch's. None when a no-newline marker joined patch lines.
fn keep_context(actual: &str, new: &str, operations: &[char]) -> Option<String> {
    let actual: Vec<&str> = actual.split_inclusive('\n').collect();
    let added: Vec<&str> = new.split_inclusive('\n').collect();
    if operations.iter().filter(|&&p| p != '+').count() != actual.len()
        || operations.iter().filter(|&&p| p != '-').count() != added.len()
    {
        return None;
    }
    let (mut a, mut n, mut rebuilt) = (0, 0, String::new());
    for &op in operations {
        match op {
            ' ' => rebuilt.push_str(actual[a]),
            '+' => rebuilt.push_str(added[n]),
            _ => (),
        }
        a += usize::from(op != '+');
        n += usize::from(op != '-');
    }
    Some(rebuilt)
}

#[cfg(test)]
mod tests;
