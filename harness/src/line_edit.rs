//! Latest-read capabilities: exact bytes and visible lines, never hash trust alone.
use super::{err, Change, PreparedPatch, Preview, Snapshot, MAX_PATCH, MAX_TEXT_BYTES};
use crate::{
    json::{self, Value},
    tools::Workspace,
};
use std::{
    hash::{Hash, Hasher},
    io,
    sync::atomic::AtomicBool,
};

pub(crate) fn tag(path: &str, text: &str) -> String {
    // A compact correlation tag; retained exact bytes enforce freshness.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    (path, text).hash(&mut hash);
    format!("lines-{:016x}", hash.finish())
}
pub(crate) struct Read {
    path: String,
    text: String,
    shown: Vec<usize>,
}
impl Read {
    pub(crate) fn capture(path: &str, text: String, output: &str) -> io::Result<Self> {
        let value = json::parse(output)?;
        let shown = value
            .get("lines")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|row| row.get("line").and_then(|n| if let Value::Number(n) = n { n.parse().ok() } else { None }))
            .collect();
        Ok(Self { path: path.into(), text, shown })
    }
    pub(crate) fn prepare(&self, workspace: &Workspace, args: &Value) -> io::Result<PreparedPatch> {
        let fail = |s: &str| io::Error::other(s);
        let Value::Object(fields) = args else {
            return Err(fail("expected an object"));
        };
        if fields.len() != 6
            || fields.iter().any(|(k, _)| !["path", "start_line", "end_line", "edit_tag", "operation", "new_text"].contains(&k.as_str()))
        {
            return Err(fail("expected path, start_line, end_line, edit_tag, operation, new_text"));
        }
        let string = |key| args.get(key).and_then(Value::as_str).ok_or_else(|| fail("expected string field"));
        let number = |key| match args.get(key) {
            Some(Value::Number(n)) => n.parse::<usize>().map_err(|_| fail("expected positive line number")),
            _ => Err(fail("expected line number")),
        };
        let (start, end) = (number("start_line")?, number("end_line")?);
        let (path, operation, new) = (string("path")?, string("operation")?, string("new_text")?);
        if path != self.path || string("edit_tag")? != tag(path, &self.text) {
            return Err(fail("tag/path does not match latest workspace_read_lines; re-read"));
        }
        let total = self.text.split_inclusive('\n').count();
        if start == 0 || end < start || end > total.max(1) || end - start >= 200 {
            return Err(fail("line range out of bounds (empty file uses 1..1); re-read"));
        }
        if total > 0 && !(start..=end).all(|n| self.shown.contains(&n)) {
            return Err(fail("range was not fully shown by latest workspace_read_lines; re-read"));
        }
        if !matches!(operation, "replace" | "delete" | "insert_after")
            || operation == "insert_after" && start != end
            || operation == "delete" && !new.is_empty()
        {
            return Err(fail("use replace, delete (new_text empty), or insert_after (start=end)"));
        }
        prepare_lines(workspace, path, &self.text, start, end, operation, new)
    }
}

pub(crate) fn current(workspace: &Workspace, args: &Value, cancel: &AtomicBool, cap: usize) -> String {
    let path = args.get("path").and_then(Value::as_str).unwrap_or("");
    let number = |key| args.get(key).and_then(|n| if let Value::Number(n) = n { n.parse::<usize>().ok() } else { None });
    let start = number("start_line").unwrap_or(1).max(1);
    let end = number("end_line").unwrap_or(start).max(start).min(start.saturating_add(199));
    workspace.read_lines(path, start, end, cap.min(4096), cancel).unwrap_or_else(|e| format!("Current lines unavailable: {e}"))
}

pub(crate) fn patch_hint(workspace: &Workspace, patch: &str, error: &str) -> String {
    if !error.contains("old text") {
        return String::new();
    }
    let mut hint = String::from(
        "\nUse workspace_edit_lines after workspace_read_lines; copy edit_tag and line numbers, supply only new_text (no old text).",
    );
    for section in patch.split("*** Update File: ").skip(1).take(32) {
        let Some((path, body)) = section.split_once('\n') else {
            continue;
        };
        let path = path.trim_end_matches('\r');
        if error.contains("patch conflict in ") && !error.contains(&format!("in {}:", json::quote(path))) {
            continue;
        }
        let Ok((_, text)) = workspace.read_line_ranges_capture(path, &[(1, 1)], 4096, &AtomicBool::new(false)) else {
            continue;
        };
        let old: Vec<&str> = body
            .lines()
            .take_while(|line| !line.starts_with("*** "))
            .filter_map(|line| line.strip_prefix(' ').or_else(|| line.strip_prefix('-')))
            .take(200)
            .collect();
        let source: Vec<&str> = text.lines().collect();
        let mut best = None;
        for (offset, needle) in old.iter().enumerate().filter(|(_, line)| line.trim().len() >= 4).take(20) {
            for (index, _) in source.iter().enumerate().filter(|(_, line)| line.trim() == needle.trim()).take(32) {
                let start = index.saturating_sub(offset);
                let score = source[start..].iter().zip(&old).filter(|(a, b)| a.trim() == b.trim()).count();
                if best.is_none_or(|(_, n)| score > n) {
                    best = Some((start, score));
                }
            }
        }
        if let Some((start, _)) = best {
            hint.push_str(&format!(" Closest match: workspace_read_lines {{\"path\":{},\"start_line\":{},\"end_line\":{}}}, then workspace_edit_lines with that range's edit_tag.", json::quote(path), start + 1, (start + old.len()).min(source.len())));
        }
    }
    hint
}

/// A separate edit form; exact-patch parsing and matching stay unchanged.
fn prepare_lines(
    workspace: &Workspace,
    path: &str,
    seen: &str,
    start: usize,
    end: usize,
    operation: &str,
    new: &str,
) -> io::Result<PreparedPatch> {
    if new.len() > MAX_PATCH {
        return Err(err("new_text exceeds 64 KiB"));
    }
    let (dir, missing, name, path) = workspace.edit_parent(path)?;
    let before = Snapshot::read(&dir, &name)?.ok_or_else(|| err("line edit requires an existing file"))?;
    if !missing.is_empty() || before.text != seen {
        return Err(err("stale edit_tag: file changed since workspace_read_lines; re-read"));
    }
    let lines: Vec<&str> = seen.split_inclusive('\n').collect();
    let (from, to) = if seen.is_empty() {
        (0, 0)
    } else {
        (lines[..start - 1].iter().map(|s| s.len()).sum(), lines[..end].iter().map(|s| s.len()).sum())
    };
    let (from, payload) = if operation == "insert_after" {
        let separator = if to > 0 && !seen[..to].ends_with('\n') && !new.is_empty() {
            if seen.contains("\r\n") {
                "\r\n"
            } else {
                "\n"
            }
        } else {
            ""
        };
        (to, format!("{separator}{new}"))
    } else {
        (from, new.to_owned())
    };
    if to < seen.len() && !payload.is_empty() && !payload.ends_with('\n') {
        return Err(err("new_text before later lines must end with a newline"));
    }
    let after = format!("{}{}{}", &seen[..from], payload, &seen[to..]);
    if after.len() > MAX_TEXT_BYTES {
        return Err(err("result exceeds 4 MiB"));
    }
    let mut diff = Preview::default();
    diff.push("Preview escapes: backslash is \\\\, tab is \\t, carriage return is \\r; nonprinting Unicode is \\u{hex}.\n--- ");
    diff.escaped(&path);
    diff.push("\n+++ ");
    diff.escaped(&path);
    diff.push(&format!("\n@@ lines {start}..{end} {operation} @@\n"));
    // An insertion after an unterminated last line also terminates that line.
    let preview_from = if operation == "insert_after" && to > 0 && !seen[..to].ends_with('\n') { to - lines[end - 1].len() } else { from };
    for line in seen[preview_from..to].split_inclusive('\n') {
        diff.line('-', line);
    }
    for line in format!("{}{}", &seen[preview_from..from], payload).split_inclusive('\n') {
        diff.line('+', line);
    }
    let patch = format!("{path}:{start}:{end}:{operation}:{}:{new}", crate::line_edit::tag(path.as_str(), seen));
    Ok(PreparedPatch {
        changes: vec![Change { dir, missing, name, path, before: Some(before), after: Some(after), target: None }],
        diff: diff.finish()?,
        patch,
        loose: Vec::new(),
    })
}
