//! The quality rubric (`rubric.txt`, next to `task.txt`): a score from 0 to 3
//! for each task run, from exact rules and no model judge.
//!
//! rubric.txt: `key: value` lines; `#` starts a comment line.
//!   may_change: PATH...          files or folders the agent may change or create
//!   max_changed_lines: N         added plus removed lines over all changed files
//!   answer_mentions: TEXT        the final reply holds TEXT (repeatable)
//!   answer_max_chars: N          the final reply has at most N characters
//!
//! The score is 0 when a check fails (the task is not correct). Otherwise it
//! is 3 minus one for each missed criterion: minimal (changed lines), no
//! collateral edits (every changed path is in may_change) and a clear final
//! answer (mentions and length).
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct Rubric { pub may_change: Vec<String>, pub max_changed_lines: usize, pub answer_mentions: Vec<String>, pub answer_max_chars: usize }

#[derive(Clone, Debug)]
pub struct Quality { pub score: u8, pub misses: Vec<String>, pub changed_lines: usize }

pub fn parse(text: &str) -> Result<Rubric, String> {
    let mut r = Rubric::default();
    let (mut lines_set, mut chars_set) = (false, false);
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') { continue; }
        let (key, value) = line.split_once(':').ok_or_else(|| format!("line {}: expected key: value", i + 1))?;
        let value = value.trim();
        let number = || value.parse::<usize>().map_err(|_| format!("line {}: {} needs a number", i + 1, key.trim()));
        match key.trim() {
            "may_change" => r.may_change.extend(value.split_whitespace().map(|p| p.trim_end_matches('/').to_string())),
            "max_changed_lines" => { r.max_changed_lines = number()?; lines_set = true; }
            "answer_mentions" if !value.is_empty() => r.answer_mentions.push(value.to_string()),
            "answer_max_chars" => { r.answer_max_chars = number()?; chars_set = true; }
            other => return Err(format!("line {}: unknown key {other}", i + 1)),
        }
    }
    if !lines_set || !chars_set { return Err("the rubric needs max_changed_lines and answer_max_chars".into()); }
    Ok(r)
}

impl Rubric {
    /// True when the workspace path is one the agent may change.
    pub fn allows(&self, path: &str) -> bool {
        let path = path.trim_start_matches("./");
        self.may_change.iter().any(|p| p == "." || path == p || path.strip_prefix(p.as_str()).is_some_and(|rest| rest.starts_with('/')))
    }

    /// Scores one task run. `removed` lists the paths that the runner deleted
    /// (`remove_before`); their removal is not the agent's edit, but a file
    /// that the agent writes again at such a path counts.
    pub fn score(&self, correct: bool, start: &Path, work: &Path, removed: &[String], reply: &str) -> Quality {
        let mut changes = changes(start, work);
        changes.retain(|(p, _)| !removed.contains(p) || std::fs::symlink_metadata(work.join(p)).is_ok());
        let changed_lines = changes.iter().map(|(_, n)| n).sum();
        if !correct { return Quality { score: 0, misses: vec!["not correct: a check failed".into()], changed_lines }; }
        let mut misses = Vec::new();
        if changed_lines > self.max_changed_lines { misses.push(format!("not minimal: {changed_lines} changed lines, at most {}", self.max_changed_lines)); }
        if let Some((path, _)) = changes.iter().find(|(p, _)| !self.allows(p)) { misses.push(format!("collateral edit: {path}")); }
        let reply = reply.trim();
        if reply.is_empty() {
            misses.push("no final answer".into());
        } else if let Some(m) = self.answer_mentions.iter().find(|m| !reply.contains(m.as_str())) {
            misses.push(format!("unclear answer: it does not mention {m}"));
        } else if reply.chars().count() > self.answer_max_chars {
            misses.push(format!("unclear answer: {} characters, at most {}", reply.chars().count(), self.answer_max_chars));
        }
        Quality { score: 3 - misses.len() as u8, misses, changed_lines }
    }
}

/// The files that differ between the two trees, with the number of added and
/// removed lines of each. `target` and `.git` are skipped, as in the checks,
/// and so is a `Cargo.lock` that a cargo command created.
pub fn changes(start: &Path, work: &Path) -> Vec<(String, usize)> {
    let (a, b) = (files(start), files(work));
    let mut out = Vec::new();
    for (path, new) in &b {
        match a.get(path) {
            Some(old) if old == new => {}
            Some(old) => out.push((path.clone(), diff_lines(old, new))),
            None if path == "Cargo.lock" || path.ends_with("/Cargo.lock") => {}
            None => out.push((path.clone(), String::from_utf8_lossy(new).lines().count().max(1))),
        }
    }
    for (path, old) in &a {
        if !b.contains_key(path) { out.push((path.clone(), String::from_utf8_lossy(old).lines().count().max(1))); }
    }
    out.sort();
    out
}

fn files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut list = Vec::new();
    super::check::files(root, &mut list);
    list.iter().filter_map(|p| Some((p.strip_prefix(root).unwrap_or(p).to_string_lossy().to_string(), super::check::content(p)?))).collect()
}

/// Added plus removed lines between two texts: both lengths minus twice the
/// longest common subsequence of lines. Large files count every line.
fn diff_lines(old: &[u8], new: &[u8]) -> usize {
    let (old, new) = (String::from_utf8_lossy(old), String::from_utf8_lossy(new));
    let (a, b): (Vec<&str>, Vec<&str>) = (old.lines().collect(), new.lines().collect());
    if a.len().saturating_mul(b.len()) > 4_000_000 { return a.len() + b.len(); }
    let mut row = vec![0usize; b.len() + 1];
    for x in &a {
        let mut diag = 0;
        for (j, y) in b.iter().enumerate() {
            let up = row[j + 1];
            row[j + 1] = if x == y { diag + 1 } else { row[j + 1].max(row[j]) };
            diag = up;
        }
    }
    a.len() + b.len() - 2 * row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rubric_lines() {
        let r = parse("# x\nmay_change: src/lib.rs tests/\nmax_changed_lines: 4\nanswer_mentions: limit\nanswer_max_chars: 200\n").unwrap();
        assert_eq!(r.may_change, vec!["src/lib.rs", "tests"]);
        assert!(r.allows("tests/a.rs") && r.allows("src/lib.rs") && !r.allows("src/lib.rsx") && !r.allows("testsx/a.rs"));
        assert!(parse("may_change: a\n").unwrap_err().contains("max_changed_lines"));
        assert!(parse("max_changed_lines: x\nanswer_max_chars: 1\n").is_err());
        assert!(parse("bogus: 1\n").unwrap_err().contains("unknown key"));
    }

    #[test]
    fn line_diff_counts_added_and_removed() {
        assert_eq!(diff_lines(b"a\nb\nc\n", b"a\nb\nc\n"), 0);
        assert_eq!(diff_lines(b"a\nb\nc\n", b"a\nX\nc\n"), 2);
        assert_eq!(diff_lines(b"a\nc\n", b"a\nb\nc\nd\n"), 2);
        assert_eq!(diff_lines(b"", b"x\n"), 1);
    }

    #[test]
    fn score_counts_each_miss() {
        let dir = std::env::temp_dir().join(format!("vade-suite-rubric-test-{}", std::process::id()));
        let (start, work) = (dir.join("start"), dir.join("work"));
        for d in [&start, &work] { std::fs::create_dir_all(d.join("src")).unwrap(); std::fs::write(d.join("src/lib.rs"), "a\nb\n").unwrap(); }
        std::fs::write(work.join("src/lib.rs"), "a\nB\n").unwrap();
        std::fs::write(work.join("Cargo.lock"), "made by cargo\n").unwrap();
        let r = parse("may_change: src/lib.rs\nmax_changed_lines: 2\nanswer_mentions: lib.rs\nanswer_max_chars: 40\n").unwrap();
        let q = r.score(true, &start, &work, &[], "Changed src/lib.rs.");
        assert_eq!((q.score, q.changed_lines), (3, 2), "{:?}", q.misses);
        assert_eq!(r.score(false, &start, &work, &[], "Changed src/lib.rs.").score, 0);
        std::fs::write(work.join("notes.txt"), "x\n").unwrap();
        let q = r.score(true, &start, &work, &[], "Done, and here is a very long answer with no file name.");
        assert_eq!(q.score, 0, "{:?}", q.misses);
        assert_eq!(q.misses.len(), 3, "{:?}", q.misses);
        assert!(q.misses[1].contains("collateral edit: notes.txt"), "{:?}", q.misses);
        std::fs::remove_file(work.join("notes.txt")).unwrap();
        std::fs::remove_file(work.join("src/lib.rs")).unwrap();
        assert_eq!(r.score(true, &start, &work, &["src/lib.rs".into()], "lib.rs").score, 3);
        // A hidden file counts, and so does a file written again at a removed path.
        std::fs::write(work.join(".env"), "x\n").unwrap();
        assert!(r.score(true, &start, &work, &[], "lib.rs").misses.iter().any(|m| m.contains("collateral edit: .env")));
        std::fs::remove_file(work.join(".env")).unwrap();
        std::fs::write(start.join("notes.txt"), "a\n").unwrap();
        std::fs::write(work.join("notes.txt"), "b\n").unwrap();
        assert!(r.score(true, &start, &work, &["notes.txt".into()], "lib.rs").misses.iter().any(|m| m.contains("collateral edit: notes.txt")));
        // A link is compared as a link and never followed.
        #[cfg(unix)]
        {
            std::fs::remove_file(work.join("notes.txt")).unwrap();
            std::os::unix::fs::symlink("..", work.join("up")).unwrap();
            let c = changes(&start, &work);
            assert!(c.iter().any(|(p, n)| p == "up" && *n == 1) && !c.iter().any(|(p, _)| p.starts_with("up/")), "{c:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
