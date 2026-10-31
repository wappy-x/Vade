//! Deterministic checks. Each `check:` line gives one result: pass or fail, and
//! a short reason. No model judges an answer.
//!
//!   last-exit-zero                       the last harness step exited 0
//!   held-closed                          a held model call was canceled by the harness (socket closed)
//!   exists PATH                          the workspace path exists
//!   missing PATH                         the workspace path does not exist
//!   file-equals PATH FIXTURE             same text after line-end normalization (see `normal`)
//!   json-equals PATH FIXTURE             both parse as JSON and are equal (member order free)
//!   contains PATH TEXT...                the file holds TEXT
//!   starts-with PATH TEXT...             the file starts with TEXT
//!   absent PATH TEXT...                  no file under PATH holds TEXT
//!   count PATH N TEXT...                 TEXT occurs exactly N times under PATH
//!   count-min PATH N TEXT...             TEXT occurs at least N times under PATH
//!   unchanged PATH                       file or folder bytes equal the starting workspace
//!   reply-contains TEXT...               the last step's printed reply holds TEXT
//!   run ARGV...                          the command exits 0 in the workspace
//!   run-with FIXTURE DEST -- ARGV...     copy FIXTURE over DEST, run, restore; must exit 0
//!   fails-with FIXTURE DEST -- ARGV...   exits 0 as is; with FIXTURE copied over DEST, exits nonzero
use super::same;
use crate::json;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub struct Ctx<'a> {
    pub work: &'a Path,
    pub start: &'a Path,
    pub task_dir: &'a Path,
    pub last_exit: Option<i32>,
    pub last_stdout: &'a str,
    pub held_closed: bool,
    pub env: &'a [(String, String)],
    pub log: &'a Path,
}

#[derive(Clone, Debug)]
pub struct Outcome { pub check: String, pub pass: bool, pub detail: String }

pub fn run(line: &str, c: &Ctx) -> Outcome {
    let words: Vec<&str> = line.split_whitespace().collect();
    let rest = |n: usize| -> String { line.split_whitespace().skip(n).collect::<Vec<_>>().join(" ") };
    let result: Result<String, String> = match words.as_slice() {
        ["last-exit-zero"] => match c.last_exit { Some(0) => Ok("exit 0".into()), other => Err(format!("exit {other:?}")) },
        ["held-closed"] => if c.held_closed { Ok("held call closed by the harness".into()) } else { Err("no held call was closed".into()) },
        ["exists", p] => if std::fs::symlink_metadata(c.work.join(p)).is_ok() { Ok("present".into()) } else { Err("missing".into()) },
        ["missing", p] => if std::fs::symlink_metadata(c.work.join(p)).is_ok() { Err("present".into()) } else { Ok("missing as required".into()) },
        ["file-equals", p, f] => file_equals(&c.work.join(p), &c.task_dir.join(f)),
        ["json-equals", p, f] => json_equals(&c.work.join(p), &c.task_dir.join(f)),
        ["contains", p, _, ..] => read(&c.work.join(p)).and_then(|t| if t.contains(&rest(2)) { Ok("found".into()) } else { Err("not found".into()) }),
        ["absent", p, _, ..] => occurrences(&c.work.join(p), &rest(2)).and_then(|(n, at)| if n == 0 { Ok("0 occurrences".into()) } else { Err(format!("{n} occurrences, first in {at}")) }),
        ["count", p, n, _, ..] => {
            let want: usize = n.parse().unwrap_or(usize::MAX);
            occurrences(&c.work.join(p), &rest(3)).and_then(|(got, _)| if got == want { Ok(format!("{got} occurrences")) } else { Err(format!("{got} occurrences, want {want}")) })
        }
        ["count-min", p, n, _, ..] => {
            let want: usize = n.parse().unwrap_or(usize::MAX);
            occurrences(&c.work.join(p), &rest(3)).and_then(|(got, _)| if got >= want { Ok(format!("{got} occurrences")) } else { Err(format!("{got} occurrences, want at least {want}")) })
        }
        ["starts-with", p, _, ..] => read(&c.work.join(p)).and_then(|t| if t.starts_with(&rest(2)) { Ok("first text matches".into()) } else { Err(format!("starts with {:?}", t.lines().next().unwrap_or(""))) }),
        ["unchanged", p] => same_tree(&c.start.join(p), &c.work.join(p)),
        ["reply-contains", _, ..] => if c.last_stdout.contains(&rest(1)) { Ok("found in reply".into()) } else { Err("not in reply".into()) },
        ["run", argv @ ..] if !argv.is_empty() => command(argv, c).and_then(|(code, tail)| if code == 0 { Ok("exit 0".into()) } else { Err(format!("exit {code}: {tail}")) }),
        ["fails-with", _, _, "--", argv @ ..] if !argv.is_empty() && !matches!(command(argv, c), Ok((0, _))) => {
            // A command that fails on the agent's own files proves nothing about the swap.
            Err("the command fails before the swap too".into())
        }
        [kind @ ("run-with" | "fails-with"), fixture, dest, "--", argv @ ..] if !argv.is_empty() => {
            swap_run(&c.task_dir.join(fixture), &c.work.join(dest), argv, c).and_then(|(code, tail)| match (*kind, code) {
                ("run-with", 0) => Ok("exit 0".into()),
                ("run-with", _) => Err(format!("exit {code}: {tail}")),
                (_, 0) => Err("exit 0, but the swapped-in file must make it fail".into()),
                _ => Ok(format!("exit {code} as required")),
            })
        }
        _ => Err("unknown check".into()),
    };
    let (pass, detail) = match result { Ok(d) => (true, d), Err(d) => (false, d) };
    Outcome { check: line.to_string(), pass, detail }
}

/// True when the check line names a check this module knows, with its words.
pub fn known(line: &str) -> bool {
    let words: Vec<&str> = line.split_whitespace().collect();
    matches!(words.as_slice(),
        ["last-exit-zero"] | ["held-closed"] | ["exists", _] | ["missing", _] | ["file-equals", _, _] | ["json-equals", _, _]
        | ["contains", _, _, ..] | ["absent", _, _, ..] | ["count", _, _, _, ..] | ["count-min", _, _, _, ..] | ["starts-with", _, _, ..]
        | ["unchanged", _] | ["reply-contains", _, ..] | ["run", _, ..] | ["run-with" | "fails-with", _, _, "--", _, ..])
}

/// CRLF becomes LF, trailing spaces leave each line, trailing empty lines go.
pub fn normal(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches(['\r', ' ', '\t'])).collect();
    let mut end = lines.len();
    while end > 0 && lines[end - 1].is_empty() { end -= 1; }
    lines[..end].join("\n")
}

fn read(p: &Path) -> Result<String, String> { std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display())) }

fn file_equals(got: &Path, want: &Path) -> Result<String, String> {
    let (g, w) = (normal(&read(got)?), normal(&read(want)?));
    if g == w { return Ok(format!("{} lines equal", w.lines().count())); }
    let (gl, wl): (Vec<_>, Vec<_>) = (g.lines().collect(), w.lines().collect());
    let i = gl.iter().zip(&wl).position(|(a, b)| a != b).unwrap_or(gl.len().min(wl.len()));
    Err(format!("line {}: got {:?}, want {:?}", i + 1, gl.get(i).unwrap_or(&"<end>"), wl.get(i).unwrap_or(&"<end>")))
}

fn json_equals(got: &Path, want: &Path) -> Result<String, String> {
    let g = json::parse(read(got)?.trim()).map_err(|e| format!("answer is not JSON: {e}"))?;
    let w = json::parse(read(want)?.trim()).map_err(|e| format!("fixture is not JSON: {e}"))?;
    if same(&g, &w) { Ok("JSON equal".into()) } else { Err(format!("got {}", super::write(&g))) }
}

/// Every file and symbolic link under `p`, sorted. Links are not followed.
/// `target` (build output) and `.git` are skipped; other dot names count.
pub fn files(p: &Path, out: &mut Vec<PathBuf>) {
    let Ok(meta) = std::fs::symlink_metadata(p) else { return };
    if meta.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| e.path()).collect();
        entries.sort();
        for e in entries {
            let name = e.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "target" || name == ".git" { continue; }
            files(&e, out);
        }
    } else {
        out.push(p.to_path_buf());
    }
}

/// The bytes of a file, or `symlink -> TARGET` for a link (never followed).
pub fn content(p: &Path) -> Option<Vec<u8>> {
    match std::fs::read_link(p) {
        Ok(to) => Some(format!("symlink -> {}", to.display()).into_bytes()),
        Err(_) => std::fs::read(p).ok(),
    }
}

fn occurrences(p: &Path, text: &str) -> Result<(usize, String), String> {
    if !p.exists() { return Err(format!("{} is missing", p.display())); }
    let mut list = Vec::new();
    files(p, &mut list);
    let (mut n, mut first) = (0, String::new());
    for f in list {
        let Some(t) = content(&f) else { continue };
        let k = String::from_utf8_lossy(&t).matches(text).count();
        if k > 0 && first.is_empty() { first = f.file_name().unwrap_or_default().to_string_lossy().into(); }
        n += k;
    }
    Ok((n, first))
}

fn same_tree(a: &Path, b: &Path) -> Result<String, String> {
    let (mut fa, mut fb) = (Vec::new(), Vec::new());
    files(a, &mut fa);
    files(b, &mut fb);
    let rel = |base: &Path, v: &[PathBuf]| v.iter().map(|p| p.strip_prefix(base).unwrap_or(p).to_path_buf()).collect::<Vec<_>>();
    let (ra, rb) = (rel(a, &fa), rel(b, &fb));
    if ra != rb {
        let added = rb.iter().find(|p| !ra.contains(p)).map(|p| format!("; {} is new", p.display()));
        let gone = ra.iter().find(|p| !rb.contains(p)).map(|p| format!("; {} is gone", p.display()));
        return Err(format!("file list differs: {} vs {} files{}", ra.len(), rb.len(), added.or(gone).unwrap_or_default()));
    }
    for (x, y) in fa.iter().zip(&fb) {
        if content(x) != content(y) { return Err(format!("{} changed", y.strip_prefix(b).unwrap_or(y).display())); }
    }
    Ok(format!("{} files unchanged", fa.len()))
}

fn command(argv: &[&str], c: &Ctx) -> Result<(i32, String), String> {
    let log = std::fs::OpenOptions::new().create(true).append(true).open(c.log).map_err(|e| e.to_string())?;
    let mut cmd = Command::new(argv[0]);
    cmd.args(&argv[1..]).current_dir(c.work).stdin(Stdio::null()).stdout(log.try_clone().map_err(|e| e.to_string())?).stderr(log);
    // A fixture crate is not Vade code: the caller's lint flags (CI sets -D warnings) do not apply to it.
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS"] { cmd.env_remove(name); }
    for (k, v) in c.env { cmd.env(k, v); }
    let mut child = cmd.spawn().map_err(|e| format!("{}: {e}", argv[0]))?;
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? { break s; }
        if start.elapsed() > Duration::from_secs(600) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("check command passed 600 s".into());
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let text = std::fs::read_to_string(c.log).unwrap_or_default();
    let tail = text.lines().rev().find(|l| l.contains("error") || l.contains("FAILED") || l.contains("panicked")).unwrap_or("").chars().take(160).collect();
    Ok((status.code().unwrap_or(-1), tail))
}

fn swap_run(fixture: &Path, dest: &Path, argv: &[&str], c: &Ctx) -> Result<(i32, String), String> {
    let saved = std::fs::read(dest).ok();
    if let Some(parent) = dest.parent() { std::fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    // Write, not copy: a copy keeps the old mtime on macOS and cargo would not rebuild.
    std::fs::read(fixture).and_then(|b| std::fs::write(dest, b)).map_err(|e| format!("{}: {e}", fixture.display()))?;
    let result = command(argv, c);
    match saved { Some(b) => std::fs::write(dest, b).map_err(|e| e.to_string())?, None => std::fs::remove_file(dest).map_err(|e| e.to_string())? }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization() {
        assert_eq!(normal("a \r\nb\t\n\n\n"), "a\nb");
        assert_eq!(normal("a\n"), normal("a"));
        assert_ne!(normal("a\n\nb"), normal("a\nb"));
    }
    #[test]
    fn known_checks() {
        assert!(known("run cargo test --offline"));
        assert!(known("fails-with mutants/a.rs src/lib.rs -- cargo test"));
        assert!(known("missing src/v1"));
        assert!(!known("fails-with mutants/a.rs src/lib.rs cargo test"));
        assert!(!known("file-equals a"));
        assert!(!known("max-turns 6"));
    }
}
