//! `vade-harness suite`: runs the agent task suite (`tests/agent-suite`) with
//! this harness. Each task runs `vade-harness agent` in a fresh copy of its
//! workspace against a loopback endpoint, then deterministic checks decide
//! pass or fail. No model judges a result.
//!
//! `--dry-run` needs no model and no GPU: the endpoint replays each task's
//! `solve.script` (every check must pass) and `wrong.script` (a check must
//! fail) through the real tool executor. `--live` records the exchanges with
//! a served model through a proxy, writes a card per run and, over several
//! runs, compares the streamed token IDs of every task.
//! `--replay` runs saved proxy replies through the same checks and rubric.
//! It writes completion and quality cards without a model or Spark.
mod card;
mod check;
mod digest;
mod replay;
mod replay_record;
mod rubric;
mod serve;
mod task;

use crate::json::{self, Value};
use card::live;
use serve::{Server, Source};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

const HELP: &str = "Usage:
  vade-harness suite --dry-run [--suite DIR] [--out DIR] [--only ID,ID] [--toolchain DIR]
  vade-harness suite --replay DIR [--suite DIR] [--out DIR] [--only ID,ID] [--toolchain DIR]
  vade-harness suite --live --endpoint URL --model NAME --out DIR [--runs N] [--suite DIR]
                     [--only ID,ID] [--toolchain DIR] [--harness-arg ARG]...

Runs the agent task suite with this harness. Each task run is graded on
completion (every check in task.txt passes) and on quality (0-3 from
rubric.txt). See docs/harness-suite.md.

  --dry-run          Replay solve.script (all checks must pass) and wrong.script
                     (a check must fail) for every task. No model, no GPU.
  --live             Run every task against a served model through a recording
                     proxy. Writes OUT/run-N/card.{json,md} and OUT/card.{json,md}.
  --replay DIR       Grade recorded replies without a model. DIR has replay.json
                     with suite_version and spark_count, and tasks/ID/transcript.jsonl.
                     Writes cards to DIR/replay-cards unless --out is set.
  --endpoint URL     The server address, for example http://127.0.0.1:18080
                     (live only). Chat calls go to /v1/chat/completions on it.
  --model NAME       The served model name (live only).
  --runs N           Live runs, 1 to 9 (default 1). Three runs give a grade.
  --suite DIR        The suite folder (default harness/tests/agent-suite or
                     tests/agent-suite under the current folder).
  --out DIR          A new folder for the results. A dry run without --out
                     uses a new temporary folder and removes it at the end.
  --only ID,ID       Run only these tasks.
  --toolchain DIR    The Rust distribution for checks and the command sandbox
                     (default: the one that runs cargo or rustc).
  --harness-arg ARG  Pass ARG to each harness run (live only).

Exit status: 0 when the dry-run gate passes or every live or replay task passes,
1 when a task fails, 2 on a usage or setup error.
";

enum Mode { DryRun, Live { endpoint: String, model: String, runs: usize }, Replay { dir: PathBuf } }

struct Opts {
    mode: Mode, suite: PathBuf, out: Option<PathBuf>, only: Vec<String>, toolchain: Option<PathBuf>, harness_args: Vec<String>, harness: PathBuf,
}

pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let args: Vec<String> = args.into_iter().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") { return crate::diagnostics::print_stdout(HELP); }
    let result = parse_opts(&args).and_then(|o| match &o.mode {
        Mode::DryRun => dry_run(&o),
        Mode::Live { .. } => live(&o),
        Mode::Replay { .. } => replay::run(&o),
    });
    match result {
        Ok(code) => ExitCode::from(code),
        Err(e) => { crate::diagnostics::stderr_line(format_args!("vade: suite: {e}")); ExitCode::from(2) }
    }
}

fn parse_opts(a: &[String]) -> Result<Opts, String> {
    let (mut dry, mut live, mut endpoint, mut model, mut runs) = (false, false, None, None, 1);
    let mut replay_dir = None;
    let mut o = Opts { mode: Mode::DryRun, suite: PathBuf::new(), out: None, only: Vec::new(), toolchain: None, harness_args: Vec::new(), harness: PathBuf::new() };
    let mut i = 0;
    while i < a.len() {
        match a[i].as_str() {
            "--dry-run" => { dry = true; i += 1; continue; }
            "--live" => { live = true; i += 1; continue; }
            _ => {}
        }
        if !matches!(a[i].as_str(), "--suite" | "--out" | "--endpoint" | "--model" | "--runs" | "--only" | "--toolchain" | "--harness-arg" | "--replay") {
            return Err(format!("unknown option {}; see vade-harness suite --help", a[i]));
        }
        let v = a.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value; see vade-harness suite --help", a[i]))?;
        match a[i].as_str() {
            "--replay" => replay_dir = Some(PathBuf::from(&v)),
            "--suite" => o.suite = PathBuf::from(&v),
            "--out" => o.out = Some(PathBuf::from(&v)),
            "--endpoint" => endpoint = Some(v),
            "--model" => model = Some(v),
            "--runs" => runs = v.parse().ok().filter(|r| (1..=9).contains(r)).ok_or("--runs takes a number from 1 to 9")?,
            "--only" => o.only = v.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect(),
            "--toolchain" => o.toolchain = Some(PathBuf::from(&v)),
            "--harness-arg" => o.harness_args.push(v),
            other => return Err(format!("unknown option {other}; see vade-harness suite --help")),
        }
        i += 2;
    }
    o.mode = if let Some(dir) = replay_dir {
        if dry || live || endpoint.is_some() || model.is_some() || a.iter().any(|v| v == "--runs") || !o.harness_args.is_empty() {
            return Err("--replay takes no --dry-run, --live, --endpoint, --model, --runs or --harness-arg".into());
        }
        Mode::Replay { dir }
    } else { match (dry, live, endpoint, model) {
        (true, false, None, None) if o.harness_args.is_empty() => Mode::DryRun,
        (true, false, ..) => return Err("--dry-run takes no --endpoint, --model or --harness-arg".into()),
        (false, true, Some(endpoint), Some(model)) => {
            if o.out.is_none() { return Err("--live needs --out DIR".into()); }
            upstream(&endpoint)?;
            Mode::Live { endpoint, model, runs }
        }
        (false, true, ..) => return Err("--live needs --endpoint URL and --model NAME".into()),
        _ => return Err("choose --dry-run, --live or --replay; see vade-harness suite --help".into()),
    } };
    if o.suite.as_os_str().is_empty() {
        o.suite = ["harness/tests/agent-suite", "tests/agent-suite"].iter().map(PathBuf::from).find(|p| p.join("tasks").is_dir())
            .ok_or("no tests/agent-suite under the current folder; give --suite DIR")?;
    }
    o.suite = std::fs::canonicalize(&o.suite).map_err(|e| format!("{}: {e}", o.suite.display()))?;
    o.toolchain = Some(match o.toolchain.take() { Some(t) => std::fs::canonicalize(&t).map_err(|e| format!("{}: {e}", t.display()))?, None => find_toolchain()? });
    o.harness = std::env::current_exe().map_err(|e| format!("cannot find this harness binary: {e}"))?;
    Ok(o)
}

/// The Rust distribution that runs this build or the `rustc` on PATH. Checks
/// and the command sandbox use its real binaries, not rustup shims.
fn find_toolchain() -> Result<PathBuf, String> {
    let real = |p: PathBuf| p.join("bin/cargo").is_file() && p.join("bin/rustc").is_file() && p.join("lib").is_dir();
    if let Some(dir) = std::env::var_os("CARGO").map(PathBuf::from).and_then(|c| Some(c.parent()?.parent()?.to_path_buf())) {
        if real(dir.clone()) { return std::fs::canonicalize(dir).map_err(|e| e.to_string()); }
    }
    // rustup puts its rustc proxy in CARGO_HOME/bin, which is not always on PATH.
    let cargo_home = std::env::var_os("CARGO_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")));
    for rustc in std::iter::once(PathBuf::from("rustc")).chain(cargo_home.map(|h| h.join("bin/rustc"))) {
        let Ok(out) = Command::new(&rustc).args(["--print", "sysroot"]).output() else { continue };
        let dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        if out.status.success() && real(dir.clone()) { return std::fs::canonicalize(dir).map_err(|e| e.to_string()); }
    }
    Err("no Rust toolchain found; give --toolchain DIR".into())
}

pub(crate) fn s(x: &str) -> Value { Value::String(x.into()) }
pub(crate) fn n<T: ToString>(x: T) -> Value { Value::Number(x.to_string()) }
pub(crate) fn obj(fields: Vec<(&str, Value)>) -> Value { Value::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect()) }
pub(crate) fn write(v: &Value) -> String { json::stringify(v).unwrap_or_default() }
pub(crate) fn u64_of(v: &Value) -> Option<u64> { if let Value::Number(x) = v { x.parse().ok() } else { None } }

/// JSON equality with free member order; numbers compare by value.
pub(crate) fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| b.get(k).is_some_and(|w| same(v, w))),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same(v, w)),
        (Value::Number(x), Value::Number(y)) => x == y || matches!((x.parse::<f64>(), y.parse::<f64>()), (Ok(p), Ok(q)) if p == q),
        _ => a == b,
    }
}

fn pretty(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent + 1);
    match v {
        Value::Object(m) if !m.is_empty() => format!("{{\n{}\n{}}}", m.iter().map(|(k, x)| format!("{pad}{}: {}", json::quote(k), pretty(x, indent + 1))).collect::<Vec<_>>().join(",\n"), "  ".repeat(indent)),
        Value::Array(a) if !a.is_empty() => format!("[\n{}\n{}]", a.iter().map(|x| format!("{pad}{}", pretty(x, indent + 1))).collect::<Vec<_>>().join(",\n"), "  ".repeat(indent)),
        other => write(other),
    }
}

/// FNV-1a, 64 bits, over little-endian `u64` words.
fn fnv(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in words { for b in w.to_le_bytes() { h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3); } }
    h
}

fn upstream(url: &str) -> Result<String, String> {
    let rest = url.strip_prefix("http://").ok_or("the endpoint must start with http://")?;
    let host = rest.split('/').next().unwrap_or("");
    if host.is_empty() { return Err("the endpoint has no host".into()); }
    // A port follows the last colon, after the closing bracket of an IPv6 address.
    let has_port = host.rsplit_once(':').is_some_and(|(h, p)| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) && (!h.starts_with('[') || h.ends_with(']')));
    Ok(if has_port { host.to_string() } else { format!("{host}:80") })
}

fn date() -> String {
    Command::new("/bin/date").arg("+%Y-%m-%d %H:%M:%S %Z").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn new_dir(p: &Path) -> Result<PathBuf, String> {
    if p.exists() { return Err(format!("{} exists; use a new output folder for each run", p.display())); }
    std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
    std::fs::canonicalize(p).map_err(|e| e.to_string())
}

/// A dry run's own folder in `tmp`: always a new one, removed when the run
/// ends, on a pass, a fail or an error. A folder that is already there, such
/// as one left by a stopped run with the same process ID, is not used and not
/// touched: the next name is tried.
struct TempOut(PathBuf);

impl TempOut {
    fn new(tmp: &Path) -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        for _ in 0..1000 {
            let p = tmp.join(format!("vade-suite-dry-run-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            match std::fs::create_dir(&p) {
                Ok(()) => return std::fs::canonicalize(&p).map(TempOut).map_err(|e| { let _ = std::fs::remove_dir(&p); format!("{}: {e}", p.display()) }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(format!("{}: {e}", p.display())),
            }
        }
        Err(format!("{}: 1000 vade-suite-dry-run folders of this process ID exist; use --out DIR", tmp.display()))
    }
}

impl Drop for TempOut { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let (src, dst) = (e.path(), to.join(e.file_name()));
        let kind = e.file_type()?;
        // A link is copied as a link (the agent may make one to a folder, or a broken one).
        if kind.is_symlink() { copy_link(&src, &dst)?; } else if kind.is_dir() { copy_tree(&src, &dst)?; } else { std::fs::copy(&src, &dst)?; }
    }
    Ok(())
}

#[cfg(unix)]
fn copy_link(src: &Path, dst: &Path) -> std::io::Result<()> { std::os::unix::fs::symlink(std::fs::read_link(src)?, dst) }

#[cfg(not(unix))]
fn copy_link(src: &Path, dst: &Path) -> std::io::Result<()> { std::fs::write(dst, format!("symlink -> {}", std::fs::read_link(src)?.display())) }

/// The Git-tracked files under `dir`, relative to it (or with `others`, the
/// untracked and ignored ones). A run uses only tracked files, so a card proves
/// the committed fixture, not files that only sit on disk.
fn tracked(dir: &Path, others: bool) -> Result<Vec<PathBuf>, String> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(dir).args(["ls-files", "-z"]);
    if others { cmd.arg("--others"); }
    let out = cmd.output().map_err(|e| format!("git: {e}"))?;
    if !out.status.success() { return Err(format!("{} is not in a Git checkout", dir.display())); }
    Ok(String::from_utf8_lossy(&out.stdout).split('\0').filter(|s| !s.is_empty() && !s.ends_with(".DS_Store")).map(PathBuf::from).collect())
}

fn copy_tracked(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for rel in tracked(from, false)? {
        let dst = to.join(&rel);
        if let Some(p) = dst.parent() { std::fs::create_dir_all(p).map_err(|e| e.to_string())?; }
        std::fs::copy(from.join(&rel), &dst).map_err(|e| format!("{}: {e}", rel.display()))?;
    }
    Ok(())
}

/// The suite's tasks in name order, parsed, with the checks of each known.
fn load_tasks(o: &Opts) -> Result<Vec<(PathBuf, task::Task)>, String> {
    let tasks_dir = o.suite.join("tasks");
    let mut ids: Vec<String> = std::fs::read_dir(&tasks_dir).map_err(|e| format!("{}: {e}", tasks_dir.display()))?
        .flatten().filter(|e| e.path().join("task.txt").is_file()).map(|e| e.file_name().to_string_lossy().into()).collect();
    ids.sort();
    for id in &o.only { if !ids.contains(id) { return Err(format!("no task {id}")); } }
    if !o.only.is_empty() { ids.retain(|id| o.only.contains(id)); }
    // Untracked or ignored files would make a card that a fresh checkout cannot repeat.
    let stray = tracked(&o.suite, true)?;
    if !stray.is_empty() { return Err(format!("untracked or ignored files in the suite (git add or remove them): {stray:?}")); }
    let mut out = Vec::new();
    for id in ids {
        let dir = tasks_dir.join(&id);
        let text = std::fs::read_to_string(dir.join("task.txt")).map_err(|e| format!("{id}/task.txt: {e}"))?;
        let mut t = task::parse(&text).map_err(|e| format!("{id}/task.txt: {e}"))?;
        let rubric = std::fs::read_to_string(dir.join("rubric.txt")).map_err(|e| format!("{id}/rubric.txt: {e}"))?;
        t.rubric = rubric::parse(&rubric).map_err(|e| format!("{id}/rubric.txt: {e}"))?;
        if t.id != id { return Err(format!("{id}/task.txt says id {}", t.id)); }
        let all = t.checks.iter().chain(t.steps.iter().flat_map(|s| &s.checks));
        if let Some(bad) = all.into_iter().find(|c| !check::known(c)) { return Err(format!("{id}/task.txt: unknown check {bad}")); }
        out.push((dir, t));
    }
    Ok(out)
}

fn suite_version(o: &Opts) -> String { std::fs::read_to_string(o.suite.join("VERSION")).unwrap_or_default().trim().to_string() }

/// The dry-run gate: every task passes with its solve.script and fails with
/// its wrong.script, and the solve replay uses every scripted reply.
fn dry_run(o: &Opts) -> Result<u8, String> {
    let tasks = load_tasks(o)?;
    // `_temp` lives to the end of this function, so its folder goes on every return.
    let (out, _temp) = match &o.out {
        Some(p) => (new_dir(p)?, None),
        None => { let t = TempOut::new(&std::env::temp_dir())?; (t.0.clone(), Some(t)) }
    };
    let version = suite_version(o);
    let (mut ok, mut skipped) = (0, 0);
    for (dir, t) in &tasks {
        let base = out.join(&t.id);
        // Without the command sandbox, workspace_shell runs nothing, so a script that needs it cannot pass here.
        let shell = |name: &str| std::fs::read_to_string(dir.join(format!("{name}.script"))).unwrap_or_default().lines().any(|l| l.trim() == ">>> call workspace_shell");
        if !crate::coding_tools::SHELL_AVAILABLE && (shell("solve") || shell("wrong")) {
            skipped += 1;
            println!("{version} dry run | {} | {} | SKIP: the scripts call workspace_shell, and this host has no command sandbox", t.id, t.category);
            continue;
        }
        let run = |name: &str| -> Result<TaskResult, String> {
            let replies = task::parse_script(&std::fs::read_to_string(dir.join(format!("{name}.script"))).map_err(|e| format!("{name}.script: {e}"))?)
                .map_err(|e| format!("{name}.script: {e}"))?;
            Ok(run_task(o, dir, t.clone(), &base.join(name), Source::Replay(Mutex::new(replies.into())), None))
        };
        let (solve, wrong) = (run("solve"), run("wrong"));
        let mut problems = Vec::new();
        let solve_text = match &solve {
            Ok(r) => {
                if r.status != "PASS" { problems.push(format!("solve.script must pass: {}", r.note)); }
                if r.replies_left.unwrap_or(0) > 0 { problems.push(format!("solve.script has {} replies the harness never asked for", r.replies_left.unwrap_or(0))); }
                if r.steps.iter().any(|s| s.timed_out) { problems.push("a solve step timed out".into()); }
                if let Some(q) = r.quality.as_ref().filter(|q| q.score < 3) { problems.push(format!("solve.script must score 3/3 on rubric.txt, got {}: {}", q.score, q.misses.join("; "))); }
                format!("solve {} {} quality {}/3 ({} changed lines)", r.status, r.score(), r.quality.as_ref().map_or(0, |q| q.score), r.quality.as_ref().map_or(0, |q| q.changed_lines))
            }
            Err(e) => { problems.push(e.clone()); "solve ERROR".into() }
        };
        let wrong_text = match &wrong {
            Ok(r) => {
                if r.status != "FAIL" { problems.push(format!("wrong.script must fail a check, got {}: {}", r.status, r.note)); }
                let first = r.checks.iter().find(|c| !c.pass).map(|c| format!(" ({} -> {})", c.check, c.detail)).unwrap_or_default();
                format!("wrong {} {} quality {}/3{first}", r.status, r.score(), r.quality.as_ref().map_or(0, |q| q.score))
            }
            Err(e) => { problems.push(e.clone()); "wrong ERROR".into() }
        };
        let verdict = if problems.is_empty() { ok += 1; "OK".to_string() } else { format!("GATE FAIL: {}", problems.join("; ")) };
        println!("{version} dry run | {} | {} | {solve_text} | {wrong_text} | {verdict}", t.id, t.category);
    }
    let pass = ok > 0 && ok + skipped == tasks.len();
    println!("DRY RUN {}: {ok} of {} tasks pass with solve.script and fail with wrong.script, {skipped} skipped ({version})", if pass { "PASS" } else { "FAIL" }, tasks.len());
    if o.out.is_some() { println!("Results: {}", out.display()); } else if !pass { println!("The temporary results folder is removed; use --out DIR to keep the results."); }
    Ok(if pass { 0 } else { 1 })
}

struct StepResult { kind: String, exit: Option<i32>, seconds: f64, native: Vec<String>, timed_out: bool }

struct TaskResult {
    task: task::Task, status: String, note: String, checks: Vec<check::Outcome>, steps: Vec<StepResult>,
    exchanges: Vec<serve::Exchange>, seconds: f64, replies_left: Option<usize>, quality: Option<rubric::Quality>,
}

impl TaskResult {
    fn score(&self) -> String { format!("{}/{}", self.checks.iter().filter(|c| c.pass).count(), self.checks.len()) }
    /// One digest per answered model call over its streamed token IDs.
    fn call_digests(&self) -> Vec<u64> { self.exchanges.iter().filter(|x| !x.held).map(|x| fnv(x.token_ids.iter().copied())).collect() }
    /// One request digest per answered model call, in the order of `call_digests`.
    fn call_requests(&self) -> Vec<u64> { self.exchanges.iter().filter(|x| !x.held).map(|x| x.request).collect() }
    fn token_count(&self) -> usize { self.exchanges.iter().map(|x| x.token_ids.len()).sum() }

    /// The first tool call that the harness refused or that failed, or that
    /// writes a path outside the rubric's may_change. Calls are numbered over
    /// the whole task; a result is in the next request of the same step.
    fn first_wrong_call(&self) -> String {
        let ex: Vec<&serve::Exchange> = self.exchanges.iter().collect();
        let (mut number, mut last) = (0, None);
        for (i, x) in ex.iter().enumerate().filter(|(_, x)| !x.held) {
            let next = ex.get(i + 1).filter(|y| y.step == x.step);
            for (k, (name, args)) in x.calls.iter().enumerate() {
                number += 1;
                let result = next.and_then(|y| y.results.get(k)).map(String::as_str).unwrap_or("");
                let head = result.lines().next().unwrap_or("").chars().take(140).collect::<String>();
                let what = format!("call {number} {name} {}", args.chars().take(100).collect::<String>().replace('\n', " "));
                let low = head.to_ascii_lowercase();
                if ["refused", "denied", "invalid arguments", "unknown tool", "unsupported"].iter().any(|w| low.contains(w)) || low.starts_with("error") {
                    return format!("{what} -> {head}");
                }
                if let Some(p) = written_paths(name, args).into_iter().find(|p| !self.task.rubric.allows(p)) {
                    return format!("{what} -> writes {p}, outside the rubric's may_change");
                }
                last = Some(what);
            }
        }
        last.map_or("no tool call".into(), |w| format!("none refused or outside the rubric; last tool call: {w}"))
    }
}

/// The workspace paths that a writing tool call names.
fn written_paths(name: &str, args: &str) -> Vec<String> {
    let v = json::parse(args).unwrap_or(Value::Null);
    match name {
        "apply_patch" => v.get("patch").and_then(Value::as_str).unwrap_or("").lines()
            .filter_map(|l| ["*** Add File: ", "*** Update File: ", "*** Delete File: ", "*** Move to: "].iter().find_map(|p| l.strip_prefix(p)))
            .map(|p| p.trim().to_string()).collect(),
        "workspace_edit_lines" | "workspace_write" => v.get("path").and_then(Value::as_str).map(|p| vec![p.to_string()]).unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Removes a folder when it goes out of scope, also on an early return.
pub(crate) struct RemoveOnDrop(pub PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

/// The note for a task whose call list may lack a call: a model call handler
/// still ran when the runner stopped waiting for it (`serve::Server::exchanges`).
const INCOMPLETE: &str = "; a model call was still running after 5 s, so the call list may lack a call";

/// Runs one task into `out` (a new folder). `work` gives a fixed workspace
/// path for live runs, so that every run sends the same prompts.
fn run_task(o: &Opts, dir: &Path, t: task::Task, out: &Path, source: Source, fixed: Option<&Path>) -> TaskResult {
    let start = Instant::now();
    let mut r = TaskResult { task: t.clone(), status: "ERROR".into(), note: String::new(), checks: Vec::new(), steps: Vec::new(), exchanges: Vec::new(), seconds: 0.0, replies_left: None, quality: None };
    let fail = |r: &mut TaskResult, e: String| { r.note = e; r.seconds = start.elapsed().as_secs_f64(); };
    let first = out.join("start");
    let (work, home) = match fixed { Some(f) => (f.join("work"), f.join("home")), None => (out.join("work"), out.join("home")) };
    // The starting workspace: the fixture folder, plus generated files.
    let prep = (|| -> Result<(), String> {
        if let Some(f) = fixed { let _ = std::fs::remove_dir_all(f.join("work")); let _ = std::fs::remove_dir_all(f.join("home")); }
        std::fs::create_dir_all(&home).map_err(|e| e.to_string())?;
        if dir.join("workspace").is_dir() { copy_tracked(&dir.join("workspace"), &first)?; } else { std::fs::create_dir_all(&first).map_err(|e| e.to_string())?; }
        match t.generate.as_deref() {
            None => {}
            Some("repo-digest-v1") => {
                std::fs::write(first.join("DIGEST.md"), digest::generate()).map_err(|e| e.to_string())?;
                let want = std::fs::read_to_string(dir.join("expected/ANSWERS.txt")).map_err(|e| e.to_string())?;
                if check::normal(&want) != check::normal(&digest::answers()) { return Err("expected/ANSWERS.txt does not match the digest generator".into()); }
            }
            Some(g) => return Err(format!("unknown generator {g}")),
        }
        for input in &t.inputs { if !first.join(input).is_file() { return Err(format!("input {input} is missing from the starting workspace")); } }
        copy_tree(&first, &work).map_err(|e| e.to_string())
    })();
    if let Err(e) = prep { fail(&mut r, e); return r; }
    let record = std::fs::File::create(out.join("transcript.jsonl")).ok();
    let sse = matches!(o.mode, Mode::DryRun);
    let server = match Server::start(source, record, sse, &work) { Ok(s) => s, Err(e) => { fail(&mut r, e.to_string()); return r; } };
    // Check builds are scratch: outside the result folder, removed after the checks.
    let scratch = std::env::temp_dir().join(format!("vade-suite-check-{}-{}", std::process::id(), t.id));
    let _scratch = RemoveOnDrop(scratch.clone());
    let mut env: Vec<(String, String)> = vec![("NO_COLOR".into(), "1".into()), ("TERM".into(), "dumb".into()), ("CARGO_TARGET_DIR".into(), scratch.display().to_string())];
    if let Some(tc) = &o.toolchain {
        let path = std::env::var("PATH").unwrap_or_default();
        env.push(("PATH".into(), format!("{}:{path}", tc.join("bin").display())));
    }
    let model = match &o.mode { Mode::Live { model, .. } => model.as_str(), Mode::DryRun | Mode::Replay { .. } => "replay" };
    let mut last_stdout = String::new();
    let mut unsupported = None;
    let checks_log = out.join("checks.log");
    if t.mode == "direct" {
        server.begin_step(1, None);
        match direct(&server.url, model, &dir.join(t.request.as_deref().unwrap_or_default()), &work) {
            Ok((status, body)) => {
                if matches!(status, 400 | 404 | 422 | 501) { unsupported = Some(format!("HTTP {status}: {}", body.chars().take(200).collect::<String>())); }
                last_stdout = body;
            }
            Err(e) => { fail(&mut r, format!("direct request: {e}")); return r; }
        }
    } else {
        for (i, s) in t.steps.iter().enumerate() {
            for p in &s.remove_before {
                if let Err(e) = std::fs::remove_file(work.join(p)) { fail(&mut r, format!("remove_before {p}: {e}")); return r; }
            }
            let res = run_step(o, model, &server, i + 1, s, &t, &work, &home, out);
            last_stdout = std::fs::read_to_string(out.join(format!("step-{}.stdout", i + 1))).unwrap_or_default();
            match res { Ok(sr) => r.steps.push(sr), Err(e) => { fail(&mut r, e); r.exchanges = server.exchanges(); if server.wait_expired() { r.note.push_str(INCOMPLETE); } return r; } }
            // Checks of this step see the workspace as the step left it.
            let held_closed = server.exchanges().iter().any(|x| x.held && x.client_closed);
            let ctx = check::Ctx { work: &work, start: &first, task_dir: dir, last_exit: r.steps.last().and_then(|s| s.exit), last_stdout: &last_stdout, held_closed, env: &env, log: &checks_log };
            for c in &s.checks {
                let mut outcome = check::run(c, &ctx);
                outcome.check = format!("after step {}: {c}", i + 1);
                r.checks.push(outcome);
            }
        }
    }
    r.exchanges = server.exchanges();
    r.replies_left = server.remaining_replies();
    let incomplete = server.wait_expired();
    drop(server);
    let held_closed = r.exchanges.iter().any(|x| x.held && x.client_closed);
    let ctx = check::Ctx { work: &work, start: &first, task_dir: dir, last_exit: r.steps.last().and_then(|s| s.exit), last_stdout: &last_stdout, held_closed, env: &env, log: &checks_log };
    r.checks.extend(t.checks.iter().map(|c| check::run(c, &ctx)));
    if fixed.is_some() {
        // Keep the final workspace with the results; the fixed folder serves the next task.
        if let Err(e) = copy_tree(&work, &out.join("work")) { r.note = format!("cannot keep the workspace: {e}"); }
    }
    let passed = r.checks.iter().filter(|c| c.pass).count();
    r.status = if let Some(u) = unsupported { r.note = u; "UNSUPPORTED".into() } else if passed == r.checks.len() { "PASS".into() } else { "FAIL".into() };
    // A direct task has no harness reply on stdout: its answer is the reply content.
    let reply = if t.mode == "direct" { std::fs::read_to_string(work.join("reply.json")).unwrap_or_default() } else { last_stdout.clone() };
    let removed: Vec<String> = t.steps.iter().flat_map(|s| s.remove_before.iter().cloned()).collect();
    r.quality = Some(t.rubric.score(r.status == "PASS", &first, &work, &removed, &reply));
    if r.note.is_empty() {
        r.note = format!("{passed}/{} checks", r.checks.len());
        if let Some(c) = r.checks.iter().find(|c| !c.pass) { r.note.push_str(&format!("; first failure: {} -> {}", c.check, c.detail)); }
    }
    if incomplete { r.note.push_str(INCOMPLETE); }
    r.seconds = start.elapsed().as_secs_f64();
    r
}

#[allow(clippy::too_many_arguments)]
fn run_step(o: &Opts, model: &str, server: &Server, n: usize, s: &task::Step, t: &task::Task, work: &Path, home: &Path, out: &Path) -> Result<StepResult, String> {
    let (tx, rx) = mpsc::channel();
    server.begin_step(n, s.cancel_at_call.map(|at| (at, tx)));
    let mut cmd = Command::new(&o.harness);
    cmd.args(["agent", "--endpoint", &server.url, "--model", model, "--workspace"]).arg(work).arg("--vade-home").arg(home)
        .args(["--verbose", "--approval", "auto", "--agent-time-limit-seconds", &t.budget_seconds.to_string()])
        // A fixed reply cap: the default scales with the measured decode speed, which changes the requests from run to run.
        .args(["--reply-tokens", &crate::contract::NATIVE_REPLY_MAX_TOKENS.to_string()]);
    // The command sandbox exists only on macOS; elsewhere workspace_shell reports an unsupported platform.
    if let Some(tc) = o.toolchain.as_ref().filter(|_| crate::coding_tools::SHELL_AVAILABLE) { cmd.arg("--shell-toolchain").arg(tc); }
    cmd.args(&o.harness_args);
    match s.kind.as_str() {
        "new" => { cmd.args(["--session", "suite", "--prompt", &s.prompt]); }
        "resume" => { cmd.args(["--resume", "suite", "--prompt", &s.prompt]); }
        _ => { cmd.args(["--session", "suite"]); }
    }
    let stdout = std::fs::File::create(out.join(format!("step-{n}.stdout"))).map_err(|e| e.to_string())?;
    // HOME is the task's home folder, so the user's own ~/.vade/instructions.md is not in the prompt.
    cmd.current_dir(work).env("HOME", home).env("NO_COLOR", "1").env("TERM", "dumb").stdout(stdout).stderr(Stdio::piped())
        .stdin(if s.kind == "cancel" { Stdio::piped() } else { Stdio::null() });
    let begun = Instant::now();
    let mut child = cmd.spawn().map_err(|e| format!("{}: {e}", o.harness.display()))?;
    let err_lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let stderr = child.stderr.take().ok_or("no harness stderr")?;
    let (lines, err_path) = (err_lines.clone(), out.join(format!("step-{n}.stderr")));
    let reader = std::thread::spawn(move || {
        let mut f = std::fs::File::create(err_path).ok();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Some(f) = f.as_mut() { let _ = writeln!(f, "{line}"); }
            if line.contains("Native run:") { if let Ok(mut l) = lines.lock() { l.push(line); } }
        }
    });
    let limit = Duration::from_secs(t.budget_seconds + 60);
    let saw_run = |l: &Arc<Mutex<Vec<String>>>| l.lock().map(|l| !l.is_empty()).unwrap_or(false);
    if let Some(mut stdin) = child.stdin.take() {
        let _ = writeln!(stdin, "{}", s.prompt.replace('\n', " "));
        let _ = stdin.flush();
        // Wait for the held call, or for the run to end without reaching it.
        let mut held = false;
        while begun.elapsed() < limit && !saw_run(&err_lines) && child.try_wait().ok().flatten().is_none() {
            if rx.recv_timeout(Duration::from_millis(100)).is_ok() { held = true; break; }
        }
        if held {
            let _ = writeln!(stdin, "/cancel");
            let _ = stdin.flush();
            while begun.elapsed() < limit && !saw_run(&err_lines) && child.try_wait().ok().flatten().is_none() { std::thread::sleep(Duration::from_millis(50)); }
        }
        drop(stdin);
    }
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {}
            Err(e) => { let _ = child.kill(); let _ = child.wait(); let _ = reader.join(); return Err(e.to_string()); }
        }
        if begun.elapsed() > limit {
            timed_out = true;
            let _ = child.kill();
            break child.wait().ok();
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let _ = reader.join();
    let native = err_lines.lock().map(|l| l.clone()).unwrap_or_default();
    Ok(StepResult { kind: s.kind.clone(), exit: status.and_then(|s| s.code()), seconds: begun.elapsed().as_secs_f64(), native, timed_out })
}

/// One streamed chat request through the task's endpoint, so that the proxy
/// records its token IDs. Writes the reply content to `work/reply.json` and
/// returns the HTTP status and the body (SSE, or JSON for an error).
fn direct(url: &str, model: &str, request: &Path, work: &Path) -> Result<(u16, String), String> {
    let mut body = json::parse(&std::fs::read_to_string(request).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if let Value::Object(m) = &mut body {
        m.retain(|(k, _)| k != "model" && k != "stream");
        m.insert(0, ("model".into(), s(model)));
        m.push(("stream".into(), Value::Bool(true)));
    }
    let payload = write(&body);
    let addr = upstream(url)?;
    let path = url.find(&addr).map(|i| &url[i + addr.len()..]).unwrap_or("/");
    let mut stream = std::net::TcpStream::connect(&addr).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(900))).map_err(|e| e.to_string())?;
    write!(stream, "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{payload}", payload.len()).map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    stream.take(serve::MAX_BODY as u64).read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let end = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("reply has no header end")?;
    let head = String::from_utf8_lossy(&raw[..end]).to_string();
    let status: u16 = head.split_whitespace().nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
    let rest = &raw[end + 4..];
    let body = String::from_utf8_lossy(&if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(rest) } else { rest.to_vec() }).to_string();
    let first = |v: &Value, key: &str| v.get("choices")?.as_array()?.first()?.get(key)?.get("content")?.as_str().map(str::to_string);
    let content = match json::parse(body.trim()) {
        Ok(v) => first(&v, "message").unwrap_or_default(),
        Err(_) => body.lines().filter_map(|l| json::parse(l.strip_prefix("data:")?.trim()).ok()).filter_map(|v| first(&v, "delta")).collect(),
    };
    std::fs::write(work.join("reply.json"), &content).map_err(|e| e.to_string())?;
    Ok((status, body))
}

/// Joins the chunks of a chunked body. Sizes count bytes, so a chunk can end
/// inside a UTF-8 character.
fn dechunk(mut rest: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(at) = rest.windows(2).position(|w| w == b"\r\n") {
        let size = String::from_utf8_lossy(&rest[..at]).to_string();
        let len = usize::from_str_radix(size.split(';').next().unwrap_or("").trim(), 16).unwrap_or(0);
        let tail = &rest[at + 2..];
        if len == 0 || tail.len() < len { break; }
        out.extend_from_slice(&tail[..len]);
        rest = tail[len..].strip_prefix(b"\r\n").unwrap_or(&tail[len..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_choose_one_mode() {
        let a = |x: &[&str]| parse_opts(&x.iter().map(|s| s.to_string()).collect::<Vec<_>>()).err().unwrap_or_default();
        assert!(a(&[]).contains("choose --dry-run, --live or --replay"));
        assert!(a(&["--dry-run", "--live"]).contains("choose"));
        assert!(a(&["--live", "--endpoint", "http://127.0.0.1:1/v1/chat/completions"]).contains("--model"));
        assert!(a(&["--live", "--endpoint", "http://h/v1", "--model", "m"]).contains("--out"));
        assert!(a(&["--live", "--endpoint", "https://h/v1", "--model", "m", "--out", "x"]).contains("http://"));
        assert!(a(&["--dry-run", "--model", "m"]).contains("takes no"));
        assert!(a(&["--dry-run", "--runs", "0"]).contains("1 to 9"));
        assert!(a(&["--dry-run", "--bogus", "1"]).contains("unknown option"));
        assert!(a(&["--dry-run", "--bogus"]).contains("unknown option --bogus"));
        assert!(a(&["--replay"]).contains("needs a value"));
        for extra in [&["--dry-run"][..], &["--live"], &["--endpoint", "unused"], &["--model", "unused"], &["--runs", "1"], &["--harness-arg", "unused"]] {
            let mut args = vec!["--replay", "records"];
            args.extend_from_slice(extra);
            assert!(a(&args).contains("--replay takes no"));
        }
        assert_eq!(upstream("http://[::1]/v1").unwrap(), "[::1]:80");
        assert_eq!(upstream("http://[::1]:8080/v1").unwrap(), "[::1]:8080");
        assert_eq!(upstream("http://h:18080/v1").unwrap(), "h:18080");
        assert_eq!(upstream("http://h/v1").unwrap(), "h:80");
    }
    #[cfg(unix)]
    #[test]
    fn copy_tree_keeps_links_as_links() {
        let dir = std::env::temp_dir().join(format!("vade-suite-copy-test-{}", std::process::id()));
        // A failed earlier run with the same process ID can leave the folder.
        let _ = std::fs::remove_dir_all(&dir);
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::write(a.join("f.txt"), "x").unwrap();
        std::os::unix::fs::symlink(".", a.join("self")).unwrap();
        std::os::unix::fs::symlink("gone.txt", a.join("broken")).unwrap();
        copy_tree(&a, &b).unwrap();
        assert_eq!(std::fs::read_link(b.join("self")).unwrap(), Path::new("."));
        assert_eq!(std::fs::read_link(b.join("broken")).unwrap(), Path::new("gone.txt"));
        assert_eq!(std::fs::read_to_string(b.join("f.txt")).unwrap(), "x");
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn a_dry_run_folder_is_new_and_goes_at_the_end() {
        let tmp = std::env::temp_dir().join(format!("vade-suite-tempout-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Folders with the first names, as a stopped run with the same process ID leaves them.
        let next = |k: u64| tmp.join(format!("vade-suite-dry-run-{}-{k}", std::process::id()));
        let mut stale = Vec::new();
        for k in 0..3 { std::fs::create_dir(next(k)).unwrap(); std::fs::write(next(k).join("keep.txt"), "x").unwrap(); stale.push(next(k)); }
        let (a, b) = (TempOut::new(&tmp).unwrap(), TempOut::new(&tmp).unwrap());
        assert_ne!(a.0, b.0);
        for t in [&a, &b] { assert!(t.0.is_dir() && std::fs::read_dir(&t.0).unwrap().next().is_none(), "{}", t.0.display()); }
        std::fs::write(a.0.join("x.txt"), "x").unwrap();
        let (pa, pb) = (a.0.clone(), b.0.clone());
        drop((a, b));
        assert!(!pa.exists() && !pb.exists());
        for s in &stale { assert_eq!(std::fs::read_to_string(s.join("keep.txt")).unwrap(), "x", "a stale folder was touched"); }
        assert!(TempOut::new(&tmp.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
    #[test]
    fn chunks_join_as_bytes() {
        // "é" is two bytes; the first chunk ends between them.
        assert_eq!(String::from_utf8(dechunk(b"2\r\nab\r\n1\r\n\xc3\r\n2;x=1\r\n\xa9!\r\n0\r\n\r\n")).unwrap(), "ab\u{e9}!");
        assert!(dechunk(b"zz\r\nab").is_empty());
    }
    #[test]
    fn json_helpers() {
        let a = json::parse(r#"{"a":[1,2.0],"b":{"c":null}}"#).unwrap();
        let b = json::parse(r#"{"b":{"c":null},"a":[1.0,2]}"#).unwrap();
        assert!(same(&a, &b));
        assert!(!same(&a, &json::parse(r#"{"a":[2,1],"b":{"c":null}}"#).unwrap()));
        assert_eq!(json::parse(&pretty(&a, 0)).unwrap(), a);
        assert_eq!(fnv([]), 0xcbf2_9ce4_8422_2325);
        assert_ne!(fnv([1, 2]), fnv([2, 1]));
    }
}
