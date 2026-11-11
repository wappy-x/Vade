//! The "qualified" checklist for a full RigMark card. A card is qualified only when every check passes.
//! Host only: it reads the receipt files of one card folder. It starts no engine, needs no GPU and writes nothing
//! beside the card (one temporary copy of the golden file, removed after use).
//! Each check prints PASS or FAIL, what it found, and the receipts it read with their sha256.
//! It is a gate: a missing, empty, corrupt or short receipt, or a missing field, is a FAIL, never a PASS.
//!
//! The folder holds the RigMark card JSON (one file with every section, or section parts that share one
//! comparison id), a `.card.txt` beside each, `health.json`, `SHA256SUMS` of the served kit,
//! `live-binary-<box>.txt` (one a rank), `checkpoint-sha256-<box>.txt` (one a rank), the gate receipts
//! `goldens/*.jsonl`, the drafting part (draft "strict" with an MTP depth, drafter "on", same comparison id)
//! and `quality-vs-maker.result`.
use std::{collections::{BTreeMap, BTreeSet}, fs, io::Write, path::{Path, PathBuf}, process::{Command, Stdio}};
use crate::{Result, json::{self, Value}, sha256};

/// Inputs of one checklist run.
#[derive(Debug, Clone)]
pub struct Options {
    /// The card folder.
    pub dir: PathBuf,
    /// The repository that holds `main` and `tools/goldens/`.
    pub repo: PathBuf,
    /// The branch the engine commit must be on. It is resolved to one commit, which the digest binds.
    pub main_ref: String,
    /// The golden manifest. Default: `tools/goldens/<model>.tsv` at the resolved `main_ref` commit, read from git.
    pub goldens: Option<PathBuf>,
    /// The `check-goldens` program.
    pub goldens_bin: PathBuf,
}

/// One line of the checklist.
#[derive(Debug, Clone, PartialEq)]
pub struct Check { pub id: &'static str, pub name: &'static str, pub pass: bool, pub detail: String, pub receipts: Vec<String> }

/// The whole checklist. `digest` is the sha256 of every receipt name and hash the checks read, the golden file
/// and the resolved main commit.
#[derive(Debug, Clone, PartialEq)]
pub struct Report { pub checks: Vec<Check>, pub info: Vec<String>, pub digest: String, pub engine: String }

/// The checks, in order. A card is qualified only with all of them.
pub const CHECKS: [(&str, &str); 12] = [
    ("Q01", "card parts: one engine, one model, one comparison id, clean RigMark tool"),
    ("Q02", "engine commit is on main"),
    ("Q03", "build identity: health, kit SHA256SUMS and the live binary on every rank agree"),
    ("Q04", "weights identity: full checkpoint sha256 on every rank, equal"),
    ("Q05", "prose, code and structured rows: 5 natural stops each, all output gates"),
    ("Q06", "prefill 8K: 3 cold reads, no cached tokens"),
    ("Q07", "concurrency 1, 2, 4 and 8: 3 rounds each, every stream ended"),
    ("Q08", "request body: plain, cache off, one body and seed for every part"),
    ("Q09", "drafting row: same prompts, 5 runs a workload, answers equal to plain"),
    ("Q10", "exact token IDs: gate receipts equal the golden file"),
    ("Q11", "quality against the maker: a PASS receipt for this engine commit"),
    ("Q12", "no fixture rates relabeled: two-Spark served rates that recompute from their own runs"),
];
const WORKLOADS: [&str; 3] = ["prose", "code", "structured"];
const FIXTURE_WORDS: [&str; 8] = ["fixture", "loopback", "resident-decode-check", "census", "servebench", "estimate", "projected", "one-box"];
/// Keys of the request body that differ between the plain and the drafting part.
const DRAFT_KEYS: [&str; 2] = ["draft", "mtp_depth"];
/// Settings that must be equal on the plain and the drafting part.
const SAME_SETTINGS: [&str; 5] = ["temperature", "seed", "top_p", "decode_tokens", "runs"];

#[derive(Clone, Copy, PartialEq)]
enum Mode { Plain, Draft, Mixed }
struct Part { name: String, value: Value, text: Option<String>, mode: Mode }
#[derive(Default)]
struct Read { files: BTreeMap<String, String> }
impl Read {
    fn bytes(&mut self, dir: &Path, name: &str) -> Option<Vec<u8>> {
        let bytes = fs::read(dir.join(name)).ok()?;
        self.files.insert(name.into(), sha256::digest(&bytes));
        Some(bytes)
    }
    fn text(&mut self, dir: &Path, name: &str) -> Option<String> { String::from_utf8(self.bytes(dir, name)?).ok() }
    fn tag(&self, names: &[&str]) -> Vec<String> {
        let mut names = names.to_vec(); names.sort(); names.dedup();
        names.iter().filter_map(|n| self.files.get(*n).map(|h| format!("{n} sha256={}", short(h, 12)))).collect()
    }
}

/// At most `n` characters of `s`, cut on a character boundary. Never panics.
fn short(s: &str, n: usize) -> &str { s.char_indices().nth(n).map_or(s, |(i, _)| &s[..i]) }
fn at<'a>(mut value: &'a Value, keys: &[&str]) -> &'a Value { for key in keys { value = value.get(key); } value }
fn num(value: &Value) -> Option<f64> { if let Value::Number(s) = value { s.parse().ok().filter(|v: &f64| v.is_finite()) } else { None } }
fn int(value: &Value) -> Option<i64> { num(value).filter(|v| v.fract() == 0. && v.abs() < 9.0e15).map(|v| v as i64) }
fn text(value: &Value) -> Option<&str> { if let Value::String(s) = value { Some(s) } else { None } }
fn runs(value: &Value) -> &[Value] { if let Value::Array(a) = value.get("runs") { a } else { &[] } }
fn hex64(value: &Value) -> bool { text(value).is_some_and(|h| crate::is_lower_hex(h, 64)) }
fn commit(s: &str) -> Option<String> {
    s.split(|c: char| !c.is_ascii_hexdigit()).find(|w| crate::is_lower_hex(w, 40)).map(str::to_string)
}
fn git(repo: &Path) -> Command { let mut c = Command::new("git"); c.arg("-C").arg(repo); c }
/// The median as the RigMark tool writes it: Python's statistics.median, rounded to 3 decimals.
fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) { return None; }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let m = if n % 2 == 1 { values[n / 2] } else { (values[n / 2 - 1] + values[n / 2]) / 2. };
    Some((m * 1000.).round() / 1000.)
}
/// The published median equals the median of its own runs (within rounding).
fn recomputes(summary: &Value, rows: &[Value], key: &str) -> std::result::Result<f64, String> {
    let shown = num(summary.get("median")).ok_or("no median")?;
    if rows.iter().any(|r| num(r.get(key)).is_none()) { return Err(format!("a run has no {key}")); }
    let mine = median(rows.iter().filter_map(|r| num(r.get(key))).collect()).ok_or("no runs")?;
    if (shown - mine).abs() > 0.0011 { return Err(format!("median {shown} does not recompute from its runs ({mine})")); }
    if shown <= 0. { return Err(format!("median {shown} is not positive")); }
    Ok(shown)
}
/// Every summary with a median in `row` recomputes from `samples`. A summary whose key no sample carries is derived
/// from deeper rows (per-stream rates) and is not checked here; a summary with no samples at all is a FAIL.
fn all_medians(row: &Value, samples: &[Value]) -> std::result::Result<(), String> {
    let Value::Object(fields) = row else { return Ok(()) };
    for (key, summary) in fields {
        if matches!(summary.get("median"), Value::Null) { continue; }
        if samples.is_empty() { return Err(format!("{key}: a median with no runs")); }
        if samples.iter().all(|s| matches!(s.get(key), Value::Null)) { continue; }
        recomputes(summary, samples, key).map_err(|e| format!("{key}: {e}"))?;
    }
    Ok(())
}
/// Run indices: integers, no duplicates.
fn run_ids(rows: &[Value]) -> std::result::Result<BTreeSet<i64>, String> {
    let mut ids = BTreeSet::new();
    for r in rows {
        let id = int(r.get("run")).ok_or(format!("a run has no integer run index ({})", short(&r.get("run").encode(), 20)))?;
        if !ids.insert(id) { return Err(format!("run {id} appears twice")); }
    }
    Ok(ids)
}
/// Every run of a row stopped by itself, sent [DONE] and passed the output check.
fn natural_stops(w: &str, rows: &[Value]) -> std::result::Result<(), String> {
    for r in rows {
        let id = r.get("run").encode();
        if text(r.get("finish_reason")) != Some("stop") { return Err(format!("{w} run {id}: finish is not stop")); }
        if *r.get("stream_done_marker") != Value::Bool(true) { return Err(format!("{w} run {id}: no [DONE] marker")); }
        if *at(r, &["completion_validation", "valid"]) != Value::Bool(true) { return Err(format!("{w} run {id}: output check failed")); }
    }
    Ok(())
}
/// The row's output gate is a full pass over 5 answers or more.
fn full_gate(w: &str, row: &Value) -> std::result::Result<(), String> {
    let g = row.get("completion_gate");
    match (int(g.get("passed")), int(g.get("total"))) {
        (Some(p), Some(t)) if t >= 5 && p == t => Ok(()),
        _ => Err(format!("{w}: output gate {}/{}", g.get("passed").encode(), g.get("total").encode())),
    }
}
/// A file whose change can change what the engine or CLI build makes. Under `engine/`, `cli/` and `harness/` (the
/// CLI links it) every file counts except Markdown and saved bench receipts under `experiments/`; source and
/// templates there always count. Toolchain and Cargo settings count anywhere. Files that the source embeds with
/// `include_str!` or `include_bytes!` count too; [`check`] adds those from git.
pub fn build_input(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    if name.starts_with("rust-toolchain") || path.contains(".cargo/") || matches!(path, "Cargo.toml" | "Cargo.lock") { return true; }
    if !["engine/", "cli/", "harness/"].iter().any(|d| path.starts_with(d)) { return false; }
    let source = [".rs", ".ptx", ".cu", ".cuh", ".c", ".h", ".jinja", ".toml", ".lock"].iter().any(|e| name.ends_with(e));
    source || !(name.ends_with(".md") || path.contains("/experiments/"))
}
/// What an embed macro points at: one file, a folder prefix (a `concat!` of a folder and more), or only a file name.
#[derive(Debug, PartialEq)]
enum Embed { File(String), Prefix(String), Name(String) }
fn normalize(dir: &str, rel: &str) -> String {
    let mut out: Vec<&str> = dir.split('/').filter(|s| !s.is_empty()).collect();
    for seg in rel.split('/') {
        match seg { "" | "." => (), ".." => { out.pop(); }, s => out.push(s) }
    }
    out.join("/") + if rel.ends_with('/') { "/" } else { "" }
}
/// The embeds in one source file's text, or an error when one names no path literal.
fn embeds(path: &str, source: &str) -> std::result::Result<Vec<Embed>, String> {
    let dir = path.rsplit_once('/').map_or("", |(d, _)| d);
    let mut out = Vec::new();
    for (start, _) in source.match_indices("include_") {
        let rest = &source[start..];
        let Some(after) = rest.strip_prefix("include_str!").or(rest.strip_prefix("include_bytes!")) else { continue };
        let after = after.trim_start();
        if !after.starts_with('(') { continue; }
        let mut depth = 0usize;
        let end = after.char_indices().find(|&(_, c)| {
            if c == '(' { depth += 1; } else if c == ')' { depth -= 1; }
            depth == 0
        }).map_or(after.len(), |(i, _)| i);
        let call = &after[..end];
        let literals: Vec<&str> = call.split('"').skip(1).step_by(2).collect();
        let Some(first) = literals.first() else { return Err(format!("{path}: an embed macro names no path literal")) };
        out.push(match *first {
            "OUT_DIR" => continue,
            "CARGO_MANIFEST_DIR" => Embed::Name(literals.last().unwrap_or(first).rsplit('/').next().unwrap_or("").to_string()),
            rel if call.contains("concat!") => Embed::Prefix(normalize(dir, rel)),
            rel => Embed::File(normalize(dir, rel)),
        });
    }
    Ok(out)
}
/// Every embed in the engine, CLI and harness source at one commit.
fn embeds_at(repo: &Path, commit: &str) -> std::result::Result<Vec<Embed>, String> {
    let listed = git(repo).args(["grep", "-l", "-I", "-E", "include_(str|bytes)!", commit, "--", "engine", "cli", "harness"])
        .output().map_err(|e| format!("git did not run: {e}"))?;
    if !listed.status.success() && listed.status.code() != Some(1) { return Err(format!("git grep failed at {}", short(commit, 9))); }
    let specs = String::from_utf8_lossy(&listed.stdout).to_string();
    let mut child = git(repo).args(["cat-file", "--batch"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
        .map_err(|e| format!("git did not run: {e}"))?;
    let mut input = child.stdin.take().ok_or("git cat-file has no input")?;
    let feed = specs.clone();
    let writer = std::thread::spawn(move || input.write_all(feed.as_bytes()));
    let out = child.wait_with_output().map_err(|e| format!("git cat-file: {e}"))?;
    if !matches!(writer.join(), Ok(Ok(()))) || !out.status.success() { return Err("git cat-file failed".into()); }
    let (mut bytes, mut found) = (&out.stdout[..], Vec::new());
    for spec in specs.lines() {
        let path = spec.split_once(':').map_or(spec, |(_, p)| p);
        let line_end = bytes.iter().position(|&b| b == b'\n').ok_or("git cat-file output is short")?;
        let header = String::from_utf8_lossy(&bytes[..line_end]).to_string();
        let size: usize = header.rsplit(' ').next().and_then(|s| s.parse().ok()).ok_or(format!("git cat-file: {header}"))?;
        let body = bytes.get(line_end + 1..line_end + 1 + size).ok_or("git cat-file output is short")?;
        found.extend(embeds(path, &String::from_utf8_lossy(body))?);
        bytes = bytes.get(line_end + 2 + size..).unwrap_or(&[]);
    }
    Ok(found)
}
/// The build inputs that differ between two commits, embeds at either commit included.
fn changed_build_inputs(repo: &Path, a: &str, b: &str) -> std::result::Result<Vec<String>, String> {
    let out = git(repo).args(["diff", "--no-renames", "--name-only", a, b]).output().map_err(|e| format!("git did not run: {e}"))?;
    if !out.status.success() { return Err(format!("git cannot compare {} with {}", short(a, 9), short(b, 9))); }
    let mut embedded = embeds_at(repo, a)?;
    embedded.extend(embeds_at(repo, b)?);
    Ok(String::from_utf8_lossy(&out.stdout).lines().filter(|f| build_input(f) || embedded.iter().any(|e| match e {
        Embed::File(p) => f == p,
        Embed::Prefix(p) => f.starts_with(p.as_str()),
        Embed::Name(n) => f.rsplit('/').next() == Some(n.as_str()),
    })).map(str::to_string).collect())
}
fn mode_of(value: &Value) -> Mode {
    let body = at(value, &["settings", "extra_body"]);
    let drafter = text(at(value, &["run", "appliance", "drafter"]));
    match (text(body.get("draft")), int(body.get("mtp_depth")), drafter) {
        (Some("plain"), Some(0), Some("off")) => Mode::Plain,
        (Some("strict"), Some(d), Some("on")) if d >= 1 => Mode::Draft,
        _ => Mode::Mixed,
    }
}
fn mode_words(p: &Part) -> String {
    let body = at(&p.value, &["settings", "extra_body"]);
    format!("{}: draft {} with mtp_depth {} and drafter {} do not agree", p.name, body.get("draft").encode(),
        body.get("mtp_depth").encode(), at(&p.value, &["run", "appliance", "drafter"]).encode())
}
/// The request body without the two drafting keys.
fn body_but_draft(value: &Value) -> String {
    match at(value, &["settings", "extra_body"]) {
        Value::Object(o) => Value::Object(o.iter().filter(|(k, _)| !DRAFT_KEYS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect()).encode(),
        other => other.encode(),
    }
}
fn gates_line(t: &str) -> Option<(u64, u64)> {
    let at = t.find(" BASIC OUTPUT GATES PASSED")?;
    let head = &t[..at];
    let start = head.char_indices().rev().find(|&(_, c)| !(c.is_ascii_digit() || c == '/')).map_or(0, |(i, c)| i + c.len_utf8());
    let (a, b) = head[start..].split_once('/')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}
/// The one complete RESULT-CARD block: BEGIN, END after it, and no key twice.
fn result_card(t: &str) -> std::result::Result<BTreeMap<String, String>, String> {
    if t.matches("RESULT-CARD BEGIN").count() != 1 { return Err("need exactly one RESULT-CARD BEGIN".into()); }
    let (_, rest) = t.split_once("RESULT-CARD BEGIN").ok_or("no RESULT-CARD BEGIN")?;
    let (body, _) = rest.split_once("RESULT-CARD END").ok_or("the RESULT-CARD block has no END (truncated)")?;
    let mut out = BTreeMap::new();
    for line in body.lines() {
        if let Some((k, v)) = line.split_once(':') && !k.trim().is_empty() && !k.trim().contains(' ')
            && out.insert(k.trim().to_string(), v.trim().to_string()).is_some() {
            return Err(format!("the RESULT-CARD block has {} twice", k.trim()));
        }
    }
    Ok(out)
}
/// The main ref as one commit. An empty ref or one that looks like an option is refused.
fn resolve(repo: &Path, name: &str) -> std::result::Result<String, String> {
    if name.trim().is_empty() || name.starts_with('-') { return Err(format!("main ref '{name}' is not a ref")); }
    let out = git(repo).args(["rev-parse", "--verify", "--quiet", &format!("{name}^{{commit}}")]).output()
        .map_err(|e| format!("git did not run: {e}"))?;
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if out.status.success() && crate::is_lower_hex(&sha, 40) { Ok(sha) } else { Err(format!("main ref '{name}' names no commit")) }
}
/// A fresh temporary path: process id, a counter and the clock, created new so two runs never share it.
fn temporary(bytes: &[u8]) -> std::io::Result<PathBuf> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let path = std::env::temp_dir().join(format!("vade-qualify-goldens-{}-{n}-{nanos}.tsv", std::process::id()));
    fs::OpenOptions::new().write(true).create_new(true).open(&path)?.write_all(bytes)?;
    Ok(path)
}

/// Run every check. Errors only on an unreadable folder; a missing receipt is a FAIL line.
pub fn check(options: &Options) -> Result<Report> {
    let dir = &options.dir;
    let mut read = Read::default();
    let mut names: Vec<String> = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok()?.file_name().into_string().ok()).collect();
    names.sort();
    let mut parts = Vec::new();
    let mut corrupt = Vec::new();
    for name in names.iter().filter(|n| n.ends_with(".json") && !n.ends_with(".card.json")) {
        // Fail closed: a JSON file that cannot be read, is not UTF-8 or does not parse may be a broken card part.
        let parsed = read.bytes(dir, name).ok_or("unreadable".to_string())
            .and_then(|b| String::from_utf8(b).map_err(|_| "not UTF-8".to_string()))
            .and_then(|t| json::parse(&t).map_err(|e| short(&e, 60).to_string()));
        let value = match parsed { Ok(v) => v, Err(e) => { corrupt.push(format!("{name}: {e}")); continue } };
        let card = matches!(value.get("run"), Value::Object(_)) && matches!(value.get("settings"), Value::Object(_))
            && ["decode", "prefill", "concurrency"].iter().any(|k| matches!(value.get(k), Value::Object(_)));
        if !card { read.files.remove(name); continue; }
        let text = read.text(dir, &name.replace(".json", ".card.txt"));
        parts.push(Part { name: name.clone(), mode: mode_of(&value), value, text });
    }
    let plain: Vec<&Part> = parts.iter().filter(|p| p.mode == Mode::Plain).collect();
    let draft: Vec<&Part> = parts.iter().filter(|p| p.mode == Mode::Draft).collect();
    let mixed: Vec<&Part> = parts.iter().filter(|p| p.mode == Mode::Mixed).collect();
    let mut checks = Vec::new();
    let mut info = Vec::new();
    let mut add = |i: usize, result: std::result::Result<String, String>, receipts: Vec<String>| {
        let (pass, detail) = match result { Ok(d) => (true, d), Err(d) => (false, d) };
        checks.push(Check { id: CHECKS[i].0, name: CHECKS[i].1, pass, detail, receipts });
    };
    let part_names: Vec<&str> = parts.iter().map(|p| p.name.as_str()).collect();
    let card_texts: Vec<String> = parts.iter().map(|p| p.name.replace(".json", ".card.txt")).collect();
    let first = plain.first().copied().or(parts.first());
    let field = |keys: &[&str]| first.and_then(|p| text(at(&p.value, keys))).unwrap_or_default().to_string();

    // Q01: the parts belong to one card.
    let engine = commit(&field(&["run", "appliance", "serving_engine"])).unwrap_or_default();
    let model = field(&["run", "model"]);
    let comparison = field(&["run", "comparison_id"]);
    let q01 = (|| {
        if let Some(c) = corrupt.first() { return Err(format!("a JSON file in the folder is broken: {c}")); }
        if plain.is_empty() { return Err("no plain card part in the folder".to_string()); }
        if engine.is_empty() { return Err("the card names no 40-digit engine commit".into()); }
        if model.is_empty() || comparison.is_empty() { return Err("the card names no model or no comparison id".into()); }
        for p in &parts {
            let v = &p.value;
            if text(at(v, &["run", "appliance", "serving_engine"])).and_then(commit).as_deref() != Some(&engine) { return Err(format!("{}: another engine commit", p.name)); }
            if text(at(v, &["run", "model"])) != Some(&model) { return Err(format!("{}: another model", p.name)); }
            if text(at(v, &["run", "comparison_id"])) != Some(&comparison) { return Err(format!("{}: another comparison id", p.name)); }
            let tool = at(v, &["run", "rigmark_release_check"]);
            if !text(tool.get("revision")).is_some_and(|r| crate::is_lower_hex(r, 40)) { return Err(format!("{}: no RigMark tool revision", p.name)); }
            if *tool.get("dirty") != Value::Bool(false) { return Err(format!("{}: RigMark tool is not clean", p.name)); }
            if !p.text.as_deref().is_some_and(|t| !t.trim().is_empty()) { return Err(format!("{}: no .card.txt beside it, or it is empty", p.name)); }
        }
        Ok(format!("{} part(s), engine {}, model {model}, comparison id {comparison}", parts.len(), short(&engine, 9)))
    })();
    let mut q01_files: Vec<&str> = part_names.iter().copied().chain(card_texts.iter().map(String::as_str)).collect();
    let corrupt_names: Vec<&str> = corrupt.iter().filter_map(|c| c.split(": ").next()).collect();
    q01_files.extend(&corrupt_names);
    add(0, q01, read.tag(&q01_files));

    // Q02: on main. The ref is resolved once; the digest binds the commit it named.
    let main = resolve(&options.repo, &options.main_ref);
    if let Ok(sha) = &main { read.files.insert(format!("git {}", options.main_ref), sha.clone()); }
    let q02 = match (&main, engine.is_empty()) {
        (_, true) => Err("no engine commit".to_string()),
        (Err(e), _) => Err(e.clone()),
        (Ok(sha), false) => match git(&options.repo).args(["merge-base", "--is-ancestor", &engine, sha]).output().map(|o| o.status) {
            Ok(s) if s.success() => Ok(format!("{} is an ancestor of {} ({})", short(&engine, 9), options.main_ref, short(sha, 9))),
            Ok(_) => Err(format!("{} is not on {} ({})", short(&engine, 9), options.main_ref, short(sha, 9))),
            Err(e) => Err(format!("git did not run: {e}")),
        },
    };
    let main_line = format!("git {} = {} in {}", options.main_ref, main.as_deref().unwrap_or("(none)"), options.repo.display());
    add(1, q02, vec![main_line]);

    // Q03: the served build is the commit, and every rank ran the same binary from the kit.
    let health = read.text(dir, "health.json").and_then(|t| json::parse(&t).ok()).unwrap_or(Value::Null);
    let sums = read.text(dir, "SHA256SUMS").unwrap_or_default();
    let kit: BTreeMap<&str, &str> = sums.lines().filter_map(|l| { let (h, n) = l.split_once(char::is_whitespace)?;
        Some((n.trim().trim_start_matches('*').rsplit('/').next()?, h)) }).collect();
    let live: Vec<String> = names.iter().filter(|n| n.starts_with("live-binary-") && n.ends_with(".txt")).cloned().collect();
    let sparks = first.and_then(|p| text(at(&p.value, &["run", "appliance", "hardware"])))
        .and_then(|h| h.split(' ').next()?.parse::<usize>().ok()).unwrap_or(0);
    let q03 = (|| {
        let sha = text(at(&health, &["vade", "build", "git_sha"])).ok_or("health.json has no vade.build.git_sha")?;
        if engine.is_empty() || sha != engine { return Err(format!("health build {} is not the card engine", short(sha, 9))); }
        if *at(&health, &["vade", "build", "dirty"]) != Value::Bool(false) { return Err("health says the build is dirty".to_string()); }
        let served = *kit.get("flash-serve").ok_or("SHA256SUMS has no flash-serve line")?;
        let ptx = *kit.get("flash.ptx").ok_or("SHA256SUMS has no flash.ptx line")?;
        if !crate::is_lower_hex(served, 64) || !crate::is_lower_hex(ptx, 64) { return Err("a SHA256SUMS hash is not 64 hex digits".into()); }
        // Each file must show the kit hash, so a stale extra file fails; one file a rank of each load is normal.
        if live.len() < sparks.max(2) { return Err(format!("{} live-binary file(s), need one a rank ({})", live.len(), sparks.max(2))); }
        for name in &live {
            let t = read.text(dir, name).unwrap_or_default();
            let h = t.split_whitespace().next().unwrap_or("");
            if h != served { return Err(format!("{name}: live binary {} is not the kit flash-serve {}", short(h, 12), short(served, 12))); }
        }
        let revision = text(at(&health, &["vade", "checkpoint_sha256"])).or(text(at(&health, &["vade", "checkpoint_fingerprint"]))).unwrap_or("");
        if revision.is_empty() { return Err("health.json names no checkpoint identity".into()); }
        for p in &parts {
            if text(at(&p.value, &["run", "appliance", "model_revision"])) != Some(revision) {
                return Err(format!("{}: model_revision is not the health checkpoint identity", p.name));
            }
        }
        Ok(format!("build {} clean; flash-serve {} in {} live-binary file(s); PTX {}", short(&engine, 9), short(served, 12), live.len(), short(ptx, 12)))
    })();
    let mut q03_files = vec!["health.json", "SHA256SUMS"]; q03_files.extend(live.iter().map(String::as_str));
    add(2, q03, read.tag(&q03_files));
    if let Some(q) = text(at(&health, &["vade", "qualification"])) {
        info.push(format!("engine health qualification field: {q} (the engine reports its start mode; this checklist is the card qualification)"));
    }

    // Q04: the weights are the same bytes on every rank, and every shard of a sharded checkpoint is hashed.
    let weights: Vec<String> = names.iter().filter(|n| n.starts_with("checkpoint-sha256-") && n.ends_with(".txt")).cloned().collect();
    let q04 = (|| {
        if weights.len() < sparks.max(2) { return Err(format!("{} checkpoint-sha256 file(s), need one a rank", weights.len())); }
        let mut first: Option<(String, Vec<String>)> = None;
        for name in &weights {
            let t = read.text(dir, name).ok_or(format!("{name}: unreadable or not UTF-8"))?;
            let mut lines = Vec::new();
            let mut shards: BTreeMap<(String, u32), BTreeSet<u32>> = BTreeMap::new();
            for l in t.lines().filter(|l| !l.trim().is_empty()) {
                let (h, n) = l.split_once(char::is_whitespace).ok_or(format!("{name}: a line is not a sha256 line"))?;
                let file = n.trim().trim_start_matches('*').rsplit('/').next().unwrap_or("");
                if !crate::is_lower_hex(h, 64) || file.is_empty() { return Err(format!("{name}: a line is not a sha256 line")); }
                // model-00001-of-00040.safetensors: shard 1 of 40.
                if let Some(stem) = file.strip_suffix(".safetensors") && let Some((head, total)) = stem.rsplit_once("-of-")
                    && let Some((prefix, index)) = head.rsplit_once('-') && let (Ok(i), Ok(t)) = (index.parse::<u32>(), total.parse::<u32>()) {
                    shards.entry((prefix.to_string(), t)).or_default().insert(i);
                }
                lines.push(format!("{h}  {file}"));
            }
            lines.sort();
            if !lines.iter().any(|l| l.ends_with(".safetensors")) { return Err(format!("{name}: no .safetensors file hashed")); }
            for ((prefix, total), seen) in &shards {
                if *total == 0 || seen.len() != *total as usize || seen.iter().any(|i| *i < 1 || i > total) {
                    return Err(format!("{name}: {} of {total} {prefix} shards hashed", seen.len()));
                }
            }
            match &first { None => first = Some((name.clone(), lines)), Some((a, b)) if *b != lines => return Err(format!("{name} differs from {a}")), _ => () }
        }
        let (_, lines) = first.ok_or("none")?;
        Ok(format!("{} files hashed, equal on {} rank(s), set sha256 {}", lines.len(), weights.len(), short(&sha256::digest(lines.join("\n").as_bytes()), 12)))
    })();
    add(3, q04, read.tag(&weights.iter().map(String::as_str).collect::<Vec<_>>()));

    // Q05: the three workloads. The part with the most samples gives each row, as on the site.
    let best = |w: &str| plain.iter().filter(|p| !runs(at(&p.value, &["decode", w])).is_empty())
        .max_by_key(|p| runs(at(&p.value, &["decode", w])).len()).copied();
    let mut rates = BTreeMap::new();
    let q05 = (|| {
        for w in WORKLOADS {
            let p = best(w).ok_or(format!("{w}: not measured"))?;
            let row = at(&p.value, &["decode", w]);
            let rs = runs(row);
            if rs.len() < 5 { return Err(format!("{w}: {} run(s), need 5", rs.len())); }
            run_ids(rs).map_err(|e| format!("{w}: {e}"))?;
            if !num(at(&p.value, &["settings", "decode_tokens"])).is_some_and(|t| t >= 8192.) { return Err(format!("{w}: output cap under 8192")); }
            natural_stops(w, rs)?;
            full_gate(w, row)?;
            match p.text.as_deref().and_then(gates_line) { Some((a, b)) if a == b && b > 0 => (), _ => return Err(format!("{}: card text does not show all gates passed", p.name)) }
            rates.insert(w, recomputes(row.get("decode_tokens_per_second"), rs, "decode_tokens_per_second").map_err(|e| format!("{w}: {e}"))?);
        }
        Ok(WORKLOADS.iter().map(|w| format!("{w} {:.3}", rates[w])).collect::<Vec<_>>().join(", ") + " tok/s; 15 natural stops, all gates")
    })();
    let q05_parts: Vec<&Part> = WORKLOADS.iter().filter_map(|w| best(w)).collect();
    let q05_texts: Vec<String> = q05_parts.iter().map(|p| p.name.replace(".json", ".card.txt")).collect();
    add(4, q05, read.tag(&q05_parts.iter().map(|p| p.name.as_str()).chain(q05_texts.iter().map(String::as_str)).collect::<Vec<_>>()));

    // Q06: an 8K cold read.
    let prefill = plain.iter().find(|p| !runs(at(&p.value, &["prefill", "8192", "cold"])).is_empty()).copied();
    let q06 = (|| {
        let p = prefill.ok_or("no 8192 prefill section")?;
        let cold = at(&p.value, &["prefill", "8192", "cold"]);
        let rs = runs(cold);
        if rs.len() < 3 { return Err(format!("{} cold read(s), need 3", rs.len())); }
        if rs.iter().any(|r| num(r.get("cached_prompt_tokens")) != Some(0.)) { return Err("a cold read reports cached prompt tokens, or none".into()); }
        if rs.iter().any(|r| num(r.get("requested_prompt_tokens")) != Some(8192.)) { return Err("a cold read is not 8192 tokens, or does not say".into()); }
        let t = recomputes(cold.get("ttft_seconds"), rs, "ttft_seconds")?;
        let r = recomputes(cold.get("effective_prefill_tokens_per_second"), rs, "effective_prefill_tokens_per_second")?;
        Ok(format!("cold 8K {t:.3} s, {r:.3} tok/s, {} reads, cached 0", rs.len()))
    })();
    add(5, q06, read.tag(&prefill.map(|p| vec![p.name.as_str()]).unwrap_or_default()));
    if let Some(p) = plain.iter().find(|p| !runs(at(&p.value, &["prefill", "32768", "cold"])).is_empty()) {
        info.push(format!("32K cold read present in {}: {} s (not required)", p.name, at(&p.value, &["prefill", "32768", "cold", "ttft_seconds", "median"]).encode()));
    }

    // Q07: chats at once.
    let conc = plain.iter().find(|p| ["1", "2", "4", "8"].iter().all(|c| matches!(at(&p.value, &["concurrency", c]), Value::Object(_)))).copied();
    let q07 = (|| {
        let p = conc.ok_or("no part measures concurrency 1, 2, 4 and 8")?;
        let mut out = Vec::new();
        for c in ["1", "2", "4", "8"] {
            let cell = at(&p.value, &["concurrency", c]);
            let Value::Array(rounds) = cell.get("rounds") else { return Err(format!("C{c}: no rounds")) };
            if rounds.len() < 3 { return Err(format!("C{c}: {} round(s), need 3", rounds.len())); }
            for round in rounds {
                let Value::Array(streams) = round.get("streams") else { return Err(format!("C{c}: a round has no streams")) };
                if streams.len().to_string() != c { return Err(format!("C{c}: a round has {} streams", streams.len())); }
                let ended = |s: &Value| *s.get("stream_done_marker") == Value::Bool(true) && matches!(text(s.get("finish_reason")), Some("stop" | "length"));
                if !streams.iter().all(ended) { return Err(format!("C{c}: a stream did not end")); }
            }
            let rate = recomputes(cell.get("aggregate_end_to_end_tokens_per_second"), rounds, "aggregate_end_to_end_tokens_per_second");
            out.push(format!("C{c} {:.3}", rate.map_err(|e| format!("C{c}: {e}"))?));
        }
        Ok(out.join(", ") + " tok/s total")
    })();
    add(6, q07, read.tag(&conc.map(|p| vec![p.name.as_str()]).unwrap_or_default()));

    // Q08: one honest request body for every plain part; no part with a drafting mode that does not agree.
    let q08 = (|| {
        if let Some(p) = mixed.first() { return Err(mode_words(p)); }
        let first = plain.first().ok_or("no plain part")?;
        let body = at(&first.value, &["settings", "extra_body"]);
        let seed = at(&first.value, &["settings", "seed"]);
        if int(seed).is_none() { return Err(format!("{}: no integer seed", first.name)); }
        for p in &plain {
            let s = |k: &str| at(&p.value, &["settings", k]).encode();
            if at(&p.value, &["settings", "extra_body"]).encode() != body.encode() { return Err(format!("{}: another request body", p.name)); }
            if num(at(&p.value, &["settings", "temperature"])) != Some(0.) { return Err(format!("{}: temperature is not 0", p.name)); }
            if s("seed") != seed.encode() { return Err(format!("{}: another seed", p.name)); }
            if s("top_p") != at(&first.value, &["settings", "top_p"]).encode() { return Err(format!("{}: another top_p", p.name)); }
        }
        if *body.get("cache") != Value::Bool(false) { return Err("body cache is not false".into()); }
        if model.to_lowercase().contains("glm") && text(body.get("reasoning_effort")) != Some("low") {
            return Err("GLM body needs reasoning_effort low (thinking cannot be switched off)".into());
        }
        Ok(format!("body {}", body.encode()))
    })();
    let mut q08_files = part_names.clone(); q08_files.retain(|n| plain.iter().chain(&mixed).any(|p| p.name == *n));
    add(7, q08, read.tag(&q08_files));

    // Q09: the drafting row. Same comparison id = same prompts a run index, so exact drafting gives the same answers.
    let q09 = (|| {
        if let Some(p) = mixed.first() { return Err(mode_words(p)); }
        if draft.is_empty() { return Err("no drafting part (draft strict, mtp_depth 1 or more, drafter on, same comparison id)".to_string()); }
        let reference = plain.first().ok_or("no plain part to compare")?;
        for d in &draft {
            if body_but_draft(&d.value) != body_but_draft(&reference.value) { return Err(format!("{}: request body differs from plain beyond draft and mtp_depth", d.name)); }
            for k in SAME_SETTINGS {
                let (a, b) = (at(&d.value, &["settings", k]), at(&reference.value, &["settings", k]));
                if matches!(a, Value::Null) || a.encode() != b.encode() { return Err(format!("{}: settings {k} is not the plain {k}", d.name)); }
            }
            match d.text.as_deref().and_then(gates_line) { Some((a, b)) if a == b && b > 0 => (), _ => return Err(format!("{}: card text does not show all gates passed", d.name)) }
        }
        let mut out = Vec::new();
        for w in WORKLOADS {
            let rows: Vec<&&Part> = draft.iter().filter(|p| !runs(at(&p.value, &["decode", w])).is_empty()).collect();
            if rows.is_empty() { return Err(format!("{w}: no drafting row")); }
            let p = best(w).ok_or(format!("{w}: no plain row to compare"))?;
            let pr = runs(at(&p.value, &["decode", w]));
            run_ids(pr).map_err(|e| format!("{w} plain: {e}"))?;
            for (i, d) in rows.iter().enumerate() {
                let row = at(&d.value, &["decode", w]);
                let dr = runs(row);
                if dr.len() < 5 { return Err(format!("{w}: drafting part {} has {} run(s), need 5", d.name, dr.len())); }
                run_ids(dr).map_err(|e| format!("{w} drafting: {e}"))?;
                natural_stops(w, dr)?;
                full_gate(w, row)?;
                for r in dr {
                    let id = r.get("run").encode();
                    let twin = pr.iter().find(|x| x.get("run") == r.get("run")).ok_or(format!("{w} run {id}: no plain run"))?;
                    for key in ["output_sha256", "reasoning_sha256"] {
                        if !hex64(r.get(key)) || !hex64(twin.get(key)) { return Err(format!("{w} run {id}: {key} is missing or not 64 hex digits")); }
                        if r.get(key) != twin.get(key) { return Err(format!("{w} run {id}: drafting answer differs from plain ({key})")); }
                    }
                }
                let rate = recomputes(row.get("decode_tokens_per_second"), dr, "decode_tokens_per_second").map_err(|e| format!("{w}: {e}"))?;
                if i == 0 { out.push(format!("{w} {rate:.3} ({:+.1}% vs plain)", rates.get(w).map_or(0., |p| (rate / p - 1.) * 100.))); }
            }
        }
        Ok(out.join(", ") + "; every answer equal to plain")
    })();
    let draft_texts: Vec<String> = draft.iter().map(|p| p.name.replace(".json", ".card.txt")).collect();
    let mut q09_files: Vec<&str> = draft.iter().map(|p| p.name.as_str()).chain(draft_texts.iter().map(String::as_str)).collect();
    q09_files.extend(WORKLOADS.iter().filter_map(|w| best(w).map(|p| p.name.as_str())));
    add(8, q09, read.tag(&q09_files));

    // Q10: the gate prompts give the golden token IDs on this load.
    let gates: Vec<String> = fs::read_dir(dir.join("goldens")).map(|it| it.filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with(".jsonl")).map(|n| format!("goldens/{n}")).collect()).unwrap_or_default();
    let mut gates = gates; gates.sort();
    let gate_bytes: Vec<(String, Option<Vec<u8>>)> = gates.iter().map(|g| (g.clone(), read.bytes(dir, g))).collect();
    // The golden file comes from git at the resolved main commit, never from the working tree; the bytes read are the
    // bytes checked and the bytes the digest binds.
    let (manifest, manifest_name) = match (&options.goldens, &main) {
        (Some(path), _) => (fs::read(path).map_err(|e| format!("golden file {}: {e}", path.display())), path.display().to_string()),
        (None, Ok(sha)) => {
            let name = format!("{sha}:tools/goldens/{model}.tsv");
            let shown = git(&options.repo).args(["show", &name]).output().ok().filter(|o| o.status.success() && !model.is_empty());
            (shown.map(|o| o.stdout).ok_or(format!("no golden file {}:tools/goldens/{model}.tsv", options.main_ref)), name)
        }
        (None, Err(e)) => (Err(format!("no golden file: {e}")), String::new()),
    };
    if let Ok(b) = &manifest { read.files.insert(format!("golden {manifest_name}"), sha256::digest(b)); }
    let q10 = (|| {
        if gates.is_empty() { return Err("no goldens/*.jsonl gate receipt".to_string()); }
        for (g, b) in &gate_bytes {
            let t = b.as_ref().and_then(|b| std::str::from_utf8(b).ok()).ok_or(format!("{g}: unreadable or not UTF-8"))?;
            if t.lines().all(|l| l.trim().is_empty()) { return Err(format!("{g}: empty gate receipt")); }
        }
        let bytes = manifest.as_ref().map_err(String::clone)?;
        if bytes.is_empty() { return Err(format!("golden file {manifest_name} is empty")); }
        let copy = temporary(bytes).map_err(|e| format!("cannot write the golden copy: {e}"))?;
        let out = Command::new(&options.goldens_bin).arg(&copy).args(gates.iter().map(|g| dir.join(g))).output();
        let _ = fs::remove_file(&copy);
        let out = out.map_err(|e| format!("check-goldens did not run: {e}"))?;
        let said = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let model_word = format!("model={model}");
        if out.status.success() && said.starts_with("GOLDENS PASS") && said.split_whitespace().any(|w| w == model_word) {
            Ok(format!("{said} ({} receipt file(s))", gates.len()))
        } else {
            let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            Err(format!("{} {} (card model {model})", said, err).trim().to_string())
        }
    })();
    let mut q10_tags = read.tag(&gates.iter().map(String::as_str).collect::<Vec<_>>());
    if let Ok(b) = &manifest { q10_tags.push(format!("{manifest_name} sha256={}", short(&sha256::digest(b), 12))); }
    add(9, q10, q10_tags);

    // Q11: quality against the maker, judged elsewhere, bound to this engine commit.
    let q11 = (|| {
        let t = read.text(dir, "quality-vs-maker.result").ok_or("no quality-vs-maker.result")?;
        let card = result_card(&t)?;
        let verdict = card.get("VERDICT").ok_or("no VERDICT line in the RESULT-CARD block")?;
        if !(verdict == "PASS" || verdict.starts_with("PASS ")) { return Err(format!("VERDICT {verdict}")); }
        if let Some(m) = card.get("MODEL") && *m != model { return Err(format!("quality receipt is for model {m}, card is {model}")); }
        let named = card.get("ENGINE").and_then(|e| commit(e)).ok_or("no 40-digit ENGINE line")?;
        if engine.is_empty() { return Err("the card names no engine commit".into()); }
        if named == engine { return Ok(format!("VERDICT {verdict} for {}", short(&named, 9))); }
        // Another commit counts only when no build input of the engine, CLI or harness differs, embedded files included.
        let changed = changed_build_inputs(&options.repo, &named, &engine)?;
        if let Some(f) = changed.first() {
            return Err(format!("quality receipt is for {}, card is {}, and {} build input(s) differ, first {f}", short(&named, 9), short(&engine, 9), changed.len()));
        }
        Ok(format!("VERDICT {verdict} for {}, same engine and CLI build inputs as {}", short(&named, 9), short(&engine, 9)))
    })();
    add(10, q11, read.tag(&["quality-vs-maker.result"]));

    // Q12: every rate is a served two-Spark rate, recomputed from its own runs, from one session.
    let q12 = (|| {
        if parts.is_empty() { return Err("no card parts".to_string()); }
        let mut starts = Vec::new();
        for p in &parts {
            let a = at(&p.value, &["run", "appliance"]);
            let hw = text(a.get("hardware")).unwrap_or("");
            if !hw.contains(" NVIDIA DGX Spark") || sparks < 2 { return Err(format!("{}: hardware '{hw}' is not two Sparks or more", p.name)); }
            if !hw.starts_with(&format!("{sparks} ")) { return Err(format!("{}: another Spark count", p.name)); }
            let words = format!("{} {} {} {}", text(at(&p.value, &["run", "label"])).unwrap_or(""), text(a.get("purpose")).unwrap_or(""),
                text(a.get("serving_engine")).unwrap_or(""), text(a.get("topology")).unwrap_or("")).to_lowercase();
            if let Some(w) = FIXTURE_WORDS.iter().find(|w| words.contains(*w)) { return Err(format!("{}: names '{w}', not a served card", p.name)); }
            if matches!(p.value.get("protocol"), Value::Null) { return Err(format!("{}: no RigMark protocol field", p.name)); }
            let start = text(at(&p.value, &["run", "started_at"])).filter(|s| s.len() >= 13).ok_or(format!("{}: no start time", p.name))?;
            starts.push(short(start, 13).to_string());
            for w in WORKLOADS {
                let row = at(&p.value, &["decode", w]);
                all_medians(row, runs(row)).map_err(|e| format!("{} {w}: {e}", p.name))?;
            }
            if let Value::Object(depths) = p.value.get("prefill") {
                for (depth, phases) in depths {
                    let Value::Object(phases) = phases else { continue };
                    for (phase, row) in phases { all_medians(row, runs(row)).map_err(|e| format!("{} prefill {depth} {phase}: {e}", p.name))?; }
                }
            }
            if let Value::Object(cells) = p.value.get("concurrency") {
                for (c, cell) in cells {
                    let rounds = if let Value::Array(r) = cell.get("rounds") { &r[..] } else { &[] };
                    all_medians(cell, rounds).map_err(|e| format!("{} C{c}: {e}", p.name))?;
                }
            }
        }
        match int(at(&health, &["vade", "numerics", "world_size"])) {
            Some(n) if n == sparks as i64 => (),
            other => return Err(format!("health world_size {} is not the card Spark count {sparks}", other.map_or("missing".into(), |n| n.to_string()))),
        }
        starts.sort(); starts.dedup();
        Ok(format!("{} part(s), {sparks} Sparks, served, every median recomputes; start hours {}", parts.len(), starts.join(", ")))
    })();
    add(11, q12, read.tag(&part_names.iter().copied().chain(["health.json"]).collect::<Vec<_>>()));

    let digest = sha256::digest(read.files.iter().map(|(n, h)| format!("{n}\t{h}\n")).collect::<String>().as_bytes());
    Ok(Report { checks, info, digest, engine })
}

impl Report {
    pub fn passed(&self) -> bool { self.checks.len() == CHECKS.len() && self.checks.iter().all(|c| c.pass) }
    /// The printed checklist. The last line is the verdict.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for c in &self.checks {
            out.push_str(&format!("{} {} {}: {}\n", if c.pass { "PASS" } else { "FAIL" }, c.id, c.name, c.detail));
            for r in &c.receipts { out.push_str(&format!("     read {r}\n")); }
        }
        for i in &self.info { out.push_str(&format!("INFO {i}\n")); }
        let passed = self.checks.iter().filter(|c| c.pass).count();
        out.push_str(&format!("CARD-QUALIFY {} {passed}/{} engine={} receipts={}\n",
            if self.passed() { "PASS" } else { "FAIL" }, CHECKS.len(), short(&self.engine, 9), short(&self.digest, 16)));
        out
    }
}

/// `vade-tools qualify-card DIR [--main-ref REF] [--goldens TSV] [--repo DIR] [--goldens-bin BIN]`.
/// Exit 1 unless every check passes.
pub fn run(args: &[String]) -> Result<()> {
    let mut dir = None;
    let exe = std::env::current_exe()?;
    let mut options = Options { dir: PathBuf::new(), repo: crate::root(), main_ref: "main".into(), goldens: None,
        goldens_bin: exe.with_file_name("check-goldens") };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--main-ref" => options.main_ref = it.next().ok_or("--main-ref needs a ref")?.clone(),
            "--goldens" => options.goldens = Some(it.next().ok_or("--goldens needs a file")?.into()),
            "--repo" => options.repo = it.next().ok_or("--repo needs a folder")?.into(),
            "--goldens-bin" => options.goldens_bin = it.next().ok_or("--goldens-bin needs a program")?.into(),
            _ if dir.is_none() && !a.starts_with("--") => dir = Some(PathBuf::from(a)),
            _ => return Err(format!("unknown argument {a}").into()),
        }
    }
    options.dir = dir.ok_or("qualify-card needs a card folder")?;
    let report = check(&options)?;
    print!("{}", report.render());
    if report.passed() { Ok(()) } else { Err("card is not qualified".into()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn short_never_cuts_inside_a_character() {
        assert_eq!(short("abc", 12), "abc");
        assert_eq!(short("ééééé", 2), "éé");
        assert_eq!(gates_line("é15/15 BASIC OUTPUT GATES PASSED"), Some((15, 15)));
        assert_eq!(gates_line("ééé/é BASIC OUTPUT GATES PASSED"), None);
    }
    #[test]
    fn embeds_resolve_relative_to_the_source_file() {
        let src = "let a = include_str!(\"../../docs/x.md\");\nlet b = include_bytes!(\n  \"fixtures/y.bin\");\n\
            let c = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/k.ptx\"));\nlet d = include_str!(concat!(\"../t/\", $n, \".md\"));";
        assert_eq!(embeds("engine/src/a.rs", src).unwrap(), vec![Embed::File("docs/x.md".into()), Embed::File("engine/src/fixtures/y.bin".into()),
            Embed::Prefix("engine/t/".into())]);
        assert!(embeds("e/a.rs", "include_str!(PATH)").is_err());
    }
}
