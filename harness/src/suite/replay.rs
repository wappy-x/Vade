//! Grade saved suite replies with the same task runner, checks and rubric.
//! These cards measure replay completion and quality, not model exactness.
use super::{Mode, Opts, TaskResult, load_tasks, n, new_dir, obj, pretty, replay_record, run_task, s, serve, suite_version};
use crate::json::{self, Value};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;

const MAX_TRANSCRIPT: u64 = 64 << 20;

fn read(path: &Path, limit: u64) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut text = String::new();
    file.take(limit + 1).read_to_string(&mut text).map_err(|e| format!("{}: {e}", path.display()))?;
    if text.len() as u64 > limit { return Err(format!("{}: record exceeds {limit} bytes", path.display())); }
    Ok(text)
}

fn metadata(text: &str, version: &str) -> Result<u64, String> {
    let v = json::parse(text).map_err(|e| format!("replay.json: {e}"))?;
    let recorded = v.get("suite_version").and_then(Value::as_str).ok_or("replay.json: suite_version must be a string")?;
    if recorded != version { return Err(format!("replay.json: suite_version {recorded:?} does not match {version:?}")); }
    v.get("spark_count").and_then(super::u64_of).ok_or_else(|| "replay.json: spark_count must be a nonnegative integer; use 0 for a host fixture".into())
}

pub(super) fn run(o: &Opts) -> Result<u8, String> {
    let Mode::Replay { dir } = &o.mode else { return Err("not a replay run".into()) };
    let version = suite_version(o);
    let sparks = metadata(&read(&dir.join("replay.json"), 64 << 10)?, &version)?;
    let tasks_dir = dir.join("tasks");
    let mut ids = BTreeSet::new();
    for entry in std::fs::read_dir(&tasks_dir).map_err(|e| format!("{}: {e}", tasks_dir.display()))? {
        let entry = entry.map_err(|e| format!("tasks: {e}"))?;
        if !entry.file_type().map_err(|e| e.to_string())?.is_dir() { return Err("replay tasks must be folders named for suite tasks".into()); }
        let id = entry.file_name().into_string().map_err(|_| "a replay task name is not UTF-8")?;
        ids.insert(id);
    }
    if ids.is_empty() { return Err("replay has no task transcripts".into()); }
    for id in &o.only { if !ids.contains(id) { return Err(format!("replay has no transcript for task {id}")); } }
    // Read all selected records before creating output or starting a harness.
    let tasks = load_tasks(o)?;
    let mut recordings = Vec::new();
    for (task_dir, task) in tasks {
        if !ids.remove(&task.id) { continue; }
        let path = tasks_dir.join(&task.id).join("transcript.jsonl");
        let recording = replay_record::parse(&read(&path, MAX_TRANSCRIPT)?, &task).map_err(|e| format!("{}/transcript.jsonl: {e}", task.id))?;
        if !safe_name(&task.category) { return Err(format!("{}: category must use letters, digits, hyphens or underscores", task.id)); }
        recordings.push((task_dir, task, recording));
    }
    if o.only.is_empty() && !ids.is_empty() { return Err(format!("replay has unknown task {}", ids.first().map(String::as_str).unwrap_or(""))); }
    if recordings.is_empty() { return Err("replay has no selected task transcripts".into()); }
    let default_out = dir.join("replay-cards");
    let out = new_dir(o.out.as_deref().unwrap_or(&default_out))?;
    let mut results = Vec::new();
    for (task_dir, task, recording) in recordings {
        let id = task.id.clone();
        let result = run_task(o, &task_dir, task, &out.join("tasks").join(&id), serve::Source::Replay(Mutex::new(recording.replies.into())), None);
        if result.status == "ERROR" { return Err(format!("{id}: replay failed: {}", result.note)); }
        if result.replies_left != Some(0) { return Err(format!("{id}: replay left {} recorded replies unused", result.replies_left.unwrap_or(recording.calls.len()))); }
        let mut actual: Vec<(usize, usize, u16)> = result.exchanges.iter().map(|x| (x.step, x.call, x.status)).collect();
        actual.sort();
        if actual != recording.calls { return Err(format!("{id}: replay calls do not match the transcript; a reply is missing or a call failed")); }
        if result.steps.iter().any(|s| s.timed_out) || result.note.contains(super::INCOMPLETE) { return Err(format!("{id}: replay did not finish all calls")); }
        results.push(result);
    }
    let all: Vec<&TaskResult> = results.iter().collect();
    save_card(&out, &version, sparks, None, &all)?;
    let categories: BTreeSet<&str> = all.iter().map(|r| r.task.category.as_str()).collect();
    for category in categories {
        let selected: Vec<&TaskResult> = all.iter().copied().filter(|r| r.task.category == category).collect();
        let dest = out.join("categories").join(category);
        std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
        save_card(&dest, &version, sparks, Some(category), &selected)?;
    }
    println!("{}", markdown(&version, sparks, None, &all));
    println!("Replay cards: {}", out.display());
    Ok(if all.iter().all(|r| r.status == "PASS") { 0 } else { 1 })
}

fn safe_name(name: &str) -> bool { !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') }

fn card(version: &str, sparks: u64, category: Option<&str>, results: &[&TaskResult]) -> Value {
    let tasks = results.iter().map(|r| obj(vec![
        ("id", s(&r.task.id)), ("category", s(&r.task.category)), ("status", s(&r.status)), ("note", s(&r.note)),
        ("quality", n(r.quality.as_ref().map_or(0, |q| q.score))),
        ("quality_misses", Value::Array(r.quality.iter().flat_map(|q| &q.misses).map(|m| s(m)).collect())),
        ("changed_lines", r.quality.as_ref().map_or(Value::Null, |q| n(q.changed_lines))),
        ("checks_passed", n(r.checks.iter().filter(|c| c.pass).count())), ("checks_total", n(r.checks.len())),
        ("checks", Value::Array(r.checks.iter().map(|c| obj(vec![("check", s(&c.check)), ("pass", Value::Bool(c.pass)), ("detail", s(&c.detail))])).collect())),
    ])).collect();
    let quality_sum: u64 = results.iter().map(|r| u64::from(r.quality.as_ref().map_or(0, |q| q.score))).sum();
    obj(vec![
        ("card", s("vade-agent-suite-replay")), ("source", s("replay")), ("suite_version", s(version)),
        ("spark_count", n(sparks)), ("category", category.map_or(Value::Null, s)),
        ("completed", n(results.iter().filter(|r| r.status == "PASS").count())), ("task_count", n(results.len())),
        ("quality_sum", n(quality_sum)), ("quality_max", n(results.len() * 3)), ("tasks", Value::Array(tasks)),
    ])
}

fn markdown(version: &str, sparks: u64, category: Option<&str>, results: &[&TaskResult]) -> String {
    let mut text = format!("# Vade suite replay card\n\nSuite: {version}. Category: {}.\n\nSpark count (recorded): {sparks}. Replay uses no model or Spark.\n\n| Task | Category | Status | Checks | Quality (0-3) |\n| --- | --- | --- | --- | --- |\n", category.unwrap_or("all"));
    for r in results {
        text.push_str(&format!("| {} | {} | {} | {} | {} |\n", r.task.id, r.task.category, r.status, r.score(), r.quality.as_ref().map_or(0, |q| q.score)));
    }
    text.push_str("\nReplay checks completion and quality. It does not prove model exactness or speed.\n");
    for r in results {
        if r.status != "PASS" { text.push_str(&format!("\n- {}: {}\n", r.task.id, r.note)); }
        for miss in r.quality.iter().flat_map(|q| &q.misses) { text.push_str(&format!("- {} quality: {miss}\n", r.task.id)); }
    }
    text
}

fn save_card(out: &Path, version: &str, sparks: u64, category: Option<&str>, results: &[&TaskResult]) -> Result<(), String> {
    std::fs::write(out.join("card.json"), pretty(&card(version, sparks, category, results), 0) + "\n").map_err(|e| e.to_string())?;
    std::fs::write(out.join("card.md"), markdown(version, sparks, category, results)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_requires_the_suite_and_recorded_spark_count() {
        assert_eq!(metadata(r#"{"suite_version":"agent-suite-2","spark_count":0}"#, "agent-suite-2").unwrap(), 0);
        for bad in [r#"{"suite_version":"old","spark_count":0}"#, r#"{"suite_version":"agent-suite-2"}"#,
            r#"{"suite_version":"agent-suite-2","spark_count":-1}"#, r#"{"suite_version":"agent-suite-2","spark_count":1.5}"#, "{"] {
            assert!(metadata(bad, "agent-suite-2").unwrap_err().starts_with("replay.json:"));
        }
    }
}
