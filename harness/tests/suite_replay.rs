//! Replay grading from small recorded replies. No model or Spark runs.
#[path = "support/home.rs"]
mod home;

use std::path::{Path, PathBuf};
use std::process::Output;
use std::sync::atomic::{AtomicUsize, Ordering};
use vade_harness::json::{self, Value};

const RECORDS: &str = "tests/fixtures/suite-replay";
const TASK: &str = "markdown-spec-qa";

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("suite-replay-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn copy_records(&self) -> PathBuf {
        let dir = self.0.join("records");
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(RECORDS);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(source.join("replay.json"), dir.join("replay.json")).unwrap();
        for id in [TASK, "refuse-outside-path", "structured-output"] {
            let rel = format!("tasks/{id}/transcript.jsonl");
            std::fs::create_dir_all(dir.join("tasks").join(id)).unwrap();
            std::fs::copy(source.join(&rel), dir.join(rel)).unwrap();
        }
        dir
    }
}
impl Drop for Scratch { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

fn replay(dir: &Path, out: Option<&Path>, only: Option<&str>) -> Output {
    let cargo = PathBuf::from(env!("CARGO"));
    let toolchain = cargo.parent().and_then(Path::parent).filter(|t| t.join("bin/rustc").is_file() && t.join("lib").is_dir());
    let mut command = home::harness();
    command.args(["suite", "--replay"]).arg(dir).current_dir(env!("CARGO_MANIFEST_DIR"));
    if let Some(t) = toolchain { command.arg("--toolchain").arg(t); }
    if let Some(p) = out { command.arg("--out").arg(p); }
    if let Some(id) = only { command.args(["--only", id]); }
    command.output().unwrap()
}

fn text(out: &Output) -> String { format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)) }
fn read_json(path: &Path) -> Value { json::parse(&std::fs::read_to_string(path).unwrap()).unwrap() }
fn number(value: usize) -> Value { Value::Number(value.to_string()) }
fn string(value: &str) -> Value { Value::String(value.into()) }
fn object(fields: Vec<(&str, Value)>) -> Value { Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect()) }
fn set(value: &mut Value, name: &str, field: Value) {
    let Value::Object(fields) = value else { panic!("expected object") };
    let entry = fields.iter_mut().find(|(k, _)| k == name).unwrap();
    entry.1 = field;
}
fn task<'a>(card: &'a Value, id: &str) -> &'a Value {
    card.get("tasks").and_then(Value::as_array).unwrap().iter().find(|t| t.get("id").and_then(Value::as_str) == Some(id)).unwrap()
}

#[test]
fn recorded_replies_repeat_the_score_and_write_category_cards() {
    let scratch = Scratch::new();
    let records = scratch.copy_records();
    let mut cards = Vec::new();
    for name in ["first", "second"] {
        let dir = if name == "first" { scratch.0.join(name) } else { records.join("replay-cards") };
        let out = replay(&records, if name == "first" { Some(&dir) } else { None }, None);
        assert!(out.status.success(), "{}", text(&out));
        let card = read_json(&dir.join("card.json"));
        assert_eq!(card.get("spark_count"), Some(&number(0)));
        assert_eq!(card.get("tasks").and_then(Value::as_array).unwrap().len(), 3);
        for (id, category) in [(TASK, "document-reading"), ("refuse-outside-path", "permission"), ("structured-output", "structured-output")] {
            let row = task(&card, id);
            assert_eq!(row.get("status"), Some(&string("PASS")));
            assert_eq!(row.get("quality"), Some(&number(3)));
            let category_dir = dir.join("categories").join(category);
            let category_card = read_json(&category_dir.join("card.json"));
            assert_eq!(category_card.get("spark_count"), Some(&number(0)));
            assert_eq!(category_card.get("tasks").and_then(Value::as_array).unwrap().len(), 1);
            assert_eq!(task(&category_card, id), row);
            let markdown = std::fs::read_to_string(category_dir.join("card.md")).unwrap();
            assert!(markdown.contains("Spark") && markdown.contains('0'), "{markdown}");
        }
        cards.push(card);
    }
    for id in [TASK, "refuse-outside-path", "structured-output"] {
        for field in ["status", "quality", "quality_misses", "changed_lines", "checks"] {
            assert_eq!(task(&cards[0], id).get(field), task(&cards[1], id).get(field), "{id}: {field}");
        }
    }
}

#[test]
fn the_card_preserves_the_recorded_spark_count() {
    let scratch = Scratch::new();
    let records = scratch.copy_records();
    // This tests metadata only. The replay still runs on the host.
    std::fs::write(records.join("replay.json"), "{\"suite_version\":\"agent-suite-2\",\"spark_count\":4}\n").unwrap();
    let dir = scratch.0.join("cards");
    let out = replay(&records, Some(&dir), Some("structured-output"));
    assert!(out.status.success(), "{}", text(&out));
    for path in ["card.json", "categories/structured-output/card.json"] {
        assert_eq!(read_json(&dir.join(path)).get("spark_count"), Some(&number(4)));
    }
    let markdown = std::fs::read_to_string(dir.join("categories/structured-output/card.md")).unwrap();
    assert!(markdown.contains("Spark count (recorded): 4"), "{markdown}");
}

fn direct_response(dir: &Path, status: usize, body: &str) {
    let path = dir.join("tasks/structured-output/transcript.jsonl");
    let mut record = read_json(&path);
    set(&mut record, "status", number(status));
    set(&mut record, "content_type", string("application/json"));
    set(&mut record, "response", string(body));
    std::fs::write(path, json::stringify(&record).unwrap() + "\n").unwrap();
}

#[test]
fn a_recorded_http_error_keeps_its_status() {
    let scratch = Scratch::new();
    let records = scratch.copy_records();
    direct_response(&records, 422, r#"{"error":{"message":"JSON mode is unsupported"}}"#);
    let dir = scratch.0.join("cards");
    let out = replay(&records, Some(&dir), Some("structured-output"));
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    let card = read_json(&dir.join("card.json"));
    let row = task(&card, "structured-output");
    assert_eq!(row.get("status"), Some(&string("UNSUPPORTED")));
    assert_eq!(row.get("quality"), Some(&number(0)));
    assert!(row.get("note").and_then(Value::as_str).unwrap().starts_with("HTTP 422:"));
    let record = read_json(&dir.join("tasks/structured-output/transcript.jsonl"));
    assert_eq!(record.get("status"), Some(&number(422)));
}

#[test]
fn recorded_text_keeps_literal_edit_tag_markers() {
    let scratch = Scratch::new();
    let records = scratch.copy_records();
    let content = r#"{"literal":"{{edit_tag note.txt}}"}"#;
    let body = object(vec![("choices", Value::Array(vec![object(vec![
        ("index", number(0)), ("message", object(vec![("role", string("assistant")), ("content", string(content))])),
        ("finish_reason", string("stop")),
    ])]))]);
    direct_response(&records, 200, &json::stringify(&body).unwrap());
    let dir = scratch.0.join("cards");
    let out = replay(&records, Some(&dir), Some("structured-output"));
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    assert_eq!(std::fs::read_to_string(dir.join("tasks/structured-output/work/reply.json")).unwrap(), content);
}

/// Keep the recorded reads. Change only the patch and final answer.
fn variant(dir: &Path, score: usize, wrong_answer: bool) {
    let path = dir.join(format!("tasks/{TASK}/transcript.jsonl"));
    let mut records: Vec<Value> = std::fs::read_to_string(&path).unwrap().lines().map(|l| json::parse(l).unwrap()).collect();
    let mut patch = "*** Begin Patch\n*** Add File: ANSWERS.txt\n+1: 7401\n+2: 65536\n+3: AUTH\n+4: 8\n+5: E117\n".to_string();
    if wrong_answer { patch = patch.replace("7401", "7400"); }
    if score < 2 { patch.push_str(&"+\n".repeat(12)); }
    if score == 0 { patch.push_str("*** Add File: notes.txt\n+Extra text.\n"); }
    patch.push_str("*** End Patch\n");
    let args = json::stringify(&object(vec![("patch", string(&patch))])).unwrap();
    let call = object(vec![("id", string("call_3")), ("type", string("function")), ("function", object(vec![("name", string("apply_patch")), ("arguments", string(&args))]))]);
    let patch_message = object(vec![("role", string("assistant")), ("tool_calls", Value::Array(vec![call]))]);
    let reply = if score == 3 { "ANSWERS.txt holds the five answers." } else { "Done." };
    let answer_message = object(vec![("role", string("assistant")), ("content", string(reply))]);
    for (index, message, finish) in [(2, patch_message, "tool_calls"), (3, answer_message, "stop")] {
        let body = object(vec![("choices", Value::Array(vec![object(vec![("index", number(0)), ("message", message), ("finish_reason", string(finish))])]))]);
        set(&mut records[index], "content_type", string("application/json"));
        set(&mut records[index], "response", string(&json::stringify(&body).unwrap()));
    }
    let data = records.iter().map(|r| json::stringify(r).unwrap()).collect::<Vec<_>>().join("\n") + "\n";
    std::fs::write(path, data).unwrap();
}

#[test]
fn recorded_patch_variants_cover_each_quality_score() {
    for score in 0..=3 {
        let scratch = Scratch::new();
        let records = scratch.copy_records();
        variant(&records, score, false);
        let dir = scratch.0.join("cards");
        let out = replay(&records, Some(&dir), Some(TASK));
        assert!(out.status.success(), "{}", text(&out));
        let card = read_json(&dir.join("card.json"));
        let row = task(&card, TASK);
        assert_eq!(row.get("status"), Some(&string("PASS")));
        assert_eq!(row.get("quality"), Some(&number(score)), "{row:?}");
        assert_eq!(row.get("quality_misses").and_then(Value::as_array).unwrap().len(), 3 - score);
    }
}

#[test]
fn a_wrong_answer_fails_completion_and_scores_zero() {
    let scratch = Scratch::new();
    let records = scratch.copy_records();
    variant(&records, 3, true);
    let out = replay(&records, None, Some(TASK));
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    let card = read_json(&records.join("replay-cards/card.json"));
    let row = task(&card, TASK);
    assert_eq!(row.get("status"), Some(&string("FAIL")));
    assert_eq!(row.get("quality"), Some(&number(0)));
    assert!(row.get("checks").and_then(Value::as_array).unwrap().iter().any(|c| c.get("pass") == Some(&Value::Bool(false))));
}

#[test]
fn a_broken_record_has_a_clear_error_without_a_panic() {
    let scratch = Scratch::new();
    let records = scratch.copy_records();
    let broken = Path::new(env!("CARGO_MANIFEST_DIR")).join(RECORDS).join("broken.jsonl");
    std::fs::copy(broken, records.join(format!("tasks/{TASK}/transcript.jsonl"))).unwrap();
    let dir = scratch.0.join("cards");
    let out = replay(&records, Some(&dir), Some(TASK));
    let message = text(&out);
    assert_eq!(out.status.code(), Some(2), "{message}");
    assert!(message.contains("transcript.jsonl") && message.contains("line 1"), "{message}");
    assert!(!message.contains("panicked"), "{message}");
    assert!(!dir.join("card.json").exists());
}

#[test]
fn a_missing_final_reply_reports_the_call_mismatch() {
    let scratch = Scratch::new();
    let records = scratch.copy_records();
    let path = records.join(format!("tasks/{TASK}/transcript.jsonl"));
    let data = std::fs::read_to_string(&path).unwrap();
    let mut lines: Vec<&str> = data.lines().collect();
    lines.pop().unwrap();
    std::fs::write(path, lines.join("\n") + "\n").unwrap();
    let dir = scratch.0.join("cards");
    let out = replay(&records, Some(&dir), Some(TASK));
    let message = text(&out);
    assert_eq!(out.status.code(), Some(2), "{message}");
    assert!(message.contains("replay calls do not match the transcript"), "{message}");
    assert!(!message.contains("panicked"), "{message}");
    assert!(!dir.join("card.json").exists());
}
